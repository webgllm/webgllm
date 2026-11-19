/**
 * What predictSpeed measures (see predict-speed.ts): a model's shapes from
 * its GGUF header or its ONNX graph and config (never the weights), one
 * layer of each kind it has with random weights in the storage the real
 * loader gives them, and the engine's own prompt and writing passes over
 * them, timed with fences.
 */
import { Engine, TUNING } from "./engine.js"
import { GL, type Gpu } from "./gpu/gl.js"
import { WebGpu } from "./gpu/webgpu.js"
import { GgmlType, readGgufHeader, tensorByteLength, type GgufHeader } from "./gguf.js"
import { bytesToFloats, halvesToFloats, toHalf } from "./half.js"
import { PROMPT_BATCH } from "./handler.js"
import { ggufConfig, matrixStorage } from "./load-gguf.js"
import { raonConfig, raonLayerMatrices, raonMatrixStorage } from "./load-raon.js"
import { OnnxModel, planOnnx, type NBits } from "./load-onnx.js"
import { parseOnnxModel } from "./onnx.js"
import { source } from "./model-files.js"
import { fileUrl, type ModelFile, type WebGlModelMeta } from "./model-meta.js"
import { storedFile } from "./storage.js"
import { newQuantMatrix } from "./repack.js"
import type { ByteSource, LayerType, LayerWeights, ModelConfig, QuantMatrix, StreamMixWeights, WeightPiece } from "./weights.js"

/** How a matrix is stored: as the loader would hand it to the engine. */
export type MatrixStorage = {
  bits: 4 | 6 | 8
  zeros: boolean
  /** 4-bit rows as MatMulNBits stores them (QuantMatrix.raw: ONNX). */
  raw: boolean
  /** 4-bit blocks' zero points as halves beside their scales (QuantMatrix.zeroHalves: a GGUF's Q4_K matrices). */
  halves?: true
}

/** A layer kind's matrices, by role (see LayerWeights). */
export type KindStorage = Record<string, MatrixStorage>

/** A model as the GPU sees it, without its weights. */
export type SpeedShape = {
  config: ModelConfig
  /** Each kind of layer the model has, in order of first appearance: its matrices' storage. */
  kinds: Array<{ type: LayerType; matrices: KindStorage }>
  /** The embedding, also the output head. */
  embedding: MatrixStorage
  /** Raon's per-layer token embeddings: the model's projection, and the tensor type of the table's rows in the file. */
  perLayer?: { proj: MatrixStorage; type: number }
}

/** (The engine's: a prompt's rows a split matmul takes.) */
const SPLIT_ROWS = 8

/**
 * Prompt tokens predictSpeed times: half a batch, the fewest that take a
 * prompt's own kernels (TUNING.quadMinBatch's 4-token matmuls; 128 took
 * others, and guessed a Pixel 10 Pro's Chrome at half its speed and an M3
 * Ultra's at 40%), a fast guess where a full batch of 512 took the Pixel's
 * Firefox 8 s of the probe's 14.
 */
export const PROMPT_TOKENS = 256

/** Most middle layers the probe's prompts run (see middleStack): about one of each kind. */
export const PROMPT_STACK = 2

/**
 * Up to `slots` of a model's middle layers (all but its first and last),
 * its kinds in proportion (largest remainders), each kind's picks spread
 * evenly over its own layers, in the model's order.
 */
export function middleStack(layerTypes: ReadonlyArray<LayerType>, slots: number): Array<LayerType> {
  const middle = layerTypes.slice(1, -1)
  if (middle.length <= slots) return middle
  const at = new Map<LayerType, Array<number>>()
  middle.forEach((t, i) => at.set(t, [...(at.get(t) ?? []), i]))
  const kinds = [...at.entries()].map(([type, indices]) => {
    const share = (indices.length * slots) / middle.length
    return { type, indices, count: Math.floor(share), rest: share - Math.floor(share) }
  })
  let left = slots - kinds.reduce((sum, k) => sum + k.count, 0)
  for (const k of [...kinds].sort((a, b) => b.rest - a.rest)) {
    if (left <= 0) break
    k.count++
    left--
  }
  const picked = kinds.flatMap((k) =>
    Array.from({ length: k.count }, (_, j) => k.indices[Math.floor(((j + 0.5) * k.indices.length) / k.count)])
  )
  return picked.sort((a, b) => a - b).map((i) => middle[i])
}
/**
 * Most vocabulary rows the probe's head holds; the rest are scaled by the
 * cost per row between this and half of it (see headPerRow). An M3 Ultra's
 * head takes the same time for 4,096 rows as for 16,384 (too few to fill
 * it) and grows in proportion only from about 32,768, so fewer rows would
 * scale from a GPU not yet full.
 */
export const PROBE_VOCAB = 8192

/** The loader's storage for a matrix made of these tensors' rows (K weights each): see load-gguf.ts's matrixStorage. */
const ggufMatrix = (header: GgufHeader, K: number, names: ReadonlyArray<string>, rule = matrixStorage): MatrixStorage => {
  const infos = names.map((name) => {
    const info = header.tensors.get(name)
    if (!info) throw new Error(`GGUF: missing tensor ${name}`)
    return info
  })
  const { bits, halves } = rule(infos, K)
  return { bits, zeros: false, raw: false, ...(halves && { halves: true as const }) }
}

/** The embedding's storage, as the loaders give it: Q6_K and Q4_0 as they are, others as 8-bit. */
const ggufEmbedding = (header: GgufHeader): MatrixStorage => {
  const embedding = header.tensors.get("token_embd.weight")
  if (!embedding) throw new Error("GGUF: missing tensor token_embd.weight")
  return { bits: embedding.type === GgmlType.Q6_K ? 6 : embedding.type === GgmlType.Q4_0 ? 4 : 8, zeros: false, raw: false }
}

/** A Raon-1 nano GGUF's shape (see load-raon.ts): each kind's matrices as loadRaon stores them, and the per-layer embeddings'. */
function raonSpeedShape(header: GgufHeader): SpeedShape {
  const config = raonConfig(header)
  const kinds: SpeedShape["kinds"] = []
  config.layerTypes.forEach((type, i) => {
    if (kinds.some((k) => k.type === type)) return
    const matrices: KindStorage = {}
    for (const [role, m] of Object.entries(raonLayerMatrices(config, type)))
      matrices[role] = ggufMatrix(header, m.K, m.tensors.map((name) => `blk.${i}.${name}.weight`), raonMatrixStorage)
    kinds.push({ type, matrices })
  })
  const table = header.tensors.get("per_layer_token_embd.weight")
  if (!table) throw new Error("GGUF: missing tensor per_layer_token_embd.weight")
  // (As loadRaon stores the embedding: Q6_K as it is, others as 8-bit.)
  const embedding = ggufEmbedding(header)
  return {
    config,
    kinds,
    embedding: { ...embedding, bits: embedding.bits === 6 ? 6 : 8 },
    perLayer: { proj: ggufMatrix(header, config.hiddenSize, ["per_layer_model_proj.weight"], raonMatrixStorage), type: table.type }
  }
}

/** A GGUF's shape from its header (a range request or two) and its rope_freqs (a few hundred bytes). */
export async function ggufSpeedShape(read: { size: number; read: (offset: number, length: number) => Promise<Uint8Array> }): Promise<SpeedShape> {
  const header = await readGgufHeader(read)
  const freqs = header.tensors.get("rope_freqs.weight")
  let factors: Float32Array | undefined
  if (freqs && header.metadata.get("general.architecture") !== "raon1nano") {
    const bytes = await read.read(freqs.offset, freqs.byteLength)
    // (As the loader reads it: F32 or F16.)
    factors = freqs.type === GgmlType.F16 ? halvesToFloats(bytes) : bytesToFloats(bytes)
  }
  return ggufHeaderSpeedShape(header, factors)
}

/** A GGUF's shape from its header as read (and its rope_freqs' values, where it has them): each matrix stored as its loader stores it. */
export function ggufHeaderSpeedShape(header: GgufHeader, factors?: Float32Array): SpeedShape {
  if (header.metadata.get("general.architecture") === "raon1nano") return raonSpeedShape(header)
  const config = ggufConfig(header, factors)
  const qwen = !!config.linear
  const lfm2 = header.metadata.get("general.architecture") === "lfm2"
  const D = config.hiddenSize
  const kinds: SpeedShape["kinds"] = []
  config.layerTypes.forEach((type, i) => {
    if (kinds.some((k) => k.type === type)) return
    // Each matrix from the tensors loadGguf makes it of: its storage follows all of them.
    const m = (K: number, ...names: Array<string>) => ggufMatrix(header, K, names.map((name) => `blk.${i}.${name}.weight`))
    const matrices: KindStorage = { gateUp: m(D, "ffn_gate", "ffn_up"), down: m(config.intermediateSize, "ffn_down") }
    if (type === "linear_attention") {
      const l = config.linear!
      matrices.inProj = m(D, "attn_qkv", "attn_gate")
      matrices.ba = m(D, "ssm_beta", "ssm_alpha")
      matrices.outProj = m(l.numValueHeads * l.valueHeadDim, "ssm_out")
    } else if (type === "conv" && !qwen) {
      matrices.inProj = m(D, "shortconv.in_proj")
      matrices.outProj = m(D, "shortconv.out_proj")
    } else {
      // (Qwen3.5's gate rows are the other half of attn_q; LFM3 has a tensor for them, LFM2 no gate.)
      matrices.qkvg = m(D, "attn_q", "attn_k", "attn_v", ...(qwen || lfm2 ? [] : ["attn_gate"]))
      matrices.o = m(config.numHeads * config.headDim, "attn_output")
    }
    kinds.push({ type, matrices })
  })
  return { config, kinds, embedding: ggufEmbedding(header) }
}

/** Storage of an ONNX matrix made of MatMulNBits parts, as loadOnnx stores it. */
const onnxMatrix = (parts: Array<NBits | undefined>): MatrixStorage => {
  const ws = parts.filter((w): w is NBits => !!w)
  if (ws.some((w) => w.bits === 8)) return { bits: 8, zeros: false, raw: false }
  return { bits: 4, zeros: ws.some((w) => !!w.zp), raw: true }
}

/** An ONNX model's shape from its graph (no external data) and config.json. */
export function onnxSpeedShape(graphBytes: Uint8Array, configJson: unknown): SpeedShape {
  const graph = parseOnnxModel(graphBytes)
  const { config, layers } = planOnnx(graph, configJson, 2048, () => undefined)
  const kinds: SpeedShape["kinds"] = []
  layers.forEach((l) => {
    if (kinds.some((k) => k.type === l.type)) return
    const [gate, up, down] = l.matrices.slice(-3)
    const matrices: KindStorage = { gateUp: onnxMatrix([gate, up]), down: onnxMatrix([down]) }
    if (l.type === "linear_attention") {
      const [qkv, z, b, a, out] = l.matrices
      matrices.inProj = onnxMatrix([qkv, z])
      matrices.ba = onnxMatrix([b, a])
      matrices.outProj = onnxMatrix([out])
    } else if (l.type === "conv") {
      matrices.inProj = onnxMatrix([l.matrices[0]])
      matrices.outProj = onnxMatrix([l.matrices[1]])
    } else {
      const [q, k, v] = l.matrices
      matrices.qkvg = onnxMatrix([q, k, v, l.gateWeight])
      matrices.o = onnxMatrix([l.matrices[l.gateWeight ? 4 : 3]])
    }
    kinds.push({ type: l.type, matrices })
  })
  // As loadOnnx stores the embedding: 4-bit rows as they are, 8-bit and
  // float ones quantized to 8 bits.
  const emb = new OnnxModel(graph).embedding(config.hiddenSize)
  const embedding: MatrixStorage = emb.kind === "q4"
    ? { bits: 4, zeros: !!emb.zp, raw: true }
    : { bits: 8, zeros: false, raw: false }
  return { config, kinds, embedding }
}

/** A model's shape: its GGUF header, or its ONNX graph and config, read from storage or the Hub. */
export async function speedShape(meta: WebGlModelMeta): Promise<SpeedShape> {
  if (meta.files.kind === "gguf") return ggufSpeedShape(await source(meta.files.model))
  const [graphBytes, configBytes] = await Promise.all([fileBytes(meta.files.graph), fileBytes(meta.files.config)])
  return onnxSpeedShape(graphBytes, JSON.parse(new TextDecoder().decode(configBytes)) as unknown)
}

/** A small file whole (an ONNX graph without its data, a config): from storage, else the Hub. */
async function fileBytes(file: ModelFile): Promise<Uint8Array> {
  const stored = await storedFile(file)
  if (stored) return new Uint8Array(await stored.arrayBuffer())
  const response = await fetch(fileUrl(file), { referrerPolicy: "no-referrer" })
  if (!response.ok) throw new Error(`${file.path}: ${response.status}`)
  return new Uint8Array(await response.arrayBuffer())
}

/** The rows and columns of each matrix role of a layer kind (see LayerWeights). */
export function matrixShapes(config: ModelConfig, type: LayerType): Record<string, { N: number; K: number }> {
  if (config.streams)
    return Object.fromEntries(Object.entries(raonLayerMatrices(config, type)).map(([role, m]) => [role, { N: m.N, K: m.K }]))
  const D = config.hiddenSize
  const F = config.intermediateSize
  const shapes: Record<string, { N: number; K: number }> = {
    gateUp: { N: 2 * F, K: D },
    down: { N: D, K: F }
  }
  if (type === "linear_attention") {
    const l = config.linear
    if (!l) throw new Error("The model has no Gated DeltaNet shapes.")
    const keys = l.numKeyHeads * l.keyHeadDim
    const values = l.numValueHeads * l.valueHeadDim
    shapes.inProj = { N: 2 * keys + 2 * values, K: D }
    shapes.ba = { N: 2 * l.numValueHeads, K: D }
    shapes.outProj = { N: D, K: values }
  } else if (type === "conv") {
    shapes.inProj = { N: 3 * D, K: D }
    shapes.outProj = { N: D, K: D }
  } else {
    const q = config.numHeads * config.headDim
    const gate = config.attentionGate ? (config.gatePerDim ? q : config.numHeads) : 0
    shapes.qkvg = { N: q + 2 * config.numKvHeads * config.headDim + gate, K: D }
    shapes.o = { N: D, K: q }
  }
  return shapes
}

/** Random words, shared by every synthetic matrix (each a view of them). */
class Noise {
  private words = new Uint32Array(0)
  private bytes = new Uint8Array(0)
  private halves = new Uint16Array(0)
  private points = new Uint8Array(0)
  private seed = 0x9e3779b9
  private grow(length: number) {
    if (this.words.length >= length) return
    const words = new Uint32Array(length)
    let s = this.seed
    for (let i = 0; i < length; i++) {
      // xorshift32
      s ^= s << 13
      s ^= s >>> 17
      s ^= s << 5
      words[i] = s >>> 0
    }
    this.seed = s
    this.words = words
    this.bytes = new Uint8Array(words.buffer)
  }
  u32(length: number) {
    this.grow(length)
    return this.words.subarray(0, length)
  }
  u8(length: number) {
    this.grow(Math.ceil(length / 4))
    return this.bytes.subarray(0, length)
  }
  /** Scales of about 0.01 (half bits). */
  scales(length: number) {
    if (this.halves.length < length) this.halves = new Uint16Array(length).fill(0x211f)
    return this.halves.subarray(0, length)
  }
  zeros(length: number) {
    if (this.points.length < length) this.points = new Uint8Array(length).fill(8)
    return this.points.subarray(0, length)
  }
  private halfPoints = new Uint16Array(0)
  /** Zero points of 8 as halves (a Q4_K block's: see QuantMatrix.zeroHalves). */
  halfZeros(length: number) {
    if (this.halfPoints.length < length) this.halfPoints = new Uint16Array(length).fill(toHalf(8))
    return this.halfPoints.subarray(0, length)
  }
  private small = new Uint16Array(0)
  /** Values of 0.01 as halves (a latent attention's per-head matrices). */
  smallHalves(length: number) {
    if (this.small.length < length) this.small = new Uint16Array(length).fill(toHalf(0.01))
    return this.small.subarray(0, length)
  }
}

/** A matrix of random weights in `storage`, as the loader would give it. */
function randomMatrix(noise: Noise, N: number, K: number, storage: MatrixStorage): QuantMatrix {
  const blocks = K / (storage.bits === 6 ? 256 : 32)
  if (storage.raw && storage.bits === 4)
    return {
      N, K, bits: 4,
      quants: new Uint32Array(0),
      raw: noise.u8((N * K) / 2),
      scales: noise.scales(N * blocks),
      zeros: storage.zeros ? noise.zeros(N * blocks) : null
    }
  const shape = newQuantMatrix(4, K, storage.bits, false)
  const wordsPerRow = shape.quants.length / 4
  return {
    N, K, bits: storage.bits,
    quants: noise.u32(N * wordsPerRow),
    scales: noise.scales(N * blocks),
    zeros: storage.bits === 4 && storage.zeros ? noise.zeros(N * blocks) : null,
    ...(storage.bits === 4 && storage.halves && { zeroHalves: noise.halfZeros(N * blocks) })
  }
}


/** One layer of `type` with random weights. */
function randomLayer(noise: Noise, config: ModelConfig, type: LayerType, matrices: KindStorage): LayerWeights {
  const D = config.hiddenSize
  const shapes = matrixShapes(config, type)
  const m = (role: string) => {
    const storage = matrices[role]
    if (!storage) throw new Error(`No storage for ${type}'s ${role}.`)
    return randomMatrix(noise, shapes[role].N, shapes[role].K, storage)
  }
  const ones = (n: number) => new Float32Array(n).fill(1)
  if (config.streams && config.latent && config.perLayer && config.linear) {
    // Raon-1 nano: a layer's two mixing sites, its operator and its per-layer part (see load-raon.ts).
    const S = config.streams.count
    const mixRows = 2 * S + (S - 1) ** 2
    const mix = (role: string): StreamMixWeights => ({
      fn: m(role),
      base: new Float32Array(mixRows),
      chart: new Float32Array(mixRows * S * S + 2 * S * S).fill(0.01)
    })
    const layer: LayerWeights = {
      type,
      operatorNorm: ones(D),
      ffnNorm: ones(D),
      gateUp: m("gateUp"),
      down: m("down"),
      mix: { operator: mix("mixOperator"), ffn: mix("mixFfn") },
      perLayer: { gate: m("perLayerGate"), proj: m("perLayerProj"), norm: ones(D) }
    }
    const l = config.linear
    const latent = config.latent
    if (type === "linear_attention") {
      const inner = l.numKeyHeads * l.keyHeadDim
      layer.linear = {
        inProj: m("inProj"),
        ba: m("ba"),
        decay: m("decay"),
        outProj: m("outProj"),
        conv: new Float32Array(3 * inner * l.convKernel).fill(0.1),
        a: new Float32Array(l.numKeyHeads),
        dtBias: new Float32Array(inner),
        norm: ones(l.valueHeadDim)
      }
    } else {
      layer.latent = {
        inProj: m("latentIn"),
        qB: m("qB"),
        kB: noise.smallHalves((latent.keyDim - latent.shared) * latent.rank * latent.heads),
        vB: noise.smallHalves(latent.rank * latent.valueDim * latent.heads),
        o: m("o"),
        qNorm: ones(latent.queryRank),
        kvNorm: ones(latent.rank)
      }
    }
    return layer
  }
  const lfm3 = config.attentionGate && !config.linear
  const layer: LayerWeights = {
    type,
    operatorNorm: ones(D),
    ffnNorm: ones(D),
    gateUp: m("gateUp"),
    down: m("down"),
    ...(lfm3 && { postOperatorNorm: ones(D), postFfnNorm: ones(D) })
  }
  if (type === "conv")
    layer.conv = { inProj: m("inProj"), outProj: m("outProj"), kernel: new Float32Array(D * config.convKernel).fill(0.1) }
  else if (type === "linear_attention") {
    const l = config.linear!
    const channels = 2 * l.numKeyHeads * l.keyHeadDim + l.numValueHeads * l.valueHeadDim
    layer.linear = {
      inProj: m("inProj"),
      ba: m("ba"),
      outProj: m("outProj"),
      conv: new Float32Array(channels * l.convKernel).fill(0.1),
      a: new Float32Array(l.numValueHeads).fill(-1),
      dtBias: new Float32Array(l.numValueHeads),
      norm: ones(l.valueHeadDim)
    }
  } else
    layer.attention = { qkvg: m("qkvg"), o: m("o"), qNorm: ones(config.headDim), kNorm: ones(config.headDim) }
  return layer
}

/** The probe's model: one layer of each kind, and a head of at most PROBE_VOCAB rows. */
export function probeConfig(shape: SpeedShape): ModelConfig {
  const vocab = Math.min(shape.config.vocabSize, PROBE_VOCAB) & ~3
  return {
    ...shape.config,
    numLayers: shape.kinds.length,
    layerTypes: shape.kinds.map((k) => k.type),
    vocabSize: vocab,
    // (Per-layer inputs for every layer of the model: each kind's one layer stands in for all of its kind, at its own place.)
    ...(shape.config.perLayer && { perLayer: { ...shape.config.perLayer, layers: shape.config.numLayers } })
  }
}

/** The per-layer table's rows in the probe: a row's bytes for its tensor type. */
function perLayerTable(shape: SpeedShape, config: ModelConfig) {
  if (!shape.perLayer || !config.perLayer) return undefined
  const width = config.perLayer.dim * (config.perLayer.layers ?? config.numLayers)
  return { offset: 0, rowBytes: tensorByteLength(shape.perLayer.type, [width, 1]), type: shape.perLayer.type, rows: config.vocabSize }
}

/**
 * What the probe's engine reads a token's per-layer row from: zero bytes
 * (every type's zero weights), as many as asked for. The read itself isn't
 * what the probe times (the GPU's passes are).
 */
export function probePerLayerSource(shape: SpeedShape): ByteSource | undefined {
  const table = perLayerTable(shape, probeConfig(shape))
  if (!table) return undefined
  return { size: table.rows * table.rowBytes, read: async (_offset, length) => new Uint8Array(length) }
}

/** The probe model's pieces, as a loader yields them. */
export async function* probePieces(shape: SpeedShape, noise = new Noise()): AsyncGenerator<WeightPiece> {
  const config = probeConfig(shape)
  yield { kind: "config", config }
  const table = perLayerTable(shape, config)
  if (table && shape.perLayer && config.perLayer)
    yield {
      kind: "perLayer",
      proj: randomMatrix(noise, config.perLayer.dim * (config.perLayer.layers ?? config.numLayers), config.hiddenSize, shape.perLayer.proj),
      norm: new Float32Array(config.perLayer.dim).fill(1),
      table
    }
  for (const [i, kind] of shape.kinds.entries())
    yield { kind: "layer", layer: i, value: randomLayer(noise, config, kind.type, kind.matrices) }
  yield { kind: "finalNorm", value: new Float32Array(config.hiddenSize).fill(1) }
  // In chunks of rows, as the loaders hand it over (the same random rows
  // each time: no head-sized buffer in memory).
  const rows = EMBEDDING_CHUNK_ROWS
  for (let row0 = 0; row0 < config.vocabSize; row0 += rows) {
    const N = Math.min(rows, config.vocabSize - row0)
    yield { kind: "embedding", value: randomMatrix(noise, N, config.hiddenSize, shape.embedding), row0, totalRows: config.vocabSize }
  }
}

/** Embedding rows per piece of the probe's load. */
const EMBEDDING_CHUNK_ROWS = 4096

/** What the probe timed, in milliseconds (see speedFromTimes). */
export type ProbeTimes = {
  /** The GPU's time for a prompt through the model's first and last layers only (prompts queued back to back). */
  promptBase: number
  /** The same with `stackLayers` of the model's middle layers between them (see middleStack). */
  promptStack: number
  stackLayers: number
  /** The GPU's time for a writing step's layers, the whole model's (steps queued back to back). */
  decodeLayers: number
  /** The GPU's time for the head over the probe's vocabulary rows (one token). */
  headAt: number
  /** What a vocabulary row adds to the head (from the probe's rows and half of them). */
  headPerRow: number
  /** Rows the probe's head had. */
  vocab: number
  /** Prompt tokens timed. */
  promptTokens: number
}

/** How the probe went: how long it took, and the most GPU memory it held. */
export type ProbeReport = {
  times: ProbeTimes
  /** Milliseconds reading the shapes. */
  shapeMs: number
  /** Milliseconds on the GPU side: the synthetic load and the timings. */
  measureMs: number
  /** Of which the synthetic load. */
  loadMs: number
  /** The engine's own load times by part (Engine.loadMs). */
  loadParts: Record<string, number>
  /** GPU programs the probe compiled. */
  programs: number
  /** The most texture bytes the probe held. */
  peakBytes: number
  /** Milliseconds in each part of the timings: prompts, writing, the head. */
  phases: { prompt: number; decode: number; head: number }
}

/** Milliseconds of GPU work before the probe's load (see warmGpu). */
export const WARM_MS = 0

/**
 * Keeps the GPU busy for `ms` (each pass reading the last one's output,
 * so no driver skips one), so the load's own timings (Engine.chooseLayout,
 * the prompt pacing's calibration) see it at the clock a real load leaves
 * it at, a gigabyte of weights converted, not idle.
 */
export async function warmGpu(gpu: Gpu | WebGpu, ms = WARM_MS) {
  const size = 512
  const textures = [0, 1].map(() => gpu.texture(GL.RGBA32F, size, size))
  const program = gpu.program(`#version 300 es
precision highp float;uniform sampler2D uX;out vec4 o;
void main(){ivec2 p=ivec2(gl_FragCoord.xy);vec4 s=vec4(0.5);for(int i=0;i<64;i++)s+=texelFetch(uX,ivec2((p.x+i*7)&${size - 1},(p.y+i*13)&${size - 1}),0)*0.01;o=s;}`)
  try {
    for (const start = performance.now(); performance.now() - start < ms;) {
      for (let i = 0; i < 8; i++) {
        gpu.run({ program, inputs: { uX: textures[0] }, output: textures[1] })
        textures.reverse()
      }
      await gpu.finish()
    }
  } finally {
    for (const t of textures) gpu.delete(t)
  }
}

/**
 * The engine's private parts the probe drives (as Engine.tune and warmUp
 * do): its layer list, swapped for each timing, and the output head.
 */
type Internals = {
  layers: Array<unknown>
  work: unknown
  head: (row0: number, rows?: number, sum?: boolean) => Promise<unknown>
}

/**
 * Times `shape` on this GPU: after warmGpu, loads the probe model (one
 * layer of each kind; the load makes its own choices of layout and pacing,
 * as a real load does), then times a PROMPT_TOKENS prompt through the
 * model's first and last layers alone and with a few of its middle layers
 * between them (middleStack); writing steps through the whole model's order
 * of layers (each kind's one layer standing in for every layer of its
 * kind), queued back to back as the worker's pipelined writing keeps the
 * GPU busy; and the head at two vocabulary sizes. Everything is freed
 * before it returns.
 */
export async function probeSpeed(
  shape: SpeedShape,
  options: {
    canvas?: OffscreenCanvas | HTMLCanvasElement
    signal?: AbortSignal
    /** Told of the probe's engine while it measures, and of null after (to report the page's frames to it). */
    onEngine?: (engine: Engine | null) => void
    /** false: the prompt alone (writing and the head untimed: decodeLayers NaN, the head's times 0). */
    decode?: boolean
    /** false: on WebGL 2 even where there's WebGPU (WebGLLMOptions.webgpu). */
    webgpu?: boolean
  } = {}
): Promise<Omit<ProbeReport, "shapeMs">> {
  const { signal, onEngine, decode = true } = options
  const started = performance.now()
  // As the worker makes its engine (handler.ts load): on WebGPU where the
  // browser has it (WebGL 2 elsewhere, or on a canvas handed in), the same
  // batch, the model's own context length (the cache starts small and grows).
  const webgpu = !options.canvas && TUNING.webgpu > 0 && options.webgpu !== false ? await WebGpu.create().catch(() => null) : null
  const engine = new Engine(options.canvas ?? new OffscreenCanvas(1, 1), {
    ...(webgpu && { gpu: webgpu }),
    batchSize: TUNING.promptBatch > 0 ? TUNING.promptBatch : PROMPT_BATCH
  })
  // A guess, fast: the load's own timed choices for prompts and the head
  // (the prompt packing, the head's layout) skipped, their defaults taken;
  // the writing matvec's too where writing isn't timed (its choice is the
  // writing speed: skipped, an M3 Ultra's was guessed at a third).
  engine.tuned.f16Pack = TUNING.f16Pack >= 0 ? TUNING.f16Pack & 3 : 3
  engine.tuned.f16Matmul = null
  if (!decode) engine.tuned.streamMatvec = false
  Reflect.set(engine, "headChoice", { expanded: false, split: TUNING.q6HeadSplit })
  let peak = 0
  const sample = () => (peak = Math.max(peak, engine.gpu.bytes))
  onEngine?.(engine)
  try {
    if (WARM_MS > 0) await warmGpu(engine.gpu)
    const perLayerSource = probePerLayerSource(shape)
    if (perLayerSource) engine.perLayerSource = perLayerSource
    await engine.load(probePieces(shape), sample, signal)
    sample()
    const loadMs = performance.now() - started
    // As the worker leaves the GPU's pacing after a load.
    engine.gpu.resetPacing(true)
    engine.gpu.setPhase("load")
    engine.gpu.allowPacingRecovery()
    let mark = performance.now()
    const lap = () => {
      const ms = performance.now() - mark
      mark = performance.now()
      return ms
    }
    const inner = engine as unknown as Internals
    const loaded = inner.layers.slice()
    const byKind = new Map(shape.kinds.map((k, i) => [k.type, loaded[i]]))
    const types = shape.config.layerTypes
    const first = byKind.get(types[0])!
    const last = byKind.get(types[types.length - 1])!
    const use = (layers: Array<unknown>) => {
      inner.layers = layers
      inner.work = null
    }
    const vocab = engine.config.vocabSize
    const tokens = Array.from({ length: PROMPT_TOKENS }, (_, i) => (i * 7919 + 13) % vocab)

    // Prompts, `count` back to back (the cache reset before each), waited
    // for once: the GPU kept busy as through a whole prompt, its clock up
    // and no readback's wait in each. The first one's reset is before the
    // timing, not in it: a model with recurrent states zeroes them all
    // there, a fifth of what an M3 Ultra took for Qwen3.5 2B's and Raon-1
    // nano's stacks, and no part of reading a prompt.
    const prompts = async (layers: Array<unknown>, count: number, rows = tokens) => {
      signal?.throwIfAborted()
      use(layers)
      engine.reset()
      await engine.gpu.finish()
      const t = performance.now()
      for (let i = 0; i < count; i++) {
        if (i > 0) engine.reset()
        // (Every row through every layer: the last layer's last-row-only
        // tail, a set of programs of its own, left out; the stack's last
        // layer then costs what a middle one does.)
        await engine.forwardHidden(rows, { fullRows: true })
      }
      await engine.gpu.finish()
      sample()
      return (performance.now() - t) / count
    }
    // Base: the first and last layers (their own variants: the embedding
    // folded into the first, the last row's tail in the last); and the
    // same with up to PROMPT_STACK of the middle layers between them, the
    // model's kinds in proportion (each kind's one layer standing in for
    // all of its kind). Their difference, scaled to the model's middle
    // layers, is what those take: a few layers' worth, so the timings'
    // noise (a few ms on an M3) is small beside it.
    const base = types.length > 1 ? [first, last] : [first]
    const middle = middleStack(types, PROMPT_STACK)
    const stack = [first, ...middle.map((t) => byKind.get(t)!), last]
    // A fast guess: the stack once untimed (programs compile; as short a
    // prompt as takes the same kernels, see TUNING.f16MinRows), then once
    // timed, every layer taken to cost what the stack's do on average (the
    // base, the first and last layers' own variants, not timed apart: they
    // differ by little, and each timing cost a Pixel 10 Pro's Firefox a
    // tenth of a second).
    const timedLayers = middle.length ? stack : base
    // (Compiled through the base where its first and last layers are of
    // every kind the stack has: their programs are the stack's, at half the
    // work; a Pixel 10 Pro's Firefox spent a quarter second more on the
    // stack's.)
    const baseKinds = new Set([types[0], types[types.length - 1]])
    const compiles = middle.every((t) => baseKinds.has(t)) ? base : timedLayers
    await prompts(compiles, 1, tokens.slice(0, Math.min(tokens.length, Math.max(SPLIT_ROWS + 1, TUNING.f16MinRows))))
    const promptStack = await prompts(timedLayers, 1)
    const promptBase = (promptStack * base.length) / timedLayers.length
    const promptPhase = lap()
    if (!decode)
      return {
        times: { promptBase, promptStack, stackLayers: middle.length, decodeLayers: NaN, headAt: 0, headPerRow: 0, vocab, promptTokens: PROMPT_TOKENS },
        measureMs: performance.now() - started,
        loadMs,
        loadParts: Object.fromEntries(Object.entries(Reflect.get(engine, "loadMs") as Record<string, number>).map(([k, v]) => [k, Math.round(v)])),
        programs: engine.gpu.compiled.count,
        peakBytes: peak,
        phases: { prompt: promptPhase, decode: 0, head: 0 }
      }

    // Writing: one-token forwards after the prompt, queued, then waited for
    // once (the worker reads each token forward while the last one's pick
    // goes back: its steps follow each other on the GPU like these),
    // through the base and the stack, the middle layers scaled from their
    // difference as the prompt's: a forward through every layer cost a
    // Pixel 10 Pro's Firefox 140 ms a token.
    const forwards = async (layers: Array<unknown>, count: number) => {
      signal?.throwIfAborted()
      use(layers)
      engine.reset()
      engine.position = PROMPT_TOKENS
      await engine.gpu.finish()
      const t = performance.now()
      for (let i = 0; i < count; i++) await engine.forwardNoLogits([tokens[i]])
      await engine.gpu.finish()
      sample()
      return (performance.now() - t) / count
    }
    // (Through every layer, six queued: a forward through a few, scaled,
    // guessed an M3 Ultra's writing at a third of its speed and Firefox's
    // at a third to a half. Prompt-only predictions skip this: see
    // PredictSpeedOptions.decode.)
    const all = types.map((t) => byKind.get(t)!)
    await forwards(all, 1)
    const decodeLayers = await forwards(all, 6)
    const decodePhase = lap()

    // The head over the probe's rows and over half of them (as chooseHead
    // times a smaller head), queued after a one-token forward.
    const half = Math.max(4, (vocab / 2) & ~3)
    const heads = async (rows: number, count: number) => {
      signal?.throwIfAborted()
      engine.config.vocabSize = rows
      await engine.gpu.finish()
      const t = performance.now()
      for (let i = 0; i < count; i++) await inner.head(0, 1, false)
      engine.gpu.flush()
      await engine.gpu.finish()
      return (performance.now() - t) / count
    }
    use(base)
    engine.position = PROMPT_TOKENS
    await engine.forwardNoLogits([tokens[0]])
    const at = { full: Infinity, half: Infinity }
    // The GPU's own time: no pacing slices (each a fence's round trip).
    engine.gpu.configurePacing(0, TUNING.submissionMs)
    try {
      // (Two sizes: a row's share from their difference; the head's time over
      // its rows, mostly fixed costs at these sizes, guessed an M3 Ultra's
      // writing at a third.)
      await heads(vocab, 1)
      await heads(half, 1)
      at.full = await heads(vocab, 4)
      at.half = await heads(half, 4)
    } finally {
      engine.config.vocabSize = vocab
      engine.gpu.configurePacing(engine.tuned.submissionReads ?? TUNING.submissionReads, TUNING.submissionMs)
    }
    const headPerRow = Math.max(0, (at.full - at.half) / (vocab - half))
    return {
      times: { promptBase, promptStack, stackLayers: middle.length, decodeLayers, headAt: at.full, headPerRow, vocab, promptTokens: PROMPT_TOKENS },
      measureMs: performance.now() - started,
      loadMs,
      loadParts: Object.fromEntries(Object.entries(Reflect.get(engine, "loadMs") as Record<string, number>).map(([k, v]) => [k, Math.round(v)])),
      programs: engine.gpu.compiled.count,
      peakBytes: peak,
      phases: { prompt: promptPhase, decode: decodePhase, head: lap() }
    }
  } finally {
    onEngine?.(null)
    engine.gpu.dispose()
  }
}
