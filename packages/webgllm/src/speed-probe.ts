/**
 * What predictSpeed measures (see predict-speed.ts): a model's shapes from
 * its GGUF header or its ONNX graph and config (never the weights), one
 * layer of each kind it has with random weights in the storage the real
 * loader gives them, and the engine's own prompt and writing passes over
 * them, timed with fences.
 */
import { Engine, TUNING } from "./engine.js"
import { GL, type Gpu } from "./gpu/gl.js"
import type { WebGpu } from "./gpu/webgpu.js"
import { GgmlType, readGgufHeader, type GgufHeader } from "./gguf.js"
import { bytesToFloats, halvesToFloats } from "./half.js"
import { PROMPT_BATCH } from "./handler.js"
import { bitsOf, ggufConfig } from "./load-gguf.js"
import { OnnxModel, planOnnx, type NBits } from "./load-onnx.js"
import { parseOnnxModel } from "./onnx.js"
import { source } from "./model-files.js"
import { fileUrl, type ModelFile, type WebGlModelMeta } from "./model-meta.js"
import { storedFile } from "./storage.js"
import { newQuantMatrix } from "./repack.js"
import type { LayerType, LayerWeights, ModelConfig, QuantMatrix, WeightPiece } from "./weights.js"

/** How a matrix is stored: as the loader would hand it to the engine. */
export type MatrixStorage = {
  bits: 4 | 6 | 8
  zeros: boolean
  /** 4-bit rows as MatMulNBits stores them (QuantMatrix.raw: ONNX). */
  raw: boolean
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
}

/** Prompt tokens predictSpeed times (one prompt batch). */
export const PROMPT_TOKENS = 512

/** Most middle layers the probe's prompts run (see middleStack). */
export const PROMPT_STACK = 6

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
export const PROBE_VOCAB = 65536

/** The engine's per-matrix storage for a GGUF tensor type. */
const ggufMatrix = (header: GgufHeader, name: string): MatrixStorage => {
  const info = header.tensors.get(name)
  if (!info) throw new Error(`GGUF: missing tensor ${name}`)
  return { bits: bitsOf(info), zeros: false, raw: false }
}

/** A GGUF's shape from its header (a range request or two) and its rope_freqs (a few hundred bytes). */
export async function ggufSpeedShape(read: { size: number; read: (offset: number, length: number) => Promise<Uint8Array> }): Promise<SpeedShape> {
  const header = await readGgufHeader(read)
  const freqs = header.tensors.get("rope_freqs.weight")
  let factors: Float32Array | undefined
  if (freqs) {
    const bytes = await read.read(freqs.offset, freqs.byteLength)
    // (As the loader reads it: F32 or F16.)
    factors = freqs.type === GgmlType.F16 ? halvesToFloats(bytes) : bytesToFloats(bytes)
  }
  const config = ggufConfig(header, factors)
  const qwen = !!config.linear
  const kinds: SpeedShape["kinds"] = []
  config.layerTypes.forEach((type, i) => {
    if (kinds.some((k) => k.type === type)) return
    const blk = `blk.${i}.`
    const m = (name: string) => ggufMatrix(header, `${blk}${name}.weight`)
    const matrices: KindStorage = { gateUp: m("ffn_gate"), down: m("ffn_down") }
    if (type === "linear_attention") {
      matrices.inProj = m("attn_qkv")
      matrices.ba = m("ssm_beta")
      matrices.outProj = m("ssm_out")
    } else if (type === "conv" && !qwen) {
      matrices.inProj = m("shortconv.in_proj")
      matrices.outProj = m("shortconv.out_proj")
    } else {
      matrices.qkvg = m("attn_q")
      matrices.o = m("attn_output")
    }
    kinds.push({ type, matrices })
  })
  const embedding = header.tensors.get("token_embd.weight")
  if (!embedding) throw new Error("GGUF: missing tensor token_embd.weight")
  const bits = embedding.type === GgmlType.Q6_K ? 6 : embedding.type === GgmlType.Q4_0 ? 4 : 8
  return { config, kinds, embedding: { bits, zeros: false, raw: false } }
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
    zeros: storage.bits === 4 && storage.zeros ? noise.zeros(N * blocks) : null
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
    vocabSize: vocab
  }
}

/** The probe model's pieces, as a loader yields them. */
async function* probePieces(shape: SpeedShape, noise: Noise): AsyncGenerator<WeightPiece> {
  const config = probeConfig(shape)
  yield { kind: "config", config }
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
  /** Of which the synthetic load (the engine's own load-time choices included). */
  loadMs: number
  /** The most texture bytes the probe held. */
  peakBytes: number
  /** Milliseconds in each part of the timings: prompts, writing, the head. */
  phases: { prompt: number; decode: number; head: number }
}

/** Milliseconds of GPU work before the probe's load (see warmGpu). */
export const WARM_MS = 150

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
  } = {}
): Promise<Omit<ProbeReport, "shapeMs">> {
  const { signal, onEngine } = options
  const started = performance.now()
  // As the worker makes its engine (handler.ts load): the same batch, the
  // model's own context length (the cache starts small and grows).
  const engine = new Engine(options.canvas ?? new OffscreenCanvas(1, 1), {
    batchSize: TUNING.promptBatch > 0 ? TUNING.promptBatch : PROMPT_BATCH
  })
  let peak = 0
  const sample = () => (peak = Math.max(peak, engine.gpu.bytes))
  onEngine?.(engine)
  try {
    await warmGpu(engine.gpu)
    await engine.load(probePieces(shape, new Noise()), sample, signal)
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
    // and no readback's wait in each.
    const prompts = async (layers: Array<unknown>, count: number) => {
      signal?.throwIfAborted()
      use(layers)
      await engine.gpu.finish()
      const t = performance.now()
      for (let i = 0; i < count; i++) {
        engine.reset()
        await engine.forwardNoLogits(tokens)
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
    // Once each untimed (programs compile), then rounds in alternating
    // order, the best of each.
    await prompts(base, 1)
    if (middle.length) await prompts(stack, 1)
    let promptBase = Infinity
    let promptStack = Infinity
    for (let round = 0; round < 3; round++) {
      const order = round % 2 ? ["stack", "base"] : ["base", "stack"]
      for (const which of order) {
        if (which === "base") promptBase = Math.min(promptBase, await prompts(base, 1))
        else if (middle.length) promptStack = Math.min(promptStack, await prompts(stack, 1))
      }
    }
    if (!middle.length) promptStack = promptBase
    const promptPhase = lap()

    // Writing: one-token forwards through every layer after the prompt,
    // queued, then waited for once. (The worker reads each token forward
    // while the last one's pick goes back: its steps follow each other on
    // the GPU like these.)
    const all = types.map((t) => byKind.get(t)!)
    use(all)
    engine.reset()
    const forwards = async (count: number) => {
      signal?.throwIfAborted()
      engine.position = PROMPT_TOKENS
      await engine.gpu.finish()
      const t = performance.now()
      for (let i = 0; i < count; i++) await engine.forwardNoLogits([tokens[i]])
      await engine.gpu.finish()
      sample()
      return (performance.now() - t) / count
    }
    await forwards(2)
    let decodeLayers = Infinity
    for (let round = 0; round < 2; round++) decodeLayers = Math.min(decodeLayers, await forwards(6))
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
      await heads(vocab, 2)
      await heads(half, 2)
      for (let round = 0; round < 3; round++)
        for (const which of round % 2 ? (["half", "full"] as const) : (["full", "half"] as const))
          at[which] = Math.min(at[which], await heads(which === "full" ? vocab : half, 8))
    } finally {
      engine.config.vocabSize = vocab
      engine.gpu.configurePacing(engine.tuned.submissionReads ?? TUNING.submissionReads, TUNING.submissionMs)
    }
    const headPerRow = Math.max(0, (at.full - at.half) / (vocab - half))
    return {
      times: { promptBase, promptStack, stackLayers: middle.length, decodeLayers, headAt: at.full, headPerRow, vocab, promptTokens: PROMPT_TOKENS },
      measureMs: performance.now() - started,
      loadMs,
      peakBytes: peak,
      phases: { prompt: promptPhase, decode: decodePhase, head: lap() }
    }
  } finally {
    onEngine?.(null)
    engine.gpu.dispose()
  }
}
