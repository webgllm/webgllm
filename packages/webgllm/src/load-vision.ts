/**
 * Loads a Qwen3.5 vision encoder from its ONNX export
 * (onnx-community/Qwen3.5-*-ONNX's onnx/vision_encoder_q4f16.onnx + its
 * .onnx_data): a ViT over 16-pixel patches (two frames of each, the image
 * twice), a learned 48 × 48 position table interpolated to the image's
 * grid, `depth` pre-norm blocks with 2D RoPE, and a merger that folds each
 * 2 × 2 block of patches into one token of the text model's width (see
 * docs/vision.md).
 *
 * Weights are found by the graph's structure, as load-onnx.ts finds the
 * text model's: a projection is the MatMulNBits node under its module path
 * (`/model/layers.0/attn/qkv/`), its bias the Add after it, a norm the
 * LayerNormalization its input comes from.
 */
import { halvesToFloats } from "./half.js"
import { checkShape, OnnxModel, tensorBytes, type NBits } from "./load-onnx.js"
import { OnnxDataType, parseOnnxModel, type OnnxNode, type OnnxTensor } from "./onnx.js"
import { ReadPlan } from "./read-plan.js"
import { packMatMulNBitsScales } from "./repack.js"
import type { ByteSource, QuantMatrix } from "./weights.js"

export type VisionConfig = {
  /** The ViT's width (0.8B: 768, 2B: 1024). */
  width: number
  heads: number
  /** Dims per head (64); RoPE rotates all of them. */
  headDim: number
  /** The MLP's inner width. */
  ffn: number
  depth: number
  /** The merger's output: the text model's hidden size. */
  outWidth: number
  /** Pixels per patch side (16). */
  patchSize: number
  /** Frames per patch (2: a still image is repeated). */
  temporal: number
  /** Patches per merged token side (2). */
  merge: number
  /** Values per patch row of pixel_values: 3 · temporal · patchSize². */
  patchValues: number
  /** The position table's side (48: 2304 rows). */
  grid: number
  /** LayerNorm's epsilon. */
  eps: number
  /** The blocks' MLP activation, and the merger's. */
  act: "tanh" | "erf"
  mergerAct: "tanh" | "erf"
}

/** A projection: 4-bit rows as MatMulNBits stores them (QuantMatrix.raw), and its bias. */
export type VisionLinear = { m: QuantMatrix; bias: Float32Array }
export type VisionNorm = { weight: Float32Array; bias: Float32Array }

export type VisionBlock = {
  norm1: VisionNorm
  qkv: VisionLinear
  proj: VisionLinear
  norm2: VisionNorm
  fc1: VisionLinear
  fc2: VisionLinear
}

export type VisionPiece =
  | { kind: "config"; config: VisionConfig }
  | {
      kind: "patch"
      patch: VisionLinear
      /** The position table, dequantized: grid² rows of `width`. */
      positions: Float32Array
      /** RoPE's 16 frequencies (headDim / 4): rows use 0..15, columns the same again. */
      invFreq: Float32Array
    }
  | { kind: "block"; index: number; value: VisionBlock }
  | { kind: "merger"; norm: VisionNorm; linear1: VisionLinear; linear2: VisionLinear }

export type LoadVisionOptions = {
  /** vision_encoder*.onnx, as bytes or a source. */
  graph: ByteSource | Uint8Array
  /** Its external data file. */
  data: ByteSource
}

type Read = { key: string; tensor: OnnxTensor }

/** The encoder's graph, read: its config and every tensor the load takes, in order. */
export function planVision(graphBytes: Uint8Array) {
  const model = new OnnxModel(parseOnnxModel(graphBytes))
  const nodes = model.graph.nodes
  const reads: Array<Read> = []
  const want = (t: OnnxTensor | null) => {
    if (t) reads.push({ key: t.name, tensor: t })
  }

  /** The node behind `name`, past Casts and Reshapes. */
  const source = (name: string): OnnxNode | undefined => {
    let node = model.producerOf(name)
    while (node && (node.opType === "Cast" || node.opType === "Reshape")) node = model.producerOf(node.inputs[0])
    return node
  }
  const linear = (path: string) => {
    const w = model.nbits(path)
    const add = model.consumersOf(w.node.outputs[0]).find((n) => n.opType === "Add")
    const bias = add?.inputs.map((i) => (model.has(i) ? model.init(i) : null)).find((t) => t !== null)
    if (!add || !bias) throw new Error(`ONNX: ${path} has no bias`)
    checkShape(bias, bias.dataType, w.N)
    return { w, bias, out: add.outputs[0] }
  }
  const normBefore = (w: NBits) => {
    const ln = source(w.node.inputs[0])
    if (ln?.opType !== "LayerNormalization") throw new Error(`ONNX: no LayerNorm before ${w.name}`)
    const weight = model.init(ln.inputs[1])
    const bias = model.init(ln.inputs[2])
    return { ln, weight, bias }
  }
  const activation = (after: string): "tanh" | "erf" => {
    const act = model.consumersOf(after).find((n) => n.opType === "Gelu" || n.opType === "FastGelu")
    if (!act) throw new Error(`ONNX: no GELU after ${after}`)
    return act.opType === "FastGelu" || act.attributes.get("approximate")?.s === "tanh" ? "tanh" : "erf"
  }

  const patch = linear("/model/embeddings/patch_embed/")
  const table = nodes.find((n) => n.opType === "GatherBlockQuantized")
  if (!table) throw new Error("ONNX: no position table (GatherBlockQuantized)")
  const attr = (n: OnnxNode, key: string) => n.attributes.get(key)?.i
  if (attr(table, "bits") !== 4 || attr(table, "block_size") !== 32 || (attr(table, "quantize_axis") ?? 1) !== 1)
    throw new Error("ONNX: the position table isn't 4-bit, blocks of 32 along its rows")
  const tableQuant = model.init(table.inputs[0])
  const tableScales = model.init(table.inputs[2])
  const tableZp = table.inputs[3] ? model.init(table.inputs[3]) : null
  const width = patch.w.N
  const rows = tableQuant.dims[0]
  const grid = Math.round(Math.sqrt(rows))
  if (grid * grid !== rows || tableQuant.dims[1] * 2 !== width)
    throw new Error(`ONNX: the position table is ${tableQuant.dims.join("×")}, expected a square grid of ${width}-wide rows`)

  // RoPE: the first block's cos table comes from a MatMul of positions by
  // the frequencies (an initializer).
  const rope = nodes.find((n) => n.opType === "RotaryEmbedding")
  if (!rope) throw new Error("ONNX: no RotaryEmbedding")
  if (attr(rope, "interleaved") === 1) throw new Error("ONNX: interleaved RoPE in the encoder")
  const freqNode = nodes.find(
    (n) => n.opType === "MatMul" && model.has(n.inputs[1]) && model.init(n.inputs[1]).dims.length === 2 && model.init(n.inputs[1]).dims[0] === 1
  )
  if (!freqNode) throw new Error("ONNX: no RoPE frequencies")
  const invFreqT = model.init(freqNode.inputs[1])

  const blocks: Array<{
    norm1: ReturnType<typeof normBefore>
    qkv: ReturnType<typeof linear>
    proj: ReturnType<typeof linear>
    norm2: ReturnType<typeof normBefore>
    fc1: ReturnType<typeof linear>
    fc2: ReturnType<typeof linear>
  }> = []
  for (let i = 0; model.hasNBits(`/model/layers.${i}/attn/qkv/`); i++) {
    const p = `/model/layers.${i}/`
    const qkv = linear(`${p}attn/qkv/`)
    const proj = linear(`${p}attn/proj/`)
    const fc1 = linear(`${p}mlp/linear_fc1/`)
    const fc2 = linear(`${p}mlp/linear_fc2/`)
    blocks.push({ norm1: normBefore(qkv.w), qkv, proj, norm2: normBefore(fc1.w), fc1, fc2 })
  }
  if (!blocks.length) throw new Error("ONNX: no encoder blocks")
  const linear1 = linear("/model/merger/linear1/")
  const linear2 = linear("/model/merger/linear2/")
  const mergerNorm = normBefore(linear1.w)
  const mha = nodes.find((n) => n.opType === "MultiHeadAttention")
  const heads = mha ? attr(mha, "num_heads") : attr(rope, "num_heads")
  if (!heads) throw new Error("ONNX: the encoder's attention has no head count")
  const headDim = width / heads
  // The shaders lay a head out as 16 texels (gpu/vision-kernels.ts).
  if (headDim !== 64) throw new Error(`ONNX: the encoder's heads are ${headDim} wide; webgllm's encoder runs 64-wide heads`)
  const rotary = attr(rope, "rotary_embedding_dim") ?? headDim
  if (rotary !== headDim || invFreqT.dims[1] * 4 !== headDim)
    throw new Error(`ONNX: the encoder's RoPE rotates ${rotary} of ${headDim} dims`)
  const merge = Math.round(Math.sqrt(linear1.w.K / width))
  if (merge * merge * width !== linear1.w.K) throw new Error("ONNX: the merger's input isn't a square block of patches")
  const pixels = patch.w.K / 3
  // temporal · patch²: 2 frames of 16 × 16 (the only build there is).
  const patchSize = 16
  const temporal = pixels / (patchSize * patchSize)
  if (!Number.isInteger(temporal)) throw new Error(`ONNX: a patch of ${patch.w.K} values isn't 3 × t × 16 × 16`)
  for (const b of blocks)
    if (b.qkv.w.N !== 3 * width || b.qkv.w.K !== width || b.proj.w.N !== width || b.fc2.w.N !== width || b.fc1.w.N !== b.fc2.w.K)
      throw new Error("ONNX: the encoder's blocks don't share one shape")

  const config: VisionConfig = {
    width,
    heads,
    headDim,
    ffn: blocks[0].fc1.w.N,
    depth: blocks.length,
    outWidth: linear2.w.N,
    patchSize,
    temporal,
    merge,
    patchValues: patch.w.K,
    grid,
    eps: blocks[0].norm1.ln.attributes.get("epsilon")?.f ?? 1e-6,
    act: activation(blocks[0].fc1.out),
    mergerAct: activation(linear1.out)
  }

  const wantLinear = (l: ReturnType<typeof linear>) => {
    want(l.w.quant)
    want(l.w.scales)
    want(l.w.zp)
    want(l.bias)
  }
  const wantNorm = (n: ReturnType<typeof normBefore>) => {
    want(n.weight)
    want(n.bias)
  }
  wantLinear(patch)
  want(tableQuant)
  want(tableScales)
  want(tableZp)
  want(invFreqT)
  for (const b of blocks) {
    wantNorm(b.norm1)
    wantLinear(b.qkv)
    wantLinear(b.proj)
    wantNorm(b.norm2)
    wantLinear(b.fc1)
    wantLinear(b.fc2)
  }
  wantNorm(mergerNorm)
  wantLinear(linear1)
  wantLinear(linear2)
  return {
    config,
    reads,
    patch,
    table: { quant: tableQuant, scales: tableScales, zp: tableZp },
    invFreq: invFreqT,
    blocks,
    merger: { norm: mergerNorm, linear1, linear2 }
  }
}

/** A tensor's values as float32 (float16 or float). */
function toFloats(t: OnnxTensor, bytes: Uint8Array): Float32Array {
  if (t.dataType === OnnxDataType.FLOAT16) return halvesToFloats(bytes)
  if (t.dataType === OnnxDataType.FLOAT) return new Float32Array(bytes.slice().buffer)
  throw new Error(`ONNX: ${t.name} has data type ${t.dataType}, expected float16 or float`)
}

/**
 * The position table dequantized: element e of a row is nibble e & 1 of
 * byte e >> 1 (low first), minus its block's zero point (packed the same
 * way; 8 without one), times the block's scale.
 */
export function dequantizeTable(
  quant: Uint8Array,
  scales: Float32Array,
  zp: Uint8Array | null,
  rows: number,
  width: number
): Float32Array {
  const blocks = width / 32
  const zpStride = Math.ceil(blocks / 2)
  const out = new Float32Array(rows * width)
  for (let r = 0; r < rows; r++) {
    for (let e = 0; e < width; e++) {
      const byte = quant[r * (width / 2) + (e >> 1)]
      const q = e & 1 ? byte >> 4 : byte & 15
      const b = e >> 5
      const z = zp ? (zp[r * zpStride + (b >> 1)] >> ((b & 1) << 2)) & 15 : 8
      out[r * width + e] = (q - z) * scales[r * blocks + b]
    }
  }
  return out
}

/** Yields the encoder's pieces: config, patch embedding and positions, each block, the merger. */
export async function* loadVision(options: LoadVisionOptions): AsyncGenerator<VisionPiece, void, undefined> {
  const graphBytes = options.graph instanceof Uint8Array ? options.graph : await options.graph.read(0, options.graph.size)
  const plan = planVision(graphBytes)
  const reads = new ReadPlan()
  for (const { key, tensor } of plan.reads) {
    const length = tensorBytes(tensor)
    if (tensor.raw) {
      if (tensor.raw.length !== length) throw new Error(`ONNX: ${key} has ${tensor.raw.length} bytes, expected ${length}`)
      reads.add({ key, source: options.data, offset: 0, length, bytes: tensor.raw })
      continue
    }
    if (!tensor.external || tensor.external.length !== length) throw new Error(`ONNX: ${key} isn't in the external data file`)
    reads.add({ key, source: options.data, offset: tensor.external.offset, length })
  }
  const floats = async (t: OnnxTensor) => toFloats(t, await reads.take(t.name))
  const linear = async (l: { w: NBits; bias: OnnxTensor }): Promise<VisionLinear> => {
    const { w } = l
    const blocks = w.K / 32
    const m: QuantMatrix = {
      N: w.N,
      K: w.K,
      bits: 4,
      quants: new Uint32Array(0),
      scales: new Uint16Array(w.N * blocks),
      zeros: w.zp ? new Uint8Array(w.N * blocks).fill(8) : null,
      raw: (await reads.take(w.quant.name)).slice()
    }
    const scales = await reads.take(w.scales.name)
    const zp = w.zp ? await reads.take(w.zp.name) : null
    packMatMulNBitsScales(m, 0, scales, zp, w.N)
    return { m, bias: await floats(l.bias) }
  }
  const norm = async (n: { weight: OnnxTensor; bias: OnnxTensor }): Promise<VisionNorm> => ({
    weight: await floats(n.weight),
    bias: await floats(n.bias)
  })
  const { config } = plan
  yield { kind: "config", config }
  const patch = await linear(plan.patch)
  const quant = await reads.take(plan.table.quant.name)
  const tableScales = await floats(plan.table.scales)
  const zp = plan.table.zp ? await reads.take(plan.table.zp.name) : null
  const positions = dequantizeTable(quant, tableScales, zp, config.grid * config.grid, config.width)
  const invFreq = await floats(plan.invFreq)
  yield { kind: "patch", patch, positions, invFreq }
  for (const [index, b] of plan.blocks.entries()) {
    const value: VisionBlock = {
      norm1: await norm(b.norm1),
      qkv: await linear(b.qkv),
      proj: await linear(b.proj),
      norm2: await norm(b.norm2),
      fc1: await linear(b.fc1),
      fc2: await linear(b.fc2)
    }
    yield { kind: "block", index, value }
  }
  yield {
    kind: "merger",
    norm: await norm(plan.merger.norm),
    linear1: await linear(plan.merger.linear1),
    linear2: await linear(plan.merger.linear2)
  }
}

