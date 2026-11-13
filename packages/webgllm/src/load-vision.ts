/**
 * Loads a vision encoder from its ONNX export (the graph and its one
 * external data file) as the pieces vision.ts runs (vision-weights.ts).
 * Three graphs are read, told apart by their modules:
 *
 * - Qwen3.5's (onnx-community/Qwen3.5-*-ONNX's vision_encoder_q4f16): a
 *   ViT over 16-pixel patches (two frames of each, the image twice), a
 *   learned 48 × 48 position table interpolated to the image's grid,
 *   pre-norm blocks with a fused q|k|v projection and 2D RoPE, and a
 *   merger that folds each 2 × 2 block of patches into one token.
 * - LFM2-VL's (LiquidAI/LFM2.5-VL-*-ONNX's vision_encoder or
 *   embed_images): SigLIP2 NaFlex. Patches of 16 × 16 pixels, a learned
 *   16 × 16 position table resized to the image's grid, pre-norm blocks
 *   with separate q, k and v projections and a tanh-GELU MLP, a final
 *   LayerNorm, then the projector: each 2 × 2 block of patches as one row,
 *   two projections with an exact GELU between.
 * - LFM3-VL's (LiquidAI/LFM3-VL-*-ONNX's vision_encoder): a patch
 *   projection and an RMS norm, then RMS-norm blocks of either a gated
 *   3 × 3 depthwise convolution over the patch grid or grouped-query
 *   attention with 2D RoPE and per-head q and k norms, each with a SwiGLU
 *   MLP; and a projector like LFM2-VL's behind a LayerNorm.
 *
 * Weights are found by the graph's structure, as load-onnx.ts finds the
 * text model's: a projection is the MatMulNBits node under its module path
 * (`/model/layers.0/attn/qkv/`), its bias the Add after it, a norm the
 * normalization its input comes from.
 */
import { halvesToFloats } from "./half.js"
import { checkShape, OnnxModel, tensorBytes, type WideNBits } from "./load-onnx.js"
import { OnnxDataType, parseOnnxModel, type OnnxNode, type OnnxTensor } from "./onnx.js"
import { ReadPlan } from "./read-plan.js"
import {
  checkVisionConfig,
  matMulNBitsMatrix,
  stackLinears,
  type VisionBlock,
  type VisionConfig,
  type VisionLinear,
  type VisionNorm,
  type VisionPiece
} from "./vision-weights.js"
import type { ByteSource } from "./weights.js"

export type { VisionBlock, VisionConfig, VisionLinear, VisionNorm, VisionPiece } from "./vision-weights.js"

export type LoadVisionOptions = {
  /** The encoder's .onnx file, as bytes or a source. */
  graph: ByteSource | Uint8Array
  /** Its external data file. */
  data: ByteSource
}

type Read = { key: string; tensor: OnnxTensor }

/** A projection in the graph: its weight and the bias added after it, if any. */
type LinearRef = { w: WideNBits; bias: OnnxTensor | null; out: string }
type NormRef = { weight: OnnxTensor; bias: OnnxTensor | null }
type BlockRef = { norm1: NormRef; norm2: NormRef; fc1: Array<LinearRef>; fc2: LinearRef } & (
  | { type: "attention"; qkv: Array<LinearRef>; qNorm?: OnnxTensor; kNorm?: OnnxTensor; proj: LinearRef }
  | { type: "conv"; inProj: LinearRef; kernel: OnnxTensor; outProj: LinearRef }
)

/** What a graph's plan found: the config and where each piece's tensors are. */
type VisionPlan = {
  config: VisionConfig
  patch: LinearRef
  patchNorm?: NormRef
  /** The position table: 4-bit rows (Qwen3.5), or floats, [1, width, grid, grid]. */
  table?: { kind: "q4"; quant: OnnxTensor; scales: OnnxTensor; zp: OnnxTensor | null } | { kind: "float"; values: OnnxTensor }
  /** RoPE's frequencies as a tensor, or the sin table they are read from (row 1 is sin of each). */
  invFreq?: { kind: "values"; tensor: OnnxTensor } | { kind: "sin"; tensor: OnnxTensor }
  blocks: Array<BlockRef>
  merger: { norm: NormRef; linear1: LinearRef; linear2: LinearRef }
}

/** Lookups the three plans share. */
class VisionGraph {
  readonly model: OnnxModel
  constructor(graphBytes: Uint8Array) {
    this.model = new OnnxModel(parseOnnxModel(graphBytes))
  }

  get nodes() {
    return this.model.graph.nodes
  }

  /** The node behind `name`, past Casts, Reshapes and Transposes (the merger's regrouping of patches). */
  source(name: string): OnnxNode | undefined {
    let node = this.model.producerOf(name)
    while (node && (node.opType === "Cast" || node.opType === "Reshape" || node.opType === "Transpose")) node = this.model.producerOf(node.inputs[0])
    return node
  }

  /** The projection under `path`: `bias` says whether an Add of a constant must follow it. */
  linear(path: string, bias: boolean): LinearRef {
    const w = this.model.wideNBits(path)
    const add = this.model.consumersOf(w.node.outputs[0]).find((n) => n.opType === "Add" && n.inputs.some((i) => this.model.has(i)))
    const tensor = add?.inputs.map((i) => (this.model.has(i) ? this.model.init(i) : null)).find((t) => t !== null) ?? null
    if (bias !== (tensor !== null)) throw new Error(`ONNX: ${path} has ${tensor ? "a" : "no"} bias`)
    if (tensor) checkShape(tensor, tensor.dataType, w.N)
    return { w, bias: tensor, out: add && tensor ? add.outputs[0] : w.node.outputs[0] }
  }

  /** The norm a projection's input comes from: a LayerNormalization (weight and bias) or an RMS norm (weight). */
  normBefore(w: WideNBits, op: "LayerNormalization" | "SimplifiedLayerNormalization"): NormRef & { node: OnnxNode } {
    const node = this.source(w.node.inputs[0])
    if (node?.opType !== op) throw new Error(`ONNX: no ${op} before ${w.name}`)
    return { node, weight: this.model.init(node.inputs[1]), bias: op === "LayerNormalization" ? this.model.init(node.inputs[2]) : null }
  }

  /** The GELU after a projection: by tanh, or exact. */
  gelu(after: string): "tanh" | "erf" {
    const act = this.model.consumersOf(after).find((n) => n.opType === "Gelu" || n.opType === "FastGelu")
    if (!act) throw new Error(`ONNX: no GELU after ${after}`)
    return act.opType === "FastGelu" || act.attributes.get("approximate")?.s === "tanh" ? "tanh" : "erf"
  }

  attr(n: OnnxNode, key: string) {
    return n.attributes.get(key)?.i
  }

  /** The graph's attention heads (every attention node agrees). */
  heads(): number {
    const mha = this.nodes.find((n) => n.opType === "MultiHeadAttention")
    const heads = mha && this.attr(mha, "num_heads")
    if (!heads) throw new Error("ONNX: the encoder's attention has no head count")
    return heads
  }

  /** The side of a merger's blocks of patches, from its first projection's input. */
  merge(linear1: LinearRef, width: number): number {
    const merge = Math.round(Math.sqrt(linear1.w.K / width))
    if (merge * merge * width !== linear1.w.K) throw new Error("ONNX: the merger's input isn't a square block of patches")
    return merge
  }
}

/** Qwen3.5's encoder (see the header). */
function planQwen(g: VisionGraph): VisionPlan {
  const { model } = g
  const patch = g.linear("/model/embeddings/patch_embed/", true)
  const table = g.nodes.find((n) => n.opType === "GatherBlockQuantized")
  if (!table) throw new Error("ONNX: no position table (GatherBlockQuantized)")
  if (g.attr(table, "bits") !== 4 || g.attr(table, "block_size") !== 32 || (g.attr(table, "quantize_axis") ?? 1) !== 1)
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
  const rope = g.nodes.find((n) => n.opType === "RotaryEmbedding")
  if (!rope) throw new Error("ONNX: no RotaryEmbedding")
  if (g.attr(rope, "interleaved") === 1) throw new Error("ONNX: interleaved RoPE in the encoder")
  const freqNode = g.nodes.find(
    (n) => n.opType === "MatMul" && model.has(n.inputs[1]) && model.init(n.inputs[1]).dims.length === 2 && model.init(n.inputs[1]).dims[0] === 1
  )
  if (!freqNode) throw new Error("ONNX: no RoPE frequencies")
  const invFreqT = model.init(freqNode.inputs[1])

  const blocks: Array<BlockRef> = []
  let eps = 1e-6
  for (let i = 0; model.hasNBits(`/model/layers.${i}/attn/qkv/`); i++) {
    const p = `/model/layers.${i}/`
    const qkv = g.linear(`${p}attn/qkv/`, true)
    const fc1 = g.linear(`${p}mlp/linear_fc1/`, true)
    const norm1 = g.normBefore(qkv.w, "LayerNormalization")
    if (i === 0) eps = norm1.node.attributes.get("epsilon")?.f ?? eps
    blocks.push({
      type: "attention",
      norm1,
      qkv: [qkv],
      proj: g.linear(`${p}attn/proj/`, true),
      norm2: g.normBefore(fc1.w, "LayerNormalization"),
      fc1: [fc1],
      fc2: g.linear(`${p}mlp/linear_fc2/`, true)
    })
  }
  if (!blocks.length) throw new Error("ONNX: no encoder blocks")
  const linear1 = g.linear("/model/merger/linear1/", true)
  const linear2 = g.linear("/model/merger/linear2/", true)
  const mergerNorm = g.normBefore(linear1.w, "LayerNormalization")
  const mha = g.nodes.find((n) => n.opType === "MultiHeadAttention")
  const heads = mha ? g.attr(mha, "num_heads") : g.attr(rope, "num_heads")
  if (!heads) throw new Error("ONNX: the encoder's attention has no head count")
  const headDim = width / heads
  const rotary = g.attr(rope, "rotary_embedding_dim") ?? headDim
  if (rotary !== headDim || invFreqT.dims[1] * 4 !== headDim)
    throw new Error(`ONNX: the encoder's RoPE rotates ${rotary} of ${headDim} dims`)
  return {
    config: {
      width,
      heads,
      kvHeads: heads,
      headDim,
      ffn: blocks[0].fc1[0].w.N,
      mergerWidth: linear1.w.N,
      outWidth: linear2.w.N,
      patchValues: patch.w.K,
      merge: g.merge(linear1, width),
      layers: blocks.map(() => "attention"),
      norm: "layer",
      eps,
      mergerEps: mergerNorm.node.attributes.get("epsilon")?.f ?? eps,
      mlp: g.gelu(blocks[0].fc1[0].out),
      mergerAct: g.gelu(linear1.out),
      grid,
      interpolation: "corners",
      rope: true,
      qkNorm: false,
      positions: "grid"
    },
    patch,
    table: { kind: "q4", quant: tableQuant, scales: tableScales, zp: tableZp },
    invFreq: { kind: "values", tensor: invFreqT },
    blocks,
    merger: { norm: mergerNorm, linear1, linear2 }
  }
}

/** LFM2-VL's SigLIP2 encoder and projector (see the header). */
function planSiglip(g: VisionGraph): VisionPlan {
  const { model } = g
  const patch = g.linear("/model/embeddings/patch_embedding/", true)
  const width = patch.w.N
  // The position table: the constant a Resize interpolates to the image's grid.
  const resize = g.nodes.find((n) => n.opType === "Resize" && model.has(n.inputs[0]))
  if (!resize) throw new Error("ONNX: no position table (a Resize of a constant)")
  const table = model.init(resize.inputs[0])
  const grid = table.dims[2]
  if (table.dims.length !== 4 || table.dims[0] !== 1 || table.dims[1] !== width || table.dims[3] !== grid)
    throw new Error(`ONNX: the position table is ${table.dims.join("×")}, expected 1×${width}× a square grid`)
  const mode = resize.attributes.get("mode")?.s
  const coordinates = resize.attributes.get("coordinate_transformation_mode")?.s ?? "half_pixel"
  if (mode !== "linear" || coordinates !== "half_pixel")
    throw new Error(`ONNX: the position table is resized ${mode}, ${coordinates}; the encoder interpolates linear, half_pixel`)

  const blocks: Array<BlockRef> = []
  const blockNorms = new Set<OnnxNode>()
  let eps = 1e-6
  for (let i = 0; model.hasNBits(`/model/layers.${i}/attn/q_proj/`); i++) {
    const p = `/model/layers.${i}/`
    const q = g.linear(`${p}attn/q_proj/`, true)
    const fc1 = g.linear(`${p}mlp/fc1/`, true)
    const norm1 = g.normBefore(q.w, "LayerNormalization")
    const norm2 = g.normBefore(fc1.w, "LayerNormalization")
    blockNorms.add(norm1.node).add(norm2.node)
    if (i === 0) eps = norm1.node.attributes.get("epsilon")?.f ?? eps
    blocks.push({
      type: "attention",
      norm1,
      qkv: [q, g.linear(`${p}attn/k_proj/`, true), g.linear(`${p}attn/v_proj/`, true)],
      proj: g.linear(`${p}attn/out_proj/`, true),
      norm2,
      fc1: [fc1],
      fc2: g.linear(`${p}mlp/fc2/`, true)
    })
  }
  if (!blocks.length) throw new Error("ONNX: no encoder blocks")
  // The projector reads the one LayerNorm that isn't a block's.
  const post = g.nodes.filter((n) => n.opType === "LayerNormalization" && !blockNorms.has(n))
  if (post.length !== 1) throw new Error(`ONNX: ${post.length} LayerNorms after the encoder's blocks, expected the projector's one`)
  const linear1 = g.linear("/model/multimodal_projector/linear_1/", true)
  const linear2 = g.linear("/model/multimodal_projector/linear_2/", true)
  const heads = g.heads()
  return {
    config: {
      width,
      heads,
      kvHeads: heads,
      headDim: width / heads,
      ffn: blocks[0].fc1[0].w.N,
      mergerWidth: linear1.w.N,
      outWidth: linear2.w.N,
      patchValues: patch.w.K,
      merge: g.merge(linear1, width),
      layers: blocks.map(() => "attention"),
      norm: "layer",
      eps,
      mergerEps: post[0].attributes.get("epsilon")?.f ?? eps,
      mlp: g.gelu(blocks[0].fc1[0].out),
      mergerAct: g.gelu(linear1.out),
      grid,
      interpolation: g.attr(resize, "antialias") === 1 ? "antialias" : "half-pixel",
      rope: false,
      qkNorm: false,
      positions: "sequence"
    },
    patch,
    table: { kind: "float", values: table },
    blocks,
    merger: { norm: { weight: model.init(post[0].inputs[1]), bias: model.init(post[0].inputs[2]) }, linear1, linear2 }
  }
}

/** LFM3-VL's tower and projector (see the header). */
function planLfm3(g: VisionGraph): VisionPlan {
  const { model } = g
  const root = "/vision_encoder/"
  const patch = g.linear(`${root}patchifier/`, false)
  const width = patch.w.N
  const rms = "SimplifiedLayerNormalization"
  const embeddingNorm = model.consumersOf(patch.out).find((n) => n.opType === rms)
  if (!embeddingNorm) throw new Error("ONNX: no norm after the patch projection")
  // RoPE's sin table (positions × frequencies), which a Gather reads by row and by column.
  const rotary = g.nodes.find((n) => n.opType === "RotaryEmbedding")
  if (!rotary) throw new Error("ONNX: no RotaryEmbedding")
  if (g.attr(rotary, "interleaved") === 1) throw new Error("ONNX: interleaved RoPE in the encoder")
  const sinGather = g.nodes.find((n) => n.opType === "Gather" && n.name.includes("/rope/sin/") && model.has(n.inputs[0]))
  if (!sinGather) throw new Error("ONNX: no RoPE sin table")
  const sin = model.init(sinGather.inputs[0])
  const heads = g.heads()
  const headDim = width / heads
  if (sin.dims.length !== 2 || sin.dims[0] < 2 || sin.dims[1] * 4 !== headDim)
    throw new Error(`ONNX: the encoder's RoPE table is ${sin.dims.join("×")} for heads of ${headDim}`)

  const blocks: Array<BlockRef> = []
  let eps = 1e-5
  let kvHeads = heads
  for (let i = 0; ; i++) {
    const p = `${root}layers.${i}/`
    const conv = model.hasNBits(`${p}conv/in_proj/`)
    if (!conv && !model.hasNBits(`${p}attn/q_proj/`)) break
    const gateUp = g.linear(`${p}mlp/gate_up/`, false)
    const mlp = { norm2: g.normBefore(gateUp.w, rms), fc1: [gateUp], fc2: g.linear(`${p}mlp/w2/`, false) }
    if (conv) {
      const inProj = g.linear(`${p}conv/in_proj/`, false)
      const node = g.nodes.find((n) => n.opType === "Conv" && n.name.startsWith(p))
      if (!node) throw new Error(`ONNX: no convolution in layer ${i}`)
      const kernel = model.init(node.inputs[1])
      const pads = node.attributes.get("pads")?.ints
      if (kernel.dims.join() !== `${width},1,3,3` || g.attr(node, "group") !== width || !pads?.every((v) => v === 1))
        throw new Error(`ONNX: layer ${i}'s convolution isn't 3 × 3 depthwise, padded by 1`)
      // B, C, x: the Split's first and third parts multiply before the convolution, the second after.
      const split = model.consumersOf(inProj.out).find((n) => n.opType === "Split")
      const gated = split && model.consumersOf(split.outputs[0]).find((n) => n.opType === "Mul")
      if (!split || split.outputs.length !== 3 || !gated?.inputs.includes(split.outputs[2]))
        throw new Error(`ONNX: layer ${i}'s in-projection isn't split into B, C, x`)
      const norm1 = g.normBefore(inProj.w, rms)
      if (!blocks.length) eps = norm1.node.attributes.get("epsilon")?.f ?? eps
      blocks.push({ type: "conv", norm1, inProj, kernel, outProj: g.linear(`${p}conv/out_proj/`, false), ...mlp })
      continue
    }
    const q = g.linear(`${p}attn/q_proj/`, false)
    const k = g.linear(`${p}attn/k_proj/`, false)
    const headNorm = (part: string) => {
      const node = g.nodes.find((n) => n.opType === rms && n.name.startsWith(`${p}attn/${part}_norm/`))
      if (!node) throw new Error(`ONNX: layer ${i} has no ${part} norm`)
      const weight = model.init(node.inputs[1])
      checkShape(weight, weight.dataType, headDim)
      return weight
    }
    kvHeads = k.w.N / headDim
    const norm1 = g.normBefore(q.w, rms)
    if (!blocks.length) eps = norm1.node.attributes.get("epsilon")?.f ?? eps
    blocks.push({
      type: "attention",
      norm1,
      qkv: [q, k, g.linear(`${p}attn/v_proj/`, false)],
      qNorm: headNorm("q"),
      kNorm: headNorm("k"),
      proj: g.linear(`${p}attn/out_proj/`, false),
      ...mlp
    })
  }
  if (!blocks.length) throw new Error("ONNX: no encoder blocks")
  const linear1 = g.linear("/vision_projector/linear_1/", true)
  const linear2 = g.linear("/vision_projector/linear_2/", true)
  const mergerNorm = g.normBefore(linear1.w, "LayerNormalization")
  // The graph takes patches in raster order within each block of merge × merge.
  return {
    config: {
      width,
      heads,
      kvHeads,
      headDim,
      ffn: blocks[0].fc2.w.K,
      mergerWidth: linear1.w.N,
      outWidth: linear2.w.N,
      patchValues: patch.w.K,
      merge: g.merge(linear1, width),
      layers: blocks.map((b) => b.type),
      norm: "rms",
      eps,
      mergerEps: mergerNorm.node.attributes.get("epsilon")?.f ?? 1e-6,
      mlp: "swiglu",
      mergerAct: g.gelu(linear1.out),
      grid: 0,
      interpolation: "corners",
      rope: true,
      qkNorm: true,
      positions: "sequence"
    },
    patch,
    patchNorm: { weight: model.init(embeddingNorm.inputs[1]), bias: null },
    invFreq: { kind: "sin", tensor: sin },
    blocks,
    merger: { norm: mergerNorm, linear1, linear2 }
  }
}

/** The encoder's graph, read: its config and every tensor the load takes, in order. */
export function planVision(graphBytes: Uint8Array) {
  const g = new VisionGraph(graphBytes)
  const plan = g.model.hasNBits("/vision_encoder/patchifier/")
    ? planLfm3(g)
    : g.model.hasNBits("/model/embeddings/patch_embedding/")
      ? planSiglip(g)
      : planQwen(g)
  checkVisionConfig(plan.config)
  const { config } = plan
  const kv = config.kvHeads * config.headDim
  const swiglu = config.mlp === "swiglu"
  for (const [i, b] of plan.blocks.entries()) {
    const rows = (parts: Array<LinearRef>) => parts.reduce((sum, l) => sum + l.w.N, 0)
    const shaped = (l: LinearRef, N: number, K: number) => l.w.N === N && l.w.K === K
    const ok =
      rows(b.fc1) === (swiglu ? 2 : 1) * config.ffn &&
      b.fc1.every((l) => l.w.K === config.width) &&
      shaped(b.fc2, config.width, config.ffn) &&
      (b.type === "attention"
        ? rows(b.qkv) === config.width + 2 * kv && b.qkv.every((l) => l.w.K === config.width) && shaped(b.proj, config.width, config.width)
        : shaped(b.inProj, 3 * config.width, config.width) && shaped(b.outProj, config.width, config.width))
    if (!ok) throw new Error(`ONNX: the encoder's block ${i} doesn't have the first one's shapes`)
  }
  const reads: Array<Read> = []
  const want = (t: OnnxTensor | null | undefined) => {
    if (t) reads.push({ key: t.name, tensor: t })
  }
  const wantLinear = (l: LinearRef) => {
    want(l.w.quant)
    want(l.w.scales)
    want(l.w.zp)
    want(l.bias)
  }
  const wantNorm = (n: NormRef) => {
    want(n.weight)
    want(n.bias)
  }
  wantLinear(plan.patch)
  if (plan.table?.kind === "q4") {
    want(plan.table.quant)
    want(plan.table.scales)
    want(plan.table.zp)
  } else want(plan.table?.values)
  want(plan.invFreq?.tensor)
  if (plan.patchNorm) wantNorm(plan.patchNorm)
  for (const b of plan.blocks) {
    wantNorm(b.norm1)
    if (b.type === "attention") {
      b.qkv.forEach(wantLinear)
      want(b.qNorm)
      want(b.kNorm)
      wantLinear(b.proj)
    } else {
      wantLinear(b.inProj)
      want(b.kernel)
      wantLinear(b.outProj)
    }
    wantNorm(b.norm2)
    b.fc1.forEach(wantLinear)
    wantLinear(b.fc2)
  }
  wantNorm(plan.merger.norm)
  wantLinear(plan.merger.linear1)
  wantLinear(plan.merger.linear2)
  // What the claim is sized by (vision.ts visionWeightBytes): the projections' bits, and whether they carry zero points.
  return { ...plan, reads, bits: plan.blocks[0].fc2.w.bits, zeros: plan.blocks[0].fc2.w.zp !== null }
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
  const linear = async (l: LinearRef): Promise<VisionLinear> => {
    const { w } = l
    const quant = await reads.take(w.quant.name)
    const scales = await reads.take(w.scales.name)
    const zp = w.zp ? await reads.take(w.zp.name) : null
    const m = matMulNBitsMatrix({ N: w.N, K: w.K, bits: w.bits, blockSize: w.blockSize, scaleBytes: w.scales.dataType === OnnxDataType.FLOAT16 ? 2 : 4 }, quant, scales, zp)
    return { m, bias: l.bias ? await floats(l.bias) : null }
  }
  const stacked = async (parts: Array<LinearRef>) => {
    const loaded: Array<VisionLinear> = []
    for (const part of parts) loaded.push(await linear(part))
    return stackLinears(loaded)
  }
  const norm = async (n: NormRef): Promise<VisionNorm> => ({ weight: await floats(n.weight), bias: n.bias ? await floats(n.bias) : null })
  const { config } = plan
  yield { kind: "config", config }
  const patch = await linear(plan.patch)
  let positions: Float32Array | undefined
  if (plan.table?.kind === "q4") {
    const quant = await reads.take(plan.table.quant.name)
    const tableScales = await floats(plan.table.scales)
    const zp = plan.table.zp ? await reads.take(plan.table.zp.name) : null
    positions = dequantizeTable(quant, tableScales, zp, config.grid * config.grid, config.width)
  } else if (plan.table) {
    // [1, width, grid, grid] as grid² rows of `width`.
    const values = await floats(plan.table.values)
    const cells = config.grid * config.grid
    positions = new Float32Array(cells * config.width)
    for (let e = 0; e < config.width; e++) for (let p = 0; p < cells; p++) positions[p * config.width + e] = values[e * cells + p]
  }
  let invFreq: Float32Array | undefined
  if (plan.invFreq) {
    const values = await floats(plan.invFreq.tensor)
    if (plan.invFreq.kind === "values") invFreq = values
    else {
      // Row p of the table is sin(p · frequency): row 1 gives each frequency (all at most 1).
      const n = config.headDim / 4
      invFreq = new Float32Array(n)
      for (let j = 0; j < n; j++) {
        if (values[j] !== 0 || !(values[n + j] > 0 && values[n + j] < 1)) throw new Error("ONNX: the encoder's RoPE table isn't sin(position · frequency)")
        invFreq[j] = Math.asin(values[n + j])
      }
    }
  }
  const patchNorm = plan.patchNorm ? await norm(plan.patchNorm) : undefined
  yield { kind: "patch", patch, ...(positions && { positions }), ...(invFreq && { invFreq }), ...(patchNorm && { norm: patchNorm }) }
  for (const [index, b] of plan.blocks.entries()) {
    const norm1 = await norm(b.norm1)
    let value: VisionBlock
    if (b.type === "attention") {
      const qkv = await stacked(b.qkv)
      const qNorm = b.qNorm ? await floats(b.qNorm) : undefined
      const kNorm = b.kNorm ? await floats(b.kNorm) : undefined
      const proj = await linear(b.proj)
      value = { type: "attention", norm1, qkv, ...(qNorm && { qNorm }), ...(kNorm && { kNorm }), proj, norm2: await norm(b.norm2), fc1: await stacked(b.fc1), fc2: await linear(b.fc2) }
    } else {
      const inProj = await linear(b.inProj)
      const kernel = await floats(b.kernel)
      const outProj = await linear(b.outProj)
      value = { type: "conv", norm1, inProj, kernel, outProj, norm2: await norm(b.norm2), fc1: await stacked(b.fc1), fc2: await linear(b.fc2) }
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
