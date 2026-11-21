/**
 * Tessera's weights as released in ONNX (cfahlgren1/tessera-small-onnx):
 * the denoiser (tessera_w8.onnx) and the CLIP text encoder
 * (clip_text_w8.onnx), read from the graphs as they are. Each parameter is
 * found by the node that uses it (most projections' weights have no name
 * of their own), and resolved through what makes it: a float initializer,
 * an fp16 one cast to float, a Constant node, or an int8 DequantizeLinear.
 *
 * Projections stay int8: their rows as [output][input], with one scale
 * per output row or per input column, whichever axis the file quantized.
 */
import { halvesToFloats } from "./half.js"
import { OnnxDataType, parseOnnxModel, type OnnxGraph, type OnnxNode, type OnnxTensor } from "./onnx.js"

/** A projection y = x · Wᵀ (+ bias), W int8 with per-row or per-column scales. */
export type QLinear = {
  /** Outputs. */
  N: number
  /** Inputs. */
  K: number
  /** W[n][k] at n · K + k. */
  q: Int8Array
  /** "out": W[n][k] = q · scale[n]; "in": W[n][k] = q · scale[k]. */
  axis: "out" | "in"
  scale: Float32Array
  bias: Float32Array | null
}

export const TESSERA_CONFIG = {
  vocab: 512,
  grid: 16,
  dim: 512,
  heads: 8,
  headDim: 64,
  depthIn: 2,
  depthCore: 8,
  depthOut: 2,
  patch: 2,
  hidden: 1536,
  registers: 4,
  textDim: 512,
  textLen: 24,
  ropeTheta: 100,
  /** RMSNorm's epsilon (model.py's default). */
  eps: 1e-6
} as const

export type TesseraBlock = {
  qkv: QLinear
  qNorm: Float32Array
  kNorm: Float32Array
  proj: QLinear
  /** a | gate rows: SwiGLU's value then its gate. */
  w12: QLinear
  w3: QLinear
  /** Depthwise 3 × 3 kernel of the gate's image rows: [channel][9]. */
  dwWeight: Float32Array
  dwBias: Float32Array
  /** Offsets added to the shared modulation: [6][dim] (shift, scale, gate, ×2). */
  modBias: Float32Array
}

export type TesseraWeights = {
  /** Each token's embedding (palette tokens, then MASK): [vocab + 1][dim]. */
  embed: Float32Array
  /** [256][dim]. */
  pos: Float32Array
  scProj: QLinear
  textNorm: Float32Array
  textProj: QLinear
  /** [textLen][dim]. */
  nullText: Float32Array
  poolNorm: Float32Array
  pool1: QLinear
  pool3: QLinear
  nullCond: Float32Array
  ratio0: QLinear
  ratio2: QLinear
  /** [registers][dim]. */
  registers: Float32Array
  tBlock: QLinear
  blocksIn: Array<TesseraBlock>
  downNorm: Float32Array
  down: QLinear
  blocksCore: Array<TesseraBlock>
  up: QLinear
  blocksOut: Array<TesseraBlock>
  finalMod: QLinear
  head: QLinear
}

export type ClipLayer = {
  ln1: { weight: Float32Array; bias: Float32Array }
  q: QLinear
  k: QLinear
  v: QLinear
  out: QLinear
  ln2: { weight: Float32Array; bias: Float32Array }
  fc1: QLinear
  fc2: QLinear
}

export type ClipWeights = {
  /** Token embeddings, int8, one scale per token: [vocab][width]. */
  tokens: { q: Int8Array; scale: Float32Array; vocab: number }
  /** Position embeddings, dequantized: [positions][width]. */
  positions: Float32Array
  layers: Array<ClipLayer>
  final: { weight: Float32Array; bias: Float32Array }
  width: number
  heads: number
  /** LayerNorm's epsilon, from the graph. */
  eps: number
}

type Value =
  | { kind: "float"; dims: Array<number>; data: Float32Array }
  | { kind: "q8"; dims: Array<number>; q: Int8Array; scale: Float32Array; axis: number }

/** A graph's values by name, and the nodes that use them. */
class Graph {
  readonly nodes = new Map<string, OnnxNode>()
  private readonly producers = new Map<string, OnnxNode>()
  private readonly consumers = new Map<string, Array<OnnxNode>>()
  private readonly g: OnnxGraph

  constructor(bytes: Uint8Array) {
    this.g = parseOnnxModel(bytes)
    for (const n of this.g.nodes) {
      this.nodes.set(n.name, n)
      for (const o of n.outputs) this.producers.set(o, n)
      for (const i of n.inputs) {
        const list = this.consumers.get(i) ?? []
        list.push(n)
        this.consumers.set(i, list)
      }
    }
  }

  node(name: string): OnnxNode {
    const n = this.nodes.get(name)
    if (!n) throw new Error(`Tessera: the graph has no node ${name}.`)
    return n
  }

  private tensorFloats(t: OnnxTensor): Float32Array {
    if (t.floatData?.length) return Float32Array.from(t.floatData)
    if (!t.raw) throw new Error(`Tessera: ${t.name} has no inline data.`)
    if (t.dataType === OnnxDataType.FLOAT16) return halvesToFloats(t.raw)
    if (t.dataType === OnnxDataType.FLOAT) return new Float32Array(t.raw.slice().buffer)
    throw new Error(`Tessera: ${t.name} has data type ${t.dataType}, expected float or float16.`)
  }

  /** What value `name` holds: constant data, or an int8 tensor and its scales. */
  value(name: string): Value {
    const init = this.g.initializers.get(name)
    if (init) return { kind: "float", dims: init.dims, data: this.tensorFloats(init) }
    const p = this.producers.get(name)
    if (!p) throw new Error(`Tessera: nothing makes ${name}.`)
    if (p.opType === "Cast") return this.value(p.inputs[0]!)
    if (p.opType === "Constant") {
      const t = p.attributes.get("value")?.t
      if (!t) throw new Error(`Tessera: the Constant ${p.name} has no tensor.`)
      return { kind: "float", dims: t.dims, data: this.tensorFloats(t) }
    }
    if (p.opType === "DequantizeLinear") {
      const q = this.g.initializers.get(p.inputs[0]!)
      const scale = this.value(p.inputs[1]!)
      if (!q?.raw || q.dataType !== OnnxDataType.INT8 || scale.kind !== "float")
        throw new Error(`Tessera: ${p.name} isn't an int8 weight with float scales.`)
      if (p.inputs[2]) {
        const zp = this.g.initializers.get(p.inputs[2])
        if (zp?.raw && zp.raw.some((b) => b !== 0)) throw new Error(`Tessera: ${p.name} has zero points.`)
      }
      const axis = p.attributes.get("axis")?.i ?? 1
      return { kind: "q8", dims: q.dims, q: new Int8Array(q.raw.buffer, q.raw.byteOffset, q.raw.byteLength), scale: scale.data, axis }
    }
    throw new Error(`Tessera: ${name} comes from a ${p.opType}, not a constant.`)
  }

  floats(name: string): Float32Array {
    const v = this.value(name)
    if (v.kind !== "float") throw new Error(`Tessera: ${name} is quantized, expected floats.`)
    return v.data
  }

  /** Input `index` of node `node`, as floats. */
  input(node: string, index: number): Float32Array {
    return this.floats(this.node(node).inputs[index]!)
  }

  /** The bias an Add right after `node` adds, if it adds a constant. */
  private addedBias(node: OnnxNode): Float32Array | null {
    for (const c of this.consumers.get(node.outputs[0]!) ?? []) {
      if (c.opType !== "Add") continue
      const other = c.inputs.find((i) => i !== node.outputs[0])
      if (!other) continue
      try {
        return this.floats(other)
      } catch {
        // An Add of two activations (a residual), not a bias.
      }
    }
    return null
  }

  /** A MatMul node's weight (input 1, [K][N]) and the bias added after it. */
  matmul(name: string, withBias = true): QLinear {
    const n = this.node(name)
    const v = this.value(n.inputs[1]!)
    if (v.kind !== "q8" || v.dims.length !== 2) throw new Error(`Tessera: ${name}'s weight isn't a 2D int8 tensor.`)
    const [K, N] = v.dims as [number, number]
    const q = new Int8Array(N * K)
    for (let k = 0; k < K; k++) for (let j = 0; j < N; j++) q[j * K + k] = v.q[k * N + j]!
    return linear(N, K, q, v.axis === 1 ? "out" : "in", v.scale, withBias ? this.addedBias(n) : null)
  }

  /** A Gemm node's weight (input 1, [N][K] with transB) and bias (input 2). */
  gemm(name: string): QLinear {
    const n = this.node(name)
    if (n.attributes.get("transB")?.i !== 1) throw new Error(`Tessera: ${name} isn't a Gemm with transB.`)
    const v = this.value(n.inputs[1]!)
    if (v.kind !== "q8" || v.dims.length !== 2) throw new Error(`Tessera: ${name}'s weight isn't a 2D int8 tensor.`)
    const [N, K] = v.dims as [number, number]
    return linear(N, K, v.q.slice(), v.axis === 0 ? "out" : "in", v.scale, n.inputs[2] ? this.floats(n.inputs[2]) : null)
  }

  /** A quantized table (an embedding), dequantized: [rows][cols]. */
  table(name: string): { rows: number; cols: number; data: Float32Array } {
    const v = this.value(name)
    if (v.kind === "float") return { rows: v.dims[0]!, cols: v.dims[1]!, data: v.data }
    const [rows, cols] = v.dims as [number, number]
    const data = new Float32Array(rows * cols)
    for (let r = 0; r < rows; r++)
      for (let c = 0; c < cols; c++) data[r * cols + c] = v.q[r * cols + c]! * v.scale[v.axis === 0 ? r : c]!
    return { rows, cols, data }
  }

  /** The int8 tensor `name`, as the file holds it. */
  quantized(name: string) {
    const v = this.value(name)
    if (v.kind !== "q8") throw new Error(`Tessera: ${name} isn't int8.`)
    return v
  }
}

function linear(N: number, K: number, q: Int8Array, axis: "out" | "in", scale: Float32Array, bias: Float32Array | null): QLinear {
  if (scale.length !== (axis === "out" ? N : K)) throw new Error(`Tessera: a ${N} × ${K} weight has ${scale.length} scales.`)
  if (bias && bias.length !== N) throw new Error(`Tessera: a ${N}-output projection has ${bias.length} biases.`)
  return { N, K, q, axis, scale, bias }
}

const silu = (x: number) => x / (1 + Math.exp(-x))

/** x · Wᵀ + b for one row, in float64 sums: the load-time projections (the color MLP). */
function project(l: QLinear, x: Float32Array, xOffset = 0): Float32Array {
  const out = new Float32Array(l.N)
  for (let n = 0; n < l.N; n++) {
    let s = 0
    for (let k = 0; k < l.K; k++) s += x[xOffset + k]! * l.q[n * l.K + k]! * (l.axis === "in" ? l.scale[k]! : 1)
    out[n] = s * (l.axis === "out" ? l.scale[n]! : 1) + (l.bias?.[n] ?? 0)
  }
  return out
}

/** Tessera's denoiser weights, from tessera_w8.onnx's bytes. */
export function readTesseraWeights(bytes: Uint8Array): TesseraWeights {
  const g = new Graph(bytes)
  const C = TESSERA_CONFIG
  // The embedding table, as embed_table() builds it: each palette token's
  // learned row plus the color MLP of its color features, then MASK's row.
  const colorFeats = g.input("/m/color_mlp/color_mlp.0/Gemm", 0)
  const color0 = g.gemm("/m/color_mlp/color_mlp.0/Gemm")
  const color2 = g.gemm("/m/color_mlp/color_mlp.2/Gemm")
  const tok = g.table(g.node("/m/Add").inputs[0]!)
  const mask = g.input("/m/Concat", 1)
  if (tok.rows !== C.vocab || tok.cols !== C.dim || mask.length !== C.dim || colorFeats.length !== C.vocab * color0.K)
    throw new Error("Tessera: the embedding table isn't the config's shape.")
  const embed = new Float32Array((C.vocab + 1) * C.dim)
  for (let t = 0; t < C.vocab; t++) {
    const hidden = project(color0, colorFeats, t * color0.K).map(silu)
    const color = project(color2, hidden)
    for (let d = 0; d < C.dim; d++) embed[t * C.dim + d] = tok.data[t * C.dim + d]! + color[d]!
  }
  embed.set(mask, C.vocab * C.dim)

  const block = (path: string): TesseraBlock => ({
    qkv: g.matmul(`/m/${path}/attn/qkv/MatMul`),
    qNorm: g.input(`/m/${path}/attn/q_norm/Mul_1`, 1),
    kNorm: g.input(`/m/${path}/attn/k_norm/Mul_1`, 1),
    proj: g.matmul(`/m/${path}/attn/proj/MatMul`),
    w12: g.matmul(`/m/${path}/mlp/w12/MatMul`),
    w3: g.matmul(`/m/${path}/mlp/w3/MatMul`),
    dwWeight: g.floats(`m.${path}.mlp.dw.weight`),
    dwBias: g.floats(`m.${path}.mlp.dw.bias`),
    modBias: g.floats(`m.${path}.mod_bias`)
  })
  const blocks = (stage: string, n: number) => Array.from({ length: n }, (_, i) => block(`${stage}.${i}`))
  const w: TesseraWeights = {
    embed,
    pos: g.floats("m.pos_emb"),
    scProj: g.matmul("/m/sc_proj/MatMul", false),
    textNorm: g.input("/m/text_norm/Mul_1", 1),
    textProj: g.matmul("/m/text_proj/MatMul"),
    nullText: g.floats("m.null_text"),
    poolNorm: g.input("/m/pool_mlp/pool_mlp.0/Mul_1", 1),
    pool1: g.gemm("/m/pool_mlp/pool_mlp.1/Gemm"),
    pool3: g.gemm("/m/pool_mlp/pool_mlp.3/Gemm"),
    nullCond: g.floats("m.null_cond"),
    ratio0: g.gemm("/m/ratio_mlp/ratio_mlp.0/Gemm"),
    ratio2: g.gemm("/m/ratio_mlp/ratio_mlp.2/Gemm"),
    registers: g.floats("m.registers"),
    tBlock: g.gemm("/m/t_block/t_block.1/Gemm"),
    blocksIn: blocks("blocks_in", C.depthIn),
    downNorm: g.input("/m/down/down.0/Mul_1", 1),
    down: g.matmul("/m/down/down.1/MatMul"),
    blocksCore: blocks("blocks_core", C.depthCore),
    up: g.matmul("/m/up/MatMul"),
    blocksOut: blocks("blocks_out", C.depthOut),
    finalMod: g.gemm("/m/final_mod/final_mod.1/Gemm"),
    head: g.matmul("/m/head/MatMul")
  }
  checkTessera(w)
  return w
}

function checkTessera(w: TesseraWeights) {
  const C = TESSERA_CONFIG
  const shape = (l: QLinear, N: number, K: number, what: string) => {
    if (l.N !== N || l.K !== K) throw new Error(`Tessera: ${what} is ${l.N} × ${l.K}, expected ${N} × ${K}.`)
  }
  shape(w.scProj, C.dim, C.vocab, "sc_proj")
  shape(w.textProj, C.dim, C.textDim, "text_proj")
  shape(w.tBlock, 6 * C.dim, C.dim, "t_block")
  shape(w.down, C.dim, C.patch * C.patch * C.dim, "down")
  shape(w.up, C.patch * C.patch * C.dim, C.dim, "up")
  shape(w.finalMod, 2 * C.dim, C.dim, "final_mod")
  shape(w.head, C.vocab, C.dim, "head")
  for (const b of [...w.blocksIn, ...w.blocksCore, ...w.blocksOut]) {
    shape(b.qkv, 3 * C.dim, C.dim, "qkv")
    shape(b.proj, C.dim, C.dim, "proj")
    shape(b.w12, 2 * C.hidden, C.dim, "w12")
    shape(b.w3, C.dim, C.hidden, "w3")
    if (b.dwWeight.length !== C.hidden * 9 || b.modBias.length !== 6 * C.dim || b.qNorm.length !== C.headDim)
      throw new Error("Tessera: a block's weights aren't the config's shape.")
  }
  if (w.pos.length !== C.grid * C.grid * C.dim || w.nullText.length !== C.textLen * C.dim || w.registers.length !== C.registers * C.dim)
    throw new Error("Tessera: the denoiser's tables aren't the config's shape.")
}

/** CLIP's text encoder weights, from clip_text_w8.onnx's bytes. */
export function readClipWeights(bytes: Uint8Array): ClipWeights {
  const g = new Graph(bytes)
  const tokName = g.node("/m/embeddings/token_embedding/Gather").inputs[0]!
  const tok = g.quantized(tokName)
  if (tok.axis !== 0) throw new Error("Tessera: CLIP's token embeddings aren't quantized per token.")
  const [vocab, width] = tok.dims as [number, number]
  const positions = g.table(g.node("/m/embeddings/position_embedding/Gather").inputs[0]!).data
  const norm = (path: string) => {
    const n = g.node(`/m/${path}/LayerNormalization`)
    return { weight: g.floats(n.inputs[1]!), bias: g.floats(n.inputs[2]!), eps: n.attributes.get("epsilon")?.f ?? 1e-5 }
  }
  const layers: Array<ClipLayer> = []
  for (let i = 0; g.nodes.has(`/m/encoder/layers.${i}/self_attn/q_proj/MatMul`); i++) {
    const p = `/m/encoder/layers.${i}`
    layers.push({
      ln1: norm(`encoder/layers.${i}/layer_norm1`),
      q: g.matmul(`${p}/self_attn/q_proj/MatMul`),
      k: g.matmul(`${p}/self_attn/k_proj/MatMul`),
      v: g.matmul(`${p}/self_attn/v_proj/MatMul`),
      out: g.matmul(`${p}/self_attn/out_proj/MatMul`),
      ln2: norm(`encoder/layers.${i}/layer_norm2`),
      fc1: g.matmul(`${p}/mlp/fc1/MatMul`),
      fc2: g.matmul(`${p}/mlp/fc2/MatMul`)
    })
  }
  const final = norm("final_layer_norm")
  if (!layers.length) throw new Error("Tessera: CLIP's graph has no layers.")
  return { tokens: { q: tok.q.slice(), scale: tok.scale, vocab }, positions, layers, final, width, heads: width / 64, eps: final.eps }
}
