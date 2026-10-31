/**
 * Loads LFM3 and LFM2/LFM2.5 weights from an onnxruntime-genai style ONNX
 * export with
 * 4-bit MatMulNBits projections and a GatherBlockQuantized embedding,
 * such as daveycodez/LFM3-1.7B-Web's onnx/model_q4f16.onnx (+ its
 * .onnx_data file).
 *
 * The graph is read to place the parts of fused matrices: the attention
 * q_proj holds q and the output gate (split by a Split node; the gate rows
 * are the ones a Sigmoid reads), and the conv in_proj holds B, C and x (C
 * is the part multiplied after the convolution). Full-attention layers in
 * this export already have each head's rotated dims first (GroupQueryAttention
 * with a rotary cache of ropeDimsFull / 2 frequencies), which is the layout
 * both loaders produce; the loader checks that against the graph.
 */
import { halvesToFloats } from "./half.js"
import {
  OnnxDataType,
  onnxElementSize,
  onnxInt64s,
  parseOnnxModel,
  type OnnxGraph,
  type OnnxNode,
  type OnnxTensor
} from "./onnx.js"
import { ReadPlan } from "./read-plan.js"
import { newQuantMatrix, packMatMulNBits } from "./repack.js"
import type {
  AttentionWeights,
  ByteSource,
  ConvWeights,
  LayerType,
  LayerWeights,
  ModelConfig,
  QuantMatrix,
  WeightPiece
} from "./weights.js"

export type LoadOnnxOptions = {
  /** The .onnx file (graph), as bytes or a source. */
  graph: ByteSource | Uint8Array
  /** The external data file (.onnx_data). */
  data: ByteSource
  /** The model's config.json, parsed. */
  config: unknown
  /** Embedding rows converted per step (default 2048). */
  embeddingChunkRows?: number
}

/** A MatMulNBits weight: its initializers and shape. */
type NBits = {
  name: string
  N: number
  K: number
  quant: OnnxTensor
  scales: OnnxTensor
  zp: OnnxTensor | null
}

/** Rows [start, start + rows) of a weight. */
type Segment = { start: number; rows: number }

/**
 * Yields the model's pieces: config, each layer in order, the final norm,
 * then the embedding (also the output head).
 */
export async function* loadOnnx(
  options: LoadOnnxOptions
): AsyncGenerator<WeightPiece, void, undefined> {
  const { data } = options
  const graphBytes =
    options.graph instanceof Uint8Array
      ? options.graph
      : await options.graph.read(0, options.graph.size)
  const graph = parseOnnxModel(graphBytes)
  const model = new OnnxModel(graph)
  const config = onnxConfig(options.config, model)
  yield { kind: "config", config }

  const {
    hiddenSize: D,
    intermediateSize: F,
    numHeads,
    numKvHeads,
    headDim,
    numLayers
  } = config

  // Check the graph against what the layout assumes, and find the fused parts.
  const layers = config.layerTypes.map((type, i) =>
    model.layerPlan(i, type, config)
  )

  const plan = new ReadPlan()
  const want = (t: OnnxTensor | null) => {
    if (!t) return
    const length = tensorBytes(t)
    if (t.raw) {
      if (t.raw.length !== length)
        throw new Error(
          `ONNX: ${t.name} has ${t.raw.length} bytes, expected ${length}`
        )
      plan.add({ key: t.name, source: data, offset: 0, length, bytes: t.raw })
      return
    }
    if (!t.external) throw new Error(`ONNX: ${t.name} has no data`)
    if (t.external.length !== length)
      throw new Error(
        `ONNX: ${t.name} has ${t.external.length} bytes, expected ${length}`
      )
    plan.add({ key: t.name, source: data, offset: t.external.offset, length })
  }
  const wantNBits = (w: NBits) => {
    want(w.quant)
    want(w.scales)
    want(w.zp)
  }
  // In the order the layer code below takes them: norms, operator matrices,
  // conv kernel or q/k norms, MLP matrices.
  for (const l of layers) {
    for (const t of l.norms) want(t)
    for (const w of l.matrices.slice(0, -3)) wantNBits(w)
    for (const t of l.extra) want(t)
    for (const w of l.matrices.slice(-3)) wantNBits(w)
  }
  const finalNorm = model.finalNorm()
  want(finalNorm)

  const emb = model.embedding(D)
  const chunkRows = options.embeddingChunkRows ?? 2048
  const blocks = D / 32
  const zpStride = Math.ceil(blocks / 2)
  for (let r = 0; r < emb.N; r += chunkRows) {
    const rows = Math.min(chunkRows, emb.N - r)
    const ext = (t: OnnxTensor, rowBytes: number, key: string) => {
      if (!t.external)
        throw new Error(`ONNX: ${t.name} is expected in the external data file`)
      plan.add({
        key: `${key}@${r}`,
        source: data,
        offset: t.external.offset + r * rowBytes,
        length: rows * rowBytes
      })
    }
    ext(emb.quant, blocks * 16, "quant")
    ext(emb.scales, blocks * 2, "scales")
    if (emb.zp) ext(emb.zp, zpStride, "zp")
  }

  const floats = async (
    t: OnnxTensor,
    length: number
  ): Promise<Float32Array> => {
    const bytes = await plan.take(t.name)
    if (t.dataType === OnnxDataType.FLOAT16) {
      const values = halvesToFloats(bytes)
      if (values.length !== length)
        throw new Error(
          `ONNX: ${t.name} has ${values.length} values, expected ${length}`
        )
      return values
    }
    if (t.dataType === OnnxDataType.FLOAT) {
      const values = new Float32Array(bytes.slice().buffer)
      if (values.length !== length)
        throw new Error(
          `ONNX: ${t.name} has ${values.length} values, expected ${length}`
        )
      return values
    }
    throw new Error(
      `ONNX: ${t.name} has data type ${t.dataType}, expected float16 or float`
    )
  }

  /**
   * MatMulNBits weights into one matrix, read in the order given; each
   * segment's rows go to destination row `to`.
   */
  const matrix = async (
    parts: Array<{ w: NBits; segments: Array<Segment & { to: number }> }>
  ): Promise<QuantMatrix> => {
    const K = parts[0].w.K
    let N = 0
    for (const { w, segments } of parts) {
      if (w.K !== K)
        throw new Error(`ONNX: ${w.name} has K ${w.K}, expected ${K}`)
      for (const s of segments) {
        if (s.start + s.rows > w.N)
          throw new Error(
            `ONNX: rows ${s.start}..${s.start + s.rows} are outside ${w.name}`
          )
        N += s.rows
      }
    }
    const m = newQuantMatrix(
      N,
      K,
      4,
      parts.some((p) => p.w.zp)
    )
    const rowBlocks = K / 32
    const rowZp = Math.ceil(rowBlocks / 2)
    for (const { w, segments } of parts) {
      const quant = await plan.take(w.quant.name)
      const scales = await plan.take(w.scales.name)
      const zp = w.zp ? await plan.take(w.zp.name) : null
      for (const { start, rows, to } of segments) {
        packMatMulNBits(
          m,
          to,
          quant.subarray(start * rowBlocks * 16),
          scales.subarray(start * rowBlocks * 2),
          zp ? zp.subarray(start * rowZp) : null,
          rows
        )
      }
    }
    return m
  }
  const whole = (w: NBits, to = 0) => ({
    w,
    segments: [{ start: 0, rows: w.N, to }]
  })

  for (let i = 0; i < numLayers; i++) {
    const l = layers[i]
    const [n0, n1, n2, n3] = l.norms as [
      OnnxTensor,
      OnnxTensor | null,
      OnnxTensor,
      OnnxTensor | null
    ]
    const operatorNorm = await floats(n0, D)
    const postOperatorNorm = n1 ? await floats(n1, D) : undefined
    const ffnNorm = await floats(n2, D)
    const postFfnNorm = n3 ? await floats(n3, D) : undefined
    let conv: ConvWeights | undefined
    let attention: AttentionWeights | undefined
    if (l.type === "conv") {
      const [inProjW, outProjW] = l.matrices as [NBits, NBits]
      const bcx = l.bcx ?? []
      const inProj = await matrix([
        { w: inProjW, segments: bcx.map((s, j) => ({ ...s, to: j * D })) }
      ])
      const outProj = await matrix([whole(outProjW)])
      // [channels, 1, width]: channel-major, oldest tap first.
      const kernel = await floats(l.extra[0], D * config.convKernel)
      conv = { inProj, outProj, kernel }
    } else {
      const [qW, kW, vW] = l.matrices as [NBits, NBits, NBits]
      const oW = l.matrices[l.gateWeight ? 4 : 3]
      const qRows = numHeads * headDim
      const kv = numKvHeads * headDim
      if (kW.N !== kv || vW.N !== kv)
        throw new Error(
          `ONNX: layer ${i} k/v projections have ${kW.N}/${vW.N} rows`
        )
      const gateTo = qRows + 2 * kv
      // Rows: q, k, v, gate (the gate from q_proj's own rows, or its own weight).
      const qkvg = await matrix([
        {
          w: qW,
          segments: [
            { ...l.q, to: 0 },
            ...(l.gate ? [{ ...l.gate, to: gateTo }] : [])
          ]
        },
        whole(kW, qRows),
        whole(vW, qRows + kv),
        ...(l.gateWeight ? [whole(l.gateWeight, gateTo)] : [])
      ])
      const o = await matrix([whole(oW)])
      const qNorm = await floats(l.extra[0], headDim)
      const kNorm = await floats(l.extra[1], headDim)
      attention = { qkvg, o, qNorm, kNorm }
    }
    const [gateW, upW, downW] = l.matrices.slice(-3) as [NBits, NBits, NBits]
    if (gateW.N !== F || upW.N !== F || downW.K !== F)
      throw new Error(`ONNX: layer ${i} MLP shapes don't match`)
    const gateUp = await matrix([whole(gateW), whole(upW, F)])
    const down = await matrix([whole(downW)])
    const value: LayerWeights = {
      type: l.type,
      operatorNorm,
      ffnNorm,
      gateUp,
      down
    }
    if (postOperatorNorm) value.postOperatorNorm = postOperatorNorm
    if (postFfnNorm) value.postFfnNorm = postFfnNorm
    if (conv) value.conv = conv
    if (attention) value.attention = attention
    yield { kind: "layer", layer: i, value }
  }

  yield { kind: "finalNorm", value: await floats(finalNorm, D) }

  // In chunks of rows, so the embedding is never all in JavaScript memory.
  for (let r = 0; r < emb.N; r += chunkRows) {
    const rows = Math.min(chunkRows, emb.N - r)
    const quant = await plan.take(`quant@${r}`)
    const scales = await plan.take(`scales@${r}`)
    const zp = emb.zp ? await plan.take(`zp@${r}`) : null
    const m = newQuantMatrix(rows, D, 4, emb.zp !== null)
    packMatMulNBits(m, 0, quant, scales, zp, rows)
    yield { kind: "embedding", value: m, row0: r, totalRows: emb.N }
  }
}

function tensorBytes(t: OnnxTensor): number {
  let n = 1
  for (const d of t.dims) n *= d
  return Math.ceil(n * onnxElementSize(t.dataType))
}

type LayerPlan = {
  type: LayerType
  /** operator, post-operator, ffn, post-ffn norms (the post ones: LFM3 only). */
  norms: Array<OnnxTensor | null>
  /** conv: in_proj, out_proj; attention: q_proj, k, v, o (+ gate_proj); then mlp gate, up, down. */
  matrices: Array<NBits>
  /** conv: the kernel; attention: q_norm, k_norm. */
  extra: Array<OnnxTensor>
  /** conv: in_proj rows of B, C, x. */
  bcx?: Array<Segment>
  /** attention: q rows within q_proj, and the gate's when q_proj holds it. */
  q: Segment
  gate?: Segment
  /** attention: a separate gate projection, when the export has one. */
  gateWeight?: NBits
}

/** Lookups over the graph. */
class OnnxModel {
  private consumers = new Map<string, Array<OnnxNode>>()
  private producers = new Map<string, OnnxNode>()

  constructor(readonly graph: OnnxGraph) {
    for (const node of graph.nodes) {
      for (const input of node.inputs) {
        const list = this.consumers.get(input)
        if (list) list.push(node)
        else this.consumers.set(input, [node])
      }
      for (const output of node.outputs) this.producers.set(output, node)
    }
  }

  init(name: string): OnnxTensor {
    const t = this.graph.initializers.get(name)
    if (!t) throw new Error(`ONNX: missing initializer ${name}`)
    return t
  }

  has(name: string): boolean {
    return this.graph.initializers.has(name)
  }

  consumersOf(name: string): Array<OnnxNode> {
    return this.consumers.get(name) ?? []
  }

  producerOf(name: string): OnnxNode | undefined {
    return this.producers.get(name)
  }

  /** The MatMulNBits weight named `{prefix}_MatMul_weight_*`. */
  nbits(prefix: string): NBits {
    const quant = this.init(`${prefix}_MatMul_weight_quant`)
    const scales = this.init(`${prefix}_MatMul_weight_scales`)
    const zpName = `${prefix}_MatMul_weight_zp`
    const zp = this.has(zpName) ? this.init(zpName) : null
    const node = this.consumersOf(quant.name).find(
      (n) => n.opType === "MatMulNBits"
    )
    if (!node) throw new Error(`ONNX: no MatMulNBits reads ${quant.name}`)
    const attr = (key: string) => node.attributes.get(key)?.i
    const N = attr("N")
    const K = attr("K")
    if (
      attr("bits") !== 4 ||
      attr("block_size") !== 32 ||
      N === undefined ||
      K === undefined
    ) {
      throw new Error(`ONNX: ${node.name} is not 4-bit with block size 32`)
    }
    if (
      node.inputs[2] !== scales.name ||
      (node.inputs[3] ?? "") !== (zp?.name ?? "")
    ) {
      throw new Error(
        `ONNX: ${node.name} doesn't use the expected scales / zero points`
      )
    }
    if (node.inputs[4] || node.inputs[5])
      throw new Error(`ONNX: ${node.name} has g_idx or bias inputs`)
    const blocks = K / 32
    checkShape(quant, OnnxDataType.UINT8, N * blocks * 16)
    checkShape(scales, OnnxDataType.FLOAT16, N * blocks)
    if (zp) checkShape(zp, OnnxDataType.UINT8, N * Math.ceil(blocks / 2))
    return { name: prefix, N, K, quant, scales, zp }
  }

  /** What a layer needs, with the graph checked against the layout assumptions. */
  layerPlan(i: number, type: LayerType, config: ModelConfig): LayerPlan {
    const D = config.hiddenSize
    const p = `model.layers.${i}.`
    const norm = (n: string) => this.init(`${p}${n}.weight`)
    const optional = (n: string) =>
      this.has(`${p}${n}.weight`) ? norm(n) : null
    const norms = [
      norm("operator_layernorm"),
      optional("post_operator_layernorm"),
      norm("ffn_layernorm"),
      optional("post_ffn_layernorm")
    ]
    const w = (name: string) => this.nbits(`model_layers_${i}_${name}`)
    const mlp = [w("mlp_gate_proj"), w("mlp_up_proj"), w("mlp_down_proj")]
    if (type === "conv") {
      const inProj = w("conv_in_proj")
      const outProj = w("conv_out_proj")
      if (
        inProj.N !== 3 * D ||
        inProj.K !== D ||
        outProj.N !== D ||
        outProj.K !== D
      ) {
        throw new Error(
          `ONNX: layer ${i} conv projections have unexpected shapes`
        )
      }
      const kernel = this.init(`${p}conv.conv.weight`)
      checkShape(kernel, kernel.dataType, D * config.convKernel)
      if (
        kernel.dims[0] !== D ||
        kernel.dims[kernel.dims.length - 1] !== config.convKernel
      ) {
        throw new Error(`ONNX: ${kernel.name} is [${kernel.dims}]`)
      }
      return {
        type,
        norms,
        matrices: [inProj, outProj, ...mlp],
        extra: [kernel],
        bcx: this.bcxSegments(inProj, D),
        q: { start: 0, rows: 0 }
      }
    }
    const q = w("attn_q_proj")
    const k = w("attn_k_proj")
    const v = w("attn_v_proj")
    const o = w("attn_o_proj")
    const qRows = config.numHeads * config.headDim
    const extra = [
      this.init(`${p}attn.q_norm.layernorm.weight`),
      this.init(`${p}attn.k_norm.layernorm.weight`)
    ]
    this.checkRotary(i, type, config)
    if (q.N === qRows && !config.attentionGate) {
      // LFM2: no output gate.
      return {
        type,
        norms,
        matrices: [q, k, v, o, ...mlp],
        extra,
        q: { start: 0, rows: qRows }
      }
    }
    if (q.N === qRows) {
      const gateWeight = w("attn_gate_proj")
      return {
        type,
        norms,
        matrices: [q, k, v, gateWeight, o, ...mlp],
        extra,
        q: { start: 0, rows: qRows },
        gateWeight
      }
    }
    const { q: qSeg, gate } = this.qGateSegments(q, config)
    return { type, norms, matrices: [q, k, v, o, ...mlp], extra, q: qSeg, gate }
  }

  /** The output of the MatMulNBits node reading `w`. */
  private outputOf(w: NBits): string {
    const node = this.consumersOf(w.quant.name).find(
      (n) => n.opType === "MatMulNBits"
    )
    if (!node) throw new Error(`ONNX: no MatMulNBits reads ${w.quant.name}`)
    return node.outputs[0]
  }

  /** A Split's part sizes, from its input or attribute. */
  private splitSizes(split: OnnxNode): Array<number> {
    const axis = split.attributes.get("axis")?.i ?? 0
    // The channels axis: last, or 1 after a transpose to channels-first.
    if (axis !== -1 && axis !== 2 && axis !== 1)
      throw new Error(`ONNX: ${split.name} doesn't split the channels axis`)
    const input = split.inputs[1]
    if (input) return onnxInt64s(this.init(input))
    const attr = split.attributes.get("split")?.ints
    if (attr) return attr
    throw new Error(`ONNX: ${split.name} has no split sizes`)
  }

  /** q_proj's q and gate rows: the Split output a Sigmoid reads is the gate. */
  private qGateSegments(
    q: NBits,
    config: ModelConfig
  ): { q: Segment; gate: Segment } {
    const split = this.consumersOf(this.outputOf(q)).find(
      (n) => n.opType === "Split"
    )
    if (!split)
      throw new Error(
        `ONNX: ${q.name} (${q.N} rows) is not split into q and gate`
      )
    const sizes = this.splitSizes(split)
    const qRows = config.numHeads * config.headDim
    if (sizes.length !== 2 || sizes[0] + sizes[1] !== q.N)
      throw new Error(`ONNX: ${split.name} sizes [${sizes}]`)
    const gateIndex = split.outputs.findIndex((out) =>
      this.consumersOf(out).some((n) => n.opType === "Sigmoid")
    )
    if (gateIndex < 0) throw new Error(`ONNX: no Sigmoid reads ${split.name}`)
    const starts = [0, sizes[0]]
    const gate = { start: starts[gateIndex], rows: sizes[gateIndex] }
    const qSeg = { start: starts[1 - gateIndex], rows: sizes[1 - gateIndex] }
    if (qSeg.rows !== qRows || gate.rows !== config.numHeads) {
      throw new Error(
        `ONNX: ${split.name} parts are ${qSeg.rows} q rows and ${gate.rows} gate rows`
      )
    }
    return { q: qSeg, gate }
  }

  /**
   * in_proj's B, C, x rows: C is the Split output multiplied by the
   * convolution's output; B and x are multiplied together before it (their
   * order can't matter; HF's chunk order B, x is kept).
   */
  private bcxSegments(inProj: NBits, D: number): Array<Segment> {
    // LFM2's export transposes to channels-first before the split.
    const out = this.outputOf(inProj)
    const transposed = this.consumersOf(out).find((n) => n.opType === "Transpose")
    const split = [out, ...(transposed ? transposed.outputs : [])]
      .flatMap((name) => this.consumersOf(name))
      .find((n) => n.opType === "Split")
    if (!split)
      throw new Error(`ONNX: ${inProj.name} is not split into B, C, x`)
    const sizes = this.splitSizes(split)
    if (sizes.length !== 3 || sizes.some((s) => s !== D))
      throw new Error(`ONNX: ${split.name} sizes [${sizes}]`)
    const conv = this.graph.nodes.find(
      (n) =>
        (n.opType.startsWith("CausalConv") || n.opType === "Conv") &&
        split.outputs.some((o) => this.feeds(o, n))
    )
    if (!conv) throw new Error(`ONNX: no convolution after ${split.name}`)
    // C multiplies the convolution's output (LFM2's export slices it first).
    const convOut = new Set([conv.outputs[0]])
    for (const n of this.consumersOf(conv.outputs[0]))
      if (n.opType === "Slice") convOut.add(n.outputs[0])
    const cIndex = split.outputs.findIndex((out) =>
      this.consumersOf(out).some(
        (n) => n.opType === "Mul" && n.inputs.some((x) => convOut.has(x))
      )
    )
    if (cIndex < 0) throw new Error(`ONNX: can't find C after ${conv.name}`)
    const others = [0, 1, 2].filter((x) => x !== cIndex)
    return [others[0], cIndex, others[1]].map((x) => ({
      start: x * D,
      rows: D
    }))
  }

  /**
   * Whether `name` reaches `node` through at most one Mul, and then at most
   * one Concat (LFM2's export prepends the past conv state).
   */
  private feeds(name: string, node: OnnxNode): boolean {
    if (node.inputs.includes(name)) return true
    return this.consumersOf(name).some(
      (n) =>
        n.opType === "Mul" &&
        n.outputs.some(
          (o) =>
            node.inputs.includes(o) ||
            this.consumersOf(o).some(
              (c) =>
                c.opType === "Concat" &&
                c.outputs.some((x) => node.inputs.includes(x))
            )
        )
    )
  }

  /**
   * GroupQueryAttention's rotary cache must rotate ropeDimsFull (full
   * layers) or ropeDimsSliding dims, non-interleaved: the layout the
   * loaders produce.
   */
  private checkRotary(i: number, type: LayerType, config: ModelConfig): void {
    const prefix = `/model/layers.${i}/`
    const gqa = this.graph.nodes.find(
      (n) => n.opType === "GroupQueryAttention" && n.name.startsWith(prefix)
    )
    if (!gqa) throw new Error(`ONNX: no GroupQueryAttention in layer ${i}`)
    if (
      gqa.attributes.get("do_rotary")?.i !== 1 ||
      (gqa.attributes.get("rotary_interleaved")?.i ?? 0) !== 0
    ) {
      throw new Error(
        `ONNX: ${gqa.name} doesn't apply non-interleaved rotary embeddings`
      )
    }
    const cos = gqa.inputs[7]
    const dims = cos ? this.graph.initializers.get(cos)?.dims : undefined
    const rotated = dims ? dims[dims.length - 1] * 2 : Number.NaN
    const expected =
      type === "full_attention" ? config.ropeDimsFull : config.ropeDimsSliding
    if (rotated !== expected) {
      throw new Error(
        `ONNX: layer ${i} rotates ${rotated} dims per head, expected ${expected}`
      )
    }
  }

  /** The final norm's weight: the norm feeding the LM head. */
  finalNorm(): OnnxTensor {
    const head = this.graph.nodes.find(
      (n) => n.name.startsWith("/lm_head/") && n.opType === "MatMulNBits"
    )
    let name = head?.inputs[0]
    for (let hops = 0; name && hops < 8; hops++) {
      const node = this.producerOf(name)
      if (!node) break
      if (node.opType === "SimplifiedLayerNormalization")
        return this.init(node.inputs[1])
      if (node.opType === "SkipSimplifiedLayerNormalization")
        return this.init(node.inputs[2])
      name = node.inputs[0]
    }
    throw new Error("ONNX: can't find the final norm")
  }

  /** The 4-bit embedding read by GatherBlockQuantized (and the LM head). */
  embedding(D: number): {
    N: number
    quant: OnnxTensor
    scales: OnnxTensor
    zp: OnnxTensor | null
  } {
    const gather = this.graph.nodes.find(
      (n) => n.opType === "GatherBlockQuantized"
    )
    if (!gather) throw new Error("ONNX: no GatherBlockQuantized embedding")
    const attr = (key: string) => gather.attributes.get(key)?.i
    if (
      attr("bits") !== 4 ||
      attr("block_size") !== 32 ||
      attr("quantize_axis") !== 1 ||
      (attr("gather_axis") ?? 0) !== 0
    ) {
      throw new Error(
        `ONNX: ${gather.name} is not 4-bit, block 32, quantized along rows`
      )
    }
    const quant = this.init(gather.inputs[0])
    const scales = this.init(gather.inputs[2])
    const zp = gather.inputs[3] ? this.init(gather.inputs[3]) : null
    const N = quant.dims[0]
    if (quant.dims[1] !== D / 2)
      throw new Error(`ONNX: ${quant.name} is [${quant.dims}]`)
    checkShape(quant, OnnxDataType.UINT8, (N * D) / 2)
    checkShape(scales, OnnxDataType.FLOAT16, (N * D) / 32)
    if (zp) checkShape(zp, OnnxDataType.UINT8, N * Math.ceil(D / 64))
    return { N, quant, scales, zp }
  }
}

function checkShape(t: OnnxTensor, dataType: number, elements: number): void {
  let n = 1
  for (const d of t.dims) n *= d
  if (t.dataType !== dataType || n !== elements) {
    throw new Error(
      `ONNX: ${t.name} is type ${t.dataType} [${t.dims}], expected type ${dataType} with ${elements} elements`
    )
  }
}

type HfRope = {
  rope_theta?: number
  partial_rotary_factor?: number
  rope_type?: string
}

/**
 * Whether attention has LFM3's output gate: its own gate_proj, or q_proj
 * rows that a Sigmoid reads after a split. LFM2 has neither.
 */
function hasAttentionGate(model: OnnxModel, layerTypes: Array<LayerType>) {
  const i = layerTypes.findIndex((t) => t !== "conv")
  if (i < 0) return false
  if (model.has(`model_layers_${i}_attn_gate_proj_MatMul_weight_quant`))
    return true
  const q = model.nbits(`model_layers_${i}_attn_q_proj`)
  const node = model
    .consumersOf(q.quant.name)
    .find((n) => n.opType === "MatMulNBits")
  return !!node && model.consumersOf(node.outputs[0]).some((n) => n.opType === "Split")
}

function onnxConfig(json: unknown, model: OnnxModel): ModelConfig {
  if (typeof json !== "object" || json === null)
    throw new Error("ONNX: config must be the parsed config.json")
  const c = json as Record<string, unknown>
  const num = (key: string, fallback?: number): number => {
    const value = c[key] ?? fallback
    if (typeof value !== "number")
      throw new Error(`config.json: ${key} is missing`)
    return value
  }
  const numLayers = num("num_hidden_layers")
  const hiddenSize = num("hidden_size")
  const numHeads = num("num_attention_heads")
  const headDim = num("head_dim", hiddenSize / numHeads)

  let layerTypes: Array<LayerType>
  if (Array.isArray(c.layer_types)) {
    layerTypes = c.layer_types.map((t) => {
      if (t === "conv" || t === "full_attention" || t === "sliding_attention")
        return t
      throw new Error(`config.json: unknown layer type ${String(t)}`)
    })
  } else if (Array.isArray(c.full_attn_idxs)) {
    const full = new Set(c.full_attn_idxs as Array<number>)
    layerTypes = Array.from({ length: numLayers }, (_, i) =>
      full.has(i) ? "full_attention" : "conv"
    )
  } else throw new Error("config.json: no layer_types")
  if (layerTypes.length !== numLayers)
    throw new Error("config.json: layer_types doesn't match num_hidden_layers")

  const rope = (c.rope_parameters ?? {}) as Record<string, HfRope | undefined>
  const fullRope: HfRope = rope.full_attention ?? (rope as HfRope)
  const slidingRope: HfRope = rope.sliding_attention ?? fullRope
  for (const r of [fullRope, slidingRope]) {
    if (r.rope_type && r.rope_type !== "default")
      throw new Error(`config.json: rope type ${r.rope_type} is not supported`)
  }
  const theta = (r: HfRope) => r.rope_theta ?? num("rope_theta")
  const rotated = (r: HfRope) =>
    Math.round((r.partial_rotary_factor ?? 1) * headDim)

  // Shapes the weights have, over what the config says.
  const gate0 = model.nbits("model_layers_0_mlp_gate_proj")
  const intermediateSize = gate0.N
  if (
    c.intermediate_size !== undefined &&
    c.intermediate_size !== intermediateSize &&
    !c.block_auto_adjust_ff_dim
  ) {
    throw new Error(
      `config.json: intermediate_size ${String(c.intermediate_size)} but the weights have ${intermediateSize}`
    )
  }
  const vocabSize = model.embedding(hiddenSize).N
  if (c.vocab_size !== undefined && c.vocab_size !== vocabSize) {
    throw new Error(
      `config.json: vocab_size ${String(c.vocab_size)} but the embedding has ${vocabSize} rows`
    )
  }

  const tokenId = (key: string): number => {
    const value = c[key]
    if (typeof value === "number") return value
    if (Array.isArray(value) && typeof value[0] === "number") return value[0]
    return -1
  }

  return {
    format: "onnx",
    hiddenSize,
    intermediateSize,
    numLayers,
    numHeads,
    numKvHeads: num("num_key_value_heads", numHeads),
    headDim,
    vocabSize,
    layerTypes,
    slidingWindow: num("sliding_window", 0),
    contextLength: num("max_position_embeddings"),
    ropeThetaFull: theta(fullRope),
    ropeThetaSliding: theta(slidingRope),
    ropeDimsFull: rotated(fullRope),
    ropeDimsSliding: rotated(slidingRope),
    normEps: num("norm_eps", 1e-5),
    convKernel: num("conv_L_cache"),
    attentionGate: hasAttentionGate(model, layerTypes),
    bosTokenId: tokenId("bos_token_id"),
    eosTokenId: tokenId("eos_token_id"),
    padTokenId: tokenId("pad_token_id")
  }
}
