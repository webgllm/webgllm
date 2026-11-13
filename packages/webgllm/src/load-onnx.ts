/**
 * Loads LFM3 and LFM2/LFM2.5 weights from an onnxruntime-genai style ONNX
 * export with
 * 4-bit MatMulNBits projections and a GatherBlockQuantized embedding,
 * such as onnx-community/LFM3-1.7B-ONNX's onnx/model_q4f16.onnx (+ its
 * .onnx_data file).
 *
 * The graph is read to place the parts of fused matrices: the attention
 * q_proj holds q and the output gate (split by a Split node; the gate rows
 * are the ones a Sigmoid reads), and the conv in_proj holds B, C and x (C
 * is the part multiplied after the convolution). Full-attention layers in
 * this export already have each head's rotated dims first (GroupQueryAttention
 * with a rotary cache of ropeDimsFull / 2 frequencies), which is the layout
 * both loaders produce; the loader checks that against the graph.
 *
 * Qwen3.5's export (onnx-community/Qwen3.5-2B-ONNX, decoder_model_merged)
 * has the same kind of weights under other names: Gated DeltaNet layers
 * (`gdn`: in_proj_qkv, in_proj_z, in_proj_b, in_proj_a, out_proj, conv1d,
 * A_neg_exp = -exp(A_log), dt_bias, norm) and full attention whose q_proj
 * holds each head's q then its gate (split per head after a Reshape). Its
 * norms already hold 1 + w. Its lm_head is byte for byte the 4-bit
 * embedding (embed_tokens_q4f16), so the embedding is read from it and
 * the separate embedding file isn't needed.
 *
 * Weights are found by the graph's structure, not their initializer names,
 * which differ between exporters: a projection is the MatMulNBits node
 * under its module path (`/model/layers.0/mlp/gate_proj/`), its weights
 * whatever that node reads. Some exports (onnx-community/LFM2-1.2B-Tool-ONNX)
 * keep the embedding in float16, read by a plain Gather, with the LM head a
 * MatMul over the same tensor transposed: it is quantized to 8 bits at
 * load, as the GGUF loader does a float embedding.
 */
import { halfTable, halvesToFloats, toHalf } from "./half.js"
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
import {
  headPermutation,
  newQuantMatrix,
  packMatMulNBits,
  packMatMulNBitsScales,
  quantizeQ8
} from "./repack.js"
import type {
  AttentionWeights,
  ByteSource,
  ConvWeights,
  LayerType,
  LayerWeights,
  LinearAttentionWeights,
  ModelConfig,
  QuantMatrix,
  WeightPiece
} from "./weights.js"

export type LoadOnnxOptions = {
  /** The .onnx file (graph), as bytes or a source. */
  graph: ByteSource | Uint8Array
  /**
   * The external data file (.onnx_data); or, for a graph whose weights are
   * in several, each file by the name the graph gives it (its `location`).
   */
  data: ByteSource | ((location: string) => ByteSource)
  /** The model's config.json, parsed. */
  config: unknown
  /** Embedding rows converted per step (default 2048). */
  embeddingChunkRows?: number
}

/** The source of a read that needs none (an inline tensor's bytes are in the graph). */
const NO_SOURCE: ByteSource = { size: 0, read: async () => new Uint8Array(0) }

/** A MatMulNBits weight: its initializers and shape. */
export type NBits = {
  name: string
  node: OnnxNode
  N: number
  K: number
  /** 4, or 8 where the export keeps a matrix at 8 bits (blocks of 32 too, zero points a byte each). */
  bits: 4 | 8
  quant: OnnxTensor
  scales: OnnxTensor
  zp: OnnxTensor | null
}

/** A MatMulNBits weight of 4 or 8 bits, in blocks of any multiple of 32 (OnnxModel.wideNBits). */
export type WideNBits = NBits & { blockSize: number }

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
  const plan = new ReadPlan()
  const { config, layers, finalNorm, emb, chunkRows } = planOnnx(
    graphBytes,
    options.config,
    options.embeddingChunkRows ?? 2048,
    // (A tensor in the graph itself has its bytes already: it is in no data file.)
    (read) => plan.add({ ...read, source: typeof data !== "function" ? data : read.bytes ? NO_SOURCE : data(read.location) })
  )
  yield { kind: "config", config }

  const {
    hiddenSize: D,
    intermediateSize: F,
    numHeads,
    numKvHeads,
    headDim,
    numLayers
  } = config

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
   * segment's rows go to destination row `to` (row `to + i` from the
   * segment's row `perm[i]` where it has a `perm`). 4-bit weights stay as
   * they are; if any part is 8-bit the whole matrix is: every part
   * dequantized (scale · (q − zero point)) and quantized to 8 bits per
   * block of 32, symmetric (quantizeQ8), as the GGUF loader stores a
   * matrix that isn't Q4_0 or Q8_0.
   */
  const matrix = async (
    parts: Array<{ w: NBits; segments: Array<Segment & { to: number; perm?: Int32Array }> }>
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
    if (N % 4 !== 0) throw new Error(`ONNX: ${N} rows isn't a multiple of 4`)
    const rowBlocks = K / 32
    if (parts.some((p) => p.w.bits === 8)) {
      const m = newQuantMatrix(N, K, 8, false)
      const row = new Float32Array(K)
      for (const { w, segments } of parts) {
        const quant = await plan.take(w.quant.name)
        const scales = halfBytes(w.scales, await plan.take(w.scales.name))
        const zp = w.zp ? await plan.take(w.zp.name) : null
        for (const { start, rows, to, perm } of segments) {
          for (let i = 0; i < rows; i++) {
            dequantizeNBitsRow(w.bits, K, quant, scales, zp, start + (perm ? perm[i] : i), row)
            quantizeQ8(m, to + i, row, 1)
          }
        }
      }
      return m
    }
    const rowZp = Math.ceil(rowBlocks / 2)
    // The quants stay as MatMulNBits stores them (rows copied into place):
    // the engine rearranges them on the GPU (QuantMatrix.raw).
    const withZeros = parts.some((p) => p.w.zp)
    const m: QuantMatrix = {
      N,
      K,
      bits: 4,
      quants: new Uint32Array(0),
      scales: new Uint16Array(N * rowBlocks),
      zeros: withZeros ? new Uint8Array(N * rowBlocks).fill(8) : null,
      raw: new Uint8Array(N * rowBlocks * 16)
    }
    const raw = m.raw as Uint8Array
    for (const { w, segments } of parts) {
      const quant = await plan.take(w.quant.name)
      const scales = halfBytes(w.scales, await plan.take(w.scales.name))
      const zp = w.zp ? await plan.take(w.zp.name) : null
      const row = rowBlocks * 16
      for (const { start, rows, to, perm } of segments) {
        // Row by row where the rows are reordered, else all at once.
        const runs = perm ? Array.from(perm.subarray(0, rows), (src, i) => ({ from: start + src, at: to + i, count: 1 })) : [{ from: start, at: to, count: rows }]
        for (const { from, at, count } of runs) {
          raw.set(quant.subarray(from * row, (from + count) * row), at * row)
          packMatMulNBitsScales(
            m,
            at,
            scales.subarray(from * rowBlocks * 2),
            zp ? zp.subarray(from * rowZp) : null,
            count
          )
        }
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
    let linear: LinearAttentionWeights | undefined
    if (l.type === "linear_attention") {
      const lc = config.linear
      if (!lc) throw new Error("ONNX: the config has no Gated DeltaNet shapes")
      const [qkvW, zW, bW, aW, outW] = l.matrices as [NBits, NBits, NBits, NBits, NBits]
      const H = lc.numValueHeads
      const inProj = await matrix([whole(qkvW), whole(zW, qkvW.N)])
      const ba = await matrix([whole(bW), whole(aW, H)])
      const outProj = await matrix([whole(outW)])
      const [convT, aT, dtT, normT] = l.extra as [OnnxTensor, OnnxTensor, OnnxTensor, OnnxTensor]
      linear = {
        inProj,
        ba,
        outProj,
        conv: await floats(convT, qkvW.N * lc.convKernel),
        a: await floats(aT, H),
        dtBias: await floats(dtT, H),
        norm: await floats(normT, lc.valueHeadDim)
      }
    } else if (l.type === "conv") {
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
      const gateTo = qRows + 2 * kv
      // Qwen3.5's q_proj: each head's q rows, then its gate rows.
      const perHead = config.gatePerDim
        ? Array.from({ length: numHeads }, (_, h) => [
            { start: 2 * h * headDim, rows: headDim, to: h * headDim },
            {
              start: (2 * h + 1) * headDim,
              rows: headDim,
              to: gateTo + h * headDim
            }
          ]).flat()
        : null
      // A graph with HF's head layout: each head's rotated dims are put
      // first, q's, k's and their norms' (see headPermutation).
      const reorder = l.hfHeads && rotatedDims(l.type, config) < headDim
      const qPerm = reorder ? headPermutation(numHeads, headDim, rotatedDims(l.type, config)) : undefined
      const kPerm = reorder ? headPermutation(numKvHeads, headDim, rotatedDims(l.type, config)) : undefined
      const normPerm = reorder ? headPermutation(1, headDim, rotatedDims(l.type, config)) : undefined
      if (l.hfHeads) checkRotaryCache(await plan.take(`rope@${i}`), l.hfHeads, rotatedDims(l.type, config), i)
      // Rows: q, k, v, gate (the gate from q_proj's own rows, or its own weight).
      const qkvg = await matrix([
        {
          w: qW,
          segments: perHead ?? [
            { ...l.q, to: 0, ...(qPerm && { perm: qPerm }) },
            ...(l.gate ? [{ ...l.gate, to: gateTo }] : [])
          ]
        },
        { w: kW, segments: [{ start: 0, rows: kW.N, to: qRows, ...(kPerm && { perm: kPerm }) }] },
        whole(vW, qRows + kv),
        ...(l.gateWeight ? [whole(l.gateWeight, gateTo)] : [])
      ])
      const o = await matrix([whole(oW)])
      const reordered = (values: Float32Array) => (normPerm ? Float32Array.from(normPerm, (src) => values[src]) : values)
      const qNorm = reordered(await floats(l.extra[0], headDim))
      const kNorm = reordered(await floats(l.extra[1], headDim))
      attention = { qkvg, o, qNorm, kNorm }
    }
    const [gateW, upW, downW] = l.matrices.slice(-3) as [NBits, NBits, NBits]
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
    if (linear) value.linear = linear
    yield { kind: "layer", layer: i, value }
  }

  yield { kind: "finalNorm", value: await floats(finalNorm, D) }

  // In chunks of rows, so the embedding is never all in JavaScript memory.
  for (let r = 0; r < emb.N; r += chunkRows) {
    const rows = Math.min(chunkRows, emb.N - r)
    if (emb.kind === "float") {
      // Floats: quantized to 8 bits (as the GGUF loader does a float embedding).
      const bytes = await plan.take(`floats@${r}`)
      const m = newQuantMatrix(rows, D, 8, false)
      quantizeEmbeddingRows(m, bytes, emb.weight.dataType, rows)
      plan.release(bytes)
      yield { kind: "embedding", value: m, row0: r, totalRows: emb.N }
      continue
    }
    const quant = await plan.take(`quant@${r}`)
    const scales = await plan.take(`scales@${r}`)
    const zp = emb.zp ? await plan.take(`zp@${r}`) : null
    if (emb.kind === "q8") {
      // 8-bit rows with zero points: dequantized, and quantized like a float embedding.
      const m = newQuantMatrix(rows, D, 8, false)
      const row = new Float32Array(D)
      for (let i = 0; i < rows; i++) {
        dequantizeNBitsRow(8, D, quant, scales, zp, i, row)
        quantizeQ8(m, i, row, 1)
      }
      yield { kind: "embedding", value: m, row0: r, totalRows: emb.N }
      continue
    }
    const m = newQuantMatrix(rows, D, 4, emb.zp !== null)
    packMatMulNBits(m, 0, quant, scales, zp, rows)
    yield { kind: "embedding", value: m, row0: r, totalRows: emb.N }
  }
}

/**
 * A scales tensor's bytes as float16 (the QuantMatrix's scales): as they
 * are, or float32 scales (a `q4` export's, where `q4f16` has float16)
 * rounded to the nearest half.
 */
export function halfBytes(t: OnnxTensor, bytes: Uint8Array): Uint8Array {
  if (t.dataType === OnnxDataType.FLOAT16) return bytes
  if (t.dataType !== OnnxDataType.FLOAT)
    throw new Error(`ONNX: ${t.name} has data type ${t.dataType}, expected float16 or float`)
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength)
  const out = new Uint8Array(bytes.length / 2)
  for (let i = 0; i < bytes.length / 4; i++) {
    const h = toHalf(view.getFloat32(i * 4, true))
    out[2 * i] = h & 255
    out[2 * i + 1] = h >> 8
  }
  return out
}

/**
 * Row `n` of a MatMulNBits (or GatherBlockQuantized) weight as floats:
 * scale · (q − zero point) per block of 32. `quant` is [rows][K / 32]
 * [bits · 4 bytes] (4-bit: element 2i of a block in byte i's low nibble),
 * `scales` float16 bytes, `zp` packed two blocks a byte for 4 bits (null:
 * 8), a byte a block for 8 (null: 128).
 */
export function dequantizeNBitsRow(
  bits: 4 | 8,
  K: number,
  quant: Uint8Array,
  scales: Uint8Array,
  zp: Uint8Array | null,
  n: number,
  out: Float32Array
): void {
  const blocks = K / 32
  const half = halfTable()
  const zpStride = bits === 4 ? Math.ceil(blocks / 2) : blocks
  for (let b = 0; b < blocks; b++) {
    const sp = (n * blocks + b) * 2
    const scale = half[scales[sp] | (scales[sp + 1] << 8)]
    const o = b * 32
    if (bits === 8) {
      const zero = zp ? zp[n * zpStride + b] : 128
      const q = (n * blocks + b) * 32
      for (let j = 0; j < 32; j++) out[o + j] = scale * (quant[q + j] - zero)
    } else {
      const zero = zp ? (zp[n * zpStride + (b >> 1)] >> ((b & 1) << 2)) & 15 : 8
      const q = (n * blocks + b) * 16
      for (let j = 0; j < 16; j++) {
        const byte = quant[q + j]
        out[o + 2 * j] = scale * ((byte & 15) - zero)
        out[o + 2 * j + 1] = scale * ((byte >> 4) - zero)
      }
    }
  }
}

/** The dims RoPE rotates per head in a layer of this type. */
const rotatedDims = (type: LayerType, config: ModelConfig) => (type === "full_attention" ? config.ropeDimsFull : config.ropeDimsSliding)

/**
 * Checks a rotary sin cache's row for position 1 (headDim / 2 frequencies)
 * against HF's layout: the first `rotated / 2` frequencies turn (sin > 0),
 * the rest don't (sin 0).
 */
function checkRotaryCache(row: Uint8Array, cache: OnnxTensor, rotated: number, layer: number): void {
  const sin = cache.dataType === OnnxDataType.FLOAT16 ? halvesToFloats(row) : new Float32Array(row.slice().buffer)
  for (let j = 0; j < sin.length; j++) {
    if (j < rotated / 2 ? sin[j] > 0 : sin[j] === 0) continue
    throw new Error(`ONNX: layer ${layer}'s rotary cache doesn't rotate its first ${rotated} dims per head`)
  }
}

/**
 * Float16 or float32 rows quantized to 8 bits into `m` (rows from 0), one
 * row at a time through one row of float32 scratch: the same values as
 * converting the whole chunk first (quantizeQ8 works per block), without
 * staging the chunk in float32.
 */
export function quantizeEmbeddingRows(
  m: QuantMatrix,
  bytes: Uint8Array,
  dataType: number,
  rows: number
): void {
  const K = m.K
  const size = onnxElementSize(dataType)
  if (dataType !== OnnxDataType.FLOAT16 && dataType !== OnnxDataType.FLOAT)
    throw new Error(`ONNX: embedding data type ${dataType}, expected float16 or float`)
  if (bytes.length < rows * K * size)
    throw new Error(`ONNX: ${bytes.length} embedding bytes for ${rows} rows of ${K}`)
  const row = new Float32Array(K)
  const half = halfTable()
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength)
  for (let i = 0; i < rows; i++) {
    let p = i * K * size
    if (size === 2)
      for (let k = 0; k < K; k++, p += 2) row[k] = half[bytes[p] | (bytes[p + 1] << 8)]
    else for (let k = 0; k < K; k++, p += 4) row[k] = view.getFloat32(p, true)
    quantizeQ8(m, i, row, 1)
  }
}

/** A read the loader will make: from the external data file at `location`, or inline bytes. */
type OnnxRead = { key: string; location: string; offset: number; length: number; bytes?: Uint8Array }

/**
 * Checks the graph against what the loader takes (4-bit MatMulNBits with
 * blocks of 32, the layer layout, the config's shapes) and lists every read
 * the load makes, in order, without reading the external data.
 */
function planOnnx(
  graphBytes: Uint8Array | OnnxGraph,
  configJson: unknown,
  chunkRows: number,
  add: (read: OnnxRead) => void
) {
  const graph =
    graphBytes instanceof Uint8Array ? parseOnnxModel(graphBytes) : graphBytes
  const model = new OnnxModel(graph)
  const config = onnxConfig(configJson, model)
  const D = config.hiddenSize
  const F = config.intermediateSize
  const kv = config.numKvHeads * config.headDim

  // Check the graph against what the layout assumes, and find the fused parts.
  const layers = config.layerTypes.map((type, i) =>
    model.layerPlan(i, type, config)
  )
  layers.forEach((l, i) => {
    if (l.type !== "conv" && l.type !== "linear_attention") {
      const [, kW, vW] = l.matrices as [NBits, NBits, NBits]
      if (kW.N !== kv || vW.N !== kv)
        throw new Error(`ONNX: layer ${i} k/v projections have ${kW.N}/${vW.N} rows`)
    }
    const [gateW, upW, downW] = l.matrices.slice(-3) as [NBits, NBits, NBits]
    if (gateW.N !== F || upW.N !== F || downW.K !== F)
      throw new Error(`ONNX: layer ${i} MLP shapes don't match`)
  })

  const want = (t: OnnxTensor | null) => {
    if (!t) return
    const length = tensorBytes(t)
    if (t.raw) {
      if (t.raw.length !== length)
        throw new Error(
          `ONNX: ${t.name} has ${t.raw.length} bytes, expected ${length}`
        )
      add({ key: t.name, location: "", offset: 0, length, bytes: t.raw })
      return
    }
    if (!t.external) throw new Error(`ONNX: ${t.name} has no data`)
    if (t.external.length !== length)
      throw new Error(
        `ONNX: ${t.name} has ${t.external.length} bytes, expected ${length}`
      )
    add({ key: t.name, location: t.external.location, offset: t.external.offset, length })
  }
  const wantNBits = (w: NBits) => {
    want(w.quant)
    want(w.scales)
    want(w.zp)
  }
  // In the order the layer code takes them: norms, operator matrices,
  // conv kernel or q/k norms, MLP matrices.
  for (const [i, l] of layers.entries()) {
    for (const t of l.norms) want(t)
    if (l.hfHeads) {
      // The sin cache's row for position 1, to check the head layout against (checkRotaryCache).
      const cache = l.hfHeads
      const row = cache.dims[1] * onnxElementSize(cache.dataType)
      if (!cache.external || cache.external.length !== tensorBytes(cache)) throw new Error(`ONNX: ${cache.name} is expected in the external data file`)
      add({ key: `rope@${i}`, location: cache.external.location, offset: cache.external.offset + row, length: row })
    }
    for (const w of l.matrices.slice(0, -3)) wantNBits(w)
    for (const t of l.extra) want(t)
    for (const w of l.matrices.slice(-3)) wantNBits(w)
  }
  const finalNorm = model.finalNorm()
  want(finalNorm)

  const emb = model.embedding(D)
  // Matrices hold rows in groups of 4 (QuantMatrix); padding would add
  // tokens the head scores, so such a vocabulary is refused here, before
  // the weights download.
  if (emb.N % 4 !== 0)
    throw new Error(`ONNX: the embedding has ${emb.N} rows, not a multiple of 4`)
  if (chunkRows < emb.N && chunkRows % 4 !== 0)
    throw new Error(`ONNX: embedding chunks of ${chunkRows} rows aren't a multiple of 4`)
  const blocks = D / 32
  const zpStride = Math.ceil(blocks / 2)
  for (let r = 0; r < emb.N; r += chunkRows) {
    const rows = Math.min(chunkRows, emb.N - r)
    const ext = (t: OnnxTensor, rowBytes: number, key: string) => {
      if (!t.external)
        throw new Error(`ONNX: ${t.name} is expected in the external data file`)
      // The declared span must hold every row the reads take.
      if (t.external.length !== tensorBytes(t) || rowBytes * emb.N !== tensorBytes(t))
        throw new Error(
          `ONNX: ${t.name} has ${t.external.length} bytes, expected ${rowBytes * emb.N}`
        )
      add({
        key: `${key}@${r}`,
        location: t.external.location,
        offset: t.external.offset + r * rowBytes,
        length: rows * rowBytes
      })
    }
    if (emb.kind === "float") {
      ext(emb.weight, D * onnxElementSize(emb.weight.dataType), "floats")
      continue
    }
    ext(emb.quant, blocks * (emb.kind === "q8" ? 32 : 16), "quant")
    ext(emb.scales, blocks * 2, "scales")
    if (emb.zp) ext(emb.zp, emb.kind === "q8" ? blocks : zpStride, "zp")
  }
  return { config, layers, finalNorm, emb, chunkRows }
}

/**
 * Whether the loader takes this graph (with this config.json), reading only
 * the graph: throws the loader's own error if not, else returns the
 * external data files its weights are in (their `location`s).
 */
export function checkOnnxGraph(graphBytes: Uint8Array, configJson: unknown): { dataFiles: Array<string> } {
  return checkOnnxModel(parseOnnxModel(graphBytes), configJson)
}

/** checkOnnxGraph on a graph already parsed. */
export function checkOnnxModel(graph: OnnxGraph, configJson: unknown): { dataFiles: Array<string> } {
  const files = new Set<string>()
  planOnnx(graph, configJson, Number.MAX_SAFE_INTEGER, (read) => {
    if (!read.bytes) files.add(read.location)
  })
  return { dataFiles: [...files] }
}

export function tensorBytes(t: OnnxTensor): number {
  let n = 1
  for (const d of t.dims) n *= d
  return Math.ceil(n * onnxElementSize(t.dataType))
}

type LayerPlan = {
  type: LayerType
  /** operator, post-operator, ffn, post-ffn norms (the post ones: LFM3 only). */
  norms: Array<OnnxTensor | null>
  /**
   * conv: in_proj, out_proj; attention: q_proj, k, v, o (+ gate_proj);
   * linear: in_proj_qkv, z, b, a, out_proj; then mlp gate, up, down.
   */
  matrices: Array<NBits>
  /** conv: the kernel; attention: q_norm, k_norm; linear: conv1d, A, dt_bias, norm. */
  extra: Array<OnnxTensor>
  /** conv: in_proj rows of B, C, x. */
  bcx?: Array<Segment>
  /** attention: q rows within q_proj, and the gate's when q_proj holds it. */
  q: Segment
  gate?: Segment
  /** attention: a separate gate projection, when the export has one. */
  gateWeight?: NBits
  /**
   * attention: the graph rotates q and k with RotaryEmbedding nodes over
   * HF's head layout (dims j and j + headDim / 2), not the engine's
   * (rotated dims first): the rotary sin cache, for the loader to check
   * and reorder the heads by.
   */
  hfHeads?: OnnxTensor
}

/** Lookups over the graph. */
export class OnnxModel {
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

  /** A constant tensor: an initializer, or a Constant node's value. */
  constant(name: string): OnnxTensor {
    const t = this.graph.initializers.get(name)
    if (t) return t
    const node = this.producerOf(name)
    const value = node?.opType === "Constant" ? node.attributes.get("value")?.t : undefined
    if (!value) throw new Error(`ONNX: ${name} is not a constant`)
    return value
  }

  consumersOf(name: string): Array<OnnxNode> {
    return this.consumers.get(name) ?? []
  }

  producerOf(name: string): OnnxNode | undefined {
    return this.producers.get(name)
  }

  /**
   * The MatMulNBits node of the module at `path` (e.g.
   * `/model/layers.0/mlp/gate_proj/`), or undefined. Either of two signals
   * names it: the node's own name (exporters prefix it with the module
   * path), or the 4-bit weight it reads, whose name is the module path in
   * any spelling (`model_layers_0_mlp_gate_proj_MatMul_weight_quant`,
   * `model.layers.0.mlp.gate_proj.MatMul.weight_Q4`). Its weights are then
   * whatever the node reads. Two nodes for one module is an error.
   */
  private nbitsNode(path: string): OnnxNode | undefined {
    const spelled = (name: string) => name.toLowerCase().replace(/[^a-z0-9]+/g, "_")
    const module = spelled(path).replace(/^_/, "")
    const nodes = this.graph.nodes.filter(
      (n) =>
        n.opType === "MatMulNBits" &&
        (n.name.startsWith(path) ||
          (this.has(n.inputs[1]) && spelled(n.inputs[1]).startsWith(module)))
    )
    if (nodes.length > 1)
      throw new Error(`ONNX: ${nodes.length} MatMulNBits nodes under ${path}`)
    return nodes[0]
  }

  hasNBits(path: string): boolean {
    return this.nbitsNode(path) !== undefined
  }

  /**
   * The MatMulNBits weight under `path`, as the vision loaders take one:
   * 4-bit in blocks of 32 or 8-bit in blocks of a multiple of 32, scales in
   * float16 or float32, zero points or none.
   */
  wideNBits(path: string): WideNBits {
    const node = this.nbitsNode(path)
    if (!node) throw new Error(`ONNX: no MatMulNBits node under ${path}`)
    const quant = this.init(node.inputs[1])
    const scales = this.init(node.inputs[2])
    const zp = node.inputs[3] ? this.init(node.inputs[3]) : null
    const attr = (key: string) => node.attributes.get(key)?.i
    const N = attr("N")
    const K = attr("K")
    const bits = attr("bits")
    const blockSize = attr("block_size")
    if (N === undefined || K === undefined || blockSize === undefined || (bits !== 4 && bits !== 8) || blockSize % 32 || (bits === 4 && blockSize !== 32))
      throw new Error(`ONNX: ${node.name} is not 4-bit in blocks of 32 or 8-bit in blocks of a multiple of 32`)
    if (node.inputs[4] || node.inputs[5]) throw new Error(`ONNX: ${node.name} has g_idx or bias inputs`)
    const blocks = Math.ceil(K / blockSize)
    checkShape(quant, OnnxDataType.UINT8, N * blocks * ((blockSize * bits) / 8))
    if (scales.dataType !== OnnxDataType.FLOAT16 && scales.dataType !== OnnxDataType.FLOAT)
      throw new Error(`ONNX: ${scales.name} has data type ${scales.dataType}, expected float16 or float`)
    checkShape(scales, scales.dataType, N * blocks)
    if (zp) checkShape(zp, OnnxDataType.UINT8, N * (bits === 4 ? Math.ceil(blocks / 2) : blocks))
    return { name: path, node, N, K, bits, blockSize, quant, scales, zp }
  }

  /**
   * The MatMulNBits weight under `path` (see nbitsNode), as the text
   * loader takes one: blocks of 32, 4-bit or 8-bit.
   */
  nbits(path: string): NBits {
    const w = this.wideNBits(path)
    if (w.blockSize !== 32) throw new Error(`ONNX: ${w.node.name} is not in blocks of 32`)
    return w
  }

  /** What a layer needs, with the graph checked against the layout assumptions. */
  layerPlan(i: number, type: LayerType, config: ModelConfig): LayerPlan {
    if (config.linear) return this.qwenLayerPlan(i, type, config)
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
    const w = (name: string) => this.nbits(`/model/layers.${i}/${name}/`)
    const mlp = [w("mlp/gate_proj"), w("mlp/up_proj"), w("mlp/down_proj")]
    if (type === "conv") {
      const inProj = w("conv/in_proj")
      const outProj = w("conv/out_proj")
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
    const q = w("attn/q_proj")
    const k = w("attn/k_proj")
    const v = w("attn/v_proj")
    const o = w("attn/o_proj")
    const qRows = config.numHeads * config.headDim
    const extra = [
      this.init(`${p}attn.q_norm.layernorm.weight`),
      this.init(`${p}attn.k_norm.layernorm.weight`)
    ]
    const hfHeads = this.checkRotary(i, type, config)
    const heads = hfHeads ? { hfHeads } : {}
    if (q.N === qRows && !config.attentionGate) {
      // LFM2: no output gate.
      return {
        type,
        norms,
        matrices: [q, k, v, o, ...mlp],
        extra,
        q: { start: 0, rows: qRows },
        ...heads
      }
    }
    if (q.N === qRows) {
      const gateWeight = w("attn/gate_proj")
      return {
        type,
        norms,
        matrices: [q, k, v, gateWeight, o, ...mlp],
        extra,
        q: { start: 0, rows: qRows },
        gateWeight,
        ...heads
      }
    }
    const { q: qSeg, gate } = this.qGateSegments(q, config)
    return { type, norms, matrices: [q, k, v, o, ...mlp], extra, q: qSeg, gate, ...heads }
  }

  /** A Qwen3.5 layer (Gated DeltaNet or gated full attention). */
  private qwenLayerPlan(
    i: number,
    type: LayerType,
    config: ModelConfig
  ): LayerPlan {
    const D = config.hiddenSize
    const p = `model.layers.${i}.`
    const norms = [
      this.init(`${p}input_layernorm.weight`),
      null,
      this.init(`${p}post_attention_layernorm.weight`),
      null
    ]
    const w = (name: string) => this.nbits(`/model/layers.${i}/${name}/`)
    const mlp = [w("mlp/gate_proj"), w("mlp/up_proj"), w("mlp/down_proj")]
    const l = config.linear
    if (!l) throw new Error("ONNX: the config has no Gated DeltaNet shapes")
    if (type === "linear_attention") {
      const keyDim = l.numKeyHeads * l.keyHeadDim
      const valueDim = l.numValueHeads * l.valueHeadDim
      const qkv = w("gdn/in_proj_qkv")
      const z = w("gdn/in_proj_z")
      const b = w("gdn/in_proj_b")
      const a = w("gdn/in_proj_a")
      const out = w("gdn/out_proj")
      const H = l.numValueHeads
      if (
        qkv.N !== 2 * keyDim + valueDim ||
        z.N !== valueDim ||
        b.N !== H ||
        a.N !== H ||
        out.N !== D ||
        out.K !== valueDim
      )
        throw new Error(`ONNX: layer ${i} Gated DeltaNet shapes don't match`)
      const conv = this.init(`${p}gdn.conv1d.weight`)
      if (conv.dims[0] !== qkv.N || conv.dims[conv.dims.length - 1] !== l.convKernel)
        throw new Error(`ONNX: ${conv.name} is [${conv.dims}]`)
      return {
        type,
        norms,
        matrices: [qkv, z, b, a, out, ...mlp],
        extra: [
          conv,
          this.init(`${p}gdn.A_neg_exp`),
          this.init(`${p}gdn.dt_bias`),
          this.init(`${p}gdn.norm.weight`)
        ],
        q: { start: 0, rows: 0 }
      }
    }
    const q = w("attn/q_proj")
    if (q.N !== 2 * config.numHeads * config.headDim)
      throw new Error(`ONNX: layer ${i} q_proj has ${q.N} rows, expected q and gate`)
    // RoPE: non-interleaved over the first ropeDimsFull dims of each head.
    const rope = this.graph.nodes.find(
      (n) =>
        n.opType === "RotaryEmbedding" && n.name.startsWith(`/model/layers.${i}/`)
    )
    if (
      !rope ||
      rope.attributes.get("rotary_embedding_dim")?.i !== config.ropeDimsFull ||
      (rope.attributes.get("interleaved")?.i ?? 0) !== 0
    )
      throw new Error(
        `ONNX: layer ${i} doesn't rotate the first ${config.ropeDimsFull} dims, non-interleaved`
      )
    return {
      type,
      norms,
      matrices: [q, w("attn/k_proj"), w("attn/v_proj"), w("attn/o_proj"), ...mlp],
      extra: [
        this.init(`${p}attn.q_norm.layernorm.weight`),
        this.init(`${p}attn.k_norm.layernorm.weight`)
      ],
      q: { start: 0, rows: config.numHeads * config.headDim }
    }
  }

  /** The output of the MatMulNBits node reading `w`. */
  private outputOf(w: NBits): string {
    return w.node.outputs[0]
  }

  /** A Split's part sizes, from its input or attribute. */
  private splitSizes(split: OnnxNode): Array<number> {
    const axis = split.attributes.get("axis")?.i ?? 0
    // The channels axis: last, or 1 after a transpose to channels-first.
    if (axis !== -1 && axis !== 2 && axis !== 1)
      throw new Error(`ONNX: ${split.name} doesn't split the channels axis`)
    const input = split.inputs[1]
    if (input) return onnxInt64s(this.constant(input))
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
   * Checks how a layer's attention rotates q and k, non-interleaved:
   * - by GroupQueryAttention itself, whose rotary cache must rotate
   *   ropeDimsFull (full layers) or ropeDimsSliding dims: each head's
   *   rotated dims first, the layout the loaders produce;
   * - or by RotaryEmbedding nodes before it, over all of each head (a
   *   cache of headDim / 2 frequencies, HF's layout): returns their sin
   *   cache, for the loader to check and reorder the heads by (see
   *   LayerPlan.hfHeads).
   */
  private checkRotary(i: number, type: LayerType, config: ModelConfig): OnnxTensor | undefined {
    const prefix = `/model/layers.${i}/`
    const gqa = this.graph.nodes.find(
      (n) => n.opType === "GroupQueryAttention" && n.name.startsWith(prefix)
    )
    if (!gqa) throw new Error(`ONNX: no GroupQueryAttention in layer ${i}`)
    const expected =
      type === "full_attention" ? config.ropeDimsFull : config.ropeDimsSliding
    if (gqa.attributes.get("do_rotary")?.i !== 1) {
      const rotary = this.graph.nodes.filter((n) => n.opType === "RotaryEmbedding" && n.name.startsWith(`${prefix}attn/`))
      const sin = rotary[0]?.inputs[3]
      const cache = sin ? this.graph.initializers.get(sin) : undefined
      if (
        rotary.length !== 2 ||
        !cache ||
        rotary.some((n) => n.inputs[3] !== sin || (n.attributes.get("interleaved")?.i ?? 0) !== 0 || (n.attributes.get("rotary_embedding_dim")?.i ?? 0) !== 0) ||
        cache.dims.length !== 2 ||
        cache.dims[1] * 2 !== config.headDim ||
        !gqa.inputs.slice(0, 2).every((input) => rotary.some((n) => n.outputs[0] === input))
      )
        throw new Error(`ONNX: layer ${i}'s attention doesn't rotate q and k over each head, non-interleaved`)
      return cache
    }
    if ((gqa.attributes.get("rotary_interleaved")?.i ?? 0) !== 0) {
      throw new Error(
        `ONNX: ${gqa.name} doesn't apply non-interleaved rotary embeddings`
      )
    }
    const cos = gqa.inputs[7]
    const dims = cos ? this.graph.initializers.get(cos)?.dims : undefined
    const rotated = dims ? dims[dims.length - 1] * 2 : Number.NaN
    if (rotated !== expected) {
      throw new Error(
        `ONNX: layer ${i} rotates ${rotated} dims per head, expected ${expected}`
      )
    }
    return undefined
  }

  /**
   * The LM head: the MatMulNBits or MatMul node under `/lm_head/`, or the
   * one that writes the graph's `logits` (through a Cast, in a float16
   * graph).
   */
  private lmHead(): OnnxNode {
    const isHead = (n: OnnxNode | undefined): n is OnnxNode => n?.opType === "MatMulNBits" || n?.opType === "MatMul"
    const heads = this.graph.nodes.filter((n) => n.name.startsWith("/lm_head/") && isHead(n))
    if (heads.length === 1) return heads[0]
    if (heads.length === 0) {
      let node = this.producerOf("logits")
      if (node?.opType === "Cast") node = this.producerOf(node.inputs[0])
      if (isHead(node)) return node
    }
    throw new Error(`ONNX: ${heads.length} LM head MatMul nodes`)
  }

  /** The final norm's weight: the norm feeding the LM head. */
  finalNorm(): OnnxTensor {
    let name: string | undefined = this.lmHead().inputs[0]
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

  /**
   * The embedding (also the output head):
   * - 4-bit, read by GatherBlockQuantized; or 8-bit with zero points
   *   (LFM3-VL's), which the loader quantizes again as it does floats;
   * - floats read by a plain Gather, when the LM head is a MatMul over the
   *   same tensor transposed (tied): the loader quantizes it to 8 bits, as
   *   it does a GGUF's float embedding;
   * - in a decoder-only graph, the tied head itself: 4-bit (Qwen3.5's), or
   *   a float MatMul over the embedding transposed (LFM2-VL's decoders,
   *   which read embeddings), quantized to 8 bits too.
   */
  embedding(D: number): OnnxEmbedding {
    const gather = this.graph.nodes.find(
      (n) => n.opType === "GatherBlockQuantized"
    )
    if (gather) {
      const attr = (key: string) => gather.attributes.get(key)?.i
      const bits = attr("bits")
      if (
        (bits !== 4 && bits !== 8) ||
        attr("block_size") !== 32 ||
        attr("quantize_axis") !== 1 ||
        (attr("gather_axis") ?? 0) !== 0
      ) {
        throw new Error(
          `ONNX: ${gather.name} is not 4-bit or 8-bit, block 32, quantized along rows`
        )
      }
      // (An export may reshape the head's [rows][blocks][bytes] to rows for the Gather.)
      const reshaped = this.has(gather.inputs[0]) ? undefined : this.producerOf(gather.inputs[0])
      const quant = this.init(reshaped?.opType === "Reshape" ? reshaped.inputs[0] : gather.inputs[0])
      const scales = this.init(gather.inputs[2])
      const zp = gather.inputs[3] ? this.init(gather.inputs[3]) : null
      const N = quant.dims[0]
      checkShape(quant, OnnxDataType.UINT8, (N * D * bits) / 8)
      checkShape(scales, OnnxDataType.FLOAT16, (N * D) / 32)
      if (zp) checkShape(zp, OnnxDataType.UINT8, bits === 4 ? N * Math.ceil(D / 64) : (N * D) / 32)
      // The engine's head is the embedding: a quantized head is taken as
      // tied (it is in these exports); a float one has its own weights.
      const head = this.lmHead()
      if (head.opType === "MatMul")
        throw new Error(
          "ONNX: the LM head is a float MatMul with its own weights, not the embedding (only a tied head is supported)"
        )
      if (bits === 8) {
        if (head.inputs[1] !== quant.name) throw new Error(`ONNX: the LM head isn't ${quant.name} (only a tied head is supported)`)
        return { kind: "q8", N, quant, scales, zp }
      }
      return { kind: "q4", N, quant, scales, zp }
    }
    const floatGather = this.graph.nodes.find(
      (n) =>
        n.opType === "Gather" &&
        n.inputs[1] === "input_ids" &&
        this.has(n.inputs[0])
    )
    if (floatGather) {
      const weight = this.init(floatGather.inputs[0])
      if (
        weight.dataType !== OnnxDataType.FLOAT16 &&
        weight.dataType !== OnnxDataType.FLOAT
      )
        throw new Error(
          `ONNX: ${weight.name} has data type ${weight.dataType}, expected float16 or float`
        )
      const [N, K] = weight.dims
      if (weight.dims.length !== 2 || K !== D)
        throw new Error(`ONNX: ${weight.name} is [${weight.dims}]`)
      // Tied: the head multiplies by this same tensor, transposed.
      const head = this.lmHead()
      const transpose = this.producerOf(head.inputs[1])
      const perm = transpose?.attributes.get("perm")?.ints
      if (
        head.opType !== "MatMul" ||
        transpose?.opType !== "Transpose" ||
        transpose.inputs[0] !== weight.name ||
        (perm && (perm[0] !== 1 || perm[1] !== 0))
      )
        throw new Error(
          `ONNX: the LM head isn't ${weight.name} (only a tied head is supported)`
        )
      return { kind: "float", N, weight }
    }
    const head = this.lmHead()
    const transposed = head.opType === "MatMul" ? this.producerOf(head.inputs[1]) : undefined
    if (transposed?.opType === "Transpose" && this.has(transposed.inputs[0])) {
      // A decoder that reads embeddings (inputs_embeds: LFM2-VL's), its
      // head a MatMul over the float embedding transposed: the tied
      // embedding, quantized to 8 bits like any float one.
      const weight = this.init(transposed.inputs[0])
      const perm = transposed.attributes.get("perm")?.ints
      if (weight.dataType !== OnnxDataType.FLOAT16 && weight.dataType !== OnnxDataType.FLOAT)
        throw new Error(`ONNX: ${weight.name} has data type ${weight.dataType}, expected float16 or float`)
      if (weight.dims.length !== 2 || weight.dims[1] !== D || (perm && (perm[0] !== 1 || perm[1] !== 0)))
        throw new Error(`ONNX: the LM head's ${weight.name} is [${weight.dims}], not the embedding transposed`)
      return { kind: "float", N: weight.dims[0], weight }
    }
    if (this.hasNBits("/lm_head/")) {
      // A decoder-only graph (Qwen3.5's): the tied head is the embedding.
      const head = this.nbits("/lm_head/")
      if (head.K !== D) throw new Error(`ONNX: lm_head has K ${head.K}`)
      return {
        kind: "q4",
        N: head.N,
        quant: head.quant,
        scales: head.scales,
        zp: head.zp
      }
    }
    throw new Error("ONNX: no embedding the loader reads")
  }
}

/** The embedding as the graph has it (OnnxModel.embedding). */
type OnnxEmbedding =
  | {
      /** 4-bit rows, kept; or 8-bit rows with zero points, quantized again like floats. */
      kind: "q4" | "q8"
      N: number
      quant: OnnxTensor
      scales: OnnxTensor
      zp: OnnxTensor | null
    }
  | { kind: "float"; N: number; weight: OnnxTensor }

export function checkShape(t: OnnxTensor, dataType: number, elements: number): void {
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
  if (model.hasNBits(`/model/layers.${i}/attn/gate_proj/`)) return true
  const q = model.nbits(`/model/layers.${i}/attn/q_proj/`)
  return model.consumersOf(q.node.outputs[0]).some((n) => n.opType === "Split")
}

function onnxConfig(json: unknown, model: OnnxModel): ModelConfig {
  if (typeof json !== "object" || json === null)
    throw new Error("ONNX: config must be the parsed config.json")
  // Qwen3.5's multimodal config keeps the language model's in text_config.
  const outer = json as Record<string, unknown>
  const c = (
    typeof outer.text_config === "object" && outer.text_config !== null
      ? outer.text_config
      : outer
  ) as Record<string, unknown>
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
      if (
        t === "conv" ||
        t === "full_attention" ||
        t === "sliding_attention" ||
        t === "linear_attention"
      )
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
  const gate0 = model.nbits("/model/layers.0/mlp/gate_proj/")
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

  const hasLinear = layerTypes.includes("linear_attention")
  const linear = hasLinear
    ? {
        numKeyHeads: num("linear_num_key_heads"),
        numValueHeads: num("linear_num_value_heads"),
        keyHeadDim: num("linear_key_head_dim"),
        valueHeadDim: num("linear_value_head_dim"),
        convKernel: num("linear_conv_kernel_dim")
      }
    : undefined
  if (linear && linear.numKeyHeads !== linear.numValueHeads)
    throw new Error(
      "ONNX: Gated DeltaNet with more value heads than key heads isn't supported"
    )

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
    // (8,192 where a config doesn't say.)
    contextLength: num("max_position_embeddings", 8192),
    ropeThetaFull: theta(fullRope),
    ropeThetaSliding: theta(slidingRope),
    ropeDimsFull: rotated(fullRope),
    ropeDimsSliding: rotated(slidingRope),
    normEps: num("norm_eps", num("rms_norm_eps", 1e-5)),
    convKernel: linear ? linear.convKernel : num("conv_L_cache"),
    attentionGate: linear ? c.attn_output_gate === true : hasAttentionGate(model, layerTypes),
    // Qwen3.5: its RoPE frequencies spread over the rotated dims, its gate per dim.
    ...(linear && {
      linear,
      gatePerDim: c.attn_output_gate === true,
      ropeFreqDims: rotated(fullRope)
    }),
    bosTokenId: tokenId("bos_token_id"),
    eosTokenId: tokenId("eos_token_id"),
    padTokenId: tokenId("pad_token_id")
  }
}
