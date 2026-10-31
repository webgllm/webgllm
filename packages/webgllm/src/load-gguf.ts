/**
 * Loads LFM3 weights from a GGUF file (llama.cpp's `lfm3` architecture),
 * such as Liquid AI's LFM3-1.7B-QAD-Q4_0.gguf.
 *
 * - Q4_0 matrices keep their blocks as they are (same nibble order); Q8_0
 *   stays 8-bit. A Q6_K embedding (Liquid AI's) is kept as it is, for the
 *   shader to read natively; a float one is quantized to 8 bits. The
 *   embedding comes in chunks of rows.
 * - Full-attention layers: llama.cpp keeps HF's head layout, where partial
 *   RoPE rotates dims j and j + headDim / 2 for j < ropeDimsFull / 2 (the
 *   other frequencies are zero; GGUF's rope_freqs.weight marks them with
 *   factors of 1e30). The loader reorders q and k rows (and q_norm, k_norm)
 *   within each head so the rotated dims come first, pairs (j, j +
 *   ropeDimsFull / 2), as the ONNX build has them: HF dims
 *   [0, R) [H/2, H/2 + R) [R, H/2) [H/2 + R, H) for R = ropeDimsFull / 2,
 *   H = headDim. Frequencies stay theta^(-2j / headDim).
 */
import { halfTable, bytesToFloats, halvesToFloats } from "./half.js"
import {
  GgmlType,
  readGgufHeader,
  type GgufHeader,
  type GgufTensorInfo,
  type GgufValue
} from "./gguf.js"
import { ReadPlan } from "./read-plan.js"
import {
  dequantizeQ6_K,
  newQuantMatrix,
  packQ4_0,
  packQ8_0,
  packQ6_K,
  quantizeQ8,
  type RequantStats
} from "./repack.js"
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

export type LoadGgufOptions = {
  /** A header read earlier (readGgufHeader), to skip reading it again. */
  header?: GgufHeader
  /** Embedding rows converted per step (default 512). */
  embeddingChunkRows?: number
  /** Filled with the error of requantizing a float embedding to 8 bits. */
  requantStats?: RequantStats
}

/**
 * Yields the model's pieces: config, each layer in order, the final norm,
 * then the embedding (also the output head).
 */
export async function* loadGguf(
  source: ByteSource,
  options: LoadGgufOptions = {}
): AsyncGenerator<WeightPiece, void, undefined> {
  const header = options.header ?? (await readGgufHeader(source))
  const tensor = (name: string): GgufTensorInfo => {
    const info = header.tensors.get(name)
    if (!info) throw new Error(`GGUF: missing tensor ${name}`)
    return info
  }

  const ropeFreqs = header.tensors.get("rope_freqs.weight")
  const freqFactors = ropeFreqs ? await readNorm(source, ropeFreqs) : undefined
  const config = ggufConfig(header, freqFactors)
  yield { kind: "config", config }

  const {
    hiddenSize: D,
    intermediateSize: F,
    numHeads,
    numKvHeads,
    headDim
  } = config
  const qPerm = headPermutation(numHeads, headDim, config.ropeDimsFull)
  const kPerm = headPermutation(numKvHeads, headDim, config.ropeDimsFull)
  const normPerm = headPermutation(1, headDim, config.ropeDimsFull)

  const plan = new ReadPlan()
  const want = (name: string) => {
    const info = tensor(name)
    plan.add({
      key: name,
      source,
      offset: info.offset,
      length: info.byteLength
    })
  }
  const layerTensors = (i: number): Array<string> => {
    const blk = `blk.${i}.`
    const names = [
      "attn_norm",
      "post_attention_norm",
      "ffn_norm",
      "post_ffw_norm"
    ]
    if (config.layerTypes[i] === "conv") {
      names.push("shortconv.in_proj", "shortconv.out_proj", "shortconv.conv")
    } else {
      names.push(
        "attn_q",
        "attn_k",
        "attn_v",
        "attn_gate",
        "attn_output",
        "attn_q_norm",
        "attn_k_norm"
      )
    }
    names.push("ffn_gate", "ffn_up", "ffn_down")
    return names.map((n) => `${blk}${n}.weight`)
  }
  for (let i = 0; i < config.numLayers; i++)
    for (const name of layerTensors(i)) want(name)
  want("output_norm.weight")

  const embedding = tensor("token_embd.weight")
  const [embK, vocab] = embedding.dims as [number, number]
  if (embK !== D)
    throw new Error(`GGUF: token_embd has ${embK} columns, expected ${D}`)
  const embRowBytes = embedding.byteLength / vocab
  const chunkRows = options.embeddingChunkRows ?? 512
  for (let r = 0; r < vocab; r += chunkRows) {
    const rows = Math.min(chunkRows, vocab - r)
    plan.add({
      key: `token_embd.weight@${r}`,
      source,
      offset: embedding.offset + r * embRowBytes,
      length: rows * embRowBytes
    })
  }

  const norm = async (
    name: string,
    length = D,
    perm?: Int32Array
  ): Promise<Float32Array> => {
    const values = toFloats(tensor(name), await plan.take(name))
    if (values.length !== length)
      throw new Error(
        `GGUF: ${name} has ${values.length} values, expected ${length}`
      )
    return perm ? permuteVector(values, perm) : values
  }

  /** Rows of a matrix: tensors with [K, rows] dims, in order, into one QuantMatrix. */
  const matrix = async (
    K: number,
    parts: Array<{ name: string; rows: number; perm?: Int32Array }>
  ): Promise<QuantMatrix> => {
    const infos = parts.map((p) => tensor(p.name))
    const bits = bitsOf(infos[0])
    for (const info of infos) {
      if (bitsOf(info) !== bits)
        throw new Error(
          `GGUF: ${info.name} can't be combined with ${infos[0].name}`
        )
    }
    const m = newQuantMatrix(
      parts.reduce((n, p) => n + p.rows, 0),
      K,
      bits,
      false
    )
    let row0 = 0
    for (let p = 0; p < parts.length; p++) {
      const { name, rows, perm } = parts[p]
      const info = infos[p]
      if (
        info.dims[0] !== K ||
        info.dims[1] !== rows ||
        info.dims.length !== 2
      ) {
        throw new Error(
          `GGUF: ${name} is [${info.dims}], expected [${K}, ${rows}]`
        )
      }
      packRows(m, row0, info, await plan.take(name), rows, perm)
      row0 += rows
    }
    return m
  }

  for (let i = 0; i < config.numLayers; i++) {
    const blk = `blk.${i}.`
    const type = config.layerTypes[i]
    const full = type === "full_attention"
    const operatorNorm = await norm(`${blk}attn_norm.weight`)
    const postOperatorNorm = await norm(`${blk}post_attention_norm.weight`)
    const ffnNorm = await norm(`${blk}ffn_norm.weight`)
    const postFfnNorm = await norm(`${blk}post_ffw_norm.weight`)
    let conv: ConvWeights | undefined
    let attention: AttentionWeights | undefined
    if (type === "conv") {
      const inProj = await matrix(D, [
        { name: `${blk}shortconv.in_proj.weight`, rows: 3 * D }
      ])
      const outProj = await matrix(D, [
        { name: `${blk}shortconv.out_proj.weight`, rows: D }
      ])
      const convInfo = tensor(`${blk}shortconv.conv.weight`)
      // ggml [width, channels]: each channel's taps are contiguous, oldest first.
      if (convInfo.dims[0] !== config.convKernel || convInfo.dims[1] !== D) {
        throw new Error(`GGUF: ${convInfo.name} is [${convInfo.dims}]`)
      }
      const kernel = toFloats(convInfo, await plan.take(convInfo.name))
      conv = { inProj, outProj, kernel }
    } else {
      const kv = numKvHeads * headDim
      const qkvg = await matrix(D, [
        {
          name: `${blk}attn_q.weight`,
          rows: numHeads * headDim,
          perm: full ? qPerm : undefined
        },
        {
          name: `${blk}attn_k.weight`,
          rows: kv,
          perm: full ? kPerm : undefined
        },
        { name: `${blk}attn_v.weight`, rows: kv },
        { name: `${blk}attn_gate.weight`, rows: numHeads }
      ])
      const o = await matrix(numHeads * headDim, [
        { name: `${blk}attn_output.weight`, rows: D }
      ])
      const qNorm = await norm(
        `${blk}attn_q_norm.weight`,
        headDim,
        full ? normPerm : undefined
      )
      const kNorm = await norm(
        `${blk}attn_k_norm.weight`,
        headDim,
        full ? normPerm : undefined
      )
      attention = { qkvg, o, qNorm, kNorm }
    }
    const gateUp = await matrix(D, [
      { name: `${blk}ffn_gate.weight`, rows: F },
      { name: `${blk}ffn_up.weight`, rows: F }
    ])
    const down = await matrix(F, [{ name: `${blk}ffn_down.weight`, rows: D }])
    const value: LayerWeights = {
      type,
      operatorNorm,
      postOperatorNorm,
      ffnNorm,
      postFfnNorm,
      gateUp,
      down
    }
    if (conv) value.conv = conv
    if (attention) value.attention = attention
    yield { kind: "layer", layer: i, value }
  }

  yield { kind: "finalNorm", value: await norm("output_norm.weight") }

  // The embedding (also the output head) in chunks of rows, so it's never
  // all in JavaScript memory at once: Q6_K as it is (the shader reads it
  // natively), Q4_0 and Q8_0 repacked, floats quantized to 8 bits.
  const q6 = embedding.type === GgmlType.Q6_K
  const native =
    embedding.type === GgmlType.Q4_0 || embedding.type === GgmlType.Q8_0
  const bits = q6 ? 6 : embedding.type === GgmlType.Q4_0 ? 4 : 8
  const floats = q6 || native ? undefined : new Float32Array(chunkRows * D)
  for (let r = 0; r < vocab; r += chunkRows) {
    const rows = Math.min(chunkRows, vocab - r)
    const bytes = await plan.take(`token_embd.weight@${r}`)
    const m = newQuantMatrix(rows, D, bits, false)
    if (q6) packQ6_K(m, 0, bytes, rows)
    else if (floats) {
      const values = floats.subarray(0, rows * D)
      rowsToFloats(embedding, bytes, rows, values)
      quantizeQ8(m, 0, values, rows, options.requantStats)
    } else packRows(m, 0, embedding, bytes, rows)
    yield { kind: "embedding", value: m, row0: r, totalRows: vocab }
  }
}

/** The bits a tensor gets in a QuantMatrix. */
function bitsOf(info: GgufTensorInfo): 4 | 8 {
  switch (info.type) {
    case GgmlType.Q4_0:
      return 4
    case GgmlType.Q8_0:
    case GgmlType.Q6_K:
    case GgmlType.F16:
    case GgmlType.F32:
      return 8
    default:
      throw new Error(
        `GGUF: ${info.name} has tensor type ${info.type}, which this loader doesn't take (Q4_0, Q8_0, Q6_K, F16, F32 do)`
      )
  }
}

function packRows(
  m: QuantMatrix,
  row0: number,
  info: GgufTensorInfo,
  bytes: Uint8Array,
  rows: number,
  perm?: Int32Array
): void {
  if (info.type === GgmlType.Q4_0) return packQ4_0(m, row0, bytes, rows, perm)
  if (info.type === GgmlType.Q8_0) return packQ8_0(m, row0, bytes, rows, perm)
  const values = new Float32Array(rows * m.K)
  rowsToFloats(info, bytes, rows, values)
  if (perm) {
    const src = values.slice()
    for (let i = 0; i < rows; i++)
      values.set(src.subarray(perm[i] * m.K, (perm[i] + 1) * m.K), i * m.K)
  }
  quantizeQ8(m, row0, values, rows)
}

/** F32, F16 or Q6_K rows as floats. */
function rowsToFloats(
  info: GgufTensorInfo,
  bytes: Uint8Array,
  rows: number,
  out: Float32Array
): void {
  const K = info.dims[0]
  if (info.type === GgmlType.Q6_K) return dequantizeQ6_K(bytes, rows, K, out)
  if (info.type === GgmlType.F16) {
    const half = halfTable()
    for (let i = 0; i < rows * K; i++)
      out[i] = half[bytes[2 * i] | (bytes[2 * i + 1] << 8)]
    return
  }
  if (info.type === GgmlType.F32)
    return out.set(bytesToFloats(bytes.subarray(0, rows * K * 4)))
  throw new Error(
    `GGUF: can't convert ${info.name} (type ${info.type}) to floats`
  )
}

function toFloats(info: GgufTensorInfo, bytes: Uint8Array): Float32Array {
  if (info.type === GgmlType.F32) return bytesToFloats(bytes)
  if (info.type === GgmlType.F16) return halvesToFloats(bytes)
  throw new Error(
    `GGUF: ${info.name} has type ${info.type}, expected F32 or F16`
  )
}

async function readNorm(
  source: ByteSource,
  info: GgufTensorInfo
): Promise<Float32Array> {
  return toFloats(info, await source.read(info.offset, info.byteLength))
}

/**
 * For each output row (head h, dim d), the HF row it comes from, so that
 * the rotated dims of each head come first (see the module comment).
 */
export function headPermutation(
  heads: number,
  headDim: number,
  ropeDims: number
): Int32Array {
  const R = ropeDims / 2
  const H = headDim / 2
  const perm = new Int32Array(heads * headDim)
  for (let h = 0; h < heads; h++) {
    for (let d = 0; d < headDim; d++) {
      let src: number
      if (d < R) src = d
      else if (d < 2 * R) src = H + d - R
      else if (d < H + R) src = d - R
      else src = d
      perm[h * headDim + d] = h * headDim + src
    }
  }
  return perm
}

function permuteVector(values: Float32Array, perm: Int32Array): Float32Array {
  const out = new Float32Array(perm.length)
  for (let i = 0; i < perm.length; i++) out[i] = values[perm[i]]
  return out
}

/** A float32 metadata value as the decimal it was written from (1e-5, not 9.9999997e-6). */
function f32Decimal(value: number): number {
  return Number(value.toPrecision(7))
}

function ggufConfig(
  header: GgufHeader,
  freqFactors: Float32Array | undefined
): ModelConfig {
  const meta = header.metadata
  const arch = meta.get("general.architecture")
  if (typeof arch !== "string") throw new Error("GGUF: no general.architecture")
  const get = (key: string): GgufValue => {
    const value = meta.get(key) ?? meta.get(`${arch}.${key}`)
    if (value === undefined)
      throw new Error(`GGUF: missing metadata ${arch}.${key}`)
    return value
  }
  const num = (key: string, fallback?: number): number => {
    if (
      fallback !== undefined &&
      meta.get(key) === undefined &&
      meta.get(`${arch}.${key}`) === undefined
    ) {
      return fallback
    }
    const value = get(key)
    if (typeof value !== "number")
      throw new Error(`GGUF: ${key} is not a number`)
    return value
  }
  const list = (key: string): Array<number | boolean | string> | undefined => {
    const value = meta.get(`${arch}.${key}`)
    if (value === undefined) return undefined
    if (Array.isArray(value)) return value
    return undefined
  }

  const numLayers = num("block_count")
  const hiddenSize = num("embedding_length")
  const numHeads = num("attention.head_count")
  const kvValue = meta.get(`${arch}.attention.head_count_kv`)
  const kvHeads: Array<number> = Array.isArray(kvValue)
    ? kvValue.map(Number)
    : Array.from({ length: numLayers }, () => Number(kvValue ?? numHeads))
  if (kvHeads.length !== numLayers)
    throw new Error("GGUF: head_count_kv doesn't have one entry per layer")
  const swa = list("attention.sliding_window_pattern")
  const layerTypes: Array<LayerType> = kvHeads.map((kv, i) =>
    kv === 0 ? "conv" : swa?.[i] ? "sliding_attention" : "full_attention"
  )
  const numKvHeads = Math.max(...kvHeads)
  if (kvHeads.some((kv) => kv !== 0 && kv !== numKvHeads)) {
    throw new Error(
      "GGUF: attention layers with different KV head counts are not supported"
    )
  }
  const headDim = num("attention.key_length", hiddenSize / numHeads)
  const ropeDimsSliding = num("rope.dimension_count", headDim)

  // rope_freqs: frequency factors over ropeDimsSliding / 2 pairs; 1 where
  // the pair rotates, 1e30 (frequency ~0) where it doesn't.
  let ropeDimsFull = ropeDimsSliding
  if (freqFactors) {
    let rotated = 0
    while (rotated < freqFactors.length && freqFactors[rotated] === 1) rotated++
    for (let j = rotated; j < freqFactors.length; j++) {
      if (!(freqFactors[j] > 1e20)) {
        throw new Error(
          "GGUF: rope_freqs holds scaling factors, which this loader doesn't support"
        )
      }
    }
    if (freqFactors.length * 2 !== ropeDimsSliding)
      throw new Error("GGUF: rope_freqs length doesn't match")
    ropeDimsFull = rotated * 2
  }

  const embedding = header.tensors.get("token_embd.weight")
  if (!embedding) throw new Error("GGUF: missing tensor token_embd.weight")
  const tokenId = (key: string, fallback: number): number => {
    const value = meta.get(`tokenizer.ggml.${key}`)
    return typeof value === "number" ? value : fallback
  }

  const ffn = meta.get(`${arch}.feed_forward_length`)
  const gate0 = header.tensors.get("blk.0.ffn_gate.weight")
  const intermediateSize =
    typeof ffn === "number" ? ffn : (gate0?.dims[1] ?? Number.NaN)
  if (gate0 && gate0.dims[1] !== intermediateSize) {
    throw new Error(
      `GGUF: feed_forward_length ${intermediateSize} doesn't match ffn_gate rows ${gate0.dims[1]}`
    )
  }

  return {
    format: "gguf",
    hiddenSize,
    intermediateSize,
    numLayers,
    numHeads,
    numKvHeads,
    headDim,
    vocabSize: embedding.dims[1],
    layerTypes,
    slidingWindow: num("attention.sliding_window", 0),
    contextLength: num("context_length", 8192),
    ropeThetaFull: f32Decimal(num("rope.freq_base")),
    ropeThetaSliding: f32Decimal(
      num("rope.freq_base_swa", num("rope.freq_base"))
    ),
    ropeDimsFull,
    ropeDimsSliding,
    normEps: f32Decimal(num("attention.layer_norm_rms_epsilon")),
    convKernel: num("shortconv.l_cache"),
    bosTokenId: tokenId("bos_token_id", -1),
    eosTokenId: tokenId("eos_token_id", -1),
    padTokenId: tokenId("padding_token_id", -1),
    // The GGUF loader reads LFM3 (its output gate and sandwich norms).
    attentionGate: true
  }
}
