/**
 * Loads KRAFTON's Raon-1 nano GGUFs (`raon1nano`), as their llama.cpp
 * converter writes them (conversion/raon1_nano.py):
 *
 * - The residual is `hyper_connection.count` streams (Raon-mHC). Each layer
 *   has two sites, before the operator and before the MLP: `hc_*_fn` picks
 *   how the branch reads the streams and writes back (its rows already
 *   scaled by alpha), `hc_*_base` its biases, `hc_*_chart` the folded
 *   mixing chart (see StreamMixWeights). post's factor of 2 is folded into
 *   attn_output and ffn_down.
 * - Raon Delta Attention layers (head_count_kv 0): attn_qkv holds q|k|v and
 *   the output gate's rows, ssm_conv1d the three convolutions' taps,
 *   ssm_f_a the decay's low-rank projection then b and b + write, ssm_f_b
 *   and ssm_dt.bias the decay logits with exp(A_log) folded in.
 * - Gated MLA layers (head_count_kv 1), absorbed: attn_k_b and attn_v_b are
 *   one matrix a head (see LatentConfig); no positions, no q/k norms.
 * - ffn_up holds gate then up.
 * - Per-layer embeddings: per_layer_token_embd (a token's row: a layer's
 *   `dim` values after another) stays in the file, read when a token is;
 *   per_layer_model_proj and each layer's inp_gate and proj are uploaded.
 *
 * Q4_K matrices keep their 4 bits (see packQ4_K) and Q8_0 ones their 8; the
 * others (Q6_K, F16) are stored as 8-bit.
 * An `_MTP` file's extra multi-token-prediction block is left out.
 */
import { toHalf } from "./half.js"
import { GgmlType, tensorByteLength, type GgufHeader, type GgufTensorInfo } from "./gguf.js"
import { bitsOf, defaultEmbeddingRows, packRows, rowsToFloats, toFloats, type LoadGgufOptions } from "./load-gguf.js"
import { ReadPlan } from "./read-plan.js"
import { newQuantMatrix, packQ4_K, packQ6_K, quantizeQ8 } from "./repack.js"
import type {
  ByteSource,
  LatentAttentionWeights,
  LayerType,
  LayerWeights,
  LinearAttentionWeights,
  ModelConfig,
  QuantMatrix,
  StreamMixWeights,
  WeightPiece
} from "./weights.js"

const ARCH = "raon1nano"

/** A float32 metadata value as the decimal it was written from. */
const f32Decimal = (value: number) => Number(value.toPrecision(7))

export function raonConfig(header: GgufHeader): ModelConfig {
  const meta = header.metadata
  const num = (key: string, fallback?: number): number => {
    const value = meta.get(`${ARCH}.${key}`)
    if (value === undefined && fallback !== undefined) return fallback
    if (typeof value !== "number") throw new Error(`GGUF: ${ARCH}.${key} is not a number`)
    return value
  }
  const list = (key: string): Array<number> => {
    const value = meta.get(`${ARCH}.${key}`)
    if (!Array.isArray(value)) throw new Error(`GGUF: ${ARCH}.${key} is not a list`)
    return value.map(Number)
  }
  // (An _MTP file's last block is the multi-token-prediction layer.)
  const numLayers = num("block_count") - num("nextn_predict_layers", 0)
  const hiddenSize = num("embedding_length")
  const heads = list("attention.head_count").slice(0, numLayers)
  const kvHeads = list("attention.head_count_kv").slice(0, numLayers)
  const layerTypes: Array<LayerType> = kvHeads.map((kv) => (kv === 0 ? "linear_attention" : "full_attention"))
  const linearHeads = heads[layerTypes.indexOf("linear_attention")]
  const latentHeads = heads[layerTypes.indexOf("full_attention")]
  if (linearHeads === undefined || latentHeads === undefined)
    throw new Error("GGUF: a Raon-1 nano model has both Raon Delta Attention and MLA layers")
  if (heads.some((h, i) => h !== (kvHeads[i] === 0 ? linearHeads : latentHeads)))
    throw new Error("GGUF: layers of one kind with different head counts are not supported")
  const rank = num("attention.kv_lora_rank")
  const shared = num("rope.dimension_count")
  const keyDim = num("attention.key_length_mla")
  const linearDim = num("kda.head_dim")
  const embedding = header.tensors.get("token_embd.weight")
  if (!embedding) throw new Error("GGUF: missing tensor token_embd.weight")
  const tokenId = (key: string): number => {
    const value = meta.get(`tokenizer.ggml.${key}`)
    return typeof value === "number" ? value : -1
  }
  return {
    format: "gguf",
    hiddenSize,
    intermediateSize: num("feed_forward_length"),
    numLayers,
    // The latent attention as the engine's attention sees it: one KV head
    // of latent rows, every head's query absorbed into it.
    numHeads: latentHeads,
    numKvHeads: 1,
    headDim: rank + shared,
    vocabSize: embedding.dims[1],
    layerTypes,
    slidingWindow: 0,
    contextLength: num("context_length", 8192),
    // No positions: nothing rotates.
    ropeThetaFull: 10000,
    ropeThetaSliding: 10000,
    ropeDimsFull: 0,
    ropeDimsSliding: 0,
    normEps: f32Decimal(num("attention.layer_norm_rms_epsilon")),
    convKernel: num("ssm.conv_kernel"),
    bosTokenId: tokenId("bos_token_id"),
    eosTokenId: tokenId("eos_token_id"),
    padTokenId: tokenId("padding_token_id"),
    // (The MLA's gate is on its values after the latent attention: see LatentAttentionWeights.)
    attentionGate: false,
    attentionScale: 1 / Math.sqrt(keyDim),
    linear: {
      numKeyHeads: linearHeads,
      numValueHeads: linearHeads,
      keyHeadDim: linearDim,
      valueHeadDim: linearDim,
      convKernel: num("ssm.conv_kernel"),
      keyDecay: { rank: num("ssm.time_step_rank"), lowerBound: num("kda.gate_lower_bound") }
    },
    streams: { count: num("hyper_connection.count"), eps: f32Decimal(num("hyper_connection.epsilon")) },
    latent: {
      heads: latentHeads,
      queryRank: num("attention.q_lora_rank"),
      rank,
      shared,
      keyDim,
      valueDim: num("attention.value_length_mla"),
      eps: f32Decimal(num("attention.latent_norm_rms_epsilon"))
    },
    perLayer: { dim: num("embedding_length_per_layer_input") }
  }
}

/** How loadRaon stores a matrix made of these tensors' rows (K weights each): Q4_K ones alone as 4-bit with half zero points, any others as 8-bit. */
export function raonMatrixStorage(infos: ReadonlyArray<GgufTensorInfo>, K: number): { bits: 4 | 8; halves: boolean } {
  const halves = infos.every((info) => info.type === GgmlType.Q4_K) && K % 256 === 0
  return { bits: halves ? 4 : 8, halves }
}

/**
 * A layer's matrices by role: the tensors each is made of (their names
 * after `blk.N.`) and the rows' width. What loadRaon reads them by, and
 * what predictSpeed's probe sizes and stores its random ones by.
 */
export function raonLayerMatrices(config: ModelConfig, type: LayerType): Record<string, { K: number; N: number; tensors: Array<string> }> {
  const { hiddenSize: D, intermediateSize: F } = config
  const lin = config.linear
  const latent = config.latent
  const streams = config.streams
  const perLayer = config.perLayer
  if (!lin?.keyDecay || !latent || !streams || !perLayer) throw new Error("The model's config isn't a Raon-1 nano's.")
  const inner = lin.numKeyHeads * lin.keyHeadDim
  const mixRows = 2 * streams.count + (streams.count - 1) ** 2
  const mix = (site: string) => ({ K: streams.count * D, N: Math.ceil(mixRows / 4) * 4, tensors: [`hc_${site}_fn`] })
  return {
    mixOperator: mix("attn"),
    mixFfn: mix("ffn"),
    ...(type === "linear_attention"
      ? {
          inProj: { K: D, N: 4 * inner, tensors: ["attn_qkv"] },
          ba: { K: D, N: lin.keyDecay.rank + 2 * lin.numKeyHeads, tensors: ["ssm_f_a"] },
          decay: { K: lin.keyDecay.rank, N: inner, tensors: ["ssm_f_b"] },
          outProj: { K: inner, N: D, tensors: ["attn_output"] }
        }
      : {
          latentIn: { K: D, N: latent.queryRank + latent.rank + latent.shared + latent.heads * latent.valueDim, tensors: ["attn_q_a", "attn_kv_a_mqa", "attn_gate"] },
          qB: { K: latent.queryRank, N: latent.heads * latent.keyDim, tensors: ["attn_q_b"] },
          o: { K: latent.heads * latent.valueDim, N: D, tensors: ["attn_output"] }
        }),
    perLayerGate: { K: D, N: perLayer.dim, tensors: ["inp_gate"] },
    perLayerProj: { K: perLayer.dim, N: D, tensors: ["proj"] },
    gateUp: { K: D, N: 2 * F, tensors: ["ffn_up"] },
    down: { K: F, N: D, tensors: ["ffn_down"] }
  }
}

export async function* loadRaon(
  source: ByteSource,
  header: GgufHeader,
  options: LoadGgufOptions = {}
): AsyncGenerator<WeightPiece, void, undefined> {
  const config = raonConfig(header)
  if (header.metadata.get(`${ARCH}.hyper_connection.magnitude`) !== 1)
    throw new Error("GGUF: a Raon-1 nano file whose post magnitude isn't folded into its weights is not supported")
  yield { kind: "config", config }
  const { hiddenSize: D, intermediateSize: F, numLayers } = config
  const lin = config.linear!
  const latent = config.latent!
  const streams = config.streams!
  const perLayer = config.perLayer!
  if (!lin.keyDecay) throw new Error("GGUF: Raon Delta Attention has a per-key decay")
  const inner = lin.numKeyHeads * lin.keyHeadDim
  const mixRows = 2 * streams.count + (streams.count - 1) ** 2
  const nope = latent.keyDim - latent.shared

  const tensor = (name: string): GgufTensorInfo => {
    const info = header.tensors.get(name)
    if (!info) throw new Error(`GGUF: missing tensor ${name}`)
    return info
  }
  const plan = new ReadPlan(options.readAhead ?? 6, 24 << 20, options.releaseReadBuffers === true)
  let largestTensorBytes = 0
  const want = (name: string) => {
    const info = tensor(name)
    largestTensorBytes = Math.max(largestTensorBytes, info.byteLength)
    plan.add({ key: name, source, offset: info.offset, length: info.byteLength })
  }
  // Every read, in the order it's taken.
  const mixNames = (blk: string, site: string) => [`${blk}hc_${site}_fn.weight`, `${blk}hc_${site}_base.weight`, `${blk}hc_${site}_chart.weight`]
  const layerNames = (i: number): Array<string> => {
    const blk = `blk.${i}.`
    const operator =
      config.layerTypes[i] === "linear_attention"
        ? ["attn_qkv.weight", "ssm_f_a.weight", "ssm_f_b.weight", "attn_output.weight", "ssm_conv1d.weight", "ssm_dt.bias", "ssm_norm.weight"]
        : [
            "attn_q_a.weight", "attn_kv_a_mqa.weight", "attn_gate.weight", "attn_q_b.weight", "attn_k_b.weight", "attn_v_b.weight",
            "attn_output.weight", "attn_q_a_norm.weight", "attn_kv_a_norm.weight"
          ]
    return [
      `${blk}attn_norm.weight`,
      `${blk}ffn_norm.weight`,
      ...mixNames(blk, "attn"),
      ...mixNames(blk, "ffn"),
      ...operator.map((n) => blk + n),
      `${blk}inp_gate.weight`,
      `${blk}proj.weight`,
      `${blk}post_norm.weight`,
      `${blk}ffn_up.weight`,
      `${blk}ffn_down.weight`
    ]
  }
  want("per_layer_model_proj.weight")
  want("per_layer_proj_norm.weight")
  for (let i = 0; i < numLayers; i++) for (const name of layerNames(i)) want(name)
  want("output_norm.weight")
  const embedding = tensor("token_embd.weight")
  const [embK, vocab] = embedding.dims as [number, number]
  if (embK !== D) throw new Error(`GGUF: token_embd has ${embK} columns, expected ${D}`)
  const embRowBytes = embedding.byteLength / vocab
  const q6 = embedding.type === GgmlType.Q6_K
  const chunkRows = options.embeddingChunkRows ?? defaultEmbeddingRows(embRowBytes, largestTensorBytes, q6 || embedding.type === GgmlType.Q8_0)
  for (let r = 0; r < vocab; r += chunkRows) {
    plan.add({ key: `token_embd.weight@${r}`, source, offset: embedding.offset + r * embRowBytes, length: Math.min(chunkRows, vocab - r) * embRowBytes })
  }

  const floats = async (name: string, length?: number): Promise<Float32Array> => {
    const bytes = await plan.take(name)
    try {
      const values = toFloats(tensor(name), bytes).slice()
      if (length !== undefined && values.length !== length)
        throw new Error(`GGUF: ${name} has ${values.length} values, expected ${length}`)
      return values
    } finally {
      plan.release(bytes)
    }
  }
  /**
   * Tensors of [K, rows] one after another as a matrix of `N` rows (zero
   * rows after them): Q4_K ones as 4-bit with their sub-blocks' own scales
   * and zero points (no weight requantized), any others as 8-bit.
   */
  const matrix = async (K: number, parts: Array<{ name: string; rows: number }>, N = parts.reduce((n, p) => n + p.rows, 0)): Promise<QuantMatrix> => {
    const q4k = raonMatrixStorage(parts.map((p) => tensor(p.name)), K).halves
    const m = q4k ? newQuantMatrix(N, K, 4, "halves") : newQuantMatrix(N, K, 8, false)
    let row0 = 0
    for (const { name, rows } of parts) {
      const info = tensor(name)
      if (info.dims.length !== 2 || info.dims[0] !== K || info.dims[1] !== rows)
        throw new Error(`GGUF: ${name} is [${info.dims}], expected [${K}, ${rows}]`)
      if (bitsOf(info) !== 8) throw new Error(`GGUF: ${name}'s type ${info.type} isn't read here`)
      const bytes = await plan.take(name)
      if (q4k) packQ4_K(m, row0, bytes, rows)
      else packRows(m, row0, info, bytes, rows)
      plan.release(bytes)
      row0 += rows
    }
    return m
  }
  /** A tensor of one small matrix a head, [K, rows, heads], as halves. */
  const perHead = async (name: string, K: number, rows: number): Promise<Uint16Array> => {
    const info = tensor(name)
    if (info.dims.length !== 3 || info.dims[0] !== K || info.dims[1] !== rows || info.dims[2] !== latent.heads)
      throw new Error(`GGUF: ${name} is [${info.dims}], expected [${K}, ${rows}, ${latent.heads}]`)
    const bytes = await plan.take(name)
    const values = new Float32Array(K * rows * latent.heads)
    rowsToFloats(info, bytes, rows * latent.heads, values)
    plan.release(bytes)
    const halves = new Uint16Array(values.length)
    for (let i = 0; i < values.length; i++) halves[i] = toHalf(values[i])
    return halves
  }
  const mix = async (blk: string, site: string): Promise<StreamMixWeights> => ({
    fn: await matrix(streams.count * D, [{ name: `${blk}hc_${site}_fn.weight`, rows: mixRows }], Math.ceil(mixRows / 4) * 4),
    base: await floats(`${blk}hc_${site}_base.weight`, mixRows),
    chart: await floats(`${blk}hc_${site}_chart.weight`, mixRows * streams.count ** 2 + 2 * streams.count ** 2)
  })

  const table = tensor("per_layer_token_embd.weight")
  if (table.dims[0] !== perLayer.dim * numLayers || table.dims[1] !== vocab)
    throw new Error(`GGUF: per_layer_token_embd is [${table.dims}]`)
  yield {
    kind: "perLayer",
    proj: await matrix(D, [{ name: "per_layer_model_proj.weight", rows: perLayer.dim * numLayers }]),
    norm: await floats("per_layer_proj_norm.weight", perLayer.dim),
    table: { offset: table.offset, rowBytes: tensorByteLength(table.type, [table.dims[0], 1]), type: table.type, rows: vocab }
  }

  for (let i = 0; i < numLayers; i++) {
    const blk = `blk.${i}.`
    const type = config.layerTypes[i]
    const operatorNorm = await floats(`${blk}attn_norm.weight`, D)
    const ffnNorm = await floats(`${blk}ffn_norm.weight`, D)
    const mixes = { operator: await mix(blk, "attn"), ffn: await mix(blk, "ffn") }
    let linear: LinearAttentionWeights | undefined
    let latentWeights: LatentAttentionWeights | undefined
    if (type === "linear_attention") {
      const inProj = await matrix(D, [{ name: `${blk}attn_qkv.weight`, rows: 4 * inner }])
      const ba = await matrix(D, [{ name: `${blk}ssm_f_a.weight`, rows: lin.keyDecay.rank + 2 * lin.numKeyHeads }])
      const decay = await matrix(lin.keyDecay.rank, [{ name: `${blk}ssm_f_b.weight`, rows: inner }])
      const outProj = await matrix(inner, [{ name: `${blk}attn_output.weight`, rows: D }])
      const convInfo = tensor(`${blk}ssm_conv1d.weight`)
      if (convInfo.dims[0] !== lin.convKernel || convInfo.dims[1] !== 3 * inner)
        throw new Error(`GGUF: ${convInfo.name} is [${convInfo.dims}]`)
      linear = {
        inProj,
        ba,
        decay,
        outProj,
        conv: await floats(convInfo.name),
        a: new Float32Array(lin.numKeyHeads),
        dtBias: await floats(`${blk}ssm_dt.bias`, inner),
        norm: await floats(`${blk}ssm_norm.weight`, lin.valueHeadDim)
      }
    } else {
      const inProj = await matrix(D, [
        { name: `${blk}attn_q_a.weight`, rows: latent.queryRank },
        { name: `${blk}attn_kv_a_mqa.weight`, rows: latent.rank + latent.shared },
        { name: `${blk}attn_gate.weight`, rows: latent.heads * latent.valueDim }
      ])
      const qB = await matrix(latent.queryRank, [{ name: `${blk}attn_q_b.weight`, rows: latent.heads * latent.keyDim }])
      const kB = await perHead(`${blk}attn_k_b.weight`, nope, latent.rank)
      const vB = await perHead(`${blk}attn_v_b.weight`, latent.rank, latent.valueDim)
      const o = await matrix(latent.heads * latent.valueDim, [{ name: `${blk}attn_output.weight`, rows: D }])
      latentWeights = {
        inProj,
        qB,
        kB,
        vB,
        o,
        qNorm: await floats(`${blk}attn_q_a_norm.weight`, latent.queryRank),
        kvNorm: await floats(`${blk}attn_kv_a_norm.weight`, latent.rank)
      }
    }
    const perLayerWeights = {
      gate: await matrix(D, [{ name: `${blk}inp_gate.weight`, rows: perLayer.dim }]),
      proj: await matrix(perLayer.dim, [{ name: `${blk}proj.weight`, rows: D }]),
      norm: await floats(`${blk}post_norm.weight`, D)
    }
    const gateUp = await matrix(D, [{ name: `${blk}ffn_up.weight`, rows: 2 * F }])
    const down = await matrix(F, [{ name: `${blk}ffn_down.weight`, rows: D }])
    const value: LayerWeights = { type, operatorNorm, ffnNorm, gateUp, down, mix: mixes, perLayer: perLayerWeights }
    if (linear) value.linear = linear
    if (latentWeights) value.latent = latentWeights
    yield { kind: "layer", layer: i, value }
  }

  yield { kind: "finalNorm", value: await floats("output_norm.weight", D) }

  // The embedding (also the output head) in chunks of rows: Q6_K as it is, others as 8-bit.
  const scratch = q6 ? undefined : new Float32Array(chunkRows * D)
  for (let r = 0; r < vocab; r += chunkRows) {
    const rows = Math.min(chunkRows, vocab - r)
    const bytes = await plan.take(`token_embd.weight@${r}`)
    const m = newQuantMatrix(rows, D, q6 ? 6 : 8, false)
    if (q6) packQ6_K(m, 0, bytes, rows)
    else if (embedding.type === GgmlType.Q8_0) packRows(m, 0, embedding, bytes, rows)
    else {
      const values = scratch!.subarray(0, rows * D)
      rowsToFloats(embedding, bytes, rows, values)
      quantizeQ8(m, 0, values, rows, options.requantStats)
    }
    plan.release(bytes)
    yield { kind: "embedding", value: m, row0: r, totalRows: vocab }
  }
}
