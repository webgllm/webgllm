import assert from "node:assert/strict"
import test from "node:test"
import { GgmlType } from "./gguf.js"
import type { GgufHeader, GgufTensorInfo, GgufValue } from "./gguf.js"
import { loadGguf } from "./load-gguf.js"
import type { QuantMatrix, WeightPiece } from "./weights.js"

/**
 * A two-layer raon1nano file in float32, shaped as KRAFTON's converter
 * writes it: a Raon Delta Attention layer, then a gated MLA layer; `mtp`
 * adds the multi-token-prediction block an _MTP file has.
 */
function fixture(mtp = false) {
  const D = 64, F = 64, hc = 4, mix = 2 * hc + (hc - 1) ** 2
  const rda = { heads: 4, dim: 8, rank: 32 }
  const mla = { heads: 4, qRank: 32, rank: 32, shared: 32, key: 64, value: 16 }
  const ple = 32, vocab = 8, layers = 2, blocks = layers + (mtp ? 1 : 0)
  const tensors = new Map<string, GgufTensorInfo>()
  const chunks: Array<Uint8Array> = []
  let offset = 0
  const add = (name: string, dims: Array<number>, value = 0.5) => {
    const bytes = new Uint8Array(new Float32Array(dims.reduce((a, b) => a * b, 1)).fill(value).buffer)
    tensors.set(name, { name, dims, type: GgmlType.F32, offset, byteLength: bytes.length })
    offset += bytes.length
    chunks.push(bytes)
  }
  add("token_embd.weight", [D, vocab])
  add("output_norm.weight", [D], 1.5)
  add("per_layer_token_embd.weight", [ple * layers, vocab], 0.25)
  add("per_layer_model_proj.weight", [D, ple * layers])
  add("per_layer_proj_norm.weight", [ple], 1.25)
  for (let i = 0; i < blocks; i++) {
    const p = `blk.${i}.`
    add(`${p}attn_norm.weight`, [D], 1 + i)
    add(`${p}ffn_norm.weight`, [D], 2 + i)
    for (const site of ["attn", "ffn"]) {
      add(`${p}hc_${site}_fn.weight`, [hc * D, mix])
      add(`${p}hc_${site}_base.weight`, [mix], 0.125)
      add(`${p}hc_${site}_chart.weight`, [mix * hc * hc + 2 * hc * hc])
    }
    if (i === 0) {
      const inner = rda.heads * rda.dim
      add(`${p}attn_qkv.weight`, [D, 4 * inner])
      add(`${p}ssm_conv1d.weight`, [4, 3 * inner])
      add(`${p}ssm_f_a.weight`, [D, rda.rank + 2 * rda.heads])
      add(`${p}ssm_f_b.weight`, [rda.rank, inner])
      add(`${p}ssm_dt.bias`, [inner])
      add(`${p}ssm_norm.weight`, [rda.dim])
      add(`${p}attn_output.weight`, [inner, D])
    } else {
      add(`${p}attn_q_a.weight`, [D, mla.qRank])
      add(`${p}attn_q_a_norm.weight`, [mla.qRank])
      add(`${p}attn_kv_a_mqa.weight`, [D, mla.rank + mla.shared])
      add(`${p}attn_kv_a_norm.weight`, [mla.rank])
      add(`${p}attn_q_b.weight`, [mla.qRank, mla.heads * mla.key])
      add(`${p}attn_k_b.weight`, [mla.key - mla.shared, mla.rank, mla.heads], 0.75)
      add(`${p}attn_v_b.weight`, [mla.rank, mla.value, mla.heads], -0.5)
      add(`${p}attn_gate.weight`, [D, mla.heads * mla.value])
      add(`${p}attn_output.weight`, [mla.heads * mla.value, D])
    }
    if (i < layers) {
      add(`${p}inp_gate.weight`, [D, ple])
      add(`${p}proj.weight`, [ple, D])
      add(`${p}post_norm.weight`, [D])
    }
    add(`${p}ffn_up.weight`, [D, 2 * F])
    add(`${p}ffn_down.weight`, [F, D])
  }
  const a = "raon1nano"
  const metadata = new Map<string, GgufValue>([
    ["general.architecture", a], [`${a}.block_count`, blocks], [`${a}.context_length`, 262144], [`${a}.embedding_length`, D],
    [`${a}.feed_forward_length`, F], [`${a}.attention.head_count`, [rda.heads, mla.heads, ...(mtp ? [mla.heads] : [])]],
    [`${a}.attention.head_count_kv`, [0, 1, ...(mtp ? [1] : [])]], [`${a}.attention.layer_norm_rms_epsilon`, Math.fround(1e-6)],
    [`${a}.ssm.conv_kernel`, 4], [`${a}.ssm.time_step_rank`, rda.rank], [`${a}.kda.head_dim`, rda.dim], [`${a}.kda.gate_lower_bound`, -5],
    [`${a}.attention.q_lora_rank`, mla.qRank], [`${a}.attention.kv_lora_rank`, mla.rank], [`${a}.rope.dimension_count`, mla.shared],
    [`${a}.attention.key_length_mla`, mla.key], [`${a}.attention.value_length_mla`, mla.value],
    [`${a}.attention.latent_norm_rms_epsilon`, Math.fround(1e-6)], [`${a}.embedding_length_per_layer_input`, ple],
    [`${a}.hyper_connection.count`, hc], [`${a}.hyper_connection.epsilon`, Math.fround(1e-5)], [`${a}.hyper_connection.magnitude`, 1],
    ...(mtp ? [[`${a}.nextn_predict_layers`, 1] as [string, GgufValue]] : []),
    ["tokenizer.ggml.eos_token_id", 5], ["tokenizer.ggml.padding_token_id", 3]
  ])
  const bytes = new Uint8Array(offset)
  let at = 0
  for (const chunk of chunks) { bytes.set(chunk, at); at += chunk.length }
  const header: GgufHeader = { version: 3, metadata, tensors, dataOffset: 0, headerLength: 0 }
  return { header, tensors, source: { size: bytes.length, read: async (o: number, n: number) => bytes.slice(o, o + n) } }
}

const collect = async (f: ReturnType<typeof fixture>) => {
  const pieces: Array<WeightPiece> = []
  for await (const piece of loadGguf(f.source, { header: f.header, embeddingChunkRows: 4 })) pieces.push(piece)
  return pieces
}

test("a Raon-1 nano GGUF's config: streams, latent attention, per-key decay, per-layer embeddings, no positions", async () => {
  const config = (await collect(fixture())).find((p) => p.kind === "config")
  assert.ok(config?.kind === "config")
  const c = config.config
  assert.deepEqual(c.layerTypes, ["linear_attention", "full_attention"])
  assert.equal(c.contextLength, 262144)
  assert.equal(c.normEps, 1e-6)
  // The latent attention as the engine's: one KV head of latent rows, every head's query absorbed into it.
  assert.deepEqual([c.numHeads, c.numKvHeads, c.headDim], [4, 1, 64])
  assert.equal(c.attentionScale, 1 / Math.sqrt(64))
  assert.equal(c.attentionGate, false)
  assert.deepEqual([c.ropeDimsFull, c.ropeDimsSliding], [0, 0])
  assert.deepEqual(c.streams, { count: 4, eps: 1e-5 })
  assert.deepEqual(c.latent, { heads: 4, queryRank: 32, rank: 32, shared: 32, keyDim: 64, valueDim: 16, eps: 1e-6 })
  assert.deepEqual(c.linear, { numKeyHeads: 4, numValueHeads: 4, keyHeadDim: 8, valueHeadDim: 8, convKernel: 4, keyDecay: { rank: 32, lowerBound: -5 } })
  assert.deepEqual(c.perLayer, { dim: 32 })
  assert.deepEqual([c.eosTokenId, c.padTokenId, c.bosTokenId], [5, 3, -1])
})

test("a Raon-1 nano GGUF's pieces: the per-layer table stays in the file, each layer has its sites, its operator and its per-layer part", async () => {
  const f = fixture()
  const pieces = await collect(f)
  assert.deepEqual(pieces.map((p) => p.kind), ["config", "perLayer", "layer", "layer", "finalNorm", "embedding", "embedding"])
  const perLayer = pieces[1]
  assert.ok(perLayer.kind === "perLayer")
  const table = f.tensors.get("per_layer_token_embd.weight")
  assert.deepEqual(perLayer.table, { offset: table?.offset, rowBytes: 64 * 4, type: GgmlType.F32, rows: 8 })
  assert.deepEqual([perLayer.proj.N, perLayer.proj.K, perLayer.norm.length], [64, 64, 32])
  const [rda, mla] = pieces.filter((p) => p.kind === "layer").map((p) => p.value)
  for (const layer of [rda, mla]) {
    // The selector's 17 rows padded to 20, over the four streams.
    assert.deepEqual([layer.mix?.operator.fn.N, layer.mix?.operator.fn.K, layer.mix?.ffn.fn.N], [20, 256, 20])
    assert.deepEqual([layer.mix?.operator.base.length, layer.mix?.operator.chart.length], [17, 17 * 16 + 32])
    assert.deepEqual([layer.perLayer?.gate.N, layer.perLayer?.gate.K, layer.perLayer?.proj.N, layer.perLayer?.proj.K], [32, 64, 64, 32])
    assert.deepEqual([layer.gateUp.N, layer.down.N], [128, 64])
  }
  assert.deepEqual([rda.operatorNorm[0], rda.ffnNorm[0], mla.operatorNorm[0]], [1, 2, 2])
  // Raon Delta Attention: q|k|v and the gate's rows; the decay's rank, then b and b + write; the rank to every key.
  assert.deepEqual([rda.linear?.inProj.N, rda.linear?.ba.N, rda.linear?.decay?.N, rda.linear?.decay?.K, rda.linear?.outProj.K], [128, 40, 32, 32, 32])
  assert.deepEqual([rda.linear?.conv.length, rda.linear?.dtBias.length, rda.linear?.norm.length], [4 * 96, 32, 8])
  assert.equal(rda.latent, undefined)
  // The MLA: q_a, kv_a and the gate in one projection; one small matrix a head for keys and for values, as halves.
  assert.deepEqual([mla.latent?.inProj.N, mla.latent?.qB.N, mla.latent?.qB.K, mla.latent?.o.K], [32 + 64 + 64, 256, 32, 64])
  assert.deepEqual([mla.latent?.kB.length, mla.latent?.vB.length], [4 * 32 * 32, 4 * 16 * 32])
  assert.equal(mla.latent?.kB[0], 0x3a00) // 0.75
  assert.equal(mla.latent?.vB[0], 0xb800) // -0.5
  assert.equal(mla.linear, undefined)
  const final = pieces.find((p) => p.kind === "finalNorm")
  assert.ok(final?.kind === "finalNorm" && final.value[0] === 1.5)
  assert.equal(pieces.filter((p) => p.kind === "embedding").reduce((n, p) => n + (p.kind === "embedding" ? p.value.N : 0), 0), 8)
})

test("an _MTP file's extra block is left out", async () => {
  const pieces = await collect(fixture(true))
  const config = pieces.find((p) => p.kind === "config")
  assert.ok(config?.kind === "config")
  assert.equal(config.config.numLayers, 2)
  assert.equal(pieces.filter((p) => p.kind === "layer").length, 2)
})

test("predictSpeed's shape of a Raon-1 nano GGUF, and the probe model made from it, are the loader's: every matrix's size and storage, every vector's length", async () => {
  const { ggufHeaderSpeedShape, probePieces, probePerLayerSource, probeConfig } = await import("./speed-probe.js")
  const { raonMatrixStorage } = await import("./load-raon.js")
  const f = fixture()
  const shape = ggufHeaderSpeedShape(f.header)
  assert.deepEqual(shape.kinds.map((k) => k.type), ["linear_attention", "full_attention"])
  assert.deepEqual(Object.keys(shape.kinds[0].matrices).sort(), ["ba", "decay", "down", "gateUp", "inProj", "mixFfn", "mixOperator", "outProj", "perLayerGate", "perLayerProj"])
  assert.deepEqual(Object.keys(shape.kinds[1].matrices).sort(), ["down", "gateUp", "latentIn", "mixFfn", "mixOperator", "o", "perLayerGate", "perLayerProj", "qB"])
  assert.ok(shape.perLayer)

  // What a piece is made of: each matrix as [rows, columns, bits, half zero points], each other array's length.
  const describe = (value: unknown): unknown => {
    if (value === null || typeof value !== "object") return value
    if (ArrayBuffer.isView(value)) return (value as unknown as { length: number }).length
    const m = value as Partial<QuantMatrix>
    if (m.quants && m.scales) return [m.N, m.K, m.bits, !!m.zeroHalves]
    return Object.fromEntries(Object.entries(value).map(([k, v]) => [k, describe(v)]))
  }
  const real = await collect(f)
  const probe: Array<WeightPiece> = []
  for await (const piece of probePieces(shape)) probe.push(piece)
  const layersOf = (pieces: Array<WeightPiece>) => pieces.flatMap((p) => (p.kind === "layer" ? [describe(p.value)] : []))
  assert.deepEqual(layersOf(probe), layersOf(real))
  const tableOf = (pieces: Array<WeightPiece>) => pieces.flatMap((p) => (p.kind === "perLayer" ? [{ proj: describe(p.proj), norm: p.norm.length, rowBytes: p.table.rowBytes, type: p.table.type }] : []))
  assert.deepEqual(tableOf(probe), tableOf(real))
  const head = (pieces: Array<WeightPiece>) => pieces.flatMap((p) => (p.kind === "embedding" ? [[p.value.K, p.value.bits]] : []))[0]
  assert.deepEqual(head(probe), head(real))
  // A token's row for the probe's engine: as many bytes as the table's rows have.
  const table = probe.find((p) => p.kind === "perLayer")
  assert.ok(table?.kind === "perLayer")
  const source = probePerLayerSource(shape)
  assert.ok(source)
  assert.equal((await source.read(3 * table.table.rowBytes, table.table.rowBytes)).length, table.table.rowBytes)
  assert.equal(probeConfig(shape).perLayer?.layers, 2)

  // Q4_K parts alone keep 4 bits (rows of whole super-blocks); anything else is 8-bit here.
  const t = (type: number) => ({ name: "t", dims: [256, 4], type, offset: 0, byteLength: 0 })
  assert.deepEqual(raonMatrixStorage([t(GgmlType.Q4_K)], 256), { bits: 4, halves: true })
  assert.deepEqual(raonMatrixStorage([t(GgmlType.Q4_K), t(GgmlType.Q6_K)], 256), { bits: 8, halves: false })
  assert.deepEqual(raonMatrixStorage([t(GgmlType.Q4_K)], 96), { bits: 8, halves: false })
})
