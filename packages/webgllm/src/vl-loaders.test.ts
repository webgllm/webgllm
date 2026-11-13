import assert from "node:assert/strict"
import { readFileSync } from "node:fs"
import { join } from "node:path"
import { test } from "node:test"
import { GgmlType } from "./gguf.js"
import { halfTable, halvesToFloats, toHalf } from "./half.js"
import { checkOnnxGraph, dequantizeNBitsRow, halfBytes, loadOnnx } from "./load-onnx.js"
import { planVision } from "./load-vision.js"
import { loadVisionGguf, planVisionGguf } from "./load-vision-gguf.js"
import { OnnxDataType, parseOnnxModel, type OnnxTensor } from "./onnx.js"
import { headPermutation } from "./repack.js"
import { projectorCandidates } from "./resolve-model.js"
import { floatMatrix, matMulNBitsMatrix, q8Matrix, stackLinears, visionBlocks, type VisionMatrix, type VisionPiece } from "./vision-weights.js"
import type { ByteSource } from "./weights.js"

// Liquid AI's real graphs (no weights) and config.json:
// LiquidAI/LFM2.5-VL-450M-ONNX's q4 decoder and q8 vision encoder, and
// LiquidAI/LFM3-VL-2B-ONNX's q4f16 decoder and q8 vision encoder.
const fixture = (dir: string, file: string) => new Uint8Array(readFileSync(join(process.cwd(), "src/test-fixtures", dir, file)))
const json = (dir: string) => JSON.parse(new TextDecoder().decode(fixture(dir, "config.json"))) as Record<string, unknown>
const LFM2_VL = "lfm2.5-vl-450m-onnx"
const LFM3_VL = "lfm3-vl-2b-onnx"

/** A weight of `m` as a float: scale · (q − zero), from the layout vision.ts uploads. */
function weightOf(m: VisionMatrix, n: number, k: number): number {
  const blocks = visionBlocks(m.K)
  const b = k >> 5
  const at = ((n >> 2) * blocks + b) * 4 + (n & 3)
  const zero = m.zeros ? m.zeros[at] : m.bits === 4 ? 8 : 128
  const j = k & 31
  const q = m.bits === 8 ? m.data[(n * blocks + b) * 32 + j] : (m.data[(n * blocks + b) * 16 + (j >> 1)] >> ((j & 1) << 2)) & 15
  return m.scales[at] * (q - zero)
}

test("LFM2.5-VL's decoder passes the graph check: float32 scales, embeddings in, a float head that is the embedding", async () => {
  const graph = fixture(LFM2_VL, "decoder_model_merged_q4.onnx")
  assert.deepEqual(checkOnnxGraph(graph, json(LFM2_VL)), { dataFiles: ["decoder_model_merged_q4.onnx_data"] })
  const zeros: ByteSource = { size: 2 ** 31, read: async (_, length) => new Uint8Array(length) }
  // Data files by name (a graph with several): asked only for tensors that are in one, not for those in the graph itself.
  const asked = new Set<string>()
  const byName = (location: string) => {
    asked.add(location)
    return zeros
  }
  for await (const piece of loadOnnx({ graph, data: byName, config: json(LFM2_VL) })) {
    assert.equal(piece.kind, "config")
    assert.deepEqual([...asked], ["decoder_model_merged_q4.onnx_data"])
    break
  }
  for await (const piece of loadOnnx({ graph, data: zeros, config: json(LFM2_VL) })) {
    assert.ok(piece.kind === "config")
    const c = piece.config
    assert.deepEqual(
      [c.hiddenSize, c.numLayers, c.numHeads, c.numKvHeads, c.headDim, c.vocabSize, c.intermediateSize, c.convKernel, c.attentionGate],
      [1024, 16, 16, 8, 64, 65536, 4608, 3, false]
    )
    break
  }
})

test("float32 scales round to float16; MatMulNBits rows dequantize by block, 4-bit and 8-bit", () => {
  const scales = new Float32Array([0.5, 0.1, -3])
  const tensor = (dataType: number): OnnxTensor => ({ name: "s", dims: [3], dataType })
  const halves = halfBytes(tensor(OnnxDataType.FLOAT), new Uint8Array(scales.buffer))
  const table = halfTable()
  const back = Array.from({ length: 3 }, (_, i) => table[halves[2 * i] | (halves[2 * i + 1] << 8)])
  assert.equal(back[0], 0.5)
  assert.ok(Math.abs(back[1] - 0.1) < 1e-4)
  assert.equal(back[2], -3)
  const same = new Uint8Array([1, 2, 3, 4])
  assert.equal(halfBytes(tensor(OnnxDataType.FLOAT16), same), same)
  assert.throws(() => halfBytes(tensor(OnnxDataType.UINT8), same))
  // Two rows of K = 64 (two blocks): 8-bit bytes with a zero point per block; 4-bit nibbles with zero points packed two a byte.
  const half = (v: number) => [toHalf(v) & 255, toHalf(v) >> 8]
  const scaleBytes = new Uint8Array([...half(0.5), ...half(2), ...half(1), ...half(0.25)])
  const out = new Float32Array(64)
  const q8 = Uint8Array.from({ length: 128 }, (_, i) => i)
  dequantizeNBitsRow(8, 64, q8, scaleBytes, new Uint8Array([10, 20, 30, 40]), 1, out)
  assert.equal(out[0], 1 * (64 - 30))
  assert.equal(out[33], 0.25 * (64 + 33 - 40))
  dequantizeNBitsRow(8, 64, q8, scaleBytes, null, 0, out)
  assert.equal(out[5], 0.5 * (5 - 128))
  const q4 = Uint8Array.from({ length: 64 }, (_, i) => (i % 16) | (((i + 1) % 16) << 4))
  dequantizeNBitsRow(4, 64, q4, scaleBytes, new Uint8Array([0x21, 0x43]), 1, out)
  // Row 1, block 0 (zero point 3): element 2j is byte j's low nibble, 2j + 1 its high one.
  assert.equal(out[0], 1 * (0 - 3))
  assert.equal(out[1], 1 * (1 - 3))
  assert.equal(out[32 + 5], 0.25 * ((2 + 1) % 16 - 4))
  dequantizeNBitsRow(4, 64, q4, scaleBytes, null, 0, out)
  assert.equal(out[2], 0.5 * (1 - 8))
})

test("LFM3-VL's decoder loads: 8-bit projections quantized again, each full-attention head's rotated dims first", async () => {
  const graph = fixture(LFM3_VL, "decoder_q4f16.onnx")
  assert.deepEqual(checkOnnxGraph(graph, json(LFM3_VL)), { dataFiles: ["decoder_q4f16.onnx_data"] })
  const init = (name: string) => {
    const t = parseOnnxModel(graph).initializers.get(name)
    assert.ok(t?.external, name)
    return t.external
  }
  // The data: zeros, but the rotary sin caches (sin(position · frequency); a
  // full layer rotates its first 64 dims of 128) and layer 2's q_proj (row
  // r: every weight (r + 1) / 4).
  const half = (v: number) => toHalf(v)
  const sin = (theta: number, rotated: number) => (i: number) => {
    const j = (i >> 1) % 64
    const h = half(j < rotated / 2 ? Math.sin(Math.floor((i >> 1) / 64) * theta ** ((-2 * j) / 128)) : 0)
    return i & 1 ? h >> 8 : h & 255
  }
  const p = "model.layers.2.attn.q_proj.MatMul.weight_"
  const fills: Array<{ at: { offset: number; length: number }; byte: (i: number) => number }> = [
    { at: init("sin_cache"), byte: sin(1e6, 64) },
    { at: init("sin_cache_local"), byte: sin(1e4, 128) },
    { at: init(`${p}Q8G32`), byte: () => 129 },
    { at: init(`${p}zp`), byte: () => 128 },
    { at: init(`${p}scale`), byte: (i) => (i & 1 ? half((Math.floor((i >> 1) / 64) + 1) / 4) >> 8 : half((Math.floor((i >> 1) / 64) + 1) / 4) & 255) }
  ]
  // (The q norm's weights are in the graph itself.)
  const qNormRaw = parseOnnxModel(graph).initializers.get("model.layers.2.attn.q_norm.layernorm.weight")?.raw
  assert.ok(qNormRaw)
  const qNorm = halvesToFloats(qNormRaw)
  const data: ByteSource = {
    size: 2 ** 31,
    read: async (offset, length) => {
      const out = new Uint8Array(length)
      for (const { at, byte } of fills) {
        const from = Math.max(offset, at.offset)
        const to = Math.min(offset + length, at.offset + at.length)
        for (let q = from; q < to; q++) out[q - offset] = byte(q - at.offset)
      }
      return out
    }
  }
  const table = halfTable()
  const perm = headPermutation(16, 128, 64)
  assert.deepEqual([perm[0], perm[31], perm[32], perm[63], perm[64], perm[95], perm[96], perm[128 + 40]], [0, 31, 64, 95, 32, 63, 96, 128 + 72])
  let checked = 0
  for await (const piece of loadOnnx({ graph, data, config: json(LFM3_VL) })) {
    if (piece.kind === "config") {
      const c = piece.config
      assert.deepEqual([c.hiddenSize, c.numLayers, c.numHeads, c.numKvHeads, c.headDim, c.ropeDimsFull, c.ropeDimsSliding, c.vocabSize, c.attentionGate], [2048, 24, 16, 4, 128, 64, 128, 128000, true])
    }
    if (piece.kind !== "layer") continue
    const { attention, down } = piece.value
    // A conv layer's 8-bit down projection (layer 0's) is an 8-bit matrix.
    if (piece.layer === 0) assert.equal(down.bits, 8)
    if (piece.layer === 1) assert.equal(piece.value.gateUp.bits, 4, "4-bit projections stay 4-bit")
    if (piece.layer !== 2) continue
    assert.ok(attention)
    const m = attention.qkvg
    // q (8-bit), k, v and the 4-bit gate as one 8-bit matrix.
    assert.deepEqual([m.bits, m.N, m.K], [8, 2048 + 512 + 512 + 16, 2048])
    const q = new Int8Array(m.quants.buffer, m.quants.byteOffset, m.quants.byteLength)
    for (const d of [0, 31, 32, 63, 64, 100, 127, 128 + 33, 2047]) {
      const at = (d >> 2) * 64 * 4 + (d & 3)
      const got = q[at * 32] * table[m.scales[at]]
      assert.ok(Math.abs(got - (perm[d] + 1) / 4) < 2e-3 * ((perm[d] + 1) / 4), `q row ${d} is the graph's row ${perm[d]}`)
    }
    // The norms' dims move with their rows.
    assert.deepEqual(Array.from(attention.qNorm), Array.from(headPermutation(1, 128, 64), (src) => qNorm[src]))
    assert.notDeepEqual(Array.from(attention.qNorm), Array.from(qNorm))
    checked++
    break
  }
  assert.equal(checked, 1)
})

test("the ONNX vision plans read Liquid AI's graphs: SigLIP2 with LFM2-VL's projector, and LFM3-VL's tower", () => {
  const siglip = planVision(fixture(LFM2_VL, "vision_encoder_q8.onnx"))
  assert.deepEqual(siglip.config, {
    width: 768, heads: 12, kvHeads: 12, headDim: 64, ffn: 3072, mergerWidth: 2048, outWidth: 1024, patchValues: 768, merge: 2,
    layers: Array.from({ length: 12 }, () => "attention"), norm: "layer", eps: siglip.config.eps, mergerEps: siglip.config.mergerEps,
    mlp: "tanh", mergerAct: "erf", grid: 16, interpolation: "half-pixel", rope: false, qkNorm: false, positions: "sequence"
  })
  assert.ok(Math.abs(siglip.config.eps - 1e-6) < 1e-9 && Math.abs(siglip.config.mergerEps - 1e-6) < 1e-9)
  assert.deepEqual([siglip.bits, siglip.zeros], [8, true])
  assert.deepEqual([...new Set(siglip.reads.flatMap((r) => (r.tensor.external ? [r.tensor.external.location] : [])))], ["vision_encoder_q8.onnx_data"])

  const tower = planVision(fixture(LFM3_VL, "vision_encoder_q8.onnx"))
  const c = tower.config
  assert.deepEqual(
    [c.width, c.heads, c.kvHeads, c.headDim, c.ffn, c.mergerWidth, c.outWidth, c.patchValues, c.merge, c.norm, c.mlp, c.mergerAct, c.grid, c.rope, c.qkNorm, c.positions],
    [1024, 16, 8, 64, 2560, 4096, 2048, 1536, 2, "rms", "swiglu", "erf", 0, true, true, "sequence"]
  )
  assert.deepEqual(c.layers, ["conv", "conv", "attention", "conv", "attention", "conv", "attention", "conv", "attention", "conv", "attention", "conv", "attention", "conv"])
  assert.ok(Math.abs(c.eps - 1e-5) < 1e-9 && Math.abs(c.mergerEps - 1e-6) < 1e-9)
  // 8-bit weights in blocks of 128, without zero points.
  assert.deepEqual([tower.bits, tower.zeros, tower.patch.w.blockSize], [8, false, 128])
  // A decoder isn't a vision encoder.
  assert.throws(() => planVision(fixture(LFM3_VL, "decoder_q4f16.onnx")))
})

test("vision matrices: MatMulNBits by bits and block size, floats and Q8_0 at 8 bits, stacked rows", () => {
  // 4-bit, K = 64: nibbles as stored, zero points two a byte.
  const quant4 = Uint8Array.from({ length: 4 * 2 * 16 }, (_, i) => (i * 7) & 255)
  const scales = new Float32Array([1, 2, 3, 4, 5, 6, 7, 8])
  const zp4 = new Uint8Array([0x21, 0x43, 0x65, 0x87])
  const four = matMulNBitsMatrix({ N: 4, K: 64, bits: 4, blockSize: 32, scaleBytes: 4 }, quant4, new Uint8Array(scales.buffer), zp4)
  assert.equal(weightOf(four, 2, 37), 6 * (((quant4[(2 * 2 + 1) * 16 + 2] >> 4) & 15) - 6))
  assert.equal(weightOf(four, 0, 0), 1 * ((quant4[0] & 15) - 1))
  // 8-bit in blocks of 128 over K = 160: a scale serves its block's four blocks of 32; the short last block reads zero points.
  const quant8 = Uint8Array.from({ length: 4 * 2 * 128 }, (_, i) => (i * 13) & 255)
  const eight = matMulNBitsMatrix({ N: 4, K: 160, bits: 8, blockSize: 128, scaleBytes: 4 }, quant8, new Uint8Array(scales.buffer), null)
  assert.equal(eight.data.length, 4 * 5 * 32)
  assert.equal(weightOf(eight, 1, 100), 3 * (quant8[256 + 100] - 128))
  assert.equal(weightOf(eight, 3, 159), 8 * (quant8[3 * 256 + 159] - 128))
  assert.equal(eight.zeros, null)
  assert.throws(() => matMulNBitsMatrix({ N: 4, K: 64, bits: 4, blockSize: 64, scaleBytes: 4 }, quant4, new Uint8Array(scales.buffer), null))
  // Floats: within half a step of each block's scale; K needn't be whole blocks.
  const values = Float32Array.from({ length: 4 * 40 }, (_, i) => Math.sin(i) * (1 + (i % 7)))
  const floats = floatMatrix(values, 4, 40)
  for (const [n, k] of [[0, 0], [1, 31], [2, 32], [3, 39]]) {
    const step = floats.scales[((n >> 2) * 2 + (k >> 5)) * 4 + (n & 3)]
    assert.ok(Math.abs(weightOf(floats, n, k) - values[n * 40 + k]) <= step / 2 + 1e-6)
  }
  // Q8_0 blocks: an fp16 scale, then 32 int8.
  const block = (scale: number, at: (j: number) => number) => [toHalf(scale) & 255, toHalf(scale) >> 8, ...Array.from({ length: 32 }, (_, j) => at(j) & 255)]
  const q8 = q8Matrix(new Uint8Array([0, 1, 2, 3].flatMap((n) => block(0.5 * (n + 1), (j) => j - 16 + n))), 4, 32)
  assert.equal(weightOf(q8, 3, 0), 2 * (-16 + 3))
  assert.equal(weightOf(q8, 1, 31), 1 * (15 + 1))
  // Rows stack in order; a part without a bias gets zeros where another has one.
  const stacked = stackLinears([{ m: q8, bias: new Float32Array([1, 2, 3, 4]) }, { m: floatMatrix(new Float32Array(4 * 32).fill(0.25), 4, 32), bias: null }])
  assert.equal(stacked.m.N, 8)
  assert.equal(weightOf(stacked.m, 3, 0), weightOf(q8, 3, 0))
  assert.ok(Math.abs(weightOf(stacked.m, 5, 7) - 0.25) < 1e-6)
  assert.deepEqual(Array.from(stacked.bias!), [1, 2, 3, 4, 0, 0, 0, 0])
  assert.equal(stackLinears([{ m: q8, bias: null }, { m: q8, bias: null }]).bias, null)
  assert.throws(() => stackLinears([{ m: q8, bias: null }, { m: four, bias: null }]))
})

/** A GGUF file's bytes (version 3): metadata, then tensors with their data. */
function ggufFile(metadata: Record<string, string | number | boolean>, tensors: Array<{ name: string; dims: Array<number>; type: number; data: Uint8Array }>): Uint8Array {
  const bytes: Array<number> = []
  const u32 = (v: number) => bytes.push(v & 255, (v >> 8) & 255, (v >> 16) & 255, (v >>> 24) & 255)
  const u64 = (v: number) => (u32(v), u32(0))
  const text = (s: string) => {
    const encoded = new TextEncoder().encode(s)
    u64(encoded.length)
    bytes.push(...encoded)
  }
  bytes.push(...new TextEncoder().encode("GGUF"))
  u32(3)
  u64(tensors.length)
  u64(Object.keys(metadata).length)
  for (const [key, value] of Object.entries(metadata)) {
    text(key)
    if (typeof value === "string") (u32(8), text(value))
    else if (typeof value === "boolean") (u32(7), bytes.push(value ? 1 : 0))
    else if (Number.isInteger(value)) (u32(4), u32(value))
    else (u32(6), bytes.push(...new Uint8Array(new Float32Array([value]).buffer)))
  }
  let offset = 0
  for (const t of tensors) {
    text(t.name)
    u32(t.dims.length)
    for (const d of t.dims) u64(d)
    u32(t.type)
    u64(offset)
    offset += Math.ceil(t.data.length / 32) * 32
  }
  while (bytes.length % 32) bytes.push(0)
  const out = new Uint8Array(bytes.length + offset)
  out.set(bytes)
  let at = bytes.length
  for (const t of tensors) {
    out.set(t.data, at)
    at += Math.ceil(t.data.length / 32) * 32
  }
  return out
}

const f32 = (name: string, dims: Array<number>, at: (i: number) => number = () => 0) => {
  const values = Float32Array.from({ length: dims.reduce((a, b) => a * b, 1) }, (_, i) => at(i))
  return { name, dims, type: GgmlType.F32, data: new Uint8Array(values.buffer) }
}
const f16 = (name: string, dims: Array<number>, at: (i: number) => number = () => 0) => {
  const values = Uint16Array.from({ length: dims.reduce((a, b) => a * b, 1) }, (_, i) => toHalf(at(i)))
  return { name, dims, type: GgmlType.F16, data: new Uint8Array(values.buffer) }
}
/** A [K, N] Q8_0 matrix whose every weight is `value` (scale value / 100, q 100). */
const q8 = (name: string, K: number, N: number, value = 0) => {
  const data = new Uint8Array(N * (K / 32) * 34)
  for (let b = 0; b < N * (K / 32); b++) {
    data[b * 34] = toHalf(value / 100) & 255
    data[b * 34 + 1] = toHalf(value / 100) >> 8
    data.fill(100, b * 34 + 2, b * 34 + 34)
  }
  return { name, dims: [K, N], type: GgmlType.Q8_0, data }
}
const source = (bytes: Uint8Array): ByteSource => ({ size: bytes.length, read: async (offset, length) => bytes.slice(offset, offset + length) })
const piecesOf = async (bytes: Uint8Array) => {
  const pieces: Array<VisionPiece> = []
  for await (const piece of loadVisionGguf(source(bytes))) pieces.push(piece)
  return pieces
}
const clip = { "general.architecture": "clip", "clip.vision.embedding_length": 32, "clip.vision.feed_forward_length": 64, "clip.vision.projection_dim": 32, "clip.vision.patch_size": 16 }

test("a GGUF lfm2 projector loads as SigLIP2: the patch kernel's columns as y, x, channel; floats at 8 bits", async () => {
  const W = 32
  const linear = (name: string, K: number, N: number, value = 0) => [q8(`${name}.weight`, K, N, value), f32(`${name}.bias`, [N], (i) => i)]
  const norm = (name: string) => [f32(`${name}.weight`, [W], () => 1), f32(`${name}.bias`, [W])]
  const file = ggufFile(
    { ...clip, "clip.projector_type": "lfm2", "clip.vision.block_count": 1, "clip.vision.attention.head_count": 2, "clip.vision.attention.layer_norm_epsilon": 1e-6, "clip.vision.projector.scale_factor": 2 },
    [
      // The kernel is [out][channel][y][x]: channel c holds c + 1.
      f32("v.patch_embd.weight", [16, 16, 3, W], (i) => (Math.floor(i / 256) % 3) + 1),
      f32("v.patch_embd.bias", [W]),
      f32("v.position_embd.weight", [W, 4], (i) => i),
      ...norm("v.blk.0.ln1"),
      ...linear("v.blk.0.attn_q", W, W, 0.5),
      ...linear("v.blk.0.attn_k", W, W, 1),
      ...linear("v.blk.0.attn_v", W, W, 2),
      ...linear("v.blk.0.attn_out", W, W),
      ...norm("v.blk.0.ln2"),
      ...linear("v.blk.0.ffn_up", W, 64),
      // A float matrix (as SigLIP2's 4,304-wide rows are, which Q8_0 can't hold).
      f16("v.blk.0.ffn_down.weight", [64, W], () => 0.75),
      f32("v.blk.0.ffn_down.bias", [W]),
      ...norm("v.post_ln"),
      ...linear("mm.1", 4 * W, 64),
      ...linear("mm.2", 64, W)
    ]
  )
  const { config, bits, zeros } = await planVisionGguf(source(file))
  assert.deepEqual(
    [config.width, config.heads, config.kvHeads, config.headDim, config.ffn, config.mergerWidth, config.outWidth, config.patchValues, config.merge, config.grid],
    [32, 2, 2, 16, 64, 64, 32, 768, 2, 2]
  )
  assert.deepEqual([config.norm, config.mlp, config.mergerAct, config.interpolation, config.rope, config.qkNorm, config.positions, bits, zeros], ["layer", "tanh", "erf", "antialias", false, false, "sequence", 8, false])
  const pieces = await piecesOf(file)
  assert.deepEqual(pieces.map((p) => p.kind), ["config", "patch", "block", "merger"])
  const patch = pieces[1] as Extract<VisionPiece, { kind: "patch" }>
  // Pixel (y 3, x 5), channel c of a patch row is column (3 · 16 + 5) · 3 + c.
  for (const c of [0, 1, 2]) assert.ok(Math.abs(weightOf(patch.patch.m, 7, (3 * 16 + 5) * 3 + c) - (c + 1)) < 0.02)
  assert.equal(patch.positions?.length, 4 * W)
  assert.equal(patch.invFreq, undefined)
  const block = (pieces[2] as Extract<VisionPiece, { kind: "block" }>).value
  assert.ok(block.type === "attention")
  // q, k, v stacked, their biases with them.
  assert.deepEqual([block.qkv.m.N, block.qkv.m.K, block.qkv.m.bits], [3 * W, W, 8])
  assert.ok(Math.abs(weightOf(block.qkv.m, 0, 0) - 0.5) < 1e-3 && Math.abs(weightOf(block.qkv.m, W, 3) - 1) < 1e-3 && Math.abs(weightOf(block.qkv.m, 2 * W + 1, 31) - 2) < 1e-3)
  assert.deepEqual(Array.from(block.qkv.bias!.subarray(W, W + 3)), [0, 1, 2])
  assert.deepEqual([block.fc1.m.N, block.fc1.m.K, block.fc2.m.N, block.fc2.m.K, block.fc2.m.bits], [64, W, W, 64, 8])
  assert.ok(Math.abs(weightOf(block.fc2.m, 5, 40) - 0.75) < 1e-6)
  const merger = pieces[3] as Extract<VisionPiece, { kind: "merger" }>
  assert.deepEqual([merger.linear1.m.N, merger.linear1.m.K, merger.linear2.m.N, merger.linear2.m.K], [64, 4 * W, W, 64])
  // Not a projector, or a type the encoder doesn't run.
  await assert.rejects(planVisionGguf(source(ggufFile({ "general.architecture": "lfm2" }, []))), /not a vision projector/)
  await assert.rejects(planVisionGguf(source(ggufFile({ ...clip, "clip.projector_type": "qwen3vl" }, []))), /"qwen3vl" projector/)
  // A tensor type the loader doesn't read (BF16) fails with the header, before any weights.
  const bf16 = { ...f16("v.blk.0.attn_out.weight", [W, W]), type: GgmlType.BF16 }
  const meta = { ...clip, "clip.projector_type": "lfm2", "clip.vision.block_count": 1, "clip.vision.attention.head_count": 2, "clip.vision.attention.layer_norm_epsilon": 1e-6, "clip.vision.projector.scale_factor": 2 }
  await assert.rejects(planVisionGguf(source(ggufFile(meta, [f32("v.position_embd.weight", [W, 4]), q8("mm.1.weight", 4 * W, 64), bf16]))), /v\.blk\.0\.attn_out\.weight has tensor type 30/)
})

test("a GGUF lfm3vl projector loads as LFM3-VL's tower: a kernel per frame, conv and grouped-query blocks, RoPE's frequencies", async () => {
  const W = 32
  const w = (name: string, K: number, N: number, value = 0) => q8(`${name}.weight`, K, N, value)
  const rms = (name: string) => f32(`${name}.weight`, [W], () => 1)
  const mlp = (p: string) => [rms(`${p}ln2`), w(`${p}ffn_gate`, W, 64, 1), w(`${p}ffn_up`, W, 64, 2), w(`${p}ffn_down`, 64, W)]
  const file = ggufFile(
    {
      ...clip, "clip.vision.projector_type": "lfm3vl", "clip.vision.block_count": 2, "clip.vision.attention.head_count": 2, "clip.vision.attention.head_count_kv": 1,
      "clip.vision.attention.layer_norm_epsilon": 1e-5, "clip.vision.spatial_merge_size": 2
    },
    [
      f16("v.patch_embd.weight", [16, 16, 3, W], () => 1),
      f16("v.patch_embd.weight.1", [16, 16, 3, W], () => 2),
      rms("v.embedding_norm"),
      rms("v.blk.0.ln1"),
      w("v.blk.0.conv2d_in_proj", W, 3 * W),
      w("v.blk.0.conv2d_out_proj", W, W),
      f32("v.blk.0.conv2d_spatial.weight", [3, 3, 1, W], (i) => i),
      ...mlp("v.blk.0."),
      rms("v.blk.1.ln1"),
      w("v.blk.1.attn_q", W, W),
      w("v.blk.1.attn_k", W, 16),
      w("v.blk.1.attn_v", W, 16),
      w("v.blk.1.attn_out", W, W),
      f32("v.blk.1.attn_q_norm.weight", [16], (i) => i),
      f32("v.blk.1.attn_k_norm.weight", [16], (i) => -i),
      ...mlp("v.blk.1."),
      f32("mm.input_norm.weight", [W], () => 1),
      f32("mm.input_norm.bias", [W]),
      w("mm.0", 4 * W, 64),
      f32("mm.0.bias", [64]),
      w("mm.2", 64, W),
      f32("mm.2.bias", [W])
    ]
  )
  const { config } = await planVisionGguf(source(file))
  assert.deepEqual(config.layers, ["conv", "attention"])
  assert.deepEqual(
    [config.heads, config.kvHeads, config.headDim, config.patchValues, config.norm, config.mlp, config.grid, config.rope, config.qkNorm, config.mergerEps],
    [2, 1, 16, 1536, "rms", "swiglu", 0, true, true, 1e-6]
  )
  const pieces = await piecesOf(file)
  const patch = pieces[1] as Extract<VisionPiece, { kind: "patch" }>
  // Channel c, frame t, pixel (y, x) is column ((c · 2 + t) · 16 + y) · 16 + x: frame 0 the first kernel, frame 1 the second.
  assert.ok(Math.abs(weightOf(patch.patch.m, 3, ((1 * 2 + 0) * 16 + 4) * 16 + 9) - 1) < 0.01)
  assert.ok(Math.abs(weightOf(patch.patch.m, 3, ((1 * 2 + 1) * 16 + 4) * 16 + 9) - 2) < 0.01)
  assert.equal(patch.patch.bias, null)
  assert.ok(patch.norm && patch.norm.bias === null)
  // Frequencies 1e6^(-j / 4) for heads of 16.
  assert.deepEqual(Array.from(patch.invFreq!).map((v) => Number(v.toPrecision(6))), [1, 1e6 ** -0.25, 1e6 ** -0.5, 1e6 ** -0.75].map((v) => Number(Math.fround(v).toPrecision(6))))
  const conv = (pieces[2] as Extract<VisionPiece, { kind: "block" }>).value
  assert.ok(conv.type === "conv")
  assert.deepEqual([conv.inProj.m.N, conv.outProj.m.N, conv.kernel.length, conv.kernel[9 * 5 + 4]], [3 * W, W, 9 * W, 49])
  // SwiGLU's gate rows, then its up rows.
  assert.deepEqual([conv.fc1.m.N, conv.fc2.m.K], [128, 64])
  assert.ok(Math.abs(weightOf(conv.fc1.m, 10, 0) - 1) < 1e-3 && Math.abs(weightOf(conv.fc1.m, 64 + 10, 0) - 2) < 1e-3)
  const attention = (pieces[3] as Extract<VisionPiece, { kind: "block" }>).value
  assert.ok(attention.type === "attention")
  assert.deepEqual([attention.qkv.m.N, attention.qNorm?.[5], attention.kNorm?.[5], attention.qkv.bias], [W + 16 + 16, 5, -5, null])
})

test("a GGUF qwen3vl_merger projector loads as Qwen3.5's ViT: fused q|k|v, a corner-aligned position table, RoPE at base 1e4", async () => {
  const W = 32
  const linear = (name: string, K: number, N: number, value: number) => [f16(`${name}.weight`, [K, N], () => value), f32(`${name}.bias`, [N], (i) => i)]
  const norm = (name: string) => [f32(`${name}.weight`, [W], () => 1), f32(`${name}.bias`, [W])]
  const meta = {
    ...clip, "clip.projector_type": "qwen3vl_merger", "clip.vision.block_count": 1, "clip.vision.attention.head_count": 2,
    "clip.vision.attention.layer_norm_epsilon": 1e-6, "clip.vision.spatial_merge_size": 2
  }
  const tensors = [
    f16("v.patch_embd.weight", [16, 16, 3, W], () => 1),
    f16("v.patch_embd.weight.1", [16, 16, 3, W], () => 2),
    f32("v.patch_embd.bias", [W], (i) => i),
    f32("v.position_embd.weight", [W, 9], (i) => i),
    ...norm("v.blk.0.ln1"),
    // (q rows 0.5, k rows 1, v rows 2.)
    f16("v.blk.0.attn_qkv.weight", [W, 3 * W], (i) => [0.5, 1, 2][Math.floor(i / (W * W))]),
    f32("v.blk.0.attn_qkv.bias", [3 * W], (i) => i),
    ...linear("v.blk.0.attn_out", W, W, 0.25),
    ...norm("v.blk.0.ln2"),
    ...linear("v.blk.0.ffn_up", W, 64, 0.5),
    ...linear("v.blk.0.ffn_down", 64, W, 0.75),
    ...norm("v.post_ln"),
    ...linear("mm.0", 4 * W, 128, 1),
    ...linear("mm.2", 128, W, 1)
  ]
  const file = ggufFile(meta, tensors)
  const { config, bits } = await planVisionGguf(source(file))
  assert.deepEqual(
    [config.width, config.heads, config.kvHeads, config.headDim, config.ffn, config.mergerWidth, config.outWidth, config.patchValues, config.merge, config.grid, bits],
    [32, 2, 2, 16, 64, 128, 32, 1536, 2, 3, 8]
  )
  assert.deepEqual([config.norm, config.mlp, config.mergerAct, config.interpolation, config.rope, config.qkNorm, config.positions], ["layer", "tanh", "erf", "corners", true, false, "grid"])
  const pieces = await piecesOf(file)
  assert.deepEqual(pieces.map((p) => p.kind), ["config", "patch", "block", "merger"])
  const patch = pieces[1] as Extract<VisionPiece, { kind: "patch" }>
  assert.ok(Math.abs(weightOf(patch.patch.m, 3, ((2 * 2 + 0) * 16 + 4) * 16 + 9) - 1) < 0.01 && Math.abs(weightOf(patch.patch.m, 3, ((2 * 2 + 1) * 16 + 4) * 16 + 9) - 2) < 0.01)
  assert.deepEqual([patch.patch.bias?.[5], patch.positions?.length, patch.norm], [5, 9 * W, undefined])
  assert.deepEqual(Array.from(patch.invFreq!).map((v) => Number(v.toPrecision(5))), [1, 0.1, 0.01, 0.001])
  const block = (pieces[2] as Extract<VisionPiece, { kind: "block" }>).value
  assert.ok(block.type === "attention")
  assert.deepEqual([block.qkv.m.N, block.qkv.m.K, block.qkv.bias?.[2 * W + 1], block.qNorm], [3 * W, W, 2 * W + 1, undefined])
  assert.ok(Math.abs(weightOf(block.qkv.m, 1, 5) - 0.5) < 1e-6 && Math.abs(weightOf(block.qkv.m, W + 1, 5) - 1) < 1e-6 && Math.abs(weightOf(block.qkv.m, 2 * W + 1, 5) - 2) < 1e-6)
  assert.deepEqual([block.fc1.m.N, block.fc2.m.N], [64, W])
  // Qwen3-VL's deepstack features aren't run: refused with the header.
  await assert.rejects(planVisionGguf(source(ggufFile(meta, [...tensors, f32("v.deepstack.0.norm.weight", [W])]))), /deepstack/)
})

test("a repository's projectors by preference: Q8_0, then floats", () => {
  const files = ["LFM2.5-VL-450M-Q4_0.gguf", "mmproj-LFM2.5-VL-450m-BF16.gguf", "mmproj-LFM2.5-VL-450m-F16.gguf", "mmproj-LFM2.5-VL-450m-F32.gguf", "mmproj-LFM2.5-VL-450m-Q8_0.gguf", "README.md"]
  // (BF16 isn't a type the loader reads.)
  assert.deepEqual(projectorCandidates(files), ["mmproj-LFM2.5-VL-450m-Q8_0.gguf", "mmproj-LFM2.5-VL-450m-F16.gguf", "mmproj-LFM2.5-VL-450m-F32.gguf"])
  assert.deepEqual(projectorCandidates(["model.gguf", "tokenizer.json"]), [])
})
