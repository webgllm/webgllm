import assert from "node:assert/strict"
import test from "node:test"
import { toHalf } from "./half.js"
import { dequantizeQ4_K, dequantizeQ8_0, matrixRow, newQuantMatrix, packQ4_K } from "./repack.js"

/** A small deterministic generator (mulberry32). */
function random(seed: number) {
  let a = seed >>> 0
  return () => {
    a = (a + 0x6d2b79f5) >>> 0
    let t = a
    t = Math.imul(t ^ (t >>> 15), t | 1)
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61)
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}

/** `supers` ggml Q4_K super-blocks: fp16 d and dmin, 8 six-bit scales and mins packed as ggml packs them, 128 bytes of nibbles. */
function q4kBlocks(seed: number, supers: number, zeroScale = false) {
  const rand = random(seed)
  const bytes = new Uint8Array(supers * 144)
  const view = new DataView(bytes.buffer)
  for (let s = 0; s < supers; s++) {
    const p = s * 144
    view.setUint16(p, toHalf(0.001 + 0.004 * rand()), true)
    view.setUint16(p + 2, toHalf(0.002 + 0.006 * rand()), true)
    const scales = Array.from({ length: 8 }, () => 1 + Math.floor(rand() * 63))
    const mins = Array.from({ length: 8 }, () => Math.floor(rand() * 64))
    if (zeroScale) scales[s % 8] = 0
    for (let j = 0; j < 4; j++) {
      bytes[p + 4 + j] = (scales[j] & 63) | ((scales[j + 4] >> 4) << 6)
      bytes[p + 8 + j] = (mins[j] & 63) | ((mins[j + 4] >> 4) << 6)
      bytes[p + 12 + j] = (scales[j + 4] & 15) | ((mins[j + 4] & 15) << 4)
    }
    for (let i = 0; i < 128; i++) bytes[p + 16 + i] = Math.floor(rand() * 256)
  }
  return bytes
}

function worst(m: ReturnType<typeof newQuantMatrix>, want: Float32Array, rows: number, K: number, srcRows?: Array<number>) {
  let error = 0
  let scale = 0
  for (let r = 0; r < rows; r++) {
    const got = matrixRow(m, r)
    assert.ok(got)
    const from = (srcRows ? srcRows[r] : r) * K
    for (let k = 0; k < K; k++) {
      error = Math.max(error, Math.abs(got[k] - want[from + k]))
      scale = Math.max(scale, Math.abs(want[from + k]))
    }
  }
  return error / scale
}

test("Q4_K rows packed as 4-bit blocks with fractional zero points are the file's weights", () => {
  const rows = 8
  const K = 512
  const bytes = q4kBlocks(1, (rows * K) / 256)
  const want = new Float32Array(rows * K)
  dequantizeQ4_K(bytes, rows, K, want)
  const m = newQuantMatrix(rows, K, 4, "halves")
  packQ4_K(m, 0, bytes, rows)
  assert.equal(m.zeros, null)
  assert.ok(m.zeroHalves)
  // (A scale and a zero point each rounded to a half: under a thousandth of the largest weight.)
  assert.ok(worst(m, want, rows, K) < 1e-3, `error ${worst(m, want, rows, K)}`)
})

test("Q4_K rows land at their row offset, in the order asked for", () => {
  const rows = 4
  const K = 256
  const bytes = q4kBlocks(2, rows)
  const want = new Float32Array(rows * K)
  dequantizeQ4_K(bytes, rows, K, want)
  const m = newQuantMatrix(8, K, 4, "halves")
  const order = [2, 0, 3, 1]
  packQ4_K(m, 4, bytes, rows, order)
  for (let r = 0; r < 4; r++) assert.ok(matrixRow(m, r)?.every((v) => v === 0), "rows before the offset stay zero")
  const shifted = { ...m, N: 4, quants: m.quants.subarray(m.quants.length / 2), scales: m.scales.subarray(m.scales.length / 2), zeroHalves: m.zeroHalves?.subarray(m.scales.length / 2) }
  assert.ok(worst(shifted, want, rows, K, order) < 1e-3)
})

test("a Q4_K sub-block with no scale keeps its constant weights", () => {
  const rows = 4
  const K = 256
  const bytes = q4kBlocks(3, rows, true)
  const want = new Float32Array(rows * K)
  dequantizeQ4_K(bytes, rows, K, want)
  // (Each row's zero-scale sub-block is -dmin · min everywhere.)
  for (let r = 0; r < rows; r++) {
    const at = r * K + (r % 8) * 32
    assert.ok(want.subarray(at, at + 32).every((v) => v === want[at]))
  }
  const m = newQuantMatrix(rows, K, 4, "halves")
  packQ4_K(m, 0, bytes, rows)
  assert.ok(worst(m, want, rows, K) < 1e-3, `error ${worst(m, want, rows, K)}`)
})

test("packQ4_K needs a matrix with fractional zero points", () => {
  assert.throws(() => packQ4_K(newQuantMatrix(4, 256, 4, true), 0, q4kBlocks(4, 4), 4), /fractional zero points/)
})

test("Q8_0 rows dequantize to scale times each signed byte", () => {
  const bytes = new Uint8Array(34 * 2)
  const view = new DataView(bytes.buffer)
  view.setUint16(0, toHalf(0.5), true)
  view.setUint16(34, toHalf(0.25), true)
  for (let j = 0; j < 32; j++) {
    bytes[2 + j] = (j - 16) & 255
    bytes[36 + j] = (4 * j - 64) & 255
  }
  const out = new Float32Array(64)
  dequantizeQ8_0(bytes, 2, 32, out)
  for (let j = 0; j < 32; j++) {
    assert.equal(out[j], 0.5 * (j - 16))
    assert.equal(out[32 + j], 0.25 * (4 * j - 64))
  }
})
