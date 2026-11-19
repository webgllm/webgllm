import assert from "node:assert/strict"
import test from "node:test"
import { halfTable } from "./half.js"
import { dequantizeQ6_K, newQuantMatrix, packQ6_K, q6SplitRows } from "./repack.js"

test("a Q6_K embedding's split layout decodes, as its shaders index it, to dequantizeQ6_K's weights", () => {
  const N = 8, K = 512, sbs = K / 256, blocks = K / 32
  const src = Uint8Array.from({ length: (N * K / 256) * 210 }, (_, i) => Math.imul(i + 11, 2654435761) >>> 24)
  for (let p = 208; p < src.length; p += 210) { src[p] = 0x66; src[p + 1] = 0x1e } // d, a sane half
  const want = new Float32Array(N * K)
  dequantizeQ6_K(src, N, K, want)
  const m = newQuantMatrix(N, K, 6, false)
  packQ6_K(m, 0, src, N)
  const groups = N / 4
  const low = new Uint16Array(groups * sbs * 256), high = new Uint16Array(groups * sbs * 128), sub = new Uint32Array(groups * sbs * 16)
  q6SplitRows(m, 0, groups, low, high, sub)
  const half = halfTable()
  for (let n = 0; n < N; n++)
    for (let j = 0; j < K; j++) {
      const g = n >> 2, r = n & 3, k = j >> 5, t = (j & 31) >> 2, s = j >> 8
      const lo = (low[((g * blocks + k) * 4 + r) * 8 + t] >> (4 * (j & 3))) & 15
      const v = (high[((g * blocks + k) * 4 + r) * 4 + (t & 3)] >> (4 * (j & 3))) & 15
      const h = t < 4 ? v & 3 : v >> 2
      const i = (j >> 4) & 15
      const sc = (sub[((g * sbs + s) * 4 + r) * 4 + (i >> 2)] << (24 - 8 * (i & 3))) >> 24
      const d = half[m.scales[(g * sbs + s) * 4 + r]]
      const w = d * sc * (lo + 16 * h - 32)
      assert.ok(Math.abs(w - want[n * K + j]) <= 1e-6 * Math.max(1, Math.abs(w)), `row ${n} weight ${j}: ${w} vs ${want[n * K + j]}`)
    }
})

test("the WebAssembly split writes q6SplitRows' texels", async () => {
  const { q6Splitter } = await import("./repack.js")
  const split = q6Splitter()
  assert.ok(split, "WebAssembly SIMD is available in Node")
  const N = 16, K = 512, sbs = K / 256, groups = N / 4
  const src = Uint8Array.from({ length: (N * K / 256) * 210 }, (_, i) => Math.imul(i + 3, 2654435761) >>> 24)
  const m = newQuantMatrix(N, K, 6, false)
  packQ6_K(m, 0, src, N)
  const low = new Uint16Array(2 * sbs * 256), high = new Uint16Array(2 * sbs * 128), sub = new Uint32Array(2 * sbs * 16)
  q6SplitRows(m, 1, 2, low, high, sub)
  const got = split(m, 1, 2)
  assert.deepEqual(Array.from(got.low), Array.from(low))
  assert.deepEqual(Array.from(got.high), Array.from(high))
  assert.deepEqual(Array.from(got.sub), Array.from(sub))
  void groups
})
