import assert from "node:assert/strict"
import test from "node:test"
import { dequantizeQ4_1, dequantizeQ5_K, newQuantMatrix, q8Converter, quantizeQ8 } from "./repack.js"

const bytes = (n: number, seed: number) => Uint8Array.from({ length: n }, (_, i) => Math.imul(i + seed, 2654435761) >>> 24)

/** Random blocks with sane, zero, tiny and negative halves for d and min. */
function blocks(n: number, size: number, seed: number) {
  const src = bytes(n * size, seed)
  for (let b = 0; b < n; b++) {
    const p = b * size
    const d = b % 13 === 0 ? 0 : b % 17 === 0 ? 0x0003 : 0x1000 + ((b * 37) % 0x2000)
    const m = (b % 2 ? 0x8000 : 0) | (0x0800 + ((b * 53) % 0x2400))
    src[p] = d & 255; src[p + 1] = d >> 8; src[p + 2] = m & 255; src[p + 3] = m >> 8
  }
  return src
}

test("the WebAssembly Q5_K and Q4_1 conversion writes quantizeQ8's bytes", () => {
  const convert = q8Converter()
  assert.ok(convert, "WebAssembly SIMD is available in Node")
  for (const [kind, rows, K] of [["q5k", 8, 512], ["q5k", 12, 2048], ["q41", 8, 256], ["q41", 16, 6144]] as const) {
    const src = kind === "q5k" ? blocks((rows * K) / 256, 176, rows + K) : blocks((rows * K) / 32, 20, rows + K)
    const floats = new Float32Array(rows * K)
    ;(kind === "q5k" ? dequantizeQ5_K : dequantizeQ4_1)(src, rows, K, floats)
    const want = newQuantMatrix(rows + 4, K, 8, false)
    quantizeQ8(want, 4, floats, rows)
    const got = newQuantMatrix(rows + 4, K, 8, false)
    assert.equal(convert(kind, got, 4, src, rows), true)
    assert.deepEqual(Array.from(got.scales), Array.from(want.scales), `${kind} ${rows}×${K} scales`)
    assert.deepEqual(Array.from(got.quants), Array.from(want.quants), `${kind} ${rows}×${K} codes`)
  }
  // Rows not in whole groups of 4: left to the caller.
  assert.equal(convert("q41", newQuantMatrix(8, 32, 8, false), 2, blocks(2, 20, 1), 2), false)
})
