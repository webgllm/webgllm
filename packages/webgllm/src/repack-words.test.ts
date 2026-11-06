import assert from "node:assert/strict"
import test from "node:test"
import { newQuantMatrix, packQ4_0, packQ8_0 } from "./repack.js"

for (const bits of [4, 8] as const) {
  test(`Q${bits}_0 word copies preserve bytes for unaligned views, row offsets and permutations`, () => {
    const bytes = bits === 4 ? 16 : 32,
      stride = bytes + 2
    for (const offset of [0, 1, 2, 3]) {
      const raw = new Uint8Array(offset + 4 * 2 * stride)
      for (let i = 0; i < raw.length; i++) raw[i] = (i * 71 + 19) & 255
      const src = raw.subarray(offset),
        perm = [3, 0, 2, 1]
      const actual = newQuantMatrix(12, 64, bits, false),
        want = newQuantMatrix(12, 64, bits, false)
      const target = new Uint8Array(want.quants.buffer)
      for (let i = 0; i < 4; i++)
        for (let b = 0; b < 2; b++) {
          const at = (((4 + i) >> 2) * 2 + b) * 4 + ((4 + i) & 3)
          const from = (perm[i] * 2 + b) * stride
          want.scales[at] = src[from] | (src[from + 1] << 8)
          target.set(src.subarray(from + 2, from + stride), at * bytes)
        }
      ;(bits === 4 ? packQ4_0 : packQ8_0)(actual, 4, src, 4, perm)
      assert.deepEqual(actual.quants, want.quants)
      assert.deepEqual(actual.scales, want.scales)
    }
  })
}
