import test from "node:test"
import assert from "node:assert/strict"
import { q4WordsToRgba4 } from "./repack.js"

test("whole-word RGBA4 packing matches independent nibble extraction for all byte values", () => {
  let seed = 1
  const words = Uint32Array.from({ length: 4096 }, (_, i) => {
    seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0
    return i < 256 ? (i | (i << 8) | (i << 16) | (i << 24)) >>> 0 : seed
  })
  const got = q4WordsToRgba4(words)
  for (let row = 0; row < words.length; row += 4)
    for (let w = 0; w < 4; w++)
      for (let high = 0; high < 2; high++) {
        let expected = 0
        for (let byte = 0; byte < 4; byte++)
          expected |=
            ((words[row + w] >>> (8 * byte + 4 * high)) & 15) << (4 * byte)
        assert.equal(got[row * 2 + w + 4 * high], expected)
      }
})
