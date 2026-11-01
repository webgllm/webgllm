import assert from "node:assert/strict"
import { describe, test } from "node:test"
import {
  newQuantMatrix,
  packMatMulNBits,
  packMatMulNBitsScales,
  quantsFromRaw,
  rawRgba4
} from "./repack.js"

/** Random MatMulNBits data: N rows of K 4-bit weights, fp16 scale bytes, packed zero points. */
function matMulNBits(N: number, K: number, seed = 3) {
  let s = seed
  const next = () => (s = (Math.imul(s, 1103515245) + 12345) >>> 0) >>> 16
  const blocks = K / 32
  const quant = Uint8Array.from({ length: N * blocks * 16 }, () => next() & 255)
  const scales = Uint8Array.from({ length: N * blocks * 2 }, () => next() & 255)
  const zp = Uint8Array.from({ length: N * Math.ceil(blocks / 2) }, () => next() & 255)
  return { quant, scales, zp, blocks }
}

/** Weight k of row n in MatMulNBits data: element 2i in byte i's low nibble, 2i + 1 in its high. */
const weight = (quant: Uint8Array, blocks: number, n: number, k: number) => {
  const byte = quant[n * blocks * 16 + (k >> 5) * 16 + ((k & 31) >> 1)]
  return k & 1 ? byte >> 4 : byte & 15
}

describe("rawRgba4", () => {
  test("texel t of row r, block b, holds weights 4t..4t+3, the first in the low nibble", () => {
    const N = 8
    const K = 96
    const { quant, blocks } = matMulNBits(N, K)
    const n0 = 4
    const rows = 4
    const texels = rawRgba4(quant, K, n0, rows)
    assert.equal(texels.length, rows * blocks * 8)
    for (let i = 0; i < rows; i++) {
      for (let b = 0; b < blocks; b++) {
        for (let t = 0; t < 8; t++) {
          // Row group by row group: block, then row, then 8 texels.
          const at = ((i >> 2) * blocks * 4 + b * 4 + (i & 3)) * 8 + t
          for (let j = 0; j < 4; j++) {
            const nibble = (texels[at] >> (4 * j)) & 15
            assert.equal(nibble, weight(quant, blocks, n0 + i, b * 32 + 4 * t + j))
          }
        }
      }
    }
  })
})

describe("MatMulNBits repacking", () => {
  test("quantsFromRaw matches packMatMulNBits' quants", () => {
    const N = 12
    const K = 128
    const { quant, scales, zp } = matMulNBits(N, K)
    const m = newQuantMatrix(N, K, 4, true)
    packMatMulNBits(m, 0, quant, scales, zp, N)
    assert.deepEqual(quantsFromRaw(quant, N, K), m.quants)
  })

  test("packMatMulNBitsScales matches packMatMulNBits' scales and zero points", () => {
    const N = 8
    const K = 96 // an odd number of blocks: the last zero point byte is half used
    const { quant, scales, zp } = matMulNBits(N, K)
    const full = newQuantMatrix(N, K, 4, true)
    packMatMulNBits(full, 0, quant, scales, zp, N)
    const split = newQuantMatrix(N, K, 4, true)
    packMatMulNBitsScales(split, 0, scales, zp, N)
    assert.deepEqual(split.scales, full.scales)
    assert.deepEqual(split.zeros, full.zeros)
  })

  test("rows land at row0 (matrices filled part by part)", () => {
    const N = 8
    const K = 64
    const a = matMulNBits(4, K, 1)
    const b = matMulNBits(4, K, 2)
    const m = newQuantMatrix(N, K, 4, true)
    packMatMulNBitsScales(m, 0, a.scales, a.zp, 4)
    packMatMulNBitsScales(m, 4, b.scales, b.zp, 4)
    const whole = newQuantMatrix(N, K, 4, true)
    const scales = new Uint8Array([...a.scales, ...b.scales])
    const zp = new Uint8Array([...a.zp, ...b.zp])
    packMatMulNBitsScales(whole, 0, scales, zp, N)
    assert.deepEqual(m.scales, whole.scales)
    assert.deepEqual(m.zeros, whole.zeros)
  })

  test("zero points unpack low nibble first, two blocks a byte", () => {
    const m = newQuantMatrix(4, 64, 4, true)
    const scales = new Uint8Array(4 * 2 * 2)
    const zp = Uint8Array.of(0x21, 0x43, 0x65, 0x87) // row r: blocks (2r+1, 2r+2)
    packMatMulNBitsScales(m, 0, scales, zp, 4)
    // Layout: (row group, block, row in group).
    const zeros = m.zeros as Uint8Array
    for (let r = 0; r < 4; r++) {
      assert.equal(zeros[0 * 4 + r], 2 * r + 1, `row ${r} block 0`)
      assert.equal(zeros[1 * 4 + r], 2 * r + 2, `row ${r} block 1`)
    }
  })
})
