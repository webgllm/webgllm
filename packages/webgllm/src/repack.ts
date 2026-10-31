/**
 * Repacking of quantized weights from file formats into the QuantMatrix
 * layout (see weights.ts). Each packer writes `rows` rows starting at row
 * `row0` of the destination, so concatenated matrices are filled part by
 * part. `srcRows`, when given, says which source row each destination row
 * comes from (for reordering head dimensions).
 */
import { halfTable, toHalf } from "./half.js"
import type { QuantMatrix } from "./weights.js"

/** Words per row per block: 32 weights of 4 or 8 bits, or a 256-weight Q6_K super-block. */
const WORDS_PER_BLOCK = { 4: 4, 6: 52, 8: 8 } as const

export function newQuantMatrix(
  N: number,
  K: number,
  bits: 4 | 6 | 8,
  withZeros: boolean
): QuantMatrix {
  if (N % 4 !== 0) throw new Error(`matrix rows (${N}) must be a multiple of 4`)
  const blockSize = bits === 6 ? 256 : 32
  if (K % blockSize !== 0)
    throw new Error(`matrix columns (${K}) must be a multiple of ${blockSize}`)
  const blocks = K / blockSize
  return {
    N,
    K,
    bits,
    quants: new Uint32Array(N * blocks * WORDS_PER_BLOCK[bits]),
    scales: new Uint16Array(N * blocks),
    zeros: withZeros ? new Uint8Array(N * blocks).fill(8) : null
  }
}

function quantBytes(m: QuantMatrix): Uint8Array {
  return new Uint8Array(
    m.quants.buffer,
    m.quants.byteOffset,
    m.quants.byteLength
  )
}

function check(
  m: QuantMatrix,
  row0: number,
  rows: number,
  bits: 4 | 6 | 8
): void {
  if (m.bits !== bits) throw new Error(`expected a ${bits}-bit matrix`)
  if (row0 < 0 || row0 + rows > m.N)
    throw new RangeError(`rows ${row0}..${row0 + rows} are outside the matrix`)
}

/**
 * llama.cpp Q4_0 blocks (18 bytes: fp16 scale, then 16 bytes of nibbles in
 * the same order as the QuantMatrix), `rows` rows of K / 32 blocks.
 */
export function packQ4_0(
  m: QuantMatrix,
  row0: number,
  src: Uint8Array,
  rows: number,
  srcRows?: ArrayLike<number>
): void {
  check(m, row0, rows, 4)
  const blocks = m.K / 32
  if (src.length < rows * blocks * 18) throw new Error("Q4_0 data is too short")
  const q = quantBytes(m)
  const scales = m.scales
  for (let i = 0; i < rows; i++) {
    const n = row0 + i
    let p = (srcRows ? srcRows[i] : i) * blocks * 18
    let si = (n >> 2) * blocks * 4 + (n & 3)
    let qi = si * 16
    for (let b = 0; b < blocks; b++, p += 18, si += 4, qi += 64) {
      scales[si] = src[p] | (src[p + 1] << 8)
      for (let j = 0; j < 16; j++) q[qi + j] = src[p + 2 + j]
    }
  }
}

/**
 * llama.cpp Q8_0 blocks (34 bytes: fp16 scale, then 32 int8), into an
 * 8-bit matrix.
 */
export function packQ8_0(
  m: QuantMatrix,
  row0: number,
  src: Uint8Array,
  rows: number,
  srcRows?: ArrayLike<number>
): void {
  check(m, row0, rows, 8)
  const blocks = m.K / 32
  if (src.length < rows * blocks * 34) throw new Error("Q8_0 data is too short")
  const q = quantBytes(m)
  const scales = m.scales
  for (let i = 0; i < rows; i++) {
    const n = row0 + i
    let p = (srcRows ? srcRows[i] : i) * blocks * 34
    let si = (n >> 2) * blocks * 4 + (n & 3)
    let qi = si * 32
    for (let b = 0; b < blocks; b++, p += 34, si += 4, qi += 128) {
      scales[si] = src[p] | (src[p + 1] << 8)
      for (let j = 0; j < 32; j++) q[qi + j] = src[p + 2 + j]
    }
  }
}

/**
 * onnxruntime MatMulNBits (bits 4, block size 32) or 4-bit
 * GatherBlockQuantized data: `quant` [rows][K / 32][16] with element 2i of
 * a block in byte i's low nibble and 2i + 1 in its high nibble; `scales`
 * [rows][K / 32] fp16 bytes; `zp` [rows][ceil(K / 64)] packed two blocks
 * per byte, low nibble first (null: 8).
 */
export function packMatMulNBits(
  m: QuantMatrix,
  row0: number,
  quant: Uint8Array,
  scaleBytes: Uint8Array,
  zp: Uint8Array | null,
  rows: number,
  srcRows?: ArrayLike<number>
): void {
  check(m, row0, rows, 4)
  const blocks = m.K / 32
  const zpStride = Math.ceil(blocks / 2)
  if (
    quant.length < rows * blocks * 16 ||
    scaleBytes.length < rows * blocks * 2
  ) {
    throw new Error("MatMulNBits data is too short")
  }
  if (zp && !m.zeros)
    throw new Error("zero points given for a matrix without them")
  if (zp && zp.length < rows * zpStride)
    throw new Error("MatMulNBits zero points are too short")
  const q = quantBytes(m)
  const scales = m.scales
  const zeros = m.zeros
  for (let i = 0; i < rows; i++) {
    const n = row0 + i
    const s = srcRows ? srcRows[i] : i
    let p = s * blocks * 16
    let sp = s * blocks * 2
    const zr = s * zpStride
    let si = (n >> 2) * blocks * 4 + (n & 3)
    let qi = si * 16
    for (let b = 0; b < blocks; b++, p += 16, sp += 2, si += 4, qi += 64) {
      // Output byte j: weight j (from byte j >> 1) low, weight j + 16 (from
      // byte 8 + (j >> 1)) high.
      for (let k = 0; k < 8; k++) {
        const lo = quant[p + k]
        const hi = quant[p + 8 + k]
        q[qi + 2 * k] = (lo & 0x0f) | ((hi & 0x0f) << 4)
        q[qi + 2 * k + 1] = (lo >> 4) | (hi & 0xf0)
      }
      scales[si] = scaleBytes[sp] | (scaleBytes[sp + 1] << 8)
      if (zp && zeros) zeros[si] = (zp[zr + (b >> 1)] >> ((b & 1) << 2)) & 15
    }
  }
}

/** Running error of the Q6_K → 8-bit conversion. */
export type RequantStats = {
  count: number
  sumSquaredError: number
  sumSquared: number
  maxError: number
}

/**
 * Dequantizes ggml Q6_K rows (super-blocks of 256: ql[128], qh[64], int8
 * scales[16], fp16 d; exactly as ggml's dequantize_row_q6_K, in float32)
 * into `out` (rows × K).
 */
export function dequantizeQ6_K(
  src: Uint8Array,
  rows: number,
  K: number,
  out: Float32Array
): void {
  const half = halfTable()
  const superBlocks = (rows * K) / 256
  for (let s = 0; s < superBlocks; s++) {
    const p = s * 210
    const d = half[src[p + 208] | (src[p + 209] << 8)]
    let y = s * 256
    for (let n = 0; n < 2; n++, y += 128) {
      const ql = p + n * 64
      const qh = p + 128 + n * 32
      const sc = p + 192 + n * 8
      for (let is = 0; is < 2; is++) {
        // d * scale is exact in float32 (11 × 8 bits); each product with q
        // is rounded once, by the Float32Array store, as ggml's float math.
        const d1 = d * ((src[sc + is] << 24) >> 24)
        const d2 = d * ((src[sc + is + 2] << 24) >> 24)
        const d3 = d * ((src[sc + is + 4] << 24) >> 24)
        const d4 = d * ((src[sc + is + 6] << 24) >> 24)
        for (let l = is * 16; l < is * 16 + 16; l++) {
          const h = src[qh + l]
          const a = src[ql + l]
          const b = src[ql + l + 32]
          out[y + l] = d1 * (((a & 15) | ((h & 3) << 4)) - 32)
          out[y + l + 32] = d2 * (((b & 15) | (((h >> 2) & 3) << 4)) - 32)
          out[y + l + 64] = d3 * (((a >> 4) | (((h >> 4) & 3) << 4)) - 32)
          out[y + l + 96] = d4 * (((b >> 4) | (((h >> 6) & 3) << 4)) - 32)
        }
      }
    }
  }
}

/**
 * Quantizes float rows (rows × K) to 8 bits: per block of 32, scale =
 * max|w| / 127 rounded to fp16, q = round(w / scale) in -127..127 (ties
 * round up).
 */
export function quantizeQ8(
  m: QuantMatrix,
  row0: number,
  values: Float32Array,
  rows: number,
  stats?: RequantStats
): void {
  check(m, row0, rows, 8)
  const half = halfTable()
  const blocks = m.K / 32
  const q = new Int8Array(
    m.quants.buffer,
    m.quants.byteOffset,
    m.quants.byteLength
  )
  const scales = m.scales
  let sse = 0
  let ss = 0
  let maxError = stats?.maxError ?? 0
  for (let i = 0; i < rows; i++) {
    const n = row0 + i
    let p = i * m.K
    let si = (n >> 2) * blocks * 4 + (n & 3)
    let qi = si * 32
    for (let b = 0; b < blocks; b++, p += 32, si += 4, qi += 128) {
      let max = 0
      for (let j = 0; j < 32; j++) max = Math.max(max, Math.abs(values[p + j]))
      const h = toHalf(max / 127)
      const scale = half[h]
      scales[si] = h
      if (!(scale > 0)) {
        for (let j = 0; j < 32; j++) q[qi + j] = 0
        if (stats) {
          for (let j = 0; j < 32; j++) {
            const w = values[p + j]
            sse += w * w
            ss += w * w
            maxError = Math.max(maxError, Math.abs(w))
          }
        }
        continue
      }
      // floor(x + 0.5) without a sign branch or Math.round (both several
      // times slower here); the clamp only matters for subnormal scales.
      for (let j = 0; j < 32; j++) {
        const v = ((values[p + j] / scale + 128.5) | 0) - 128
        q[qi + j] = v > 127 ? 127 : v < -127 ? -127 : v
      }
      if (stats) {
        for (let j = 0; j < 32; j++) {
          const w = values[p + j]
          const e = Math.abs(q[qi + j] * scale - w)
          sse += e * e
          ss += w * w
          maxError = Math.max(maxError, e)
        }
      }
    }
  }
  if (stats) {
    stats.count += rows * m.K
    stats.sumSquaredError += sse
    stats.sumSquared += ss
    stats.maxError = maxError
  }
}

/**
 * ggml Q6_K rows (super-blocks of 210 bytes: ql[128], qh[64], int8
 * scales[16], fp16 d) into a 6-bit QuantMatrix, bytes as they are: the
 * shader dequantizes them as dequantize_row_q6_K does.
 */
export function packQ6_K(
  m: QuantMatrix,
  row0: number,
  src: Uint8Array,
  rows: number
): void {
  check(m, row0, rows, 6)
  const superBlocks = m.K / 256
  const bytes = quantBytes(m)
  for (let i = 0; i < rows; i++) {
    const n = row0 + i
    const g = n >> 2
    const r = n & 3
    for (let s = 0; s < superBlocks; s++) {
      const from = (i * superBlocks + s) * 210
      const at = (g * superBlocks + s) * 4 + r
      bytes.set(src.subarray(from, from + 208), at * 208)
      m.scales[at] = src[from + 208] | (src[from + 209] << 8)
    }
  }
}
