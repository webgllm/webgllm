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
  const started = performance.now(),
    blocks = m.K / 32
  if (src.length < rows * blocks * 18) throw new Error("Q4_0 data is too short")
  const input = new DataView(src.buffer, src.byteOffset, src.byteLength)
  const q = m.quants,
    scales = m.scales
  for (let i = 0; i < rows; i++) {
    const n = row0 + i
    let p = (srcRows ? srcRows[i] : i) * blocks * 18
    let si = (n >> 2) * blocks * 4 + (n & 3)
    let qi = si * 4
    for (let b = 0; b < blocks; b++, p += 18, si += 4, qi += 16) {
      scales[si] = input.getUint16(p, true)
      q[qi] = input.getUint32(p + 2, true)
      q[qi + 1] = input.getUint32(p + 6, true)
      q[qi + 2] = input.getUint32(p + 10, true)
      q[qi + 3] = input.getUint32(p + 14, true)
    }
  }
  repackTime.ggufQ4Ms += performance.now() - started
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
  const started = performance.now(),
    blocks = m.K / 32
  if (src.length < rows * blocks * 34) throw new Error("Q8_0 data is too short")
  const input = new DataView(src.buffer, src.byteOffset, src.byteLength)
  const q = m.quants,
    scales = m.scales
  for (let i = 0; i < rows; i++) {
    const n = row0 + i
    let p = (srcRows ? srcRows[i] : i) * blocks * 34
    let si = (n >> 2) * blocks * 4 + (n & 3)
    let qi = si * 8
    for (let b = 0; b < blocks; b++, p += 34, si += 4, qi += 32) {
      scales[si] = input.getUint16(p, true)
      for (let j = 0; j < 8; j++)
        q[qi + j] = input.getUint32(p + 2 + j * 4, true)
    }
  }
  repackTime.ggufQ8Ms += performance.now() - started
}

/**
 * onnxruntime MatMulNBits (bits 4, block size 32) or 4-bit
 * GatherBlockQuantized data: `quant` [rows][K / 32][16] with element 2i of
 * a block in byte i's low nibble and 2i + 1 in its high nibble; `scales`
 * [rows][K / 32] fp16 bytes; `zp` [rows][ceil(K / 64)] packed two blocks
 * per byte, low nibble first (null: 8).
 */
/**
 * MatMulNBits rows (see QuantMatrix.raw) as `quants` in this module's
 * layout: what packMatMulNBits does to the quants, for a matrix whose
 * quants were left raw.
 */
export function quantsFromRaw(raw: Uint8Array, N: number, K: number) {
  const blocks = K / 32
  const quants = new Uint32Array(N * blocks * 4)
  const q = new Uint8Array(quants.buffer)
  for (let n = 0; n < N; n++) {
    let p = n * blocks * 16
    let qi = ((n >> 2) * blocks * 4 + (n & 3)) * 16
    for (let b = 0; b < blocks; b++, p += 16, qi += 64) {
      for (let k = 0; k < 8; k++) {
        const lo = raw[p + k]
        const hi = raw[p + 8 + k]
        q[qi + 2 * k] = (lo & 0x0f) | ((hi & 0x0f) << 4)
        q[qi + 2 * k + 1] = (lo >> 4) | (hi & 0xf0)
      }
    }
  }
  return quants
}

/**
 * Rows [n0, n0 + rows) (multiples of 4) of MatMulNBits rows as RGBA4 texels
 * in the engine's layout (row group by row group: block, then row, then 8
 * texels of 4 weights, the first in the low nibble): each row's 16-byte
 * block is already those 8 texels, so it's moved into place, not repacked.
 */
export function rawRgba4(raw: Uint8Array, K: number, n0: number, rows: number) {
  const blocks = K / 32
  const src = new Uint32Array(raw.buffer, raw.byteOffset, raw.byteLength >> 2)
  const out = new Uint32Array(rows * blocks * 4)
  for (let i = 0; i < rows; i++) {
    let s = (n0 + i) * blocks * 4
    let o = ((i >> 2) * blocks * 4 + (i & 3)) * 4
    for (let b = 0; b < blocks; b++, s += 4, o += 16) {
      out[o] = src[s]
      out[o + 1] = src[s + 1]
      out[o + 2] = src[s + 2]
      out[o + 3] = src[s + 3]
    }
  }
  return new Uint16Array(out.buffer)
}

/**
 * packMatMulNBits' scales and zero points only, for a matrix whose quants
 * stay raw (QuantMatrix.raw).
 */
export function packMatMulNBitsScales(
  m: QuantMatrix,
  row0: number,
  scaleBytes: Uint8Array,
  zp: Uint8Array | null,
  rows: number
): void {
  const blocks = m.K / 32
  const zpStride = Math.ceil(blocks / 2)
  if (scaleBytes.length < rows * blocks * 2)
    throw new Error("MatMulNBits scales are too short")
  if (zp && zp.length < rows * zpStride)
    throw new Error("MatMulNBits zero points are too short")
  const scales = m.scales
  const zeros = m.zeros
  for (let i = 0; i < rows; i++) {
    const n = row0 + i
    let sp = i * blocks * 2
    const zr = i * zpStride
    let si = (n >> 2) * blocks * 4 + (n & 3)
    for (let b = 0; b < blocks; b++, sp += 2, si += 4) {
      scales[si] = scaleBytes[sp] | (scaleBytes[sp + 1] << 8)
      if (zp && zeros) zeros[si] = (zp[zr + (b >> 1)] >> ((b & 1) << 2)) & 15
    }
  }
}

/** Milliseconds spent in packMatMulNBits (load profiling). */
export const repackTime = { ms: 0, ggufQ4Ms: 0, ggufQ8Ms: 0, ggufQ6Ms: 0 }

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
  const started = performance.now()
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
  repackTime.ms += performance.now() - started
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
 * Dequantizes ggml Q4_1 rows (blocks of 32: fp16 d, fp16 m, 16 bytes of
 * nibbles, weight j in byte j's low nibble and j + 16 in its high one;
 * w = d · q + m, as dequantize_row_q4_1) into `out` (rows × K).
 */
export function dequantizeQ4_1(
  src: Uint8Array,
  rows: number,
  K: number,
  out: Float32Array
): void {
  const half = halfTable()
  const blocks = (rows * K) / 32
  for (let b = 0; b < blocks; b++) {
    const p = b * 20
    const d = half[src[p] | (src[p + 1] << 8)]
    const m = half[src[p + 2] | (src[p + 3] << 8)]
    const y = b * 32
    for (let j = 0; j < 16; j++) {
      const q = src[p + 4 + j]
      out[y + j] = (q & 15) * d + m
      out[y + j + 16] = (q >> 4) * d + m
    }
  }
}

/**
 * Dequantizes ggml Q5_K rows (super-blocks of 256: fp16 d, fp16 dmin, 12
 * bytes of 6-bit scales and mins, qh[32], qs[128]; as dequantize_row_q5_K)
 * into `out` (rows × K).
 */
export function dequantizeQ5_K(
  src: Uint8Array,
  rows: number,
  K: number,
  out: Float32Array
): void {
  const half = halfTable()
  const superBlocks = (rows * K) / 256
  // get_scale_min_k4: sub-block j's 6-bit scale and min.
  const scaleMin = (p: number, j: number): [number, number] =>
    j < 4
      ? [src[p + j] & 63, src[p + j + 4] & 63]
      : [
          (src[p + j + 4] & 15) | ((src[p + j - 4] >> 6) << 4),
          (src[p + j + 4] >> 4) | ((src[p + j] >> 6) << 4)
        ]
  for (let s = 0; s < superBlocks; s++) {
    const p = s * 176
    const d = half[src[p] | (src[p + 1] << 8)]
    const dmin = half[src[p + 2] | (src[p + 3] << 8)]
    const sc = p + 4
    const qh = p + 16
    let y = s * 256
    for (let is = 0; is < 8; is += 2) {
      const ql = p + 48 + is * 16
      const u1 = 1 << is
      const u2 = 2 << is
      const [s1, m1] = scaleMin(sc, is)
      const [s2, m2] = scaleMin(sc, is + 1)
      const d1 = d * s1
      const min1 = dmin * m1
      const d2 = d * s2
      const min2 = dmin * m2
      for (let l = 0; l < 32; l++)
        out[y++] =
          d1 * ((src[ql + l] & 15) + (src[qh + l] & u1 ? 16 : 0)) - min1
      for (let l = 0; l < 32; l++)
        out[y++] =
          d2 * ((src[ql + l] >> 4) + (src[qh + l] & u2 ? 16 : 0)) - min2
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
  const started = performance.now()
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
  repackTime.ggufQ6Ms += performance.now() - started
}

/** Q4_0 bytes to RGBA4 upload order: four low-nibble texels then four high.
 * Each word is compacted with whole-word masks/shifts; shaders read .wzyx.
 */
export function q4WordsToRgba4(words: Uint32Array) {
  const out = new Uint16Array(words.length * 2)
  const four = (value: number) => {
    let packed = value & 0x0f0f0f0f
    packed = (packed | (packed >>> 4)) & 0x00ff00ff
    return (packed | (packed >>> 8)) & 0xffff
  }
  for (let row = 0; row < words.length; row += 4) {
    const at = row * 2
    for (let word = 0; word < 4; word++) {
      out[at + word] = four(words[row + word])
      out[at + 4 + word] = four(words[row + word] >>> 4)
    }
  }
  return out
}
