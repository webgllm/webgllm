/** IEEE half-precision conversions for the weight loaders. */

const f32 = new Float32Array(1)
const u32 = new Uint32Array(f32.buffer)

/** A float as IEEE half bits, rounded to nearest even. */
export function toHalf(value: number): number {
  f32[0] = value
  const x = u32[0]
  const sign = (x >>> 16) & 0x8000
  const exp = (x >>> 23) & 0xff
  let mant = x & 0x7fffff
  if (exp === 0xff) return sign | 0x7c00 | (mant ? 0x200 : 0)
  const e = exp - 127 + 15
  if (e >= 31) return sign | 0x7c00
  if (e <= 0) {
    // Subnormal half (or zero).
    if (e < -10) return sign
    mant |= 0x800000
    const shift = 14 - e
    const half = 1 << (shift - 1)
    const rest = mant & ((1 << shift) - 1)
    let h = mant >>> shift
    if (rest > half || (rest === half && h & 1)) h++
    return sign | h
  }
  let h = (e << 10) | (mant >>> 13)
  const rest = mant & 0x1fff
  // A carry out of the mantissa rounds up to the next exponent (or Infinity).
  if (rest > 0x1000 || (rest === 0x1000 && h & 1)) h++
  return sign | h
}

let table: Float32Array | undefined

/** Every half's value, indexed by its bits (256 KB, built on first use). */
export function halfTable(): Float32Array {
  if (table) return table
  const t = new Float32Array(65536)
  for (let h = 0; h < 65536; h++) {
    const sign = h & 0x8000 ? -1 : 1
    const exp = (h >>> 10) & 0x1f
    const mant = h & 0x3ff
    t[h] =
      exp === 0
        ? sign * mant * 2 ** -24
        : exp === 31
          ? mant
            ? NaN
            : sign * Infinity
          : sign * (1 + mant / 1024) * 2 ** (exp - 15)
  }
  table = t
  return t
}

/** IEEE half bits to a float. */
export function fromHalf(h: number): number {
  return halfTable()[h & 0xffff]
}

/** Little-endian halves (2 bytes each) to floats. */
export function halvesToFloats(bytes: Uint8Array): Float32Array {
  const t = halfTable()
  const n = bytes.length >> 1
  const out = new Float32Array(n)
  for (let i = 0, p = 0; i < n; i++, p += 2) {
    out[i] = t[bytes[p] | (bytes[p + 1] << 8)]
  }
  return out
}

/** Little-endian float32s to a Float32Array that owns its memory. */
export function bytesToFloats(bytes: Uint8Array): Float32Array {
  const n = bytes.length >> 2
  if (bytes.byteOffset % 4 === 0) {
    return new Float32Array(bytes.buffer, bytes.byteOffset, n).slice()
  }
  return new Float32Array(bytes.slice(0, n * 4).buffer)
}
