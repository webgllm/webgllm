/**
 * Seeded sampling: one uniform number per position in the reply, from the
 * seed and the position alone. Speculative decoding and pipelining decide
 * by timing how many tokens each step checks, so a stream of numbers drawn
 * in order would shift with them; a number per position doesn't (with
 * sampler.ts's drawAt and verifyAt, a drafted token is kept exactly when it
 * is the token drawn there).
 */

/** A 32-bit integer hash with full avalanche (lowbias32). */
function mix(h: number) {
  h ^= h >>> 16
  h = Math.imul(h, 0x7feb352d)
  h ^= h >>> 15
  h = Math.imul(h, 0x846ca68b)
  h ^= h >>> 16
  return h >>> 0
}

/** The uniform number in [0, 1) for each reply position, for `seed` (any integer). */
export function seededUniforms(seed: number): (position: number) => number {
  if (!Number.isFinite(seed)) throw new RangeError("seed must be a finite number.")
  // Both 32-bit halves count, so 64-bit seeds (OpenAI's) stay distinct.
  const whole = BigInt(Math.trunc(seed))
  const low = Number(BigInt.asUintN(32, whole))
  const high = Number(BigInt.asUintN(32, whole >> 32n))
  const key = mix(mix(low ^ 0x9e3779b9) ^ high)
  return (position) => mix(key ^ mix(position + 0x632be5ab)) / 2 ** 32
}
