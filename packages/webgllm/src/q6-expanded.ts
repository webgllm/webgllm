import type { QuantMatrix } from "./weights.js"

/** Expand six-bit integers into bytes exactly; keep signed scales and fp16 d. */
export function expandQ6(
  m: QuantMatrix,
  group0: number,
  groups: number,
  storage?: Uint8Array
) {
  const blocks = m.K / 256
  const src = new Uint8Array(
    m.quants.buffer,
    m.quants.byteOffset,
    m.quants.byteLength
  )
  const bytes = groups * blocks * 4 * 256
  const quants =
    storage && storage.length >= bytes
      ? storage.subarray(0, bytes)
      : new Uint8Array(bytes)
  for (let g = 0; g < groups; g++)
    for (let sb = 0; sb < blocks; sb++)
      for (let r = 0; r < 4; r++) {
        const from = (((group0 + g) * blocks + sb) * 4 + r) * 208
        const to = ((g * blocks + sb) * 4 + r) * 256
        for (let n = 0; n < 2; n++)
          for (let l = 0; l < 32; l++) {
            const a = src[from + n * 64 + l],
              b = src[from + n * 64 + l + 32],
              h = src[from + 128 + n * 32 + l]
            const at = to + n * 128 + l
            quants[at] = (a & 15) | ((h & 3) << 4)
            quants[at + 32] = (b & 15) | (((h >> 2) & 3) << 4)
            quants[at + 64] = (a >> 4) | (((h >> 4) & 3) << 4)
            quants[at + 96] = (b >> 4) | (((h >> 6) & 3) << 4)
          }
        // Four otherwise unused two-bit fields carry each signed subscale.
        for (let group = 0; group < 16; group++)
          for (let part = 0; part < 4; part++)
            quants[to + group * 16 + part] |=
              ((src[from + 192 + group] >> (2 * part)) & 3) << 6
      }
  return { quants }
}

/** Expansion is permitted only when other scratch savings cover every added byte. */
export function fitsExpandedHead(
  extraHeadBytes: number,
  previousScratchBytes: number,
  nextScratchBytes: number,
  activationBytes: number
) {
  return (
    Number.isFinite(previousScratchBytes) &&
    extraHeadBytes <= previousScratchBytes - nextScratchBytes - activationBytes
  )
}

/** Split=1 writes directly to the logits buffer and needs no extra head target. */
export function headScratchBytes(
  split: number,
  rows: number,
  logitRows: number,
  width: number
) {
  return split > 1 ? split * rows * logitRows * width * 16 : 0
}

/** Fund streaming scratch from the former automatic tuner's retained-layout ceiling. */
export function fundedF16Scratch(
  previousScratch: number,
  previousActivations: number,
  nextActivations: number,
  headGrowth: number,
  minimum: number
) {
  if (!Number.isFinite(previousScratch)) return Math.max(minimum, 4 * 2 ** 20)
  const available = Math.min(
    previousScratch,
    previousScratch + previousActivations - nextActivations - headGrowth
  )
  const quantum = 4 * 2 ** 20
  const rounded =
    headGrowth > 0 && available >= quantum
      ? Math.floor(available / quantum) * quantum
      : available
  return Math.max(minimum, rounded)
}
