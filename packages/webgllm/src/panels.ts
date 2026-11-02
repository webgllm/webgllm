/**
 * Writing's attention over a long cache (see attendPanel in gpu/kernels.ts):
 * a token's keys in short chunks, each a fragment per 8 dims of a row group
 * of heads, laid out in panels side by side so the pass covers many tiles
 * of its render target (a tile-based GPU shades each tile on one core),
 * then one combine pass over every chunk. This file holds the layout's
 * arithmetic and a CPU model of both passes' math, for the tests.
 */

export type PanelLayout = {
  /** Score texels (4 keys each) per chunk. */
  chunkTexels: number
  /** Chunks per key segment (the last ones of a short segment empty). */
  segChunks: number
  /** Chunks in all: segments · segChunks. */
  chunks: number
  /** Panels across (chunk k in panel column k % across). */
  across: number
  /** The pass's rectangle: across panels of panelW texels, rows of slice queries. */
  width: number
  height: number
}

/**
 * The layout for `kt` score texels of keys, in `segments` segments of at
 * most `segTexels`, for `slice` queries: chunks of `chunkTexels` (more
 * where that would make over `maxChunks`), as many panels across as
 * `maxWidth` texels hold (and at most `maxPanels`, if set).
 */
export function panelLayout(o: {
  kt: number
  segTexels: number
  segments: number
  slice: number
  panelW: number
  maxWidth: number
  chunkTexels: number
  maxChunks: number
  maxPanels?: number
}): PanelLayout {
  const segKt = Math.min(o.kt, o.segTexels)
  const chunkTexels = Math.min(
    segKt,
    Math.max(o.chunkTexels, Math.ceil(o.kt / o.maxChunks))
  )
  const segChunks = Math.ceil(segKt / chunkTexels)
  const chunks = o.segments * segChunks
  const across = Math.max(
    1,
    Math.min(chunks, Math.floor(o.maxWidth / o.panelW), o.maxPanels || Infinity)
  )
  return {
    chunkTexels,
    segChunks,
    chunks,
    across,
    width: across * o.panelW,
    height: Math.ceil(chunks / across) * o.slice
  }
}

/** Chunk k of query qi: its panel's first texel (x) and its row (as the combine reads it). */
export function panelAt(
  layout: PanelLayout,
  panelW: number,
  slice: number,
  k: number,
  qi: number
) {
  const row = Math.floor(k / layout.across)
  return { x: (k - row * layout.across) * panelW, y: row * slice + qi }
}

/** Which chunk and query a fragment of the panel pass works on (as attendPanel finds them). */
export function panelChunk(
  layout: PanelLayout,
  panelW: number,
  slice: number,
  px: number,
  y: number
) {
  const op = Math.floor(px / panelW)
  return {
    x: px - op * panelW,
    qi: y % slice,
    k: Math.floor(y / slice) * layout.across + op
  }
}

/** A chunk's partial result: weighted sum relative to its own max, and the weights' sum. */
export type Partial = { max: number; sum: number; acc: Float64Array }

/**
 * attendPanel's math for one query and head over a chunk: scores (masked
 * ones -1e30) and their values, the softmax kept online `step` scores at a
 * time as the kernel reads them (4 a texel, times its unroll), in f32
 * where the kernel's exp and sums are.
 */
export function chunkPartial(
  scores: ArrayLike<number>,
  values: Array<ArrayLike<number>>,
  step = 4
): Partial {
  const f = Math.fround
  const dims = values[0]?.length ?? 0
  let max = -1e30
  let sum = 0
  const acc = new Float64Array(dims)
  for (let j0 = 0; j0 < scores.length; j0 += step) {
    const j1 = Math.min(scores.length, j0 + step)
    let m = max
    for (let j = j0; j < j1; j++) m = Math.max(m, scores[j])
    const scale = f(Math.exp(max - m))
    max = m
    sum = f(sum * scale)
    for (let d = 0; d < dims; d++) acc[d] = f(acc[d] * scale)
    for (let j = j0; j < j1; j++) {
      const e = f(Math.exp(scores[j] - m))
      sum = f(sum + e)
      for (let d = 0; d < dims; d++) acc[d] = f(acc[d] + e * values[j][d])
    }
  }
  // A chunk with no live key is written empty.
  if (max < -1e29) return { max: -1e30, sum: 0, acc: new Float64Array(dims) }
  return { max, sum, acc }
}

/**
 * attendPanelCombine's math: the chunks, `unroll` at a time, rescaled to
 * a running max and added; the softmax-weighted sum of values.
 */
export function combinePartials(parts: Array<Partial>, unroll = 8) {
  const f = Math.fround
  const dims = parts[0]?.acc.length ?? 0
  let max = -1e30
  let sum = 0
  const acc = new Float64Array(dims)
  for (let c0 = 0; c0 < parts.length;) {
    const n = c0 + unroll <= parts.length ? unroll : 1
    let m = max
    for (let c = c0; c < c0 + n; c++) m = Math.max(m, parts[c].max)
    const scale = f(Math.exp(max - m))
    sum = f(sum * scale)
    for (let d = 0; d < dims; d++) acc[d] = f(acc[d] * scale)
    for (let c = c0; c < c0 + n; c++) {
      const w = f(Math.exp(parts[c].max - m))
      sum = f(sum + parts[c].sum * w)
      for (let d = 0; d < dims; d++) acc[d] = f(acc[d] + w * parts[c].acc[d])
    }
    max = m
    c0 += n
  }
  return acc.map((v) => v / sum)
}
