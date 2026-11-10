/**
 * Image preprocessing for Qwen's vision encoders, as Qwen2-VL's image
 * processor does it (Transformers.js `Qwen2VLImageProcessor`): the image
 * resized to multiples of patch · merge pixels within a pixel budget,
 * scaled to [0, 1] and normalized, then cut into patches ordered so each
 * merge × merge block of patches is consecutive (the merger folds them into
 * one token). Each patch row is channel, frame, y, x: a still image is its
 * own two frames.
 */

/** preprocessor_config.json's fields this reads. */
export type ImageProcessorConfig = {
  patchSize: number
  mergeSize: number
  temporalPatchSize: number
  minPixels: number
  maxPixels: number
  mean: [number, number, number]
  std: [number, number, number]
}

/** The fields of a Qwen2-VL-style preprocessor_config.json, checked. */
export function imageProcessorConfig(json: unknown): ImageProcessorConfig {
  const c = (json ?? {}) as Record<string, unknown>
  const size = (c.size ?? {}) as Record<string, unknown>
  const num = (v: unknown, name: string) => {
    if (typeof v !== "number" || !(v > 0)) throw new Error(`preprocessor_config.json has no ${name}.`)
    return v
  }
  const triple = (v: unknown, name: string): [number, number, number] => {
    if (!Array.isArray(v) || v.length !== 3 || !v.every((x) => typeof x === "number"))
      throw new Error(`preprocessor_config.json has no ${name}.`)
    return v as [number, number, number]
  }
  return {
    patchSize: num(c.patch_size, "patch_size"),
    mergeSize: num(c.merge_size, "merge_size"),
    temporalPatchSize: num(c.temporal_patch_size, "temporal_patch_size"),
    minPixels: num(c.min_pixels ?? size.shortest_edge, "min_pixels"),
    maxPixels: num(c.max_pixels ?? size.longest_edge, "max_pixels"),
    mean: triple(c.image_mean, "image_mean"),
    std: triple(c.image_std, "image_std")
  }
}

/**
 * The size an image is resized to: each side a multiple of `factor`,
 * nearest to the original, then scaled (keeping the aspect ratio) into
 * [minPixels, maxPixels] (Qwen2-VL's smart_resize), never past maxPixels
 * (see the end).
 */
export function smartResize(
  height: number,
  width: number,
  factor: number,
  minPixels: number,
  maxPixels: number
): { height: number; width: number } {
  if (height < factor || width < factor) {
    const scale = Math.max(factor / height, factor / width)
    height = Math.round(height * scale)
    width = Math.round(width * scale)
  }
  if (Math.max(height, width) / Math.min(height, width) > 200)
    throw new RangeError("An image's longer side must be under 200 times its shorter side.")
  let h = Math.round(height / factor) * factor
  let w = Math.round(width / factor) * factor
  if (h * w > maxPixels) {
    const beta = Math.sqrt((height * width) / maxPixels)
    h = Math.max(factor, Math.floor(height / beta / factor) * factor)
    w = Math.max(factor, Math.floor(width / beta / factor) * factor)
  } else if (h * w < minPixels) {
    const beta = Math.sqrt(minPixels / (height * width))
    h = Math.ceil((height * beta) / factor) * factor
    w = Math.ceil((width * beta) / factor) * factor
  }
  // Rounding to whole blocks can overshoot the maximum (both sides rounded
  // up past the minimum; a thin strip's short side held at one block): the
  // maximum wins, the longer side giving a block at a time, then the aspect
  // ratio and the minimum as they may. One block is the least there is.
  while (h * w > maxPixels && (h > factor || w > factor)) {
    if (h >= w && h > factor) h -= factor
    else w -= factor
  }
  return { height: h, width: w }
}

/**
 * OpenAI's `detail` as a pixel budget: "low" 256², "auto" 640², "high"
 * 1024² (64, 400 and 1,024 tokens for a square image), within the
 * processor's own range.
 */
export function pixelBudget(detail: ImageDetail | undefined, config: ImageProcessorConfig) {
  const side = detail === "low" ? 256 : detail === "high" ? 1024 : 640
  const max = Math.min(config.maxPixels, side * side)
  return { minPixels: Math.min(config.minPixels, max), maxPixels: max }
}

export type ImageDetail = "low" | "high" | "auto"

/** A patch's row and column in the patch grid, from its place in merge order. */
export function patchPlace(p: number, gridW: number, merge: number): { row: number; col: number } {
  const per = merge * merge
  const block = Math.floor(p / per)
  const inner = p % per
  const blocksW = gridW / merge
  return {
    row: Math.floor(block / blocksW) * merge + Math.floor(inner / merge),
    col: (block % blocksW) * merge + (inner % merge)
  }
}

/** An image ready for the encoder: patches in merge order, and the patch grid. */
export type ImagePatches = {
  /** patches × (3 · temporal · patch²) values. */
  pixels: Float32Array
  gridH: number
  gridW: number
}

/**
 * RGBA pixels (already resized to multiples of patch · merge) as patches:
 * normalized, in merge order, each patch row channel-major then frame,
 * y, x, the frame repeated temporalPatchSize times.
 */
export function toPatches(
  rgba: ArrayLike<number>,
  height: number,
  width: number,
  config: ImageProcessorConfig
): ImagePatches {
  const p = config.patchSize
  const m = config.mergeSize
  const t = config.temporalPatchSize
  if (height % (p * m) || width % (p * m))
    throw new RangeError(`A ${width}×${height} image isn't in whole ${p * m}-pixel blocks.`)
  const gridH = height / p
  const gridW = width / p
  const count = gridH * gridW
  const per = 3 * t * p * p
  const pixels = new Float32Array(count * per)
  const scale = config.std.map((s) => 1 / (255 * s))
  const shift = config.mean.map((mu, c) => mu / config.std[c])
  for (let i = 0; i < count; i++) {
    const { row, col } = patchPlace(i, gridW, m)
    const base = i * per
    for (let c = 0; c < 3; c++) {
      for (let y = 0; y < p; y++) {
        const src = ((row * p + y) * width + col * p) * 4 + c
        for (let x = 0; x < p; x++) {
          const v = rgba[src + 4 * x] * scale[c] - shift[c]
          for (let f = 0; f < t; f++) pixels[base + ((c * t + f) * p + y) * p + x] = v
        }
      }
    }
  }
  return { pixels, gridH, gridW }
}
