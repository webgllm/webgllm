/**
 * Image preprocessing, as the models' own processors do it.
 *
 * Qwen's vision encoders and LFM3-VL's (Transformers.js
 * `Qwen2VLImageProcessor`, `Lfm3VlImageProcessor`): the image resized to
 * multiples of patch · merge pixels within a pixel budget, scaled to [0, 1]
 * and normalized, then cut into patches ordered so each merge × merge block
 * of patches is consecutive (the merger folds them into one token). Each
 * patch row is channel, frame, y, x: a still image is its own two frames.
 *
 * LFM2-VL's (`Lfm2VlImageProcessor`): an image that fits one view is
 * resized the same way, into 64 to 256 tokens; a larger one is resized to
 * a grid of 512-pixel tiles (the grid nearest its aspect ratio), each tile
 * a view of its own, plus the whole image again as a thumbnail. Its
 * encoder (SigLIP2 NaFlex) takes patch rows as y, x, channel.
 */

/** A processor config's fields this reads (preprocessor_config.json, or processor_config.json's image_processor). */
export type ImageProcessorConfig = {
  patchSize: number
  mergeSize: number
  temporalPatchSize: number
  minPixels: number
  maxPixels: number
  mean: [number, number, number]
  std: [number, number, number]
  /** A patch row's order: channel, frame, y, x, or y, x, channel (one frame). */
  layout: "channel-first" | "channel-last"
  /** LFM2-VL's tiling of images too large for one view; undefined where there's none. */
  tiles?: {
    /** A tile's side, in pixels (512). */
    size: number
    min: number
    max: number
    /** An image is too large for one view past maxPixels times this. */
    tolerance: number
    /** Whether a tiled image is also read whole, as a thumbnail. */
    thumbnail: boolean
  }
}

/**
 * A processor config's fields, checked. Two shapes are read, told apart by
 * their fields: Qwen2-VL's (`patch_size`, `merge_size`,
 * `temporal_patch_size`, pixel limits) and LFM2-VL's (`encoder_patch_size`,
 * `downsample_factor`, token limits, tiles). `json` is
 * preprocessor_config.json, or a processor_config.json holding it as
 * `image_processor`.
 */
export function imageProcessorConfig(json: unknown): ImageProcessorConfig {
  const outer = (json ?? {}) as Record<string, unknown>
  const c = (typeof outer.image_processor === "object" && outer.image_processor !== null ? outer.image_processor : outer) as Record<string, unknown>
  const size = (c.size ?? {}) as Record<string, unknown>
  const num = (v: unknown, name: string) => {
    if (typeof v !== "number" || !(v > 0)) throw new Error(`The image processor's config has no ${name}.`)
    return v
  }
  const triple = (v: unknown, name: string): [number, number, number] => {
    if (!Array.isArray(v) || v.length !== 3 || !v.every((x) => typeof x === "number"))
      throw new Error(`The image processor's config has no ${name}.`)
    return v as [number, number, number]
  }
  const mean = triple(c.image_mean, "image_mean")
  const std = triple(c.image_std, "image_std")
  if (c.encoder_patch_size !== undefined) {
    const patchSize = num(c.encoder_patch_size, "encoder_patch_size")
    const mergeSize = num(c.downsample_factor, "downsample_factor")
    const block = (patchSize * mergeSize) ** 2
    const split = c.do_image_splitting !== false
    const min = split ? num(c.min_tiles, "min_tiles") : 1
    const max = split ? num(c.max_tiles, "max_tiles") : 1
    if (min > max) throw new Error("The image processor's min_tiles is past its max_tiles.")
    return {
      patchSize,
      mergeSize,
      temporalPatchSize: 1,
      minPixels: num(c.min_image_tokens, "min_image_tokens") * block,
      maxPixels: num(c.max_image_tokens, "max_image_tokens") * block,
      mean,
      std,
      layout: "channel-last",
      ...(max > 1 && {
        tiles: {
          size: num(c.tile_size, "tile_size"),
          min,
          max,
          tolerance: num(c.max_pixels_tolerance ?? 1, "max_pixels_tolerance"),
          thumbnail: c.use_thumbnail !== false
        }
      })
    }
  }
  return {
    patchSize: num(c.patch_size, "patch_size"),
    mergeSize: num(c.merge_size, "merge_size"),
    temporalPatchSize: num(c.temporal_patch_size, "temporal_patch_size"),
    minPixels: num(c.min_pixels ?? size.shortest_edge, "min_pixels"),
    maxPixels: num(c.max_pixels ?? size.longest_edge, "max_pixels"),
    mean,
    std,
    layout: "channel-first"
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

/** LFM2-VL's resize of an image that is one view (its `smart_resize`): whole blocks, within the pixel limits. */
export function tileResize(height: number, width: number, config: ImageProcessorConfig): { height: number; width: number } {
  const factor = config.patchSize * config.mergeSize
  const round = (v: number) => Math.round(v / factor) * factor
  let h = Math.max(factor, round(height))
  let w = Math.max(factor, round(width))
  if (h * w > config.maxPixels) {
    const beta = Math.sqrt((height * width) / config.maxPixels)
    h = Math.max(factor, Math.floor(height / beta / factor) * factor)
    w = Math.max(factor, Math.floor(width / beta / factor) * factor)
  } else if (h * w < config.minPixels) {
    const beta = Math.sqrt(config.minPixels / (height * width))
    h = Math.ceil((height * beta) / factor) * factor
    w = Math.ceil((width * beta) / factor) * factor
  }
  return { height: h, width: w }
}

/**
 * The grid of tiles (columns, rows) nearest an image's aspect ratio, among
 * those of min to max tiles (LFM2-VL's `find_closest_aspect_ratio`): of
 * two grids as near, the larger one when the image has more than half its
 * pixels.
 */
export function tileGrid(height: number, width: number, tiles: NonNullable<ImageProcessorConfig["tiles"]>): { cols: number; rows: number } {
  const ratios: Array<[number, number]> = []
  for (let w = 1; w <= tiles.max; w++)
    for (let h = 1; h <= tiles.max; h++) if (w * h >= tiles.min && w * h <= tiles.max) ratios.push([w, h])
  ratios.sort((a, b) => a[0] * a[1] - b[0] * b[1] || a[0] - b[0])
  const aspect = width / height
  const area = width * height
  let best: [number, number] = [1, 1]
  let bestDiff = Infinity
  for (const ratio of ratios) {
    const diff = Math.abs(aspect - ratio[0] / ratio[1])
    if (diff < bestDiff) {
      bestDiff = diff
      best = ratio
    } else if (diff === bestDiff && area > 0.5 * tiles.size * tiles.size * ratio[0] * ratio[1]) best = ratio
  }
  return { cols: best[0], rows: best[1] }
}

/** A view of an image: what the encoder reads at once. */
export type ViewPlan = {
  /** Its pixels. */
  height: number
  width: number
  /** The whole image's size when this view was cut from it, and where: a tile. */
  from?: { height: number; width: number; x: number; y: number }
  /** A tile's row and column, from 1; "thumbnail" for a tiled image read whole. */
  place?: { row: number; col: number } | "thumbnail"
}

/**
 * The views an image is read as. A processor without tiles reads one, sized
 * by smartResize within `detail`'s budget and the GPU's patch limit.
 * LFM2-VL's reads one when the image fits (or `detail` is "low"), else its
 * tiles, row by row, then the thumbnail.
 */
export function planViews(
  height: number,
  width: number,
  detail: ImageDetail | undefined,
  config: ImageProcessorConfig,
  maxPatches: number
): Array<ViewPlan> {
  const factor = config.patchSize * config.mergeSize
  if (config.layout === "channel-first") {
    const budget = pixelBudget(detail, config)
    // The encoder reads at most maxPatches (a texture's height).
    const maxPixels = Math.min(budget.maxPixels, maxPatches * config.patchSize * config.patchSize)
    return [smartResize(height, width, factor, Math.min(budget.minPixels, maxPixels), maxPixels)]
  }
  if (Math.max(height, width) / Math.min(height, width) > 200)
    throw new RangeError("An image's longer side must be under 200 times its shorter side.")
  const whole = tileResize(height, width, config)
  const { tiles } = config
  const round = (v: number) => Math.max(config.patchSize, Math.round(v / factor) * factor)
  const large = tiles && round(height) * round(width) > config.maxPixels * tiles.tolerance
  if (!tiles || !large || detail === "low") return [whole]
  const { cols, rows } = tileGrid(height, width, tiles)
  if (cols * rows === 1) return [{ height: tiles.size, width: tiles.size }]
  const views: Array<ViewPlan> = []
  const from = { height: rows * tiles.size, width: cols * tiles.size }
  for (let row = 0; row < rows; row++)
    for (let col = 0; col < cols; col++)
      views.push({ height: tiles.size, width: tiles.size, from: { ...from, x: col * tiles.size, y: row * tiles.size }, place: { row: row + 1, col: col + 1 } })
  if (tiles.thumbnail) views.push({ ...whole, place: "thumbnail" })
  return views
}

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
 * normalized, in merge order, each patch row in the config's layout
 * (channel, frame, y, x with the frame repeated temporalPatchSize times, or
 * y, x, channel).
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
  const last = config.layout === "channel-last"
  if (last && t !== 1) throw new RangeError("Patches ordered y, x, channel have one frame.")
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
          if (last) pixels[base + (y * p + x) * 3 + c] = v
          else for (let f = 0; f < t; f++) pixels[base + ((c * t + f) * p + y) * p + x] = v
        }
      }
    }
  }
  return { pixels, gridH, gridW }
}
