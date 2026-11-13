/**
 * A prompt's images in the worker: fetched and resized into views as the
 * model's processor says (image.ts), each image slot the chat template
 * wrote expanded to the image's tokens, and keys for the prompt cache that tell
 * images apart (see handler.ts). Embedding rows are kept for a few recent
 * images, by content, so a chat that reads an image again doesn't encode it
 * again.
 */
import { planViews, toPatches, type ImageDetail, type ImagePatches, type ImageProcessorConfig } from "./image.js"
import type { ChatMessage } from "./tokenizer.js"

/** An image part of a chat message, as the worker gets it. */
export type ChatImage = { url: string; detail?: ImageDetail }

/** A view of an image (the whole image, a tile of it, or its thumbnail): what the encoder reads at once. */
export type ImageView = {
  /** Merged tokens: rows and columns. */
  height: number
  width: number
  /** A tile's row and column (from 1), or "thumbnail", where the image is tiled. */
  place?: { row: number; col: number } | "thumbnail"
  /** Its resized pixels (RGBA), until it's encoded. */
  pixels: { rgba: Uint8ClampedArray; height: number; width: number } | null
  /** Its embedding rows, once encoded. */
  features: Float32Array | null
}

/** An image ready for a prompt. */
export type PreparedImage = {
  /** The prompt cache's key for its rows: negative (never a token id), from its bytes, detail and size. */
  key: number
  /** What it's read as, in prompt order: one view, or its tiles then its thumbnail. */
  views: Array<ImageView>
}

/** What of an image its place in a prompt needs: each view's size, and where a tile is. */
export type ImageLayout = { views: Array<Pick<ImageView, "height" | "width" | "place">> }

/** An image's tokens in the prompt that are its rows (the views' tokens; the tokens around them aside). */
export const imageRows = (image: ImageLayout) => image.views.reduce((sum, v) => sum + v.height * v.width, 0)

/** A view's patches, cut from its resized pixels when it's encoded. */
export function patchesOf(view: ImageView, config: ImageProcessorConfig): ImagePatches {
  if (!view.pixels) throw new Error("An image's pixels were dropped before it was encoded.")
  const { rgba, height, width } = view.pixels
  return toPatches(rgba, height, width, config)
}

/** The images of a chat, in the order its template writes them (message by message, part by part). */
export function chatImages(chat: Array<ChatMessage>): Array<ChatImage> {
  const images: Array<ChatImage> = []
  for (const message of chat) {
    if (typeof message.content === "string") continue
    for (const part of message.content)
      if (part.type === "image") images.push({ url: part.image, ...(part.detail && { detail: part.detail }) })
  }
  return images
}

/** FNV-1a and a second multiplicative hash of the bytes and size, as one 52-bit negative key. */
export function imageKey(bytes: Uint8Array, height: number, width: number): number {
  let a = 0x811c9dc5
  let b = 0x9747b28c ^ height ^ (width << 16)
  for (let i = 0; i < bytes.length; i++) {
    a = Math.imul(a ^ bytes[i], 0x01000193)
    b = Math.imul(b ^ bytes[i], 0x5bd1e995)
    b ^= b >>> 15
  }
  return -(1 + (a >>> 0) * 2 ** 20 + ((b >>> 0) & 0xfffff))
}

/**
 * How a chat template writes an image, as token ids. Qwen3.5 and LFM3-VL
 * write a placeholder between a vision start and end, which becomes the
 * image's tokens (`wrapped`). LFM2-VL writes a bare placeholder, which
 * becomes `<|image_start|>`, the image's tokens and `<|image_end|>`; a
 * tiled image's are each tile's after its `<|img_row_R_col_C|>`, then the
 * thumbnail's after `<|img_thumbnail|>`.
 */
export type ImageSlot =
  | { kind: "wrapped"; start: number; pad: number; end: number }
  | { kind: "bare"; pad: number; start: number; end: number; thumbnail: number; tile: (row: number, col: number) => number }

/** What stands for a placeholder a message's own text has, in imagePlaceholders' render. */
const LITERAL = "@@webgllm-literal-placeholder@@"

/** `value` with `change` applied to every string in it (arrays and plain objects, keys as they are). */
function mapStrings<T>(value: T, change: (text: string) => string): T {
  if (typeof value === "string") return change(value) as T
  if (Array.isArray(value)) return value.map((item) => mapStrings(item, change)) as T
  if (value && typeof value === "object") {
    return Object.fromEntries(Object.entries(value).map(([key, item]) => [key, mapStrings(item, change)])) as T
  }
  return value
}

/**
 * For each `placeholder` in a chat's rendered text, in order: whether the
 * template wrote it for an image part (true), or it is a message's own
 * text (false). A user can write "<image>" in a message, and the tokenizer
 * makes the template's placeholder token of it: that one is text, not an
 * attachment. Told apart by rendering `input` (what `render` reads: the
 * messages, the tools) with the placeholder in every string of it replaced
 * by a marker: the placeholders left are the template's own.
 */
export function imagePlaceholders<T>(input: T, placeholder: string, render: (input: T) => string): Array<boolean> {
  const probe = render(mapStrings(input, (text) => text.split(placeholder).join(LITERAL)))
  const written: Array<boolean> = []
  let at = 0
  for (;;) {
    const part = probe.indexOf(placeholder, at)
    const literal = probe.indexOf(LITERAL, at)
    if (part < 0 && literal < 0) return written
    const isPart = literal < 0 || (part >= 0 && part < literal)
    written.push(isPart)
    at = isPart ? part + placeholder.length : literal + LITERAL.length
  }
}

/**
 * Where the prompt's image slots are. `written` says, for each placeholder
 * token in order, whether the template wrote it for an image part (see
 * imagePlaceholders); without it every placeholder token is taken for one
 * (a wrapped one only between its start and end tokens).
 */
function slotPositions(prompt: ArrayLike<number>, slot: ImageSlot, written?: ReadonlyArray<boolean>): Set<number> {
  const at = new Set<number>()
  let placeholder = 0
  for (let i = 0; i < prompt.length; i++) {
    if (prompt[i] !== slot.pad) continue
    const part = written ? written[placeholder] === true : true
    placeholder++
    if (part && (slot.kind === "bare" || (prompt[i - 1] === slot.start && prompt[i + 1] === slot.end))) at.add(i)
  }
  return at
}

/** How many image slots the template wrote (a wrapped placeholder alone in a message's text isn't one; see slotPositions for `written`). */
export function countImageSlots(prompt: ArrayLike<number>, slot: ImageSlot, written?: ReadonlyArray<boolean>): number {
  return slotPositions(prompt, slot, written).size
}

/** The tokens an image's slot becomes around and between its views' rows, and in all. */
export function imageTokenCount(image: ImageLayout, slot: ImageSlot): number {
  const rows = imageRows(image)
  if (slot.kind === "wrapped") return rows
  return rows + 2 + image.views.filter((v) => v.place).length
}

/**
 * The prompt with each image slot expanded to its image's tokens, and the
 * keys the cache compares: the tokens, with each image's rows its key.
 * `starts` is where each view's rows begin, images and their views in
 * order. A placeholder that isn't a slot (`written`: see slotPositions)
 * stays the one token it is, as text.
 */
export function expandImageTokens(
  prompt: Array<number>,
  slot: ImageSlot,
  images: Array<Pick<PreparedImage, "key"> & ImageLayout>,
  written?: ReadonlyArray<boolean>
): { tokens: Array<number>; keys: Array<number>; starts: Array<number> } {
  const slots = slotPositions(prompt, slot, written)
  const tokens: Array<number> = []
  const keys: Array<number> = []
  const starts: Array<number> = []
  const literal = (token: number) => {
    tokens.push(token)
    keys.push(token)
  }
  let next = 0
  for (let i = 0; i < prompt.length; i++) {
    if (!slots.has(i)) {
      literal(prompt[i])
      continue
    }
    const image = images[next++]
    if (!image) throw new Error("The prompt has more image slots than images.")
    if (slot.kind === "bare") literal(slot.start)
    for (const view of image.views) {
      if (slot.kind === "bare" && view.place) literal(view.place === "thumbnail" ? slot.thumbnail : slot.tile(view.place.row, view.place.col))
      starts.push(tokens.length)
      for (let r = 0; r < view.height * view.width; r++) {
        tokens.push(slot.pad)
        keys.push(image.key)
      }
    }
    if (slot.kind === "bare") literal(slot.end)
  }
  if (next !== images.length) throw new Error(`The prompt has ${next} image slots for ${images.length} images.`)
  return { tokens, keys, starts }
}

/** An image's bytes: fetched (the browser's HTTP cache decides if they're fresh), or a data URL's. */
export async function fetchImage(url: string, signal?: AbortSignal): Promise<Blob> {
  try {
    const response = await fetch(url, signal ? { signal } : {})
    if (!response.ok) throw new Error(`HTTP ${response.status}`)
    return await response.blob()
  } catch (error) {
    if (signal?.aborted) throw signal.reason
    throw new Error(`Couldn't fetch an image (${error instanceof Error ? error.message : String(error)}).`)
  }
}

/**
 * Decodes an image's bytes and resizes it into its views (on white, as
 * transformers' convert_to_rgb composites transparency). `fits` sees its
 * views before any pixels are made, and may throw to stop there.
 */
export async function decodeImage(
  blob: Blob,
  detail: ImageDetail | undefined,
  config: ImageProcessorConfig,
  maxPatches: number,
  fits: (image: ImageLayout) => void,
  signal?: AbortSignal
): Promise<PreparedImage> {
  let bitmap: ImageBitmap
  try {
    bitmap = await createImageBitmap(blob)
  } catch {
    throw new Error("An image couldn't be decoded: give a PNG, JPEG, WebP or GIF.")
  }
  try {
    signal?.throwIfAborted()
    const merged = config.patchSize * config.mergeSize
    const plans = planViews(bitmap.height, bitmap.width, detail, config, maxPatches)
    const views: Array<ImageView> = plans.map((plan) => ({
      height: plan.height / merged,
      width: plan.width / merged,
      ...(plan.place && { place: plan.place }),
      pixels: null,
      features: null
    }))
    fits({ views })
    for (const [i, plan] of plans.entries()) {
      signal?.throwIfAborted()
      const canvas = new OffscreenCanvas(plan.width, plan.height)
      const context = canvas.getContext("2d", { willReadFrequently: true })
      if (!context) throw new Error("No 2D canvas to resize an image on.")
      context.fillStyle = "#fff"
      context.fillRect(0, 0, plan.width, plan.height)
      context.imageSmoothingEnabled = true
      context.imageSmoothingQuality = "high"
      // A tile: the image resized to the whole grid, drawn so the tile's part lands on the canvas.
      if (plan.from) context.drawImage(bitmap, -plan.from.x, -plan.from.y, plan.from.width, plan.from.height)
      else context.drawImage(bitmap, 0, 0, plan.width, plan.height)
      views[i].pixels = { rgba: context.getImageData(0, 0, plan.width, plan.height).data, height: plan.height, width: plan.width }
    }
    const bytes = new Uint8Array(await blob.arrayBuffer())
    const first = plans[0]
    return { key: imageKey(bytes, first.height * plans.length, first.width), views }
  } finally {
    bitmap.close()
  }
}

/**
 * Recently prepared images, least recent first: by content (the bytes'
 * hash, detail), and data URLs (whose bytes never change) by URL too.
 */
export class ImageCache {
  private byContent = new Map<string, PreparedImage>()
  private byUrl = new Map<string, string>()
  constructor(private readonly size = 8) {}

  /** A data URL's image, if it was prepared lately. */
  getUrl(image: ChatImage) {
    const content = this.byUrl.get(ImageCache.urlId(image))
    return content === undefined ? undefined : this.get(content)
  }

  get(content: string) {
    const found = this.byContent.get(content)
    if (found) {
      this.byContent.delete(content)
      this.byContent.set(content, found)
    }
    return found
  }

  set(image: ChatImage, content: string, prepared: PreparedImage) {
    this.byContent.delete(content)
    this.byContent.set(content, prepared)
    if (image.url.startsWith("data:")) this.byUrl.set(ImageCache.urlId(image), content)
    while (this.byContent.size > this.size) this.byContent.delete(this.byContent.keys().next().value!)
    for (const [url, entry] of this.byUrl) if (!this.byContent.has(entry)) this.byUrl.delete(url)
  }

  clear() {
    this.byContent.clear()
    this.byUrl.clear()
  }

  private static urlId(image: ChatImage) {
    return `${image.detail ?? "auto"}|${image.url}`
  }
}

/** An image's content id: its bytes' hash and its detail. */
export async function contentId(blob: Blob, detail: ImageDetail | undefined): Promise<string> {
  return `${detail ?? "auto"}|${imageKey(new Uint8Array(await blob.arrayBuffer()), 0, 0)}`
}
