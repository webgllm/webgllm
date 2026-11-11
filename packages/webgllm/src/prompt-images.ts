/**
 * A prompt's images in the worker: fetched and resized as the model's
 * processor says (image.ts), each image slot the chat template wrote
 * expanded to the image's tokens, and keys for the prompt cache that tell
 * images apart (see handler.ts). Embedding rows are kept for a few recent
 * images, by content, so a chat that reads an image again doesn't encode it
 * again.
 */
import { pixelBudget, smartResize, toPatches, type ImageDetail, type ImagePatches, type ImageProcessorConfig } from "./image.js"
import type { ChatMessage } from "./tokenizer.js"

/** An image part of a chat message, as the worker gets it. */
export type ChatImage = { url: string; detail?: ImageDetail }

/** An image ready for a prompt. */
export type PreparedImage = {
  /** The prompt cache's key for its rows: negative (never a token id), from its bytes, detail and size. */
  key: number
  /** Merged tokens: rows and columns. */
  height: number
  width: number
  /** Its resized pixels (RGBA), until it's encoded. */
  pixels: { rgba: Uint8ClampedArray; height: number; width: number } | null
  /** Its embedding rows, once encoded. */
  features: Float32Array | null
}

/** An image's patches, cut from its resized pixels when it's encoded. */
export function patchesOf(image: PreparedImage, config: ImageProcessorConfig): ImagePatches {
  if (!image.pixels) throw new Error("An image's pixels were dropped before it was encoded.")
  const { rgba, height, width } = image.pixels
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

/** The chat template's image slot: `<|vision_start|><|image_pad|><|vision_end|>`, as token ids. */
export type ImageSlot = { start: number; pad: number; end: number }

/** How many image slots the template wrote (a pad between a start and an end; a pad in text isn't one). */
export function countImageSlots(prompt: ArrayLike<number>, slot: ImageSlot): number {
  let count = 0
  for (let i = 1; i + 1 < prompt.length; i++)
    if (prompt[i] === slot.pad && prompt[i - 1] === slot.start && prompt[i + 1] === slot.end) count++
  return count
}

/**
 * The prompt with each image slot's pad expanded to its image's tokens,
 * and the keys the cache compares: the tokens, with each image's rows its
 * key. Only pads the template put between a vision start and end are
 * slots; the same token written in a message's text stays as it is.
 */
export function expandImageTokens(
  prompt: Array<number>,
  slot: ImageSlot,
  images: Array<Pick<PreparedImage, "key" | "height" | "width">>
): { tokens: Array<number>; keys: Array<number>; starts: Array<number> } {
  const tokens: Array<number> = []
  const keys: Array<number> = []
  const starts: Array<number> = []
  let next = 0
  for (let i = 0; i < prompt.length; i++) {
    const token = prompt[i]
    const isSlot = token === slot.pad && prompt[i - 1] === slot.start && prompt[i + 1] === slot.end
    if (!isSlot) {
      tokens.push(token)
      keys.push(token)
      continue
    }
    const image = images[next++]
    if (!image) throw new Error("The prompt has more image slots than images.")
    starts.push(tokens.length)
    for (let r = 0; r < image.height * image.width; r++) {
      tokens.push(token)
      keys.push(image.key)
    }
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

/** The size an image is resized to, and its tokens, for its detail and the GPU's patch limit. */
export function resizedSize(
  height: number,
  width: number,
  detail: ImageDetail | undefined,
  config: ImageProcessorConfig,
  maxPatches: number
) {
  const factor = config.patchSize * config.mergeSize
  const budget = pixelBudget(detail, config)
  // The encoder reads at most maxPatches (a texture's height).
  const maxPixels = Math.min(budget.maxPixels, maxPatches * config.patchSize * config.patchSize)
  const size = smartResize(height, width, factor, Math.min(budget.minPixels, maxPixels), maxPixels)
  return { ...size, tokens: (size.height / factor) * (size.width / factor) }
}

/**
 * Decodes an image's bytes and resizes it into its pixel budget (on white,
 * as transformers' convert_to_rgb composites transparency). `fits` sees its
 * token count before any pixels are made, and may throw to stop there.
 */
export async function decodeImage(
  blob: Blob,
  detail: ImageDetail | undefined,
  config: ImageProcessorConfig,
  maxPatches: number,
  fits: (tokens: number) => void,
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
    const size = resizedSize(bitmap.height, bitmap.width, detail, config, maxPatches)
    fits(size.tokens)
    const canvas = new OffscreenCanvas(size.width, size.height)
    const context = canvas.getContext("2d", { willReadFrequently: true })
    if (!context) throw new Error("No 2D canvas to resize an image on.")
    context.fillStyle = "#fff"
    context.fillRect(0, 0, size.width, size.height)
    context.imageSmoothingEnabled = true
    context.imageSmoothingQuality = "high"
    context.drawImage(bitmap, 0, 0, size.width, size.height)
    const rgba = context.getImageData(0, 0, size.width, size.height).data
    const bytes = new Uint8Array(await blob.arrayBuffer())
    const merged = config.patchSize * config.mergeSize
    return {
      key: imageKey(bytes, size.height, size.width),
      height: size.height / merged,
      width: size.width / merged,
      pixels: { rgba, height: size.height, width: size.width },
      features: null
    }
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
