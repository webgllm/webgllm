/**
 * A prompt's images in the worker: fetched and resized as the model's
 * processor says (image.ts), each image's `<|image_pad|>` token expanded to
 * its tokens, and keys for the prompt cache that tell images apart (see
 * worker.ts). Embedding rows are kept for a few recent images, so a chat
 * that reads an image again doesn't encode it again.
 */
import { pixelBudget, smartResize, toPatches, type ImageDetail, type ImagePatches, type ImageProcessorConfig } from "./image.js"
import type { ChatMessage } from "./tokenizer.js"

/** An image part of a chat message, as the worker gets it. */
export type ChatImage = { url: string; detail?: ImageDetail }

/** An image ready for a prompt. */
export type PreparedImage = {
  /** The prompt cache's key for its rows: negative (never a token id), from its pixels and size. */
  key: number
  /** Merged tokens: rows and columns. */
  height: number
  width: number
  /** Its patches, until it's encoded. */
  patches: ImagePatches | null
  /** Its embedding rows, once encoded. */
  features: Float32Array | null
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

/** FNV-1a and a second multiplicative hash of the bytes, as one 52-bit negative key. */
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
 * The prompt with each image token expanded to its image's tokens, and the
 * keys the cache compares: the tokens, with each image's rows its key.
 */
export function expandImageTokens(
  prompt: Array<number>,
  imageToken: number,
  images: Array<Pick<PreparedImage, "key" | "height" | "width">>
): { tokens: Array<number>; keys: Array<number>; starts: Array<number> } {
  const tokens: Array<number> = []
  const keys: Array<number> = []
  const starts: Array<number> = []
  let next = 0
  for (const token of prompt) {
    if (token !== imageToken) {
      tokens.push(token)
      keys.push(token)
      continue
    }
    const image = images[next++]
    if (!image) throw new Error("The prompt has more image tokens than images.")
    starts.push(tokens.length)
    for (let i = 0; i < image.height * image.width; i++) {
      tokens.push(imageToken)
      keys.push(image.key)
    }
  }
  if (next !== images.length) throw new Error(`The prompt has ${next} image tokens for ${images.length} images.`)
  return { tokens, keys, starts }
}

/**
 * Fetches an image (an http(s) or data URL), resizes it into the
 * processor's pixel budget for its detail (on white, as transformers'
 * convert_to_rgb composites transparency) and cuts it into patches.
 */
export async function decodeImage(
  image: ChatImage,
  config: ImageProcessorConfig,
  maxPatches: number,
  signal?: AbortSignal
): Promise<PreparedImage> {
  let blob: Blob
  try {
    const response = await fetch(image.url, signal ? { signal } : {})
    if (!response.ok) throw new Error(`HTTP ${response.status}`)
    blob = await response.blob()
  } catch (error) {
    throw new Error(`Couldn't fetch an image (${error instanceof Error ? error.message : String(error)}).`)
  }
  let bitmap: ImageBitmap
  try {
    bitmap = await createImageBitmap(blob)
  } catch {
    throw new Error("An image couldn't be decoded: give a PNG, JPEG, WebP or GIF.")
  }
  const factor = config.patchSize * config.mergeSize
  const budget = pixelBudget(image.detail, config)
  // The encoder reads at most maxPatches (a texture's height).
  const maxPixels = Math.min(budget.maxPixels, maxPatches * config.patchSize * config.patchSize)
  const size = smartResize(bitmap.height, bitmap.width, factor, Math.min(budget.minPixels, maxPixels), maxPixels)
  const canvas = new OffscreenCanvas(size.width, size.height)
  const context = canvas.getContext("2d", { willReadFrequently: true })
  if (!context) throw new Error("No 2D canvas to resize an image on.")
  context.fillStyle = "#fff"
  context.fillRect(0, 0, size.width, size.height)
  context.imageSmoothingEnabled = true
  context.imageSmoothingQuality = "high"
  context.drawImage(bitmap, 0, 0, size.width, size.height)
  bitmap.close()
  const rgba = context.getImageData(0, 0, size.width, size.height).data
  const patches = toPatches(rgba, size.height, size.width, config)
  const merge = config.mergeSize
  return {
    key: imageKey(new Uint8Array(rgba.buffer, rgba.byteOffset, rgba.byteLength), size.height, size.width),
    height: patches.gridH / merge,
    width: patches.gridW / merge,
    patches,
    features: null
  }
}

/** Recently prepared images by URL and detail, least recent first. */
export class ImageCache {
  private entries = new Map<string, PreparedImage>()
  constructor(private readonly size = 8) {}

  private static id(image: ChatImage) {
    return `${image.detail ?? "auto"}|${image.url}`
  }

  get(image: ChatImage) {
    const id = ImageCache.id(image)
    const found = this.entries.get(id)
    if (found) {
      this.entries.delete(id)
      this.entries.set(id, found)
    }
    return found
  }

  set(image: ChatImage, prepared: PreparedImage) {
    const id = ImageCache.id(image)
    this.entries.delete(id)
    this.entries.set(id, prepared)
    while (this.entries.size > this.size) this.entries.delete(this.entries.keys().next().value!)
  }

  clear() {
    this.entries.clear()
  }
}
