/**
 * The image model messages between the page (images.ts) and a worker
 * (image-worker.ts), beside the text models' (protocol.ts). They carry the
 * same mark (WEBGLLM), so a worker of the caller's can hold both.
 */
import { WEBGLLM } from "./protocol.js"

export type ImageRequest =
  | { type: "image-load"; id: number; model: string }
  | {
      type: "image-generate"
      id: number
      model: string
      prompt: string
      n: number
      seed: number
      steps: number
      guidanceScale: number
      /** PNG files (with each pixel `scale` × `scale`) besides the raw pixels. */
      png: boolean
      scale: number
    }
  | { type: "image-stop"; id: number }
  | { type: "image-unload"; id: number; model: string }
  /** The page's answer to claim `id` (an image-claim's `claim`): granted, or refused with why. */
  | { type: "image-claimed"; id: number; refused?: string }

export type ImageResponse =
  /** Before allocating: the bytes the model will hold in all (replacing its last claim), for request `id`. */
  | { type: "image-claim"; id: number; claim: number; model: string; bytes: number }
  | { type: "image-progress"; id: number; model: string; phase: "download" | "load"; progress: number; cached: boolean }
  | { type: "image-ready"; id: number; model: string; engine: "webgpu" | "webgl"; weightBytes: number; selftest: number }
  | { type: "image-step"; id: number; step: number; steps: number }
  | { type: "image-done"; id: number; images: Array<{ rgba: Uint8Array; png?: Uint8Array }>; timings: { textMs: number; drawMs: number }; workBytes: number }
  | { type: "image-error"; id: number; message: string }
  | { type: "image-unloaded"; id: number }

const TYPES: ReadonlySet<string> = new Set<ImageRequest["type"]>(["image-load", "image-generate", "image-stop", "image-unload", "image-claimed"])

/** Whether a message is one of the page's image requests (marked as webgllm's, with an id). */
export function isImageRequest(data: unknown): data is ImageRequest {
  if (typeof data !== "object" || data === null) return false
  const d = data as Record<string, unknown>
  return d[WEBGLLM] === true && typeof d.type === "string" && TYPES.has(d.type) && typeof d.id === "number"
}

/** Whether a message is a worker's answer to an image request. */
export function isImageResponse(data: unknown): data is ImageResponse {
  if (typeof data !== "object" || data === null) return false
  const d = data as Record<string, unknown>
  return d[WEBGLLM] === true && typeof d.type === "string" && d.type.startsWith("image-") && typeof d.id === "number"
}
