/** Classifying the worker's failures into the protocol's error codes. */
import { StructuredOutputError } from "./constraint/index.js"
import { GpuError } from "./gpu/gl.js"
import type { WorkerErrorCode } from "./protocol.js"
import { isQuotaError } from "./storage.js"

/** An error whose code the worker knew where it threw it. */
export class CodedError extends Error {
  constructor(message: string, readonly code: WorkerErrorCode, options?: ErrorOptions) {
    super(message, options)
  }
}

/**
 * GpuErrors are thrown with a message only, by the engine and the GL layer:
 * their causes are told apart by it. A browser or GPU short of what the
 * engine needs at all counts as WebGL 2 being unavailable.
 */
const GPU_CODES: Array<[RegExp, WorkerErrorCode]> = [
  [/context was lost/i, "gpu-context-lost"],
  [/out of memory/i, "out-of-memory"],
  [/no WebGL 2|no OffscreenCanvas|EXT_color_buffer_float|textures go up to/i, "webgl2-unavailable"]
]

/** The code for `error`, if the worker can tell what it was. */
export function errorCode(error: unknown): WorkerErrorCode | undefined {
  if (error instanceof CodedError) return error.code
  if (error instanceof StructuredOutputError) return error.code === "unsupported-schema" ? "invalid-schema" : undefined
  if (isQuotaError(error)) return "storage-full"
  if (error instanceof GpuError) return GPU_CODES.find(([pattern]) => pattern.test(error.message))?.[1]
  return undefined
}

/**
 * A failure while downloading or storing a model's files: storage-full when
 * the origin's quota ran out, download-failed otherwise. Aborts (another
 * model asked for) pass through as they are.
 */
export function downloadError(error: unknown, signal?: AbortSignal): unknown {
  if (signal?.aborted || error instanceof CodedError) return error
  const message = (error as { message?: unknown } | null)?.message ?? String(error)
  return isQuotaError(error)
    ? new CodedError(`There's no room left to store the model: ${message}`, "storage-full", { cause: error })
    : new CodedError(`Couldn't download the model: ${message}`, "download-failed", { cause: error })
}
