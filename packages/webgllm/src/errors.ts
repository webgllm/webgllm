import type { WorkerErrorCode } from "./protocol.js"

/** What went wrong, for code to act on (the message is for people). */
export type WebGLLMErrorCode =
  | WorkerErrorCode
  | "model-not-supported"
  | "model-not-loaded"
  | "unsupported-parameter"

/**
 * A failure webgllm can name. Other failures (a worker that crashed, a
 * malformed request value) are plain `Error`s, `TypeError`s or
 * `RangeError`s; an aborted call rejects with a DOMException named
 * "AbortError".
 */
export class WebGLLMError extends Error {
  readonly code: WebGLLMErrorCode

  constructor(code: WebGLLMErrorCode, message: string, options?: ErrorOptions) {
    super(message, options)
    this.name = "WebGLLMError"
    this.code = code
  }
}

/** A worker's failure: a WebGLLMError when it carries a code, else an Error. */
export function workerError(message: string, code?: WorkerErrorCode): Error {
  return code ? new WebGLLMError(code, message) : new Error(message)
}

/** The rejection an aborted call gets: the signal's reason when it's an AbortError. */
export function abortError(signal?: AbortSignal): unknown {
  const reason: unknown = signal?.reason
  if (reason instanceof DOMException && reason.name === "AbortError") return reason
  return new DOMException("The operation was aborted.", "AbortError")
}

export function throwIfAborted(signal?: AbortSignal) {
  if (signal?.aborted) throw abortError(signal)
}

/**
 * `promise`, or an AbortError as soon as `signal` aborts (the work itself
 * goes on; the caller stops it if it should).
 */
export function abortable<T>(promise: Promise<T>, signal?: AbortSignal): Promise<T> {
  if (!signal) return promise
  if (signal.aborted) {
    promise.catch(() => undefined)
    return Promise.reject(abortError(signal))
  }
  return new Promise<T>((resolve, reject) => {
    const onAbort = () => reject(abortError(signal))
    signal.addEventListener("abort", onAbort, { once: true })
    promise.then(
      (value) => {
        signal.removeEventListener("abort", onAbort)
        resolve(value)
      },
      (error: unknown) => {
        signal.removeEventListener("abort", onAbort)
        reject(error)
      }
    )
  })
}
