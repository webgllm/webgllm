import type { WorkerErrorCode } from "./protocol.js"

/** What went wrong, for code to act on (the message is for people). */
export type WebGLLMErrorCode =
  | WorkerErrorCode
  | "model-not-supported"
  | "model-not-loaded"
  | "unsupported-parameter"
  /** A model's worker failed: its script didn't load, or its code threw. */
  | "worker-failed"

/**
 * A failure webgllm can name. Other failures (a malformed request value)
 * are plain `Error`s, `TypeError`s or
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

/**
 * Whether a worker's "error" event says its script never loaded: such an
 * event carries nothing (no message, no file), where an error thrown by
 * running code names both. A worker that didn't load never runs; one whose
 * code threw is still running.
 */
export const failedToLoad = (event: { message?: string; filename?: string }): boolean => !event.message && !event.filename

/** The failure a worker's "error" event reports. */
export function workerFailure(event: { message?: string; filename?: string }): WebGLLMError {
  return new WebGLLMError(
    "worker-failed",
    failedToLoad(event) ? "The worker failed to load its script, so it can't run models." : `The model's worker failed: ${event.message || "an error in its code."}`
  )
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
