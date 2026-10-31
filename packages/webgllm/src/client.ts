/**
 * The page's side of the WebGL 2 worker (worker.ts), which runs the model
 * off the main thread on an OffscreenCanvas. One worker, one model in it at
 * a time: loading another frees the previous one.
 */
import { MIN_TEXTURE_SIZE } from "./limits.js"
import type { WebGlTextModel } from "./model-meta.js"
import type { RunTimings, WorkerRequest, WorkerResponse } from "./protocol.js"
import type { Sampling } from "./sampler.js"
import { createStore } from "./store.js"
import type { Store } from "./store.js"
import type { ChatMessage, ChatTool } from "./tokenizer.js"

export type ModelStatus =
  | { state: "idle" }
  | {
      state: "loading"
      /** 0-1 across downloading and uploading to the GPU. */
      progress: number
      /** True when the files come from the browser's storage, not the network. */
      cached: boolean
      loadedBytes?: number
      totalBytes?: number
    }
  | { state: "ready" }
  | { state: "error"; message: string }

const statuses = new Map<string, Store<ModelStatus>>()

/** Loading status of a model, by its name. */
export function modelStatus(model: WebGlTextModel): Store<ModelStatus> {
  let status = statuses.get(model)
  if (!status) {
    status = createStore<ModelStatus>({ state: "idle" })
    statuses.set(model, status)
  }
  return status
}

type Generation = {
  onToken: (text: string) => void
  onEnd: (error?: string, stats?: RunTimings) => void
}
const generations = new Map<string, Generation>()

type Pending = {
  promise: Promise<void>
  resolve: () => void
  reject: (error: Error) => void
}
let loaded: { model: string; load: Pending } | null = null
let worker: Worker | null = null

function handleMessage({ data }: MessageEvent<WorkerResponse>) {
  if (data.type === "progress") {
    modelStatus(data.model as WebGlTextModel).set({
      state: "loading",
      progress: data.progress,
      cached: data.cached,
      ...(data.totalBytes !== undefined && {
        loadedBytes: data.loadedBytes ?? 0,
        totalBytes: data.totalBytes
      })
    })
  } else if (data.type === "ready") {
    modelStatus(data.model as WebGlTextModel).set({ state: "ready" })
    if (loaded?.model === data.model) loaded.load.resolve()
  } else if (data.type === "load-error") {
    modelStatus(data.model as WebGlTextModel).set({
      state: "error",
      message: data.message
    })
    if (loaded?.model === data.model) {
      loaded.load.reject(new Error(data.message))
      loaded = null
    }
  } else if (data.type === "token") {
    generations.get(data.id)?.onToken(data.text)
  } else if (data.type === "log") {
    console.info(data.text)
  } else {
    const generation = generations.get(data.id)
    generations.delete(data.id)
    if (data.type === "error") generation?.onEnd(data.message)
    else generation?.onEnd(undefined, data.stats)
  }
}

function post(message: WorkerRequest) {
  if (!worker) {
    worker = new Worker(new URL("./worker.js", import.meta.url), {
      type: "module",
      name: "webgl"
    })
    worker.onmessage = handleMessage
    worker.onerror = (event) =>
      failWorker(event.message || "The model's worker stopped.")
  }
  worker.postMessage(message)
}

/**
 * The worker failed to start or crashed: everything waiting on it fails,
 * and the next request starts a new one.
 */
function failWorker(message: string) {
  worker?.terminate()
  worker = null
  if (loaded) {
    modelStatus(loaded.model as WebGlTextModel).set({ state: "error", message })
    loaded.load.reject(new Error(message))
    loaded = null
  }
  for (const [id, generation] of generations) {
    generations.delete(id)
    generation.onEnd(message)
  }
}

let askedToKeep = false
/**
 * Asks the browser to keep this site's storage, so a downloaded model isn't
 * deleted to free space. Browsers may say no; only pages can ask.
 */
function keepDownloads() {
  if (askedToKeep || typeof navigator === "undefined") return
  askedToKeep = true
  navigator.storage?.persist?.().catch(() => undefined)
}


/**
 * Whether this browser can run the engine: WebGL 2 with float render
 * targets and big enough textures, on an OffscreenCanvas in a worker.
 */
export function hasWebGl2(): boolean {
  if (typeof OffscreenCanvas === "undefined") return false
  try {
    const gl = new OffscreenCanvas(1, 1).getContext("webgl2")
    const ok =
      !!gl?.getExtension("EXT_color_buffer_float") &&
      (gl.getParameter(gl.MAX_TEXTURE_SIZE) as number) >= MIN_TEXTURE_SIZE
    gl?.getExtension("WEBGL_lose_context")?.loseContext()
    return ok
  } catch {
    return false
  }
}

/** Loads a model (downloading it the first time), replacing any other. */
export function loadModel(model: WebGlTextModel): Promise<void> {
  if (loaded?.model === model) return loaded.load.promise
  if (loaded) {
    modelStatus(loaded.model as WebGlTextModel).set({ state: "idle" })
    loaded.load.reject(new Error("Switched to another model."))
  }
  let resolve!: () => void
  let reject!: (error: Error) => void
  const promise = new Promise<void>((res, rej) => {
    resolve = res
    reject = rej
  })
  promise.catch(() => undefined)
  loaded = { model, load: { promise, resolve, reject } }
  keepDownloads()
  modelStatus(model).set({ state: "loading", progress: 0, cached: true })
  post({ type: "load", model })
  return promise
}

export type GenerationEnd = { failure?: string; stats?: RunTimings }

/**
 * Runs one generation with the loaded model, yielding its text as it comes
 * (thinking and tool-call markup included, as the model writes them).
 */
export async function* generate(request: {
  id: string
  model: WebGlTextModel
  chat: Array<ChatMessage>
  tools: Array<ChatTool>
  thinking?: boolean
  sampling: Sampling
  maxTokens?: number
  /** Speculative decoding by prompt lookup (default on). */
  speculative?: boolean
  /** Reading the likeliest next token ahead (default on). */
  pipeline?: boolean
  signal?: AbortSignal
}): AsyncGenerator<string, GenerationEnd> {
  const { id, signal } = request
  // Stopped before it started (say, while the model downloaded).
  if (signal?.aborted) return {}
  const queue: Array<string> = []
  let wake: (() => void) | null = null
  let finished = false
  let end: GenerationEnd = {}
  generations.set(id, {
    onToken: (text) => {
      queue.push(text)
      wake?.()
    },
    onEnd: (failure, stats) => {
      end = { failure, stats }
      finished = true
      wake?.()
    }
  })
  const onAbort = () => post({ type: "stop", id })
  signal?.addEventListener("abort", onAbort)
  post({
    type: "generate",
    id,
    model: request.model,
    chat: request.chat,
    tools: request.tools,
    ...(request.thinking !== undefined && { thinking: request.thinking }),
    sampling: request.sampling,
    ...(request.maxTokens !== undefined && { maxTokens: request.maxTokens }),
    ...(request.speculative !== undefined && {
      speculative: request.speculative
    }),
    ...(request.pipeline !== undefined && { pipeline: request.pipeline })
  })
  try {
    while (!finished || queue.length > 0) {
      const text = queue.shift()
      if (text !== undefined) {
        yield text
        continue
      }
      await new Promise<void>((resolve) => {
        wake = resolve
      })
      wake = null
    }
  } finally {
    signal?.removeEventListener("abort", onAbort)
    // The caller stopped reading before the end: stop the worker too, so it
    // doesn't keep generating into the next request.
    if (!finished) {
      generations.delete(id)
      post({ type: "stop", id })
    }
  }
  return end
}
