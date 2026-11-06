/**
 * The page's side of one WebGL 2 worker (worker.ts), which runs a model off
 * the main thread on an OffscreenCanvas. One worker holds one model; the
 * pool (pool.ts) keeps a worker per loaded model.
 */
import { workerError } from "./errors.js"
import { MIN_TEXTURE_SIZE } from "./limits.js"
import type { FinishReason, RunTimings, WorkerRequest, WorkerResponse } from "./protocol.js"

/** What a worker sends that the pool acts on. */
export type WorkerEvents = {
  onProgress: (progress: Extract<WorkerResponse, { type: "progress" }>) => void
  /** The model needs about `bytes`: resolve once there's room. */
  onClaim: (bytes: number) => Promise<void>
  onLog: (text: string) => void
  /** The worker failed to start or crashed: it's gone. */
  onCrash: (error: Error) => void
}

/** How a generation ended, as the worker reports it. */
export type WorkerGenerationEnd = {
  stats: RunTimings
  finishReason: FinishReason
  toolCall?: { name: string; arguments: Record<string, unknown> }
}

type Pending<T> = {
  promise: Promise<T>
  resolve: (value: T) => void
  reject: (error: unknown) => void
}

function pending<T>(): Pending<T> {
  let resolve!: (value: T) => void
  let reject!: (error: unknown) => void
  const promise = new Promise<T>((res, rej) => {
    resolve = res
    reject = rej
  })
  // Callers may stop waiting (an abort); an unread failure isn't an error.
  promise.catch(() => undefined)
  return { promise, resolve, reject }
}

type Generation = Pending<WorkerGenerationEnd> & { onToken: (text: string) => void }

/** Engine knobs every new worker starts with (see setPromptBatch). */
const tuning: Record<string, number> = {}
const live = new Set<ModelWorker>()

/** The bundled worker. Written out here so bundlers see the `new URL(…, import.meta.url)`. */
export const defaultWorker = () =>
  new Worker(new URL("./worker.js", import.meta.url), { type: "module", name: "webgllm" })

export class ModelWorker {
  private readonly worker: Worker
  private load: (Pending<void> & { contextLength: number | undefined }) | null = null
  private downloading: Pending<void> | null = null
  private readonly generations = new Map<string, Generation>()
  /** Loaded and ready, for the context it was asked for (see contextLength). */
  ready = false
  /** Tokens the loaded model holds, as the worker reported it. */
  loadedContextLength: number | undefined
  /** The context the model was asked for: undefined for the model's own. */
  contextLength: number | undefined
  private gone: Error | null = null

  constructor(
    readonly model: string,
    create: () => Worker,
    private readonly events: WorkerEvents
  ) {
    this.worker = create()
    this.worker.onmessage = (event: MessageEvent<WorkerResponse>) => this.handle(event.data)
    this.worker.onerror = (event: ErrorEvent) => {
      const error = new Error(event.message || "The model's worker stopped.")
      this.terminate(error)
      this.events.onCrash(error)
    }
    live.add(this)
    if (Object.keys(tuning).length) this.post({ type: "tune", values: { ...tuning } })
  }

  get loading(): boolean {
    return this.load !== null && !this.ready
  }

  get busy(): boolean {
    return this.generations.size > 0
  }

  /** The load in progress or done, if one was asked for. */
  get loadPromise(): Promise<void> | null {
    return this.load?.promise ?? null
  }

  private idlers: Array<() => void> = []

  /** Resolves once no generation is running (at once if none is). */
  idle(): Promise<void> {
    if (!this.busy || this.gone) return Promise.resolve()
    return new Promise((resolve) => this.idlers.push(resolve))
  }

  private settled() {
    if (this.busy) return
    for (const resolve of this.idlers.splice(0)) resolve()
  }

  post(message: WorkerRequest) {
    if (!this.gone) this.worker.postMessage(message)
  }

  /**
   * Loads the model for chats of up to `contextLength` tokens (the model's
   * own context when undefined); a different context loads it again.
   */
  loadModel(contextLength: number | undefined): Promise<void> {
    if (this.gone) return Promise.reject(this.gone)
    if (this.load && this.contextLength === contextLength) return this.load.promise
    this.load?.reject(new Error("The model was loaded again for another context length."))
    this.ready = false
    this.contextLength = contextLength
    this.load = { ...pending<void>(), contextLength }
    this.post({ type: "load", model: this.model, ...(contextLength !== undefined && { contextLength }) })
    return this.load.promise
  }

  /** Downloads the model's files to storage, without loading it. */
  download(): Promise<void> {
    if (this.gone) return Promise.reject(this.gone)
    if (!this.downloading) {
      this.downloading = pending<void>()
      this.post({ type: "download", model: this.model })
    }
    return this.downloading.promise
  }

  /**
   * Runs a generation, calling `onToken` with the model's raw text as it
   * comes; settles when the worker ends it (stopped ones included).
   */
  generate(
    request: Omit<Extract<WorkerRequest, { type: "generate" }>, "type" | "model" | "contextLength">,
    onToken: (text: string) => void
  ): Promise<WorkerGenerationEnd> {
    if (this.gone) return Promise.reject(this.gone)
    const generation = { ...pending<WorkerGenerationEnd>(), onToken }
    this.generations.set(request.id, generation)
    this.post({
      type: "generate",
      ...request,
      model: this.model,
      // Exactly what it was loaded for: another value loads it again.
      ...(this.contextLength !== undefined && { contextLength: this.contextLength })
    })
    return generation.promise
  }

  stop(id: string) {
    this.post({ type: "stop", id })
  }

  /** Ends the worker, and with it the model and all its GPU memory. */
  terminate(reason: Error) {
    if (this.gone) return
    this.gone = reason
    this.ready = false
    live.delete(this)
    this.worker.terminate()
    this.load?.reject(reason)
    this.load = null
    this.downloading?.reject(reason)
    this.downloading = null
    for (const generation of this.generations.values()) generation.reject(reason)
    this.generations.clear()
    this.settled()
  }

  get terminated(): boolean {
    return this.gone !== null
  }

  private handle(data: WorkerResponse) {
    switch (data.type) {
      case "progress":
        this.events.onProgress(data)
        break
      case "claim":
        void this.events
          .onClaim(data.bytes)
          .catch(() => undefined)
          .then(() => this.post({ type: "claimed" }))
        break
      case "ready":
        if (this.load) {
          this.ready = true
          this.loadedContextLength = data.contextLength
          this.load.resolve()
        }
        break
      case "load-error":
        if (this.load) {
          this.load.reject(workerError(data.message, data.code))
          this.load = null
          this.ready = false
        }
        break
      case "downloaded":
        this.downloading?.resolve()
        this.downloading = null
        break
      case "download-error":
        this.downloading?.reject(workerError(data.message, data.code))
        this.downloading = null
        break
      case "token":
        this.generations.get(data.id)?.onToken(data.text)
        break
      case "done": {
        const generation = this.generations.get(data.id)
        this.generations.delete(data.id)
        generation?.resolve({
          stats: data.stats,
          // (A worker from before finish reasons: its end-of-turn token.)
          finishReason: data.finishReason ?? "stop",
          ...(data.toolCall && { toolCall: data.toolCall })
        })
        this.settled()
        break
      }
      case "error": {
        const generation = this.generations.get(data.id)
        this.generations.delete(data.id)
        generation?.reject(workerError(data.message, data.code))
        this.settled()
        break
      }
      case "log":
        this.events.onLog(data.text)
        break
      case "phase":
        break
    }
  }
}

/**
 * Prompt tokens per forward batch from each model's next load on
 * (undefined: the default, 512 on every device). Smaller batches take less
 * memory and read prompts slower: 128 holds Qwen3.5 2B's 8k-token chats in
 * ~270 MiB less (an iPhone page has ~2 GB), at about half the prompt speed.
 * A loaded model keeps its batch until it loads again.
 */
export function setPromptBatch(tokens: number | undefined) {
  if (tokens === undefined) delete tuning.promptBatch
  else tuning.promptBatch = tokens
  for (const worker of live) worker.post({ type: "tune", values: { promptBatch: tokens ?? 0 } })
}

/**
 * Whether this browser can run the engine: WebGL 2 with float render
 * targets and big enough textures, on an OffscreenCanvas in a worker.
 */
export function hasWebGl2(): boolean {
  if (typeof OffscreenCanvas === "undefined" || typeof Worker === "undefined") return false
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
