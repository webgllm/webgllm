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

type Load = Pending<void> & {
  contextLength: number | undefined
  id: number
  /** Not sent yet (it waits for the reply running): the load to go back to if it's called off. */
  previous?: Load
}

/** Engine knobs every new worker starts with (see setPromptBatch). */
const tuning: Record<string, number> = {}
const live = new Set<ModelWorker>()

/** The bundled worker. Written out here so bundlers see the `new URL(…, import.meta.url)`. */
export const defaultWorker = () =>
  new Worker(new URL("./worker.js", import.meta.url), { type: "module", name: "webgllm" })

export class ModelWorker {
  private readonly worker: Worker
  private load: Load | null = null
  /** The last load's id: the worker's "ready" or "load-error" names the load it answers. */
  private loads = 0
  /** Loads sent and not answered yet. */
  private readonly open = new Set<number>()
  private downloading: Pending<void> | null = null
  private readonly stored = pending<void>()
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

  /** Resolves once a load has the model's files in storage (a worker that says so). */
  get storedPromise(): Promise<void> {
    return this.stored.promise
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
    const replaced = new Error("The model was loaded again for another context length.")
    // Back to what it's loaded for, before the resize was sent: call that off.
    const previous = this.load?.previous ?? (this.ready ? this.load ?? undefined : undefined)
    if (this.load?.previous && this.load.previous.contextLength === contextLength) {
      this.cancelLoad(replaced)
      return this.load.promise
    }
    this.load?.reject(replaced)
    this.ready = false
    this.contextLength = contextLength
    const load: Load = { ...pending<void>(), contextLength, id: ++this.loads }
    this.load = load
    if (previous && this.busy) {
      // The worker resizes after the reply running anyway: send it then, and
      // until then it can be called off with the model staying as it is.
      load.previous = previous
      void this.idle().then(() => {
        if (this.load !== load) return
        delete load.previous
        this.sendLoad(load)
      })
    } else this.sendLoad(load)
    return load.promise
  }

  private sendLoad(load: Load) {
    if (this.gone) return
    this.open.add(load.id)
    this.post({ type: "load", model: this.model, loadId: load.id, ...(load.contextLength !== undefined && { contextLength: load.contextLength }) })
  }

  /**
   * Calls off a load not sent yet (a resize waiting for a reply to end): the
   * model stays loaded for the context it was. False if it was sent.
   */
  cancelLoad(reason: Error): boolean {
    const load = this.load
    if (!load?.previous || this.gone) return false
    this.load = load.previous
    this.contextLength = load.previous.contextLength
    this.ready = true
    load.reject(reason)
    return true
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

  private failLoad(load: Load, error: Error) {
    load.reject(error)
    this.load = null
    this.ready = false
  }

  /**
   * The current load, if a "ready" or "load-error" answers it. A worker that
   * names no load (one from before loadId) answers the current load only
   * when it's the one load open: with others open it may be a late answer
   * to one this replaced, so the load fails rather than guess.
   */
  private answered(loadId: number | undefined): Load | null {
    const load = this.load
    if (loadId !== undefined) {
      // The worker answers loads in order: the ones before are done too.
      for (const id of this.open) if (id <= loadId) this.open.delete(id)
      return load?.id === loadId ? load : null
    }
    if (!load || !this.open.has(load.id)) {
      if (this.open.size === 1) this.open.clear()
      return null
    }
    const only = this.open.size === 1
    this.open.clear()
    if (only) return load
    this.failLoad(load, new Error("The model's worker answered a load without naming it (loadId) while another was open."))
    return null
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
      case "stored":
        this.stored.resolve()
        break
      case "ready": {
        // (A late answer to a load this one replaced is ignored.)
        const load = this.answered(data.loadId)
        if (load) {
          this.ready = true
          this.loadedContextLength = data.contextLength
          load.resolve()
        }
        break
      }
      case "load-error": {
        const load = this.answered(data.loadId)
        if (load) this.failLoad(load, workerError(data.message, data.code))
        break
      }
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
