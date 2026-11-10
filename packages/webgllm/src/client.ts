/**
 * The page's side of one model in a WebGL 2 worker (handler.ts), which runs
 * it off the main thread on an OffscreenCanvas. One worker holds one model;
 * the pool (pool.ts) starts a worker per loaded model, or keeps its one
 * model in the worker the caller gave it.
 */
import { workerError } from "./errors.js"
import { MIN_TEXTURE_SIZE } from "./limits.js"
import { isCatalogModel, knownModelMeta } from "./model-meta.js"
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

/** The bundled worker. Written out here so bundlers see the `new URL(…, import.meta.url)`. */
export const defaultWorker = () =>
  new Worker(new URL("./worker.js", import.meta.url), { type: "module", name: "webgllm" })

/**
 * Ids of loads, unloads and downloads: one sequence for the page, so no
 * two attachments to a worker (a model unloaded and loaded again in the
 * caller's worker) ever share one, and a late answer can't pass for a new
 * request's.
 */
let ids = 0

const isResponse = (data: unknown): data is WorkerResponse =>
  typeof data === "object" && data !== null && typeof (data as { type?: unknown }).type === "string"

export class ModelWorker {
  private load: Load | null = null
  /** Loads sent and not answered yet. */
  private readonly open = new Set<number>()
  private downloading: (Pending<void> & { id: string }) | null = null
  /** What a worker of the caller's still owes once this is gone ("unloaded", a stopped download's end), by id. */
  private readonly owed = new Map<number | string, () => void>()
  private freed: Promise<void> | undefined
  private readonly stored = pending<void>()
  private readonly generations = new Map<string, Generation>()
  /** Loaded and ready, for the context it was asked for (see contextLength). */
  ready = false
  /** Tokens the loaded model holds, as the worker reported it. */
  loadedContextLength: number | undefined
  /** The context the model was asked for: undefined for the model's own. */
  contextLength: number | undefined
  private gone: Error | null = null

  private readonly onMessage = (event: MessageEvent<unknown>) => this.handle(event.data)
  private readonly onError = (event: ErrorEvent) => {
    // (Gone already: a worker that fails owes nothing more.)
    if (this.gone) {
      for (const settle of [...this.owed.values()]) settle()
      return
    }
    const error = new Error(event.message || "The model's worker stopped.")
    this.terminate(error)
    for (const settle of [...this.owed.values()]) settle()
    this.events.onCrash(error)
  }

  /**
   * `owned`: a worker started for this model, ended with it. Otherwise it's
   * the caller's: it's left running (asked to free the model), its other
   * listeners alone, and it may carry another model's messages too.
   */
  constructor(
    readonly model: string,
    private readonly worker: Worker,
    private readonly owned: boolean,
    private readonly events: WorkerEvents
  ) {
    worker.addEventListener("message", this.onMessage)
    worker.addEventListener("error", this.onError)
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
    const load: Load = { ...pending<void>(), contextLength, id: ++ids }
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
    this.post({ type: "load", model: this.model, loadId: load.id, ...(load.contextLength !== undefined && { contextLength: load.contextLength }), ...this.meta() })
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

  /** A model resolved from the Hub: its meta, so the worker needn't ask the Hub again. */
  private meta() {
    const meta = isCatalogModel(this.model) ? undefined : knownModelMeta(this.model)
    return meta ? { meta } : {}
  }

  /** Downloads the model's files to storage, without loading it. */
  download(): Promise<void> {
    if (this.gone) return Promise.reject(this.gone)
    if (!this.downloading) {
      const id = `dl_${(++ids).toString(36)}`
      this.downloading = { ...pending<void>(), id }
      this.post({ type: "download", id, model: this.model, ...this.meta() })
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

  /**
   * Frees the model and all its GPU memory: by ending the worker, or in a
   * worker of the caller's by stopping its generations and its download
   * and asking the worker to free the model, which leaves it running.
   * What waits on it fails with `reason` at once. Returns a promise for
   * the caller's worker having freed the model and closed the download's
   * files; undefined when that's done already (its own worker is ended).
   */
  terminate(reason: Error): Promise<void> | undefined {
    if (this.gone) return this.freed
    const detach = () => {
      this.worker.removeEventListener("message", this.onMessage)
      this.worker.removeEventListener("error", this.onError)
    }
    if (this.owned) {
      detach()
      this.worker.terminate()
    } else {
      const waits: Array<Promise<void>> = []
      const owe = (id: number | string) =>
        waits.push(new Promise<void>((resolve) => this.owed.set(id, () => (this.owed.delete(id), resolve()))))
      for (const id of this.generations.keys()) this.post({ type: "stop", id })
      // (Only if it was asked to load: a download leaves the worker's model alone.)
      if (this.load) {
        const id = ++ids
        this.post({ type: "unload", model: this.model, id })
        owe(id)
      }
      if (this.downloading) {
        this.post({ type: "stop", id: this.downloading.id })
        owe(this.downloading.id)
      }
      if (waits.length) this.freed = Promise.all(waits).then(detach)
      else detach()
    }
    this.gone = reason
    this.ready = false
    this.load?.reject(reason)
    this.load = null
    this.downloading?.reject(reason)
    this.downloading = null
    for (const generation of this.generations.values()) generation.reject(reason)
    this.generations.clear()
    this.settled()
    return this.freed
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

  private handle(data: unknown) {
    // A worker of the caller's carries its own messages too, and those of a download of another model.
    if (!isResponse(data)) return
    if (this.gone) {
      if (data.type === "unloaded" || data.type === "downloaded" || data.type === "download-error") this.owed.get(data.id)?.()
      return
    }
    if (!this.owned && "model" in data && data.model !== this.model) return
    switch (data.type) {
      case "progress":
        this.events.onProgress(data)
        break
      case "claim":
        void this.events
          .onClaim(data.bytes)
          .catch(() => undefined)
          .then(() => this.post({ type: "claimed", id: data.id }))
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
        if (data.id !== this.downloading?.id) break
        this.downloading.resolve()
        this.downloading = null
        break
      case "download-error":
        if (data.id !== this.downloading?.id) break
        this.downloading.reject(workerError(data.message, data.code))
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
      case "unloaded":
        break
    }
  }
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
