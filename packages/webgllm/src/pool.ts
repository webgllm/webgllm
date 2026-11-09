/**
 * The loaded models of one WebGLLM: a worker per model, as many as fit the
 * memory budget (the least recently used unloaded to make room, unless kept
 * loaded), and one queue for their generations. Only one generation runs
 * at a time across all models: they'd contend for the GPU, and frame
 * pacing assumes one.
 */
import { ModelWorker } from "./client.js"
import type { WorkerGenerationEnd } from "./client.js"
import { abortable, abortError, throwIfAborted, WebGLLMError } from "./errors.js"
import { monitorFrames } from "./frame-pacing.js"
import type { ModelStorage } from "./model-files.js"
import { MemoryBudget } from "./model-memory.js"
import type { LoadProgress, ModelUnloadReason, WebGLLMEventMap } from "./openai.js"
import type { WorkerRequest, WorkerResponse } from "./protocol.js"

export type LogLevel = "debug" | "info" | "warn" | "error" | "silent"

export type PoolOptions = {
  createWorker: () => Worker
  /** Bytes all loaded models may take together. */
  memoryBudget: () => number
  storage: ModelStorage
  /** Every load's and download's progress (the instance's listeners). */
  onProgress: (progress: LoadProgress) => void
  /** Models starting to load, loaded, failing and leaving memory (the instance's listeners). */
  onModelEvent?: (event: ModelEvent) => void
  /** The host's own budget, asked after webgllm's for each claim (see WebGLLMOptions). */
  onMemoryClaim?: MemoryClaimHook
  log: (level: Exclude<LogLevel, "silent">, text: string) => void
}

type ProgressListener = (progress: LoadProgress) => void

/** WebGLLMOptions.onMemoryClaim. */
export type MemoryClaimHook = (claim: { model: string; bytes: number; signal: AbortSignal }) => Promise<void> | void

/** A model lifecycle event: WebGLLMEventMap's, tagged with its type. */
export type ModelEvent = {
  [K in Exclude<keyof WebGLLMEventMap, "progress">]: { type: K } & WebGLLMEventMap[K]
}[Exclude<keyof WebGLLMEventMap, "progress">]

type Entry = {
  model: string
  worker: ModelWorker
  keepLoaded: boolean
  /** Callers waiting on its load: the last to abort stops the load. */
  waiters: number
  /** Per-call listeners of its load in progress. */
  listeners: Set<ProgressListener>
  /** Being unloaded to make room (once its generation ends). */
  evicting: Promise<void> | null
  /** The load "loadstart" was dispatched for (one per load, however many wait for it). */
  watched: Promise<void> | null
  /** Aborted when it's dropped: its claims for memory stop. */
  gone: AbortController
}

/** Getting a model's files into storage: one at a time per model. */
type Download = {
  /** Stops it, before its worker starts or after. */
  abort: AbortController
  /** Its own worker, once it needs one (not when a load or storage has the files). */
  worker: ModelWorker | null
  promise: Promise<void>
  waiters: number
  listeners: Set<ProgressListener>
}

type GenerateFields = Omit<
  Extract<WorkerRequest, { type: "generate" }>,
  "type" | "id" | "model" | "contextLength"
>

/** One task at a time, in the order they came. */
export class TaskQueue {
  private tail: Promise<void> = Promise.resolve()

  /** Runs `task` after the ones before it; one aborted while it waits is skipped. */
  run<T>(task: () => Promise<T>, signal?: AbortSignal): Promise<T> {
    const result = this.tail.then(() => {
      throwIfAborted(signal)
      return task()
    })
    this.tail = result.then(
      () => undefined,
      () => undefined
    )
    return result
  }
}

const unloaded = (model: string) =>
  new WebGLLMError("model-not-loaded", `${model} was unloaded.`)

let nextId = 0

export class ModelPool {
  private readonly entries = new Map<string, Entry>()
  private readonly downloads = new Map<string, Download>()
  private readonly budget: MemoryBudget
  private readonly queue = new TaskQueue()
  private stopFrames: (() => void) | undefined
  private askedToKeep = false
  /** Unloads asked for, by model and for every model: a load waiting meanwhile stops (see unloadMark). */
  private readonly unloads = new Map<string, number>()
  private unloadsOfAll = 0

  constructor(private readonly options: PoolOptions) {
    this.budget = new MemoryBudget(options.memoryBudget, (model, signal) => this.evict(model, signal))
  }

  /** Loaded and ready (not being unloaded). */
  isLoaded(model: string): boolean {
    const entry = this.entries.get(model)
    return !!entry && entry.worker.ready && !entry.evicting
  }

  /** The context a loaded model was asked for (undefined: its own). */
  requestedContext(model: string): number | undefined {
    return this.entries.get(model)?.worker.contextLength
  }

  /** The context a loaded model holds, as its worker reported it. */
  loadedContext(model: string): number | undefined {
    return this.isLoaded(model) ? this.entries.get(model)?.worker.loadedContextLength : undefined
  }

  loadedModels(): Array<string> {
    return [...this.entries.keys()].filter((model) => this.isLoaded(model))
  }

  /** Bytes the loaded models claim together. */
  get memoryUsed(): number {
    return this.budget.total
  }

  /**
   * Loads `model` for chats of up to `contextLength` tokens (its own when
   * undefined), downloading it the first time; loaded for another context,
   * it loads again.
   */
  async load(
    model: string,
    options: {
      contextLength?: number
      keepLoaded?: boolean
      onProgress?: ProgressListener
      signal?: AbortSignal
      /** unloadMark when the caller started waiting (ensureLoaded's own waits count too). */
      since?: number
    } = {}
  ): Promise<void> {
    const { contextLength, signal } = options
    const since = options.since ?? this.unloadMark(model)
    throwIfAborted(signal)
    // A download of it in another worker: let it finish (they'd write the same files).
    const download = this.downloads.get(model)
    if (download) {
      await abortable(download.promise.catch(() => undefined), signal)
      this.throwIfUnloaded(model, since)
    }
    let entry = this.entries.get(model)
    if (entry?.evicting) {
      await abortable(entry.evicting, signal)
      this.throwIfUnloaded(model, since)
      entry = this.entries.get(model)
    }
    this.throwIfUnloaded(model, since)
    entry ??= this.spawn(model)
    if (options.keepLoaded !== undefined) {
      entry.keepLoaded = options.keepLoaded
      this.budget.setKeepLoaded(model, options.keepLoaded)
    }
    if (entry.worker.ready && entry.worker.contextLength === contextLength) {
      this.budget.touch(model)
      return
    }
    const load = entry.worker.loadModel(contextLength)
    this.watch(entry, load, contextLength)
    await this.waitForLoad(entry, load, options.onProgress, signal)
  }

  /** Counts the unloads of `model` asked for so far: a load compares it before and after it waits. */
  unloadMark(model: string): number {
    return this.unloadsOfAll + (this.unloads.get(model) ?? 0)
  }

  /** Rejects as an unloaded model's load does if `model` was unloaded since `mark`. */
  private throwIfUnloaded(model: string, mark: number) {
    if (this.unloadMark(model) !== mark) throw unloaded(model)
  }

  /** Dispatches "loadstart" for a new load, and "load" when it's ready. */
  private watch(entry: Entry, load: Promise<void>, contextLength: number | undefined) {
    if (entry.watched === load) return
    entry.watched = load
    this.options.onModelEvent?.({ type: "loadstart", model: entry.model, ...(contextLength !== undefined && { contextLength }) })
    load.then(
      () => {
        if (this.entries.get(entry.model) !== entry || entry.watched !== load || !entry.worker.ready) return
        const held = entry.worker.loadedContextLength ?? contextLength ?? 0
        this.options.onModelEvent?.({ type: "load", model: entry.model, contextLength: held })
      },
      // Failures dispatch from drop(); a load superseded by another context has its own events.
      () => undefined
    )
  }

  /**
   * `model` loaded for whatever context it is (or loading for), else loaded
   * for `contextLength`: a generation's load.
   */
  async ensureLoaded(
    model: string,
    contextLength: number | undefined,
    onProgress: ProgressListener | undefined,
    signal: AbortSignal | undefined,
    /** unloadMark when the caller started waiting (before resolving the model, say). */
    since = this.unloadMark(model)
  ): Promise<Entry> {
    for (;;) {
      const entry = this.entries.get(model)
      if (entry?.evicting) {
        await abortable(entry.evicting, signal)
        this.throwIfUnloaded(model, since)
        continue
      }
      if (entry?.worker.ready) return entry
      const loading = entry?.worker.loadPromise
      if (!entry || !loading) {
        await this.load(model, { contextLength, onProgress, signal, since })
        continue
      }
      try {
        await this.waitForLoad(entry, loading, onProgress, signal)
      } catch (error) {
        // Superseded by a load for another context: wait for that one instead.
        const next = entry.worker.loadPromise
        if (signal?.aborted || !next || next === loading || entry.worker.terminated) throw error
      }
    }
  }

  private async waitForLoad(
    entry: Entry,
    load: Promise<void>,
    onProgress: ProgressListener | undefined,
    signal: AbortSignal | undefined
  ) {
    entry.waiters++
    if (onProgress) entry.listeners.add(onProgress)
    try {
      await abortable(load, signal)
      this.budget.touch(entry.model)
    } catch (error) {
      if (signal?.aborted) {
        // Nobody wants it any more: call off a resize not sent yet (the model
        // stays loaded as it was), else stop the load.
        if (entry.waiters === 1 && !entry.worker.ready && entry.worker.loadPromise === load) {
          if (entry.worker.cancelLoad(abortError(signal) as Error)) this.restored(entry)
          else this.drop(entry, abortError(signal) as Error, "aborted")
        }
      } else if (!entry.worker.ready && entry.worker.loadPromise === null) {
        // A failed load frees its worker (and what it claimed); the next try starts afresh.
        // (Not one superseded by a load for another context: that one goes on.)
        this.drop(entry, error as Error, "failed")
      }
      throw error
    } finally {
      entry.waiters--
      if (onProgress) entry.listeners.delete(onProgress)
    }
  }

  /** A resize called off: the "load" its "loadstart" waits for, at the context it holds. */
  private restored(entry: Entry) {
    entry.watched = entry.worker.loadPromise
    const contextLength = entry.worker.loadedContextLength ?? entry.worker.contextLength ?? 0
    this.options.onModelEvent?.({ type: "load", model: entry.model, contextLength })
  }

  /** Unloads `model`, or every model; their running generations fail with "model-not-loaded". */
  unload(model?: string) {
    // Loads still waiting (for a download or an eviction of it) have no
    // entry to drop: they see this and stop.
    if (model === undefined) this.unloadsOfAll++
    else this.unloads.set(model, (this.unloads.get(model) ?? 0) + 1)
    for (const entry of [...this.entries.values()]) {
      if (model === undefined || entry.model === model) this.drop(entry, unloaded(entry.model), "unloaded")
    }
  }

  /**
   * Downloads `model`'s files to storage without loading it: settles once
   * they're there, whatever a load of it meanwhile does on the GPU.
   */
  async download(
    model: string,
    options: { onProgress?: ProgressListener; signal?: AbortSignal } = {}
  ): Promise<void> {
    const { onProgress, signal } = options
    throwIfAborted(signal)
    // Loaded means downloaded.
    if (this.entries.get(model)?.worker.ready) return
    let download = this.downloads.get(model)
    if (!download) {
      // Registered at once, so a load starting meanwhile waits for it.
      const created: Download = {
        abort: new AbortController(),
        worker: null,
        promise: Promise.resolve(),
        waiters: 0,
        listeners: new Set()
      }
      created.promise = this.fetchFiles(model, created).finally(() => {
        created.worker?.terminate(new Error("The download ended."))
        if (this.downloads.get(model) === created) this.downloads.delete(model)
      })
      created.promise.catch(() => undefined)
      download = created
      this.downloads.set(model, download)
    }
    download.waiters++
    if (onProgress) download.listeners.add(onProgress)
    try {
      await abortable(download.promise, signal)
    } catch (error) {
      // The last caller gave up: stop it (storage keeps what came, to resume from).
      if (signal?.aborted && download.waiters === 1) this.stopDownload(model, download, abortError(signal) as Error)
      throw error
    } finally {
      download.waiters--
      if (onProgress) download.listeners.delete(onProgress)
    }
  }

  /**
   * Gets `model`'s files into storage: a load in progress fetches them (done
   * once its worker says they're stored or the load ends; its GPU doesn't
   * matter), else storage has them, else a worker of the download's own
   * fetches them.
   */
  private async fetchFiles(model: string, download: Download): Promise<void> {
    const { signal } = download.abort
    const entry = this.entries.get(model)
    const loading = entry?.worker.loadPromise
    if (entry && loading) {
      const stored = entry.worker.storedPromise
      // Its downloading, as a download's 0–1.
      const forward = (progress: LoadProgress) => {
        if (progress.phase !== "download") return
        const share = progress.totalBytes ? (progress.loadedBytes ?? 0) / progress.totalBytes : progress.progress / 0.9
        for (const listener of download.listeners) listener({ ...progress, progress: Math.min(Math.max(share, 0), 1) })
      }
      entry.listeners.add(forward)
      try {
        await abortable(Promise.race([loading, stored]).catch(() => undefined), signal)
      } finally {
        entry.listeners.delete(forward)
      }
    }
    if (await abortable(this.options.storage.downloaded(model), signal)) return
    // Canceled as the check came back: there was no worker to stop yet.
    throwIfAborted(signal)
    const worker = new ModelWorker(model, this.options.createWorker, {
      onProgress: (data) => this.progress(downloadProgress(data), download.listeners),
      onClaim: () => Promise.resolve(),
      onLog: (text) => this.options.log("info", text),
      onCrash: () => {
        if (this.downloads.get(model) === download) this.downloads.delete(model)
      }
    })
    download.worker = worker
    await worker.download()
  }

  private stopDownload(model: string, download: Download, reason: Error) {
    download.abort.abort(reason)
    download.worker?.terminate(reason)
    if (this.downloads.get(model) === download) this.downloads.delete(model)
  }

  /** Stops a download of `model` in progress, if there is one. */
  cancelDownload(model: string) {
    const download = this.downloads.get(model)
    if (download) this.stopDownload(model, download, abortError() as Error)
  }

  /**
   * Runs one generation on `model`, loading it first if it isn't loaded
   * (for `contextLength`), after the generations queued before it. Settles
   * when the worker ends it; aborting `signal` stops it.
   */
  async generate(
    model: string,
    fields: GenerateFields,
    options: {
      contextLength?: number
      onToken: (text: string) => void
      onProgress?: ProgressListener
      signal?: AbortSignal
    }
  ): Promise<WorkerGenerationEnd> {
    const { signal } = options
    // Load outside the queue, so other models generate meanwhile.
    await this.ensureLoaded(model, options.contextLength, options.onProgress, signal)
    return this.queue.run(async () => {
      // (Unloaded while it waited: load it again.)
      const entry = await this.ensureLoaded(model, options.contextLength, options.onProgress, signal)
      throwIfAborted(signal)
      this.budget.touch(model)
      const id = `gen_${(nextId++).toString(36)}`
      const ended = entry.worker.generate({ id, ...fields }, options.onToken)
      const stop = () => entry.worker.stop(id)
      signal?.addEventListener("abort", stop, { once: true })
      try {
        return await ended
      } finally {
        signal?.removeEventListener("abort", stop)
      }
    }, signal)
  }

  private spawn(model: string): Entry {
    const entry: Entry = {
      model,
      keepLoaded: false,
      waiters: 0,
      listeners: new Set(),
      evicting: null,
      watched: null,
      gone: new AbortController(),
      worker: new ModelWorker(model, this.options.createWorker, {
        onProgress: (data) => this.progress(loadProgress(data), entry.listeners),
        onClaim: async (bytes) => {
          if (this.entries.get(model) !== entry) return
          await this.budget.claim(model, bytes, entry.keepLoaded, entry.gone.signal)
          // (Unloaded while it claimed: what it claimed isn't held.)
          if (this.entries.get(model) !== entry) {
            if (!this.entries.has(model)) this.budget.release(model)
            return
          }
          await this.hostClaim(entry, bytes)
        },
        onLog: (text) => this.options.log("info", text),
        onCrash: (error) => {
          this.options.log("error", `${model}'s worker stopped: ${error.message}`)
          this.drop(entry, error, "crashed")
        }
      })
    }
    this.entries.set(model, entry)
    this.stopFrames ??= monitorFrames((intervalMs, expectedMs, frames, startedAt, meanMs) => {
      const message = { type: "frames", intervalMs, expectedMs, frames, startedAt, meanMs } as const
      for (const { worker } of this.entries.values()) worker.post(message)
    })
    return entry
  }

  /**
   * Asks the host's budget for `entry`'s claim, after webgllm's own and
   * outside its queue of claims, so a host making room by unloading another
   * model can't wait on a claim in progress. A refusal fails the model (its
   * load or the reply whose cache grew) and frees what webgllm's budget held
   * for it; unloading it stops the wait.
   */
  private async hostClaim(entry: Entry, bytes: number) {
    const hook = this.options.onMemoryClaim
    if (!hook) return
    const { signal } = entry.gone
    try {
      await abortable(Promise.resolve().then(() => hook({ model: entry.model, bytes, signal })), signal)
    } catch (error) {
      if (signal.aborted) return
      const reason = error instanceof Error ? error.message : String(error)
      this.drop(entry, new WebGLLMError("out-of-memory", `${entry.model}'s memory claim was refused: ${reason}`, { cause: error }), "failed")
    }
  }

  /** Ends `entry`'s worker, freeing its model and memory; what waits on it fails with `reason`. */
  private drop(entry: Entry, reason: Error, why: ModelUnloadReason) {
    entry.worker.terminate(reason)
    entry.gone.abort(reason)
    if (this.entries.get(entry.model) !== entry) return
    this.entries.delete(entry.model)
    this.budget.release(entry.model)
    if (why === "failed" || why === "crashed") this.options.onModelEvent?.({ type: "error", model: entry.model, error: reason })
    this.options.onModelEvent?.({ type: "unload", model: entry.model, reason: why })
    if (this.entries.size === 0) {
      this.stopFrames?.()
      this.stopFrames = undefined
    }
  }

  /**
   * Unloads `model` to make room for another: at once, or when its running
   * generation ends (queued ones load it again when their turn comes).
   * Resolves false, the model staying, if `signal` (the claimant's) aborts
   * before then.
   */
  private evict(model: string, signal?: AbortSignal): Promise<boolean> {
    const entry = this.entries.get(model)
    if (!entry) return Promise.resolve(true)
    if (entry.evicting) return entry.evicting.then(() => this.entries.get(model) !== entry)
    const evicting = abortable(entry.worker.idle(), signal).then(
      () => true,
      () => false
    ).then((go) => {
      if (this.entries.get(model) !== entry) return true
      // The claimant went away while it waited: the model stays.
      if (!go || signal?.aborted) {
        if (entry.evicting === done) entry.evicting = null
        return false
      }
      this.options.log("info", `Unloaded ${model} to make room for another model.`)
      this.drop(entry, unloaded(model), "evicted")
      return true
    })
    const done = evicting.then(() => undefined)
    entry.evicting = done
    return evicting
  }

  private progress(progress: LoadProgress, listeners: Set<ProgressListener>) {
    if (!progress.cached) this.keepDownloads()
    this.options.onProgress(progress)
    for (const listener of listeners) listener(progress)
  }

  /**
   * Asks the browser to keep this site's storage, so a downloaded model isn't
   * deleted to free space: once, when a model first downloads. Not in
   * Firefox, which asks the user with a permission prompt. Browsers may say
   * no; only pages can ask.
   */
  private keepDownloads() {
    if (this.askedToKeep || typeof navigator === "undefined") return
    this.askedToKeep = true
    if (/Firefox\//.test(navigator.userAgent)) return
    navigator.storage?.persist?.().catch(() => undefined)
  }
}

type WorkerProgress = Extract<WorkerResponse, { type: "progress" }>

const bytes = (data: WorkerProgress) =>
  data.totalBytes !== undefined ? { loadedBytes: data.loadedBytes ?? 0, totalBytes: data.totalBytes } : {}

/**
 * A load's progress as the worker counts it: downloading takes 0–0.9 and
 * reading into the GPU the rest; from storage, reading takes 0–1.
 */
export function loadProgress(data: WorkerProgress): LoadProgress {
  return {
    model: data.model,
    phase: data.cached || data.progress >= 0.9 ? "load" : "download",
    progress: data.progress,
    cached: data.cached,
    ...bytes(data)
  }
}

/** A download's progress across 0–1 (as a "download" request counts it): its bytes when known. */
export function downloadProgress(data: WorkerProgress): LoadProgress {
  const progress =
    data.totalBytes && data.loadedBytes !== undefined
      ? data.loadedBytes / data.totalBytes
      : data.progress
  return {
    model: data.model,
    phase: "download",
    progress: Math.min(Math.max(progress, 0), 1),
    cached: data.cached,
    ...bytes(data)
  }
}
