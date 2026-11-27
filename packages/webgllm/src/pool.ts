/**
 * The loaded models of one WebGLLM: a worker per model, as many as fit the
 * memory budget (the least recently used unloaded to make room, unless kept
 * loaded), and one queue for their generations. Given a worker of the
 * caller's, it keeps one model, in that worker: loading another replaces it. Only one generation runs
 * at a time across all models: they'd contend for the GPU, and frame
 * pacing assumes one.
 */
import { ModelWorker } from "./client.js"
import type { OnToken, WorkerEvents, WorkerGenerationEnd } from "./client.js"
import { abortable, abortError, failedToLoad, throwIfAborted, WebGLLMError, workerFailure } from "./errors.js"
import { monitorFrames } from "./frame-pacing.js"
import type { ModelStorage } from "./model-files.js"
import { MemoryBudget } from "./model-memory.js"
import type { LoadProgress, ModelUnloadReason, WebGLLMEventMap } from "./openai.js"
import type { WorkerRequest, WorkerResponse } from "./protocol.js"
import type { Sampling } from "./sampler.js"

export type LogLevel = "debug" | "info" | "warn" | "error" | "silent"

export type PoolOptions = {
  /** Starts a worker for a model (not used with `worker`). */
  createWorker: () => Worker
  /** The caller's worker (WebGLLMOptions.worker): every model runs in it, one at a time, and it's never ended. */
  worker?: Worker
  /** Bytes all loaded models may take together. */
  memoryBudget: () => number
  /** WebGLLMOptions.webgpu: false loads on WebGL 2 even where there's WebGPU. */
  webgpu?: "auto" | false
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
  /** Once dropped: settles when its memory is free and "unload" says so (see drop). */
  dropped?: Promise<void>
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

/** How long a spare worker of webgllm's waits for the next load (see ModelPool.spare). */
export const SPARE_MS = 30_000

export class ModelPool {
  private readonly entries = new Map<string, Entry>()
  private readonly downloads = new Map<string, Download>()
  private readonly budget: MemoryBudget
  private readonly queue = new TaskQueue()
  private stopFrames: (() => void) | undefined
  /** Whether the browser agreed, when first asked, to keep this site's storage (see keepDownloads). */
  private keeping: Promise<boolean> | undefined
  /** Unloads asked for, by model and for every model: a load waiting meanwhile stops (see unloadMark). */
  private readonly unloads = new Map<string, number>()
  private unloadsOfAll = 0
  /** Told of each unload: waits that end when their model is unloaded (see unlessUnloaded). */
  private readonly unloadListeners = new Set<() => void>()
  /** Models dropped from the caller's worker that it hasn't freed yet: their memory is held, and no load starts, until it has. */
  private readonly freeing = new Set<Promise<void>>()
  /** Downloads stopped in the caller's worker whose files aren't closed yet, by model. */
  private readonly closing = new Map<string, Promise<void>>()
  /**
   * A worker of webgllm's started when a model was unloaded, for the next
   * load (see startSpare): the unloaded model's own worker is ended, as only
   * that frees its memory in Safari on an iPhone at once (a worker that
   * frees a model keeps its memory until its collector runs, which an idle
   * one never does), and a worker started ahead loads as soon as one that's
   * reused (LFM3 1.7B: 1.0 s; 1.3 to 1.5 s in one started for the load).
   * Ended once idle for SPARE_MS.
   */
  private spare: Worker | null = null
  private spareTimer: ReturnType<typeof setTimeout> | undefined
  /** Models dropped whose memory isn't free yet, by model: settles once it is (see drop, evict). */
  private readonly releasing = new Map<string, Promise<void>>()
  /** The caller's worker's completions, one at a time from load to last token (see turn). */
  private turns: Promise<void> = Promise.resolve()

  /**
   * Why the caller's worker can't run anything, once it has said its script
   * failed to load: every request rejects with it, at once.
   */
  private dead: WebGLLMError | null = null

  constructor(private readonly options: PoolOptions) {
    this.budget = new MemoryBudget(options.memoryBudget, (model, signal) => {
      const ext = this.external.get(model)
      return ext ? ext.unload().then(() => true) : this.evict(model, signal)
    })
    // The caller's worker may fail before anything is asked of it: heard from
    // now on, as nothing says later that it had. (A worker that failed before
    // this can't be told from one that's slow to answer.) Its models' own
    // listeners, added after this one, fail what waits on them.
    options.worker?.addEventListener("error", (event: ErrorEvent) => {
      if (failedToLoad(event)) this.dead ??= workerFailure(event)
    })
  }

  /** Rejects as every request to a worker that failed to load does. */
  private throwIfDead() {
    if (this.dead) throw this.dead
  }

  /** Loaded, loading or downloading (or a stopped download's files still closing): its files are in use. */
  inUse(model: string): boolean {
    return this.entries.has(model) || this.downloads.has(model) || this.closing.has(model)
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
   * Models outside the pool sharing its budget (image models: images.ts):
   * what unloads each to make room, the bytes its last claim to go through
   * admitted, whether it's kept loaded, and its claims, one at a time.
   */
  private readonly external = new Map<string, { unload: () => Promise<void>; admitted: number | undefined; keepLoaded: boolean; claims: Promise<void> }>()

  /**
   * Claims `bytes` for a model outside the pool (all it will hold, in
   * place of its last claim), after its claims before: room in the budget
   * first, unloading the least recently used models (text or image), then
   * the host's onMemoryClaim. `unload` frees it when another claim needs
   * its room. Rejects if the host refuses or `signal` aborts, and the model
   * then holds what its last claim admitted (its retention and last use as
   * they are).
   */
  async claimExternal(model: string, bytes: number, unload: () => Promise<void>, keepLoaded: boolean, signal?: AbortSignal): Promise<void> {
    let ext = this.external.get(model)
    if (!ext) this.external.set(model, (ext = { unload, admitted: undefined, keepLoaded, claims: Promise.resolve() }))
    const own = ext
    own.unload = unload
    own.keepLoaded = keepLoaded
    const run = own.claims.then(async () => {
      throwIfAborted(signal)
      // (Unloaded while it waited: it holds nothing to claim for.)
      if (this.external.get(model) !== own) throw new WebGLLMError("model-not-loaded", `${model} was unloaded.`)
      try {
        await this.budget.claim(model, bytes, own.keepLoaded, signal)
        // (Set while it waited for room, when the budget held no entry to set it on.)
        this.budget.setKeepLoaded(model, own.keepLoaded)
        const hook = this.options.onMemoryClaim
        if (hook) await abortable(Promise.resolve().then(() => hook({ model, bytes, signal: signal ?? new AbortController().signal })), signal)
        own.admitted = bytes
      } catch (error) {
        if (this.external.get(model) === own) {
          if (own.admitted === undefined) this.budget.release(model)
          else {
            this.budget.setBytes(model, own.admitted, own.keepLoaded)
            this.budget.setKeepLoaded(model, own.keepLoaded)
          }
        }
        throw error
      }
    })
    own.claims = run.catch(() => undefined)
    return run
  }

  /** Whether a model outside the pool may be unloaded to make room. */
  setExternalKeepLoaded(model: string, keep: boolean) {
    const ext = this.external.get(model)
    if (ext) ext.keepLoaded = keep
    this.budget.setKeepLoaded(model, keep)
  }

  /** Marks a model outside the pool as just used (least recently used order). */
  touchExternal(model: string) {
    this.budget.touch(model)
  }

  /** Forgets a model outside the pool: it's unloaded, or failed to load. */
  releaseExternal(model: string) {
    this.external.delete(model)
    this.budget.release(model)
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
    const since = options.since ?? this.unloadMark(model)
    await this.withStorageRetry(() => this.loadOnce(model, { ...options, since }), options.signal, { model, since })
  }

  private async loadOnce(
    model: string,
    options: Parameters<ModelPool["load"]>[1] & { since: number }
  ): Promise<void> {
    const { contextLength, signal, since } = options
    throwIfAborted(signal)
    this.throwIfDead()
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
    // The caller's worker holds one model: the one there leaves first, as one
    // evicted for room does (kept loaded or not), and the last load asked for wins.
    while (!entry && this.options.worker && (this.entries.size || this.freeing.size)) {
      for (const other of [...this.entries.values()]) await abortable(this.evict(other.model, signal), signal)
      // (Evicted or unloaded: until the worker has freed it.)
      await abortable(Promise.all(this.freeing), signal)
      throwIfAborted(signal)
      this.throwIfUnloaded(model, since)
      entry = this.entries.get(model)
    }
    // A model still being freed in a worker of webgllm's: its worker is the one
    // to reuse, and its memory goes before the next model's comes.
    if (!entry && !this.options.worker && this.freeing.size) {
      await abortable(Promise.all(this.freeing), signal)
      throwIfAborted(signal)
      this.throwIfUnloaded(model, since)
      entry = this.entries.get(model)
    }
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

  /** `wait`, unless `model` is unloaded first (since `mark`): then rejects as an unloaded model's load does. */
  private unlessUnloaded<T>(wait: Promise<T>, model: string, mark: number): Promise<T> {
    if (this.unloadMark(model) !== mark) {
      // (`wait` is this call's: its rejection is handled here, not left unhandled.)
      wait.catch(() => undefined)
      throw unloaded(model)
    }
    return new Promise<T>((resolve, reject) => {
      const check = () => {
        if (this.unloadMark(model) === mark) return
        this.unloadListeners.delete(check)
        reject(unloaded(model))
      }
      this.unloadListeners.add(check)
      wait.then(
        (value) => {
          this.unloadListeners.delete(check)
          resolve(value)
        },
        (error: unknown) => {
          this.unloadListeners.delete(check)
          reject(error)
        }
      )
    })
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

  /**
   * Unloads `model`, or every model; their running generations fail with
   * "model-not-loaded" at once. Resolves once their memory is free: at
   * once for webgllm's own workers (they're ended), and for the caller's
   * worker when it says it has freed the model (see collectReleased in
   * handler.ts).
   */
  async unload(model?: string): Promise<void> {
    // Loads still waiting (for a download or an eviction of it) have no
    // entry to drop: they see this and stop.
    if (model === undefined) this.unloadsOfAll++
    else this.unloads.set(model, (this.unloads.get(model) ?? 0) + 1)
    for (const listener of [...this.unloadListeners]) listener()
    for (const entry of [...this.entries.values()]) {
      if (model === undefined || entry.model === model) this.drop(entry, unloaded(entry.model), "unloaded")
    }
    // (One being freed already, by an earlier unload or an eviction, too.)
    const freeing = [...this.freeing]
    await Promise.all(freeing)
    // The worker failed to load while it was asked to free them: it never did (it never held them either).
    if (freeing.length) this.throwIfDead()
  }

  /**
   * A completion's turn in the caller's worker, which holds one model: from
   * before its load to its last token, so a completion for another model
   * can't replace this one's between its load and its generation. Resolves
   * to what ends the turn. With webgllm's own workers (a model each) there
   * are no turns.
   */
  async turn(signal?: AbortSignal): Promise<() => void> {
    if (!this.options.worker) return () => undefined
    const before = this.turns
    let end!: () => void
    this.turns = new Promise<void>((resolve) => (end = resolve))
    try {
      await abortable(before, signal)
    } catch (error) {
      void before.then(end)
      throw error
    }
    return end
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
    this.throwIfDead()
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
      created.promise = this.withStorageRetry(() => this.fetchFiles(model, created), created.abort.signal).finally(() => {
        this.endDownload(model, created, new Error("The download ended."))
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
    // (One stopped just before, in the caller's worker: once its files are closed.)
    await abortable(this.closing.get(model) ?? Promise.resolve(), signal)
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
    // (A retry: the failed attempt's worker freed first, and, the caller's,
    // its files closed before the same worker writes them again.)
    if (download.worker) {
      this.releaseDownloadWorker(model, download.worker, new Error("The download is tried again."))
      download.worker = null
      await abortable(this.closing.get(model) ?? Promise.resolve(), signal)
    }
    const worker = this.modelWorker(model, null, {
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
    this.endDownload(model, download, reason)
  }

  /** Frees a download's worker (the caller's: stops the download there, and waits for its files to close). */
  private endDownload(model: string, download: Download, reason: Error) {
    if (this.downloads.get(model) === download) this.downloads.delete(model)
    this.releaseDownloadWorker(model, download.worker, reason)
  }

  /** Frees one download attempt's worker: webgllm's own is ended; the caller's is stopped, and its files' closing tracked (see closing). */
  private releaseDownloadWorker(model: string, worker: ModelWorker | null, reason: Error) {
    const closed = worker?.terminate(reason)
    if (!closed) return
    const both = Promise.all([this.closing.get(model), closed]).then(() => {
      if (this.closing.get(model) === both) this.closing.delete(model)
    })
    this.closing.set(model, both)
  }

  /**
   * Stops a download of `model` in progress, if there is one. Resolves once
   * nothing writes its files any more (at once with webgllm's own workers,
   * which are ended).
   */
  async cancelDownload(model: string): Promise<void> {
    const download = this.downloads.get(model)
    if (download) this.stopDownload(model, download, abortError() as Error)
    await this.closing.get(model)
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
      onToken: OnToken
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

  /**
   * `model`'s side of a worker: the caller's, left running when the model
   * goes; else one of webgllm's (`own`, or one started now), ended with it.
   */
  private modelWorker(model: string, own: Worker | null, events: WorkerEvents): ModelWorker {
    const worker = this.options.worker ?? own ?? this.options.createWorker()
    return new ModelWorker(model, worker, !this.options.worker, events, this.options.webgpu ?? "auto")
  }

  /** The spare worker, taken for a new model (see spare). */
  private takeSpare(): Worker | null {
    const worker = this.spare
    this.spare = null
    clearTimeout(this.spareTimer)
    worker?.removeEventListener("error", this.spareError)
    return worker
  }

  /**
   * Starts a spare worker for the next load, unless there's one (see spare).
   * Never throws: a worker that can't be started now is logged, and the
   * next load starts its own.
   */
  private startSpare() {
    if (this.spare || this.options.worker) return
    let worker: Worker
    try {
      worker = this.options.createWorker()
    } catch (error) {
      this.options.log("warn", `A spare worker didn't start: ${error instanceof Error ? error.message : String(error)}`)
      return
    }
    this.spare = worker
    worker.addEventListener("error", this.spareError)
    this.spareTimer = setTimeout(() => this.takeSpare()?.terminate(), SPARE_MS)
    // (Node: a pending timer doesn't keep the process alive.)
    ;(this.spareTimer as { unref?: () => void }).unref?.()
  }

  /** A spare worker that fails is ended, not used (it listens only while it's the spare). */
  private readonly spareError = () => this.takeSpare()?.terminate()

  /**
   * WebLLM's forwardTokensAndSample on `model` (loaded first if it isn't):
   * in the queue with the generations, as it reads the same cache.
   */
  async forward(
    model: string,
    inputIds: Array<number>,
    sampling: Sampling,
    /** `since`: unloadMark when the caller asked (an unload since then stops it, loaded or not, however long it waited). */
    options: { contextLength?: number; signal?: AbortSignal; since?: number } = {}
  ): Promise<number> {
    const { signal } = options
    const since = options.since ?? this.unloadMark(model)
    await this.ensureLoaded(model, options.contextLength, undefined, signal, since)
    return this.queue.run(async () => {
      // (Unloaded while it waited in the queue: it doesn't load the model again.)
      const entry = await this.ensureLoaded(model, options.contextLength, undefined, signal, since)
      throwIfAborted(signal)
      this.budget.touch(model)
      return entry.worker.forward(`fwd_${(nextId++).toString(36)}`, inputIds, sampling)
    }, signal)
  }

  /** WebLLM's resetChat: `model` forgets its cache, after the generations queued before (nothing to forget if it isn't loaded). */
  reset(model: string): Promise<void> {
    return this.queue.run(async () => {
      this.throwIfDead()
      const entry = this.entries.get(model)
      if (entry?.worker.ready && !entry.evicting) await entry.worker.reset(`rst_${(nextId++).toString(36)}`)
    })
  }

  private spawn(model: string): Entry {
    // A worker of webgllm's own: the spare, else a new one.
    const own = this.options.worker ? null : this.takeSpare()
    const entry: Entry = {
      model,
      keepLoaded: false,
      waiters: 0,
      listeners: new Set(),
      evicting: null,
      watched: null,
      gone: new AbortController(),
      worker: this.modelWorker(model, own, {
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

  /**
   * Drops `entry`: its model's worker stops and frees it. Once only (a load
   * failing after an unload asked for it drops it again): the first
   * reason stands. Settles when the memory is free and "unload" says so.
   */
  private drop(entry: Entry, reason: Error, why: ModelUnloadReason): Promise<void> {
    if (entry.dropped) return entry.dropped
    // A worker of webgllm's is ended (its memory is free at once); the
    // caller's worker frees the model (see collectReleased in handler.ts).
    const freed = entry.worker.terminate(reason)
    entry.gone.abort(reason)
    if (this.entries.get(entry.model) !== entry) return (entry.dropped = freed ?? Promise.resolve())
    this.entries.delete(entry.model)
    if (why === "failed" || why === "crashed") this.options.onModelEvent?.({ type: "error", model: entry.model, error: reason })
    // Its memory is free, and "unload" says so, once it is: the caller's
    // worker frees the model after the generation running there has stopped.
    const release = () => {
      this.budget.release(entry.model)
      this.options.onModelEvent?.({ type: "unload", model: entry.model, reason: why })
    }
    let done = Promise.resolve()
    if (freed) {
      done = freed.then(() => {
        this.freeing.delete(done)
        if (this.releasing.get(entry.model) === done) this.releasing.delete(entry.model)
        release()
      })
      this.freeing.add(done)
      this.releasing.set(entry.model, done)
    } else release()
    entry.dropped = done
    if (this.entries.size === 0) {
      this.stopFrames?.()
      this.stopFrames = undefined
    }
    // A spare for the next load, if the caller unloaded it: once all the above is done.
    if (why === "unloaded") this.startSpare()
    return done
  }

  /**
   * Unloads `model` to make room for another: at once, or when its running
   * generation ends (queued ones load it again when their turn comes).
   * Resolves false, the model staying, if `signal` (the claimant's) aborts
   * before then.
   */
  private evict(model: string, signal?: AbortSignal): Promise<boolean> {
    const entry = this.entries.get(model)
    // Gone already (an unload asked for it): done once its memory is free.
    if (!entry) return (this.releasing.get(model) ?? Promise.resolve()).then(() => true)
    if (entry.evicting) return entry.evicting.then(() => this.entries.get(model) !== entry)
    const evicting = abortable(entry.worker.idle(), signal).then(
      () => true,
      () => false
    ).then((go) => {
      // Dropped meanwhile (an unload asked for it): done once its memory is free.
      if (this.entries.get(model) !== entry) return (entry.dropped ?? this.releasing.get(model) ?? Promise.resolve()).then(() => true)
      // The claimant went away while it waited: the model stays.
      if (!go || signal?.aborted) {
        if (entry.evicting === done) entry.evicting = null
        return false
      }
      this.options.log("info", `Unloaded ${model} to make room for another model.`)
      // (Done once its memory is free: the claim waiting on this counts it out then.)
      return this.drop(entry, unloaded(model), "evicted").then(() => true)
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
   * deleted to free space: once, when a model first downloads. Some browsers
   * also give a site more room once it's kept (Firefox for Android stores
   * about 360 MB for a site otherwise, less than a model), and some ask the
   * user first (Firefox, with a permission prompt). Browsers may say no;
   * only pages can ask.
   */
  private keepDownloads() {
    if (this.keeping || typeof navigator === "undefined") return
    const storage = navigator.storage
    this.keeping = (async () => {
      if (!storage?.persist || (await storage.persisted?.())) return false
      return storage.persist()
    })().catch(() => false)
  }

  /**
   * Runs `attempt`, and once more if it ran out of room and the browser has
   * since agreed to keep this site's storage (keepDownloads): a user can
   * answer the prompt after a download has filled the room it had. The
   * retry resumes the download where it stopped. `signal` (the load's, or
   * the download's own) ends the wait for that answer, as does unloading
   * a load's model (`load`: the model, and its unloadMark at the start).
   */
  private async withStorageRetry(
    attempt: () => Promise<void>,
    signal?: AbortSignal,
    load?: { model: string; since: number }
  ): Promise<void> {
    try {
      await attempt()
    } catch (error) {
      if (!(error instanceof WebGLLMError && error.code === "storage-full")) throw error
      // (The browser's prompt may never be answered: a canceled or unloaded
      // load, or a canceled download, stops waiting for it, and doesn't retry.)
      // (Canceled or unloaded already, as a listener of the failure can do: no wait at all.)
      throwIfAborted(signal)
      if (load) this.throwIfUnloaded(load.model, load.since)
      const answer = abortable(this.keeping ?? Promise.resolve(false), signal)
      if (!(await (load ? this.unlessUnloaded(answer, load.model, load.since) : answer))) throw error
      throwIfAborted(signal)
      if (load) this.throwIfUnloaded(load.model, load.since)
      await attempt()
    }
  }
}

type WorkerProgress = Extract<WorkerResponse, { type: "progress" }>

const bytes = (data: WorkerProgress) =>
  data.totalBytes !== undefined ? { loadedBytes: data.loadedBytes ?? 0, totalBytes: data.totalBytes } : {}

/**
 * A load's progress as the worker counts it: downloading takes 0–0.9 and
 * loading (reading into the GPU, and a first load's calibration) the rest;
 * from storage, loading takes 0–1. It reaches 1 once the model is ready.
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
