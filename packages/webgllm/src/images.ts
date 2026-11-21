/**
 * Image generation (webgllm.images): an image model loaded in a worker,
 * one of webgllm's for each model (ended when it's unloaded, which frees
 * its memory at once), or the caller's worker (which frees the model and
 * stays running). Requests take turns in the worker. The model's memory is
 * claimed, before the worker allocates it, within the instance's memory
 * budget (with the text models) and through the host's onMemoryClaim.
 */
import { abortError, failedToLoad, throwIfAborted, WebGLLMError, workerFailure } from "./errors.js"
import { isImageModel, TESSERA, type WebGlImageModel } from "./image-models.js"
import { isImageResponse, type ImageRequest, type ImageResponse } from "./image-protocol.js"
import type { LoadProgress, ModelUnloadReason, WebGLLMEventMap } from "./openai.js"
import { WEBGLLM } from "./protocol.js"

export type ImageGenerateParams = {
  /** The image model (Tessera-small by default). */
  model?: WebGlImageModel
  prompt: string
  /** Images to draw, 1–8 (1 by default): drawn together, in one batch. */
  n?: number
  /** The same prompt, seed and settings draw the same images on the same GPU. Random when left out. */
  seed?: number
  /** Denoising steps (16 by default). */
  steps?: number
  /** Classifier-free guidance (3 by default): higher sticks closer to the prompt. */
  guidance_scale?: number
  /** "b64_json" (the default): PNG files, base64; "rgba": the raw pixels, 4 bytes a pixel, row by row. */
  response_format?: "b64_json" | "rgba"
  /** A PNG's size: the model's own (16x16) by default, or a multiple of it, each pixel repeated ("128x128"). */
  size?: `${number}x${number}`
}

export type ImageData = {
  /** The PNG, base64 (response_format "b64_json"). */
  b64_json?: string
  /** The pixels at the model's own size (response_format "rgba"). */
  rgba?: Uint8Array
}

export type ImagesResponse = {
  /** Seconds since the epoch. */
  created: number
  model: WebGlImageModel
  data: Array<ImageData>
  /** The seed the images were drawn with. */
  seed: number
  /** Width and height of `rgba` (the model's own). */
  width: number
  height: number
  timings: { text_ms: number; draw_ms: number }
  /** GPU bytes the model holds: its weights, and the work textures this many images took. */
  memory: { weights_bytes: number; work_bytes: number }
}

export type ImageRequestOptions = {
  /** Aborting rejects with an AbortError and stops the drawing between steps. */
  signal?: AbortSignal
  /** The model's load progress, when this call loads it. */
  onProgress?: (progress: LoadProgress) => void
  /** After each denoising step. */
  onStep?: (step: number, steps: number) => void
}

type Host = {
  createWorker: () => Worker
  worker?: Worker
  webgpu?: "auto" | false
  checkSupport: () => void
  onProgress: (progress: LoadProgress) => void
  dispatch: <K extends keyof WebGLLMEventMap>(type: K, event: WebGLLMEventMap[K]) => void
  /**
   * Room for `bytes` (all the model will hold): in the shared memory budget
   * (`unload` frees the model when another claim needs its room), then
   * the host's onMemoryClaim. Rejects if refused or `signal` aborts.
   */
  claim: (model: string, bytes: number, unload: () => Promise<void>, keepLoaded: boolean, signal: AbortSignal) => Promise<void>
  /** Whether the model may be unloaded to make room for another. */
  setKeepLoaded: (model: string, keep: boolean) => void
  /** The model was just used (least recently used order). */
  touch: (model: string) => void
  /** The model holds nothing any more. */
  release: (model: string) => void
}

type Waiter = { onMessage: (message: ImageResponse) => boolean; reject: (error: unknown) => void }

let ids = 0

/** One worker's image side: its requests and their answers. */
class ImageWorker {
  private readonly waiters = new Map<number, Waiter>()
  gone: Error | null = null
  /** No longer listened to (see close). */
  closed = false

  constructor(
    readonly worker: Worker,
    readonly owned: boolean,
    webgpu: "auto" | false,
    /** A claim the worker makes before allocating. */
    private readonly onClaim: (message: Extract<ImageResponse, { type: "image-claim" }>) => void,
    /** The worker failed: every request fails with `error`. */
    private readonly onFail: (error: Error) => void
  ) {
    worker.addEventListener("message", this.onMessage)
    worker.addEventListener("error", this.onError)
    // A new worker loads on WebGPU unless told otherwise; the caller's is told every time (another client may have told it otherwise).
    if (webgpu === false || !owned) this.post({ type: "configure", webgpu })
  }

  post(message: Record<string, unknown>) {
    this.worker.postMessage({ ...message, [WEBGLLM]: true })
  }

  private readonly onMessage = ({ data }: MessageEvent) => {
    if (!isImageResponse(data)) return
    if (data.type === "image-claim") return void this.onClaim(data)
    const waiter = this.waiters.get(data.id)
    if (waiter?.onMessage(data)) this.waiters.delete(data.id)
  }

  private readonly onError = (event: ErrorEvent) => {
    // A worker of the caller's that only threw in its own code keeps running; one whose script failed never does.
    if (!this.owned && !failedToLoad(event)) return
    const error = workerFailure(event)
    this.fail(error)
    this.onFail(error)
  }

  /** Fails every request waiting, and any later one, with `error`. */
  fail(error: Error) {
    this.gone ??= error
    for (const waiter of this.waiters.values()) waiter.reject(error)
    this.waiters.clear()
  }

  /** Fails request `id` here, and stops it in the worker. */
  cancel(id: number, error: unknown) {
    const waiter = this.waiters.get(id)
    this.waiters.delete(id)
    waiter?.reject(error)
    if (!this.gone) this.post({ type: "image-stop", id })
  }

  /** Sends `request`; `onMessage` gets its answers until it returns true (the last one). */
  send(request: ImageRequest, onMessage: (message: ImageResponse, resolve: (value: ImageResponse) => void, reject: (error: unknown) => void) => boolean) {
    return new Promise<ImageResponse>((resolve, reject) => {
      if (this.gone) return reject(this.gone)
      this.waiters.set(request.id, { onMessage: (m) => onMessage(m, resolve, reject), reject })
      this.post(request)
    })
  }

  /** Ends it (webgllm's own), or stops listening (the caller's). */
  close(error: Error) {
    this.closed = true
    this.fail(error)
    this.worker.removeEventListener("message", this.onMessage)
    this.worker.removeEventListener("error", this.onError)
    if (this.owned) this.worker.terminate()
  }
}

type Entry = {
  model: WebGlImageModel
  worker: ImageWorker
  /** The load's request id. */
  loadId: number
  ready: Promise<void>
  loaded: boolean
  /** Calls waiting for the load: the last to abort stops it. */
  waiters: number
  listeners: Set<(progress: LoadProgress) => void>
  /** Drawings in progress, by request id. */
  drawing: Set<number>
  /** Aborted once it's gone: its claims stop waiting. */
  gone: AbortController
  /** Never unloaded to make room for another model. */
  keepLoaded: boolean
  /** Its teardown, once it began (see teardown). */
  closing: Promise<void> | null
  /** Each request's claims still waiting for room: stopping the request stops them. */
  claims: Map<number, Set<AbortController>>
  /** GPU bytes of its weights, once loaded. */
  weightBytes: number
  engine: "webgpu" | "webgl" | null
}

const toBase64 = (bytes: Uint8Array) => {
  let text = ""
  for (let i = 0; i < bytes.length; i += 0x8000) text += String.fromCharCode(...bytes.subarray(i, i + 0x8000))
  return btoa(text)
}

export class Images {
  /** Models loaded or loading, usable by requests. */
  private readonly entries = new Map<WebGlImageModel, Entry>()
  /** Models being torn down: a new load waits for its model's to finish (see teardown). */
  private readonly tearing = new Map<string, Promise<void>>()
  /** Unloads asked for, by model and of every model: a load asked for before one stops (see unloadMark). */
  private readonly unloads = new Map<string, number>()
  private unloadsOfAll = 0

  /** Counts the unloads of `model` asked for so far: a load compares it before and after it waits. */
  private unloadMark(model: string) {
    return this.unloadsOfAll + (this.unloads.get(model) ?? 0)
  }

  /** Stops request `id`'s claims still waiting for room (it was stopped). */
  private stopClaims(entry: Entry, id: number) {
    for (const c of entry.claims.get(id) ?? []) c.abort(abortError())
    entry.claims.delete(id)
  }

  constructor(private readonly host: Host) {}

  /** Whether `model` is loaded and ready. */
  isLoaded(model: string): boolean {
    return !!this.entries.get(model as WebGlImageModel)?.loaded
  }

  /** The GPU API `model` runs on, once loaded. */
  engineOf(model: string): "webgpu" | "webgl" | null {
    return this.entries.get(model as WebGlImageModel)?.engine ?? null
  }

  /** The image models loaded. */
  loadedModels(): Array<WebGlImageModel> {
    return [...this.entries.values()].filter((e) => e.loaded).map((e) => e.model)
  }

  /**
   * Tears `entry` down, once: it's taken out of use at once (a request for
   * its model waits for this, then loads it again), its claims stop, and
   * its requests in progress fail. webgllm's own worker is ended; the
   * caller's is told to stop each request and to free the model, and is
   * listened to until it says it has. Then its memory is released from the
   * budget and its events say why.
   */
  private teardown(entry: Entry, error: Error, why: ModelUnloadReason): Promise<void> {
    if (entry.closing) return entry.closing
    let done!: () => void
    entry.closing = new Promise<void>((resolve) => (done = resolve))
    this.tearing.set(entry.model, entry.closing)
    if (this.entries.get(entry.model) === entry) this.entries.delete(entry.model)
    entry.gone.abort(error)
    const finish = () => {
      entry.worker.close(error)
      entry.ready.catch(() => undefined)
      this.host.release(entry.model)
      if (this.tearing.get(entry.model) === entry.closing) this.tearing.delete(entry.model)
      if (why === "failed" || why === "crashed") this.host.dispatch("error", { model: entry.model, error })
      this.host.dispatch("unload", { model: entry.model, reason: why })
      done()
    }
    if (entry.worker.owned || entry.worker.gone) {
      finish()
      return entry.closing
    }
    // Every request stops first (else the unload waits behind them), the load too (it may have finished: the unload frees what it loaded).
    for (const id of entry.drawing) entry.worker.cancel(id, error)
    entry.worker.cancel(entry.loadId, error)
    void entry.worker
      .send({ type: "image-unload", id: ++ids, model: entry.model }, (m, resolve) => {
        if (m.type !== "image-unloaded" && m.type !== "image-error") return false
        resolve(m)
        return true
      })
      .catch(() => undefined)
      .then(finish)
    return entry.closing
  }

  /** Answers a claim of `entry`'s worker: granted, or refused (its request then fails). */
  private claim(entry: Entry, m: Extract<ImageResponse, { type: "image-claim" }>) {
    // (Answered while the worker is listened to, torn down or not: it waits for the answer.)
    const answer = (refused?: string) => {
      if (!entry.worker.closed) entry.worker.post({ type: "image-claimed", id: m.claim, ...(refused !== undefined && { refused }) })
    }
    if (entry.closing) return answer(`${entry.model} was unloaded`)
    // Given up when the request is stopped, or the model goes.
    const stop = new AbortController()
    const claims = entry.claims.get(m.id) ?? new Set<AbortController>()
    claims.add(stop)
    entry.claims.set(m.id, claims)
    const gone = entry.gone.signal
    const onGone = () => stop.abort(gone.reason)
    if (gone.aborted) onGone()
    else gone.addEventListener("abort", onGone, { once: true })
    const signal = stop.signal
    const settled = () => {
      gone.removeEventListener("abort", onGone)
      claims.delete(stop)
      if (!claims.size && entry.claims.get(m.id) === claims) entry.claims.delete(m.id)
    }
    this.host.claim(entry.model, m.bytes, () => this.unload(entry.model, "evicted"), entry.keepLoaded, signal).then(
      () => (settled(), answer()),
      (error: unknown) => (settled(), answer(error instanceof Error ? error.message : String(error)))
    )
  }

  /**
   * Loads `model` (downloading it the first time): on WebGPU where the
   * browser has it and the GPU passes the model's self-test, else on WebGL 2.
   * Aborting stops the load, unless another call is waiting for it too.
   */
  async load(
    model: WebGlImageModel,
    options: { onProgress?: (progress: LoadProgress) => void; signal?: AbortSignal; keepLoaded?: boolean } = {}
  ): Promise<void> {
    const { signal } = options
    throwIfAborted(signal)
    if (!isImageModel(model)) throw new WebGLLMError("model-not-supported", `${String(model)} isn't an image model webgllm runs.`)
    this.host.checkSupport()
    // An unload asked for while this waits stops it.
    const since = this.unloadMark(model)
    let entry = this.entries.get(model)
    while (!entry) {
      // A teardown of this model, or of another in the caller's worker (which holds one), finishes first; the teardown goes on if this stops waiting.
      const waits = [this.tearing.get(model), ...(this.host.worker ? [...this.tearing.values(), ...[...this.entries.keys()].map((other) => this.unload(other))] : [])]
      if (waits.some(Boolean)) {
        const all = Promise.all(waits)
        await (signal ? abortableWait(all, signal) : all)
        if (this.unloadMark(model) !== since) throw new WebGLLMError("model-not-loaded", `${model} was unloaded.`)
        entry = this.entries.get(model)
        continue
      }
      entry = this.start(model, options.keepLoaded ?? false)
    }
    const made = entry
    if (options.keepLoaded !== undefined && made.keepLoaded !== options.keepLoaded) {
      made.keepLoaded = options.keepLoaded
      this.host.setKeepLoaded(model, options.keepLoaded)
    }
    const listener = options.onProgress
    if (listener) made.listeners.add(listener)
    made.waiters++
    try {
      await (signal ? abortableWait(made.ready, signal) : made.ready)
    } catch (error) {
      // The last call waiting gave up: the load stops.
      if (signal?.aborted && made.waiters === 1 && !made.loaded) void this.teardown(made, abortError(signal) as Error, "aborted")
      throw error
    } finally {
      made.waiters--
      if (listener) made.listeners.delete(listener)
    }
  }

  /** Starts loading `model` in a worker. */
  private start(model: WebGlImageModel, keepLoaded: boolean): Entry {
    const loadId = ++ids
    // (The entry's callbacks reach it once it's made, below.)
    let entry!: Entry
    const worker = new ImageWorker(
      this.host.worker ?? this.host.createWorker(),
      !this.host.worker,
      this.host.webgpu ?? "auto",
      (m) => this.claim(entry, m),
      (error) => void this.teardown(entry, error, "crashed")
    )
    const listeners = new Set<(progress: LoadProgress) => void>()
    this.host.dispatch("loadstart", { model })
    entry = {
      model,
      worker,
      loadId,
      loaded: false,
      waiters: 0,
      listeners,
      drawing: new Set(),
      gone: new AbortController(),
      keepLoaded,
      closing: null,
      claims: new Map(),
      weightBytes: 0,
      engine: null,
      ready: Promise.resolve()
    }
    this.entries.set(model, entry)
    entry.ready = worker
      .send({ type: "image-load", id: loadId, model }, (m, resolve, reject) => {
        if (m.type === "image-progress") {
          const progress: LoadProgress = { model, phase: m.phase, progress: m.progress, cached: m.cached }
          this.host.onProgress(progress)
          for (const l of listeners) l(progress)
          return false
        }
        if (m.type === "image-ready") {
          entry.weightBytes = m.weightBytes
          entry.engine = m.engine
          resolve(m)
        } else if (m.type === "image-error") reject(new Error(m.message))
        else return false
        return true
      })
      .then(
        () => {
          entry.loaded = true
          this.host.dispatch("load", { model, contextLength: 0 })
        },
        (error: unknown) => {
          // (Torn down already, its teardown stands: this is its load failing.)
          void this.teardown(entry, error as Error, "failed")
          throw error
        }
      )
    return entry
  }

  /**
   * Unloads `model`, or every image model: webgllm's own worker is ended,
   * which frees the memory at once; the caller's stops the drawings in
   * progress, then frees the model. Their drawings fail. Resolves after
   * the "unload" event.
   */
  async unload(model?: string, why: ModelUnloadReason = "unloaded"): Promise<void> {
    // Loads waiting (for a teardown) see an unload asked for and stop (not an eviction: that's room made, not a request).
    if (why === "unloaded") {
      if (model === undefined) this.unloadsOfAll++
      else this.unloads.set(model, (this.unloads.get(model) ?? 0) + 1)
    }
    const reason = (m: string) => new WebGLLMError("model-not-loaded", `${m} was unloaded.`)
    const entries = [...this.entries.values()].filter((e) => model === undefined || e.model === model)
    // (Teardowns in progress too: it resolves once the model is freed.)
    const tearing = [...this.tearing.entries()].filter(([m]) => model === undefined || m === model).map(([, p]) => p)
    await Promise.all([...entries.map((e) => this.teardown(e, reason(e.model), why)), ...tearing])
  }

  /** Draws `n` images for `prompt` (loading the model first if it isn't). */
  async generate(params: ImageGenerateParams, options: ImageRequestOptions = {}): Promise<ImagesResponse> {
    const { signal } = options
    throwIfAborted(signal)
    const model = params.model ?? TESSERA
    if (typeof params.prompt !== "string") throw new TypeError("prompt must be a string.")
    const n = params.n ?? 1
    if (!Number.isInteger(n) || n < 1 || n > 8) throw new TypeError("n must be a whole number from 1 to 8.")
    const steps = params.steps ?? 16
    if (!Number.isInteger(steps) || steps < 1 || steps > 64) throw new TypeError("steps must be a whole number from 1 to 64.")
    const guidanceScale = params.guidance_scale ?? 3
    if (typeof guidanceScale !== "number" || !Number.isFinite(guidanceScale)) throw new TypeError("guidance_scale must be a number.")
    const seed = params.seed ?? Math.floor(Math.random() * 2 ** 32)
    if (!Number.isInteger(seed) || seed < 0 || seed >= 2 ** 32) throw new TypeError("seed must be a whole number from 0 to 2³² − 1.")
    const format = params.response_format ?? "b64_json"
    if (format !== "b64_json" && format !== "rgba") throw new TypeError(`response_format must be "b64_json" or "rgba".`)
    const size = /^(\d+)x(\d+)$/.exec(params.size ?? "16x16")
    const side = size ? Number(size[1]) : NaN
    if (!size || size[1] !== size[2] || side % 16 || side < 16 || side > 1024) throw new TypeError(`size must be square, a multiple of 16 up to 1024 ("16x16", "128x128").`)
    await this.load(model, { ...(options.onProgress && { onProgress: options.onProgress }), ...(signal && { signal }) })
    // (Nothing awaited from here to the request: an unload either took the entry out already, or stops this drawing.)
    const entry = this.entries.get(model)
    if (!entry || entry.closing) throw new WebGLLMError("model-not-loaded", `${model} was unloaded.`)
    this.host.touch(model)
    const id = ++ids
    const stop = () => {
      entry.worker.post({ type: "image-stop", id })
      this.stopClaims(entry, id)
    }
    signal?.addEventListener("abort", stop, { once: true })
    entry.drawing.add(id)
    try {
      const done = await entry.worker.send(
        { type: "image-generate", id, model, prompt: params.prompt, n, seed, steps, guidanceScale, png: format === "b64_json", scale: side / 16 },
        (m, resolve, reject) => {
          if (m.type === "image-step") {
            options.onStep?.(m.step, m.steps)
            return false
          }
          if (m.type === "image-done") resolve(m)
          else if (m.type === "image-error") reject(signal?.aborted ? abortError(signal) : new Error(m.message))
          else return false
          return true
        }
      )
      if (done.type !== "image-done") throw new Error("Unexpected answer.")
      return {
        created: Math.floor(Date.now() / 1000),
        model,
        data: done.images.map((image) => (format === "rgba" ? { rgba: image.rgba } : { b64_json: toBase64(image.png!) })),
        seed,
        width: 16,
        height: 16,
        timings: { text_ms: done.timings.textMs, draw_ms: done.timings.drawMs },
        memory: { weights_bytes: entry.weightBytes, work_bytes: done.workBytes }
      }
    } finally {
      entry.drawing.delete(id)
      signal?.removeEventListener("abort", stop)
    }
  }
}

/** `promise`, unless `signal` aborts first (the load goes on for others). */
function abortableWait<T>(promise: Promise<T>, signal: AbortSignal): Promise<T> {
  return new Promise<T>((resolve, reject) => {
    const onAbort = () => reject(abortError(signal))
    if (signal.aborted) return onAbort()
    signal.addEventListener("abort", onAbort, { once: true })
    promise.then(
      (v) => {
        signal.removeEventListener("abort", onAbort)
        resolve(v)
      },
      (e: unknown) => {
        signal.removeEventListener("abort", onAbort)
        reject(e)
      }
    )
  })
}
