/**
 * Image generation (webgllm.images): an image model loaded in a worker,
 * one of webgllm's for each model (ended when it's unloaded, which frees
 * its memory at once), or the caller's worker (which frees the model and
 * stays running). Requests take turns in the worker.
 */
import { abortError, failedToLoad, throwIfAborted, WebGLLMError, workerFailure } from "./errors.js"
import { isImageModel, TESSERA, type WebGlImageModel } from "./image-models.js"
import { isImageResponse, type ImageRequest, type ImageResponse } from "./image-protocol.js"
import type { LoadProgress, WebGLLMEventMap } from "./openai.js"
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
}

type Waiter = { onMessage: (message: ImageResponse) => boolean; reject: (error: unknown) => void }

let ids = 0

/** One worker's image side: its requests and their answers. */
class ImageWorker {
  private readonly waiters = new Map<number, Waiter>()
  gone: Error | null = null

  constructor(
    readonly worker: Worker,
    readonly owned: boolean,
    webgpu: "auto" | false
  ) {
    worker.addEventListener("message", this.onMessage)
    worker.addEventListener("error", this.onError)
    // A new worker loads on WebGPU unless told otherwise; the caller's is told every time (another client may have told it otherwise).
    if (webgpu === false || !owned) worker.postMessage({ type: "configure", webgpu, [WEBGLLM]: true })
  }

  private readonly onMessage = ({ data }: MessageEvent) => {
    if (!isImageResponse(data)) return
    const waiter = this.waiters.get(data.id)
    if (waiter?.onMessage(data)) this.waiters.delete(data.id)
  }

  private readonly onError = (event: ErrorEvent) => {
    // A worker of the caller's that only threw in its own code keeps running; one whose script failed never does.
    if (!this.owned && !failedToLoad(event)) return
    this.fail(workerFailure(event))
  }

  /** Fails every request waiting, and any later one, with `error`. */
  fail(error: Error) {
    this.gone ??= error
    for (const waiter of this.waiters.values()) waiter.reject(error)
    this.waiters.clear()
  }

  /** Sends `request`; `onMessage` gets its answers until it returns true (the last one). */
  send(request: ImageRequest, onMessage: (message: ImageResponse, resolve: (value: ImageResponse) => void, reject: (error: unknown) => void) => boolean) {
    return new Promise<ImageResponse>((resolve, reject) => {
      if (this.gone) return reject(this.gone)
      this.waiters.set(request.id, { onMessage: (m) => onMessage(m, resolve, reject), reject })
      this.worker.postMessage({ ...request, [WEBGLLM]: true })
    })
  }

  /** Ends it (webgllm's own), or stops listening (the caller's). */
  close(error: Error) {
    this.fail(error)
    this.worker.removeEventListener("message", this.onMessage)
    this.worker.removeEventListener("error", this.onError)
    if (this.owned) this.worker.terminate()
  }
}

type Entry = {
  model: WebGlImageModel
  worker: ImageWorker
  ready: Promise<void>
  loaded: boolean
  listeners: Set<(progress: LoadProgress) => void>
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
  private readonly entries = new Map<WebGlImageModel, Entry>()

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
   * Loads `model` (downloading it the first time): on WebGPU where the
   * browser has it and the GPU passes the model's self-test, else on WebGL 2.
   */
  async load(model: WebGlImageModel, options: { onProgress?: (progress: LoadProgress) => void; signal?: AbortSignal } = {}): Promise<void> {
    throwIfAborted(options.signal)
    if (!isImageModel(model)) throw new WebGLLMError("model-not-supported", `${String(model)} isn't an image model webgllm runs.`)
    this.host.checkSupport()
    let entry = this.entries.get(model)
    if (!entry) {
      // The caller's worker holds one image model: another is unloaded first.
      if (this.host.worker) for (const other of [...this.entries.keys()]) await this.unload(other)
      const worker = new ImageWorker(this.host.worker ?? this.host.createWorker(), !this.host.worker, this.host.webgpu ?? "auto")
      const listeners = new Set<(progress: LoadProgress) => void>()
      const id = ++ids
      this.host.dispatch("loadstart", { model })
      const made: Entry = {
        model,
        worker,
        loaded: false,
        listeners,
        weightBytes: 0,
        engine: null,
        ready: worker
          .send({ type: "image-load", id, model }, (m, resolve, reject) => {
            if (m.type === "image-progress") {
              const progress: LoadProgress = { model, phase: m.phase, progress: m.progress, cached: m.cached }
              this.host.onProgress(progress)
              for (const listener of listeners) listener(progress)
              return false
            }
            if (m.type === "image-ready") {
              made.weightBytes = m.weightBytes
              made.engine = m.engine
              resolve(m)
            } else if (m.type === "image-error") reject(new Error(m.message))
            else return false
            return true
          })
          .then(
            () => {
              made.loaded = true
              this.host.dispatch("load", { model, contextLength: 0 })
            },
            (error: unknown) => {
              if (this.entries.get(model) === made) {
                this.entries.delete(model)
                worker.close(error as Error)
                this.host.dispatch("error", { model, error: error as Error })
                this.host.dispatch("unload", { model, reason: "failed" })
              }
              throw error
            }
          )
      }
      entry = made
      this.entries.set(model, entry)
    }
    const { listeners, ready } = entry
    const listener = options.onProgress
    if (listener) listeners.add(listener)
    try {
      await (options.signal ? abortableWait(ready, options.signal) : ready)
    } finally {
      if (listener) listeners.delete(listener)
    }
  }

  /**
   * Unloads `model`, or every image model: webgllm's own worker is ended,
   * which frees the memory at once; the caller's frees the model. Resolves
   * after its "unload" event.
   */
  async unload(model?: string): Promise<void> {
    const entries = [...this.entries.values()].filter((e) => model === undefined || e.model === model)
    await Promise.all(
      entries.map(async (entry) => {
        this.entries.delete(entry.model)
        const reason = new WebGLLMError("model-not-loaded", `${entry.model} was unloaded.`)
        if (!entry.worker.owned) {
          // The caller's worker frees it after the request running there.
          await entry.worker
            .send({ type: "image-unload", id: ++ids, model: entry.model }, (m, resolve) => {
              if (m.type !== "image-unloaded" && m.type !== "image-error") return false
              resolve(m)
              return true
            })
            .catch(() => undefined)
        }
        entry.worker.close(reason)
        entry.ready.catch(() => undefined)
        this.host.dispatch("unload", { model: entry.model, reason: "unloaded" })
      })
    )
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
    const entry = this.entries.get(model)
    if (!entry) throw new WebGLLMError("model-not-loaded", `${model} was unloaded.`)
    const id = ++ids
    const stop = () => entry.worker.worker.postMessage({ type: "image-stop", id, [WEBGLLM]: true })
    signal?.addEventListener("abort", stop, { once: true })
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
