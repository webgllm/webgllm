/**
 * predictSpeed: how fast a model would read prompts and write on this
 * device, before its weights download, for an app deciding whether to run
 * it here (say, only at 1,000 prompt tokens/s or more).
 *
 * Measured, not guessed: the model's shapes come from its GGUF header or
 * its ONNX graph and config (see speed-probe.ts), one layer of each kind it
 * has is built with random weights in the storage the real load gives
 * them, and the engine's own prompt and writing passes are timed on it.
 * The timings scale to the whole model by its layer counts and vocabulary
 * (speedFromTimes). The result is remembered per GPU, browser, model and
 * engine code, so later calls answer at once.
 */
import { Engine, TUNING } from "./engine.js"
import { monitorFrames } from "./frame-pacing.js"
import { throwIfAborted } from "./errors.js"
import type { WebGlModelId } from "./model-meta.js"
import { resolveModel } from "./resolve-model.js"
import { probeSpeed, speedShape, type ProbeReport, type ProbeTimes, type SpeedShape } from "./speed-probe.js"
import type { LayerType } from "./weights.js"

/** Tokens per second on this device. */
export type SpeedPrediction = {
  /** Prompt tokens read per second, for a prompt of about 512 tokens. */
  prompt: number
  /** Tokens written per second, one at a time (no speculative drafts). */
  decode: number
}

export type PredictSpeedOptions = {
  /** Measure again, even if this device's result for the model is remembered. */
  force?: boolean
  signal?: AbortSignal
}

/**
 * The prompt and writing speed of `model` on this device, in tokens per
 * second, without downloading its weights: it reads the model's shapes (a
 * GGUF's header, or an ONNX model's graph and config), times one layer of
 * each kind it has with random weights on this GPU (about a second on an
 * M3 Ultra, longer on a slower GPU; a few hundred MB of GPU memory, freed
 * after), and scales that to the whole model. Remembered per GPU, browser, model and engine version: a
 * later call answers at once (`force` measures again).
 *
 * In a page it runs in a worker of its own. It shares the GPU with
 * anything else running: call it while no model is loading or answering,
 * or both are slowed and the prediction is low.
 *
 * @example
 * ```ts
 * const { prompt, decode } = await predictSpeed("LiquidAI/LFM3-1.7B-GGUF")
 * const local = prompt >= 200 && decode >= 10
 * ```
 */
export async function predictSpeed(model: WebGlModelId, options: PredictSpeedOptions = {}): Promise<SpeedPrediction> {
  throwIfAborted(options.signal)
  if (typeof model !== "string" || !model) throw new TypeError("model is required.")
  const remembered = !options.force && inPage.get(model)
  if (remembered) return remembered
  const run = typeof document !== "undefined" && typeof Worker !== "undefined"
    ? inWorker(model, options).then((r) => r.prediction)
    : predictHere(model, options).then((r) => r.prediction)
  const prediction = await run
  inPage.set(model, prediction)
  return prediction
}

/** This page's answers (the GPU, browser and engine code can't change under it). */
const inPage = new Map<string, SpeedPrediction>()

/** What a worker of predictSpeed's own is sent, and answers. */
export type SpeedWorkerRequest =
  | { type: "predict-speed"; model: string; force: boolean }
  | { type: "frames"; intervalMs: number; expectedMs: number; frames: number; startedAt?: number; meanMs?: number }
export type SpeedWorkerReply =
  | { type: "speed"; prediction: SpeedPrediction; report: SpeedReport }
  | { type: "speed-error"; name: string; message: string }

/**
 * predictSpeed in a worker of its own (predict-speed-worker.ts), as a
 * model runs: its GPU waits aren't the page's, and the page's frames are
 * reported to it as webgllm's own workers get them (the GPU's pacing
 * depends on them).
 */
export function inWorker(
  model: string,
  options: PredictSpeedOptions = {},
  worker: Worker = new Worker(new URL("./predict-speed-worker.js", import.meta.url), { type: "module", name: "webgllm-speed" })
): Promise<{ prediction: SpeedPrediction; report: SpeedReport }> {
  return new Promise((resolve, reject) => {
    const post = (message: SpeedWorkerRequest) => worker.postMessage(message)
    const stopFrames = monitorFrames((intervalMs, expectedMs, frames, startedAt, meanMs) =>
      post({ type: "frames", intervalMs, expectedMs, frames, startedAt, meanMs })
    )
    const done = () => {
      stopFrames()
      options.signal?.removeEventListener("abort", abort)
      worker.terminate()
    }
    const abort = () => {
      done()
      reject(options.signal?.reason ?? new DOMException("Aborted", "AbortError"))
    }
    options.signal?.addEventListener("abort", abort, { once: true })
    worker.onmessage = ({ data }: MessageEvent<SpeedWorkerReply>) => {
      done()
      if (data.type === "speed") resolve({ prediction: data.prediction, report: data.report })
      else reject(Object.assign(new Error(data.message), { name: data.name }))
    }
    worker.onerror = (event) => {
      done()
      reject(new Error(event.message || "predictSpeed's worker failed to start."))
    }
    post({ type: "predict-speed", model, force: !!options.force })
  })
}

/** How a prediction was made: the shapes, the timings, how long it took (absent when remembered). */
export type SpeedReport = ProbeReport & { cached: false } | { cached: true }

/** A remembered prediction. */
export type SpeedNote = SpeedPrediction & { at: number }

/** Where predictions are remembered (OPFS by default; see opfsSpeedStore). */
export type SpeedStore = {
  read: (key: string) => Promise<SpeedNote | undefined>
  write: (key: string, note: SpeedNote) => Promise<void>
}

/** What predictHere runs on: the real ones, or a test's. */
export type SpeedDeps = {
  /** This GPU's renderer string (as GPU notes key it). */
  renderer: () => string
  userAgent: string
  store: SpeedStore
  /** Shapes, then timings: the GPU's part. */
  measure: (model: string, signal?: AbortSignal) => Promise<{ shape: SpeedShape; report: ProbeReport }>
}

/** FNV-1a of a string, as hex (as the worker's note keys). */
function hash(text: string) {
  let h = 0x811c9dc5
  for (let i = 0; i < text.length; i++) h = Math.imul(h ^ text.charCodeAt(i), 0x01000193)
  return (h >>> 0).toString(16)
}

/**
 * The key a prediction is remembered by: GPU, browser, model, and the
 * engine's code and settings (Engine.fingerprint and TUNING, as GPU notes
 * are keyed), so a new engine or browser measures again.
 */
export function speedKey(renderer: string, userAgent: string, model: string, code = `${Engine.fingerprint}.${hash(JSON.stringify(TUNING))}`) {
  return `${renderer}|${userAgent}|${model}|${code}`
}

/**
 * Tokens per second from the probe's timings. The head over the model's
 * vocabulary: the probe's, plus its rows past the probe's at the measured
 * cost per row. A prompt: through the first and last layers, plus what
 * the probe's stack of middle layers added, scaled to the model's middle
 * layers, then the head for its last token. A writing step: every layer,
 * then the head.
 */
export function speedFromTimes(times: ProbeTimes, layerTypes: ReadonlyArray<LayerType>, vocabSize: number): SpeedPrediction {
  const headExtra = Math.max(0, vocabSize - times.vocab) * times.headPerRow
  const head = times.headAt + headExtra
  const middle = Math.max(0, layerTypes.length - 2)
  const middleMs = times.stackLayers > 0 ? (Math.max(0, times.promptStack - times.promptBase) * middle) / times.stackLayers : 0
  // (The prompt's last token goes through the head too.)
  const promptMs = times.promptBase + middleMs + head
  const decodeMs = times.decodeLayers + head
  return { prompt: (1000 * times.promptTokens) / promptMs, decode: 1000 / decodeMs }
}

const SPEED_NOTES = "speed-notes.json"

/**
 * Predictions in the origin's storage (webgl-models/speed-notes.json),
 * best effort: no storage means measuring each time. Written through a
 * synchronous access handle in a worker, else createWritable.
 */
export const opfsSpeedStore: SpeedStore = {
  async read(key) {
    try {
      const notes = JSON.parse(await (await (await speedFile()).getFile()).text()) as Record<string, SpeedNote>
      const note = notes[key]
      return note && note.prompt > 0 && note.decode > 0 ? note : undefined
    } catch {
      return undefined
    }
  },
  async write(key, note) {
    try {
      const file = await speedFile()
      let notes: Record<string, SpeedNote> = {}
      try {
        notes = JSON.parse(await (await file.getFile()).text()) as Record<string, SpeedNote>
      } catch {
        // None yet, or unreadable: written over.
      }
      // This GPU, browser and model's notes from other engine code are stale.
      const prefix = key.slice(0, key.lastIndexOf("|") + 1)
      for (const k of Object.keys(notes)) if (k.startsWith(prefix)) delete notes[k]
      notes[key] = note
      const bytes = new TextEncoder().encode(JSON.stringify(notes))
      const handle = file as FileSystemFileHandle & {
        createSyncAccessHandle?: () => Promise<{ truncate: (n: number) => void; write: (b: Uint8Array, o: { at: number }) => number; flush: () => void; close: () => void }>
      }
      if (typeof document === "undefined" && handle.createSyncAccessHandle) {
        const access = await handle.createSyncAccessHandle()
        try {
          access.truncate(0)
          access.write(bytes, { at: 0 })
          access.flush()
        } finally {
          access.close()
        }
      } else {
        const writable = await file.createWritable()
        await writable.write(bytes)
        await writable.close()
      }
    } catch {
      // Remembering is optional.
    }
  }
}

async function speedFile() {
  const root = await navigator.storage.getDirectory()
  const dir = await root.getDirectoryHandle("webgl-models", { create: true })
  return dir.getFileHandle(SPEED_NOTES, { create: true })
}

/** The GPU's renderer string, from a context of its own. */
function rendererHere() {
  const gl = new OffscreenCanvas(1, 1).getContext("webgl2")
  if (!gl) throw new Error("This browser has no WebGL 2.")
  const debug = gl.getExtension("WEBGL_debug_renderer_info")
  const renderer = String(gl.getParameter(debug ? debug.UNMASKED_RENDERER_WEBGL : gl.RENDERER))
  gl.getExtension("WEBGL_lose_context")?.loseContext()
  return renderer
}

/**
 * The real dependencies; `onEngine` is told of the probe's engine while it
 * measures (and of null after), for the page's frame reports.
 */
export const realSpeedDeps = (onEngine?: (engine: Engine | null) => void): SpeedDeps => ({
  renderer: rendererHere,
  userAgent: typeof navigator !== "undefined" ? navigator.userAgent : "",
  store: opfsSpeedStore,
  measure: async (model, signal) => {
    const started = performance.now()
    const meta = await resolveModel(model, signal)
    const shape = await speedShape(meta)
    const shapeMs = performance.now() - started
    throwIfAborted(signal)
    const report = await probeSpeed(shape, { signal, onEngine })
    return { shape, report: { ...report, shapeMs } }
  }
})

/** Measures at most once at a time per model in this context. */
const pending = new Map<string, Promise<{ prediction: SpeedPrediction; report: SpeedReport }>>()

/**
 * predictSpeed in this context (a worker's, or a test's with `deps`):
 * the remembered prediction unless `force`, else measured and remembered.
 */
export function predictHere(
  model: string,
  options: PredictSpeedOptions = {},
  deps: SpeedDeps = realSpeedDeps()
): Promise<{ prediction: SpeedPrediction; report: SpeedReport }> {
  const key = speedKey(deps.renderer(), deps.userAgent, model)
  const running = pending.get(key)
  if (running) return running
  const run = (async () => {
    if (!options.force) {
      const note = await deps.store.read(key)
      if (note) return { prediction: { prompt: note.prompt, decode: note.decode }, report: { cached: true as const } }
    }
    const { shape, report } = await deps.measure(model, options.signal)
    const prediction = speedFromTimes(report.times, shape.config.layerTypes, shape.config.vocabSize)
    await deps.store.write(key, { ...prediction, at: Date.now() })
    return { prediction, report: { ...report, cached: false as const } }
  })()
  pending.set(key, run)
  const clear = () => pending.delete(key)
  run.then(clear, clear)
  return run
}
