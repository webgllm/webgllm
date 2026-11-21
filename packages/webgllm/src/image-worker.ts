/**
 * The image models' side of a worker (beside handler.ts's text models):
 * loads Tessera's files (from storage, else downloading them into it),
 * runs it on WebGPU, else WebGL 2, and draws sprites. A load checks the
 * GPU against the model's own self-test (selftest.json: the denoiser's
 * argmax for one prompt on a blank canvas); a GPU that gets fewer than
 * 80% of its pixels right isn't used, and WebGL 2 is tried next.
 */
import { Tokenizer } from "@huggingface/tokenizers"
import { Gpu } from "./gpu/gl.js"
import { WebGpu } from "./gpu/webgpu.js"
import { IMAGE_MODEL_META, isImageModel, type ImageModelMeta } from "./image-models.js"
import type { ImageRequest, ImageResponse } from "./image-protocol.js"
import type { ModelFile } from "./model-meta.js"
import { encodePng } from "./png.js"
import { download, storedFile } from "./storage.js"
import { TesseraGpu, tesseraWeightBytes } from "./tessera.js"
import { DEFAULT_SAMPLER, drawSprites, fixedLengthIds, GRID, MASK, PALETTE, PIXELS, spritePixels } from "./tessera-sampler.js"
import { readClipWeights, readTesseraWeights } from "./tessera-weights.js"

/** Pixels of selftest.json's canvas that must match its argmax for a GPU to be used. */
export const SELFTEST_AGREEMENT = 0.8

export type ImageHost = {
  post: (message: ImageResponse, transfer?: Array<Transferable>) => void
  /** Whether loads may run on WebGPU ("configure"). */
  webgpu: () => boolean
  /** Runs the worker's collector after a model's memory was released (see collectReleased in handler.ts). */
  collect: () => Promise<void>
}

type Loaded = {
  model: string
  /** Bytes claimed for it: weights and the work textures of the largest batch so far. */
  claimed: number
  weightBytes: number
  gpu: Gpu | WebGpu
  tessera: TesseraGpu
  engine: "webgpu" | "webgl"
  tokenize: (text: string) => Array<number>
  endToken: number
  palette: Array<Array<number>>
}

let current: Loaded | null = null
/** Requests run one at a time, in order (loads, generations, unloads). */
let queue: Promise<void> = Promise.resolve()
const aborts = new Map<number, AbortController>()

/** Claims waiting for the page's answer, by claim id. */
const claims = new Map<number, { resolve: () => void; reject: (error: unknown) => void }>()
let claimIds = 0

/**
 * Asks the page for room for `bytes` (all the model will hold), before
 * allocating them: it makes room within the instance's memory budget
 * (unloading other models) and asks the host's onMemoryClaim. Rejects if
 * either refuses, or `signal` aborts.
 */
function claim(host: ImageHost, id: number, model: string, bytes: number, signal: AbortSignal): Promise<void> {
  signal.throwIfAborted()
  const c = ++claimIds
  return new Promise<void>((resolve, reject) => {
    const onAbort = () => {
      if (claims.delete(c)) reject(signal.reason)
    }
    claims.set(c, {
      resolve: () => (signal.removeEventListener("abort", onAbort), resolve()),
      reject: (error) => (signal.removeEventListener("abort", onAbort), reject(error))
    })
    signal.addEventListener("abort", onAbort, { once: true })
    host.post({ type: "image-claim", id, claim: c, model, bytes })
  })
}

/** An image request from the page. */
export function receiveImage(data: ImageRequest, host: ImageHost) {
  if (data.type === "image-stop") {
    aborts.get(data.id)?.abort()
    return
  }
  if (data.type === "image-claimed") {
    const waiting = claims.get(data.id)
    claims.delete(data.id)
    if (data.refused === undefined) waiting?.resolve()
    else waiting?.reject(new Error(`Tessera's memory claim was refused: ${data.refused}`))
    return
  }
  const abort = new AbortController()
  aborts.set(data.id, abort)
  queue = queue.then(async () => {
    try {
      if (data.type === "image-load") {
        const loaded = await load(data.model, data.id, host, abort.signal)
        host.post({ type: "image-ready", id: data.id, model: loaded.model, engine: loaded.engine, weightBytes: loaded.tessera.bytes.weights, selftest: lastSelftest })
      } else if (data.type === "image-generate") {
        await generate(data, host, abort.signal)
      } else if (data.type === "image-unload") {
        if (current?.model === data.model) await release(host)
        host.post({ type: "image-unloaded", id: data.id })
      }
    } catch (error) {
      host.post({ type: "image-error", id: data.id, message: error instanceof Error ? error.message : String(error) })
    } finally {
      aborts.delete(data.id)
    }
  })
}

let lastSelftest = 0

/**
 * Frees the loaded model, then runs the collector (see disposeThenCollect):
 * nothing of the model is held while it runs, so the collection can take it.
 */
function release(host: ImageHost): Promise<void> {
  return disposeThenCollect(
    () => {
      const was = current
      current = null
      return was
    },
    (was) => {
      was.tessera.dispose()
      was.gpu.dispose()
    },
    host.collect
  )
}

/**
 * Takes a thing out (`take`), frees it (`dispose`) and drops it, all in
 * this synchronous call, then starts `collect`: no suspended function holds
 * the thing while the collector runs, as an async function awaiting the
 * collection would.
 */
export function disposeThenCollect<T>(take: () => T | null, dispose: (thing: T) => void, collect: () => Promise<void>): Promise<void> {
  let thing = take()
  if (thing === null) return Promise.resolve()
  dispose(thing)
  thing = null
  return collect()
}

/** A file's bytes: from storage, else downloaded into it, reporting its bytes as they come. */
async function fileBytes(file: ModelFile, onBytes: (loaded: number) => void, signal: AbortSignal): Promise<Uint8Array> {
  const stored = await storedFile(file)
  const blob = stored ?? (await download(file, (loaded) => onBytes(loaded), signal))
  onBytes(blob.size)
  return new Uint8Array(await blob.arrayBuffer())
}

async function load(model: string, id: number, host: ImageHost, signal: AbortSignal): Promise<Loaded> {
  if (current?.model === model) return current
  if (!isImageModel(model)) throw new Error(`${model} isn't an image model webgllm runs.`)
  if (current) await release(host)
  const meta: ImageModelMeta = IMAGE_MODEL_META[model]
  const files = Object.values(meta.files)
  const total = files.reduce((s, f) => s + f.bytes, 0)
  const cached = (await Promise.all(files.map((f) => storedFile(f)))).every(Boolean)
  const got = new Map<ModelFile, number>()
  const report = (phase: "download" | "load", progress: number) =>
    host.post({ type: "image-progress", id, model, phase, progress, cached })
  const onBytes = (file: ModelFile) => (loaded: number) => {
    got.set(file, loaded)
    let sum = 0
    for (const n of got.values()) sum += n
    report("download", Math.min(sum / total, 1) * (cached ? 0.5 : 0.9))
  }
  const [denoiser, textEncoder, tokenizerJson, paletteJson, selftestJson] = await Promise.all(
    [meta.files.denoiser, meta.files.textEncoder, meta.files.tokenizer, meta.files.palette, meta.files.selftest].map((f) => fileBytes(f, onBytes(f), signal))
  )
  signal.throwIfAborted()
  const text = new TextDecoder()
  const tokenizer = new Tokenizer(JSON.parse(text.decode(tokenizerJson)), { clean_up_tokenization_spaces: false })
  const endToken = tokenizer.token_to_id("<|endoftext|>")
  if (endToken == null) throw new Error("Tessera's tokenizer has no <|endoftext|> token.")
  const palette = (JSON.parse(text.decode(paletteJson)) as { palette_rgba: Array<Array<number>> }).palette_rgba
  if (palette.length !== PALETTE) throw new Error(`Tessera's palette has ${palette.length} colors, not ${PALETTE}.`)
  const selftest = JSON.parse(text.decode(selftestJson)) as { ids: Array<number>; argmax: Array<number> }
  const weights = readTesseraWeights(denoiser)
  const clip = readClipWeights(textEncoder)
  report("load", cached ? 0.6 : 0.92)

  // WebGPU first, where it's allowed and passes the self-test; else WebGL 2.
  const failures: Array<string> = []
  for (const engine of host.webgpu() ? (["webgpu", "webgl"] as const) : (["webgl"] as const)) {
    signal.throwIfAborted()
    let gpu: Gpu | WebGpu | null = null
    let tessera: TesseraGpu | null = null
    try {
      gpu = engine === "webgpu" ? await WebGpu.create() : new Gpu(new OffscreenCanvas(1, 1))
      if (!gpu) {
        failures.push("WebGPU: no adapter")
        continue
      }
      // Room for the weights and a one-canvas pass (the self-test's), before any of it is allocated.
      const sizes = tesseraWeightBytes(weights, clip)
      const weightBytes = sizes.denoiser + sizes.text
      tessera = new TesseraGpu(gpu)
      const claimed = weightBytes + tessera.workBytesFor(1)
      await claim(host, id, model, claimed, signal)
      await tessera.load(weights, clip, signal)
      report("load", cached ? 0.8 : 0.96)
      const agreement = await selftestAgreement(tessera, selftest, signal)
      if (agreement < SELFTEST_AGREEMENT) throw new Error(`its self-test matched ${(agreement * 100).toFixed(1)}% of pixels (at least ${SELFTEST_AGREEMENT * 100}% needed)`)
      lastSelftest = agreement
      const tokenize = (prompt: string) => fixedLengthIds(prompt === "" ? [] : tokenizer.encode(prompt, { add_special_tokens: true }).ids, endToken)
      current = { model, claimed, weightBytes, gpu, tessera, engine, tokenize, endToken, palette }
      report("load", 1)
      return current
    } catch (error) {
      tessera?.dispose()
      gpu?.dispose()
      if (signal.aborted) throw error
      failures.push(`${engine === "webgpu" ? "WebGPU" : "WebGL 2"}: ${error instanceof Error ? error.message : String(error)}`)
    }
  }
  throw new Error(`Tessera couldn't run on this GPU (${failures.join("; ")}).`)
}

/** The share of selftest.json's pixels whose argmax the GPU matches. */
async function selftestAgreement(tessera: TesseraGpu, selftest: { ids: Array<number>; argmax: Array<number> }, signal: AbortSignal) {
  const text = await tessera.encodeText(selftest.ids, signal)
  await tessera.setText(text.states, text.pooled, signal)
  const logits = await tessera.denoise(new Int32Array(PIXELS).fill(MASK), new Float32Array([1]), null, 1, 1, signal)
  let agree = 0
  for (let i = 0; i < PIXELS; i++) {
    let best = 0
    for (let k = 1; k < PALETTE; k++) if (logits[i * PALETTE + k]! > logits[i * PALETTE + best]!) best = k
    if (best === selftest.argmax[i]) agree++
  }
  return agree / PIXELS
}

async function generate(data: Extract<ImageRequest, { type: "image-generate" }>, host: ImageHost, signal: AbortSignal) {
  const model = current?.model === data.model ? current : await load(data.model, data.id, host, signal)
  const { tessera } = model
  // Room for this batch's work textures, before they're made (two rows a sprite: with the prompt and without).
  const needed = model.weightBytes + tessera.workBytesFor(2 * data.n)
  if (needed > model.claimed) {
    await claim(host, data.id, model.model, needed, signal)
    model.claimed = needed
  }
  const started = performance.now()
  const text = await tessera.encodeText(model.tokenize(data.prompt), signal)
  await tessera.setText(text.states, text.pooled, signal)
  const textMs = performance.now() - started
  const tokens = await drawSprites(
    (x, ratio, sc, batch, count) => tessera.denoise(x, ratio, sc, batch, count, signal),
    data.n,
    { ...DEFAULT_SAMPLER, steps: data.steps, guidanceScale: data.guidanceScale, seed: data.seed },
    signal,
    (step) => host.post({ type: "image-step", id: data.id, step: step + 1, steps: data.steps })
  )
  const images: Array<{ rgba: Uint8Array; png?: Uint8Array }> = []
  for (const t of tokens) {
    const rgba = spritePixels(t, model.palette)
    images.push({ rgba, ...(data.png && { png: await encodePng(rgba, GRID, GRID, data.scale) }) })
  }
  const drawMs = performance.now() - started - textMs
  host.post(
    { type: "image-done", id: data.id, images, timings: { textMs, drawMs }, workBytes: tessera.bytes.work },
    images.flatMap((i) => [i.rgba.buffer, ...(i.png ? [i.png.buffer] : [])]) as Array<Transferable>
  )
}
