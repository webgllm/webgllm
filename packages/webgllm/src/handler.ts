import { adaptivePrefixFor } from "./checkpoint-policy.js"
import type { GpuNote } from "./storage.js"
/**
 * The worker's side of webgllm: downloads a model to the Origin Private File
 * System, uploads its weights to the GPU on an OffscreenCanvas, and runs
 * generations one at a time, streaming each one's text back. One model at a
 * time: loading another frees the previous one. WebGLLMWorkerHandler puts it
 * behind a worker's messages (the package's worker.ts, or a caller's).
 */
import { uploadedGgufWeights, canReleaseStaging } from "./upload-staging.js"
import { blobSource } from "./byte-source.js"
import { Engine, MAX_CHUNKS, TUNING } from "./engine.js"
import { MIN_TEXTURE_SIZE } from "./limits.js"
import { GpuError } from "./gpu/gl.js"
import { loadGguf } from "./load-gguf.js"
import { loadOnnx } from "./load-onnx.js"
import { modelMeta, rememberModel } from "./model-meta.js"
import { resolveModel } from "./resolve-model.js"
import { anchorFor, readPrompt, systemPrefixFor, templateFramingFor } from "./prompt.js"
import { repackTime } from "./repack.js"
import type { ModelFile } from "./model-meta.js"
import type { FinishReason, RunTimings, WorkerRequest, WorkerResponse } from "./protocol.js"
import { DraftPolicy, lookupDraft } from "./draft.js"
import { createSampler, draw, drawAt, verify, verifyAt } from "./sampler.js"
import { download, previousGpuTuning, readGpuNotes, saveManifest, storedFile, writeGpuNote } from "./storage.js"
import { loadTokenizer } from "./tokenizer.js"
import type { Tokenizer } from "./tokenizer.js"
import { THINK_OPEN } from "./reply.js"
import { budgetWrapUp, createThinkingBudget, templateClosing } from "./thinking-budget.js"
import { outputTokenLimit } from "./generation-limits.js"
import { createConstraint } from "./constraint/index.js"
import { StructuredGeneration, maskLogits } from "./structured-generation.js"
import { withPromptCachePolicy } from "./prompt-cache-policy.js"
import { planToolChoice } from "./tool-choice.js"
import type { ToolCall } from "./tool-choice.js"
import { createStopFilter } from "./stop-sequences.js"
import { seededUniforms } from "./seeded-random.js"
import { CodedError, downloadError, errorCode } from "./worker-errors.js"
import { createSharedDownloads } from "./shared-download.js"
import { imageProcessorConfig, type ImageProcessorConfig } from "./image.js"
import { loadVision, planVision } from "./load-vision.js"
import {
  chatImages,
  contentId,
  countImageSlots,
  decodeImage,
  expandImageTokens,
  fetchImage,
  ImageCache,
  patchesOf,
  type ImageSlot,
  type PreparedImage
} from "./prompt-images.js"
import { VisionEncoder, visionWeightBytes, visionWorkBytes } from "./vision.js"

const post = (message: WorkerResponse) => postMessage(message)

/**
 * Runs webgllm's models in a Web Worker of your own: construct one in the
 * worker and hand it the worker's messages, then give the `Worker` to
 * `new WebGLLM({ worker })`. One handler per worker; it holds one model.
 *
 * @example
 * ```ts
 * import { WebGLLMWorkerHandler } from "webgllm"
 *
 * const handler = new WebGLLMWorkerHandler()
 * self.onmessage = (event) => handler.onmessage(event)
 * ```
 */
export class WebGLLMWorkerHandler {
  /** A message from the page: `self.onmessage = (event) => handler.onmessage(event)`. */
  onmessage(event: { data: unknown }): void {
    // The worker's own messages come this way too: only webgllm's are taken.
    if (isRequest(event.data)) receive(event.data)
  }
}

const REQUESTS: ReadonlySet<string> = new Set<WorkerRequest["type"]>([
  "load", "generate", "stop", "unload", "download", "frames", "claimed", "tune"
])

/** Whether a message is one of the page's requests (by its type, and the fields every handler reads first). */
function isRequest(data: unknown): data is WorkerRequest {
  if (typeof data !== "object" || data === null) return false
  const { type, model, id } = data as { type?: unknown; model?: unknown; id?: unknown }
  if (typeof type !== "string" || !REQUESTS.has(type)) return false
  if (type === "load" || type === "generate" || type === "unload" || type === "download") {
    if (typeof model !== "string") return false
  }
  if (type === "generate" || type === "stop" || type === "download") return typeof id === "string"
  if (type === "claimed" || type === "unload") return typeof id === "number"
  return true
}

/** Claims the page hasn't answered yet, by id. */
const claims = new Map<number, { model: string; resolve: () => void }>()
let claimIds = 0
/** Models asked to unload and not freed yet: they claim nothing more (the page has stopped listening). */
const leaving = new Set<string>()

/**
 * Asks the page to make room for the model's `bytes` (in place of what it
 * claimed before), and waits until it has. Resolves at once, asking
 * nothing, if `signal` (its load's or generation's) has aborted or the
 * model is being unloaded, and when either happens while it waits: the
 * caller checks its signal after.
 */
function claimMemory(model: string, bytes: number, signal?: AbortSignal) {
  if (signal?.aborted || leaving.has(model)) return Promise.resolve()
  return new Promise<void>((resolve) => {
    const id = ++claimIds
    const settle = () => {
      claims.delete(id)
      signal?.removeEventListener("abort", settle)
      resolve()
    }
    claims.set(id, { model, resolve: settle })
    signal?.addEventListener("abort", settle, { once: true })
    post({ type: "claim", id, model, bytes: Math.round(bytes) })
  })
}

/** A claim again once the textures grow this much past the last one (see generate). */
const RECLAIM_BYTES = 32 * 2 ** 20

/**
 * The textures a model takes loaded, per byte of its weight files: what's
 * claimed before a load (then the textures' bytes, measured). Qwen3.5 2B
 * and LFM3 1.7B take 1.3–1.4× at 512-token batches.
 */
const TEXTURES_PER_FILE_BYTE = 1.4

type Loaded = {
  model: string
  /** The contextLength it was loaded for (the model's own when undefined). */
  contextLength: number | undefined
  engine: Engine
  tokenizer: Tokenizer
  /**
   * What the engine's cache holds, in order: token ids, and for an image's
   * rows its key (see prompt-images.ts), so another image doesn't match.
   */
  cached: Array<number>
  /** Images of recent prompts, with their embedding rows once encoded. */
  images: ImageCache
  /** The image processor's settings, read the first time a prompt has images. */
  imageConfig?: ImageProcessorConfig
  /** Recent verified common content, for conservative short-prefix promotion. */
  previousShared?: number
  /** Milliseconds each phase of the load took. */
  timings?: Record<string, number>
  /** The texture bytes last claimed from the page's memory budget (see claimMemory). */
  claimed: number
  noteKey: string
  note?: GpuNote
}

let current: Loaded | null = null
let pacingEngine: Engine | null = null
let loading: {
  model: string
  contextLength: number | undefined
  promise: Promise<Loaded>
  abort: AbortController
} | null = null
/** A load stopped by an unload, until it has ended: the next load waits for it (see startLoad). */
let stopping: Promise<unknown> | undefined

/**
 * Drafted tokens checked with each written one (speculative decoding by
 * prompt lookup; see draft.ts). A batch of a token and its drafts is read
 * as fast as one token, and the tokens kept are distributed exactly as
 * without drafts.
 */
const DRAFTS = 7
/**
 * Guessing (see Engine.forwardGuess) stops while fewer than this share of
 * the last GUESS_WINDOW guesses were the token drawn: a wrong guess holds
 * the GPU up for the next token.
 */
const GUESS_MIN_HITS = 0.6
const GUESS_WINDOW = 16


/** FNV-1a of a string, as hex (for note keys). */
function hash(text: string) {
  let h = 0x811c9dc5
  for (let i = 0; i < text.length; i++)
    h = Math.imul(h ^ text.charCodeAt(i), 0x01000193)
  return (h >>> 0).toString(16)
}

/** Frees an engine's GPU memory now (see Gpu.dispose). */
const release = (engine: Engine) => {
  if (pacingEngine === engine) pacingEngine = null
  engine.gpu.dispose()
}

/**
 * Prompt tokens per forward batch, the same on every device and model (the
 * engine scales it up for smaller models; TUNING.promptBatch overrides it
 * for benchmarks): bigger batches read prompts faster on every GPU tried
 * (2.4× on a Mac, 15% on a Pixel than 128). 256 or 128 would save 120 or
 * 270 MiB of Qwen3.5 2B's buffers, but LFM3 then reads prompts 32% and 56%
 * slower on an M3, and nothing the GPU reports tells where the room is
 * short (an M3 and an iPhone report the same limits).
 */
const PROMPT_BATCH = 512

class Superseded extends Error {
  constructor() {
    super("Switched to another model.")
  }
}

/** Downloads in progress, shared by a "download" request and a load (see shared-download.ts). */
const downloads = createSharedDownloads<File>()

/**
 * The model's files, downloaded unless they're stored. Progress goes up to
 * `share` (a load's uploading takes the rest). Needs no GPU: a worker may
 * only download.
 */
async function fetchFiles(model: string, signal: AbortSignal, share = 0.9) {
  // The page sends a resolved model's meta; a page that doesn't has it resolved here.
  const meta = await resolveModel(model)
  const weights: Array<ModelFile> =
    meta.files.kind === "gguf"
      ? [meta.files.model]
      : [meta.files.config, meta.files.graph, meta.files.data]
  const template = meta.tokenizer.chatTemplate
    ? [meta.tokenizer.chatTemplate]
    : []
  const all = [
    meta.tokenizer.tokenizer,
    meta.tokenizer.config,
    ...template,
    ...weights
  ]
  const stored = await Promise.all(all.map(storedFile))
  const cached = stored.every(Boolean)
  // Sizes of the files not known up front count as 0 until they're fetched.
  const totals = all.map(() => 0)
  const loaded = all.map(() => 0)
  const total = () => totals.reduce((a, b) => a + b, 0)
  const report = () => {
    const sum = loaded.reduce((a, b) => a + b, 0)
    post({
      type: "progress",
      model,
      progress: total() ? Math.min(sum / total(), 1) * share : 0,
      cached: false,
      loadedBytes: sum,
      totalBytes: total()
    })
  }
  const files = await Promise.all(
    all.map((file, i) =>
      stored[i]
        ? Promise.resolve(stored[i])
        : downloads(
            `${file.repo}/${file.path}`,
            (onProgress, shared) => download(file, onProgress, shared),
            (bytes, size) => {
              loaded[i] = bytes
              totals[i] = Math.max(totals[i], size)
              report()
            },
            signal
          )
    )
  ).catch((error: unknown) => {
    throw downloadError(error, signal)
  })
  // Every file is stored: a model outside the catalog keeps its meta beside them.
  await saveManifest(meta)
  const [tokenizerFile, tokenizerConfigFile, ...rest] = files
  // tokenizer_config.json's template when the model has no file for it.
  const templateFile = template.length ? rest.shift() : undefined
  return {
    meta,
    cached,
    tokenizerFile,
    tokenizerConfigFile,
    templateFile,
    weightFiles: rest
  }
}

/** Attention's largest relative error the self-check passes (right: ~2e-4). */
const SELF_CHECK_MAX = 1e-2

async function load(
  model: string,
  signal: AbortSignal,
  /** Most tokens a chat may hold: the model's own context length when undefined. */
  contextLength: number | undefined,
  /** Without multiple render targets or the chunked recurrence: after a failed self-check. */
  plain = false,
  reuseTuning = true
): Promise<Loaded> {
  const meta = await resolveModel(model)
  // The GPU first: a browser or GPU that can't run the model says so before
  // a gigabyte downloads. (The page's hasWebGl2 checks the page; this is
  // the worker, where the engine runs.)
  if (typeof OffscreenCanvas === "undefined")
    throw new GpuError("This browser has no OffscreenCanvas in workers.")
  const t0 = performance.now()
  const timings: Record<string, number> = {}
  let mark = t0
  const lap = (name: string) => {
    const now = performance.now()
    timings[name] = Math.round(now - mark)
    mark = now
  }
  const engine = new Engine(new OffscreenCanvas(1, 1), {
    batchSize: TUNING.promptBatch > 0 ? TUNING.promptBatch : PROMPT_BATCH,
    ...(contextLength !== undefined && { contextLength }),
    ...(plain && {
      noMrt: ["scaled", "quad", "conv", "chunks"],
      chunkedRecurrence: false
    })
  })
  pacingEngine = engine
  lap("context")
  let reusedTuning = false
  try {
    // Before a gigabyte downloads: the weight textures' width.
    const { maxTextureSize } = engine.gpu.limits
    if (maxTextureSize < MIN_TEXTURE_SIZE) {
      throw new GpuError(
        `This GPU's textures go up to ${maxTextureSize} texels; the model needs ${MIN_TEXTURE_SIZE}.`
      )
    }
    const {
      cached,
      tokenizerFile,
      tokenizerConfigFile,
      templateFile,
      weightFiles
    } = await fetchFiles(model, signal)
    lap("files")
    // Downloads waiting on this load needn't wait for the GPU.
    post({ type: "stored", model })
    post({ type: "progress", model, progress: cached ? 0 : 0.9, cached })
    // Room for it first: the page unloads other runtimes' models past the
    // memory budget. An estimate until it's loaded (then the textures it
    // took): the weights, the cache and the working textures.
    const weightBytes = weightFiles.reduce((sum, f) => sum + f.size, 0)
    if (signal.aborted) throw new Superseded()
    await claimMemory(model, weightBytes * TEXTURES_PER_FILE_BYTE, signal)
    if (signal.aborted) throw new Superseded()
    const tokenizer = loadTokenizer({
      tokenizerJson: JSON.parse(await tokenizerFile.text()) as object,
      tokenizerConfig: JSON.parse(await tokenizerConfigFile.text()) as Record<
        string,
        unknown
      >,
      ...(templateFile && { chatTemplate: await templateFile.text() })
    })
    const scope = [
      engine.gpu.limits.renderer,
      navigator.userAgent,
      model,
      plain ? "plain" : "mrt"
    ].join("|") + "|"
    const settings = hash(JSON.stringify(TUNING))
    const noteKey = `${scope}${Engine.fingerprint}.${settings}`
    const notes = await readGpuNotes()
    const checkedNote = notes[noteKey]
    // A code update may only change scheduling. Keep compatible tuning
    // choices and check the new engine before saving them under its key.
    const note = checkedNote ?? (reuseTuning
      ? previousGpuTuning(notes, scope, settings, SELF_CHECK_MAX)
      : undefined)
    reusedTuning = !checkedNote && !!note
    timings.calibrationCached = checkedNote ? 1 : 0
    timings.calibrationReused = reusedTuning ? 1 : 0
    timings.engineCode = parseInt(Engine.fingerprint, 16)
    if (note) {
      engine.preferLayout = note.layout
      engine.headChoice = note.head ?? null
      if (note.tuned) engine.tuned = { ...note.tuned }
    }
    const pieces =
      meta.files.kind === "gguf"
        ? uploadedGgufWeights(loadGguf(blobSource(weightFiles[0]), {
            embeddingChunkRows: TUNING.ggufEmbeddingRows > 0 ? TUNING.ggufEmbeddingRows : canReleaseStaging ? undefined : 512,
            readAhead: canReleaseStaging ? TUNING.ggufReadAhead : Math.min(3, TUNING.ggufReadAhead),
            releaseReadBuffers: canReleaseStaging
          }))
        : loadOnnx({
            config: JSON.parse(await weightFiles[0].text()) as Record<
              string,
              unknown
            >,
            graph: blobSource(weightFiles[1]),
            data: blobSource(weightFiles[2])
          })
    let layers = 0
    await engine.load(pieces, (piece) => {
      if (signal.aborted) throw new Superseded()
      if (piece.kind === "config") layers = piece.config.numLayers + 2
      if (
        piece.kind === "layer" ||
        piece.kind === "embedding" ||
        piece.kind === "finalNorm"
      ) {
        const count = piece.kind === "layer" ? piece.layer + 1 : layers
        post({
          type: "progress",
          model,
          progress:
            (cached ? 0 : 0.9) +
            (cached ? 1 : 0.1) * Math.min(count / layers, 1),
          cached
        })
      }
    }, signal)
    if (signal.aborted) throw new Superseded()
    lap("weights")
    timings.headExpanded = engine.headChoice?.expanded ? 1 : 0
    timings.headSplit = engine.headChoice?.split ?? TUNING.q6HeadSplit
    timings.promptPacking = engine.tuned.f16Pack ?? TUNING.f16Pack & 3
    post({
      type: "log",
      text: `load tuning: ${JSON.stringify({ head: engine.headChoice, headReport: engine.headReport, packing: engine.packingReport, rms: engine.rmsReport, matvec: engine.matvecReport })}`
    })
    timings.weightsReading = Math.round(engine.loadMs.reading)
    timings.weightsUploading = Math.round(engine.loadMs.uploading)
    timings.f16Init = Math.round(engine.loadMs.scratch)
    timings.packingProbe = Math.round(engine.loadMs.packing)
    timings.matvecProbe = Math.round(engine.loadMs.matvec)
    timings.packingProbeRows = engine.packingReport[0]?.rows ?? 0
    for (const sample of engine.rmsReport) timings[`rmsProbe${sample.parts}Ms`] = Math.round(sample.ms)
    timings.rmsParts = engine.tuned.rmsParts ?? TUNING.rmsParts
    for (const sample of engine.packingReport)
      timings[`packingProbe${sample.matmul == null ? "" : `Mode${sample.matmul}`}Pack${sample.pack}Ms`] = Math.round(sample.ms)
    timings.toRgba4 = Math.round(engine.loadMs.convert)
    timings.repackOnnx = Math.round(repackTime.ms)
    timings.repackGgufQ4 = Math.round(repackTime.ggufQ4Ms)
    timings.repackGgufQ8 = Math.round(repackTime.ggufQ8Ms)
    timings.repackGgufQ6 = Math.round(repackTime.ggufQ6Ms)
    for (const key of Object.keys(repackTime) as Array<keyof typeof repackTime>) repackTime[key] = 0
    if (engine.layoutChoice?.ms)
      timings.layoutProbe = Math.round(engine.layoutChoice.ms)
    if (engine.layoutChoice) { timings.layoutRgba4DecodeMs = engine.layoutChoice.rgba4Ms; timings.layoutBytesDecodeMs = engine.layoutChoice.bytesMs }
    if (engine.layoutChoice?.rgba4PrefillMs != null) timings.layoutRgba4PrefillMs = engine.layoutChoice.rgba4PrefillMs
    if (engine.layoutChoice?.bytesPrefillMs != null) timings.layoutBytesPrefillMs = engine.layoutChoice.bytesPrefillMs
    // Resolve all kernel choices during initial loading, then warm the
    // selected variants. No calibration work races with user generations.
    if (!note?.calibrated) {
      // tune warms its measured variants itself. Warm the final choices
      // once, instead of running a second complete startup warm-up.
      await engine.tune(() => signal.aborted, { alreadyBusy: true })
      if (signal.aborted) throw new Superseded()
      post({ type: "log", text: `startup tuning: ${JSON.stringify(engine.tuneReport)}` })
      lap("initialTuning")
    }
    await engine.warmUp()
    lap("warmUp")
    timings.programs = engine.gpu.compiled.count
    timings.compileMs = Math.round(engine.gpu.compiled.ms)
    for (const [k, v] of Object.entries(engine.warmCounts))
      timings[`warm_${k}`] = v
    if (signal.aborted) throw new Superseded()
    timings.tunedShortSplit = engine.tuned.shortSplit ?? 0
    timings.tunedF16Matmul = engine.tuned.f16Matmul ?? -1
    timings.tunedF16Pack = engine.tuned.f16Pack ?? -1
    timings.tunedFlashDecode = engine.tuned.flashDecode ?? -1
    // Attention against the CPU's (and Qwen3.5's chunked recurrence against
    // a token at a time): a GPU whose compiler gets a kernel wrong
    // tries again with the simplest passes, then says so (not gibberish).
    // A pass is remembered (same GPU, browser, model and code).
    const error =
      checkedNote && checkedNote.selfCheck <= SELF_CHECK_MAX
        ? checkedNote.selfCheck
        : await engine.selfCheck()
    lap("selfCheck")
    const measured =
      engine.layoutChoice && error <= SELF_CHECK_MAX
        ? {
            layout: engine.layoutChoice.layout,
            selfCheck: error,
            calibrated: true,
            pacing: note?.pacing,
            head: engine.headChoice ?? undefined,
            tuned: { ...engine.tuned }
          }
        : null
    timings.total = Math.round(performance.now() - t0)
    if (signal.aborted) throw new Superseded()
    if (!(error <= SELF_CHECK_MAX)) {
      if (reusedTuning) throw new GpuError("Previous GPU tuning failed validation.")
      if (plain) {
        throw new GpuError(
          `This GPU computes attention wrong (relative error ${error.toExponential(1)}).`
        )
      }
      release(engine)
      return await load(model, signal, contextLength, true)
    }
    engine.gpu.resetPacing(true)
    engine.gpu.setPhase("load")
    engine.gpu.allowPacingRecovery()
    const overlap = engine.tuned.submissionOverlap
    if (overlap != null && Number.isFinite(overlap) && overlap >= 0 && overlap <= 2 ** 34)
      engine.gpu.primePresentation(engine.tuned.submissionWindow ?? TUNING.presentationFrames, Math.max(TUNING.submissionReads, overlap), overlap > 0)
    const pacing = overlap === 0 ? undefined : note?.pacing
    if (pacing && Number.isFinite(pacing.reads) && pacing.reads > 0 && pacing.reads <= 2 ** 34 && pacing.window >= 2 && pacing.window <= 128 && pacing.frameMs >= 4 && pacing.frameMs <= 100 && pacing.window * pacing.frameMs <= 450) {
      const scale = Math.min(1, engine.gpu.expectedFrameMs / pacing.frameMs)
      const window = Math.max(TUNING.presentationFrames, Math.min(engine.gpu.presentationAdvice.cap, Math.round(pacing.window * pacing.frameMs / engine.gpu.expectedFrameMs)))
      engine.gpu.primePresentation(window, Math.max(TUNING.submissionReads, Math.round(pacing.reads * scale)))
    }
    // What it took, in place of the estimate.
    await claimMemory(model, engine.gpu.bytes, signal)
    if (signal.aborted) throw new Superseded()
    if (measured && JSON.stringify(measured) !== JSON.stringify(checkedNote))
      void writeGpuNote(noteKey, measured)
    return {
      model,
      contextLength,
      engine,
      tokenizer,
      cached: [],
      images: new ImageCache(),
      claimed: engine.gpu.bytes,
      noteKey,
      note: measured ?? note,
      timings
    }
  } catch (error) {
    // A failed or superseded load frees its textures now.
    release(engine)
    if (reusedTuning && !signal.aborted && error instanceof GpuError)
      return load(model, signal, contextLength, plain, false)
    throw error
  }
}

/**
 * Loads `model` for chats of up to `contextLength` tokens unless it's loaded
 * (or loading) for them, replacing any other. Rejects with Superseded if
 * another load replaces this one first.
 */
function startLoad(
  model: string,
  contextLength: number | undefined
): Promise<Loaded> {
  if (
    current?.model === model &&
    current.contextLength === contextLength &&
    !current.engine.gpu.isLost
  ) {
    return Promise.resolve(current)
  }
  if (loading?.model === model && loading.contextLength === contextLength) {
    return loading.promise
  }
  // Stop a load of another model, and free the loaded one. The new load
  // starts once the stopped one has ended: two never run at once (their
  // startup probes set worker-wide tuning, and each holds GPU memory).
  const previous = loading?.promise ?? stopping
  loading?.abort.abort()
  if (current) {
    release(current.engine)
    current = null
  }
  const abort = new AbortController()
  const run = async () => {
    await previous?.catch(() => undefined)
    if (abort.signal.aborted) throw new Superseded()
    return load(model, abort.signal, contextLength)
  }
  const entry: NonNullable<typeof loading> = {
    model,
    contextLength,
    abort,
    promise: run().then(
      (loaded) => {
        if (loading !== entry) {
          release(loaded.engine)
          throw new Superseded()
        }
        loading = null
        current = loaded
        return loaded
      },
      (error: unknown) => {
        if (loading !== entry) throw new Superseded()
        loading = null
        throw error
      }
    )
  }
  loading = entry
  return entry.promise
}

/**
 * Frees `model` and stops a load of it, leaving the worker running: what
 * the page asks of a worker it didn't start (one it started, it ends). The
 * loaded model goes after the generations asked for before this.
 */
function unload(model: string, id: number) {
  leaving.add(model)
  if (loading?.model === model) {
    const ended = loading.promise.catch(() => undefined)
    stopping = ended
    void ended.then(() => {
      if (stopping === ended) stopping = undefined
    })
    loading.abort.abort()
    // (Its promise rejects as Superseded, and frees what it loaded.)
    loading = null
  }
  // The page has stopped listening to this model: nothing answers its claims.
  for (const claim of [...claims.values()]) if (claim.model === model) claim.resolve()
  const stopped = stopping
  queue = queue.then(async () => {
    if (current?.model === model) {
      release(current.engine)
      current = null
    }
    // A load of it that was stopped has freed its textures once it ends.
    await stopped
    leaving.delete(model)
    post({ type: "unloaded", id })
  })
}

/**
 * Downloads `model`'s files to storage without loading it: no GL context,
 * so a worker spawned just for this works where WebGL 2 doesn't.
 */
async function downloadModel(model: string, id: string) {
  // Its "stop" aborts it; a load or another download of the same files goes on (see shared-download.ts).
  const signal = abortFor(id)
  try {
    const { cached } = await fetchFiles(model, signal, 1)
    post({ type: "progress", model, progress: 1, cached })
    post({ type: "downloaded", model, id })
  } catch (error) {
    // (Stopped: sent once the files' writers have closed.)
    post({ type: "download-error", model, id, message: errorMessage(error), code: errorCode(error) })
  } finally {
    stopped.delete(id)
    aborts.delete(id)
  }
}

const stopped = new Set<string>()
/** Each generation's abort, signalled by its "stop" (image fetches, downloads and encoding take it). */
const aborts = new Map<string, AbortController>()
/** Generation `id`'s abort signal (aborted already if it was stopped). */
function abortFor(id: string): AbortSignal {
  let abort = aborts.get(id)
  if (!abort) {
    abort = new AbortController()
    aborts.set(id, abort)
  }
  if (stopped.has(id)) abort.abort()
  return abort.signal
}

const errorMessage = (error: unknown) =>
  error instanceof GpuError
    ? `The GPU failed: ${error.message}`
    : error instanceof Error
      ? error.message
      : String(error)

/** One of a model's files: stored, or downloaded now. */
async function modelFile(file: ModelFile, signal?: AbortSignal): Promise<File> {
  return (
    (await storedFile(file)) ??
    downloads(`${file.repo}/${file.path}`, (onProgress, shared) => download(file, onProgress, shared), () => undefined, signal).catch(
      (error: unknown) => {
        throw downloadError(error, signal ?? new AbortController().signal)
      }
    )
  )
}

/**
 * The prompt's images, prepared (fetched and resized) or found among
 * recent ones, and the prompt with each image slot expanded to its image's
 * tokens, with the cache's keys (see prompt-images.ts). Images are taken
 * one at a time, each checked against the context before its pixels are
 * made, so a prompt that can't fit fails before it holds them all. With
 * `reuse` false (cache_prompt: false), images are shared within the request
 * only, and none is kept for later ones.
 */
async function promptImages(model: Loaded, prompt: Array<number>, chat: GenerateRequest["chat"], reuse: boolean, signal: AbortSignal) {
  const wanted = chatImages(chat)
  if (!wanted.length) return { prompt, keys: prompt, images: [] as Array<{ start: number; image: PreparedImage }> }
  const files = (await resolveModel(model.model, signal)).files
  if (files.kind !== "onnx" || !files.vision) throw new Error(`${model.model} doesn't read images.`)
  const { tokenizer, engine } = model
  const slot = {
    start: tokenizer.tokenToId("<|vision_start|>"),
    pad: tokenizer.tokenToId("<|image_pad|>"),
    end: tokenizer.tokenToId("<|vision_end|>")
  }
  if (slot.start == null || slot.pad == null || slot.end == null) throw new Error("The tokenizer has no image tokens.")
  const imageSlot = slot as ImageSlot
  const slots = countImageSlots(prompt, imageSlot)
  if (slots !== wanted.length) throw new Error(`The prompt has ${slots} image slots for ${wanted.length} images.`)
  model.imageConfig ??= imageProcessorConfig(JSON.parse(await (await modelFile(files.vision.preprocessor, signal)).text()))
  const config = model.imageConfig
  const maxPatches = engine.gpu.limits.maxTextureSize
  // Tokens so far: the text, and each image's (its slot's pad stands in until it's sized).
  let tokens = prompt.length - slots
  const fits = (more: number) => {
    tokens += more
    if (tokens >= engine.contextLength)
      throw new CodedError(
        `This session is too long for the model: its images alone take it past ${engine.contextLength} tokens, the most it can read, reply included. Send fewer or smaller images (detail: "low"), or start a new session.`,
        "context-length-exceeded"
      )
  }
  const local = new Map<string, PreparedImage>()
  const prepared: Array<PreparedImage> = []
  for (const image of wanted) {
    signal.throwIfAborted()
    const urlId = `${image.detail ?? "auto"}|${image.url}`
    let found = local.get(urlId) ?? (reuse && image.url.startsWith("data:") ? model.images.getUrl(image) : undefined)
    let content: string | undefined
    let blob: Blob | undefined
    if (!found) {
      blob = await fetchImage(image.url, signal)
      content = await contentId(blob, image.detail)
      found = local.get(content) ?? (reuse ? model.images.get(content) : undefined)
    }
    if (found) fits(found.height * found.width)
    else {
      found = await decodeImage(blob!, image.detail, config, maxPatches, fits, signal)
      if (reuse) model.images.set(image, content!, found)
    }
    local.set(urlId, found)
    if (content) local.set(content, found)
    prepared.push(found)
  }
  const expanded = expandImageTokens(prompt, imageSlot, prepared)
  return { prompt: expanded.tokens, keys: expanded.keys, images: expanded.starts.map((start, i) => ({ start, image: prepared[i] })) }
}

/**
 * Encodes the images whose rows the prompt reads from `start` on (the
 * cache holds the rest), loading the vision encoder for them and freeing
 * it after: text-only prompts never load it. Returns the ms the load took
 * (its download included; 0 without one).
 */
async function encodeImages(model: Loaded, images: Array<{ start: number; image: PreparedImage }>, start: number, signal: AbortSignal) {
  const needed = images.filter(
    ({ start: at, image }) => at + image.height * image.width > start && !image.features
  )
  if (!needed.length) return 0
  const loadAt = performance.now()
  const meta = (await resolveModel(model.model)).files
  if (meta.kind !== "onnx" || !meta.vision) throw new Error(`${model.model} doesn't read images.`)
  const [graph, data] = await Promise.all([modelFile(meta.vision.graph, signal), modelFile(meta.vision.data, signal)])
  const { engine } = model
  // Room for the encoder's weights and its largest image's work while it's
  // loaded, sized from the graph as the encoder allocates them.
  const graphBytes = new Uint8Array(await graph.arrayBuffer())
  const { config } = planVision(graphBytes)
  const maxTexture = engine.gpu.limits.maxTextureSize
  const work = Math.max(
    ...needed.map(({ image }) => visionWorkBytes(config, image.height * config.merge, image.width * config.merge, maxTexture))
  )
  const encoder = new VisionEncoder(engine.gpu)
  try {
    signal.throwIfAborted()
    await claimMemory(model.model, engine.gpu.bytes + visionWeightBytes(config) + work, signal)
    signal.throwIfAborted()
    await encoder.load(loadVision({ graph: graphBytes, data: blobSource(data) }), signal)
    if (encoder.config.outWidth !== engine.config.hiddenSize)
      throw new Error(`The vision encoder writes ${encoder.config.outWidth}-wide rows; the model reads ${engine.config.hiddenSize}.`)
    const loadMs = performance.now() - loadAt
    for (const { image } of needed) {
      signal.throwIfAborted()
      image.features = await encoder.encode(patchesOf(image, model.imageConfig!), signal)
      image.pixels = null
    }
    return loadMs
  } finally {
    encoder.dispose()
    void claimMemory(model.model, model.claimed)
  }
}

type GenerateRequest = Extract<WorkerRequest, { type: "generate" }>

async function generate(request: GenerateRequest) {
  const { id } = request
  const requested = performance.now()
  const empty: RunTimings = {
    inputTokens: 0,
    cachedTokens: 0,
    outputTokens: 0,
    firstTokenMs: null,
    promptMs: 0,
    decodeMs: 0
  }
  if (stopped.has(id)) return post({ type: "done", id, stats: empty, finishReason: "stop" })
  // The model this request is for: loaded again if its GPU context was
  // lost, or loaded if another was asked for since.
  const model = await startLoad(request.model, request.contextLength)
  if (stopped.has(id)) return post({ type: "done", id, stats: empty, finishReason: "stop" })
  const calibration = model.engine.gpu.calibratePromptPacing()
  if (calibration) {
    await calibration
    if (stopped.has(id)) return post({ type: "done", id, stats: empty, finishReason: "stop" })
  }
  return withPromptCachePolicy(model, request.cachePrompt, async () => {
  if (request.reportPhases) post({ type: "phase", id, phase: "processing" })
  const { engine, tokenizer } = model
  const compiledBefore = engine.gpu.compiled.count
  const toolPlan = planToolChoice(request.tools, request.toolChoice)
  const tools = toolPlan.tools.length ? toolPlan.tools : undefined
  const forced = toolPlan.mode === "forced" ? toolPlan : undefined
  // A forced call is the reply: its arguments' schema takes the answer's place.
  const answerSchema = forced ? forced.schema : request.outputSchema
  const tokenizeAt = performance.now()
  const encoded = tokenizer.encodeChat(request.chat, {
    tools,
    ...(request.thinking !== undefined && { enableThinking: request.thinking }),
    addGenerationPrompt: true
  })
  // Each image's token expanded to its tokens (the image fetched and
  // resized here); `keys` is what the cache compares.
  const imagesAt = performance.now()
  const signal = abortFor(id)
  let prepared: Awaited<ReturnType<typeof promptImages>>
  try {
    prepared = await promptImages(model, encoded, request.chat, request.cachePrompt !== false, signal)
  } catch (error) {
    // Stopped while its images were fetched or resized: the cache is as it was.
    if (stopped.has(id)) return post({ type: "done", id, stats: empty, finishReason: "stop" })
    throw error
  }
  const { prompt, keys, images } = prepared
  const prepareImagesMs = performance.now() - imagesAt
  const cacheEnabled = request.cachePrompt !== false
  const prefix = cacheEnabled ? systemPrefixFor(tokenizer, prompt, request.chat, request.thinking, tools) : undefined
  const anchor = cacheEnabled ? anchorFor(tokenizer, prompt, request.chat, request.thinking, tools) : undefined
  const tokenizeMs = performance.now() - tokenizeAt - prepareImagesMs
  const think = tokenizer.specialTokens.think
  const last = prompt[prompt.length - 1]
  const startsInThinking = think !== null &&
    (last === think ||
      (prompt[prompt.length - 2] === think &&
        tokenizer.decode([last]).trim() === ""))
  const thinkingBudget = createThinkingBudget(
    request.thinkingBudgetTokens,
    tokenizer.specialTokens,
    startsInThinking,
    request.thinkingBudgetTokens === undefined ? undefined : templateClosing(tokenizer),
    request.thinkingBudgetTokens === undefined ? undefined : budgetWrapUp(tokenizer)
  )
  // LFM2.5 Thinking writes its opening marker itself; its template ends at
  // assistant\n. Seed that expected marker so the answer grammar never suppresses reasoning.
  const reasoning = modelMeta(request.model).reasoning
  const mandatoryThinking = !!reasoning && "levels" in reasoning && !("off" in reasoning.levels)
  const openThinking = answerSchema !== undefined && !startsInThinking && think !== null &&
    (request.thinking === true || (request.thinking === undefined && mandatoryThinking))
  const schemaAt = performance.now()
  const structured = answerSchema === undefined ? undefined : new StructuredGeneration(
    await createConstraint({ schema: answerSchema, tokenizer: tokenizer.constraintTokenizer }),
    tokenizer.specialTokens.eos,
    tokenizer.specialTokens.thinkEnd,
    startsInThinking || openThinking,
    tokenizer.vocabSize
  )
  const schemaMs = performance.now() - schemaAt
  try {
  if (stopped.has(id)) return post({ type: "done", id, stats: empty, finishReason: "stop" })
  engine.setAllowedTokens(null)
  const room = engine.contextLength - prompt.length
  if (room <= 0) {
    throw new CodedError(
      `This session is too long for the model: ${prompt.length} tokens, and it can read ${engine.contextLength} at most, reply included. Start a new session to keep going.`,
      "context-length-exceeded"
    )
  }

  const maxTokens = outputTokenLimit(room, request.maxTokens)

  // Reuse the cache for the longest prefix this prompt shares with it. The
  // whole prompt only if its end was saved with its last hidden row (the
  // same prompt again: a retry); else the last token is read again, for
  // the logits after it.
  const restoreAt = performance.now()
  let common = 0
  while (
    common < model.cached.length &&
    common < prompt.length &&
    model.cached[common] === keys[common]
  ) {
    common++
  }
  const whole = common === prompt.length && engine.canResume(common)
  const start = whole
    ? engine.resume(common)
    : engine.rewind(Math.min(common, prompt.length - 1))
  model.cached.length = start
  let natural: number | undefined
  if (cacheEnabled && common > 0 && !whole && prefix !== undefined && prefix > start) {
    let at = start
    for (const size of engine.batchSizes(prompt.length - start)) { at += size; if (at > prefix) break; natural = at }
  }
  const framing = cacheEnabled && common > 0 && !whole
    ? templateFramingFor(tokenizer, prompt, request.chat[0]?.role ?? "user", request.thinking) : 0
  const plan = adaptivePrefixFor({ shared: common, declared: prefix, retained: engine.snapshotPosition("prefix"), previousShared: model.previousShared, natural, framing })
  if (!whole) model.previousShared = plan.previousShared
  const cacheRestoreMs = performance.now() - restoreAt

  // Read the rest of the prompt (see readPrompt), with the anchor snapshot
  // where the next question's prompt will stop sharing it (see anchorFor).
  // First the images' embedding rows where the prompt reads them (the
  // vision encoder loaded for that only), and where every image is.
  const encodeAt = performance.now()
  let visionLoadMs: number
  try {
    visionLoadMs = await encodeImages(model, images, start, signal)
  } catch (error) {
    // Stopped while the encoder downloaded, loaded or ran: its textures and
    // claim are freed, and the cache holds the prompt up to `start`.
    if (stopped.has(id)) return post({ type: "done", id, stats: { ...empty, inputTokens: prompt.length, cachedTokens: start }, finishReason: "stop" })
    throw error
  }
  engine.setImages(images.map(({ start: at, image }) => ({ start: at, height: image.height, width: image.width, features: image.features })))
  const encodeMs = performance.now() - encodeAt
  const promptStart = performance.now()
  const reusablePrefix = plan.prefix
  if (!stopped.has(id)) await engine.preparePrompt(prompt.length)
  let row = await readPrompt(engine, prompt, start, model.cached, {
    keys,
    stop: () => stopped.has(id),
    // After a changed request reveals a shorter common boundary, retain it
    // during this ordinary forward. One-off cold calls incur no extra work.
    prefix: reusablePrefix,
    ...(cacheEnabled && reusablePrefix !== undefined && reusablePrefix > start && reusablePrefix <= common && common > 0 && { observedPrefix: reusablePrefix }),
    anchor,
    checkpoints: cacheEnabled
  })
  const stats: RunTimings = {
    inputTokens: prompt.length,
    cachedTokens: start,
    outputTokens: 0,
    firstTokenMs: null,
    promptMs: performance.now() - promptStart,
    tokenizeMs,
    schemaMs,
    ...(images.length && { imagesMs: prepareImagesMs + encodeMs, visionLoadMs }),
    cacheRestoreMs,
    cacheMatchedTokens: common,
    cachePrefixTokens: engine.snapshotPosition("prefix") ?? 0,
    cacheAnchorTokens: engine.snapshotPosition("anchor") ?? 0,
    decodeMs: 0,
    ...thinkingBudget?.stats
  }
  if (
    stopped.has(id) ||
    maxTokens <= 0 ||
    model.cached.length < prompt.length
  ) {
    if (structured && !stopped.has(id)) throw new Error("Structured output was truncated before generation could produce a schema-valid answer.")
    return post({ type: "done", id, stats, finishReason: maxTokens <= 0 && !stopped.has(id) ? "length" : "stop" })
  }

  const { sampling } = request
  // With top-k sampling (LFM3's card: 50), the GPU applies the repetition
  // penalty and picks the candidates, and only those are read back. Ids
  // past the tokenizer's vocabulary are padding rows, never drawn.
  const vocab = tokenizer.vocabSize
  // Greedy (temperature 0, or top_k 1): the GPU picks the token itself and
  // only its id is read back (see Engine.argmax).
  const greedy = sampling.temperature <= 0 || sampling.top_k === 1
  const onGpu = greedy || (sampling.top_k > 0 && sampling.top_k <= MAX_CHUNKS)
  const sampler = createSampler(
    onGpu ? { ...sampling, repetition_penalty: 1, presence_penalty: 0 } : sampling
  )
  // The repetition penalty counts the prompt's tokens and the reply's, as
  // Transformers.js does: the model cards' values assume it. (The presence
  // penalty counts only the reply's.) setSeen also bans the padding ids.
  if (onGpu) engine.setSeen(prompt, vocab)
  else sampler.see(prompt)
  // With a seed, the number each token is drawn with comes from the seed
  // and the token's place in the reply (see seeded-random.ts); else
  // Math.random, as before. Greedy picks don't depend on either.
  const uniform = request.seed === undefined ? undefined : seededUniforms(request.seed)
  const pick = (d: { ids: Int32Array; p: Float32Array }) =>
    uniform ? drawAt(d, uniform(stats.outputTokens)) : draw(d)
  const check = (d: { ids: Int32Array; p: Float32Array }, drafted: number) =>
    uniform ? verifyAt(d, drafted, uniform(stats.outputTokens)) : verify(d, drafted)
  // The likeliest token is read forward while each token is drawn (see
  // Engine.forwardGuess), while that's usually the token drawn.
  const guesses: Array<boolean> = []
  const guessing = () =>
    onGpu &&
    request.pipeline !== false &&
    (guesses.length < GUESS_WINDOW / 2 ||
      guesses.filter(Boolean).length >= GUESS_MIN_HITS * guesses.length)
  // The token guessed with the last distribution, or -1.
  let guessed = -1
  // While guessing is off, what it would have guessed (the likeliest
  // token) is still scored against the token drawn, so it comes back on
  // when guesses would hit again.
  const scoreGuess = (
    d: { ids: Int32Array; p: Float32Array },
    drawn: number
  ) => {
    if (!onGpu || guessing()) return
    let best = 0
    for (let j = 1; j < d.p.length; j++) if (d.p[j] > d.p[best]) best = j
    guesses.push(d.ids[best] === drawn)
    if (guesses.length > GUESS_WINDOW) guesses.shift()
  }
  // How many drafts to check with each token, by what they've cost and
  // earned (see DraftPolicy).
  const policy = new DraftPolicy(DRAFTS)
  // Drafts for the reply so far, at most `max`: as many of those found as
  // the policy expects to pay.
  const draftsFor = (max: number) => {
    // (Gated DeltaNet models read one token at a time: see Engine.canDraft.)
    if (request.speculative === false || max <= 0 || !engine.canDraft) return []
    const found = lookupDraft(history, max, TUNING.draftMatch)
    return found.slice(0, policy.choose(found))
  }
  // Whether a draft batch would follow (without changing what the policy
  // remembers).
  const draftsNext = () => {
    if (request.speculative === false || !engine.canDraft) return false
    const found = lookupDraft(history, DRAFTS, TUNING.draftMatch)
    return found.length > 0 && policy.peek(found) > 0
  }
  // The distribution of the token after `r`, a row of the last batch, as
  // token ids and probabilities; with `guess`, the likeliest token is read
  // forward meanwhile.
  let appliedMask: Uint32Array | null | undefined
  // Thinking switched off: the template closed it (LFM3's `<think></think>`)
  // or left it out, and the reply can't open it again. A repetition penalty
  // on the prompt's markers used to hide LFM3 reopening it.
  const markers = [tokenizer.specialTokens.think, tokenizer.specialTokens.thinkEnd].filter((t): t is number => t !== null && t >= 0 && t < vocab)
  let thinkingOff: Uint32Array | null = null
  if (request.thinking === false && markers.length) {
    thinkingOff = new Uint32Array(Math.ceil(vocab / 32)).fill(0xffffffff)
    for (const t of markers) thinkingOff[t >>> 5] &= ~(1 << (t & 31))
  }
  const updateAllowed = () => {
    // JSON literals are governed by their grammar. Only unconstrained replies
    // reserve control markers: with thinking off, or after an explicit
    // budgeted reasoning span closes.
    const mask = structured ? structured.mask() : thinkingOff ?? thinkingBudget?.closedControlMask(vocab) ?? null
    if (structured?.answering || appliedMask !== mask) engine.setAllowedTokens(mask)
    appliedMask = mask
    return mask ?? undefined
  }
  // No penalties inside a constrained answer: its quotes, keys and the values
  // it copies from the prompt repeat by design.
  const penalize = () => !structured?.answering
  const penalty = () => penalize() ? sampling.repetition_penalty : 1
  const presence = () => penalize() ? sampling.presence_penalty : 0
  const distribution = async (r: number, guess = false) => {
    const mask = updateAllowed()
    if (greedy) {
      const [token] = await engine.argmax(
        r,
        1,
        penalty(),
        [],
        guess,
        presence()
      )
      if (guess && engine.guessing) guessed = engine.guessedToken
      return { ids: Int32Array.of(token), p: Float32Array.of(1) }
    }
    if (onGpu) {
      const { ids, logits } = await engine.candidates(
        r,
        sampling.top_k,
        penalty(),
        guess,
        presence()
      )
      if (guess && engine.guessing) guessed = engine.guessedToken
      const d = sampler.distribution(logits)
      return { ids: Int32Array.from(d.ids, (i) => ids[i]), p: d.p }
    }
    const logits = await engine.logits(r)
    if (logits.length > vocab) logits.fill(-Infinity, vocab)
    if (mask) maskLogits(logits, mask)
    return sampler.distribution(logits, penalize())
  }
  // On the GPU: the distributions after rows 0..rows-1 of a batch whose
  // inputs after the first are `drafts`, in one pass and one readback.
  const distributions = async (
    rows: number,
    drafts: Array<number>,
    guess: boolean,
    masks?: readonly Uint32Array[]
  ) => {
    if (masks) engine.setAllowedTokenRows(masks)
    else updateAllowed()
    if (greedy) {
      const tokens = await engine.argmax(
        0,
        rows,
        penalty(),
        drafts,
        guess,
        presence()
      )
      if (guess && engine.guessing) guessed = engine.guessedToken
      return Array.from(tokens, (token) => ({
        ids: Int32Array.of(token),
        p: Float32Array.of(1)
      }))
    }
    const picked = await engine.pick(
      0,
      rows,
      sampling.top_k,
      penalty(),
      drafts,
      guess,
      0,
      presence()
    )
    if (guess && engine.guessing) guessed = engine.guessedToken
    return picked.map(({ ids, logits }) => {
      const d = sampler.distribution(logits)
      return { ids: Int32Array.from(d.ids, (i) => ids[i]), p: d.p }
    })
  }

  // Stop sequences end the answer (see stop-sequences.ts); a forced call's
  // JSON isn't text: it's kept, and sent as the call (see "done").
  // Neither applies inside a constrained answer: the schema ends it, and a
  // stop string cut from it would leave JSON that fails the schema.
  const stopFilter = createStopFilter(request.stop)
  let stopHit = false
  let sawEos = false
  let callText = ""
  let unfiltered = false
  /** Sends reply text; false once a stop sequence ended the reply. */
  const send = (text: string, constrained = false) => {
    if (constrained && stopFilter && !unfiltered) {
      // What the filter holds back (whitespace after the thinking) first.
      const held = stopFilter.flush()
      if (held) post({ type: "token", id, text: held })
      unfiltered = true
    }
    if (!stopFilter || unfiltered) {
      post({ type: "token", id, text })
      return true
    }
    const out = stopFilter.push(text)
    if (out.text) post({ type: "token", id, text: out.text })
    stopHit = out.stopped
    return !stopHit
  }
  // When the template opens the thinking itself (LFM3 with thinking on ends
  // the prompt with `<think>`, Qwen3.5 with `<think>\n`), the reply starts
  // inside it: say so first, as the model would have, so the thinking
  // reads as thinking.
  if (startsInThinking) send(THINK_OPEN)

  // A zero budget may close a template-opened span before any sampling.
  let token = openThinking ? think as number : thinkingBudget?.forcedToken ?? pick(await distribution(row, guessing()))
  stats.promptMs = performance.now() - promptStart
  if (request.reportPhases && !stopped.has(id)) {
    post({ type: "phase", id, phase: "generating" })
  }
  const firstTokenAt = performance.now()
  stats.firstTokenMs = firstTokenAt - requested
  const decoder = tokenizer.createStreamDecoder({ skipSpecialTokens: false })
  const eos = tokenizer.specialTokens.eos
  // Prompt and reply so far, where drafts are looked up.
  const history = prompt.slice()
  // Writes out a token of the reply; false once the reply is over.
  const emit = (t: number) => {
    stats.outputTokens++
    // (Whether this token is the answer's, before it may close the thinking.)
    const constrained = !!structured?.answering
    const calling = !!forced && constrained
    structured?.observe(t)
    if (t === eos) {
      sawEos = true
      return false
    }
    thinkingBudget?.observe(t)
    if (onGpu) engine.markSeen(t)
    else sampler.see([t], true)
    policy.drawn(t)
    history.push(t)
    const text = decoder.push(t)
    if (text) {
      if (calling) callText += text
      else if (!send(text, constrained)) return false
    }
    return stats.outputTokens < maxTokens
  }
  let going = emit(token)
  while (going && !stopped.has(id)) {
    const stepStart = performance.now()
    if (engine.guessing) {
      const hit = guessed === token
      guesses.push(hit)
      if (guesses.length > GUESS_WINDOW) guesses.shift()
      if (hit) {
        // Already read: draw the next token (and guess the one after).
        engine.keepGuess()
        model.cached.push(token)
        // Guess the next token unless drafts follow (then a batch of them
        // gets more done than overlapping the wait).
        const more =
          stats.outputTokens < maxTokens - 1 &&
          engine.position < engine.contextLength - 1 &&
          !draftsNext()
        const guess = more && guessing()
        const d = await distribution(0, guess)
        token = thinkingBudget?.forcedToken ?? pick(d)
        if (!guess) scoreGuess(d, token)
        policy.alone(performance.now() - stepStart)
        going = emit(token)
        continue
      }
      engine.dropGuess()
    }
    // The token and its drafts in one batch: drafts the reply can still
    // hold (one more token is drawn after them) and the cache too.
    const proposed = draftsFor(
      Math.min(
        DRAFTS,
        maxTokens - stats.outputTokens - 1,
        engine.contextLength - engine.position - 1
      )
    )
    const answeringAtBatchStart = !!structured?.answering
    const preview = structured?.answering ? structured.previewAnswerDrafts(proposed) : undefined
    const drafts = preview?.drafts ?? proposed
    const batch = [token, ...drafts]
    row = await engine.forwardHidden(batch, { draft: drafts.length > 0 })
    // Row i's distribution checks draft i + 1; the first draft it rejects
    // is replaced by a token drawn from what's left, and the rest dropped.
    // Guess the token after the last row (good if every draft is kept).
    const guessLast =
      guessing() &&
      (drafts.length === 0 || TUNING.guessDrafts > 0) &&
      stats.outputTokens + drafts.length < maxTokens - 1 &&
      engine.position < engine.contextLength - 1
    const all = onGpu
      ? await distributions(batch.length, drafts, guessLast, preview?.masks)
      : null
    let kept = 1
    let next = -1
    let constrainedRow = -1
    for (let i = 0; i < batch.length && going; i++) {
      // A natural close accepted from a draft activates the answer grammar.
      // Its precomputed thinking distributions cannot select answer tokens.
      if (!answeringAtBatchStart && structured?.answering && drafts.length > 0) {
        constrainedRow = kept - 1
        break
      }
      // A drafted thinking token may have just used the remaining budget.
      // Stop before accepting another; accept(kept) below rewinds its cache.
      const forced = thinkingBudget?.forcedToken
      if (forced !== undefined && forced !== null) {
        next = forced
        break
      }
      const last = i + 1 === batch.length
      const d = all ? all[i] : await distribution(i, last && guessLast)
      if (last) {
        next = pick(d)
        if (!guessLast) scoreGuess(d, next)
        break
      }
      const verdict = check(d, batch[i + 1])
      if (!verdict.kept) {
        next = verdict.token
        break
      }
      kept++
      going = emit(batch[i + 1]) && !stopped.has(id)
    }
    const stepMs = performance.now() - stepStart
    // (A reply that ended mid-batch judged only the drafts it wrote.)
    if (drafts.length) {
      policy.checkedBatch(
        drafts.length,
        kept - 1,
        stepMs,
        going ? drafts.length : kept - 1
      )
      stats.draftBatches = (stats.draftBatches ?? 0) + 1
      stats.draftsChecked =
        (stats.draftsChecked ?? 0) + (going ? drafts.length : kept - 1)
      stats.draftsKept = (stats.draftsKept ?? 0) + kept - 1
    } else policy.alone(stepMs)
    // A draft turned down: the guess (made for all drafts kept) goes first,
    // then the batch back to the tokens kept.
    if (drafts.length) {
      if (engine.guessing && kept < batch.length) engine.dropGuess()
      engine.accept(kept)
    }
    model.cached.push(...batch.slice(0, kept))
    if (constrainedRow >= 0 && going) {
      engine.dropGuess()
      next = pick(await distribution(constrainedRow, false))
    }
    if (!going || next < 0) break
    token = next
    going = emit(token)
  }
  // A guess still pending is for a token that wasn't drawn.
  engine.dropGuess()
  const rest = decoder.flush()
  if (rest && !stopHit) {
    if (forced && structured!.answering) callText += rest
    else send(rest, !!structured?.answering)
  }
  const held = stopHit || unfiltered ? "" : stopFilter?.flush()
  if (held) post({ type: "token", id, text: held })
  if (cacheEnabled) engine.snapshot()
  stats.decodeMs = performance.now() - firstTokenAt
  if (thinkingBudget) Object.assign(stats, thinkingBudget.stats)
  // Programs the warm-up missed compile mid-request: worth knowing.
  const compiled = engine.gpu.compiled.count - compiledBefore
  if (compiled > 0) {
    const sources = [...engine.gpu.compiled.times.keys()].slice(-compiled)
    const names = sources.map((s) =>
      s
        .split("\n")
        .filter((l) => l.startsWith("uniform"))
        .map((l) => l.split(" ")[2]?.replace(";", ""))
        .filter((n) => n && !/^u(PairSplit|SliceStride|SplitIn)$/.test(n))
        .slice(0, 4)
        .join(",")
    )
    post({
      type: "log",
      text: `${compiled} programs compiled during request ${id}: ${names.join(" | ")}`
    })
  }
  // The cache grew with the chat: claim what the textures take now (the
  // page may unload other runtimes' models; the reply needn't wait).
  if (engine.gpu.bytes > model.claimed + RECLAIM_BYTES) {
    model.claimed = engine.gpu.bytes
    void claimMemory(model.model, model.claimed)
  }
  if (structured && !stopped.has(id) && !structured.complete) {
    throw new Error("Structured output was truncated before the JSON Schema was satisfied (output token or context limit).")
  }
  let toolCall: ToolCall | undefined
  if (forced && structured?.complete) toolCall = forced.read(JSON.parse(callText))
  // "length": maxTokens or the context ended it, not its end-of-turn token,
  // a stop sequence or a stop request.
  const finishReason: FinishReason =
    !sawEos && !stopHit && !stopped.has(id) && stats.outputTokens >= maxTokens ? "length" : "stop"
  // Extend the startup-tested overlap only from completed real inference.
  // Persist in the background so OPFS never delays completion.
  const pacing = engine.gpu.validatedPresentation
  const recommendation = pacing ?? (engine.gpu.presentationAdvice.relaxation ? model.note?.pacing : undefined)
  const overlapFailed = !engine.gpu.presentationAdvice.relaxation && engine.tuned.submissionOverlap !== 0
  if (overlapFailed) engine.tuned.submissionOverlap = 0
  if (model.note && (overlapFailed || JSON.stringify(model.note.pacing) !== JSON.stringify(recommendation))) {
    model.note = { ...model.note, tuned: { ...engine.tuned }, pacing: recommendation }
    void writeGpuNote(model.noteKey, model.note)
  }
  post({ type: "done", id, stats, finishReason, ...(toolCall && { toolCall }) })
  } finally {
    structured?.dispose()
    engine.setAllowedTokens(null)
    engine.dropGuess()
  }
  })
}

/**
 * Generations run one after another: they share the engine and its cache,
 * so one must finish (or stop) before the next reads its prompt.
 */
let queue: Promise<void> = Promise.resolve()
function receive(data: WorkerRequest) {
  if (data.type === "frames") {
    pacingEngine?.gpu.observeFrames(
      data.intervalMs,
      data.expectedMs,
      data.frames,
      data.startedAt,
      data.meanMs
    )
  } else if (data.type === "claimed") {
    claims.get(data.id)?.resolve()
  } else if (data.type === "tune") {
    Object.assign(TUNING, data.values)
  } else if (data.type === "load") {
    // After the generations asked for before it: another context length
    // frees the engine they run on. Each answer names its load.
    const { model, contextLength, loadId } = data
    if (data.meta) rememberModel(data.meta)
    queue = queue.then(() => {
      const before = current
      startLoad(model, contextLength).then(
        (loaded) =>
          post({
            type: "ready",
            model,
            loadId,
            contextLength: loaded.engine.contextLength,
            ...(loaded !== before && { timings: loaded.timings })
          }),
        (error: unknown) => {
          if (!(error instanceof Superseded))
            post({ type: "load-error", model, loadId, message: errorMessage(error), code: errorCode(error) })
        }
      )
    })
  } else if (data.type === "stop") {
    stopped.add(data.id)
    aborts.get(data.id)?.abort()
  } else if (data.type === "unload") {
    unload(data.model, data.id)
  } else if (data.type === "download") {
    if (data.meta) rememberModel(data.meta)
    void downloadModel(data.model, data.id)
  } else if (data.type === "generate") {
    queue = queue
      .then(() => generate(data))
      .catch((error: unknown) => {
        // A lost GPU context can't be used again: the next generation
        // loads the model afresh. Otherwise the cache may hold part of a
        // batch (or a failed growth): forget it.
        const lost = !!current?.engine.gpu.isLost
        if (lost) current = null
        else if (current) {
          current.cached.length = 0
          try {
            current.engine.reset()
          } catch {
            current = null
          }
        }
        post({ type: "error", id: data.id, message: errorMessage(error), code: lost ? "gpu-context-lost" : errorCode(error) })
      })
      .finally(() => {
        stopped.delete(data.id)
        aborts.delete(data.id)
      })
  }
}
