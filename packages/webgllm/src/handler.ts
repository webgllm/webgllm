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
import { blobSource, keptSource, storedSources } from "./byte-source.js"
import { ggufShape, readGgufHeader, type GgufHeader } from "./gguf.js"
import { Engine, MAX_CHUNKS, retimedTuning, timesAgain, TUNING, tuningToRetime } from "./engine.js"
import { MIN_TEXTURE_SIZE } from "./limits.js"
import { GpuError } from "./gpu/gl.js"
import { WebGpu } from "./gpu/webgpu.js"
import { loadGguf } from "./load-gguf.js"
import { loadOnnx } from "./load-onnx.js"
import { modelMeta, onnxDataSources, rememberModel } from "./model-meta.js"
import { resolveModel } from "./resolve-model.js"
import { anchorFor, readPrompt, systemPrefixFor, templateFramingFor } from "./prompt.js"
import { repackTime } from "./repack.js"
import type { ModelFile } from "./model-meta.js"
import { WEBGLLM } from "./protocol.js"
import type { FinishReason, RunTimings, TokenLogprob, WorkerRequest, WorkerResponse } from "./protocol.js"
import { DraftPolicy, draftTimes, lookupDraft, type DraftTimes } from "./draft.js"
import { createSampler, draw, drawAt, logprobsOf, verify, verifyAt } from "./sampler.js"
import { download, incumbentGpuTuning, previousGpuTuning, readGpuNotes, saveManifest, sharedGpuTuning, storedFile, writeGpuNote } from "./storage.js"
import { loadTokenizer } from "./tokenizer.js"
import type { Tokenizer } from "./tokenizer.js"
import { createThinkingSplitter, THINK_OPEN } from "./reply.js"
import { withCallsInText } from "./tool-calls.js"
import { callVocab, ToolCallGrammar } from "./tool-call-grammar.js"
import { budgetWrapUp, createThinkingBudget, templateClosing } from "./thinking-budget.js"
import { outputTokenLimit } from "./generation-limits.js"
import { createConstraint } from "./constraint/index.js"
import { StructuredGeneration, maskLogits } from "./structured-generation.js"
import { withPromptCachePolicy } from "./prompt-cache-policy.js"
import { PrefixStore, savePromptPrefix } from "./prefix-store.js"
import { opfsFolder, persistPrefix, PrefixDisk, untilAborted } from "./prefix-disk.js"
import type { SavedPrefix } from "./engine.js"
import { planToolChoice } from "./tool-choice.js"
import { createStopFilter } from "./stop-sequences.js"
import { DEFAULT_REPEAT_LAST_N, PenaltyWindow } from "./penalty-window.js"
import { seededUniforms } from "./seeded-random.js"
import { CodedError, downloadError, errorCode, isFilesError } from "./worker-errors.js"
import { createSharedDownloads } from "./shared-download.js"
import { isImageRequest } from "./image-protocol.js"
import { receiveImage, type ImageHost } from "./image-worker.js"
import { imageProcessorConfig, type ImageProcessorConfig } from "./image.js"
import { loadVision, planVision } from "./load-vision.js"
import { loadVisionGguf, planVisionGguf } from "./load-vision-gguf.js"
import {
  chatImages,
  contentId,
  countImageSlots,
  decodeImage,
  expandImageTokens,
  fetchImage,
  ImageCache,
  imagePlaceholders,
  imageTokenCount,
  patchesOf,
  type ImageLayout,
  type ImageSlot,
  type ImageView,
  type PreparedImage
} from "./prompt-images.js"
import { VisionEncoder, visionStreamBytes } from "./vision.js"
import type { VisionConfig, VisionPiece } from "./vision-weights.js"

const post = (message: WorkerResponse) => postMessage({ ...message, [WEBGLLM]: true })

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
  /**
   * Logit processors by model id, as WebLLM's handler takes them: each
   * model's is given every logit after each forward pass, before the next
   * token is sampled from what it returns.
   */
  setLogitProcessorRegistry(registry?: Map<string, LogitProcessor>): void {
    logitProcessors = registry
  }

  /** A message from the page: `self.onmessage = (event) => handler.onmessage(event)`. */
  onmessage(event: { data: unknown }): void {
    // The worker's own messages come this way too: only webgllm's are taken.
    if (isImageRequest(event.data)) receiveImage(event.data, imageHost)
    else if (isRequest(event.data)) receive(event.data)
  }
}

/**
 * Changes a model's logits before each token is sampled: WebLLM's
 * LogitProcessor. It runs in the worker, on the CPU, so a model with one
 * reads every logit back for each token.
 */
export interface LogitProcessor {
  /** The logits right after a forward pass; returns the logits to sample from. */
  processLogits: (logits: Float32Array) => Float32Array
  /** The token sampled from them (or written without sampling), to update the processor's state. */
  processSampledToken: (token: number) => void
  /** Clears its state: the chat was reset (see resetChat). */
  resetState: () => void
}

let logitProcessors: Map<string, LogitProcessor> | undefined

/** What the image models' side (image-worker.ts) asks of the worker. */
const imageHost: ImageHost = {
  post: (message, transfer) => postMessage({ ...message, [WEBGLLM]: true }, { transfer: transfer ?? [] }),
  webgpu: () => TUNING.webgpu > 0 && webgpuAllowed,
  collect: async () => {
    uncollected = true
    await collectReleased()
  }
}

const REQUESTS: ReadonlySet<string> = new Set<WorkerRequest["type"]>([
  "load", "generate", "forward", "reset", "stop", "unload", "download", "frames", "claimed", "tune", "configure"
])

/** Whether loads may run on WebGPU (WebGLLMOptions.webgpu: "configure" with false says not). */
let webgpuAllowed = true

/**
 * Whether a message is one of the page's requests: marked as webgllm's
 * (the worker's own messages aren't, whatever their type), with the
 * fields every handler reads first and the id its answer names.
 */
function isRequest(data: unknown): data is WorkerRequest {
  if (typeof data !== "object" || data === null || (data as Record<string, unknown>)[WEBGLLM] !== true) return false
  const { type, model, id, loadId } = data as { type?: unknown; model?: unknown; id?: unknown; loadId?: unknown }
  if (typeof type !== "string" || !REQUESTS.has(type)) return false
  if (type === "load") return typeof model === "string" && typeof loadId === "number"
  if (type === "load" || type === "generate" || type === "forward" || type === "reset" || type === "unload" || type === "download") {
    if (typeof model !== "string") return false
  }
  if (type === "generate" || type === "forward" || type === "reset" || type === "stop" || type === "download") return typeof id === "string"
  if (type === "claimed" || type === "unload") return typeof id === "number"
  if (type === "configure") return (data as { webgpu?: unknown }).webgpu === "auto" || (data as { webgpu?: unknown }).webgpu === false
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

/** A GGUF tensor the engine reads from the file a row at a time, never as textures (Raon's per-layer token embeddings). */
const DISK_ONLY_TENSOR = "per_layer_token_embd.weight"

/** The memory claimed before a load: its files' bytes as textures (TEXTURES_PER_FILE_BYTE), without what stays in the file. */
export function loadEstimate(weightBytes: number, header?: Pick<GgufHeader, "tensors">): number {
  return (weightBytes - (header?.tensors.get(DISK_ONLY_TENSOR)?.byteLength ?? 0)) * TEXTURES_PER_FILE_BYTE
}

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
  /** How the model places an image's rows (the vision encoder's config.positions), known once it has encoded one. */
  imagePositions?: VisionConfig["positions"]
  /** Recent verified common content, for conservative short-prefix promotion. */
  previousShared?: number
  /** Prompt prefixes saved apart from the live history (see prefix-store.ts). */
  prefixes: PrefixStore<SavedPrefix>
  /** The same on disk, where there's OPFS (see prefix-disk.ts). */
  disk?: PrefixDisk
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
/**
 * What tokens and draft batches have cost on each loaded engine, kept from
 * reply to reply (see DraftPolicy). On a Pixel 10 Pro a batch of a token
 * and 4 drafts took 5.6× a token alone; each reply paid one to find out.
 */
const draftCosts = new WeakMap<object, DraftTimes>()


/** FNV-1a of a string, as hex (for note keys). */
function hash(text: string) {
  let h = 0x811c9dc5
  for (let i = 0; i < text.length; i++)
    h = Math.imul(h ^ text.charCodeAt(i), 0x01000193)
  return (h >>> 0).toString(16)
}

/** Frees an engine's GPU memory now (see Gpu.dispose); the rest of it goes at the next collectReleased. */
const release = (engine: Engine) => {
  if (pacingEngine === engine) pacingEngine = null
  // (Its file's lock too: see keptSource.)
  void keptFiles.get(engine)?.()
  keptFiles.delete(engine)
  engine.gpu.dispose()
  uncollected = true
}

/** Whether an engine was released since the last collectReleased. */
let uncollected = false

/** Rounds of garbage collectReleased makes (a 16 MB buffer and 50,000 objects each). */
const GARBAGE_ROUNDS = 64

/**
 * Makes garbage after an engine was released, so the worker's collector
 * runs: until it does, Safari on an iPhone keeps the released model's
 * memory (its buffers, textures and device are destroyed, but not yet
 * collected), and an idle worker never collects. An unloaded LFM3 1.7B
 * kept 1.3 GB that way, and the next model, loaded at once, crossed the
 * tab's 2 GB. What the page asks of a worker of its own (webgllm ends its
 * own workers instead, which frees it all at once). Nothing in a worker
 * there says when something is collected (FinalizationRegistry and WeakRef
 * said nothing in 256 rounds), so the rounds are counted, not watched, and
 * it's chance: 32 rounds or more freed LFM3 in 9 runs of 10 (64: 5 of 6,
 * in 0.4 to 0.6 s), 8 or 16 in 8 of 10; nothing, in none at once. The
 * buffers aren't written, so they take no memory of their own.
 */
async function collectReleased() {
  // (One at a time: a caller waits for the one making garbage, then makes its own if more was released since.)
  while (collecting) await collecting
  if (!uncollected) return
  uncollected = false
  collecting = (async () => {
    for (let round = 0; round < GARBAGE_ROUNDS; round++) {
      garbage(round)
      await new Promise((resolve) => setTimeout(resolve, 0))
    }
  })()
  try {
    await collecting
  } finally {
    collecting = undefined
  }
}

/** collectReleased's garbage being made. */
let collecting: Promise<void> | undefined

/** One round of collectReleased's garbage, dropped as soon as it's made. */
function garbage(round: number) {
  const objects: Array<{ round: number }> = []
  for (let i = 0; i < 50_000; i++) objects.push({ round })
  return new ArrayBuffer(16 << 20).byteLength + objects.length
}

/** A load's closeStored with nothing to close: made here, so it holds nothing of the load. */
const closeNothing = async (): Promise<void> => {}

/** `loading`, after which `close` runs if it fails. */
function closingOnFailure<T>(loading: Promise<T>, close: () => Promise<unknown>): Promise<T> {
  return loading.catch((error: unknown) => {
    void close()
    throw error
  })
}

/** What closes the stored file an engine reads while its model is loaded (Raon's per-layer table: see keptSource). */
const keptFiles = new Map<Engine, () => Promise<void>>()


/**
 * Prompt tokens per forward batch, the same on every device and model (the
 * engine scales it up for smaller models; TUNING.promptBatch overrides it
 * for benchmarks): bigger batches read prompts faster on every GPU tried
 * (2.4× on a Mac, 15% on a Pixel than 128). 256 or 128 would save 120 or
 * 270 MiB of Qwen3.5 2B's buffers, but LFM3 then reads prompts 32% and 56%
 * slower on an M3, and nothing the GPU reports tells where the room is
 * short (an M3 and an iPhone report the same limits).
 */
export const PROMPT_BATCH = 512

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
      : [meta.files.config, meta.files.graph, meta.files.data, ...(meta.files.moreData ?? [])]
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
  // This request's part in each file's download: stopped with the request,
  // and when one of its files fails (the others would be for nothing). A
  // download another caller still needs goes on (see shared-download.ts).
  const own = new AbortController()
  const stop = () => own.abort(signal.reason)
  if (signal.aborted) stop()
  else signal.addEventListener("abort", stop, { once: true })
  const settled = await Promise.allSettled(
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
            own.signal
          ).catch((error: unknown) => {
            own.abort(error)
            throw error
          })
    )
  )
  signal.removeEventListener("abort", stop)
  // Every file has settled: none of this request's writers is open when it
  // fails or says it was stopped. The failure is the first that wasn't the
  // stop it caused.
  const failures = settled.filter((r): r is PromiseRejectedResult => r.status === "rejected").map((r) => r.reason as unknown)
  if (failures.length) throw downloadError(signal.aborted ? failures[0] : own.signal.reason ?? failures[0], signal)
  const files = settled.map((r) => (r as PromiseFulfilledResult<File>).value)
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
  reuseTuning = true,
  /** On WebGL 2 even where there's WebGPU: after WebGPU failed this model. */
  webgl = false,
  /** A second try after the model's files failed to download (see the catch below). */
  again = false
): Promise<Loaded> {
  // What a model loaded before left (replaced, or this load's own first try) is free before this one claims its own.
  await collectReleased()
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
  const webgpu = TUNING.webgpu > 0 && webgpuAllowed && !webgl ? await WebGpu.create().catch(() => null) : null
  const engine = new Engine(new OffscreenCanvas(1, 1), {
    ...(webgpu && { gpu: webgpu }),
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
  // Closes the weights' synchronous access handles, which lock their files (see storedSources).
  let closeStored = closeNothing
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
    // The weights' header before the claim: what stays in the file isn't
    // claimed (a claim of the whole file had the page unload models that
    // fit beside this one, and a smaller claim after it brings none back).
    // The weights read through synchronous access handles where they can be (see storedSources).
    const stored = TUNING.syncReads > 0 ? await storedSources(weightFiles, { reader: TUNING.readerWorker > 0 }) : { sources: weightFiles.map(blobSource), close: closeNothing }
    closeStored = stored.close
    timings.readerWorker = "reader" in stored && stored.reader ? 1 : 0
    const weightSources = stored.sources
    // (The header is read once: the load below takes it too.)
    let ggufHeader = meta.files.kind === "gguf" ? await readGgufHeader(weightSources[0]) : undefined
    if (signal.aborted) throw new Superseded()
    await claimMemory(model, loadEstimate(weightBytes, ggufHeader), signal)
    if (signal.aborted) throw new Superseded()
    const tokenizer = loadTokenizer({
      tokenizerJson: JSON.parse(await tokenizerFile.text()) as object,
      tokenizerConfig: JSON.parse(await tokenizerConfigFile.text()) as Record<
        string,
        unknown
      >,
      ...(templateFile && { chatTemplate: await templateFile.text() })
    })
    const gpuScope = `${engine.gpu.limits.renderer}|${navigator.userAgent}|`
    const passes = plain ? "plain" : "mrt"
    const scope = `${gpuScope}${model}|${passes}|`
    const settings = hash(JSON.stringify(TUNING))
    const code = `${Engine.fingerprint}.${settings}`
    const noteKey = `${scope}${code}`
    const notes = await readGpuNotes()
    const checkedNote = notes[noteKey]
    // Per-layer embeddings (Raon) stay in the file, a token's row read when
    // the token is: from the stored file itself, which outlives the load's
    // access handles.
    const perLayerTable = !!ggufHeader?.tensors.has(DISK_ONLY_TENSOR)
    // (As a Blob while the load holds the file's handles: its probes may run a forward.)
    if (perLayerTable) engine.perLayerSource = blobSource(weightFiles[0])
    // The weights' shape, from their header: a model of the same shape as
    // one already tuned here starts from its tuning.
    const shape = ggufHeader && ggufShape(ggufHeader)
    // A code update may only change scheduling. Keep compatible tuning
    // choices and check the new engine before saving them under its key.
    // Likewise another model's with the same weights' shape.
    const note = checkedNote ?? (reuseTuning
      ? previousGpuTuning(notes, scope, settings, SELF_CHECK_MAX) ??
        (shape ? sharedGpuTuning(notes, gpuScope, `|${passes}|${code}`, shape, SELF_CHECK_MAX) : undefined)
      : undefined)
    reusedTuning = !checkedNote && !!note
    timings.calibrationCached = checkedNote ? 1 : 0
    timings.calibrationReused = reusedTuning ? 1 : 0
    timings.engineCode = parseInt(Engine.fingerprint, 16)
    // A stored timing past two frames is taken once more (a GPU busy with
    // something else at the first load times a fast pipeline slow, and the
    // pacing stored from that would slow every later session's prompts),
    // and so is one stored before timings were kept (see timesAgain). A
    // marked one keeps its kernels and has them alone timed, so its pacing
    // is always a timing of the pipeline it runs (see retimedTuning).
    const retime = note?.tuned && timesAgain(note.tuned) ? note.tuned : undefined
    if (note) {
      engine.preferLayout = note.layout
      engine.headChoice = note.head ?? null
      if (note.tuned) engine.tuned = retime ? tuningToRetime(note.tuned) : { ...note.tuned }
    } else if (reuseTuning) {
      // Tuned afresh (another browser version, engine code or settings):
      // this GPU's last choices for the model stand unless another way is
      // clearly faster (see Engine.incumbent).
      const incumbent = incumbentGpuTuning(notes, engine.gpu.limits.renderer, model, passes, SELF_CHECK_MAX)
      if (incumbent) engine.incumbent = { ...(incumbent.tuned && { tuned: incumbent.tuned }), ...(incumbent.head && { head: incumbent.head }) }
      timings.incumbent = incumbent ? 1 : 0
    }
    const pieces =
      meta.files.kind === "gguf"
        ? uploadedGgufWeights(loadGguf(weightSources[0], {
            header: ggufHeader,
            embeddingChunkRows: TUNING.ggufEmbeddingRows > 0 ? TUNING.ggufEmbeddingRows : canReleaseStaging ? undefined : 512,
            readAhead: canReleaseStaging ? TUNING.ggufReadAhead : Math.min(3, TUNING.ggufReadAhead),
            releaseReadBuffers: canReleaseStaging,
            // (The engine uploads each piece before it asks for the next.)
            reusePieces: true
          }))
        : loadOnnx({
            config: JSON.parse(await weightFiles[0].text()) as Record<
              string,
              unknown
            >,
            ...(stored.reader && { readAhead: { reads: TUNING.readerAhead, bytes: TUNING.readerAheadMiB << 20 } }),
            releaseReadBuffers: canReleaseStaging && TUNING.releaseOnnxReads > 0,
            reusePieces: true,
            graph: weightSources[1],
            data: onnxDataSources(meta.files, weightSources.slice(2))
          })
    ggufHeader = undefined
    // The load's progress, from 0 to 1 of its share (after the download's
    // 0.9, or all of it from storage): 1 only once it's ready. A first load
    // on this GPU calibrates after the weights (probes, tuning, warm-up and
    // the self-check), which takes as long as reading them or longer: the
    // weights take half the share then, and each step after them a part of
    // the rest. Otherwise the weights take nearly all of it.
    const calibrating = !note?.calibrated
    const loaded = (fraction: number) =>
      post({ type: "progress", model, progress: (cached ? 0 : 0.9) + (cached ? 1 : 0.1) * fraction, cached })
    const weightsShare = calibrating ? 0.5 : 0.95
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
        // (The last piece is the weights' end: probes may still run after it.)
        loaded(weightsShare * 0.9 * Math.min(count / layers, 1))
      }
    }, signal).finally(stored.close)
    if (signal.aborted) throw new Superseded()
    loaded(weightsShare)
    if (signal.aborted) throw new Superseded()
    if (perLayerTable && TUNING.syncReads > 0) {
      // Through a handle kept open from here on (see keptSource), now that the load's are closed: they lock the same file.
      const kept = await keptSource(weightFiles[0], (error) =>
        post({ type: "log", text: `${model}'s open file handle was closed under it (${error instanceof Error ? error.message : String(error)}); another is opened for its rows.` }))
      keptFiles.set(engine, kept.close)
      engine.perLayerSource = kept.source
    }
    if (retime) engine.tuned = retimedTuning(retime, engine.tuned)
    lap("weights")
    timings.headExpanded = engine.headChoice?.expanded ? 1 : 0
    timings.headSplit = engine.headChoice?.split ?? engine.q6HeadSplit
    timings.promptPacking = engine.tuned.f16Pack ?? TUNING.f16Pack & 3
    post({
      type: "log",
      text: `load tuning: ${JSON.stringify({ head: engine.headChoice, headReport: engine.headReport, packing: engine.packingReport, rms: engine.rmsReport, matvec: engine.matvecReport })}`
    })
    timings.weightsReading = Math.round(engine.loadMs.reading)
    timings.weightsUploading = Math.round(engine.loadMs.uploading)
    timings.weightsPacing = Math.round(engine.loadMs.pacing)
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
      loaded(0.8)
    }
    await engine.warmUp()
    lap("warmUp")
    loaded(calibrating ? 0.9 : 0.97)
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
            tuned: { ...engine.tuned },
            ...(shape && { shape }),
            // (A note written again unchanged keeps its time: see incumbentGpuTuning.)
            at: checkedNote && JSON.stringify(checkedNote.tuned) === JSON.stringify(engine.tuned) ? (checkedNote.at ?? Date.now()) : Date.now()
          }
        : null
    timings.total = Math.round(performance.now() - t0)
    if (signal.aborted) throw new Superseded()
    if (!(error <= SELF_CHECK_MAX)) {
      if (reusedTuning) throw new GpuError("Previous GPU tuning failed validation.")
      if (webgpu) throw new GpuError(`WebGPU computes the model wrong (relative error ${error.toExponential(1)}).`)
      if (plain) {
        throw new GpuError(
          `This GPU computes the model wrong: its startup check against the CPU found a relative error of ${error.toExponential(1)}.`
        )
      }
      release(engine)
      // Not awaited: this attempt ends here, and with it what holds its
      // engine, so the collection before the retry can take it (see
      // collectReleased). The retry failing closes this attempt's files.
      return closingOnFailure(load(model, signal, contextLength, true, true, webgl), closeStored)
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
    loaded(1)
    if (measured && JSON.stringify(measured) !== JSON.stringify(checkedNote))
      void writeGpuNote(noteKey, measured)
    return {
      model,
      contextLength,
      engine,
      tokenizer,
      cached: [],
      images: new ImageCache(),
      prefixes: new PrefixStore<SavedPrefix>((saved) => engine.freePrefix(saved)),
      // A file is for these weights (their files as stored), these kernels and settings, on this GPU.
      ...(typeof navigator !== "undefined" && typeof navigator.storage?.getDirectory === "function" && {
        disk: new PrefixDisk(opfsFolder(model), [
          code, scope, engine.contextLength,
          ...weightFiles.map((f) => `${f.name}:${f.size}:${f.lastModified}`)
        ].join("|"))
      }),
      claimed: engine.gpu.bytes,
      noteKey,
      note: measured ?? note,
      timings
    }
  } catch (error) {
    void closeStored()
    // A failed or superseded load frees its textures now.
    release(engine)
    if (reusedTuning && !signal.aborted && error instanceof GpuError) {
      post({ type: "log", text: `${model} didn't load with another load's GPU tuning (${error.message}); tuning afresh.` })
      return load(model, signal, contextLength, plain, false, webgl)
    }
    // Whatever WebGPU couldn't do, WebGL 2 may: but not download the model's
    // files or find room for them. A failed download is tried once more as it
    // was (it goes on from the bytes stored), then reported: an iPad whose
    // download met a passing storage error loaded Raon-1 nano on WebGL 2
    // instead, and wrote 26 tokens/s where WebGPU's 47.
    if (webgpu && !signal.aborted && isFilesError(error)) {
      if (again || errorCode(error) === "storage-full") throw error
      post({ type: "log", text: `${model}'s files didn't download (${error instanceof Error ? error.message : String(error)}); trying once more.` })
      return load(model, signal, contextLength, plain, reuseTuning, webgl, true)
    }
    if (webgpu && !signal.aborted && !(error instanceof Superseded)) {
      post({ type: "log", text: `${model} didn't load on WebGPU (${error instanceof Error ? error.message : String(error)}); loading it on WebGL 2.` })
      return load(model, signal, contextLength, plain, reuseTuning, true)
    }
    throw error
  }
}

/**
 * The context length a request runs at: its own, or (left out) the one its
 * model is loaded or loading at. A request that left it out used to load the
 * model again at its whole window: a second copy of LFM3 1.7B beside one
 * loaded at 8,192 tokens took an iPhone's tab past its 2 GB.
 */
function requestedContext(request: { model: string; contextLength?: number }) {
  if (request.contextLength !== undefined) return request.contextLength
  if (loading?.model === request.model) return loading.contextLength
  if (current?.model === request.model) return current.contextLength
  return undefined
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

/** Drops the loaded model after a failure it can't go on from, its engine released (a lost context's too: what it holds isn't only the GPU's). */
function discard() {
  const dropped = current
  current = null
  if (!dropped) return
  try {
    release(dropped.engine)
  } catch {
    // A lost context may not dispose: the next load starts afresh either way.
  }
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
    // Its memory is free once the worker's collector has run (see collectReleased).
    await collectReleased()
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
 * How the model's chat template writes an image, in its tokenizer's ids
 * (see prompt-images.ts ImageSlot): LFM2-VL's processor (the one whose
 * patch rows end in the channel) replaces a bare `<image>`; Qwen3.5's and
 * LFM3-VL's fill a placeholder between `<|vision_start|>` and
 * `<|vision_end|>`.
 */
function imageSlotOf(tokenizer: Tokenizer, config: ImageProcessorConfig): ImageSlot {
  const id = (token: string) => {
    const found = tokenizer.tokenToId(token)
    if (found == null) throw new Error(`The tokenizer has no ${token} token for images.`)
    return found
  }
  if (config.layout === "channel-last") {
    return { kind: "bare", pad: id("<image>"), start: id("<|image_start|>"), end: id("<|image_end|>"), thumbnail: id("<|img_thumbnail|>"), tile: (row, col) => id(`<|img_row_${row}_col_${col}|>`) }
  }
  return { kind: "wrapped", start: id("<|vision_start|>"), pad: tokenizer.tokenToId("<|image_pad|>") ?? id("<|image|>"), end: id("<|vision_end|>") }
}

/** A view of an image in a prompt: where its rows start. */
type PromptView = { start: number; view: ImageView }

/** What a chat template renders a prompt from: the messages and the tools. */
type TemplateSource = { chat: GenerateRequest["chat"]; tools: GenerateRequest["tools"] | undefined }

/**
 * The prompt's images, prepared (fetched and resized) or found among
 * recent ones, and the prompt with each image slot expanded to its image's
 * tokens, with the cache's keys (see prompt-images.ts). Images are taken
 * one at a time, each checked against the context before its pixels are
 * made, so a prompt that can't fit fails before it holds them all.
 * `render` is the template's text for a source (the one `prompt` is the
 * tokens of, for `source`). With
 * `reuse` false (cache_prompt: false), images are shared within the request
 * only, and none is kept for later ones.
 */
async function promptImages(
  model: Loaded,
  prompt: Array<number>,
  source: TemplateSource,
  render: (source: TemplateSource) => string,
  reuse: boolean,
  signal: AbortSignal
) {
  const wanted = chatImages(source.chat)
  if (!wanted.length) return { prompt, keys: prompt, images: [] as Array<PromptView> }
  const files = (await resolveModel(model.model, signal)).files
  if (!files.vision) throw new Error(`${model.model} doesn't read images.`)
  const { tokenizer, engine } = model
  model.imageConfig ??= imageProcessorConfig(JSON.parse(await (await modelFile(files.vision.preprocessor, signal)).text()))
  const config = model.imageConfig
  const slot = imageSlotOf(tokenizer, config)
  // Which placeholder tokens the template wrote for image parts: one a
  // message's own text has (a literal "<image>") is text, here and in an
  // earlier turn. (Asked of the template's text: one per placeholder token,
  // or the tokens aren't that text's and every placeholder is taken as before.)
  const placeholder = tokenizer.idToToken(slot.pad)
  const inText = placeholder === undefined ? undefined : imagePlaceholders(source, placeholder, render)
  const written = inText?.length === prompt.reduce((n, token) => n + (token === slot.pad ? 1 : 0), 0) ? inText : undefined
  const slots = countImageSlots(prompt, slot, written)
  if (slots !== wanted.length) throw new Error(`The prompt has ${slots} image slots for ${wanted.length} images.`)
  const maxPatches = engine.gpu.limits.maxTextureSize
  // Tokens so far: the text, and each image's (its slot's placeholder stands in until it's sized).
  let tokens = prompt.length - slots
  const fits = (image: ImageLayout) => {
    tokens += imageTokenCount(image, slot)
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
    if (found) fits(found)
    else {
      found = await decodeImage(blob!, image.detail, config, maxPatches, fits, signal)
      if (reuse) model.images.set(image, content!, found)
    }
    local.set(urlId, found)
    if (content) local.set(content, found)
    prepared.push(found)
  }
  const expanded = expandImageTokens(prompt, slot, prepared, written)
  const views = prepared.flatMap((image) => image.views)
  return { prompt: expanded.tokens, keys: expanded.keys, images: expanded.starts.map((start, i): PromptView => ({ start, view: views[i] })) }
}

/**
 * Encodes the image views whose rows the prompt reads from `start` on (the
 * cache holds the rest), reading the vision encoder's weights for them a
 * piece at a time: text-only prompts never read them. Returns the ms it
 * took to have the encoder's files (their download included; 0 with no
 * view to encode).
 */
async function encodeImages(model: Loaded, images: Array<PromptView>, start: number, signal: AbortSignal) {
  const needed = images.filter(({ start: at, view }) => at + view.height * view.width > start && !view.features)
  if (!needed.length) return 0
  const loadAt = performance.now()
  const { vision } = (await resolveModel(model.model)).files
  if (!vision) throw new Error(`${model.model} doesn't read images.`)
  const { engine } = model
  // The encoder's shapes, read before its weights: from a GGUF projector's header, or an ONNX graph.
  let config: VisionConfig
  let bits: 4 | 8
  let zeros: boolean
  let pieces: () => AsyncIterable<VisionPiece>
  if ("projector" in vision) {
    const source = blobSource(await modelFile(vision.projector, signal))
    ;({ config, bits, zeros } = await planVisionGguf(source))
    pieces = () => loadVisionGguf(source)
  } else {
    const [graph, data] = await Promise.all([modelFile(vision.graph, signal), modelFile(vision.data, signal)])
    const graphBytes = new Uint8Array(await graph.arrayBuffer())
    ;({ config, bits, zeros } = planVision(graphBytes))
    pieces = () => loadVision({ graph: graphBytes, data: blobSource(data) })
  }
  if (config.outWidth !== engine.config.hiddenSize)
    throw new Error(`The vision encoder writes ${config.outWidth}-wide rows; the model reads ${engine.config.hiddenSize}.`)
  model.imagePositions = config.positions
  // The views to encode (one the prompt has twice is encoded once), and their patch grids.
  const views = [...new Set(needed.map(({ view }) => view))]
  const grids = views.map((view): [number, number] => [view.height * config.merge, view.width * config.merge])
  // Room for what the encode holds at once: the views' work and one piece
  // of the encoder (its weights pass through a block at a time, so the
  // whole tower is never on the GPU beside the model).
  const encoder = new VisionEncoder(engine.gpu)
  try {
    signal.throwIfAborted()
    await claimMemory(model.model, engine.gpu.bytes + visionStreamBytes(config, bits, zeros, grids, engine.gpu.limits.maxTextureSize), signal)
    signal.throwIfAborted()
    const loadMs = performance.now() - loadAt
    // A view's patches are cut from its pixels when its patch projection
    // reads them, one view at a time (the views don't all hold their floats).
    const sources = views.map((view, i) => ({
      gridH: grids[i][0],
      gridW: grids[i][1],
      pixels: () => {
        const patches = patchesOf(view, model.imageConfig!)
        if (patches.gridH !== grids[i][0] || patches.gridW !== grids[i][1]) throw new Error("An image's patches aren't its view's grid.")
        return patches.pixels
      }
    }))
    const features = await encoder.encodeStream(pieces(), sources, signal)
    views.forEach((view, i) => {
      view.features = features[i]
      view.pixels = null
    })
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
  const model = await startLoad(request.model, requestedContext(request))
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
  // "required", as llama.cpp has it: the model writes its call in its own
  // format, text may come first, and the reply can't end before a call is
  // written to its closing marker (LFM3's `</tool_call>`, LFM2's `<|tool_call_end|>`).
  const callEnds = toolPlan.mode === "required"
    ? [tokenizer.specialTokens.toolCallEnd, tokenizer.tokenToId("<|tool_call_end|>")].filter((t): t is number => typeof t === "number" && t >= 0)
    : []
  let called = toolPlan.mode !== "required"
  // A required call and a JSON answer can't both be the reply (chat-completions refuses it too).
  if (request.outputSchema && toolPlan.mode === "required") {
    throw new Error("An output schema can't be combined with a required or named tool choice.")
  }
  const answerSchema = request.outputSchema
  const tokenizeAt = performance.now()
  // A raw prompt (OpenAI's Completions) is read as it is: no chat template, so nothing of a chat's framing to cache by.
  const raw = request.prompt !== undefined
  const variables = request.templateVariables
  // Past calls as `tool_calls`, or in their turn's text where the template doesn't write those (see withCallsInText).
  const chat = raw || tokenizer.writesToolCalls() ? request.chat : withCallsInText(request.chat, modelMeta(request.model).toolFormat ?? "lfm3")
  // A last assistant message is continued (llama.cpp's prefill_assistant, on by default there).
  const continues = chat?.[chat.length - 1]?.role === "assistant"
  const templateOptions = {
    ...variables,
    ...(request.thinking !== undefined && { enableThinking: request.thinking }),
    addGenerationPrompt: !continues,
    ...(continues && { continueFinalMessage: true })
  }
  const encoded = raw ? tokenizer.encode(request.prompt!) : tokenizer.encodeChat(chat, { tools, ...templateOptions })
  // The text those tokens are of, for another source: how the image step
  // tells the template's image placeholders from a message's own text.
  const source: TemplateSource = { chat, tools }
  const render = (from: TemplateSource) => tokenizer.applyChatTemplate(from.chat, { ...(from.tools && { tools: from.tools }), ...templateOptions })
  // Each image's token expanded to its tokens (the image fetched and
  // resized here); `keys` is what the cache compares.
  const imagesAt = performance.now()
  const signal = abortFor(id)
  let prepared: Awaited<ReturnType<typeof promptImages>>
  try {
    prepared = await promptImages(model, encoded, source, render, request.cachePrompt !== false, signal)
  } catch (error) {
    // Stopped while its images were fetched or resized: the cache is as it was.
    if (stopped.has(id)) return post({ type: "done", id, stats: empty, finishReason: "stop" })
    throw error
  }
  const { prompt, keys, images } = prepared
  // (A reply is sampled from the logits after the prompt's last token: with none, there's nothing to read them from.)
  if (!prompt.length) throw new Error("The prompt has no tokens.")
  const prepareImagesMs = performance.now() - imagesAt
  const cacheEnabled = request.cachePrompt !== false
  const prefix = cacheEnabled && !raw ? systemPrefixFor(tokenizer, prompt, chat, request.thinking, tools, variables) : undefined
  const anchor = cacheEnabled && !raw ? anchorFor(tokenizer, prompt, chat, request.thinking, tools, variables) : undefined
  const tokenizeMs = performance.now() - tokenizeAt - prepareImagesMs
  const think = tokenizer.specialTokens.think
  // The reply starts inside a thought when the prompt's last turn opened one
  // and didn't close it: the template opened it (LFM3's `<think>`, Qwen3.5's
  // `<think>\n`), or a continued message's thinking has no text after it yet.
  // (A raw prompt is written back as it is: no thinking of the chat template's to announce.)
  const startsInThinking = !raw && think !== null &&
    prompt.lastIndexOf(think) > Math.max(prompt.lastIndexOf(tokenizer.specialTokens.thinkEnd ?? -1), prompt.lastIndexOf(tokenizer.specialTokens.imStart ?? -1))
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
  // Automatic and required calls held to their tools while they're written,
  // in the model's own format, as llama.cpp's grammars hold them (see
  // tool-call-grammar.ts). Not beside a JSON answer: that reply is the JSON.
  const callFormat = modelMeta(request.model).toolFormat ?? "lfm3"
  const [callStart, callEnd] = callFormat === "lfm2"
    ? [tokenizer.tokenToId("<|tool_call_start|>"), tokenizer.tokenToId("<|tool_call_end|>")]
    : [tokenizer.specialTokens.toolCall, tokenizer.specialTokens.toolCallEnd]
  const grammar = !raw && answerSchema === undefined && (toolPlan.mode === "auto" || toolPlan.mode === "required") && toolPlan.tools.length &&
    typeof callStart === "number" && callStart >= 0 && typeof callEnd === "number" && callEnd >= 0
    ? new ToolCallGrammar({
      format: callFormat,
      tools: toolPlan.callable ?? toolPlan.tools,
      // Unset, as llama.cpp defaults it: whether the template writes a turn's second call.
      parallel: request.parallelToolCalls ?? tokenizer.writesParallelToolCalls(),
      vocab: callVocab(tokenizer),
      start: callStart,
      end: callEnd,
      eos: tokenizer.specialTokens.eos,
      think: tokenizer.specialTokens.think,
      thinkEnd: tokenizer.specialTokens.thinkEnd,
      startsInThinking,
      json: (schema) => createConstraint({ schema, tokenizer: tokenizer.constraintTokenizer })
    })
    : undefined
  const schemaMs = performance.now() - schemaAt
  let fresh: ReturnType<typeof savePromptPrefix<SavedPrefix>>
  // A saved prefix this request restored, to mark used on disk too.
  let used: Array<number> | undefined
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
  // A saved prefix of this prompt's (another system prompt's history may
  // have replaced it since), where it reaches further than the live history.
  // (Not the whole prompt: the last token is read for the logits after it.)
  // The request's budgets first: they decide what may be matched, read from disk and kept.
  if (request.promptCacheBytes !== undefined) model.prefixes.setBudget(request.promptCacheBytes)
  if (request.promptCacheDiskBytes !== undefined && model.disk) await model.disk.setBudget(request.promptCacheDiskBytes)
  const storing = cacheEnabled && !raw && !whole
  const live = engine.rewindPosition(Math.min(common, prompt.length - 1))
  // On disk, one longer than what memory and the live history have (a page
  // loaded again starts with none in memory), that fits the memory budget:
  // into the store first, room made before its textures are.
  if (storing && model.disk && model.prefixes.budget > 0) {
    try {
      const inMemory = model.prefixes.match(keys, prompt.length - 1)?.keys.length ?? 0
      // (A stop meanwhile ends the wait: the read finishes unused, nothing allocated.)
      const found = await untilAborted(model.disk.match(keys, prompt.length - 1, Math.max(live, inMemory), model.prefixes.budget), signal)
      if (stopped.has(id)) return post({ type: "done", id, stats: empty, finishReason: "stop" })
      if (found) {
        const bytes = found.textures.reduce((sum, t) => sum + t.bytes.byteLength, 0)
        if (model.prefixes.makeRoom(bytes)) {
          const saved = engine.importPrefix(found.position, found.textures)
          if (saved) model.prefixes.add(found.keys, bytes, saved)
        }
      }
    } catch (error) {
      if (engine.gpu.isLost) throw error
      post({ type: "log", text: `Couldn't read a prompt prefix from disk: ${error instanceof Error ? error.message : String(error)}` })
    }
  }
  // (After the disk: making room may have freed what memory matched before.)
  const stored = storing ? model.prefixes.match(keys, prompt.length - 1) : undefined
  const restored = stored && stored.keys.length > live ? stored : undefined
  let start: number
  if (restored) {
    // (Nothing held until it's back: a failure leaves an empty cache.)
    model.cached.length = 0
    start = await engine.restorePrefix(restored.state)
    model.cached.push(...restored.keys)
    used = restored.keys
    common = start
  } else {
    start = whole
      ? engine.resume(common)
      : engine.rewind(Math.min(common, prompt.length - 1))
    model.cached.length = start
  }
  let natural: number | undefined
  if (cacheEnabled && common > 0 && !whole && prefix !== undefined && prefix > start) {
    let at = start
    for (const size of engine.batchSizes(prompt.length - start)) { at += size; if (at > prefix) break; natural = at }
  }
  const framing = cacheEnabled && !raw && common > 0 && !whole
    ? templateFramingFor(tokenizer, prompt, chat[0]?.role ?? "user", request.thinking, request.templateVariables) : 0
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
  engine.setImages(
    images.map(({ start: at, view }) => ({ start: at, height: view.height, width: view.width, features: view.features })),
    model.imagePositions
  )
  const encodeMs = performance.now() - encodeAt
  const promptStart = performance.now()
  const reusablePrefix = plan.prefix
  const learned = cacheEnabled && reusablePrefix !== undefined && reusablePrefix > start && reusablePrefix <= common && common > 0
  // The system prompt's end, saved for the store where it lacks it, even
  // inside a batch (see Engine.forwardHidden's prefixAt): a few copies. It
  // takes the place of a shorter boundary learned from the live history.
  const storable = cacheEnabled && prefix !== undefined && prefix > start && prefix < prompt.length &&
    model.prefixes.budget > 0 && !images.some((image) => image.start < prefix) && !model.prefixes.has(keys.slice(0, prefix))
  if (!stopped.has(id)) await engine.preparePrompt(prompt.length)
  let row = await readPrompt(engine, prompt, start, model.cached, {
    keys,
    stop: () => stopped.has(id),
    // After a changed request reveals a shorter common boundary, retain it
    // during this ordinary forward. One-off cold calls incur no extra work.
    prefix: reusablePrefix,
    // (The store's first: the live history's shorter shared boundary has other requests to come back to.)
    ...(storable ? { observedPrefix: prefix } : learned && { observedPrefix: reusablePrefix }),
    anchor,
    checkpoints: cacheEnabled
  })
  if (cacheEnabled && !raw && !stopped.has(id)) fresh = savePromptPrefix(engine, model.prefixes, model.cached.length, keys, images, prefix, {
    lost: () => engine.gpu.isLost,
    log: (text) => post({ type: "log", text })
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
  // Log probabilities, a logit bias and a logit processor need every logit
  // read back: the token is then picked here, not on the GPU.
  const processor = logitProcessors?.get(request.model)
  const bias = request.logitBias ? Object.entries(request.logitBias).map(([token, value]) => [Number(token), value] as const).filter(([token]) => Number.isInteger(token) && token >= 0 && token < vocab) : []
  // So does a frequency penalty: the GPU's penalties know whether a token
  // is in the window, not how many times.
  const allLogits = !!request.logprobs || bias.length > 0 || !!processor || (sampling.frequency_penalty ?? 0) !== 0
  const onGpu = !allLogits && (greedy || (sampling.top_k > 0 && sampling.top_k <= MAX_CHUNKS))
  // The log probability of the token just chosen, until it's written (see emit).
  let scored: { token: number; of: (token: number) => TokenLogprob } | undefined
  // A token's own bytes: a character split across tokens decodes to nothing alone.
  const logprobEntry = (token: number, logprob: number) => ({
    token: tokenizer.decode([token]),
    logprob,
    bytes: Array.from(tokenizer.tokenBytes(token))
  })
  // Written tokens' log probabilities go out with their text, once all of
  // it is sent: a token a stop sequence cut, wholly or partly, isn't
  // reported (its bytes would say more than the reply does). `spans` are
  // the pieces of text given to the stop filter and not yet all sent, each
  // with the tokens that made it (several, when a character is split across
  // them). Kept only when log probabilities were asked for.
  const scoring = !!request.logprobs
  const spans: Array<{ entries: Array<TokenLogprob>; end: number }> = []
  let waiting: Array<TokenLogprob> = []
  let produced = 0
  let sentChars = 0
  const track = (text: string) => {
    if (!scoring) return
    spans.push({ entries: waiting, end: (produced += text.length) })
    waiting = []
  }
  /** The tokens all of whose text is sent. */
  const sentEntries = () => {
    const out: Array<TokenLogprob> = []
    while (spans.length && spans[0].end <= sentChars) out.push(...spans.shift()!.entries)
    return out
  }
  const postText = (text: string) => {
    if (!scoring) return post({ type: "token", id, text })
    sentChars += text.length
    const logprobs = sentEntries()
    post({ type: "token", id, text, ...(logprobs.length && { logprobs }) })
  }
  const sampler = createSampler(
    onGpu ? { ...sampling, repetition_penalty: 1, presence_penalty: 0 } : sampling
  )
  // The penalties look at the last tokens read or written, the prompt's
  // included, as llama.cpp's do (see penalty-window.ts). setSeen also bans
  // the padding ids.
  const seen = new PenaltyWindow(sampling.repeat_last_n ?? DEFAULT_REPEAT_LAST_N)
  if (onGpu) {
    seen.push(prompt)
    engine.setSeen(seen.entries().map(([token]) => token), vocab)
  } else sampler.see(prompt)
  // With a seed, the number each token is drawn with comes from the seed
  // and the token's place in the reply (see seeded-random.ts); else
  // Math.random, as before. Greedy picks don't depend on either.
  const uniform = request.seed === undefined ? undefined : seededUniforms(request.seed)
  type Distribution = { ids: Int32Array; p: Float32Array; of?: (token: number) => TokenLogprob }
  const pick = (d: Distribution) => {
    const token = uniform ? drawAt(d, uniform(stats.outputTokens)) : draw(d)
    if (d.of) scored = { token, of: d.of }
    return token
  }
  const check = (d: Distribution, drafted: number) => {
    const verdict = uniform ? verifyAt(d, drafted, uniform(stats.outputTokens)) : verify(d, drafted)
    if (d.of) scored = { token: verdict.kept ? drafted : verdict.token, of: d.of }
    return verdict
  }
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
  // Drafts stop short of a call's closing marker until a call is written:
  // the rows after it may end the reply, which the batch's one mask (no
  // end yet) wouldn't let them.
  // None inside a call held to its tools (or after it, where it ends the
  // reply), and they stop short of a call's opening marker: each token of a
  // call has its own mask.
  const untilCallEnd = (drafts: Array<number>) => {
    if (grammar?.holding) return []
    const opens = grammar ? drafts.indexOf(callStart as number) : -1
    if (opens >= 0) drafts = drafts.slice(0, opens)
    if (called) return drafts
    const at = drafts.findIndex((t) => callEnds.includes(t))
    return at < 0 ? drafts : drafts.slice(0, at)
  }
  // How many drafts to check with each token, by what they've cost and
  // earned (see DraftPolicy).
  let costs = draftCosts.get(engine)
  if (!costs) draftCosts.set(engine, (costs = draftTimes(DRAFTS)))
  const policy = new DraftPolicy(DRAFTS, undefined, undefined, undefined, undefined, costs)
  // Drafts for the reply so far, at most `max`: as many of those found as
  // the policy expects to pay.
  // (A long match starts drafting: prose repeats two or three tokens by
  // chance, and each batch that finds so costs it speed. While a batch's
  // drafts keep being right, shorter ones carry on.)
  let rolling = false
  const lookup = (max: number) =>
    untilCallEnd(lookupDraft(history, max, rolling ? TUNING.draftMatch : TUNING.draftStart))
  // (Not every model reads draft batches: see Engine.canDraft.)
  const drafting = () => request.speculative !== false && engine.canDraft
  const draftsFor = async (max: number) => {
    if (max <= 0 || !drafting()) return []
    policy.startAfter = engine.draftsReady ? 1 : 2
    const found = lookup(max)
    // The first batch's shaders compile in the background (no guess is
    // pending here); tokens go on one at a time until they're done.
    if (!engine.draftsReady && policy.peek(found) > 0) {
      await engine.prepareDraftBatches()
      policy.watch(found)
      return []
    }
    return found.slice(0, policy.choose(found))
  }
  // Whether a draft batch would follow. If not, the drafts found are still
  // scored: with batch times remembered from an earlier reply, a reply whose
  // guesses all hit would otherwise never learn that its drafts are right.
  const draftsNext = () => {
    if (!drafting()) return false
    policy.startAfter = engine.draftsReady ? 1 : 2
    const found = lookup(DRAFTS)
    const next = found.length > 0 && !engine.draftsCompiling && policy.peek(found) > 0
    if (!next) policy.watch(found)
    return next
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
  // "required" before its call: every token but the end of the reply.
  let noEnd: Uint32Array | null = null
  const eosToken = tokenizer.specialTokens.eos
  if (!called && eosToken >= 0 && eosToken < vocab) {
    noEnd = new Uint32Array(Math.ceil(vocab / 32)).fill(0xffffffff)
    noEnd[eosToken >>> 5] &= ~(1 << (eosToken & 31))
  }
  // (The last mask with the end taken out, by the mask it came from: the same array while that is.)
  let endless: { from: Uint32Array; mask: Uint32Array } | undefined
  const withoutEnd = (mask: Uint32Array | null) => {
    if (!mask) return noEnd!
    if (endless?.from !== mask) endless = { from: mask, mask: mask.map((word, i) => word & noEnd![i]) }
    return endless.mask
  }
  const updateAllowed = () => {
    // JSON literals are governed by their grammar. Only unconstrained replies
    // reserve control markers: with thinking off, or after an explicit
    // budgeted reasoning span closes.
    // A tool call's mask, where one is being written, decides alone.
    const call = grammar?.mask() ?? null
    const open = call ?? (structured ? structured.mask() : thinkingOff ?? thinkingBudget?.closedControlMask(vocab) ?? null)
    const mask = !called && noEnd && !structured && !call ? withoutEnd(open) : open
    if (structured?.answering || call || appliedMask !== mask) engine.setAllowedTokens(mask)
    appliedMask = mask
    return mask ?? undefined
  }
  // Penalties inside a constrained answer too, as llama.cpp applies them:
  // its grammar keeps the JSON valid, and a window of the last tokens (see
  // penalty-window.ts) keeps quotes and keys from being penalized for long.
  // Without them LFM3 went on copying the prompt's examples into an array
  // that llama.cpp ended after the one item asked for.
  const penalize = () => true
  const penalty = () => penalize() ? sampling.repetition_penalty : 1
  const presence = () => penalize() ? sampling.presence_penalty : 0
  const distribution = async (r: number, guess = false) => {
    await grammar?.ready()
    const mask = updateAllowed()
    if (greedy && onGpu) {
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
    let logits = await engine.logits(r)
    if (logits.length > vocab) logits.fill(-Infinity, vocab)
    // The caller's processor and bias first, as WebLLM applies them; then
    // what the model's distribution is, before the sampler's own settings.
    if (processor) logits = processor.processLogits(logits)
    for (const [token, value] of bias) logits[token] += value
    if (mask) maskLogits(logits, mask)
    const of = request.logprobs ? logprobsOf(logits, request.topLogprobs ?? 0, logprobEntry) : undefined
    const d = sampler.distribution(logits, penalize())
    return of ? { ...d, of } : d
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
    else {
      await grammar?.ready()
      updateAllowed()
    }
    // Each row's penalties as drawing the drafts one by one would give them.
    const seenRows = penalize() ? seen.ahead(drafts) : []
    if (greedy) {
      const tokens = await engine.argmax(
        0,
        rows,
        penalty(),
        seenRows,
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
      seenRows,
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

  // Stop sequences end the answer (see stop-sequences.ts), but not inside a
  // constrained answer: the schema ends it, and a stop string cut from it
  // would leave JSON that fails the schema.
  const stopFilter = createStopFilter(request.stop)
  let stopHit = false
  let sawEos = false
  // The answer's last characters while a "required" call isn't written yet,
  // and whether its opening marker is written; the reply read into thinking
  // and answer as the reply reader reads it (it gets the same text).
  let callTail = ""
  let callOpen = false
  const callReader = called ? undefined : createThinkingSplitter()
  let unfiltered = false
  /** Sends reply text; false once a stop sequence ended the reply. */
  const send = (text: string, constrained = false) => {
    if (constrained && stopFilter && !unfiltered) {
      // What the filter holds back (whitespace after the thinking) first.
      const held = stopFilter.flush()
      if (held) postText(held)
      unfiltered = true
    }
    track(text)
    if (!stopFilter || unfiltered) {
      postText(text)
      return true
    }
    const out = stopFilter.push(text)
    if (out.text) postText(out.text)
    stopHit = out.stopped
    return !stopHit
  }
  // When the template opens the thinking itself (LFM3 with thinking on ends
  // the prompt with `<think>`, Qwen3.5 with `<think>\n`), the reply starts
  // inside it: say so first, as the model would have, so the thinking
  // reads as thinking.
  if (startsInThinking) {
    send(THINK_OPEN)
    callReader?.push(THINK_OPEN)
  }

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
    structured?.observe(t)
    grammar?.observe(t)
    processor?.processSampledToken(t)
    // (A token the model didn't choose, like a forced close of its thinking, has no distribution: probability 1.)
    if (scoring && t !== eos) waiting.push(scored?.token === t ? scored.of(t) : { ...logprobEntry(t, 0), top_logprobs: [] })
    scored = undefined
    if (t === eos) {
      sawEos = true
      return false
    }
    thinkingBudget?.observe(t)
    if (onGpu) for (const [token, state] of seen.push([t], true)) engine.setSeenState(token, state)
    else sampler.see([t], true)
    policy.drawn(t)
    history.push(t)
    const text = decoder.push(t)
    // A "required" call is written once the answer has a call's opening
    // marker and then its closing one. What's answer and what's thinking is
    // read from the text as the reply reader reads it (tags split across
    // tokens too), so a call the reader counts as thinking never counts here.
    if (!called && callReader && text) {
      for (const piece of callReader.push(text)) {
        if (piece.kind !== "answer") continue
        callTail += piece.text
        if (!callOpen) {
          const at = Math.max(callTail.lastIndexOf("<tool_call>"), callTail.lastIndexOf("<|tool_call_start|>"))
          if (at >= 0) {
            callOpen = true
            callTail = callTail.slice(at)
          }
        }
        if (callOpen && (callTail.includes("</tool_call>") || callTail.includes("<|tool_call_end|>"))) called = true
        callTail = callTail.slice(-32)
      }
    }
    if (text && !send(text, constrained)) return false
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
    const proposed = await draftsFor(
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
    const compiledBefore = engine.gpu.compiled.count
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
        going ? drafts.length : kept - 1,
        engine.gpu.compiled.count !== compiledBefore
      )
      rolling = kept > 1
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
  if (rest && !stopHit) send(rest, !!structured?.answering)
  const held = stopHit || unfiltered ? "" : stopFilter?.flush()
  if (held) postText(held)
  // What's left: tokens whose text never made a message (an unfinished
  // character's); after a stop sequence, nothing past it.
  const unsent = !scoring ? [] : stopHit ? sentEntries() : [...sentEntries(), ...spans.splice(0).flatMap((span) => span.entries), ...waiting]
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
  post({ type: "done", id, stats, finishReason, ...(unsent.length && { logprobs: unsent }) })
  } finally {
    structured?.dispose()
    grammar?.dispose()
    engine.setAllowedTokens(null)
    engine.dropGuess()
    // A system prompt read for the first time, to disk once the reply is out (the next request waits for it).
    if (used && model.disk) await model.disk.touch(used).catch(() => undefined)
    if (fresh && !engine.gpu.isLost)
      await persistPrefix(model.disk, (saved) => model.prefixes.holds(saved), engine, fresh, (text) => post({ type: "log", text }))
    // What the textures take now, on every way out (a prefill's too, which returns before the reply's claim).
    if (!engine.gpu.isLost && engine.gpu.bytes > model.claimed + RECLAIM_BYTES) {
      model.claimed = engine.gpu.bytes
      void claimMemory(model.model, model.claimed)
    }
  }
  })
}

/**
 * WebLLM's forwardTokensAndSample: reads `inputIds` after what the model's
 * cache holds and samples the next token from the logits after them (its
 * logit processor's, if it has one). The token isn't read into the cache:
 * the caller passes it in next.
 */
async function forwardTokens(request: Extract<WorkerRequest, { type: "forward" }>) {
  const model = await startLoad(request.model, requestedContext(request))
  const { engine } = model
  const ids = request.inputIds
  if (!ids.length) throw new Error("forwardTokensAndSample needs at least one token.")
  const vocab = model.tokenizer.vocabSize
  if (ids.some((t) => !Number.isInteger(t) || t < 0 || t >= vocab)) throw new Error(`Token ids go from 0 to ${vocab - 1}.`)
  if (engine.position + ids.length > engine.contextLength) {
    throw new CodedError(
      `This session is too long for the model: ${engine.position + ids.length} tokens, and it can read ${engine.contextLength} at most. Call resetChat() to start again.`,
      "context-length-exceeded"
    )
  }
  engine.dropGuess()
  engine.setAllowedTokens(null)
  let logits = await engine.forward(ids)
  // What the cache holds now, for the next chat completion to compare its
  // prompt with. (One at a time: a long prompt is more than a call takes as arguments.)
  for (const t of ids) model.cached.push(t)
  // The cache grew with them: claim what the textures take now, as a reply does.
  if (engine.gpu.bytes > model.claimed + RECLAIM_BYTES) {
    model.claimed = engine.gpu.bytes
    void claimMemory(model.model, model.claimed)
  }
  if (logits.length > vocab) logits = logits.subarray(0, vocab)
  const processor = logitProcessors?.get(request.model)
  if (processor) logits = processor.processLogits(logits)
  // The penalties look at the last tokens the cache holds, as a reply's look at its prompt's.
  const sampler = createSampler(request.sampling)
  sampler.see(model.cached)
  const token = draw(sampler.distribution(logits))
  processor?.processSampledToken(token)
  post({ type: "forwarded", id: request.id, token })
}

/** WebLLM's resetChat: forgets what the model's cache holds, and its logit processor's state. */
function resetChat(request: Extract<WorkerRequest, { type: "reset" }>) {
  if (current?.model === request.model) {
    current.cached.length = 0
    current.prefixes.clear()
    current.engine.reset()
  }
  logitProcessors?.get(request.model)?.resetState()
  post({ type: "reset-done", id: request.id })
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
  } else if (data.type === "configure") {
    webgpuAllowed = data.webgpu !== false
  } else if (data.type === "load") {
    // After the generations asked for before it: another context length
    // frees the engine they run on. Each answer names its load.
    const { model, contextLength, loadId } = data
    if (data.meta) rememberModel(data.meta)
    queue = queue.then(() => {
      // Weakly: a load that replaces it releases its engine, which the
      // collection before that load must be able to take (see collectReleased).
      const before = current && new WeakRef(current)
      startLoad(model, contextLength).then(
        (loaded) => {
          post({
            type: "ready",
            model,
            loadId,
            contextLength: loaded.engine.contextLength,
            ...(loaded !== before?.deref() && { timings: loaded.timings })
          })
        },
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
  } else if (data.type === "reset") {
    queue = queue.then(() => {
      try {
        resetChat(data)
      } catch (error) {
        post({ type: "error", id: data.id, message: errorMessage(error), code: errorCode(error) })
      }
    })
  } else if (data.type === "generate" || data.type === "forward") {
    queue = queue
      .then(() => (data.type === "generate" ? generate(data) : forwardTokens(data)))
      .catch((error: unknown) => {
        // A lost GPU context can't be used again: the next generation
        // loads the model afresh. Otherwise the cache may hold part of a
        // batch (or a failed growth): forget it.
        // (Dropped, its engine is released: the file it keeps open is locked until then.)
        const lost = !!current?.engine.gpu.isLost
        if (lost) discard()
        else if (current) {
          current.cached.length = 0
          try {
            current.engine.reset()
          } catch {
            discard()
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
