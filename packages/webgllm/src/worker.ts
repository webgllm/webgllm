/**
 * The WebGL 2 worker: downloads a model to the Origin Private File System,
 * uploads its weights to the GPU on an OffscreenCanvas, and runs
 * generations one at a time, streaming each one's text back. One model at a
 * time: loading another frees the previous one.
 */
import { blobSource } from "./byte-source.js"
import { Engine, MAX_CHUNKS, TUNING } from "./engine.js"
import { MIN_TEXTURE_SIZE } from "./limits.js"
import { GpuError } from "./gpu/gl.js"
import { loadGguf } from "./load-gguf.js"
import { loadOnnx } from "./load-onnx.js"
import { modelMeta } from "./model-meta.js"
import { anchorFor, prefixAnchorFor, readPrompt } from "./prompt.js"
import { repackTime } from "./repack.js"
import type { ModelFile, WebGlTextModel } from "./model-meta.js"
import type { RunTimings, WorkerRequest, WorkerResponse } from "./protocol.js"
import { DraftPolicy, lookupDraft } from "./draft.js"
import { createSampler, draw, verify } from "./sampler.js"
import { download, readGpuNotes, storedFile, writeGpuNote } from "./storage.js"
import type { GpuNote } from "./storage.js"
import { loadTokenizer } from "./tokenizer.js"
import type { Tokenizer } from "./tokenizer.js"
import { THINK_OPEN } from "./reply.js"
import { createThinkingBudget } from "./thinking-budget.js"
import { outputTokenLimit } from "./generation-limits.js"
import { createConstraint } from "@tanstack-ai-browser/structured-output"
import { StructuredGeneration, maskLogits } from "./structured-generation.js"

const post = (message: WorkerResponse) => postMessage(message)

/** Claims the page hasn't answered yet, oldest first (it answers in order). */
const claims: Array<() => void> = []

/**
 * Asks the page to make room for the model's `bytes` (in place of what it
 * claimed before), and waits until it has.
 */
function claimMemory(model: string, bytes: number) {
  return new Promise<void>((resolve) => {
    claims.push(resolve)
    post({ type: "claim", model, bytes: Math.round(bytes) })
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
  /** Token ids in the engine's cache, in order. */
  cached: Array<number>
  /** Milliseconds each phase of the load took. */
  timings?: Record<string, number>
  /** The texture bytes last claimed from the page's memory budget (see claimMemory). */
  claimed: number
  /**
   * Engine.tune not run yet on this GPU: it runs when the worker is idle
   * (see tuneWhenIdle), then its picks join this note.
   */
  untuned?: { key: string; note: GpuNote }
  /** The last tuning left choices for an empty cache (tried again once it's empty). */
  tunedPartly?: boolean
}

let current: Loaded | null = null
let loading: {
  model: string
  contextLength: number | undefined
  promise: Promise<Loaded>
  abort: AbortController
} | null = null

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
  for (let i = 0; i < text.length; i++) h = Math.imul(h ^ text.charCodeAt(i), 0x01000193)
  return (h >>> 0).toString(16)
}

/** Frees an engine's GPU memory now (see Gpu.dispose). */
const release = (engine: Engine) => engine.gpu.dispose()

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

async function fetchFiles(model: WebGlTextModel, signal: AbortSignal) {
  const meta = modelMeta(model)
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
      progress: total() ? Math.min(sum / total(), 1) * 0.9 : 0,
      cached: false,
      loadedBytes: sum,
      totalBytes: total()
    })
  }
  const files = await Promise.all(
    all.map((file, i) =>
      stored[i]
        ? Promise.resolve(stored[i])
        : download(
            file,
            (bytes, size) => {
              loaded[i] = bytes
              totals[i] = Math.max(totals[i], size)
              report()
            },
            signal
          )
    )
  )
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
  model: WebGlTextModel,
  signal: AbortSignal,
  /** Most tokens a chat may hold: the model's own context length when undefined. */
  contextLength: number | undefined,
  /** Without multiple render targets or the chunked recurrence: after a failed self-check. */
  plain = false
): Promise<Loaded> {
  const meta = modelMeta(model)
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
    ...(plain && { noMrt: ["scaled", "quad", "conv", "chunks"], chunkedRecurrence: false })
  })
  lap("context")
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
    post({ type: "progress", model, progress: cached ? 0 : 0.9, cached })
    // Room for it first: the page unloads other runtimes' models past the
    // memory budget. An estimate until it's loaded (then the textures it
    // took): the weights, the cache and the working textures.
    const weightBytes = weightFiles.reduce((sum, f) => sum + f.size, 0)
    await claimMemory(model, weightBytes * TEXTURES_PER_FILE_BYTE)
    const tokenizer = loadTokenizer({
      tokenizerJson: JSON.parse(await tokenizerFile.text()) as object,
      tokenizerConfig: JSON.parse(await tokenizerConfigFile.text()) as Record<
        string,
        unknown
      >,
      ...(templateFile && { chatTemplate: await templateFile.text() })
    })
    // What an earlier load measured on this GPU with this model and code.
    const noteKey = [
      engine.gpu.limits.renderer,
      navigator.userAgent,
      model,
      plain ? "plain" : "mrt",
      // The engine's code and its settings (a remembered check or pick
      // doesn't carry over to other defaults).
      `${Engine.fingerprint}.${hash(JSON.stringify(TUNING))}`
    ].join("|")
    const note = (await readGpuNotes())[noteKey]
    if (note) engine.preferLayout = note.layout
    const pieces =
      meta.files.kind === "gguf"
        ? loadGguf(blobSource(weightFiles[0]))
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
    })
    if (signal.aborted) throw new Superseded()
    lap("weights")
    timings.weightsReading = Math.round(engine.loadMs.reading)
    timings.weightsUploading = Math.round(engine.loadMs.uploading)
    timings.toRgba4 = Math.round(engine.loadMs.convert)
    timings.repackOnnx = Math.round(repackTime.ms)
    repackTime.ms = 0
    if (engine.layoutChoice?.ms) timings.layoutProbe = Math.round(engine.layoutChoice.ms)
    // How this GPU splits prompt matmuls, if timed on an earlier load (else
    // timed when the worker is next idle: see tuneWhenIdle); before the
    // warm-up, which compiles the programs those splits read with.
    if (note?.tuned) engine.tuned = note.tuned
    await engine.warmUp()
    lap("warmUp")
    timings.programs = engine.gpu.compiled.count
    timings.compileMs = Math.round(engine.gpu.compiled.ms)
    for (const [k, v] of Object.entries(engine.warmCounts)) timings[`warm_${k}`] = v
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
      note && note.selfCheck <= SELF_CHECK_MAX
        ? note.selfCheck
        : await engine.selfCheck()
    lap("selfCheck")
    const fresh =
      !note && engine.layoutChoice && error <= SELF_CHECK_MAX
        ? { layout: engine.layoutChoice.layout, selfCheck: error }
        : null
    if (fresh) void writeGpuNote(noteKey, fresh)
    timings.total = Math.round(performance.now() - t0)
    if (signal.aborted) throw new Superseded()
    if (!(error <= SELF_CHECK_MAX)) {
      if (plain) {
        throw new GpuError(
          `This GPU computes attention wrong (relative error ${error.toExponential(1)}).`
        )
      }
      release(engine)
      return await load(model, signal, contextLength, true)
    }
    // (A note whose tuning left choices for later is tuned again.)
    const tunedAll = note?.tuned && "f16Matmul" in note.tuned && "f16Pack" in note.tuned
    const base = fresh ?? (tunedAll ? null : note)
    // What it took, in place of the estimate.
    await claimMemory(model, engine.gpu.bytes)
    if (signal.aborted) throw new Superseded()
    return {
      model,
      contextLength,
      engine,
      tokenizer,
      cached: [],
      claimed: engine.gpu.bytes,
      timings,
      ...(base && { untuned: { key: noteKey, note: base } })
    }
  } catch (error) {
    // A failed or superseded load frees its textures now.
    release(engine)
    throw error
  }
}

/**
 * Loads `model` for chats of up to `contextLength` tokens unless it's loaded
 * (or loading) for them, replacing any other.
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
    post({ type: "ready", model })
    return Promise.resolve(current)
  }
  if (loading?.model === model && loading.contextLength === contextLength) {
    return loading.promise
  }
  // Stop a load of another model, and free the loaded one.
  loading?.abort.abort()
  if (current) {
    release(current.engine)
    current = null
  }
  const abort = new AbortController()
  const promise = load(model as WebGlTextModel, abort.signal, contextLength)
  const entry = { model, contextLength, promise, abort }
  loading = entry
  promise.then(
    (loaded) => {
      if (loading !== entry) {
        release(loaded.engine)
        return
      }
      loading = null
      current = loaded
      post({ type: "ready", model, timings: loaded.timings })
      tuneWhenIdle()
    },
    (error: unknown) => {
      if (loading !== entry) return
      loading = null
      post({ type: "load-error", model, message: errorMessage(error) })
    }
  )
  return promise
}

const stopped = new Set<string>()

const errorMessage = (error: unknown) =>
  error instanceof GpuError
    ? `The GPU failed: ${error.message}`
    : error instanceof Error
      ? error.message
      : String(error)

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
  if (stopped.has(id)) return post({ type: "done", id, stats: empty })
  // The model this request is for: loaded again if its GPU context was
  // lost, or loaded if another was asked for since.
  const model = await startLoad(request.model, request.contextLength)
  if (stopped.has(id)) return post({ type: "done", id, stats: empty })
  if (request.reportPhases) post({ type: "phase", id, phase: "processing" })
  const { engine, tokenizer } = model
  const compiledBefore = engine.gpu.compiled.count
  const prompt = tokenizer.encodeChat(request.chat, {
    tools: request.tools.length ? request.tools : undefined,
    ...(request.thinking !== undefined && { enableThinking: request.thinking }),
    addGenerationPrompt: true
  })
  const think = tokenizer.specialTokens.think
  const last = prompt[prompt.length - 1]
  const startsInThinking = think !== null &&
    (last === think ||
      (prompt[prompt.length - 2] === think &&
        tokenizer.decode([last]).trim() === ""))
  const thinkingBudget = createThinkingBudget(
    request.thinkingBudgetTokens,
    tokenizer.specialTokens,
    startsInThinking
  )
  // LFM2.5 Thinking writes its opening marker itself; its template ends at
  // assistant\n. Seed that expected marker so the answer grammar never suppresses reasoning.
  const reasoning = modelMeta(request.model as WebGlTextModel).reasoning
  const mandatoryThinking = !!reasoning && "levels" in reasoning && !("off" in reasoning.levels)
  const openThinking = request.outputSchema !== undefined && !startsInThinking && think !== null &&
    (request.thinking === true || (request.thinking === undefined && mandatoryThinking))
  const structured = request.outputSchema === undefined ? undefined : new StructuredGeneration(
    await createConstraint({ schema: request.outputSchema, tokenizer: tokenizer.constraintTokenizer }),
    tokenizer.specialTokens.eos,
    tokenizer.specialTokens.thinkEnd,
    startsInThinking || openThinking,
    tokenizer.vocabSize
  )
  engine.setAllowedTokens(null)
  try {
  const room = engine.contextLength - prompt.length
  if (room <= 0) {
    throw new Error(
      `This session is too long for the model: ${prompt.length} tokens, and it can read ${engine.contextLength} at most, reply included. Start a new session to keep going.`
    )
  }

  const maxTokens = outputTokenLimit(room, request.maxTokens)

  // Reuse the cache for the longest prefix this prompt shares with it. The
  // whole prompt only if its end was saved with its last hidden row (the
  // same prompt again: a retry); else the last token is read again, for
  // the logits after it.
  let common = 0
  while (
    common < model.cached.length &&
    common < prompt.length &&
    model.cached[common] === prompt[common]
  ) {
    common++
  }
  const whole = common === prompt.length && engine.canResume(common)
  const start = whole
    ? engine.resume(common)
    : engine.rewind(Math.min(common, prompt.length - 1))
  model.cached.length = start

  // Read the rest of the prompt (see readPrompt), with the anchor snapshot
  // where the next question's prompt will stop sharing it (see anchorFor).
  const promptStart = performance.now()
  let row = await readPrompt(engine, prompt, start, model.cached, {
    stop: () => stopped.has(id),
    anchor: (request.cachePrefix === undefined
      ? undefined
      : prefixAnchorFor(tokenizer, prompt, request.cachePrefix, request.thinking)) ?? anchorFor(
        tokenizer,
        prompt,
        request.chat,
        request.thinking,
        request.tools.length ? request.tools : undefined
      )
  })
  const stats: RunTimings = {
    inputTokens: prompt.length,
    cachedTokens: start,
    outputTokens: 0,
    firstTokenMs: null,
    promptMs: performance.now() - promptStart,
    decodeMs: 0,
    ...thinkingBudget?.stats
  }
  if (
    stopped.has(id) ||
    maxTokens <= 0 ||
    model.cached.length < prompt.length
  ) {
    if (structured && !stopped.has(id)) throw new Error("Structured output was truncated before generation could produce a schema-valid answer.")
    return post({ type: "done", id, stats })
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
  if (onGpu) engine.setSeen(prompt, vocab)
  else sampler.see(prompt)
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
  const scoreGuess = (d: { ids: Int32Array; p: Float32Array }, drawn: number) => {
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
  const updateAllowed = () => {
    // JSON literals are governed by their grammar. Only unconstrained replies
    // reserve control markers after an explicit budgeted reasoning span closes.
    const mask = structured ? structured.mask() : thinkingBudget?.closedControlMask(vocab) ?? null
    if (structured?.answering || appliedMask !== mask) engine.setAllowedTokens(mask)
    appliedMask = mask
    return mask ?? undefined
  }
  const distribution = async (r: number, guess = false) => {
    const mask = updateAllowed()
    if (greedy) {
      const [token] = await engine.argmax(r, 1, sampling.repetition_penalty, [], guess, sampling.presence_penalty)
      if (guess && engine.guessing) guessed = engine.guessedToken
      return { ids: Int32Array.of(token), p: Float32Array.of(1) }
    }
    if (onGpu) {
      const { ids, logits } = await engine.candidates(
        r,
        sampling.top_k,
        sampling.repetition_penalty,
        guess,
        sampling.presence_penalty
      )
      if (guess && engine.guessing) guessed = engine.guessedToken
      const d = sampler.distribution(logits)
      return { ids: Int32Array.from(d.ids, (i) => ids[i]), p: d.p }
    }
    const logits = await engine.logits(r)
    if (logits.length > vocab) logits.fill(-Infinity, vocab)
    if (mask) maskLogits(logits, mask)
    return sampler.distribution(logits)
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
        sampling.repetition_penalty,
        drafts,
        guess,
        sampling.presence_penalty
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
      sampling.repetition_penalty,
      drafts,
      guess,
      0,
      sampling.presence_penalty
    )
    if (guess && engine.guessing) guessed = engine.guessedToken
    return picked.map(({ ids, logits }) => {
      const d = sampler.distribution(logits)
      return { ids: Int32Array.from(d.ids, (i) => ids[i]), p: d.p }
    })
  }

  // When the template opens the thinking itself (LFM3 with thinking on ends
  // the prompt with `<think>`, Qwen3.5 with `<think>\n`), the reply starts
  // inside it: say so first, as the model would have, so the thinking
  // reads as thinking.
  if (startsInThinking) {
    post({ type: "token", id, text: THINK_OPEN })
  }

  // A zero budget may close a template-opened span before any sampling.
  let token = openThinking ? think as number : thinkingBudget?.forcedToken ?? draw(await distribution(row, guessing()))
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
    structured?.observe(t)
    if (t === eos) return false
    thinkingBudget?.observe(t)
    if (onGpu) engine.markSeen(t)
    else sampler.see([t], true)
    policy.drawn(t)
    history.push(t)
    const text = decoder.push(t)
    if (text) post({ type: "token", id, text })
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
        token = thinkingBudget?.forcedToken ?? draw(d)
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
        next = draw(d)
        if (!guessLast) scoreGuess(d, next)
        break
      }
      const check = verify(d, batch[i + 1])
      if (!check.kept) {
        next = check.token
        break
      }
      kept++
      going = emit(batch[i + 1]) && !stopped.has(id)
    }
    const stepMs = performance.now() - stepStart
    // (A reply that ended mid-batch judged only the drafts it wrote.)
    if (drafts.length) {
      policy.checkedBatch(drafts.length, kept - 1, stepMs, going ? drafts.length : kept - 1)
      stats.draftBatches = (stats.draftBatches ?? 0) + 1
      stats.draftsChecked =
        (stats.draftsChecked ?? 0) + (going ? drafts.length : kept - 1)
      stats.draftsKept = (stats.draftsKept ?? 0) + kept - 1
    }
    else policy.alone(stepMs)
    // A draft turned down: the guess (made for all drafts kept) goes first,
    // then the batch back to the tokens kept.
    if (drafts.length) {
      if (engine.guessing && kept < batch.length) engine.dropGuess()
      engine.accept(kept)
    }
    model.cached.push(...batch.slice(0, kept))
    if (constrainedRow >= 0 && going) {
      engine.dropGuess()
      next = draw(await distribution(constrainedRow, false))
    }
    if (!going || next < 0) break
    token = next
    going = emit(token)
  }
  // A guess still pending is for a token that wasn't drawn.
  engine.dropGuess()
  const rest = decoder.flush()
  if (rest) post({ type: "token", id, text: rest })
  engine.snapshot()
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
  post({ type: "done", id, stats })
  } finally {
    structured?.dispose()
    engine.setAllowedTokens(null)
    engine.dropGuess()
  }
}

/**
 * Generations run one after another: they share the engine and its cache,
 * so one must finish (or stop) before the next reads its prompt.
 */
let queue: Promise<void> = Promise.resolve()
/** Generations queued or running. */
let pending = 0
/** A background Engine.tune, while it runs. */
let tuning: Promise<void> | null = null

/**
 * Times how this GPU splits prompt matmuls (Engine.tune) while nothing
 * waits on the worker, not during the load: it gives way to the first
 * generation that arrives (and runs again once the queue is empty), and
 * its picks are remembered with the GPU's note.
 */
function tuneWhenIdle() {
  const loaded = current
  const untuned = loaded?.untuned
  if (!loaded || !untuned || tuning || pending > 0) return
  if (loaded.tunedPartly && loaded.cached.length > 0) return
  tuning = (async () => {
    // (Its prompt-batch timings reset the cache: not while it holds a chat.)
    const picked = await loaded.engine.tune(() => pending > 0 || current !== loaded, {
      keepCache: loaded.cached.length > 0
    })
    post({
      type: "log",
      text: `tuned in the background: ${picked ? JSON.stringify({ ...picked, ...loaded.engine.tuneReport }) : "stopped for a request"}`
    })
    if (!picked || current !== loaded) return
    // Choices left for an empty cache (undefined) are tried again then.
    const complete = picked.f16Matmul !== undefined && picked.f16Pack !== undefined
    if (complete) loaded.untuned = undefined
    else loaded.tunedPartly = true
    // A new pick reads short batches' results with programs the warm-up
    // didn't compile: compiled now while the cache is empty (else on the
    // first short prompt).
    if (picked.shortSplit !== null && loaded.cached.length === 0 && pending === 0)
      await loaded.engine.warmUp({ shortOnly: true })
    // (Not awaited: a request needn't wait on storage.)
    void writeGpuNote(untuned.key, { ...untuned.note, tuned: picked })
  })()
    .catch((error: unknown) => {
      // Not tried again after every reply: TUNING's splits stay.
      if (current === loaded) loaded.untuned = undefined
      post({ type: "log", text: `background tuning failed: ${errorMessage(error)}` })
    })
    .finally(() => {
      tuning = null
      // Another model may have loaded meanwhile, waiting on this one.
      tuneWhenIdle()
    })
}

onmessage = ({ data }: MessageEvent<WorkerRequest>) => {
  if (data.type === "claimed") {
    claims.shift()?.()
  } else if (data.type === "tune") {
    Object.assign(TUNING, data.values)
  } else if (data.type === "load") {
    void startLoad(data.model, data.contextLength).catch(() => undefined)
  } else if (data.type === "stop") {
    stopped.add(data.id)
  } else {
    pending++
    queue = queue
      // (A background tune stops at its next step.)
      .then(() => tuning)
      .then(() => generate(data))
      .catch((error: unknown) => {
        // A lost GPU context can't be used again: the next generation
        // loads the model afresh. Otherwise the cache may hold part of a
        // batch (or a failed growth): forget it.
        if (current?.engine.gpu.isLost) current = null
        else if (current) {
          current.cached.length = 0
          try {
            current.engine.reset()
          } catch {
            current = null
          }
        }
        post({ type: "error", id: data.id, message: errorMessage(error) })
      })
      .finally(() => {
        stopped.delete(data.id)
        pending--
        tuneWhenIdle()
      })
  }
}
