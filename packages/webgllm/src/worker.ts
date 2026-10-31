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
import type { ModelFile, WebGlTextModel } from "./model-meta.js"
import type { RunTimings, WorkerRequest, WorkerResponse } from "./protocol.js"
import { lookupDraft } from "./draft.js"
import { createSampler, draw, verify } from "./sampler.js"
import { download, storedFile } from "./storage.js"
import { loadTokenizer } from "./tokenizer.js"
import type { Tokenizer } from "./tokenizer.js"
import { THINK_OPEN } from "./reply.js"

const post = (message: WorkerResponse) => postMessage(message)

type Loaded = {
  model: string
  engine: Engine
  tokenizer: Tokenizer
  /** Token ids in the engine's cache, in order. */
  cached: Array<number>
}

let current: Loaded | null = null
let loading: {
  model: string
  promise: Promise<Loaded>
  abort: AbortController
} | null = null

/**
 * Drafted tokens checked with each written one (speculative decoding by
 * prompt lookup; see draft.ts). A batch of a token and its drafts is read
 * as fast as one token, and the tokens kept are distributed exactly as
 * without drafts.
 */
const DRAFTS = 3
/**
 * Guessing (see Engine.forwardGuess) stops while fewer than this share of
 * the last GUESS_WINDOW guesses were the token drawn: a wrong guess holds
 * the GPU up for the next token.
 */
const GUESS_MIN_HITS = 0.6
const GUESS_WINDOW = 16
/**
 * Drafts are used while recent draft batches kept at least
 * TUNING.draftMinKept drafts on average (over the last DRAFT_WINDOW):
 * otherwise guessing, which a draft batch gives up, does more. (Drafts
 * follow an earlier occurrence of TUNING.draftMatch last tokens.)
 */
const DRAFT_WINDOW = 8
/**
 * Tokens after which drafting tries again though recent batches kept too
 * few: a reply may start quoting (code, a tool's result) halfway through.
 */
const DRAFT_RETRY = 32

/** Most tokens a reply may have, unless the caller says otherwise. */
const DEFAULT_MAX_TOKENS = 4096
/** Below this much room for a reply, the chat is too long to answer. */
const MIN_REPLY_TOKENS = 32

/** Drops an engine's GPU memory now rather than whenever it's collected. */
const release = (engine: Engine) =>
  engine.gpu.gl.getExtension("WEBGL_lose_context")?.loseContext()

/** The page's touch points (sent by the client: a worker can't read them). */
let touchPoints = 0

/**
 * Prompt tokens per forward batch (the engine scales it up for smaller
 * models). A big GPU needs big batches to stay busy; phones and tablets
 * keep small ones, for memory (iOS caps a page's GPU memory). iPads report
 * a Mac's user agent: touch points give them away.
 */
const promptBatch = () =>
  /Android|iPhone|iPad|Mobile/.test(navigator.userAgent) || touchPoints > 1
    ? 128
    : 512

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
  const totals = all.map((f) => f.bytes ?? 0)
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

async function load(
  model: WebGlTextModel,
  signal: AbortSignal
): Promise<Loaded> {
  const meta = modelMeta(model)
  // The GPU first: a browser or GPU that can't run the model says so before
  // a gigabyte downloads.
  const engine = new Engine(new OffscreenCanvas(1, 1), {
    batchSize: promptBatch()
  })
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
    post({ type: "progress", model, progress: cached ? 0 : 0.9, cached })
    const tokenizer = loadTokenizer({
      tokenizerJson: JSON.parse(await tokenizerFile.text()) as object,
      tokenizerConfig: JSON.parse(await tokenizerConfigFile.text()) as Record<
        string,
        unknown
      >,
      ...(templateFile && { chatTemplate: await templateFile.text() })
    })
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
    await engine.warmUp()
    if (signal.aborted) throw new Superseded()
    return { model, engine, tokenizer, cached: [] }
  } catch (error) {
    // A failed or superseded load frees its textures now.
    release(engine)
    throw error
  }
}

/** Loads `model` unless it's loaded (or loading), replacing any other. */
function startLoad(model: string): Promise<Loaded> {
  if (current?.model === model && !current.engine.gpu.isLost) {
    post({ type: "ready", model })
    return Promise.resolve(current)
  }
  if (loading?.model === model) return loading.promise
  // Stop a load of another model, and free the loaded one.
  loading?.abort.abort()
  if (current) {
    release(current.engine)
    current = null
  }
  const abort = new AbortController()
  const promise = load(model as WebGlTextModel, abort.signal)
  const entry = { model, promise, abort }
  loading = entry
  promise.then(
    (loaded) => {
      if (loading !== entry) {
        release(loaded.engine)
        return
      }
      loading = null
      current = loaded
      post({ type: "ready", model })
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
  const model = await startLoad(request.model)
  if (stopped.has(id)) return post({ type: "done", id, stats: empty })
  const { engine, tokenizer } = model
  const prompt = tokenizer.encodeChat(request.chat, {
    tools: request.tools.length ? request.tools : undefined,
    ...(request.thinking !== undefined && { enableThinking: request.thinking }),
    addGenerationPrompt: true
  })
  const room = engine.contextLength - prompt.length
  if (room < MIN_REPLY_TOKENS) {
    throw new Error(
      `This session is too long for the model: ${prompt.length} tokens, and it can read ${engine.contextLength} at most, reply included. Start a new session to keep going.`
    )
  }

  // Reuse the cache for the longest prefix this prompt shares with it.
  let common = 0
  while (
    common < model.cached.length &&
    common < prompt.length - 1 &&
    model.cached[common] === prompt[common]
  ) {
    common++
  }
  const start = engine.rewind(common)
  model.cached.length = start

  // Read the rest of the prompt, saving the convolutions' state after each
  // batch so a later prompt can resume from there.
  const promptStart = performance.now()
  let row = 0
  let i = start
  for (const size of engine.batchSizes(prompt.length - start)) {
    const batch = prompt.slice(i, i + size)
    i += size
    row = await engine.forwardHidden(batch)
    model.cached.push(...batch)
    engine.snapshot()
    if (stopped.has(id)) break
  }
  const maxTokens = Math.min(request.maxTokens ?? DEFAULT_MAX_TOKENS, room)
  const stats: RunTimings = {
    inputTokens: prompt.length,
    cachedTokens: start,
    outputTokens: 0,
    firstTokenMs: null,
    promptMs: performance.now() - promptStart,
    decodeMs: 0
  }
  if (
    stopped.has(id) ||
    maxTokens <= 0 ||
    model.cached.length < prompt.length
  ) {
    return post({ type: "done", id, stats })
  }

  const { sampling } = request
  // With top-k sampling (LFM3's card: 50), the GPU applies the repetition
  // penalty and picks the candidates, and only those are read back. Ids
  // past the tokenizer's vocabulary are padding rows, never drawn.
  const vocab = tokenizer.vocabSize
  const onGpu = sampling.top_k > 0 && sampling.top_k <= MAX_CHUNKS
  const sampler = createSampler(
    onGpu ? { ...sampling, repetition_penalty: 1 } : sampling
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
  // Drafts kept per recent draft batch (see DRAFT_WINDOW).
  const draftsKept: Array<number> = []
  // Tokens written since the last draft batch.
  let sinceDrafts = 0
  const drafting = () =>
    request.speculative !== false &&
    (draftsKept.length < DRAFT_WINDOW / 2 ||
      sinceDrafts >= DRAFT_RETRY ||
      draftsKept.reduce((a, b) => a + b, 0) >=
        TUNING.draftMinKept * draftsKept.length)
  // The distribution of the token after `r`, a row of the last batch, as
  // token ids and probabilities; with `guess`, the likeliest token is read
  // forward meanwhile.
  const distribution = async (r: number, guess = false) => {
    if (onGpu) {
      const { ids, logits } = await engine.candidates(
        r,
        sampling.top_k,
        sampling.repetition_penalty,
        guess
      )
      if (guess && engine.guessing) guessed = engine.guessedToken
      const d = sampler.distribution(logits)
      return { ids: Int32Array.from(d.ids, (i) => ids[i]), p: d.p }
    }
    const logits = await engine.logits(r)
    if (logits.length > vocab) logits.fill(-Infinity, vocab)
    return sampler.distribution(logits)
  }
  // On the GPU: the distributions after rows 0..rows-1 of a batch whose
  // inputs after the first are `drafts`, in one pass and one readback.
  const distributions = async (
    rows: number,
    drafts: Array<number>,
    guess: boolean
  ) => {
    const picked = await engine.pick(
      0,
      rows,
      sampling.top_k,
      sampling.repetition_penalty,
      drafts,
      guess
    )
    if (guess && engine.guessing) guessed = engine.guessedToken
    return picked.map(({ ids, logits }) => {
      const d = sampler.distribution(logits)
      return { ids: Int32Array.from(d.ids, (i) => ids[i]), p: d.p }
    })
  }

  // When the template opens the thinking itself (LFM3 with thinking on ends
  // the prompt with `<think>`), the reply starts inside it: say so first,
  // as the model would have, so the thinking reads as thinking.
  if (prompt[prompt.length - 1] === tokenizer.specialTokens.think) {
    post({ type: "token", id, text: THINK_OPEN })
  }

  let token = draw(await distribution(row, guessing()))
  stats.promptMs = performance.now() - promptStart
  const firstTokenAt = performance.now()
  stats.firstTokenMs = firstTokenAt - requested
  const decoder = tokenizer.createStreamDecoder({ skipSpecialTokens: false })
  const eos = tokenizer.specialTokens.eos
  // Prompt and reply so far, where drafts are looked up.
  const history = prompt.slice()
  // Writes out a token of the reply; false once the reply is over.
  const emit = (t: number) => {
    stats.outputTokens++
    if (t === eos) return false
    if (onGpu) engine.markSeen(t)
    else sampler.see([t])
    history.push(t)
    const text = decoder.push(t)
    if (text) post({ type: "token", id, text })
    return stats.outputTokens < maxTokens
  }
  let going = emit(token)
  while (going && !stopped.has(id)) {
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
          (!drafting() || lookupDraft(history, 1, TUNING.draftMatch).length === 0)
        token = draw(await distribution(0, more && guessing()))
        going = emit(token)
        sinceDrafts++
        continue
      }
      engine.dropGuess()
    }
    // The token and its drafts in one batch: drafts the reply can still
    // hold (one more token is drawn after them) and the cache too.
    const drafts = lookupDraft(
      history,
      Math.min(
        drafting() ? DRAFTS : 0,
        maxTokens - stats.outputTokens - 1,
        engine.contextLength - engine.position - 1
      ),
      TUNING.draftMatch
    )
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
      ? await distributions(batch.length, drafts, guessLast)
      : null
    let kept = 1
    let next = -1
    for (let i = 0; i < batch.length && going; i++) {
      const last = i + 1 === batch.length
      const d = all ? all[i] : await distribution(i, last && guessLast)
      if (last) {
        next = draw(d)
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
    // A draft turned down: the guess (made for all drafts kept) goes first,
    // then the batch back to the tokens kept.
    if (drafts.length) {
      if (engine.guessing && kept < batch.length) engine.dropGuess()
      engine.accept(kept)
      draftsKept.push(kept - 1)
      if (draftsKept.length > DRAFT_WINDOW) draftsKept.shift()
      sinceDrafts = 0
    } else sinceDrafts += kept
    model.cached.push(...batch.slice(0, kept))
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
  post({ type: "done", id, stats })
}

/**
 * Generations run one after another: they share the engine and its cache,
 * so one must finish (or stop) before the next reads its prompt.
 */
let queue: Promise<void> = Promise.resolve()

onmessage = ({ data }: MessageEvent<WorkerRequest>) => {
  if (data.type === "device") {
    touchPoints = data.touchPoints
  } else if (data.type === "tune") {
    Object.assign(TUNING, data.values)
  } else if (data.type === "load") {
    void startLoad(data.model).catch(() => undefined)
  } else if (data.type === "stop") {
    stopped.add(data.id)
  } else {
    queue = queue
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
      .finally(() => stopped.delete(data.id))
  }
}
