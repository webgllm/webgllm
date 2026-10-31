/**
 * The WebGL 2 worker: downloads a model to the Origin Private File System,
 * uploads its weights to the GPU on an OffscreenCanvas, and runs
 * generations one at a time, streaming each one's text back. One model at a
 * time: loading another frees the previous one.
 */
import { blobSource } from "./byte-source.js"
import { Engine, MAX_CHUNKS } from "./engine.js"
import { GpuError } from "./gpu/gl.js"
import { loadGguf } from "./load-gguf.js"
import { loadOnnx } from "./load-onnx.js"
import { modelMeta } from "./model-meta.js"
import type { ModelFile, WebGlTextModel } from "./model-meta.js"
import type { RunTimings, WorkerRequest, WorkerResponse } from "./protocol.js"
import { createSampler } from "./sampler.js"
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

/** Most tokens a reply may have, unless the caller says otherwise. */
const DEFAULT_MAX_TOKENS = 4096
/** Below this much room for a reply, the chat is too long to answer. */
const MIN_REPLY_TOKENS = 32

/** Drops an engine's GPU memory now rather than whenever it's collected. */
const release = (engine: Engine) =>
  engine.gpu.gl.getExtension("WEBGL_lose_context")?.loseContext()

/**
 * Prompt tokens per forward batch. The prompt matmuls compute two tokens
 * per fragment, so a big GPU needs big batches to stay busy; phones keep
 * small ones, for memory (~45 MB less).
 */
const promptBatch = () =>
  /Android|iPhone|iPad|Mobile/.test(navigator.userAgent) ? 128 : 512

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
    contextLength: meta.context,
    batchSize: promptBatch()
  })
  try {
    const { maxTextureSize } = engine.gpu.limits
    if (maxTextureSize < meta.context) {
      throw new GpuError(
        `This GPU's textures go up to ${maxTextureSize} texels; the model needs ${meta.context}.`
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
    engine.reset()
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
  for (let i = start; i < prompt.length; i += engine.batchSize) {
    const batch = prompt.slice(i, i + engine.batchSize)
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
  const next = async () => {
    if (onGpu) {
      const { ids, logits } = await engine.candidates(
        row,
        sampling.top_k,
        sampling.repetition_penalty
      )
      return ids[sampler.sample(logits)]
    }
    const logits = await engine.logits(row)
    if (logits.length > vocab) logits.fill(-Infinity, vocab)
    return sampler.sample(logits)
  }

  // When the template opens the thinking itself (LFM3 with thinking on ends
  // the prompt with `<think>`), the reply starts inside it: say so first,
  // as the model would have, so the thinking reads as thinking.
  if (prompt[prompt.length - 1] === tokenizer.specialTokens.think) {
    post({ type: "token", id, text: THINK_OPEN })
  }

  let token = await next()
  stats.promptMs = performance.now() - promptStart
  const firstTokenAt = performance.now()
  stats.firstTokenMs = firstTokenAt - requested
  const decoder = tokenizer.createStreamDecoder({ skipSpecialTokens: false })
  const eos = tokenizer.specialTokens.eos
  while (!stopped.has(id)) {
    stats.outputTokens++
    if (token === eos) break
    if (onGpu) engine.markSeen(token)
    else sampler.see([token])
    const text = decoder.push(token)
    if (text) post({ type: "token", id, text })
    if (stats.outputTokens >= maxTokens) break
    row = await engine.forwardHidden([token])
    model.cached.push(token)
    token = await next()
  }
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
  if (data.type === "load") {
    void startLoad(data.model).catch(() => undefined)
  } else if (data.type === "stop") {
    stopped.add(data.id)
  } else {
    queue = queue
      .then(() => generate(data))
      .catch((error: unknown) => {
        // A lost GPU context can't be used again: the next generation
        // loads the model afresh.
        if (current?.engine.gpu.isLost) current = null
        post({ type: "error", id: data.id, message: errorMessage(error) })
      })
      .finally(() => stopped.delete(data.id))
  }
}
