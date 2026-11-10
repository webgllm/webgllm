/**
 * WebGLLM: the public API. OpenAI's Chat Completions and Models on models
 * running in the page, each in a Web Worker on WebGL 2.
 */
import {
  buildCompletion,
  createChunkWriter,
  createReplyReader,
  planCompletion,
  planRepairs,
  randomId,
  repairRequest,
  timingsOf,
  usageOf
} from "./chat-completions.js"
import type { CompletionPlan, ReplyDelta } from "./chat-completions.js"
import type { ParsedToolCall } from "./tool-calls.js"
import type { RunTimings } from "./protocol.js"
import { defaultWorker, hasWebGl2 } from "./client.js"
import { abortable, abortError, throwIfAborted, WebGLLMError } from "./errors.js"
import { browserStorage, fileKey, storedModelFiles } from "./model-files.js"
import type { ModelStorage } from "./model-files.js"
import { defaultMemoryBudget } from "./model-memory.js"
import { knownModelMeta, knownModels, modelMeta, rememberModel } from "./model-meta.js"
import type { WebGlModelId } from "./model-meta.js"
import { isModelId, resolveModel } from "./resolve-model.js"
import type {
  ChatCompletion,
  ChatCompletionChunk,
  ChatCompletionCreateParams,
  ChatCompletionCreateParamsNonStreaming,
  ChatCompletionCreateParamsStreaming,
  LoadProgress,
  ModelInfo,
  WebGLLMEventMap
} from "./openai.js"
import { ModelPool } from "./pool.js"
import type { LogLevel } from "./pool.js"

export type { LogLevel }

export type WebGLLMOptions = {
  /** Every load's progress, however it started (loadModel, a completion, downloadModel). */
  onProgress?: (progress: LoadProgress) => void
  /** The context a completion loads a model for: the model's own when left out. */
  contextLength?: number
  /** Bytes all loaded models may take together: picked from the device when left out. */
  memoryBudget?: number
  /** Which worker and engine messages reach the console ("warn" by default). */
  logLevel?: LogLevel
  /**
   * A Web Worker of your own to run models in, in place of the ones webgllm
   * starts: one running `WebGLLMWorkerHandler`. It holds one model, so
   * loading another replaces the one there (whatever `keepLoaded` says), and
   * unloading frees the model and leaves the worker running.
   */
  worker?: Worker
  /**
   * Called each time a model's worker claims memory, with the bytes the
   * model will hold in all (replacing its previous claim). The worker
   * allocates only after the returned promise resolves, so the host can
   * make room elsewhere first. Rejecting fails the load (or the cache
   * growth) as a refused claim does. `signal` aborts when the model is
   * unloaded or the claimant goes away; stop waiting then.
   * The model's memory is free again at its "unload" event.
   */
  onMemoryClaim?: (claim: { model: string; bytes: number; signal: AbortSignal }) => Promise<void> | void
}

export type LoadModelOptions = {
  /**
   * Most tokens a chat may hold, prompt and reply together: the instance's
   * `contextLength`, else the model's own, when left out (never more than
   * the model's own). Loaded for another value, the model loads again.
   */
  contextLength?: number
  /** Never unloaded to make room for another model past the memory budget. */
  keepLoaded?: boolean
  onProgress?: (progress: LoadProgress) => void
  /** Aborting stops the load (unless another call is waiting for it too). */
  signal?: AbortSignal
}

export type DownloadModelOptions = {
  onProgress?: (progress: LoadProgress) => void
  /** Aborting stops the download; what came is kept, and the next one resumes. */
  signal?: AbortSignal
}

export type RequestOptions = {
  /** Aborting rejects with an AbortError and stops the generation. */
  signal?: AbortSignal
  /** The model's load progress, when this call loads it. */
  onProgress?: (progress: LoadProgress) => void
}

const LEVELS: Record<LogLevel, number> = { debug: 0, info: 1, warn: 2, error: 3, silent: 4 }

/**
 * `model` resolved (the catalog's meta, or one built from the Hub) so its
 * meta can be read synchronously from here on.
 */
async function resolved(model: unknown, signal?: AbortSignal): Promise<string> {
  throwIfAborted(signal)
  if (typeof model === "string" && knownModelMeta(model)) return model
  if (typeof model !== "string") {
    throw new WebGLLMError("model-not-supported", `${String(model)} isn't a model webgllm runs (see models.list()).`)
  }
  return (await resolveModel(model, signal)).name
}

const notSupported = (model: unknown) =>
  new WebGLLMError("model-not-supported", `${String(model)} isn't a model webgllm runs (see models.list()).`)

/** What the tests swap (storage, the workers webgllm starts, the support check); not exported from the package. */
export const internals = {
  storage: browserStorage as ModelStorage,
  createWorker: defaultWorker as () => Worker,
  hasWebGl2
}

export class WebGLLM {
  /** Whether this browser can run the engine (WebGL 2 with float render targets, in a worker). */
  static isSupported(): boolean {
    return internals.hasWebGl2()
  }

  /**
   * Where requests go, as an OpenAI client reports it: in this page, not over
   * the network. With `chat.completions.create`, it's what an OpenAI client
   * offers that TanStack AI's Chat Completions adapters use for streaming and
   * structured output; it isn't a whole OpenAI client (no `withOptions`, so no
   * `wrapFetch`), and its type isn't theirs.
   */
  readonly baseURL = "webgllm:"
  readonly chat: { readonly completions: Completions }
  readonly models: Models
  private readonly pool: ModelPool
  /** Listeners by event type (typed at addEventListener and dispatch). */
  private readonly listeners = new Map<keyof WebGLLMEventMap, Set<(event: unknown) => void>>()
  private readonly storage: ModelStorage
  private checked = false

  constructor(private readonly options: WebGLLMOptions = {}) {
    const level = LEVELS[options.logLevel ?? "warn"]
    this.storage = internals.storage
    this.pool = new ModelPool({
      createWorker: internals.createWorker,
      ...(options.worker && { worker: options.worker }),
      memoryBudget: () => options.memoryBudget ?? defaultMemoryBudget(),
      storage: this.storage,
      onProgress: (progress) => {
        options.onProgress?.(progress)
        this.dispatch("progress", progress)
      },
      onModelEvent: ({ type, ...event }) => this.dispatch(type, event as never),
      ...(options.onMemoryClaim && { onMemoryClaim: options.onMemoryClaim }),
      log: (at, text) => {
        if (LEVELS[at] < level) return
        if (at === "error") console.error(text)
        else if (at === "warn") console.warn(text)
        else console.info(text)
      }
    })
    this.chat = {
      completions: new Completions({ checkSupport: () => this.checkSupport(), contextLength: options.contextLength }, this.pool)
    }
    this.models = new Models(this, this.pool, this.storage)
  }

  /**
   * Listens for load progress ("progress") and models' lifecycle:
   * "loadstart", "load" (ready), "error" and "unload", whatever started
   * them: loadModel, a completion loading on demand, or room made for
   * another model.
   */
  addEventListener<K extends keyof WebGLLMEventMap>(type: K, listener: (event: WebGLLMEventMap[K]) => void): void {
    let set = this.listeners.get(type)
    if (!set) this.listeners.set(type, (set = new Set()))
    set.add(listener as (event: unknown) => void)
  }

  removeEventListener<K extends keyof WebGLLMEventMap>(type: K, listener: (event: WebGLLMEventMap[K]) => void): void {
    this.listeners.get(type)?.delete(listener as (event: unknown) => void)
  }

  private dispatch<K extends keyof WebGLLMEventMap>(type: K, event: WebGLLMEventMap[K]) {
    // A throwing listener mustn't break the engine's bookkeeping.
    for (const listener of this.listeners.get(type) ?? []) {
      try {
        listener(event)
      } catch (error) {
        console.error(error)
      }
    }
  }

  /**
   * Loads `model` (downloading it the first time) for chats of up to
   * `contextLength` tokens. Other models stay loaded while they fit the
   * memory budget; past it, the least recently used go (not ones kept loaded).
   * In a worker of your own (`worker`), it replaces the model loaded there.
   */
  async loadModel(model: WebGlModelId, options: LoadModelOptions = {}): Promise<void> {
    throwIfAborted(options.signal)
    // Before resolving: an unload asked for meanwhile stops this load.
    const since = typeof model === "string" ? this.pool.unloadMark(model) : 0
    // (Without a wait for a known model: its load starts in this call.)
    const id = typeof model === "string" && knownModelMeta(model) ? model : await resolved(model, options.signal)
    this.checkSupport()
    await this.pool.load(id, {
      since,
      contextLength: options.contextLength ?? this.options.contextLength,
      ...(options.keepLoaded !== undefined && { keepLoaded: options.keepLoaded }),
      ...(options.onProgress && { onProgress: options.onProgress }),
      ...(options.signal && { signal: options.signal })
    })
  }

  /**
   * Unloads `model`, or every model, freeing its GPU memory; its running
   * generation fails. webgllm's own worker for the model ends; a worker of
   * your own (`worker`) frees the model and stays running.
   */
  async unloadModel(model?: WebGlModelId): Promise<void> {
    this.pool.unload(model)
  }

  /**
   * Downloads `model`'s files to the browser's storage, without loading it:
   * what it loads for text. A model that reads images downloads its vision
   * encoder with the first request that has an image (so images need the
   * network once).
   */
  async downloadModel(model: WebGlModelId, options: DownloadModelOptions = {}): Promise<void> {
    await this.pool.download(await resolved(model, options.signal), options)
  }

  /** The page can't run the engine: say so before a worker fails at it. */
  private checkSupport() {
    if (this.checked) return
    if (!internals.hasWebGl2()) {
      throw new WebGLLMError("webgl2-unavailable", "This browser can't run webgllm: it needs WebGL 2 with float render targets in a worker.")
    }
    this.checked = true
  }
}

/** What completions need of their WebGLLM. */
type CompletionContext = {
  checkSupport: () => void
  /** The context completions load models for (undefined: the model's own). */
  contextLength: number | undefined
}

class Completions {
  constructor(
    private readonly llm: CompletionContext,
    private readonly pool: ModelPool
  ) {}

  /**
   * A chat completion, as OpenAI's `chat.completions.create`: a whole
   * completion, or with `stream: true` its chunks. Loads `body.model`
   * first if it isn't loaded.
   */
  create(body: ChatCompletionCreateParamsStreaming, options?: RequestOptions): Promise<AsyncIterable<ChatCompletionChunk>>
  create(body: ChatCompletionCreateParamsNonStreaming, options?: RequestOptions): Promise<ChatCompletion>
  create(body: ChatCompletionCreateParams, options?: RequestOptions): Promise<ChatCompletion | AsyncIterable<ChatCompletionChunk>>
  async create(body: ChatCompletionCreateParams, options: RequestOptions = {}): Promise<ChatCompletion | AsyncIterable<ChatCompletionChunk>> {
    // Any Hugging Face model: resolved before the request is checked against it.
    const { signal, onProgress } = options
    throwIfAborted(signal)
    const named = body && typeof body === "object" && typeof body.model === "string" && body.model ? body.model : undefined
    // Before resolving: an unload asked for meanwhile stops this call.
    const since = named === undefined ? undefined : this.pool.unloadMark(named)
    if (named !== undefined && !knownModelMeta(named)) await resolveModel(named, signal)
    const plan = planCompletion(body)
    if (!this.pool.isLoaded(plan.model)) this.llm.checkSupport()
    // Loaded before the call resolves, so load failures reject it.
    await this.pool.ensureLoaded(plan.model, this.llm.contextLength, onProgress, signal, since)
    const meta = { id: randomId("chatcmpl-"), created: Math.floor(Date.now() / 1000), model: plan.model }
    return plan.stream ? this.stream(plan, meta, options) : this.complete(plan, meta, options)
  }

  private async complete(plan: CompletionPlan, meta: { id: string; created: number; model: string }, options: RequestOptions): Promise<ChatCompletion> {
    const reader = readerFor(plan)
    const deltas: Array<ReplyDelta> = []
    const end = await abortable(
      this.pool.generate(plan.model, plan.generate, {
        contextLength: this.llm.contextLength,
        onToken: (text) => {
          for (const d of reader.push(text)) deltas.push(d)
        },
        ...(options.onProgress && { onProgress: options.onProgress }),
        ...(options.signal && { signal: options.signal })
      }),
      options.signal
    )
    const last = reader.finish(end)
    for (const d of last.deltas) deltas.push(d)
    const settled = await this.settleToolCalls(plan, last, options.signal)
    for (const d of settled.deltas) deltas.push(d)
    const toolCalls = reader.toolCalls(settled.calls)
    return buildCompletion(meta, deltas, toolCalls, toolCalls.length ? "tool_calls" : last.finishReason, end.stats, settled.repairs)
  }

  /**
   * Checks an automatic reply's tool calls and generates bad ones again,
   * each constrained to its tool's schema (see planRepairs). Unrepaired
   * malformed markup is shown as the text it is.
   */
  private async settleToolCalls(
    plan: CompletionPlan,
    last: { calls: Array<ParsedToolCall>; malformed?: string },
    signal: AbortSignal | undefined
  ): Promise<{ calls: Array<ParsedToolCall>; deltas: Array<ReplyDelta>; repairs: Array<RunTimings> }> {
    const calls = [...last.calls]
    const repairs: Array<RunTimings> = []
    let malformed = last.malformed
    if (plan.parseTools && plan.repairToolCalls) {
      for (const repair of await planRepairs(calls, malformed, plan.generate.tools)) {
        throwIfAborted(signal)
        const end = await abortable(
          this.pool.generate(plan.model, repairRequest(plan, repair), {
            contextLength: this.llm.contextLength,
            // A forced call comes back in "done"; any thinking before it isn't shown.
            onToken: () => undefined,
            ...(signal && { signal })
          }),
          signal
        )
        if (!end.toolCall) throw new Error("A tool call couldn't be repaired.")
        repairs.push(end.stats)
        calls[repair.index] = { name: end.toolCall.name, args: end.toolCall.arguments }
        malformed = undefined
      }
    }
    return { calls, deltas: malformed !== undefined ? [{ content: malformed }] : [], repairs }
  }

  /**
   * The chunks as the worker writes the reply. The generation starts at
   * once; chunks wait for the reader. Leaving the loop early stops it.
   */
  private stream(plan: CompletionPlan, meta: { id: string; created: number; model: string }, options: RequestOptions): AsyncIterable<ChatCompletionChunk> {
    const reader = readerFor(plan)
    const writer = createChunkWriter(meta)
    const stop = new AbortController()
    const signal = options.signal
    const chunks: Array<ChatCompletionChunk> = []
    let finished = false
    // Set by return(): reads end, and later messages are dropped.
    let closed = false
    let failure: { error: unknown } | null = null
    let wake: (() => void) | null = null
    const emit = (...more: Array<ChatCompletionChunk>) => {
      if (!closed) chunks.push(...more)
    }
    const notify = () => {
      wake?.()
      wake = null
    }
    const onAbort = () => {
      stop.abort(signal?.reason)
      notify()
    }
    if (signal?.aborted) stop.abort(signal.reason)
    else signal?.addEventListener("abort", onAbort, { once: true })
    this.pool
      .generate(plan.model, plan.generate, {
        contextLength: this.llm.contextLength,
        onToken: (text) => {
          for (const d of reader.push(text)) emit(writer.delta(d))
          notify()
        },
        ...(options.onProgress && { onProgress: options.onProgress }),
        signal: stop.signal
      })
      .then(async (end) => {
        const last = reader.finish(end)
        for (const d of last.deltas) emit(writer.delta(d))
        notify()
        const settled = await this.settleToolCalls(plan, last, stop.signal)
        for (const d of settled.deltas) emit(writer.delta(d))
        const toolCalls = reader.toolCalls(settled.calls)
        emit(...writer.toolCalls(toolCalls))
        const timings = timingsOf(end.stats)
        emit(writer.finish(toolCalls.length ? "tool_calls" : last.finishReason, plan.includeUsage ? undefined : timings))
        if (plan.includeUsage) emit(writer.usage(usageOf(end.stats, settled.repairs), timings))
      })
      .then(
        () => undefined,
        (error: unknown) => {
          failure = { error }
        }
      )
      .finally(() => {
        finished = true
        signal?.removeEventListener("abort", onAbort)
        notify()
      })

    const read = async (): Promise<IteratorResult<ChatCompletionChunk>> => {
      for (;;) {
        if (closed) return { value: undefined, done: true }
        if (signal?.aborted) throw abortError(signal)
        const chunk = chunks.shift()
        if (chunk) return { value: chunk, done: false }
        if (finished) {
          if (failure) {
            const { error } = failure
            failure = null
            throw error
          }
          return { value: undefined, done: true }
        }
        await new Promise<void>((resolve) => {
          wake = resolve
        })
      }
    }
    // Reads run one at a time, in call order: two waiting at once would
    // share the one wake callback, and one would never wake.
    let reads: Promise<unknown> = Promise.resolve()
    const iterator: AsyncIterator<ChatCompletionChunk> = {
      next() {
        const result = reads.then(read)
        reads = result.catch(() => undefined)
        return result
      },
      async return() {
        // The reader left early: stop the worker too.
        if (!finished) stop.abort(abortError())
        closed = true
        chunks.length = 0
        notify()
        return { value: undefined, done: true }
      }
    }
    return { [Symbol.asyncIterator]: () => iterator }
  }
}

const readerFor = (plan: CompletionPlan) =>
  createReplyReader({
    format: modelMeta(plan.model).toolFormat ?? "lfm3",
    parseTools: plan.parseTools,
    parallelToolCalls: plan.parallelToolCalls
  })

class Models {
  constructor(
    private readonly llm: WebGLLM,
    private readonly pool: ModelPool,
    private readonly storage: ModelStorage
  ) {}

  /**
   * The catalog's models, and any other Hugging Face model used on this
   * page, with whether each is downloaded and loaded.
   */
  async list(): Promise<Array<ModelInfo>> {
    return Promise.all((await this.allModels()).map((model) => this.info(model)))
  }

  /** The catalog, models used on the page, and downloaded ones (from their stored meta). */
  private async allModels(): Promise<Array<string>> {
    for (const meta of await this.storage.manifests()) if (!knownModelMeta(meta.name)) rememberModel(meta)
    return knownModels()
  }

  /** `model`'s info: any Hugging Face model with a supported architecture, resolved first. */
  async retrieve(model: WebGlModelId): Promise<ModelInfo> {
    return this.info(await resolved(model))
  }

  /**
   * Whether `model` is ready to load without a download: every file it
   * loads (for text; see downloadModel) is in storage, or it's loaded. Reads storage only, with no network
   * request (unlike retrieve, which also asks for sizes and context length).
   */
  async downloaded(model: WebGlModelId): Promise<boolean> {
    if (!isModelId(model)) throw notSupported(model)
    // A model outside the catalog is downloaded only with its stored meta: no Hub request.
    let meta = knownModelMeta(model)
    if (!meta) {
      meta = await this.storage.manifest(model)
      if (!meta) return false
      rememberModel(meta)
    }
    return this.pool.isLoaded(model) || this.storage.downloaded(model)
  }

  /** Removes `model`'s files from storage, unloading it first if it's loaded. */
  async delete(model: WebGlModelId): Promise<void> {
    const id = await resolved(model)
    await this.llm.unloadModel(id)
    this.pool.cancelDownload(id)
    // Files another downloaded or loaded model also reads stay.
    const keep = new Set<string>()
    for (const other of await this.allModels()) {
      if (other === id) continue
      if (this.pool.isLoaded(other) || (await this.storage.downloaded(other))) {
        for (const file of storedModelFiles(modelMeta(other))) keep.add(fileKey(file))
      }
    }
    // A stored meta is a downloaded model, even partly: its files stay.
    for (const meta of await this.storage.manifests()) {
      if (meta.name !== id) for (const file of storedModelFiles(meta)) keep.add(fileKey(file))
    }
    await this.storage.remove(id, keep)
  }

  private async info(model: string): Promise<ModelInfo> {
    const meta = modelMeta(model)
    const loaded = this.pool.isLoaded(model)
    const [contextLength, size, downloaded] = await Promise.all([
      this.storage.contextLength(model),
      this.storage.size(model),
      loaded ? Promise.resolve(true) : this.storage.downloaded(model)
    ])
    const requested = this.pool.requestedContext(model)
    return {
      id: model,
      format: meta.format,
      contextLength,
      size,
      downloaded,
      loaded,
      ...(loaded && {
        loadedContextLength:
          this.pool.loadedContext(model) ??
          (requested === undefined ? contextLength : contextLength ? Math.min(requested, contextLength) : requested)
      }),
      supports: {
        tools: meta.supports.capabilities.includes("tools"),
        thinking: meta.supports.capabilities.includes("thinking"),
        images: meta.supports.input.includes("image")
      }
    }
  }
}
