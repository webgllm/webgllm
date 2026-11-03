/**
 * A TanStack AI text adapter for LFM3 running on WebGL 2 in a Web Worker,
 * for browsers without WebGPU. Hand it to `chat()` like any hosted model's
 * adapter: `chat()` runs tools and the agent loop, and each `chatStream`
 * call here is one generation in the worker, streamed back as AG-UI events.
 */
import { buildBaseUsage, EventType, normalizeSystemPrompts } from "@tanstack/ai"
import type {
  AdapterYieldChunk,
  ModelMessage,
  ReasoningRequest,
  TextOptions,
  TokenUsage,
  Tool
} from "@tanstack/ai"
import { resolveReasoning } from "@tanstack/ai/adapter-internals"
import { BaseTextAdapter } from "@tanstack/ai/adapters"
import { chooseGuidedTool, addGuidedUsage, validateStructuredValue, structuredAnswer } from "@tanstack-ai-browser/structured-output"
import type {
  StructuredOutputOptions,
  StructuredOutputResult
} from "@tanstack/ai/adapters"

import { generate, loadModel, setKeepLoaded } from "./client.js"
import { modelMeta } from "./model-meta.js"
import type { WebGlModelReasoningByName, WebGlTextModel } from "./model-meta.js"
import type { GenerationPhase, RunTimings } from "./protocol.js"
import { createThinkingSplitter } from "./reply.js"
import type { ReplyPiece } from "./reply.js"
import type { Sampling } from "./sampler.js"
import type { ChatMessage, ChatTool } from "./tokenizer.js"
import {
  createToolCallSplitter,
  formatLfm2ToolCalls,
  parseToolCalls
} from "./tool-calls.js"
import type { ParsedToolCall, ToolFormat } from "./tool-calls.js"

/** Per-call `modelOptions`: sampling over the model card's, and the most tokens to generate. */
export type WebGlTextProviderOptions = Partial<Sampling> & {
  /** Maximum generated tokens, thinking included; otherwise EOS or remaining context. */
  maxTokens?: number
  /** Legacy alias for maxTokens. maxTokens takes precedence when both are supplied. */
  max_new_tokens?: number
  /**
   * Draft tokens by prompt lookup and check them in one batch (speculative
   * decoding; the tokens are the same as without). On by default.
   */
  speculative?: boolean
  /**
   * Read the likeliest next token forward while each token is drawn (the
   * tokens are the same as without). On by default.
   */
  pipeline?: boolean
}

/** Timings of one model call, in milliseconds, in `RUN_FINISHED`'s `usage.providerUsageDetails`. */
export type WebGlProviderUsageDetails = {
  /** From the request to the first generated token, reading the prompt included. */
  firstTokenMs?: number
  /** Reading the part of the prompt that wasn't cached. */
  promptMs?: number
  /** From the first generated token to the last. */
  decodeMs?: number
  /** Tokens generated after the first one, over `decodeMs`. */
  decodeTokens?: number
  /** Model-generated tokens inside the thinking block. */
  thinkingTokens?: number
  /** Explicit maximum thinking tokens requested for this generation. */
  thinkingBudgetTokens?: number
  /** Whether thinking reached the explicit budget. */
  thinkingBudgetExhausted?: boolean
}

type ResolveReasoning<TModel extends string> =
  TModel extends keyof WebGlModelReasoningByName
    ? WebGlModelReasoningByName[TModel]
    : never

type MetadataByModality = {
  text: unknown
  image: unknown
  audio: unknown
  video: unknown
  document: unknown
}

/** Resolve the template switch and only an explicitly requested thinking cap. */
function thinkingOptions(
  request: ReasoningRequest | undefined,
  model: WebGlTextModel
): { thinking: boolean | undefined; thinkingBudgetTokens?: number } {
  const budget = request?.budgetTokens
  if (budget !== undefined && (!Number.isSafeInteger(budget) || budget < 0)) {
    throw new RangeError("reasoning.budgetTokens must be a non-negative safe integer.")
  }
  const capability = modelMeta(model).reasoning
  const resolved = resolveReasoning(request, capability)
  const thinking = resolved?.value === undefined ? undefined : resolved.value === "true"
  // String levels and omitted budgets preserve unlimited thinking; off ignores a valid cap.
  if (budget === undefined || thinking === false) return { thinking }
  if (!capability || !capability.budget || !resolved) {
    throw new Error(`${model} does not support a thinking token budget.`)
  }
  return { thinking, thinkingBudgetTokens: resolved.budgetTokens }
}

/** Resolve a caller's output cap without introducing a provider-wide default. */
function requestedOutputTokens(options: WebGlTextProviderOptions | undefined) {
  const limit = options?.maxTokens ?? options?.max_new_tokens
  if (limit !== undefined && (!Number.isSafeInteger(limit) || limit < 0)) {
    throw new RangeError("maxTokens must be a nonnegative safe integer.")
  }
  return limit
}

function waitForModel(model: WebGlTextModel, contextLength: number | undefined, signal: AbortSignal | undefined): Promise<void> {
  signal?.throwIfAborted()
  if (!signal) return loadModel(model, contextLength)
  return new Promise((resolve, reject) => {
    const abort = () => reject(signal.reason)
    signal.addEventListener("abort", abort, { once: true })
    loadModel(model, contextLength).then(() => {
      signal.removeEventListener("abort", abort)
      if (signal.aborted) reject(signal.reason)
      else resolve()
    }, error => { signal.removeEventListener("abort", abort); reject(error) })
  })
}

function contentText(content: ModelMessage["content"]) {
  if (typeof content === "string") return content
  return (content ?? [])
    .map((part) => (part.type === "text" ? part.content : ""))
    .join("")
}

/** A tool call's text content, or the error a failed tool gave. */
function toolResultText(m: ModelMessage) {
  const text = contentText(m.content)
  if (m.error) return `Error: ${m.error}`
  return text && text !== "null" ? text : "The tool returned nothing."
}

function parseArguments(json: string): Record<string, unknown> {
  try {
    const args = JSON.parse(json) as unknown
    return args && typeof args === "object"
      ? (args as Record<string, unknown>)
      : {}
  } catch {
    return {}
  }
}

/**
 * An assistant turn with tool calls, written the way the model's template
 * takes them: LFM2's template doesn't read `tool_calls`, so its calls go in
 * the content in its own format; LFM3's template writes `tool_calls` itself.
 */
function assistantToolTurn(
  content: string,
  calls: Array<ParsedToolCall & { id: string }>,
  format: ToolFormat
): ChatMessage {
  if (format === "lfm2") {
    return { role: "assistant", content: content + formatLfm2ToolCalls(calls) }
  }
  return {
    role: "assistant",
    content,
    tool_calls: calls.map((c) => ({
      id: c.id,
      type: "function",
      function: { name: c.name, arguments: c.args }
    }))
  }
}

/** The current turn's thinking, under the key the format's template reads. */
function withThinking(
  turn: ChatMessage,
  thinking: string,
  format: ToolFormat
): ChatMessage {
  // Liquid's guide: keep the reasoning of the current tool-use turn.
  if (format === "lfm3") return { ...turn, thinking }
  if (format === "qwen3.5") return { ...turn, reasoning_content: thinking }
  return turn
}

/**
 * The conversation as chat-template messages. Tool calls and results from
 * earlier turns are left out and their answers kept: small models copy the
 * shape of past calls badly and get lost in old results. The current
 * turn's calls and results stay, with its thinking (Liquid's guidance for
 * tool use), since the model answers from them.
 */
function toChatMessages(
  messages: Array<ModelMessage>,
  format: ToolFormat
): Array<ChatMessage> {
  const chat: Array<ChatMessage> = []
  const currentTurn = messages.map((m) => m.role).lastIndexOf("user")
  const toolNames = new Map<string, string>()
  for (const [i, m] of messages.entries()) {
    const content = contentText(m.content)
    const earlier = i < currentTurn
    if (m.role === "tool") {
      if (earlier) continue
      const name = m.toolCallId ? toolNames.get(m.toolCallId) : undefined
      chat.push({
        role: "tool",
        content: toolResultText(m),
        ...(m.toolCallId && { tool_call_id: m.toolCallId }),
        ...(name && { name })
      })
      continue
    }
    const calls = (m.toolCalls ?? []).map((call) => {
      toolNames.set(call.id, call.function.name)
      return {
        id: call.id,
        name: call.function.name,
        args: parseArguments(call.function.arguments)
      }
    })
    if (m.role === "user" || calls.length === 0 || earlier) {
      if (content) chat.push({ role: m.role, content })
      continue
    }
    const turn = assistantToolTurn(content, calls, format)
    const thinking = m.thinking?.map((t) => t.content).join("")
    chat.push(thinking ? withThinking(turn, thinking, format) : turn)
  }
  return chat
}

/** Tools as chat templates take them (OpenAI's function format). */
function toTools(tools: Array<Tool> | undefined): Array<ChatTool> {
  return (tools ?? []).map((tool) => ({
    type: "function",
    function: {
      name: tool.name,
      description: tool.description,
      parameters: tool.inputSchema ?? {
        type: "object",
        properties: {},
        required: []
      }
    }
  }))
}

function usageOf(stats: RunTimings): TokenUsage {
  const usage = buildBaseUsage<WebGlProviderUsageDetails>({
    promptTokens: stats.inputTokens,
    completionTokens: stats.outputTokens,
    totalTokens: stats.inputTokens + stats.outputTokens
  })
  if (stats.cachedTokens > 0)
    usage.promptTokensDetails = { cachedTokens: stats.cachedTokens }
  usage.providerUsageDetails = {
    ...(stats.firstTokenMs !== null && { firstTokenMs: stats.firstTokenMs }),
    promptMs: stats.promptMs,
    ...(stats.thinkingTokens !== undefined && { thinkingTokens: stats.thinkingTokens }),
    ...(stats.thinkingBudgetTokens !== undefined && { thinkingBudgetTokens: stats.thinkingBudgetTokens }),
    ...(stats.thinkingBudgetExhausted !== undefined && { thinkingBudgetExhausted: stats.thinkingBudgetExhausted }),
    ...(stats.firstTokenMs !== null && {
      decodeMs: stats.decodeMs,
      decodeTokens: Math.max(stats.outputTokens - 1, 0)
    })
  }
  return usage
}

const errorText = (error: unknown) =>
  error instanceof Error ? error.message : String(error)

let calls = 0
const nextId = (prefix: string) =>
  `${prefix}_${Date.now().toString(36)}_${(calls++).toString(36)}`

/** Options for `webglText`. */
export type WebGlTextConfig = {
  /** Real worker prefill/decode transitions, independent of streamed text. */
  onGenerationPhase?: (phase: GenerationPhase) => void
  /** Raw model-generated thinking text, without tags or answer content. */
  onThinking?: (delta: string) => void
  /**
   * Most tokens a chat may hold, prompt and reply together: the model's
   * whole context length by default (from its config), never more. The
   * attention cache grows as a chat fills it either way; a smaller limit
   * refuses longer chats sooner. Changing it loads the model again.
   */
  contextLength?: number
  /**
   * Keeps the model loaded when another runtime (Transformers.js, wllama)
   * loads one past the page's memory budget (see `setMemoryBudget`); by
   * default the least recently used model is unloaded to make room.
   */
  keepLoaded?: boolean
}

export class WebGlTextAdapter<
  TModel extends WebGlTextModel
> extends BaseTextAdapter<
  TModel,
  WebGlTextProviderOptions,
  readonly ["text"],
  MetadataByModality,
  ReadonlyArray<string>,
  unknown,
  never,
  ResolveReasoning<TModel>
> {
  override readonly kind = "text" as const
  readonly name = "webgl" as const
  override readonly api = "webgl" as const
  override readonly inputModalities = ["text"] as const

  private readonly contextLength: number | undefined
  private readonly onGenerationPhase: WebGlTextConfig["onGenerationPhase"]
  private readonly onThinking: WebGlTextConfig["onThinking"]

  constructor(model: TModel, config: WebGlTextConfig = {}) {
    super({}, model)
    this.contextLength = config.contextLength
    this.onGenerationPhase = config.onGenerationPhase
    this.onThinking = config.onThinking
    setKeepLoaded(model, config.keepLoaded ?? false)
  }

  /** How the model writes tool calls. */
  private get toolFormat(): ToolFormat {
    return modelMeta(this.model).toolFormat ?? "lfm3"
  }

  private sampling(
    modelOptions: WebGlTextProviderOptions | undefined,
    thinking: boolean | undefined
  ): Sampling {
    const meta = modelMeta(this.model)
    const base = thinking ? meta.thinkingSampling ?? meta.sampling : meta.sampling
    const top_p = modelOptions?.top_p ?? base.top_p
    const min_p = modelOptions?.min_p ?? base.min_p
    const presence_penalty = modelOptions?.presence_penalty ?? base.presence_penalty
    return {
      temperature: modelOptions?.temperature ?? base.temperature,
      top_k: modelOptions?.top_k ?? base.top_k,
      repetition_penalty:
        modelOptions?.repetition_penalty ?? base.repetition_penalty,
      ...(top_p !== undefined && { top_p }),
      ...(min_p !== undefined && { min_p }),
      ...(presence_penalty !== undefined && { presence_penalty })
    }
  }

  async *chatStream(
    options: TextOptions<WebGlTextProviderOptions>
  ): AsyncIterable<AdapterYieldChunk> {
    const thinking = thinkingOptions(options.reasoning, this.model)
    const maxTokens = requestedOutputTokens(options.modelOptions)
    const runId = options.runId ?? nextId("run")
    const threadId = options.threadId
    const messageId = nextId("msg")
    const reasoningId = nextId("reasoning")
    const signal =
      options.abortController?.signal ??
      (options.request as RequestInit | undefined)?.signal ??
      undefined
    if (signal?.aborted) return
    const model = this.model
    const timestamp = () => Date.now()

    yield {
      type: EventType.RUN_STARTED,
      runId,
      threadId,
      model,
      timestamp: timestamp(),
      ...(options.parentRunId && { parentRunId: options.parentRunId })
    }
    if (options.tools?.length && options.toolChoice && options.toolChoice !== "auto" && options.toolChoice !== "none") {
      try {
        const { call, usage } = await chooseGuidedTool(options, (outputSchema, instruction) =>
          this.structuredOutput({
            chatOptions: { ...options, systemPrompts: [...normalizeSystemPrompts(options.systemPrompts), { content: instruction }] },
            outputSchema
          })
        )
        if (signal?.aborted) return
        if (call) {
          const toolCallId = nextId("tool")
          const args = JSON.stringify(call.args)
          yield { type: EventType.TOOL_CALL_START, toolCallId, toolCallName: call.name, parentMessageId: messageId, index: 0, model, timestamp: timestamp() }
          yield { type: EventType.TOOL_CALL_ARGS, toolCallId, delta: args, args, model, timestamp: timestamp() }
          yield { type: EventType.TOOL_CALL_END, toolCallId, toolCallName: call.name, args, input: call.args, model, timestamp: timestamp() }
          yield { type: EventType.RUN_FINISHED, runId, threadId, model, timestamp: timestamp(), finishReason: "tool_calls", ...(usage && { usage }) }
        } else {
          for await (const chunk of this.chatStream({ ...options, runId, tools: [], toolChoice: "none" })) {
            if (chunk.type === EventType.RUN_STARTED) continue
            if (chunk.type === EventType.RUN_FINISHED) {
              const reported = Array.isArray(chunk.usage) ? chunk.usage[0] : chunk.usage
              yield { ...chunk, usage: addGuidedUsage(usage, reported) }
            } else yield chunk
          }
        }
      } catch (error) {
        if (signal?.aborted) return
        const message = errorText(error)
        yield { type: EventType.RUN_ERROR, model, timestamp: timestamp(), message, error: { message } }
      }
      return
    }
    try {
      await waitForModel(model, this.contextLength, signal)
    } catch (error) {
      const message = `Couldn't load ${model}: ${errorText(error)}`
      yield {
        type: EventType.RUN_ERROR,
        model,
        timestamp: timestamp(),
        message,
        error: { message }
      }
      return
    }

    const system = normalizeSystemPrompts(options.systemPrompts)
      .map((p) => p.content)
      .join("\n\n")
    const chat: Array<ChatMessage> = [
      ...(system ? [{ role: "system" as const, content: system }] : []),
      ...toChatMessages(options.messages, this.toolFormat)
    ]

    let reasoning: "none" | "open" | "closed" = "none"
    let textStarted = false
    let text = ""
    const thinkingSplitter = createThinkingSplitter()
    const toolSplitter = createToolCallSplitter()
    const events: Array<AdapterYieldChunk> = []
    const endReasoning = () => {
      if (reasoning !== "open") return
      reasoning = "closed"
      events.push(
        {
          type: EventType.REASONING_MESSAGE_END,
          messageId: reasoningId,
          model,
          timestamp: timestamp()
        },
        {
          type: EventType.REASONING_END,
          messageId: reasoningId,
          model,
          timestamp: timestamp()
        }
      )
    }
    const addText = (delta: string) => {
      if (!delta) return
      if (!textStarted) {
        textStarted = true
        events.push({
          type: EventType.TEXT_MESSAGE_START,
          messageId,
          role: "assistant",
          model,
          timestamp: timestamp()
        })
      }
      text += delta
      events.push({
        type: EventType.TEXT_MESSAGE_CONTENT,
        messageId,
        delta,
        content: text,
        model,
        timestamp: timestamp()
      })
    }
    const addPieces = (pieces: ReturnType<typeof thinkingSplitter.push>) => {
      for (const piece of pieces) {
        if (piece.kind === "thinking") {
          if (!signal?.aborted) this.onThinking?.(piece.text)
          if (reasoning === "none") {
            events.push(
              {
                type: EventType.REASONING_START,
                messageId: reasoningId,
                model,
                timestamp: timestamp()
              },
              {
                type: EventType.REASONING_MESSAGE_START,
                messageId: reasoningId,
                role: "reasoning",
                model,
                timestamp: timestamp()
              }
            )
          }
          reasoning = "open"
          events.push({
            type: EventType.REASONING_MESSAGE_CONTENT,
            messageId: reasoningId,
            delta: piece.text,
            model,
            timestamp: timestamp()
          })
        } else {
          endReasoning()
          addText(toolSplitter.push(piece.text))
        }
      }
    }

    const stream = generate({
      onPhase: this.onGenerationPhase,
      id: nextId("call"),
      model: this.model,
      contextLength: this.contextLength,
      chat,
      tools: options.toolChoice === "none" ? [] : toTools(options.tools),
      ...thinking,
      sampling: this.sampling(options.modelOptions, thinking.thinking),
      maxTokens,
      speculative: options.modelOptions?.speculative,
      pipeline: options.modelOptions?.pipeline,
      signal
    })
    let next = await stream.next()
    try {
      while (!next.done) {
        addPieces(thinkingSplitter.push(next.value))
        yield* events.splice(0)
        next = await stream.next()
      }
    } finally {
      // Stopped reading early (a break, or return() on this stream): close
      // the generation, which tells the worker to stop.
      if (!next.done) await stream.return({})
    }
    const { failure, stats } = next.value
    addPieces(thinkingSplitter.flush())
    endReasoning()
    const { text: rest, markup } = toolSplitter.flush()
    addText(rest)
    const parsed = markup && options.tools?.length && options.toolChoice !== "none" ? parseToolCalls(markup) : []
    // Markup that isn't any tool call is shown as the text it is.
    if (markup && parsed.length === 0) addText(markup)
    yield* events.splice(0)

    let repairedUsage: TokenUsage | undefined
    const validated: typeof parsed = []
    try {
      if (markup && options.tools?.length && options.toolChoice !== "none" && parsed.length === 0) {
        throw new Error("The model produced a malformed tool-call envelope.")
      }
      for (const call of parsed) {
        const tool = options.tools?.find(tool => tool.name === call.name)
        if (!tool) throw new Error(`The model requested an unknown tool: ${call.name}`)
        const schema = tool.inputSchema ?? { type: "object", properties: {}, additionalProperties: false }
        let args = call.args
        if (!await validateStructuredValue(schema, args)) {
          const repair = await this.structuredOutput({
            chatOptions: { ...options, systemPrompts: [...normalizeSystemPrompts(options.systemPrompts), { content: `Return valid arguments for ${tool.name}. Previous draft: ${JSON.stringify(args)}` }] },
            outputSchema: schema
          })
          if (!repair.data || typeof repair.data !== "object" || Array.isArray(repair.data) || !await validateStructuredValue(schema, repair.data)) throw new Error(`Invalid arguments for tool: ${tool.name}`)
          args = repair.data as Record<string, unknown>
          repairedUsage = addGuidedUsage(repairedUsage, repair.usage)
        }
        if (signal?.aborted) return
        validated.push({ ...call, args })
      }
    } catch (error) {
      if (signal?.aborted) return
      const message = errorText(error)
      yield { type: EventType.RUN_ERROR, model, timestamp: timestamp(), message, error: { message } }
      return
    }

    for (const [index, call] of validated.entries()) {
      const toolCallId = nextId("tool")
      const args = JSON.stringify(call.args)
      yield {
        type: EventType.TOOL_CALL_START,
        toolCallId,
        toolCallName: call.name,
        parentMessageId: messageId,
        index,
        model,
        timestamp: timestamp()
      }
      yield {
        type: EventType.TOOL_CALL_ARGS,
        toolCallId,
        delta: args,
        args,
        model,
        timestamp: timestamp()
      }
      yield {
        type: EventType.TOOL_CALL_END,
        toolCallId,
        toolCallName: call.name,
        args,
        input: call.args,
        model,
        timestamp: timestamp()
      }
    }
    if (textStarted) {
      yield {
        type: EventType.TEXT_MESSAGE_END,
        messageId,
        model,
        timestamp: timestamp()
      }
    }
    if (failure) {
      yield {
        type: EventType.RUN_ERROR,
        model,
        timestamp: timestamp(),
        message: failure,
        error: { message: failure }
      }
      return
    }
    yield {
      type: EventType.RUN_FINISHED,
      runId,
      threadId,
      model,
      timestamp: timestamp(),
      finishReason: parsed.length > 0 ? "tool_calls" : "stop",
      ...((stats || repairedUsage) && { usage: repairedUsage ? addGuidedUsage(repairedUsage, stats ? usageOf(stats) : undefined) : stats ? usageOf(stats) : undefined })
    }
  }

  /**
   * A JSON answer for `chat({ outputSchema })`: the model is asked for JSON
   * matching the schema and its answer parsed; `chat()` validates it.
   */
  async structuredOutput(
    options: StructuredOutputOptions<WebGlTextProviderOptions>
  ): Promise<StructuredOutputResult<unknown>> {
    const { chatOptions, outputSchema } = options
    const thinking = thinkingOptions(chatOptions.reasoning, this.model)
    const maxTokens = requestedOutputTokens(chatOptions.modelOptions)
    const signal = chatOptions.abortController?.signal ?? (chatOptions.request as RequestInit | undefined)?.signal ?? undefined
    signal?.throwIfAborted()
    await waitForModel(this.model, this.contextLength, signal)
    signal?.throwIfAborted()
    const system = normalizeSystemPrompts(chatOptions.systemPrompts)
      .map((p) => p.content)
      .concat(
        `Answer with only a JSON value matching this JSON Schema:\n${JSON.stringify(outputSchema)}`
      )
      .join("\n\n")
    const stream = generate({
      onPhase: this.onGenerationPhase,
      id: nextId("call"),
      model: this.model,
      contextLength: this.contextLength,
      chat: [
        { role: "system", content: system },
        ...toChatMessages(chatOptions.messages, this.toolFormat)
      ],
      tools: [],
      outputSchema,
      ...thinking,
      sampling: this.sampling(chatOptions.modelOptions, thinking.thinking),
      maxTokens,
      speculative: chatOptions.modelOptions?.speculative,
      pipeline: chatOptions.modelOptions?.pipeline,
      signal
    })
    let raw = ""
    const thinkingSplitter = this.onThinking ? createThinkingSplitter() : undefined
    const addThinking = (pieces: Array<ReplyPiece>) => {
      if (signal?.aborted) return
      for (const piece of pieces) {
        if (piece.kind === "thinking") this.onThinking?.(piece.text)
      }
    }
    let next = await stream.next()
    try {
      while (!next.done) {
        signal?.throwIfAborted()
        raw += next.value
        if (thinkingSplitter) addThinking(thinkingSplitter.push(next.value))
        next = await stream.next()
      }
      if (thinkingSplitter) addThinking(thinkingSplitter.flush())
    } finally {
      if (!next.done) await stream.return({})
    }
    signal?.throwIfAborted()
    if (next.value.failure) throw new Error(next.value.failure)
    const rawText = structuredAnswer(raw)
    const json = rawText.replace(/^```(?:json)?\s*|\s*```$/g, "")
    try {
      return {
        data: JSON.parse(json) as unknown,
        rawText,
        ...(next.value.stats && { usage: usageOf(next.value.stats) })
      }
    } catch {
      throw new Error(
        `The model's answer isn't JSON: ${rawText.slice(0, 200)}${rawText.length > 200 ? "…" : ""}`
      )
    }
  }
}

/**
 * A TanStack AI text adapter for LFM3 on WebGL 2, for browsers without
 * WebGPU (Firefox on Android, for one).
 *
 * @example
 * ```ts
 * chat({ adapter: webglText("LiquidAI/LFM3-1.7B-GGUF"), messages, threadId })
 * ```
 */
export function webglText<TModel extends WebGlTextModel>(
  model: TModel,
  config?: WebGlTextConfig
): WebGlTextAdapter<TModel> {
  return new WebGlTextAdapter(model, config)
}
