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
import type {
  StructuredOutputOptions,
  StructuredOutputResult
} from "@tanstack/ai/adapters"

import { generate, loadModel } from "./client.js"
import { modelMeta } from "./model-meta.js"
import type { WebGlModelReasoningByName, WebGlTextModel } from "./model-meta.js"
import type { RunTimings } from "./protocol.js"
import {
  createThinkingSplitter,
  createToolCallSplitter,
  parseToolCalls,
  splitThinking
} from "./reply.js"
import type { Sampling } from "./sampler.js"
import type { ChatMessage, ChatTool } from "./tokenizer.js"

/** Per-call `modelOptions`: sampling over the model card's, and the most tokens to generate. */
export type WebGlTextProviderOptions = Partial<Sampling> & {
  /** Most tokens to generate: 4,096 by default. */
  max_new_tokens?: number
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

/** The template's `enable_thinking` for a request; the template's default when it doesn't ask. */
function enableThinking(
  request: ReasoningRequest | undefined,
  model: WebGlTextModel
) {
  const value = resolveReasoning(request, modelMeta(model).reasoning)?.value
  return value === undefined ? undefined : value === "true"
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
 * The conversation as chat-template messages. Tool calls and results from
 * earlier turns are left out and their answers kept: small models copy the
 * shape of past calls badly and get lost in old results. The current
 * turn's calls and results stay, with its thinking (Liquid's guidance for
 * tool use), since the model answers from them.
 */
function toChatMessages(messages: Array<ModelMessage>): Array<ChatMessage> {
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
    const thinking = m.thinking?.map((t) => t.content).join("")
    chat.push({
      role: "assistant",
      content,
      tool_calls: calls.map((c) => ({
        id: c.id,
        type: "function",
        function: { name: c.name, arguments: c.args }
      })),
      ...(thinking && { thinking })
    })
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

  constructor(model: TModel) {
    super({}, model)
  }

  private sampling(
    modelOptions: WebGlTextProviderOptions | undefined
  ): Sampling {
    const base = modelMeta(this.model).sampling
    const top_p = modelOptions?.top_p ?? base.top_p
    return {
      temperature: modelOptions?.temperature ?? base.temperature,
      top_k: modelOptions?.top_k ?? base.top_k,
      repetition_penalty:
        modelOptions?.repetition_penalty ?? base.repetition_penalty,
      ...(top_p !== undefined && { top_p })
    }
  }

  async *chatStream(
    options: TextOptions<WebGlTextProviderOptions>
  ): AsyncIterable<AdapterYieldChunk> {
    const runId = options.runId ?? nextId("run")
    const threadId = options.threadId
    const messageId = nextId("msg")
    const reasoningId = nextId("reasoning")
    const signal =
      options.abortController?.signal ??
      (options.request as RequestInit | undefined)?.signal ??
      undefined
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
    try {
      await loadModel(model)
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
      ...toChatMessages(options.messages)
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
      id: nextId("call"),
      model: this.model,
      chat,
      tools: toTools(options.tools),
      thinking: enableThinking(options.reasoning, model),
      sampling: this.sampling(options.modelOptions),
      maxTokens: options.modelOptions?.max_new_tokens,
      signal
    })
    let next = await stream.next()
    while (!next.done) {
      addPieces(thinkingSplitter.push(next.value))
      yield* events.splice(0)
      next = await stream.next()
    }
    const { failure, stats } = next.value
    addPieces(thinkingSplitter.flush())
    endReasoning()
    const { text: rest, markup } = toolSplitter.flush()
    addText(rest)
    const parsed = markup ? parseToolCalls(markup) : []
    // Markup that isn't any tool call is shown as the text it is.
    if (markup && parsed.length === 0) addText(markup)
    yield* events.splice(0)

    for (const [index, call] of parsed.entries()) {
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
      ...(stats && { usage: usageOf(stats) })
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
    await loadModel(this.model)
    const system = normalizeSystemPrompts(chatOptions.systemPrompts)
      .map((p) => p.content)
      .concat(
        `Answer with only a JSON value matching this JSON Schema:\n${JSON.stringify(outputSchema)}`
      )
      .join("\n\n")
    const stream = generate({
      id: nextId("call"),
      model: this.model,
      chat: [
        { role: "system", content: system },
        ...toChatMessages(chatOptions.messages)
      ],
      tools: [],
      thinking: enableThinking(chatOptions.reasoning, this.model),
      sampling: this.sampling(chatOptions.modelOptions),
      maxTokens: chatOptions.modelOptions?.max_new_tokens,
      signal: chatOptions.abortController?.signal
    })
    let raw = ""
    let next = await stream.next()
    while (!next.done) {
      raw += next.value
      next = await stream.next()
    }
    if (next.value.failure) throw new Error(next.value.failure)
    const rawText = splitThinking(raw).answer.trim()
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
  model: TModel
): WebGlTextAdapter<TModel> {
  return new WebGlTextAdapter(model)
}
