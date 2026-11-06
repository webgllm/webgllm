/**
 * OpenAI's Chat Completions on the engine: a request checked and turned
 * into the worker's generate request, and the worker's raw text (thinking
 * and tool-call markup inline, as the model writes them) turned back into
 * a completion or its chunks. No worker here: webgllm.ts runs it.
 */
import { validateStructuredValue } from "./constraint/index.js"
import { WebGLLMError } from "./errors.js"
import { modelMeta, WEBGL_MODEL_META } from "./model-meta.js"
import type { WebGlModelMeta, WebGlTextModel } from "./model-meta.js"
import type {
  ChatCompletion,
  ChatCompletionChunk,
  ChatCompletionChunkDelta,
  ChatCompletionContent,
  ChatCompletionCreateParams,
  ChatCompletionMessageParam,
  ChatCompletionMessageToolCall,
  ChatCompletionTool,
  CompletionTimings,
  CompletionUsage,
  FinishReason
} from "./openai.js"
import type { FinishReason as WorkerFinishReason, RunTimings, ToolChoice, WorkerRequest } from "./protocol.js"
import { createThinkingSplitter } from "./reply.js"
import type { Sampling } from "./sampler.js"
import type { ChatMessage, ChatTool } from "./tokenizer.js"
import { createToolCallSplitter, formatLfm2ToolCalls, parseToolCalls } from "./tool-calls.js"
import type { ParsedToolCall, ToolFormat } from "./tool-calls.js"

/** The generate request's fields a completion decides (the page adds id, model and context). */
export type GenerateFields = Omit<
  Extract<WorkerRequest, { type: "generate" }>,
  "type" | "id" | "model" | "contextLength" | "reportPhases"
>

/** A checked request: what to send the worker and how to read its reply. */
export type CompletionPlan = {
  model: WebGlTextModel
  stream: boolean
  includeUsage: boolean
  generate: GenerateFields
  /** Read tool-call markup out of the answer. */
  parseTools: boolean
  /** False keeps only the first tool call. */
  parallelToolCalls: boolean
  /** Check parsed tool calls against their tools, and regenerate bad ones (see planRepairs). */
  repairToolCalls: boolean
}

export const isCatalogModel = (model: unknown): model is WebGlTextModel =>
  typeof model === "string" && Object.hasOwn(WEBGL_MODEL_META, model)

const unsupported = (message: string) => new WebGLLMError("unsupported-parameter", message)

const REASONING_EFFORTS = new Set(["none", "minimal", "low", "medium", "high", "xhigh"])

/** A number field: unset (undefined or null), or a finite number in range. */
function numberField(
  body: Record<string, unknown>,
  key: string,
  range: { min?: number; max?: number; integer?: boolean } = {}
): number | undefined {
  const value = body[key]
  if (value === undefined || value === null) return undefined
  if (typeof value !== "number" || !Number.isFinite(value)) {
    throw new TypeError(`${key} must be a number.`)
  }
  if (range.integer && !Number.isSafeInteger(value)) {
    throw new RangeError(`${key} must be an integer.`)
  }
  if ((range.min !== undefined && value < range.min) || (range.max !== undefined && value > range.max)) {
    throw new RangeError(`${key} must be between ${range.min ?? "-∞"} and ${range.max ?? "∞"}.`)
  }
  return value
}

/** A message's text: a string, or its text parts joined. Images and audio are refused. */
export function contentText(content: ChatCompletionContent | null | undefined): string {
  if (content === undefined || content === null) return ""
  if (typeof content === "string") return content
  if (!Array.isArray(content)) throw new TypeError("A message's content must be a string or an array of parts.")
  let text = ""
  for (const part of content as Array<{ type?: unknown; text?: unknown; refusal?: unknown }>) {
    if (part?.type === "text" && typeof part.text === "string") text += part.text
    else if (part?.type === "refusal" && typeof part.refusal === "string") text += part.refusal
    else throw unsupported(`Content parts of type "${String(part?.type)}" aren't supported: the models read text only.`)
  }
  return text
}

function parseArguments(json: string): Record<string, unknown> {
  try {
    const args = JSON.parse(json) as unknown
    return args && typeof args === "object" && !Array.isArray(args) ? (args as Record<string, unknown>) : {}
  } catch {
    return {}
  }
}

/**
 * An assistant turn with tool calls, written the way the model's template
 * takes them: LFM2's template doesn't read `tool_calls`, so its calls go in
 * the content in its own format; LFM3's and Qwen3.5's write `tool_calls`.
 */
function assistantToolTurn(content: string, calls: Array<ParsedToolCall & { id: string }>, format: ToolFormat): ChatMessage {
  if (format === "lfm2") return { role: "assistant", content: content + formatLfm2ToolCalls(calls) }
  return {
    role: "assistant",
    content,
    tool_calls: calls.map((c) => ({ id: c.id, type: "function", function: { name: c.name, arguments: c.args } }))
  }
}

/** The current turn's thinking, under the key the format's template reads. */
function withThinking(turn: ChatMessage, thinking: string, format: ToolFormat): ChatMessage {
  if (format === "lfm3") return { ...turn, thinking }
  if (format === "qwen3.5") return { ...turn, reasoning_content: thinking }
  return turn
}

/**
 * OpenAI messages as chat-template messages. System and developer messages
 * join into one leading system message (templates take one, first). Tool
 * calls and results of earlier turns are left out and their answers kept:
 * small models copy the shape of past calls badly and get lost in old
 * results. The current turn's calls and results stay, with its thinking
 * (Liquid's guidance for tool use), since the model answers from them.
 */
export function toChatMessages(messages: Array<ChatCompletionMessageParam>, format: ToolFormat): Array<ChatMessage> {
  if (!Array.isArray(messages) || messages.length === 0) throw new TypeError("messages must be a non-empty array.")
  const system: Array<string> = []
  const chat: Array<ChatMessage> = []
  let currentTurn = -1
  for (let i = messages.length - 1; i >= 0; i--) {
    if (messages[i]?.role === "user") {
      currentTurn = i
      break
    }
  }
  const toolNames = new Map<string, string>()
  for (const [i, m] of messages.entries()) {
    const earlier = i < currentTurn
    switch (m?.role) {
      case "system":
      case "developer": {
        const text = contentText(m.content)
        if (text) system.push(text)
        break
      }
      case "user": {
        const text = contentText(m.content)
        if (text) chat.push({ role: "user", content: text })
        break
      }
      case "tool": {
        if (earlier) break
        const text = contentText(m.content)
        const name = toolNames.get(m.tool_call_id)
        chat.push({
          role: "tool",
          content: text && text !== "null" ? text : "The tool returned nothing.",
          ...(m.tool_call_id && { tool_call_id: m.tool_call_id }),
          ...(name && { name })
        })
        break
      }
      case "assistant": {
        const text = contentText(m.content)
        const calls = (m.tool_calls ?? []).map((call) => {
          if (call?.type !== "function" || typeof call.function?.name !== "string") {
            throw unsupported("Assistant tool calls must be function calls.")
          }
          toolNames.set(call.id, call.function.name)
          return { id: call.id, name: call.function.name, args: parseArguments(call.function.arguments) }
        })
        if (calls.length === 0 || earlier) {
          if (text) chat.push({ role: "assistant", content: text })
          break
        }
        const turn = assistantToolTurn(text, calls, format)
        chat.push(m.reasoning_content ? withThinking(turn, m.reasoning_content, format) : turn)
        break
      }
      default:
        throw unsupported(`Messages with role "${String((m as { role?: unknown })?.role)}" aren't supported.`)
    }
  }
  return system.length ? [{ role: "system", content: system.join("\n\n") }, ...chat] : chat
}

function toTools(tools: Array<ChatCompletionTool>): Array<ChatTool> {
  return tools.map((tool) => {
    if (tool?.type !== "function" || typeof tool.function?.name !== "string") {
      throw unsupported(`Tools of type "${String(tool?.type)}" aren't supported: only "function".`)
    }
    return {
      type: "function",
      function: {
        name: tool.function.name,
        ...(tool.function.description !== undefined && { description: tool.function.description }),
        parameters: tool.function.parameters ?? { type: "object", properties: {}, required: [] }
      }
    }
  })
}

function toToolChoice(choice: unknown, tools: Array<ChatTool>): ToolChoice {
  if (choice === undefined || choice === null || choice === "auto") return "auto"
  if (choice === "none") return "none"
  if (choice === "required") {
    if (!tools.length) throw new TypeError('tool_choice "required" needs tools.')
    return "required"
  }
  const named = choice as { type?: unknown; function?: { name?: unknown } }
  if (named?.type === "function" && typeof named.function?.name === "string") {
    const name = named.function.name
    if (!tools.some((t) => t.function.name === name)) throw new TypeError(`tool_choice names "${name}", which isn't in tools.`)
    return { name }
  }
  throw unsupported(`tool_choice ${JSON.stringify(choice)} isn't supported.`)
}

/**
 * The template's `enable_thinking` for `reasoning_effort`: "none" turns
 * thinking off where the model can switch it off (a model that always
 * thinks ignores it: OpenAI treats effort as a hint), any other level turns
 * it on where the model thinks at all; unset leaves the template's default.
 */
function thinkingFor(effort: unknown, meta: WebGlModelMeta): boolean | undefined {
  if (effort === undefined || effort === null) return undefined
  if (typeof effort !== "string" || !REASONING_EFFORTS.has(effort)) {
    throw new TypeError(`reasoning_effort must be one of ${[...REASONING_EFFORTS].join(", ")}.`)
  }
  if (!meta.reasoning) return undefined
  if (effort === "none") return meta.reasoning.levels.off ? false : undefined
  return true
}

/** Unset fields from the model card (its thinking sampling when thinking is on). */
function samplingFor(body: Record<string, unknown>, meta: WebGlModelMeta, thinking: boolean | undefined): Sampling {
  const base = thinking ? (meta.thinkingSampling ?? meta.sampling) : meta.sampling
  const top_p = numberField(body, "top_p", { min: 0, max: 1 }) ?? base.top_p
  const min_p = numberField(body, "min_p", { min: 0, max: 1 }) ?? base.min_p
  const presence_penalty = numberField(body, "presence_penalty", { min: -2, max: 2 }) ?? base.presence_penalty
  return {
    temperature: numberField(body, "temperature", { min: 0, max: 2 }) ?? base.temperature,
    top_k: numberField(body, "top_k", { min: 0, integer: true }) ?? base.top_k,
    repetition_penalty: numberField(body, "repetition_penalty", { min: 0 }) ?? base.repetition_penalty,
    ...(top_p !== undefined && { top_p }),
    ...(min_p !== undefined && { min_p }),
    ...(presence_penalty !== undefined && { presence_penalty })
  }
}

function outputSchemaFor(format: unknown): object | undefined {
  if (format === undefined || format === null) return undefined
  const f = format as { type?: unknown; json_schema?: { schema?: unknown } }
  if (f.type === "text") return undefined
  // Any JSON object: the grammar of OpenAI's JSON mode.
  if (f.type === "json_object") return { type: "object" }
  if (f.type === "json_schema") {
    const schema = f.json_schema?.schema ?? {}
    if (typeof schema !== "object" || schema === null || Array.isArray(schema)) {
      throw new WebGLLMError("invalid-schema", "response_format.json_schema.schema must be a JSON Schema object.")
    }
    return schema
  }
  throw unsupported(`response_format type "${String(f.type)}" isn't supported.`)
}

/** Checks a request and plans its generation. Throws on unsupported or malformed fields. */
export function planCompletion(params: ChatCompletionCreateParams): CompletionPlan {
  if (!params || typeof params !== "object") throw new TypeError("The request must be an object.")
  const body = params as unknown as Record<string, unknown>
  if (typeof body.model !== "string" || !body.model) throw new TypeError("model is required.")
  if (!isCatalogModel(body.model)) {
    throw new WebGLLMError("model-not-supported", `${body.model} isn't a model webgllm runs (see models.list()).`)
  }
  const model = body.model
  const meta = modelMeta(model)

  const n = numberField(body, "n", { min: 1, integer: true })
  if (n !== undefined && n > 1) throw unsupported("n > 1 isn't supported: one choice per request.")
  if (body.logprobs === true) throw unsupported("logprobs aren't supported.")
  if (body.top_logprobs !== undefined && body.top_logprobs !== null) throw unsupported("top_logprobs aren't supported.")
  if (body.logit_bias && Object.keys(body.logit_bias as object).length > 0) throw unsupported("logit_bias isn't supported.")
  if ((numberField(body, "frequency_penalty", { min: -2, max: 2 }) ?? 0) !== 0) {
    throw unsupported("frequency_penalty isn't supported; presence_penalty and repetition_penalty are.")
  }
  if (body.audio || (Array.isArray(body.modalities) && body.modalities.some((m) => m !== "text"))) {
    throw unsupported("Only text output is supported.")
  }
  if (body.prediction) throw unsupported("Predicted outputs aren't supported.")

  const format = meta.toolFormat ?? "lfm3"
  const chat = toChatMessages(body.messages as Array<ChatCompletionMessageParam>, format)

  if (body.tools !== undefined && body.tools !== null && !Array.isArray(body.tools)) throw new TypeError("tools must be an array.")
  const offered = toTools((body.tools as Array<ChatCompletionTool> | null | undefined) ?? [])
  if (offered.length && !meta.supports.capabilities.includes("tools")) throw unsupported(`${model} doesn't take tools.`)
  const toolChoice = toToolChoice(body.tool_choice, offered)
  // "none" offers no tools at all: the prompt doesn't list them.
  const tools = toolChoice === "none" ? [] : offered

  const thinking = thinkingFor(body.reasoning_effort, meta)
  const budget = numberField(body, "thinking_budget_tokens", { min: 0, integer: true })
  if (budget !== undefined && thinking !== false && !meta.reasoning?.budget) {
    throw unsupported(`${model} doesn't take a thinking budget.`)
  }

  const maxTokens =
    numberField(body, "max_completion_tokens", { min: 0, integer: true }) ??
    numberField(body, "max_tokens", { min: 0, integer: true })
  const seed = numberField(body, "seed", { integer: true })
  const stop = typeof body.stop === "string" ? [body.stop] : body.stop
  if (stop !== undefined && stop !== null && (!Array.isArray(stop) || stop.some((s) => typeof s !== "string"))) {
    throw new TypeError("stop must be a string or an array of strings.")
  }
  const stops = (stop ?? []).filter(Boolean) as Array<string>
  const outputSchema = outputSchemaFor(body.response_format)
  if (body.cache_prompt !== undefined && body.cache_prompt !== null && typeof body.cache_prompt !== "boolean") {
    throw new TypeError("cache_prompt must be a boolean.")
  }
  if (body.repair_tool_calls !== undefined && body.repair_tool_calls !== null && typeof body.repair_tool_calls !== "boolean") {
    throw new TypeError("repair_tool_calls must be a boolean.")
  }

  const generate: GenerateFields = {
    chat,
    tools,
    sampling: samplingFor(body, meta, thinking),
    ...(thinking !== undefined && { thinking }),
    // Thinking switched off ignores a budget.
    ...(budget !== undefined && thinking !== false && { thinkingBudgetTokens: budget }),
    ...(outputSchema && { outputSchema }),
    ...(maxTokens !== undefined && { maxTokens }),
    ...(stops.length && { stop: stops }),
    ...(seed !== undefined && { seed }),
    ...(toolChoice !== "auto" && { toolChoice }),
    ...(body.cache_prompt === false && { cachePrompt: false })
  }
  return {
    model,
    stream: body.stream === true,
    includeUsage: (body.stream_options as { include_usage?: boolean } | undefined)?.include_usage === true,
    generate,
    // A JSON answer can't hold markup; a forced call comes back in "done".
    parseTools: tools.length > 0 && !outputSchema && (toolChoice === "auto"),
    parallelToolCalls: body.parallel_tool_calls !== false,
    repairToolCalls: body.repair_tool_calls !== false
  }
}

const ID_CHARS = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"

/** `prefix` and 24 random characters, like OpenAI's ids. */
export function randomId(prefix: string): string {
  const bytes = crypto.getRandomValues(new Uint8Array(24))
  let id = prefix
  for (const b of bytes) id += ID_CHARS[b % ID_CHARS.length]
  return id
}

/** A piece of the reply: thinking or answer text. */
export type ReplyDelta = { reasoning_content: string } | { content: string }

/** How the worker ended a generation. */
export type GenerationEnd = {
  finishReason: WorkerFinishReason
  toolCall?: { name: string; arguments: Record<string, unknown> }
}

/**
 * Reads the worker's raw text as it streams: thinking into
 * `reasoning_content`, the answer into `content` (shown as it comes, up to
 * the first tool call's markup), and the tool calls parsed once the reply
 * ends. Linear in the text: each piece is looked at once (the splitters
 * hold back only a possible partial tag).
 */
export function createReplyReader(options: {
  format?: ToolFormat
  parseTools: boolean
  parallelToolCalls?: boolean
  newId?: () => string
}) {
  const thinking = createThinkingSplitter()
  const tools = options.parseTools ? createToolCallSplitter() : null
  const newId = options.newId ?? (() => randomId("call_"))

  const read = (pieces: ReturnType<typeof thinking.push>, out: Array<ReplyDelta>) => {
    for (const piece of pieces) {
      if (piece.kind === "thinking") out.push({ reasoning_content: piece.text })
      else {
        const text = tools ? tools.push(piece.text) : piece.text
        if (text) out.push({ content: text })
      }
    }
    return out
  }

  return {
    push(text: string): Array<ReplyDelta> {
      return read(thinking.push(text), [])
    },
    /**
     * The rest of the reply and its tool calls as parsed, not yet checked
     * (see settleToolCalls). `malformed` is tool-call markup no call parsed from.
     */
    finish(end: GenerationEnd): {
      deltas: Array<ReplyDelta>
      calls: Array<ParsedToolCall>
      malformed?: string
      finishReason: WorkerFinishReason
    } {
      const deltas = read(thinking.flush(), [])
      let calls: Array<ParsedToolCall> = []
      let malformed: string | undefined
      if (tools) {
        const { text, markup } = tools.flush()
        if (text) deltas.push({ content: text })
        calls = markup ? parseToolCalls(markup) : []
        if (markup && calls.length === 0) malformed = markup
      }
      if (end.toolCall) calls = [{ name: end.toolCall.name, args: end.toolCall.arguments }, ...calls]
      return { deltas, calls, ...(malformed !== undefined && { malformed }), finishReason: end.finishReason }
    },
    /** Checked calls as OpenAI's tool calls (one only when parallel calls are off). */
    toolCalls(calls: Array<ParsedToolCall>): Array<ChatCompletionMessageToolCall> {
      return (options.parallelToolCalls === false ? calls.slice(0, 1) : calls).map((call) => ({
        id: newId(),
        type: "function" as const,
        function: { name: call.name, arguments: JSON.stringify(call.args) }
      }))
    }
  }
}

/** Token counts, summed over the reply and any tool-call repairs it took. */
export function usageOf(stats: RunTimings, repairs: ReadonlyArray<RunTimings> = []): CompletionUsage {
  const all = [stats, ...repairs]
  const sum = (pick: (s: RunTimings) => number) => all.reduce((total, s) => total + pick(s), 0)
  const prompt = sum((s) => s.inputTokens)
  const completion = sum((s) => s.outputTokens)
  return {
    prompt_tokens: prompt,
    completion_tokens: completion,
    total_tokens: prompt + completion,
    prompt_tokens_details: { cached_tokens: sum((s) => s.cachedTokens) },
    completion_tokens_details: { reasoning_tokens: sum((s) => s.thinkingTokens ?? 0) }
  }
}

/**
 * A tool call to generate again, constrained to its tool: `index` replaces
 * that call. `draft` is the call as the model wrote it, or `markup` the
 * call text that didn't parse.
 */
export type ToolRepair =
  | { index: number; problem: "invalid-arguments" | "unknown-tool"; choice: ToolChoice; draft: ParsedToolCall }
  | { index: number; problem: "malformed"; choice: ToolChoice; markup: string }

const EMPTY_ARGUMENTS = { type: "object", properties: {}, additionalProperties: false }

/**
 * Which of an automatic reply's tool calls need generating again: a call
 * to a tool that wasn't offered (any offered tool will do), arguments that
 * fail their tool's schema (that tool), or call markup that didn't parse
 * as any call (any tool). A schema the validator can't check passes.
 */
export async function planRepairs(
  calls: ReadonlyArray<ParsedToolCall>,
  malformed: string | undefined,
  tools: ReadonlyArray<ChatTool>
): Promise<Array<ToolRepair>> {
  const repairs: Array<ToolRepair> = []
  for (const [index, call] of calls.entries()) {
    const tool = tools.find((t) => t.function.name === call.name)
    if (!tool) {
      repairs.push({ index, problem: "unknown-tool", choice: "required", draft: call })
      continue
    }
    let valid = true
    try {
      valid = await validateStructuredValue(tool.function.parameters ?? EMPTY_ARGUMENTS, call.args)
    } catch {
      // Unsupported by the validator: nothing to check against.
    }
    if (!valid) repairs.push({ index, problem: "invalid-arguments", choice: { name: call.name }, draft: call })
  }
  if (malformed !== undefined && calls.length === 0 && tools.length > 0) {
    repairs.push({ index: 0, problem: "malformed", choice: "required", markup: malformed })
  }
  return repairs
}

/**
 * The request again with the model's own failed call appended as its turn,
 * a note on what was wrong, and the call forced to the tool's schema. The
 * draft keeps what the model got right (forcing from the bare prompt
 * loses it), and appending keeps the prompt's cache. Stop strings don't
 * apply to a call, and thinking closes at once where the model takes a
 * budget: it already thought in the reply being repaired.
 */
export function repairRequest(plan: CompletionPlan, repair: ToolRepair): GenerateFields {
  const { stop: _stop, thinkingBudgetTokens: _budget, ...rest } = plan.generate
  const meta = modelMeta(plan.model)
  const format = meta.toolFormat ?? "lfm3"
  const names = plan.generate.tools.map((t) => t.function.name).join(", ")
  const attempt: ChatMessage =
    repair.problem === "malformed"
      ? { role: "assistant", content: repair.markup }
      : assistantToolTurn("", [{ ...repair.draft, id: "call_0" }], format)
  const note =
    repair.problem === "invalid-arguments"
      ? `The arguments of that ${repair.draft.name} call don't match its parameters. Call ${repair.draft.name} again with arguments that do.`
      : repair.problem === "unknown-tool"
        ? `There's no tool named ${repair.draft.name}. Call one of these instead: ${names}.`
        : `That tool call couldn't be read. Call one of these tools again: ${names}.`
  const budget = plan.generate.thinking !== false && meta.reasoning?.budget
  return {
    ...rest,
    chat: [...plan.generate.chat, attempt, { role: "user", content: note }],
    toolChoice: repair.choice,
    ...(budget && { thinkingBudgetTokens: 0 })
  }
}

const perSecond = (tokens: number, ms: number) => (ms > 0 && tokens > 0 ? (tokens * 1000) / ms : 0)

export function timingsOf(stats: RunTimings): CompletionTimings {
  return {
    first_token_ms: stats.firstTokenMs,
    prompt_ms: stats.promptMs,
    decode_ms: stats.decodeMs,
    prompt_tokens_per_second: perSecond(stats.inputTokens - stats.cachedTokens, stats.promptMs),
    // The first token comes from reading the prompt; decoding makes the rest.
    decode_tokens_per_second: perSecond(stats.outputTokens - 1, stats.decodeMs)
  }
}

/** Builds one stream's chunks: the role on the first, finish_reason and usage at the end. */
export function createChunkWriter(meta: { id: string; created: number; model: string }) {
  let roleSent = false
  const chunk = (
    delta: ChatCompletionChunkDelta,
    finish_reason: FinishReason | null = null
  ): ChatCompletionChunk => {
    if (!roleSent) {
      roleSent = true
      delta = { role: "assistant", ...delta }
    }
    return {
      id: meta.id,
      object: "chat.completion.chunk",
      created: meta.created,
      model: meta.model,
      choices: [{ index: 0, delta, finish_reason, logprobs: null }]
    }
  }
  return {
    delta: (delta: ReplyDelta) => chunk(delta),
    toolCalls: (calls: Array<ChatCompletionMessageToolCall>) =>
      calls.map((call, index) => chunk({ tool_calls: [{ index, id: call.id, type: "function", function: { ...call.function } }] })),
    finish(reason: FinishReason, timings?: CompletionTimings): ChatCompletionChunk {
      const last = chunk({}, reason)
      return timings ? { ...last, timings } : last
    },
    usage(usage: CompletionUsage, timings: CompletionTimings): ChatCompletionChunk {
      return { id: meta.id, object: "chat.completion.chunk", created: meta.created, model: meta.model, choices: [], usage, timings }
    }
  }
}

/** A whole completion from a reply's deltas and its end. */
export function buildCompletion(
  meta: { id: string; created: number; model: string },
  deltas: Array<ReplyDelta>,
  toolCalls: Array<ChatCompletionMessageToolCall>,
  finishReason: FinishReason,
  stats: RunTimings,
  repairs: ReadonlyArray<RunTimings> = []
): ChatCompletion {
  let content = ""
  let reasoning = ""
  for (const d of deltas) {
    if ("content" in d) content += d.content
    else reasoning += d.reasoning_content
  }
  return {
    id: meta.id,
    object: "chat.completion",
    created: meta.created,
    model: meta.model,
    choices: [
      {
        index: 0,
        message: {
          role: "assistant",
          content: content || !toolCalls.length ? content : null,
          ...(reasoning && { reasoning_content: reasoning }),
          ...(toolCalls.length && { tool_calls: toolCalls }),
          refusal: null
        },
        finish_reason: finishReason,
        logprobs: null
      }
    ],
    usage: usageOf(stats, repairs),
    timings: timingsOf(stats)
  }
}
