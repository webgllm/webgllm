/**
 * The public request and response types: OpenAI's Chat Completions shapes
 * (the fields webgllm takes), plus a few llama.cpp-style extensions.
 */
import type { WebGlModelId } from "./model-meta.js"

/** A text part of a message's content. Other part types (images, audio) are refused. */
export type ChatCompletionContentPartText = { type: "text"; text: string }

/** A message's content: text, or text parts. */
export type ChatCompletionContent = string | Array<ChatCompletionContentPartText>

export type ChatCompletionMessageToolCall = {
  id: string
  type: "function"
  /** `arguments` is a JSON string, as OpenAI sends it. */
  function: { name: string; arguments: string }
}

export type ChatCompletionSystemMessageParam = {
  role: "system" | "developer"
  content: ChatCompletionContent
  name?: string
}

export type ChatCompletionUserMessageParam = {
  role: "user"
  content: ChatCompletionContent
  name?: string
}

export type ChatCompletionAssistantMessageParam = {
  role: "assistant"
  content?: ChatCompletionContent | null
  /** The reply's thinking; given back to templates that keep the current turn's. */
  reasoning_content?: string | null
  tool_calls?: Array<ChatCompletionMessageToolCall>
  name?: string
  refusal?: string | null
}

export type ChatCompletionToolMessageParam = {
  role: "tool"
  content: ChatCompletionContent
  tool_call_id: string
}

export type ChatCompletionMessageParam =
  | ChatCompletionSystemMessageParam
  | ChatCompletionUserMessageParam
  | ChatCompletionAssistantMessageParam
  | ChatCompletionToolMessageParam

export type ChatCompletionTool = {
  type: "function"
  function: {
    name: string
    description?: string
    /** JSON Schema of the arguments object. */
    parameters?: Record<string, unknown>
    strict?: boolean | null
  }
}

export type ChatCompletionToolChoiceOption =
  | "none"
  | "auto"
  | "required"
  | { type: "function"; function: { name: string } }

export type ResponseFormat =
  | { type: "text" }
  | { type: "json_object" }
  | {
      type: "json_schema"
      json_schema: {
        name: string
        description?: string
        schema?: Record<string, unknown>
        strict?: boolean | null
      }
    }

/** OpenAI's effort levels: "none" turns thinking off, any other turns it on. */
export type ReasoningEffort = "none" | "minimal" | "low" | "medium" | "high" | "xhigh"

export type ChatCompletionCreateParamsBase = {
  /**
   * A catalog model (see `models.list()`), or any Hugging Face repository
   * (`owner/name`) with an architecture webgllm runs; loaded first if it isn't.
   */
  model: WebGlModelId
  messages: Array<ChatCompletionMessageParam>
  stream?: boolean | null
  stream_options?: { include_usage?: boolean } | null
  /** Most tokens to generate, thinking included. */
  max_completion_tokens?: number | null
  /** Older name of `max_completion_tokens` (which wins when both are set). */
  max_tokens?: number | null
  temperature?: number | null
  top_p?: number | null
  presence_penalty?: number | null
  /** Only 0 (or unset) is supported. */
  frequency_penalty?: number | null
  stop?: string | Array<string> | null
  seed?: number | null
  response_format?: ResponseFormat
  tools?: Array<ChatCompletionTool>
  tool_choice?: ChatCompletionToolChoiceOption
  /** False keeps only the first tool call of a reply. */
  parallel_tool_calls?: boolean
  /** A hint: models that can't switch thinking off ignore "none". */
  reasoning_effort?: ReasoningEffort | null
  /** Only 1 (or unset) is supported. */
  n?: number | null
  /** Unsupported when true. */
  logprobs?: boolean | null
  /** Unsupported. */
  top_logprobs?: number | null
  /** Unsupported. */
  logit_bias?: Record<string, number> | null
  /** Accepted and ignored, like the rest of OpenAI's bookkeeping fields. */
  user?: string
  metadata?: Record<string, string> | null
  store?: boolean | null

  // webgllm extensions (llama.cpp's names).
  top_k?: number | null
  min_p?: number | null
  repetition_penalty?: number | null
  /** Close thinking after this many tokens, then answer. */
  thinking_budget_tokens?: number | null
  /** False: don't reuse or keep the prompt's cache across calls. */
  cache_prompt?: boolean | null
  /**
   * Automatic tool calls are checked against their tools; a bad one (an
   * unknown tool, arguments that fail the schema, markup that doesn't
   * parse) is generated again constrained to the schema. False returns
   * them as the model wrote them. Default true.
   */
  repair_tool_calls?: boolean | null
}

export type ChatCompletionCreateParamsStreaming = ChatCompletionCreateParamsBase & {
  stream: true
}
export type ChatCompletionCreateParamsNonStreaming = ChatCompletionCreateParamsBase & {
  stream?: false | null
}
export type ChatCompletionCreateParams =
  | ChatCompletionCreateParamsStreaming
  | ChatCompletionCreateParamsNonStreaming

export type FinishReason = "stop" | "length" | "tool_calls"

export type CompletionUsage = {
  prompt_tokens: number
  completion_tokens: number
  total_tokens: number
  prompt_tokens_details: { cached_tokens: number }
  completion_tokens_details: { reasoning_tokens: number }
}

/** How long the call took on this device (llama.cpp's `timings`, in ms and tokens/s). */
export type CompletionTimings = {
  /** From the request reaching the worker to the first generated token; null if none. */
  first_token_ms: number | null
  /** Reading the part of the prompt that wasn't cached. */
  prompt_ms: number
  /** From the first generated token to the last. */
  decode_ms: number
  prompt_tokens_per_second: number
  decode_tokens_per_second: number
}

export type ChatCompletionMessage = {
  role: "assistant"
  /** Null when the reply is only tool calls. */
  content: string | null
  reasoning_content?: string
  tool_calls?: Array<ChatCompletionMessageToolCall>
  refusal: null
}

export type ChatCompletion = {
  id: string
  object: "chat.completion"
  /** Unix seconds. */
  created: number
  model: string
  choices: Array<{
    index: 0
    message: ChatCompletionMessage
    finish_reason: FinishReason
    logprobs: null
  }>
  usage: CompletionUsage
  timings: CompletionTimings
}

export type ChatCompletionChunkToolCall = {
  index: number
  id?: string
  type?: "function"
  function: { name?: string; arguments?: string }
}

export type ChatCompletionChunkDelta = {
  role?: "assistant"
  content?: string
  reasoning_content?: string
  tool_calls?: Array<ChatCompletionChunkToolCall>
}

export type ChatCompletionChunk = {
  id: string
  object: "chat.completion.chunk"
  created: number
  model: string
  /** Empty in the usage chunk (`stream_options.include_usage`). */
  choices: Array<{
    index: 0
    delta: ChatCompletionChunkDelta
    finish_reason: FinishReason | null
    logprobs: null
  }>
  /** On the last chunk, with `stream_options.include_usage`. */
  usage?: CompletionUsage
  /** On the last chunk. */
  timings?: CompletionTimings
}

/** A model load's progress, from any load however it started. */
export type LoadProgress = {
  model: string
  /** Downloading files, or reading them into the GPU. */
  phase: "download" | "load"
  /**
   * 0–1 across both phases: downloading takes 0–0.9 and loading the rest;
   * a model already in storage loads across 0–1. `downloadModel` reports
   * its download across 0–1.
   */
  progress: number
  /** True when the files come from the browser's storage, not the network. */
  cached: boolean
  loadedBytes?: number
  totalBytes?: number
}

export type ModelInfo = {
  /** The id to pass as `model`. */
  id: string
  format: "gguf" | "onnx"
  /** Most tokens the model reads (its own context window); 0 if it couldn't be read. */
  contextLength: number
  /** Bytes to download, every file together; 0 when unknown. */
  size: number
  /** All its files are in the browser's storage. */
  downloaded: boolean
  loaded: boolean
  /** The context the loaded model holds, when loaded. */
  loadedContextLength?: number
  supports: { tools: boolean; thinking: boolean }
}

/** Why a model left memory: unloadModel, room for another model, nobody waiting for its load, a failed load, or its worker stopped. */
export type ModelUnloadReason = "unloaded" | "evicted" | "aborted" | "failed" | "crashed"

/** The events a WebGLLM instance dispatches, by type. */
export type WebGLLMEventMap = {
  /** Every load's and download's progress. */
  progress: LoadProgress
  /** A model began loading (from loadModel, or on demand for a completion). */
  loadstart: { model: string; contextLength?: number }
  /** A model is loaded and ready, holding `contextLength` tokens. */
  load: { model: string; contextLength: number }
  /** A model's load failed, or its worker stopped. */
  error: { model: string; error: Error }
  /** A model left memory. */
  unload: { model: string; reason: ModelUnloadReason }
}
