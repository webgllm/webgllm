/**
 * webgllm: LLM inference in the browser on WebGL 2, for devices without
 * WebGPU, behind OpenAI's Chat Completions API. Each model runs in a Web
 * Worker on an OffscreenCanvas: one webgllm starts, or one of your own
 * running WebGLLMWorkerHandler.
 *
 * @example
 * ```ts
 * import { WebGLLM } from "webgllm"
 *
 * const llm = new WebGLLM()
 * const completion = await llm.chat.completions.create({
 *   model: "LiquidAI/LFM2.5-1.2B-Instruct-GGUF",
 *   messages: [{ role: "user", content: "Hello!" }]
 * })
 * console.log(completion.choices[0].message.content)
 * ```
 */
export { WebGLLM } from "./webgllm.js"
export type { DownloadModelOptions, LoadModelOptions, LogLevel, RequestOptions, WebGLLMOptions } from "./webgllm.js"
export { WebGLLMWorkerHandler } from "./handler.js"
export { WebGLLMError } from "./errors.js"
export type { WebGLLMErrorCode } from "./errors.js"
export type {
  ChatCompletion,
  ChatCompletionAssistantMessageParam,
  ChatCompletionChunk,
  ChatCompletionChunkDelta,
  ChatCompletionChunkToolCall,
  ChatCompletionContent,
  ChatCompletionContentPartImage,
  ChatCompletionContentPartText,
  ChatCompletionUserContent,
  ChatCompletionCreateParams,
  ChatCompletionCreateParamsBase,
  ChatCompletionCreateParamsNonStreaming,
  ChatCompletionCreateParamsStreaming,
  ChatCompletionMessage,
  ChatCompletionMessageParam,
  ChatCompletionMessageToolCall,
  ChatCompletionSystemMessageParam,
  ChatCompletionTool,
  ChatCompletionToolChoiceOption,
  ChatCompletionToolMessageParam,
  ChatCompletionUserMessageParam,
  CompletionTimings,
  CompletionUsage,
  FinishReason,
  LoadProgress,
  ModelUnloadReason,
  WebGLLMEventMap,
  ModelInfo,
  ReasoningEffort,
  ResponseFormat
} from "./openai.js"

// The model catalog.
export { WEBGL_MODEL_META, WEBGL_TEXT_MODELS } from "./model-meta.js"
export type { WebGlModelId, WebGlModelMeta, WebGlModelReasoning, WebGlTextModel } from "./model-meta.js"
