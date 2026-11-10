/**
 * webgllm: LLM inference in the browser on WebGL 2, for devices without
 * WebGPU, behind OpenAI's Chat Completions API. Each model runs in a Web
 * Worker on an OffscreenCanvas: one webgllm starts, or one of your own
 * running WebGLLMWorkerHandler. The engine underneath (Engine, the weights
 * loaders, the tokenizer) is exported too, for code that runs a model itself.
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
export type { ModelFile, ModelReasoning, VisionFiles, WebGlModelId, WebGlModelMeta, WebGlModelReasoning, WebGlTextModel } from "./model-meta.js"

// Settings beside a WebGLLM's own.
export { setPromptBatch } from "./client.js"
export { defaultMemoryBudget } from "./model-memory.js"

// The engine underneath, for code that runs a model itself (benchmarks,
// experiments). Not covered by semver while the package is 0.x.
export { Engine } from "./engine.js"
export type { EngineImage, EngineOptions } from "./engine.js"
export type { Sampling } from "./sampler.js"
export type { ByteSource, ModelConfig, ModelWeights } from "./weights.js"
export { loadTokenizer } from "./tokenizer.js"
export type {
  ChatContentPart,
  ChatMessage,
  ChatTemplateOptions,
  ChatTool,
  ChatToolCall,
  SpecialTokenIds,
  StreamDecoder,
  Tokenizer,
  TokenizerFiles
} from "./tokenizer.js"
export { blobSource, httpSource } from "./byte-source.js"
export type { HttpSourceOptions } from "./byte-source.js"
export { loadGguf } from "./load-gguf.js"
export type { LoadGgufOptions } from "./load-gguf.js"
export { loadOnnx } from "./load-onnx.js"
export type { LoadOnnxOptions } from "./load-onnx.js"
export { loadVision } from "./load-vision.js"
export type { LoadVisionOptions, VisionConfig } from "./load-vision.js"
export { VisionEncoder } from "./vision.js"
export { imageProcessorConfig, smartResize, toPatches } from "./image.js"
export type { ImageDetail, ImagePatches, ImageProcessorConfig } from "./image.js"
