/**
 * TanStack AI adapter for LFM3 on an LLM inference engine for WebGL 2, for
 * browsers without WebGPU (Firefox on Android, for one). Loads Liquid AI's
 * GGUF and the ONNX build for the web.
 */
export { webglText, WebGlTextAdapter } from "./adapter.js"
export type {
  WebGlProviderUsageDetails,
  WebGlTextProviderOptions
} from "./adapter.js"
export { hasWebGl2, loadModel, modelStatus } from "./client.js"
export type { ModelStatus } from "./client.js"
export {
  WEBGL_MODEL_META,
  WEBGL_MODEL_REASONING,
  WEBGL_TEXT_MODELS
} from "./model-meta.js"
export type {
  ModelFile,
  WebGlModelMeta,
  WebGlModelReasoningByName,
  WebGlTextModel
} from "./model-meta.js"
export type { ToolFormat } from "./tool-calls.js"
export { Engine } from "./engine.js"
export type { EngineOptions, LayerProbe } from "./engine.js"
export type { Sampling } from "./sampler.js"
export type { Store } from "./store.js"
export type {
  ByteSource,
  ModelConfig,
  ModelWeights,
  QuantMatrix,
  WeightPiece
} from "./weights.js"
export { loadTokenizer } from "./tokenizer.js"
export type {
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
export { readGgufHeader } from "./gguf.js"
export type { GgufHeader, GgufTensorInfo, GgufValue } from "./gguf.js"
