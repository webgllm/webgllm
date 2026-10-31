/**
 * The models this package runs: LFM3 1.7B, as Liquid AI's QAD Q4_0 GGUF or
 * as the 4-bit ONNX build for the web, each with its files pinned and the
 * sampling from Liquid AI's card.
 */
import type { ModelReasoning } from "@tanstack/ai"
import type { ModelReasoningCapability } from "@tanstack/ai/adapter-internals"

import type { Sampling } from "./sampler.js"

/** A file of a model: its repository, pinned commit and path. */
export type ModelFile = {
  repo: string
  revision: string
  path: string
  bytes?: number
}

export type WebGlModelMeta = {
  /** The model's name in this package. */
  name: string
  format: "gguf" | "onnx"
  /**
   * The weights: one GGUF file, or the ONNX graph and its external data
   * (plus config.json for the architecture).
   */
  files:
    | { kind: "gguf"; model: ModelFile }
    | { kind: "onnx"; graph: ModelFile; data: ModelFile; config: ModelFile }
  /** The tokenizer and chat template. */
  /** The tokenizer and chat template (tokenizer_config.json's when there's no file for it). */
  tokenizer: {
    tokenizer: ModelFile
    config: ModelFile
    chatTemplate?: ModelFile
  }
  supports: {
    input: ReadonlyArray<"text">
    output: ReadonlyArray<"text">
    capabilities: ReadonlyArray<"tools" | "thinking">
  }
  size: string
  /** Tokens a chat can hold, prompt and reply together, in this runtime. */
  context: number
  /** Reasoning levels for `chat({ reasoning })`: the template's enable_thinking. */
  reasoning?: ModelReasoning
  sampling: Sampling
}

const THINKING_SWITCH = {
  levels: { off: "false", high: "true" },
  budget: false
} as const

const TOOLS_AND_THINKING = {
  input: ["text"],
  output: ["text"],
  capabilities: ["tools", "thinking"]
} as const

/** LFM3's card, "General": temperature 0.1, top-k 50, repetition penalty 1.1. */
const LFM3_SAMPLING = {
  temperature: 0.1,
  top_k: 50,
  repetition_penalty: 1.1
} as const satisfies Sampling

/**
 * Tokens a chat may hold: the attention cache for 8,192 tokens is 88 MB in
 * halves, and a full context already takes a long time to read on a phone.
 */
const CONTEXT = 8192

const LFM3_REPO = {
  repo: "LiquidAI/LFM3-1.7B",
  revision: "ac2586a34f1a1c03329dbdcdefabd8b6478ddaa7"
}

/**
 * Liquid AI's official LFM3 1.7B GGUF, quantization-aware trained for Q4_0
 * (QAD): every projection is Q4_0, the embedding Q6_K. Its tokenizer and
 * chat template come from the original repository (the GGUF embeds the
 * same ones).
 */
const LFM3_1_7B_GGUF = {
  name: "LiquidAI/LFM3-1.7B-GGUF",
  format: "gguf",
  files: {
    kind: "gguf",
    model: {
      repo: "LiquidAI/LFM3-1.7B-GGUF",
      revision: "bad5729bba61d076fff20d2548e62526665af215",
      path: "LFM3-1.7B-QAD-Q4_0.gguf",
      bytes: 1_049_615_680
    }
  },
  tokenizer: {
    tokenizer: { ...LFM3_REPO, path: "tokenizer.json" },
    config: { ...LFM3_REPO, path: "tokenizer_config.json" },
    chatTemplate: { ...LFM3_REPO, path: "chat_template.jinja" }
  },
  supports: TOOLS_AND_THINKING,
  size: "1.7b",
  context: CONTEXT,
  reasoning: THINKING_SWITCH,
  sampling: LFM3_SAMPLING
} as const satisfies WebGlModelMeta

const LFM3_WEB_REPO = {
  repo: "daveycodez/LFM3-1.7B-Web",
  revision: "9cda8fa27706d239b9c58f53a867d398e313b961"
}

/**
 * LFM3 1.7B for the web: GPTQ 4-bit with block size 32 and a 49k
 * vocabulary (900 MB), so writing each token costs less than with the GGUF's
 * 128k-row output head.
 */
const LFM3_1_7B_WEB = {
  name: "daveycodez/LFM3-1.7B-Web",
  format: "onnx",
  files: {
    kind: "onnx",
    graph: { ...LFM3_WEB_REPO, path: "onnx/model_q4f16.onnx" },
    data: {
      ...LFM3_WEB_REPO,
      path: "onnx/model_q4f16.onnx_data",
      bytes: 900_061_184
    },
    config: { ...LFM3_WEB_REPO, path: "config.json" }
  },
  tokenizer: {
    tokenizer: { ...LFM3_WEB_REPO, path: "tokenizer.json" },
    config: { ...LFM3_WEB_REPO, path: "tokenizer_config.json" },
    chatTemplate: { ...LFM3_WEB_REPO, path: "chat_template.jinja" }
  },
  supports: TOOLS_AND_THINKING,
  size: "1.7b",
  context: CONTEXT,
  reasoning: THINKING_SWITCH,
  sampling: LFM3_SAMPLING
} as const satisfies WebGlModelMeta

const LFM3_ONNX_REPO = {
  repo: "onnx-community/LFM3-1.7B-ONNX",
  revision: "ab9fec4c681a25bec3b48b0b97a62fc2660a2985"
}

/**
 * The onnx-community export of LFM3 1.7B (q4f16), the one the
 * Transformers.js package runs: 4-bit weights in blocks of 32 and the full
 * 128k vocabulary. Its chat template is in tokenizer_config.json.
 */
const LFM3_1_7B_ONNX = {
  name: "onnx-community/LFM3-1.7B-ONNX",
  format: "onnx",
  files: {
    kind: "onnx",
    graph: { ...LFM3_ONNX_REPO, path: "onnx/model_q4f16.onnx" },
    data: {
      ...LFM3_ONNX_REPO,
      path: "onnx/model_q4f16.onnx_data",
      bytes: 1_087_789_056
    },
    config: { ...LFM3_ONNX_REPO, path: "config.json" }
  },
  tokenizer: {
    tokenizer: { ...LFM3_ONNX_REPO, path: "tokenizer.json" },
    config: { ...LFM3_ONNX_REPO, path: "tokenizer_config.json" }
  },
  supports: TOOLS_AND_THINKING,
  size: "1.7b",
  context: CONTEXT,
  reasoning: THINKING_SWITCH,
  sampling: LFM3_SAMPLING
} as const satisfies WebGlModelMeta

export const WEBGL_TEXT_MODELS = [
  LFM3_1_7B_GGUF.name,
  LFM3_1_7B_WEB.name,
  LFM3_1_7B_ONNX.name
] as const

/** A model this package runs. */
export type WebGlTextModel = (typeof WEBGL_TEXT_MODELS)[number]

export const WEBGL_MODEL_META: Readonly<
  Record<WebGlTextModel, WebGlModelMeta>
> = {
  [LFM3_1_7B_GGUF.name]: LFM3_1_7B_GGUF,
  [LFM3_1_7B_WEB.name]: LFM3_1_7B_WEB,
  [LFM3_1_7B_ONNX.name]: LFM3_1_7B_ONNX
}

const WEBGL_REASONING = {
  [LFM3_1_7B_GGUF.name]: LFM3_1_7B_GGUF.reasoning,
  [LFM3_1_7B_WEB.name]: LFM3_1_7B_WEB.reasoning,
  [LFM3_1_7B_ONNX.name]: LFM3_1_7B_ONNX.reasoning
} as const

/** The reasoning levels of each model. */
export type WebGlModelReasoningByName = {
  [K in keyof typeof WEBGL_REASONING]: ModelReasoningCapability<
    (typeof WEBGL_REASONING)[K]
  >
}

/** Reasoning data of each model, for `supportedReasoningLevels()`. */
export const WEBGL_MODEL_REASONING: Readonly<Record<string, ModelReasoning>> =
  WEBGL_REASONING

export const modelMeta = (model: WebGlTextModel): WebGlModelMeta =>
  WEBGL_MODEL_META[model]

/** Where a file downloads from: its pinned commit on Hugging Face. */
export const fileUrl = (file: ModelFile) =>
  `https://huggingface.co/${file.repo}/resolve/${file.revision}/${file.path}`
