/**
 * The models this package runs: LFM3 1.7B (Liquid AI's QAD Q4_0 GGUF and two
 * 4-bit ONNX builds) and the LFM2.5 / LFM2 ONNX builds the Transformers.js
 * package runs, each with its files pinned and the sampling from Liquid
 * AI's cards.
 */
import type { ModelReasoning } from "@tanstack/ai"
import type { ModelReasoningCapability } from "@tanstack/ai/adapter-internals"

import type { Sampling } from "./sampler.js"
import type { ToolFormat } from "./tool-calls.js"

/** A file of a model: its repository and path, at its latest version (main). */
export type ModelFile = {
  repo: string
  path: string
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
  /** Reasoning levels for `chat({ reasoning })`: the template's enable_thinking. */
  reasoning?: ModelReasoning
  sampling: Sampling
  /** How the model writes tool calls: LFM3's when left out. */
  toolFormat?: ToolFormat
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
const TOOLS = {
  input: ["text"],
  output: ["text"],
  capabilities: ["tools"]
} as const

/** LFM2.5 Thinking always thinks: its template takes no switch. */
const ALWAYS_THINKS = { levels: { high: "true" }, budget: false } as const

/** LFM2.5's card: temperature 0.1, top-k 50, repetition penalty 1.05. */
const LFM2_5_SAMPLING = {
  temperature: 0.1,
  top_k: 50,
  repetition_penalty: 1.05
} as const satisfies Sampling
const LFM2_5_THINKING_SAMPLING = {
  temperature: 0.05,
  top_k: 50,
  repetition_penalty: 1.05
} as const satisfies Sampling

/** LFM3's card, "General": temperature 0.1, top-k 50, repetition penalty 1.1. */
const LFM3_SAMPLING = {
  temperature: 0.1,
  top_k: 50,
  repetition_penalty: 1.1
} as const satisfies Sampling

const LFM3_REPO = {
  repo: "LiquidAI/LFM3-1.7B"
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
      path: "LFM3-1.7B-QAD-Q4_0.gguf"
    }
  },
  tokenizer: {
    tokenizer: { ...LFM3_REPO, path: "tokenizer.json" },
    config: { ...LFM3_REPO, path: "tokenizer_config.json" },
    chatTemplate: { ...LFM3_REPO, path: "chat_template.jinja" }
  },
  supports: TOOLS_AND_THINKING,
  size: "1.7b",
  reasoning: THINKING_SWITCH,
  sampling: LFM3_SAMPLING,
  toolFormat: "lfm3"
} as const satisfies WebGlModelMeta

const LFM3_WEB_REPO = {
  repo: "daveycodez/LFM3-1.7B-Web"
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
      path: "onnx/model_q4f16.onnx_data"
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
  reasoning: THINKING_SWITCH,
  sampling: LFM3_SAMPLING,
  toolFormat: "lfm3"
} as const satisfies WebGlModelMeta

const LFM3_ONNX_REPO = {
  repo: "onnx-community/LFM3-1.7B-ONNX"
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
      path: "onnx/model_q4f16.onnx_data"
    },
    config: { ...LFM3_ONNX_REPO, path: "config.json" }
  },
  tokenizer: {
    tokenizer: { ...LFM3_ONNX_REPO, path: "tokenizer.json" },
    config: { ...LFM3_ONNX_REPO, path: "tokenizer_config.json" }
  },
  supports: TOOLS_AND_THINKING,
  size: "1.7b",
  reasoning: THINKING_SWITCH,
  sampling: LFM3_SAMPLING,
  toolFormat: "lfm3"
} as const satisfies WebGlModelMeta

/**
 * An LFM2.5 ONNX build in Liquid AI's export (4-bit MatMulNBits projections,
 * a 4-bit GatherBlockQuantized embedding that is also the output head),
 * at its latest version (main).
 */
function lfm2Onnx(o: {
  name: string
  size: string
  thinking: boolean
}) {
  const repo = { repo: o.name }
  return {
    name: o.name,
    format: "onnx",
    files: {
      kind: "onnx",
      graph: { ...repo, path: "onnx/model_q4f16.onnx" },
      data: { ...repo, path: "onnx/model_q4f16.onnx_data" },
      config: { ...repo, path: "config.json" }
    },
    tokenizer: {
      tokenizer: { ...repo, path: "tokenizer.json" },
      config: { ...repo, path: "tokenizer_config.json" },
      chatTemplate: { ...repo, path: "chat_template.jinja" }
    },
    supports: o.thinking ? TOOLS_AND_THINKING : TOOLS,
    size: o.size,
    ...(o.thinking && { reasoning: ALWAYS_THINKS }),
    sampling: o.thinking ? LFM2_5_THINKING_SAMPLING : LFM2_5_SAMPLING,
    toolFormat: "lfm2"
  } as const satisfies WebGlModelMeta
}

const LFM2_5_1_2B = lfm2Onnx({
  name: "LiquidAI/LFM2.5-1.2B-Instruct-ONNX",
  size: "1.2b",
  thinking: false
})

const LFM2_5_1_2B_THINKING = lfm2Onnx({
  name: "LiquidAI/LFM2.5-1.2B-Thinking-ONNX",
  size: "1.2b",
  thinking: true
})

/** LFM2.5 1.2B re-exported to compute the prompt's last-token logits only (the WebGL engine does that anyway). */
const LFM2_5_1_2B_WEB = lfm2Onnx({
  name: "daveycodez/LFM2.5-1.2B-Web",
  size: "1.2b",
  thinking: false
})

const LFM2_5_1_2B_THINKING_WEB = lfm2Onnx({
  name: "daveycodez/LFM2.5-1.2B-Thinking-Web",
  size: "1.2b",
  thinking: true
})

/** LFM2.5 350M: the smallest and quickest, for short answers and tool calls. */
const LFM2_5_350M = lfm2Onnx({
  name: "LiquidAI/LFM2.5-350M-ONNX",
  size: "350m",
  thinking: false
})

export const WEBGL_TEXT_MODELS = [
  LFM3_1_7B_GGUF.name,
  LFM3_1_7B_WEB.name,
  LFM3_1_7B_ONNX.name,
  LFM2_5_1_2B.name,
  LFM2_5_1_2B_THINKING.name,
  LFM2_5_1_2B_WEB.name,
  LFM2_5_1_2B_THINKING_WEB.name,
  LFM2_5_350M.name
] as const

/** A model this package runs. */
export type WebGlTextModel = (typeof WEBGL_TEXT_MODELS)[number]

export const WEBGL_MODEL_META: Readonly<
  Record<WebGlTextModel, WebGlModelMeta>
> = {
  [LFM3_1_7B_GGUF.name]: LFM3_1_7B_GGUF,
  [LFM3_1_7B_WEB.name]: LFM3_1_7B_WEB,
  [LFM3_1_7B_ONNX.name]: LFM3_1_7B_ONNX,
  [LFM2_5_1_2B.name]: LFM2_5_1_2B,
  [LFM2_5_1_2B_THINKING.name]: LFM2_5_1_2B_THINKING,
  [LFM2_5_1_2B_WEB.name]: LFM2_5_1_2B_WEB,
  [LFM2_5_1_2B_THINKING_WEB.name]: LFM2_5_1_2B_THINKING_WEB,
  [LFM2_5_350M.name]: LFM2_5_350M
}

const WEBGL_REASONING = {
  [LFM3_1_7B_GGUF.name]: LFM3_1_7B_GGUF.reasoning,
  [LFM3_1_7B_WEB.name]: LFM3_1_7B_WEB.reasoning,
  [LFM3_1_7B_ONNX.name]: LFM3_1_7B_ONNX.reasoning,
  [LFM2_5_1_2B_THINKING.name]: ALWAYS_THINKS,
  [LFM2_5_1_2B_THINKING_WEB.name]: ALWAYS_THINKS
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

/** Where a file downloads from: its latest version (main) on Hugging Face. */
export const fileUrl = (file: ModelFile) =>
  `https://huggingface.co/${file.repo}/resolve/main/${file.path}`
