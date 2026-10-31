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
  sampling: LFM3_SAMPLING,
  toolFormat: "lfm3"
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
  sampling: LFM3_SAMPLING,
  toolFormat: "lfm3"
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
  sampling: LFM3_SAMPLING,
  toolFormat: "lfm3"
} as const satisfies WebGlModelMeta

/**
 * An LFM2.5 ONNX build in Liquid AI's export (4-bit MatMulNBits projections,
 * a 4-bit GatherBlockQuantized embedding that is also the output head),
 * with its files at one pinned commit.
 */
function lfm2Onnx(o: {
  name: string
  revision: string
  dataBytes: number
  size: string
  thinking: boolean
}) {
  const repo = { repo: o.name, revision: o.revision }
  return {
    name: o.name,
    format: "onnx",
    files: {
      kind: "onnx",
      graph: { ...repo, path: "onnx/model_q4f16.onnx" },
      data: { ...repo, path: "onnx/model_q4f16.onnx_data", bytes: o.dataBytes },
      config: { ...repo, path: "config.json" }
    },
    tokenizer: {
      tokenizer: { ...repo, path: "tokenizer.json" },
      config: { ...repo, path: "tokenizer_config.json" },
      chatTemplate: { ...repo, path: "chat_template.jinja" }
    },
    supports: o.thinking ? TOOLS_AND_THINKING : TOOLS,
    size: o.size,
    context: CONTEXT,
    ...(o.thinking && { reasoning: ALWAYS_THINKS }),
    sampling: o.thinking ? LFM2_5_THINKING_SAMPLING : LFM2_5_SAMPLING,
    toolFormat: "lfm2"
  } as const satisfies WebGlModelMeta
}

const LFM2_5_1_2B = lfm2Onnx({
  name: "LiquidAI/LFM2.5-1.2B-Instruct-ONNX",
  revision: "10f72e70abf67ac0fd7ebf15bc5854726891d864",
  dataBytes: 760_279_040,
  size: "1.2b",
  thinking: false
})

const LFM2_5_1_2B_THINKING = lfm2Onnx({
  name: "LiquidAI/LFM2.5-1.2B-Thinking-ONNX",
  revision: "e7fe61974e3a167dff77c5722db9a1cb7b57140f",
  dataBytes: 760_279_040,
  size: "1.2b",
  thinking: true
})

/** LFM2.5 1.2B re-exported to compute the prompt's last-token logits only (the WebGL engine does that anyway). */
const LFM2_5_1_2B_WEB = lfm2Onnx({
  name: "daveycodez/LFM2.5-1.2B-Web",
  revision: "f8df717df1ee01ccb899484d7f25cddfa78dd49f",
  dataBytes: 760_279_040,
  size: "1.2b",
  thinking: false
})

const LFM2_5_1_2B_THINKING_WEB = lfm2Onnx({
  name: "daveycodez/LFM2.5-1.2B-Thinking-Web",
  revision: "db6d0161174b69db3bd4ed789e9a1cb888344f32",
  dataBytes: 760_279_040,
  size: "1.2b",
  thinking: true
})

/** LFM2.5 350M: the smallest and quickest, for short answers and tool calls. */
const LFM2_5_350M = lfm2Onnx({
  name: "LiquidAI/LFM2.5-350M-ONNX",
  revision: "d11593fd9eb408e322667926656598896c2d5ff9",
  dataBytes: 254_965_760,
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

/** Where a file downloads from: its pinned commit on Hugging Face. */
export const fileUrl = (file: ModelFile) =>
  `https://huggingface.co/${file.repo}/resolve/${file.revision}/${file.path}`
