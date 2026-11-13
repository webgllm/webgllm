/**
 * The catalog: known defaults for models this package runs and is tuned
 * for. Any other Hugging Face repository with a supported architecture
 * loads too (resolve-model.ts builds its meta from the Hub); the catalog
 * holds what the Hub can't say, such as the sampling on a model card.
 *
 * The models: LFM3 1.7B (Liquid AI's QAD Q4_0 GGUF and two
 * 4-bit ONNX builds), official LFM2.5 GGUFs, the LFM2.5 / LFM2 ONNX builds
 * the Transformers.js package runs, Qwen3.5 2B (unsloth's Q4_0 GGUF and onnx-community's
 * q4f16 ONNX), and Liquid AI's vision-language models (LFM3-VL 2B and
 * LFM2.5-VL 3B, 1.6B and 450M, each in its GGUF and its ONNX build), each
 * with its files (at their latest version) and the sampling from its card.
 */

import type { Sampling } from "./sampler.js"
import type { ToolFormat } from "./tool-calls.js"

/** A file of a model: its repository and path, at its latest version (main). */
export type ModelFile = {
  repo: string
  path: string
}

/**
 * A model's vision encoder files (see WebGlModelMeta.files): an ONNX graph
 * and its data, or a GGUF projector (`mmproj-*.gguf`), and the image
 * processor's config (preprocessor_config.json, or a processor_config.json
 * that holds it).
 */
export type VisionFiles =
  | { graph: ModelFile; data: ModelFile; preprocessor: ModelFile }
  | { projector: ModelFile; preprocessor: ModelFile }

/**
 * An ONNX graph's external data as loadOnnx takes it, from `sources` (one
 * per data file, `data` then `moreData`): the one source, or, where the
 * graph's weights are in several files, each by the location the graph
 * gives it. A location is the file's path from the graph's folder, its
 * subfolders kept (`weights/layers.bin`): two files of the same name in
 * different folders are different files.
 */
export function onnxDataSources<S>(files: Extract<WebGlModelMeta["files"], { kind: "onnx" }>, sources: ReadonlyArray<S>): S | ((location: string) => S) {
  if (!files.moreData?.length) return sources[0]
  const folder = files.graph.path.slice(0, files.graph.path.lastIndexOf("/") + 1)
  const byLocation = new Map(
    [files.data, ...files.moreData].map((file, i) => [file.path.startsWith(folder) ? file.path.slice(folder.length) : file.path, sources[i]])
  )
  return (location: string) => {
    const source = byLocation.get(location.replace(/^(\.\/)+/, ""))
    if (source === undefined) throw new Error(`The graph reads ${location}, which isn't one of the model's files.`)
    return source
  }
}

/** A vision encoder's files, each one (as storedModelFiles lists them). */
export const visionFileList = (vision: VisionFiles): Array<ModelFile> =>
  "projector" in vision ? [vision.projector, vision.preprocessor] : [vision.graph, vision.data, vision.preprocessor]

/**
 * Whether thinking can be switched off (`off`) and capped by a token budget.
 * Each level's value is what to send as `reasoning_effort` for it.
 */
export type ModelReasoning = {
  readonly levels: { readonly off?: "none"; readonly high: "high" }
  readonly budget: boolean
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
    | {
        kind: "gguf"
        model: ModelFile
        /** The vision encoder (the repository's `mmproj-*.gguf`), for models that read images. */
        vision?: VisionFiles
      }
    | {
        kind: "onnx"
        graph: ModelFile
        data: ModelFile
        /** The graph's other external data files, when its weights are in several (`data` is the first it names). */
        moreData?: ReadonlyArray<ModelFile>
        config: ModelFile
        /**
         * The vision encoder (the graph beside the decoder, such as
         * `onnx/vision_encoder_q4.onnx`, and its data), for models that
         * read images: loaded only for a request with images (see
         * vision.ts).
         */
        vision?: VisionFiles
      }
  /** The tokenizer and chat template. */
  /** The tokenizer and chat template (tokenizer_config.json's when there's no file for it). */
  tokenizer: {
    tokenizer: ModelFile
    config: ModelFile
    chatTemplate?: ModelFile
  }
  supports: {
    /** "image" for models with a vision encoder webgllm runs (see `files.vision`). */
    input: ReadonlyArray<"text" | "image">
    output: ReadonlyArray<"text">
    capabilities: ReadonlyArray<"tools" | "thinking">
  }
  size: string
  /** Thinking levels, each with its `reasoning_effort` value, and whether a budget applies. */
  reasoning?: ModelReasoning
  sampling: Sampling
  /** Recommended text sampling when the chat template enables thinking. */
  thinkingSampling?: Sampling
  /** How the model writes tool calls: LFM3's when left out. */
  toolFormat?: ToolFormat
  /**
   * Whether the resolver looked for an ONNX model's vision encoder
   * (`files.vision`): a stored meta from before it did, or whose look
   * failed, is looked at again when the Hub can be reached.
   */
  visionChecked?: boolean
  /**
   * The same for a GGUF model: whether the resolver looked for a vision
   * projector in its repository (a stored meta from before it did is
   * looked at again).
   */
  projectorChecked?: boolean
}

const THINKING_SWITCH = {
  levels: { off: "none", high: "high" },
  budget: true
} as const

/** What a model reads, from its files: images where it has a vision encoder (`files.vision`). */
export function inputsOf(files: WebGlModelMeta["files"]): ReadonlyArray<"text" | "image"> {
  return files.vision ? ["text", "image"] : ["text"]
}

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
const ALWAYS_THINKS = { levels: { high: "high" }, budget: true } as const

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
/** LFM2.5's reasoning as a type: Thinking builds always think, the rest don't. */
type Lfm2Reasoning<Thinking extends boolean> = Thinking extends true
  ? { readonly reasoning: typeof ALWAYS_THINKS }
  : { readonly reasoning?: undefined }

function lfm2Onnx<const Name extends string, const Thinking extends boolean>(o: {
  name: Name
  size: string
  thinking: Thinking
}) {
  const repo = { repo: o.name }
  const meta = {
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
  // `thinking` decides `reasoning` at run time; this says so in the type.
  return meta as unknown as Omit<typeof meta, "reasoning"> & Lfm2Reasoning<Thinking>
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

/** Official Q4_0 GGUFs; QAD tokenizers/templates accompany those exact builds. */
function lfm2Gguf<const Name extends string, const Thinking extends boolean>(o: { name: Name; file: string; size: string; thinking: Thinking; tokenizerRepo: string; tokenizerPrefix?: string }) {
  const tokenizer = { repo: o.tokenizerRepo }
  const prefix = o.tokenizerPrefix ?? ""
  const meta = {
    name: o.name, format: "gguf",
    files: { kind: "gguf", model: { repo: o.name, path: o.file } },
    tokenizer: {
      tokenizer: { ...tokenizer, path: `${prefix}tokenizer.json` },
      config: { ...tokenizer, path: `${prefix}tokenizer_config.json` },
      chatTemplate: { ...tokenizer, path: `${prefix}chat_template.jinja` }
    },
    supports: o.thinking ? TOOLS_AND_THINKING : TOOLS,
    size: o.size,
    ...(o.thinking && { reasoning: ALWAYS_THINKS }),
    sampling: o.thinking ? LFM2_5_THINKING_SAMPLING : LFM2_5_SAMPLING,
    toolFormat: "lfm2"
  } as const satisfies WebGlModelMeta
  // `thinking` decides `reasoning` at run time; this says so in the type.
  return meta as unknown as Omit<typeof meta, "reasoning"> & Lfm2Reasoning<Thinking>
}
const LFM2_5_350M_GGUF = lfm2Gguf({
  name: "LiquidAI/LFM2.5-350M-GGUF", file: "LFM2.5-350M-QAD-Q4_0.gguf", size: "350m", thinking: false,
  tokenizerRepo: "LiquidAI/LFM2.5-350M-GGUF", tokenizerPrefix: "qad/"
})
const LFM2_5_1_2B_GGUF = lfm2Gguf({
  name: "LiquidAI/LFM2.5-1.2B-Instruct-GGUF", file: "LFM2.5-1.2B-Instruct-QAD-Q4_0.gguf", size: "1.2b", thinking: false,
  tokenizerRepo: "LiquidAI/LFM2.5-1.2B-Instruct-GGUF", tokenizerPrefix: "qad/"
})
const LFM2_5_1_2B_THINKING_GGUF = lfm2Gguf({
  name: "LiquidAI/LFM2.5-1.2B-Thinking-GGUF", file: "LFM2.5-1.2B-Thinking-Q4_0.gguf", size: "1.2b", thinking: true,
  tokenizerRepo: "LiquidAI/LFM2.5-1.2B-Thinking"
})

/** Qwen3.5-2B model card: non-thinking text recommendations (all our inputs are text). */
const QWEN_3_5_SAMPLING = {
  temperature: 1,
  top_k: 20,
  top_p: 1,
  min_p: 0,
  presence_penalty: 2,
  repetition_penalty: 1
} as const satisfies Sampling
/** The same card's thinking text recommendations. */
const QWEN_3_5_THINKING_SAMPLING = {
  ...QWEN_3_5_SAMPLING,
  top_p: 0.95,
  presence_penalty: 1.5
} as const satisfies Sampling

const QWEN_3_5_2B_REPO = { repo: "Qwen/Qwen3.5-2B" }

/**
 * Qwen3.5 2B in unsloth's Q4_0 GGUF (1.21 GB): Gated DeltaNet layers with
 * full attention every fourth. Its projections are Q4_0; a few matrices
 * (Q4_1, Q5_K) load as 8-bit. Tokenizer, chat template and the image
 * processor's config from Qwen's own repository. Its vision projector is
 * the repository's float16 one (668 MB: there is no smaller), stored at 8
 * bits. Thinking is off unless turned on.
 */
const QWEN_3_5_2B_GGUF_FILES = {
  kind: "gguf",
  model: { repo: "unsloth/Qwen3.5-2B-GGUF", path: "Qwen3.5-2B-Q4_0.gguf" },
  vision: {
    projector: { repo: "unsloth/Qwen3.5-2B-GGUF", path: "mmproj-F16.gguf" },
    preprocessor: { ...QWEN_3_5_2B_REPO, path: "preprocessor_config.json" }
  }
} as const

const QWEN_3_5_2B_GGUF = {
  name: "unsloth/Qwen3.5-2B-GGUF",
  format: "gguf",
  files: QWEN_3_5_2B_GGUF_FILES,
  tokenizer: {
    tokenizer: { ...QWEN_3_5_2B_REPO, path: "tokenizer.json" },
    config: { ...QWEN_3_5_2B_REPO, path: "tokenizer_config.json" },
    chatTemplate: { ...QWEN_3_5_2B_REPO, path: "chat_template.jinja" }
  },
  supports: { ...TOOLS_AND_THINKING, input: inputsOf(QWEN_3_5_2B_GGUF_FILES) },
  size: "2b",
  reasoning: THINKING_SWITCH,
  sampling: QWEN_3_5_SAMPLING,
  thinkingSampling: QWEN_3_5_THINKING_SAMPLING,
  toolFormat: "qwen3.5"
} as const satisfies WebGlModelMeta

const QWEN_3_5_2B_ONNX_REPO = { repo: "onnx-community/Qwen3.5-2B-ONNX" }

/**
 * Qwen3.5 2B in onnx-community's q4f16 build, the one the Transformers.js
 * package runs: only its decoder (1.09 GB), whose 4-bit lm_head is the
 * embedding too (the separate embed_tokens file holds the same bytes).
 */
const QWEN_3_5_2B_ONNX_FILES = {
  kind: "onnx",
  graph: { ...QWEN_3_5_2B_ONNX_REPO, path: "onnx/decoder_model_merged_q4f16.onnx" },
  data: {
    ...QWEN_3_5_2B_ONNX_REPO,
    path: "onnx/decoder_model_merged_q4f16.onnx_data"
  },
  config: { ...QWEN_3_5_2B_ONNX_REPO, path: "config.json" },
  vision: {
    graph: { ...QWEN_3_5_2B_ONNX_REPO, path: "onnx/vision_encoder_q4f16.onnx" },
    data: { ...QWEN_3_5_2B_ONNX_REPO, path: "onnx/vision_encoder_q4f16.onnx_data" },
    preprocessor: { ...QWEN_3_5_2B_ONNX_REPO, path: "preprocessor_config.json" }
  }
} as const

const QWEN_3_5_2B_ONNX = {
  name: "onnx-community/Qwen3.5-2B-ONNX",
  format: "onnx",
  files: QWEN_3_5_2B_ONNX_FILES,
  tokenizer: {
    tokenizer: { ...QWEN_3_5_2B_ONNX_REPO, path: "tokenizer.json" },
    config: { ...QWEN_3_5_2B_ONNX_REPO, path: "tokenizer_config.json" },
    chatTemplate: { ...QWEN_3_5_2B_ONNX_REPO, path: "chat_template.jinja" }
  },
  supports: { ...TOOLS_AND_THINKING, input: inputsOf(QWEN_3_5_2B_ONNX_FILES) },
  size: "2b",
  reasoning: THINKING_SWITCH,
  sampling: QWEN_3_5_SAMPLING,
  thinkingSampling: QWEN_3_5_THINKING_SAMPLING,
  toolFormat: "qwen3.5"
} as const satisfies WebGlModelMeta

/** LFM2.5 350M: the smallest and quickest, for short answers and tool calls. */
const LFM2_5_350M = lfm2Onnx({
  name: "LiquidAI/LFM2.5-350M-ONNX",
  size: "350m",
  thinking: false
})

/** LFM2 1.2B Tool's card: greedy decoding (temperature 0). */
const GREEDY = { temperature: 0, top_k: 1, repetition_penalty: 1 } as const satisfies Sampling

const LFM2_1_2B_TOOL_REPO = { repo: "LiquidAI/LFM2-1.2B-Tool" }

/**
 * Liquid AI's LFM2 1.2B Tool, tuned for tool calls, in its official Q4_0
 * GGUF; tokenizer and chat template from the original repository (the GGUF
 * repository has none). It writes LFM2's Pythonic calls and doesn't think.
 */
const LFM2_1_2B_TOOL_GGUF = {
  name: "LiquidAI/LFM2-1.2B-Tool-GGUF",
  format: "gguf",
  files: { kind: "gguf", model: { repo: "LiquidAI/LFM2-1.2B-Tool-GGUF", path: "LFM2-1.2B-Tool-Q4_0.gguf" } },
  tokenizer: {
    tokenizer: { ...LFM2_1_2B_TOOL_REPO, path: "tokenizer.json" },
    config: { ...LFM2_1_2B_TOOL_REPO, path: "tokenizer_config.json" },
    chatTemplate: { ...LFM2_1_2B_TOOL_REPO, path: "chat_template.jinja" }
  },
  supports: TOOLS,
  size: "1.2b",
  reasoning: undefined,
  sampling: GREEDY,
  toolFormat: "lfm2"
} as const satisfies WebGlModelMeta

const LFM2_1_2B_TOOL_ONNX_REPO = { repo: "onnx-community/LFM2-1.2B-Tool-ONNX" }

/** The onnx-community export of LFM2 1.2B Tool (q4f16, one external data file). */
const LFM2_1_2B_TOOL_ONNX = {
  name: "onnx-community/LFM2-1.2B-Tool-ONNX",
  format: "onnx",
  files: {
    kind: "onnx",
    graph: { ...LFM2_1_2B_TOOL_ONNX_REPO, path: "onnx/model_q4f16.onnx" },
    data: { ...LFM2_1_2B_TOOL_ONNX_REPO, path: "onnx/model_q4f16.onnx_data" },
    config: { ...LFM2_1_2B_TOOL_ONNX_REPO, path: "config.json" }
  },
  tokenizer: {
    tokenizer: { ...LFM2_1_2B_TOOL_ONNX_REPO, path: "tokenizer.json" },
    config: { ...LFM2_1_2B_TOOL_ONNX_REPO, path: "tokenizer_config.json" },
    chatTemplate: { ...LFM2_1_2B_TOOL_ONNX_REPO, path: "chat_template.jinja" }
  },
  supports: TOOLS,
  size: "1.2b",
  reasoning: undefined,
  sampling: GREEDY,
  toolFormat: "lfm2"
} as const satisfies WebGlModelMeta

/** LFM2.5-VL 450M's and 1.6B's cards, for text: temperature 0.1, min-p 0.15, repetition penalty 1.05 (Transformers' top-k, 50). */
const LFM2_5_VL_SAMPLING = {
  temperature: 0.1,
  top_k: 50,
  min_p: 0.15,
  repetition_penalty: 1.05
} as const satisfies Sampling

/** LFM2.5-VL 3B's card: temperature 0.2, top-k 50, repetition penalty 1.0. */
const LFM2_5_VL_3B_SAMPLING = {
  temperature: 0.2,
  top_k: 50,
  repetition_penalty: 1
} as const satisfies Sampling

const IMAGES_AND_TOOLS = { input: ["text", "image"], output: ["text"], capabilities: ["tools"] } as const
const IMAGES = { input: ["text", "image"], output: ["text"], capabilities: [] } as const
const IMAGES_TOOLS_AND_THINKING = { input: ["text", "image"], output: ["text"], capabilities: ["tools", "thinking"] } as const

/**
 * A Liquid AI vision-language model in its official GGUF build: the text
 * model's Q4_0 file (quantization-aware where there is one) and the
 * repository's Q8_0 vision projector (`mmproj`). The tokenizer, the chat
 * template and the image processor's config come from the original
 * repository (the GGUF repository has none).
 */
function vlGguf<const Name extends string, const Supports extends WebGlModelMeta["supports"]>(o: {
  name: Name
  /** The original repository. */
  base: string
  file: string
  projector: string
  size: string
  supports: Supports
  sampling: Sampling
  toolFormat: ToolFormat
}) {
  const repo = { repo: o.name }
  const base = { repo: o.base }
  return {
    name: o.name,
    format: "gguf",
    files: {
      kind: "gguf",
      model: { ...repo, path: o.file },
      vision: { projector: { ...repo, path: o.projector }, preprocessor: { ...base, path: "processor_config.json" } }
    },
    tokenizer: {
      tokenizer: { ...base, path: "tokenizer.json" },
      config: { ...base, path: "tokenizer_config.json" },
      chatTemplate: { ...base, path: "chat_template.jinja" }
    },
    supports: o.supports,
    size: o.size,
    reasoning: undefined,
    sampling: o.sampling,
    toolFormat: o.toolFormat
  } as const satisfies WebGlModelMeta
}

/**
 * A Liquid AI vision-language model in its official ONNX export: the 4-bit
 * decoder (its float embedding, which is also its head, is quantized to 8
 * bits at load) and the 8-bit vision encoder beside it (`vision_encoder_q8`,
 * or `embed_images_q8`), whose features are within a few percent of the
 * float encoder's (the 4-bit one's are far off). The separate embed_tokens
 * file isn't needed.
 */
function vlOnnx<const Name extends string, const Supports extends WebGlModelMeta["supports"]>(o: {
  name: Name
  /** The decoder graph, and the vision encoder graph, under onnx/. */
  decoder: string
  /** The decoder's external data files, in the order the loader reads them. */
  data: [string, ...Array<string>]
  encoder: string
  /** The image processor's config: preprocessor_config.json, or the processor_config.json that holds it. */
  preprocessor: string
  size: string
  supports: Supports
  sampling: Sampling
  toolFormat: ToolFormat
}) {
  const repo = { repo: o.name }
  const [data, ...more] = o.data
  return {
    name: o.name,
    format: "onnx",
    files: {
      kind: "onnx",
      graph: { ...repo, path: `onnx/${o.decoder}.onnx` },
      data: { ...repo, path: `onnx/${data}` },
      ...(more.length > 0 && { moreData: more.map((path) => ({ ...repo, path: `onnx/${path}` })) }),
      config: { ...repo, path: "config.json" },
      vision: {
        graph: { ...repo, path: `onnx/${o.encoder}.onnx` },
        data: { ...repo, path: `onnx/${o.encoder}.onnx_data` },
        preprocessor: { ...repo, path: o.preprocessor }
      }
    },
    tokenizer: {
      tokenizer: { ...repo, path: "tokenizer.json" },
      config: { ...repo, path: "tokenizer_config.json" },
      chatTemplate: { ...repo, path: "chat_template.jinja" }
    },
    supports: o.supports,
    size: o.size,
    reasoning: undefined,
    sampling: o.sampling,
    toolFormat: o.toolFormat
  } as const satisfies WebGlModelMeta
}

/** LFM3-VL 2B: LFM3 with a vision tower of its own; thinks unless turned off. Its card's sampling is LFM3's. */
const LFM3_VL_2B_GGUF = {
  ...vlGguf({
    name: "LiquidAI/LFM3-VL-2B-GGUF",
    base: "LiquidAI/LFM3-VL-2B",
    file: "LFM3-VL-2B-QAD-Q4_0.gguf",
    projector: "mmproj-LFM3-VL-2B-Q8_0.gguf",
    size: "2b",
    supports: IMAGES_TOOLS_AND_THINKING,
    sampling: LFM3_SAMPLING,
    toolFormat: "lfm3"
  }),
  reasoning: THINKING_SWITCH
} as const satisfies WebGlModelMeta

/** LFM3-VL 2B's ONNX export: a decoder of 4- and 8-bit projections (q4f16), which reads token ids. */
const LFM3_VL_2B_ONNX = {
  ...vlOnnx({
    name: "LiquidAI/LFM3-VL-2B-ONNX",
    decoder: "decoder_q4f16",
    data: ["decoder_q4f16.onnx_data"],
    encoder: "vision_encoder_q8",
    preprocessor: "processor_config.json",
    size: "2b",
    supports: IMAGES_TOOLS_AND_THINKING,
    sampling: LFM3_SAMPLING,
    toolFormat: "lfm3"
  }),
  reasoning: THINKING_SWITCH
} as const satisfies WebGlModelMeta

/** LFM2.5-VL 3B: LFM2.5 with SigLIP2 (so400m); large images are read in 512-pixel tiles. */
const LFM2_5_VL_3B_GGUF = vlGguf({
  name: "LiquidAI/LFM2.5-VL-3B-GGUF",
  base: "LiquidAI/LFM2.5-VL-3B",
  file: "LFM2.5-VL-3B-Q4_0.gguf",
  projector: "mmproj-LFM2.5-VL-3B-Q8_0.gguf",
  size: "3b",
  supports: IMAGES_AND_TOOLS,
  sampling: LFM2_5_VL_3B_SAMPLING,
  toolFormat: "lfm2"
})

/** LFM2.5-VL 3B's ONNX export: its decoder's weights are in four data files. */
const LFM2_5_VL_3B_ONNX = vlOnnx({
  name: "LiquidAI/LFM2.5-VL-3B-ONNX",
  decoder: "decoder_model_merged_q4",
  data: [
    "decoder_model_merged_q4.onnx_data_2",
    "decoder_model_merged_q4.onnx_data_3",
    "decoder_model_merged_q4.onnx_data_4",
    "decoder_model_merged_q4.onnx_data_1"
  ],
  encoder: "vision_encoder_q8",
  preprocessor: "preprocessor_config.json",
  size: "3b",
  supports: IMAGES_AND_TOOLS,
  sampling: LFM2_5_VL_3B_SAMPLING,
  toolFormat: "lfm2"
})

/** LFM2.5-VL 1.6B: LFM2.5 1.2B with SigLIP2 (so400m). */
const LFM2_5_VL_1_6B_GGUF = vlGguf({
  name: "LiquidAI/LFM2.5-VL-1.6B-GGUF",
  base: "LiquidAI/LFM2.5-VL-1.6B",
  file: "LFM2.5-VL-1.6B-Q4_0.gguf",
  projector: "mmproj-LFM2.5-VL-1.6b-Q8_0.gguf",
  size: "1.6b",
  supports: IMAGES_AND_TOOLS,
  sampling: LFM2_5_VL_SAMPLING,
  toolFormat: "lfm2"
})

/** LFM2.5-VL 1.6B's ONNX export, whose chat template takes no tools. */
const LFM2_5_VL_1_6B_ONNX = vlOnnx({
  name: "LiquidAI/LFM2.5-VL-1.6B-ONNX",
  decoder: "decoder_q4",
  data: ["decoder_q4.onnx_data"],
  encoder: "embed_images_q8",
  preprocessor: "processor_config.json",
  size: "1.6b",
  supports: IMAGES,
  sampling: LFM2_5_VL_SAMPLING,
  toolFormat: "lfm2"
})

/** LFM2.5-VL 450M: LFM2.5 350M with SigLIP2 (base), the smallest that reads images. */
const LFM2_5_VL_450M_GGUF = vlGguf({
  name: "LiquidAI/LFM2.5-VL-450M-GGUF",
  base: "LiquidAI/LFM2.5-VL-450M",
  file: "LFM2.5-VL-450M-Q4_0.gguf",
  projector: "mmproj-LFM2.5-VL-450m-Q8_0.gguf",
  size: "450m",
  supports: IMAGES_AND_TOOLS,
  sampling: LFM2_5_VL_SAMPLING,
  toolFormat: "lfm2"
})

const LFM2_5_VL_450M_ONNX = vlOnnx({
  name: "LiquidAI/LFM2.5-VL-450M-ONNX",
  decoder: "decoder_model_merged_q4",
  data: ["decoder_model_merged_q4.onnx_data"],
  encoder: "vision_encoder_q8",
  preprocessor: "preprocessor_config.json",
  size: "450m",
  supports: IMAGES_AND_TOOLS,
  sampling: LFM2_5_VL_SAMPLING,
  toolFormat: "lfm2"
})

export const WEBGL_TEXT_MODELS = [
  LFM3_1_7B_GGUF.name,
  LFM3_1_7B_ONNX.name,
  LFM2_5_1_2B.name,
  LFM2_5_1_2B_THINKING.name,
  LFM2_5_350M.name,
  LFM2_5_350M_GGUF.name,
  LFM2_5_1_2B_GGUF.name,
  LFM2_5_1_2B_THINKING_GGUF.name,
  QWEN_3_5_2B_GGUF.name,
  QWEN_3_5_2B_ONNX.name,
  LFM2_1_2B_TOOL_GGUF.name,
  LFM2_1_2B_TOOL_ONNX.name,
  LFM3_VL_2B_GGUF.name,
  LFM3_VL_2B_ONNX.name,
  LFM2_5_VL_3B_GGUF.name,
  LFM2_5_VL_3B_ONNX.name,
  LFM2_5_VL_1_6B_GGUF.name,
  LFM2_5_VL_1_6B_ONNX.name,
  LFM2_5_VL_450M_GGUF.name,
  LFM2_5_VL_450M_ONNX.name
] as const

/** A model this package runs. */
export type WebGlTextModel = (typeof WEBGL_TEXT_MODELS)[number]

/**
 * Each model's files, capabilities, reasoning and sampling. `satisfies`
 * rather than a type annotation keeps each entry's exact type, so a
 * model's reasoning levels can be read per model (see WebGlModelReasoning).
 */
export const WEBGL_MODEL_META = {
  [LFM3_1_7B_GGUF.name]: LFM3_1_7B_GGUF,
  [LFM3_1_7B_ONNX.name]: LFM3_1_7B_ONNX,
  [LFM2_5_1_2B.name]: LFM2_5_1_2B,
  [LFM2_5_1_2B_THINKING.name]: LFM2_5_1_2B_THINKING,
  [LFM2_5_350M.name]: LFM2_5_350M,
  [LFM2_5_350M_GGUF.name]: LFM2_5_350M_GGUF,
  [LFM2_5_1_2B_GGUF.name]: LFM2_5_1_2B_GGUF,
  [LFM2_5_1_2B_THINKING_GGUF.name]: LFM2_5_1_2B_THINKING_GGUF,
  [QWEN_3_5_2B_GGUF.name]: QWEN_3_5_2B_GGUF,
  [QWEN_3_5_2B_ONNX.name]: QWEN_3_5_2B_ONNX,
  [LFM2_1_2B_TOOL_GGUF.name]: LFM2_1_2B_TOOL_GGUF,
  [LFM2_1_2B_TOOL_ONNX.name]: LFM2_1_2B_TOOL_ONNX,
  [LFM3_VL_2B_GGUF.name]: LFM3_VL_2B_GGUF,
  [LFM3_VL_2B_ONNX.name]: LFM3_VL_2B_ONNX,
  [LFM2_5_VL_3B_GGUF.name]: LFM2_5_VL_3B_GGUF,
  [LFM2_5_VL_3B_ONNX.name]: LFM2_5_VL_3B_ONNX,
  [LFM2_5_VL_1_6B_GGUF.name]: LFM2_5_VL_1_6B_GGUF,
  [LFM2_5_VL_1_6B_ONNX.name]: LFM2_5_VL_1_6B_ONNX,
  [LFM2_5_VL_450M_GGUF.name]: LFM2_5_VL_450M_GGUF,
  [LFM2_5_VL_450M_ONNX.name]: LFM2_5_VL_450M_ONNX
} as const satisfies Readonly<Record<WebGlTextModel, WebGlModelMeta>>

/** A model's reasoning, exactly: its levels, or undefined for a model that doesn't think. */
export type WebGlModelReasoning<Model extends WebGlTextModel> = (typeof WEBGL_MODEL_META)[Model]["reasoning"]

/**
 * A model id: a catalog model (autocompleted), or any Hugging Face repository
 * (`owner/name`) with a supported architecture, resolved from the Hub.
 */
export type WebGlModelId = WebGlTextModel | (string & {})

/** Models resolved from the Hub in this thread (see resolve-model.ts), by name. */
const resolved = new Map<string, WebGlModelMeta>()

/** Whether `model` is in the catalog. */
export const isCatalogModel = (model: unknown): model is WebGlTextModel =>
  typeof model === "string" && Object.hasOwn(WEBGL_MODEL_META, model)

/** A model's meta: the catalog's, or one resolved earlier in this thread; undefined otherwise. */
export function knownModelMeta(model: string): WebGlModelMeta | undefined {
  return isCatalogModel(model) ? WEBGL_MODEL_META[model] : resolved.get(model)
}

/** Keeps a model's resolved meta for the sync lookups (modelMeta) of this thread. */
export function rememberModel(meta: WebGlModelMeta) {
  if (!isCatalogModel(meta.name)) resolved.set(meta.name, meta)
}

/** Forgets resolved models, as a page reload does (for tests). */
export const forgetResolvedModels = () => resolved.clear()

/** Every model this thread knows: the catalog's, then those resolved since. */
export const knownModels = (): Array<string> => [...WEBGL_TEXT_MODELS, ...resolved.keys()]

/**
 * A model's meta, for code that runs after the model was resolved
 * (resolveModel): the catalog's or the resolved one. Throws for a model
 * that wasn't.
 */
export function modelMeta(model: string): WebGlModelMeta {
  const meta = knownModelMeta(model)
  if (!meta) throw new Error(`${model} wasn't resolved before its meta was read.`)
  return meta
}

/**
 * Where a file downloads from: its latest version (main) on Hugging Face,
 * or, for a model given by URL (whose `repo` is the folder's URL), the
 * folder.
 */
export const fileUrl = (file: ModelFile) =>
  /^https?:\/\//.test(file.repo) ? `${file.repo}/${file.path}` : `https://huggingface.co/${file.repo}/resolve/main/${file.path}`
