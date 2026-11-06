/** Messages between the page and the WebGL 2 worker. */
import type { Sampling } from "./sampler.js"
import type { ChatMessage, ChatTool } from "./tokenizer.js"

export type GenerationPhase = "processing" | "generating"

/** Why a generation ended: its end-of-turn token or a stop sequence, or maxTokens. */
export type FinishReason = "stop" | "length"

/** Error codes the worker attaches to failures (WebGLLMError.code on the page). */
export type WorkerErrorCode =
  | "webgl2-unavailable"
  | "download-failed"
  | "storage-full"
  | "out-of-memory"
  | "gpu-context-lost"
  | "context-length-exceeded"
  | "invalid-schema"
  /** The output couldn't satisfy the schema's pattern, format or numeric constraints. */
  | "invalid-output"

/**
 * OpenAI's tool_choice: "auto" lets the model decide, "none" offers no tools,
 * "required" makes it call one of them, a name makes it call that one.
 */
export type ToolChoice = "auto" | "none" | "required" | { name: string }

/** Token counts and timings of one generation, measured in the worker. */
export type RunTimings = {
  /** Tokens of the prompt, including any already in the cache. */
  inputTokens: number
  /** Tokens at the start of the prompt read on an earlier run. */
  cachedTokens: number
  /** Tokens generated (the end-of-turn token included). */
  outputTokens: number
  /** From the request to the first generated token, in ms; null if none. */
  firstTokenMs: number | null
  /** Reading the part of the prompt that wasn't cached, in ms. */
  promptMs: number
  /** Chat-template encoding and token-verified prefix-boundary discovery. */
  tokenizeMs?: number
  /** Generation-local schema/constraint setup, before prefill. */
  schemaMs?: number
  /** CPU prefix matching and restore-command submission, without a GPU synchronization. */
  cacheRestoreMs?: number
  /** Tokens matching the previous request before checkpoint selection. */
  cacheMatchedTokens?: number
  /** Actual retained system/tools prefix checkpoint selected for this request. */
  cachePrefixTokens?: number
  /** Actual retained ordinary conversational checkpoint. */
  cacheAnchorTokens?: number
  /** From the first generated token to the last, in ms. */
  decodeMs: number
  /** Speculative decoding: batches with drafts, drafts checked, drafts kept. */
  draftBatches?: number
  draftsChecked?: number
  draftsKept?: number
  /** Thinking content tokens, excluding opening and closing markers. */
  thinkingTokens?: number
  /** Requested thinking ceiling; omitted when generation is unbudgeted. */
  thinkingBudgetTokens?: number
  thinkingBudgetExhausted?: boolean
}

export type WorkerRequest =
  | {
      type: "load"
      model: string
      /**
       * Most tokens a chat may hold, prompt and reply together: the model's
       * own context length when left out, and never more. A different value
       * loads the model again.
       */
      contextLength?: number
    }
  | {
      type: "generate"
      id: string
      /** Send real prefill/decode lifecycle messages only when requested. */
      reportPhases?: boolean
      /** The model to run it on: loaded first if it isn't (or its GPU context was lost). */
      model: string
      /** The context to load it for, if it's loaded first (see "load"). */
      contextLength?: number
      chat: Array<ChatMessage>
      /** Explicit false disables cross-call inference-cache reuse and retention. */
      cachePrompt?: boolean
      tools: Array<ChatTool>
      /** The chat template's `enable_thinking`; its default when left out. */
      thinking?: boolean
      /** Close thinking at this many content tokens, then continue the answer. */
      thinkingBudgetTokens?: number
      /** JSON Schema enforced during answer token selection. */
      outputSchema?: object
      sampling: Sampling
      /** Most tokens to generate; EOS or remaining model context when left out. */
      maxTokens?: number
      /**
       * End the reply where any of these strings first appears in its
       * answer (not in its thinking); the stop string itself is not sent.
       */
      stop?: Array<string>
      /** Seeds the sampler: the same seed, prompt and settings give the same reply. */
      seed?: number
      /** "auto" when left out. A forced call is sent as `toolCall` in "done", not as text. */
      toolChoice?: ToolChoice
      /** Draft tokens by prompt lookup (speculative decoding); on by default. */
      speculative?: boolean
      /**
       * Read the likeliest next token forward while each token is drawn
       * (lossless); on by default.
       */
      pipeline?: boolean
    }
  | { type: "stop"; id: string }
  /** Download a model's files to storage without loading it ("progress", then "downloaded"). */
  | { type: "download"; model: string }
  | { type: "frames"; intervalMs: number; expectedMs: number; frames: number; startedAt?: number; meanMs?: number }
  /** The page made room for the model (see "claim"): the worker loads it. */
  | { type: "claimed" }
  /** Engine tuning knobs (see TUNING in engine.ts): for benchmarks, and setPromptBatch. */
  | { type: "tune"; values: Record<string, number> }

export type WorkerResponse =
  | {
      type: "progress"
      model: string
      progress: number
      cached: boolean
      loadedBytes?: number
      totalBytes?: number
    }
  | {
      type: "ready"
      model: string
      /** Tokens the loaded model holds, prompt and reply together. */
      contextLength?: number
      /** Milliseconds each phase of the load took (a fresh load only). */
      timings?: Record<string, number>
    }
  | { type: "load-error"; model: string; message: string; code?: WorkerErrorCode }
  | { type: "downloaded"; model: string }
  | { type: "download-error"; model: string; message: string; code?: WorkerErrorCode }
  /**
   * The model takes about `bytes` once loaded: the page makes room for it
   * (unloading other runtimes' models past the memory budget, see
   * model-memory.ts) and answers "claimed".
   */
  | { type: "claim"; model: string; bytes: number }
  | { type: "phase"; id: string; phase: GenerationPhase }
  | { type: "token"; id: string; text: string }
  | {
      type: "done"
      id: string
      stats: RunTimings
      finishReason: FinishReason
      /** The call a forced toolChoice ("required" or a name) produced. */
      toolCall?: { name: string; arguments: Record<string, unknown> }
    }
  | { type: "error"; id: string; message: string; code?: WorkerErrorCode }
  | { type: "log"; text: string }
