/** Messages between the page and the WebGL 2 worker. */

/**
 * The property that marks a message as webgllm's, set true on every one
 * either side sends: a worker of the caller's carries its own messages
 * too, and those are never taken for these, whatever their type.
 */
export const WEBGLLM = "webgllm"
import type { WebGlModelMeta } from "./model-meta.js"
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
  /**
   * The prompt's images (only with images): fetching, resizing and encoding
   * the ones not encoded before, the vision encoder's load included.
   */
  imagesMs?: number
  /** Of imagesMs, getting the vision encoder's files (their download included; 0 when no image needed it). */
  visionLoadMs?: number
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

/** A token's log probability, as OpenAI's chat completions report it. */
export type TokenLogprob = {
  token: string
  logprob: number
  /** The token's UTF-8 bytes. */
  bytes: Array<number>
  /** The likeliest tokens at this position (`top_logprobs` of them). */
  top_logprobs: Array<{ token: string; logprob: number; bytes: Array<number> }>
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
      /**
       * Echoed in the "ready" or "load-error" that answers this load: never
       * used twice in one worker, so a late answer to an earlier load of the
       * same model can't pass for this one's.
       */
      loadId: number
      /** A model that isn't in the catalog: its meta, as the page resolved it from the Hub. */
      meta?: WebGlModelMeta
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
      /** A raw prompt in place of `chat`: tokenized as it is, with no chat template (OpenAI's Completions). */
      prompt?: string
      /** Send each written token's log probability, with its `topLogprobs` likeliest alternatives. */
      logprobs?: boolean
      topLogprobs?: number
      /** OpenAI's logit_bias: token ids to a bias added to their logits before sampling. */
      logitBias?: Record<string, number>
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
       * answer (not in its thinking, nor in an answer constrained to a
       * schema); the stop string itself is not sent.
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
  /**
   * WebLLM's forwardTokensAndSample: read `inputIds` into the model's cache
   * after what it holds, and sample the next token ("forwarded"). In the
   * queue with the generations; the model loads first if it isn't loaded.
   */
  | { type: "forward"; id: string; model: string; contextLength?: number; inputIds: Array<number>; sampling: Sampling }
  /** WebLLM's resetChat: forget the model's cache ("reset-done"). */
  | { type: "reset"; id: string; model: string }
  /** Stops generation `id`, or download `id`. */
  | { type: "stop"; id: string }
  /**
   * Free `model` (stopping a load of it) and stay running: sent to a worker
   * the caller started, where the page's own worker would be ended.
   * Answered with "unloaded" once its GPU memory is free.
   */
  | { type: "unload"; model: string; id: number }
  /** Download a model's files to storage without loading it ("progress", then "downloaded"). */
  | { type: "download"; id: string; model: string; meta?: WebGlModelMeta }
  | { type: "frames"; intervalMs: number; expectedMs: number; frames: number; startedAt?: number; meanMs?: number }
  /** The page made room for claim `id` (see "claim"): the worker goes on. */
  | { type: "claimed"; id: number }
  /** Engine tuning knobs (see TUNING in engine.ts): for benchmarks. */
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
      /** The load this answers (see "load"). */
      loadId: number
      /** Tokens the loaded model holds, prompt and reply together. */
      contextLength?: number
      /** Milliseconds each phase of the load took (a fresh load only). */
      timings?: Record<string, number>
    }
  | { type: "load-error"; model: string; loadId: number; message: string; code?: WorkerErrorCode }
  /** A load has the model's files in storage, before the GPU reads them (downloads waiting on it settle). */
  | { type: "stored"; model: string }
  | { type: "downloaded"; model: string; id: string }
  /** A download failed, or was stopped (sent once its files' writers have closed). */
  | { type: "download-error"; model: string; id: string; message: string; code?: WorkerErrorCode }
  /** The model of "unload" `id` is freed, with its GPU memory. */
  | { type: "unloaded"; id: number }
  /**
   * The model takes about `bytes` once loaded: the page makes room for it
   * (unloading other runtimes' models past the memory budget, see
   * model-memory.ts) and answers "claimed".
   */
  | { type: "claim"; id: number; model: string; bytes: number }
  | { type: "phase"; id: string; phase: GenerationPhase }
  | { type: "token"; id: string; text: string; logprobs?: Array<TokenLogprob> }
  | {
      type: "done"
      id: string
      stats: RunTimings
      finishReason: FinishReason
      /** Log probabilities of tokens written after the last "token" message. */
      logprobs?: Array<TokenLogprob>
      /** The call a forced toolChoice ("required" or a name) produced. */
      toolCall?: { name: string; arguments: Record<string, unknown> }
    }
  /** A generation, a "forward" or a "reset" failed. */
  | { type: "error"; id: string; message: string; code?: WorkerErrorCode }
  /** "forward" `id`'s next token. */
  | { type: "forwarded"; id: string; token: number }
  | { type: "reset-done"; id: string }
  | { type: "log"; text: string }
