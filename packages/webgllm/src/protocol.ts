/** Messages between the page and the WebGL 2 worker. */
import type { Sampling } from "./sampler.js"
import type { ChatMessage, ChatTool } from "./tokenizer.js"

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
  /** From the first generated token to the last, in ms. */
  decodeMs: number
}

export type WorkerRequest =
  | { type: "load"; model: string }
  | {
      type: "generate"
      id: string
      /** The model to run it on: loaded first if it isn't (or its GPU context was lost). */
      model: string
      chat: Array<ChatMessage>
      tools: Array<ChatTool>
      /** The chat template's `enable_thinking`; its default when left out. */
      thinking?: boolean
      sampling: Sampling
      /** Most tokens to generate; the worker's default when left out. */
      maxTokens?: number
      /** Draft tokens by prompt lookup (speculative decoding); on by default. */
      speculative?: boolean
      /**
       * Read the likeliest next token forward while each token is drawn
       * (lossless); on by default.
       */
      pipeline?: boolean
    }
  | { type: "stop"; id: string }
  /** The page's navigator.maxTouchPoints (a worker can't read it). */
  | { type: "device"; touchPoints: number }
  /** For benchmarks: engine tuning knobs (see TUNING in engine.ts). */
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
  | { type: "ready"; model: string }
  | { type: "load-error"; model: string; message: string }
  | { type: "token"; id: string; text: string }
  | { type: "done"; id: string; stats: RunTimings }
  | { type: "error"; id: string; message: string }
  | { type: "log"; text: string }
