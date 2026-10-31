/**
 * LFM3's tokenizer and chat template, on Tokenizers.js and Hugging Face's
 * Jinja. Serves both builds: the GGUF one (LiquidAI/LFM3-1.7B's
 * tokenizer.json, 128k ids) and the ONNX one (a 49k-token trimmed
 * vocabulary). No Node APIs: it runs in a Web Worker.
 *
 * Matches Python `tokenizers` (ids, decoding) and transformers'
 * apply_chat_template (prompts) byte for byte; scripts/check-tokenizer.ts
 * checks both.
 */

import { Template } from "@huggingface/jinja"
import { Tokenizer as HfTokenizer } from "@huggingface/tokenizers"

/** A tool call in a past assistant turn; arguments as an object. */
export type ChatToolCall = {
  id?: string
  type?: "function"
  function: { name: string; arguments: Record<string, unknown> }
}

/** A chat message as the chat template reads it (transformers' shape). */
export type ChatMessage = {
  role: "system" | "user" | "assistant" | "tool"
  content: string
  /** The reply's thinking; LFM3's template keeps it for the current turn. */
  thinking?: string
  /** Another name for the thinking (LFM3's template reads either). */
  reasoning?: string
  tool_calls?: Array<ChatToolCall>
  /** A tool result's call, and its tool's name. */
  tool_call_id?: string
  name?: string
}

/** A tool as chat templates take it (OpenAI's function format). */
export type ChatTool = {
  type: "function"
  function: { name: string; description?: string; parameters?: unknown }
}

export type ChatTemplateOptions = {
  tools?: Array<ChatTool>
  /** The template's `enable_thinking`; its default (true) when left out. */
  enableThinking?: boolean
  /** Ends the prompt with the assistant's turn opening; false by default. */
  addGenerationPrompt?: boolean
  /** Keeps the thinking of earlier turns too (`preserve_thinking`). */
  preserveThinking?: boolean
  /** More template variables, by their names in the template. */
  [variable: string]: unknown
}

export type SpecialTokenIds = {
  /** `<|startoftext|>`; the chat template writes it itself. */
  bos: number
  /** `<|im_end|>`, which ends every turn: the model's end token. */
  eos: number
  pad: number
  /** `<|endoftext|>` */
  endOfText: number | null
  /** `<|im_start|>` */
  imStart: number | null
  /** `<think>` and `</think>` */
  think: number | null
  thinkEnd: number | null
  /** `<tool_call>` and `</tool_call>` */
  toolCall: number | null
  toolCallEnd: number | null
}

/** Turns generated ids into text as they come. */
export type StreamDecoder = {
  /**
   * Adds the next id and returns the text it completes, "" while a
   * character is still split across ids (or for a skipped special token).
   */
  push: (id: number) => string
  /** Returns whatever is held back (an unfinished character as U+FFFD). */
  flush: () => string
  /** Starts over, for the next reply. */
  reset: () => void
}

export type Tokenizer = {
  /**
   * Ids of `text`. Special tokens in it (`<|im_start|>`, `<think>`, ...)
   * become their ids, as transformers does. LFM3's post-processor adds no
   * tokens, so `addSpecialTokens` changes nothing; the chat template writes
   * `<|startoftext|>` itself.
   */
  encode: (
    text: string,
    options?: { addSpecialTokens?: boolean }
  ) => Array<number>
  /** Text of `ids`; ids the vocabulary doesn't have are left out. */
  decode: (
    ids: ArrayLike<number>,
    options?: { skipSpecialTokens?: boolean }
  ) => string
  createStreamDecoder: (options?: {
    skipSpecialTokens?: boolean
  }) => StreamDecoder
  /** The prompt for `messages`, as transformers' apply_chat_template writes it. */
  applyChatTemplate: (
    messages: Array<ChatMessage>,
    options?: ChatTemplateOptions
  ) => string
  /** The prompt's ids (apply_chat_template with tokenize=True). */
  encodeChat: (
    messages: Array<ChatMessage>,
    options?: ChatTemplateOptions
  ) => Array<number>
  tokenToId: (token: string) => number | undefined
  idToToken: (id: number) => string | undefined
  /** Ids `isSpecial` is true of: those decode skips when asked to. */
  isSpecial: (id: number) => boolean
  specialTokens: SpecialTokenIds
  /** One past the highest id the tokenizer knows (the model's may be more). */
  vocabSize: number
}

export type TokenizerFiles = {
  /** tokenizer.json, parsed. */
  tokenizerJson: object
  /** tokenizer_config.json, parsed: the special tokens' names. */
  tokenizerConfig?: Record<string, unknown>
  /** chat_template.jinja; tokenizer_config.json's `chat_template` if left out. */
  chatTemplate?: string
}

/** LFM3's special tokens, for a tokenizer_config.json that doesn't name them. */
const DEFAULT_SPECIAL_TOKENS: Record<string, string> = {
  bos_token: "<|startoftext|>",
  eos_token: "<|im_end|>",
  pad_token: "<|pad|>"
}

type AddedTokenJson = { id: number; content: string; special?: boolean }

export function loadTokenizer(files: TokenizerFiles): Tokenizer {
  const config = files.tokenizerConfig ?? {}
  const chatTemplate =
    files.chatTemplate ??
    (typeof config.chat_template === "string" ? config.chat_template : null)
  if (chatTemplate === null) throw new Error("No chat template.")

  const hf = new HfTokenizer(files.tokenizerJson, {
    ...config,
    // The text as the model wrote it: no "cleaning" of spaces before
    // punctuation (Python `tokenizers` does none either).
    clean_up_tokenization_spaces: false
  })
  const template = new Template(chatTemplate)

  const added =
    (files.tokenizerJson as { added_tokens?: Array<AddedTokenJson> })
      .added_tokens ?? []
  const specialIds = new Set(added.filter((t) => t.special).map((t) => t.id))
  let vocabSize = 0
  for (const id of hf.get_vocab(true).values()) {
    vocabSize = Math.max(vocabSize, id + 1)
  }

  // The named special tokens, as the template sees them: transformers
  // passes its special_tokens_map (bos_token, image_token, ...).
  const named: Record<string, string> = { ...DEFAULT_SPECIAL_TOKENS }
  for (const [key, value] of Object.entries(config)) {
    const name = key.endsWith("_token") ? tokenName(value) : undefined
    if (name !== undefined) named[key] = name
  }
  const idOf = (token: string | undefined) =>
    token === undefined ? null : (hf.token_to_id(token) ?? null)
  const required = (key: string) => {
    const id = idOf(named[key])
    if (id === null) throw new Error(`The tokenizer has no ${key}.`)
    return id
  }
  const specialTokens: SpecialTokenIds = {
    bos: required("bos_token"),
    eos: required("eos_token"),
    pad: required("pad_token"),
    endOfText: idOf("<|endoftext|>"),
    imStart: idOf("<|im_start|>"),
    think: idOf("<think>"),
    thinkEnd: idOf("</think>"),
    toolCall: idOf("<tool_call>"),
    toolCallEnd: idOf("</tool_call>")
  }

  const encode = (text: string, options?: { addSpecialTokens?: boolean }) =>
    text === ""
      ? []
      : hf.encode(text, {
          add_special_tokens: options?.addSpecialTokens ?? true
        }).ids

  const decode = (
    ids: ArrayLike<number>,
    options?: { skipSpecialTokens?: boolean }
  ) => {
    // Like Python `tokenizers`, leave out ids without a token (the GGUF
    // build's padding rows past the vocabulary, for one).
    const known: Array<number> = []
    for (let i = 0; i < ids.length; i++) {
      const id = ids[i] as number
      if (hf.id_to_token(id) !== undefined) known.push(id)
    }
    if (known.length === 0) return ""
    return hf.decode(known, {
      skip_special_tokens: options?.skipSpecialTokens ?? false
    })
  }

  /*
   * Hugging Face's DecodeStream: decode the ids since the last piece, with
   * the id before it as context, and hand out what's new once it doesn't
   * end in U+FFFD (a character split across ids, which byte-fallback and
   * byte-level tokens do).
   */
  const createStreamDecoder = (options?: { skipSpecialTokens?: boolean }) => {
    let ids: Array<number> = []
    let prefix = ""
    let prefixIndex = 0
    const run = (all: Array<number>) => decode(all, options)
    return {
      push(id: number) {
        ids.push(id)
        const text = run(ids)
        if (text.length <= prefix.length || text.endsWith("�")) return ""
        const piece = text.startsWith(prefix)
          ? text.slice(prefix.length)
          : text.slice(commonPrefix(text, prefix))
        const nextIndex = ids.length - prefixIndex
        ids = ids.slice(prefixIndex)
        prefix = run(ids)
        prefixIndex = nextIndex
        return piece
      },
      flush() {
        const text = run(ids)
        const piece =
          text.length > prefix.length && text.startsWith(prefix)
            ? text.slice(prefix.length)
            : ""
        ids = []
        prefix = ""
        prefixIndex = 0
        return piece
      },
      reset() {
        ids = []
        prefix = ""
        prefixIndex = 0
      }
    }
  }

  const applyChatTemplate = (
    messages: Array<ChatMessage>,
    options: ChatTemplateOptions = {}
  ) => {
    const {
      tools,
      enableThinking,
      addGenerationPrompt,
      preserveThinking,
      ...extra
    } = options
    // Plain JSON, as transformers' templates get from Python: no undefined
    // values (absent keys instead), no class instances.
    const context = toJson({
      ...named,
      ...extra,
      messages,
      add_generation_prompt: addGenerationPrompt ?? false,
      ...(enableThinking !== undefined && { enable_thinking: enableThinking }),
      ...(preserveThinking !== undefined && {
        preserve_thinking: preserveThinking
      })
    }) as Record<string, unknown>
    // transformers passes `tools=None` when there are none.
    context.tools = tools && tools.length > 0 ? toJson(tools) : null
    return template.render(context)
  }

  return {
    encode,
    decode,
    createStreamDecoder,
    applyChatTemplate,
    encodeChat: (messages, options) =>
      encode(applyChatTemplate(messages, options), { addSpecialTokens: false }),
    tokenToId: (token) => hf.token_to_id(token),
    idToToken: (id) => hf.id_to_token(id),
    isSpecial: (id) => specialIds.has(id),
    specialTokens,
    vocabSize
  }
}

/** A special token's text from tokenizer_config.json (a string or an AddedToken). */
function tokenName(value: unknown) {
  if (typeof value === "string") return value
  if (value && typeof value === "object" && "content" in value) {
    const content = (value as { content: unknown }).content
    if (typeof content === "string") return content
  }
  return undefined
}

function toJson(value: unknown): unknown {
  return value === undefined ? undefined : JSON.parse(JSON.stringify(value))
}

function commonPrefix(a: string, b: string) {
  let i = 0
  while (i < a.length && i < b.length && a[i] === b[i]) i++
  return i
}
