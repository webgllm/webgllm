/** JSON's whitespace bytes: space, newline, carriage return, tab. */
const WHITESPACE = new Set([32, 10, 13, 9])
/** The same bytes as HF ByteLevel vocabularies spell them (Ġ, Ċ, č, ĉ). */
const BYTE_LEVEL_WHITESPACE = /^[ĠĊčĉ]+$/
const QUOTE = 34
const BACKSLASH = 92

/** Where a JSON lexer is: outside strings, inside one, or just after a backslash in one. */
export type StringState = 0 | 1 | 2
/** A token's effect on StringState: the state after it from each state, two bits each. */
const IDENTITY = 0 | 1 << 2 | 2 << 4

export type JsonTokens = {
  /** Tokens that are only JSON whitespace. */
  whitespace: readonly number[]
  /** How tokens with a quote or backslash move StringState; every other token leaves it. */
  strings: ReadonlyMap<number, number>
}

/** The state after a token, from the state before it. */
export const nextStringState = (tokens: JsonTokens, token: number, state: StringState): StringState =>
  ((tokens.strings.get(token) ?? IDENTITY) >> state * 2 & 3) as StringState

/** A byte (or ByteLevel character) code's effect from each state. */
function transitions(codes: Iterable<number>): number {
  let packed = 0
  for (let from = 0; from < 3; from++) {
    let state = from
    for (const code of codes) {
      if (state === 2) state = 1
      else if (code === QUOTE) state = state === 0 ? 1 : 0
      else if (code === BACKSLASH && state === 1) state = 2
    }
    packed |= state << from * 2
  }
  return packed
}

const cache = new WeakMap<object, JsonTokens | null>()

const record = (value: unknown): value is Record<string, unknown> =>
  value !== null && typeof value === "object" && !Array.isArray(value)

/** A ByteLevel decoder, alone or as a Sequence's only step. */
const byteLevel = (decoder: unknown): boolean =>
  record(decoder) && (decoder.type === "ByteLevel" ||
    (decoder.type === "Sequence" && Array.isArray(decoder.decoders) && decoder.decoders.length === 1 && byteLevel(decoder.decoders[0])))

/**
 * The tokens that matter to JSON layout, from the tokenizer's exact bytes
 * (or its HF ByteLevel vocabulary); null when they can't be read.
 */
export function jsonTokens(tokenizer: object): JsonTokens | null {
  if (cache.has(tokenizer)) return cache.get(tokenizer)!
  const tokens = read(tokenizer as Record<string, unknown>)
  cache.set(tokenizer, tokens)
  return tokens
}

function read(source: Record<string, unknown>): JsonTokens | null {
  const special = new Set<number>()
  for (const name of ["special_token_ids", "specialTokenIds", "all_special_ids", "allSpecialIds"]) {
    const list = source[name]
    if (Array.isArray(list)) for (const id of list) if (Number.isSafeInteger(id)) special.add(id)
  }
  const whitespace: number[] = []
  const strings = new Map<number, number>()
  // Quote and backslash are single bytes, and ByteLevel spells them as themselves.
  const add = (id: number, codes: number[], blank: boolean) => {
    if (codes.length === 0 || special.has(id)) return
    if (blank) whitespace.push(id)
    else if (codes.includes(QUOTE) || codes.includes(BACKSLASH)) strings.set(id, transitions(codes))
  }
  if (Array.isArray(source.tokens)) {
    source.tokens.forEach((bytes: unknown, id) => {
      if (!Array.isArray(bytes) && !(bytes instanceof Uint8Array)) return
      const codes = Array.from(bytes as ArrayLike<number>)
      add(id, codes, codes.every((byte) => WHITESPACE.has(byte)))
    })
    return { whitespace, strings }
  }
  const json = source.tokenizerJSON
  if (!record(json) || !byteLevel(json.decoder) || !record(json.model) || !record(json.model.vocab)) return null
  const added = new Map<number, Record<string, unknown>>()
  if (Array.isArray(json.added_tokens)) for (const token of json.added_tokens) if (record(token) && Number.isSafeInteger(token.id)) added.set(token.id as number, token)
  const codes = (text: string) => Array.from(text, (c) => c.charCodeAt(0))
  for (const [piece, id] of Object.entries(json.model.vocab)) {
    // Added tokens replace vocabulary pieces.
    if (typeof id === "number" && !added.has(id)) add(id, codes(piece), BYTE_LEVEL_WHITESPACE.test(piece))
  }
  for (const [id, token] of added) {
    if (token.special || typeof token.content !== "string") continue
    const text = codes(token.content)
    add(id, text, text.every((c) => WHITESPACE.has(c)))
  }
  return { whitespace, strings }
}
