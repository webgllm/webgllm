/** JSON's whitespace bytes: space, newline, carriage return, tab. */
const WHITESPACE = new Set([32, 10, 13, 9])
/** The same bytes as HF ByteLevel vocabularies spell them (Ġ, Ċ, č, ĉ). */
const BYTE_LEVEL_WHITESPACE = /^[ĠĊčĉ]+$/

const cache = new WeakMap<object, readonly number[] | null>()

const record = (value: unknown): value is Record<string, unknown> =>
  value !== null && typeof value === "object" && !Array.isArray(value)

/** A ByteLevel decoder, alone or as a Sequence's only step. */
const byteLevel = (decoder: unknown): boolean =>
  record(decoder) && (decoder.type === "ByteLevel" ||
    (decoder.type === "Sequence" && Array.isArray(decoder.decoders) && decoder.decoders.length === 1 && byteLevel(decoder.decoders[0])))

/**
 * The ids of tokens that are only JSON whitespace, from the tokenizer's exact
 * bytes (or its HF ByteLevel vocabulary); null when they can't be read.
 */
export function whitespaceTokenIds(tokenizer: object): readonly number[] | null {
  if (cache.has(tokenizer)) return cache.get(tokenizer)!
  const ids = read(tokenizer as Record<string, unknown>)
  cache.set(tokenizer, ids)
  return ids
}

function read(source: Record<string, unknown>): number[] | null {
  const special = new Set<number>()
  for (const name of ["special_token_ids", "specialTokenIds", "all_special_ids", "allSpecialIds"]) {
    const list = source[name]
    if (Array.isArray(list)) for (const id of list) if (Number.isSafeInteger(id)) special.add(id)
  }
  const ids: number[] = []
  if (Array.isArray(source.tokens)) {
    source.tokens.forEach((bytes: unknown, id) => {
      if ((Array.isArray(bytes) || bytes instanceof Uint8Array) && bytes.length > 0 && !special.has(id) &&
        Array.from(bytes as ArrayLike<number>).every((byte) => WHITESPACE.has(byte))) ids.push(id)
    })
    return ids
  }
  const json = source.tokenizerJSON
  if (!record(json) || !byteLevel(json.decoder) || !record(json.model) || !record(json.model.vocab)) return null
  const added = new Map<number, Record<string, unknown>>()
  if (Array.isArray(json.added_tokens)) for (const token of json.added_tokens) if (record(token) && Number.isSafeInteger(token.id)) added.set(token.id as number, token)
  for (const [piece, id] of Object.entries(json.model.vocab)) {
    // Added tokens replace vocabulary pieces.
    if (typeof id === "number" && !added.has(id) && !special.has(id) && BYTE_LEVEL_WHITESPACE.test(piece)) ids.push(id)
  }
  for (const [id, token] of added) {
    if (!token.special && !special.has(id) && typeof token.content === "string" && token.content.length > 0 &&
      [...token.content].every((c) => WHITESPACE.has(c.charCodeAt(0)))) ids.push(id)
  }
  return ids
}
