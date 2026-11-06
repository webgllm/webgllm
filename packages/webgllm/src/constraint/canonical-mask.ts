/** A format restriction for a deliberately small, finite JSON Schema subset.
 * Every permitted value retains a canonical spelling; escaped spellings are
 * excluded because the matcher can admit impossible escape/UTF-8 prefixes.
 */
const annotations = new Set(["description", "title", "$comment", "default", "examples"])
const printableLiteral = (value: unknown): value is string =>
  typeof value === "string" && /^[\x20-\x21\x23-\x5b\x5d-\x7e]*$/.test(value)
const record = (value: unknown): value is Record<string, unknown> =>
  value !== null && typeof value === "object" && !Array.isArray(value)
const keysWithin = (value: Record<string, unknown>, keys: string[]) =>
  Object.keys(value).every(key => keys.includes(key) || annotations.has(key))

/** Unknown assertions, Unicode and strings requiring escapes use the normal backend. */
export function canonicalAsciiObject(schema: unknown): boolean {
  if (!record(schema) || schema.type !== "object" || schema.additionalProperties !== false ||
    !keysWithin(schema, ["type", "properties", "required", "additionalProperties", "$schema"])) return false
  const properties = schema.properties ?? {}
  if (!record(properties)) return false
  const names = Object.keys(properties)
  if (schema.required !== undefined && (!Array.isArray(schema.required) ||
    schema.required.some(name => typeof name !== "string" || !names.includes(name)))) return false
  return Object.entries(properties).every(([name, property]) => {
    if (!printableLiteral(name) || !record(property) || !keysWithin(property, ["type", "enum", "const"]) ||
      (property.type !== undefined && property.type !== "string")) return false
    if ("enum" in property && "const" in property) return false
    if ("const" in property) return printableLiteral(property.const)
    return Array.isArray(property.enum) && property.enum.length > 0 && property.enum.every(printableLiteral)
  })
}

const canonicalByte = (byte: number) => Number.isInteger(byte) && byte >= 32 && byte <= 126 && byte !== 92
const byteLevelPiece = (piece: string) => /^[\x21-\x5b\x5d-\x7e\u0120]*$/.test(piece)
const literalPiece = (piece: string) => /^[\x20-\x5b\x5d-\x7e]*$/.test(piece)
const exactByteLevel = (value: unknown): boolean => record(value) && value.type === "ByteLevel" &&
  Object.entries(value).every(([key, entry]) => key === "type" ||
    (["add_prefix_space", "trim_offsets", "use_regex"].includes(key) && typeof entry === "boolean"))
const masks = new WeakMap<object, Map<string, Uint32Array | undefined>>()

/** Exact explicit bytes or HF ByteLevel metadata only; never independently decode tokens.
 * Requiring a normal one-byte token for every canonical byte guarantees that all
 * permitted values remain representable, even with an unusual model vocabulary.
 */
export function canonicalTokenMask(tokenizer: object, vocabSize: number, eos: number): Uint32Array | undefined {
  let cached = masks.get(tokenizer)
  if (!cached) { cached = new Map(); masks.set(tokenizer, cached) }
  const key = `${vocabSize}:${eos}`
  if (cached.has(key)) return cached.get(key)
  const mask = buildMask(tokenizer, vocabSize, eos)
  cached.set(key, mask)
  return mask
}

function buildMask(tokenizer: object, vocabSize: number, eos: number): Uint32Array | undefined {
  const source = tokenizer as Record<string, unknown>
  const mask = new Uint32Array(Math.ceil(vocabSize / 32))
  const singles = new Map<number, number>()
  const special = new Set([eos])
  const validId = (id: unknown): id is number => Number.isSafeInteger(id) && (id as number) >= 0 && (id as number) < vocabSize
  const set = (id: unknown) => { if (validId(id)) mask[id >>> 5] |= 1 << (id & 31) }
  for (const name of ["special_token_ids", "specialTokenIds", "all_special_ids", "allSpecialIds"]) {
    const ids = source[name]
    if (Array.isArray(ids)) for (const id of ids) if (validId(id)) special.add(id)
  }
  for (const name of ["bos_token_id", "bosTokenId", "pad_token_id", "padTokenId", "eos_token_id", "eosTokenId"]) {
    const id = source[name]
    if (validId(id)) special.add(id)
  }
  if (Array.isArray(source.tokens)) {
    for (let id = 0; id < source.tokens.length; id++) {
      const bytes: unknown = source.tokens[id]
      if (!Array.isArray(bytes) && !(bytes instanceof Uint8Array)) return undefined
      if (bytes.length > 0 && Array.from(bytes as ArrayLike<number>).every(canonicalByte)) set(id)
      if (validId(id) && bytes.length === 1 && canonicalByte(bytes[0] as number)) singles.set(id, bytes[0] as number)
    }
  } else {
    const json = source.tokenizerJSON
    if (!record(json) || !exactByteLevel(json.decoder) ||
      !record(json.model) || !record(json.model.vocab)) return undefined
    for (const [piece, id] of Object.entries(json.model.vocab)) {
      if (piece.length > 0 && byteLevelPiece(piece)) set(id)
      if (validId(id) && piece.length === 1 && byteLevelPiece(piece)) singles.set(id, piece === "Ġ" ? 32 : piece.charCodeAt(0))
    }
    if (Array.isArray(json.added_tokens)) for (const token of json.added_tokens) {
      if (!record(token) || !validId(token.id)) continue
      // Added tokens replace vocabulary bytes, even when they are not special.
      mask[token.id >>> 5] &= ~(1 << (token.id & 31))
      singles.delete(token.id)
      if (typeof token.content === "string") {
        if (token.content.length > 0 && literalPiece(token.content)) set(token.id)
        if (token.content.length === 1 && canonicalByte(token.content.charCodeAt(0))) singles.set(token.id, token.content.charCodeAt(0))
      }
      if (token.special) special.add(token.id)
    }
  }
  const covered = new Set([...singles].filter(([id]) => !special.has(id)).map(([, byte]) => byte))
  for (let byte = 32; byte <= 126; byte++) if (canonicalByte(byte) && !covered.has(byte)) return undefined
  set(eos)
  return mask
}
