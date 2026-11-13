/*
 * Strings the schema fixes are held to their one plain spelling.
 *
 * In llguidance 0.2.1 a string's mask takes any JSON escape and any first
 * byte of a UTF-8 character, also where the schema leaves one text only: a
 * closed object's key, an enum's or const's value. After `{"cut` of the key
 * `cuts` it allows `s`, 31 tokens with a backslash and 592 partial
 * characters. None of those can become `cuts`: an escape is then taken for
 * ever (`{"cut\"\"\"\"…` to the output limit) and a partial character
 * dead-ends on the next token. A model reaches them whenever the token it
 * wants is refused (the closing quote after `cut`, a `}` before a required
 * key it skipped), as they are what is left that looks like it.
 *
 * So the bytes written are followed through the document and its schema,
 * and inside a key of a closed object with plain names, or a value whose
 * enum or const strings are plain, tokens that put a backslash or a byte
 * past ASCII there are taken out of the mask. Plain is printable ASCII
 * needing no escape: such a text has one spelling without them, which stays
 * allowed. Every other string is left to llguidance: free values keep their
 * escapes and characters, open objects their keys. Where the schema can't be
 * followed plainly (allOf, a choice of several shapes), nothing is fixed.
 */

const QUOTE = 34
const BACKSLASH = 92
/** Stands for any byte past ASCII. */
const HIGH = 0x80

const isRecord = (value: unknown): value is Record<string, unknown> =>
  value !== null && typeof value === "object" && !Array.isArray(value)
/** Printable ASCII that JSON writes as itself: one spelling without escapes. */
const plain = (value: unknown): value is string => typeof value === "string" && /^[\x20-\x21\x23-\x5b\x5d-\x7e]*$/.test(value)
const ANNOTATIONS = new Set(["description", "title", "$comment", "default", "examples", "deprecated", "readOnly", "writeOnly"])
/** Keywords under which a value's shape isn't the node's own to read. */
const INDIRECT = ["allOf", "not", "if", "then", "else", "dependentSchemas"]

/** What the tokens of a tokenizer write, as far as JSON's structure and these strings need it. */
export type FixedStringTokens = {
  /** A token's bytes, with every byte past ASCII as HIGH; nothing for a special token. */
  codes(id: number): ArrayLike<number> | undefined
  /** Tokens with a backslash or a byte past ASCII, as a bitset. */
  escapes: Uint32Array
  /** Those of them that also have a quote: the ones that can enter a string. */
  quoted: Uint32Array
}

/** A ByteLevel piece's characters as bytes: 0x21 to 0x7e are themselves, 0x100 to 0x121 the bytes up to space and DEL. */
const byteLevelCode = (char: number) => (char >= 0x21 && char <= 0x7e ? char : char >= 0x100 && char <= 0x120 ? char - 0x100 : char === 0x121 ? 0x7f : HIGH)

const byteLevel = (decoder: unknown): boolean =>
  isRecord(decoder) && (decoder.type === "ByteLevel" ||
    (decoder.type === "Sequence" && Array.isArray(decoder.decoders) && decoder.decoders.length === 1 && byteLevel(decoder.decoders[0])))

const cache = new WeakMap<object, Map<number, FixedStringTokens | null>>()

/**
 * The tokenizer's tokens, from its exact bytes or its HF ByteLevel
 * vocabulary; null when they can't be read, or when some printable ASCII
 * character has no token of its own (the plain spelling must stay writable
 * whatever tokens are taken out).
 */
export function fixedStringTokens(tokenizer: object, vocabSize: number): FixedStringTokens | null {
  let sizes = cache.get(tokenizer)
  if (!sizes) cache.set(tokenizer, sizes = new Map())
  if (!sizes.has(vocabSize)) sizes.set(vocabSize, read(tokenizer as Record<string, unknown>, vocabSize))
  return sizes.get(vocabSize)!
}

function read(source: Record<string, unknown>, vocabSize: number): FixedStringTokens | null {
  const special = new Set<number>()
  for (const name of ["special_token_ids", "specialTokenIds", "all_special_ids", "allSpecialIds"]) {
    const list = source[name]
    if (Array.isArray(list)) for (const id of list) if (Number.isSafeInteger(id)) special.add(id)
  }
  for (const name of ["eos_token_id", "eosTokenId", "bos_token_id", "bosTokenId", "pad_token_id", "padTokenId"]) {
    const id = source[name]
    if (Number.isSafeInteger(id)) special.add(id as number)
  }
  let codes: (id: number) => ArrayLike<number> | undefined
  if (Array.isArray(source.tokens)) {
    const tokens = source.tokens as unknown[]
    codes = (id) => {
      const bytes = tokens[id]
      if (special.has(id) || (!Array.isArray(bytes) && !(bytes instanceof Uint8Array))) return undefined
      return Array.from(bytes as ArrayLike<number>, (byte) => (byte < HIGH ? byte : HIGH))
    }
  } else {
    const json = source.tokenizerJSON
    if (!isRecord(json) || !byteLevel(json.decoder) || !isRecord(json.model) || !isRecord(json.model.vocab)) return null
    // Each id's piece, and whether it is a ByteLevel one (an added token's content is its text as it is).
    const pieces: Array<string | undefined> = []
    const literal = new Set<number>()
    for (const [piece, id] of Object.entries(json.model.vocab)) if (typeof id === "number") pieces[id] = piece
    if (Array.isArray(json.added_tokens)) for (const token of json.added_tokens) {
      if (!isRecord(token) || !Number.isSafeInteger(token.id)) continue
      const id = token.id as number
      // Added tokens replace vocabulary pieces.
      pieces[id] = typeof token.content === "string" ? token.content : undefined
      literal.add(id)
      if (token.special) special.add(id)
    }
    codes = (id) => {
      const piece = pieces[id]
      if (piece === undefined || special.has(id)) return undefined
      return literal.has(id)
        ? Array.from(piece, (c) => (c.charCodeAt(0) < HIGH ? c.charCodeAt(0) : HIGH))
        : Array.from(piece, (c) => byteLevelCode(c.charCodeAt(0)))
    }
  }
  const words = Math.ceil(vocabSize / 32)
  const escapes = new Uint32Array(words), quoted = new Uint32Array(words)
  const single = new Set<number>()
  for (let id = 0; id < vocabSize; id++) {
    const token = codes(id)
    if (!token || token.length === 0) continue
    if (token.length === 1) single.add(token[0])
    let escape = false, quote = false
    for (let i = 0; i < token.length; i++) {
      if (token[i] === BACKSLASH || token[i] >= HIGH) escape = true
      else if (token[i] === QUOTE) quote = true
    }
    if (!escape) continue
    escapes[id >>> 5] |= 1 << (id & 31)
    if (quote) quoted[id >>> 5] |= 1 << (id & 31)
  }
  for (let byte = 0x20; byte <= 0x7e; byte++) if (byte !== BACKSLASH && !single.has(byte)) return null
  return { codes, escapes, quoted }
}

/** A JSON pointer's target in the schema (`#/$defs/name`), or undefined. */
function pointer(root: unknown, ref: string): unknown {
  if (ref === "#") return root
  if (!ref.startsWith("#/")) return undefined
  let node = root
  for (const part of ref.slice(2).split("/")) {
    const key = decodeURIComponent(part).replace(/~1/g, "/").replace(/~0/g, "~")
    if (isRecord(node) && Object.hasOwn(node, key)) node = node[key]
    else if (Array.isArray(node) && /^\d+$/.test(key)) node = node[Number(key)]
    else return undefined
  }
  return node
}

/**
 * The one schema a value there has to satisfy, when it can be read
 * plainly: through local references and through a choice whose other
 * branches are null. Null where it can't.
 */
function resolve(root: unknown, node: unknown, depth = 0): Record<string, unknown> | null {
  if (!isRecord(node) || depth > 32) return null
  const alone = (key: string) => Object.keys(node).every((other) => other === key || ANNOTATIONS.has(other))
  if ("$ref" in node) return typeof node.$ref === "string" && alone("$ref") ? resolve(root, pointer(root, node.$ref), depth + 1) : null
  for (const key of ["anyOf", "oneOf"]) {
    if (!(key in node)) continue
    const branches = node[key]
    if (!Array.isArray(branches) || !alone(key)) return null
    const rest = branches.filter((branch) => !(isRecord(branch) && branch.type === "null"))
    return rest.length === 1 ? resolve(root, rest[0], depth + 1) : null
  }
  return INDIRECT.some((key) => key in node) ? null : node
}

/** An object whose keys are its listed properties, all plain. */
const fixedKeys = (node: Record<string, unknown> | null): boolean =>
  !!node && node.additionalProperties === false && !("patternProperties" in node) &&
  (node.properties === undefined || (isRecord(node.properties) && Object.keys(node.properties).every(plain)))

/** A value whose strings are the const's or the enum's, all plain. */
const fixedValue = (node: Record<string, unknown> | null): boolean => {
  if (!node) return false
  if ("const" in node) return plain(node.const)
  return Array.isArray(node.enum) && node.enum.every((value) => typeof value !== "string" || plain(value))
}

type Frame = {
  object: boolean
  schema: Record<string, unknown> | null
  /** An object's key being written or last written; null when it isn't plain text. */
  key: string | null
  /** An object's next string is a key. */
  expectKey: boolean
  /** An array's item number. */
  index: number
}

/** Where the bytes so far end in the document and in its schema. */
class Position {
  frames: Frame[] = []
  inString = false
  escaped = false
  /** The open string is a key. */
  isKey = false
  /** The open string is one the schema fixes. */
  fixed = false

  constructor(private readonly root: unknown) {}

  clone(): Position {
    const copy = new Position(this.root)
    copy.frames = this.frames.map((frame) => ({ ...frame }))
    copy.inString = this.inString
    copy.escaped = this.escaped
    copy.isKey = this.isKey
    copy.fixed = this.fixed
    return copy
  }

  /** The schema of the value that comes next. */
  private valueSchema(): Record<string, unknown> | null {
    const top = this.frames[this.frames.length - 1]
    if (!top) return resolve(this.root, this.root)
    if (!top.schema) return null
    if (top.object) {
      const properties = top.schema.properties
      return top.key !== null && isRecord(properties) && Object.hasOwn(properties, top.key) ? resolve(this.root, properties[top.key]) : null
    }
    const prefix = top.schema.prefixItems
    if (Array.isArray(prefix) && top.index < prefix.length) return resolve(this.root, prefix[top.index])
    return "items" in top.schema ? resolve(this.root, top.schema.items) : null
  }

  /** Takes one byte. True when it is a backslash or a byte past ASCII inside a string the schema fixes. */
  feed(code: number): boolean {
    const top = this.frames[this.frames.length - 1]
    if (this.inString) {
      if (this.escaped) {
        this.escaped = false
        return this.fixed
      }
      if (code === QUOTE) {
        this.inString = false
        return false
      }
      if (code === BACKSLASH || code >= HIGH) {
        this.escaped = code === BACKSLASH
        // (A key written with them isn't looked up: what follows it is left alone.)
        if (this.isKey && top) top.key = null
        return this.fixed
      }
      if (this.isKey && top && top.key !== null) top.key += String.fromCharCode(code)
      return false
    }
    switch (code) {
      case QUOTE:
        this.inString = true
        this.isKey = !!top && top.object && top.expectKey
        if (this.isKey) {
          top.key = ""
          this.fixed = fixedKeys(top.schema)
        } else this.fixed = fixedValue(this.valueSchema())
        break
      case 0x7b:
        this.frames.push({ object: true, schema: this.valueSchema(), key: null, expectKey: true, index: 0 })
        break
      case 0x5b:
        this.frames.push({ object: false, schema: this.valueSchema(), key: null, expectKey: false, index: 0 })
        break
      case 0x7d:
      case 0x5d:
        this.frames.pop()
        break
      case 0x2c:
        if (top?.object) top.expectKey = true
        else if (top) top.index++
        break
      case 0x3a:
        if (top) top.expectKey = false
        break
    }
    return false
  }
}

/** Follows a generation's tokens and takes out of its masks what a fixed string can't hold. */
export class FixedStrings {
  private position: Position

  constructor(private readonly tokens: FixedStringTokens, schema: unknown, position?: Position) {
    this.position = position ?? new Position(schema)
  }

  /** The mask without the tokens that would write a backslash or a byte past ASCII into a fixed string; the mask itself when there are none. */
  filter(mask: Uint32Array): Uint32Array {
    // Inside a fixed string any of them does. Elsewhere only one that has a quote can get into one.
    const suspects = this.position.inString && this.position.fixed ? this.tokens.escapes : this.tokens.quoted
    let out: Uint32Array | undefined
    for (let word = 0; word < mask.length; word++) {
      let bits = mask[word] & suspects[word]
      while (bits !== 0) {
        const bit = bits & -bits
        bits ^= bit
        if (!this.spoils((word << 5) + 31 - Math.clz32(bit))) continue
        out ??= mask.slice()
        out[word] &= ~bit
      }
    }
    return out ?? mask
  }

  private spoils(token: number): boolean {
    const codes = this.tokens.codes(token)
    if (!codes) return false
    const position = this.position.clone()
    for (let i = 0; i < codes.length; i++) if (position.feed(codes[i])) return true
    return false
  }

  accept(token: number) {
    const codes = this.tokens.codes(token)
    if (codes) for (let i = 0; i < codes.length; i++) this.position.feed(codes[i])
  }

  clone(): FixedStrings {
    return new FixedStrings(this.tokens, undefined, this.position.clone())
  }
}
