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
export const plain = (value: unknown): value is string => typeof value === "string" && /^[\x20-\x21\x23-\x5b\x5d-\x7e]*$/.test(value)
const ANNOTATIONS = new Set(["description", "title", "$comment", "default", "examples", "deprecated", "readOnly", "writeOnly"])
/** Keywords under which a value's shape isn't the node's own to read. */
const INDIRECT = ["allOf", "not", "if", "then", "else", "dependentSchemas"]

/** What the tokens of a tokenizer write, as far as JSON's structure and these strings need it. */
export type FixedStringTokens = {
  /** Tokens with whitespace in them (see Position.run). */
  spaces: Uint32Array
  /** A token's bytes (HIGH for one that can't be read); nothing for a special token. */
  codes(id: number): ArrayLike<number> | undefined
  /** Tokens with a backslash or a byte past ASCII, as a bitset. */
  escapes: Uint32Array
  /** Those of them that also have a quote: the ones that can enter a string. */
  quoted: Uint32Array
  /** Tokens with a quote: the ones that can open or close a string. */
  quotes: Uint32Array
  /** Each token's length in bytes (0 for a special token), and the longest. */
  lengths: Uint16Array
  longest: number
}

/**
 * A ByteLevel piece's characters as its bytes (GPT-2's bytes_to_unicode
 * reversed): printable bytes are themselves, the others 0x100 onward in
 * order (bytes to space, then DEL to NBSP, then the soft hyphen).
 */
const byteLevelCode = (char: number) =>
  (char >= 0x21 && char <= 0x7e) || (char >= 0xa1 && char <= 0xac) || (char >= 0xae && char <= 0xff) ? char
    : char >= 0x100 && char <= 0x120 ? char - 0x100
      : char >= 0x121 && char <= 0x142 ? 0x7f + char - 0x121
        : char === 0x143 ? 0xad : HIGH

const byteLevel = (decoder: unknown): boolean =>
  isRecord(decoder) && (decoder.type === "ByteLevel" ||
    (decoder.type === "Sequence" && Array.isArray(decoder.decoders) && decoder.decoders.length === 1 && byteLevel(decoder.decoders[0])))

const utf8 = new TextEncoder()

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
      return Array.from(bytes as ArrayLike<number>)
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
      return literal.has(id) ? Array.from(utf8.encode(piece)) : Array.from(piece, (c) => byteLevelCode(c.charCodeAt(0)))
    }
  }
  const words = Math.ceil(vocabSize / 32)
  const escapes = new Uint32Array(words), quoted = new Uint32Array(words), quotes = new Uint32Array(words), spaces = new Uint32Array(words)
  const lengths = new Uint16Array(vocabSize)
  let longest = 0
  const single = new Set<number>()
  for (let id = 0; id < vocabSize; id++) {
    const token = codes(id)
    if (!token || token.length === 0) continue
    lengths[id] = Math.min(token.length, 0xffff)
    longest = Math.max(longest, token.length)
    if (token.length === 1) single.add(token[0])
    let escape = false, quote = false, space = false
    for (let i = 0; i < token.length; i++) {
      if (token[i] === BACKSLASH || token[i] >= HIGH) escape = true
      else if (token[i] === QUOTE) quote = true
      else if (isSpace(token[i])) space = true
    }
    if (space) spaces[id >>> 5] |= 1 << (id & 31)
    if (quote) quotes[id >>> 5] |= 1 << (id & 31)
    if (!escape) continue
    escapes[id >>> 5] |= 1 << (id & 31)
    if (quote) quoted[id >>> 5] |= 1 << (id & 31)
  }
  for (let byte = 0x20; byte <= 0x7e; byte++) if (byte !== BACKSLASH && !single.has(byte)) return null
  return { codes, escapes, quoted, quotes, spaces, lengths, longest }
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

/**
 * How the document walker reads a node (see resolve): as itself, through
 * its one branch that isn't null (a choice with nothing else beside it but
 * annotations), or not at all (a $ref, composition, or a choice of several).
 * What it can't read, it holds nothing in (see prunableSchema).
 */
export function walked(node: unknown): "self" | { branch: number; key: "anyOf" | "oneOf" } | null {
  if (!isRecord(node)) return null
  if ("$ref" in node || INDIRECT.some((key) => key in node)) return null
  for (const key of ["anyOf", "oneOf"] as const) {
    if (!(key in node)) continue
    const branches = node[key]
    if (!Array.isArray(branches) || !Object.keys(node).every((other) => other === key || ANNOTATIONS.has(other))) return null
    const rest = branches.map((branch, i) => ({ branch, i })).filter(({ branch }) => !(isRecord(branch) && branch.type === "null"))
    return rest.length === 1 ? { branch: rest[0].i, key } : null
  }
  return "self"
}

/** An object whose keys are its listed properties, all plain. */
const fixedKeys = (node: Record<string, unknown> | null): boolean =>
  !!node && node.additionalProperties === false && !("patternProperties" in node) &&
  (node.properties === undefined || (isRecord(node.properties) && Object.keys(node.properties).every(plain)))

/**
 * What an ordered object may hold: its keys and which are required, and
 * annotations. A rule over its keys as a whole (minProperties,
 * dependentRequired, …) can need a key the order has already passed, or
 * forbid closing where the order allows nothing else: a valid object left
 * with no way to finish. Those objects' keys go in any order (llguidance
 * still holds them to their rules).
 */
const ORDERABLE = new Set(["type", "properties", "required", "additionalProperties", ...ANNOTATIONS])

/** What an array may hold for the walker to know when it can end: its items' one schema, and bounds on how many. */
const COUNTABLE = new Set(["type", "items", "minItems", "maxItems", ...ANNOTATIONS])

const isSpace = (code: number) => code === 0x20 || code === 0x0a || code === 0x0d || code === 0x09

/**
 * A closed object's keys in llama.cpp's order (see Frame.order), and how
 * many of them are required; undefined unless it has two or more, and
 * nothing beside them but annotations (see ORDERABLE).
 */
function keyOrder(node: Record<string, unknown> | null): { order: Array<string>; required: number } | undefined {
  if (!fixedKeys(node) || !isRecord(node!.properties)) return undefined
  if (!Object.keys(node!).every((key) => ORDERABLE.has(key))) return undefined
  const names = Object.keys(node!.properties)
  if (names.length < 2) return undefined
  const required = Array.isArray(node!.required) ? node!.required : []
  const first = names.filter((name) => required.includes(name))
  return { order: [...first, ...names.filter((name) => !required.includes(name))], required: first.length }
}

/** The keys an ordered object may write next: the next required one, or once those are written, any optional one after the last. */
function nextKeys(frame: Frame): Array<string> | undefined {
  if (!frame.order) return undefined
  const next = frame.next ?? 0
  return next < (frame.required ?? 0) ? [frame.order[next]] : frame.order.slice(next)
}

/** A value whose strings are the const's or the enum's, all plain. */
const fixedValue = (node: Record<string, unknown> | null): boolean => {
  if (!node) return false
  if ("const" in node) return plain(node.const)
  return Array.isArray(node.enum) && node.enum.every((value) => typeof value !== "string" || plain(value))
}

type Frame = {
  object: boolean
  schema: Record<string, unknown> | null
  /**
   * A closed object's keys in the order llama.cpp's grammar writes them
   * (required ones as declared, then optional ones as declared), and the
   * place of the next one; undefined for any other object.
   */
  order?: Array<string>
  required?: number
  next?: number
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
  /** The open value string's minLength and maxLength, when the schema gives them (see filter). */
  min: number | undefined
  max: number | undefined
  /** Characters the open string holds so far (an escape is one, a surrogate pair of \u escapes one). */
  chars = 0
  /** Hex digits left of a \u escape, and its value so far. */
  hex = 0
  hexValue = 0
  /** The last character was a \u escape of a high surrogate: a low one next pairs with it. */
  high = false
  /** A value just ended (its closing quote, brace or bracket), or a number or literal is being written. */
  ended = false
  /**
   * Whitespace followed a value in an object or array that may end there:
   * what comes next is its end, not a comma. llama.cpp's grammar takes
   * whitespace after a value only before the end of what holds it (a
   * comma comes right after the value), so a value that's followed by more
   * is written `"a",` and never `"a"\n  ,` (see closable).
   */
  spaced = false
  /**
   * The whitespace run being written outside strings, when generating:
   * llama.cpp's grammar takes one space, or one or two newlines and then
   * up to 20 spaces or tabs (its `space` rule), nothing else.
   */
  run: { space: boolean; newlines: number; indent: number } | null = null

  /** With `ordered`, closed objects' keys are held to llama.cpp's order (see keyOrder), and whitespace to llama.cpp's grammar (see spaced, run): generation, not checking. */
  constructor(private readonly root: unknown, readonly ordered = false) {}

  clone(): Position {
    const copy = new Position(this.root, this.ordered)
    copy.frames = this.frames.map((frame) => ({ ...frame }))
    copy.inString = this.inString
    copy.escaped = this.escaped
    copy.isKey = this.isKey
    copy.fixed = this.fixed
    copy.min = this.min
    copy.max = this.max
    copy.chars = this.chars
    copy.hex = this.hex
    copy.hexValue = this.hexValue
    copy.high = this.high
    copy.ended = this.ended
    copy.spaced = this.spaced
    copy.run = this.run && { ...this.run }
    return copy
  }

  /**
   * Whether the object or array the last value is in may end now: an
   * ordered object (see keyOrder) once its required keys are written, an
   * array of items once it holds minItems. Undefined where the walker can't
   * tell (other rules over it, or no container): there whitespace and
   * commas are left as llguidance takes them.
   */
  private closable(): boolean | undefined {
    const top = this.frames[this.frames.length - 1]
    if (!top) return undefined
    if (top.object) return top.order ? (top.next ?? 0) >= (top.required ?? 0) : undefined
    const schema = top.schema
    if (!schema || !Object.keys(schema).every((key) => COUNTABLE.has(key))) return undefined
    return top.index + 1 >= (typeof schema.minItems === "number" ? schema.minItems : 0)
  }

  /** The open string holds more characters than its maxLength. */
  private long() {
    return this.max !== undefined && this.chars > this.max
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

  /**
   * Takes one byte. True when it can't be written there: a backslash or a
   * byte past ASCII inside a string the schema fixes, a character past the
   * open string's maxLength, or its closing quote before its minLength.
   */
  feed(code: number): boolean {
    const top = this.frames[this.frames.length - 1]
    if (this.inString) {
      if (this.escaped) {
        this.escaped = false
        if (code === 0x75) {
          // A \u escape: counted once its digits say what it is.
          this.hex = 4
          this.hexValue = 0
          return this.fixed
        }
        this.high = false
        this.chars++
        return this.fixed || this.long()
      }
      if (this.hex > 0) {
        this.hex--
        const digit = code >= 0x30 && code <= 0x39 ? code - 0x30 : (code | 0x20) >= 0x61 && (code | 0x20) <= 0x66 ? (code | 0x20) - 0x57 : 0
        this.hexValue = this.hexValue * 16 + digit
        // At maxLength only a low surrogate may follow a high one: \udc00 to \udfff.
        const pairing = this.high && this.max !== undefined && this.chars >= this.max
        if (pairing && this.hex === 3 && digit !== 0xd) return true
        if (pairing && this.hex === 2 && digit < 0xc) return true
        if (this.hex > 0) return false
        const v = this.hexValue
        if (this.high && v >= 0xdc00 && v <= 0xdfff) {
          // The second half of a pair: one character with the first.
          this.high = false
          return false
        }
        this.high = v >= 0xd800 && v <= 0xdbff
        this.chars++
        return this.long()
      }
      if (code === QUOTE) {
        this.inString = false
        this.ended = !this.isKey
        if (this.isKey && top) {
          // An ordered object's key: one of those it may write next, after which only the later ones are.
          const keys = nextKeys(top)
          if (!keys) return false
          if (top.key === null || !keys.includes(top.key)) return true
          top.next = top.order!.indexOf(top.key) + 1
          return false
        }
        return this.min !== undefined && this.chars < this.min
      }
      if (code === BACKSLASH) {
        // An escape: counted at its next byte (see above). At maxLength, only one
        // that can be the second half of a surrogate pair fits.
        this.escaped = true
        if (this.isKey && top) top.key = null
        return this.fixed || (this.max !== undefined && this.chars >= this.max && !this.high)
      }
      // A character starts at any byte but a UTF-8 continuation byte (10xxxxxx).
      if (code < 0x80 || code >= 0xc0) {
        this.chars++
        this.high = false
      }
      const long = this.long()
      if (code >= HIGH) {
        // (A key written with them isn't looked up: what follows it is left alone.)
        if (this.isKey && top) top.key = null
        return this.fixed || long
      }
      if (this.isKey && top && top.key !== null) {
        top.key += String.fromCharCode(code)
        // Nothing but the start of a key the object may write next.
        const keys = nextKeys(top)
        if (keys && !keys.some((key) => key.startsWith(top.key!))) return true
      }
      return long
    }
    // Whitespace after a value only before the end of what holds it (see spaced): when generating.
    if (this.ordered && isSpace(code)) {
      // llama.cpp's `space`: " " | "\n"{1,2} [ \t]{0,20}.
      const run = (this.run ??= { space: false, newlines: 0, indent: 0 })
      if (code === 0x0a) {
        if (run.space || run.indent > 0 || run.newlines >= 2) return true
        run.newlines++
      } else if (run.newlines > 0 && (code === 0x20 || code === 0x09)) {
        if (run.indent >= 20) return true
        run.indent++
      } else if (code === 0x20 && !run.space) run.space = true
      else return true
      if (this.ended) {
        this.ended = false
        const close = this.closable()
        if (close === false) return true
        this.spaced = close === true
      }
      return false
    }
    if (this.ordered) {
      this.run = null
      if (code === 0x2c && this.spaced) return true
      this.spaced = false
      // A number or literal being written: whitespace after it follows a value.
      this.ended = (code >= 0x30 && code <= 0x39) || ((code | 0x20) >= 0x61 && (code | 0x20) <= 0x7a) || code === 0x2e || code === 0x2b || code === 0x2d
    }
    switch (code) {
      case QUOTE:
        this.inString = true
        this.isKey = !!top && top.object && top.expectKey
        this.chars = 0
        this.hex = 0
        this.high = false
        this.min = undefined
        this.max = undefined
        if (this.isKey) {
          top.key = ""
          this.fixed = fixedKeys(top.schema)
        } else {
          const schema = this.valueSchema()
          this.fixed = fixedValue(schema)
          if (schema) {
            this.min = typeof schema.minLength === "number" ? schema.minLength : undefined
            this.max = typeof schema.maxLength === "number" ? schema.maxLength : undefined
          }
        }
        break
      case 0x7b: {
        const schema = this.valueSchema()
        const order = this.ordered ? keyOrder(schema) : undefined
        this.frames.push({ object: true, schema, key: null, expectKey: true, index: 0, ...(order && { order: order.order, required: order.required, next: 0 }) })
        break
      }
      case 0x5b:
        this.frames.push({ object: false, schema: this.valueSchema(), key: null, expectKey: false, index: 0 })
        break
      case 0x7d:
      case 0x5d:
        this.frames.pop()
        this.ended = this.ordered
        break
      case 0x2c:
        if (top?.object) {
          top.expectKey = true
          // No key may follow the last one an ordered object may write: no comma either.
          const keys = nextKeys(top)
          if (keys && keys.length === 0) return true
        } else if (top) top.index++
        break
      case 0x3a:
        if (top) top.expectKey = false
        break
    }
    return false
  }
}

/** Whether a schema has a closed object of two or more plain keys anywhere (whose order is held: see keyOrder). */
export function orderedKeys(schema: unknown, depth = 0): boolean {
  if (depth > 64) return false
  if (Array.isArray(schema)) return schema.some((item) => orderedKeys(item, depth + 1))
  if (!isRecord(schema)) return false
  if (keyOrder(schema)) return true
  return Object.entries(schema).some(([key, value]) => key !== "enum" && key !== "const" && key !== "default" && key !== "examples" && orderedKeys(value, depth + 1))
}

/** Whether a schema has a minLength or maxLength anywhere. */
export function lengthBounds(schema: unknown, depth = 0): boolean {
  if (depth > 64) return false
  if (Array.isArray(schema)) return schema.some((item) => lengthBounds(item, depth + 1))
  if (!isRecord(schema)) return false
  if (typeof schema.minLength === "number" || typeof schema.maxLength === "number") return true
  return Object.entries(schema).some(([key, value]) => key !== "enum" && key !== "const" && key !== "default" && key !== "examples" && lengthBounds(value, depth + 1))
}

/** Follows a generation's tokens and takes out of its masks what a fixed string can't hold, and what a bounded string can't. */
export class FixedStrings {
  private position: Position

  constructor(
    private readonly tokens: FixedStringTokens,
    schema: unknown,
    /** Whether closed objects' keys are held to their order (see keyOrder). */
    ordered = false,
    position?: Position,
    /** Whether the schema has string lengths, or keys in an order, to hold (see lengthBounds, keyOrder). */
    private readonly bounded = schema !== undefined && (lengthBounds(schema) || (ordered && orderedKeys(schema)))
  ) {
    this.position = position ?? new Position(schema, ordered)
  }

  /**
   * The mask without the tokens that can't be written next: one that would
   * put a backslash or a byte past ASCII into a fixed string, run a string
   * past its maxLength or close it before its minLength (see feed). The mask
   * itself when there are none.
   */
  filter(mask: Uint32Array): Uint32Array {
    const p = this.position
    const { escapes, quoted, quotes, spaces, lengths } = this.tokens
    // Inside a fixed string any escape does. Elsewhere (a free string too) only one
    // that has a quote can get into a fixed string, or into a bounded one and break its bounds.
    const suspects = p.inString && p.fixed ? escapes : this.bounded ? quotes : quoted
    // Inside a bounded string: a quote before minLength, and near maxLength, any
    // token with more bytes than characters left (a character is at least one byte).
    const short = p.inString && p.min !== undefined && p.chars < p.min
    const left = p.inString && p.max !== undefined ? p.max - p.chars : Infinity
    // Inside an ordered object's key, any token may start a key it may not write yet; between
    // its keys and values (a few tokens: whitespace, comma, brace, quote), any may add a key after its last.
    const top = p.frames[p.frames.length - 1]
    // After a value (or in a string that ends in one), when generating: whitespace and commas (see Position.spaced).
    const spacing = p.ordered && (p.inString ? !p.isKey : p.ended || p.spaced)
    const ordering = (!!top?.order && (p.inString ? p.isKey : true)) || (!p.inString && spacing)
    const closing = spacing && p.inString
    // Whitespace runs held to llama.cpp's rule (see Position.run): a token with whitespace in it, outside strings.
    const runs = p.ordered
    const near = left < this.tokens.longest || ordering
    let out: Uint32Array | undefined
    for (let word = 0; word < mask.length; word++) {
      let bits = mask[word] & ((suspects ? suspects[word] : 0) | (short ? quotes[word] : 0) | (closing ? quotes[word] : 0) | (runs ? (p.inString ? quotes[word] & spaces[word] : spaces[word]) : 0) | (near ? ~0 : 0))
      while (bits !== 0) {
        const bit = bits & -bits
        bits ^= bit
        const token = (word << 5) + 31 - Math.clz32(bit)
        // (Near maxLength, a token no longer than what's left can't run past it, unless it's suspect anyway.)
        if (near && !ordering && lengths[token] <= left && !((suspects ? suspects[word] : 0) & bit) && !(((short || closing) ? quotes[word] : 0) & bit) && !((runs ? spaces[word] : 0) & bit)) continue
        if (!this.spoils(token)) continue
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
    return new FixedStrings(this.tokens, undefined, false, this.position.clone(), this.bounded)
  }
}
