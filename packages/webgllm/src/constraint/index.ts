import type { JsonGuidanceRuntime } from "llguidance/json-schema"
import { canonicalAsciiObject, canonicalTokenMask } from "./canonical-mask.js"
import { whitespaceTokenIds } from "./whitespace.js"

/** Exact model tokenizer, not token strings decoded independently. */
export type ConstraintTokenizer = object
export type ConstraintOptions = {
  schema: unknown
  tokenizer: ConstraintTokenizer
  /** Override when a runtime keeps EOS on its model config rather than tokenizer. */
  eosTokenId?: number
  maxTokens?: number
  /**
   * How the JSON may be laid out. "model" (the default) takes whitespace
   * where models write it, as they write it: held to compact JSON, a model
   * that writes `"key": "value"` (LFM3) gets no ` "` token after the colon
   * or a comma, picks `":` instead and starts a value (or, in an open
   * object, a key) with a stray colon. "compact" is JSON.stringify's
   * layout, for checking parsed values.
   */
  layout?: "model" | "compact"
}

/**
 * Whitespace-only tokens in a row before whitespace is refused until the
 * JSON goes on: a laid-out document needs a few between two tokens, and a
 * model stuck writing whitespace would otherwise run to its output limit.
 */
export const MAX_WHITESPACE_RUN = 8

export class StructuredOutputError extends Error {
  constructor(readonly code: "unsupported-schema" | "invalid-tokenizer" | "constraint-dead-end" | "constraint-violation" | "schema-mismatch", message: string, options?: ErrorOptions) {
    super(message, options)
    this.name = "StructuredOutputError"
  }
}

export type Constraint = {
  readonly vocabSize: number
  readonly eosTokenId: number
  /** True when the current bytes form an accepted complete schema-valid value. */
  readonly complete: boolean
  /** A bitset, with token id's bit in word id >>> 5. Never mutate this array. */
  computeMask(): Uint32Array
  accept(token: number): void
  clone(): Constraint
  dispose(): void
}

export function tokenAllowed(mask: Uint32Array, token: number): boolean {
  return Number.isSafeInteger(token) && token >= 0 && token >>> 5 < mask.length &&
    (mask[token >>> 5] & (1 << (token & 31))) !== 0
}

type Interpreter = ReturnType<JsonGuidanceRuntime["createInterpreter"]>
let runtimePromise: Promise<JsonGuidanceRuntime> | undefined
const loadRuntime = () => runtimePromise ??= import("llguidance/json-schema").then(module => module.loadJsonGuidance())
const eosOverrides = new WeakMap<object, Map<number, object>>()
function withEos(tokenizer: ConstraintTokenizer, eos: number | undefined): object {
  if (eos === undefined) return tokenizer
  let byEos = eosOverrides.get(tokenizer)
  if (!byEos) { byEos = new Map(); eosOverrides.set(tokenizer, byEos) }
  let source = byEos.get(eos)
  if (!source) {
    source = new Proxy(tokenizer, { get(target, key, receiver) { return key === "eos_token_id" || key === "eosTokenId" ? eos : Reflect.get(target, key, receiver) } })
    byEos.set(eos, source)
  }
  return source
}

class JsonConstraint implements Constraint {
  private mask: Uint32Array | undefined
  private accepting = false
  private disposed = false
  constructor(
    private readonly interpreter: Interpreter,
    readonly vocabSize: number,
    readonly eosTokenId: number,
    private readonly endAtAcceptance: boolean,
    private readonly canonicalMask?: Uint32Array,
    /** The full schema, when the mask runs on a looser one (see prunableSchema). */
    private readonly shadow?: Interpreter,
    /** Whitespace-only tokens, when the layout takes whitespace (see MAX_WHITESPACE_RUN). */
    private readonly whitespace?: Uint32Array
  ) {}

  /** Whitespace-only tokens accepted in a row. */
  private whitespaceRun = 0

  /**
   * Whether the full schema takes the value as it stands. The shadow only
   * commits tokens and checks EOS as a single candidate: its whole mask is
   * the slow path (https://github.com/nico-martin/llguidance-js/issues/4).
   */
  private fullSchemaAccepts(): boolean {
    // A shadow that took EOS has stopped, and a stopped interpreter validates no candidate.
    return !this.shadow || this.shadowEnded || this.shadow.validateCandidates(Uint32Array.of(this.eosTokenId))[0] === 1
  }

  /** The shadow committed EOS: the full schema accepted the value. */
  private shadowEnded = false

  private mismatch(): never {
    throw new StructuredOutputError("schema-mismatch", "The output can't satisfy the JSON Schema's pattern, format or numeric constraints.")
  }

  private check() {
    if (this.disposed) throw new StructuredOutputError("constraint-violation", "The structured-output constraint has been disposed.")
  }

  computeMask(): Uint32Array {
    this.check()
    if (this.mask) return this.mask
    const result = this.interpreter.computeMask()
    if ("stop" in result) {
      if (result.reason !== "accepted") throw new StructuredOutputError("constraint-dead-end", "No token can complete the requested output schema.")
      if (!this.fullSchemaAccepts()) this.mismatch()
      this.accepting = true
      const mask = new Uint32Array(Math.ceil(this.vocabSize / 32))
      mask[this.eosTokenId >>> 5] |= 1 << (this.eosTokenId & 31)
      return this.mask = mask
    }
    const canonical = this.canonicalMask
    const whitespace = this.whitespace && this.whitespaceRun >= MAX_WHITESPACE_RUN ? this.whitespace : undefined
    const allowed = canonical || whitespace
      ? result.mask.map((word, index) => word & (canonical ? canonical[index] : ~0) & (whitespace ? ~whitespace[index] : ~0))
      : result.mask
    // The looser mask may end a value the full schema refuses: keep generating, or fail fast.
    if (this.shadow && tokenAllowed(allowed, this.eosTokenId) && !this.fullSchemaAccepts()) {
      allowed[this.eosTokenId >>> 5] &= ~(1 << (this.eosTokenId & 31))
      if (!allowed.some(word => word !== 0)) this.mismatch()
    }
    if (!allowed.some(word => word !== 0)) throw new StructuredOutputError("constraint-dead-end", "No token can complete the requested output schema.")
    this.accepting = tokenAllowed(allowed, this.eosTokenId)
    if (this.accepting && this.endAtAcceptance) {
      // End at a complete value: never spend an unbounded output allowance on whitespace.
      const mask = new Uint32Array(allowed.length)
      mask[this.eosTokenId >>> 5] |= 1 << (this.eosTokenId & 31)
      return this.mask = mask
    }
    return this.mask = allowed
  }

  get complete() {
    this.computeMask()
    return this.accepting
  }

  accept(token: number) {
    this.check()
    if (!tokenAllowed(this.computeMask(), token)) throw new StructuredOutputError("constraint-violation", `Token ${token} violates the requested output schema.`)
    const result = this.interpreter.commitToken(token)
    if (result.backtrack || result.ffTokens.length) throw new StructuredOutputError("constraint-violation", "Unexpected constraint backtracking or token insertion.")
    this.mask = undefined
    if (this.whitespace) this.whitespaceRun = tokenAllowed(this.whitespace, token) ? this.whitespaceRun + 1 : 0
    if (this.shadow) {
      // The full schema refuses some impossible values as they're written: fail fast.
      try {
        this.shadow.commitToken(token)
      } catch {
        this.mismatch()
      }
      if (token === this.eosTokenId) this.shadowEnded = true
    }
  }

  clone(): Constraint {
    this.check()
    const copy = new JsonConstraint(this.interpreter.clone(), this.vocabSize, this.eosTokenId, this.endAtAcceptance, this.canonicalMask, this.shadow?.clone(), this.whitespace)
    copy.shadowEnded = this.shadowEnded
    copy.whitespaceRun = this.whitespaceRun
    return copy
  }

  dispose() {
    if (this.disposed) return
    this.disposed = true
    this.interpreter.dispose()
    this.shadow?.dispose()
    this.mask = undefined
  }
}

/** Numbers can be accepted prefixes of longer numbers: preserve their continuations. */
function excludesNumericRoot(schema: unknown): boolean {
  if (!schema || typeof schema !== "object" || Array.isArray(schema)) return false
  const s = schema as Record<string, unknown>
  if ("const" in s) return typeof s.const !== "number"
  if (Array.isArray(s.enum)) return s.enum.every(value => typeof value !== "number")
  if (typeof s.type === "string") return s.type !== "number" && s.type !== "integer"
  if (Array.isArray(s.type)) return !s.type.includes("number") && !s.type.includes("integer")
  for (const key of ["anyOf", "oneOf"]) {
    const branches = s[key]
    if (Array.isArray(branches) && branches.every(excludesNumericRoot)) return true
  }
  return Array.isArray(s.allOf) && s.allOf.some(excludesNumericRoot)
}

type Layout = { whitespace_flexible: boolean; key_separator: string; item_separator: string }
/** JSON.stringify's layout. */
const COMPACT: Layout = { whitespace_flexible: false, key_separator: ":", item_separator: "," }
/**
 * Whitespace between any two tokens, and the key separator's space required
 * (flexible alone also takes `":` again after a key's colon). Items stay ","
 * apart: llguidance 0.2.1 takes a comma after a closed object's last key when
 * items are ", " apart (its pruning is conservative), leaving no way on.
 */
const MODEL: Layout = { whitespace_flexible: true, key_separator: ": ", item_separator: "," }
/** MODEL when the tokenizer's whitespace tokens can't be read to cap their runs. */
const MODEL_FIXED: Layout = { whitespace_flexible: false, key_separator: ": ", item_separator: "," }

/** Formatting-only compiler hints; instance assertions and string contents stay intact. */
function compactSchema(schema: unknown, compact: Layout): unknown {
  if (schema && typeof schema === "object" && !Array.isArray(schema)) {
    const source = schema as Record<string, unknown>
    const hints = source["x-guidance"]
    return { ...source, "x-guidance": { ...(hints && typeof hints === "object" && !Array.isArray(hints) ? hints : {}), ...compact } }
  }
  return { allOf: [schema], "x-guidance": compact }
}

/*
 * Workaround for https://github.com/nico-martin/llguidance-js/issues/4: in
 * llguidance 0.2.1, `pattern`, `format`, numeric bounds and numeric enums
 * aren't pruned while a value is generated, only checked once it's complete,
 * so a value that has gone wrong can never close and generation runs to the
 * output limit (and those masks take 80 ms to seconds). Masks run on the
 * schema without them; a shadow interpreter with the full schema decides
 * whether a value may end, and fails fast when it can't. Remove this (and
 * the shadow in JsonConstraint) once that issue is fixed upstream.
 */
const UNPRUNED = new Set(["pattern", "format", "minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum", "multipleOf"])
const SCHEMA_MAPS = ["properties", "patternProperties", "$defs", "definitions", "dependentSchemas"]
const SCHEMA_VALUES = ["items", "additionalItems", "additionalProperties", "contains", "propertyNames", "unevaluatedItems", "unevaluatedProperties"]
const SCHEMA_LISTS = ["prefixItems", "anyOf", "oneOf", "allOf"]
const isRecord = (value: unknown): value is Record<string, unknown> => !!value && typeof value === "object" && !Array.isArray(value)
const jsonType = (value: unknown) =>
  value === null ? "null" : Array.isArray(value) ? "array" : typeof value === "number" ? (Number.isInteger(value) ? "integer" : "number") : typeof value

/**
 * The schema with what llguidance doesn't prune taken out, and whether
 * anything was. Only ever looser than the original: a `not` or `if` that
 * would change meaning without those keywords is left out whole.
 */
export function prunableSchema(schema: unknown): { schema: unknown; loosened: boolean } {
  let loosened = false
  const visit = (node: unknown): unknown => {
    if (Array.isArray(node)) return node.map(visit)
    if (!isRecord(node)) return node
    const out: Record<string, unknown> = {}
    for (const [key, value] of Object.entries(node)) {
      if (UNPRUNED.has(key)) {
        loosened = true
        continue
      }
      if (SCHEMA_MAPS.includes(key) && isRecord(value)) out[key] = Object.fromEntries(Object.entries(value).map(([k, v]) => [k, visit(v)]))
      else if (SCHEMA_VALUES.includes(key) || SCHEMA_LISTS.includes(key)) out[key] = visit(value)
      else if (key === "not" || key === "if" || key === "then" || key === "else") {
        const before = loosened
        loosened = false
        const inner = visit(value)
        const changed = loosened
        loosened = before || changed
        if (!changed) out[key] = inner
        else if (key === "then" || key === "else") out[key] = inner
        // A changed `not` or `if` is dropped below.
        else out[key] = { __drop: true }
      } else out[key] = value
    }
    if (isRecord(out.not) && out.not.__drop) delete out.not
    if (isRecord(out.if) && out.if.__drop) {
      delete out.if
      delete out.then
      delete out.else
    }
    // Numeric enum and const values: keep their types, check the values at the end.
    const literals = "enum" in out ? out.enum : "const" in out ? [out.const] : undefined
    if (Array.isArray(literals) && literals.some((value) => typeof value === "number")) {
      loosened = true
      delete out.enum
      delete out.const
      if (out.type === undefined) {
        const types = [...new Set(literals.map(jsonType))]
        out.type = types.length === 1 ? types[0] : types
      }
    }
    return out
  }
  const result = visit(schema)
  return { schema: result, loosened }
}

/** Compile a fresh, generation-local schema matcher. No inference or upstream patches. */
export async function createConstraint(options: ConstraintOptions): Promise<Constraint> {
  const runtime = await loadRuntime()
  // Preserve identity so repeated generations reuse converted tokenizer bytes.
  const source = withEos(options.tokenizer, options.eosTokenId)
  const limits = { maxTokens: options.maxTokens ?? Number.MAX_SAFE_INTEGER, maxOutputBytes: Number.MAX_SAFE_INTEGER }
  const prunable = prunableSchema(options.schema)
  const whitespaceIds = options.layout === "compact" ? null : whitespaceTokenIds(source)
  const layout = options.layout === "compact" ? COMPACT : whitespaceIds?.length ? MODEL : MODEL_FIXED
  let interpreter: Interpreter
  let shadow: Interpreter | undefined
  try {
    // The full schema first, so what it doesn't support still fails here.
    const full = runtime.createInterpreter({ tokenizer: source, response_format: { type: "json_schema", json_schema: compactSchema(options.schema, layout) }, limits })
    if (prunable.loosened) {
      shadow = full
      interpreter = runtime.createInterpreter({ tokenizer: source, response_format: { type: "json_schema", json_schema: compactSchema(prunable.schema, layout) }, limits })
    } else interpreter = full
  } catch (cause) {
    shadow?.dispose()
    const message = cause instanceof Error ? cause.message : String(cause)
    const tokenizer = /tokenizer|vocabulary|token id|EOS/i.test(message)
    throw new StructuredOutputError(tokenizer ? "invalid-tokenizer" : "unsupported-schema", `Cannot enforce structured output: ${message}`, { cause })
  }
  let vocabSize: number
  let eos: number
  try {
    const initial = interpreter.computeMask()
    if ("stop" in initial) throw new StructuredOutputError("constraint-dead-end", "No token can complete the requested output schema.")
    vocabSize = initial.vocabSize
    const metadata = source as { eos_token_id?: number; eosTokenId?: number; eos_token?: number; tokenizer?: { eos_token_id?: number } }
    eos = options.eosTokenId ?? metadata.eos_token_id ?? metadata.eosTokenId ?? metadata.eos_token ?? metadata.tokenizer?.eos_token_id ?? -1
    if (!Number.isSafeInteger(eos) || eos < 0 || eos >= vocabSize) throw new Error("An exact EOS token id is required for structured output.")
  } catch (cause) {
    interpreter.dispose()
    shadow?.dispose()
    if (cause instanceof StructuredOutputError) throw cause
    throw new StructuredOutputError("invalid-tokenizer", cause instanceof Error ? cause.message : String(cause), { cause })
  }
  let whitespace: Uint32Array | undefined
  if (layout === MODEL && whitespaceIds) {
    whitespace = new Uint32Array(Math.ceil(vocabSize / 32))
    for (const id of whitespaceIds) if (id >= 0 && id < vocabSize) whitespace[id >>> 5] |= 1 << (id & 31)
  }
  return new JsonConstraint(interpreter, vocabSize, eos, excludesNumericRoot(options.schema),
    canonicalAsciiObject(options.schema) ? canonicalTokenMask(source, vocabSize, eos) : undefined, shadow, whitespace)
}
const VALIDATION_EOS = 256
const validationTokenizer = {
  tokens: Array.from({ length: 257 }, (_, id) => id === VALIDATION_EOS ? [] : [id]),
  eos_token_id: VALIDATION_EOS,
  special_token_ids: [VALIDATION_EOS]
}
function jsonValue(value: unknown, ancestors = new Set<object>()): boolean {
  if (value === null || typeof value === "string" || typeof value === "boolean") return true
  if (typeof value === "number") return Number.isFinite(value)
  if (typeof value !== "object" || ancestors.has(value)) return false
  const prototype = Object.getPrototypeOf(value)
  if (!Array.isArray(value) && prototype !== Object.prototype && prototype !== null) return false
  ancestors.add(value)
  const valid = Array.isArray(value)
    ? Array.from({ length: value.length }, (_, i) => i in value && jsonValue(value[i], ancestors)).every(Boolean)
    : Object.values(value).every(item => jsonValue(item, ancestors))
  ancestors.delete(value)
  return valid
}

/** Validate parsed tool arguments locally; unsupported schemas remain explicit errors. */
export async function validateStructuredValue(schema: unknown, value: unknown): Promise<boolean> {
  let constraint: Constraint
  try {
    constraint = await createConstraint({ schema, tokenizer: validationTokenizer, layout: "compact" })
  } catch (error) {
    // A supported but unsatisfiable schema has no accepted value.
    if (error instanceof StructuredOutputError && error.code === "constraint-dead-end") return false
    throw error
  }
  try {
    if (!jsonValue(value)) return false
    const text = JSON.stringify(value)
    for (const token of new TextEncoder().encode(text)) {
      if (!tokenAllowed(constraint.computeMask(), token)) return false
      constraint.accept(token)
    }
    return constraint.complete
  } catch (error) {
    if (error instanceof StructuredOutputError && (error.code === "constraint-violation" || error.code === "constraint-dead-end" || error.code === "schema-mismatch")) return false
    throw error
  } finally {
    constraint.dispose()
  }
}
