/**
 * Tool calls held to their tools while they're written, in the model's own
 * call format, as llama.cpp's grammars hold them. Outside a call nothing is
 * constrained; once the model writes its call's opening marker (`<tool_call>`
 * for LFM3 and Qwen3.5, `<|tool_call_start|>` for LFM2), every token until
 * the closing marker must keep the call one that names an offered tool,
 * with that tool's parameters, each value valid against its schema.
 *
 * The two formats, as llama.cpp builds their grammars:
 *
 * - XML (LFM3, Qwen3.5; llama.cpp's tag-tagged autoparser):
 *   `<tool_call>\n<function=NAME>\n<parameter=KEY>\nVALUE\n</parameter>\n...</function>\n</tool_call>`.
 *   Required parameters first, in the order the schema declares them, then
 *   optional ones in any order (once each here). A value that may be a
 *   string is raw text up to `\n</parameter>`; any other value is JSON held
 *   to the parameter's schema.
 * - Python style (LFM2; llama.cpp's python_style_tool_calls):
 *   `<|tool_call_start|>[name(key="value", n=1), ...]<|tool_call_end|>`.
 *   Arguments in any order (once each here). A string parameter's value is a
 *   quoted string, in single or double quotes; any other value is JSON held
 *   to the parameter's schema (llama.cpp takes any Python or JSON value
 *   there: the schema holds it to what the tool takes).
 *
 * After a call, the reply ends, as llama.cpp's grammar ends it: with
 * parallel calls, XML calls may follow (whitespace between them).
 *
 * Built by hand, from the JSON constraint and a byte-level matcher over the
 * vocabulary, because llguidance-js 0.2.1's grammar entry can't embed a JSON
 * Schema (`%json` is refused). When it can
 * (https://github.com/nico-martin/llguidance-js/pull/7), swap this whole file
 * for one Lark grammar per format with `%json` for each value, as llama.cpp
 * writes them: see https://github.com/webgllm/webgllm/issues/17.
 */
import type { Constraint } from "./constraint/index.js"
import type { ChatTool } from "./tokenizer.js"
import { closeObjects } from "./tool-choice.js"
import type { ToolFormat } from "./tool-calls.js"
import { isString, mayBeString, toolSchema, valueSchema } from "./tool-schemas.js"
import type { ToolSchema } from "./tool-schemas.js"

/** The vocabulary as the matcher reads it: each token's bytes, sorted. */
export type CallVocab = {
  /** One past the highest id: the masks' size in bits. */
  size: number
  /** Each id's bytes (empty for special ids). */
  bytes: Array<Uint8Array>
  /** The ids of text tokens (not special, not empty), sorted by their bytes. */
  order: Int32Array
  /** Text tokens made of whitespace only. */
  whitespace: Int32Array
}

const encoder = new TextEncoder()
const decoder = new TextDecoder()
const enc = (text: string) => encoder.encode(text)

/** Lexicographic order of two byte strings, a prefix first. */
function compareBytes(a: Uint8Array, b: Uint8Array) {
  const n = Math.min(a.length, b.length)
  for (let i = 0; i < n; i++) if (a[i] !== b[i]) return a[i] - b[i]
  return a.length - b.length
}

const vocabs = new WeakMap<object, CallVocab>()

/** The vocabulary the matcher reads, built once per tokenizer. */
export function callVocab(tokenizer: { vocabSize: number; tokenBytes(id: number): Uint8Array; isSpecial(id: number): boolean }): CallVocab {
  const known = vocabs.get(tokenizer)
  if (known) return known
  const size = tokenizer.vocabSize
  const bytes: Array<Uint8Array> = new Array(size)
  const text: Array<number> = []
  const space: Array<number> = []
  for (let id = 0; id < size; id++) {
    const b = tokenizer.isSpecial(id) ? new Uint8Array(0) : tokenizer.tokenBytes(id)
    bytes[id] = b
    if (!b.length) continue
    text.push(id)
    if (b.every((x) => x === 0x20 || x === 0x0a || x === 0x09 || x === 0x0d)) space.push(id)
  }
  text.sort((a, b) => compareBytes(bytes[a], bytes[b]))
  const vocab = { size, bytes, order: Int32Array.from(text), whitespace: Int32Array.from(space) }
  vocabs.set(tokenizer, vocab)
  return vocab
}

type Next = () => State
/** Where the matcher is in a call. States are never changed: a step makes a new one. */
type State =
  /** One of several texts (no one a prefix of another), `i` bytes in. */
  | { k: "pick"; options: Array<{ s: Uint8Array; next: Next }>; i: number }
  /** Raw text, up to and including `suffix` (`m` of its bytes matched so far). */
  | { k: "until"; suffix: Uint8Array; m: number; next: Next }
  /** A quoted string: `q` its quote once opened, `esc` after a backslash. */
  | { k: "quoted"; q: number; esc: boolean; next: Next }
  /** A JSON value held to `schema`, token by token (see ToolCallGrammar.sub). */
  | { k: "json"; schema: object; next: Next }
  /** A special token. */
  | { k: "special"; id: number; next: Next }
  /** The call is closed. */
  | { k: "end" }

const lit = (text: string, next: Next): State => ({ k: "pick", options: [{ s: enc(text), next }], i: 0 })
const pick = (options: Array<{ text: string; next: Next }>): State => ({ k: "pick", options: options.map((o) => ({ s: enc(o.text), next: o.next })), i: 0 })
const byteLevel = (s: State) => s.k === "pick" || s.k === "until" || s.k === "quoted"

/** The state after byte `b`, or null where `b` can't come next. */
function step(s: State, b: number): State | null {
  switch (s.k) {
    case "pick": {
      let alive: Array<{ s: Uint8Array; next: Next }> | undefined
      for (const o of s.options) {
        if (o.s[s.i] !== b) continue
        if (s.i + 1 === o.s.length) return o.next()
        ;(alive ??= []).push(o)
      }
      return alive ? { k: "pick", options: alive, i: s.i + 1 } : null
    }
    case "until": {
      if (s.suffix[s.m] === b) return s.m + 1 === s.suffix.length ? s.next() : { ...s, m: s.m + 1 }
      // The longest start of the suffix the text now ends with.
      for (let m = s.m; m > 0; m--) {
        let ok = s.suffix[m - 1] === b
        for (let j = 0; ok && j < m - 1; j++) ok = s.suffix[j] === s.suffix[s.m - m + 1 + j]
        if (ok) return { ...s, m }
      }
      return s.m === 0 ? s : { ...s, m: 0 }
    }
    case "quoted":
      if (s.q === 0) return b === 0x22 || b === 0x27 ? { ...s, q: b } : null
      if (s.esc) return { ...s, esc: false }
      if (b === 0x5c) return { ...s, esc: true }
      return b === s.q ? s.next() : s
    default:
      return null
  }
}

/** The state after all of `bytes`, or null; past a JSON value or special token, nothing more. */
function run(s: State, bytes: Uint8Array): State | null {
  let at: State | null = s
  for (let i = 0; i < bytes.length && at; i++) {
    if (!byteLevel(at)) return null
    at = step(at, bytes[i])
  }
  return at
}

const setBit = (bits: Uint32Array, id: number) => { bits[id >>> 5] |= 1 << (id & 31) }

/**
 * Sets the bit of every text token whose bytes keep `start` alive, walking
 * the sorted vocabulary as a trie. A token may end where a JSON value or a
 * special token begins, never go into one.
 */
function allow(vocab: CallVocab, start: State, bits: Uint32Array, order = vocab.order) {
  const { bytes } = vocab
  const visit = (lo: number, hi: number, depth: number, s: State) => {
    let i = lo
    while (i < hi && bytes[order[i]].length === depth) {
      if (depth > 0) setBit(bits, order[i])
      i++
    }
    if (!byteLevel(s)) return
    while (i < hi) {
      const b = bytes[order[i]][depth]
      // The group's end: the first token after it whose byte at `depth` is larger.
      let lo2 = i + 1, hi2 = hi
      while (lo2 < hi2) {
        const mid = (lo2 + hi2) >>> 1
        if (bytes[order[mid]][depth] > b) hi2 = mid
        else lo2 = mid + 1
      }
      const next = step(s, b)
      if (next) visit(i, lo2, depth + 1, next)
      i = lo2
    }
  }
  visit(0, order.length, 0, start)
}

/** Text tokens without any of `triggers` (always fine in raw or quoted text), and the ones with. */
const splits = new WeakMap<CallVocab, Map<string, { safe: Uint32Array; rest: Int32Array }>>()
function split(vocab: CallVocab, triggers: Array<number>) {
  let byKey = splits.get(vocab)
  if (!byKey) splits.set(vocab, (byKey = new Map()))
  const key = triggers.join(",")
  let found = byKey.get(key)
  if (!found) {
    const safe = new Uint32Array(Math.ceil(vocab.size / 32))
    const rest: Array<number> = []
    for (const id of vocab.order) {
      if (vocab.bytes[id].some((b) => triggers.includes(b))) rest.push(id)
      else setBit(safe, id)
    }
    found = { safe, rest: Int32Array.from(rest) }
    byKey.set(key, found)
  }
  return found
}

type Tool = ToolSchema

/** A value's JSON Schema for the constraint: the parameter's own, references resolved, objects closed as llama.cpp reads them. */
const jsonSchema = (tool: Tool, schema: unknown) => closeObjects(valueSchema(tool, schema)) as object

/** The XML format's call, after its opening marker. */
function xmlCall(tools: Array<Tool>, end: number): State {
  const args = (tool: Tool, used: Array<string>): State => {
    const value = (name: string, schema: unknown): State => {
      const after = () => args(tool, [...used, name])
      return mayBeString(tool, schema)
        ? { k: "until", suffix: enc("\n</parameter>\n"), m: 0, next: after }
        : { k: "json", schema: jsonSchema(tool, schema), next: () => lit("\n</parameter>\n", after) }
    }
    const open = ([name, schema]: [string, unknown]) => ({ text: `<parameter=${name}>\n`, next: () => value(name, schema) })
    const required = tool.required.find((r) => !used.includes(r))
    if (required !== undefined) return pick([open(tool.properties.find(([name]) => name === required)!)])
    return pick([
      { text: "</function>\n", next: (): State => ({ k: "special", id: end, next: () => ({ k: "end" }) }) },
      ...tool.properties.filter(([name]) => !used.includes(name)).map(open)
    ])
  }
  return pick(tools.map((tool) => ({ text: `\n<function=${tool.name}>\n`, next: () => args(tool, []) })))
}

/** The Python-style list of calls, after its opening marker. */
function pythonCalls(tools: Array<Tool>, end: number, parallel: boolean): State {
  const closed = (): State => ({ k: "special", id: end, next: () => ({ k: "end" }) })
  // After a call: the list ends, or (parallel) a comma, a space or not, and the next call.
  const afterCall = (): State =>
    parallel ? pick([{ text: "]", next: closed }, ...[", ", ","].flatMap((sep) => calls(sep))]) : lit("]", closed)
  const args = (tool: Tool, used: Array<string>): State => {
    const value = (name: string, schema: unknown): State => {
      const after = () => args(tool, [...used, name])
      return isString(tool, schema) ? { k: "quoted", q: 0, esc: false, next: after } : { k: "json", schema: jsonSchema(tool, schema), next: after }
    }
    const unused = tool.properties.filter(([name]) => !used.includes(name))
    const seps = used.length ? [", ", ","] : [""]
    // The call closes once its required arguments are written (llama.cpp's
    // Python-style grammar doesn't check them; its XML one does, and a call
    // without them is no call the tool takes).
    const complete = tool.required.every((name) => used.includes(name))
    return pick([
      ...(complete ? [{ text: ")", next: afterCall }] : []),
      ...seps.flatMap((sep) => unused.map(([name, schema]) => ({ text: `${sep}${name}=`, next: () => value(name, schema) })))
    ])
  }
  const calls = (sep: string) => tools.map((tool) => ({ text: `${sep}${tool.name}(`, next: () => args(tool, []) }))
  return lit("[", () => pick(calls("")))
}

export type ToolCallGrammarOptions = {
  format: ToolFormat
  /** The tools a call may name. */
  tools: Array<ChatTool>
  /** False: one call per reply. */
  parallel: boolean
  vocab: CallVocab
  /** The call's opening and closing markers (special tokens). */
  start: number
  end: number
  eos: number
  think: number | null
  thinkEnd: number | null
  /** Whether the reply starts inside a thought (no call is read there). */
  startsInThinking: boolean
  /** A JSON constraint for a value's schema (the worker's createConstraint). */
  json: (schema: object) => Promise<Constraint>
}

/** A reply's tool calls, held to their tools as they're written (see the file's comment). */
export class ToolCallGrammar {
  private state: State | null = null
  private thinking: boolean
  /** The reply's last few characters outside a call, for thinking tags split across ordinary tokens. */
  private tail = ""
  /** The JSON value being written, and the state after it. */
  private sub: Constraint | undefined
  private after: State | undefined
  private readonly tools: Array<Tool>

  constructor(private readonly o: ToolCallGrammarOptions) {
    this.thinking = o.startsInThinking
    this.tools = o.tools.map(toolSchema)
  }

  /** Inside a call. */
  get calling() { return this.state !== null && this.state.k !== "end" }
  /** Inside a call or past one: each token has its own mask (no drafts). */
  get holding() { return this.state !== null }

  private begin(): State {
    return this.o.format === "lfm2" ? pythonCalls(this.tools, this.o.end, this.o.parallel) : xmlCall(this.tools, this.o.end)
  }

  /** Readies the JSON constraint of the value about to be written, before the next mask. */
  async ready() {
    if (this.state?.k === "json" && !this.sub) {
      this.sub = await this.o.json(this.state.schema)
      this.after = this.state.next()
    }
  }

  /** The tokens that may come next, or null where the call doesn't decide (outside a call). */
  mask(): Uint32Array | null {
    const s = this.state
    if (s === null) return null
    const bits = new Uint32Array(Math.ceil(this.o.vocab.size / 32))
    switch (s.k) {
      case "end":
        setBit(bits, this.o.eos)
        if (this.o.parallel && this.o.format !== "lfm2") {
          setBit(bits, this.o.start)
          for (const id of this.o.vocab.whitespace) setBit(bits, id)
        }
        return bits
      case "special":
        setBit(bits, s.id)
        return bits
      case "json": {
        if (!this.sub) throw new Error("ToolCallGrammar.ready() must run before mask().")
        const value = this.sub.computeMask()
        bits.set(value.subarray(0, bits.length))
        bits[this.o.eos >>> 5] &= ~(1 << (this.o.eos & 31))
        if (this.sub.complete) allow(this.o.vocab, this.after!, bits)
        return bits
      }
      case "until":
      case "quoted": {
        const open = s.k === "until" ? s.m === 0 : s.q !== 0 && !s.esc
        if (open) {
          const { safe, rest } = split(this.o.vocab, s.k === "until" ? [s.suffix[0]] : [s.q, 0x5c])
          bits.set(safe)
          allow(this.o.vocab, s, bits, rest)
          return bits
        }
        allow(this.o.vocab, s, bits)
        return bits
      }
      default:
        allow(this.o.vocab, s, bits)
        return bits
    }
  }

  /** Takes in a token of the reply. */
  observe(token: number) {
    const s = this.state
    if (s === null || s.k === "end") {
      // Thinking as the reply reader reads it: its special tokens, or its tags written as text.
      if (token === this.o.think) this.thinking = true
      else if (token === this.o.thinkEnd) this.thinking = false
      else if (!this.thinking && token === this.o.start && (s === null || this.o.parallel)) this.state = this.begin()
      else {
        this.tail = (this.tail + decoder.decode(this.o.vocab.bytes[token] ?? new Uint8Array(0))).slice(-16)
        const open = this.tail.lastIndexOf("<think>"), close = this.tail.lastIndexOf("</think>")
        if (open > close) this.thinking = true
        else if (close > open) this.thinking = false
        if (open >= 0 || close >= 0) this.tail = ""
      }
      return
    }
    if (s.k === "special") {
      if (token !== s.id) throw new Error("The tool call's grammar didn't allow that token.")
      this.state = s.next()
      return
    }
    const bytes = this.o.vocab.bytes[token] ?? new Uint8Array(0)
    if (s.k === "json") {
      // The value ends where the text after it begins.
      const out = this.sub!.complete && bytes.length ? run(this.after!, bytes) : null
      if (out) {
        this.sub!.dispose()
        this.sub = this.after = undefined
        this.state = out
      } else this.sub!.accept(token)
      return
    }
    const out = bytes.length ? run(s, bytes) : null
    if (!out) throw new Error("The tool call's grammar didn't allow that token.")
    this.state = out
  }

  dispose() {
    this.sub?.dispose()
    this.sub = undefined
  }
}
