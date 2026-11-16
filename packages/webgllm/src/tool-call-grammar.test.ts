import assert from "node:assert/strict"
import { test } from "node:test"
import type { Constraint } from "./constraint/index.js"
import { callVocab, ToolCallGrammar } from "./tool-call-grammar.js"
import type { ChatTool } from "./tokenizer.js"

// A vocabulary of every printable ASCII byte and newline (ids = the byte),
// some longer pieces, and special tokens from 1000.
const pieces = ["\n<function=", "get_weather", "<parameter=", ">\n", "\n</parameter>\n", "</function>\n", "Oslo", "Oslo\n", "city", "units", "(", ")", "=", "[", "]", ", ", "12", "get_time"]
const START = 1000, END = 1001, EOS = 1002, THINK = 1003, THINK_END = 1004
const texts = new Map<number, string>()
for (let b = 32; b < 127; b++) texts.set(b, String.fromCharCode(b))
texts.set(10, "\n")
pieces.forEach((p, i) => texts.set(200 + i, p))
const specials = new Set([START, END, EOS, THINK, THINK_END])
const tokenizer = {
  vocabSize: 1005,
  tokenBytes: (id: number) => new TextEncoder().encode(texts.get(id) ?? ""),
  isSpecial: (id: number) => specials.has(id)
}
const idOf = (text: string) => [...texts].find(([, t]) => t === text)![0]
const allowed = (mask: Uint32Array | null, id: number) => !!mask && (mask[id >>> 5] & (1 << (id & 31))) !== 0

/** A JSON constraint for integers: digits, complete after one. */
function digits(): Constraint {
  let n = 0
  const make = (): Constraint => ({
    vocabSize: 1005,
    eosTokenId: EOS,
    get complete() { return n > 0 },
    computeMask() {
      const m = new Uint32Array(Math.ceil(1005 / 32))
      for (let b = 48; b < 58; b++) m[b >>> 5] |= 1 << (b & 31)
      const twelve = idOf("12")
      m[twelve >>> 5] |= 1 << (twelve & 31)
      if (n > 0) m[EOS >>> 5] |= 1 << (EOS & 31)
      return m
    },
    accept() { n++ },
    clone: () => make(),
    dispose() {}
  })
  return make()
}

const weather: ChatTool = {
  type: "function",
  function: {
    name: "get_weather",
    description: "",
    parameters: { type: "object", properties: { city: { type: "string" }, days: { type: "integer" } }, required: ["city"] }
  }
}
const time: ChatTool = { type: "function", function: { name: "get_time", description: "", parameters: {} } }

function grammar(format: "lfm3" | "lfm2", parallel = true, tools = [weather, time]) {
  return new ToolCallGrammar({
    format, tools, parallel, vocab: callVocab(tokenizer), start: START, end: END, eos: EOS, think: THINK, thinkEnd: THINK_END,
    startsInThinking: false, json: async () => digits()
  })
}

/** Writes `text` one known piece or byte at a time, checking each against the mask. */
async function write(g: ToolCallGrammar, tokens: Array<number | string>) {
  for (const t of tokens) {
    const piece = typeof t === "string" ? [...texts].find(([, text]) => text === t)?.[0] : undefined
    const ids = typeof t === "number" ? [t] : piece !== undefined ? [piece] : [...t].map((c) => c.charCodeAt(0))
    for (const id of ids) {
      await g.ready()
      assert.ok(allowed(g.mask(), id), `${JSON.stringify(texts.get(id))} should be allowed`)
      g.observe(id)
    }
  }
}

test("outside a call nothing is constrained; inside a thought, the marker isn't a call", async () => {
  const g = grammar("lfm3")
  assert.equal(g.mask(), null)
  g.observe(THINK)
  g.observe(START)
  assert.equal(g.calling, false)
  g.observe(THINK_END)
  g.observe(START)
  assert.equal(g.calling, true)
})

test("XML: a call to an offered tool, required parameters first, strings raw, other values held to their schema", async () => {
  const g = grammar("lfm3")
  g.observe(START)
  await g.ready()
  assert.ok(allowed(g.mask(), idOf("\n<function=")))
  assert.ok(!allowed(g.mask(), "x".charCodeAt(0)))
  await write(g, ["\n<function=", "get_weather", ">\n"])
  // `city` is required: it comes first, and the function can't close before it.
  await g.ready()
  assert.ok(!allowed(g.mask(), idOf("</function>\n")))
  await write(g, ["<parameter=", "city", ">\n"])
  // A raw string: any text, up to its closing tag; the token spanning both is fine.
  await g.ready()
  assert.ok(allowed(g.mask(), "<".charCodeAt(0)), "raw text may hold <")
  assert.ok(!allowed(g.mask(), START), "no special tokens in raw text")
  await write(g, ["Oslo", "\n</parameter>\n"])
  await write(g, ["<parameter=", "days", ">\n"])
  // An integer: digits only, then its closing tag once it's a whole value.
  await g.ready()
  assert.ok(!allowed(g.mask(), "x".charCodeAt(0)))
  assert.ok(!allowed(g.mask(), idOf("\n</parameter>\n")), "a value can't be empty")
  await write(g, ["12"])
  await g.ready()
  assert.ok(allowed(g.mask(), idOf("\n</parameter>\n")))
  assert.ok(!allowed(g.mask(), EOS))
  await write(g, ["\n</parameter>\n", "</function>\n"])
  await g.ready()
  assert.deepEqual([...Array(1005).keys()].filter((id) => allowed(g.mask(), id)), [END])
  g.observe(END)
  assert.equal(g.calling, false)
  // After the call: the reply ends, or (parallel) another call follows.
  assert.ok(allowed(g.mask(), EOS) && allowed(g.mask(), START) && allowed(g.mask(), 10))
  assert.ok(!allowed(g.mask(), "x".charCodeAt(0)))
  const one = grammar("lfm3", false)
  one.observe(START)
  await write(one, ["\n<function=", "get_time", ">\n", "</function>\n", END])
  assert.ok(allowed(one.mask(), EOS) && !allowed(one.mask(), START), "one call only")
})

test("XML: no tool that wasn't offered, no parameter it doesn't have", async () => {
  const g = grammar("lfm3", true, [time])
  g.observe(START)
  await g.ready()
  await write(g, ["\n<function="])
  await g.ready()
  assert.ok(!allowed(g.mask(), idOf("get_weather")))
  assert.ok(allowed(g.mask(), idOf("get_time")))
  await write(g, ["get_time", ">\n"])
  await g.ready()
  assert.ok(!allowed(g.mask(), idOf("<parameter=")), "get_time takes no parameters")
})

test("Python style: [name(key=value, ...)], strings quoted either way, other values held to their schema", async () => {
  const g = grammar("lfm2")
  g.observe(START)
  await write(g, ["[", "get_weather", "(", "city", "="])
  await g.ready()
  assert.ok(allowed(g.mask(), "'".charCodeAt(0)) && allowed(g.mask(), '"'.charCodeAt(0)))
  assert.ok(!allowed(g.mask(), "O".charCodeAt(0)), "a string is quoted")
  await write(g, ["'", "Oslo", "'", ", ", "days", "=", "12"])
  await g.ready()
  assert.ok(allowed(g.mask(), idOf(")")))
  await write(g, [")"])
  // Parallel: another call, or the list ends.
  await g.ready()
  assert.ok(allowed(g.mask(), idOf(", ")) && allowed(g.mask(), idOf("]")))
  await write(g, ["]", END])
  assert.ok(allowed(g.mask(), EOS) && !allowed(g.mask(), START), "the reply ends after the list")
})

test("the vocabulary is read once per tokenizer", () => {
  assert.equal(callVocab(tokenizer), callVocab(tokenizer))
  const vocab = callVocab(tokenizer)
  assert.ok(!vocab.order.includes(START), "special tokens aren't text")
  assert.ok(vocab.whitespace.includes(10))
})

test("Python style: the call can't close before its required arguments, written before or after an optional one", async () => {
  const g = grammar("lfm2")
  g.observe(START)
  await write(g, ["[", "get_weather", "("])
  await g.ready()
  assert.ok(!allowed(g.mask(), idOf(")")), "city is required")
  await write(g, ["days", "=", "12"])
  await g.ready()
  assert.ok(!allowed(g.mask(), idOf(")")), "still no city")
  await write(g, [", ", "city", "=", "'", "Oslo", "'"])
  await g.ready()
  assert.ok(allowed(g.mask(), idOf(")")))
})

test("a parameter's references resolve against the whole tool, and a root $ref is followed to its object", async () => {
  const { toolSchema, valueSchema, mayBeString } = await import("./tool-schemas.js")
  const local = toolSchema({ type: "function", function: { name: "f", parameters: { type: "object", properties: { a: { type: "integer", minimum: 1 }, b: { $ref: "#/properties/a" } }, required: ["b"] } } })
  assert.deepEqual(valueSchema(local, local.properties[1]![1]), { type: "integer", minimum: 1 })
  // It compiles with the real JSON constraint (the reference alone didn't resolve).
  const { validateStructuredValue } = await import("./constraint/index.js")
  assert.equal(await validateStructuredValue(valueSchema(local, local.properties[1]![1]), 3), true)
  assert.equal(await validateStructuredValue(valueSchema(local, local.properties[1]![1]), 0), false)
  const rooted = toolSchema({
    type: "function",
    function: { name: "g", parameters: { $ref: "#/$defs/args", $defs: { args: { type: "object", properties: { city: { $ref: "#/$defs/city" } }, required: ["city"] }, city: { type: "string" } } } }
  })
  assert.deepEqual(rooted.properties.map(([name]) => name), ["city"])
  assert.deepEqual(rooted.required, ["city"])
  assert.equal(mayBeString(rooted, rooted.properties[0]![1]), true)
  // A definition reference stays, with the definitions beside it (a recursive one still compiles).
  const value = valueSchema(rooted, rooted.properties[0]![1]) as Record<string, unknown>
  assert.equal(value.$ref, "#/$defs/city")
  assert.ok((value.$defs as Record<string, unknown>).city)
  // The grammar asks for the root $ref's required parameter.
  const g = new ToolCallGrammar({
    format: "lfm3", tools: [{ type: "function", function: { name: "get_weather", description: "", parameters: { $ref: "#/$defs/args", $defs: { args: { type: "object", properties: { city: { type: "string" } }, required: ["city"] } } } } }],
    parallel: true, vocab: callVocab(tokenizer), start: START, end: END, eos: EOS, think: THINK, thinkEnd: THINK_END, startsInThinking: false, json: async () => digits()
  })
  g.observe(START)
  await write(g, ["\n<function=", "get_weather", ">\n"])
  await g.ready()
  assert.ok(!allowed(g.mask(), idOf("</function>\n")), "city is required")
  assert.ok(allowed(g.mask(), idOf("<parameter=")))
})
