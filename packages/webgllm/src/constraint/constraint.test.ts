import assert from "node:assert/strict"
import { test } from "node:test"
import { createConstraint, tokenAllowed, type Constraint } from "./index.js"
const EOS = 256
const tokenizer = { tokens: Array.from({ length: 257 }, (_, id) => id === EOS ? [] : [id]), eos_token_id: EOS, special_token_ids: [EOS] }
const schema = { type: "object", properties: { moveId: { type: "string", enum: ["b6-a5", "d6-c5"] } }, required: ["moveId"], additionalProperties: false }
const bytes = (text: string) => new TextEncoder().encode(text)
function accept(c: Constraint, text: string) {
  for (const token of bytes(text)) {
    assert.ok(tokenAllowed(c.computeMask(), token), `token ${token} of ${text} is allowed`)
    c.accept(token)
  }
}

test("a move object enforces the enum and permits only EOS after completion", async () => {
  const c = await createConstraint({ schema, tokenizer })
  assert.equal(c.vocabSize, 257)
  assert.equal(c.complete, false)
  assert.equal(tokenAllowed(c.computeMask(), EOS), false)
  accept(c, '{"moveId":"')
  assert.equal(tokenAllowed(c.computeMask(), "x".charCodeAt(0)), false)
  accept(c, 'b6-a5"}')
  assert.equal(c.complete, true)
  assert.equal(tokenAllowed(c.computeMask(), EOS), true)
  assert.equal(tokenAllowed(c.computeMask(), 32), false, "no unlimited trailing whitespace")
  assert.throws(() => c.accept(123), /violates/)
  c.accept(EOS)
  assert.equal(c.complete, true)
  c.dispose(); c.dispose()
  assert.throws(() => c.computeMask(), /disposed/)
})

test("clones maintain separate speculative prefixes", async () => {
  const a = await createConstraint({ schema, tokenizer })
  accept(a, '{"moveId":"')
  const b = a.clone()
  accept(a, 'b6-a5"}')
  accept(b, 'd6-c5"}')
  assert.equal(a.complete, true); assert.equal(b.complete, true)
  a.dispose(); b.dispose()
})

test("exact UTF-8 bytes and escaped strings are accepted across token boundaries", async () => {
  const value = { answer: 'café ☃ "quoted"\n' }
  const c = await createConstraint({ schema: { const: value }, tokenizer })
  accept(c, JSON.stringify(value))
  assert.equal(c.complete, true)
  c.dispose()
})

test("HF ByteLevel vocabulary bridge preserves raw byte tokens and added special ids", async () => {
  const bridge = {
    tokenizerJSON: { model: { type: "BPE", vocab: { '{': 0, '}': 1, '"': 2, ':': 3, 'x': 4, '1': 5 } }, decoder: { type: "ByteLevel" }, added_tokens: [{ id: 6, content: "<eos>", special: true }] },
    eos_token_id: 6
  }
  const c = await createConstraint({ schema: { const: { x: 1 } }, tokenizer: bridge })
  for (const token of [0,2,4,2,3,5,1]) c.accept(token)
  assert.equal(c.complete, true)
  assert.equal(tokenAllowed(c.computeMask(), 6), true)
  c.dispose()
})

test("EOS override supports runtimes that keep EOS on model configuration", async () => {
  const c = await createConstraint({ schema: { const: null }, tokenizer: { tokens: tokenizer.tokens }, eosTokenId: EOS })
  accept(c, "null")
  assert.equal(c.complete, true)
  c.dispose()
})

test("unsupported assertions fail explicitly", async () => {
  await assert.rejects(createConstraint({ schema: { type: "object", unevaluatedProperties: false }, tokenizer }), /unsupported|Unsupported/)
})

test("invalid tokenizers and dead ends fail explicitly", async () => {
  await assert.rejects(createConstraint({ schema, tokenizer: { tokens: [[120], []], eos_token_id: 1 } }).then(c => c.computeMask()), /complete|dead|EOS|inspect/)
  await assert.rejects(createConstraint({ schema, tokenizer: { tokens: [[120]] } }), /EOS|tokenizer/)
})


test("numeric roots preserve valid extensions after an accepting prefix", async () => {
  const c = await createConstraint({ schema: { type: "integer", minimum: 1, maximum: 123 }, tokenizer })
  accept(c, "1")
  assert.equal(c.complete, true)
  assert.ok(tokenAllowed(c.computeMask(), 50), "12 remains possible after the accepted prefix 1")
  accept(c, "23")
  assert.equal(c.complete, true)
  c.accept(EOS);c.dispose()
  const bounded = await createConstraint({ schema: { type: "number", minimum: 42, maximum: 43 }, tokenizer })
  accept(bounded, "4")
  assert.equal(bounded.complete, false)
  assert.equal(tokenAllowed(bounded.computeMask(), EOS), false)
  accept(bounded, "2.5")
  assert.equal(bounded.complete, true)
  bounded.dispose()
})

test("compact punctuation forbids whitespace loops outside strings while preserving string spaces", async () => {
  const source = { type: "object", properties: { answer: { const: "a b" } }, required: ["answer"], additionalProperties: false, "x-guidance": { whitespace_flexible: true, key_separator: ": ", item_separator: ", " } }
  const before = JSON.stringify(source)
  const c = await createConstraint({ schema: source, tokenizer })
  assert.equal(tokenAllowed(c.computeMask(), 32), false)
  accept(c, "{")
  assert.equal(tokenAllowed(c.computeMask(), 32), false)
  assert.equal(tokenAllowed(c.computeMask(), 10), false)
  accept(c, '"answer"')
  assert.equal(tokenAllowed(c.computeMask(), 32), false)
  accept(c, ":")
  assert.equal(tokenAllowed(c.computeMask(), 32), false)
  accept(c, '"a b"}')
  assert.equal(c.complete, true)
  assert.equal(JSON.stringify(source), before, "caller schema is never mutated")
  c.dispose()
  const array = await createConstraint({ schema: { const: ["a b", "c d"] }, tokenizer })
  accept(array, '["a b",')
  assert.equal(tokenAllowed(array.computeMask(), 32), false)
  accept(array, '"c d"]')
  assert.equal(array.complete, true)
  array.dispose()
})

test("finite tool-routing and argument enums reject wrong prefixes before closing quotes", async () => {
  const route = await createConstraint({ schema: { type: "object", properties: { action: { type: "string", enum: ["answer", "lookup_weather"] } }, required: ["action"], additionalProperties: false }, tokenizer })
  accept(route, '{"action":"')
  for (const invalid of ["x", " ", "L", "0"]) assert.equal(tokenAllowed(route.computeMask(), invalid.charCodeAt(0)), false)
  assert.equal(tokenAllowed(route.computeMask(), "a".charCodeAt(0)), true)
  assert.equal(tokenAllowed(route.computeMask(), "l".charCodeAt(0)), true)
  const tool = route.clone()
  accept(route, 'answer')
  assert.equal(tokenAllowed(route.computeMask(), "x".charCodeAt(0)), false)
  accept(route, '"}')
  accept(tool, 'lookup_')
  assert.equal(tokenAllowed(tool.computeMask(), "z".charCodeAt(0)), false)
  accept(tool, 'weather"}')
  assert.equal(route.complete, true); assert.equal(tool.complete, true)
  route.dispose();tool.dispose()
  const args = await createConstraint({ schema: { type: "object", properties: { city: { type: "string", enum: ["London"] } }, required: ["city"], additionalProperties: false }, tokenizer })
  accept(args, '{"city":"')
  assert.equal(tokenAllowed(args.computeMask(), "B".charCodeAt(0)), false)
  assert.equal(tokenAllowed(args.computeMask(), "l".charCodeAt(0)), false)
  accept(args, 'London"}')
  assert.equal(args.complete, true)
  args.dispose()
})

test("parsed structured values validate schemas without a model tokenizer", async () => {
  const { validateStructuredValue } = await import("./index.js")
  assert.equal(await validateStructuredValue(schema, { moveId: "b6-a5" }), true)
  assert.equal(await validateStructuredValue(schema, { moveId: "illegal" }), false)
  assert.equal(await validateStructuredValue(schema, { moveId: "b6-a5", extra: 1 }), false)
  assert.equal(await validateStructuredValue(schema, {}), false)
  const args = { type: "object", properties: { city: { type: "string", minLength: 3, pattern: "^[A-Z]" }, count: { type: "integer", minimum: 1, maximum: 5 }, tags: { type: "array", items: { enum: ["café", "☃"] }, uniqueItems: true } }, required: ["city", "count"], additionalProperties: false }
  assert.equal(await validateStructuredValue(args, { city: "London", count: 2, tags: ["café", "☃"] }), true)
  assert.equal(await validateStructuredValue(args, { city: "london", count: 2 }), false)
  assert.equal(await validateStructuredValue(args, { city: "London", count: 7 }), false)
  assert.equal(await validateStructuredValue(args, { city: "London", count: 2, tags: ["café", "café"] }), false)
  assert.equal(await validateStructuredValue({ type: "integer", minimum: 12, maximum: 15 }, 12), true)
  assert.equal(await validateStructuredValue({ type: "integer", minimum: 12, maximum: 15 }, 1), false)
  await assert.rejects(validateStructuredValue({ type: "object", unevaluatedProperties: false }, {}), /unsupported|Unsupported/)
})

test("parsed value validation rejects non-JSON coercions and cycles", async () => {
  const { validateStructuredValue } = await import("./index.js")
  const cycle: Record<string, unknown> = {}; cycle.self = cycle
  for (const value of [undefined, NaN, Infinity, 1n, Symbol("value"), () => 1, new Date(), { x: undefined }, [undefined], Array(1), cycle]) {
    assert.equal(await validateStructuredValue(true, value), false)
  }
  assert.equal(await validateStructuredValue(true, { note: "a b </think> café" }), true)
  assert.equal(await validateStructuredValue(false, null), false)
})

test("parsed native arguments validate independent of property insertion order and string whitespace", async () => {
  const { validateStructuredValue } = await import("./index.js")
  const properties = { a: { type: "string" }, b: { type: "string" } }
  const schema = { type: "object", properties, required: ["a", "b"], additionalProperties: false }
  const original = { b: "second\nline\t☃", a: "  leading and trailing  " }
  assert.equal(await validateStructuredValue(schema, original), true)
  assert.equal(await validateStructuredValue(schema, { a: original.a, b: original.b }), true)
  assert.deepEqual(original, { b: "second\nline\t☃", a: "  leading and trailing  " })
  const constant = { const: { a: "a b", b: "c d" } }
  assert.equal(await validateStructuredValue(constant, { b: "c d", a: "a b" }), true)
  assert.equal(await validateStructuredValue({ enum: [{ a: "a b", b: "c d" }] }, { b: "c d", a: "a b" }), true)
})


test("explicit EOS replaces existing camel-case and snake-case metadata consistently", async () => {
  const source = { tokens: [...tokenizer.tokens, []], eosTokenId: 256, eos_token_id: 256, special_token_ids: [256, 257] }
  for (const eos of [257, 256, 257]) {
    const c = await createConstraint({ schema: { const: null }, tokenizer: source, eosTokenId: eos })
    accept(c, "null")
    assert.equal(c.eosTokenId, eos)
    assert.equal(c.complete, true)
    assert.equal(tokenAllowed(c.computeMask(), eos), true)
    assert.equal(tokenAllowed(c.computeMask(), eos === 256 ? 257 : 256), false)
    c.accept(eos)
    c.dispose()
  }
  assert.equal(source.eosTokenId, 256)
  assert.equal(source.eos_token_id, 256)
})
