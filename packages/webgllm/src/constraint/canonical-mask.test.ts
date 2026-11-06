import assert from "node:assert/strict"
import { test } from "node:test"
import { canonicalAsciiObject, canonicalTokenMask } from "./canonical-mask.js"
import { createConstraint, tokenAllowed, type Constraint } from "./index.js"

const EOS = 256
const tokenizer = {
  tokens: [...Array.from({ length: 257 }, (_, id) => id === EOS ? [] : [id]), [34, 92], [240, 159]],
  eos_token_id: EOS,
  special_token_ids: [EOS]
}
const schema = { type: "object", properties: { city: { type: "string", enum: ["Paris", "Tokyo", "London", "Berlin"] } }, required: ["city"], additionalProperties: false }
const accept = (constraint: Constraint, text: string) => { for (const byte of new TextEncoder().encode(text)) constraint.accept(byte) }

test("finite printable string objects reject impossible escape and UTF-8 prefixes before accepting them", async () => {
  const c = await createConstraint({ schema, tokenizer })
  accept(c, "{")
  assert.equal(tokenAllowed(c.computeMask(), 257), false, "quote plus backslash cannot start an escaped key")
  assert.throws(() => c.accept(257), /violates/)
  accept(c, '"city":"P')
  assert.equal(tokenAllowed(c.computeMask(), 225), false, "the E1 lead byte cannot extend an ASCII enum")
  assert.equal(tokenAllowed(c.computeMask(), 92), false, "escaped wrong characters cannot bypass prefix checks")
  assert.throws(() => c.accept(225), /violates/)
  accept(c, 'aris"}')
  assert.equal(c.complete, true)
  assert.equal(tokenAllowed(c.computeMask(), EOS), true)
  c.dispose()
})

test("forks retain immutable canonical masks, independent prefixes, and EOS", async () => {
  const c = await createConstraint({ schema, tokenizer })
  accept(c, '{"city":"')
  const originalMask = c.computeMask().slice(), clone = c.clone()
  assert.deepEqual(clone.computeMask(), originalMask)
  accept(clone, 'London"}')
  assert.deepEqual(c.computeMask(), originalMask)
  accept(c, 'Paris"}')
  assert.equal(c.complete, true)
  assert.equal(clone.complete, true)
  c.accept(EOS); clone.accept(EOS)
  c.dispose(); clone.dispose()
})

test("recognition leaves Unicode, escaped literals, unknown assertions and other schemas unchanged", async () => {
  const unsupported = [true, { ...schema, "x-guidance": { whitespace_flexible: false } }, {}, { type: "string" }, { ...schema, additionalProperties: true }, { ...schema, required: ["missing"] }, { ...schema, patternProperties: { ".*": { const: "x" } } }, { ...schema, properties: { city: { enum: ["Paris"], minLength: 20 } } }, { ...schema, properties: { city: { type: "string" } } }, { ...schema, properties: { city: { enum: [] } } }, { ...schema, properties: { city: { enum: ["Paris"], const: "Paris" } } }]
  for (const source of unsupported) assert.equal(canonicalAsciiObject(source), false)
  for (const [key, value] of [["city", "café ☃"], ["city", 'a"b'], ["city", "a\\b"], ["city", "a\nb"], ["café", "Paris"]]) {
    const source = { type: "object", properties: { [key]: { const: value } }, required: [key], additionalProperties: false }
    assert.equal(canonicalAsciiObject(source), false)
    const c = await createConstraint({ schema: source, tokenizer })
    accept(c, JSON.stringify({ [key]: value }))
    assert.equal(c.complete, true)
    c.dispose()
  }
  const c = await createConstraint({ schema: { type: "string" }, tokenizer })
  accept(c, '"')
  assert.equal(tokenAllowed(c.computeMask(), 225), true, "unbounded strings retain the unchanged backend")
  c.dispose()
})

test("mask conversion caches exact bytes and bypasses vocabularies lacking canonical byte coverage", async () => {
  const mask = canonicalTokenMask(tokenizer, tokenizer.tokens.length, EOS)
  assert.ok(mask)
  assert.equal(canonicalTokenMask(tokenizer, tokenizer.tokens.length, EOS), mask)
  for (const token of [92, 225, 257, 258]) assert.equal(tokenAllowed(mask, token), false)
  assert.equal(tokenAllowed(mask, EOS), true)
  assert.equal(canonicalTokenMask({}, 2, 1), undefined)
  assert.equal(canonicalTokenMask({ tokens: [[34, 92, 117, 48, 48, 54, 49], []], eos_token_id: 1 }, 2, 1), undefined, "an escape-only vocabulary must keep its valid values")
  const escapeOnly = { tokens: [[...new TextEncoder().encode('{"city":"\\u0050aris"}')], []], eos_token_id: 1, special_token_ids: [1] }
  const c = await createConstraint({ schema, tokenizer: escapeOnly })
  c.accept(0)
  assert.equal(c.complete, true, "a legitimate escaped-only value remains available through the unchanged backend")
  c.accept(1); c.dispose()
  const emptyToken = { ...tokenizer, tokens: [...tokenizer.tokens, []] }
  const noEmpty = canonicalTokenMask(emptyToken, emptyToken.tokens.length, EOS)
  assert.ok(noEmpty)
  assert.equal(tokenAllowed(noEmpty, emptyToken.tokens.length - 1), false)
  assert.equal(tokenAllowed(noEmpty, EOS), true)
  const specialA = { ...tokenizer, special_token_ids: [EOS, 65] }
  assert.equal(canonicalTokenMask(specialA, tokenizer.tokens.length, EOS), undefined, "special bytes cannot prove normal coverage")
})

test("HF ByteLevel metadata uses byte-level spaces and added-token byte overrides", () => {
  const bytes: number[] = []
  for (let byte = 33; byte <= 126; byte++) bytes.push(byte)
  for (let byte = 161; byte <= 172; byte++) bytes.push(byte)
  for (let byte = 174; byte <= 255; byte++) bytes.push(byte)
  const characters = [...bytes]; let extra = 0
  for (let byte = 0; byte < 256; byte++) if (!bytes.includes(byte)) { bytes.push(byte); characters.push(256 + extra++) }
  const vocab = Object.fromEntries(bytes.map((byte, index) => [String.fromCodePoint(characters[index]), byte]))
  const source = { tokenizerJSON: { model: { vocab }, decoder: { type: "ByteLevel" }, added_tokens: [{ id: 65, content: "café", special: false }, { id: EOS, content: "<eos>", special: true }, { id: 257, content: "A", special: false }] }, eos_token_id: EOS }
  const mask = canonicalTokenMask(source, 258, EOS)
  assert.ok(mask)
  assert.equal(tokenAllowed(mask, 32), true)
  assert.equal(tokenAllowed(mask, 65), false)
  assert.equal(tokenAllowed(mask, 257), true)
  assert.equal(tokenAllowed(mask, 92), false)
  for (const decoder of [{ type: "Sequence", decoders: [{ type: "ByteLevel" }, { type: "Replace", pattern: "A", content: "B" }] }, { type: "Fuse" }, { type: "ByteLevel", unknownTransform: true }]) {
    assert.equal(canonicalTokenMask({ ...source, tokenizerJSON: { ...source.tokenizerJSON, decoder, pre_tokenizer: { type: "ByteLevel" } } }, 258, EOS), undefined)
  }
})

test("every canonical prefix mask matches an independent finite-language oracle", async () => {
  const fixtures = [
    { schema, docs: ["Paris", "Tokyo", "London", "Berlin"].map(city => JSON.stringify({ city })) },
    { schema: { type: "object", properties: { a: { enum: ["", "x", "xyz"] }, b: { const: "</think> <tool_call>" } }, required: ["a", "b"], additionalProperties: false }, docs: ["", "x", "xyz"].flatMap(a => [JSON.stringify({ a, b: "</think> <tool_call>" }), JSON.stringify({ b: "</think> <tool_call>", a })]) },
    { schema: { type: "object", properties: { a: { enum: ["x", "y"] }, b: { const: "z" } }, additionalProperties: false }, docs: ["{}", '{"a":"x"}', '{"a":"y"}', '{"b":"z"}', '{"a":"x","b":"z"}', '{"a":"y","b":"z"}', '{"b":"z","a":"x"}', '{"b":"z","a":"y"}'] }
  ]
  for (const fixture of fixtures) {
    assert.equal(canonicalAsciiObject(fixture.schema), true)
    const prefixes = new Set(fixture.docs.flatMap(doc => Array.from({ length: doc.length + 1 }, (_, length) => doc.slice(0, length))))
    for (const prefix of prefixes) {
      const c = await createConstraint({ schema: fixture.schema, tokenizer })
      accept(c, prefix)
      const mask = c.computeMask()
      for (let token = 0; token < tokenizer.tokens.length; token++) {
        const expected = token === EOS ? fixture.docs.includes(prefix) :
          tokenizer.tokens[token].length > 0 && tokenizer.tokens[token].every(byte => byte >= 32 && byte <= 126 && byte !== 92) &&
          fixture.docs.some(doc => doc.startsWith(prefix + String.fromCharCode(...tokenizer.tokens[token])))
        assert.equal(tokenAllowed(mask, token), expected, `token ${token} at ${JSON.stringify(prefix)}`)
      }
      c.dispose()
    }
  }
})
