import assert from "node:assert/strict"
import { test } from "node:test"
import { createConstraint, inlineRefs, prunableSchema, tokenAllowed, validateStructuredValue, type Constraint } from "./index.js"

const EOS = 256
const encoder = new TextEncoder()
// Every byte as a token, then tokens of several characters as models have them.
const EXTRA = ["abc", "abcd", '"', 'x"', "ab\"}", "é", "\\n", "\\u00e9"]
const tokenizer = {
  tokens: [...Array.from({ length: 257 }, (_, id) => (id === EOS ? [] : [id])), ...EXTRA.map((text) => [...encoder.encode(text)])],
  eos_token_id: EOS,
  special_token_ids: [EOS]
}
const extra = (text: string) => 257 + EXTRA.indexOf(text)
const byte = (char: string) => char.charCodeAt(0)
const allows = (c: Constraint, token: number) => tokenAllowed(c.computeMask(), token)
function accept(c: Constraint, text: string) {
  for (const token of encoder.encode(text)) {
    assert.ok(allows(c, token), `byte ${token} of ${JSON.stringify(text)} is allowed`)
    c.accept(token)
  }
}
const closed = (properties: Record<string, unknown>) => ({ type: "object", properties, required: Object.keys(properties), additionalProperties: false })

test("maxLength: no character past it, from a token of one byte or several; the closing quote stays", async () => {
  const c = await createConstraint({ schema: closed({ s: { type: "string", maxLength: 3 } }), tokenizer })
  accept(c, '{"s": "')
  assert.equal(allows(c, extra("abc")), true)
  assert.equal(allows(c, extra("abcd")), false, "four characters into three")
  accept(c, "ab")
  assert.equal(allows(c, byte("c")), true)
  assert.equal(allows(c, extra("abc")), false)
  assert.equal(allows(c, extra('x"')), true, "a third character, then the quote")
  accept(c, "c")
  assert.equal(allows(c, byte("d")), false)
  assert.equal(allows(c, extra("\\n")), false, "an escape is a character too")
  assert.equal(allows(c, byte('"')), true)
  accept(c, '"}')
  assert.equal(c.complete, true)
  c.dispose()
})

test("minLength: no closing quote before it, alone or at the end of a token", async () => {
  const c = await createConstraint({ schema: closed({ s: { type: "string", minLength: 2 } }), tokenizer })
  accept(c, '{"s": "')
  assert.equal(allows(c, byte('"')), false, "an empty string")
  accept(c, "a")
  assert.equal(allows(c, byte('"')), false)
  assert.equal(allows(c, extra('x"')), true, "the second character, then the quote")
  assert.equal(allows(c, extra('ab"}')), true)
  accept(c, "b")
  assert.equal(allows(c, byte('"')), true)
  c.dispose()
})

test("characters, not bytes: a two-byte character and an escape each count once", async () => {
  const c = await createConstraint({ schema: closed({ s: { type: "string", maxLength: 2 } }), tokenizer })
  accept(c, '{"s": "')
  assert.equal(allows(c, extra("é")), true)
  c.accept(extra("é"))
  assert.equal(allows(c, extra("\\u00e9")), true, "\\u00e9 is one character")
  c.accept(extra("\\u00e9"))
  assert.equal(allows(c, byte("a")), false)
  accept(c, '"}')
  assert.equal(c.complete, true)
  c.dispose()
})

test("lengths are held through a $ref, items and a nullable choice", async () => {
  const schema = {
    ...closed({ tags: { type: "array", items: { $ref: "#/$defs/tag" } }, note: { anyOf: [{ type: "string", maxLength: 1 }, { type: "null" }] } }),
    $defs: { tag: { type: "string", maxLength: 2 } }
  }
  const c = await createConstraint({ schema, tokenizer })
  accept(c, '{"tags": ["ab')
  assert.equal(allows(c, byte("c")), false)
  accept(c, '", "x')
  assert.equal(allows(c, byte("y")), true)
  accept(c, '"], "note": "z')
  assert.equal(allows(c, byte("z")), false)
  c.dispose()
})

test("where the document can't be followed plainly (allOf), lengths stay with llguidance: still held", async () => {
  const schema = closed({ s: { allOf: [{ type: "string" }, { maxLength: 2 }] } })
  assert.equal(JSON.stringify(prunableSchema(schema, { lengths: true }).schema).includes("maxLength"), true)
  assert.equal(await validateStructuredValue(schema, { s: "ab" }), true)
  assert.equal(await validateStructuredValue(schema, { s: "abc" }), false)
  // Followed ones go.
  assert.equal(JSON.stringify(prunableSchema(closed({ s: { type: "string", maxLength: 2 } }), { lengths: true }).schema).includes("maxLength"), false)
})

test("inlineRefs writes local references in place; recursive ones, and ones with siblings, stay", () => {
  const tag = { type: "string", enum: ["a", "b"] }
  assert.deepEqual(
    inlineRefs({ type: "object", properties: { x: { $ref: "#/$defs/tag", description: "X" }, default: { $ref: "#/definitions/tag" } }, $defs: { tag }, definitions: { tag } }),
    { type: "object", properties: { x: { ...tag, description: "X" }, default: tag } }
  )
  const tree = { type: "object", properties: { children: { type: "array", items: { $ref: "#/$defs/node" } } } }
  const recursive = { type: "object", properties: { root: { $ref: "#/$defs/node" } }, $defs: { node: tree } }
  const out = inlineRefs(recursive) as { properties: { root: { properties: { children: { items: unknown } } } }; $defs: unknown }
  // Written out once; the reference inside it, back to itself, stays, with its definitions.
  assert.deepEqual(out.properties.root.properties.children.items, { $ref: "#/$defs/node" })
  assert.deepEqual(out.$defs, { node: tree })
  // A root that is a reference has its definitions beside it: it stays as it is.
  const rooted = { $ref: "#/$defs/node", $defs: { node: tree } }
  assert.deepEqual(inlineRefs(rooted), rooted)
  const sibling = { properties: { x: { $ref: "#/$defs/tag", minLength: 1 } }, $defs: { tag } }
  assert.deepEqual(inlineRefs(sibling), sibling)
  // Values are values: a $ref in an enum or a default isn't followed.
  const values = { properties: { x: { enum: [{ $ref: "#/$defs/tag" }], default: { $ref: "#/$defs/tag" } } }, $defs: { tag } }
  assert.deepEqual(inlineRefs(values), { properties: values.properties })
})

test("where the walker can't read the schema, lengths stay with llguidance: allOf beside, allOf above, a key not spelled plainly", async () => {
  const cases = [
    closed({ s: { type: "string", maxLength: 2, allOf: [{}] } }),
    { ...closed({ s: { type: "string", maxLength: 2 } }), allOf: [{}] },
    closed({ é: { type: "string", maxLength: 2 } })
  ]
  for (const schema of cases) {
    assert.equal(JSON.stringify(prunableSchema(schema, { lengths: true }).schema).includes('"maxLength":2'), true, JSON.stringify(schema))
  }
  // And the mask still holds them: the base's behavior, a third character refused.
  const c = await createConstraint({ schema: cases[0], tokenizer })
  accept(c, '{"s": "ab')
  assert.equal(allows(c, byte("c")), false)
  c.dispose()
})

test("a surrogate pair of \\u escapes is one character, also written a byte at a time; raw UTF-8 too", async () => {
  const c = await createConstraint({ schema: closed({ s: { type: "string", maxLength: 1 } }), tokenizer })
  accept(c, '{"s": "\\uD83D')
  // At maxLength, only the low half may follow.
  accept(c, "\\u")
  assert.equal(allows(c, byte("0")), false, "\\u0… isn't a low surrogate")
  accept(c, "DE00")
  assert.equal(allows(c, byte("a")), false)
  accept(c, '"}')
  assert.equal(c.complete, true)
  c.dispose()

  const raw = await createConstraint({ schema: closed({ s: { type: "string", maxLength: 1 } }), tokenizer })
  accept(raw, '{"s": "😀"}')
  assert.equal(raw.complete, true)
  raw.dispose()

  // After one ordinary character at maxLength, no escape fits at all.
  const full = await createConstraint({ schema: closed({ s: { type: "string", maxLength: 1 } }), tokenizer })
  accept(full, '{"s": "a')
  assert.equal(allows(full, byte("\\")), false)
  full.dispose()
})

test("minLength counts a surrogate pair once", async () => {
  const c = await createConstraint({ schema: closed({ s: { type: "string", minLength: 2 } }), tokenizer })
  accept(c, '{"s": "\\ud83d\\ude00')
  assert.equal(allows(c, byte('"')), false, "one character so far")
  accept(c, "x")
  assert.equal(allows(c, byte('"')), true)
  c.dispose()
})

test("a $ref to what isn't the schema's own definition (Object.prototype's) stays, and fails to compile as before", async () => {
  const schema = closed({ x: { $ref: "#/$defs/__proto__" } })
  const withDefs = { ...schema, $defs: {} }
  assert.deepEqual((inlineRefs(withDefs) as { properties: unknown }).properties, { x: { $ref: "#/$defs/__proto__" } })
  await assert.rejects(createConstraint({ schema: withDefs, tokenizer }), (e: Error & { code?: string }) => e.code === "unsupported-schema")
})
