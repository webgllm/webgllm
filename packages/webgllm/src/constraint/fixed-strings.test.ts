import assert from "node:assert/strict"
import { test } from "node:test"
import { createConstraint, tokenAllowed, type Constraint } from "./index.js"

const EOS = 256
// Every byte as a token, then tokens of several bytes as models have them.
const EXTRA = ['{"', "cut", "cuts", '\\"', '":', '","\\', '": "é', "\\n", '"}']
const encoder = new TextEncoder()
const tokenizer = {
  tokens: [...Array.from({ length: 257 }, (_, id) => (id === EOS ? [] : [id])), ...EXTRA.map((text) => [...encoder.encode(text)])],
  eos_token_id: EOS,
  special_token_ids: [EOS]
}
const extra = (text: string) => 257 + EXTRA.indexOf(text)
const BACKSLASH = 92
/** The first byte of a two-byte UTF-8 character, alone. */
const PARTIAL = 0xc3
const byte = (char: string) => char.charCodeAt(0)
const closed = (properties: Record<string, unknown>) => ({ type: "object", properties, required: Object.keys(properties), additionalProperties: false })
const strings = { type: "array", items: { type: "string" } }

function accept(c: Constraint, text: string) {
  for (const token of encoder.encode(text)) {
    assert.ok(tokenAllowed(c.computeMask(), token), `byte ${token} of ${JSON.stringify(text)} is allowed`)
    c.accept(token)
  }
}
const allows = (c: Constraint, token: number) => tokenAllowed(c.computeMask(), token)
/** The single-byte tokens allowed, as text (bytes past ASCII as \xNN). */
const allowed = (c: Constraint) => {
  const out: string[] = []
  for (let id = 0; id < 256; id++) if (allows(c, id)) out.push(id < 0x80 ? String.fromCharCode(id) : `\\x${id.toString(16)}`)
  return out.join("")
}

test("a key the schema fixes takes its next character only: no escapes, no partial UTF-8", async () => {
  // Reported with LFM3 1.7B: {"cut\"\"\"\"… to the output limit, and "No token can complete the requested output schema."
  const c = await createConstraint({ schema: closed({ cuts: { type: "string" } }), tokenizer })
  accept(c, '{"')
  assert.equal(allowed(c), "c")
  assert.equal(allows(c, extra("cut")), true)
  assert.equal(allows(c, extra("cuts")), true)
  assert.equal(allows(c, extra('\\"')), false)
  c.accept(extra("cut"))
  // The model wants the key to end here. It can't: the one way on is the "s".
  assert.equal(allowed(c), "s")
  for (const token of [BACKSLASH, PARTIAL, extra('\\"'), extra("\\n"), byte('"')]) assert.equal(allows(c, token), false, `token ${token} after {"cut`)
  accept(c, 's": "')
  // The value is free: its escapes and its characters past ASCII stay.
  for (const token of [BACKSLASH, PARTIAL, extra('\\"'), extra("\\n")]) assert.equal(allows(c, token), true, `token ${token} in the value`)
  accept(c, 'café \\"x\\" \\u00e9"}')
  assert.equal(c.complete, true)
  c.dispose()
})

test("an enum or const value beside free strings is fixed too, in objects and arrays", async () => {
  const schema = closed({
    intent: { type: "string", enum: ["trim", "keep"] },
    kind: { const: "cut" },
    tags: { type: "array", items: { enum: ["intro", "outro", null] } },
    note: { type: "string" }
  })
  const c = await createConstraint({ schema, tokenizer })
  accept(c, '{"intent": "')
  assert.equal(allowed(c), "kt")
  accept(c, "tri")
  assert.equal(allowed(c), "m")
  accept(c, 'm","kind": "')
  assert.equal(allowed(c), "c")
  accept(c, 'cut","tags": ["')
  assert.equal(allowed(c), "io")
  accept(c, 'intro","ou')
  assert.equal(allowed(c), "t")
  accept(c, 'tro",null],"note": "')
  assert.equal(allows(c, BACKSLASH), true)
  assert.equal(allows(c, PARTIAL), true)
  accept(c, 'a \\\\ ☃"}')
  assert.equal(c.complete, true)
  c.dispose()
})

test("a required key the model skipped is still added, and nothing else is", async () => {
  // Reported: "replace": []\n \t \t, "m\n\n\n… without ever closing.
  const c = await createConstraint({ schema: closed({ keep: strings, merge: strings, replace: strings }), tokenizer })
  accept(c, '{"keep": [],"replace": []')
  assert.equal(allows(c, byte("}")), false)
  accept(c, ' , "')
  assert.equal(allowed(c), "m")
  accept(c, "m")
  assert.equal(allowed(c), "e")
  accept(c, 'erge": []}')
  assert.equal(c.complete, true)
  c.dispose()
})

test("a token of several bytes is judged by where each byte lands", async () => {
  const c = await createConstraint({ schema: closed({ a: { type: "string" }, b: { type: "string" } }), tokenizer })
  accept(c, '{"a')
  // Closes the key and opens the value, where the character past ASCII is content.
  assert.equal(allows(c, extra('": "é')), true)
  c.accept(extra('": "é'))
  assert.equal(allows(c, extra('\\"')), true)
  // Closes the value and opens the next key with a backslash in it.
  assert.equal(allows(c, extra('","\\')), false)
  accept(c, '","b": "x')
  assert.equal(allows(c, extra('"}')), true)
  c.accept(extra('"}'))
  assert.equal(c.complete, true)
  c.dispose()
})

test("nested objects, nullable values and local references are followed", async () => {
  const schema = {
    $defs: { step: closed({ op: { enum: ["cut", "keep"] }, at: { type: "string" } }) },
    ...closed({
      steps: { type: "array", items: { $ref: "#/$defs/step" } },
      mode: { anyOf: [{ type: "string", enum: ["fast", "slow"] }, { type: "null" }] }
    })
  }
  const c = await createConstraint({ schema, tokenizer })
  accept(c, '{"steps": [{"')
  assert.equal(allowed(c), "ao")
  accept(c, 'op": "')
  assert.equal(allowed(c), "ck")
  accept(c, 'keep","at": "')
  assert.equal(allows(c, BACKSLASH), true)
  accept(c, '\\t"}],"mode": "')
  assert.equal(allowed(c), "fs")
  accept(c, 'slow"}')
  assert.equal(c.complete, true)
  c.dispose()
})

test("strings the schema doesn't fix are left as they were", async () => {
  // An open object's keys are the model's to write.
  const open = await createConstraint({ schema: { type: "object", properties: { a: { type: "string" } } }, tokenizer })
  accept(open, '{"')
  assert.equal(allows(open, BACKSLASH), true)
  assert.equal(allows(open, PARTIAL), true)
  accept(open, 'café": 1, "a": "\\n"}')
  assert.equal(open.complete, true)
  open.dispose()
  // A key that isn't plain ASCII has no one spelling to hold it to.
  const named = await createConstraint({ schema: closed({ "café": { type: "string" } }), tokenizer })
  accept(named, '{"caf')
  assert.equal(allows(named, PARTIAL), true)
  accept(named, 'é": "x"}')
  assert.equal(named.complete, true)
  named.dispose()
  // Nor an enum with a value that needs an escape.
  const quoted = await createConstraint({ schema: closed({ a: { enum: ['say "hi"', "x"] } }), tokenizer })
  accept(quoted, '{"a": "say ')
  assert.equal(allows(quoted, BACKSLASH), true)
  accept(quoted, '\\"hi\\""}')
  assert.equal(quoted.complete, true)
  quoted.dispose()
})

test("a clone carries where it is in the document", async () => {
  const a = await createConstraint({ schema: closed({ cuts: { type: "string" } }), tokenizer })
  accept(a, '{"cut')
  const b = a.clone()
  accept(a, 's": "')
  assert.equal(allows(a, BACKSLASH), true)
  assert.equal(allows(b, BACKSLASH), false)
  assert.equal(allowed(b), "s")
  a.dispose()
  b.dispose()
})

test("HF ByteLevel vocabularies: the same, from their pieces", async () => {
  const pieces = [...Array.from({ length: 94 }, (_, i) => String.fromCharCode(33 + i)), "Ġ", "Ã", "©", '\\"', "cut", '{"', '":', "Ġ\""]
  const eos = pieces.length
  const bridge = {
    tokenizerJSON: {
      model: { type: "BPE", vocab: Object.fromEntries(pieces.map((piece, id) => [piece, id])) },
      decoder: { type: "ByteLevel" },
      added_tokens: [{ id: eos, content: "<eos>", special: true }]
    },
    eos_token_id: eos,
    special_token_ids: [eos]
  }
  const id = (piece: string) => pieces.indexOf(piece)
  const c = await createConstraint({ schema: closed({ cuts: { type: "string" } }), tokenizer: bridge })
  c.accept(id('{"'))
  c.accept(id("cut"))
  for (const piece of ["\\", '\\"', "Ã", '"']) assert.equal(allows(c, id(piece)), false, `piece ${piece} after {"cut`)
  assert.equal(allows(c, id("s")), true)
  c.accept(id("s"))
  c.accept(id('":'))
  c.accept(id("Ġ\""))
  for (const piece of ["\\", '\\"', "Ã"]) assert.equal(allows(c, id(piece)), true, `piece ${piece} in the value`)
  c.accept(id("Ã"))
  c.accept(id("©"))
  c.accept(id('"'))
  c.accept(id("}"))
  assert.equal(c.complete, true)
  c.dispose()
})
