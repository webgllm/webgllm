import assert from "node:assert/strict"
import { test } from "node:test"
import { jsonTokens, nextStringState, type StringState } from "./whitespace.js"

const whitespaceTokenIds = (tokenizer: object) => jsonTokens(tokenizer)?.whitespace ?? null

test("exact token bytes: tokens of only JSON whitespace, not special ones", () => {
  const tokens = [[32], [10, 32, 32], [9], [13, 10], [32, 34], [65], [], [32]]
  assert.deepEqual(whitespaceTokenIds({ tokens, special_token_ids: [7] }), [0, 1, 2, 3])
})

test("HF ByteLevel vocabularies, alone or in a Sequence, with added tokens", () => {
  const vocab = { "Ġ": 0, "ĠĠ": 1, "Ċ": 2, "ĉ": 3, "č": 4, "ĠĠĠĠĊ": 5, "Ġa": 6, "a": 7, "Ġ\"": 8, "ĊĊ": 9 }
  const added = [{ id: 9, content: "<pad>", special: true }, { id: 10, content: "\n\n\n", special: false }, { id: 11, content: "<eos>", special: true }]
  const ids = (decoder: object) => whitespaceTokenIds({ tokenizerJSON: { model: { vocab }, decoder, added_tokens: added } })
  assert.deepEqual(ids({ type: "ByteLevel" }), [0, 1, 2, 3, 4, 5, 10])
  assert.deepEqual(ids({ type: "Sequence", decoders: [{ type: "ByteLevel", add_prefix_space: true }] }), [0, 1, 2, 3, 4, 5, 10])
})

test("null when the bytes can't be read", () => {
  assert.equal(whitespaceTokenIds({}), null)
  assert.equal(whitespaceTokenIds({ tokenizerJSON: { model: { vocab: { "▁": 0 } }, decoder: { type: "Metaspace" } } }), null)
  assert.equal(whitespaceTokenIds({ tokenizerJSON: { model: { vocab: {} }, decoder: { type: "Sequence", decoders: [{ type: "ByteLevel" }, { type: "Fuse" }] } } }), null)
})

test("tokens move in and out of strings, through escapes and across boundaries", () => {
  const texts = [' "', '",', '":', '\\"', '\\', '"', 'a', ' ', '"a"']
  const exact = jsonTokens({ tokens: texts.map((t) => [...new TextEncoder().encode(t)]) })!
  // ByteLevel spells quote and backslash as themselves.
  const byteLevel = jsonTokens({ tokenizerJSON: { model: { vocab: Object.fromEntries(texts.map((t, id) => [t.replace(/ /g, "Ġ"), id])) }, decoder: { type: "ByteLevel" } } })!
  for (const tokens of [exact, byteLevel]) {
    const walk = (ids: number[]) => ids.reduce<StringState>((state, id) => nextStringState(tokens, id, state), 0)
    assert.equal(walk([0]), 1, "a space then a quote opens a string")
    assert.equal(walk([0, 7, 6, 1]), 0, "a quote then a comma closes it")
    assert.equal(walk([5, 2]), 0, "a key closes on its colon")
    assert.equal(walk([5, 3]), 1, "an escaped quote stays inside")
    assert.equal(walk([5, 4]), 2, "a backslash escapes what follows")
    assert.equal(walk([5, 4, 4]), 1, "an escaped backslash")
    assert.equal(walk([5, 4, 4, 5]), 0, "then a quote closes")
    assert.equal(walk([8, 7]), 0, "a whole string in one token")
    assert.deepEqual(tokens.whitespace, [7])
  }
})
