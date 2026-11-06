import assert from "node:assert/strict"
import { test } from "node:test"
import { whitespaceTokenIds } from "./whitespace.js"

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
