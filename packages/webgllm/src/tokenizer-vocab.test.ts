import assert from "node:assert/strict"
import test from "node:test"
import { loadTokenizer } from "./tokenizer.js"

test("vocabulary size includes sparse added-token IDs without copying the vocabulary Map", () => {
  const tokenizer = loadTokenizer({
    tokenizerJson: {
      model: { type: "BPE", vocab: { x: 0 }, merges: [], unk_token: "x" },
      normalizer: null,
      pre_tokenizer: null,
      post_processor: null,
      decoder: null,
      added_tokens: [
        { id: 9, content: "<|im_end|>", special: true, normalized: false },
        { id: 17, content: "<|pad|>", special: true, normalized: false }
      ]
    },
    chatTemplate: "{{ messages[0].content }}"
  })
  assert.equal(tokenizer.vocabSize, 18)
  assert.equal(tokenizer.specialTokens.eos, 9)
  assert.equal(tokenizer.specialTokens.pad, 17)
  assert.deepEqual(tokenizer.encode("x"), [0])
})

test("a token's bytes come from the vocabulary: a character split across tokens keeps its bytes", () => {
  // Byte-level BPE writes a byte as a character: 🧪 is F0 9F A7 AA, here one token a byte.
  const chars = [0xf0, 321, 0xa7, 0xaa].map((c) => String.fromCodePoint(c))
  const tokenizer = loadTokenizer({
    tokenizerJson: {
      model: { type: "BPE", vocab: { [chars[0]]: 0, [chars[1]]: 1, [chars[2]]: 2, [chars[3]]: 3, "Ġhi": 4, [chars[2] + chars[3]]: 5 }, merges: [], unk_token: chars[0] },
      normalizer: null,
      pre_tokenizer: { type: "ByteLevel", add_prefix_space: false, trim_offsets: true, use_regex: true },
      post_processor: null,
      decoder: { type: "ByteLevel", add_prefix_space: true, trim_offsets: true, use_regex: true },
      added_tokens: [
        { id: 9, content: "<|im_end|>", special: true, normalized: false },
        { id: 10, content: "<|pad|>", special: true, normalized: false }
      ]
    },
    chatTemplate: "{{ messages[0].content }}"
  })
  const bytes = [0, 1, 2, 3].map((id) => [...tokenizer.tokenBytes(id)])
  assert.deepEqual(bytes, [[0xf0], [0x9f], [0xa7], [0xaa]])
  assert.equal(new TextDecoder().decode(Uint8Array.from(bytes.flat())), "🧪")
  assert.deepEqual([...tokenizer.tokenBytes(5)], [0xa7, 0xaa])
  // "Ġ" is the space byte; an added token is its text; an id without a token has no bytes.
  assert.deepEqual([...tokenizer.tokenBytes(4)], [...new TextEncoder().encode(" hi")])
  assert.deepEqual([...tokenizer.tokenBytes(9)], [...new TextEncoder().encode("<|im_end|>")])
  assert.deepEqual([...tokenizer.tokenBytes(7)], [])
})
