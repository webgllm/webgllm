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
