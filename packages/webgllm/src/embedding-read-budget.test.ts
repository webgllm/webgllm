import assert from "node:assert/strict"
import test from "node:test"
import { defaultEmbeddingRows } from "./load-gguf.js"

test("native embedding reads amortize I/O within the existing raw-plus-packed tensor budget", () => {
  for (const rowBytes of [576, 1680, 3360, 6720, 13440])
    for (const largest of [4, 8, 16, 32].map((x) => x * 2 ** 20)) {
      const rows = defaultEmbeddingRows(rowBytes, largest, true)
      assert.ok(rows >= 512 && rows <= 2048 && rows % 4 === 0)
      if (rows > 512) {
        assert.ok(rows * rowBytes <= 4 * 2 ** 20)
        assert.ok(rows * rowBytes * 5 <= largest * 2)
      }
    }
  assert.ok(defaultEmbeddingRows(1680, 9 * 2 ** 20, true) > 512)
  assert.equal(defaultEmbeddingRows(8192, 9 * 2 ** 20, false), 512)
})
