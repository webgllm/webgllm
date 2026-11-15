import assert from "node:assert/strict"
import { test } from "node:test"
import { DEFAULT_REPEAT_LAST_N, PenaltyWindow } from "./penalty-window.js"
import { createSampler } from "./sampler.js"

test("the window holds the last tokens, each once; one leaving it is reported unseen", () => {
  const w = new PenaltyWindow(3)
  assert.deepEqual(w.push([1, 2, 2]), [[1, 1], [2, 1]])
  // 1 leaves; 2 is still in (once).
  assert.deepEqual(new Map(w.push([3, 4])), new Map([[3, 1], [1, 0], [4, 1]]))
  assert.deepEqual(new Map(w.entries()), new Map([[2, 1], [3, 1], [4, 1]]))
  // A written token is 3 (the presence penalty's too), and back to 1 when only the prompt's copy is left.
  assert.deepEqual(new Map(w.push([3], true)), new Map([[3, 3], [2, 0]]))
  // (In no particular order.)
  assert.deepEqual(new Map(w.push([5, 6, 7])), new Map([[5, 1], [3, 0], [4, 0], [6, 1], [7, 1]]))
  assert.equal(DEFAULT_REPEAT_LAST_N, 64)
})

test("size 0 penalizes nothing; a negative size keeps every token", () => {
  assert.deepEqual(new PenaltyWindow(0).push([1, 2, 3]), [])
  const all = new PenaltyWindow(-1)
  all.push(Array.from({ length: 500 }, (_, i) => i))
  assert.equal(all.entries().length, 500)
})

test("the sampler penalizes only what's in the window: a token read long ago isn't", () => {
  const sampler = createSampler({ temperature: 0, top_k: 0, repetition_penalty: 2, repeat_last_n: 2 })
  sampler.see([0, 1, 2])
  const logits = Float32Array.of(4, 4, 4, 3.9)
  const { ids } = sampler.distribution(logits)
  // 1 and 2 halved; 0 left the window: it wins.
  assert.equal(ids[0], 0)
  assert.deepEqual(Array.from(logits), [4, 2, 2, Math.fround(3.9)])
})
