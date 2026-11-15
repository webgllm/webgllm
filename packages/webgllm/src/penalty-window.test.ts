import assert from "node:assert/strict"
import { test } from "node:test"
import { DEFAULT_REPEAT_LAST_N, MAX_SEEN_ROWS, PenaltyWindow } from "./penalty-window.js"
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

/** The state at each row of a batch, pushing its drafts one by one: the reference ahead() must match. */
function sequentialRows(size: number, prompt: Array<number>, written: Array<number>, drafts: Array<number>) {
  const window = new PenaltyWindow(size)
  window.push(prompt)
  window.push(written, true)
  const rows: Array<Map<number, number>> = [new Map(window.entries())]
  for (const draft of drafts) {
    window.push([draft], true)
    rows.push(new Map(window.entries()))
  }
  return rows
}

/** The state at each row from ahead()'s segments over the window's state now. */
function aheadRows(size: number, prompt: Array<number>, written: Array<number>, drafts: Array<number>) {
  const window = new PenaltyWindow(size)
  window.push(prompt)
  window.push(written, true)
  const base = new Map(window.entries())
  const segments = window.ahead(drafts)
  assert.ok(segments.length <= MAX_SEEN_ROWS, `${segments.length} segments`)
  assert.deepEqual(new Map(window.entries()), base, "ahead() changes nothing")
  return Array.from({ length: drafts.length + 1 }, (_, r) => {
    const row = new Map(base)
    for (const [token, state, from, to] of segments) if (r >= from && r < to) row.set(token, state)
    for (const [token, state] of [...row]) if (state === 0) row.delete(token)
    return row
  })
}

test("a batch's rows see the window as pushing its drafts one by one would: zero, finite and unlimited windows", () => {
  let seed = 7
  const random = (n: number) => ((seed = (seed * 1103515245 + 12345) % 2 ** 31) % n)
  for (const size of [0, 1, 3, 8, 64, -1]) {
    for (let trial = 0; trial < 300; trial++) {
      const prompt = Array.from({ length: random(12) }, () => random(6))
      const written = Array.from({ length: random(5) }, () => random(6))
      const drafts = Array.from({ length: 1 + random(8) }, () => random(6))
      assert.deepEqual(aheadRows(size, prompt, written, drafts), sequentialRows(size, prompt, written, drafts), JSON.stringify({ size, prompt, written, drafts }))
    }
  }
})

test("a zero window gives drafts no penalties; a draft pushes the oldest token out of a full window", () => {
  const none = new PenaltyWindow(0)
  none.push([1, 2])
  assert.deepEqual(none.ahead([5, 6]), [])
  const window = new PenaltyWindow(2)
  window.push([1, 2])
  // Row 1 has read draft 5 (written) and lost token 1; row 2 has read 6 and lost 2.
  assert.deepEqual(window.ahead([5, 6]), [[5, 3, 1, 3], [1, 0, 1, 3], [6, 3, 2, 3], [2, 0, 2, 3]])
})
