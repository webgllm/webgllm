import assert from "node:assert/strict"
import { describe, test } from "node:test"
import { DraftPolicy, lookupDraft } from "./draft.js"

describe("lookupDraft", () => {
  test("drafts what followed the latest earlier occurrence of the last 4 tokens", () => {
    // ... 1 2 3 4 [5 6 7 8 9] ... 1 2 3 4
    const tokens = [9, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4]
    assert.deepEqual(lookupDraft(tokens, 8), [5, 6, 7, 8, 9, 0, 1, 2])
  })

  test("an n-token match drafts at most 2^(n-1) tokens, and at most max", () => {
    // Only the last 2 tokens (3 4) recur: up to 2 drafts.
    assert.deepEqual(lookupDraft([3, 4, 5, 6, 7, 1, 3, 4], 8), [5, 6])
    assert.deepEqual(lookupDraft([9, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4], 3), [5, 6, 7])
  })

  test("the latest occurrence wins", () => {
    assert.deepEqual(lookupDraft([1, 2, 7, 1, 2, 8, 1, 2], 1, 2), [8])
  })

  test("nothing below minMatch, nothing with max 0", () => {
    // Only a 1-token match (2): none at minMatch 2.
    assert.deepEqual(lookupDraft([2, 5, 3, 2], 4, 2), [])
    assert.deepEqual(lookupDraft([2, 5, 3, 2], 4, 1), [5])
    assert.deepEqual(lookupDraft([1, 2, 3, 1, 2, 3], 0), [])
  })

  test("drafts stop at the end of the text", () => {
    assert.deepEqual(lookupDraft([1, 2, 1, 2], 8, 2), [1, 2])
  })
})

/** A seeded uniform random number generator in [0, 1). */
function rng(seed: number) {
  let s = seed >>> 0
  return () => {
    s = (Math.imul(s, 1664525) + 1013904223) >>> 0
    return s / 2 ** 32
  }
}

/**
 * Runs `steps` writing steps against a policy: drafts are right with
 * probability `accept` each, a token alone takes `singleMs`, a batch of d
 * drafts `batchMs(d)`. Returns tokens per ms and the draft counts chosen.
 */
function simulate(
  policy: DraftPolicy,
  accept: number,
  singleMs: number,
  batchMs: (d: number) => number,
  steps = 400
) {
  const random = rng(7)
  const available = [101, 102, 103, 104, 105, 106, 107]
  let tokens = 0
  let ms = 0
  const chosen: Array<number> = []
  for (let step = 0; step < steps; step++) {
    const d = policy.choose(available)
    chosen.push(d)
    if (d === 0) {
      policy.alone(singleMs)
      ms += singleMs
      tokens++
      // The token drawn matches the first unchecked draft with probability `accept`.
      policy.drawn(random() < accept ? available[0] : -1)
    } else {
      let kept = 0
      while (kept < d && random() < accept) kept++
      policy.checkedBatch(d, kept, batchMs(d))
      ms += batchMs(d)
      tokens += kept + 1
    }
  }
  const tail = chosen.slice(-100)
  return { rate: tokens / ms, meanDrafts: tail.reduce((a, b) => a + b, 0) / tail.length }
}

describe("DraftPolicy", () => {
  test("times a token alone before drafting", () => {
    const p = new DraftPolicy(7)
    assert.equal(p.choose([1, 2, 3]), 0)
    p.alone(10)
    assert.ok(p.choose([1, 2, 3]) > 0, "drafts get tried once a single token is timed")
  })

  test("drafts the most when drafts are usually right and batches cost about a token", () => {
    const r = simulate(new DraftPolicy(7), 0.85, 10, (d) => 10.5 + 0.2 * d)
    assert.ok(r.meanDrafts > 5, `mean drafts ${r.meanDrafts}`)
    assert.ok(r.rate > 0.25, `rate ${r.rate} tokens/ms`)
  })

  test("stops drafting when drafts are rarely right", () => {
    const r = simulate(new DraftPolicy(7), 0.05, 10, (d) => 11 + 0.3 * d)
    assert.ok(r.meanDrafts < 0.5, `mean drafts ${r.meanDrafts}`)
    assert.ok(r.rate >= 0.095, `no slower than plain writing: ${r.rate}`)
  })

  test("stops drafting when a batch costs more than it earns", () => {
    // Drafts right 40% of the time, but a batch costs twice a token.
    const r = simulate(new DraftPolicy(7), 0.4, 10, (d) => 20 + d)
    assert.ok(r.meanDrafts < 0.5, `mean drafts ${r.meanDrafts}`)
  })

  test("never writes slower than plain decoding by much", () => {
    for (const accept of [0, 0.2, 0.5, 0.8, 1]) {
      for (const cost of [1.05, 1.5, 2.5]) {
        const r = simulate(new DraftPolicy(7), accept, 10, (d) => 10 * cost + 0.3 * d)
        assert.ok(r.rate > 0.1 * 0.93, `accept ${accept}, batch ×${cost}: ${r.rate}`)
      }
    }
  })

  test("drafts again once unchecked drafts start being right", () => {
    const p = new DraftPolicy(7)
    p.alone(10)
    // Learn that drafts are wrong: batches of 1 rejected.
    for (let i = 0; i < 40; i++) p.checkedBatch(1, 0, 11)
    assert.equal(p.choose([5, 6, 7]), 0)
    // The drafts it skipped turn out right, scored as tokens are drawn.
    for (let i = 0; i < 60 && p.choose([5, 6, 7]) === 0; i++) {
      p.alone(10)
      p.drawn(5)
      p.drawn(6)
      p.drawn(7)
    }
    assert.ok(p.choose([5, 6, 7]) > 0)
  })

  test("peek answers like choose without changing what it remembers", () => {
    const a = new DraftPolicy(7)
    const b = new DraftPolicy(7)
    for (const p of [a, b]) {
      p.alone(10)
      for (let i = 0; i < 20; i++) p.checkedBatch(1, 0, 11)
    }
    // a peeks (no shadow), b chooses (shadow of the drafts it skipped).
    assert.equal(a.peek([5, 6]), b.choose([5, 6]))
    a.drawn(5)
    b.drawn(5)
    assert.ok(b.rate(1) > a.rate(1), "only choose's skipped drafts are scored")
  })

  test("a batch cut short scores only the drafts it judged", () => {
    const p = new DraftPolicy(7)
    // 3 drafts, the first kept, the reply over after it: draft 2 wasn't judged
    // (not counted as turned down), so drafts 2 and 3 have no outcomes of
    // their own and lean on draft 1's rate.
    p.checkedBatch(3, 1, 10, 1)
    assert.ok(p.rate(1) > 0.5)
    assert.equal(p.rate(2), p.rate(1))
    assert.equal(p.rate(3), p.rate(2))
    // Judged in full, draft 2 counts as turned down.
    const q = new DraftPolicy(7)
    q.checkedBatch(3, 1, 10)
    assert.ok(q.rate(2) < q.rate(1))
  })

  test("one stalled batch doesn't stop drafting for the rest of the reply", () => {
    const p = new DraftPolicy(7)
    p.alone(15)
    // The first draft batch stalls (a compile, say): 200 ms against 15.
    const d = p.choose([1, 2, 3, 4, 5, 6, 7])
    assert.ok(d > 0)
    p.checkedBatch(d, d, 200)
    assert.ok(p.choose([1, 2, 3, 4, 5, 6, 7]) > 0, "still drafting after the stall")
    // ...and one stalled token alone is soon forgotten: with tokens back at
    // 15 ms, batches at 20 ms that drafts rarely pay for aren't drawn.
    const q = new DraftPolicy(7)
    q.alone(15)
    q.alone(500)
    for (let i = 0; i < 10; i++) q.alone(15)
    for (let i = 0; i < 30; i++) q.checkedBatch(1, 0, 20)
    assert.equal(q.choose([1, 2]), 0)
  })

  test("drafts past those checked are scored once a batch keeps every one", () => {
    const p = new DraftPolicy(7)
    p.alone(10)
    // Draft 3 has been turned down: batches settle on 2 drafts.
    for (let i = 0; i < 10; i++) p.checkedBatch(3, 2, 10.5)
    const before = p.rate(3)
    const available = [11, 12, 13, 14, 15, 16, 17]
    const d = p.choose(available)
    assert.ok(d > 0 && d < 3, `chose ${d}`)
    for (let i = 0; i < d; i++) p.drawn(available[i]) // the kept drafts, as emitted
    p.checkedBatch(d, d, 10.5)
    // The tokens drawn after the batch match the drafts it didn't check.
    for (let i = d; i < 7; i++) p.drawn(available[i])
    assert.ok(p.rate(3) > before, `rate(3) ${before} -> ${p.rate(3)}`)
  })

  test("rates stay between 0 and 1", () => {
    const p = new DraftPolicy(7)
    for (let i = 0; i < 200; i++) p.checkedBatch(7, i % 8, 10)
    for (let i = 1; i <= 7; i++) {
      assert.ok(p.rate(i) >= 0 && p.rate(i) <= 1, `rate(${i}) = ${p.rate(i)}`)
    }
  })
})
