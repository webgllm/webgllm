import assert from "node:assert/strict"
import { describe, test } from "node:test"
import { createThinkingBudget } from "./thinking-budget.js"

// Synthetic ids for LFM3's single-token thinking delimiters and generated text.
const markers = { think: 64400, thinkEnd: 64401 }
const budget = (limit: number, startsInThinking = true) =>
  createThinkingBudget(limit, markers, startsInThinking)!

describe("LFM3 thinking budget", () => {
  test("an omitted budget leaves generation unchanged, even without markers", () => {
    assert.equal(createThinkingBudget(undefined, { think: null, thinkEnd: null }, true), null)
  })

  test("zero closes a template-opened thinking block before the first draw", () => {
    const b = budget(0)
    assert.equal(b.forcedToken, markers.thinkEnd)
    b.observe(markers.thinkEnd)
    assert.equal(b.forcedToken, null)
    b.observe(42) // Answer tokens are outside the thinking budget.
    assert.deepEqual(b.stats, {
      thinkingTokens: 0, thinkingBudgetTokens: 0, thinkingBudgetExhausted: true
    })
  })

  test("the exact cap forces the next token, leaving answer generation available", () => {
    const b = budget(2)
    b.observe(10)
    assert.equal(b.forcedToken, null)
    b.observe(11)
    assert.equal(b.forcedToken, markers.thinkEnd)
    b.observe(markers.thinkEnd)
    assert.equal(b.forcedToken, null)
    b.observe(12)
    assert.equal(b.stats.thinkingTokens, 2)
    assert.equal(b.stats.thinkingBudgetExhausted, true)
  })

  test("natural early closing never spends the remaining budget or forces a delimiter", () => {
    const b = budget(5)
    b.observe(10)
    b.observe(markers.thinkEnd)
    for (const token of [20, 21, 22, 23, 24, 25]) b.observe(token)
    assert.equal(b.forcedToken, null)
    assert.equal(b.stats.thinkingTokens, 1)
    assert.equal(b.stats.thinkingBudgetExhausted, false)
  })

  test("the model can open thinking later; delimiters are not content tokens", () => {
    const b = budget(1, false)
    b.observe(10)
    b.observe(markers.think)
    assert.equal(b.stats.thinkingTokens, 0)
    assert.equal(b.forcedToken, null)
    b.observe(11)
    assert.equal(b.forcedToken, markers.thinkEnd)
  })

  test("reopened thinking spans share one budget instead of resetting it", () => {
    const b = budget(2)
    b.observe(10)
    b.observe(markers.thinkEnd)
    b.observe(20)
    b.observe(markers.think)
    b.observe(11)
    assert.equal(b.forcedToken, markers.thinkEnd)
    b.observe(markers.thinkEnd)
    b.observe(markers.think)
    assert.equal(b.forcedToken, markers.thinkEnd)
    assert.equal(b.stats.thinkingTokens, 2)
  })

  test("zero without any thinking does not claim exhaustion", () => {
    const b = budget(0, false)
    b.observe(10)
    assert.equal(b.forcedToken, null)
    assert.equal(b.stats.thinkingBudgetExhausted, false)
    b.observe(markers.think)
    assert.equal(b.forcedToken, markers.thinkEnd)
    assert.equal(b.stats.thinkingBudgetExhausted, true)
  })

  test("an overall output limit may stop before the thinking budget", () => {
    const b = budget(5)
    const totalLimit = 2
    let emitted = 0
    while (emitted < totalLimit) {
      const token = b.forcedToken ?? 10
      b.observe(token)
      emitted++
    }
    assert.equal(emitted, 2)
    assert.equal(b.stats.thinkingTokens, 2)
    assert.equal(b.stats.thinkingBudgetExhausted, false)
  })

  test("a speculative batch stops acceptance immediately at the cap", () => {
    const b = budget(2)
    b.observe(10) // Current token, already emitted before the forward batch.
    const drafts = [11, 12, 13, 14]
    let kept = 1
    let next = -1
    for (const token of drafts) {
      const forced = b.forcedToken
      if (forced !== null) { next = forced; break }
      kept++
      b.observe(token)
    }
    assert.equal(kept, 2, "only the current token and first draft stay in KV/cache")
    assert.equal(next, markers.thinkEnd)
    assert.equal(b.stats.thinkingTokens, 2)
  })

  test("rejects invalid budgets and missing or invalid single-token delimiters", () => {
    for (const limit of [-1, 0.5, NaN, Infinity, -Infinity, Number.MAX_SAFE_INTEGER + 1]) {
      assert.throws(() => createThinkingBudget(limit, markers, true), /nonnegative safe integer/)
    }
    for (const invalid of [
      { think: null, thinkEnd: 64401 }, { think: 64400, thinkEnd: null },
      { think: 1, thinkEnd: 1 }, { think: -1, thinkEnd: 2 },
      { think: 1, thinkEnd: 2.5 }, { think: NaN, thinkEnd: 2 }
    ]) {
      assert.throws(() => createThinkingBudget(1, invalid, true), /single-token/)
    }
  })
})
