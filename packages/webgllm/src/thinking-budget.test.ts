import assert from "node:assert/strict"
import { describe, test } from "node:test"
import { BUDGET_WRAP_UP, budgetWrapUp, createThinkingBudget, templateClosing } from "./thinking-budget.js"

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


test("an exhausted budget allows its closing marker once, then masks both control tokens", () => {
  const b = createThinkingBudget(2, { think: 31, thinkEnd: 32 }, true)!
  b.observe(10)
  b.observe(11)
  assert.equal(b.forcedToken, 32)
  assert.equal(b.closedControlMask(64), undefined, "forced close must remain available")
  b.observe(32)
  const mask = b.closedControlMask(64)!
  assert.equal((mask[0] & (1 << 31)) >>> 0, 0)
  assert.equal(mask[1] & 1, 0)
  assert.equal(mask[0] & 1, 1, "ordinary answer tokens stay allowed")
  assert.equal(b.closedControlMask(64), mask, "the stable mask needs no per-token GPU upload")
})

test("natural early closing and omitted budgets preserve normal control-token behavior", () => {
  const b = createThinkingBudget(10, { think: 31, thinkEnd: 32 }, true)!
  b.observe(10)
  b.observe(32)
  assert.equal(b.closedControlMask(64), undefined)
  assert.equal(createThinkingBudget(undefined, { think: 31, thinkEnd: 32 }, true), null)
  const zero = createThinkingBudget(0, { think: 31, thinkEnd: 32 }, true)!
  assert.equal(zero.closedControlMask(64), undefined)
  zero.observe(32)
  assert.ok(zero.closedControlMask(64))
})

describe("closing the way the chat template does", () => {
  // Qwen3.5's template writes "\n</think>\n\n" between a reply's reasoning
  // and its answer: a newline, the marker, a blank line (ids 5, 32, 6 here).
  const markers = { think: 31, thinkEnd: 32 }
  const closing = [5, 32, 6]

  test("an exhausted budget forces the template's whole closing, in order", () => {
    const b = createThinkingBudget(2, markers, true, closing)!
    b.observe(10)
    b.observe(11)
    const forced: number[] = []
    for (let token = b.forcedToken; token !== null; token = b.forcedToken) {
      forced.push(token)
      b.observe(token)
    }
    assert.deepEqual(forced, closing)
    b.observe(12) // The answer is the model's own.
    assert.equal(b.forcedToken, null)
    assert.equal(b.stats.thinkingTokens, 2, "the forced newline isn't counted as thinking")
    assert.equal(b.stats.thinkingBudgetExhausted, true)
  })

  test("a natural close forces nothing", () => {
    const b = createThinkingBudget(5, markers, true, closing)!
    b.observe(10)
    b.observe(32)
    b.observe(12)
    assert.equal(b.forcedToken, null)
  })

  test("drafts stop at the cap and the close continues from there", () => {
    const b = createThinkingBudget(1, markers, true, closing)!
    b.observe(10)
    assert.equal(b.forcedToken, 5)
    b.observe(5)
    assert.equal(b.forcedToken, 32)
    assert.equal(b.closedControlMask(64), undefined, "the marker is still to come")
    b.observe(32)
    assert.equal(b.forcedToken, 6)
    b.observe(6)
    assert.equal(b.forcedToken, null)
    assert.ok(b.closedControlMask(64), "both markers masked once it has closed")
  })

  test("a closing without exactly one end marker is refused", () => {
    for (const bad of [[5, 6], [32, 5, 32], []]) {
      assert.throws(() => createThinkingBudget(1, markers, true, bad), /closing/)
    }
  })
})

describe("templateClosing", () => {
  // A tokenizer stand-in: one id per character, the marker as one id.
  const tokenizerFor = (between: string) => ({
    specialTokens: { think: 31, thinkEnd: 32 },
    applyChatTemplate: (messages: Array<{ role: "user" | "assistant"; content: string; reasoning_content?: string }>) =>
      messages.map((m) => m.role === "assistant" ? `<think>${m.reasoning_content}${between}${m.content}` : m.content).join("|"),
    encode: (text: string) => {
      const ids: number[] = []
      for (let i = 0; i < text.length; i++) {
        if (text.startsWith("</think>", i)) { ids.push(32); i += "</think>".length - 1 }
        else ids.push(text.charCodeAt(i))
      }
      return ids
    }
  })

  test("reads what the template writes between a reply's reasoning and its answer", () => {
    assert.deepEqual(templateClosing(tokenizerFor("\n</think>\n\n")), [10, 32, 10, 10])
    assert.deepEqual(templateClosing(tokenizerFor("</think>")), [32])
  })

  test("falls back to the marker alone when the template shows no single close", () => {
    assert.deepEqual(templateClosing(tokenizerFor("")), [32])
    assert.deepEqual(templateClosing(tokenizerFor("</think></think>")), [32])
  })
})

describe("wrapping up before the close", () => {
  const markers = { think: 31, thinkEnd: 32 }
  const closing = [5, 32, 6]
  const wrapUp = [7, 8]
  const drain = (b: ReturnType<typeof createThinkingBudget> & object) => {
    const forced: number[] = []
    for (let token = b.forcedToken; token !== null; token = b.forcedToken) {
      forced.push(token)
      b.observe(token)
    }
    return forced
  }

  test("after some thinking, the wrap-up goes into the thinking before the close", () => {
    const b = createThinkingBudget(2, markers, true, closing, wrapUp)!
    b.observe(10)
    b.observe(11)
    assert.deepEqual(drain(b), [7, 8, 5, 32, 6])
    assert.equal(b.stats.thinkingTokens, 2, "the wrap-up isn't counted as thinking")
    assert.ok(b.closedControlMask(64))
  })

  test("a zero budget closes with no wrap-up: there's nothing to wrap up", () => {
    const b = createThinkingBudget(0, markers, true, closing, wrapUp)!
    assert.deepEqual(drain(b), closing)
  })

  test("a wrap-up holding a marker is refused", () => {
    assert.throws(() => createThinkingBudget(1, markers, true, closing, [7, 32]), /wrap-up/)
    assert.throws(() => createThinkingBudget(1, markers, true, closing, [31]), /wrap-up/)
  })
})

test("budgetWrapUp encodes the wrap-up once per tokenizer", () => {
  let calls = 0
  const tokenizer = { encode: (text: string) => { calls++; return [text.length] } }
  assert.deepEqual(budgetWrapUp(tokenizer), [BUDGET_WRAP_UP.length])
  budgetWrapUp(tokenizer)
  assert.equal(calls, 1)
})
