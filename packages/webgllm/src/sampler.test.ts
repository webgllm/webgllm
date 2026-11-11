import assert from "node:assert/strict"
import { describe, test } from "node:test"
import { createSampler, draw, verify, logprobsOf } from "./sampler.js"

/** A seeded uniform random number generator in [0, 1). */
function rng(seed: number) {
  let s = seed >>> 0
  return () => {
    s = (Math.imul(s, 1664525) + 1013904223) >>> 0
    return s / 2 ** 32
  }
}

const sum = (p: Float32Array) => p.reduce((a, b) => a + b, 0)

describe("createSampler", () => {
  test("temperature 0 is the argmax at probability 1", () => {
    const s = createSampler({ temperature: 0, top_k: 0, repetition_penalty: 1 })
    const d = s.distribution(Float32Array.of(0.1, 3, -1, 2.9))
    assert.deepEqual([...d.ids], [1])
    assert.deepEqual([...d.p], [1])
  })

  test("the repetition penalty divides positive logits and multiplies negative ones", () => {
    const s = createSampler({ temperature: 0, top_k: 0, repetition_penalty: 2 })
    s.see([1, 2])
    // Token 1: 3 → 1.5, below token 3's 2. Token 2: -1 → -2.
    const logits = Float32Array.of(0, 3, -1, 2)
    assert.deepEqual([...s.distribution(logits).ids], [3])
    assert.equal(logits[1], 1.5)
    assert.equal(logits[2], -2)
  })

  test("presence penalty applies once to generated tokens, without penalizing prompt-only tokens", () => {
    const s = createSampler({ temperature: 0, top_k: 0, repetition_penalty: 2, presence_penalty: 1.5 })
    s.see([0, 1])
    s.see([1, 1, 2], true)
    const logits = Float32Array.of(4, 6, -1, 1)
    s.distribution(logits)
    assert.deepEqual([...logits], [2, 1.5, -3.5, 1])
    const fresh = createSampler({ temperature: 0, top_k: 0, repetition_penalty: 1, presence_penalty: 2 })
    const next = Float32Array.of(4, 6, -1, 1)
    fresh.distribution(next)
    assert.deepEqual([...next], [4, 6, -1, 1], "generated history resets for every reply")
  })

  test("min_p filters relative to the highest probability and renormalizes before top_p", () => {
    const s = createSampler({ temperature: 1, top_k: 0, repetition_penalty: 1, min_p: 0.5, top_p: 0.9 })
    const d = s.distribution(Float32Array.of(0, Math.log(0.6), Math.log(0.1)))
    assert.equal(d.p[2], 0)
    assert.ok(Math.abs(d.p[0] - 0.625) < 1e-6)
    assert.ok(Math.abs(sum(d.p) - 1) < 1e-6)
  })

  test("top_k keeps the k likeliest, normalized", () => {
    const s = createSampler({ temperature: 1, top_k: 2, repetition_penalty: 1 })
    const d = s.distribution(Float32Array.of(1, 4, 2, 3))
    assert.deepEqual([...d.ids].sort(), [1, 3])
    assert.ok(Math.abs(sum(d.p) - 1) < 1e-6)
    const p4 = d.p[[...d.ids].indexOf(1)]
    assert.ok(Math.abs(p4 - Math.exp(4) / (Math.exp(4) + Math.exp(3))) < 1e-6)
  })

  test("top_p drops the tail once the kept share reaches it", () => {
    const s = createSampler({ temperature: 1, top_k: 0, repetition_penalty: 1, top_p: 0.5 })
    // Softmax of (5, 0, 0, 0): the first holds ~98%, enough alone.
    const d = s.distribution(Float32Array.of(5, 0, 0, 0))
    const kept = [...d.ids].filter((_, i) => d.p[i] > 0)
    assert.deepEqual(kept, [0])
    assert.ok(Math.abs(sum(d.p) - 1) < 1e-6)
  })
})

describe("verify (speculative sampling)", () => {
  const d = { ids: Int32Array.of(10, 11, 12, 13), p: Float32Array.of(0.5, 0.25, 0.15, 0.1) }

  /** How often each token comes out of verify with `drafted` proposed, over `n` trials. */
  function outcomes(drafted: number, n: number) {
    const random = rng(42)
    const counts = new Map<number, number>()
    for (let i = 0; i < n; i++) {
      const r = verify(d, drafted, random)
      const token = r.kept ? drafted : r.token
      counts.set(token, (counts.get(token) ?? 0) + 1)
    }
    return counts
  }

  for (const drafted of [10, 11, 13, 99]) {
    test(`tokens come out distributed as the distribution, draft ${drafted}`, () => {
      const n = 200_000
      const counts = outcomes(drafted, n)
      for (let i = 0; i < d.ids.length; i++) {
        const got = (counts.get(d.ids[i]) ?? 0) / n
        assert.ok(Math.abs(got - d.p[i]) < 0.005, `token ${d.ids[i]}: ${got} vs ${d.p[i]}`)
      }
      assert.equal(counts.get(99) ?? 0, 0, "a draft outside the distribution never comes out")
    })
  }

  test("a draft at probability 1 is always kept", () => {
    const one = { ids: Int32Array.of(5), p: Float32Array.of(1) }
    assert.deepEqual(verify(one, 5, () => 0.999), { kept: true })
  })

  test("greedy: a one-token distribution keeps its token, replaces any other", () => {
    const one = { ids: Int32Array.of(5), p: Float32Array.of(1) }
    assert.deepEqual(verify(one, 6, () => 0.5), { kept: false, token: 5 })
  })
})

describe("draw", () => {
  test("draws in proportion to p", () => {
    const d = { ids: Int32Array.of(1, 2, 3), p: Float32Array.of(0.2, 0.3, 0.5) }
    const random = rng(1)
    const counts = [0, 0, 0]
    const n = 100_000
    for (let i = 0; i < n; i++) counts[draw(d, random) - 1]++
    for (let i = 0; i < 3; i++) assert.ok(Math.abs(counts[i] / n - d.p[i]) < 0.006)
  })

  test("never returns a token at probability 0", () => {
    const d = { ids: Int32Array.of(1, 2, 3), p: Float32Array.of(0, 1, 0) }
    for (const x of [0, 0.5, 0.999999]) assert.equal(draw(d, () => x), 2)
  })
})

/**
 * The selection the sampler used before its heap and radix sort: the k
 * largest by rescanning for the smallest kept, and top-p by a comparator
 * sort. The new one must give the same distribution, bit for bit.
 */
function reference(s: { temperature: number; top_k: number; top_p: number; min_p: number }, logits: Float32Array) {
  const k = Math.max(1, Math.min(s.top_k > 0 ? s.top_k : logits.length, logits.length))
  let ids: Int32Array
  let kept: Float32Array
  if (k >= logits.length) {
    ids = Int32Array.from({ length: logits.length }, (_, i) => i)
    kept = logits.slice()
  } else {
    ids = new Int32Array(k)
    kept = new Float32Array(k).fill(-Infinity)
    let smallest = 0
    for (let i = 0; i < logits.length; i++) {
      const x = logits[i]
      if (!(x > kept[smallest])) continue
      kept[smallest] = x
      ids[smallest] = i
      smallest = 0
      for (let j = 1; j < k; j++) if (kept[j] < kept[smallest]) smallest = j
    }
  }
  let best = 0
  for (let j = 1; j < k; j++) if (kept[j] > kept[best]) best = j
  const max = kept[best]
  let total = 0
  for (let j = 0; j < k; j++) {
    kept[j] = Math.exp((kept[j] - max) / s.temperature)
    total += kept[j]
  }
  for (let j = 0; j < k; j++) {
    if (kept[j] < s.min_p) {
      total -= kept[j]
      kept[j] = 0
    }
  }
  if (s.top_p < 1) {
    const order = Array.from({ length: k }, (_, j) => j).sort((a, b) => kept[b] - kept[a])
    let share = 0
    let cut = false
    for (const j of order) {
      if (cut) {
        total -= kept[j]
        kept[j] = 0
      } else {
        share += kept[j] / total
        cut = share >= s.top_p
      }
    }
  }
  for (let j = 0; j < k; j++) kept[j] /= total
  return { ids, p: kept }
}

describe("selection matches the rescanning reference", () => {
  test("same ids, order and probabilities on random logits, with ties and masked tokens", () => {
    const random = rng(7)
    const bytes = (a: ArrayBufferView) => Buffer.from(a.buffer, a.byteOffset, a.byteLength)
    for (let round = 0; round < 60; round++) {
      const n = 50 + Math.floor(random() * 5000)
      const logits = new Float32Array(n)
      // Every third round quantized, for many ties; some tokens masked.
      const step = round % 3 === 0 ? 0.5 : 0
      for (let i = 0; i < n; i++) {
        const x = (random() - 0.5) * 12
        logits[i] = random() < 0.05 ? -Infinity : step ? Math.round(x / step) * step : x
      }
      const s = {
        temperature: 0.3 + random() * 1.5,
        top_k: [0, 1, 7, 40, 512, n, n + 3][round % 7],
        top_p: [1, 0.9, 0.5, 0, 0.99][round % 5],
        min_p: [0, 0, 0.05][round % 3]
      }
      const got = createSampler({ ...s, repetition_penalty: 1 }).distribution(logits.slice())
      const want = reference(s, logits.slice())
      assert.ok(bytes(got.ids).equals(bytes(want.ids)), `ids, round ${round}`)
      assert.ok(bytes(got.p).equals(bytes(want.p)), `p, round ${round}`)
    }
  })
})

test("distribution(logits, false) leaves out the repetition and presence penalties", () => {
  const s = createSampler({ temperature: 0, top_k: 1, repetition_penalty: 10, presence_penalty: 5 })
  s.see([0], true)
  // Token 0 leads by a little: penalized it loses, unpenalized it wins.
  assert.deepEqual(Array.from(s.distribution(Float32Array.of(2, 1.9)).ids), [1])
  assert.deepEqual(Array.from(s.distribution(Float32Array.of(2, 1.9), false).ids), [0])
})

test("log probabilities: the model's distribution, with only tokens that can be written as alternatives", () => {
  const logits = Float32Array.of(2, -Infinity, 1, -Infinity, 0)
  const before = logits.slice()
  const of = logprobsOf(logits, 5, (token, logprob) => ({ token, logprob }))
  const z = Math.log(Math.exp(2) + Math.exp(1) + 1)
  const entry = of(2)
  assert.ok(Math.abs(entry.logprob - (1 - z)) < 1e-6)
  // Three tokens are allowed: three alternatives, likeliest first, none masked.
  assert.deepEqual(entry.top_logprobs.map((a) => a.token), [0, 2, 4])
  assert.ok(entry.top_logprobs.every((a) => Number.isFinite(a.logprob)))
  assert.deepEqual(logits, before)
  assert.deepEqual(logprobsOf(logits, 0, (token, logprob) => ({ token, logprob }))(0).top_logprobs, [])
})
