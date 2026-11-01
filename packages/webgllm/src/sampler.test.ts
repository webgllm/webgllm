import assert from "node:assert/strict"
import { describe, test } from "node:test"
import { createSampler, draw, verify } from "./sampler.js"

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
