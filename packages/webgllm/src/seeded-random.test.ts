import assert from "node:assert/strict"
import { test } from "node:test"
import { seededUniforms } from "./seeded-random.js"
import { draw, drawAt, verify, verifyAt } from "./sampler.js"

test("the same seed gives the same numbers, in [0, 1)", () => {
  const a = seededUniforms(42), b = seededUniforms(42)
  for (let i = 0; i < 1000; i++) {
    const u = a(i)
    assert.equal(u, b(i))
    assert.ok(u >= 0 && u < 1)
  }
})

test("other seeds, and both halves of 64-bit seeds, give other numbers", () => {
  const first = (seed: number) => Array.from({ length: 8 }, (_, i) => seededUniforms(seed)(i)).join()
  const seen = new Set([0, 1, 2, -1, 2 ** 32, 2 ** 32 + 1, 2 ** 52, Number.MAX_SAFE_INTEGER].map(first))
  assert.equal(seen.size, 8)
  assert.throws(() => seededUniforms(NaN), RangeError)
})

test("roughly uniform", () => {
  const u = seededUniforms(7)
  const bins = new Array(10).fill(0)
  for (let i = 0; i < 100_000; i++) bins[Math.floor(u(i) * 10)]++
  for (const n of bins) assert.ok(Math.abs(n - 10_000) < 500, String(bins))
})

const dist = (ids: number[], p: number[]) => ({ ids: Int32Array.from(ids), p: Float32Array.from(p) })

test("drawAt is the inverse CDF in id order, whatever order candidates come in", () => {
  const a = dist([3, 7, 9], [0.5, 0.25, 0.25])
  const b = dist([9, 3, 7], [0.25, 0.5, 0.25])
  for (const u of [0, 0.2, 0.49, 0.5, 0.74, 0.75, 0.99]) {
    const expected = u < 0.5 ? 3 : u < 0.75 ? 7 : 9
    assert.equal(drawAt(a, u), expected)
    assert.equal(drawAt(b, u), expected)
  }
  // Rounding past the total: the likeliest token (lowest id on a tie).
  assert.equal(drawAt(dist([5, 2], [0.4, 0.4]), 0.9), 2)
  // Zero-probability tokens are never drawn.
  assert.equal(drawAt(dist([1, 2], [0, 1]), 0), 2)
  // Greedy distributions are one token at probability 1.
  assert.equal(drawAt(dist([11], [1]), 0.999), 11)
})

test("verifyAt keeps a draft exactly when it's the token drawn there", () => {
  const d = dist([3, 7, 9], [0.5, 0.25, 0.25])
  assert.deepEqual(verifyAt(d, 7, 0.6), { kept: true })
  assert.deepEqual(verifyAt(d, 7, 0.1), { kept: false, token: 3 })
  assert.deepEqual(verifyAt(d, 4, 0.9), { kept: false, token: 9 })
})

test("seeded verification keeps drafts as often as verify does, and draws the same distribution", () => {
  const d = dist([1, 2, 3, 4], [0.1, 0.2, 0.3, 0.4])
  const u = seededUniforms(1)
  const n = 200_000
  const seeded = new Map<number, number>(), plain = new Map<number, number>()
  let keptSeeded = 0, keptPlain = 0
  let state = 1
  const random = () => ((state = Math.imul(state, 48271) % 0x7fffffff) & 0x7fffffff) / 0x7fffffff
  for (let i = 0; i < n; i++) {
    const a = verifyAt(d, 3, u(i))
    const b = verify(d, 3, random)
    const ta = a.kept ? 3 : a.token, tb = b.kept ? 3 : b.token
    if (a.kept) keptSeeded++
    if (b.kept) keptPlain++
    seeded.set(ta, (seeded.get(ta) ?? 0) + 1)
    plain.set(tb, (plain.get(tb) ?? 0) + 1)
  }
  assert.ok(Math.abs(keptSeeded / n - 0.3) < 0.01)
  assert.ok(Math.abs(keptPlain / n - 0.3) < 0.01)
  for (const [id, p] of [[1, 0.1], [2, 0.2], [3, 0.3], [4, 0.4]]) {
    assert.ok(Math.abs(seeded.get(id)! / n - p) < 0.01)
    assert.ok(Math.abs(plain.get(id)! / n - p) < 0.01)
  }
  // Unseeded draw is unchanged.
  assert.equal(draw(d, () => 0), 1)
})

test("a seeded reply doesn't depend on what was drafted", () => {
  // The same position's number decides the token whether it's drawn or verified.
  const d = dist([1, 2, 3, 4], [0.1, 0.2, 0.3, 0.4])
  const u = seededUniforms(99)
  for (let i = 0; i < 1000; i++) {
    const drawn = drawAt(d, u(i))
    for (const drafted of [1, 2, 3, 4, 5]) {
      const v = verifyAt(d, drafted, u(i))
      assert.equal(v.kept ? drafted : v.token, drawn)
    }
  }
})
