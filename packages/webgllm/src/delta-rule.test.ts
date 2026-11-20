import assert from "node:assert/strict"
import { describe, test } from "node:test"
import { deltaChunked, deltaChunkedKeys, deltaSequential, deltaSequentialKeys, type DeltaInputs, type KeyedDeltaInputs } from "./delta-rule.js"

/** A small deterministic generator (mulberry32). */
function random(seed: number) {
  let a = seed >>> 0
  return () => {
    a = (a + 0x6d2b79f5) >>> 0
    let t = a
    t = Math.imul(t ^ (t >>> 15), t | 1)
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61)
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}

/**
 * Inputs like the model's: q̂ and k̂ unit vectors (q̂ over √dk), beta in
 * (0, 1), decays in (0, 1). `similar` mixes every key toward one shared
 * direction (nearly collinear keys: the solve's hardest case); `decay`
 * scales the log-decays (large: state forgotten within a few tokens).
 */
function inputs(
  seed: number,
  o: { dk: number; dv: number; tokens: number; similar?: number; decay?: number }
): DeltaInputs {
  const rand = random(seed)
  const normal = () => {
    const u = Math.max(rand(), 1e-12)
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * rand())
  }
  const { dk, dv, tokens } = o
  const shared = Array.from({ length: dk }, normal)
  const unit = (scale: number) => {
    const x = Array.from({ length: dk }, (_, i) => (o.similar ?? 0) * shared[i] + normal())
    const n = Math.hypot(...x)
    return x.map((v) => (v / n) * scale)
  }
  const q = new Float64Array(tokens * dk)
  const k = new Float64Array(tokens * dk)
  const v = new Float64Array(tokens * dv)
  const beta = new Float64Array(tokens)
  const g = new Float64Array(tokens)
  for (let t = 0; t < tokens; t++) {
    q.set(unit(1 / Math.sqrt(dk)), t * dk)
    k.set(unit(1), t * dk)
    for (let j = 0; j < dv; j++) v[t * dv + j] = normal()
    beta[t] = 1 / (1 + Math.exp(-2 * normal()))
    g[t] = -(o.decay ?? 0.2) * Math.log(1 + Math.exp(normal()))
  }
  return { dk, dv, tokens, q, k, v, beta, g }
}

function randomState(seed: number, dk: number, dv: number) {
  const rand = random(seed)
  return Float64Array.from({ length: dk * dv }, () => rand() - 0.5)
}

/** max |a - b| over max |b|. */
function relError(a: Float64Array, b: Float64Array) {
  let diff = 0
  let scale = 0
  for (let i = 0; i < a.length; i++) {
    diff = Math.max(diff, Math.abs(a[i] - b[i]))
    scale = Math.max(scale, Math.abs(b[i]))
  }
  return diff / scale
}

describe("chunked gated delta rule", () => {
  for (const [tokens, chunk] of [
    [64, 16],
    [64, 32],
    [50, 16], // a partial last chunk
    [37, 32],
    [5, 16], // one partial chunk
    [1, 16],
    [96, 64]
  ]) {
    test(`matches token by token: ${tokens} tokens, chunks of ${chunk}`, () => {
      const x = inputs(tokens * 31 + chunk, { dk: 16, dv: 12, tokens })
      const s0 = randomState(tokens + chunk, 16, 12)
      const seq = deltaSequential(x, s0)
      const chunked = deltaChunked(x, s0, chunk)
      assert.ok(relError(chunked.o, seq.o) < 1e-10, `outputs ${relError(chunked.o, seq.o)}`)
      assert.ok(
        relError(chunked.state, seq.state) < 1e-10,
        `state ${relError(chunked.state, seq.state)}`
      )
    })
  }

  test("nearly collinear keys and slow decay stay exact", () => {
    const x = inputs(7, { dk: 32, dv: 8, tokens: 64, similar: 6, decay: 0.01 })
    const s0 = randomState(8, 32, 8)
    const seq = deltaSequential(x, s0)
    const chunked = deltaChunked(x, s0, 32)
    assert.ok(relError(chunked.o, seq.o) < 1e-8, `outputs ${relError(chunked.o, seq.o)}`)
    assert.ok(relError(chunked.state, seq.state) < 1e-8)
  })

  test("fast decay (exp of big negative sums) doesn't overflow", () => {
    const x = inputs(9, { dk: 16, dv: 8, tokens: 48, decay: 20 })
    const s0 = randomState(10, 16, 8)
    const seq = deltaSequential(x, s0)
    const chunked = deltaChunked(x, s0, 16)
    for (const v of chunked.o) assert.ok(Number.isFinite(v))
    assert.ok(relError(chunked.o, seq.o) < 1e-10)
    assert.ok(relError(chunked.state, seq.state) < 1e-10)
  })

  for (const [tokens, saveAt] of [
    [50, 7],
    [50, 39],
    [64, 1],
    [37, 31],
    [5, 3]
  ]) {
    // As Engine.recurrenceChunks: a shorter first chunk, so one ends at saveAt.
    const first = saveAt % 16 || 16
    test(`chunks ending at token ${saveAt} of ${tokens} (the first ${first}, then 16): the same outputs, and the state there`, () => {
      const x = inputs(tokens * 7 + saveAt, { dk: 16, dv: 12, tokens })
      const s0 = randomState(tokens + saveAt, 16, 12)
      const seq = deltaSequential(x, s0)
      let at: Float64Array | null = null
      const chunked = deltaChunked(x, s0, 16, first, (end, state) => {
        if (end === saveAt) at = state
      })
      assert.ok(relError(chunked.o, seq.o) < 1e-10, `outputs ${relError(chunked.o, seq.o)}`)
      assert.ok(relError(chunked.state, seq.state) < 1e-10)
      const prefix = deltaSequential({ ...x, tokens: saveAt }, s0)
      assert.ok(at && relError(at, prefix.state) < 1e-10, "the state where the chunk ends")
    })
  }

  test("a chunked batch then token steps continue the same state", () => {
    const x = inputs(11, { dk: 16, dv: 16, tokens: 40 })
    const s0 = randomState(12, 16, 16)
    const first = { ...x, tokens: 33 }
    const after = deltaChunked(first, s0, 16).state
    const slice = (a: Float64Array, w: number) => a.slice(33 * w, 40 * w)
    const rest: DeltaInputs = {
      ...x,
      tokens: 7,
      q: slice(x.q, 16),
      k: slice(x.k, 16),
      v: slice(x.v, 16),
      beta: slice(x.beta, 1),
      g: slice(x.g, 1)
    }
    const whole = deltaSequential(x, s0)
    const resumed = deltaSequential(rest, after)
    assert.ok(relError(resumed.state, whole.state) < 1e-10)
    assert.ok(relError(resumed.o, whole.o.slice(33 * 16)) < 1e-10)
  })
})

/** Raon Delta Attention's inputs: `inputs` with a log-decay per key in (lower, 0) and a write ratio. */
function keyedInputs(seed: number, o: { dk: number; dv: number; tokens: number; similar?: number; lower?: number }): KeyedDeltaInputs {
  const base = inputs(seed, o)
  const rand = random(seed ^ 0x9e3779b9)
  const g = new Float64Array(o.tokens * o.dk)
  // (lower · sigmoid(logit): Raon-1 nano's lower bound is -5.)
  for (let i = 0; i < g.length; i++) g[i] = (o.lower ?? -5) / (1 + Math.exp(-6 * (rand() - 0.5)))
  const ratio = Float64Array.from({ length: o.tokens }, () => 0.25 + 1.5 * rand())
  return { ...base, g, ratio }
}

describe("chunked delta rule with a decay per key and a write gate (Raon Delta Attention)", () => {
  for (const [tokens, chunk, first] of [
    [64, 16, 16],
    [64, 32, 32],
    [50, 16, 16],
    [37, 32, 5], // a short first chunk, as before a saved state
    [5, 16, 16],
    [1, 16, 16],
    [96, 32, 32]
  ]) {
    test(`matches token by token: ${tokens} tokens, chunks of ${chunk}, the first of ${first}`, () => {
      const x = keyedInputs(tokens * 17 + chunk, { dk: 16, dv: 12, tokens })
      const s0 = randomState(tokens + chunk, 16, 12)
      const seq = deltaSequentialKeys(x, s0)
      const ends: Array<number> = []
      const chunked = deltaChunkedKeys(x, s0, chunk, first, (end) => ends.push(end))
      assert.ok(relError(chunked.o, seq.o) < 1e-9, `outputs ${relError(chunked.o, seq.o)}`)
      assert.ok(relError(chunked.state, seq.state) < 1e-9, `state ${relError(chunked.state, seq.state)}`)
      assert.equal(ends[ends.length - 1], tokens)
      if (tokens > first) assert.equal(ends[0], first)
    })
  }

  test("the state after a chunk is the token-by-token state there", () => {
    const x = keyedInputs(3, { dk: 16, dv: 8, tokens: 40 })
    const s0 = randomState(4, 16, 8)
    const states = new Map<number, Float64Array>()
    deltaChunkedKeys(x, s0, 16, 8, (end, state) => states.set(end, Float64Array.from(state)))
    for (const [end, state] of states) {
      const cut = (a: Float64Array, width: number) => a.subarray(0, end * width)
      const seq = deltaSequentialKeys({ ...x, tokens: end, q: cut(x.q, 16), k: cut(x.k, 16), v: cut(x.v, 8), g: cut(x.g, 16), beta: x.beta.subarray(0, end), ratio: x.ratio.subarray(0, end) }, s0)
      assert.ok(relError(state, seq.state) < 1e-9, `state after ${end}: ${relError(state, seq.state)}`)
    }
  })

  test("a scalar decay and a ratio of 1 are the Gated DeltaNet rule", () => {
    const base = inputs(21, { dk: 16, dv: 12, tokens: 48 })
    const g = new Float64Array(48 * 16)
    for (let t = 0; t < 48; t++) g.fill(base.g[t], t * 16, (t + 1) * 16)
    const x: KeyedDeltaInputs = { ...base, g, ratio: new Float64Array(48).fill(1) }
    const s0 = randomState(22, 16, 12)
    const keyed = deltaChunkedKeys(x, s0, 16)
    const plain = deltaSequential(base, s0)
    assert.ok(relError(keyed.o, plain.o) < 1e-10)
    assert.ok(relError(keyed.state, plain.state) < 1e-10)
  })

  test("nearly collinear keys with the strongest decays stay exact and finite", () => {
    const x = keyedInputs(5, { dk: 32, dv: 8, tokens: 64, similar: 6, lower: -5 })
    const s0 = randomState(6, 32, 8)
    const seq = deltaSequentialKeys(x, s0)
    const chunked = deltaChunkedKeys(x, s0, 32)
    for (const v of chunked.o) assert.ok(Number.isFinite(v))
    assert.ok(relError(chunked.o, seq.o) < 1e-7, `outputs ${relError(chunked.o, seq.o)}`)
    assert.ok(relError(chunked.state, seq.state) < 1e-7)
  })
})
