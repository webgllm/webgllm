import assert from "node:assert/strict"
import test from "node:test"
import { Engine } from "./engine.js"
import { readPrompt, templateFramingFor } from "./prompt.js"
import { reusablePrefixFor, adaptivePrefixFor } from "./checkpoint-policy.js"

type Point = { prefix?: boolean; anchor?: boolean; hidden?: object; conv: object[]; windows: object[]; linear: object[] }
function harness(limit = 8, inside = false) {
  const engine = Object.create(Engine.prototype) as Engine
  const snapshots = new Map<number, Point>()
  const calls: { length: number; saveAt?: number; prefixAt?: number }[] = []
  const live = new Set<object>()
  let fail = false
  const texture = () => { const t = { width: 1, height: 1, bytes: 4 }; live.add(t); return t }
  Object.defineProperties(engine, { snapshotLimit: { value: limit }, canSaveInside: { value: inside } })
  Object.assign(engine, { position: 0, guessAt: null, draft: null, layers: [], config: { slidingWindow: 0 }, snapshots, gpu: { delete(t: object) { live.delete(t) } } })
  Reflect.set(engine, "copyHidden", texture)
  Reflect.set(engine, "reset", () => { for (const saved of snapshots.values()) Reflect.get(engine, "freeSnapshot").call(engine, saved); snapshots.clear(); engine.position = 0 })
  Reflect.set(engine, "batchSizes", (n: number) => { const out = []; while (n > 576) { out.push(512); n -= 512 }; if (n) out.push(n); return out })
  const forward = Engine.prototype.forwardHidden
  Reflect.set(engine, "forwardHidden", function(tokens: number[], options: { saveAt?: number; prefixAt?: number } = {}) { calls.push({ length: tokens.length, ...options }); return forward.call(engine, tokens, options) })
  Reflect.set(engine, "forwardBatch", async (tokens: number[], _probe: unknown, _ids: boolean, saveAt?: number) => {
    const prefix = Reflect.get(engine, "prefixSaving") as { state: Point } | null
    if (prefix) { prefix.state.conv.push(texture()); prefix.state.windows.push(texture()); prefix.state.linear.push(texture(), texture()) }
    if (saveAt !== undefined) (Reflect.get(engine, "saving") as object[]).push(texture(), texture())
    if (fail) throw new Error("interrupted forward")
    engine.position += tokens.length
    return 0
  })
  return { engine, snapshots, calls, live, fail() { fail = true } }
}

test("a one-off short or long request retains original forwards and allocates no learned state", async () => {
  for (const length of [1, 67, 256, 560, 800, 1800]) {
    const { engine, calls, snapshots } = harness()
    await readPrompt(engine, Array(length).fill(1), 0, [], { prefix: Math.min(21, length - 1), anchor: length })
    assert.deepEqual(calls.map(c => c.length), engine.batchSizes(length))
    assert.ok(calls.every(c => c.prefixAt === undefined))
    assert.ok([...snapshots.values()].every(s => !s.prefix))
    assert.ok(engine.canResume(length), "an exact short-input retry retains its hidden endpoint")
  }
})

test("a revealed sub-batch boundary is captured without splitting and reused by the next changed input", async () => {
  for (const boundary of [1, 21, 128, 255, 256, 389, 511]) {
    const { engine, calls, snapshots } = harness()
    const first = Array(800).fill(1)
    await readPrompt(engine, first, 0, [], { prefix: boundary, anchor: 800 })
    assert.equal(engine.rewind(boundary), 0, "first request cannot predict an unknown shared boundary")
    calls.length = 0
    await readPrompt(engine, first, 0, [], { prefix: boundary, observedPrefix: boundary, anchor: 800 })
    assert.deepEqual(calls.map(c => c.length), [512, 288])
    assert.equal(calls[0].prefixAt, boundary)
    assert.equal(engine.snapshotPosition("prefix"), boundary)
    assert.equal(engine.rewind(boundary), boundary)
    assert.ok(snapshots.size <= engine.snapshotLimit)
    calls.length = 0
    await readPrompt(engine, first, boundary, first.slice(0, boundary), { prefix: boundary, anchor: 800 })
    assert.deepEqual(calls.map(c => c.length), engine.batchSizes(800 - boundary))
    assert.ok(calls.every(c => c.prefixAt === undefined), "warm reuse does not re-copy the learned state")
    assert.equal(engine.snapshotPosition("prefix"), boundary)
  }
})

test("natural boundaries and whole short-input endpoints need no in-batch extraction", async () => {
  for (const boundary of [256, 512, 800]) {
    const length = boundary === 256 ? 256 : 800
    const { engine, calls } = harness()
    await readPrompt(engine, Array(length).fill(1), 0, [], { prefix: boundary, observedPrefix: boundary, anchor: length })
    assert.ok(calls.every(c => c.prefixAt === undefined))
    assert.equal(engine.snapshotPosition("prefix"), boundary)
  }
})

test("simultaneous conversational anchor and learned prefix obey the existing two-snapshot bound", async () => {
  for (const anchor of [67, 256, 500]) {
    const { engine, snapshots } = harness(2, true)
    await readPrompt(engine, Array(560).fill(1), 0, [], { prefix: 256, observedPrefix: 256, anchor })
    assert.equal(engine.snapshotPosition("prefix"), 256)
    assert.ok(snapshots.size <= 2)
    for (let i = 0; i < 10; i++) { engine.position++; engine.snapshot() }
    assert.equal(engine.snapshotPosition("prefix"), 256)
    assert.ok(snapshots.size <= 2)
  }
})

test("one-snapshot models and explicit cache disable keep the original forward path", async () => {
  for (const limit of [1, 2]) {
    const { engine, calls, snapshots } = harness(limit)
    await readPrompt(engine, Array(800).fill(1), 0, [], { prefix: 256, observedPrefix: 256, anchor: 800, ...(limit === 2 && { checkpoints: false }) })
    assert.ok(calls.every(c => c.prefixAt === undefined))
    assert.equal(engine.snapshotPosition("prefix"), undefined)
    if (limit === 2) assert.equal(snapshots.size, 0)
  }
})

test("interrupted in-batch extraction frees every partial texture and clears transient capture state", async () => {
  for (const anchor of [undefined, 500]) {
    const { engine, live, snapshots, fail } = harness(2, true)
    fail()
    await assert.rejects(readPrompt(engine, Array(560).fill(1), 0, [], { prefix: 256, observedPrefix: 256, anchor }), /interrupted forward/)
    assert.equal(live.size, 0)
    assert.equal(snapshots.size, 0)
    assert.equal(Reflect.get(engine, "saving"), null)
    assert.equal(Reflect.get(engine, "prefixSaving"), null)
  }
})

test("stop after a batch retains only committed prefix state and reset frees it", async () => {
  const { engine, snapshots, live } = harness(2)
  const cached: number[] = []
  const forward = engine.forwardHidden.bind(engine)
  let stopped = false
  engine.forwardHidden = async (...args) => { const row = await forward(...args); stopped = true; return row }
  await readPrompt(engine, Array(800).fill(1), 0, cached, { prefix: 256, observedPrefix: 256, anchor: 800, stop: () => stopped })
  assert.equal(cached.length, 512)
  assert.equal(engine.snapshotPosition("prefix"), 256)
  assert.ok(snapshots.size <= 2)
  engine.reset()
  assert.equal(live.size, 0)
})


test("a longer observed prefix can improve an existing shorter hit without changing the tail batches", async () => {
  const { engine, calls } = harness(2)
  const ids = Array(800).fill(1)
  await readPrompt(engine, ids, 0, [], { prefix: 21, observedPrefix: 21, anchor: 800 })
  assert.equal(engine.rewind(256), 21)
  calls.length = 0
  await readPrompt(engine, ids, 21, ids.slice(0, 21), { prefix: 256, observedPrefix: 256, anchor: 800 })
  assert.deepEqual(calls.map(c => c.length), engine.batchSizes(779))
  assert.equal(calls[0].prefixAt, 235)
  assert.equal(engine.rewind(256), 256)
})


test("minor schema prefix extensions preserve the safer retained fallback without imposing a token minimum", () => {
  assert.equal(reusablePrefixFor(0, 180), 180, "cold planning is unchanged")
  assert.equal(reusablePrefixFor(21, 900), 21, "tiny learned prefixes are supported")
  assert.equal(reusablePrefixFor(278, 900, 4), 278, "a new larger workload can replace a trivial old hit")
  assert.equal(reusablePrefixFor(286, 900, 262), 262, "small schema variations keep the verified common fallback")
  assert.equal(reusablePrefixFor(46, 70, 46), 46)
  assert.equal(reusablePrefixFor(48, 70, 46), 46)
  assert.equal(reusablePrefixFor(600, 900, 512), 512, "already useful natural checkpoints do not add copy work")
  assert.equal(reusablePrefixFor(1200, 1300, 512), 1200, "a different size band may improve reuse")
  assert.equal(reusablePrefixFor(240, 900), 240, "rewind has already discarded incompatible longer snapshots")
  assert.equal(reusablePrefixFor(256, undefined), 256, "no system prompt is required")
})


test("a fluctuating dynamic schema reuses its learned fallback instead of repeatedly discarding it", async () => {
  const { engine, calls } = harness(2)
  const ids = Array(800).fill(1)
  await readPrompt(engine, ids, 0, [], { prefix: 900, anchor: 800 })
  const starts: number[] = []
  for (const shared of [278, 262, 286, 262, 280]) {
    const start = engine.rewind(shared)
    starts.push(start)
    const prefix = reusablePrefixFor(shared, 900, engine.snapshotPosition("prefix"))
    calls.length = 0
    await readPrompt(engine, ids, start, ids.slice(0, start), { prefix, ...(prefix !== undefined && prefix > start && { observedPrefix: prefix }), anchor: 800 })
    if (start === 262) assert.ok(calls.every(c => c.prefixAt === undefined), "small extensions do not add copy work")
  }
  assert.deepEqual(starts, [0, 0, 262, 262, 262])
})


test("a profile switch keeps the normal long-prefix checkpoint despite a short previous match", async () => {
  const { engine, calls } = harness(8)
  const ids = Array(10000).fill(1)
  const plan = adaptivePrefixFor({ shared: 50, declared: 1400, previousShared: 4, natural: 1024, framing: 4 })
  assert.equal(plan.prefix, 1400)
  await readPrompt(engine, ids, 0, [], { prefix: plan.prefix, anchor: ids.length })
  assert.ok(calls.every(c => c.prefixAt === undefined))
  assert.equal(engine.snapshotPosition("prefix"), 1024)
  assert.equal(engine.rewind(1200), 1024)
})

test("repeated short content supersedes an unsuitable natural checkpoint without changing batches", async () => {
  const { engine, calls } = harness(2)
  const ids = Array(800).fill(1)
  let previousShared: number | undefined
  for (const shared of [389, 402]) {
    engine.rewind(shared)
    const plan = adaptivePrefixFor({ shared, declared: 700, previousShared, natural: 512, framing: 4 })
    previousShared = plan.previousShared
    const observedPrefix = plan.prefix !== undefined && plan.prefix <= shared ? plan.prefix : undefined
    calls.length = 0
    await readPrompt(engine, ids, 0, [], { prefix: plan.prefix, observedPrefix, anchor: 800 })
    assert.deepEqual(calls.map(c => c.length), [512, 288])
  }
  assert.equal(engine.rewind(389), 389)
})

test("template-only matches do not replace natural checkpoints and short content has no fixed cutoff", () => {
  assert.equal(adaptivePrefixFor({ shared: 4, declared: 1400, framing: 4 }).prefix, 1400)
  assert.equal(adaptivePrefixFor({ shared: 5, framing: 4 }).prefix, 5)
  assert.equal(adaptivePrefixFor({ shared: 21, framing: 4 }).prefix, 21)
})

test("framing probes are memoized, token-verified, and never alter the real request", () => {
  let calls = 0
  const tokenizer = { encodeChat(chat: Array<{ content: string }>) { calls++; return [1, 2, chat[0].content.charCodeAt(0), 3] } } as unknown as Parameters<typeof templateFramingFor>[0]
  const input = [1, 2, 90, 3], original = [...input]
  assert.equal(templateFramingFor(tokenizer, input, "system", false), 2)
  assert.equal(calls, 2)
  assert.equal(templateFramingFor(tokenizer, input, "system", false), 2)
  assert.equal(calls, 2)
  assert.equal(templateFramingFor(tokenizer, [9, 2, 90], "system", false), 0)
  assert.deepEqual(input, original)
})


test("a reusable ordinary checkpoint is promoted without another GPU-state copy", async () => {
  const { engine, calls } = harness(8)
  const ids = Array(10000).fill(1)
  await readPrompt(engine, ids.slice(0, 800), 0, [], { prefix: 20, anchor: 800 })
  assert.equal(engine.rewind(512), 512)
  assert.equal(engine.snapshotPosition("prefix"), undefined)
  calls.length = 0
  await readPrompt(engine, ids, 512, ids.slice(0, 512), { prefix: 512, anchor: ids.length })
  assert.ok(calls.every(c => c.prefixAt === undefined))
  assert.equal(engine.snapshotPosition("prefix"), 512)
  assert.equal(engine.rewind(600), 512, "the long tail cannot evict the promoted reusable point")
})


test("an exact retry does not promote the full endpoint over a useful shorter prefix", async () => {
  const { engine } = harness(2)
  const ids = Array(800).fill(1)
  await readPrompt(engine, ids, 0, [], { prefix: 21, observedPrefix: 21, anchor: 800 })
  await readPrompt(engine, ids, 800, [...ids], { prefix: 800, anchor: 800 })
  assert.equal(engine.snapshotPosition("prefix"), 21)
  engine.position = 810; engine.snapshot()
  assert.equal(engine.rewind(21), 21)
})
