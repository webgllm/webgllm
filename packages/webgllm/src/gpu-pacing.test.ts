import assert from "node:assert/strict"
import test from "node:test"
import { Gpu, type Pass } from "./gpu/gl.js"
import { SubmissionBudget } from "./gpu/submission-budget.js"

function harness(readsPerSlice: number) {
  const gpu = Object.create(Gpu.prototype) as Gpu
  Object.assign(gpu, {
    readsPerSlice,
    submittedWork: 0,
    sliceReads: 0,
    sliceMatrixReads: 0,
    sliceCalibrates: true
  })
  const passes: Pass[] = []
  let finishes = 0
  gpu.run = (pass) => {
    passes.push(pass)
  }
  gpu.pace = async () => {
    finishes++
  }
  return { gpu, passes, finishes: () => finishes }
}
const pass = {
  program: {},
  inputs: {},
  output: { width: 20, height: 20 },
  rect: { x: 3, y: 7, width: 8, height: 5 },
  work: 4
} as Pass

test("paced bands cover exactly the original rectangle, using absolute coordinates", async () => {
  const h = harness(64)
  await h.gpu.runAsync(pass)
  assert.deepEqual(
    h.passes.map((p) => p.rect),
    [
      { x: 3, y: 7, width: 8, height: 2 },
      { x: 3, y: 9, width: 8, height: 2 },
      { x: 3, y: 11, width: 8, height: 1 }
    ]
  )
  assert.equal(h.finishes(), 2)
  assert.ok(
    h.passes.every(
      (p) => p.output === pass.output && p.program === pass.program
    )
  )
})
test("disabled pacing preserves a single submission without a fence", async () => {
  const h = harness(0)
  await h.gpu.runAsync(pass)
  assert.deepEqual(h.passes, [pass])
  assert.equal(h.finishes(), 0)
})
test("a row larger than the budget still progresses, and MRT uses the full target", async () => {
  const h = harness(1)
  await h.gpu.runAsync({
    ...pass,
    rect: undefined,
    output: [pass.output, pass.output]
  } as Pass)
  assert.equal(h.passes.length, 20)
  assert.deepEqual(h.passes[19].rect, { x: 0, y: 19, width: 20, height: 1 })
})

test("short output heads are sliced even below the read budget", async () => {
  const h = harness(100000)
  await h.gpu.runAsync(pass, 2)
  assert.deepEqual(
    h.passes.map((p) => p.rect?.height),
    [2, 2, 1]
  )
  assert.equal(h.finishes(), 3)
})

test("paced matrix bands align to raster tiles without consuming the leftover sliver", async () => {
  const h = harness(68 * 128 * 1024)
  await h.gpu.runAsync({
    ...pass,
    name: "matmul",
    work: 1024,
    rect: { x: 0, y: 0, width: 128, height: 128 }
  })
  assert.deepEqual(
    h.passes.map((p) => p.rect),
    [
      { x: 0, y: 0, width: 128, height: 64 },
      { x: 0, y: 64, width: 128, height: 64 }
    ]
  )
  assert.equal(h.finishes(), 1)
})

test("a delayed load-frame window cannot reduce the new prompt's budget", () => {
  let observed = 0,
    heartbeats = 0
  const budget = {
    reads: 256,
    heartbeat() {
      heartbeats++
    },
    frame() {
      observed++
    }
  }
  const gpu = Object.create(Gpu.prototype) as Gpu
  Object.assign(gpu, {
    phaseAt: 1000,
    budgets: { load: budget, prompt: budget, decode: budget },
    submissionBudget: budget
  })
  gpu.observeFrames(150, 16.7, 24, 900)
  assert.equal(observed, 0)
  assert.equal(heartbeats, 3)
  gpu.observeFrames(16.7, 16.7, 24, 1001)
  assert.equal(observed, 1)
})


test("a small projection starts a new chunk instead of splitting across its boundary", async () => {
  const h = harness(1024 * 128 * 1024)
  Object.assign(h.gpu, { sliceReads: 1000 * 128 * 1024 })
  await h.gpu.runAsync({ ...pass, name: "matmul", work: 1024,
    rect: { x: 0, y: 0, width: 128, height: 64 } })
  assert.deepEqual(h.passes.map(p => p.rect), [{ x: 0, y: 0, width: 128, height: 64 }])
  assert.equal(h.finishes(), 1)
})

test("pacing recovery is limited to a collapsed quota during high-refresh inference", () => {
  for (const [phase, cadence, quotaCollapsed, expected] of [
    ["prompt", 16.7, true, false],
    ["prompt", 8.34, false, false],
    ["load", 8.34, true, false],
    ["prompt", 8.34, true, true]
  ] as const) {
    const budget = {
      reads: 2 ** 25,
      recovery: false,
      quotaCollapsed,
      heartbeat() { return undefined },
      frame() { return undefined },
      enableRecovery() { this.recovery = true }
    }
    const gpu = Object.create(Gpu.prototype) as Gpu
    Object.assign(gpu, {
      phase,
      phaseAt: -Infinity,
      canRecoverPacing: true,
      submissionBudget: budget,
      budgets: { load: budget, prompt: budget, decode: budget }
    })
    gpu.observeFrames(cadence, cadence, 24)
    assert.equal(budget.recovery, expected)
  }
})

test("prompt pacing calibrates only a high-refresh session with repeated slow empty fences", async (t) => {
  let now = 100
  t.mock.method(performance, "now", () => now)
  for (const [expectedFrameMs, costs, qualifies] of [
    [16.7, [5, 5, 5], false],
    [8.34, [0.2, 5, 0.2], false],
    [8.34, [5, 0.2, 5], true]
  ] as const) {
    const gpu = Object.create(Gpu.prototype) as Gpu
    const starts: number[] = []
    let finishes = 0, decode = false
    Object.assign(gpu, {
      expectedFrameMs,
      lastObservedFrame: now,
      canRecoverPacing: true,
      budgets: {
        prompt: { enableRecovery: (reads: number) => starts.push(reads) },
        decode: { enableRecovery: () => { decode = true } }
      },
      finish: async () => { now += costs[finishes++] }
    })
    await gpu.calibratePromptPacing()
    assert.equal(finishes, expectedFrameMs > 12 ? 0 : 3)
    assert.equal(decode, qualifies)
    assert.deepEqual(starts, qualifies ? [160 * 2 ** 20] : [])
    if (expectedFrameMs <= 12) {
      assert.equal(gpu.calibratePromptPacing(), undefined)
    }
  }
})

test("writing earns its queue back on frames that are slow on average but have no long one", () => {
  const budget = new SubmissionBudget(true)
  budget.configure(2 ** 28, 4, 2)
  budget.enableRecovery()
  // Backed off as far as it goes, by an earlier burst of long frames.
  budget.ahead = 0
  budget.layers = 1
  budget.waiting(0)
  let now = 1000
  const report = (worstMs: number, meanMs: number) => {
    budget.decode(now)
    budget.frame(worstMs, 8.3, 24, now, meanMs, now - meanMs * 24)
    now += meanMs * 24
  }
  // A 120 Hz page painting about 100 frames a second: no frame over 32 ms.
  for (let i = 0; i < 4; i++) report(25, 10)
  assert.equal(budget.ahead, 1)
  assert.equal(budget.layers, 2)
  for (let i = 0; i < 4; i++) report(25, 10)
  assert.equal(budget.ahead, 2)
  // Long frames one after another still back it off.
  for (let i = 0; i < 2; i++) {
    budget.frame(60, 8.3, 1, now, 60, now - 60)
    now += 60
  }
  assert.equal(budget.ahead, 1)
})
