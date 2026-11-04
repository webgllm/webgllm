import test from "node:test"
import assert from "node:assert/strict"
import {
  SubmissionBudget,
  calibratedReadQuota
} from "./gpu/submission-budget.js"

test("fence fallback adjusts unobserved work within its calibrated quota", () => {
  const b = new SubmissionBudget()
  b.configure(2 ** 27, 4)
  b.sample(2 ** 27, 32, 2)
  const low = b.reads
  b.sample(2 ** 27, 2, 2)
  assert.ok(b.reads > low && b.reads <= 2 ** 27)
  b.configure(2 ** 27, 4)
  assert.ok(b.reads <= 2 ** 27)
})

test("fixed/disabled pacing ignores feedback and invalid fence samples", () => {
  const b = new SubmissionBudget()
  b.configure(2 ** 27, 0)
  b.sample(2 ** 27, 1, 2)
  assert.equal(b.reads, 2 ** 27)
  b.configure(0, 4)
  b.sample(2 ** 27, 1, 2)
  assert.equal(b.reads, 0)
  b.configure(2 ** 27, 4)
  for (const ms of [0, NaN, Infinity, 1000]) b.sample(2 ** 27, ms, 2)
  assert.equal(b.reads, 2 ** 27)
})

test("isolated presentation misses change queue drains before shrinking efficient matrix bands", () => {
  const b = new SubmissionBudget()
  b.configure(2 ** 28, 4, 4)
  b.work(10)
  b.frame(25, 16.7, 24, 20, 18)
  assert.equal(b.reads, 2 ** 28)
  assert.equal(b.presentationWindow, 4)
  b.frame(150, 16.7, 1, 30, 150)
  assert.ok(b.reads < 2 ** 28)
  const low = b.reads
  b.heartbeat(40, true)
  b.sample(2 ** 27, 0.5, 2, 50)
  assert.equal(b.reads, low)
  b.heartbeat(2000, true)
  b.sample(2 ** 27, 0.5, 2, 2010)
  assert.ok(b.reads > low && b.reads <= 2 ** 28)
})

test("healthy frames keep the calibrated matrix quota until sustained presentation fairness is proven", () => {
  const b = new SubmissionBudget()
  b.configure(2 ** 28, 4)
  for (let i = 0; i < 4; i++) {
    b.work(i * 100)
    b.frame(16.7, 16.7, 24, i * 100 + 10)
    b.sample(2 ** 28, 0.2, 2, i * 100 + 20)
  }
  assert.equal(b.reads, 2 ** 28)
})

test("decode quanta grow with smooth frames and back off on stutter", () => {
  const b = new SubmissionBudget()
  b.configure(2 ** 28, 4)
  for (let i = 0; i < 4; i++) {
    b.decode(i * 100)
    b.frame(16.7, 16.7, 24, i * 100 + 10)
  }
  assert.equal(b.layers, 2)
  b.frame(50, 16.7, 1, 450)
  assert.equal(b.layers, 1)
})

test("recent cadence suppresses slow fence fallback after idle", () => {
  const b = new SubmissionBudget()
  b.configure(2 ** 28, 4)
  b.heartbeat(5000)
  b.sample(2 ** 28, 40, 2, 5010)
  assert.equal(b.reads, 2 ** 28)
  b.sample(2 ** 28, 40, 2, 6100)
  assert.ok(b.reads < 2 ** 28)
})

test("fast complete-layer fences scale decode without device limits", () => {
  const b = new SubmissionBudget()
  b.configure(2 ** 28, 4)
  b.heartbeat(10, true)
  b.layerSample(2, 1, 3)
  assert.equal(b.layers, 4)
  b.layerSample(4, 1, 3)
  assert.equal(b.layers, 8)
  b.layerSample(8, 20, 3)
  assert.equal(b.layers, 8)
  b.frame(80, 16.7, 1, 30)
  assert.equal(b.layers, 4)
})

test("whole-workload calibration bounds invalid estimates and scales fast workloads", () => {
  const base = 2 ** 28
  assert.equal(calibratedReadQuota(base, 3e9, 200, 16.7), base)
  assert.ok(calibratedReadQuota(base, 3e9, 2, 16.7) > base)
  assert.equal(calibratedReadQuota(base, 3e9, 0.01, 16.7), 2 ** 34)
  for (const ms of [0, NaN, Infinity])
    assert.equal(calibratedReadQuota(base, 3e9, ms, 16.7), base)
})

test("240 Hz cadence lowers the four-millisecond completion target", () => {
  const high = new SubmissionBudget(),
    normal = new SubmissionBudget()
  for (const b of [high, normal]) {
    b.configure(2 ** 28, 4)
    b.frame(150, 16.7, 1, 20)
  }
  high.heartbeat(2000, true, 1000 / 240)
  normal.heartbeat(2000, true, 1000 / 60)
  high.sample(2 ** 27, 1.6, 2, 2010)
  normal.sample(2 ** 27, 1.6, 2, 2010)
  assert.equal(high.reads, 2 ** 27)
  assert.ok(normal.reads > high.reads)
})

test("presentation policy learns only from active work and startup reset discards compiler stalls", () => {
  const b = new SubmissionBudget()
  b.configure(2 ** 28, 4, 4)
  b.frame(16.7, 16.7, 96, 10, 16.7)
  assert.equal(b.presentationWindow, 4)
  for (let i = 0; i < 4; i++) {
    b.work(100 + i * 100)
    b.frame(16.7, 16.7, 24, 110 + i * 100, 16.7)
  }
  assert.equal(b.presentationWindow, 12)
  b.frame(33.4, 16.7, 24, 600, 20)
  assert.equal(b.presentationWindow, 4)
  assert.equal(b.presentationMax, 11)
  b.reset()
  assert.equal(b.presentationMax, 11)
  b.reset(true)
  assert.equal(b.presentationWindow, 4)
  assert.equal(b.presentationMax, 24)
})

test("mean cadence detects an isolated stall hidden below p95", () => {
  const b = new SubmissionBudget()
  b.configure(2 ** 28, 4, 4)
  b.work(10)
  b.frame(16.7, 16.7, 24, 20, 18)
  assert.equal(b.presentationMax, 4)
  assert.equal(b.reads, 2 ** 28)
})

test("larger queues require proven presentation fairness and roll back without shrinking efficient bands", () => {
  const b = new SubmissionBudget()
  b.configure(2 ** 28, 4, 4)
  for (let i = 0; i < 4; i++) {
    b.work(i * 100)
    b.frame(16.7, 16.7, 24, i * 100 + 10)
  }
  assert.equal(b.reads, 2 ** 28)
  for (let i = 4; i < 14; i++) {
    b.work(i * 100)
    b.frame(16.7, 16.7, 24, i * 100 + 10)
    b.sample(b.reads, 0.5, 2, i * 100 + 11)
  }
  assert.ok(b.reads > 2 ** 28)
  const beforeMiss = b.reads
  b.frame(33.4, 16.7, 1, 1500)
  assert.equal(b.reads, beforeMiss)
  for (let i = 0; i < 50; i++) {
    b.work(2000 + i * 100)
    b.frame(16.7, 16.7, 24, 2010 + i * 100)
  }
  assert.ok(b.reads >= beforeMiss && b.reads <= 2 ** 34)
})

test("only completed smooth real-compute history can become a persisted pacing recommendation", () => {
  const b = new SubmissionBudget()
  b.configure(2 ** 28, 4, 4)
  assert.equal(Reflect.get(b, "validatedPresentation"), null)
  for (let i = 0; i < 26; i++) {
    b.work(i * 100)
    b.frame(8.34, 8.34, 24, i * 100 + 10, 8.34)
  }
  const measured = b.validatedPresentation
  assert.ok(measured)
  assert.equal(measured.frameMs, 8.3)
  assert.equal(measured.window, 48)
  b.frame(20, 8.34, 1, 2000, 20)
  assert.equal(Reflect.get(b, "validatedPresentation"), null)
  b.reset(true)
  assert.equal(Reflect.get(b, "validatedPresentation"), null)
})

test("saved overlap restores the calibrated quota immediately after a critical stall", () => {
  const b = new SubmissionBudget()
  b.configure(2 ** 28, 4, 4)
  b.primePresentation(24, 2 ** 33)
  assert.equal(b.reads, 2 ** 33)
  b.frame(150, 8.34, 1, 2000, 150)
  assert.equal(b.reads, 2 ** 28)
  assert.equal(Reflect.get(b, "validatedPresentation"), null)
})

test("smooth presentation alone does not increase matrix bands without a fast matrix fence", () => {
  const b = new SubmissionBudget()
  b.configure(2 ** 28, 4, 4)
  for (let i = 0; i < 100; i++) {
    b.work(i * 100)
    b.frame(16.7, 16.7, 24, i * 100 + 10)
    b.sample(b.reads, 8, 2, i * 100 + 11)
  }
  assert.equal(b.reads, 2 ** 28)
  assert.equal(b.presentationWindow, 24)
})

test("isolated missed frames preserve efficient bands and recover the drain window", () => {
  const b = new SubmissionBudget()
  b.configure(2 ** 28, 4, 4)
  for (const at of [10, 1000, 2000]) b.frame(33.4, 16.7, 1, at)
  assert.equal(b.reads, 2 ** 28)
  assert.equal(b.presentationMax, 4)
  for (let i = 0; i < 24; i++) {
    b.work(3000 + i * 100)
    b.frame(16.7, 16.7, 24, 3010 + i * 100)
  }
  assert.equal(b.presentationMax, 24)
  assert.equal(b.presentationWindow, 24)
})

test("repeated stalls in one burst still reduce the matrix quota", () => {
  const b = new SubmissionBudget()
  b.configure(2 ** 28, 4, 4)
  for (const at of [10, 50, 90]) b.frame(33.4, 16.7, 1, at)
  assert.ok(b.reads < 2 ** 28)
})

test("efficient matrix samples recover even after backoff below half the initial quota", () => {
  const b = new SubmissionBudget()
  b.configure(2 ** 28, 4, 4)
  b.frame(150, 16.7, 1, 10)
  b.frame(150, 16.7, 1, 20)
  const reduced = b.reads
  assert.ok(reduced < 2 ** 27)
  b.heartbeat(2000, true, 16.7)
  b.sample(reduced, 0.5, 2, 2010)
  assert.ok(b.reads > reduced)
})

test("moderate callback jitter with healthy mean cadence does not revoke proven overlap", () => {
  const b = new SubmissionBudget()
  b.configure(2 ** 28, 4, 4)
  b.primePresentation(24, 2 ** 30)
  b.frame(13.4, 8.33, 24, 100, 8.34)
  assert.equal(b.reads, 2 ** 30)
  assert.equal(b.relaxationAllowed, true)
})

test("an isolated missed frame rolls back immediately without permanently disabling recovery", () => {
  const b = new SubmissionBudget()
  b.configure(2 ** 28, 4, 4)
  b.primePresentation(24, 2 ** 30)
  b.frame(20, 8.33, 1, 100, 20)
  assert.equal(b.reads, 2 ** 30)
  assert.equal(b.relaxationAllowed, true)
  b.frame(20, 8.33, 1, 130, 20)
  b.frame(20, 8.33, 1, 160, 20)
  assert.equal(b.reads, 2 ** 28)
  assert.equal(b.relaxationAllowed, false)
})

for (const fps of [60, 120]) {
  test(`healthy ${fps} FPS rendering uses the same maximum drain interval`, () => {
    const b = new SubmissionBudget()
    b.configure(2 ** 28, 4, 4)
    const frame = 1000 / fps
    for (let i = 0; i < 40; i++) {
      b.work(i * 100)
      b.frame(frame, frame, 24, i * 100 + 10)
    }
    assert.equal(b.presentationWindow, Math.round(400 / frame))
    assert.ok(Math.abs(b.presentationWindow * frame - 400) < frame)
  })
}
