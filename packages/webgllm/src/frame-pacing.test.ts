import test from "node:test"
import assert from "node:assert/strict"
import { monitorFrames } from "./frame-pacing.js"
test("frame monitoring is optional outside a window", () => {
  const stop = monitorFrames(() => assert.fail("unexpected frame feedback"))
  stop()
})

test("frame feedback reports 120 Hz and the absolute start of its observation window", () => {
  const frames: FrameRequestCallback[] = [],
    sent: number[][] = []
  const previous = {
    raf: globalThis.requestAnimationFrame,
    cancel: globalThis.cancelAnimationFrame,
    document: globalThis.document
  }
  Reflect.set(
    globalThis,
    "requestAnimationFrame",
    (fn: FrameRequestCallback) => {
      frames.push(fn)
      return frames.length
    }
  )
  Reflect.set(globalThis, "cancelAnimationFrame", () => undefined)
  Reflect.set(globalThis, "document", Object.assign(new EventTarget(), { visibilityState: "visible" }))
  try {
    const stop = monitorFrames((...values) => sent.push(values as number[]))
    for (let i = 0; i <= 24; i++) frames.shift()?.(100 + (i * 1000) / 120)
    assert.equal(sent.length, 1)
    assert.ok(Math.abs(sent[0][1] - 1000 / 120) < 1e-6)
    assert.equal(sent[0][2], 24)
    assert.ok(Math.abs(sent[0][3] - (performance.timeOrigin + 100)) < 1)
    stop()
  } finally {
    Reflect.set(globalThis, "requestAnimationFrame", previous.raf)
    Reflect.set(globalThis, "cancelAnimationFrame", previous.cancel)
    Reflect.set(globalThis, "document", previous.document)
  }
})

test("short catch-up callbacks remain in the mean instead of inventing a slowdown", () => {
  const queued: FrameRequestCallback[] = [],
    sent: number[][] = []
  const previous = {
    raf: globalThis.requestAnimationFrame,
    cancel: globalThis.cancelAnimationFrame,
    document: globalThis.document
  }
  Reflect.set(
    globalThis,
    "requestAnimationFrame",
    (fn: FrameRequestCallback) => {
      queued.push(fn)
      return queued.length
    }
  )
  Reflect.set(globalThis, "cancelAnimationFrame", () => undefined)
  Reflect.set(globalThis, "document", Object.assign(new EventTarget(), { visibilityState: "visible" }))
  try {
    const stop = monitorFrames((...values) => sent.push(values as number[]))
    let now = 100
    queued.shift()?.(now)
    for (let i = 0; i < 24; i++) {
      now += i % 6 === 0 ? (2 * 1000) / 120 - 3 : i % 6 === 1 ? 3 : 1000 / 120
      queued.shift()?.(now)
    }
    assert.equal(sent.length, 1)
    assert.equal(sent[0][2], 24)
    assert.ok(Math.abs(sent[0][1] - 1000 / 120) < 1e-6)
    assert.ok(Math.abs(sent[0][4] - 1000 / 120) < 1e-6)
    stop()
  } finally {
    Reflect.set(globalThis, "requestAnimationFrame", previous.raf)
    Reflect.set(globalThis, "cancelAnimationFrame", previous.cancel)
    Reflect.set(globalThis, "document", previous.document)
  }
})


for (const fps of [60, 120])
  test(`whole-millisecond timestamps retain healthy ${fps} Hz cadence`, () => {
    const frames: FrameRequestCallback[] = [], sent: number[][] = []
    const previous = { raf: globalThis.requestAnimationFrame, cancel: globalThis.cancelAnimationFrame, document: globalThis.document }
    Reflect.set(globalThis, "requestAnimationFrame", (fn: FrameRequestCallback) => { frames.push(fn); return frames.length })
    Reflect.set(globalThis, "cancelAnimationFrame", () => undefined)
    Reflect.set(globalThis, "document", Object.assign(new EventTarget(), { visibilityState: "visible" }))
    try {
      const stop = monitorFrames((...values) => sent.push(values as number[]))
      for (let i = 0; i <= 120; i++) frames.shift()?.(Math.floor(100 + i * 1000 / fps))
      assert.equal(sent.length, 5)
      for (const sample of sent) {
        assert.ok(Math.abs(sample[1] - 1000 / fps) < 0.1)
        assert.ok(sample[4] <= sample[1] * 1.03)
      }
      stop()
    } finally {
      Reflect.set(globalThis, "requestAnimationFrame", previous.raf)
      Reflect.set(globalThis, "cancelAnimationFrame", previous.cancel)
      Reflect.set(globalThis, "document", previous.document)
    }
  })

test("visible long gaps are reported, hidden gaps are skipped, and 120 Hz can settle at 60 Hz", (t) => {
  const queued: FrameRequestCallback[] = [], sent: number[][] = []
  const doc = Object.assign(new EventTarget(), { visibilityState: "visible" })
  for (const [key, value] of Object.entries({
    requestAnimationFrame: (fn: FrameRequestCallback) => queued.push(fn),
    cancelAnimationFrame: () => undefined,
    document: doc
  })) {
    const original = Object.getOwnPropertyDescriptor(globalThis, key)
    Object.defineProperty(globalThis, key, { configurable: true, value })
    t.after(() => {
      if (original) Object.defineProperty(globalThis, key, original)
      else Reflect.deleteProperty(globalThis, key)
    })
  }
  const stop = monitorFrames((...values) => sent.push(values as number[]))
  let now = 100
  const next = (elapsed: number) => queued.shift()?.(now += elapsed)
  next(0)
  for (let i = 0; i < 24; i++) next(1000 / 120)
  next(1200)
  assert.equal(sent.at(-1)?.[0], 1200)
  assert.equal(sent.at(-1)?.[2], 1)
  for (let i = 0; i < 8; i++) next(1000 / 60)
  const settled = sent.at(-1)
  assert.ok(settled && Math.abs(settled[1] - 1000 / 60) < 1e-6)
  assert.ok(sent.slice(2).every(([interval, expected]) => interval < expected * 1.8))
  const before = sent.length
  for (const visibilityState of ["hidden", "visible"]) {
    doc.visibilityState = visibilityState
    doc.dispatchEvent(new Event("visibilitychange"))
  }
  next(5000)
  assert.equal(sent.length, before)
  stop()
})
