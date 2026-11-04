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
  Reflect.set(globalThis, "document", { visibilityState: "visible" })
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
  Reflect.set(globalThis, "document", { visibilityState: "visible" })
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
