import test from "node:test"
import assert from "node:assert/strict"
import { Gpu } from "./gpu/gl.js"
import { SubmissionBudget } from "./gpu/submission-budget.js"

test("short completed fences avoid timer sleeps even at the periodic polling threshold", async (t) => {
  let now = 0,
    polls = 0,
    yields = 0
  t.mock.method(performance, "now", () => (now += 0.01))
  const gpu = Object.create(Gpu.prototype) as Gpu
  Object.assign(gpu, {
    lost: false,
    submissionBudget: new SubmissionBudget(),
    phase: "prompt",
    pollsBeforeSleep: 32,
    polls: 31,
    yieldOnce: async () => {
      yields++
    },
    gl: {
      isContextLost: () => false,
      WAIT_FAILED: 1,
      TIMEOUT_EXPIRED: 2,
      ALREADY_SIGNALED: 3,
      CONDITION_SATISFIED: 4,
      clientWaitSync: () => (++polls < 3 ? 2 : 3)
    }
  })
  await Reflect.get(Gpu.prototype, "wait").call(gpu, {})
  assert.equal(yields, 2)
  assert.equal(Reflect.get(gpu, "polls"), 31)
})

test("a prompt's wait sleeps on timers once it isn't a short one", async (t) => {
  let now = 0,
    polls = 0,
    yields = 0,
    sleeps = 0
  t.mock.method(performance, "now", () => (now += 0.05))
  t.mock.method(globalThis, "setTimeout", (resolve: () => void) => {
    sleeps++
    now += 1
    resolve()
  })
  const gpu = Object.create(Gpu.prototype) as Gpu
  Object.assign(gpu, {
    lost: false,
    submissionBudget: new SubmissionBudget(),
    phase: "prompt",
    pollsBeforeSleep: 32,
    polls: 0,
    promptSpinMs: 0.25,
    throttledUntil: 0,
    yieldOnce: async () => {
      yields++
    },
    gl: {
      isContextLost: () => false,
      WAIT_FAILED: 1,
      TIMEOUT_EXPIRED: 2,
      ALREADY_SIGNALED: 3,
      CONDITION_SATISFIED: 4,
      // Done after 12 looks: a few on messages, the rest between sleeps.
      clientWaitSync: () => (++polls < 12 ? 2 : 3)
    }
  })
  await Reflect.get(Gpu.prototype, "wait").call(gpu, {})
  assert.ok(yields >= 1 && yields <= 5, `${yields} polls on messages`)
  assert.equal(yields + sleeps, 11)
  assert.ok(sleeps >= 6)
})

test("a wait's fence polls wait inside the call where the browser allows a timeout, and not where it doesn't", () => {
  const timeouts: number[] = []
  const gpu = Object.create(Gpu.prototype) as Gpu
  Object.assign(gpu, {
    timedPolls: true,
    maxPollNs: 1e6,
    gl: {
      WAIT_FAILED: 1,
      TIMEOUT_EXPIRED: 2,
      ALREADY_SIGNALED: 3,
      CONDITION_SATISFIED: 4,
      clientWaitSync: (_sync: object, _flags: number, timeout: number) => (timeouts.push(timeout), 4)
    }
  })
  const ready = Reflect.get(Gpu.prototype, "fenceReady")
  // A wait's poll, then a look in passing.
  assert.equal(ready.call(gpu, {}, true), true)
  ready.call(gpu, {})
  // A browser that allows no timeout (Chrome, Safari).
  Object.assign(gpu, { maxPollNs: 0 })
  ready.call(gpu, {}, true)
  // Switched off (TUNING.timedFencePolls 0).
  Object.assign(gpu, { maxPollNs: 1e6, timedPolls: false })
  ready.call(gpu, {}, true)
  assert.deepEqual(timeouts, [1e6, 0, 0, 0])
})
