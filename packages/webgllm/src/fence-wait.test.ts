import test from "node:test"
import assert from "node:assert/strict"
import { Gpu } from "./gpu/gl.js"

test("short completed fences avoid timer sleeps even at the periodic polling threshold", async (t) => {
  let now = 0,
    polls = 0,
    yields = 0
  t.mock.method(performance, "now", () => (now += 0.01))
  const gpu = Object.create(Gpu.prototype) as Gpu
  Object.assign(gpu, {
    lost: false,
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
