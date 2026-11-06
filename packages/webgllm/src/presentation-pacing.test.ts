import test from "node:test"
import assert from "node:assert/strict"
import { Gpu } from "./gpu/gl.js"
import { SubmissionBudget } from "./gpu/submission-budget.js"

test("presentation drains periodically empty the compute queue while ordinary bands overlap", async () => {
  const gpu = Object.create(Gpu.prototype) as Gpu,
    budget = new SubmissionBudget()
  budget.configure(2 ** 28, 4, 2)
  let deleted = 0
  Object.assign(gpu, {
    lost: false,
    paced: [],
    submissionBudget: budget,
    expectedFrameMs: 16.7,
    presentationFrames: 2,
    lastPresentationDrain: -Infinity,
    gl: {
      SYNC_GPU_COMMANDS_COMPLETE: 1,
      SYNC_STATUS: 2,
      SIGNALED: 3,
      fenceSync: () => ({}),
      flush: () => undefined,
      ALREADY_SIGNALED: 3,
      CONDITION_SATISFIED: 4,
      WAIT_FAILED: 5,
      clientWaitSync: (_sync: unknown, flags: number, timeout: number) => {
        assert.equal(flags, 0)
        assert.equal(timeout, 0)
        return 3
      },
      deleteSync: () => {
        deleted++
      },
      isContextLost: () => false
    }
  })
  await gpu.pace(1, true, 1024)
  assert.equal(deleted, 1)
  assert.equal(Reflect.get(gpu, "paced").length, 0)
  await gpu.pace(1, true, 1024)
  assert.equal(deleted, 1)
  assert.equal(Reflect.get(gpu, "paced").length, 1)
  Reflect.set(gpu, "lastPresentationDrain", -Infinity)
  await gpu.pace(1, true, 1024)
  assert.equal(deleted, 3)
  assert.equal(Reflect.get(gpu, "paced").length, 0)
  Reflect.set(gpu, "phase", "load")
  Reflect.set(gpu, "lastPresentationDrain", -Infinity)
  await gpu.pace(1, true, 0)
  assert.equal(deleted, 4)
  assert.equal(Reflect.get(gpu, "paced").length, 0)
})
