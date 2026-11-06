// A pending readback's pixel buffer and fence are freed when the guess read
// forward meanwhile (Engine.forwardGuess) throws, not only once it's read.
import assert from "node:assert/strict"
import test from "node:test"
import { Engine } from "./engine.js"
import { Gpu } from "./gpu/gl.js"
import { SubmissionBudget } from "./gpu/submission-budget.js"

function fakeGpu() {
  const live = { buffers: 0, syncs: 0 }
  const gpu = Object.create(Gpu.prototype) as Gpu
  Object.assign(gpu, {
    lost: false,
    submissionBudget: new SubmissionBudget(),
    framebuffer: () => ({}),
    submitGpuWork: () => undefined,
    run: () => undefined,
    program: () => ({}),
    texture: () => ({}),
    gl: {
      PIXEL_PACK_BUFFER: 1,
      FRAMEBUFFER: 2,
      createBuffer: () => (live.buffers++, {}),
      deleteBuffer: () => void live.buffers--,
      fenceSync: () => (live.syncs++, {}),
      deleteSync: () => void live.syncs--,
      bindBuffer() {},
      bindFramebuffer() {},
      bufferData() {},
      readPixels() {},
      flush() {}
    }
  })
  return { gpu, live }
}

function fakeEngine() {
  const { gpu, live } = fakeGpu()
  const engine = Object.create(Engine.prototype) as Engine
  Object.assign(engine, {
    gpu,
    config: { vocabSize: 1024 },
    buffers: { chunkMax: {}, chunkRank: {}, chunkByRank: {}, candidates: {}, ids: {}, mask: {} },
    allowedEnabled: false,
    allowedTokens: null,
    allowedRows: 0,
    allowedStride: 0,
    greedyOut: {},
    head: async () => ({ logits: {}, split: 1, stride: 1 }),
    forwardGuess: async () => {
      throw new Error("The cache is full.")
    }
  })
  return { engine, live }
}

test("greedy readback frees its buffer and fence when the guess throws", async () => {
  const { engine, live } = fakeEngine()
  await assert.rejects(engine.argmax(0, 1, 1, [], true), /cache is full/)
  assert.deepEqual(live, { buffers: 0, syncs: 0 })
})

test("top-k readback frees its buffer and fence when the guess throws", async () => {
  const { engine, live } = fakeEngine()
  await assert.rejects(engine.pick(0, 1, 4, 1, [], true), /cache is full/)
  assert.deepEqual(live, { buffers: 0, syncs: 0 })
})
