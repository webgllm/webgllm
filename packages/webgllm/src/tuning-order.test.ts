import test from "node:test"
import assert from "node:assert/strict"
import { Engine } from "./engine.js"

test("tuning awaits an asynchronous attention sample before fencing or starting another sample", async () => {
  const descriptor = Object.getOwnPropertyDescriptor(globalThis, "performance")
  if (!descriptor) throw new Error("Missing performance descriptor")
  let clock = 0,
    active = 0,
    calls = 0,
    maxActive = 0
  Object.defineProperty(globalThis, "performance", {
    configurable: true,
    value: { now: () => (clock += 25) }
  })
  const state = {
    tuned: {
      shortSplit: null,
      f16Matmul: null,
      f16Pack: null,
      flashDecode: null
    },
    tuneReport: {},
    quadMatmuls: false,
    buffers: {},
    layers: [{ type: "full_attention", attention: {}, gateUp: {}, down: {} }],
    mrt: { chunks: true },
    contextLength: 4096,
    gpu: {
      finish: async () => {
        assert.equal(active, 0, "fence raced with pending attention")
      },
      delete: () => undefined
    },
    matmul: async () => undefined,
    layerMats: () => [],
    kvTexture: () => ({}),
    reset: () => undefined,
    f16Trim: () => undefined,
    attention: async () => {
      active++
      calls++
      maxActive = Math.max(maxActive, active)
      await new Promise((r) => setTimeout(r, 1))
      active--
    }
  }
  try {
    await Engine.prototype.tune.call(state as unknown as Engine)
    assert.ok(calls > 1)
    assert.equal(maxActive, 1)
  } finally {
    Object.defineProperty(globalThis, "performance", descriptor)
  }
})
