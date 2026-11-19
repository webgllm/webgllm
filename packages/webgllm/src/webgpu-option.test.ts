import assert from "node:assert/strict"
import test from "node:test"
import { predictSpeed } from "./predict-speed.js"
import { WebGLLM } from "./webgllm.js"

test('webgpu takes only "auto" or false', async () => {
  assert.doesNotThrow(() => new WebGLLM({ webgpu: "auto" }))
  assert.doesNotThrow(() => new WebGLLM({ webgpu: false }))
  for (const value of [true, "false", "on", 0, null])
    assert.throws(() => new WebGLLM({ webgpu: value as never }), /webgpu must be "auto" or false/)
  await assert.rejects(predictSpeed("LiquidAI/LFM3-1.7B-GGUF", { webgpu: true as never }), /webgpu must be "auto" or false/)
})
