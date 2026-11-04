import assert from "node:assert/strict"
import test from "node:test"
import { Engine } from "./engine.js"
import { attentionError } from "./attention-check.js"

function fake(linear = false) {
  return {
    config: {
      vocabSize: 128,
      headDim: 1,
      numHeads: 1,
      numKvHeads: 1,
      ...(linear && { linear: {} })
    },
    layers: [{ type: "full_attention" }, { type: "full_attention" }],
    work: null,
    position: 0,
    forwards: 0,
    chunkErrors: null as number[] | null,
    debug: null,
    kvDebug: null,
    reset() {
      this.position = 0
    },
    async forward(tokens: number[]) {
      this.forwards++
      this.position += tokens.length
      this.chunkErrors?.push(0.1)
      return new Float32Array(1)
    },
    async recurrenceError() {
      return 0.7
    },
    async panelError(_tokens: number[], reuse: boolean) {
      if (!linear) {
        assert.equal(reuse, true)
        assert.equal(this.position, 40)
      }
      this.position = 0
      return 0.2
    },
    gpu: {
      delete() {
        return undefined
      }
    }
  }
}
test("startup checks reuse real reference inputs and leave no model cache", async () => {
  const state = fake()
  const error = await Engine.prototype.selfCheck.call(
    state as unknown as Engine
  )
  assert.equal(error, 0.2)
  assert.equal(state.forwards, 1)
  assert.equal(state.position, 0)
  assert.equal(state.chunkErrors, null)
})
test("recurrent models retain their existing numerical rejection result", async () => {
  const state = fake(true)
  assert.equal(
    await Engine.prototype.selfCheck.call(state as unknown as Engine),
    0.7
  )
})
test("a failed reference forward always clears its cache even when reuse was requested", async () => {
  const state = fake()
  state.forward = async () => {
    state.position = 40
    throw new Error("GPU read failed")
  }
  await assert.rejects(
    attentionError(state as unknown as Engine, [1, 2, 3], true),
    /GPU read failed/
  )
  assert.equal(state.position, 0)
})

test("the fused startup check conservatively converts GPU-relative error to a CPU-relative bound", async () => {
  const state = fake()
  state.panelError = async () => 0
  const error = await Engine.prototype.selfCheck.call(
    state as unknown as Engine
  )
  assert.ok(Math.abs(error - 0.1 / 0.9) < 1e-12)
  assert.equal(state.position, 0)
})

test("dense startup checks retain every attention layout and restore layer indexing on failure", async () => {
  const state = fake()
  const layers = [
    { type: "full_attention" },
    { type: "full_attention" },
    { type: "sliding_attention" },
    { type: "sliding_attention", attention: { ring: true } },
    { type: "full_attention" }
  ]
  Object.assign(state, { layers })
  const config = state.config
  state.forward = async () => {
    assert.equal(state.layers.length, 4)
    assert.deepEqual(Reflect.get(state.config, "layerTypes"), [
      "full_attention",
      "sliding_attention",
      "sliding_attention",
      "full_attention"
    ])
    throw new Error("reference read failed")
  }
  await assert.rejects(
    Engine.prototype.selfCheck.call(state as unknown as Engine),
    /reference read failed/
  )
  assert.equal(state.layers, layers)
  assert.equal(state.config, config)
  assert.equal(state.position, 0)
})
