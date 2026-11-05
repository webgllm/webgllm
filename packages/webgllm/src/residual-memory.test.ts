import assert from "node:assert/strict"
import test from "node:test"
import { Engine } from "./engine.js"
import { GL } from "./gpu/gl.js"

test("decode-only MRT storage is bounded and its complete replacement cost is accounted", () => {
  const state = {
    capacity: 576,
    baseBatch: 512,
    contextLength: 8192,
    config: {
      hiddenSize: 2048,
      intermediateSize: 7680,
      numHeads: 16,
      headDim: 128,
      vocabSize: 128000,
      layerTypes: ["full_attention"]
    },
    gpu: {
      texture(internal: number, width: number, height: number) {
        return { internal, width, height }
      },
      canRender() {
        return false
      }
    },
    qkvgDims: () => 2560,
    linDims: () => null,
    convRows: () => 8,
    headGroup: () => 1,
    scoreRows: () => 128,
    h2: () => ({}),
    pair: () => ({}),
    f32: () => ({})
  }
  const allocate = Reflect.get(Engine.prototype, "allocate")
  const b = allocate.call(state)
  assert.equal(b.residual, b.residualLarge)
  assert.equal(b.residualLarge[0].height, 576)
  assert.equal(b.residualSmall[0].height, 8)
  assert.equal(b.scaled[0].height, 8)
  assert.equal(b.scaled[0].width, b.residualSmall[0].width)
  const savings = Reflect.get(Engine.prototype, "residualMemorySavings").call(
    state
  )
  const width = b.scaled[0].width
  assert.equal(savings, 2 * width * 576 * 16 - 4 * width * 8 * 16)
  assert.ok(savings > 0)
  assert.equal(b.scaled[0].internal, GL.RGBA32UI)
  state.capacity = 9
  const tiny = allocate.call(state)
  assert.equal(tiny.residualSmall, tiny.residualLarge)
  assert.equal(tiny.scaled[0].height, 9)
  assert.equal(
    Reflect.get(Engine.prototype, "residualMemorySavings").call(state),
    0
  )
})

test("attention slicing funds scratch only with bytes actually removed from score buffers", () => {
  const rows = Reflect.get(Engine.prototype, "scoreRows")
  const state = {
    baseBatch: 512,
    capacity: 576,
    contextLength: 8192,
    config: { numHeads: 16 },
    scoreRows(expanded: boolean) {
      return rows.call(this, expanded)
    }
  }
  assert.equal(state.scoreRows(false), 128)
  assert.equal(state.scoreRows(true), 64)
  const saved = Reflect.get(Engine.prototype, "scoreMemorySavings").call(
    state,
    true
  )
  assert.equal(saved, 32 * 2 ** 20)
  assert.equal(
    Reflect.get(Engine.prototype, "scoreMemorySavings").call(state, false),
    0
  )
})
