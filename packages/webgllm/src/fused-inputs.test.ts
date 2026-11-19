import assert from "node:assert/strict"
import test from "node:test"
import { Engine } from "./engine.js"

test("fused float inputs require every consumer to accept the selected FP16 layout", () => {
  const texture = {},
    quadOut = { id: 1 },
    plainOut = { id: 2 },
    state = {
      quadMatmuls: true,
      // (Outputs with a partner texture take the 4-token f16 matmul.)
      quads: new Map([[quadOut.id, {}]]),
      f16Pack: () => 0,
      takesF16: (m: { bits: number }) => m.bits === 4,
      f16Scratch: () => ({ a: [texture], next: 0 })
    }
  const get = Reflect.get(Engine.prototype, "floatInput")
  const q4 = { bits: 4 }
  assert.equal(get.call(state, 512, [[q4, quadOut]]), texture)
  assert.equal(get.call(state, 8, [[q4, quadOut]]), null)
  assert.equal(get.call(state, 512, [[q4, quadOut], [{ bits: 8 }, quadOut]]), null)
  assert.equal(get.call(state, 512, [[q4, quadOut]], false), null)
  // A 4-bit matmul into an output without a partner (Gated DeltaNet's
  // `ba`) reads h2: the input stays h2 for every consumer.
  assert.equal(get.call(state, 512, [[q4, quadOut], [q4, plainOut]]), null)
  state.f16Pack = () => 3
  assert.equal(get.call(state, 512, [[q4, quadOut]]), null)
})
