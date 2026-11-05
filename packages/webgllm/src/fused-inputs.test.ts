import assert from "node:assert/strict"
import test from "node:test"
import { Engine } from "./engine.js"

test("fused float inputs require every consumer to accept the selected FP16 layout", () => {
  const texture = {},
    state = {
      quadMatmuls: true,
      f16Pack: () => 0,
      takesF16: (m: { bits: number }) => m.bits === 4,
      f16Scratch: () => ({ a: [texture], next: 0 })
    }
  const get = Reflect.get(Engine.prototype, "floatInput")
  assert.equal(get.call(state, 512, [{ bits: 4 }]), texture)
  assert.equal(get.call(state, 8, [{ bits: 4 }]), null)
  assert.equal(get.call(state, 512, [{ bits: 4 }, { bits: 8 }]), null)
  assert.equal(get.call(state, 512, [{ bits: 4 }], false), null)
  state.f16Pack = () => 3
  assert.equal(get.call(state, 512, [{ bits: 4 }]), null)
})
