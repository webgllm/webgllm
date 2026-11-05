import assert from "node:assert/strict"
import test from "node:test"
import { Engine } from "./engine.js"

for (const fail of [false, true]) {
  test(`warm-up drops cached full-layer costs before selecting a subset${fail ? " and restores layers on failure" : ""}`, async () => {
    const layers = [
      { type: "conv", conv: {} },
      { type: "conv", conv: {} },
      { type: "full_attention", attention: {} },
      { type: "conv", conv: {} }
    ]
    const state = {
      layers,
      work: [1, 2, 3, 4] as number[] | null,
      quadMatmuls: false,
      f16Trim() {
        return undefined
      },
      reset() {
        return undefined
      },
      async warmUpLayers() {
        assert.equal(this.work, null)
        assert.equal(this.layers.length, 3)
        if (fail) throw new Error("cancelled warm-up")
      }
    }
    const run = Engine.prototype.warmUp.call(state as unknown as Engine)
    if (fail) await assert.rejects(run, /cancelled warm-up/)
    else await run
    assert.equal(state.layers, layers)
    assert.equal(state.work, null)
  })
}
