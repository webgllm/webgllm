import assert from "node:assert/strict"
import test from "node:test"
import { Engine, TUNING } from "./engine.js"

test("an unsplit expanded Q6 head uses its format-aware kernel rather than the old byte shader", async () => {
  const previous = TUNING.q6HeadSplit
  TUNING.q6HeadSplit = 1
  let specialized = 0
  const state = {
    lastRowOnly: false,
    headScaled: false,
    buffers: { residual: [{}], lastNormed: {} },
    finalNorm: {},
    embedding: { bits: 6, bytes: true, q6Expanded: true },
    normToH2() {
      return undefined
    },
    async headQ6K() {
      specialized++
      return {}
    },
    gpu: {
      run() {
        throw new Error("used the native byte shader")
      }
    }
  }
  try {
    const head = Reflect.get(Engine.prototype, "head") as (
      row: number
    ) => Promise<object>
    await head.call(state, 0)
    assert.equal(specialized, 1)
  } finally {
    TUNING.q6HeadSplit = previous
  }
})
