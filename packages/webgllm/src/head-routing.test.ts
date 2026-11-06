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


test("head work stays constant when the automatic submission quota grows", async () => {
  const work: number[] = []
  const state = {
    logitRows: 4,
    gpu: {
      readsPerSlice: 2 ** 28,
      async runAsync(pass: { work: number }) { work.push(pass.work) }
    }
  }
  const matrix = { groups: 4096, rowsPerColumn: 4096,
    quantPanels: [{ height: 2048 }, { height: 2048 }] }
  const pass = { inputs: {}, uniforms: {} }
  const head = Reflect.get(Engine.prototype, "weightHead")
  await head.call(state, matrix, pass, 1, 2)
  const initial = work.slice()
  work.length = 0
  state.gpu.readsPerSlice = 2 ** 34
  await head.call(state, matrix, pass, 1, 2)
  assert.deepEqual(work, initial)
  assert.ok(work.every(n => n === TUNING.submissionReads / (TUNING.headPaceRows * 1024)))
})
