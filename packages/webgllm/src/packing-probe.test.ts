import test from "node:test"
import assert from "node:assert/strict"
import { Engine, TUNING } from "./engine.js"

function fixture(native = false, fail = false) {
  const layers = [
    { type: "conv", conv: {} },
    { type: "full_attention" },
    { type: "sliding_attention" }
  ]
  let calls = 0
  const state = {
    quadMatmuls: true,
    tuned: {} as {
      f16Pack?: number
      f16Matmul?: number
      submissionReads?: number
      submissionOverlap?: number
      submissionWindow?: number
    },
    batchSize: 512,
    config: { vocabSize: 128000, hiddenSize: 2048 },
    layers,
    work: [1, 2, 3] as number[] | null,
    buffers: { residual: [{}], rms: {} },
    packingReport: [] as { pack: number; matmul?: number; ms: number }[],
    layerMats: () => [{ rgba4: native }],
    legacyF16Layout: () => ({ half: false }),
    usesF16: () => !native,
    activationCopyFits: () => true,
    f16Pack: () => 3,
    reset() {
      return undefined
    },
    f16Trim() {
      return undefined
    },
    gpu: {
      activePhase: "load",
      submittedWork: 0,
      readsPerSlice: 2 ** 30,
      presentationAdvice: { relaxation: true, cap: 24 },
      primePresentation(_window: number, reads: number, allow = true) {
        this.readsPerSlice = reads
        this.presentationAdvice.relaxation = allow
      },
      expectedFrameMs: 16.7,
      resetPacing() {
        return undefined
      },
      configurePacing() {
        return undefined
      },
      setPhase(phase: string) {
        this.activePhase = phase
      },
      program() {
        return {}
      },
      async runAsync() {
        return undefined
      },
      async read() {
        return new Float32Array(1)
      }
    },
    async forwardHidden(tokens: number[]) {
      assert.equal(this.layers.length, native ? 3 : 2)
      assert.equal(TUNING.lastRowOnly, 0)
      assert.ok(this.work === null || this.work.length === 1)
      this.work = [1]
      if (fail && ++calls === 2) throw Error("probe failed")
      if (!fail) calls++
      this.gpu.submittedWork += tokens.length * 1000
    }
  }
  return { state, layers, calls: () => calls }
}
for (const fail of [false, true])
  test(`packing calibration restores all operator kinds and pacing${fail ? " after failure" : ""}`, async () => {
    const { state, layers, calls } = fixture(false, fail),
      beforeLast = TUNING.lastRowOnly
    const result = Reflect.get(Engine.prototype, "choosePromptPacking").call(
      state
    )
    if (fail) await assert.rejects(result, /probe failed/)
    else {
      await result
      assert.equal(calls(), 8)
    }
    assert.equal(state.layers, layers)
    assert.equal(state.work, null)
    assert.equal(TUNING.lastRowOnly, beforeLast)
    assert.equal(state.gpu.activePhase, "load")
  })

test("native layout comparisons preserve the real model operator mixture", async (t) => {
  const { state, layers } = fixture(true)
  let clock = 0
  t.mock.method(performance, "now", () => clock)
  const forward = state.forwardHidden.bind(state)
  state.forwardHidden = async (tokens) => {
    await forward(tokens)
    clock +=
      state.tuned.f16Matmul === 1 ? 10 : state.tuned.f16Pack === 3 ? 7 : 8
  }
  await Reflect.get(Engine.prototype, "choosePromptPacking").call(state)
  assert.deepEqual(
    state.packingReport.map(({ pack, matmul }) => ({ pack, matmul })),
    [
      { pack: 3, matmul: 1 },
      { pack: 3, matmul: 2 }
    ]
  )
  assert.equal(state.tuned.f16Matmul, 2)
  assert.equal(state.tuned.f16Pack, 3)
  assert.equal(Reflect.get(state, "promptMatmulMeasured"), true)
  assert.equal(state.layers, layers)
})

test("a layout winning only one reversed timing pair does not become the production choice", async (t) => {
  const { state } = fixture(true)
  let clock = 0
  const seen = new Map<string, number>()
  t.mock.method(performance, "now", () => clock)
  const forward = state.forwardHidden.bind(state)
  state.forwardHidden = async (tokens) => {
    await forward(tokens)
    const key = `${state.tuned.f16Matmul}/${state.tuned.f16Pack}`
    const n = (seen.get(key) ?? 0) + 1
    seen.set(key, n)
    clock +=
      state.tuned.f16Matmul === 1
        ? 10
        : state.tuned.f16Pack === 3
          ? n === 3
            ? 6
            : 12
          : 11
  }
  await Reflect.get(Engine.prototype, "choosePromptPacking").call(state)
  assert.equal(state.tuned.f16Matmul, 1)
})

test("a failed presentation trial is not repeated by later packing candidates", async () => {
  const { state } = fixture(false)
  const forward = state.forwardHidden.bind(state)
  let failed = false
  state.forwardHidden = async (tokens) => {
    await forward(tokens)
    if (!failed && state.gpu.presentationAdvice.relaxation) {
      state.gpu.readsPerSlice = 2 ** 28
      state.gpu.presentationAdvice.relaxation = false
      failed = true
    } else if (failed) {
      assert.equal(state.gpu.readsPerSlice, 2 ** 28)
      assert.equal(state.gpu.presentationAdvice.relaxation, false)
    }
  }
  await Reflect.get(Engine.prototype, "choosePromptPacking").call(state)
  assert.equal(state.tuned.submissionOverlap, 0)
  assert.equal(state.tuned.submissionWindow, TUNING.presentationFrames)
})


test("a fast successful complete pipeline retains its tested presentation window", async (t) => {
  const { state } = fixture(false)
  let clock = 0
  t.mock.method(performance, "now", () => clock)
  const forward = state.forwardHidden.bind(state)
  state.forwardHidden = async tokens => { await forward(tokens); clock += 5 }
  await Reflect.get(Engine.prototype, "choosePromptPacking").call(state)
  assert.equal(state.tuned.submissionWindow, 24)
})
