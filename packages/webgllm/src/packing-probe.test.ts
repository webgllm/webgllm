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
    contextLength: 8192,
    embedding: {},
    finalNorm: {},
    choosePromptPacking: Reflect.get(Engine.prototype, "choosePromptPacking"),
    async chooseMatvec() { return undefined },
    f16Scratch() { return undefined },
    config: { vocabSize: 128000, hiddenSize: 2048, numLayers: layers.length },
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
      compiled: { count: 0 },
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
      async finish() { return undefined },
      async runAsync() {
        return undefined
      },
      async read() {
        return new Float32Array(1)
      }
    },
    async forwardHidden(tokens: number[], options?: { fullRows?: boolean }) {
      assert.equal(this.layers.length, native ? 3 : 2)
      assert.equal(options?.fullRows, true)
      assert.ok(tokens.length <= this.contextLength)
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
      assert.ok(calls() >= 6)
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
  state.forwardHidden = async (...args) => {
    await forward(...args)
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
  state.forwardHidden = async (...args) => {
    await forward(...args)
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
  state.forwardHidden = async (...args) => {
    await forward(...args)
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
  state.forwardHidden = async (...args) => { await forward(...args); clock += 5 }
  await Reflect.get(Engine.prototype, "choosePromptPacking").call(state)
  assert.equal(state.tuned.submissionWindow, 24)
})


test("a large packing win requires agreement from the reversed short pair", async (t) => {
  const { state, calls } = fixture(false)
  let clock=0
  t.mock.method(performance,"now",()=>clock)
  const forward=state.forwardHidden.bind(state)
  state.forwardHidden=async (...args)=>{await forward(...args);clock+=args[0].length*(state.tuned.f16Pack===3?1:0.6)}
  await Reflect.get(Engine.prototype,"choosePromptPacking").call(state)
  assert.equal(state.tuned.f16Pack,0)
  assert.equal(calls(),6)
})

test("a short-size disagreement rejects a misleading first full-batch win", async (t) => {
  const { state } = fixture(false)
  let clock=0,full=0
  t.mock.method(performance,"now",()=>clock)
  const forward=state.forwardHidden.bind(state)
  state.forwardHidden=async (...args)=>{await forward(...args);if(state.tuned.f16Pack===3)clock+=10;else if(args[0].length===512)clock+=++full===1?6:12;else clock+=12}
  await Reflect.get(Engine.prototype,"choosePromptPacking").call(state)
  assert.equal(state.tuned.f16Pack,3)
})

test("new full-batch compilation retains complete warm-up and full reversed pairs", async (t) => {
  const { state, calls }=fixture(false)
  let clock=0,compiled=false
  t.mock.method(performance,"now",()=>clock)
  const forward=state.forwardHidden.bind(state)
  state.forwardHidden=async (...args)=>{await forward(...args);clock+=10;if(args[0].length===512&&!compiled){state.gpu.compiled.count++;compiled=true}}
  await Reflect.get(Engine.prototype,"choosePromptPacking").call(state)
  assert.equal(calls(),10)
  assert.equal(state.tuned.f16Pack,3)
})

for (const contextLength of [256, 32])
  test(`cold weight load calibrates within a ${contextLength}-token context`, async () => {
    const { state, calls } = fixture()
    state.contextLength = contextLength
    const loadedWeights = {
      [Symbol.asyncIterator]: () => ({ next: async () => ({ done: true as const, value: undefined }) })
    }
    await Engine.prototype.load.call(state as unknown as Engine, loadedWeights)
    if (contextLength === 256) {
      assert.ok(calls() > 0)
      assert.ok(state.packingReport.every(sample => Reflect.get(sample, "rows") === 256))
    } else {
      assert.equal(calls(), 0)
    }
  })

test("a superseded packing probe leaves a concurrent load and global tuning alone", async () => {
  const { state: first, calls } = fixture()
  const { state: second } = fixture()
  const abort = new AbortController()
  const before = TUNING.lastRowOnly
  const forward = first.forwardHidden.bind(first)
  first.forwardHidden = async (...args) => {
    await forward(...args)
    abort.abort()
  }
  const old = first.choosePromptPacking.call(first, abort.signal)
  const replacement = second.choosePromptPacking.call(second)
  assert.equal(TUNING.lastRowOnly, before)
  await assert.rejects(old, { name: "AbortError" })
  await replacement
  assert.equal(calls(), 1)
  assert.equal(first.packingReport.length, 0)
  assert.ok(second.packingReport.length > 0)
  assert.equal(TUNING.lastRowOnly, before)
})
