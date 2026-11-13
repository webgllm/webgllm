import test from "node:test"
import assert from "node:assert/strict"
import { Engine, retimedTuning, timesAgain, TUNING, tuningToRetime, type Tuned } from "./engine.js"

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
  // (And two more pairs: at 10 ms a timing they fit the probe's quarter second.)
  assert.equal(calls(),14)
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

test("the pacing stored from the probe's timing: whole within two frames, graded past them, and marked for one more timing there", async (t) => {
  const stored = async (ms: number) => {
    const { state } = fixture()
    let clock = 0
    const now = t.mock.method(performance, "now", () => clock)
    const forward = state.forwardHidden.bind(state)
    // Both packings take the same time: the first (the default) is kept.
    state.forwardHidden = async (...args) => {
      await forward(...args)
      if (args[0].length === 512) clock += ms
    }
    await Reflect.get(Engine.prototype, "choosePromptPacking").call(state)
    now.mock.restore()
    return state.tuned as Tuned
  }
  // 16.7 ms a frame, a window of up to 24 frames, 512,000 of work a full batch.
  const quiet = await stored(26)
  assert.deepEqual(
    [quiet.submissionWindow, quiet.submissionMs, quiet.submissionRetime, quiet.f16Pack],
    [24, 26, 0, 3]
  )
  // Beside a game the same GPU timed 35 ms: nearly the same pacing (it was a twelfth of the quota and a window of 4), and one more timing.
  const busy = await stored(35)
  assert.deepEqual([busy.submissionMs, busy.submissionRetime], [35, 1])
  assert.ok(busy.submissionWindow! >= 20, `window ${busy.submissionWindow}`)
  // Four frames or more: the starting window, as before, and one more timing too.
  const slow = await stored(70)
  assert.deepEqual([slow.submissionWindow, slow.submissionMs, slow.submissionRetime], [TUNING.presentationFrames, 70, 1])
})

test("a second timing keeps the faster one's pacing where both are of the same kernels, and isn't taken a third time", () => {
  const first: Tuned = { shortSplit: 2, f16Pack: 0, f16Matmul: null, flashDecode: 1, rmsParts: 64, submissionReads: 3e8, submissionOverlap: 2 ** 29, submissionWindow: 4, submissionMs: 70, submissionRetime: 1 }
  // What the load that times again starts from: the kernels and the mark, without the timing's own values, so the probe runs on those kernels.
  assert.deepEqual(tuningToRetime(first), { shortSplit: 2, f16Pack: 0, f16Matmul: null, flashDecode: 1, rmsParts: 64, submissionRetime: 1 })
  // The second timing was quiet: its pacing.
  const second: Tuned = { shortSplit: 2, f16Pack: 0, f16Matmul: null, flashDecode: 1, rmsParts: 64, submissionReads: 1e10, submissionOverlap: 2 ** 29, submissionWindow: 24, submissionMs: 26, submissionRetime: 0 }
  assert.deepEqual(retimedTuning(first, second), second)
  // The second was the held-up one: the first's pacing stays, and it's settled.
  const worse: Tuned = { ...second, submissionReads: 2.7e8, submissionWindow: 4, submissionMs: 90, submissionRetime: 1 }
  assert.deepEqual(retimedTuning(first, worse), { ...first, submissionRetime: 0 })
  // A GPU that is honestly past two frames times about the same again: settled either way.
  assert.equal(retimedTuning(first, { ...second, submissionMs: 71, submissionWindow: 4, submissionReads: 3e8, submissionRetime: 1 }).submissionRetime, 0)
  // The second timing found larger submissions miss frames: that stands, whichever was faster.
  const missed: Tuned = { ...second, submissionOverlap: 0, submissionWindow: 4, submissionMs: 90 }
  assert.deepEqual(retimedTuning(first, missed), { ...missed, submissionRetime: 0 })
  // A load that timed nothing (no probe for this model): the stored pacing stays.
  assert.deepEqual(retimedTuning(first, tuningToRetime(first)), { ...first, submissionRetime: 0 })
  // A timing of other kernels is never paired with the stored ones: kernels and pacing go together, faster or slower.
  for (const other of [{ ...second, f16Pack: 3 }, { ...worse, f16Pack: 3 }, { ...second, f16Matmul: 1 }])
    assert.deepEqual(retimedTuning(first, other), { ...other, submissionRetime: 0 })
  // Which stored tunings are timed again: one marked for it, and one from before timings were kept.
  const before: Tuned = { shortSplit: 2, f16Pack: 0, f16Matmul: null, flashDecode: 1, rmsParts: 64, submissionReads: 2 ** 28, submissionOverlap: 2 ** 29, submissionWindow: 4 }
  assert.equal(timesAgain(first), true)
  assert.equal(timesAgain({ ...first, submissionRetime: 0 }), false)
  assert.equal(timesAgain(before), true)
  // (One with no pacing at all was never timed for it: nothing to take again.)
  assert.equal(timesAgain({ shortSplit: null }), false)
  // An older tuning has no timing to compare and its packing was two pairs' pick: the packing is chosen again, and the new timing's pacing goes with it.
  assert.deepEqual(tuningToRetime(before), { shortSplit: 2, f16Pack: null, f16Matmul: null, flashDecode: 1, rmsParts: 64 })
  const chosen: Tuned = { ...second, f16Pack: 3 }
  assert.deepEqual(retimedTuning(before, chosen), chosen)
  assert.deepEqual(retimedTuning(before, second), second)
  assert.deepEqual(retimedTuning(before, worse), { ...worse, submissionRetime: 0 })
  assert.equal(timesAgain(retimedTuning(before, worse)), false)
})

test("a fast pipeline is timed in more pairs, all of which the other packing must win; a slow one in two, as before", async (t) => {
  // Each way's full-batch timings (ms) in the order the probe takes them.
  const pick = async (timings: { 3: Array<number>; 0: Array<number> }) => {
    const { state } = fixture()
    let clock = 0
    const now = t.mock.method(performance, "now", () => clock)
    const taken: Record<number, number> = { 3: 0, 0: 0 }
    const order: Array<number> = []
    const forward = state.forwardHidden.bind(state)
    state.forwardHidden = async (...args) => {
      await forward(...args)
      const pack = state.tuned.f16Pack as 3 | 0
      if (args[0].length !== 512) return
      order.push(pack)
      const ms = timings[pack][taken[pack]++]
      assert.ok(ms !== undefined, `a ${taken[pack]}th timing of packing ${pack}`)
      clock += ms
    }
    await Reflect.get(Engine.prototype, "choosePromptPacking").call(state)
    now.mock.restore()
    return { pack: state.tuned.f16Pack, order, report: state.packingReport }
  }
  // Measured on an M3 Ultra beside a game: packing 0 won the first two pairs by 2% (38.2 and 42.7 against 37.1 and
  // 40.1 ms), which used to settle it. A third pair fits the quarter second, in the first's order, and it loses it.
  const busy = await pick({ 3: [38.2, 42.7, 26.1], 0: [37.1, 40.1, 28.9] })
  assert.equal(busy.pack, 3)
  assert.deepEqual(busy.order, [3, 0, 0, 3, 3, 0], JSON.stringify(busy.order))
  // On a quiet GPU (26 ms a timing) four pairs fit, each in the reverse order of the last.
  const quiet = await pick({ 3: [26.0, 27.4, 25.4, 26.1], 0: [25.4, 26.1, 25.9, 25.7] })
  assert.equal(quiet.pack, 3)
  assert.deepEqual(quiet.order, [3, 0, 0, 3, 3, 0, 0, 3], JSON.stringify(quiet.order))
  // Winning every pair still takes it: all four, or all three where a fourth doesn't fit.
  assert.equal((await pick({ 3: [30, 31, 30, 32], 0: [26, 27, 26, 27] })).pack, 0)
  const three = await pick({ 3: [36, 36, 36], 0: [34, 34, 34] })
  assert.equal(three.pack, 0)
  assert.deepEqual(three.order, [3, 0, 0, 3, 3, 0], JSON.stringify(three.order))
  // Measured on a Pixel 10 Pro: half a second a timing. Two pairs, as before, and packing 0's lead is taken.
  for (const timings of [{ 3: [569.8, 570.6], 0: [474.7, 497.5] }, { 3: [517.9, 572.6], 0: [498.9, 481.5] }]) {
    const slow = await pick(timings)
    assert.equal(slow.pack, 0)
    assert.deepEqual(slow.order, [3, 0, 0, 3])
    assert.deepEqual(slow.report.map((r) => (r as { rounds?: Array<number> }).rounds), [timings[3], timings[0]])
  }
})

test("a second timing times the stored kernels only: the pacing kept is of the pipeline prompts run", async (t) => {
  // A stored tuning marked for one more timing, and each kernel's full-batch time at this load.
  const again = async (native: boolean, stored: Tuned, ms: (tuned: Tuned) => number) => {
    const { state } = fixture(native)
    let clock = 0
    const now = t.mock.method(performance, "now", () => clock)
    const timed: Array<string> = []
    const forward = state.forwardHidden.bind(state)
    state.forwardHidden = async (...args) => {
      await forward(...args)
      if (args[0].length !== 512) return
      timed.push(`${state.tuned.f16Pack}/${state.tuned.f16Matmul}`)
      clock += ms(state.tuned as Tuned)
    }
    Object.assign(state, { tuned: tuningToRetime(stored) })
    await Reflect.get(Engine.prototype, "choosePromptPacking").call(state)
    now.mock.restore()
    return { tuned: retimedTuning(stored, state.tuned as Tuned), timed: [...new Set(timed)] }
  }
  const first = { shortSplit: null, rmsParts: 64, submissionReads: 2.7e8, submissionOverlap: 2 ** 29, submissionWindow: 4, submissionMs: 90, submissionRetime: 1 }
  // RGBA4 weights whose stored matmul is 2: the other one would time 25 ms now, the stored one 80 ms.
  const matmul = await again(true, { ...first, f16Pack: 3, f16Matmul: 2 }, (tuned) => (tuned.f16Matmul === 1 ? 25 : 80))
  assert.deepEqual(matmul.timed, ["3/2"])
  assert.deepEqual(
    [matmul.tuned.f16Pack, matmul.tuned.f16Matmul, matmul.tuned.submissionMs, matmul.tuned.submissionWindow, matmul.tuned.submissionRetime],
    [3, 2, 80, TUNING.presentationFrames, 0]
  )
  // The same for a stored packing of 0 where packing 3 would time 26 ms and it 60 ms.
  const pack = await again(false, { ...first, f16Pack: 0, f16Matmul: null }, (tuned) => (tuned.f16Pack === 3 ? 26 : 60))
  assert.deepEqual(pack.timed, ["0/null"])
  assert.deepEqual([pack.tuned.f16Pack, pack.tuned.submissionMs, pack.tuned.submissionRetime], [0, 60, 0])
  assert.ok(pack.tuned.submissionWindow! < 12, `window ${pack.tuned.submissionWindow}`)
  // A quiet second timing of the stored kernel: its pacing, in full.
  const quiet = await again(false, { ...first, f16Pack: 0, f16Matmul: null }, (tuned) => (tuned.f16Pack === 3 ? 26 : 27))
  assert.deepEqual([quiet.tuned.f16Pack, quiet.tuned.submissionMs, quiet.tuned.submissionWindow], [0, 27, 24])
  // A held-up one (slower than the stored timing): the stored pacing stays.
  const held = await again(false, { ...first, f16Pack: 0, f16Matmul: null }, () => 120)
  assert.deepEqual([held.tuned.f16Pack, held.tuned.submissionMs, held.tuned.submissionReads, held.tuned.submissionWindow], [0, 90, 2.7e8, 4])
})
