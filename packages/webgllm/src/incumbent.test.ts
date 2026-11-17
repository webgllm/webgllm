import assert from "node:assert/strict"
import test from "node:test"
import { KEEP_MARGIN, keepMargin, replaces } from "./engine.js"
import { incumbentGpuTuning, type GpuNote } from "./storage.js"

test("a way replaces the standing choice only when it's faster in every round by the margin", () => {
  assert.equal(replaces([80, 85], [100, 100], KEEP_MARGIN), true)
  // One round within 5%: the standing choice stays.
  assert.equal(replaces([80, 96], [100, 100], KEEP_MARGIN), false)
  // 10% faster in every round (Firefox's f16Pack 0 on a Pixel): replaced.
  assert.equal(replaces([466, 466], [518, 518], KEEP_MARGIN), true)
  // Faster on average but not in every round.
  assert.equal(replaces([50, 105], [100, 100], KEEP_MARGIN), false)
  assert.equal(replaces([], [], KEEP_MARGIN), false)
  assert.equal(replaces([80], [100, 100], KEEP_MARGIN), false)
})

test("the incumbent is the newest checked note for the GPU, model and passes, from any browser version or settings", () => {
  const tuned = (f16Pack: number) => ({ shortSplit: null, f16Matmul: null, f16Pack, flashDecode: null })
  const note = (f16Pack: number, at: number | undefined, more: Partial<GpuNote> = {}): GpuNote =>
    ({ layout: "bytes", calibrated: true, selfCheck: 1e-4, tuned: tuned(f16Pack), ...(at !== undefined && { at }), ...more })
  const notes: Record<string, GpuNote> = {
    "Apple M3|Chrome/141|LiquidAI/LFM3-1.7B-GGUF|mrt|abc.111": note(3, 1),
    "Apple M3|Chrome/142|LiquidAI/LFM3-1.7B-GGUF|mrt|def.222": note(0, 5),
    // Not checked, another model, another GPU, other passes: never.
    "Apple M3|Chrome/143|LiquidAI/LFM3-1.7B-GGUF|mrt|ghi.333": note(2, 9, { selfCheck: 0.5 }),
    "Apple M3|Chrome/143|LiquidAI/LFM2.5-350M-GGUF|mrt|ghi.333": note(1, 9),
    "Apple M4|Chrome/143|LiquidAI/LFM3-1.7B-GGUF|mrt|ghi.333": note(1, 9),
    "Apple M3|Chrome/143|LiquidAI/LFM3-1.7B-GGUF|plain|ghi.333": note(1, 9)
  }
  assert.equal(incumbentGpuTuning(notes, "Apple M3", "LiquidAI/LFM3-1.7B-GGUF", "mrt", 1e-2)?.tuned?.f16Pack, 0)
  assert.equal(incumbentGpuTuning(notes, "Apple M2", "LiquidAI/LFM3-1.7B-GGUF", "mrt", 1e-2), undefined)
  // Notes from before notes had times: any of them.
  const old = { "Apple M3|Chrome/141|LiquidAI/LFM3-1.7B-GGUF|mrt|abc.111": note(3, undefined) }
  assert.equal(incumbentGpuTuning(old, "Apple M3", "LiquidAI/LFM3-1.7B-GGUF", "mrt", 1e-2)?.tuned?.f16Pack, 3)
})

test("a noisy timer raises the margin: a way must win by more than the timings' own spread", () => {
  assert.equal(keepMargin([[100, 102], [90, 91]]), KEEP_MARGIN)
  // One way's rounds 30% apart: the other must be over 30% faster.
  assert.ok(Math.abs(keepMargin([[100, 130], [70, 71]]) - 1 / 1.3) < 1e-9)
  // About 20% faster against 30% noise: the standing choice stays.
  assert.equal(replaces([80, 81], [100, 130], keepMargin([[100, 130], [80, 81]])), false)
  // Over 40% faster: it doesn't.
  assert.equal(replaces([55, 56], [100, 130], keepMargin([[100, 130], [55, 56]])), true)
})
