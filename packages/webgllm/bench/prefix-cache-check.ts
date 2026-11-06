/** Model/device-independent numerical check; it does not load a model or run on import. */
import type { Engine } from "../src/engine.js"
import { naturalPromptPlan, readPrompt, sharedPrefix } from "../src/prompt.js"

const top = (values: Float32Array) => {
  let at = 0
  for (let i = 1; i < values.length; i++) if (values[i] > values[at]) at = i
  return at
}
const distribution = (values: Float32Array) => {
  let maximum = -Infinity
  for (const value of values) maximum = Math.max(maximum, value)
  const out = Float64Array.from(values, value => Math.exp(value - maximum))
  const sum = out.reduce((a, b) => a + b, 0)
  return out.map(value => value / sum)
}
function compare(cold: Float32Array, warm: Float32Array) {
  let maximumAbsoluteDifference = 0
  for (let i = 0; i < cold.length; i++) {
    if (cold[i] === warm[i]) continue
    maximumAbsoluteDifference = Math.max(maximumAbsoluteDifference, Math.abs(cold[i] - warm[i]))
  }
  const p = distribution(cold), q = distribution(warm)
  let klDivergence = 0
  for (let i = 0; i < p.length; i++) if (p[i] > 0) klDivergence += p[i] * Math.log(p[i] / Math.max(q[i], Number.MIN_VALUE))
  if (!Number.isFinite(maximumAbsoluteDifference) || !Number.isFinite(klDivergence)) throw new Error("Cache comparison produced non-finite numerical differences.")
  return { top1Equal: top(cold) === top(warm), maximumAbsoluteDifference, klDivergence }
}

/**
 * The ordinary cold/production-warm comparison may change batch arithmetic.
 * A separate matched-boundary control computes both prefixes with the same
 * segmentation, disturbs only the warm state, restores it, then feeds identical
 * second-request tails. This isolates snapshot copying from chunk differences.
 * It does not by itself prove an inside-batch recurrent extraction is exact.
 */
export async function checkPrefixCache(
  engine: Engine,
  first: Array<number>,
  second: Array<number>,
  boundaries: { firstPrefix: number; secondPrefix: number; firstAnchor?: number; secondAnchor?: number }
) {
  const common = sharedPrefix(first, second)
  if (boundaries.firstPrefix > common || boundaries.secondPrefix > common) throw new Error("Prefix boundary extends past matching tokens.")
  const opts = (prefix: number, anchor?: number) => ({ prefix, anchor })
  engine.reset()
  try {
    const coldStart = performance.now()
    const coldRow = await readPrompt(engine, second, 0, [], opts(boundaries.secondPrefix, boundaries.secondAnchor))
    const cold = await engine.logits(coldRow)
    const coldMs = performance.now() - coldStart
    engine.reset()
    const cached: Array<number> = []
    await readPrompt(engine, first, 0, cached, opts(boundaries.firstPrefix, boundaries.firstAnchor))
    const expectedRestore = engine.snapshotPosition("prefix") ?? 0
    const beforeRestoreBytes = engine.gpu.bytes
    const restored = engine.rewind(Math.min(common, second.length - 1))
    if (restored !== expectedRestore) throw new Error(`Expected selected checkpoint ${expectedRestore}, restored ${restored}.`)
    cached.length = restored
    const warmStart = performance.now()
    const warmRow = await readPrompt(engine, second, restored, cached, opts(boundaries.secondPrefix, boundaries.secondAnchor))
    const warm = await engine.logits(warmRow)
    const warmMs = performance.now() - warmStart
    const afterWarmBytes = engine.gpu.bytes
    const ordinaryComparison = compare(cold, warm)
    let matchedBoundaryControl: (ReturnType<typeof compare> & { boundaryTokens: number; productionFirstAnchorInsideBatch: boolean }) | undefined
    if (restored > 0) {
      // Reference: compute the common prefix and second tail without a rewind.
      engine.reset()
      const coldIds: Array<number> = []
      await readPrompt(engine, second.slice(0, restored), 0, coldIds, opts(boundaries.secondPrefix, restored))
      const referenceRow = await readPrompt(engine, second, restored, coldIds, opts(boundaries.secondPrefix, boundaries.secondAnchor))
      const reference = await engine.logits(referenceRow)
      // Same prefix segmentation, then different data to disturb all run state.
      engine.reset()
      const warmIds: Array<number> = []
      await readPrompt(engine, first.slice(0, restored), 0, warmIds, opts(boundaries.firstPrefix, restored))
      await readPrompt(engine, first, restored, warmIds, { checkpoints: false })
      const back = engine.rewind(restored)
      if (back !== restored) throw new Error(`Matched control lost checkpoint ${restored}.`)
      warmIds.length = back
      const checkedRow = await readPrompt(engine, second, back, warmIds, opts(boundaries.secondPrefix, boundaries.secondAnchor))
      const checked = await engine.logits(checkedRow)
      const plan = naturalPromptPlan(n => engine.batchSizes(n), first.length, 0, boundaries.firstAnchor, engine.snapshotLimit < 4, engine.canSaveInside, boundaries.firstPrefix)
      matchedBoundaryControl = { boundaryTokens: restored, productionFirstAnchorInsideBatch: plan.some(step => step.saveAt !== undefined), ...compare(reference, checked) }
    }
    return {
      commonTokens: common,
      prefixTokens: boundaries.secondPrefix,
      expectedRestore,
      restoredTokens: restored,
      inputTokens: second.length,
      planMode: "legacy-batches-protected-prefix",
      coldMs,
      warmMs,
      ...ordinaryComparison,
      matchedBoundaryControl,
      beforeRestoreBytes,
      afterWarmBytes,
      snapshotLimit: engine.snapshotLimit
    }
  } finally {
    engine.reset()
  }
}
