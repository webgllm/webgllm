import assert from "node:assert/strict"
import { test } from "node:test"
import { setFlagsFromString } from "node:v8"
import { runInNewContext } from "node:vm"

import { disposeThenCollect } from "./image-worker.js"

setFlagsFromString("--expose-gc")
const gc = runInNewContext("gc") as () => void
const tick = () => new Promise((resolve) => setTimeout(resolve, 0))

/** Whether `ref`'s target is collected by a collection run inside `collect` (in tasks of their own: deref keeps its target to the end of its task). */
async function collectedDuring(release: (collect: () => Promise<void>) => Promise<void>, ref: WeakRef<object>) {
  let collected = false
  await release(async () => {
    await tick()
    gc()
    await tick()
    collected = ref.deref() === undefined
  })
  return collected
}

test("a released model is collectable while the collector runs: nothing of the release holds it", async () => {
  let current: { weights: Float64Array } | null = { weights: new Float64Array(1 << 20) }
  const ref = new WeakRef(current)
  await tick()
  let disposed = 0
  const release = (collect: () => Promise<void>) =>
    disposeThenCollect(
      () => {
        const was = current
        current = null
        return was
      },
      () => void disposed++,
      collect
    )
  assert.equal(await collectedDuring(release, ref), true)
  assert.equal(disposed, 1)
  // Nothing loaded: nothing to free, and no collection.
  let collections = 0
  await disposeThenCollect(() => null, () => void disposed++, async () => void collections++)
  assert.deepEqual([disposed, collections], [1, 0])
})

test("(control) an async release that awaits the collection holds the model through it", async () => {
  let current: { weights: Float64Array } | null = { weights: new Float64Array(1 << 20) }
  const ref = new WeakRef(current)
  await tick()
  const release = async (collect: () => Promise<void>) => {
    const was = current
    current = null
    if (!was) return
    void was.weights.length
    await collect()
    void was.weights.length
  }
  assert.equal(await collectedDuring(release, ref), false)
})
