// The worker's collection after a released engine (handler.ts collectReleased):
// none when nothing was released, its rounds otherwise, and one at a time.
import assert from "node:assert/strict"
import { test } from "node:test"
import { readFile } from "node:fs/promises"
import { join } from "node:path"
import { pathToFileURL } from "node:url"
import { createRequire } from "node:module"

const require = createRequire(join(process.cwd(), "package.json"))
const { transform } = await import(pathToFileURL(require.resolve("esbuild")).href) as typeof import("esbuild")

const source = await readFile(join(process.cwd(), "src/handler.ts"), "utf8")
const start = source.indexOf("/** Whether an engine was released since the last collectReleased. */")
const end = source.indexOf("/** What closes the stored file")
assert.ok(start >= 0 && end > start)
const { code: js } = await transform(source.slice(start, end), { loader: "ts" })
const make = new Function(`${js}
  return { collectReleased, release: () => { uncollected = true }, rounds: GARBAGE_ROUNDS, closingOnFailure }`) as () => {
  collectReleased: () => Promise<void>
  release: () => void
  rounds: number
  closingOnFailure: <T>(loading: Promise<T>, close: () => Promise<unknown>) => Promise<T>
}

/** Counts the timers a collection waits on: one a round. */
async function timed(t: { mock: { method: (o: object, k: string, f: unknown) => void; restoreAll: () => void } }, run: () => Promise<void>) {
  let timers = 0
  const real = globalThis.setTimeout
  t.mock.method(globalThis, "setTimeout", ((fn: () => void, ms?: number) => {
    timers++
    return real(fn, ms)
  }) as typeof setTimeout)
  try {
    await run()
  } finally {
    t.mock.restoreAll()
  }
  return timers
}

test("nothing released: no garbage", async (t) => {
  const c = make()
  assert.equal(await timed(t, () => c.collectReleased()), 0)
})

test("an engine released: GARBAGE_ROUNDS rounds, once", async (t) => {
  const c = make()
  c.release()
  assert.equal(await timed(t, () => c.collectReleased()), c.rounds)
  assert.equal(await timed(t, () => c.collectReleased()), 0)
})

test("a second caller waits for the collection running, and one released meanwhile is collected after it", async () => {
  const c = make()
  c.release()
  const order: Array<string> = []
  const first = c.collectReleased().then(() => order.push("first"))
  c.release()
  const second = c.collectReleased().then(() => order.push("second"))
  const third = c.collectReleased().then(() => order.push("third"))
  await Promise.all([first, second, third])
  assert.equal(order[0], "first")
  assert.deepEqual(order.slice(1).sort(), ["second", "third"])
})

test("a retried load (the self-check's) closes the first attempt's files only if it fails, and fails with its own error", async () => {
  const { closingOnFailure } = make()
  let closed = 0
  const close = async () => void closed++
  assert.equal(await closingOnFailure(Promise.resolve("loaded"), close), "loaded")
  assert.equal(closed, 0)
  const error = new Error("retry failed")
  await assert.rejects(closingOnFailure(Promise.reject(error), close), (e) => e === error)
  assert.equal(closed, 1)
})
