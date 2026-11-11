import assert from "node:assert/strict"
import { test } from "node:test"
import { createSharedDownloads } from "./shared-download.js"

function controllable() {
  let resolve!: (value: string) => void
  let progress!: (loaded: number, total: number) => void
  let signal!: AbortSignal
  let starts = 0
  const start = (onProgress: typeof progress, s: AbortSignal) => {
    starts++
    progress = onProgress
    signal = s
    return new Promise<string>((res, reject) => {
      resolve = res
      s.addEventListener("abort", () => reject(s.reason))
    })
  }
  return { start, get starts() { return starts }, resolve: (v: string) => resolve(v), progress: (a: number, b: number) => progress(a, b), get signal() { return signal } }
}

test("callers of one key share one download and each hear its progress", async () => {
  const shared = createSharedDownloads<string>()
  const c = controllable()
  const heard: string[] = []
  const a = shared("f", c.start, (l, t) => heard.push(`a${l}/${t}`))
  c.progress(1, 4)
  const b = shared("f", c.start, (l, t) => heard.push(`b${l}/${t}`))
  c.progress(2, 4)
  c.resolve("file")
  assert.deepEqual(await Promise.all([a, b]), ["file", "file"])
  assert.equal(c.starts, 1)
  assert.deepEqual(heard, ["a1/4", "b1/4", "a2/4", "b2/4"])
  // Settled: a later caller starts afresh.
  const again = shared("f", c.start, () => undefined)
  c.resolve("new")
  assert.equal(await again, "new")
  assert.equal(c.starts, 2)
})

test("one caller aborting leaves the download to the others", async () => {
  const shared = createSharedDownloads<string>()
  const c = controllable()
  const abort = new AbortController()
  const a = shared("f", c.start, () => undefined, abort.signal)
  const b = shared("f", c.start, () => undefined)
  abort.abort(new Error("superseded"))
  await assert.rejects(a, /superseded/)
  assert.equal(c.signal.aborted, false)
  c.resolve("file")
  assert.equal(await b, "file")
})

test("the download stops once every caller has aborted", async () => {
  const shared = createSharedDownloads<string>()
  const c = controllable()
  const one = new AbortController(), two = new AbortController()
  const a = shared("f", c.start, () => undefined, one.signal)
  const b = shared("f", c.start, () => undefined, two.signal)
  one.abort(new Error("one"))
  two.abort(new Error("two"))
  await assert.rejects(a, /one/)
  await assert.rejects(b, /two/)
  assert.equal(c.signal.aborted, true)
  // A new caller doesn't join the aborted download.
  const next = shared("f", c.start, () => undefined)
  c.resolve("fresh")
  assert.equal(await next, "fresh")
  assert.equal(c.starts, 2)
})

test("an already aborted caller doesn't start anything", async () => {
  const shared = createSharedDownloads<string>()
  const c = controllable()
  const abort = new AbortController()
  abort.abort(new Error("gone"))
  await assert.rejects(shared("f", c.start, () => undefined, abort.signal), /gone/)
  assert.equal(c.starts, 0)
})

test("the caller whose abort stops the download is rejected once it has ended", async () => {
  const shared = createSharedDownloads<string>()
  let closed!: () => void
  let signal!: AbortSignal
  // A download that takes a moment to close its file after it's aborted.
  const start = (_: unknown, s: AbortSignal) => {
    signal = s
    return new Promise<string>((_, reject) => { closed = () => reject(s.reason) })
  }
  const abort = new AbortController()
  let rejected = false
  const a = shared("f", start, () => undefined, abort.signal).catch(() => { rejected = true })
  abort.abort(new Error("stopped"))
  await new Promise((r) => setTimeout(r, 0))
  assert.equal(signal.aborted, true)
  assert.equal(rejected, false, "rejected while the file was still open")
  closed()
  await a
  assert.equal(rejected, true)
})
