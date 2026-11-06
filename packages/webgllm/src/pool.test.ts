import assert from "node:assert/strict"
import { test } from "node:test"
import { WebGLLMError } from "./errors.js"
import { fakeStorage, fakeWorkers, settle } from "./fake-worker.test.js"
import type { FakeOptions } from "./fake-worker.test.js"
import { MemoryBudget } from "./model-memory.js"
import type { LoadProgress } from "./openai.js"
import { downloadProgress, loadProgress, ModelPool, TaskQueue } from "./pool.js"

const A = "LiquidAI/LFM2.5-350M-GGUF"
const B = "LiquidAI/LFM2.5-1.2B-Instruct-GGUF"
const C = "LiquidAI/LFM3-1.7B-GGUF"

const fields = { chat: [{ role: "user" as const, content: "hi" }], tools: [], sampling: { temperature: 0, top_k: 1, repetition_penalty: 1 } }

function pool(options: FakeOptions & { budget?: number; downloaded?: Set<string> } = {}) {
  const workers = fakeWorkers({ bytes: () => 60, ...options })
  const progress: Array<LoadProgress> = []
  const logs: Array<string> = []
  const storage = fakeStorage(options.downloaded)
  const p = new ModelPool({
    createWorker: workers.create,
    memoryBudget: () => options.budget ?? 100,
    storage,
    onProgress: (event) => progress.push(event),
    log: (_, text) => logs.push(text)
  })
  return { pool: p, workers, progress, logs, storage }
}

test("the budget unloads the least recently used model, keeping models kept loaded", async () => {
  const unloaded: Array<string> = []
  const budget = new MemoryBudget(() => 100, (model) => {
    unloaded.push(model)
  })
  await budget.claim("a", 40, false)
  await budget.claim("b", 40, true)
  await budget.claim("c", 10, false)
  budget.touch("a")
  // 40 + 40 + 10 + 20 > 100: c is the least recently used one that may go.
  await budget.claim("d", 20, false)
  assert.deepEqual(unloaded, ["c"])
  // A model's own claims replace its earlier one, and keep its place.
  await budget.claim("d", 10, false)
  assert.equal(budget.total, 90)
  // Past the budget with only kept models left: they stay.
  await budget.claim("e", 100, false)
  assert.deepEqual(unloaded, ["c", "a", "d"])
  assert.deepEqual(budget.models().sort(), ["b", "e"])
})

test("loading past the budget unloads the least recently used model", async () => {
  const { pool: p, workers } = pool()
  await p.load(A)
  assert.equal(p.isLoaded(A), true)
  await p.load(B)
  assert.equal(p.isLoaded(A), false)
  assert.equal(p.isLoaded(B), true)
  assert.equal(workers.workers[0]?.terminated, true)
  assert.equal(p.memoryUsed, 60)
})

test("models kept loaded stay, even past the budget", async () => {
  const { pool: p } = pool()
  await p.load(A, { keepLoaded: true })
  await p.load(B)
  assert.deepEqual(p.loadedModels().sort(), [A, B].sort())
})

test("using a model makes it the most recently used", async () => {
  const { pool: p } = pool({ budget: 130, reply: () => ({ tokens: ["x"] }) })
  await p.load(A)
  await p.load(B)
  await p.generate(A, fields, { onToken: () => undefined })
  await p.load(C)
  assert.deepEqual(p.loadedModels().sort(), [A, C].sort())
})

test("generations run one at a time across models", async () => {
  const { pool: p, workers } = pool({ budget: 1000 })
  await p.load(A)
  await p.load(B)
  const tokens: Array<string> = []
  const first = p.generate(A, fields, { onToken: (t) => tokens.push(`a:${t}`) })
  const second = p.generate(B, fields, { onToken: (t) => tokens.push(`b:${t}`) })
  await settle()
  const a = workers.of(A)!
  const b = workers.of(B)!
  assert.equal(a.generations().length, 1)
  assert.equal(b.generations().length, 0, "B waits for A's generation")
  a.finish(a.generations()[0]!.id, ["1"])
  await first
  await settle()
  assert.equal(b.generations().length, 1)
  b.finish(b.generations()[0]!.id, ["2"])
  const end = await second
  assert.equal(end.finishReason, "stop")
  assert.deepEqual(tokens, ["a:1", "b:2"])
})

test("a model being unloaded for room finishes its running generation first", async () => {
  const { pool: p, workers } = pool()
  await p.load(A)
  const running = p.generate(A, fields, { onToken: () => undefined })
  await settle()
  const a = workers.of(A)!
  const loadB = p.load(B)
  await settle()
  assert.equal(a.terminated, false, "A generates on")
  a.finish(a.generations()[0]!.id, ["done"])
  await running
  await loadB
  assert.equal(a.terminated, true)
  assert.equal(p.isLoaded(B), true)
})

test("a generation for an unloaded model loads it again", async () => {
  const { pool: p, workers } = pool({ reply: () => ({ tokens: ["ok"] }) })
  await p.load(A)
  await p.load(B)
  assert.equal(p.isLoaded(A), false)
  const tokens: Array<string> = []
  await p.generate(A, fields, { onToken: (t) => tokens.push(t) })
  assert.deepEqual(tokens, ["ok"])
  assert.equal(p.isLoaded(A), true)
  assert.equal(workers.workers.filter((w) => w.model === A).length, 2)
})

test("aborting stops the worker's generation", async () => {
  const { pool: p, workers } = pool()
  await p.load(A)
  const abort = new AbortController()
  const running = p.generate(A, fields, { onToken: () => undefined, signal: abort.signal })
  await settle()
  abort.abort()
  await running
  assert.ok(workers.of(A)!.sent.some((m) => m.type === "stop"))
})

test("a generation aborted while queued never reaches the worker", async () => {
  const { pool: p, workers } = pool()
  await p.load(A)
  const first = p.generate(A, fields, { onToken: () => undefined })
  const abort = new AbortController()
  const second = p.generate(A, fields, { onToken: () => undefined, signal: abort.signal })
  await settle()
  abort.abort()
  const a = workers.of(A)!
  a.finish(a.generations()[0]!.id)
  await first
  await assert.rejects(second, { name: "AbortError" })
  assert.equal(a.generations().length, 1)
})

test("unloading fails the running generation with model-not-loaded", async () => {
  const { pool: p } = pool()
  await p.load(A)
  const running = p.generate(A, fields, { onToken: () => undefined })
  await settle()
  p.unload()
  await assert.rejects(running, (error: unknown) => error instanceof WebGLLMError && error.code === "model-not-loaded")
  assert.equal(p.memoryUsed, 0)
})

test("a load error keeps the worker's code; the next load starts a fresh worker", async () => {
  let fail = true
  const { pool: p, workers } = pool({ loadError: () => (fail ? { message: "no room", code: "storage-full" } : undefined) })
  await assert.rejects(p.load(A), (error: unknown) => error instanceof WebGLLMError && error.code === "storage-full")
  assert.equal(workers.workers[0]?.terminated, true)
  fail = false
  await p.load(A)
  assert.equal(workers.workers.length, 2)
})

test("an uncoded load error is a plain Error", async () => {
  const { pool: p } = pool({ loadError: () => ({ message: "boom" }) })
  await assert.rejects(p.load(A), (error: unknown) => !(error instanceof WebGLLMError) && (error as Error).message === "boom")
})

test("aborting the only wait for a load stops it", async () => {
  const { pool: p, workers } = pool()
  const abort = new AbortController()
  const load = p.load(A, { signal: abort.signal })
  abort.abort()
  await assert.rejects(load, { name: "AbortError" })
  await settle()
  assert.equal(workers.workers[0]?.terminated, true)
  assert.equal(p.isLoaded(A), false)
})

test("a load for another context loads again, and generations carry the loaded context", async () => {
  const { pool: p, workers } = pool({ reply: () => ({ tokens: [] }) })
  await p.load(A, { contextLength: 4096 })
  await p.load(A, { contextLength: 4096 })
  const a = workers.of(A)!
  assert.equal(a.sent.filter((m) => m.type === "load").length, 1)
  await p.load(A, { contextLength: 8192 })
  assert.equal(a.sent.filter((m) => m.type === "load").length, 2)
  // A generation takes the model as loaded, whatever context it asks for.
  await p.generate(A, fields, { contextLength: 1024, onToken: () => undefined })
  const generate = a.generations()[0]!
  assert.equal(generate.contextLength, 8192)
  assert.equal(p.requestedContext(A), 8192)
})

test("progress reaches the instance and the loading call", async () => {
  const { pool: p, progress } = pool()
  const mine: Array<LoadProgress> = []
  await p.load(A, { onProgress: (event) => mine.push(event) })
  assert.deepEqual(mine, [{ model: A, phase: "load", progress: 0.5, cached: true }])
  assert.deepEqual(progress, mine)
  // A later load that's already done reports nothing.
  await p.load(A, { onProgress: (event) => mine.push(event) })
  assert.equal(mine.length, 1)
})

test("downloads run in a worker of their own, ended after", async () => {
  const { pool: p, workers } = pool()
  const events: Array<LoadProgress> = []
  await p.download(A, { onProgress: (event) => events.push(event) })
  assert.equal(workers.workers.length, 1)
  assert.equal(workers.workers[0]?.terminated, true)
  assert.deepEqual(workers.workers[0]?.sent, [{ type: "download", model: A }])
  assert.deepEqual(events, [{ model: A, phase: "download", progress: 0.5, cached: false, loadedBytes: 50, totalBytes: 100 }])
  assert.equal(p.isLoaded(A), false)
})

test("a downloaded or loaded model doesn't download again", async () => {
  const { pool: p, workers } = pool({ downloaded: new Set([A]) })
  await p.download(A)
  await p.load(B)
  await p.download(B)
  assert.equal(workers.workers.length, 1)
})

test("progress phases: downloading is 0–0.9 of a load, then reading into the GPU", () => {
  assert.equal(loadProgress({ type: "progress", model: A, progress: 0.3, cached: false }).phase, "download")
  assert.equal(loadProgress({ type: "progress", model: A, progress: 0.95, cached: false }).phase, "load")
  assert.equal(loadProgress({ type: "progress", model: A, progress: 0.3, cached: true }).phase, "load")
  assert.equal(downloadProgress({ type: "progress", model: A, progress: 0.45, cached: false, loadedBytes: 1, totalBytes: 4 }).progress, 0.25)
  assert.equal(downloadProgress({ type: "progress", model: A, progress: 1, cached: true }).progress, 1)
})

test("the task queue runs tasks in order, one at a time", async () => {
  const queue = new TaskQueue()
  const order: Array<string> = []
  let release!: () => void
  const first = queue.run(async () => {
    order.push("first start")
    await new Promise<void>((resolve) => (release = resolve))
    order.push("first end")
  })
  const second = queue.run(async () => {
    order.push("second")
  })
  await settle()
  assert.deepEqual(order, ["first start"])
  release()
  await Promise.all([first, second])
  assert.deepEqual(order, ["first start", "first end", "second"])
})

test("a download waiting on a load settles when the files are there, whatever the GPU says", async () => {
  // Stored files: the load's GPU failure isn't the download's.
  const gpu = { message: "no WebGL 2", code: "webgl2-unavailable" as const }
  const stored = pool({ loadError: () => gpu, downloaded: new Set([A]) })
  const load = stored.pool.load(A)
  const download = stored.pool.download(A)
  await assert.rejects(load, { message: "no WebGL 2" })
  await download
  // Not stored: the download goes on in a worker of its own once the load fails.
  const fresh = pool({ loadError: () => gpu })
  const failing = fresh.pool.load(A)
  const downloading = fresh.pool.download(A)
  await assert.rejects(failing, { message: "no WebGL 2" })
  await downloading
  assert.deepEqual(fresh.workers.workers.map((w) => w.sent[0]?.type), ["load", "download"])
})

test("a load started while a download checks storage waits for that download", async () => {
  const { pool: p, workers } = pool()
  const download = p.download(A)
  const load = p.load(A)
  await Promise.all([download, load])
  // One worker fetches the files at a time: the download's, then the load's.
  assert.deepEqual(workers.workers.map((w) => w.sent[0]?.type), ["download", "load"])
  assert.equal(p.isLoaded(A), true)
})

test("a load canceled while it waits for room doesn't unload the model it was waiting on", async () => {
  const { pool: p, workers } = pool()
  await p.load(A)
  const running = p.generate(A, fields, { onToken: () => undefined })
  await settle()
  const abort = new AbortController()
  const loadB = p.load(B, { signal: abort.signal })
  await settle()
  abort.abort()
  await assert.rejects(loadB, { name: "AbortError" })
  const a = workers.of(A)!
  a.finish(a.generations()[0]!.id)
  await running
  await settle()
  assert.equal(a.terminated, false)
  assert.equal(p.isLoaded(A), true)
  assert.equal(p.isLoaded(B), false)
  assert.equal(p.memoryUsed, 60)
  // And the next claim isn't stuck behind it.
  await p.load(C)
  assert.equal(p.isLoaded(C), true)
})

test("a download canceled after its storage check starts no worker", async () => {
  // Cancel at each step of the way: none may start writing after it.
  for (let hops = 0; hops < 12; hops++) {
    const { pool: p, workers } = pool()
    const download = p.download(A).catch(() => undefined)
    for (let i = 0; i < hops; i++) await Promise.resolve()
    const before = workers.workers.length
    p.cancelDownload(A)
    await download
    await settle()
    assert.equal(workers.workers.length, before, `canceled after ${hops} microtasks`)
  }
})

test("canceling a resize queued behind a reply keeps the model loaded and the reply going", async () => {
  const { pool: p, workers } = pool()
  await p.load(A, { contextLength: 8192 })
  const running = p.generate(A, fields, { onToken: () => undefined })
  await settle()
  const abort = new AbortController()
  const resize = p.load(A, { contextLength: 4096, signal: abort.signal })
  await settle()
  abort.abort()
  await assert.rejects(resize, { name: "AbortError" })
  const a = workers.of(A)!
  assert.equal(a.terminated, false)
  assert.equal(p.isLoaded(A), true)
  assert.equal(p.requestedContext(A), 8192)
  a.finish(a.generations()[0]!.id, ["ok"])
  const end = await running
  assert.equal(end.finishReason, "stop")
  await settle()
  // The resize never reached the worker; the next generation runs as loaded.
  assert.equal(a.sent.filter((m) => m.type === "load").length, 1)
  const next = p.generate(A, fields, { onToken: () => undefined })
  await settle()
  assert.equal(a.generations()[1]?.contextLength, 8192)
  a.finish(a.generations()[1]!.id)
  await next
})

test("a download settles once a concurrent load has stored the files, before the GPU is done", async () => {
  let release!: () => void
  const held = new Promise<void>((resolve) => (release = resolve))
  const downloaded = new Set<string>()
  const { pool: p, workers } = pool({ downloaded, stored: (model) => downloaded.add(model), hold: () => held })
  const load = p.load(A)
  const download = p.download(A)
  let done = false
  void download.then(() => (done = true))
  await settle()
  assert.equal(done, true)
  assert.equal(workers.workers.length, 1, "no download worker of its own")
  release()
  await load
})

test("a resize queued behind a reply is sent when the reply ends", async () => {
  const { pool: p, workers } = pool()
  await p.load(A, { contextLength: 8192 })
  const running = p.generate(A, fields, { onToken: () => undefined })
  await settle()
  const resize = p.load(A, { contextLength: 4096 })
  await settle()
  const a = workers.of(A)!
  assert.equal(a.sent.filter((m) => m.type === "load").length, 1)
  a.finish(a.generations()[0]!.id)
  await running
  await resize
  assert.deepEqual(a.sent.filter((m) => m.type === "load").map((m) => m.type === "load" && m.contextLength), [8192, 4096])
  assert.equal(p.requestedContext(A), 4096)
})
