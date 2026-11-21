import assert from "node:assert/strict"
import { test } from "node:test"

import { TESSERA } from "./image-models.js"
import { isImageRequest, type ImageRequest } from "./image-protocol.js"
import { Images } from "./images.js"
import type { LoadProgress } from "./openai.js"
import { WEBGLLM } from "./protocol.js"

/** A worker that answers image requests as image-worker.ts does, by hand or at once. */
class FakeImageWorker {
  private listeners = { message: new Set<(e: { data: unknown }) => void>(), error: new Set<(e: unknown) => void>() }
  readonly sent: Array<Record<string, unknown>> = []
  terminated = false
  /** Answers each request at once unless set (then a test answers). */
  auto = true

  addEventListener(type: "message" | "error", listener: (e: never) => void) {
    this.listeners[type].add(listener as never)
  }
  removeEventListener(type: "message" | "error", listener: (e: never) => void) {
    this.listeners[type].delete(listener as never)
  }
  terminate() {
    this.terminated = true
  }
  emit(data: Record<string, unknown>) {
    if (this.terminated) return
    for (const l of [...this.listeners.message]) l({ data: { ...data, [WEBGLLM]: true } })
  }
  failToLoad() {
    for (const l of [...this.listeners.error]) l({ type: "error" })
  }
  /** Claims waiting for the page's answer: what each request does once granted. */
  private claims = new Map<number, { request: number; then: () => void }>()
  private claimIds = 0
  /** The bytes each claim asked for. */
  readonly claimed: Array<number> = []
  postMessage(data: Record<string, unknown>) {
    if (this.terminated) return
    this.sent.push(data)
    if (!isImageRequest(data)) return
    const r = data as ImageRequest
    if (r.type === "image-claimed") {
      const waiting = this.claims.get(r.id)
      this.claims.delete(r.id)
      if (!waiting) return
      if (r.refused === undefined) waiting.then()
      else this.emit({ type: "image-error", id: waiting.request, message: `Tessera's memory claim was refused: ${r.refused}` })
      return
    }
    if (!this.auto) return
    queueMicrotask(() => this.answer(r))
  }
  /** Claims `bytes` for request `id`, as the worker does before allocating; `then` runs once it's granted. */
  claim(id: number, model: string, bytes: number, then: () => void) {
    const c = ++this.claimIds
    this.claims.set(c, { request: id, then })
    this.claimed.push(bytes)
    this.emit({ type: "image-claim", id, claim: c, model, bytes })
  }
  answer(r: ImageRequest) {
    if (r.type === "image-load") {
      this.emit({ type: "image-progress", id: r.id, model: r.model, phase: "download", progress: 0.5, cached: false })
      this.claim(r.id, r.model, 90, () =>
        this.emit({ type: "image-ready", id: r.id, model: r.model, engine: "webgpu", weightBytes: 84 << 20, selftest: 0.965 })
      )
    } else if (r.type === "image-generate") {
      this.claim(r.id, r.model, 90 + 10 * r.n, () => {
        this.emit({ type: "image-step", id: r.id, step: 1, steps: r.steps })
        const images = Array.from({ length: r.n }, (_, i) => ({ rgba: new Uint8Array(1024).fill(i), ...(r.png && { png: new Uint8Array([137, 80, 78, 71]) }) }))
        this.emit({ type: "image-done", id: r.id, images, timings: { textMs: 1, drawMs: 2 }, workBytes: 36 << 20 })
      })
    } else if (r.type === "image-unload") this.emit({ type: "image-unloaded", id: r.id })
  }
  requests(type: string) {
    return this.sent.filter((m) => m.type === type)
  }
}

function setup(own = true, refuse?: (bytes: number) => string | undefined, holdClaims = false) {
  const made: Array<FakeImageWorker> = []
  const caller = new FakeImageWorker()
  const events: Array<string> = []
  const progress: Array<LoadProgress> = []
  /** Bytes held by model, as the budget would. */
  const held = new Map<string, number>()
  const touched: Array<string> = []
  const images = new Images({
    claim: async (model, bytes, _unload, signal) => {
      // (Held: a claim waiting for room, until the claimant goes.)
      if (holdClaims) await new Promise((_, reject) => signal.addEventListener("abort", () => reject(signal.reason), { once: true }))
      const why = refuse?.(bytes)
      if (why) throw new Error(why)
      held.set(model, bytes)
    },
    touch: (model) => touched.push(model),
    release: (model) => held.delete(model),
    createWorker: () => {
      const w = new FakeImageWorker()
      made.push(w)
      return w as unknown as Worker
    },
    ...(!own && { worker: caller as unknown as Worker }),
    checkSupport: () => undefined,
    onProgress: (p) => progress.push(p),
    dispatch: (type, event) => events.push(`${type} ${(event as { model: string }).model}${"reason" in event ? ` ${event.reason}` : ""}`)
  })
  return { images, made, caller, events, progress, held, touched }
}

test("a load starts a worker, reports progress and the model's events, once for calls at once", async () => {
  const { images, made, events, progress } = setup()
  await Promise.all([images.load(TESSERA), images.load(TESSERA)])
  assert.equal(made.length, 1)
  assert.equal(made[0]!.requests("image-load").length, 1)
  assert.equal(images.isLoaded(TESSERA), true)
  assert.equal(images.engineOf(TESSERA), "webgpu")
  assert.deepEqual(events, [`loadstart ${TESSERA}`, `load ${TESSERA}`])
  assert.deepEqual(progress, [{ model: TESSERA, phase: "download", progress: 0.5, cached: false }])
})

test("generate loads the model, draws n images as PNG (base64) or raw pixels, with its seed and memory", async () => {
  const { images, made } = setup()
  const steps: Array<number> = []
  const png = await images.generate({ prompt: "rubber duck", n: 2, seed: 7, size: "128x128" }, { onStep: (s) => steps.push(s) })
  const req = made[0]!.requests("image-generate")[0]!
  assert.deepEqual([req.prompt, req.n, req.seed, req.steps, req.guidanceScale, req.png, req.scale], ["rubber duck", 2, 7, 16, 3, true, 8])
  assert.deepEqual(png.data, [{ b64_json: "iVBORw==" }, { b64_json: "iVBORw==" }])
  assert.deepEqual([png.seed, png.width, png.height, png.model], [7, 16, 16, TESSERA])
  assert.deepEqual(png.memory, { weights_bytes: 84 << 20, work_bytes: 36 << 20 })
  assert.deepEqual(steps, [1])
  const raw = await images.generate({ prompt: "fire", response_format: "rgba" })
  assert.equal(raw.data.length, 1)
  assert.equal(raw.data[0]!.rgba!.length, 1024)
  assert.ok(Number.isInteger(raw.seed))
})

test("bad requests are refused before anything runs", async () => {
  const { images, made } = setup()
  for (const params of [{ n: 0 }, { n: 9 }, { seed: -1 }, { steps: 0 }, { size: "20x20" }, { size: "32x16" }, { response_format: "url" }])
    await assert.rejects(images.generate({ prompt: "x", ...params } as never), TypeError)
  assert.equal(made.length, 0)
})

test("unloading webgllm's own worker ends it at once, failing a drawing in progress", async () => {
  const { images, made, events } = setup()
  await images.load(TESSERA)
  made[0]!.auto = false
  const drawing = images.generate({ prompt: "cat" })
  await new Promise((r) => setTimeout(r, 0))
  await images.unload(TESSERA)
  assert.equal(made[0]!.terminated, true)
  await assert.rejects(drawing, { code: "model-not-loaded" })
  assert.equal(images.isLoaded(TESSERA), false)
  assert.deepEqual(events.slice(-1), [`unload ${TESSERA} unloaded`])
  // The next load starts a new worker.
  await images.load(TESSERA)
  assert.equal(made.length, 2)
})

test("aborting a drawing tells the worker to stop it", async () => {
  const { images, made } = setup()
  await images.load(TESSERA)
  made[0]!.auto = false
  const abort = new AbortController()
  const drawing = images.generate({ prompt: "cat" }, { signal: abort.signal })
  await new Promise((r) => setTimeout(r, 0))
  abort.abort()
  const stop = made[0]!.requests("image-stop")[0]!
  const gen = made[0]!.requests("image-generate")[0]!
  assert.equal(stop.id, gen.id)
  made[0]!.emit({ type: "image-error", id: gen.id, message: "aborted" })
  await assert.rejects(drawing, { name: "AbortError" })
})

test("a worker of the caller's frees the model and keeps running", async () => {
  const { images, made, caller, events } = setup(false)
  await images.load(TESSERA)
  assert.equal(made.length, 0)
  // Told the webgpu setting first ("auto" too: another client may have told it otherwise).
  assert.deepEqual(caller.sent[0], { type: "configure", webgpu: "auto", [WEBGLLM]: true })
  await images.unload()
  assert.equal(caller.requests("image-unload").length, 1)
  assert.equal(caller.terminated, false)
  assert.deepEqual(events.slice(-1), [`unload ${TESSERA} unloaded`])
})

test("a load that fails reports it and leaves nothing loaded; a worker that fails to start fails the load", async () => {
  const { images, made, events } = setup()
  const loading = images.load(TESSERA)
  made[0]!.auto = false
  const id = made[0]!.requests("image-load")[0]!.id as number
  made[0]!.emit({ type: "image-error", id, message: "Tessera couldn't run on this GPU" })
  await assert.rejects(loading, /couldn't run/)
  assert.equal(images.isLoaded(TESSERA), false)
  assert.equal(made[0]!.terminated, true)
  assert.deepEqual(events.slice(-2), [`error ${TESSERA}`, `unload ${TESSERA} failed`])
  const again = images.load(TESSERA)
  made[1]!.auto = false
  made[1]!.failToLoad()
  await assert.rejects(again, { code: "worker-failed" })
})

test("the worker's claims go to the shared budget before it allocates, and an unload releases them", async () => {
  const { images, made, held, touched } = setup()
  await images.load(TESSERA)
  assert.deepEqual(made[0]!.claimed, [90])
  assert.equal(held.get(TESSERA), 90)
  await images.generate({ prompt: "duck", n: 4 })
  assert.deepEqual(made[0]!.claimed, [90, 130])
  assert.equal(held.get(TESSERA), 130)
  assert.deepEqual(touched, [TESSERA])
  await images.unload(TESSERA)
  assert.equal(held.has(TESSERA), false)
})

test("a refused claim fails the load (or the drawing) and holds nothing", async () => {
  const refusing = setup(true, () => "the host has no room")
  await assert.rejects(refusing.images.load(TESSERA), /no room/)
  assert.equal(refusing.images.isLoaded(TESSERA), false)
  assert.equal(refusing.held.size, 0)
  assert.equal(refusing.made[0]!.terminated, true)
  // A drawing whose batch needs more than the host allows fails; the model stays loaded.
  const big = setup(true, (bytes) => (bytes > 100 ? "too big" : undefined))
  await big.images.load(TESSERA)
  await assert.rejects(big.images.generate({ prompt: "duck", n: 4 }), /too big/)
  assert.equal(big.images.isLoaded(TESSERA), true)
})

test("the last call waiting for a load aborting stops it; one of two leaves it loading", async () => {
  const { images, made, events } = setup(true, undefined, true)
  const a = new AbortController()
  const b = new AbortController()
  const first = images.load(TESSERA, { signal: a.signal })
  const second = images.load(TESSERA, { signal: b.signal })
  await new Promise((r) => setTimeout(r, 0))
  a.abort()
  await assert.rejects(first, { name: "AbortError" })
  assert.equal(made[0]!.requests("image-stop").length, 0, "stopped while another call waits")
  b.abort()
  await assert.rejects(second, { name: "AbortError" })
  assert.equal(made[0]!.terminated, true)
  assert.equal(images.isLoaded(TESSERA), false)
  assert.deepEqual(events.slice(-1), [`unload ${TESSERA} aborted`])
  // The next load starts afresh.
  const next = images.load(TESSERA)
  await new Promise((r) => setTimeout(r, 0))
  assert.equal(made.length, 2)
  void images.unload()
  await assert.rejects(next)
})

test("unloading the caller's worker stops its drawing first, which fails", async () => {
  const { images, caller } = setup(false)
  await images.load(TESSERA)
  caller.auto = false
  const drawing = images.generate({ prompt: "cat" })
  await new Promise((r) => setTimeout(r, 0))
  const gen = caller.requests("image-generate")[0]!
  const unloading = images.unload(TESSERA)
  const order = caller.sent.map((m) => m.type)
  assert.ok(order.indexOf("image-stop") >= 0 && order.indexOf("image-stop") < order.indexOf("image-unload"), "the unload went before the stop")
  assert.equal(caller.requests("image-stop")[0]!.id, gen.id)
  await assert.rejects(drawing, { code: "model-not-loaded" })
  caller.answer(caller.requests("image-unload")[0] as unknown as ImageRequest)
  await unloading
})

test("a worker that fails after loading takes its model out; the next load starts a new worker", async () => {
  const { images, made, events, held } = setup()
  await images.load(TESSERA)
  made[0]!.failToLoad()
  assert.equal(images.isLoaded(TESSERA), false)
  assert.equal(held.has(TESSERA), false)
  assert.deepEqual(events.slice(-2), [`error ${TESSERA}`, `unload ${TESSERA} crashed`])
  await images.generate({ prompt: "fire" })
  assert.equal(made.length, 2)
})
