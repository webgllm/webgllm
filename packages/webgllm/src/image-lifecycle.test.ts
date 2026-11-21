// The page's side (images.ts) and the worker's (image-worker.ts) together,
// over an in-memory worker of the caller's, with the model's files, GPU and
// model stubbed: unloads, aborts and drawings racing each other.
import assert from "node:assert/strict"
import { test } from "node:test"

import { TESSERA } from "./image-models.js"
import { isImageRequest } from "./image-protocol.js"
import { imageDeps, receiveImage, type ImageHost, type ImageModelOnGpu } from "./image-worker.js"
import { Images } from "./images.js"
import { WEBGLLM } from "./protocol.js"

const tick = () => new Promise((resolve) => setTimeout(resolve, 0))

function gate() {
  let open!: () => void
  const promise = new Promise<void>((resolve) => (open = resolve))
  return { promise, open }
}

/** The stubbed model: what it was asked, and gates a test can hold its steps at. */
const stub = {
  gpus: 0,
  disposed: 0,
  denoises: 0,
  /** Holds the next denoiser pass until opened (null: none held). */
  hold: null as ReturnType<typeof gate> | null,
  /** Called as the model loads onto the GPU. */
  onLoad: (): void | Promise<void> => undefined
}

imageDeps.prepare = async () => ({
  weights: {} as never,
  clip: {} as never,
  weightBytes: 90,
  tokenize: () => new Array(24).fill(0),
  palette: Array.from({ length: 512 }, () => [0, 0, 0, 255]),
  selftest: { ids: new Array(24).fill(0), argmax: new Array(256).fill(3) },
  cached: true
})
imageDeps.gpu = async () => {
  stub.gpus++
  return { dispose: () => void stub.disposed++ }
}
imageDeps.model = (): ImageModelOnGpu => ({
  load: async () => {
    await stub.onLoad()
  },
  encodeText: async () => ({ states: new Float32Array(24 * 512), pooled: new Float32Array(512) }),
  setText: async () => undefined,
  denoise: async (_tokens, _ratio, _sc, batch) => {
    stub.denoises++
    const held = stub.hold
    if (held) {
      stub.hold = null
      await held.promise
    }
    const logits = new Float32Array(batch * 256 * 512)
    for (let i = 0; i < batch * 256; i++) logits[i * 512 + 3] = 30
    return logits
  },
  dispose: () => undefined,
  workBytesFor: (batch: number) => 10 * batch,
  bytes: { weights: 90, work: 0 }
})

/** A worker of the caller's running image-worker.ts: messages each way in a task of their own; `holdReplies` keeps the worker's answers back. */
class BridgeWorker {
  private listeners = { message: new Set<(e: { data: unknown }) => void>(), error: new Set<(e: unknown) => void>() }
  readonly toWorker: Array<string> = []
  readonly toPage: Array<string> = []
  holdReplies = false
  private held: Array<unknown> = []
  readonly host: ImageHost = {
    post: (message) => this.deliver({ ...message, [WEBGLLM]: true } as unknown as { type: string }),
    webgpu: () => true,
    collect: async () => undefined
  }
  addEventListener(type: "message" | "error", l: (e: never) => void) {
    this.listeners[type].add(l as never)
  }
  removeEventListener(type: "message" | "error", l: (e: never) => void) {
    this.listeners[type].delete(l as never)
  }
  terminate() {}
  postMessage(data: { type: string }) {
    this.toWorker.push(data.type)
    if (isImageRequest(data)) setTimeout(() => receiveImage(data, this.host), 0)
  }
  private deliver(message: { type: string }) {
    if (this.holdReplies) return void this.held.push(message)
    setTimeout(() => {
      this.toPage.push(message.type)
      for (const l of [...this.listeners.message]) l({ data: message })
    }, 0)
  }
  flush() {
    this.holdReplies = false
    for (const m of this.held.splice(0)) this.deliver(m as { type: string })
  }
}

function setup() {
  const worker = new BridgeWorker()
  const held = new Map<string, number>()
  const events: Array<string> = []
  const images = new Images({
    createWorker: () => {
      throw new Error("the caller's worker is used")
    },
    worker: worker as unknown as Worker,
    checkSupport: () => undefined,
    onProgress: () => undefined,
    dispatch: (type, event) => events.push(`${type}${"reason" in event ? ` ${event.reason}` : ""}`),
    claim: async (model, bytes) => void held.set(model, bytes),
    setKeepLoaded: () => undefined,
    touch: () => undefined,
    release: (model) => void held.delete(model)
  })
  return { images, worker, held, events }
}

test("a drawing asked for while the caller's worker unloads waits for the unload, then loads the model again", { timeout: 5000 }, async () => {
  const { images, worker, held } = setup()
  await images.load(TESSERA)
  stub.hold = gate()
  const held1 = stub.hold
  const first = images.generate({ prompt: "cat" })
  first.catch(() => undefined)
  while (stub.hold) await tick()
  const unloading = images.unload(TESSERA)
  const second = images.generate({ prompt: "duck" })
  await assert.rejects(first, { code: "model-not-loaded" })
  held1.open()
  await unloading
  const res = await second
  assert.equal(res.data.length, 1)
  // The worker freed the model before it loaded it again: the second load comes after the unload, and nothing waits.
  assert.ok(worker.toWorker.lastIndexOf("image-load") > worker.toWorker.indexOf("image-unload"))
  // (Its claim: weights 90, then a one-sprite batch's two rows, 10 each.)
  assert.equal(held.get(TESSERA), 110)
  await images.unload()
  assert.equal(held.size, 0)
})

test("unloading the caller's worker during a load resolves once the worker has freed it, and releases it then", { timeout: 5000 }, async () => {
  const { images, worker, held } = setup()
  const loadHeld = gate()
  stub.onLoad = () => loadHeld.promise
  const loading = images.load(TESSERA)
  loading.catch(() => undefined)
  while (!held.has(TESSERA)) await tick()
  let unloaded = false
  const unloading = images.unload(TESSERA).then(() => (unloaded = true))
  await assert.rejects(loading)
  await tick()
  assert.equal(unloaded, false, "resolved before the worker said it freed the model")
  assert.equal(held.has(TESSERA), true, "released before the worker freed it")
  loadHeld.open()
  stub.onLoad = () => undefined
  await unloading
  assert.ok(worker.toPage.includes("image-unloaded"))
  assert.equal(held.has(TESSERA), false)
})

test("the last load waiter aborting as the model becomes ready still frees it in the caller's worker", { timeout: 5000 }, async () => {
  const { images, worker, held, events } = setup()
  const gpusBefore = stub.disposed
  // The worker finishes the load, but its answer is still on its way when the call gives up.
  stub.onLoad = () => void (worker.holdReplies = true)
  const abort = new AbortController()
  const loading = images.load(TESSERA, { signal: abort.signal })
  while (!worker.holdReplies) await tick()
  for (let i = 0; i < 5; i++) await tick()
  abort.abort()
  await assert.rejects(loading, { name: "AbortError" })
  stub.onLoad = () => undefined
  worker.flush()
  while (!events.includes("unload aborted")) await tick()
  assert.ok(worker.toWorker.includes("image-unload"))
  assert.equal(stub.disposed, gpusBefore + 1, "the worker kept the model")
  assert.equal(held.has(TESSERA), false)
  assert.equal(images.isLoaded(TESSERA), false)
})

test("a load waiting behind a teardown stops at once when aborted; the teardown goes on", { timeout: 5000 }, async () => {
  const { images, worker, events } = setup()
  await images.load(TESSERA)
  // The worker's answer to the unload is held: the teardown waits for it.
  worker.holdReplies = true
  const unloading = images.unload(TESSERA)
  const abort = new AbortController()
  const reload = images.load(TESSERA, { signal: abort.signal })
  await tick()
  abort.abort()
  await assert.rejects(reload, { name: "AbortError" })
  assert.equal(events.includes("unload unloaded"), false, "the teardown ended before its worker answered")
  worker.flush()
  await unloading
  assert.equal(images.isLoaded(TESSERA), false)
})

test("an unload of every model stops a reload still waiting behind an earlier teardown", { timeout: 5000 }, async () => {
  const { images, worker } = setup()
  await images.load(TESSERA)
  worker.holdReplies = true
  const unloading = images.unload(TESSERA)
  const reload = images.load(TESSERA)
  reload.catch(() => undefined)
  await tick()
  const all = images.unload()
  worker.flush()
  await Promise.all([unloading, all])
  await assert.rejects(reload, { code: "model-not-loaded" })
  for (let i = 0; i < 5; i++) await tick()
  assert.equal(images.isLoaded(TESSERA), false)
  assert.equal(worker.toWorker.filter((t) => t === "image-load").length, 1, "the reload started after the unload")
})
