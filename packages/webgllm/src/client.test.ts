// The page's side of a worker: a load answers only the request that made it.
import assert from "node:assert/strict"
import test from "node:test"
import { ModelWorker } from "./client.js"
import type { WorkerRequest, WorkerResponse } from "./protocol.js"
import { settle } from "./fake-worker.test.js"

function manual() {
  const sent: WorkerRequest[] = []
  const raw: unknown[] = []
  let listener: ((event: { data: WorkerResponse }) => void) | undefined
  const fake = {
    addEventListener: (type: string, added: (event: { data: WorkerResponse }) => void) => {
      if (type === "message") listener = added
    },
    removeEventListener() {},
    postMessage: (message: WorkerRequest) => {
      raw.push(message)
      sent.push(message)
    },
    terminate() {}
  }
  const emitOwn = (data: unknown) => listener?.({ data: data as WorkerResponse })
  const emit = (data: WorkerResponse) => emitOwn({ ...data, webgllm: true })
  const worker = new ModelWorker("LiquidAI/LFM3-1.7B-GGUF", fake as unknown as Worker, true, {
    onProgress() {}, onClaim: async () => undefined, onLog() {}, onCrash() {}
  })
  const loads = () => sent.filter((m): m is Extract<WorkerRequest, { type: "load" }> => m.type === "load")
  return { worker, emit, emitOwn, loads, raw }
}

test("a late ready for a replaced load doesn't resolve the new one", async () => {
  const { worker, emit, loads } = manual()
  void worker.loadModel(undefined)
  let resized = false
  void worker.loadModel(64).then(() => { resized = true })
  const [first, second] = loads()
  emit({ type: "ready", model: worker.model, loadId: first.loadId, contextLength: 4096 })
  await settle()
  assert.equal(resized, false)
  assert.equal(worker.ready, false)
  emit({ type: "ready", model: worker.model, loadId: second.loadId, contextLength: 64 })
  await settle()
  assert.equal(resized, true)
  assert.equal(worker.loadedContextLength, 64)
})

test("a late load-error for a replaced load doesn't fail the new one", async () => {
  const { worker, emit, loads } = manual()
  void worker.loadModel(undefined)
  const resized = worker.loadModel(64)
  const [first, second] = loads()
  emit({ type: "load-error", model: worker.model, loadId: first.loadId, message: "Switched." })
  emit({ type: "ready", model: worker.model, loadId: second.loadId, contextLength: 64 })
  await resized
  assert.equal(worker.ready, true)
})

test("an answer without webgllm's mark, or naming no load of this one's, settles nothing", async () => {
  const { worker, emit, emitOwn, loads } = manual()
  let loaded = false
  const load = worker.loadModel(2048).then(() => { loaded = true })
  const [sent] = loads()
  // The worker's own message, shaped like an answer (even with the load's id).
  emitOwn({ type: "ready", model: worker.model, contextLength: 8192 })
  emitOwn({ type: "ready", model: worker.model, loadId: sent.loadId, contextLength: 8192 })
  // webgllm's, for another load.
  emit({ type: "ready", model: worker.model, loadId: sent.loadId + 1000, contextLength: 8192 })
  await settle()
  assert.equal(loaded, false)
  assert.equal(worker.ready, false)
  emit({ type: "ready", model: worker.model, loadId: sent.loadId, contextLength: 2048 })
  await load
  assert.equal(worker.loadedContextLength, 2048)
})

test("every request carries webgllm's mark", () => {
  const { worker, raw } = manual()
  void worker.loadModel(undefined)
  void worker.download()
  assert.equal(raw.length, 2)
  assert.ok(raw.every((message) => (message as { webgllm?: unknown }).webgllm === true))
})
