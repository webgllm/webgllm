// The page's side of a worker: a load answers only the request that made it.
import assert from "node:assert/strict"
import test from "node:test"
import { ModelWorker } from "./client.js"
import type { WorkerRequest, WorkerResponse } from "./protocol.js"
import { settle } from "./fake-worker.test.js"

function manual() {
  const sent: WorkerRequest[] = []
  const fake = {
    onmessage: null as ((event: { data: WorkerResponse }) => void) | null,
    onerror: null,
    postMessage: (message: WorkerRequest) => void sent.push(message),
    terminate() {}
  }
  const emit = (data: WorkerResponse) => fake.onmessage?.({ data })
  const worker = new ModelWorker("LiquidAI/LFM3-1.7B-GGUF", () => fake as unknown as Worker, {
    onProgress() {}, onClaim: async () => undefined, onLog() {}, onCrash() {}
  })
  const loads = () => sent.filter((m): m is Extract<WorkerRequest, { type: "load" }> => m.type === "load")
  return { worker, emit, loads }
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
