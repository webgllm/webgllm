import assert from "node:assert/strict"
import { test } from "node:test"
import type { WorkerRequest, WorkerResponse } from "./protocol.js"

/**
 * A stand-in for the model's worker: ready at once, and a generation that
 * writes tokens until it's told to stop.
 */
class FakeWorker {
  static last: FakeWorker | null = null
  onmessage: ((event: { data: WorkerResponse }) => void) | null = null
  onerror: ((event: { message: string }) => void) | null = null
  posted: Array<WorkerRequest> = []
  private writing = new Map<string, ReturnType<typeof setInterval>>()
  constructor() {
    FakeWorker.last = this
  }
  postMessage(message: WorkerRequest) {
    this.posted.push(message)
    const send = (data: WorkerResponse) => this.onmessage?.({ data })
    if (message.type === "load") {
      setTimeout(() => send({ type: "ready", model: message.model }), 0)
    } else if (message.type === "generate") {
      const { id } = message
      const timer = setInterval(() => send({ type: "token", id, text: "word " }), 1)
      // (A generation that's never stopped mustn't keep the test run alive.)
      timer.unref()
      this.writing.set(id, timer)
    } else if (message.type === "stop") {
      clearInterval(this.writing.get(message.id))
      this.writing.delete(message.id)
    }
  }
  terminate() {
    for (const timer of this.writing.values()) clearInterval(timer)
  }
}
;(globalThis as { Worker?: unknown }).Worker = FakeWorker

test("leaving the stream early stops the generation in the worker", { timeout: 10_000 }, async () => {
  const { webglText } = await import("./adapter.js")
  const adapter = webglText("onnx-community/LFM3-1.7B-ONNX")
  const stream = adapter.chatStream({
    model: "onnx-community/LFM3-1.7B-ONNX",
    messages: [{ role: "user", content: "hi" }]
  } as Parameters<typeof adapter.chatStream>[0])
  let content = 0
  for await (const chunk of stream) {
    if ((chunk as { type: string }).type === "TEXT_MESSAGE_CONTENT" && ++content === 2) break
  }
  // (The stop is posted as the stream closes.)
  await new Promise((resolve) => setTimeout(resolve, 5))
  const worker = FakeWorker.last
  assert.ok(worker, "a worker was started")
  const generate = worker.posted.find((m) => m.type === "generate")
  assert.ok(generate && generate.type === "generate")
  assert.ok(
    worker.posted.some((m) => m.type === "stop" && m.id === generate.id),
    `stop sent for ${generate.id}: ${JSON.stringify(worker.posted.map((m) => m.type))}`
  )
  worker.terminate()
})
