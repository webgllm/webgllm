import { resolveDebugOption } from "@tanstack/ai/adapter-internals"
import assert from "node:assert/strict"
import { test } from "node:test"
import type { GenerationPhase, RunTimings, WorkerRequest, WorkerResponse } from "./protocol.js"

/**
 * A stand-in for the model's worker: ready at once, and a generation that
 * writes tokens until it's told to stop.
 */
class FakeWorker {
  static last: FakeWorker | null = null
  static onGenerate?: (worker: FakeWorker, request: Extract<WorkerRequest, { type: "generate" }>) => void
  onmessage: ((event: { data: WorkerResponse }) => void) | null = null
  onerror: ((event: { message: string }) => void) | null = null
  posted: Array<WorkerRequest> = []
  private writing = new Map<string, ReturnType<typeof setInterval>>()
  constructor() {
    FakeWorker.last = this
  }
  send(data: WorkerResponse) {
    this.onmessage?.({ data })
  }
  postMessage(message: WorkerRequest) {
    this.posted.push(message)
    const send = (data: WorkerResponse) => this.onmessage?.({ data })
    if (message.type === "load") {
      setTimeout(() => send({ type: "ready", model: message.model }), 0)
    } else if (message.type === "generate") {
      if (FakeWorker.onGenerate) return FakeWorker.onGenerate(this, message)
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


const STATS: RunTimings = {
  inputTokens: 1,
  cachedTokens: 0,
  outputTokens: 1,
  firstTokenMs: 1,
  promptMs: 1,
  decodeMs: 1
}

test("structured output forwards real phases, ignoring synthetic thinking text and stale ids", { timeout: 10_000 }, async () => {
  const { webglText } = await import("./adapter.js")
  const { unloadModel } = await import("./client.js")
  unloadModel()
  const phases: GenerationPhase[] = []
  const adapter = webglText("onnx-community/LFM3-1.7B-ONNX", {
    onGenerationPhase: (phase) => phases.push(phase)
  })
  FakeWorker.onGenerate = (worker, request) => {
    assert.equal(request.reportPhases, true)
    assert.equal("onPhase" in request, false, "callbacks stay on the page")
    worker.send({ type: "phase", id: request.id, phase: "processing" })
    worker.send({ type: "token", id: request.id, text: "<think>" })
    assert.deepEqual(phases, ["processing"], "template text cannot mark prefill complete")
    worker.send({ type: "phase", id: "stale-generation", phase: "generating" })
    assert.deepEqual(phases, ["processing"])
    worker.send({ type: "phase", id: request.id, phase: "generating" })
    worker.send({ type: "token", id: request.id, text: 'considering</think>{"moveId":"b6-a5"}' })
    worker.send({ type: "done", id: request.id, stats: STATS })
  }
  try {
    const result = await adapter.structuredOutput({
      chatOptions: { logger: resolveDebugOption(false), threadId: "adapter-test", model: adapter.model, messages: [{ role: "user", content: "choose a move" }] },
      outputSchema: { type: "object", properties: { moveId: { type: "string" } } }
    } as Parameters<typeof adapter.structuredOutput>[0])
    assert.deepEqual(result.data, { moveId: "b6-a5" })
    assert.deepEqual(phases, ["processing", "generating"])
    const request = FakeWorker.last?.posted.find((message) => message.type === "generate")
    assert.ok(request && request.type === "generate")
    FakeWorker.last?.send({ type: "phase", id: request.id, phase: "processing" })
    assert.deepEqual(phases, ["processing", "generating"], "completed generations cannot update phase")
  } finally {
    FakeWorker.onGenerate = undefined
    unloadModel()
  }
})

test("structured output streams only thinking text across split tags", { timeout: 10_000 }, async () => {
  const { webglText } = await import("./adapter.js")
  const { unloadModel } = await import("./client.js")
  unloadModel()
  const thinking: string[] = []
  const adapter = webglText("onnx-community/LFM3-1.7B-ONNX", {
    onThinking: (delta) => thinking.push(delta)
  })
  FakeWorker.onGenerate = (worker, request) => {
    assert.equal(request.reportPhases, undefined, "thinking text needs no extra worker telemetry")
    for (const text of ['<th', 'ink>Analyze ', 'a capture.</th', 'ink>{"move', 'Id":"b6-a5"}']) {
      worker.send({ type: "token", id: request.id, text })
    }
    worker.send({ type: "done", id: request.id, stats: STATS })
  }
  try {
    const result = await adapter.structuredOutput({
      chatOptions: { logger: resolveDebugOption(false), threadId: "adapter-test", model: adapter.model, messages: [{ role: "user", content: "choose a move" }] },
      outputSchema: { type: "object" }
    } as Parameters<typeof adapter.structuredOutput>[0])
    assert.equal(thinking.join(""), "Analyze a capture.")
    assert.deepEqual(result.data, { moveId: "b6-a5" })
    assert.equal(result.rawText, '{"moveId":"b6-a5"}')
  } finally {
    FakeWorker.onGenerate = undefined
    unloadModel()
  }
})

test("aborted structured generations cannot publish later phases", { timeout: 10_000 }, async () => {
  const { webglText } = await import("./adapter.js")
  const { unloadModel } = await import("./client.js")
  unloadModel()
  const controller = new AbortController()
  const phases: GenerationPhase[] = []
  const thinking: string[] = []
  const adapter = webglText("onnx-community/LFM3-1.7B-ONNX", {
    onGenerationPhase: (phase) => phases.push(phase),
    onThinking: (delta) => thinking.push(delta)
  })
  FakeWorker.onGenerate = (worker, request) => {
    worker.send({ type: "phase", id: request.id, phase: "processing" })
    worker.send({ type: "token", id: request.id, text: "<think>before " })
    setTimeout(() => {
      controller.abort()
      worker.send({ type: "phase", id: request.id, phase: "generating" })
      worker.send({ type: "token", id: request.id, text: 'after</think>{"moveId":"b6-a5"}' })
      worker.send({ type: "done", id: request.id, stats: STATS })
    }, 0)
  }
  try {
    await adapter.structuredOutput({
      chatOptions: {
        logger: resolveDebugOption(false),
        threadId: "adapter-test",
        model: adapter.model,
        messages: [{ role: "user", content: "choose a move" }],
        abortController: controller
      },
      outputSchema: { type: "object" }
    } as Parameters<typeof adapter.structuredOutput>[0])
    assert.deepEqual(phases, ["processing"])
    assert.deepEqual(thinking, ["before "], "aborted turns cannot append thinking text")
    assert.ok(FakeWorker.last?.posted.some((message) => message.type === "stop"))
  } finally {
    FakeWorker.onGenerate = undefined
    unloadModel()
  }
})
