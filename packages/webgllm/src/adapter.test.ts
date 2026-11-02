import { chat } from "@tanstack/ai"
import type { AdapterReasoning, ReasoningRequest, TextOptions } from "@tanstack/ai"
import type { WebGlTextAdapter, WebGlTextProviderOptions } from "./adapter.js"
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
    logger: resolveDebugOption(false),
    threadId: "adapter-test",
    model: "onnx-community/LFM3-1.7B-ONNX",
    messages: [{ role: "user", content: "hi" }]
  })
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
    })
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
    })
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
    })
    assert.deepEqual(phases, ["processing"])
    assert.deepEqual(thinking, ["before "], "aborted turns cannot append thinking text")
    assert.ok(FakeWorker.last?.posted.some((message) => message.type === "stop"))
  } finally {
    FakeWorker.onGenerate = undefined
    unloadModel()
  }
})


const LFM3_GGUF = "LiquidAI/LFM3-1.7B-GGUF" as const
const LFM3_ONNX = "onnx-community/LFM3-1.7B-ONNX" as const

type Assert<T extends true> = T
export type GgufBudgetCapability = Assert<AdapterReasoning<WebGlTextAdapter<typeof LFM3_GGUF>>["budget"]>
export type OnnxBudgetCapability = Assert<AdapterReasoning<WebGlTextAdapter<typeof LFM3_ONNX>>["budget"]>

function budgetOptions(reasoning?: ReasoningRequest): TextOptions<WebGlTextProviderOptions> {
  return {
    logger: resolveDebugOption(false),
    threadId: "adapter-thinking-budget-test",
    model: LFM3_GGUF,
    messages: [{ role: "user", content: "Choose a move." }],
    ...(reasoning && { reasoning }),
    modelOptions: { max_new_tokens: 1024 }
  }
}

async function collect(stream: AsyncIterable<unknown>) {
  const chunks: unknown[] = []
  for await (const chunk of stream) chunks.push(chunk)
  return chunks
}

test("LFM3 GGUF forwards an explicit thinking budget through shared chat and structured output", { timeout: 10_000 }, async () => {
  const { webglText } = await import("./adapter.js")
  const { unloadModel } = await import("./client.js")
  const expectedStats: RunTimings = {
    ...STATS,
    outputTokens: 270,
    thinkingTokens: 256,
    thinkingBudgetTokens: 256,
    thinkingBudgetExhausted: true
  }
  const requests: Array<Extract<WorkerRequest, { type: "generate" }>> = []
  FakeWorker.onGenerate = (worker, request) => {
    requests.push(request)
    worker.send({ type: "token", id: request.id, text: '<think>Plan.</think>{"moveId":"b6-a5"}' })
    worker.send({ type: "done", id: request.id, stats: expectedStats })
  }
  try {
    unloadModel()
    const adapter = webglText(LFM3_GGUF)
    const result = await chat({
      adapter,
      threadId: "shared-budget-test",
      messages: [{ role: "user", content: "Choose a move." }],
      reasoning: { level: "high", budgetTokens: 256 },
      modelOptions: { max_new_tokens: 1024 },
      stream: false
    })
    const reportedUsage = result.chunks.find(chunk => chunk.type === "RUN_FINISHED")?.usage
    const usage = Array.isArray(reportedUsage) ? reportedUsage[0] : reportedUsage
    assert.ok(usage && "providerUsageDetails" in usage)
    assert.equal(usage.providerUsageDetails?.thinkingTokens, 256)
    assert.equal(usage.providerUsageDetails?.thinkingBudgetTokens, 256)
    assert.equal(usage.providerUsageDetails?.thinkingBudgetExhausted, true)
    const structured = await adapter.structuredOutput({
      chatOptions: budgetOptions({ level: "high", summary: true, budgetTokens: 256 }),
      outputSchema: { type: "object" }
    })
    assert.deepEqual(structured.data, { moveId: "b6-a5" })
    assert.equal(structured.usage?.providerUsageDetails?.thinkingTokens, 256)
    assert.equal(structured.usage?.providerUsageDetails?.thinkingBudgetTokens, 256)
    assert.equal(structured.usage?.providerUsageDetails?.thinkingBudgetExhausted, true)
    assert.equal(requests.length, 2)
    for (const request of requests) {
      assert.equal(request.thinking, true)
      assert.equal(request.thinkingBudgetTokens, 256)
      assert.equal(request.maxTokens, 1024, "max_new_tokens remains the total output limit")
    }
  } finally {
    FakeWorker.onGenerate = undefined
    unloadModel()
  }
})

test("LFM3 reasoning shorthand and omitted budgets remain unlimited", { timeout: 10_000 }, async () => {
  const { webglText } = await import("./adapter.js")
  const { unloadModel } = await import("./client.js")
  const requests: Array<Extract<WorkerRequest, { type: "generate" }>> = []
  FakeWorker.onGenerate = (worker, request) => {
    requests.push(request)
    worker.send({ type: "token", id: request.id, text: '{"moveId":"b6-a5"}' })
    worker.send({ type: "done", id: request.id, stats: STATS })
  }
  try {
    unloadModel()
    const adapter = webglText(LFM3_GGUF)
    await chat({ adapter, threadId: "shorthand-budget-test", messages: [{ role: "user", content: "Choose a move." }], reasoning: "high", stream: false })
    for (const reasoning of [undefined, { level: "high", summary: true }, { level: "off", summary: true } ] satisfies Array<ReasoningRequest | undefined>) {
      await collect(adapter.chatStream(budgetOptions(reasoning)))
      await adapter.structuredOutput({ chatOptions: budgetOptions(reasoning), outputSchema: { type: "object" } })
    }
    assert.equal(requests[0].thinking, true)
    assert.ok(requests.every(request => request.thinkingBudgetTokens === undefined), "no 16k default may be introduced")
    assert.deepEqual(requests.slice(1).map(request => request.thinking), [undefined, undefined, true, true, false, false])
  } finally {
    FakeWorker.onGenerate = undefined
    unloadModel()
  }
})

test("LFM3 rejects invalid thinking budgets before starting a worker", { timeout: 10_000 }, async () => {
  const { webglText } = await import("./adapter.js")
  const { unloadModel } = await import("./client.js")
  unloadModel()
  const previousWorker = FakeWorker.last
  const adapter = webglText(LFM3_GGUF)
  for (const value of [-1, 1.5, NaN, Infinity, Number.MAX_SAFE_INTEGER + 1, "256"]) {
    const reasoning = { level: "high", summary: true, budgetTokens: value } as unknown as ReasoningRequest
    await assert.rejects(collect(adapter.chatStream(budgetOptions(reasoning))), /budgetTokens.*non-negative safe integer/)
    await assert.rejects(adapter.structuredOutput({ chatOptions: budgetOptions(reasoning), outputSchema: { type: "object" } }), /budgetTokens.*non-negative safe integer/)
    assert.equal(FakeWorker.last, previousWorker)
  }
})

test("every thinking-capable WebGL model advertises budget support", async () => {
  const { WEBGL_MODEL_META } = await import("./model-meta.js")
  for (const [model, meta] of Object.entries(WEBGL_MODEL_META)) {
    if (meta.supports.capabilities.includes("thinking")) {
      assert.ok(meta.reasoning && "budget" in meta.reasoning, model)
      assert.equal(meta.reasoning.budget, true, model)
    } else assert.equal(meta.reasoning, undefined, model)
  }
})

test("LFM3 forwards zero budgets and ignores a valid cap when thinking is off", { timeout: 10_000 }, async () => {
  const { webglText } = await import("./adapter.js")
  const { unloadModel } = await import("./client.js")
  const requests: Array<Extract<WorkerRequest, { type: "generate" }>> = []
  FakeWorker.onGenerate = (worker, request) => {
    requests.push(request)
    worker.send({ type: "token", id: request.id, text: '{"moveId":"b6-a5"}' })
    worker.send({ type: "done", id: request.id, stats: STATS })
  }
  try {
    unloadModel()
    const adapter = webglText(LFM3_GGUF)
    for (const reasoning of [
      { level: "high", summary: true, budgetTokens: 0 },
      { level: "off", summary: true, budgetTokens: 256 }
    ] satisfies ReasoningRequest[]) {
      await collect(adapter.chatStream(budgetOptions(reasoning)))
      await adapter.structuredOutput({ chatOptions: budgetOptions(reasoning), outputSchema: { type: "object" } })
    }
    assert.deepEqual(requests.map(request => [request.thinking, request.thinkingBudgetTokens]), [
      [true, 0], [true, 0], [false, undefined], [false, undefined]
    ])
    await assert.rejects(adapter.structuredOutput({ chatOptions: budgetOptions({ level: "off", summary: true, budgetTokens: -1 }), outputSchema: { type: "object" } }), /non-negative safe integer/)
  } finally {
    FakeWorker.onGenerate = undefined
    unloadModel()
  }
})

test("all thinking models forward the budget and caller output cap in both adapter paths", { timeout: 10_000 }, async () => {
  const { webglText } = await import("./adapter.js")
  const { WEBGL_MODEL_META, WEBGL_TEXT_MODELS } = await import("./model-meta.js")
  const { unloadModel } = await import("./client.js")
  const requests: Array<Extract<WorkerRequest, { type: "generate" }>> = []
  FakeWorker.onGenerate = (worker, request) => {
    requests.push(request)
    worker.send({ type: "token", id: request.id, text: '{"moveId":"b6-a5"}' })
    worker.send({ type: "done", id: request.id, stats: STATS })
  }
  try {
    for (const model of WEBGL_TEXT_MODELS) {
      unloadModel()
      const adapter = webglText(model)
      const reasons = WEBGL_MODEL_META[model].reasoning
      const options = {
        ...budgetOptions(reasons ? { level: "high", summary: true, budgetTokens: 512 } : undefined),
        model, modelOptions: { maxTokens: 8192 }
      }
      await collect(adapter.chatStream(options))
      await adapter.structuredOutput({ chatOptions: options, outputSchema: { type: "object" } })
      for (const request of requests.slice(-2)) {
        assert.equal(request.thinkingBudgetTokens, reasons ? 512 : undefined, model)
        assert.equal(request.maxTokens, 8192, model)
        assert.equal(request.contextLength, undefined, "caller did not override the model context")
      }
    }
    unloadModel()
    const adapter = webglText(LFM3_GGUF)
    const answer = await chat({ adapter, threadId: "shared-max-tokens", messages: [{ role: "user", content: "Choose a move." }], reasoning: { level: "high", budgetTokens: 512 }, modelOptions: { maxTokens: 8192, max_new_tokens: 64 }, stream: false })
    assert.ok(answer)
    assert.equal(requests.at(-1)?.maxTokens, 8192, "canonical maxTokens wins over the legacy alias")
  } finally {
    FakeWorker.onGenerate = undefined
    unloadModel()
  }
})


test("every WebGL model uses its recommended text sampling in both adapter paths", { timeout: 10_000 }, async () => {
  const { webglText } = await import("./adapter.js")
  const { WEBGL_TEXT_MODELS } = await import("./model-meta.js")
  const { unloadModel } = await import("./client.js")
  const requests: Array<Extract<WorkerRequest, { type: "generate" }>> = []
  FakeWorker.onGenerate = (worker, request) => {
    requests.push(request)
    worker.send({ type: "token", id: request.id, text: '{"ok":true}' })
    worker.send({ type: "done", id: request.id, stats: STATS })
  }
  try {
    for (const model of WEBGL_TEXT_MODELS) {
      const qwen = model.includes("Qwen3.5")
      const modes = qwen ? [undefined, false, true] : [undefined]
      for (const thinking of modes) {
        unloadModel()
        const options: TextOptions<WebGlTextProviderOptions> = {
          logger: resolveDebugOption(false), threadId: "model-defaults", model,
          messages: [{ role: "user", content: "Hello" }],
          ...(thinking !== undefined && { reasoning: { level: thinking ? "high" : "off", summary: true } })
        }
        const adapter = webglText(model)
        await collect(adapter.chatStream(options))
        await adapter.structuredOutput({ chatOptions: options, outputSchema: { type: "object" } })
        const expected = qwen
          ? { temperature: 1, top_k: 20, top_p: thinking ? 0.95 : 1, min_p: 0, presence_penalty: thinking ? 1.5 : 2, repetition_penalty: 1 }
          : { temperature: model.includes("Thinking") ? 0.05 : 0.1, top_k: 50, repetition_penalty: model.includes("LFM3") ? 1.1 : 1.05 }
        for (const request of requests.slice(-2)) {
          assert.deepEqual(request.sampling, expected, `${model}, thinking=${thinking}`)
          assert.equal(request.maxTokens, undefined, "no caller output cap is introduced")
        }
      }
    }
    unloadModel()
    const adapter = webglText("unsloth/Qwen3.5-2B-GGUF")
    await collect(adapter.chatStream({
      logger: resolveDebugOption(false), threadId: "sampling-overrides", model: adapter.model,
      messages: [{ role: "user", content: "Hello" }],
      reasoning: { level: "high", summary: true },
      modelOptions: { temperature: 0, top_k: 1, top_p: 0.8, min_p: 0.1, presence_penalty: 0, repetition_penalty: 1.2 }
    }))
    assert.deepEqual(requests.at(-1)?.sampling, { temperature: 0, top_k: 1, top_p: 0.8, min_p: 0.1, presence_penalty: 0, repetition_penalty: 1.2 })
  } finally {
    FakeWorker.onGenerate = undefined
    unloadModel()
  }
})


test("invalid output caps reject before model loading", { timeout: 10_000 }, async () => {
  const { webglText } = await import("./adapter.js")
  const { unloadModel } = await import("./client.js")
  unloadModel()
  const previous = FakeWorker.last
  const adapter = webglText(LFM3_GGUF)
  for (const value of [-1, 1.5, NaN, Infinity]) {
    const options = { ...budgetOptions(), modelOptions: { maxTokens: value } }
    await assert.rejects(collect(adapter.chatStream(options)), /nonnegative safe integer/)
    await assert.rejects(adapter.structuredOutput({ chatOptions: options, outputSchema: { type: "object" } }), /nonnegative safe integer/)
    assert.equal(FakeWorker.last, previous)
  }
})
