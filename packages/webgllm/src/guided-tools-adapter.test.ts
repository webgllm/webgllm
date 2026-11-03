import assert from "node:assert/strict"
import { test } from "node:test"
import { resolveDebugOption } from "@tanstack/ai/adapter-internals"
import type { TextOptions } from "@tanstack/ai"
import type { WebGlTextProviderOptions } from "./adapter.js"
import type { RunTimings, WorkerRequest, WorkerResponse } from "./protocol.js"

class ToolWorker {
  static last: ToolWorker
  static generate: (worker: ToolWorker, request: Extract<WorkerRequest, { type: "generate" }>) => void
  onmessage?: (event: { data: WorkerResponse }) => void
  requests: WorkerRequest[] = []
  constructor() { ToolWorker.last = this }
  postMessage(request: WorkerRequest) {
    this.requests.push(request)
    if (request.type === "load") setTimeout(() => this.send({ type: "ready", model: request.model }), 0)
    if (request.type === "generate") ToolWorker.generate(this, request)
  }
  send(data: WorkerResponse) { this.onmessage?.({ data }) }
  terminate() { this.onmessage = undefined }
}
Object.defineProperty(globalThis, "Worker", { configurable: true, writable: true, value: ToolWorker })
const MODEL = "onnx-community/LFM3-1.7B-ONNX"
const stats: RunTimings = { inputTokens: 10, cachedTokens: 0, outputTokens: 20, firstTokenMs: 1, promptMs: 1, decodeMs: 1 }
const tool = { name: "lookup", description: "Look up a value", inputSchema: { type: "object", properties: { query: { type: "string" } }, required: ["query"], additionalProperties: false } }
function options(): TextOptions<WebGlTextProviderOptions> {
  return { logger: resolveDebugOption(false), threadId: "guided-tool-test", model: MODEL, messages: [{ role: "user", content: "Look up checkers." }], tools: [tool], modelOptions: { maxTokens: 8192 }, reasoning: { level: "high", summary: true, budgetTokens: 512 } }
}
function reply(worker: ToolWorker, id: string, value: unknown) {
  worker.send({ type: "token", id, text: typeof value === "string" ? value : JSON.stringify(value) })
  worker.send({ type: "done", id, stats })
}

test("guided auto, required, and named tools emit exactly one coherent AGUI call", async () => {
  const { webglText } = await import("./adapter.js")
  const { unloadModel } = await import("./client.js")
  for (const toolChoice of ["auto", "required", { type: "tool", name: "lookup" }] as const) {
    unloadModel()
    let requests = 0
    ToolWorker.generate = (worker, request) => {
      requests++
      assert.equal(request.maxTokens, 8192)
      assert.equal(request.thinkingBudgetTokens, 512)
      const schema = request.outputSchema as { properties: Record<string, { enum?: string[] }> } | undefined
      if (!schema) {
        assert.equal(toolChoice, "auto")
        reply(worker, request.id, '<tool_call>{"name":"lookup","arguments":{"query":"checkers"}}</tool_call>')
      } else if (schema.properties.action) {
        if (toolChoice === "required") assert.deepEqual(schema.properties.action.enum, ["lookup"])
        reply(worker, request.id, { action: "lookup" })
      } else {
        assert.deepEqual(request.outputSchema, tool.inputSchema)
        reply(worker, request.id, { query: "checkers" })
      }
    }
    try {
      const chunks = []
      for await (const chunk of webglText(MODEL).chatStream({ ...options(), toolChoice })) chunks.push(chunk)
      assert.equal(requests, 1)
      assert.equal(chunks.filter(c => c.type === "RUN_STARTED").length, 1)
      const starts = chunks.filter(c => c.type === "TOOL_CALL_START")
      const args = chunks.filter(c => c.type === "TOOL_CALL_ARGS")
      const ends = chunks.filter(c => c.type === "TOOL_CALL_END")
      assert.equal(starts.length, 1); assert.equal(args.length, 1); assert.equal(ends.length, 1)
      assert.equal(starts[0].toolCallId, ends[0].toolCallId)
      assert.equal(args[0].toolCallId, starts[0].toolCallId)
      assert.equal(ends[0].toolCallName, "lookup")
      assert.deepEqual(ends[0].input, { query: "checkers" })
      const final = chunks.find(c => c.type === "RUN_FINISHED")
      assert.ok(final && final.type === "RUN_FINISHED")
      assert.equal(final.finishReason, "tool_calls")
      const usage = Array.isArray(final.usage) ? final.usage[0] : final.usage
      assert.equal(usage?.totalTokens, requests * 30)
      assert.ok(!chunks.some(c => c.type === "RUN_ERROR"))
    } finally { unloadModel() }
  }
})

test("auto can use an existing tool result to answer without another tool call", async () => {
  const { webglText } = await import("./adapter.js")
  const { unloadModel } = await import("./client.js")
  unloadModel(); let requests = 0
  ToolWorker.generate = (worker, request) => {
    requests++
    assert.ok(request.chat.some(turn => turn.role === "tool" && turn.content.includes("42")))
    reply(worker, request.id, request.outputSchema ? { action: "answer" } : "The answer is 42.")
  }
  try {
    const chunks = []
    for await (const chunk of webglText(MODEL).chatStream({ ...options(), messages: [{ role: "user", content: "Look up the value." }, { role: "tool", content: "42", toolCallId: "prior-call" }] })) chunks.push(chunk)
    assert.equal(requests, 1)
    assert.equal(chunks.filter(c => c.type === "RUN_STARTED").length, 1)
    assert.ok(chunks.some(c => c.type === "TEXT_MESSAGE_CONTENT" && c.delta === "The answer is 42."))
    assert.ok(!chunks.some(c => c.type === "TOOL_CALL_START"))
    const final = chunks.find(c => c.type === "RUN_FINISHED")
    const usage = final?.type === "RUN_FINISHED" && (Array.isArray(final.usage) ? final.usage[0] : final.usage)
    assert.equal(usage && usage.totalTokens, 30)
  } finally { unloadModel() }
})

test("toolChoice none bypasses guided routing", async () => {
  const { webglText } = await import("./adapter.js")
  const { unloadModel } = await import("./client.js")
  unloadModel(); let requests = 0
  ToolWorker.generate = (worker, request) => {
    requests++; assert.equal(request.outputSchema, undefined)
    assert.deepEqual(request.tools, [])
    reply(worker, request.id, "Hello")
  }
  try {
    for await (const chunk of webglText(MODEL).chatStream({ ...options(), toolChoice: "none" })) void chunk
    assert.equal(requests, 1)
  } finally { unloadModel() }
})

test("request-signal cancellation prevents guided generation before model loading", async () => {
  const { webglText } = await import("./adapter.js")
  const { unloadModel } = await import("./client.js")
  unloadModel(); const previous = ToolWorker.last; const controller = new AbortController(); controller.abort()
  const chunks = []
  for await (const chunk of webglText(MODEL).chatStream({ ...options(), request: new Request("http://localhost", { signal: controller.signal }) })) chunks.push(chunk)
  assert.equal(ToolWorker.last, previous)
  assert.ok(!chunks.some(c => c.type === "TOOL_CALL_START"))
})

test("a real tool named answer remains callable in auto mode", async () => {
  const { webglText } = await import("./adapter.js")
  const { unloadModel } = await import("./client.js")
  unloadModel(); let requests = 0
  ToolWorker.generate = (worker, request) => {
    requests++
    reply(worker, request.id, '<tool_call>{"name":"answer","arguments":{"query":"checkers"}}</tool_call>')
  }
  try {
    const chunks = []
    for await (const chunk of webglText(MODEL).chatStream({ ...options(), tools: [{ ...tool, name: "answer" }] })) chunks.push(chunk)
    const end = chunks.find(c => c.type === "TOOL_CALL_END")
    assert.ok(end && end.type === "TOOL_CALL_END")
    assert.equal(end.toolCallName, "answer")
    assert.equal(requests, 1)
  } finally { unloadModel() }
})


test("no declared tools means hallucinated native calls remain text", async () => {
  const { webglText } = await import("./adapter.js")
  const { unloadModel } = await import("./client.js")
  const { formatLfm2ToolCalls, parseToolCalls } = await import("./tool-calls.js")
  const markup = formatLfm2ToolCalls([{ name: "unavailable", args: {} }])
  assert.equal(parseToolCalls(markup).length, 1)
  for (const tools of [undefined, []]) {
    unloadModel()
    ToolWorker.generate = (worker, request) => {
      assert.deepEqual(request.tools, [])
      reply(worker, request.id, markup)
    }
    try {
      const chunks = []
      for await (const chunk of webglText(MODEL).chatStream({ ...options(), tools })) chunks.push(chunk)
      assert.ok(!chunks.some(chunk => chunk.type === "TOOL_CALL_START" || chunk.type === "TOOL_CALL_END"))
      assert.equal(chunks.filter(chunk => chunk.type === "TEXT_MESSAGE_CONTENT").map(chunk => chunk.delta).join(""), markup)
      const final = chunks.find(chunk => chunk.type === "RUN_FINISHED")
      assert.ok(final && final.type === "RUN_FINISHED")
      assert.equal(final.finishReason, "stop")
    } finally { unloadModel() }
  }
})

test("failed native generations never publish tool calls or trigger repair",async()=>{
  const {webglText}=await import("./adapter.js"),{unloadModel}=await import("./client.js")
  unloadModel();let requests=0
  ToolWorker.generate=(worker,request)=>{requests++;worker.send({type:"token",id:request.id,text:'<tool_call>{"name":"lookup","arguments":{"query":"checkers"}}</tool_call>'});worker.send({type:"error",id:request.id,message:"No token can complete the requested output schema."})}
  try{
    const chunks=[];for await(const chunk of webglText(MODEL).chatStream({...options(),toolChoice:"auto"}))chunks.push(chunk)
    assert.equal(requests,1)
    assert.equal(chunks.filter(c=>c.type==="RUN_ERROR").length,1)
    assert.ok(!chunks.some(c=>c.type==="TOOL_CALL_START"||c.type==="TOOL_CALL_END"||c.type==="RUN_FINISHED"))
  }finally{unloadModel()}
})
