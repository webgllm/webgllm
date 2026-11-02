import assert from "node:assert/strict"
import { test } from "node:test"
import { resolveDebugOption } from "@tanstack/ai/adapter-internals"
import type { WorkerRequest, WorkerResponse } from "./protocol.js"

class SchemaWorker {
  static requests: WorkerRequest[] = []
  static failure: string | undefined
  onmessage: ((event: { data: WorkerResponse }) => void) | null = null
  postMessage(message: WorkerRequest) {
    SchemaWorker.requests.push(message)
    if (message.type === "load") this.onmessage?.({ data: { type: "ready", model: message.model } })
    if (message.type === "generate") {
      if (SchemaWorker.failure) this.onmessage?.({ data: { type: "error", id: message.id, message: SchemaWorker.failure } })
      else {
        this.onmessage?.({ data: { type: "token", id: message.id, text: '{"moveId":"b6-a5"}' } })
        this.onmessage?.({ data: { type: "done", id: message.id, stats: { inputTokens: 10, cachedTokens: 0, outputTokens: 8, firstTokenMs: 1, promptMs: 1, decodeMs: 1 } } })
      }
    }
  }
  terminate() { this.onmessage = null }
}
;(globalThis as { Worker?: unknown }).Worker = SchemaWorker

test("structured output forwards the actual schema and limits to worker generation", async () => {
  const { webglText } = await import("./adapter.js")
  const { unloadModel } = await import("./client.js")
  unloadModel()
  const adapter = webglText("LiquidAI/LFM3-1.7B-GGUF")
  const schema = { type: "object", properties: { moveId: { type: "string", enum: ["b6-a5"] } }, required: ["moveId"], additionalProperties: false }
  const chatOptions = { model: adapter.model, threadId: "schema-forwarding", logger: resolveDebugOption(false), messages: [{ role: "user" as const, content: "Choose" }], modelOptions: { maxTokens: 8192 }, reasoning: { level: "high" as const, summary: true, budgetTokens: 512 } }
  const result = await adapter.structuredOutput({ chatOptions, outputSchema: schema })
  assert.deepEqual(result.data, { moveId: "b6-a5" })
  const request = SchemaWorker.requests.find(r => r.type === "generate")
  assert.ok(request && request.type === "generate")
  assert.deepEqual(request.outputSchema, schema)
  assert.equal(request.maxTokens, 8192)
  assert.equal(request.thinkingBudgetTokens, 512)
  SchemaWorker.failure = "Structured output was truncated before the JSON Schema was satisfied."
  await assert.rejects(adapter.structuredOutput({ chatOptions, outputSchema: schema }), /truncated/)
  unloadModel()
})
