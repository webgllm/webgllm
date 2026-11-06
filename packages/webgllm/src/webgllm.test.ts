import assert from "node:assert/strict"
import { test } from "node:test"
import { WebGLLMError } from "./errors.js"
import { fakeStorage, fakeWorkers, settle } from "./fake-worker.test.js"
import type { FakeOptions } from "./fake-worker.test.js"
import type { ChatCompletionChunk, LoadProgress } from "./openai.js"
import { internals, WebGLLM } from "./webgllm.js"

const QWEN = "unsloth/Qwen3.5-2B-GGUF"
const LFM2 = "LiquidAI/LFM2.5-350M-GGUF"
const hi = [{ role: "user" as const, content: "hi" }]

function llm(options: FakeOptions = {}, downloaded = new Set<string>()) {
  const workers = fakeWorkers(options)
  const storage = fakeStorage(downloaded)
  internals.storage = storage
  const progress: Array<LoadProgress> = []
  const instance = new WebGLLM({ worker: workers.create, onProgress: (p) => progress.push(p), memoryBudget: 1000 })
  return { llm: instance, workers, progress, storage }
}

async function collect(stream: AsyncIterable<ChatCompletionChunk>) {
  const chunks: Array<ChatCompletionChunk> = []
  for await (const chunk of stream) chunks.push(chunk)
  return chunks
}

test("a completion loads its model, then answers with thinking split out", async () => {
  const { llm: client, progress, workers } = llm({ reply: () => ({ tokens: ["<think>", "Greet.", "</think>", "\n\nHello", "!"] }) })
  const mine: Array<LoadProgress> = []
  const completion = await client.chat.completions.create(
    { model: QWEN, messages: hi, reasoning_effort: "high" },
    { onProgress: (p) => mine.push(p) }
  )
  assert.equal(completion.object, "chat.completion")
  assert.equal(completion.model, QWEN)
  assert.match(completion.id, /^chatcmpl-/)
  assert.deepEqual(completion.choices[0]?.message, { role: "assistant", content: "Hello!", reasoning_content: "Greet.", refusal: null })
  assert.equal(completion.choices[0]?.finish_reason, "stop")
  assert.equal(completion.usage.total_tokens, 13)
  assert.equal(mine.length, 1)
  assert.equal(progress.length, 1)
  const generate = workers.of(QWEN)!.generations()[0]!
  assert.equal(generate.thinking, true)
  // Loaded already: no progress for the next call.
  await client.chat.completions.create({ model: QWEN, messages: hi }, { onProgress: (p) => mine.push(p) })
  assert.equal(mine.length, 1)
})

test("streamed chunks: role first, content, tool calls, finish, usage", async () => {
  const raw = ["Let me check.", "<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n</function>\n</tool_call>"]
  const { llm: client } = llm({ reply: () => ({ tokens: raw }) })
  const stream = await client.chat.completions.create({
    model: QWEN,
    messages: hi,
    stream: true,
    stream_options: { include_usage: true },
    tools: [{ type: "function", function: { name: "get_weather", parameters: { type: "object" } } }]
  })
  const chunks = await collect(stream)
  assert.deepEqual(chunks[0]?.choices[0]?.delta, { role: "assistant", content: "Let me check." })
  const call = chunks[1]?.choices[0]?.delta.tool_calls?.[0]
  assert.equal(call?.index, 0)
  assert.match(call?.id ?? "", /^call_/)
  assert.deepEqual(call?.function, { name: "get_weather", arguments: '{"city":"Paris"}' })
  assert.equal(chunks[2]?.choices[0]?.finish_reason, "tool_calls")
  assert.equal(chunks[2]?.timings, undefined)
  assert.deepEqual(chunks[3]?.choices, [])
  assert.equal(chunks[3]?.usage?.completion_tokens, 3)
  assert.equal(chunks[3]?.timings?.decode_ms, 20)
  assert.equal(chunks.length, 4)
  assert.ok(chunks.every((c) => c.object === "chat.completion.chunk" && c.id === chunks[0]?.id))
})

test("without include_usage the timings ride on the finish chunk", async () => {
  const { llm: client } = llm({ reply: () => ({ tokens: ["Hi"], done: { finishReason: "length" } }) })
  const chunks = await collect(await client.chat.completions.create({ model: LFM2, messages: hi, stream: true }))
  const last = chunks.at(-1)!
  assert.equal(last.choices[0]?.finish_reason, "length")
  assert.equal(last.usage, undefined)
  assert.equal(last.timings?.prompt_ms, 4)
})

test("a forced tool call comes back as tool_calls", async () => {
  const { llm: client, workers } = llm({ reply: () => ({ tokens: [], done: { toolCall: { name: "pick", arguments: { n: 1 } } } }) })
  const completion = await client.chat.completions.create({
    model: LFM2,
    messages: hi,
    tools: [{ type: "function", function: { name: "pick" } }],
    tool_choice: "required"
  })
  assert.equal(completion.choices[0]?.finish_reason, "tool_calls")
  assert.equal(completion.choices[0]?.message.content, null)
  assert.equal(completion.choices[0]?.message.tool_calls?.[0]?.function.arguments, '{"n":1}')
  assert.equal(workers.of(LFM2)!.generations()[0]!.toolChoice, "required")
})

test("aborting a stream rejects with AbortError and stops the worker", async () => {
  const { llm: client, workers } = llm()
  const abort = new AbortController()
  const stream = await client.chat.completions.create({ model: LFM2, messages: hi, stream: true }, { signal: abort.signal })
  await settle()
  const worker = workers.of(LFM2)!
  const id = worker.generations()[0]!.id
  worker.emit({ type: "token", id, text: "Hel" })
  const iterator = stream[Symbol.asyncIterator]()
  assert.equal((await iterator.next()).value?.choices[0]?.delta.content, "Hel")
  const pending = iterator.next()
  abort.abort()
  await assert.rejects(pending, { name: "AbortError" })
  assert.ok(worker.sent.some((m) => m.type === "stop" && m.id === id))
})

test("leaving a stream early stops the worker", async () => {
  const { llm: client, workers } = llm()
  const stream = await client.chat.completions.create({ model: LFM2, messages: hi, stream: true })
  await settle()
  const worker = workers.of(LFM2)!
  const id = worker.generations()[0]!.id
  worker.emit({ type: "token", id, text: "a" })
  for await (const chunk of stream) {
    assert.equal(chunk.choices[0]?.delta.content, "a")
    break
  }
  assert.ok(worker.sent.some((m) => m.type === "stop" && m.id === id))
})

test("aborting a whole completion rejects at once", async () => {
  const { llm: client } = llm()
  const abort = new AbortController()
  const pending = client.chat.completions.create({ model: LFM2, messages: hi }, { signal: abort.signal })
  await settle()
  abort.abort()
  await assert.rejects(pending, { name: "AbortError" })
})

test("worker errors keep their codes", async () => {
  const { llm: client, workers } = llm()
  const pending = client.chat.completions.create({ model: LFM2, messages: hi })
  await settle()
  const worker = workers.of(LFM2)!
  worker.emit({ type: "error", id: worker.generations()[0]!.id, message: "too long", code: "context-length-exceeded" })
  await assert.rejects(pending, (error: unknown) => error instanceof WebGLLMError && error.code === "context-length-exceeded")
})

test("bad requests reject before anything loads", async () => {
  const { llm: client, workers } = llm()
  await assert.rejects(client.chat.completions.create({ model: "gpt-4o", messages: hi }), { code: "model-not-supported" })
  await assert.rejects(client.chat.completions.create({ model: LFM2, messages: hi, n: 3 }), { code: "unsupported-parameter" })
  await assert.rejects(client.loadModel("nope"), { code: "model-not-supported" })
  assert.equal(workers.workers.length, 0)
})

test("models: the catalog with what's downloaded and loaded", async () => {
  const { llm: client } = llm({}, new Set([LFM2]))
  await client.loadModel(QWEN, { contextLength: 4096 })
  const models = await client.models.list()
  assert.ok(models.length >= 10)
  const qwen = models.find((m) => m.id === QWEN)!
  assert.deepEqual(qwen, {
    id: QWEN,
    format: "gguf",
    contextLength: 32768,
    size: 1234,
    downloaded: true,
    loaded: true,
    loadedContextLength: 4096,
    supports: { tools: true, thinking: true }
  })
  const lfm2 = await client.models.retrieve(LFM2)
  assert.equal(lfm2.downloaded, true)
  assert.equal(lfm2.loaded, false)
  assert.equal("loadedContextLength" in lfm2, false)
  assert.equal(lfm2.supports.thinking, false)
})

test("deleting a model unloads it and removes its files", async () => {
  const { llm: client, storage } = llm({}, new Set([QWEN]))
  await client.loadModel(QWEN)
  await client.models.delete(QWEN)
  assert.equal((await client.models.retrieve(QWEN)).loaded, false)
  assert.equal(storage.removed[0]?.[0], QWEN)
})

test("progress listeners can be added and removed", async () => {
  const { llm: client } = llm()
  const seen: Array<LoadProgress> = []
  const listener = (p: LoadProgress) => seen.push(p)
  client.addEventListener("progress", listener)
  await client.loadModel(LFM2)
  client.removeEventListener("progress", listener)
  await client.loadModel(QWEN)
  assert.deepEqual(seen.map((p) => p.model), [LFM2])
})

test("unloadModel fails a running completion with model-not-loaded", async () => {
  const { llm: client } = llm()
  const pending = client.chat.completions.create({ model: LFM2, messages: hi })
  await settle()
  await client.unloadModel(LFM2)
  await assert.rejects(pending, { code: "model-not-loaded" })
})
