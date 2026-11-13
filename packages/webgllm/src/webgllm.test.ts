import assert from "node:assert/strict"
import { test } from "node:test"
import { WebGLLMError } from "./errors.js"
import { FakeWorker, fakeStorage, fakeWorkers, settle, STATS } from "./fake-worker.test.js"
import type { FakeOptions } from "./fake-worker.test.js"
import type { ChatCompletionChunk, LoadProgress } from "./openai.js"
import { internals, WebGLLM } from "./webgllm.js"
import { fileKey, storedModelFiles } from "./model-files.js"
import { modelMeta, rememberModel } from "./model-meta.js"
import { hub } from "./resolve-model.js"
import hubFixture from "./test-fixtures/hub.json"

const QWEN = "unsloth/Qwen3.5-2B-GGUF"
const LFM2 = "LiquidAI/LFM2.5-350M-GGUF"
const hi = [{ role: "user" as const, content: "hi" }]

// (Node has no WebGL 2: the fake workers stand in for the engine.)
internals.hasWebGl2 = () => true

function llm(options: FakeOptions = {}, downloaded = new Set<string>()) {
  const workers = fakeWorkers(options)
  const storage = fakeStorage(downloaded)
  internals.storage = storage
  internals.createWorker = workers.create
  const progress: Array<LoadProgress> = []
  const instance = new WebGLLM({ onProgress: (p) => progress.push(p), memoryBudget: 1000 })
  return { llm: instance, workers, progress, storage }
}

async function collect(stream: AsyncIterable<ChatCompletionChunk>) {
  const chunks: Array<ChatCompletionChunk> = []
  for await (const chunk of stream) chunks.push(chunk)
  return chunks
}

test("the instance has a baseURL and chat.completions.create, as an OpenAI client does", () => {
  const { llm: instance } = llm()
  assert.equal(instance.baseURL, "webgllm:")
  assert.equal(typeof instance.chat.completions.create, "function")
})

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

test("reads made before any chunk arrives each get a chunk, in order", async () => {
  const { llm: client, workers } = llm()
  const stream = await client.chat.completions.create({ model: LFM2, messages: hi, stream: true })
  await settle()
  const worker = workers.of(LFM2)!
  const id = worker.generations()[0]!.id
  const iterator = stream[Symbol.asyncIterator]()
  const reads = [iterator.next(), iterator.next(), iterator.next()]
  worker.emit({ type: "token", id, text: "a" })
  worker.finish(id)
  const timeout = new Promise<"hung">((resolve) => setTimeout(() => resolve("hung"), 200).unref())
  const results = await Promise.race([Promise.all(reads), timeout])
  assert.notEqual(results, "hung")
  const [first, second, third] = results as Array<IteratorResult<ChatCompletionChunk>>
  assert.equal(first.value?.choices[0]?.delta.content, "a")
  assert.equal(second.value?.choices[0]?.finish_reason, "stop")
  assert.equal(third.done, true)
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

test("leaving a stream ends a read still waiting, and later messages are dropped", async () => {
  const { llm: client, workers } = llm()
  const stream = await client.chat.completions.create({ model: LFM2, messages: hi, stream: true })
  await settle()
  const worker = workers.of(LFM2)!
  const id = worker.generations()[0]!.id
  const iterator = stream[Symbol.asyncIterator]()
  const pending = iterator.next()
  assert.deepEqual(await iterator.return!(), { value: undefined, done: true })
  worker.emit({ type: "token", id, text: "late" })
  worker.finish(id)
  assert.deepEqual(await pending, { value: undefined, done: true })
  assert.deepEqual(await iterator.next(), { value: undefined, done: true })
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
    // (Its repository has a vision projector webgllm runs.)
    supports: { tools: true, thinking: true, images: true }
  })
  // The ONNX build has a vision encoder; a text model has neither.
  assert.equal((await client.models.retrieve("onnx-community/Qwen3.5-2B-ONNX")).supports.images, true)
  assert.equal((await client.models.retrieve("LiquidAI/LFM3-1.7B-GGUF")).supports.images, false)
  const lfm2 = await client.models.retrieve(LFM2)
  assert.equal(lfm2.downloaded, true)
  assert.equal(lfm2.loaded, false)
  assert.equal("loadedContextLength" in lfm2, false)
  assert.equal(lfm2.supports.thinking, false)
})

test("models.downloaded: stored or loaded, asking storage nothing else", async () => {
  const { llm: client, storage } = llm({}, new Set([LFM2]))
  let other = 0
  storage.size = async () => (other++, 1234)
  storage.contextLength = async () => (other++, 32768)
  assert.equal(await client.models.downloaded(LFM2), true)
  assert.equal(await client.models.downloaded(QWEN), false)
  await client.loadModel(QWEN)
  assert.equal(await client.models.downloaded(QWEN), true)
  await assert.rejects(client.models.downloaded("gpt-4o"), { code: "model-not-supported" })
  assert.equal(other, 0, "no size or context length lookups")
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

const weatherTool = {
  type: "function" as const,
  function: { name: "get_weather", parameters: { type: "object", properties: { city: { type: "string" } }, required: ["city"] } }
}
const qwenCall = (body: string) => `<tool_call>\n<function=get_weather>\n${body}\n</function>\n</tool_call>`

test("tool-call repair: arguments that fail the schema are generated again for that tool", async () => {
  const { llm: client, workers } = llm({
    reply: (request) =>
      request.toolChoice
        ? { tokens: [], done: { toolCall: { name: "get_weather", arguments: { city: "Paris" } } } }
        : { tokens: [qwenCall("<parameter=town>\nParis\n</parameter>")] }
  })
  const completion = await client.chat.completions.create({ model: QWEN, messages: hi, tools: [weatherTool] })
  const [first, repair] = workers.of(QWEN)!.generations()
  assert.equal(first!.toolChoice, undefined)
  assert.deepEqual(repair!.toolChoice, { name: "get_weather" })
  // The prompt so far stays; the failed call and a note follow; thinking closes at once.
  assert.deepEqual(repair!.chat.slice(0, first!.chat.length), first!.chat)
  assert.equal(repair!.chat.length, first!.chat.length + 2)
  assert.equal(repair!.thinkingBudgetTokens, 0)
  const call = completion.choices[0]?.message.tool_calls?.[0]
  assert.equal(call?.function.arguments, '{"city":"Paris"}')
  assert.equal(completion.choices[0]?.finish_reason, "tool_calls")
  // Both generations count.
  assert.equal(completion.usage.completion_tokens, 2 * STATS.outputTokens)
  assert.equal(completion.usage.prompt_tokens, 2 * STATS.inputTokens)
})

test("tool-call repair: unknown tools and unparsed markup are forced to any tool; off returns them as written", async () => {
  const forced = (request: { toolChoice?: unknown }) =>
    request.toolChoice ? { tokens: [], done: { toolCall: { name: "get_weather", arguments: { city: "Rome" } } } } : undefined
  const unknown = llm({ reply: (r) => forced(r) ?? { tokens: ["<tool_call>\n<function=get_time>\n<parameter=zone>\nUTC\n</parameter>\n</function>\n</tool_call>"] } })
  const a = await unknown.llm.chat.completions.create({ model: QWEN, messages: hi, tools: [weatherTool] })
  assert.equal(unknown.workers.of(QWEN)!.generations()[1]!.toolChoice, "required")
  assert.equal(a.choices[0]?.message.tool_calls?.[0]?.function.name, "get_weather")

  const broken = llm({ reply: (r) => forced(r) ?? { tokens: ["<tool_call>\nnot a call\n</tool_call>"] } })
  const b = await broken.llm.chat.completions.create({ model: QWEN, messages: hi, tools: [weatherTool] })
  assert.equal(b.choices[0]?.message.tool_calls?.length, 1)
  assert.equal(b.choices[0]?.message.content, null)

  const off = llm({ reply: (r) => forced(r) ?? { tokens: [qwenCall("<parameter=town>\nParis\n</parameter>")] } })
  const c = await off.llm.chat.completions.create({ model: QWEN, messages: hi, tools: [weatherTool], repair_tool_calls: false })
  assert.equal(off.workers.of(QWEN)!.generations().length, 1)
  assert.equal(c.choices[0]?.message.tool_calls?.[0]?.function.arguments, '{"town":"Paris"}')
})

test("tool-call repair runs before a stream's tool calls, and valid calls aren't repeated", async () => {
  const { llm: client, workers } = llm({
    reply: (request) =>
      request.toolChoice
        ? { tokens: [], done: { toolCall: { name: "get_weather", arguments: { city: "Oslo" } } } }
        : { tokens: ["Checking.", qwenCall("<parameter=city>\nOslo\n</parameter>")] }
  })
  const chunks = await collect(
    await client.chat.completions.create({ model: QWEN, messages: hi, tools: [weatherTool], stream: true, stream_options: { include_usage: true } })
  )
  assert.equal(workers.of(QWEN)!.generations().length, 1)
  const call = chunks.flatMap((c) => c.choices[0]?.delta.tool_calls ?? [])[0]
  assert.equal(call?.function?.arguments, '{"city":"Oslo"}')
})

function recordEvents(client: WebGLLM) {
  const events: Array<string> = []
  client.addEventListener("loadstart", (e) => events.push(`loadstart ${e.model}`))
  client.addEventListener("load", (e) => events.push(`load ${e.model}`))
  client.addEventListener("error", (e) => events.push(`error ${e.model}: ${e.error.message}`))
  client.addEventListener("unload", (e) => events.push(`unload ${e.model} ${e.reason}`))
  return events
}

test("lifecycle events: loadModel, an on-demand load, and unloadModel", async () => {
  const { llm: client } = llm({ reply: () => ({ tokens: ["hi"] }) })
  const events = recordEvents(client)
  await client.loadModel(LFM2)
  assert.deepEqual(events, [`loadstart ${LFM2}`, `load ${LFM2}`])
  // Loaded already: no events. A completion for another model loads it on demand.
  await client.chat.completions.create({ model: LFM2, messages: hi })
  await client.chat.completions.create({ model: QWEN, messages: hi })
  assert.deepEqual(events.slice(2), [`loadstart ${QWEN}`, `load ${QWEN}`])
  await client.unloadModel(LFM2)
  assert.deepEqual(events.slice(4), [`unload ${LFM2} unloaded`])
})

test("lifecycle events: a model evicted for room, a failed load, an aborted load, a crash", async () => {
  // Each model claims 600 of the 1,000-byte budget: loading the second evicts the first.
  const evicting = llm({ bytes: () => 600 })
  const evictions = recordEvents(evicting.llm)
  await evicting.llm.loadModel(LFM2)
  await evicting.llm.loadModel(QWEN)
  assert.ok(evictions.includes(`unload ${LFM2} evicted`), evictions.join(", "))

  const failing = llm({ loadError: () => ({ message: "no luck", code: "download-failed" }) })
  const failures = recordEvents(failing.llm)
  await assert.rejects(failing.llm.loadModel(LFM2))
  assert.deepEqual(failures, [`loadstart ${LFM2}`, `error ${LFM2}: no luck`, `unload ${LFM2} failed`])

  const aborting = llm()
  const aborts = recordEvents(aborting.llm)
  const controller = new AbortController()
  const load = aborting.llm.loadModel(LFM2, { signal: controller.signal })
  controller.abort()
  await assert.rejects(load)
  await settle()
  assert.deepEqual(aborts, [`loadstart ${LFM2}`, `unload ${LFM2} aborted`])

  const crashing = llm()
  const crashes = recordEvents(crashing.llm)
  await crashing.llm.loadModel(LFM2)
  crashing.workers.of(LFM2)!.fail("boom")
  await settle()
  assert.equal(crashes[2]?.startsWith(`error ${LFM2}:`), true, crashes.join(", "))
  assert.equal(crashes[3], `unload ${LFM2} crashed`)
})

test("a throwing listener doesn't break loading", async () => {
  const { llm: client } = llm()
  client.addEventListener("loadstart", () => {
    throw new Error("listener bug")
  })
  const original = console.error
  console.error = () => undefined
  try {
    await client.loadModel(LFM2)
  } finally {
    console.error = original
  }
  assert.equal((await client.models.retrieve(LFM2)).loaded, true)
})

test("onMemoryClaim forwards each model's claims, released at its unload event", async () => {
  const workers = fakeWorkers({ bytes: () => 700 })
  internals.storage = fakeStorage(new Set([QWEN]))
  internals.createWorker = workers.create
  const host = new Map<string, number>()
  const instance = new WebGLLM({
    memoryBudget: 1000,
    onMemoryClaim: ({ model, bytes }) => void host.set(model, bytes)
  })
  instance.addEventListener("unload", ({ model }) => host.delete(model))
  await instance.loadModel(QWEN)
  assert.deepEqual([...host], [[QWEN, 700]])
  await instance.unloadModel(QWEN)
  assert.deepEqual([...host], [])
})

test("any Hugging Face model: resolved from the Hub, its meta sent to the worker, listed after", async () => {
  const name = "LiquidAI/LFM2-350M-GGUF"
  const answers = hubFixture as Record<string, { status: number; body: unknown }>
  hub.fetch = async (url: string) => {
    const answer = answers[url]
    if (!answer) return new Response("Not found", { status: 404 })
    return new Response(typeof answer.body === "string" ? answer.body : JSON.stringify(answer.body), { status: answer.status })
  }
  const { llm: client, workers } = llm({ reply: () => ({ tokens: ["Hello", "!"] }) })
  const completion = await client.chat.completions.create({ model: name, messages: hi })
  assert.equal(completion.choices[0]?.message.content, "Hello!")
  const load = workers.of(name)!.sent.find((m) => m.type === "load")
  assert.ok(load?.type === "load" && load.meta?.name === name)
  assert.deepEqual(load.meta.files, { kind: "gguf", model: { repo: name, path: "LFM2-350M-Q4_0.gguf" } })
  // Greedy, as its generation_config.json says nothing about sampling.
  assert.equal(workers.of(name)!.generations()[0]!.sampling.temperature, 0)
  const listed = (await client.models.list()).find((m) => m.id === name)
  assert.equal(listed?.loaded, true)
  assert.equal(listed?.format, "gguf")
  // A catalog model's load carries no meta (the worker has the catalog).
  await client.loadModel(LFM2)
  assert.equal(workers.of(LFM2)!.sent.find((m) => m.type === "load" && "meta" in m), undefined)
})

/**
 * The Hub, holding every answer for a model outside the catalog until
 * `release`: `alias` is answered as LFM2 350M's GGUF repository.
 */
function heldHub(alias: string) {
  const answers = hubFixture as Record<string, { status: number; body: unknown }>
  const source = "LiquidAI/LFM2-350M-GGUF"
  let release!: () => void
  const held = new Promise<void>((resolve) => {
    release = resolve
  })
  const calls: Array<string> = []
  hub.fetch = async (url: string) => {
    calls.push(url)
    await held
    const answer = answers[url.replace(alias, source)]
    if (!answer) return new Response("Not found", { status: 404 })
    return new Response(typeof answer.body === "string" ? answer.body : JSON.stringify(answer.body), { status: answer.status })
  }
  return { calls, release }
}

const isAbort = (error: unknown) => error instanceof DOMException && error.name === "AbortError"

test("aborting stops a call waiting on the Hub; the shared resolution goes on for others", async () => {
  const name = "someone/held-abort-GGUF"
  const { release } = heldHub(name)
  const { llm: client, workers } = llm({ reply: () => ({ tokens: ["ok"] }) })
  const calls = [
    (signal: AbortSignal) => client.loadModel(name, { signal }),
    (signal: AbortSignal) => client.downloadModel(name, { signal }),
    (signal: AbortSignal) => client.chat.completions.create({ model: name, messages: hi }, { signal })
  ]
  for (const call of calls) {
    const controller = new AbortController()
    const pending = call(controller.signal)
    let settled = false
    pending.then(
      () => (settled = true),
      () => (settled = true)
    )
    await settle()
    controller.abort()
    await settle()
    assert.equal(settled, true, "settled while the Hub still holds its answer")
    await assert.rejects(pending, isAbort)
  }
  // Another caller, not aborted, gets the model once the Hub answers.
  const waiting = client.loadModel(name)
  release()
  await waiting
  assert.ok(workers.of(name))
})

test("a load aborted before it starts asks the Hub nothing", async () => {
  const name = "someone/pre-aborted-GGUF"
  const { calls, release } = heldHub(name)
  release()
  const { llm: client } = llm()
  await assert.rejects(client.loadModel(name, { signal: AbortSignal.abort() }), isAbort)
  await assert.rejects(client.downloadModel(name, { signal: AbortSignal.abort() }), isAbort)
  await assert.rejects(client.chat.completions.create({ model: name, messages: hi }, { signal: AbortSignal.abort() }), isAbort)
  assert.equal(calls.length, 0)
})

test("unloading a model while it resolves stops its load", async () => {
  const name = "someone/held-unload-GGUF"
  const { release } = heldHub(name)
  const { llm: client, workers } = llm({ reply: () => ({ tokens: ["ok"] }) })
  const load = client.loadModel(name)
  const reply = client.chat.completions.create({ model: name, messages: hi })
  await settle()
  await client.unloadModel(name)
  release()
  await assert.rejects(load, { code: "model-not-loaded" })
  await assert.rejects(reply, { code: "model-not-loaded" })
  assert.equal(workers.of(name), undefined, "no worker started")
})

test("completions.create: a raw prompt, whole and streamed, with echo and log probabilities", async () => {
  const entry = { token: "Hello", logprob: -0.25, bytes: [72, 101, 108, 108, 111], top_logprobs: [] }
  const { llm: client, workers } = llm({ reply: () => ({ tokens: ["Hello", "!"] }) })
  const whole = await client.completions.create({ model: LFM2, prompt: "Say hi: ", echo: true })
  assert.equal(whole.object, "text_completion")
  assert.equal(whole.choices[0]!.text, "Say hi: Hello!")
  assert.equal(whole.choices[0]!.finish_reason, "stop")
  assert.equal(whole.choices[0]!.logprobs, null)
  assert.equal(whole.usage!.completion_tokens, STATS.outputTokens)
  const sent = workers.of(LFM2)!.generations()[0]!
  assert.equal(sent.prompt, "Say hi: ")
  assert.deepEqual(sent.chat, [])
  // Streamed: the worker's log probabilities ride on the chunks.
  const held = llm()
  const stream = await held.llm.completions.create({ model: LFM2, prompt: "x", stream: true, logprobs: true, stream_options: { include_usage: true } })
  await settle()
  const worker = held.workers.of(LFM2)!
  const id = worker.generations()[0]!.id
  assert.equal(worker.generations()[0]!.logprobs, true)
  worker.emit({ type: "token", id, text: "Hello", logprobs: [entry] })
  worker.finish(id)
  const chunks = []
  for await (const chunk of stream) chunks.push(chunk)
  assert.equal(chunks[0]!.choices[0]!.text, "Hello")
  assert.deepEqual(chunks[0]!.choices[0]!.logprobs, { content: [entry], refusal: null })
  assert.equal(chunks.at(-2)!.choices[0]!.finish_reason, "stop")
  assert.deepEqual(chunks.at(-1)!.choices, [])
  assert.equal(chunks.at(-1)!.usage!.completion_tokens, STATS.outputTokens)
})

test("chat completions return log probabilities when asked for", async () => {
  const entry = { token: "Hi", logprob: -0.5, bytes: [72, 105], top_logprobs: [{ token: "Hi", logprob: -0.5, bytes: [72, 105] }] }
  const { llm: client, workers } = llm()
  const pending = client.chat.completions.create({ model: LFM2, messages: hi, logprobs: true, top_logprobs: 1 })
  await settle()
  const worker = workers.of(LFM2)!
  const sent = worker.generations()[0]!
  assert.equal(sent.logprobs, true)
  assert.equal(sent.topLogprobs, 1)
  worker.emit({ type: "token", id: sent.id, text: "Hi", logprobs: [entry] })
  worker.finish(sent.id)
  const completion = await pending
  assert.deepEqual(completion.choices[0]!.logprobs, { content: [entry], refusal: null })
  // Not asked for: null, as before.
  const plain = llm({ reply: () => ({ tokens: ["Hi"] }) })
  assert.equal((await plain.llm.chat.completions.create({ model: LFM2, messages: hi })).choices[0]!.logprobs, null)
})

test("forwardTokensAndSample and resetChat, as WebLLM has them", async () => {
  const { llm: client, workers } = llm()
  await assert.rejects(client.forwardTokensAndSample([1, 2], true), { code: "model-not-loaded" })
  await client.loadModel(LFM2)
  // The one loaded model, when modelId is left out.
  assert.equal(await client.forwardTokensAndSample([5, 6, 7], true), 8)
  assert.equal(await client.forwardTokensAndSample([8], false, LFM2), 9)
  const worker = workers.of(LFM2)!
  const forwards = worker.sent.filter((m) => m.type === "forward")
  assert.deepEqual(forwards.map((m) => m.inputIds), [[5, 6, 7], [8]])
  assert.equal(typeof forwards[0]!.sampling.temperature, "number")
  await client.resetChat()
  assert.equal(worker.sent.filter((m) => m.type === "reset").length, 1)
  await assert.rejects(client.forwardTokensAndSample([1.5], true), TypeError)
  // A model that isn't loaded is loaded first.
  assert.equal(await client.forwardTokensAndSample([1], true, QWEN), 2)
  assert.ok(workers.of(QWEN))
  // Two loaded: which one must be said.
  await assert.rejects(client.forwardTokensAndSample([1], true), { code: "model-not-loaded" })
})

test("a worker of the caller's: a token forward for another model waits its turn behind a completion still loading", async () => {
  let release!: () => void
  const held = new Promise<void>((resolve) => { release = resolve })
  const worker = new FakeWorker({ hold: (model) => (model === LFM2 ? held : Promise.resolve()), reply: () => ({ tokens: ["ok"] }) })
  internals.storage = fakeStorage(new Set([LFM2, QWEN]))
  const client = new WebGLLM({ worker: worker as unknown as Worker, memoryBudget: 1000 })
  // Both models cold: the chat's load is held, then a forward for the other model is asked.
  const chat = client.chat.completions.create({ model: LFM2, messages: hi })
  await settle()
  const forwarded = client.forwardTokensAndSample([5], true, QWEN)
  await settle()
  assert.equal(worker.sent.some((m) => m.type === "load" && m.model === QWEN), false, "the forward's model replaced the chat's while it loaded")
  release()
  assert.equal((await chat).choices[0]!.message.content, "ok")
  assert.equal(await forwarded, 6)
  assert.deepEqual(worker.sent.filter((m) => m.type === "generate" || m.type === "forward").map((m) => m.model), [LFM2, QWEN])
})

test("a token forward queued behind a chat doesn't load its model after unloadModel() has resolved", async () => {
  const { llm: client, workers } = llm()
  const chat = client.chat.completions.create({ model: LFM2, messages: hi })
  chat.catch(() => undefined)
  await settle()
  // Queued behind the chat's generation, for another model.
  const forwarded = client.forwardTokensAndSample([5], true, QWEN)
  forwarded.catch(() => undefined)
  await settle()
  await client.unloadModel()
  await assert.rejects(chat, { code: "model-not-loaded" })
  await assert.rejects(forwarded, { code: "model-not-loaded" })
  await settle()
  assert.deepEqual((await client.models.list()).filter((m) => m.loaded).map((m) => m.id), [])
  assert.equal(workers.workers.some((w) => !w.terminated && w.sent.some((m) => m.type === "forward")), false)
})

test("resetChat called after a token forward resets after it: loaded, and still loading", async () => {
  const { llm: client, workers } = llm()
  await client.loadModel(LFM2)
  const forwarded = client.forwardTokensAndSample([5], true)
  const reset = client.resetChat()
  await Promise.all([forwarded, reset])
  assert.deepEqual(workers.of(LFM2)!.sent.filter((m) => m.type === "forward" || m.type === "reset").map((m) => m.type), ["forward", "reset"])
  // Cold: the forward's load is held when reset is called.
  let release!: () => void
  const held = new Promise<void>((resolve) => { release = resolve })
  const cold = llm({ hold: () => held })
  const first = cold.llm.forwardTokensAndSample([5], true, LFM2)
  await settle()
  let done = false
  const after = cold.llm.resetChat(false, LFM2).then(() => { done = true })
  await settle()
  assert.equal(done, false, "the reset resolved before the forward it was called after")
  release()
  assert.equal(await first, 6)
  await after
  assert.deepEqual(cold.workers.of(LFM2)!.sent.filter((m) => m.type === "forward" || m.type === "reset").map((m) => m.type), ["forward", "reset"])
})

test("deleting a model keeps the files another model is still downloading", async () => {
  // Two models outside the catalog with their own weights and one tokenizer between them.
  const base = modelMeta(LFM2)
  assert.equal(base.files.kind, "gguf")
  const variant = (name: string) => ({ ...base, name, files: { ...base.files, model: { ...(base.files as { model: object }).model, path: `${name}.gguf` } } }) as typeof base
  const [a, b] = ["shared-tokenizer/a", "shared-tokenizer/b"]
  rememberModel(variant(a))
  rememberModel(variant(b))
  const keys = (model: string) => new Set(storedModelFiles(modelMeta(model)).map(fileKey))
  assert.ok([...keys(a)].some((k) => keys(b).has(k)) && [...keys(a)].some((k) => !keys(b).has(k)))
  let release!: () => void
  const held = new Promise<void>((resolve) => { release = resolve })
  const { llm: client, storage } = llm({ holdDownload: () => held })
  const downloads = [client.downloadModel(a), client.downloadModel(b)]
  downloads[0]!.catch(() => undefined)
  await settle()
  await client.models.delete(a)
  const [removed, kept] = storage.removed.at(-1)!
  assert.equal(removed, a)
  const shared = [...keys(a)].filter((k) => keys(b).has(k))
  assert.ok(shared.every((k) => kept.has(k)), "a file the other download is writing wasn't protected")
  release()
  await downloads[1]
})

test("log probabilities reach whole and streamed replies as the worker reports them, a cut token left out", async () => {
  const hello = { token: "Hello", logprob: -0.1, bytes: [72, 101, 108, 108, 111], top_logprobs: [] }
  for (const stream of [false, true]) {
    const { llm: client, workers } = llm()
    const pending = client.chat.completions.create({ model: LFM2, messages: hi, logprobs: true, stop: ["STOP"], stream } as never) as Promise<unknown>
    await settle()
    const worker = workers.of(LFM2)!
    const sent = worker.generations()[0]!
    // As the worker sends a reply whose last token (" STOP") the stop sequence cut into.
    worker.emit({ type: "token", id: sent.id, text: "Hello", logprobs: [hello] })
    worker.emit({ type: "token", id: sent.id, text: " " })
    worker.finish(sent.id)
    const result = await pending
    const entries: unknown[] = []
    let content = ""
    if (stream) {
      for await (const chunk of result as AsyncIterable<ChatCompletionChunk>) {
        content += chunk.choices[0]?.delta.content ?? ""
        entries.push(...(chunk.choices[0]?.logprobs?.content ?? []))
      }
    } else {
      const whole = result as { choices: Array<{ message: { content: string }; logprobs: { content: unknown[] } }> }
      content = whole.choices[0]!.message.content
      entries.push(...whole.choices[0]!.logprobs.content)
    }
    assert.equal(content, "Hello ")
    assert.deepEqual(entries, [hello])
  }
})
