import assert from "node:assert/strict"
import { test } from "node:test"
import {
  buildCompletion,
  createChunkWriter,
  createReplyReader,
  planCompletion,
  planRepairs,
  planTextCompletion,
  randomId,
  repairRequest,
  timingsOf,
  toChatMessages,
  usageOf
} from "./chat-completions.js"
import type { ReplyDelta } from "./chat-completions.js"
import { WebGLLMError } from "./errors.js"
import type { ChatCompletionCreateParams } from "./openai.js"
import type { RunTimings } from "./protocol.js"

const QWEN = "unsloth/Qwen3.5-2B-GGUF"
const LFM3 = "LiquidAI/LFM3-1.7B-GGUF"
const LFM2 = "LiquidAI/LFM2.5-1.2B-Instruct-GGUF"
const LFM2_THINKING = "LiquidAI/LFM2.5-1.2B-Thinking-GGUF"
const hi = [{ role: "user" as const, content: "hi" }]

const weather = {
  type: "function" as const,
  function: { name: "get_weather", parameters: { type: "object", properties: { city: { type: "string" } } } }
}

function code(fn: () => unknown): string | undefined {
  try {
    fn()
  } catch (error) {
    return error instanceof WebGLLMError ? error.code : (error as Error).name
  }
  return undefined
}

test("a plain request takes the model card's sampling and nothing else", () => {
  const plan = planCompletion({ model: LFM3, messages: hi })
  assert.equal(plan.stream, false)
  assert.deepEqual(plan.generate, {
    chat: [{ role: "user", content: "hi" }],
    tools: [],
    sampling: { temperature: 0.1, top_k: 50, repetition_penalty: 1.1 }
  })
  assert.equal(plan.parseTools, false)
})

test("request fields map onto the generate request", () => {
  const plan = planCompletion({
    model: QWEN,
    messages: hi,
    stream: true,
    stream_options: { include_usage: true },
    max_tokens: 10,
    max_completion_tokens: 20,
    temperature: 0.5,
    top_p: 0.9,
    presence_penalty: 0,
    stop: "END",
    seed: 7,
    top_k: 5,
    min_p: 0.05,
    repetition_penalty: 1.2,
    cache_prompt: false,
    frequency_penalty: 0,
    n: 1,
    logprobs: false,
    user: "u"
  })
  assert.equal(plan.stream, true)
  assert.equal(plan.includeUsage, true)
  assert.equal(plan.generate.maxTokens, 20)
  assert.deepEqual(plan.generate.stop, ["END"])
  assert.equal(plan.generate.seed, 7)
  assert.equal(plan.generate.cachePrompt, false)
  assert.deepEqual(plan.generate.sampling, {
    temperature: 0.5,
    top_k: 5,
    repetition_penalty: 1.2,
    top_p: 0.9,
    min_p: 0.05,
    presence_penalty: 0
  })
})

test("max_tokens alone is the limit", () => {
  assert.equal(planCompletion({ model: LFM3, messages: hi, max_tokens: 3 }).generate.maxTokens, 3)
})

test("reasoning_effort switches thinking and its sampling", () => {
  const on = planCompletion({ model: QWEN, messages: hi, reasoning_effort: "high" })
  assert.equal(on.generate.thinking, true)
  assert.equal(on.generate.sampling.top_p, 0.95)
  assert.equal(on.generate.sampling.presence_penalty, 1.5)
  const off = planCompletion({ model: QWEN, messages: hi, reasoning_effort: "none" })
  assert.equal(off.generate.thinking, false)
  assert.equal(off.generate.sampling.top_p, 1)
  assert.equal("thinking" in planCompletion({ model: QWEN, messages: hi }).generate, false)
  // A model that always thinks ignores "none"; one that never thinks ignores any level.
  assert.equal("thinking" in planCompletion({ model: LFM2_THINKING, messages: hi, reasoning_effort: "none" }).generate, false)
  assert.equal("thinking" in planCompletion({ model: LFM2, messages: hi, reasoning_effort: "high" }).generate, false)
  assert.equal(code(() => planCompletion({ model: QWEN, messages: hi, reasoning_effort: "max" as never })), "TypeError")
})

test("a thinking budget (reasoning.max_tokens) needs a model that takes one, turns thinking on, and is dropped with thinking off", () => {
  const budgeted = planCompletion({ model: QWEN, messages: hi, reasoning: { max_tokens: 64 } }).generate
  assert.equal(budgeted.thinkingBudgetTokens, 64)
  assert.equal(budgeted.thinking, true)
  assert.equal(planCompletion({ model: QWEN, messages: hi, reasoning: { effort: "high", max_tokens: 64 } }).generate.thinkingBudgetTokens, 64)
  assert.equal(
    "thinkingBudgetTokens" in planCompletion({ model: QWEN, messages: hi, reasoning: { effort: "none", max_tokens: 64 } }).generate,
    false
  )
  assert.equal(code(() => planCompletion({ model: LFM2, messages: hi, reasoning: { max_tokens: 64 } })), "unsupported-parameter")
  // The old field is refused, never dropped: a cap the caller asked for isn't silently lost.
  assert.equal(code(() => planCompletion({ model: QWEN, messages: hi, reasoning_effort: "high", thinking_budget_tokens: 64 } as never)), "unsupported-parameter")
})

test("reasoning.effort works as reasoning_effort; the two must agree, and other reasoning fields are refused", () => {
  assert.equal(planCompletion({ model: QWEN, messages: hi, reasoning: { effort: "none" } }).generate.thinking, false)
  assert.equal(planCompletion({ model: QWEN, messages: hi, reasoning_effort: "high", reasoning: { effort: "high" } }).generate.thinking, true)
  assert.throws(() => planCompletion({ model: QWEN, messages: hi, reasoning_effort: "none", reasoning: { effort: "high" } }), TypeError)
  assert.equal(code(() => planCompletion({ model: QWEN, messages: hi, reasoning: { exclude: true } as never })), "unsupported-parameter")
  assert.throws(() => planCompletion({ model: QWEN, messages: hi, reasoning: "high" as never }), TypeError)
})

test("unsupported OpenAI fields throw unsupported-parameter", () => {
  const bad: Array<Partial<ChatCompletionCreateParams> & Record<string, unknown>> = [
    { n: 2 },
    { frequency_penalty: 0.5 },
    { messages: [{ role: "user", content: [{ type: "image_url", image_url: { url: "x" } } as never] }] },
    { response_format: { type: "grammar" } as never },
    { tools: [{ type: "custom", custom: { name: "x" } } as never] },
    { modalities: ["text", "audio"] }
  ]
  for (const fields of bad) {
    assert.equal(code(() => planCompletion({ model: LFM3, messages: hi, ...fields } as ChatCompletionCreateParams)), "unsupported-parameter", JSON.stringify(fields))
  }
})

test("models outside the catalog and malformed values are refused", () => {
  assert.equal(code(() => planCompletion({ model: "gpt-4o", messages: hi })), "model-not-supported")
  assert.equal(code(() => planCompletion({ messages: hi } as never)), "TypeError")
  assert.equal(code(() => planCompletion({ model: LFM3, messages: [] })), "TypeError")
  assert.equal(code(() => planCompletion({ model: LFM3, messages: hi, temperature: 3 })), "RangeError")
  assert.equal(code(() => planCompletion({ model: LFM3, messages: hi, max_tokens: 1.5 })), "RangeError")
  assert.equal(code(() => planCompletion({ model: LFM3, messages: hi, top_k: "5" as never })), "TypeError")
  // Zero would divide the seen logits by zero (and turn masked ones into NaN).
  assert.equal(code(() => planCompletion({ model: LFM3, messages: hi, repetition_penalty: 0 } as ChatCompletionCreateParams)), "RangeError")
  assert.equal(
    code(() => planCompletion({ model: LFM3, messages: hi, response_format: { type: "json_schema", json_schema: { name: "x", schema: [] as never } } })),
    "invalid-schema"
  )
})

test("response_format becomes the output schema", () => {
  assert.deepEqual(planCompletion({ model: LFM3, messages: hi, response_format: { type: "json_object" } }).generate.outputSchema, { type: "object" })
  const schema = { type: "object", properties: { a: { type: "number" } }, required: ["a"] }
  const plan = planCompletion({
    model: LFM3,
    messages: hi,
    tools: [weather],
    response_format: { type: "json_schema", json_schema: { name: "a", schema, strict: true } }
  })
  assert.deepEqual(plan.generate.outputSchema, { ...schema, additionalProperties: false })
  assert.equal(plan.parseTools, false)
  assert.equal("outputSchema" in planCompletion({ model: LFM3, messages: hi, response_format: { type: "text" } }).generate, false)
})

test("json_schema objects that don't set additionalProperties are closed, as tools' are", () => {
  const outputSchema = (schema: Record<string, unknown>) =>
    planCompletion({ model: LFM3, messages: hi, response_format: { type: "json_schema", json_schema: { name: "a", schema } } }).generate.outputSchema
  // Nested objects too: an open object's free-form keys are slow to constrain and let a model invent keys.
  assert.deepEqual(
    outputSchema({ type: "object", properties: { author: { type: "object", properties: { name: { type: "string" } } } } }),
    { type: "object", properties: { author: { type: "object", properties: { name: { type: "string" } }, additionalProperties: false } }, additionalProperties: false }
  )
  // Saying so keeps an object open.
  const open = { type: "object", properties: { a: { type: "string" } }, additionalProperties: true }
  assert.deepEqual(outputSchema(open), open)
  // JSON mode is any object.
  assert.deepEqual(planCompletion({ model: LFM3, messages: hi, response_format: { type: "json_object" } }).generate.outputSchema, { type: "object" })
})

test("tools and tool_choice", () => {
  const auto = planCompletion({ model: LFM3, messages: hi, tools: [weather] })
  assert.equal(auto.parseTools, true)
  assert.equal("toolChoice" in auto.generate, false)
  assert.deepEqual(auto.generate.tools[0], { type: "function", function: weather.function })
  const none = planCompletion({ model: LFM3, messages: hi, tools: [weather], tool_choice: "none" })
  assert.deepEqual(none.generate.tools, [])
  assert.equal(none.generate.toolChoice, "none")
  assert.equal(none.parseTools, false)
  const named = planCompletion({ model: LFM3, messages: hi, tools: [weather], tool_choice: { type: "function", function: { name: "get_weather" } } })
  assert.deepEqual(named.generate.toolChoice, { name: "get_weather" })
  assert.equal(named.parseTools, false)
  assert.equal(planCompletion({ model: LFM3, messages: hi, tools: [weather], tool_choice: "required" }).generate.toolChoice, "required")
  assert.equal(code(() => planCompletion({ model: LFM3, messages: hi, tool_choice: "required" })), "TypeError")
  assert.equal(
    code(() => planCompletion({ model: LFM3, messages: hi, tools: [weather], tool_choice: { type: "function", function: { name: "nope" } } })),
    "TypeError"
  )
  assert.equal(planCompletion({ model: LFM3, messages: hi, tools: [weather], parallel_tool_calls: false }).parallelToolCalls, false)
  // A tool without parameters gets an empty object schema.
  const bare = planCompletion({ model: LFM3, messages: hi, tools: [{ type: "function", function: { name: "now" } }] })
  assert.deepEqual(bare.generate.tools[0]?.function.parameters, { type: "object", properties: {}, required: [] })
})

test("messages: system and developer join first; parts join; tool calls get object arguments", () => {
  const chat = toChatMessages(
    [
      { role: "system", content: "Be brief." },
      { role: "user", content: [{ type: "text", text: "Weather " }, { type: "text", text: "in Paris?" }] },
      { role: "developer", content: "Use tools." },
      {
        role: "assistant",
        content: null,
        reasoning_content: "Need the tool.",
        tool_calls: [{ id: "call_1", type: "function", function: { name: "get_weather", arguments: '{"city":"Paris"}' } }]
      },
      { role: "tool", tool_call_id: "call_1", content: "18C" }
    ],
    "qwen3.5"
  )
  assert.deepEqual(chat, [
    { role: "system", content: "Be brief.\n\nUse tools." },
    { role: "user", content: "Weather in Paris?" },
    {
      role: "assistant",
      content: "",
      reasoning_content: "Need the tool.",
      tool_calls: [{ id: "call_1", type: "function", function: { name: "get_weather", arguments: { city: "Paris" } } }]
    },
    { role: "tool", content: "18C", tool_call_id: "call_1", name: "get_weather" }
  ])
})

test("messages: LFM3 keeps thinking as `thinking`; LFM2 writes calls into the content", () => {
  const turn = [
    { role: "user" as const, content: "Weather?" },
    {
      role: "assistant" as const,
      content: "",
      reasoning_content: "hmm",
      tool_calls: [{ id: "c", type: "function" as const, function: { name: "get_weather", arguments: '{"city":"Oslo"}' } }]
    },
    { role: "tool" as const, tool_call_id: "c", content: "" }
  ]
  assert.equal(toChatMessages(turn, "lfm3")[1]?.thinking, "hmm")
  const lfm2 = toChatMessages(turn, "lfm2")
  assert.equal(lfm2[1]?.content, '<|tool_call_start|>[get_weather(city="Oslo")]<|tool_call_end|>')
  assert.equal(lfm2[1]?.tool_calls, undefined)
  // An empty tool result reads as a sentence, not nothing.
  assert.equal(lfm2[2]?.content, "The tool returned nothing.")
})

test("messages: earlier turns keep their answers, not their tool calls", () => {
  const chat = toChatMessages(
    [
      { role: "user", content: "Weather?" },
      { role: "assistant", content: "", tool_calls: [{ id: "c", type: "function", function: { name: "get_weather", arguments: "{}" } }] },
      { role: "tool", tool_call_id: "c", content: "18C" },
      { role: "assistant", content: "It's 18C." },
      { role: "user", content: "Thanks" }
    ],
    "lfm3"
  )
  assert.deepEqual(chat, [
    { role: "user", content: "Weather?" },
    { role: "assistant", content: "It's 18C." },
    { role: "user", content: "Thanks" }
  ])
})

const end = { finishReason: "stop" as const }
const ids = () => {
  let n = 0
  return () => `call_${n++}`
}

function readAll(pieces: Array<string>, options: Parameters<typeof createReplyReader>[0], last: Parameters<ReturnType<typeof createReplyReader>["finish"]>[0] = end) {
  const reader = createReplyReader({ newId: ids(), ...options })
  const deltas: Array<ReplyDelta> = pieces.flatMap((p) => reader.push(p))
  const done = reader.finish(last)
  // Unrepaired markup shows as text (repair is tested apart: planRepairs).
  const all = [...deltas, ...done.deltas, ...(done.malformed !== undefined ? [{ content: done.malformed }] : [])]
  const toolCalls = reader.toolCalls(done.calls)
  return {
    streamed: deltas,
    content: all.map((d) => ("content" in d ? d.content : "")).join(""),
    reasoning: all.map((d) => ("reasoning_content" in d ? d.reasoning_content : "")).join(""),
    toolCalls,
    finishReason: toolCalls.length ? "tool_calls" : done.finishReason
  }
}

test("the reader splits thinking from the answer, tags split across pieces", () => {
  const r = readAll(["<th", "ink>Let me ", "think.</th", "ink>\n\nHello", " there"], { parseTools: false })
  assert.equal(r.reasoning, "Let me think.")
  assert.equal(r.content, "Hello there")
  assert.equal(r.finishReason, "stop")
  // The answer streams as it comes.
  assert.ok(r.streamed.some((d) => "content" in d && d.content === "Hello"))
})

test("the reader parses LFM3/Qwen XML tool calls and streams the text before them", () => {
  const raw = "Checking.<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n</function>\n</tool_call>"
  const r = readAll([...raw], { parseTools: true })
  assert.equal(r.content, "Checking.")
  assert.deepEqual(r.toolCalls, [{ id: "call_0", type: "function", function: { name: "get_weather", arguments: '{"city":"Paris"}' } }])
  assert.equal(r.finishReason, "tool_calls")
})

test("the reader parses LFM2 calls, several per reply unless parallel calls are off", () => {
  const raw = '<|tool_call_start|>[get_weather(city="Paris"), get_weather(city="Rome")]<|tool_call_end|>'
  assert.equal(readAll([raw], { parseTools: true, format: "lfm2" }).toolCalls.length, 2)
  const one = readAll([raw], { parseTools: true, format: "lfm2", parallelToolCalls: false })
  assert.deepEqual(one.toolCalls.map((c) => c.function.arguments), ['{"city":"Paris"}'])
})

test("markup that isn't a tool call stays text; no tools means no parsing", () => {
  assert.equal(readAll(["<tool_call>oops"], { parseTools: true }).content, "<tool_call>oops")
  const r = readAll(["a <tool_call>\n<function=x>\n</function>\n</tool_call>"], { parseTools: false })
  assert.equal(r.toolCalls.length, 0)
  assert.ok(r.content.includes("<function=x>"))
})

test("a structured answer after thinking is just the JSON", () => {
  const r = readAll(["<think>plan</think>", "\n", '{"a":', "1}"], { parseTools: false })
  assert.equal(r.reasoning, "plan")
  assert.deepEqual(JSON.parse(r.content), { a: 1 })
})

test("a forced tool call from the worker becomes the reply's tool call", () => {
  const r = readAll(["<think>pick</think>"], { parseTools: false }, { finishReason: "stop", toolCall: { name: "get_weather", arguments: { city: "Oslo" } } })
  assert.equal(r.reasoning, "pick")
  assert.equal(r.content, "")
  assert.deepEqual(r.toolCalls, [{ id: "call_0", type: "function", function: { name: "get_weather", arguments: '{"city":"Oslo"}' } }])
  assert.equal(r.finishReason, "tool_calls")
})

test("length passes through as the finish reason", () => {
  assert.equal(readAll(["abc"], { parseTools: false }, { finishReason: "length" }).finishReason, "length")
})

const stats: RunTimings = {
  inputTokens: 100,
  cachedTokens: 40,
  outputTokens: 11,
  firstTokenMs: 250,
  promptMs: 200,
  decodeMs: 500,
  thinkingTokens: 4
}

test("usage and timings from the worker's stats", () => {
  assert.deepEqual(usageOf(stats), {
    prompt_tokens: 100,
    completion_tokens: 11,
    total_tokens: 111,
    prompt_tokens_details: { cached_tokens: 40 },
    completion_tokens_details: { reasoning_tokens: 4 }
  })
  assert.deepEqual(timingsOf(stats), {
    first_token_ms: 250,
    prompt_ms: 200,
    decode_ms: 500,
    prompt_tokens_per_second: 300,
    decode_tokens_per_second: 20
  })
  assert.equal(timingsOf({ ...stats, decodeMs: 0, outputTokens: 1 }).decode_tokens_per_second, 0)
})

test("chunks: role on the first, tool calls indexed, finish then usage", () => {
  const w = createChunkWriter({ id: "chatcmpl-x", created: 1, model: LFM3 })
  const first = w.delta({ content: "Hi" })
  assert.deepEqual(first.choices[0]?.delta, { role: "assistant", content: "Hi" })
  assert.equal(first.object, "chat.completion.chunk")
  assert.deepEqual(w.delta({ reasoning_content: "x" }).choices[0]?.delta, { reasoning_content: "x" })
  const calls = w.toolCalls([
    { id: "a", type: "function", function: { name: "f", arguments: "{}" } },
    { id: "b", type: "function", function: { name: "g", arguments: "{}" } }
  ])
  assert.deepEqual(calls.map((c) => c.choices[0]?.delta.tool_calls?.[0]?.index), [0, 1])
  const finish = w.finish("tool_calls")
  assert.equal(finish.choices[0]?.finish_reason, "tool_calls")
  assert.equal(finish.timings, undefined)
  const usage = w.usage(usageOf(stats), timingsOf(stats))
  assert.deepEqual(usage.choices, [])
  assert.equal(usage.usage?.total_tokens, 111)
  // A reply with nothing to stream still sends the role, on its finish chunk.
  const empty = createChunkWriter({ id: "c", created: 1, model: LFM3 }).finish("stop", timingsOf(stats))
  assert.deepEqual(empty.choices[0]?.delta, { role: "assistant" })
  assert.equal(empty.timings?.prompt_ms, 200)
})

test("a whole completion: content null when only tool calls", () => {
  const meta = { id: "chatcmpl-x", created: 1, model: LFM3 }
  const text = buildCompletion(meta, [{ reasoning_content: "r" }, { content: "a" }, { content: "b" }], [], "stop", stats)
  assert.deepEqual(text.choices[0]?.message, { role: "assistant", content: "ab", reasoning_content: "r", refusal: null })
  assert.equal(text.object, "chat.completion")
  const call = { id: "c", type: "function" as const, function: { name: "f", arguments: "{}" } }
  const calls = buildCompletion(meta, [], [call], "tool_calls", stats)
  assert.deepEqual(calls.choices[0]?.message, { role: "assistant", content: null, tool_calls: [call], refusal: null })
})

test("ids look like OpenAI's", () => {
  assert.match(randomId("call_"), /^call_[A-Za-z0-9]{24}$/)
})

const weatherTool = {
  type: "function" as const,
  function: { name: "get_weather", parameters: { type: "object", properties: { city: { type: "string" } }, required: ["city"] } }
}

test("planRepairs: valid calls pass; bad arguments, unknown tools and unparsed markup are repaired", async () => {
  const tools = [weatherTool]
  const ok = { name: "get_weather", args: { city: "Paris" } }
  assert.deepEqual(await planRepairs([ok], undefined, tools), [])
  // Open objects take extra keys in the tool's own schema; a missing required one fails.
  const town = { name: "get_weather", args: { town: "Paris" } }
  assert.deepEqual(await planRepairs([town], undefined, tools), [{ index: 0, problem: "invalid-arguments", choice: { name: "get_weather" }, draft: town }])
  const typed = { name: "get_weather", args: { city: 3 } }
  assert.equal((await planRepairs([typed], undefined, tools))[0]?.problem, "invalid-arguments")
  const unknown = { name: "get_time", args: {} }
  assert.deepEqual(await planRepairs([ok, unknown], undefined, tools), [{ index: 1, problem: "unknown-tool", choice: "required", draft: unknown }])
  assert.deepEqual(await planRepairs([], "<tool_call>junk</tool_call>", tools), [{ index: 0, problem: "malformed", choice: "required", markup: "<tool_call>junk</tool_call>" }])
  // No tools offered: markup stays text.
  assert.deepEqual(await planRepairs([], "<tool_call>junk</tool_call>", []), [])
  // A schema the validator can't check passes.
  const odd = { type: "function" as const, function: { name: "odd", parameters: { type: "object", unevaluatedProperties: false } } }
  assert.deepEqual(await planRepairs([{ name: "odd", args: { a: 1 } }], undefined, [odd]), [])
})

test("repairRequest appends the failed call and a note, drops stop strings and closes thinking", () => {
  const qwen = planCompletion({ model: "unsloth/Qwen3.5-2B-GGUF", messages: hi, tools: [weatherTool], stop: ["END"], reasoning: { effort: "high", max_tokens: 300 } })
  const draft = { name: "get_weather", args: { town: "Paris" } }
  const request = repairRequest(qwen, { index: 0, problem: "invalid-arguments", choice: { name: "get_weather" }, draft })
  // The prompt so far is kept as it was (the cache covers it).
  assert.deepEqual(request.chat.slice(0, qwen.generate.chat.length), qwen.generate.chat)
  const [attempt, note] = request.chat.slice(qwen.generate.chat.length)
  assert.deepEqual(attempt?.tool_calls?.[0]?.function, { name: "get_weather", arguments: { town: "Paris" } })
  assert.equal(note?.role, "user")
  assert.match(String(note?.content ?? ""), /get_weather/)
  assert.deepEqual(request.tools, qwen.generate.tools)
  assert.deepEqual(request.toolChoice, { name: "get_weather" })
  assert.equal(request.stop, undefined)
  assert.equal(request.thinkingBudgetTokens, 0)
  assert.equal(request.thinking, true)
  // Thinking off stays off, with no budget.
  const off = planCompletion({ model: "unsloth/Qwen3.5-2B-GGUF", messages: hi, tools: [weatherTool], reasoning_effort: "none" })
  const malformed = repairRequest(off, { index: 0, problem: "malformed", choice: "required", markup: "<tool_call>junk" })
  assert.equal(malformed.thinking, false)
  assert.equal(malformed.thinkingBudgetTokens, undefined)
  assert.equal(malformed.chat.at(-2)?.content, "<tool_call>junk")
  // LFM2 writes calls in its own format, in the content.
  const lfm2 = planCompletion({ model: "LiquidAI/LFM2.5-350M-GGUF", messages: hi, tools: [weatherTool] })
  const written = repairRequest(lfm2, { index: 0, problem: "unknown-tool", choice: "required", draft: { name: "get_time", args: {} } })
  assert.match(String(written.chat.at(-2)?.content ?? ""), /get_time/)
  assert.match(String(written.chat.at(-1)?.content ?? ""), /no tool named get_time/)
})

test("usage sums the reply and its repairs", () => {
  const stats = { inputTokens: 100, cachedTokens: 0, outputTokens: 20, firstTokenMs: 1, promptMs: 1, decodeMs: 1, thinkingTokens: 5 }
  const repair = { ...stats, cachedTokens: 99, outputTokens: 8, thinkingTokens: 0 }
  assert.deepEqual(usageOf(stats, [repair]), {
    prompt_tokens: 200,
    completion_tokens: 28,
    total_tokens: 228,
    prompt_tokens_details: { cached_tokens: 99 },
    completion_tokens_details: { reasoning_tokens: 5 }
  })
})

test("repair_tool_calls must be a boolean and defaults to on", () => {
  assert.equal(planCompletion({ model: "unsloth/Qwen3.5-2B-GGUF", messages: hi }).repairToolCalls, true)
  assert.equal(planCompletion({ model: "unsloth/Qwen3.5-2B-GGUF", messages: hi, repair_tool_calls: false }).repairToolCalls, false)
  assert.throws(() => planCompletion({ model: "unsloth/Qwen3.5-2B-GGUF", messages: hi, repair_tool_calls: "yes" as never }), TypeError)
})

test("logprobs, top_logprobs and logit_bias go to the worker", () => {
  const plan = planCompletion({ model: LFM3, messages: hi, logprobs: true, top_logprobs: 3, logit_bias: { "12": 5, "13": 0 } })
  assert.equal(plan.logprobs, true)
  assert.equal(plan.generate.logprobs, true)
  assert.equal(plan.generate.topLogprobs, 3)
  assert.deepEqual(plan.generate.logitBias, { "12": 5 })
  const plain = planCompletion({ model: LFM3, messages: hi, logprobs: false, logit_bias: {} })
  assert.equal(plain.logprobs, false)
  assert.equal("logprobs" in plain.generate, false)
  assert.equal("logitBias" in plain.generate, false)
  assert.equal(code(() => planCompletion({ model: LFM3, messages: hi, top_logprobs: 2 })), "TypeError")
  assert.equal(code(() => planCompletion({ model: LFM3, messages: hi, logprobs: true, top_logprobs: 21 })), "RangeError")
  assert.equal(code(() => planCompletion({ model: LFM3, messages: hi, logit_bias: { hello: 1 } })), "TypeError")
  assert.equal(code(() => planCompletion({ model: LFM3, messages: hi, logit_bias: { "1": 101 } })), "TypeError")
})

test("chunks and whole completions carry log probabilities when asked for", () => {
  const entry = (token: string) => ({ token, logprob: -0.5, bytes: [...new TextEncoder().encode(token)], top_logprobs: [] })
  const w = createChunkWriter({ id: "chatcmpl-x", created: 1, model: LFM3 }, true)
  w.logprobs([entry("Hel")])
  // Its text is still held by the reader: it goes with the next chunk.
  w.logprobs([entry("lo")])
  assert.deepEqual(w.delta({ content: "Hello" }).choices[0]?.logprobs, { content: [entry("Hel"), entry("lo")], refusal: null })
  assert.deepEqual(w.finish("stop").choices[0]?.logprobs, { content: [], refusal: null })
  const whole = buildCompletion({ id: "chatcmpl-x", created: 1, model: LFM3 }, [{ content: "Hello" }], [], "stop", stats, [], [entry("Hello")])
  assert.deepEqual(whole.choices[0]?.logprobs, { content: [entry("Hello")], refusal: null })
  assert.equal(buildCompletion({ id: "chatcmpl-x", created: 1, model: LFM3 }, [{ content: "Hello" }], [], "stop", stats).choices[0]?.logprobs, null)
})

test("a Completions request: a raw prompt, no chat, the card's sampling", () => {
  const plan = planTextCompletion({ model: LFM3, prompt: "Once upon a time", max_tokens: 8, stop: "\n", echo: true, logprobs: true, top_logprobs: 1 })
  assert.equal(plan.generate.prompt, "Once upon a time")
  assert.deepEqual(plan.generate.chat, [])
  assert.deepEqual(plan.generate.tools, [])
  assert.equal(plan.generate.maxTokens, 8)
  assert.deepEqual(plan.generate.stop, ["\n"])
  assert.equal(plan.echo, "Once upon a time")
  assert.equal(plan.generate.topLogprobs, 1)
  assert.equal(typeof plan.generate.sampling.temperature, "number")
  assert.equal(planTextCompletion({ model: LFM3, prompt: "x" }).echo, undefined)
  assert.equal(code(() => planTextCompletion({ model: LFM3 } as never)), "TypeError")
  assert.equal(code(() => planTextCompletion({ model: LFM3, prompt: ["a"] } as never)), "TypeError")
  assert.equal(code(() => planTextCompletion({ model: LFM3, prompt: "x", n: 2 })), "unsupported-parameter")
  assert.equal(code(() => planTextCompletion({ model: LFM3, prompt: "x", suffix: "y" })), "unsupported-parameter")
  assert.equal(code(() => planTextCompletion({ model: "gpt-4o", prompt: "x" })), "model-not-supported")
})
