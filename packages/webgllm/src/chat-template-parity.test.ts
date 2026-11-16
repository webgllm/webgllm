import assert from "node:assert/strict"
import { test } from "node:test"
import hub from "./test-fixtures/hub.json" with { type: "json" }
import { loadTokenizer } from "./tokenizer.js"
import type { ChatMessage } from "./tokenizer.js"
import { withCallsInText } from "./tool-calls.js"

const TEMPLATES = {
  lfm3: "https://huggingface.co/LiquidAI/LFM3-1.7B/resolve/main/chat_template.jinja",
  qwen: "https://huggingface.co/Qwen/Qwen3.5-2B/resolve/main/chat_template.jinja",
  lfm25: "https://huggingface.co/LiquidAI/LFM2.5-1.2B-Instruct-GGUF/resolve/main/qad/chat_template.jinja",
  lfm25Onnx: "https://huggingface.co/LiquidAI/LFM2.5-1.2B-Instruct-ONNX/resolve/main/chat_template.jinja"
}

/** A tokenizer with the model's real chat template, over a one-token vocabulary: only its prompts are checked. */
function withTemplate(url: string) {
  const chatTemplate = (hub as Record<string, { body: unknown }>)[url]?.body as string
  const specials = ["<|startoftext|>", "<|im_start|>", "<|im_end|>", "<|pad|>", "<think>", "</think>"]
  return loadTokenizer({
    tokenizerJson: {
      model: { type: "BPE", vocab: { x: 0 }, merges: [], unk_token: "x" },
      normalizer: null,
      pre_tokenizer: null,
      post_processor: null,
      decoder: null,
      added_tokens: specials.map((content, i) => ({ id: 10 + i, content, special: true, normalized: false }))
    },
    tokenizerConfig: { chat_template: chatTemplate, bos_token: "<|startoftext|>", eos_token: "<|im_end|>", pad_token: "<|pad|>" }
  } as never)
}

const call: ChatMessage = {
  role: "assistant",
  content: "",
  tool_calls: [{ id: "c", type: "function", function: { name: "get_weather", arguments: { city: "Oslo" } } }]
}

test("the templates that write past tool calls, and a turn's second one (llama.cpp's default for parallel calls)", () => {
  for (const url of [TEMPLATES.lfm3, TEMPLATES.qwen, TEMPLATES.lfm25]) {
    assert.equal(withTemplate(url).writesToolCalls(), true, url)
    assert.equal(withTemplate(url).writesParallelToolCalls(), true, url)
  }
  assert.equal(withTemplate(TEMPLATES.lfm25Onnx).writesToolCalls(), false)
  assert.equal(withTemplate(TEMPLATES.lfm25Onnx).writesParallelToolCalls(), false)
})

test("LFM2.5's template writes a past call itself; where a template can't, the call goes into the turn's text", () => {
  const chat: Array<ChatMessage> = [{ role: "user", content: "Weather?" }, call, { role: "tool", content: "18C", tool_call_id: "c" }]
  assert.match(withTemplate(TEMPLATES.lfm25).applyChatTemplate(chat), /<\|tool_call_start\|>\[get_weather\(city='Oslo'\)\]<\|tool_call_end\|>/)
  const onnx = withTemplate(TEMPLATES.lfm25Onnx)
  assert.doesNotMatch(onnx.applyChatTemplate(chat), /get_weather/, "the ONNX template drops tool_calls")
  assert.match(onnx.applyChatTemplate(withCallsInText(chat, "lfm2")), /<\|tool_call_start\|>\[get_weather\(city="Oslo"\)\]<\|tool_call_end\|>/)
  // Formats whose templates write tool_calls are left as they are.
  assert.equal(withCallsInText(chat, "lfm3"), chat)
})

test("a last assistant message is continued as llama.cpp continues it", () => {
  const hi: ChatMessage = { role: "user", content: "Hi" }
  const lfm3 = withTemplate(TEMPLATES.lfm3)
  const asked = lfm3.applyChatTemplate([hi], { addGenerationPrompt: true })
  // Text: the generation prompt cut where it opens a thought, the thought closed, then the text.
  const text = lfm3.applyChatTemplate([hi, { role: "assistant", content: "Hel" }], { continueFinalMessage: true })
  assert.equal(text, asked.slice(0, asked.lastIndexOf("<think>")) + "<think></think>Hel")
  // Thinking and no text yet: the thought stays open.
  const thinking = lfm3.applyChatTemplate([hi, { role: "assistant", content: "", reasoning_content: "Hmm" }], { continueFinalMessage: true })
  assert.ok(thinking.endsWith("<|im_start|>assistant\n<think>Hmm"), JSON.stringify(thinking.slice(-40)))
  // Qwen3.5 with thinking off writes `<think>\n\n</think>\n\n`; a continuation writes its own thought instead.
  const qwen = withTemplate(TEMPLATES.qwen)
  const continued = qwen.applyChatTemplate([hi, { role: "assistant", content: "Hel" }], { continueFinalMessage: true })
  assert.ok(continued.endsWith("<|im_start|>assistant\n<think></think>Hel"), JSON.stringify(continued.slice(-40)))
  // A template without thinking tags: the generation prompt, then the text.
  const onnx = withTemplate(TEMPLATES.lfm25Onnx)
  assert.equal(
    onnx.applyChatTemplate([hi, { role: "assistant", content: "Hel" }], { continueFinalMessage: true }),
    onnx.applyChatTemplate([hi], { addGenerationPrompt: true }) + "Hel"
  )
})
