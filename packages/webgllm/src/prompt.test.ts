import assert from "node:assert/strict"
import { describe, test } from "node:test"
import { anchorFor, promptPlan, sharedPrefix, unsharedTail } from "./prompt.js"
import type { ChatMessage, Tokenizer } from "./tokenizer.js"

// Token ids for a toy chat template: words map to ids, specials below 10.
const IM_START = 1
const IM_END = 2
const THINK = 3
const END_THINK = 4
const NL = 5
const words = new Map<string, number>()
const word = (w: string) => {
  if (!words.has(w)) words.set(w, 100 + words.size)
  return words.get(w) as number
}
const text = (s: string) => s.split(" ").filter(Boolean).map(word)

/**
 * A tokenizer whose template writes each message as <|im_start|> role \n
 * content <|im_end|>. Qwen-like (`rewrites`): the generation prompt opens
 * <think> \n </think> that a past reply loses; LFM3-like: the generation
 * prompt is <think> </think> and past replies keep it.
 */
function toyTokenizer(rewrites: boolean): Tokenizer {
  const encodeChat = (messages: Array<ChatMessage>, options?: { addGenerationPrompt?: boolean }) => {
    const out: Array<number> = []
    for (const m of messages) {
      out.push(IM_START, word(m.role), NL)
      if (m.role === "assistant" && !rewrites) out.push(THINK, END_THINK)
      out.push(...text(typeof m.content === "string" ? m.content : ""), IM_END, NL)
    }
    if (options?.addGenerationPrompt) {
      out.push(IM_START, word("assistant"), NL)
      out.push(...(rewrites ? [THINK, NL, END_THINK, NL] : [THINK, END_THINK]))
    }
    return out
  }
  return {
    encodeChat,
    specialTokens: { imStart: IM_START }
  } as unknown as Tokenizer
}

describe("sharedPrefix", () => {
  test("counts the leading items both have", () => {
    assert.equal(sharedPrefix([1, 2, 3], [1, 2, 4, 5]), 2)
    assert.equal(sharedPrefix([1, 2], [1, 2, 3]), 2)
    assert.equal(sharedPrefix([], [1]), 0)
  })
})

describe("unsharedTail and anchorFor", () => {
  test("a template that rewrites past replies: the generation prompt's tail, anchored at its <|im_start|>", () => {
    const tok = toyTokenizer(true)
    assert.equal(unsharedTail(tok, false), 4)
    const chat: Array<ChatMessage> = [{ role: "user", content: "a long question" }]
    const prompt = tok.encodeChat(chat, { addGenerationPrompt: true })
    const anchor = anchorFor(tok, prompt, chat, false)
    assert.equal(prompt[anchor as number], IM_START)
    assert.equal(anchor, prompt.length - 7)
    // The next question's prompt shares everything before the anchor.
    const next = tok.encodeChat(
      [...chat, { role: "assistant", content: "an answer" }, { role: "user", content: "more" }],
      { addGenerationPrompt: true }
    )
    assert.ok(sharedPrefix(prompt, next) >= (anchor as number))
  })

  test("a template that writes replies back as asked: the anchor at the prompt's end", () => {
    const tok = toyTokenizer(false)
    assert.equal(unsharedTail(tok, false), 0)
    const chat: Array<ChatMessage> = [{ role: "user", content: "a question" }]
    const prompt = tok.encodeChat(chat, { addGenerationPrompt: true })
    assert.equal(anchorFor(tok, prompt, chat, false), prompt.length)
  })

  test("no anchor after a tool's result: the last question's stays", () => {
    const tok = toyTokenizer(true)
    const chat: Array<ChatMessage> = [
      { role: "user", content: "weather" },
      { role: "assistant", content: "call" },
      { role: "tool", content: "sunny" }
    ]
    const prompt = tok.encodeChat(chat, { addGenerationPrompt: true })
    assert.equal(anchorFor(tok, prompt, chat, false), undefined)
  })
})

describe("promptPlan", () => {
  const sizes = (n: number) => {
    const out: Array<number> = []
    for (; n > 576; n -= 512) out.push(512)
    if (n > 0) out.push(n)
    return out
  }

  test("a batch ends at the anchor, the rest read after it", () => {
    assert.deepEqual(promptPlan(sizes, 560, 0, 553, false), [
      { size: 553, snapshot: "anchor" },
      { size: 7, snapshot: "end" }
    ])
    assert.deepEqual(promptPlan(sizes, 600, 0, 593, false), [
      { size: 512, snapshot: "batch" },
      { size: 81, snapshot: "anchor" },
      { size: 7, snapshot: "end" }
    ])
  })

  test("with few snapshots kept, only the anchor's", () => {
    assert.deepEqual(promptPlan(sizes, 1300, 0, 1290, true), [
      { size: 512, snapshot: null },
      { size: 512, snapshot: null },
      { size: 266, snapshot: "anchor" },
      { size: 10, snapshot: null }
    ])
  })

  test("an anchor saved from inside its batch splits nothing", () => {
    assert.deepEqual(promptPlan(sizes, 560, 0, 553, true, true), [
      { size: 560, snapshot: null, saveAt: 553 }
    ])
    assert.deepEqual(promptPlan(sizes, 1300, 100, 1290, false, true), [
      { size: 512, snapshot: "batch" },
      { size: 512, snapshot: "batch" },
      { size: 176, snapshot: "end", saveAt: 166 }
    ])
    // At a batch's end: a snapshot after it, as without `inside`.
    assert.deepEqual(promptPlan(sizes, 1100, 0, 512, true, true), [
      { size: 512, snapshot: "anchor" },
      { size: 512, snapshot: null },
      { size: 76, snapshot: null }
    ])
  })

  test("an anchor at the end, before the start or missing splits nothing", () => {
    assert.deepEqual(promptPlan(sizes, 300, 0, 300, true), [{ size: 300, snapshot: "anchor" }])
    assert.deepEqual(promptPlan(sizes, 300, 100, 50, false), [{ size: 200, snapshot: "end" }])
    assert.deepEqual(promptPlan(sizes, 1100, 0, undefined, false), [
      { size: 512, snapshot: "batch" },
      { size: 512, snapshot: "batch" },
      { size: 76, snapshot: "end" }
    ])
  })
})
