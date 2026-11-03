/**
 * How the worker reads a prompt into the engine: in forward batches from
 * where the cache stops sharing it, with snapshots so a later prompt can
 * resume (see Engine.snapshot). The bench pages read prompts through this
 * too, so they measure what the worker does.
 */
import type { Engine } from "./engine.js"
import type { ChatMessage, ChatTool, Tokenizer } from "./tokenizer.js"

/** How many leading items `a` and `b` share. */
export function sharedPrefix(a: ArrayLike<number>, b: ArrayLike<number>) {
  let n = 0
  while (n < a.length && n < b.length && a[n] === b[n]) n++
  return n
}

const tails = new WeakMap<Tokenizer, { key: string; tail: number }>()
const prefixes = new WeakMap<Tokenizer, { key: string; tokens: Array<number> }>()

/**
 * A reusable system/tools boundary, measured from the actual chat template and
 * verified token-for-token against this prompt. User-required templates use two
 * tokenize-only probes; unsupported templates keep conversational caching.
 */
export function systemPrefixFor(
  tokenizer: Tokenizer,
  prompt: Array<number>,
  chat: Array<ChatMessage>,
  thinking: boolean | undefined,
  tools?: Array<ChatTool>
): number | undefined {
  const first = chat.findIndex(message => message.role !== "system")
  const system = first < 0 ? chat : chat.slice(0, first)
  if (!system.length && !tools?.length) return undefined
  const key = JSON.stringify({ system, thinking, tools })
  let known = prefixes.get(tokenizer)
  if (!known || known.key !== key) {
    const options = {
      addGenerationPrompt: false,
      ...(tools?.length && { tools }),
      ...(thinking !== undefined && { enableThinking: thinking })
    }
    let tokens: Array<number>
    try {
      tokens = tokenizer.encodeChat(system, options)
    } catch {
      // Some templates (Qwen3.5) require a user query even when only their
      // system/tool prefix is wanted. Probe tokenization only, never inference.
      // The actual request stays unchanged and is checked against this prefix.
      try {
        const a = tokenizer.encodeChat([...system, { role: "user", content: "A" }], options)
        const b = tokenizer.encodeChat([...system, { role: "user", content: "B" }], options)
        tokens = a.slice(0, sharedPrefix(a, b))
      } catch { tokens = [] }
    }
    known = { key, tokens }
    prefixes.set(tokenizer, known)
  }
  const length = Math.min(sharedPrefix(known.tokens, prompt), prompt.length - 1)
  return length > 0 ? length : undefined
}


/**
 * How many tokens at the end of a prompt the next user turn's prompt won't
 * share: the generation prompt, where the template writes a past reply
 * back differently. Qwen3.5's opens `<think>` tags (empty with thinking
 * off) that a reply before the latest question loses, so a follow-up
 * stops sharing the chat right after `<|im_start|>assistant\n`; LFM3's
 * writes its reply back as it was asked for (0). Found once per template
 * and thinking setting, from a short probe chat with a reply and a
 * question after it.
 */
export function unsharedTail(
  tokenizer: Tokenizer,
  thinking: boolean | undefined,
  tools?: Array<ChatTool>
) {
  const key = JSON.stringify({ thinking, tools })
  const known = tails.get(tokenizer)
  if (known?.key === key) return known.tail
  const options = {
    addGenerationPrompt: true,
    ...(tools?.length && { tools }),
    ...(thinking !== undefined && { enableThinking: thinking })
  }
  const asked: Array<ChatMessage> = [{ role: "user", content: "Hi." }]
  const prompt = tokenizer.encodeChat(asked, options)
  const next = tokenizer.encodeChat(
    [...asked, { role: "assistant", content: "Hello." }, { role: "user", content: "Thanks." }],
    options
  )
  const tail = prompt.length - sharedPrefix(prompt, next)
  tails.set(tokenizer, { key, tail })
  return tail
}

/**
 * Where to keep the anchor snapshot for `prompt` (see Engine.snapshot):
 * where the next user turn's prompt will stop sharing it, if it asks a
 * question (a tool's result leaves the last question's anchor in place:
 * the reply after it is written back as it was). Past the unshared tail,
 * at its `<|im_start|>` where it has one (a special token: no merge with
 * the reply can move the boundary). Undefined for no anchor.
 */
export function anchorFor(
  tokenizer: Tokenizer,
  prompt: Array<number>,
  chat: Array<{ role: string }>,
  thinking: boolean | undefined,
  tools?: Array<ChatTool>
) {
  if (chat[chat.length - 1]?.role !== "user") return undefined
  const tail = unsharedTail(tokenizer, thinking, tools)
  if (tail <= 0) return prompt.length
  const imStart = tokenizer.specialTokens.imStart
  const from = prompt.length - tail
  for (let i = prompt.length - 1; i >= Math.max(0, from - 8); i--)
    if (prompt[i] === imStart && i <= from) return i
  return from
}

/**
 * The batches for reading prompt[start..]: Engine.batchSizes, and whether
 * to snapshot after each (the anchor always; with `few` snapshots kept,
 * nothing else, else after every batch, the last with its hidden row). An
 * anchor inside a batch is saved from inside it (`saveAt` tokens in) where
 * the engine can (`inside`), else the batches split there.
 */
export type PromptReadStep = {
  size: number
  snapshot: "prefix" | "anchor" | "end" | "batch" | null
  saveAt?: number
  /** A natural endpoint can serve both ordinary conversation and static prefix. */
  prefix?: boolean
}

export function promptPlan(
  batchSizes: (count: number) => Array<number>,
  length: number,
  start: number,
  anchor: number | undefined,
  few: boolean,
  inside = false
): Array<PromptReadStep> {
  const within = anchor !== undefined && anchor > start && anchor < length
  const sizes =
    within && !inside
      ? [...batchSizes(anchor - start), ...batchSizes(length - anchor)]
      : batchSizes(length - start)
  const plan: Array<PromptReadStep> = []
  let at = start
  for (const size of sizes) {
    const from = at
    at += size
    const saveAt =
      within && inside && anchor > from && anchor < at ? anchor - from : undefined
    plan.push({
      size,
      snapshot:
        at === anchor ? "anchor" : few ? null : at === length ? "end" : "batch",
      ...(saveAt !== undefined && { saveAt })
    })
  }
  return plan
}

/** Protect an existing legacy batch endpoint; never change batching to capture a prefix. */
export function naturalPromptPlan(
  batchSizes: (count: number) => Array<number>, length: number, start: number,
  anchor: number | undefined, few: boolean, inside: boolean, prefix?: number
): Array<PromptReadStep> {
  const plan = promptPlan(batchSizes, length, start, anchor, few, inside)
  let at = start
  let prefixAt: number | undefined
  for (const step of plan) { at += step.size; if (prefix !== undefined && at <= prefix) prefixAt = at }
  at = start
  return plan.map(step => {
    at += step.size
    const stable = at === prefixAt
    return { ...step, snapshot: stable && step.snapshot !== "anchor" ? "prefix" : step.snapshot,
      ...(stable && step.snapshot === "anchor" && { prefix: true }) }
  })
}

/**
 * Reads prompt[start..] into `engine` (whose cache holds prompt[..start])
 * as promptPlan says, appending each batch to `cached` once it's read.
 * Stops after a batch if `stop` says so. Returns the last batch's row of
 * the last token read (see Engine.forwardHidden).
 */
export async function readPrompt(
  engine: Engine,
  prompt: Array<number>,
  start: number,
  cached: Array<number>,
  options: { stop?: () => boolean; anchor?: number; prefix?: number; checkpoints?: boolean } = {}
): Promise<number> {
  const few = engine.snapshotLimit < 4
  let row = 0
  let i = start
  const plan: Array<PromptReadStep> = options.checkpoints === false
    ? engine.batchSizes(prompt.length - start).map(size => ({ size, snapshot: null }))
    : naturalPromptPlan(n => engine.batchSizes(n), prompt.length, start, options.anchor, few, engine.canSaveInside, engine.snapshotLimit >= 2 ? options.prefix : undefined)
  for (const { size, snapshot, saveAt, prefix: alsoPrefix } of plan) {
    const batch = prompt.slice(i, i + size)
    i += size
    row = await engine.forwardHidden(batch, saveAt !== undefined ? { saveAt } : {})
    cached.push(...batch)
    const end = i === prompt.length ? row : undefined
    if (snapshot === "prefix") engine.snapshot(end, "prefix")
    else if (snapshot === "anchor") engine.snapshot(end, true)
    else if (snapshot) engine.snapshot(end)
    if (alsoPrefix) engine.snapshot(end, "prefix")
    if (options.stop?.()) break
  }
  return row
}
