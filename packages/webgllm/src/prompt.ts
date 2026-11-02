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

const tails = new WeakMap<Tokenizer, Map<string, number>>()

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
  const key = `${thinking}`
  let byKey = tails.get(tokenizer)
  if (!byKey) tails.set(tokenizer, (byKey = new Map()))
  const known = byKey.get(key)
  if (known !== undefined) return known
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
  byKey.set(key, tail)
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
export function promptPlan(
  batchSizes: (count: number) => Array<number>,
  length: number,
  start: number,
  anchor: number | undefined,
  few: boolean,
  inside = false
) {
  const within = anchor !== undefined && anchor > start && anchor < length
  const sizes =
    within && !inside
      ? [...batchSizes(anchor - start), ...batchSizes(length - anchor)]
      : batchSizes(length - start)
  const plan: Array<{
    size: number
    snapshot: "anchor" | "end" | "batch" | null
    saveAt?: number
  }> = []
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
  options: { stop?: () => boolean; anchor?: number } = {}
): Promise<number> {
  const few = engine.snapshotLimit < 4
  let row = 0
  let i = start
  for (const { size, snapshot, saveAt } of promptPlan(
    (n) => engine.batchSizes(n),
    prompt.length,
    start,
    options.anchor,
    few,
    engine.canSaveInside
  )) {
    const batch = prompt.slice(i, i + size)
    i += size
    row = await engine.forwardHidden(batch, saveAt !== undefined ? { saveAt } : {})
    cached.push(...batch)
    const end = i === prompt.length ? row : undefined
    if (snapshot === "anchor") engine.snapshot(end, true)
    else if (snapshot) engine.snapshot(end)
    if (options.stop?.()) break
  }
  return row
}
