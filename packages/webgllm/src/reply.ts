/**
 * Reading LFM3's replies as they stream: thinking (`<think>…</think>`), the
 * answer, and tool calls in LFM3's format:
 *
 * `<tool_call>\n<function=get_weather>\n<parameter=city>\nTokyo\n</parameter>\n</function>\n</tool_call>`
 */

export type ParsedToolCall = { name: string; args: Record<string, unknown> }

export const THINK_OPEN = "<think>"
export const THINK_CLOSE = "</think>"
const TOOL_CALL = "<tool_call>"

/** Length of the longest end of `text` that could be the start of `tag`. */
function partialTag(text: string, tag: string) {
  for (let n = Math.min(tag.length - 1, text.length); n > 0; n--) {
    if (tag.startsWith(text.slice(-n))) return n
  }
  return 0
}

export type ReplyPiece = { kind: "thinking" | "answer"; text: string }

/**
 * Splits a reply into thinking and answer while it streams. A reply that
 * starts with `<think>` is thinking until `</think>`; the rest is the
 * answer, with any stray close left out. Tags split across tokens are held
 * back until they're complete.
 */
export function createThinkingSplitter() {
  let state: "start" | "thinking" | "answer" = "start"
  let buffer = ""
  let thought = false
  let answered = false

  const answer = (text: string): Array<ReplyPiece> => {
    const out = answered ? text : text.trimStart()
    if (!out) return []
    answered = true
    return [{ kind: "answer", text: out }]
  }

  function push(text: string): Array<ReplyPiece> {
    buffer += text
    const pieces: Array<ReplyPiece> = []
    for (;;) {
      if (state === "start") {
        const lead = buffer.trimStart()
        if (lead.startsWith(THINK_OPEN)) {
          buffer = lead.slice(THINK_OPEN.length)
          state = "thinking"
        } else if (THINK_OPEN.startsWith(lead)) {
          return pieces
        } else {
          state = "answer"
        }
      } else if (state === "thinking") {
        const end = buffer.indexOf(THINK_CLOSE)
        if (end === -1) {
          const keep = partialTag(buffer, THINK_CLOSE)
          const done = buffer.slice(0, buffer.length - keep)
          buffer = buffer.slice(done.length)
          const text = thought ? done : done.trimStart()
          if (text) {
            thought = true
            pieces.push({ kind: "thinking", text })
          }
          return pieces
        }
        const text = thought
          ? buffer.slice(0, end)
          : buffer.slice(0, end).trimStart()
        if (text) {
          thought = true
          pieces.push({ kind: "thinking", text })
        }
        buffer = buffer.slice(end + THINK_CLOSE.length)
        state = "answer"
      } else {
        // A close in the answer is dropped: told not to think, the model
        // sometimes still thinks a little and closes it.
        buffer = buffer.split(THINK_CLOSE).join("")
        const keep = partialTag(buffer, THINK_CLOSE)
        pieces.push(...answer(buffer.slice(0, buffer.length - keep)))
        buffer = buffer.slice(buffer.length - keep)
        return pieces
      }
    }
  }

  function flush(): Array<ReplyPiece> {
    const rest = buffer
    buffer = ""
    if (!rest) return []
    return state === "thinking"
      ? [{ kind: "thinking", text: rest }]
      : answer(rest)
  }

  return { push, flush }
}

/** Splits thinking from the answer in a whole reply. */
export function splitThinking(text: string) {
  const splitter = createThinkingSplitter()
  const pieces = [...splitter.push(text), ...splitter.flush()]
  return {
    thinking: pieces
      .filter((p) => p.kind === "thinking")
      .map((p) => p.text)
      .join("")
      .trim(),
    answer: pieces
      .filter((p) => p.kind === "answer")
      .map((p) => p.text)
      .join("")
      .trim()
  }
}

/**
 * Splits a streaming answer into the text before the first tool call (shown
 * as it comes) and the tool-call markup after it (parsed at the end).
 */
export function createToolCallSplitter() {
  let held = ""
  let markup: string | null = null
  return {
    push(text: string) {
      if (markup !== null) {
        markup += text
        return ""
      }
      held += text
      const at = held.indexOf(TOOL_CALL)
      if (at >= 0) {
        const before = held.slice(0, at)
        markup = held.slice(at)
        held = ""
        return before
      }
      const keep = partialTag(held, TOOL_CALL)
      const out = held.slice(0, held.length - keep)
      held = held.slice(out.length)
      return out
    },
    flush() {
      const rest = held
      held = ""
      return { text: rest, markup }
    }
  }
}

/** A parameter's text as a value: JSON where it parses, else the text. */
function parseValue(raw: string): unknown {
  const s = raw.trim()
  if (s === "") return ""
  try {
    return JSON.parse(s)
  } catch {
    return raw
  }
}

/** Every tool call in LFM3's markup. */
export function parseToolCalls(text: string): Array<ParsedToolCall> {
  const calls: Array<ParsedToolCall> = []
  for (const m of text.matchAll(/<tool_call>([\s\S]*?)(?:<\/tool_call>|$)/g)) {
    for (const f of m[1].matchAll(
      /<function=([^>\n]+)>([\s\S]*?)(?:<\/function>|$)/g
    )) {
      const args: Record<string, unknown> = {}
      for (const p of f[2].matchAll(
        /<parameter=([^>\n]+)>\n?([\s\S]*?)\n?<\/parameter>/g
      )) {
        args[p[1].trim()] = parseValue(p[2])
      }
      calls.push({ name: f[1].trim(), args })
    }
  }
  return calls
}
