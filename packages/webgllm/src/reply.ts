/**
 * Reading Liquid AI models' replies as they stream: thinking
 * (`<think>…</think>`, in LFM2 and LFM3 alike) and the answer. Tool calls
 * are read in tool-calls.ts.
 */

export const THINK_OPEN = "<think>"
export const THINK_CLOSE = "</think>"

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
        if (!answered) {
          // A budgeted model may reopen thinking and immediately be closed again.
          // Strip only complete empty blocks before the first visible answer.
          const lead = buffer.trimStart()
          if ((lead.length < THINK_OPEN.length && THINK_OPEN.startsWith(lead)) || (lead.length < THINK_CLOSE.length && THINK_CLOSE.startsWith(lead))) return pieces
          if (lead.startsWith(THINK_CLOSE)) {
            buffer = lead.slice(THINK_CLOSE.length)
            continue
          }
          if (lead.startsWith(THINK_OPEN)) {
            const afterOpen = lead.slice(THINK_OPEN.length).trimStart()
            if (afterOpen.length < THINK_CLOSE.length && THINK_CLOSE.startsWith(afterOpen)) return pieces
            if (afterOpen.startsWith(THINK_CLOSE)) {
              buffer = afterOpen.slice(THINK_CLOSE.length)
              continue
            }
          }
        }
        // Once answer text starts, marker-shaped content is literal data.
        pieces.push(...answer(buffer))
        buffer = ""
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
