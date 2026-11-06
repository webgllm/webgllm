/**
 * Stop sequences on a streaming reply: the reply ends where any of them
 * first appears in its answer. Thinking (`<think>…</think>`) passes through
 * unmatched, read as reply.ts reads it: a reply starting with `<think>` is
 * thinking until `</think>`, and whitespace and empty `<think></think>`
 * blocks before the first answer text aren't answer (the page trims them).
 * Text is passed through raw, markers and all; answer text that could be
 * the start of a stop sequence is held back until it's resolved.
 */
import { THINK_CLOSE, THINK_OPEN } from "./reply.js"

export type StopFilter = {
  /** The text that can be sent now; `stopped` once a stop sequence ended the reply. */
  push(text: string): { text: string; stopped: boolean }
  /** What was held back, at the end of a reply no stop sequence ended. */
  flush(): string
}

/** Whether `text` is a proper start of `tag` (more may complete it). */
const partOf = (text: string, tag: string) => text.length < tag.length && tag.startsWith(text)

/** Length of the longest end of `text` that is a proper start of one of `tags`. */
function heldEnd(text: string, tags: ReadonlyArray<string>, longest: number) {
  for (let n = Math.min(longest - 1, text.length); n > 0; n--) {
    const end = text.slice(-n)
    if (tags.some((tag) => partOf(end, tag))) return n
  }
  return 0
}

/** A filter for `stops` (empty strings ignored), or null if there are none. */
export function createStopFilter(stops: ReadonlyArray<string> | undefined): StopFilter | null {
  const sequences = [...new Set(stops ?? [])].filter((s) => s.length > 0)
  if (!sequences.length) return null
  const longest = Math.max(...sequences.map((s) => s.length))
  let state: "start" | "thinking" | "before" | "answer" = "start"
  // Text not yet sent: undecided markers, or answer that may start a stop.
  let buffer = ""
  let stopped = false

  const firstStop = () => {
    let at = -1
    for (const s of sequences) {
      const i = buffer.indexOf(s)
      if (i >= 0 && (at < 0 || i < at)) at = i
    }
    return at
  }

  // The answer so far is matched; returns what can be sent.
  const answer = () => {
    const at = firstStop()
    if (at >= 0) {
      stopped = true
      const out = buffer.slice(0, at)
      buffer = ""
      return out
    }
    const keep = heldEnd(buffer, sequences, longest)
    const out = buffer.slice(0, buffer.length - keep)
    buffer = buffer.slice(out.length)
    return out
  }

  function push(text: string) {
    if (stopped) return { text: "", stopped }
    buffer += text
    let out = ""
    const send = (n: number) => {
      out += buffer.slice(0, n)
      buffer = buffer.slice(n)
    }
    for (;;) {
      if (state === "start") {
        const lead = buffer.trimStart()
        if (lead.startsWith(THINK_OPEN)) {
          send(buffer.length - lead.length + THINK_OPEN.length)
          state = "thinking"
        } else if (THINK_OPEN.startsWith(lead)) break
        else state = "before"
      } else if (state === "thinking") {
        const end = buffer.indexOf(THINK_CLOSE)
        if (end < 0) {
          send(buffer.length - heldEnd(buffer, [THINK_CLOSE], THINK_CLOSE.length))
          break
        }
        send(end + THINK_CLOSE.length)
        state = "before"
      } else if (state === "before") {
        const lead = buffer.trimStart()
        const space = buffer.length - lead.length
        if (!lead || partOf(lead, THINK_OPEN) || partOf(lead, THINK_CLOSE)) {
          send(space)
          break
        }
        if (lead.startsWith(THINK_CLOSE)) {
          send(space + THINK_CLOSE.length)
          continue
        }
        if (lead.startsWith(THINK_OPEN)) {
          const after = lead.slice(THINK_OPEN.length).trimStart()
          const through = buffer.length - after.length
          if (partOf(after, THINK_CLOSE)) {
            send(space)
            break
          }
          if (after.startsWith(THINK_CLOSE)) {
            send(through + THINK_CLOSE.length)
            continue
          }
        }
        send(space)
        state = "answer"
      } else {
        out += answer()
        break
      }
    }
    return { text: out, stopped }
  }

  function flush() {
    if (stopped) return ""
    let out = ""
    // Text still undecided at the end is answer (as reply.ts reads it).
    if (state === "start" || state === "before") {
      const lead = buffer.trimStart()
      out = buffer.slice(0, buffer.length - lead.length)
      buffer = lead
      state = "answer"
    }
    if (state === "answer") {
      const at = firstStop()
      if (at >= 0) {
        stopped = true
        buffer = buffer.slice(0, at)
      }
    }
    out += buffer
    buffer = ""
    return out
  }

  return { push, flush }
}
