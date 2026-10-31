/**
 * Reading and writing tool calls in Liquid AI's formats:
 *
 * - LFM2 and LFM2.5: `<|tool_call_start|>[get_weather(city="Tokyo")]<|tool_call_end|>`
 * - LFM3: `<tool_call>\n<function=get_weather>\n<parameter=city>\nTokyo\n</parameter>\n</function>\n</tool_call>`
 */

export type ParsedToolCall = { name: string; args: Record<string, unknown> }

/** How a model writes tool calls. */
export type ToolFormat = "lfm2" | "lfm3"

/** Where a tool call starts, in either format. */
const MARKERS = ["<|tool_call_start|>", "<tool_call>"]

/** A JSON number exactly (so "02134" or "0x1F" stay strings). */
const NUMBER = /^-?(?:0|[1-9]\d*)(?:\.\d+)?(?:[eE][+-]?\d+)?$/

/** A Python string literal's text: its escapes decoded, quotes unescaped. */
function stringText(raw: string) {
  const inner = raw.slice(1, -1)
  let out = ""
  for (let i = 0; i < inner.length; i++) {
    const c = inner[i]
    if (c !== "\\" || i + 1 === inner.length) {
      out += c
      continue
    }
    const next = inner[++i]
    if (next === "n") out += "\n"
    else if (next === "t") out += "\t"
    else if (next === "r") out += "\r"
    else if (next === "u" && /^[0-9a-fA-F]{4}$/.test(inner.slice(i + 1, i + 5))) {
      out += String.fromCharCode(parseInt(inner.slice(i + 1, i + 5), 16))
      i += 4
    } else out += next
  }
  return out
}

/**
 * A Python literal list or dict as JSON: single-quoted strings double-quoted,
 * True / False / None as true / false / null (outside strings).
 */
function pythonToJson(text: string) {
  let out = ""
  for (let i = 0; i < text.length; i++) {
    const c = text[i]
    if (c === '"' || c === "'") {
      const end = stringEnd(text, i, INNER_END)
      if (end < 0) return text
      out += JSON.stringify(stringText(text.slice(i, end + 1)))
      i = end
      continue
    }
    const word = /^(True|False|None)\b/.exec(text.slice(i))
    if (word && !/[\w.]/.test(text[i - 1] ?? "")) {
      out += word[1] === "True" ? "true" : word[1] === "False" ? "false" : "null"
      i += word[1].length - 1
      continue
    }
    out += c
  }
  return out
}

/** A bare Python value: a number, True, None, a list… or else the text. */
function parseValue(raw: string): unknown {
  const s = raw.trim()
  if (s === "True" || s === "true") return true
  if (s === "False" || s === "false") return false
  if (s === "None" || s === "null") return null
  if (NUMBER.test(s)) return Number(s)
  if (s.startsWith("[") || s.startsWith("{")) {
    try {
      return JSON.parse(pythonToJson(s))
    } catch {
      return s
    }
  }
  return s
}

/** An LFM3 XML parameter's value: JSON if it parses, else the text as it is. */
function parseXmlValue(raw: string): unknown {
  try {
    return JSON.parse(raw)
  } catch {
    return raw
  }
}

/** After an argument's value: the next argument (`, name=`), or the call's end. */
const VALUE_END = /^\s*(?:,\s*[A-Za-z_]\w*\s*=|,?\s*\)|$)/
/** After a string inside a list or dict: a separator or the bracket's end. */
const INNER_END = /^\s*[,\]}:]/

/**
 * Where the string starting at `body[start]` ends (the index of its closing
 * quote). The model doesn't escape quotes inside its strings ('JWST's
 * findings'), so a quote only closes the string when what follows can come
 * after one (`after`: an argument's end, or a list or dict separator);
 * otherwise it's part of the text.
 */
function stringEnd(body: string, start: number, after = VALUE_END) {
  const quote = body[start]
  let last = -1
  for (let i = start + 1; i < body.length; i++) {
    if (body[i] === "\\") i++
    else if (body[i] === quote) {
      if (after.test(body.slice(i + 1))) return i
      last = i
    }
  }
  return last
}

/** Where a `[…]` or `{…}` starting at `body[start]` ends, strings skipped. */
function bracketEnd(body: string, start: number) {
  let depth = 0
  for (let i = start; i < body.length; i++) {
    const c = body[i]
    if (c === '"' || c === "'") {
      const end = stringEnd(body, i, INNER_END)
      if (end < 0) return -1
      i = end
    } else if (c === "[" || c === "{") depth++
    else if ((c === "]" || c === "}") && --depth === 0) return i
  }
  return -1
}

/** `[name(a="x", b=2), other()]` → calls. */
function parsePythonic(body: string) {
  const calls: Array<ParsedToolCall> = []
  const callStart = /([A-Za-z_][\w.-]*)\(/g
  let m: RegExpExecArray | null
  while ((m = callStart.exec(body))) {
    const args: Record<string, unknown> = {}
    let i = m.index + m[0].length
    for (;;) {
      const key = /^\s*([A-Za-z_]\w*)\s*=\s*/.exec(body.slice(i))
      if (!key) break
      i += key[0].length
      const first = body[i]
      let end: number
      if (first === '"' || first === "'") {
        end = stringEnd(body, i)
        if (end < 0) break
        args[key[1]] = stringText(body.slice(i, end + 1))
      } else if (first === "[" || first === "{") {
        end = bracketEnd(body, i)
        if (end < 0) break
        args[key[1]] = parseValue(body.slice(i, end + 1))
      } else {
        const rest = /^[^,)]*/.exec(body.slice(i))?.[0] ?? ""
        end = i + rest.length - 1
        args[key[1]] = parseValue(rest)
      }
      i = end + 1
      const comma = /^\s*,/.exec(body.slice(i))
      if (!comma) break
      i += comma[0].length
    }
    calls.push({ name: m[1], args })
    // The next call starts after this one's closing parenthesis.
    const close = body.indexOf(")", i)
    callStart.lastIndex = close < 0 ? body.length : close + 1
  }
  return calls
}

/** `<function=name><parameter=k>v</parameter></function>` → calls. */
function parseXmlish(body: string) {
  const calls: Array<ParsedToolCall> = []
  for (const f of body.matchAll(
    /<function=([^>\n]+)>([\s\S]*?)(?:<\/function>|$)/g
  )) {
    const args: Record<string, unknown> = {}
    for (const p of f[2].matchAll(
      /<parameter=([^>\n]+)>\n?([\s\S]*?)\n?<\/parameter>/g
    )) {
      args[p[1].trim()] = parseXmlValue(p[2])
    }
    calls.push({ name: f[1].trim(), args })
  }
  return calls
}

/** Every tool call in `text`, in LFM3's or LFM2's format. */
export function parseToolCalls(text: string): Array<ParsedToolCall> {
  const calls: Array<ParsedToolCall> = []
  for (const m of text.matchAll(/<tool_call>([\s\S]*?)(?:<\/tool_call>|$)/g)) {
    const body = m[1].trim()
    if (body.includes("<function=")) {
      calls.push(...parseXmlish(body))
      continue
    }
    // A `<tool_call>` that isn't LFM3's markup: JSON, or a Python-style call.
    try {
      const json = JSON.parse(body) as {
        name: string
        arguments?: unknown
        parameters?: unknown
      }
      const args =
        typeof json.arguments === "string"
          ? JSON.parse(json.arguments)
          : (json.arguments ?? json.parameters ?? {})
      calls.push({ name: json.name, args: args as Record<string, unknown> })
    } catch {
      calls.push(...parsePythonic(body))
    }
  }
  for (const m of text.matchAll(
    /<\|tool_call_start\|>([\s\S]*?)(?:<\|tool_call_end\|>|$)/g
  )) {
    calls.push(...parsePythonic(m[1]))
  }
  return calls
}

/** Length of the longest end of `text` that could start one of `markers`. */
function partialMarker(text: string, markers: Array<string>) {
  const longest = Math.max(...markers.map((m) => m.length))
  for (let n = Math.min(text.length, longest - 1); n > 0; n--) {
    const end = text.slice(-n)
    if (markers.some((m) => m.startsWith(end))) return n
  }
  return 0
}

/**
 * Splits a streaming answer into the text before the first tool call (shown
 * as it comes) and the tool-call markup after it (parsed at the end).
 */
export function createToolCallSplitter(markers: Array<string> = MARKERS) {
  let held = ""
  let markup: string | null = null
  return {
    push(text: string) {
      if (markup !== null) {
        markup += text
        return ""
      }
      held += text
      const at = Math.min(
        ...markers.map((m) => held.indexOf(m)).filter((i) => i >= 0)
      )
      if (Number.isFinite(at)) {
        const before = held.slice(0, at)
        markup = held.slice(at)
        held = ""
        return before
      }
      const keep = partialMarker(held, markers)
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

/** A value written the way Python (and LFM2) writes it. */
function pythonValue(value: unknown): string {
  if (value === true) return "True"
  if (value === false) return "False"
  if (value === null || value === undefined) return "None"
  return JSON.stringify(value)
}

/** Past tool calls in LFM2's own format, for the assistant turn's text. */
export function formatLfm2ToolCalls(calls: Array<ParsedToolCall>) {
  const list = calls
    .map(
      (c) =>
        `${c.name}(${Object.entries(c.args)
          .map(([k, v]) => `${k}=${pythonValue(v)}`)
          .join(", ")})`
    )
    .join(", ")
  return `<|tool_call_start|>[${list}]<|tool_call_end|>`
}
