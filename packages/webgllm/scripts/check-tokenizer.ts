/**
 * Checks src/tokenizer.ts against Python: ids and decoding against
 * `tokenizers`, prompts against transformers' apply_chat_template, the
 * stream decoder against decode().
 *
 *   node packages/webgl/scripts/check-tokenizer.ts \
 *     --python /path/to/venv/bin/python NAME=DIR [NAME=DIR ...]
 *
 * DIR holds tokenizer.json, tokenizer_config.json and chat_template.jinja
 * (LiquidAI/LFM3-1.7B's for the GGUF build, daveycodez/LFM3-1.7B-Web's for
 * the ONNX one). The Python needs `tokenizers`, `jinja2` and `transformers`.
 * Exits 1 on any mismatch.
 */

import { execFileSync } from "node:child_process"
import { mkdtempSync, readFileSync } from "node:fs"
import { tmpdir } from "node:os"
import { dirname, join } from "node:path"
import { fileURLToPath } from "node:url"
import { loadTokenizer } from "../src/tokenizer.ts"
import type { ChatMessage, ChatTool, Tokenizer } from "../src/tokenizer.ts"

type TextTruth = {
  text: string
  ids: Array<number>
  idsPlain: Array<number>
  decoded: string
  decodedSkip: string
}
type IdsTruth = { ids: Array<number>; decoded: string; decodedSkip: string }
type ChatTruth = {
  name: string
  messages: Array<ChatMessage>
  tools: Array<ChatTool> | null
  kwargs: Record<string, unknown>
  text?: string
  ids?: Array<number>
  error?: string
}
type ModelTruth = {
  name: string
  dir: string
  vocabSize: number
  texts: Array<TextTruth>
  idSequences: Array<IdsTruth>
  chats: Array<ChatTruth>
}

const args = process.argv.slice(2)
const pythonAt = args.indexOf("--python")
const python = pythonAt >= 0 ? args.splice(pythonAt, 2)[1] : "python3"
if (args.length === 0) {
  console.error(
    "Usage: check-tokenizer.ts [--python PATH] NAME=DIR [NAME=DIR ...]"
  )
  process.exit(2)
}

const here = dirname(fileURLToPath(import.meta.url))
const out = join(mkdtempSync(join(tmpdir(), "check-tokenizer-")), "truth.json")
execFileSync(python, ["-I", join(here, "check-tokenizer.py"), out, ...args], {
  stdio: ["ignore", "inherit", "inherit"]
})
const truth = JSON.parse(readFileSync(out, "utf8")) as {
  models: Array<ModelTruth>
}

let failures = 0
const fail = (model: string, what: string, detail: unknown) => {
  failures++
  if (failures <= 40) console.log(`FAIL [${model}] ${what}`, detail)
}
const same = (a: Array<number>, b: Array<number>) =>
  a.length === b.length && a.every((x, i) => x === b[i])

/** Feeds `ids` one at a time; checks the pieces against decode(). */
function checkStream(
  model: string,
  tokenizer: Tokenizer,
  ids: Array<number>,
  skipSpecialTokens: boolean,
  expected: string
) {
  const stream = tokenizer.createStreamDecoder({ skipSpecialTokens })
  const pieces: Array<string> = []
  for (const id of ids) pieces.push(stream.push(id))
  pieces.push(stream.flush())
  const text = pieces.join("")
  if (text !== expected) {
    fail(model, `stream (skip=${skipSpecialTokens})`, { ids, text, expected })
  }
  // No piece splits a character: each is well-formed UTF-16, and only ends
  // in U+FFFD where the whole text has one.
  let at = 0
  for (const piece of pieces) {
    at += piece.length
    if (!piece.isWellFormed())
      fail(model, "stream piece splits a surrogate pair", { ids, piece })
    if (piece.endsWith("�") && expected[at - 1] !== "�") {
      fail(model, "stream piece ends in U+FFFD", { ids, piece })
    }
  }
  return pieces.filter((p) => p !== "").length
}

for (const model of truth.models) {
  const read = (file: string) => readFileSync(join(model.dir, file), "utf8")
  const started = performance.now()
  const tokenizer = loadTokenizer({
    tokenizerJson: JSON.parse(read("tokenizer.json")),
    tokenizerConfig: JSON.parse(read("tokenizer_config.json")),
    chatTemplate: read("chat_template.jinja")
  })
  const loadMs = performance.now() - started
  const counts = { texts: 0, ids: 0, chats: 0, chatErrors: 0, pieces: 0 }

  if (tokenizer.vocabSize !== model.vocabSize) {
    fail(model.name, "vocabSize", {
      ours: tokenizer.vocabSize,
      python: model.vocabSize
    })
  }

  for (const t of model.texts) {
    counts.texts++
    const ids = tokenizer.encode(t.text)
    const plain = tokenizer.encode(t.text, { addSpecialTokens: false })
    if (!same(ids, t.ids))
      fail(model.name, "encode", { text: t.text, ids, expected: t.ids })
    if (!same(plain, t.idsPlain)) {
      fail(model.name, "encode (no special tokens)", {
        text: t.text,
        plain,
        expected: t.idsPlain
      })
    }
    const decoded = tokenizer.decode(t.ids)
    const decodedSkip = tokenizer.decode(t.ids, { skipSpecialTokens: true })
    if (decoded !== t.decoded)
      fail(model.name, "decode", { text: t.text, decoded, expected: t.decoded })
    if (decoded !== t.text)
      fail(model.name, "round trip", { text: t.text, decoded })
    if (decodedSkip !== t.decodedSkip) {
      fail(model.name, "decode (skip special)", {
        text: t.text,
        decodedSkip,
        expected: t.decodedSkip
      })
    }
    counts.pieces += checkStream(model.name, tokenizer, t.ids, false, t.decoded)
    checkStream(model.name, tokenizer, t.ids, true, t.decodedSkip)
  }

  for (const s of model.idSequences) {
    counts.ids++
    const decoded = tokenizer.decode(s.ids)
    const decodedSkip = tokenizer.decode(s.ids, { skipSpecialTokens: true })
    if (decoded !== s.decoded)
      fail(model.name, "decode ids", {
        ids: s.ids,
        decoded,
        expected: s.decoded
      })
    if (decodedSkip !== s.decodedSkip) {
      fail(model.name, "decode ids (skip special)", {
        ids: s.ids,
        decodedSkip,
        expected: s.decodedSkip
      })
    }
    checkStream(model.name, tokenizer, s.ids, false, s.decoded)
    checkStream(model.name, tokenizer, s.ids, true, s.decodedSkip)
  }

  for (const c of model.chats) {
    counts.chats++
    const {
      add_generation_prompt,
      enable_thinking,
      preserve_thinking,
      ...rest
    } = c.kwargs
    const options = {
      ...rest,
      ...(c.tools && { tools: c.tools }),
      addGenerationPrompt: add_generation_prompt as boolean,
      ...(enable_thinking !== undefined && {
        enableThinking: enable_thinking as boolean
      }),
      ...(preserve_thinking !== undefined && {
        preserveThinking: preserve_thinking as boolean
      })
    }
    let text: string | undefined
    let error: string | undefined
    try {
      text = tokenizer.applyChatTemplate(c.messages, options)
    } catch (e) {
      error = e instanceof Error ? e.message : String(e)
    }
    if (c.error !== undefined) {
      counts.chatErrors++
      if (error === undefined)
        fail(model.name, `chat ${c.name}: should throw`, {
          expected: c.error,
          text
        })
      continue
    }
    if (error !== undefined) {
      fail(model.name, `chat ${c.name}: threw`, error)
      continue
    }
    if (text !== c.text) {
      fail(model.name, `chat ${c.name}`, {
        ours: JSON.stringify(text),
        python: JSON.stringify(c.text)
      })
      continue
    }
    const ids = tokenizer.encodeChat(c.messages, options)
    if (!same(ids, c.ids ?? []))
      fail(model.name, `chat ids ${c.name}`, { ids, expected: c.ids })
  }

  const { specialTokens } = tokenizer
  console.log(
    `[${model.name}] ${counts.texts} texts (${counts.pieces} stream pieces), ` +
      `${counts.ids} random id sequences, ${counts.chats} chats ` +
      `(${counts.chatErrors} expected errors); vocab ${tokenizer.vocabSize}; ` +
      `loaded in ${loadMs.toFixed(0)} ms`
  )
  console.log(`[${model.name}] special tokens ${JSON.stringify(specialTokens)}`)
}

console.log(failures === 0 ? "All checks passed." : `${failures} failures.`)
process.exit(failures === 0 ? 0 : 1)
