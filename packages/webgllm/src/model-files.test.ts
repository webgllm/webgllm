import assert from "node:assert/strict"
import { test } from "node:test"
import { blobSource } from "./byte-source.js"
import { browserStorage, ggufContextLength, modelFiles } from "./model-files.js"
import { WEBGL_MODEL_META, WEBGL_TEXT_MODELS } from "./model-meta.js"

/** A GGUF header with these keys (strings, u32s and a string array). */
function gguf(entries: Array<[string, string | number | Array<string>]>) {
  const parts: Array<Uint8Array> = []
  const u32 = (v: number) => new Uint8Array(new Uint32Array([v]).buffer)
  const u64 = (v: number) => new Uint8Array(new BigUint64Array([BigInt(v)]).buffer)
  const str = (s: string) => {
    const bytes = new TextEncoder().encode(s)
    parts.push(u64(bytes.length), bytes)
  }
  parts.push(new TextEncoder().encode("GGUF"), u32(3), u64(0), u64(entries.length))
  for (const [key, value] of entries) {
    str(key)
    if (typeof value === "string") {
      parts.push(u32(8))
      str(value)
    } else if (typeof value === "number") {
      parts.push(u32(4), u32(value))
    } else {
      parts.push(u32(9), u32(8), u64(value.length))
      for (const s of value) str(s)
    }
  }
  return new Blob(parts as Array<BlobPart>)
}

test("a GGUF's context length is read from its metadata, stopping there", async () => {
  const file = gguf([
    ["general.architecture", "lfm2"],
    ["general.name", "x"],
    ["tokenizer.ggml.tokens", Array.from({ length: 3000 }, (_, i) => `t${i}`)],
    ["lfm2.context_length", 128000],
    ["lfm2.block_count", 16]
  ])
  let reads = 0
  const source = blobSource(file)
  const counted = { size: source.size, read: (offset: number, length: number) => (reads++, source.read(offset, length)) }
  assert.equal(await ggufContextLength(counted, 1024), 128000)
  assert.ok(reads > 1, "small chunks walk the string array")
  assert.equal(await ggufContextLength(blobSource(gguf([["general.architecture", "qwen35"]]))), undefined)
})

test("a model's files are the ones the worker fetches", () => {
  const gguf = modelFiles(WEBGL_MODEL_META["unsloth/Qwen3.5-2B-GGUF"])
  assert.deepEqual(gguf.map((f) => f.path), ["tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "Qwen3.5-2B-Q4_0.gguf"])
  const onnx = modelFiles(WEBGL_MODEL_META["onnx-community/LFM3-1.7B-ONNX"])
  assert.equal(onnx.length, 5)
})

test("a cold model listing asks the server for each GGUF's size once", async () => {
  const header = new Uint8Array(await gguf([["general.architecture", "lfm2"], ["lfm2.context_length", 32768]]).arrayBuffer())
  const file = new Uint8Array(1 << 20)
  file.set(header)
  const calls: Array<string> = []
  const real = globalThis.fetch
  globalThis.fetch = (async (input: string, init: RequestInit = {}) => {
    const method = init.method ?? "GET"
    const range = new Headers(init.headers).get("range")
    calls.push(`${method} ${input}${range ? ` ${range}` : ""}`)
    const body = input.endsWith(".gguf") ? file : new TextEncoder().encode(JSON.stringify({ max_position_embeddings: 4096 }))
    if (method === "HEAD") return new Response(null, { headers: { "content-length": String(body.length) } })
    const match = range?.match(/bytes=(\d+)-(\d+)/)
    if (!match) return new Response(body)
    const slice = body.slice(Number(match[1]), Number(match[2]) + 1)
    return new Response(slice, { status: 206, headers: { "content-range": `bytes ${match[1]}-${match[2]}/${body.length}` } })
  }) as typeof fetch
  try {
    // (No OPFS here: nothing is stored.)
    await Promise.all(WEBGL_TEXT_MODELS.map((model) => Promise.all([browserStorage.contextLength(model), browserStorage.size(model), browserStorage.downloaded(model)])))
  } finally {
    globalThis.fetch = real
  }
  const heads = calls.filter((c) => c.startsWith("HEAD"))
  console.log(`fetch calls: ${calls.length}`)
  assert.equal(new Set(heads).size, heads.length, `duplicate size probes: ${heads.filter((c, i) => heads.indexOf(c) !== i).join(", ")}`)
})
