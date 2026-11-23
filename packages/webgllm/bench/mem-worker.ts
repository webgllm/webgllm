/**
 * Memory and speed over a chat, in a worker (bench/mem.html): downloads a
 * model's files as the package's worker does (to the Origin Private File
 * System, once), loads it with the worker's settings, then chats the way
 * handler.ts does (cache reuse, readPrompt's batches and snapshots, the
 * one-token guess), logging every live texture by what it is and the peak
 * since the last checkpoint. ?mode=ab instead loads the model once per
 * variant (TUNING knobs, batch sizes) and times a prompt and a reply on
 * each, rounds alternating. With ?cdp=1 it waits at each checkpoint for a
 * driver to measure the JS heaps and the GPU process.
 */
import { blobSource, httpSource } from "../src/byte-source.js"
import { Engine, TUNING } from "../src/engine.js"
import { WebGpu } from "../src/gpu/webgpu.js"
import { names } from "./memory-names.js"
import type { Texture } from "../src/gpu/gl.js"
import { loadGguf } from "../src/load-gguf.js"
import { loadOnnx } from "../src/load-onnx.js"
import { fileUrl, modelMeta, onnxDataSources, type WebGlTextModel } from "../src/model-meta.js"
import { resolveModel } from "../src/resolve-model.js"
import { anchorFor, readPrompt, systemPrefixFor } from "../src/prompt.js"
import { download } from "../src/storage.js"
import { checkPrefixCache } from "./prefix-cache-check.js"
import { loadTokenizer, type ChatMessage, type Tokenizer } from "../src/tokenizer.js"
import type { ByteSource } from "../src/weights.js"

type Params = {
  model: WebGlTextModel
  /** chat: a chat with checkpoints; ab: a load per variant, timed. */
  mode: "chat" | "ab" | "prefix" | "draft" | "kv8"
  /** Prompt batch (0: the worker's, 512). */
  batch: number
  /** The first message's length in tokens. */
  length: number
  /** Follow-ups' length in tokens (0: short questions). */
  followLength: number
  /** Turns (chat), or until the cache holds `target` tokens if that's set. */
  turns: number
  target: number
  /** Tokens per reply. */
  reply: number
  contextLength?: number
  thinking: boolean
  /** TUNING knobs for every load ("knob:value,…"). */
  tune: string
  /** ab: variants, "|"-separated, each "knob:value,…" or "batch:N" ("default": none). */
  variants: string
  rounds: number
  cdp: boolean
  /** chat: print every texture group at each checkpoint, not just the totals. */
  detail: boolean
  /** Run on WebGPU (?webgpu=1); WebGL 2 otherwise. */
  webgpu: boolean
}

const MiB = 2 ** 20
const mib = (b: number) => (b / MiB).toFixed(1)
const log = (text: string) => postMessage({ type: "log", text })
/** Live numbers for the page's header. */
const status = (s: Record<string, unknown>) => postMessage({ type: "status", ...s })


let peak = 0
let peakLabel = ""
let phase = ""
let peakSet: Array<Texture> = []

/** Tracks the most texture bytes alive at once (Gpu.bytes) and what they were. */
function instrument(engine: Engine) {
  const gpu = engine.gpu as unknown as {
    texture: (...a: Array<unknown>) => Texture
    live: Set<Texture>
    bytes: number
  }
  const make = gpu.texture.bind(gpu)
  gpu.texture = (...args: Array<unknown>) => {
    const t = make(...args)
    if (gpu.bytes > peak) {
      peak = gpu.bytes
      peakLabel = phase
      peakSet = [...gpu.live]
    }
    return t
  }
}

/** The category of a texture name: what the summary line adds up. */
const category = (name: string) =>
  name.startsWith("weights")
    ? "weights"
    : name.startsWith("KV")
      ? "KV"
      : name.startsWith("snapshots")
        ? "snapshots"
        : name.startsWith("state")
          ? "state"
          : name.startsWith("scratch")
            ? "scratch"
            : "other"

function breakdown(engine: Engine, set?: Array<Texture>) {
  const gpu = engine.gpu as unknown as { live: Set<Texture> }
  const named = names(engine)
  const by = new Map<string, { bytes: number; n: number }>()
  for (const t of set ?? gpu.live) {
    const name = named.get(t) ?? "other (freed since, or unnamed)"
    const g = by.get(name) ?? { bytes: 0, n: 0 }
    g.bytes += t.bytes
    g.n++
    by.set(name, g)
  }
  let total = 0
  const cats: Record<string, number> = {}
  for (const [name, g] of by) {
    total += g.bytes
    const c = category(name)
    cats[c] = (cats[c] ?? 0) + g.bytes
  }
  return { total, by, cats }
}

function printBreakdown(label: string, b: ReturnType<typeof breakdown>, detail: boolean) {
  const cats = Object.entries(b.cats)
    .sort((x, y) => y[1] - x[1])
    .map(([c, v]) => `${c} ${mib(v)}`)
    .join(", ")
  let text = `MEM ${label}: ${mib(b.total)} MiB of textures (${cats})`
  if (detail) {
    const rows = [...b.by.entries()].sort((x, y) => y[1].bytes - x[1].bytes)
    text += `\n${rows.map(([name, g]) => `  ${mib(g.bytes).padStart(8)} MiB ${String(g.n).padStart(4)}× ${name}`).join("\n")}`
  }
  log(text)
  log(`MEMJSON ${JSON.stringify({ label, total: b.total, cats: b.cats, by: Object.fromEntries([...b.by].map(([n, g]) => [n, g.bytes])) })}`)
}

let goResolve: (() => void) | null = null
onmessage = ({ data }) => {
  if (data.type === "go") {
    goResolve?.()
    goResolve = null
    return
  }
  run(data as Params).then(
    () => postMessage({ type: "done" }),
    (error: unknown) =>
      postMessage({
        type: "error",
        message: error instanceof Error ? `${error.message}\n${error.stack}` : String(error)
      })
  )
}

async function checkpoint(engine: Engine, params: Params, label: string) {
  await engine.gpu.finish()
  const now = breakdown(engine)
  printBreakdown(label, now, params.detail)
  const most = Math.max(peak, now.total)
  if (peak > now.total) {
    printBreakdown(`${label}, PEAK since the last checkpoint (${peakLabel})`, breakdown(engine, peakSet), params.detail)
  }
  log(`MEMPOINT ${label} | now ${mib(now.total)} MiB | peak ${mib(most)} MiB`)
  status({ textures: now.total, peak: most })
  peak = now.total
  peakSet = [...(engine.gpu as unknown as { live: Set<Texture> }).live]
  if (params.cdp) {
    await new Promise<void>((resolve) => {
      goResolve = resolve
      postMessage({ type: "memwait", label })
    })
  }
}

/** The model's files, from the Origin Private File System (downloaded the first time). */
async function files(model: WebGlTextModel) {
  // (Any Hugging Face repository, not just the catalog: resolved first.)
  await resolveModel(model)
  const meta = modelMeta(model)
  const t = meta.tokenizer
  const weights =
    meta.files.kind === "gguf" ? [meta.files.model] : [meta.files.config, meta.files.graph, meta.files.data, ...(meta.files.moreData ?? [])]
  const all = [t.tokenizer, t.config, ...(t.chatTemplate ? [t.chatTemplate] : []), ...weights]
  // Without the Origin Private File System (a page that isn't a secure
  // context: a phone on http://<the Mac's address>), the files stream
  // from Hugging Face on every load instead.
  const opfs = typeof navigator.storage?.getDirectory === "function"
  if (!opfs) log("no Origin Private File System here: the weights stream from Hugging Face on every load")
  let last = -10
  const loaded = all.map(() => 0)
  const totals = all.map(() => 0)
  const got: Array<Weight> = await Promise.all(
    all.map((f, i) =>
      opfs
        ? download(f, (bytes, size) => {
            loaded[i] = bytes
            totals[i] = size
            const pct = Math.floor((loaded.reduce((a, b) => a + b, 0) / Math.max(1, totals.reduce((a, b) => a + b, 0))) * 100)
            if (pct >= last + 10) {
              last = pct
              log(`downloading ${model}: ${pct}%`)
            }
          })
        : Promise.resolve(fileUrl(f))
    )
  )
  const text = (w: Weight) => (typeof w === "string" ? fetch(w).then((r) => r.text()) : w.text())
  const [tokenizerFile, configFile, ...rest] = got
  const templateFile = t.chatTemplate ? rest.shift() : undefined
  const tokenizer = loadTokenizer({
    tokenizerJson: JSON.parse(await text(tokenizerFile)) as object,
    tokenizerConfig: JSON.parse(await text(configFile)) as Record<string, unknown>,
    ...(templateFile && { chatTemplate: await text(templateFile) })
  })
  return { meta, tokenizer, weightFiles: rest, text }
}

/** A model file: from the Origin Private File System, or its URL. */
type Weight = File | string
const source = (w: Weight) => (typeof w === "string" ? httpSource(w) : Promise.resolve(blobSource(w)))

/** Applies "knob:value,…" to TUNING; returns the batch if one says "batch:N". */
function applyTune(spec: string) {
  let batch = 0
  for (const pair of spec.split(",")) {
    const [knob, value] = pair.split(":")
    if (!knob || value === undefined || knob === "default") continue
    if (knob === "batch") batch = Number(value)
    else if (knob in TUNING) (TUNING as Record<string, number>)[knob] = Number(value)
    else throw new Error(`No TUNING knob ${knob}.`)
  }
  return batch
}

async function loadEngine(params: Params, got: Awaited<ReturnType<typeof files>>, batch: number) {
  const engine = new Engine(new OffscreenCanvas(1, 1), {
    ...(params.webgpu && { gpu: await WebGpu.create() }),
    batchSize: batch || params.batch || (TUNING.promptBatch > 0 ? TUNING.promptBatch : 512),
    ...(params.contextLength && { contextLength: params.contextLength })
  })
  instrument(engine)
  const { meta, weightFiles, text } = got
  const pieces =
    meta.files.kind === "gguf"
      ? loadGguf(await source(weightFiles[0]))
      : loadOnnx({
          config: JSON.parse(await text(weightFiles[0])) as Record<string, unknown>,
          graph: await source(weightFiles[1]),
          data: await onnxData(meta.files, weightFiles.slice(2), source)
        })
  const t0 = performance.now()
  await engine.load(pieces)
  const loadMs = performance.now() - t0
  return { engine, loadMs }
}

/** An ONNX graph's external data: its one file, or each of several by the location the graph gives it. */
async function onnxData<F>(files: Extract<ReturnType<typeof modelMeta>["files"], { kind: "onnx" }>, data: Array<F>, source: (file: F) => Promise<ByteSource>) {
  return onnxDataSources(files, await Promise.all(data.map(source)))
}

const FILLER =
  "The river runs past the old mill, where the miller's daughter counts sacks of grain each morning before the market opens in the square. "
const FOLLOW_UPS = [
  "Now tell it again from the daughter's point of view.",
  "What happens at the market the next day?",
  "Give the story a different ending.",
  "Summarize everything so far in three sentences.",
  "Write a short poem about the river.",
  "Who else lives in the village?"
]

/**
 * `ask` after filler: the chat's next user message, padded so the prompt
 * grows by about `length` tokens (the first message: a prompt of about
 * `length` tokens in all).
 */
function padded(tokenizer: Tokenizer, before: Array<ChatMessage>, ask: string, length: number, thinking: boolean) {
  if (length <= 0) return ask
  const options = { addGenerationPrompt: true, enableThinking: thinking }
  const count = (pad: string) =>
    tokenizer.encodeChat([...before, { role: "user", content: pad + ask }], options).length
  const goal = (before.length ? tokenizer.encodeChat(before, options).length : 0) + length
  const base = count("")
  if (base >= goal) return ask
  const per = count(FILLER) - base
  let pad = FILLER.repeat(Math.max(0, Math.floor((goal - base) / per)))
  while (count(pad) < goal) pad += FILLER
  return `${pad}\n\n${ask}`
}

type Turn = { prompt: number; read: number; promptMs: number; firstMs: number; reply: number; writeMs: number }

/**
 * One turn as handler.ts's generate runs it: the longest prefix the cache
 * shares reused (the whole prompt only if its end was saved with its
 * hidden row), the rest read by readPrompt, then `reply` tokens written
 * with the model's top-k and the one-token guess (greedy among the
 * candidates, so runs repeat).
 */
async function turn(engine: Engine, tokenizer: Tokenizer, messages: Array<ChatMessage>, cached: Array<number>, params: Params, topK: number, penalty: number): Promise<Turn & { text: string }> {
  const prompt = tokenizer.encodeChat(messages, { addGenerationPrompt: true, enableThinking: params.thinking })
  let common = 0
  while (common < cached.length && common < prompt.length && cached[common] === prompt[common]) common++
  const t0 = performance.now()
  const start =
    common === prompt.length && engine.canResume(common)
      ? engine.resume(common)
      : engine.rewind(Math.min(common, prompt.length - 1))
  cached.length = start
  let row = await readPrompt(engine, prompt, start, cached, {
    anchor: anchorFor(tokenizer, prompt, messages, params.thinking)
  })
  engine.setSeen(prompt, tokenizer.vocabSize)
  const k = Math.max(1, Math.min(topK, 64))
  const top = (d: { ids: Int32Array; logits: Float32Array }) => {
    let best = 0
    for (let j = 1; j < d.logits.length; j++) if (d.logits[j] > d.logits[best]) best = j
    return d.ids[best]
  }
  let token = top(await engine.candidates(row, k, penalty, params.reply > 1))
  const firstMs = performance.now() - t0
  // (The prompt's time: up to the first token's candidates, as the worker counts it.)
  const promptMs = firstMs
  const reply: Array<number> = []
  const eos = tokenizer.specialTokens.eos
  const w0 = performance.now()
  const room = engine.contextLength - engine.position - 1
  while (reply.length < Math.min(params.reply, room) && token !== eos) {
    reply.push(token)
    engine.markSeen(token)
    const more = reply.length < Math.min(params.reply, room)
    if (engine.guessing) {
      if (engine.guessedToken === token) {
        engine.keepGuess()
        cached.push(token)
        if (!more) break
        token = top(await engine.candidates(0, k, penalty, true))
        continue
      }
      engine.dropGuess()
    }
    row = await engine.forwardHidden([token])
    cached.push(token)
    if (!more) break
    token = top(await engine.candidates(row, k, penalty, true))
  }
  engine.dropGuess()
  engine.snapshot()
  const writeMs = performance.now() - w0
  return { prompt: prompt.length, read: prompt.length - start, promptMs, firstMs, reply: reply.length, writeMs, text: tokenizer.decode(reply) }
}

const describe = (t: Turn) =>
  `prompt ${t.prompt} tokens, ${t.read} read in ${t.promptMs.toFixed(0)} ms (${((t.read / t.promptMs) * 1000).toFixed(0)} tok/s), reply ${t.reply} tokens at ${(((t.reply - 1) / t.writeMs) * 1000).toFixed(1)} tok/s`

async function chat(params: Params, got: Awaited<ReturnType<typeof files>>) {
  applyTune(params.tune)
  phase = "load"
  const { engine, loadMs } = await loadEngine(params, got, 0)
  log(`GPU: ${engine.gpu.limits.renderer}, max texture ${engine.gpu.limits.maxTextureSize}; ${params.model}, batch ${engine.batchSize} (capacity ${engine.capacity}), context ${engine.contextLength}; layout ${engine.layoutChoice?.layout}; loaded in ${(loadMs / 1000).toFixed(1)} s`)
  await checkpoint(engine, params, "load")
  phase = "warm-up and self-check"
  await engine.warmUp()
  await engine.selfCheck()
  await checkpoint(engine, params, "warm")
  const { tokenizer, meta } = got
  const messages: Array<ChatMessage> = [
    { role: "user", content: padded(tokenizer, [], "Tell me a short story about the mill, in a few paragraphs.", params.length, params.thinking) }
  ]
  const cached: Array<number> = []
  const turns = params.target > 0 ? 1000 : params.turns
  for (let n = 1; n <= turns; n++) {
    phase = `turn ${n}`
    const t = await turn(engine, tokenizer, messages, cached, params, meta.sampling.top_k || 20, meta.sampling.repetition_penalty ?? 1)
    log(`TURN ${n}: ${describe(t)}; cache ${engine.position}`)
    status({ turn: n, position: engine.position, promptTps: (t.read / t.promptMs) * 1000, writeTps: ((t.reply - 1) / t.writeMs) * 1000, read: t.read })
    await checkpoint(engine, params, `turn ${n}, cache ${engine.position} tokens`)
    if (params.target > 0 && engine.position >= params.target) break
    if (engine.contextLength - engine.position < params.reply + params.followLength + 64) {
      log(`stopping: the context (${engine.contextLength}) is nearly full`)
      break
    }
    messages.push({ role: "assistant", content: t.text })
    const ask = FOLLOW_UPS[(n - 1) % FOLLOW_UPS.length]
    messages.push({ role: "user", content: params.followLength > 0 ? padded(tokenizer, messages, ask, params.followLength, params.thinking) : ask })
  }
  engine.gpu.dispose()
}

/** ?mode=ab: each variant in its own load, timed on the same prompt; rounds alternate the order. */
async function ab(params: Params, got: Awaited<ReturnType<typeof files>>) {
  const variants = params.variants.split("|").filter(Boolean)
  const base = { ...TUNING }
  const results = new Map<string, Array<Turn & { textures: number; peak: number }>>()
  const { tokenizer, meta } = got
  const content = padded(tokenizer, [], "Tell me a short story about the mill, in a few paragraphs.", params.length, params.thinking)
  for (let round = 0; round < params.rounds; round++) {
    const order = round % 2 ? [...variants].reverse() : variants
    for (const v of order) {
      Object.assign(TUNING, base)
      applyTune(params.tune)
      const batch = applyTune(v)
      phase = `${v} load`
      peak = 0
      const { engine } = await loadEngine(params, got, batch)
      await engine.warmUp()
      const loaded = engine.gpu.bytes
      phase = `${v} chat`
      // A short turn first (the GPU's clocks up, programs compiled), then the timed one.
      const cached: Array<number> = []
      await turn(engine, tokenizer, [{ role: "user", content: "Hi" }], cached, { ...params, reply: 8 }, meta.sampling.top_k || 20, 1)
      const t = await turn(engine, tokenizer, [{ role: "user", content: `Round ${round + 1}.\n\n${content}` }], cached, params, meta.sampling.top_k || 20, meta.sampling.repetition_penalty ?? 1)
      await engine.gpu.finish()
      const r = { ...t, textures: loaded, peak }
      results.set(v, [...(results.get(v) ?? []), r])
      log(`AB ${v} round ${round + 1}: ${describe(t)}; textures ${mib(loaded)} MiB at load, peak ${mib(peak)} MiB (batch ${engine.batchSize})`)
      status({ variant: v, round: round + 1, promptTps: (t.read / t.promptMs) * 1000, writeTps: ((t.reply - 1) / t.writeMs) * 1000, textures: loaded, peak })
      engine.gpu.dispose()
      // (A moment for the browser to let the textures go.)
      await new Promise((r) => setTimeout(r, 500))
    }
  }
  const median = (xs: Array<number>) => [...xs].sort((a, b) => a - b)[Math.floor(xs.length / 2)]
  const lines = variants.map((v) => {
    const rs = results.get(v) ?? []
    return `  ${v.padEnd(28)} prompt ${median(rs.map((r) => (r.read / r.promptMs) * 1000)).toFixed(0).padStart(6)} tok/s, first token ${(median(rs.map((r) => r.firstMs)) / 1000).toFixed(2)} s, writing ${median(rs.map((r) => ((r.reply - 1) / r.writeMs) * 1000)).toFixed(1)} tok/s, textures ${mib(median(rs.map((r) => r.textures)))} MiB, peak ${mib(median(rs.map((r) => r.peak)))} MiB`
  })
  log(`AB SUMMARY (${params.model}, ${params.length}-token prompt, ${params.reply}-token reply, median of ${params.rounds}):\n${lines.join("\n")}`)
}

/** ?mode=prefix: verify real GPU restoration after an independent task. */
async function prefixCheck(params: Params, got: Awaited<ReturnType<typeof files>>) {
  applyTune(params.tune)
  const { engine } = await loadEngine(params, got, 0)
  try {
    await engine.warmUp()
    // Stable context spans a natural batch so the test exercises a retained point.
    const system: ChatMessage = { role: "system", content: "Follow the current request and reply accurately. " + padded(got.tokenizer, [], "Shared context for independent tasks.", params.length, params.thinking) }
    const chats: Array<Array<ChatMessage>> = ["First independent task.", "Different independent task."].map(task => [
      system,
      { role: "user" as const, content: task + "\n" + padded(got.tokenizer, [], task, params.length, params.thinking) }
    ])
    const prompts = chats.map(chat => got.tokenizer.encodeChat(chat, { addGenerationPrompt: true, enableThinking: params.thinking }))
    const prefixes = chats.map((chat,index) => systemPrefixFor(got.tokenizer, prompts[index], chat, params.thinking))
    if (!prefixes[0] || !prefixes[1]) throw new Error("This template has no verified system-prefix checkpoint.")
    const anchors = chats.map((chat,index) => anchorFor(got.tokenizer,prompts[index],chat,params.thinking))
    const result = await checkPrefixCache(engine, prompts[0], prompts[1], {
      firstPrefix: prefixes[0], secondPrefix: prefixes[1], firstAnchor: anchors[0], secondAnchor: anchors[1]
    })
    if (result.restoredTokens <= 0) throw new Error("Numerical cache check did not restore a nonempty selected checkpoint.")
    const reported = { model: params.model, mode: params.mode, ...result, peakBytes: peak }
    log(`PREFIX CACHE CHECK ${JSON.stringify(reported)}`)
    postMessage({ type: "result", value: reported })
  } finally {
    engine.gpu.dispose()
  }
}

/**
 * ?mode=draft: draft batches cut back to every prefix against the same
 * tokens read one at a time: the batch's kept rows, then six tokens read
 * after `accept`, which only match if the state went back correctly; and
 * two draft batches in a row, each cut short.
 */
async function draftCheck(params: Params, got: Awaited<ReturnType<typeof files>>) {
  applyTune(params.tune)
  const { engine } = await loadEngine(params, got, 0)
  try {
    await engine.warmUp()
    if (!engine.canDraft) throw new Error("This model reads no draft batches (tune=linearDrafts:1 for Gated DeltaNet).")
    const { tokenizer } = got
    const content = padded(tokenizer, [], "Tell me a short story about the mill, in a few paragraphs.", params.length + 32, false)
    const all = Array.from(tokenizer.encodeChat([{ role: "user", content }], { addGenerationPrompt: true, enableThinking: false }))
    const prompt = all.slice(0, -32)
    const rest = all.slice(-32)
    const distribution = (logits: Float32Array) => {
      let max = -Infinity
      for (const l of logits) if (l > max) max = l
      const p = new Float64Array(logits.length)
      let sum = 0
      for (let i = 0; i < p.length; i++) sum += p[i] = Math.exp(logits[i] - max)
      for (let i = 0; i < p.length; i++) p[i] /= sum
      return p
    }
    const argmax = (x: ArrayLike<number>) => {
      let at = 0
      for (let i = 1; i < x.length; i++) if (x[i] > x[at]) at = i
      return at
    }
    engine.reset()
    await engine.forwardHidden(prompt)
    const want: Array<Float64Array> = []
    const wantTop: Array<number> = []
    for (const token of rest) {
      const logits = await engine.logits(await engine.forwardHidden([token]))
      want.push(distribution(logits))
      wantTop.push(argmax(logits))
    }
    let worst = 0
    let compared = 0
    let same = 0
    const check = async (row: number, at: number) => {
      const logits = await engine.logits(row)
      const p = want[at]
      const q = distribution(logits)
      let kl = 0
      for (let i = 0; i < p.length; i++) if (p[i] > 1e-12) kl += p[i] * Math.log(p[i] / Math.max(q[i], 1e-30))
      if (!(kl <= worst)) worst = kl
      compared++
      if (argmax(logits) === wantTop[at]) same++
    }
    const before = engine.gpu.bytes
    let cases = 0
    for (const n of [2, 3, 4, 5, 8]) {
      for (let keep = 1; keep < n; keep++) {
        engine.reset()
        await engine.forwardHidden(prompt)
        const last = await engine.forwardHidden(rest.slice(0, n), { draft: true })
        for (let r = 0; r < keep; r++) await check(last - n + 1 + r, r)
        engine.accept(keep)
        for (let at = keep; at < keep + 6; at++) await check(await engine.forwardHidden([rest[at]]), at)
        cases++
      }
    }
    engine.reset()
    await engine.forwardHidden(prompt)
    await engine.forwardHidden(rest.slice(0, 5), { draft: true })
    engine.accept(2)
    const last = await engine.forwardHidden(rest.slice(2, 7), { draft: true })
    for (let r = 0; r < 3; r++) await check(last - 4 + r, 2 + r)
    engine.accept(3)
    for (let at = 5; at < 11; at++) await check(await engine.forwardHidden([rest[at]]), at)
    cases++
    log(
      `DRAFT CHECK ${params.model} (${prompt.length}-token prompt, ${cases} cut-short batches, ${compared} rows): worst KL ${worst.toExponential(2)}, same top token ${same}/${compared}, draft rows ${mib(engine.gpu.bytes - before)} MiB`
    )
  } finally {
    engine.gpu.dispose()
  }
}

/**
 * ?mode=kv8: what an 8-bit KV cache would cost in quality, before building
 * one. A real document is read in batches of 256 and then 32 tokens one at
 * a time, the cache's keys and values rounded after each to int8 with a
 * scale per 32 values (llama.cpp's Q8_0), against the same run at f16. The
 * cache stays f16 here: no memory is saved and a token's own batch reads
 * its keys unrounded, so this is a lower bound on the error.
 */
async function kv8Check(params: Params, got: Awaited<ReturnType<typeof files>>) {
  applyTune(params.tune)
  const { engine } = await loadEngine(params, got, 0)
  try {
    await engine.warmUp()
    const layers = (engine as unknown as { layers: Array<{ attention?: { kv: Texture } }> }).layers
    const { tokenizer } = got
    const doc = await (await fetch("/README.md")).text()
    const ids = Array.from(tokenizer.encodeChat(
      [{ role: "user", content: doc.slice(0, (params.length + 64) * 8) }],
      { addGenerationPrompt: false, enableThinking: false }
    ))
    if (ids.length < params.length + 32) throw new Error(`The document is only ${ids.length} tokens.`)
    const prompt = ids.slice(0, params.length)
    const steps = Math.max(32, params.reply)
    const rest = ids.slice(params.length, params.length + steps)
    // Halves per cache row: the keys, then the values.
    const rowHalves = (engine as unknown as { kvWidth(): number }).kvWidth() * 8
    const round = async (part: string) => {
      for (const layer of layers) {
        const kv = layer.attention?.kv
        if (!kv) continue
        const bytes = (await engine.gpu.readBytes(kv)).slice()
        const halves = new (globalThis as unknown as { Float16Array: new (b: ArrayBuffer) => Float32Array }).Float16Array(bytes.buffer as ArrayBuffer)
        for (let b = 0; b + 32 <= halves.length; b += 32) {
          const isKey = b % rowHalves < rowHalves / 2
          if ((part === "keys" && !isKey) || (part === "values" && isKey)) continue
          let most = 0
          for (let i = b; i < b + 32; i++) {
            const a = Math.abs(halves[i])
            if (Number.isFinite(a) && a > most) most = a
          }
          if (most === 0) continue
          const step = most / 127
          for (let i = b; i < b + 32; i++) if (Number.isFinite(halves[i])) halves[i] = Math.round(halves[i] / step) * step
        }
        engine.gpu.write(kv, { x: 0, y: 0, width: kv.width, height: kv.height }, {
          format: 0x8d99, // RGBA_INTEGER
          type: 0x1405, // UNSIGNED_INT
          pixels: new Uint32Array(bytes.buffer as ArrayBuffer)
        })
      }
    }
    const read = async (part: string | null) => {
      engine.reset()
      const out: Array<Float32Array> = []
      let row = 0
      for (let at = 0; at < prompt.length; at += 256) {
        row = await engine.forwardHidden(prompt.slice(at, at + 256), { fullRows: true })
        if (part) await round(part)
      }
      out.push((await engine.logits(row)).slice())
      for (const token of rest) {
        out.push((await engine.logits(await engine.forwardHidden([token]))).slice())
        if (part) await round(part)
      }
      return out
    }
    const compare = (a: Array<Float32Array>, b: Array<Float32Array>) => {
      const dist = (l: Float32Array) => {
        let max = -Infinity
        for (const v of l) if (v > max) max = v
        const p = new Float64Array(l.length)
        let sum = 0
        for (let i = 0; i < p.length; i++) sum += p[i] = Math.exp(l[i] - max)
        for (let i = 0; i < p.length; i++) p[i] /= sum
        return p
      }
      const top = (l: Float32Array) => {
        let at = 0
        for (let i = 1; i < l.length; i++) if (l[i] > l[at]) at = i
        return at
      }
      let worst = 0
      let total = 0
      let same = 0
      for (let r = 0; r < a.length; r++) {
        const p = dist(a[r])
        const q = dist(b[r])
        let kl = 0
        for (let i = 0; i < p.length; i++) if (p[i] > 1e-12) kl += p[i] * Math.log(p[i] / Math.max(q[i], 1e-30))
        total += kl
        worst = Math.max(worst, kl)
        if (top(a[r]) === top(b[r])) same++
      }
      return `mean KL ${(total / a.length).toExponential(2)}, worst ${worst.toExponential(2)}, same top token ${same}/${a.length}`
    }
    const plain = await read(null)
    const again = await read(null)
    log(`KV8 CHECK ${params.model} (${prompt.length} tokens + ${rest.length} steps): f16 against itself ${compare(plain, again)}`)
    for (const part of ["keys", "values", "both"])
      log(`KV8 CHECK ${params.model} (${prompt.length} tokens + ${rest.length} steps): int8 ${part} against f16 ${compare(plain, await read(part))}`)
  } finally {
    engine.gpu.dispose()
  }
}

async function run(params: Params) {
  if (params.cdp) {
    // The GPU process before any of this page's textures: what the
    // driver's later footprints are measured against.
    log("MEMPOINT empty | now 0.0 MiB | peak 0.0 MiB")
    await new Promise<void>((resolve) => {
      goResolve = resolve
      postMessage({ type: "memwait", label: "empty" })
    })
  }
  const got = await files(params.model)
  if (params.mode === "ab") await ab(params, got)
  else if (params.mode === "prefix") await prefixCheck(params, got)
  else if (params.mode === "draft") await draftCheck(params, got)
  else if (params.mode === "kv8") await kv8Check(params, got)
  else await chat(params, got)
}
