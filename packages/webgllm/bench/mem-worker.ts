/**
 * Memory and speed over a chat, in a worker (bench/mem.html): downloads a
 * model's files as the package's worker does (to the Origin Private File
 * System, once), loads it with the worker's settings, then chats the way
 * worker.ts does (cache reuse, readPrompt's batches and snapshots, the
 * one-token guess), logging every live texture by what it is and the peak
 * since the last checkpoint. ?mode=ab instead loads the model once per
 * variant (TUNING knobs, batch sizes) and times a prompt and a reply on
 * each, rounds alternating. With ?cdp=1 it waits at each checkpoint for a
 * driver to measure the JS heaps and the GPU process.
 */
import { blobSource } from "../src/byte-source.js"
import { Engine, TUNING } from "../src/engine.js"
import type { Texture } from "../src/gpu/gl.js"
import { loadGguf } from "../src/load-gguf.js"
import { loadOnnx } from "../src/load-onnx.js"
import { modelMeta, type WebGlTextModel } from "../src/model-meta.js"
import { readPrompt } from "../src/prompt.js"
import { download } from "../src/storage.js"
import { loadTokenizer, type ChatMessage, type Tokenizer } from "../src/tokenizer.js"

type Params = {
  model: WebGlTextModel
  /** chat: a chat with checkpoints; ab: a load per variant, timed. */
  mode: "chat" | "ab"
  /** Prompt batch (0: the worker's). */
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
}

const MiB = 2 ** 20
const mib = (b: number) => (b / MiB).toFixed(1)
const log = (text: string) => postMessage({ type: "log", text })
/** Live numbers for the page's header. */
const status = (s: Record<string, unknown>) => postMessage({ type: "status", ...s })

/** Same as worker.ts's PROMPT_BATCH where the page doesn't say. */
const WORKER_BATCH = 512

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

// biome-ignore lint: the engine's private fields, read for naming.
type Any = Record<string, any>

/** What each of the engine's textures is, by walking its fields. */
function names(engine: Engine) {
  const e = engine as unknown as Any
  const map = new Map<Texture, string>()
  const put = (t: unknown, name: string) => {
    if (t && typeof t === "object" && "tex" in t) map.set(t as Texture, name)
  }
  const matrix = (m: Any | undefined, kind: string) => {
    if (!m) return
    put(m.quants, `weights ${kind}`)
    put(m.scales, `weights ${kind}`)
    put(m.zeros, `weights ${kind}`)
  }
  matrix(e.embedding, "embedding/head")
  for (const l of e.layers as Array<Any>) {
    matrix(l.gateUp, "ffn")
    matrix(l.down, "ffn")
    for (const n of ["operatorNorm", "postOperatorNorm", "ffnNorm", "postFfnNorm"])
      put(l[n], "weights norms+vectors")
    if (l.conv) {
      matrix(l.conv.inProj, "conv")
      matrix(l.conv.outProj, "conv")
      put(l.conv.kernel, "weights norms+vectors")
      l.conv.state.forEach((t: Texture) => put(t, "state conv"))
      put(l.conv.prefixes, "state conv prefixes")
    }
    if (l.attention) {
      matrix(l.attention.qkvg, "attention")
      matrix(l.attention.o, "attention")
      put(l.attention.qNorm, "weights norms+vectors")
      put(l.attention.kNorm, "weights norms+vectors")
      put(l.attention.kv, l.attention.ring ? "KV ring (sliding)" : "KV full attention")
    }
    if (l.linear) {
      matrix(l.linear.inProj, "deltanet")
      matrix(l.linear.ba, "deltanet")
      matrix(l.linear.outProj, "deltanet")
      for (const n of ["kernel", "a", "dtBias", "norm"]) put(l.linear[n], "weights norms+vectors")
      l.linear.convState.forEach((t: Texture) => put(t, "state deltanet"))
      l.linear.state.forEach((t: Texture) => put(t, "state deltanet"))
    }
  }
  put(e.finalNorm, "weights norms+vectors")
  put(e.ropeFull, "rope tables")
  put(e.ropeSliding, "rope tables")
  const ids = new Map<number, string>()
  for (const [k, v] of Object.entries((e.buffers ?? {}) as Any)) {
    const each = (t: unknown, name: string) => {
      put(t, `scratch ${name}`)
      if (t && typeof t === "object" && "id" in t) ids.set((t as Texture).id, name)
    }
    if (Array.isArray(v)) v.forEach((t) => each(t, k))
    else if (v && typeof v === "object" && !("tex" in v))
      for (const [k2, t] of Object.entries(v as Any)) each(t, `${k}.${k2}`)
    else each(v, k)
  }
  for (const [id, t] of e.quads as Map<number, Texture>)
    put(t, `scratch quad partner (${ids.get(id) ?? "?"})`)
  if (e.f16) {
    put(e.f16.w, "scratch f16 weights")
    e.f16.a.forEach((t: Texture) => put(t, "scratch f16 activations"))
  }
  for (const t of Object.values((e.linChunks?.textures ?? {}) as Any)) put(t, "scratch deltanet chunks")
  for (const bufs of (e.chunkBufs as Map<number, Array<Texture>>).values())
    bufs.forEach((t) => put(t, "scratch attention chunks"))
  for (const s of (e.snapshots as Map<number, Any>).values()) {
    s.conv.forEach((t: Texture) => put(t, "snapshots"))
    s.windows.forEach((t: Texture) => put(t, "snapshots"))
    s.linear.forEach((t: Texture) => put(t, "snapshots"))
    put(s.hidden, "snapshots")
  }
  put(e.headScratch, "scratch head slices")
  put(e.greedyOut, "scratch greedy")
  put(e.debug, "debug")
  put(e.kvDebug, "debug")
  return map
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
  const meta = modelMeta(model)
  const t = meta.tokenizer
  const weights =
    meta.files.kind === "gguf" ? [meta.files.model] : [meta.files.config, meta.files.graph, meta.files.data]
  const all = [t.tokenizer, t.config, ...(t.chatTemplate ? [t.chatTemplate] : []), ...weights]
  let last = -10
  const loaded = all.map(() => 0)
  const totals = all.map(() => 0)
  const got = await Promise.all(
    all.map((f, i) =>
      download(f, (bytes, size) => {
        loaded[i] = bytes
        totals[i] = size
        const pct = Math.floor((loaded.reduce((a, b) => a + b, 0) / Math.max(1, totals.reduce((a, b) => a + b, 0))) * 100)
        if (pct >= last + 10) {
          last = pct
          log(`downloading ${model}: ${pct}%`)
        }
      })
    )
  )
  const [tokenizerFile, configFile, ...rest] = got
  const templateFile = t.chatTemplate ? rest.shift() : undefined
  const tokenizer = loadTokenizer({
    tokenizerJson: JSON.parse(await tokenizerFile.text()) as object,
    tokenizerConfig: JSON.parse(await configFile.text()) as Record<string, unknown>,
    ...(templateFile && { chatTemplate: await templateFile.text() })
  })
  return { meta, tokenizer, weightFiles: rest }
}

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
    batchSize: batch || params.batch || WORKER_BATCH,
    ...(params.contextLength && { contextLength: params.contextLength })
  })
  instrument(engine)
  const { meta, weightFiles } = got
  const pieces =
    meta.files.kind === "gguf"
      ? loadGguf(blobSource(weightFiles[0]))
      : loadOnnx({
          config: JSON.parse(await weightFiles[0].text()) as Record<string, unknown>,
          graph: blobSource(weightFiles[1]),
          data: blobSource(weightFiles[2])
        })
  const t0 = performance.now()
  await engine.load(pieces)
  const loadMs = performance.now() - t0
  return { engine, loadMs }
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
 * One turn as worker.ts's generate runs it: the longest prefix the cache
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
  let row = await readPrompt(engine, prompt, start, cached)
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
  else await chat(params, got)
}
