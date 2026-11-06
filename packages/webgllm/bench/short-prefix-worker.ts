/** Native Chrome benchmark. Runs only after the benchmark page's explicit Run. */
import { blobSource } from "../src/byte-source.js"
import { loadGguf } from "../src/load-gguf.js"
import { loadOnnx } from "../src/load-onnx.js"
import { modelMeta } from "../src/model-meta.js"
import type { WebGlTextModel } from "../src/model-meta.js"
import { download, readGpuNotes } from "../src/storage.js"
import { loadTokenizer } from "../src/tokenizer.js"
import type { Engine } from "../src/engine.js"

type Runtime = typeof import("../src/engine.js") & typeof import("../src/prompt.js")
const emit = (type: string, data: unknown) => postMessage({ type, data })
const median = (xs: number[]) => { const a = [...xs].sort((x, y) => x - y); return a[Math.floor(a.length / 2)] }
const common = (a: number[], b: number[]) => { let i = 0; while (i < a.length && i < b.length && a[i] === b[i]) i++; return i }
function compare(a: Float32Array, b: Float32Array) {
  let maxAbs = 0, ia = 0, ib = 0, ma = -Infinity, mb = -Infinity
  for (let i = 0; i < a.length; i++) { maxAbs = Math.max(maxAbs, Math.abs(a[i] - b[i])); if (a[i] > ma) { ma = a[i]; ia = i }; if (b[i] > mb) { mb = b[i]; ib = i } }
  let sa = 0, sb = 0
  for (let i = 0; i < a.length; i++) { sa += Math.exp(a[i] - ma); sb += Math.exp(b[i] - mb) }
  let kl = 0
  for (let i = 0; i < a.length; i++) { const p = Math.exp(a[i] - ma) / sa, q = Math.exp(b[i] - mb) / sb; if (p) kl += p * Math.log(p / Math.max(q, Number.MIN_VALUE)) }
  const aBits = new Uint32Array(a.buffer, a.byteOffset, a.length), bBits = new Uint32Array(b.buffer, b.byteOffset, b.length)
  const bitwise = aBits.every((x, i) => x === bBits[i])
  return { maxAbs, kl, top1Equal: ia === ib, bitwise, finite: Number.isFinite(maxAbs) && Number.isFinite(kl) }
}
function referenceFor(values: Map<string, Float32Array>, key: string): Float32Array {
  const value = values.get(key)
  if (!value) throw new Error(`Missing baseline logits for ${key}`)
  return value
}
function accounting(engine: Engine) {
  const points = [...(Reflect.get(engine, "snapshots") as Map<number, { conv: { bytes: number }[]; windows: { bytes: number }[]; linear: { bytes: number }[]; hidden?: { bytes: number }; prefix?: boolean; anchor?: boolean }>)].map(([at, s]) => ({ at, prefix: !!s.prefix, anchor: !!s.anchor, bytes: [...s.conv, ...s.windows, ...s.linear, ...(s.hidden ? [s.hidden] : [])].reduce((a, t) => a + t.bytes, 0) }))
  return { count: points.length, limit: engine.snapshotLimit, bytes: points.reduce((a, p) => a + p.bytes, 0), points, gpuBytes: engine.gpu.bytes }
}
/** Each variant's engine and prompt modules, bundled by bench/server.mjs: "before" is the snapshot prepare-short-prefix.mjs wrote. */
async function runtime(variant: string): Promise<Runtime> {
  const url = variant === "before" ? "/bundle/.short-prefix-qa/before/runtime.js" : "/bundle/bench/short-prefix-runtime.js"
  return await import(url) as Runtime
}
let active = false
self.addEventListener("message", async ({ data }: MessageEvent<{ model: WebGlTextModel; rounds?: number; pairedOnly?: boolean; blockingWait?: boolean }>) => {
  if (active) return
  active = true
  const report: Record<string, unknown> = { model: data.model, startedAt: new Date().toISOString(), results: [], errors: [], scope: "Cold pp/s includes GPU completion, excludes output head. Forced decode includes logits readback. Five samples per cold length per variant/round; two warmups." }
  const results = report.results as Record<string, unknown>[]
  try {
    emit("status", "Loading model/tokenizer files from the normal browser cache…")
    const meta = modelMeta(data.model), descriptors = [meta.tokenizer.tokenizer, meta.tokenizer.config, ...(meta.tokenizer.chatTemplate ? [meta.tokenizer.chatTemplate] : []), ...(meta.files.kind === "gguf" ? [meta.files.model] : [meta.files.config, meta.files.graph, meta.files.data])]
    const files = await Promise.all(descriptors.map(file => download(file, () => undefined)))
    const rest = files.slice(2), template = meta.tokenizer.chatTemplate ? rest.shift() : undefined
    const tokenizer = loadTokenizer({ tokenizerJson: JSON.parse(await files[0].text()) as object, tokenizerConfig: JSON.parse(await files[1].text()) as Record<string, unknown>, ...(template && { chatTemplate: await template.text() }) })
    const base = tokenizer.encode("The king crossed the quiet garden. Each legal action has a separate checked outcome. ".repeat(300))
    const alternate = tokenizer.encode("A different path leads toward the river. Compare the available moves carefully. ".repeat(300))
    if (data.pairedOnly) {
      report.scope = "Alternating before/after methods in one GPU context, same loaded weights/settings/buffers. Production forward and pacing; final gl.finish blocks only this benchmark worker to exclude asynchronous end-fence scheduling. Ten measured samples per variant/length, two warmup pairs."
      const before = await runtime("before"), after = await runtime("after")
      const engine = new after.Engine(new OffscreenCanvas(1, 1), { batchSize: 512 })
      const note = Object.entries(await readGpuNotes()).find(([key]) => key.includes(`|${data.model}|mrt|`))?.[1]
      if (note) engine.preferLayout = note.layout
      let draws = 0
      const run = engine.gpu.run.bind(engine.gpu)
      engine.gpu.run = options => { draws++; return run(options) }
      try {
        const pieces = meta.files.kind === "gguf" ? loadGguf(blobSource(rest[0])) : loadOnnx({ config: JSON.parse(await rest[0].text()) as Record<string, unknown>, graph: blobSource(rest[1]), data: blobSource(rest[2]) })
        await engine.load(pieces)
        if (note?.tuned) engine.tuned = note.tuned
        await engine.warmUp()
        const gl = Reflect.get(engine.gpu, "gl") as WebGL2RenderingContext
        if (data.blockingWait) {
          // Benchmark-only: use the GPU wrapper's existing hidden-timer fallback.
          // No production method, shader, batch size or tuning option is changed.
          Reflect.set(engine.gpu, "throttledUntil", Infinity)
          report.waitControl = "existing blocking readback fallback, benchmark only; normal asynchronous runs retained separately"
        }
        report.configuration = { batchSize: engine.batchSize, tuned: engine.tuned, layout: engine.layoutChoice, renderer: engine.gpu.limits.renderer }
        for (const length of [67, 256, 2048]) {
          let reference: Float32Array | undefined
          for (let sample = -2; sample < 10; sample++) for (const variant of sample % 2 ? ["after", "before"] : ["before", "after"]) {
            const mod = variant === "before" ? before : after
            Object.setPrototypeOf(engine, mod.Engine.prototype)
            engine.reset(); gl.finish()
            emit("status", `paired ${length} tokens · sample ${sample + 1}/10 · ${variant}`)
            const began = performance.now(), count = draws
            const row = await mod.readPrompt(engine, base.slice(0, length), 0, [], { prefix: 21, anchor: length })
            gl.finish()
            const ms = performance.now() - began, forwardDraws = draws - count
            let numerical: ReturnType<typeof compare> | undefined
            if (sample === 0) {
              const logits = await engine.logits(row)
              if (!reference) reference = logits.slice()
              numerical = compare(reference, logits)
            }
            if (sample >= 0) {
              const result = { round: 1, variant, kind: "paired-cold", length, sample, ms, ppS: length / ms * 1000, draws: forwardDraws, batches: engine.batchSizes(length), numerical }
              results.push(result); emit("result", result)
            }
          }
        }
        report.finishedAt = new Date().toISOString(); emit("done", report)
      } finally { engine.gpu.dispose() }
      return
    }
    const reference = new Map<string, Float32Array>()
    let settings: { layout: "rgba4" | "bytes"; tuned: Engine["tuned"] } | undefined
    for (let round = 1; round <= (data.rounds ?? 2); round++) for (const variant of round % 2 ? ["before", "after"] : ["after", "before"]) {
      emit("status", `${data.model} · round ${round} · loading ${variant}`)
      const mod = await runtime(variant), engine = new mod.Engine(new OffscreenCanvas(1, 1), { batchSize: 512 })
      const note = Object.entries(await readGpuNotes()).find(([key]) => key.includes(`|${data.model}|mrt|`))?.[1]
      if (settings) engine.preferLayout = settings.layout
      else if (note) engine.preferLayout = note.layout
      let draws = 0, peak = engine.gpu.bytes
      const run = engine.gpu.run.bind(engine.gpu), allocate = engine.gpu.texture.bind(engine.gpu)
      engine.gpu.run = options => { draws++; return run(options) }
      engine.gpu.texture = (...args) => { const t = allocate(...args); peak = Math.max(peak, engine.gpu.bytes); return t }
      const add = (value: Record<string, unknown>) => { results.push({ model: data.model, round, variant, ...value }); emit("result", results.at(-1)) }
      try {
        const pieces = meta.files.kind === "gguf" ? loadGguf(blobSource(rest[0])) : loadOnnx({ config: JSON.parse(await rest[0].text()) as Record<string, unknown>, graph: blobSource(rest[1]), data: blobSource(rest[2]) })
        await engine.load(pieces)
        if (settings) engine.tuned = settings.tuned
        else if (note?.tuned) engine.tuned = note.tuned
        await engine.warmUp()
        if (!settings) settings = { layout: engine.layoutChoice?.layout ?? "rgba4", tuned: engine.tuned }
        if (data.blockingWait) {
          Reflect.set(engine.gpu, "throttledUntil", Infinity)
          report.waitControl = "existing blocking readback fallback; separate real engine instances; benchmark only"
        }
        const sync = async () => {
          if (data.blockingWait) (Reflect.get(engine.gpu, "gl") as WebGL2RenderingContext).finish()
          else await engine.gpu.finish()
        }
        add({ kind: "configuration", batchSize: engine.batchSize, tuned: engine.tuned, layout: engine.layoutChoice, renderer: engine.gpu.limits.renderer })
        for (const length of [67, 256, 2048]) {
          emit("status", `${variant} round ${round} · cold ${length} tokens`)
          const ids = base.slice(0, length), times: number[] = [], counts: number[] = [], batches: number[][] = []
          let row = 0
          for (let iteration = 0; iteration < 7; iteration++) {
            engine.reset(); await sync(); const began = performance.now(), n = draws
            row = await mod.readPrompt(engine, ids, 0, [], { prefix: Math.min(21, length - 1), anchor: length })
            await sync()
            if (iteration >= 2) { times.push(performance.now() - began); counts.push(draws - n); batches.push(engine.batchSizes(length)) }
          }
          const logits = await engine.logits(row), key = `cold-${length}`
          if (!reference.has(key)) reference.set(key, logits.slice())
          add({ kind: "cold", length, times, medianMs: median(times), ppS: length / median(times) * 1000, draws: counts, batches, numerical: compare(referenceFor(reference, key), logits), memory: accounting(engine) })
          const back = engine.canResume(length) ? engine.resume(length) : 0
          add({ kind: "exact", length, restored: back, numerical: back ? compare(logits, await engine.logits(0)) : null })
        }
        for (const boundary of [21, 256, 389, 733]) {
          const length = boundary > 512 ? 1800 : 800
          const first = base.slice(0, length), second = [...base.slice(0, boundary), ...alternate.slice(0, length - boundary)], third = [...base.slice(0, boundary), ...alternate.slice(7, 7 + length - boundary)]
          while (second[boundary] === first[boundary]) second[boundary] = (second[boundary] + 1) % tokenizer.vocabSize
          while (third[boundary] === second[boundary]) third[boundary] = (third[boundary] + 1) % tokenizer.vocabSize
          engine.reset()
          let previous: number[] = []
          for (const [phase, ids] of [["first", first], ["learn", second], ["reuse", third]] as const) {
            const shared = common(previous, ids), start = engine.rewind(Math.min(shared, ids.length - 1)); previous = [...ids]
            const opts = { prefix: shared ? Math.min(1100, shared) : 1100, anchor: ids.length, ...(variant === "after" && Math.min(shared, 1100) > start && { observedPrefix: Math.min(shared, 1100) }) }
            await sync(); const began = performance.now(), n = draws
            const row = await mod.readPrompt(engine, ids, start, ids.slice(0, start), opts)
            await sync(); const ms = performance.now() - began, logits = await engine.logits(row)
            const key = `prefix-${boundary}-${phase}`
            if (!reference.has(key)) reference.set(key, logits.slice())
            add({ kind: "prefix", boundary, phase, shared, restored: start, forwarded: ids.length - start, ms, draws: draws - n, numerical: compare(referenceFor(reference, key), logits), memory: accounting(engine) })
            if (variant === "after" && phase === "reuse" && start < boundary) throw new Error(`Short prefix ${boundary} was not reusable: ${start}`)
          }
        }
        emit("status", `${variant} round ${round} · decode throughput`)
        const decode: number[] = []
        for (let sample = 0; sample < 3; sample++) {
          engine.reset(); await mod.readPrompt(engine, base.slice(0, 256), 0, [], { anchor: 256 }); await sync()
          const began = performance.now()
          for (let i = 0; i < 32; i++) { const row = await engine.forwardHidden([base[300 + i]]); await engine.logits(row) }
          decode.push(performance.now() - began)
        }
        add({ kind: "decode", tokens: 32, times: decode, tokS: 32000 / median(decode), peakGpuBytes: peak, memory: accounting(engine) })
      } finally { engine.gpu.dispose() }
    }
    report.finishedAt = new Date().toISOString(); emit("done", report)
  } catch (error) { report.error = error instanceof Error ? error.stack : String(error); emit("done", report) }
  finally { active = false }
})
