/**
 * raon-check.html's worker: Raon-1 nano's GGUF through the engine against
 * the CPU reference (scripts/raon-reference.ts): each layer's streams for
 * the prompt's last token, the first logits, then greedy tokens.
 */
import { httpSource } from "../src/byte-source.js"
import { Engine, TUNING } from "../src/engine.js"
import { WebGpu } from "../src/gpu/webgpu.js"
import { loadGguf } from "../src/load-gguf.js"

const log = (text: string) => postMessage({ type: "log", text })

type Params = {
  model: string
  /** The reference's name in /models/: <ref>.json, <ref>.layers.bin (+ .json), <ref>.logits.bin. */
  ref: string
  webgpu: boolean
  batch: number
  ctx: number
  /** The prompt a token at a time (writing's passes), not as one batch. */
  single: boolean
  tokens: number
  layers: boolean
  selfCheck: boolean
  profile: boolean
  tune: string
  timing: string
  promptProfile: number
  tuned: boolean
}

const relative = (got: ArrayLike<number>, want: ArrayLike<number>, g0 = 0, w0 = 0, n = want.length) => {
  let err = 0
  let ref = 0
  for (let i = 0; i < n; i++) {
    err += (got[g0 + i] - want[w0 + i]) ** 2
    ref += want[w0 + i] ** 2
  }
  return Math.sqrt(err / Math.max(ref, 1e-30))
}

const top = (logits: Float32Array, n: number) => {
  const best: Array<number> = []
  for (let i = 0; i < logits.length; i++) {
    if (best.length < n || logits[i] > logits[best[best.length - 1]]) {
      best.push(i)
      best.sort((a, b) => logits[b] - logits[a])
      if (best.length > n) best.pop()
    }
  }
  return best
}

/** KL(reference ‖ engine) of the two softmaxes. */
const kl = (want: Float32Array, got: Float32Array) => {
  const lse = (x: Float32Array) => {
    let max = -Infinity
    for (const v of x) max = Math.max(max, v)
    let sum = 0
    for (const v of x) sum += Math.exp(v - max)
    return max + Math.log(sum)
  }
  const a = lse(want)
  const b = lse(got)
  let total = 0
  for (let i = 0; i < want.length; i++) {
    const p = Math.exp(want[i] - a)
    if (p > 0) total += p * (want[i] - a - (got[i] - b))
  }
  return total
}

async function run(p: Params) {
  // ?tune=knob:value,…: TUNING knobs for this run.
  for (const pair of p.tune ? p.tune.split(",") : []) {
    const [knob, value] = pair.split(":")
    if (!(knob in TUNING)) throw new Error(`No TUNING knob ${knob}.`)
    ;(TUNING as Record<string, number>)[knob] = Number(value)
  }
  const source = await httpSource(`/models/${p.model}`)
  const ref = (await fetch(`/models/${p.ref}.json`).then((r) => r.json())) as { ids: Array<number>; written: Array<number>; steps: Array<{ top: Array<[number, number]> }> }
  const wantLogits = new Float32Array(await fetch(`/models/${p.ref}.logits.bin`).then((r) => r.arrayBuffer()))
  const index = (await fetch(`/models/${p.ref}.layers.bin.json`).then((r) => r.json())) as Record<string, { offset: number; length: number }>
  const wantLayers = new Float32Array(await fetch(`/models/${p.ref}.layers.bin`).then((r) => r.arrayBuffer()))
  const gpu = p.webgpu ? await WebGpu.create() : null
  if (p.webgpu && !gpu) throw new Error("This browser has no WebGPU.")
  const engine = new Engine(new OffscreenCanvas(1, 1), { batchSize: p.batch, contextLength: p.ctx, ...(gpu && { gpu }) })
  // ?timing=a|b,c|…: a written token's time with passes of those names skipped (each set after the plain run; the model's output is wrong while skipping).
  const timing = async (label: string) => {
    const steps = 48
    const ids = [682, 9920, 182368, 310, 19997, 46]
    await engine.forwardHidden([ids[0]])
    await engine.gpu.finish()
    const before = engine.gpu.passes
    const t0 = performance.now()
    for (let n = 0; n < steps; n++) await engine.forwardHidden([ids[n % ids.length]])
    await engine.gpu.finish()
    log(`TIMING ${label}: ${((performance.now() - t0) / steps).toFixed(2)} ms a token, ${((engine.gpu.passes - before) / steps).toFixed(0)} passes`)
  }
  log(`GPU: ${engine.gpu.limits.renderer} (${p.webgpu ? "WebGPU" : "WebGL 2"}), max texture ${engine.gpu.limits.maxTextureSize}`)
  engine.perLayerSource = source
  let t = performance.now()
  await engine.load(loadGguf(source))
  log(`LOAD ${(performance.now() - t).toFixed(0)} ms; GPU bytes ${(engine.gpu.bytes / 2 ** 20).toFixed(0)} MiB`)
  if (p.promptProfile) {
    // A prompt batch's GPU time by kind of pass (WebGPU timestamps, each pass in a compute pass of its own), then its plain time.
    await engine.warmUp()
    // ?tuned=1: after the engine's own tuning, as the worker loads it.
    if (p.tuned) {
      await engine.tune()
      log(`TUNED ${JSON.stringify(engine.tuned)} ${JSON.stringify(engine.tuneReport)}`)
      await engine.warmUp()
    }
    const n = p.promptProfile
    const tokens = Array.from({ length: n }, (_, i) => 1000 + ((i * 7919) % 200000))
    const kinds = async () => {
      if (!gpu) return
      const profile = (await gpu.profiled()) ?? new Map()
      const byKind = new Map<string, { ms: number; n: number }>()
      for (const [name, e] of profile) {
        const kind = name.split(" ").slice(0, name.startsWith("native") ? 2 : 1).join(" ") || "(unnamed)"
        const k = byKind.get(kind) ?? { ms: 0, n: 0 }
        k.ms += e.ms
        k.n += e.n
        byKind.set(kind, k)
      }
      const total = [...byKind.values()].reduce((sum, k) => sum + k.ms, 0)
      log(`PROMPT PROFILE GPU ${total.toFixed(0)} ms: ${[...byKind].sort((a, b) => b[1].ms - a[1].ms).map(([kind, k]) => `${kind} ${k.ms.toFixed(1)} ms/${k.n}`).join(", ")}`)
      log(`PROMPT PROFILE largest: ${[...profile].sort((a, b) => b[1].ms - a[1].ms).slice(0, 24).map(([name, e]) => `${name} ${e.ms.toFixed(1)} ms/${e.n}`).join(", ")}`)
    }
    for (let round = 0; round < 3; round++) {
      engine.reset()
      const t0 = performance.now()
      await engine.forwardHidden(tokens)
      await engine.gpu.finish()
      const ms = performance.now() - t0
      log(`PROMPT ${n} tokens in ${ms.toFixed(0)} ms: ${((n * 1000) / ms).toFixed(0)} tokens/s`)
    }
    if (gpu) {
      engine.reset()
      gpu.profile = new Map()
      await engine.forwardHidden(tokens)
      await engine.gpu.finish()
      await kinds()
      gpu.profile = null
    }
    return
  }
  if (p.timing) {
    await engine.warmUp()
    // A token as the worker writes it greedily: the forward, the head and the pick's readback, each waiting on the last.
    {
      const steps = 48
      let row = await engine.forwardHidden([682])
      let [token] = await engine.argmax(row, 1, 1, [])
      const t0 = performance.now()
      let heads = 0
      for (let n = 0; n < steps; n++) {
        row = await engine.forwardHidden([token])
        const t1 = performance.now()
        ;[token] = await engine.argmax(row, 1, 1, [])
        heads += performance.now() - t1
      }
      log(`TIMING written greedily: ${((performance.now() - t0) / steps).toFixed(2)} ms a token, of which the head and its readback ${(heads / steps).toFixed(2)} ms`)
    }
    await timing("all passes")
    for (const set of p.timing.split("|")) {
      for (const name of set.split(",")) engine.gpu.skip.add(name)
      await timing(`without ${set}`)
      for (const name of set.split(",")) engine.gpu.skip.delete(name)
    }
    await timing("all passes again")
    return
  }
  const c = engine.config
  const ids = ref.ids
  const last = ids.length - 1
  // Each probed tensor's last row against the reference's (which has one entry a layer, in order).
  const seen = new Map<string, number>()
  const errors: Array<string> = []
  const probe = async (layer: number, name: string, read: () => Promise<{ data: Float32Array; rows: number; dims: number; offset: number }>) => {
    // (The engine's l_out is the hidden state; the reference's, the streams.)
    const at = index[name === "streams_out" ? "l_out" : name === "l_out" ? "" : name]
    if (!at) return
    const { data, rows, dims, offset } = await read()
    if (offset + rows - 1 !== last) return
    const n = seen.get(name) ?? 0
    seen.set(name, n + 1)
    const e = relative(data, wantLayers, (rows - 1) * dims, at.offset + n * dims, dims)
    errors.push(`${layer}:${name} ${e.toExponential(1)}`)
  }
  t = performance.now()
  let logits = new Float32Array(0)
  if (p.single) {
    for (const [i, id] of ids.entries()) logits = await engine.forward([id], p.layers && i === last ? probe : undefined)
  } else logits = await engine.forward(ids, p.layers ? probe : undefined)
  log(`PROMPT ${ids.length} tokens ${p.single ? "one at a time" : "as a batch"} in ${(performance.now() - t).toFixed(0)} ms`)
  if (p.layers) for (let i = 0; i < errors.length; i += 8) log(`LAYERS ${errors.slice(i, i + 8).join("  ")}`)
  const best = top(logits, 5)
  log(`TOP engine ${best.map((id) => `${id}:${logits[id].toFixed(3)}`).join(" ")}`)
  log(`TOP reference ${ref.steps[0].top.map(([id, v]) => `${id}:${v.toFixed(3)}`).join(" ")}`)
  log(`LOGITS relative error ${relative(logits, wantLogits).toExponential(2)}, KL ${kl(wantLogits, logits).toExponential(2)}, vocab ${c.vocabSize}`)
  const written: Array<number> = []
  t = performance.now()
  for (let n = 0; n < p.tokens; n++) {
    const token = top(logits, 1)[0]
    written.push(token)
    if (token === c.eosTokenId || token === c.padTokenId) break
    logits = await engine.forward([token])
  }
  const ms = (performance.now() - t) / Math.max(1, written.length - 1)
  const same = written.every((id, i) => id === ref.written[i]) && written.length === Math.min(p.tokens, ref.written.length)
  log(`WRITTEN ${written.join(" ")} (${ms.toFixed(1)} ms a token with logits read back)`)
  log(`REFERENCE ${ref.written.join(" ")}`)
  log(same ? "PASS: the engine writes the reference's tokens" : "FAIL: the engine's tokens differ from the reference's")
  if (p.profile) {
    // A written token's time as the worker spends it: issuing its passes (JavaScript, and the per-layer row's read), then the GPU finishing them.
    {
      const steps = 32
      let issue = 0
      let total = 0
      let token = ids[0]
      for (let n = 0; n < steps; n++) {
        const t0 = performance.now()
        await engine.forwardHidden([ids[(n * 7) % ids.length] ?? token])
        const t1 = performance.now()
        await engine.gpu.finish()
        issue += t1 - t0
        total += performance.now() - t0
      }
      log(`TIMING a token without its head: ${(total / steps).toFixed(2)} ms, of which issuing ${(issue / steps).toFixed(2)} ms`)
      void token
    }
    // Where a written token's GPU time goes: each pass timed in a compute pass of its own (WebGPU), by its kind.
    const steps = 16
    const before = engine.gpu.passes
    if (gpu) gpu.profile = new Map()
    t = performance.now()
    let token = written[written.length - 2] ?? ids[0]
    for (let n = 0; n < steps; n++) {
      logits = await engine.forward([token])
      token = top(logits, 1)[0]
    }
    const wall = (performance.now() - t) / steps
    log(`PROFILE ${((engine.gpu.passes - before) / steps).toFixed(0)} passes a token, ${wall.toFixed(1)} ms a token profiled`)
    if (gpu) {
      const profile = (await gpu.profiled()) ?? new Map()
      gpu.profile = null
      const kinds = new Map<string, { ms: number; n: number }>()
      for (const [name, e] of profile) {
        const kind = name.split(" ").slice(0, name.startsWith("native") ? 2 : 1).join(" ") || "(unnamed)"
        const k = kinds.get(kind) ?? { ms: 0, n: 0 }
        k.ms += e.ms
        k.n += e.n
        kinds.set(kind, k)
      }
      const total = [...kinds.values()].reduce((sum, k) => sum + k.ms, 0)
      log(`PROFILE GPU ${(total / steps).toFixed(2)} ms a token: ${[...kinds].sort((a, b) => b[1].ms - a[1].ms).map(([kind, k]) => `${kind} ${(k.ms / steps).toFixed(2)} ms/${(k.n / steps).toFixed(0)}`).join(", ")}`)
      const worst = [...profile].sort((a, b) => b[1].ms - a[1].ms).slice(0, 14)
      log(`PROFILE largest: ${worst.map(([name, e]) => `${name} ${(e.ms / steps).toFixed(2)} ms/${(e.n / steps).toFixed(0)}`).join(", ")}`)
    }
  }
  if (p.selfCheck) {
    await engine.warmUp()
    t = performance.now()
    log(`SELF-CHECK ${(await engine.selfCheck()).toExponential(2)} in ${(performance.now() - t).toFixed(0)} ms`)
  }
}

onmessage = ({ data }: MessageEvent<Params>) => {
  run(data).then(
    () => postMessage({ type: "done" }),
    (error: unknown) => postMessage({ type: "error", text: error instanceof Error ? `${error.message}\n${error.stack ?? ""}` : String(error) })
  )
}
