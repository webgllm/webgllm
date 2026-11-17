/** Phase instrumentation around the same worker and engine used by the app. */
import { Gpu } from "../src/gpu/gl.js"
import { WebGpu } from "../src/gpu/webgpu.js"
import { profileGpu, type UploadSample } from "./gpu-profile.js"
import { expandQ6 } from "../src/q6-expanded.js"
import { newQuantMatrix } from "../src/repack.js"
import type { QuantMatrix, WeightPiece } from "../src/weights.js"
import { Engine } from "../src/engine.js"
let loadedEngine: Engine | null = null
let peakGpuBytes = 0
let captureSample = false
let nativeSample: QuantMatrix | undefined
const uploadSamples: UploadSample[] = []
const captureEngine = (engine: Engine) => {
  loadedEngine = engine
}
for (const name of ["load", "warmUp", "selfCheck", "tune"] as const) {
  const original = Engine.prototype[name]
  // Benchmark-only wrappers preserve all arguments and return values.
  Object.assign(Engine.prototype, {
    [name]: async function (this: Engine, ...args: unknown[]) {
      if (name === "load") {
        captureEngine(this)
        if (captureSample) {
          const pieces = args[0] as AsyncIterable<WeightPiece>
          args[0] = (async function* () {
            for await (const piece of pieces) {
              if (
                captureSample &&
                piece.kind === "embedding" &&
                piece.value.bits === 6
              ) {
                const m = piece.value,
                  row = piece.row0 ?? 0
                for (const group of new Set([0, m.N / 4 - 1]))
                  uploadSamples.push({
                    group: row / 4 + group,
                    bytes: expandQ6(m, group, 1).quants
                  })
                if (row < 8192) {
                  nativeSample ??= newQuantMatrix(
                    Math.min(8192, piece.totalRows ?? m.N),
                    m.K,
                    6,
                    false
                  )
                  const rows = Math.min(m.N, nativeSample.N - row),
                    blocks = m.K / 256
                  nativeSample.quants.set(
                    m.quants.subarray(0, rows * blocks * 52),
                    row * blocks * 52
                  )
                  nativeSample.scales.set(
                    m.scales.subarray(0, rows * blocks),
                    row * blocks
                  )
                }
              }
              yield piece
            }
          })()
        }
      }
      postMessage({ type: "bench-phase", name, gpuBytes: this.gpu.bytes })
      const value = await Reflect.apply(original, this, args)
      postMessage({ type: "bench-phase", name: `${name}:end`, gpuBytes: this.gpu.bytes })
      return value
    }
  })
}

for (const name of ["texture", "write"] as const) {
  const original = Gpu.prototype[name]
  Object.assign(Gpu.prototype, {
    [name]: function (this: Gpu, ...args: unknown[]) {
      const start = performance.now()
      const value = Reflect.apply(original, this, args)
      peakGpuBytes = Math.max(peakGpuBytes, this.bytes)
      if (performance.now() - start > 10)
        postMessage({
          type: "bench-gpu",
          name,
          start: performance.timeOrigin + start,
          ms: performance.now() - start,
          shape: name === "write" ? args[1] : args.slice(0, 4)
        })
      return value
    }
  })
}


// Shader corpus (tune "dumpShaders:1"): each program's source with the formats
// and uniforms it ran with, posted to /dump (bench/shaders.jsonl) when the GPU
// is disposed. webgpu-translate.html compiles the corpus as WGSL.
let dumpShaders = false
type Shape = { internal: number; width: number; height: number }
const corpus = new Map<string, {
  names: Set<string>
  inputs: Record<string, number>
  outputs: number[]
  uniforms: Record<string, string>
  /** The last pass's, to replay it (kernel-ab.html). */
  last?: Replay
  /** The largest pass's (by rectangle). */
  biggest?: Replay
}>()
type Replay = { uniforms: unknown; inputs: Record<string, Shape>; outputs: Array<Shape>; rect: { width: number; height: number }; name?: string }
const record = (gpu: Gpu | WebGpu, pass: Parameters<Gpu["run"]>[0]) => {
    const own = (pass.program as { source?: string }).source
    const source = own ?? [...Reflect.get(gpu, "programs") as Map<string, unknown>].find(([, p]) => p === pass.program)?.[0]
    if (source !== undefined) {
      let e = corpus.get(source)
      if (!e) corpus.set(source, (e = { names: new Set(), inputs: {}, outputs: [], uniforms: {} }))
      if (pass.name) e.names.add(pass.name)
      for (const [k, t] of Object.entries(pass.inputs)) e.inputs[k] = t.internal
      e.outputs = (Array.isArray(pass.output) ? pass.output : [pass.output]).map((t) => t.internal)
      for (const [k, v] of Object.entries(pass.uniforms ?? {}))
        e.uniforms[k] = typeof v === "number" ? "float" : Array.isArray(v) ? `vec${v.length}` : Object.keys(v)[0]
      const shape = (t: { internal: number; width: number; height: number }) => ({ internal: t.internal, width: t.width, height: t.height })
      const outs = Array.isArray(pass.output) ? pass.output : [pass.output]
      e.last = {
        name: pass.name,
        uniforms: Object.fromEntries(Object.entries(pass.uniforms ?? {}).map(([k, v]) => [k, typeof v === "object" && "ivec4s" in v ? { ivec4s: Array.from(v.ivec4s) } : v])),
        inputs: Object.fromEntries(Object.entries(pass.inputs).map(([k, t]) => [k, shape(t)])),
        outputs: outs.map(shape),
        rect: pass.rect ?? { x: 0, y: 0, width: outs[0].width, height: outs[0].height }
      }
      const area = (r?: Replay) => (r ? r.rect.width * r.rect.height : -1)
      if (area(e.last) > area(e.biggest)) e.biggest = e.last
    }
}
// Draws (WebGL) or dispatches (WebGPU) and passes issued while writing.
const writing = { draws: 0, passes: 0, at: null as null | { draws: number; passes: number } }
for (const C of [Gpu, WebGpu] as const) {
  const setPhase = C.prototype.setPhase
  C.prototype.setPhase = function (this: Gpu & WebGpu, phase) {
    if (this.activePhase === "decode" && phase !== "decode" && writing.at) {
      writing.draws += this.draws - writing.at.draws
      writing.passes += this.passes - writing.at.passes
      writing.at = null
    }
    if (phase === "decode" && this.activePhase !== "decode") writing.at = { draws: this.draws, passes: this.passes }
    return setPhase.call(this, phase)
  }
}
const instrumented = () => countWriting || dumpShaders || burnUs > 0 || skipWhileWriting.size > 0 || twiceWhileWriting.size > 0
// Passes while writing, by kind and size, and in independent groups (tune "countWriting:1").
let countWriting = false
const writingRuns = new Map<string, number>()
let group = { written: new Set<number>(), read: new Set<number>(), n: 0 }
const groups = { count: 0, passes: 0, sizes: new Map<number, number>() }
const NOTES_DIR = "webgl-models"
// Lanes forced past the engine's choice (tune "coopThreads:<n>").
let forcedLanes: number | undefined
{
  const create = WebGpu.create
  WebGpu.create = async () => {
    const gpu = await create()
    if (gpu && forcedLanes !== undefined) {
      const n = forcedLanes
      Object.defineProperty(gpu, "coopThreads", { get: () => n, set: () => {} })
    }
    return gpu
  }
}
// Pass kinds run twice while writing (tune "twice_<name>:1"): what each costs a token, in place.
const twiceWhileWriting = new Set<string>()
// CPU time added to each pass while writing (tune "burnUs:<µs>"): whether the GPU waits on the CPU.
let burnUs = 0
// Pass kinds not run while writing (tune "skip_<name>:1"): what each costs a token.
const skipWhileWriting = new Set<string>()
for (const C of [Gpu, WebGpu] as const) {
  const run = C.prototype.run
  C.prototype.run = function (this: Gpu & WebGpu, pass) {
    if (!instrumented()) return run.call(this, pass)
    if (countWriting && this.activePhase === "decode") {
      // Greedy groups of consecutive passes none of which reads or writes
      // what an earlier one in its group wrote (or writes what it read).
      const ins = Object.values(pass.inputs).map((t) => t.id)
      const outs = (Array.isArray(pass.output) ? pass.output : [pass.output]).map((t) => t.id)
      if (ins.some((id) => group.written.has(id)) || outs.some((id) => group.written.has(id) || group.read.has(id))) {
        groups.count++
        groups.sizes.set(group.n, (groups.sizes.get(group.n) ?? 0) + 1)
        group = { written: new Set(), read: new Set(), n: 0 }
      }
      for (const id of ins) group.read.add(id)
      for (const id of outs) group.written.add(id)
      group.n++
      groups.passes++
      const r = pass.rect ?? { width: 0, height: 0 }
      const key = `${pass.name ?? "?"} ${r.width}x${r.height}`
      writingRuns.set(key, (writingRuns.get(key) ?? 0) + 1)
    }
    if (pass.name && skipWhileWriting.has(pass.name) && this.activePhase === "decode") return
    if (pass.name && twiceWhileWriting.has(pass.name) && this.activePhase === "decode") run.call(this, pass)
    if (burnUs > 0 && this.activePhase === "decode") for (const t = performance.now(); performance.now() - t < burnUs / 1000;);
    if (dumpShaders) record(this, pass)
    return run.call(this, pass)
  }
}

const dumpCorpus = async () =>
  dumpShaders && fetch("/dump", { method: "POST", body: JSON.stringify([...corpus].map(([source, e]) => ({ source, ...e, names: [...e.names] }))) })

// WebGPU per-pass GPU times, by phase (tune "wgprofile:1"): logged when a reply is done.
let wgProfiling = false
const wgProfiles = { prompt: new Map<string, { ms: number; n: number }>(), decode: new Map<string, { ms: number; n: number }>() }
{
  const setPhase = WebGpu.prototype.setPhase
  WebGpu.prototype.setPhase = function (this: WebGpu, phase) {
    setPhase.call(this, phase)
    if (wgProfiling) this.profile = phase === "load" ? null : wgProfiles[phase]
  }
}
// CPU time spent issuing WebGPU work, by phase.
const wgCpu = { run: 0, runs: 0, submit: 0, submits: 0 }
{
  const run = WebGpu.prototype.run
  WebGpu.prototype.run = function (this: WebGpu, pass) {
    const t = performance.now()
    try { return run.call(this, pass) } finally { wgCpu.run += performance.now() - t; wgCpu.runs++ }
  }
  const submit = Reflect.get(WebGpu.prototype, "submit") as () => void
  Reflect.set(WebGpu.prototype, "submit", function (this: WebGpu) {
    const t = performance.now()
    try { return submit.call(this) } finally { wgCpu.submit += performance.now() - t; wgCpu.submits++ }
  })
}
const logWgProfile = () =>
  logWgProfileNow().catch((e) => fetch("/log", { method: "POST", body: `WGPROFILE failed: ${(e as Error).stack ?? e}` }))
const logWgProfileNow = async () => {
  const gpu = loadedEngine?.gpu
  await fetch("/log", { method: "POST", body: `GROUPS ${groups.passes} passes in ${groups.count} independent groups; sizes ${[...groups.sizes].sort((x, y) => x[0] - y[0]).map(([k, n]) => `${k}:${n}`).join(" ")}` })
  await fetch("/log", { method: "POST", body: `WRITINGRUNS ${[...writingRuns].sort((x, y) => y[1] - x[1]).map(([k, n]) => `${k}=${n}`).join(", ")}` })
  await fetch("/log", { method: "POST", body: `WRITING draws ${writing.draws} passes ${writing.passes}` })
  await fetch("/log", { method: "POST", body: `TUNED ${JSON.stringify(loadedEngine?.tuned)} report ${JSON.stringify(loadedEngine?.tuneReport)}` })
  if (!(gpu instanceof WebGpu)) return
  await fetch("/log", { method: "POST", body: `WGFEATURES ${[...gpu.device.features].join(",")}` })
  await fetch("/log", { method: "POST", body: `WGCPU run ${wgCpu.run.toFixed(1)} ms / ${wgCpu.runs}, submit ${wgCpu.submit.toFixed(1)} ms / ${wgCpu.submits}` })
  Object.assign(wgCpu, { run: 0, runs: 0, submit: 0, submits: 0 })
  if (gpu.errors.length) await fetch("/log", { method: "POST", body: `WGERRORS ${gpu.errors.join(" || ")}` })
  if (!wgProfiling) return
  await gpu.profiled()
  for (const [phase, map] of Object.entries(wgProfiles)) {
    const rows = [...map].sort((a, b) => b[1].ms - a[1].ms)
    const total = rows.reduce((s, [, e]) => s + e.ms, 0)
    const text = `WGPROFILE ${phase} total ${total.toFixed(1)} ms: ` + rows.map(([n, e]) => `${n || "?"} ${e.ms.toFixed(1)}/${e.n}`).join(", ")
    await fetch("/log", { method: "POST", body: text })
    map.clear()
  }
}

await import("../src/worker.js")

const productionHandler = onmessage
// Messages handled one at a time, in order (some handling awaits).
let handling = Promise.resolve()
onmessage = (event) => {
  handling = handling.then(() => handle(event)).catch((e) => console.error(e))
}
const handle = async (event: MessageEvent) => {
  if (event.data.type === "bench-dispose") {
    await logWgProfile()
    await dumpCorpus()
    await loadedEngine?.gpu.disposePaced()
    postMessage({ type: "bench-disposed" })
  } else if (event.data.type === "bench-profile-config") {
    captureSample = true
    // Keep calibration from changing settings during an isolated profile.
    Engine.prototype.tune = async function () {
      return { ...this.tuned }
    }
  } else if (event.data.type === "bench-profile") {
    if (!loadedEngine) throw new Error("No model loaded for profile")
    void profileGpu(
      loadedEngine,
      (data) => postMessage({ type: "bench-profile-data", data }),
      nativeSample,
      event.data.qualityOnly === true,
      uploadSamples
    )
      .then(() => postMessage({ type: "bench-profile-done" }))
      .catch((error) =>
        postMessage({ type: "bench-profile-error", message: String(error) })
      )
  } else {
    if (event.data.type === "tune" && event.data.values?.wgprofile) {
      wgProfiling = true
      delete event.data.values.wgprofile
    }
    // (coopThreads:<n> forces lanes past the tuned choice; lanes:<n> is TUNING's.)
    if (event.data.type === "tune" && event.data.values?.coopThreads !== undefined) {
      forcedLanes = event.data.values.coopThreads
      delete event.data.values.coopThreads
    }
    if (event.data.type === "tune")
      for (const key of Object.keys(event.data.values ?? {}))
        if (key.startsWith("skip_")) {
          skipWhileWriting.add(key.slice(5))
          delete event.data.values[key]
        }
        else if (key.startsWith("twice_")) {
          twiceWhileWriting.add(key.slice(6))
          delete event.data.values[key]
        }
    if (event.data.type === "tune" && event.data.values?.burnUs !== undefined) {
      burnUs = event.data.values.burnUs
      delete event.data.values.burnUs
    }
    if (event.data.type === "tune" && event.data.values?.wgx !== undefined) {
      WebGpu.workgroup = [event.data.values.wgx, event.data.values.wgy ?? 1]
      delete event.data.values.wgx
      delete event.data.values.wgy
    }
    // (dropNotes:1 forgets WebGPU's stored tuning, so this load tunes afresh.)
    if (event.data.type === "tune" && event.data.values?.dropNotes) {
      delete event.data.values.dropNotes
      const dir = await (await navigator.storage.getDirectory()).getDirectoryHandle(NOTES_DIR, { create: true })
      const handle = await dir.getFileHandle("gpu-notes.json", { create: true })
      const notes = JSON.parse((await (await handle.getFile()).text()) || "{}") as Record<string, unknown>
      for (const k of Object.keys(notes)) if (k.startsWith("WebGPU")) delete notes[k]
      const access = await handle.createSyncAccessHandle()
      const bytes = new TextEncoder().encode(JSON.stringify(notes))
      access.truncate(0)
      access.write(bytes, { at: 0 })
      access.flush()
      access.close()
    }
    if (event.data.type === "tune" && event.data.values?.countWriting) {
      countWriting = true
      delete event.data.values.countWriting
    }
    if (event.data.type === "tune" && event.data.values?.coarse) {
      WebGpu.coarse = true
      delete event.data.values.coarse
    }
    if (event.data.type === "tune" && event.data.values?.dumpShaders) {
      dumpShaders = true
      delete event.data.values.dumpShaders
    }
    productionHandler?.call(self, event)
  }
}

const send = postMessage.bind(self)
const reported = new Set<string>()
postMessage = (message) => {
  send(message)
  if (
    loadedEngine &&
    (message.type === "done" ||
      (message.type === "token" && !reported.has(message.id)))
  ) {
    reported.add(message.id)
    const states = Reflect.get(loadedEngine.gpu, "budgets") as Record<
      string,
      { reads: number; ahead: number; layers: number }
    >
    send({
      type: "bench-pacing",
      when: message.type,
      states: Object.fromEntries(
        Object.entries(states).map(([name, b]) => [
          name,
          {
            reads: b.reads,
            ahead: b.ahead,
            layers: b.layers,
            ceiling: Reflect.get(b, "ceiling"),
            initial: Reflect.get(b, "initial"),
            healthy: Reflect.get(b, "healthyFrame"),
            frameMs: Reflect.get(b, "frameMs")
          }
        ])
      )
    })
  }
}

Object.assign(globalThis, {
  __benchMemory: () => {
    const f = loadedEngine && Reflect.get(loadedEngine, "f16")
    return {
      gpuBytes: loadedEngine?.gpu.bytes ?? 0,
      peakGpuBytes,
      scratchBytes: f ? (f.w?.bytes ?? f.wh2?.bytes ?? 0) : 0,
      activationBytes:
        f?.a?.reduce((n: number, t: { bytes: number }) => n + t.bytes, 0) ?? 0
    }
  }
})
