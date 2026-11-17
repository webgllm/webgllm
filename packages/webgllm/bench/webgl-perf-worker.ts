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
const corpus = new Map<string, { names: Set<string>; inputs: Record<string, number>; outputs: number[]; uniforms: Record<string, string> }>()
{
  const run = Gpu.prototype.run
  Gpu.prototype.run = function (this: Gpu, pass) {
    if (!dumpShaders) return run.call(this, pass)
    const source = [...Reflect.get(this, "programs") as Map<string, { program: unknown }>].find(([, p]) => p === pass.program)?.[0]
    if (source !== undefined) {
      let e = corpus.get(source)
      if (!e) corpus.set(source, (e = { names: new Set(), inputs: {}, outputs: [], uniforms: {} }))
      if (pass.name) e.names.add(pass.name)
      for (const [k, t] of Object.entries(pass.inputs)) e.inputs[k] = t.internal
      e.outputs = (Array.isArray(pass.output) ? pass.output : [pass.output]).map((t) => t.internal)
      for (const [k, v] of Object.entries(pass.uniforms ?? {}))
        e.uniforms[k] = typeof v === "number" ? "float" : Array.isArray(v) ? `vec${v.length}` : Object.keys(v)[0]
    }
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
const logWgProfile = async () => {
  const gpu = loadedEngine?.gpu
  if (!(gpu instanceof WebGpu)) return
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
onmessage = async (event) => {
  if (event.data.type === "bench-dispose") {
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
  if (message.type === "done") void logWgProfile()
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
