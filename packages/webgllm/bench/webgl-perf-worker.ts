/** Phase instrumentation around the same worker and engine used by the app. */
import { Gpu } from "../src/gpu/gl.js"
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
      return Reflect.apply(original, this, args)
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

await import("../src/worker.js")

const productionHandler = onmessage
onmessage = (event) => {
  if (event.data.type === "bench-dispose") {
    loadedEngine?.gpu.dispose()
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
  } else productionHandler?.call(self, event)
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
