/** predictSpeed's measurements, each model afresh, then the public API's remembered answer (see predict-speed.html). */
import { inWorker } from "../src/predict-speed.js"

// The package from its own bundle, so predictSpeed's worker URL (next to
// index.js) resolves as an app's bundler would place it.
const entry = "/bundle/src/index.js"
const { predictSpeed } = (await import(entry)) as typeof import("../src/index.js")

const params = new URLSearchParams(location.search)
const models = (params.get("models") ?? "LiquidAI/LFM3-1.7B-GGUF,onnx-community/LFM3-1.7B-ONNX").split(",")
const rounds = Number(params.get("rounds") ?? 1)
const out = document.getElementById("out")!
const log = (text: string) => {
  out.textContent += `${text}\n`
  void fetch("/log", { method: "POST", body: text }).catch(() => undefined)
}
const round1 = (n: number) => Math.round(n * 10) / 10

/** One forced measurement in predictSpeed's own worker (the page's frames reported to it), with its report. */
const measure = (model: string) => inWorker(model, { force: true }, new Worker("/bundle/src/predict-speed-worker.js", { type: "module" }))

try {
  log(`UA ${navigator.userAgent}`)
  for (const model of models) {
    for (let r = 0; r < rounds; r++) {
      const started = performance.now()
      const { prediction, report } = await measure(model)
      const wall = performance.now() - started
      const detail = report && !report.cached
        ? {
            shapeMs: Math.round(report.shapeMs),
            loadMs: Math.round(report.loadMs),
            measureMs: Math.round(report.measureMs),
            peakMiB: Math.round(report.peakBytes / 2 ** 20),
            phases: Object.fromEntries(Object.entries(report.phases).map(([k, v]) => [k, Math.round(v)])),
            times: {
              ...report.times,
              promptBase: round1(report.times.promptBase),
              decodeLayers: round1(report.times.decodeLayers),
              headAt: round1(report.times.headAt),
              headPerRow: report.times.headPerRow,
              promptStack: round1(report.times.promptStack)
            }
          }
        : report
      log(`PREDICT ${model} round ${r} prompt ${round1(prediction.prompt)} tok/s decode ${round1(prediction.decode)} tok/s wall ${Math.round(wall)} ms ${JSON.stringify(detail)}`)
    }
    // The public API: remembered (OPFS) on the first call in this page, then in memory.
    for (const which of ["remembered", "in page"]) {
      const t = performance.now()
      const p = await predictSpeed(model)
      log(`API ${model} ${which} prompt ${round1(p.prompt)} decode ${round1(p.decode)} in ${round1(performance.now() - t)} ms`)
    }
  }
  log("ALL DONE")
} catch (error) {
  log(`ERROR: ${error instanceof Error ? error.stack ?? error.message : String(error)}`)
}
