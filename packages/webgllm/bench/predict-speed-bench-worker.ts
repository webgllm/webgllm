/**
 * Bench only: predictSpeed's worker (src/predict-speed-worker.ts, the same
 * steps) with the backend from its URL (?webgpu=0|1) set before it measures.
 * (Its own code, not an import of that worker: the package is side-effect
 * free, so a bare import of it is dropped from the bundle.)
 */
import { TUNING, type Engine } from "../src/engine.js"
import { predictHere, realSpeedDeps, type SpeedWorkerReply, type SpeedWorkerRequest } from "../src/predict-speed.js"

const webgpu = new URLSearchParams(self.location.search).get("webgpu")
if (webgpu !== null) TUNING.webgpu = Number(webgpu)

let probing: Engine | null = null
self.onmessage = async ({ data }: MessageEvent<SpeedWorkerRequest>) => {
  if (data?.type === "frames") {
    probing?.gpu.observeFrames(data.intervalMs, data.expectedMs, data.frames, data.startedAt, data.meanMs)
    return
  }
  if (data?.type !== "predict-speed") return
  let reply: SpeedWorkerReply
  try {
    const { prediction, report } = await predictHere(data.model, { force: data.force }, realSpeedDeps((engine) => (probing = engine)))
    reply = { type: "speed", prediction, report }
  } catch (error) {
    const e = error instanceof Error ? error : new Error(String(error))
    reply = { type: "speed-error", name: e.name, message: e.message }
  }
  self.postMessage(reply)
}
