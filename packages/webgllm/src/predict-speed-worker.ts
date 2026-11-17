/**
 * The worker predictSpeed starts in a page (see predict-speed.ts): it
 * measures as a model's own worker runs, its GPU's pacing told of the
 * page's frames, then answers once.
 */
import type { Engine } from "./engine.js"
import { predictHere, realSpeedDeps, type SpeedWorkerReply, type SpeedWorkerRequest } from "./predict-speed.js"

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
