/**
 * A worker of the page's own for unload-headroom.html: webgllm's handler,
 * plus a "churn" message that makes and drops garbage in the worker, to
 * see whether what an unloaded model leaves is held until a collection.
 */
import { WebGLLMWorkerHandler } from "../src/index.js"

const handler = new WebGLLMWorkerHandler()
self.onmessage = (event) => handler.onmessage(event)

self.addEventListener("message", async ({ data }) => {
  if (data?.type !== "churn") return
  // 32 rounds of 16 MB made, touched and dropped, with a turn of the event loop between.
  let sum = 0
  for (let i = 0; i < 32; i++) {
    const block = new Uint8Array(16 << 20)
    for (let j = 0; j < block.length; j += 4096) block[j] = i
    sum += block[4096]!
    // Many small objects too, for the collector's young generation.
    const small: Array<object> = []
    for (let j = 0; j < 100_000; j++) small.push({ j })
    sum += small.length
    await new Promise((resolve) => setTimeout(resolve, 20))
  }
  postMessage({ type: "churned", sum })
})

// "probe": does a FinalizationRegistry in this worker report objects the collector took? A plain object made a
// while ago, and a destroyed WebGPU device, each dropped, then garbage until each is reported (at most 256 rounds).
self.addEventListener("message", async ({ data }) => {
  if (data?.type !== "probe") return
  // mode "promote": each round's garbage is kept for `keep` rounds (so it outlives young collections and the old
  // generation grows, as a full collection needs); else dropped at once.
  const promote = data.mode === "promote"
  const keep = Number(data.keep ?? 4)
  const ring: Array<unknown> = []
  const seen = new Set<string>()
  const registry = new FinalizationRegistry<string>((what) => seen.add(what))
  let plain: object | null = { values: new Array(10_000).fill(1) }
  const adapter = await (navigator as Navigator & { gpu?: GPU }).gpu?.requestAdapter()
  let device: GPUDevice | null = (await adapter?.requestDevice()) ?? null
  // (Aged first: through a few collections, as an engine is by the time it's unloaded.)
  for (let i = 0; i < 16; i++) {
    const objects: Array<object> = []
    for (let j = 0; j < 50_000; j++) objects.push({ j })
    void new ArrayBuffer(16 << 20)
    await new Promise((resolve) => setTimeout(resolve, 0))
  }
  registry.register(plain, "plain")
  const refs: Record<string, WeakRef<object>> = { plain: new WeakRef(plain) }
  plain = null
  if (device) {
    device.destroy()
    registry.register(device, "device")
    refs.device = new WeakRef(device)
    device = null
  }
  const rounds: Record<string, number> = {}
  const started = performance.now()
  for (let round = 0; round < 256; round++) {
    const objects: Array<object> = []
    for (let j = 0; j < 50_000; j++) objects.push({ round })
    if (promote) {
      ring.push(objects, new Array(1 << 20).fill(round))
      while (ring.length > 2 * keep) ring.shift()
    } else void new ArrayBuffer(16 << 20)
    await new Promise((resolve) => setTimeout(resolve, 0))
    // Checked in a task of its own: deref keeps its target alive to the end of the task it's called in.
    for (const what of seen) rounds[what] ??= round
    for (const [what, ref] of Object.entries(refs)) if (!ref.deref()) {
      rounds[`weak ${what}`] ??= round
      rounds[`weak ${what} ms`] ??= Math.round(performance.now() - started)
    }
    if (Object.keys(refs).every((what) => `weak ${what}` in rounds) && round >= 8) break
    await new Promise((resolve) => setTimeout(resolve, 0))
  }
  postMessage({ type: "probed", rounds })
})
