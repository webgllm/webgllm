/** On-device production-worker benchmark: painted frames during every phase. */
import { monitorFrames } from "../src/frame-pacing.js"
import type { WorkerResponse, RunTimings } from "../src/protocol.js"
const params = new URLSearchParams(location.search)
const id = params.get("id") ?? String(Date.now())
const baseline = params.get("baseline") === "1"
const heading = document.querySelector("h1")
if (!heading) throw new Error("Benchmark heading missing")
heading.textContent = baseline
  ? "BEFORE · Original engine"
  : "AFTER · Optimized engine"
heading.style.color = baseline ? "#ffb4a8" : "#71e4ba"
document.title = baseline
  ? "BEFORE — WebGL performance"
  : "AFTER — WebGL performance"
const log = (s: string) => {
  document.querySelector("pre")!.textContent += `${s}\n`
  void fetch("/log", { method: "POST", body: `${id} ${s}` })
}
const result = (value: object) =>
  fetch("/result", {
    method: "POST",
    body: JSON.stringify({ id, at: new Date().toISOString(), ...value })
  })
const canvas = document.querySelector("canvas")!
const ctx = canvas.getContext("2d")!
let phase = "idle",
  phaseStart = performance.now(),
  last = performance.now(),
  loadingStartedAt = 0
let gaps: number[] = [],
  taps = 0
const gpuEvents: unknown[] = []
let stageEvents: Array<{ name: string; at: number }> = []
const frameTimes: Array<{ at: number; gap: number }> = []
const tap = document.querySelector("button")!
tap.onclick = (event) => {
  tap.textContent = `Test touch response: ${++taps}`
  void result({
    phase: "touch",
    during: phase,
    inputDelayMs: performance.now() - event.timeStamp,
    taps
  })
}
function frame(t: number) {
  if (phase !== "complete") {
    gaps.push(t - last)
    frameTimes.push({ at: t, gap: t - last })
  }
  last = t
  ctx.fillStyle = "#13171c"
  ctx.fillRect(0, 0, 700, 180)
  ctx.fillStyle = "#71e4ba"
  ctx.beginPath()
  ctx.arc(350 + 290 * Math.sin(t / 600), 90, 28, 0, Math.PI * 2)
  ctx.fill()
  ctx.fillStyle = "#fff"
  ctx.font = "24px sans-serif"
  ctx.fillText(`${phase} ${((t - phaseStart) / 1000).toFixed(1)}s`, 10, 35)
  requestAnimationFrame(frame)
}
requestAnimationFrame(frame)
function begin(name: string) {
  document.documentElement.dataset.phase = name
  phase = name
  phaseStart = performance.now()
  if (name === "load") loadingStartedAt = phaseStart
  last = phaseStart
  gaps = []
  log(`BEGIN ${name}`)
}
const visibilityEvents = [{ name: document.visibilityState, startTime: 0 }]
function visibilityHistory() {
  return PerformanceObserver.supportedEntryTypes.includes("visibility-state")
    ? performance
        .getEntriesByType("visibility-state")
        .map((e) => ({ name: e.name, startTime: e.startTime }))
    : visibilityEvents
}
document.addEventListener("visibilitychange", () => {
  visibilityEvents.push({
    name: document.visibilityState,
    startTime: performance.now()
  })
  void result({
    phase: "visibility",
    state: document.visibilityState,
    time: performance.now()
  })
})
async function end(extra: object = {}) {
  // Include a frame after the last worker response to capture a trailing stall.
  await new Promise<void>((r) => requestAnimationFrame(() => r()))
  const sorted = gaps.slice().sort((a, b) => a - b),
    durationMs = performance.now() - phaseStart
  const percentile = (p: number) =>
    sorted[Math.min(sorted.length - 1, Math.floor(sorted.length * p))]
  const history = visibilityHistory()
  const atStart =
    history.filter((e) => e.startTime <= phaseStart).at(-1)?.name ??
    document.visibilityState
  const data = {
    phase,
    foregroundValid:
      atStart === "visible" &&
      document.visibilityState === "visible" &&
      !history.some((e) => e.name === "hidden" && e.startTime >= phaseStart),
    visibilityHistory: history,
    durationMs,
    frames: gaps.length,
    fps: (1000 * gaps.length) / durationMs,
    p50: percentile(0.5),
    p95: percentile(0.95),
    p99: percentile(0.99),
    max: sorted.at(-1),
    over34: gaps.filter((x) => x > 34).length,
    over100: gaps.filter((x) => x > 100).length,
    ...extra,
    stageEvents: phase === "load" ? stageEvents : undefined,
    gpuEvents: gpuEvents.filter(
      (e) =>
        (e as { start: number }).start >= performance.timeOrigin + phaseStart
    ),
    stalls: frameTimes
      .filter((f) => f.at >= phaseStart && f.gap > 34)
      .map((f) => ({ at: performance.timeOrigin + f.at, gap: f.gap }))
  }
  log(JSON.stringify(data))
  await result({
    ...data,
    subphases:
      phase === "load"
        ? stageEvents.map((e, i) => {
            const until = stageEvents[i + 1]?.at ?? performance.now()
            const fs = frameTimes.filter((f) => f.at >= e.at && f.at < until)
            return {
              name: e.name,
              durationMs: until - e.at,
              frames: fs.length,
              max: Math.max(0, ...fs.map((f) => f.gap)),
              over34: fs.filter((f) => f.gap > 34).length,
              over100: fs.filter((f) => f.gap > 100).length
            }
          })
        : undefined
  })
}
const isModelNote = (key: string) => key.includes("|LiquidAI/LFM3-1.7B-GGUF|")
async function calibrationStore(
  action: "cold" | "save" | "restore",
  name = ""
) {
  if (action !== "cold" && !/^[a-z0-9-]{1,64}$/.test(name))
    throw Error("Invalid benchmark calibration name")
  const directory = await (
    await navigator.storage.getDirectory()
  ).getDirectoryHandle("webgl-models", { create: true })
  const read = async (file: string): Promise<Record<string, unknown>> => {
    try {
      return JSON.parse(
        await (await (await directory.getFileHandle(file)).getFile()).text()
      ) as Record<string, unknown>
    } catch (error) {
      if ((error as DOMException).name === "NotFoundError") return {}
      throw error
    }
  }
  const write = async (file: string, value: object) => {
    const writable = await (
      await directory.getFileHandle(file, { create: true })
    ).createWritable()
    await writable.write(JSON.stringify(value))
    await writable.close()
  }
  const notes = await read("gpu-notes.json")
  const other = Object.fromEntries(
    Object.entries(notes).filter(([key]) => !isModelNote(key))
  )
  if (action === "save")
    await write(
      `bench-calibration-${name}.json`,
      Object.fromEntries(
        Object.entries(notes).filter(([key]) => isModelNote(key))
      )
    )
  else
    await write(
      "gpu-notes.json",
      action === "cold"
        ? other
        : { ...other, ...(await read(`bench-calibration-${name}.json`)) }
    )
  log(`Calibration ${action}: ${name || "model notes only"}`)
  await result({ phase: "calibration-storage", action, name })
}
async function capabilityInfo() {
  const c = document.createElement("canvas"),
    gl = c.getContext("webgl2", { powerPreference: "high-performance" })
  const debug = gl?.getExtension("WEBGL_debug_renderer_info")
  const info = {
    phase: "capabilities",
    renderer: gl?.getParameter(
      debug ? debug.UNMASKED_RENDERER_WEBGL : gl.RENDERER
    ) as string | undefined,
    maxTextureSize: gl?.getParameter(gl.MAX_TEXTURE_SIZE) as number | undefined,
    crossOriginIsolated,
    uaMemorySupported:
      typeof (
        performance as Performance & {
          measureUserAgentSpecificMemory?: unknown
        }
      ).measureUserAgentSpecificMemory === "function"
  }
  gl?.getExtension("WEBGL_lose_context")?.loseContext()
  log(JSON.stringify(info))
  await result(info)
}
async function userAgentMemory(when: string) {
  const measure = (
    performance as Performance & {
      measureUserAgentSpecificMemory?: () => Promise<{
        bytes: number
        breakdown?: unknown[]
      }>
    }
  ).measureUserAgentSpecificMemory
  if (!crossOriginIsolated || !measure)
    throw Error("User-agent memory measurement unavailable")
  const measured = await measure.call(performance)
  log(
    `User-agent memory (${when}): ${(measured.bytes / 2 ** 20).toFixed(2)} MiB`
  )
  await result({ phase: "ua-memory", when, ...measured })
}
async function run() {
  await result({
    phase: "environment",
    query: Object.fromEntries(params),
    ua: navigator.userAgent,
    initialVisibility: document.visibilityState,
    screen: {
      width: screen.width,
      height: screen.height,
      dpr: devicePixelRatio
    }
  })
  if (params.get("cold") === "1") await calibrationStore("cold")
  const restore = params.get("restoreCalibration")
  if (restore) await calibrationStore("restore", restore)
  if (params.get("info") === "1") await capabilityInfo()
  begin("idle")
  await new Promise((r) => setTimeout(r, Number(params.get("idleMs") ?? 2500)))
  await end()
  const worker = new Worker(
    params.get("baseline") === "1"
      ? "/dist/baseline-worker.js"
      : params.get("production") === "1"
        ? "/dist/current-worker.js"
        : "/bundle/bench/webgl-perf-worker.js",
    { type: "module" }
  )
  Object.assign(globalThis, { __benchWorker: worker })
  const stopFrames =
    params.get("baseline") === "1"
      ? () => undefined
      : monitorFrames((intervalMs, expectedMs, frames, startedAt, meanMs) =>
          worker.postMessage({
            type: "frames",
            intervalMs,
            expectedMs,
            frames,
            startedAt,
            meanMs
          })
        )
  let disposed: (() => void) | undefined
  let shutdownPromise: Promise<boolean> | undefined
  const shutdown = () =>
    (shutdownPromise ??= (async () => {
      stopFrames()
      await new Promise<void>((resolve, reject) => {
        const timeout = setTimeout(
          () => reject(new Error("GPU cleanup timed out")),
          5000
        )
        disposed = () => {
          clearTimeout(timeout)
          resolve()
        }
        worker.postMessage({ type: "bench-dispose" })
      }).finally(() => worker.terminate())
      return true
    })())
  Object.assign(globalThis, { __benchShutdown: shutdown })
  const release = document.querySelector<HTMLButtonElement>("#release")
  if (!release) throw Error("Missing benchmark release control")
  release.disabled = false
  release.onclick = () => {
    void shutdown().then(() => {
      release.textContent = "Model released"
      release.disabled = true
    })
  }
  let tuningFinished: (() => void) | undefined
  const tuning = new Promise<void>((resolve) => {
    tuningFinished = resolve
  })
  let profileDone: (() => void) | undefined
  let ready: () => void,
    reject: (e: Error) => void,
    done: (s: RunTimings) => void
  let first = true,
    reply = "",
    lastProgress = -1,
    gpuBytes = 0
  worker.onmessage = ({
    data
  }: {
    data:
      | { type: "bench-profile-data"; data: object }
      | {
          type: "bench-memory"
          when: string
          gpuBytes: number
          peakGpuBytes: number
          pacing?: object
        }
      | { type: "bench-pacing"; when: string; states: object }
      | { type: "bench-disposed" }
      | { type: "bench-profile-done" }
      | { type: "bench-profile-error"; message: string }
      | WorkerResponse
      | { type: "bench-phase"; name: string }
      | {
          type: "bench-gpu"
          name: string
          start: number
          ms: number
          shape: unknown
        }
  }) => {
    if (data.type === "bench-disposed") {
      disposed?.()
      return
    }
    if (data.type === "bench-memory") {
      void result({
        phase: "memory",
        when: data.when,
        gpuBytes: data.gpuBytes,
        peakGpuBytes: data.peakGpuBytes,
        pacing: data.pacing
      })
      return
    }
    if (data.type === "bench-pacing") {
      void result({ phase: "pacing", when: data.when, states: data.states })
      return
    }
    if (data.type === "bench-profile-data") {
      void result({ phase: "profile", ...data.data })
      log(JSON.stringify(data.data))
      return
    }
    if (data.type === "bench-profile-done") {
      profileDone?.()
      return
    }
    if (data.type === "bench-profile-error") {
      reject(new Error(data.message))
      return
    }
    if (data.type === "bench-gpu") {
      gpuEvents.push(data)
      return
    }
    if (data.type === "bench-phase") {
      stageEvents.push({ name: data.name, at: performance.now() })
      return
    }
    if (data.type === "claim") {
      gpuBytes = data.bytes
      worker.postMessage({ type: "claimed" })
    } else if (data.type === "ready" && phase === "load") {
      void end({ timings: data.timings, gpuBytes }).then(() => ready())
    } else if (data.type === "token") {
      reply += data.text
      if (first) {
        first = false
        void end().then(() => begin("decode"))
      }
    } else if (data.type === "done") done(data.stats)
    else if (data.type === "load-error" || data.type === "error")
      reject(new Error(data.message))
    else if (data.type === "log") {
      log(data.text)
      if (data.text.startsWith("tuned in the background:")) {
        void result({
          phase: "calibration-complete",
          durationMs: performance.now() - loadingStartedAt
        })
        tuningFinished?.()
      }
    } else if (data.type === "progress") {
      const pct = Math.floor(data.progress * 10)
      if (pct !== lastProgress) {
        lastProgress = pct
        log(`loading ${pct * 10}% cached=${data.cached}`)
      }
    }
  }
  worker.onerror = (e) => reject(new Error(e.message))
  const tune = Object.fromEntries(
    (params.get("tune") ?? "")
      .split(",")
      .filter(Boolean)
      .map((s) => {
        const [k, v] = s.split(":")
        return [k, Number(v)]
      })
  )
  if (Object.keys(tune).length)
    worker.postMessage({ type: "tune", values: tune })
  if (params.get("profile") === "1" || params.get("notune") === "1")
    worker.postMessage({ type: "bench-profile-config" })
  const model = "LiquidAI/LFM3-1.7B-GGUF"
  stageEvents = []
  begin("load")
  await new Promise<void>((r, j) => {
    ready = r
    reject = j
    worker.postMessage({ type: "load", model })
  })
  if (baseline && params.get("waitTune") === "1") {
    begin("calibration")
    await tuning
    await end()
  }
  // Runtime-only controls keep startup/calibration identical in isolated A/B tests.
  const inferenceTune = Object.fromEntries(
    (params.get("inferenceTune") ?? "")
      .split(",")
      .filter(Boolean)
      .map((value) => {
        const [key, setting] = value.split(":")
        return [key, Number(setting)]
      })
  )
  if (Object.keys(inferenceTune).length)
    worker.postMessage({ type: "tune", values: inferenceTune })
  const rounds = Number(params.get("rounds") ?? 3),
    length = Number(params.get("len") ?? 512)
  const filler =
    "The river runs past the old mill, where the miller's daughter counts sacks of grain each morning before the market opens in the square. "
  for (let i = 0; i < rounds; i++) {
    const rest = Number(params.get("restMs") ?? 0)
    if (i && rest > 0) {
      begin("rest")
      await new Promise((resolve) => setTimeout(resolve, rest))
      await end()
    }
    first = true
    reply = ""
    begin("prefill")
    const stats = await new Promise<RunTimings>((r, j) => {
      done = r
      reject = j
      worker.postMessage({
        type: "generate",
        id: `run${i}`,
        model,
        chat: [
          {
            role: "user",
            content: `Trial ${i}.\n${filler.repeat(Math.ceil(length / 29))}\nDescribe the river scene in a detailed paragraph.`
          }
        ],
        tools: [],
        thinking: false,
        sampling: {
          temperature: Number(params.get("temperature") ?? 0),
          top_k: Number(params.get("topk") ?? 1),
          repetition_penalty: 1.1
        },
        maxTokens: Number(params.get("tokens") ?? 64),
        pipeline: params.get("pipeline") !== "0",
        speculative: params.get("speculative") !== "0"
      })
    })
    await end({
      round: i,
      stats,
      promptTps:
        (1000 * (stats.inputTokens - stats.cachedTokens)) / stats.promptMs,
      decodeTps: (1000 * (stats.outputTokens - 1)) / stats.decodeMs,
      reply
    })
  }
  if (params.get("profile") === "1") {
    begin("profile")
    await new Promise<void>((resolve, rejectProfile) => {
      profileDone = resolve
      reject = rejectProfile
      worker.postMessage({
        type: "bench-profile",
        qualityOnly: params.get("qualityOnly") === "1"
      })
    })
    await end()
  }
  const save = params.get("saveCalibration")
  if (save) await calibrationStore("save", save)
  if (params.get("uaMemory") === "1") {
    begin("memory")
    await userAgentMemory("retained")
    await end()
  }
  stopFrames()
  if (params.get("hold") !== "1") await shutdown()
  begin("complete")
  await result({ phase: "complete" })
  log("ALL DONE")
  // Completed telemetry is already persisted by /result. Do not retain
  // every frame through Safari's back/forward cache between test pages.
  frameTimes.length = gaps.length = gpuEvents.length = stageEvents.length = 0
}
void run().catch(async (e) => {
  document.documentElement.dataset.phase = "error"
  log(`ERROR ${e.stack}`)
  try {
    await (
      globalThis as { __benchShutdown?: () => Promise<boolean> }
    ).__benchShutdown?.()
  } catch {
    /* A failed worker may already be gone. */
  }
  await result({ phase: "error", error: String(e) })
})
