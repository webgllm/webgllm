/** Sequential load benchmarks on the already-forwarded Chrome CDP tab. */
import fs from "node:fs"
const [
  mode,
  prefix,
  output,
  countArg = "3",
  stateArg = "both",
  cdpOrigin = "http://127.0.0.1:9223"
] = process.argv.slice(2)
if (!["baseline", "production"].includes(mode) || !prefix || !output)
  throw new Error(
    "Usage: node bench/load-suite-cdp.mjs baseline|production PREFIX OUTPUT [COUNT]"
  )
const count = Number(countArg)
const states =
  stateArg === "cached" ? [false] : stateArg === "cold" ? [true] : [true, false]
const targets = await fetch(`${cdpOrigin}/json/list`).then((r) => r.json())
const tab = targets.find(
  (t) => t.type === "page" && t.url.startsWith("http://localhost:8788/")
)
if (!tab) throw new Error("No WebGL benchmark tab")
const ws = new WebSocket(tab.webSocketDebuggerUrl)
await new Promise((r) => (ws.onopen = r))
let sequence = 0
const pending = new Map()
ws.onmessage = ({ data }) => {
  const m = JSON.parse(data)
  if (m.id) {
    pending.get(m.id)?.(m)
    pending.delete(m.id)
  }
}
const call = (method, params = {}) =>
  new Promise((resolve, reject) => {
    const id = ++sequence
    pending.set(id, (m) =>
      m.error ? reject(new Error(JSON.stringify(m.error))) : resolve(m)
    )
    ws.send(JSON.stringify({ id, method, params }))
  })
const evaluate = async (expression) =>
  (
    await call("Runtime.evaluate", {
      expression,
      returnByValue: true,
      awaitPromise: true
    })
  ).result?.result?.value
const pause = (ms) => new Promise((r) => setTimeout(r, ms))
const results = []
const suiteId = `${prefix}-${Date.now().toString(36)}`
try {
  for (const cold of states)
    for (let i = 1; i <= count; i++) {
      const id = `${suiteId}-${cold ? "cold" : "cached"}-${i}`
      const url = `http://localhost:8788/webgl-perf.html?id=${id}&${mode}=1&hold=1&rounds=0&len=512${cold ? "&cold=1" : ""}`
      await evaluate("globalThis.__benchShutdown?.()")
      await call("Page.bringToFront")
      await call("Page.navigate", { url })
      await call("Page.bringToFront")
      let finished = false
      for (let poll = 0; poll < 720; poll++) {
        await pause(250)
        let state
        try {
          state = await evaluate(
            `({url:location.href,phase:document.documentElement.dataset.phase,text:document.querySelector('pre')?.textContent??''})`
          )
        } catch {
          continue
        }
        if (state?.url !== url) continue
        if (state.phase === "error") throw new Error(`Benchmark failed: ${id}`)
        if (state.phase === "complete") {
          if (
            mode === "baseline" &&
            cold &&
            !state.text.includes("tuned in the background:")
          )
            continue
          finished = true
          break
        }
      }
      if (!finished) throw new Error(`Benchmark timed out: ${id}`)
      const rows = fs
        .readFileSync(
          new URL("./results/webgl-live.jsonl", import.meta.url),
          "utf8"
        )
        .trim()
        .split("\n")
        .map(JSON.parse)
      const load = rows.filter((r) => r.id === id && r.phase === "load").at(-1)
      if (!load) throw new Error(`Missing load measurement: ${id}`)
      if (!load.foregroundValid)
        throw new Error(`Benchmark was not continuously visible: ${id}`)
      const calibration = rows
        .filter((r) => r.id === id && r.phase === "calibration-complete")
        .at(-1)
      results.push({
        id,
        at: load.at,
        cold,
        loadMs: load.durationMs,
        engineLoadMs: load.timings.total,
        fullyCalibratedMs:
          calibration?.durationMs ??
          (mode === "production" || !cold ? load.durationMs : null),
        fps: load.fps,
        maxFrameMs: load.max,
        timings: load.timings,
        gpuBytes: load.gpuBytes,
        foregroundValid: load.foregroundValid,
        visibilityHistory: load.visibilityHistory
      })
      console.log(JSON.stringify(results.at(-1)))
      // The old engine calibrates after ready. Let its derived note finish
      // before measuring a subsequent load; the new engine is already idle.
      await pause(1800)
      await evaluate("globalThis.__benchShutdown?.()")
    }
} finally {
  fs.writeFileSync(
    output,
    JSON.stringify(
      { mode, results, complete: results.length === count * states.length },
      null,
      2
    ) + "\n"
  )
  ws.close()
}
