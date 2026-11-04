/** Sequential startup tests; navigate only the explicitly selected Safari tab. */
import fs from "node:fs"
import { execFile } from "node:child_process"
import { promisify } from "node:util"
const runFile = promisify(execFile)
const [
  socket,
  origin,
  mode,
  prefix,
  output,
  countArg = "3",
  stateArg = "both"
] = process.argv.slice(2)
if (
  !socket ||
  !origin ||
  !["baseline", "production"].includes(mode) ||
  !prefix ||
  !output
)
  throw Error(
    "Expected WebKit socket, origin, baseline|production, prefix, output, [count] [both|cold|cached]"
  )
const helper = new URL("./safari-bench.mjs", import.meta.url).pathname
const states =
  stateArg === "cold" ? [true] : stateArg === "cached" ? [false] : [true, false]
const count = Number(countArg),
  results = []
const suiteId = `${prefix}-${Date.now().toString(36)}`
const call = async (args) => {
  const r = await runFile(process.execPath, [helper, socket, ...args], {
    maxBuffer: 4 * 1024 * 1024
  })
  if (r.stdout) process.stdout.write(r.stdout)
  return r
}
try {
  for (const cold of states)
    for (let i = 1; i <= count; i++) {
      const id = `${suiteId}-${cold ? "cold" : "cached"}-${i}`
      const url = `${origin}/webgl-perf.html?id=${id}&${mode}=1&hold=1&rounds=0&len=512${cold ? "&cold=1" : ""}`
      await call([
        "run",
        url,
        new URL(`./results/${id}.json`, import.meta.url).pathname,
        cold && mode === "baseline" ? "background-tune" : "no-memory"
      ])
      const rows = fs
        .readFileSync(
          new URL("./results/webgl-live.jsonl", import.meta.url),
          "utf8"
        )
        .trim()
        .split("\n")
        .map(JSON.parse)
        .filter((r) => r.id === id)
      const load = rows.filter((r) => r.phase === "load").at(-1)
      if (
        !load?.foregroundValid ||
        !rows.some((r) => r.phase === "complete") ||
        rows.some((r) => r.phase === "visibility" && r.state === "hidden")
      )
        throw Error(`Invalid startup capture: ${id}`)
      const calibration = rows
        .filter((r) => r.phase === "calibration-complete")
        .at(-1)
      results.push({
        id,
        cold,
        loadMs: load.durationMs,
        engineLoadMs: load.timings.total,
        fullyCalibratedMs: calibration?.durationMs ?? load.durationMs,
        fps: load.fps,
        maxFrameMs: load.max,
        timings: load.timings,
        gpuBytes: load.gpuBytes,
        foregroundValid: load.foregroundValid,
        visibilityHistory: load.visibilityHistory
      })
    }
} finally {
  fs.writeFileSync(
    output,
    JSON.stringify(
      { mode, results, complete: results.length === states.length * count },
      null,
      2
    ) + "\n"
  )
}
