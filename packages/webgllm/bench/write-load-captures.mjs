/** Summarize independent app/worker ready events selected from benchmark captures. */
import fs from "node:fs"
const [rawFile, mode, output, ...ids] = process.argv.slice(2)
if (
  !rawFile ||
  !["baseline", "production"].includes(mode) ||
  !output ||
  !ids.length
)
  throw Error("Expected RAW.jsonl baseline|production OUTPUT.json ID ...")
const raw = fs.readFileSync(rawFile, "utf8").trim().split("\n").map(JSON.parse)
const results = ids.map((id) => {
  const all = raw.filter((row) => row.id === id)
  const rows = all.slice(
    Math.max(
      0,
      all.findLastIndex((row) => row.phase === "environment")
    )
  )
  const load = rows.find((row) => row.phase === "load")
  const end = rows.find((row) => row.phase === "complete")
  if (
    !load ||
    !end ||
    !load.foregroundValid ||
    rows.some(
      (row) =>
        row.phase === "visibility" && row.state === "hidden" && row.at < end.at
    )
  )
    throw Error(`Incomplete/backgrounded load: ${id}`)
  const cold = rows[0].query.cold === "1"
  const calibration = rows.find((row) => row.phase === "calibration-complete")
  return {
    id,
    cold,
    at: load.at,
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
  }
})
fs.writeFileSync(
  output,
  JSON.stringify({ mode, complete: true, results }, null, 2) + "\n"
)
