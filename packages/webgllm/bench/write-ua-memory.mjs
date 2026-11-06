/** Turn an app-reported user-agent memory sample into a named evidence file. */
import fs from "node:fs"
const [rawFile, id, output] = process.argv.slice(2)
if (!rawFile || !id || !output) throw Error("Expected RAW.jsonl ID OUTPUT.json")
const all = fs
  .readFileSync(rawFile, "utf8")
  .trim()
  .split("\n")
  .map(JSON.parse)
  .filter((row) => row.id === id)
const start = all.findLastIndex((row) => row.phase === "environment")
const rows = all.slice(Math.max(start, 0))
const ua = rows.findLast(
  (row) => row.phase === "ua-memory" && row.when === "retained"
)
const complete = rows.some((row) => row.phase === "complete")
if (
  !complete ||
  !ua ||
  rows.some(
    (row) =>
      row.phase === "visibility" &&
      row.state === "hidden" &&
      row.at < rows.find((row) => row.phase === "complete").at
  )
)
  throw Error(`Incomplete or backgrounded memory capture: ${id}`)
const memory = rows.filter((row) => row.phase === "memory" && Number.isFinite(row.peakGpuBytes))
const done = memory.findLast((row) => row.when === "done") ?? memory.at(-1)
if (!done) throw Error("Missing GPU counters")
fs.writeFileSync(
  output,
  JSON.stringify(
    {
      kind: "ua-page",
      id,
      url:
        "http://localhost:8788/webgl-perf.html?" +
        new URLSearchParams(rows[0].query),
      complete,
      retainedPageBytes: ua.bytes,
      breakdown: ua.breakdown,
      gpuBytes: done.gpuBytes,
      peakGpuBytes: Math.max(...memory.map((row) => row.peakGpuBytes)),
      scope:
        "measureUserAgentSpecificMemory after GC: this page plus its dedicated worker; GPU texture bytes separate. Not natural CPU peak or process/phone RSS."
    },
    null,
    2
  ) + "\n"
)
