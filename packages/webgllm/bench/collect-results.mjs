/** Preserve the latest capture for exactly the IDs referenced by device manifests. */
import fs from "node:fs"
import path from "node:path"
const [input, output, ...manifests] = process.argv.slice(2)
if (!input || !output || !manifests.length)
  throw Error(
    "Usage: node bench/collect-results.mjs RAW.jsonl SELECTED.jsonl MANIFEST.json ..."
  )
const rows = fs.readFileSync(input, "utf8").trim().split("\n").map(JSON.parse)
const ids = new Set()
for (const file of manifests) {
  const manifest = JSON.parse(fs.readFileSync(file, "utf8"))
  for (const run of Object.values(manifest.runs))
    for (const id of Array.isArray(run) ? run : [run]) ids.add(id)
  ids.add(manifest.qualityRun)
  for (const control of manifest.controls ?? [])
    for (const id of Array.isArray(control.ids) ? control.ids : [control.ids])
      ids.add(id)
  for (const id of manifest.evidenceRunIds ?? []) ids.add(id)
  for (const [kind, name] of Object.entries(manifest.files)) {
    const capture = JSON.parse(
      fs.readFileSync(path.join(path.dirname(file), name), "utf8")
    )
    if (kind.startsWith("load"))
      for (const result of capture.results) ids.add(result.id)
    else ids.add(new URL(capture.url).searchParams.get("id"))
  }
}
const selected = new Set()
for (const id of ids) {
  const indices = rows.flatMap((row, i) => (row.id === id ? [i] : []))
  const start = indices.findLast((i) => rows[i].phase === "environment")
  if (
    start === undefined ||
    !indices.some((i) => i >= start && rows[i].phase === "complete")
  )
    throw Error(`Incomplete capture: ${id}`)
  for (const i of indices) if (i >= start) selected.add(i)
}
fs.writeFileSync(
  output,
  rows
    .filter((_, i) => selected.has(i))
    .map(JSON.stringify)
    .join("\n") + "\n"
)
console.log(
  `Preserved ${selected.size} records from ${ids.size} selected captures.`
)
