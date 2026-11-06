/** Summarize selected device measurements; no inference runs on the host. */
import { readFileSync, writeFileSync } from "node:fs"
const input = process.argv[2]
if (!input)
  throw new Error(
    "Usage: node bench/summarize-bench.mjs RAW.jsonl SUMMARY.json MANIFEST.json"
  )
const rows = readFileSync(input, "utf8").trim().split("\n").map(JSON.parse)
const median = (values) => {
  const a = values.toSorted((x, y) => x - y),
    m = Math.floor(a.length / 2)
  return a.length % 2 ? a[m] : (a[m - 1] + a[m]) / 2
}
const read = (name) =>
  JSON.parse(
    readFileSync(new URL(`./results/${name}`, import.meta.url), "utf8")
  )
// Navigation sends a final hidden event after ALL DONE. Only visibility during
// the measured capture invalidates it; use the last environment block for an id.
function selectedRows(ids) {
  return (Array.isArray(ids) ? ids : [ids]).flatMap((id) => {
    const all = rows.filter((r) => r.id === id)
    const start = all.findLastIndex((r) => r.phase === "environment")
    const capture = all.slice(Math.max(0, start))
    const end = capture.findIndex(
      (r) => r.phase === "complete" || r.phase === "error"
    )
    return end < 0 ? capture : capture.slice(0, end + 1)
  })
}
function summarize(id) {
  const runIds = Array.isArray(id) ? id : [id]
  const chosen = selectedRows(runIds),
    decode = chosen.filter((r) => r.phase === "decode"),
    prefill = chosen.filter((r) => r.phase === "prefill" && r.durationMs > 100),
    load = chosen.find((r) => r.phase === "load")
  if (!chosen.some((r) => r.phase === "complete") || !decode.length)
    throw new Error(`Incomplete performance capture: ${id}`)
  if (
    chosen.some((r) => r.phase === "visibility" && r.state === "hidden") ||
    [...decode, ...prefill, load].some((r) => r?.foregroundValid === false)
  )
    throw new Error(`Backgrounded performance capture: ${id}`)
  return {
    id,
    rounds: decode.length,
    inputTokens: decode.map((r) => r.stats.inputTokens),
    cachedTokens: decode.map((r) => r.stats.cachedTokens),
    promptTps: median(decode.map((r) => r.promptTps)),
    aggregatePromptTps:
      (1000 *
        decode.reduce(
          (n, r) => n + r.stats.inputTokens - r.stats.cachedTokens,
          0
        )) /
      decode.reduce((n, r) => n + r.stats.promptMs, 0),
    aggregateDecodeTps:
      (1000 * decode.reduce((n, r) => n + r.stats.outputTokens - 1, 0)) /
      decode.reduce((n, r) => n + r.stats.decodeMs, 0),
    decodeTps: median(decode.map((r) => r.decodeTps)),
    promptTpsRange: decode.map((r) => r.promptTps),
    decodeTpsRange: decode.map((r) => r.decodeTps),
    prefillFps: median(prefill.map((r) => r.fps)),
    decodeFps: median(decode.map((r) => r.fps)),
    worstPrefillFrameMs: Math.max(...prefill.map((r) => r.max)),
    worstDecodeFrameMs: Math.max(...decode.map((r) => r.max)),
    framesOver100: [...prefill, ...decode].reduce((n, r) => n + r.over100, 0),
    loadMs: load?.durationMs,
    engineCode: load?.timings.engineCode,
    loadFps: load?.fps,
    loadWorstFrameMs: load?.max,
    gpuBytes: load?.gpuBytes,
    idleMs: chosen.find((r) => r.phase === "environment")?.query.idleMs,
    restMs: chosen.find((r) => r.phase === "environment")?.query.restMs,
    firstTokenMs: median(decode.map((r) => r.stats.firstTokenMs)),
    compiledDuringLoad: load?.timings.programs,
    kernelChoices: load?.timings
  }
}
function memory(file) {
  const x = read(file)
  if (x.kind === "ua-page") {
    if (!x.complete || !Number.isFinite(x.retainedPageBytes) || !Number.isFinite(x.peakGpuBytes))
      throw Error(`Incomplete user-agent memory capture: ${file}`)
    return { file, ...x }
  }
  if (typeof x.tracking === "boolean") {
    if (
      !x.complete ||
      !x.tracking ||
      !x.samples.length ||
      x.errors.length ||
      !x.gcCollected ||
      !x.workerTargets ||
      x.workerErrors?.length
    )
      throw new Error(`Incomplete WebKit memory capture: ${file}`)
    const id = new URL(x.url).searchParams.get("id")
    const measured = selectedRows(id)
    if (measured.some((r) => r.phase === "visibility" && r.state === "hidden"))
      throw new Error(`Backgrounded memory capture: ${file}`)
    const gpu = measured.filter((r) => r.phase === "memory")
    const done = gpu.filter((r) => r.when === "done").at(-1) ?? gpu.at(-1)
    if (!done) throw new Error(`Missing GPU memory capture: ${file}`)
    const total = (sample) =>
      sample.categories.reduce((sum, category) => sum + category.size, 0)
    const retained = x.samples.filter((sample) => sample.phase === "retained")
    if (!retained.length)
      throw new Error(`Missing retained WebKit memory capture: ${file}`)
    const idle = x.samples.filter((sample) => sample.phase === "idle")
    const unloaded = x.samples.filter((sample) => sample.phase === "unloaded")
    const baseline = idle.length ? median(idle.map(total)) : null
    return {
      file,
      kind: "webkit",
      idlePageBytes: baseline,
      incrementalPeakPageBytes:
        baseline == null ? null : Math.max(...x.samples.map(total)) - baseline,
      incrementalRetainedPageBytes:
        baseline == null ? null : median(retained.map(total)) - baseline,
      unloadedPageBytes: unloaded.length ? median(unloaded.map(total)) : null,
      retainedOverUnloadedBytes: unloaded.length
        ? median(retained.map(total)) - median(unloaded.map(total))
        : null,
      complete: true,
      samples: x.samples.length,
      gpuBytes: x.workerGpu?.gpuBytes ?? done.gpuBytes,
      peakGpuBytes:
        x.workerGpu?.peakGpuBytes ??
        Math.max(...gpu.map((r) => r.peakGpuBytes)),
      workerGcTargets: x.workerTargets ?? 0,
      gcCollected: x.gcCollected ?? false,
      memoryPressure: x.memoryPressure ?? [],
      peakPageBytes: Math.max(...x.samples.map(total)),
      retainedPageBytes: median(retained.map(total))
    }
  }
  if (!x.complete || !x.gcCollected || !x.gpu || !x.retainedAfterGc)
    throw new Error(`Incomplete memory capture: ${file}`)
  const total = (s) =>
    s.usedSize + (s.backingStorageSize ?? 0) + (s.embedderHeapUsedSize ?? 0)
  return {
    file,
    kind: "v8-worker",
    complete: x.complete,
    samples: x.samples.length,
    gpuBytes: x.gpu.gpuBytes,
    peakGpuBytes: x.gpu.peakGpuBytes,
    peakWorkerBytes: Math.max(...x.samples.map(total)),
    peakJsHeapCapacityBytes: Math.max(...x.samples.map((s) => s.totalSize)),
    retainedWorkerBytes: total(x.retainedAfterGc),
    retainedTrackedTotalBytes: x.gpu.gpuBytes + total(x.retainedAfterGc)
  }
}
function loads(file) {
  const x = read(file)
  if (!x.complete) throw new Error(`Incomplete load suite: ${file}`)
  const select = (cold) => {
    const a = x.results.filter((r) => r.cold === cold)
    if (a.some((r) => !r.foregroundValid))
      throw new Error(`Backgrounded load suite: ${file}`)
    return {
      ids: a.map((r) => r.id),
      trials: a.length,
      medianMs: median(a.map((r) => r.loadMs)),
      minMs: Math.min(...a.map((r) => r.loadMs)),
      maxMs: Math.max(...a.map((r) => r.loadMs)),
      fps: median(a.map((r) => r.fps)),
      worstFrameMs: Math.max(...a.map((r) => r.maxFrameMs)),
      allUnder10s: a.every((r) => r.loadMs < 10000),
      fullyCalibratedMedianMs: a.every(
        (r) => r.fullyCalibratedMs != null || x.mode === "production" || !r.cold
      )
        ? median(a.map((r) => r.fullyCalibratedMs ?? r.loadMs))
        : null,
      fullyCalibratedRangeMs: a.every(
        (r) => r.fullyCalibratedMs != null || x.mode === "production" || !r.cold
      )
        ? a.map((r) => r.fullyCalibratedMs ?? r.loadMs)
        : null,
      headExpanded: a.map((r) => r.timings.headExpanded),
      engineCodes: [...new Set(a.map((r) => r.timings.engineCode))]
    }
  }
  return { file, cold: select(true), cached: select(false) }
}
const manifestPath = process.argv[4]
if (!manifestPath) throw new Error("A device result manifest is required")
const manifest = JSON.parse(readFileSync(manifestPath, "utf8"))
const ids = manifest.runs
const runs = Object.fromEntries(
  Object.entries(ids).map(([key, id]) => [key, summarize(id)])
)
function compare(before, after) {
  const select = (key) =>
    selectedRows(ids[key]).filter((r) => r.phase === "decode")
  const b = select(before),
    a = select(after)
  return {
    promptGainPct: 100 * (runs[after].promptTps / runs[before].promptTps - 1),
    decodeGainPct: 100 * (runs[after].decodeTps / runs[before].decodeTps - 1),
    matchedReplies: a.map(
      (r) => b.find((s) => s.round === r.round)?.reply === r.reply
    )
  }
}
for (const id of Object.values(manifest.runs)) {
  for (const env of selectedRows(id).filter(r => r.phase === "environment"))
    if (["tune", "inferenceTune", "profile", "notune"].some(key => env.query[key]))
      throw Error(`Diagnostic override in primary comparison: ${env.id}`)
}
const expectedCode = manifest.metadata.optimizedEngineCode
if (expectedCode != null) {
  for (const capture of read(manifest.files.loadAfter).results)
    if (capture.timings.engineCode !== expectedCode)
      throw Error(`Stale optimized load capture: ${capture.id}`)
  for (const name of ["afterShort", "afterLong", "sampled"])
    if (runs[name]?.engineCode !== expectedCode)
      throw Error(`Stale optimized capture: ${name}`)
  for (const file of [
    manifest.files.memoryAfterShort,
    manifest.files.memoryAfterLong
  ]) {
    const id = new URL(read(file).url).searchParams.get("id")
    if (
      selectedRows(id).find((r) => r.phase === "load")?.timings.engineCode !==
      expectedCode
    )
      throw Error(`Stale optimized memory capture: ${file}`)
  }
}
const summary = {
  ...manifest.metadata,
  runs,
  short: compare("beforeShort", "afterShort"),
  long: compare("beforeLong", "afterLong"),
  load: {
    before: loads(manifest.files.loadBefore),
    after: loads(manifest.files.loadAfter)
  },
  memory: {
    before: memory(manifest.files.memoryBeforeShort),
    after: memory(manifest.files.memoryAfterShort),
    beforeLong: memory(manifest.files.memoryBeforeLong),
    afterLong: memory(manifest.files.memoryAfterLong)
  },
  quality: selectedRows(manifest.qualityRun).filter(
    (r) => r.phase === "profile" && r.stage === "quality"
  ),
  controls: (manifest.controls ?? []).map((control) => ({
    ...control,
    metrics: summarize(control.ids)
  })),
  interrupted: manifest.interrupted ?? [],
  methods: {
    fps: "requestAnimationFrame with a repainted canvas and CSS spinner; not a system-wide presentation trace",
    load: "Cached GGUF files. Cold means derived calibration notes reset, not a network download or an empty OS/driver cache.",
    memory:
      manifest.metadata.memoryKind === "webkit"
        ? "WebKit resource categories share the WebContent process and main VM; adjacent captures include pre-load and post-unload baselines, plus page and worker GC. GPU texture counters remain separate. Excludes whole-phone RSS and is not a worker heap counter."
        : manifest.metadata.memoryKind === "ua-page"
          ? "measureUserAgentSpecificMemory after GC includes this page and its dedicated worker. Retained CPU measurement only; natural CPU peak not captured. GPU texture peak separate, excluding driver overhead and browser RSS."
          : "Source-map-free workers; texture allocation bytes and worker V8 used + backing storage + embedder GC heap. Peaks are simultaneous sums within a worker sample. Excludes driver allocation overhead, main-page and whole Chrome/phone RSS."
  }
}
const json = JSON.stringify(summary, null, 2) + "\n"
if (process.argv[3]) writeFileSync(process.argv[3], json)
else process.stdout.write(json)
