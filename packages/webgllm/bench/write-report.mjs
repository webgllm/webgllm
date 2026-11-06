/** Render the selected on-device measurements; never runs inference on the host. */
import fs from "node:fs"
import path from "node:path"
const [output, ...files] = process.argv.slice(2)
if (!output || !files.length)
  throw Error("Usage: node bench/write-report.mjs REPORT.md SUMMARY.json ...")
const summaries = files.map((file) => ({
  file,
  data: JSON.parse(fs.readFileSync(file, "utf8"))
}))
const n = (value, d = 1) =>
  Number.isFinite(value) ? value.toFixed(d) : "not captured"
const mib = (value) =>
  Number.isFinite(value) ? n(value / 2 ** 20, 1) : "not captured"
const range = (a, d = 1) => `${n(Math.min(...a), d)}–${n(Math.max(...a), d)}`
const interval = (x) =>
  `${n(x.medianMs / 1000, 2)} (${n(x.minMs / 1000, 2)}–${n(x.maxMs / 1000, 2)})`
let report = `# WebGL performance evaluation\n\nBranch: \`perf/webgl-performance\`, based on untouched \`2c10f9e\`.\nModel measurements ran on the connected phones and, after the phone checks, Chrome on the M3 Ultra.\nThe Pixel used one existing Chrome tab. The iPhone used exactly one new Safari tab,\nreused for every test; the automation did not close any existing tabs.\nThe Mac used one new benchmark tab in the user's existing Chrome window.\n\nThe workload uses official \`LFM3-1.7B-QAD-Q4_0.gguf\`, batch 512, thinking off,\nspeculative decoding and pipelining enabled, 64 output tokens, greedy sampling and\nrepetition penalty 1.1. Prompt reuse is recorded in the summaries. Both workers are\nbundled without source maps. The original completed its tuner before throughput\nand cached-calibration memory comparisons.\n\nOverall throughput is total processed tokens divided by total measured model time.\nMedians and every trial are also shown. Short phase windows can estimate slightly\nabove the physical refresh rate because a boundary-crossing callback is counted;\nthe frame-gap rows provide the complementary timing evidence. The phone throughput runs pause for one minute before loading and between prompts;\nMac runs use 10 seconds before loading and 5 seconds between prompts. Pauses are excluded\nfrom model timing. These controls do not establish an identical measured temperature.\nContinuous stress results and earlier candidates remain in the local diagnostic log.\n\n`
for (const { file, data: s } of summaries) {
  const runs = [
    s.runs.beforeShort,
    s.runs.afterShort,
    s.runs.beforeLong,
    s.runs.afterLong
  ]
  const row = (label, key, d = 1) =>
    `| ${label} | ${runs.map((r) => n(r[key], d)).join(" | ")} |\n`
  report += `## ${s.device}\n\n${s.os}; ${s.browser}; ${s.renderer}.\n\n`
  if (s.hardware) report += `${s.hardware}.\n\n`
  if (s.optimizedCommit) report += `Optimized checkpoint: \`${s.optimizedCommit}\`, engine fingerprint \`${s.optimizedEngineCode}\`.\n\n`
  if (s.versionLimitation) report += `**Version coverage:** ${s.versionLimitation}\n\n`
  if (s.pageRenderingPreference)
    report += `Safari's “Prefer Page Rendering Updates near 60fps” preference was disabled by the user,\nthen Safari was reopened without closing tabs. A 120 Hz idle cadence was verified.\nThe engine follows observed rendering cadence and does not impose a 60 FPS ceiling.\nThis browser setting, rather than an engine change, enabled the 120 Hz test.\n[WebKit describes the preference](https://bugs.webkit.org/show_bug.cgi?id=290671#c4).\n\n`
  report += `| Metric | Before, short | After, short | Before, long | After, long |\n| --- | ---: | ---: | ---: | ---: |\n`
  report += row("Input tokens", "inputTokens") // replaced below for array-valued tokens
  report = report.replace(
    /\| Input tokens \|[^\n]*\n$/,
    `| Input tokens | ${runs.map((r) => [...new Set(r.inputTokens)].join(", ")).join(" | ")} |\n`
  )
  report +=
    row("Trials", "rounds", 0) +
    row("Overall prompt tokens/s", "aggregatePromptTps", 2) +
    row("Median prompt tokens/s", "promptTps", 2) +
    row("Overall generated tokens/s", "aggregateDecodeTps", 2) +
    row("Median generated tokens/s", "decodeTps", 2) +
    row("Prefill animation FPS (rAF)", "prefillFps", 2) +
    row("Generation animation FPS (rAF)", "decodeFps", 2) +
    row("Worst prefill frame gap, ms", "worstPrefillFrameMs", 2) +
    row("Worst generation frame gap, ms", "worstDecodeFrameMs", 2) +
    row("Frames over 100 ms during inference", "framesOver100", 0)
  report += `\n| Throughput range | Before, short | After, short | Before, long | After, long |\n| --- | ---: | ---: | ---: | ---: |\n| Prompt tokens/s | ${runs.map((r) => range(r.promptTpsRange, 2)).join(" | ")} |\n| Generated tokens/s | ${runs.map((r) => range(r.decodeTpsRange, 2)).join(" | ")} |\n\n`
  const b = s.load.before,
    a = s.load.after
  report += `Loading is cached GGUF files to ready, including worker startup and a painted frame\nafter ready. Fresh calibration removes derived GPU notes; it does not empty the OS\nor shader-driver cache. Values are median seconds, with the full observed range. Trial counts are shown.\n\n| Startup metric | Before | After |\n| --- | ---: | ---: |\n| Fresh / saved calibration trials | ${b.cold.trials} / ${b.cached.trials} | ${a.cold.trials} / ${a.cached.trials} |\n| Fresh calibration: first ready, s | ${interval(b.cold)} | ${interval(a.cold)} |\n| Saved calibration: ready, s | ${interval(b.cached)} | ${interval(a.cached)} |\n| Fresh calibration: tuner completion, s | ${n(b.cold.fullyCalibratedMedianMs == null ? NaN : b.cold.fullyCalibratedMedianMs / 1000, 2)} | ${n(a.cold.fullyCalibratedMedianMs / 1000, 2)} |\n| Fresh-calibration animation FPS (rAF) | ${n(b.cold.fps, 2)} | ${n(a.cold.fps, 2)} |\n| Saved-calibration animation FPS (rAF) | ${n(b.cached.fps, 2)} | ${n(a.cached.fps, 2)} |\n| Worst fresh-calibration frame gap, ms | ${n(b.cold.worstFrameMs, 2)} | ${n(a.cold.worstFrameMs, 2)} |\n| Worst saved-calibration frame gap, ms | ${n(b.cached.worstFrameMs, 2)} | ${n(a.cached.worstFrameMs, 2)} |\n\nThe original signals ready before its background tuner completes; the updated\nworker resolves its selections before ready. The two startup milestones are kept\nseparate. The sub-10-second target concerns cached-file startup, not transfer of\nthe roughly 1.05 GB model over the network.\n\n`
  if (s.memoryKind === "webkit") report += `On the tested iPhone checkpoint, fresh-calibration first ready increased from ${n(b.cold.medianMs / 1000, 2)} to ${n(a.cold.medianMs / 1000, 2)} seconds. Full calibration finished sooner (${n(b.cold.fullyCalibratedMedianMs / 1000, 2)} to ${n(a.cold.fullyCalibratedMedianMs / 1000, 2)} seconds), and saved-calibration loading changed from ${n(b.cached.medianMs / 1000, 2)} to ${n(a.cached.medianMs / 1000, 2)} seconds. The earlier ready milestone is therefore a tradeoff, not an across-the-board loading improvement.\n\n`
  if (s.device.startsWith("Pixel")) {
    const generated = [...s.runs.afterShort.decodeTpsRange, ...s.runs.afterLong.decodeTpsRange]
    const prompt = [...s.runs.afterShort.promptTpsRange, ...s.runs.afterLong.promptTpsRange]
    report += `Measured Pixel targets: generation stayed at or above ${n(Math.min(...generated), 2)} tokens/s, and prefill at or above ${n(Math.min(...prompt), 2)} tokens/s. Overall prefill changed by ${n(100 * (s.runs.afterShort.aggregatePromptTps / s.runs.beforeShort.aggregatePromptTps - 1), 2)}% short and ${n(100 * (s.runs.afterLong.aggregatePromptTps / s.runs.beforeLong.aggregatePromptTps - 1), 2)}% long. All ${a.cold.trials + a.cached.trials} final cached-file loads were below 10 seconds. Cold startup still showed a ${n(a.cold.worstFrameMs, 2)} ms worst frame gap; the inference and saved-calibration rows quantify the smoother steady behavior.\n\n`
  }
  const m = [
    s.memory.before,
    s.memory.after,
    s.memory.beforeLong,
    s.memory.afterLong
  ]
  const mrow = (label, key) =>
    `| ${label} | ${m.map((v) => mib(v[key])).join(" | ")} |\n`
  report += `Memory was profiled separately from throughput, with completed calibration and\none prompt per worker. MiB means 2^20 bytes.\n\n| Memory, MiB | Before, short | After, short | Before, long | After, long |\n| --- | ---: | ---: | ---: | ---: |\n`
  report +=
    mrow("Peak allocated GPU textures", "peakGpuBytes") +
    mrow("GPU textures after generation", "gpuBytes")
  if (m[0].kind === "webkit")
    report +=
      mrow("Peak WebKit resource categories", "peakPageBytes") +
      mrow(
        "Retained WebKit resource categories after GC",
        "retainedPageBytes"
      ) +
      mrow(
        "Resource categories after model unload and GC",
        "unloadedPageBytes"
      ) +
      mrow(
        "Loaded minus post-unload retained categories",
        "retainedOverUnloadedBytes"
      )
  else if (m[0].kind === "ua-page")
    report += mrow(
      "Retained page + dedicated worker after GC",
      "retainedPageBytes"
    )
  else
    report +=
      mrow("Peak worker memory", "peakWorkerBytes") +
      mrow("Retained worker memory after GC", "retainedWorkerBytes") +
      mrow("Retained textures + worker memory", "retainedTrackedTotalBytes")
  report += `\nGPU values count allocated textures, excluding driver overhead. `
  if (m[0].kind === "webkit")
    report += `Safari values use\nthe [WebKit Memory protocol](https://raw.githubusercontent.com/WebKit/WebKit/main/Source/JavaScriptCore/inspector/protocol/Memory.json),\nsumming categories within each sample before taking its peak. The\n[WebKit collector](https://raw.githubusercontent.com/WebKit/WebKit/main/Source/WebCore/page/cocoa/ResourceUsageThreadCocoa.mm)\nuses shared WebContent-process allocation categories and the main VM; it is not a\ndedicated-worker heap counter. Adjacent original/optimized captures include a\npost-unload baseline so accumulated browser state is visible. Collection is requested\nin the page and available worker inspector targets. The page counter and GPU allocation\ncounter are shown separately; their independent peaks are not added together.\n`
  else if (m[0].kind === "ua-page")
    report += `The browser\n[measureUserAgentSpecificMemory API](https://web.dev/articles/monitor-total-page-memory-usage)\nincludes this page and its dedicated worker and measures after garbage collection.\nThese are retained measurements; natural CPU peaks were not captured in this existing\nChrome profile. The GPU texture peak is recorded independently.\n`
  else
    report += `Worker values\nuse CDP Runtime.getHeapUsage usedSize + backingStorageSize + embedderHeapUsedSize,\nsummed within each sample before taking its peak. Retained totals are measured\nafter collection while the finished model remains loaded.\n`
  report += `These counters are not total phone RAM or whole-browser RSS. Memory scopes differ\nbetween Chrome and Safari, so compare before/after within one device.\n\n`
  if (m[0].kind === "webkit") report += `The WebKit resource peaks do not establish a CPU-memory reduction: short-prompt peak categories increased, while the long-prompt peaks were close. Loaded-minus-post-unload retained categories rose by about 2–3 MiB; allocated GPU textures fell by about 15 MiB. These separately sampled counters do not establish a combined RSS saving.\n\n`
  if (s.runs.sampled)
    report += `The additional sampled run (temperature 0.1, top-k 50) measured\n${n(s.runs.sampled.aggregatePromptTps, 2)} prompt tokens/s and ${n(s.runs.sampled.aggregateDecodeTps, 2)}\ngenerated tokens/s, with ${n(s.runs.sampled.prefillFps, 2)}/${n(s.runs.sampled.decodeFps, 2)} prefill/generation FPS.\n\n`
  report += `Greedy replies matched by trial: short ${JSON.stringify(s.short.matchedReplies)},\nlong ${JSON.stringify(s.long.matchedReplies)}.\n\n| Numerical check | Maximum absolute difference | Relative RMS | KL divergence | Same top token |\n| --- | ---: | ---: | ---: | --- |\n`
  for (const q of s.quality) {
    if (q.kind === "upload-byte-exact")
      report += `| Upload byte checks (${q.bytesChecked} bytes, ${q.mismatches} mismatches) | 0 | 0 | — | — |\n`
    else
      report += `| ${q.kind} | ${q.maxAbs?.toExponential(3) ?? "—"} | ${q.relativeRms?.toExponential(3) ?? "—"} | ${q.kl?.toExponential(3) ?? "—"} | ${q.top1} |\n`
  }
  const replies = [...s.short.matchedReplies, ...s.long.matchedReplies]
  if (replies.some(matched => !matched)) report += `${replies.filter(Boolean).length}/${replies.length} complete greedy replies matched the original. One iPhone short-trial reply differed; its cause was not isolated. The numerical rows above compare specific alternate paths and are not a claim that every original/optimized reply is bit-identical.\n\n`
  if (s.device.startsWith("Mac")) report += `Measured overall prefill changed by ${n(100 * (s.runs.afterShort.aggregatePromptTps / s.runs.beforeShort.aggregatePromptTps - 1), 2)}% short and ${n(100 * (s.runs.afterLong.aggregatePromptTps / s.runs.beforeLong.aggregatePromptTps - 1), 2)}% long. Individual trial ranges overlap; these measurements support roughly flat prefill, not a proof of exactly zero overhead.\n\n`
  if (s.quality.some(q => q.kind === "streamed-vs-original-full-model")) report += `The streamed-kernel numerical row is an experimental path kept off by default; it is not the primary generation path.\n\n`
  if (s.presentationMotionFile) {
    const motion = JSON.parse(fs.readFileSync(path.join(path.dirname(file), s.presentationMotionFile), "utf8"))
    const before = motion.results.find(r => r.mode === "before"), after = motion.results.find(r => r.mode === "after")
    report += `Additional visible-motion validation used equal Android screenrecord settings in this same tab, separately from primary throughput. The encoder/compositor adds work. A green-ball centroid changing by 0.25 px counts as motion; 350 ms is excluded at phase boundaries. Near a natural turning point the ball can remain within that threshold for about 50 ms even at idle. These are motion events, not an exact display FPS counter.\n\n| Recorded window | Before motion events/s | After motion events/s | Before longest position hold, ms | After longest position hold, ms |\n| --- | ---: | ---: | ---: | ---: |\n`
    for (const phase of ["idle", "prefill", "decode"]) {
      const b = before.phases[phase], a = after.phases[phase]
      report += `| ${phase} | ${n(b.motionEventsPerSecond, 2)} | ${n(a.motionEventsPerSecond, 2)} | ${n(b.longestPositionHoldMs, 2)} | ${n(a.longestPositionHoldMs, 2)} |\n`
    }
    report += `\nCanvas-only prefill clips: [before](results/presentation-before.mp4), [after](results/presentation-after.mp4). [Motion method and counts](results/presentation-motion-summary.json) and [per-frame centroids](results/presentation-motion-frames.jsonl) preserve the diagnostic evidence.\n\n`
  }
  if (s.controls?.length) {
    report += `\nDiagnostic controls are kept separate from the final build.\n\n| Earlier control | Prompt tokens/s range | Generated tokens/s range | Interpretation |\n| --- | ---: | ---: | --- |\n`
    for (const control of s.controls)
      report += `| ${control.name} | ${range(control.metrics.promptTpsRange, 2)} | ${range(control.metrics.decodeTpsRange, 2)} | ${control.interpretation} |\n`
  }
  report += `\n[Machine-readable metrics and selected run IDs](results/${path.basename(file)}).\n\n`
}
report += `## Engine changes and validation\n\nThe engine uses measured frame cadence, fence latency, workload shape and memory\nbudgets. There are no GPU-vendor, phone-name or user-agent pacing gates. Pacing is\nautomatically active through the normal worker path. Fixed tuning values belong\nto isolated benchmark experiments, with no new product or developer setting.\n\n- Bounded, tile-aligned GPU submissions preserve the full prompt batch. Queue drains\n  adapt to real rendering deadlines, and startup shader/tuner history is discarded\n  before user inference. Larger overlap is tested only after full-batch shader warm-up and requires\n  compositor progress. Confirmed failures keep the conservative quota; isolated\n  callback jitter does not permanently revoke an otherwise smooth queue. Isolated missed frames preserve efficient matrix\n  bands; repeated stalls can reduce them, and healthy work restores queue overlap.\n  Fast complete pipelines retain their tested presentation window. Head work charges\n  stay constant as the quota grows, and small projections avoid crossing chunk boundaries.\n  Known prompt capacity is reserved once, avoiding repeated cache allocation/copies\n  between prefill batches without increasing the final cache capacity.\n- Program, uniform, texture, sampler and framebuffer state caching removes redundant\n  WebGL calls, with invalidation after uploads/deletes. Fence checks use zero-timeout\n  clientWaitSync, and short fences avoid unnecessary timer sleeps.\n- Direct FP16 norm/activation outputs remove intermediate packing passes. Real\n  dependent prompt pipelines select matrix layouts using reversed timing pairs.\n  Native-versus-dequantized comparisons retain the real model operator mixture.\n  RMS loops compile with fixed bounds and keep the original 64-part reduction;\n  extra RMS calibration was removed because its gain was inconsistent.\n- A losslessly expanded Q6_K byte head is permitted only when other GPU scratch\n  savings fund its complete allocation. Its upload reuses existing integer staging.\n  Small residual/MRT buffers and smaller score working sets fund the faster layout.\n- Exact word copies, whole-word nibble packing, bounded contiguous file reads and\n  direct tokenizer vocabulary sizing reduce startup work. Exclusively owned GGUF\n  staging arrays and upload temporaries are released after WebGL consumes them.\n  Dependent storage probes consume every output group; a layout ruled out by decode\n  skips its redundant full-batch prefill probe.\n  Platforms without ArrayBuffer.transfer retain conservative loading buffers.\n- Startup correctness checks cover real inputs on every distinct attention/cache\n  variant and both model ends, against CPU attention and the alternate GPU paths.\n  Recurrent models keep their full sequence/state checks. No correctness check is\n  replaced by a successful speed result.\n\nBuild, typecheck, test and lint results are recorded in\n[validation.txt](results/validation.txt). Lint warnings are reported separately\nfrom errors. Host work included compilation, mathematical tests, serving and device\ncontrol, followed by the user-authorized M3 Ultra Chrome model benchmark.\n\nRejected experiments included smaller prompt batches, draining every small band,\nextra activation buffers, unrolled/transposed float kernels, mixed-precision dots,\nsynchronous FileReaderSync loading and large fixed submission budgets. Automatic\nbyte-activation streaming was also rejected after Mac full-model timing disagreed\nwith the isolated matvec probe. A storage-probe candidate picked native RGBA4 on\nPixel and caused a 20-second cold load; the final guarded selection was checked\nin repeated fresh-calibration loads. The earlier Mac native-layout/controller\ncombination also regressed and is retained as a rejected control. A broad\neight-layer scratch-layout search took 26.25 seconds on iPhone and was rejected;\nthe retained packed comparison avoids those alternate scratch allocations. Intermediate\nresults are not combined with the final build. A user-confirmed tab-closing\ninterruption produced a roughly 30-second pause and is excluded from controlled\nmetrics. Earlier 60/80 Hz iPhone measurements are superseded by the 120 Hz comparison.\n\nThe FPS metric is requestAnimationFrame while repainting the canvas and animating\nthe CSS spinner, not a system-wide presentation trace. Device heat and OS scheduling\ncan change throughput; sustained stress measurements must not be compared with a\nrested run as though only engine code changed. Older iPhones and macOS Safari were not measured; the same engine follows the\nobserved display cadence, with unit coverage for both 60 and 120 FPS. A final client-side\nguard for whole-millisecond frame timestamps was added after the device timing runs\nand verified with synthetic 60/120 Hz sequences. The reported device runs used\nfractional-millisecond timestamps, whose estimator is unchanged.\n[WebKit time precision implementation](https://github.com/WebKit/WebKit/blob/main/Source/WebCore/page/Performance.cpp)\nprovides the reason to cover coarse clocks; the rounded-clock branch was not\nphysically retested on the iPhone.\n\n[Selected raw phase records](results/webgl-performance-2026-11-03.jsonl) and\n[source/build provenance](results/source-manifest.json) preserve the evidence.\nThe dedicated worktree contains the changes; publishing or pushing is not part of\nthis measurement report.\n`
fs.writeFileSync(output, report)
