# WebGL performance evaluation

Branch: `perf/webgl-performance`, based on untouched `2c10f9e`.
Model measurements ran on the connected phones and, after the phone checks, Chrome on the M3 Ultra.
The Pixel used one existing Chrome tab. The iPhone used exactly one new Safari tab,
reused for every test; the automation did not close any existing tabs.
The Mac used one new benchmark tab in the user's existing Chrome window.

The workload uses official `LFM3-1.7B-QAD-Q4_0.gguf`, batch 512, thinking off,
speculative decoding and pipelining enabled, 64 output tokens, greedy sampling and
repetition penalty 1.1. Prompt reuse is recorded in the summaries. Both workers are
bundled without source maps. The original completed its tuner before throughput
and cached-calibration memory comparisons.

Overall throughput is total processed tokens divided by total measured model time.
Medians and every trial are also shown. Short phase windows can estimate slightly
above the physical refresh rate because a boundary-crossing callback is counted;
the frame-gap rows provide the complementary timing evidence. The phone throughput runs pause for one minute before loading and between prompts;
Mac runs use 10 seconds before loading and 5 seconds between prompts. Pauses are excluded
from model timing. These controls do not establish an identical measured temperature.
Continuous stress results and earlier candidates remain in the local diagnostic log.

## Pixel 10 Pro

Android 17; Chrome 156.0.8078.73; ANGLE (Imagination Technologies, PowerVR D-Series DXT-48-1536, OpenGL ES 3.2).

Optimized checkpoint: `51b0400`, engine fingerprint `3321799229`.

| Metric | Before, short | After, short | Before, long | After, long |
| --- | ---: | ---: | ---: | ---: |
| Input tokens | 548 | 548 | 2085 | 2085 |
| Trials | 3 | 3 | 2 | 2 |
| Overall prompt tokens/s | 116.27 | 121.25 | 118.88 | 120.29 |
| Median prompt tokens/s | 116.42 | 121.49 | 118.88 | 120.29 |
| Overall generated tokens/s | 10.57 | 13.14 | 10.29 | 12.82 |
| Median generated tokens/s | 10.59 | 13.16 | 10.29 | 12.82 |
| Prefill animation FPS (rAF) | 6.14 | 60.05 | 7.15 | 59.60 |
| Generation animation FPS (rAF) | 36.53 | 60.11 | 36.59 | 60.04 |
| Worst prefill frame gap, ms | 1333.37 | 24.99 | 1000.17 | 33.38 |
| Worst generation frame gap, ms | 283.38 | 17.35 | 291.67 | 16.89 |
| Frames over 100 ms during inference | 41 | 0 | 73 | 0 |

| Throughput range | Before, short | After, short | Before, long | After, long |
| --- | ---: | ---: | ---: | ---: |
| Prompt tokens/s | 115.88–116.52 | 120.34–121.94 | 118.67–119.09 | 119.93–120.65 |
| Generated tokens/s | 10.50–10.62 | 13.01–13.24 | 10.29–10.29 | 12.69–12.95 |

Loading is cached GGUF files to ready, including worker startup and a painted frame
after ready. Fresh calibration removes derived GPU notes; it does not empty the OS
or shader-driver cache. Values are median seconds, with the full observed range. Trial counts are shown.

| Startup metric | Before | After |
| --- | ---: | ---: |
| Fresh / saved calibration trials | 3 / 3 | 6 / 3 |
| Fresh calibration: first ready, s | 9.94 (9.88–10.04) | 8.37 (8.25–8.45) |
| Saved calibration: ready, s | 6.22 (6.09–6.39) | 4.54 (4.49–4.66) |
| Fresh calibration: tuner completion, s | not captured | 8.37 |
| Fresh-calibration animation FPS (rAF) | 55.29 | 58.46 |
| Saved-calibration animation FPS (rAF) | 58.55 | 59.91 |
| Worst fresh-calibration frame gap, ms | 558.36 | 75.01 |
| Worst saved-calibration frame gap, ms | 141.66 | 25.01 |

The original signals ready before its background tuner completes; the updated
worker resolves its selections before ready. The two startup milestones are kept
separate. The sub-10-second target concerns cached-file startup, not transfer of
the roughly 1.05 GB model over the network.

Measured Pixel targets: generation stayed at or above 12.69 tokens/s, and prefill at or above 119.93 tokens/s. Overall prefill changed by 4.28% short and 1.18% long. All 9 final cached-file loads were below 10 seconds. Cold startup still showed a 75.01 ms worst frame gap; the inference and saved-calibration rows quantify the smoother steady behavior.

Memory was profiled separately from throughput, with completed calibration and
one prompt per worker. MiB means 2^20 bytes.

| Memory, MiB | Before, short | After, short | Before, long | After, long |
| --- | ---: | ---: | ---: | ---: |
| Peak allocated GPU textures | 1376.1 | 1334.9 | 1382.6 | 1362.9 |
| GPU textures after generation | 1354.7 | 1334.9 | 1382.6 | 1362.9 |
| Peak worker memory | 269.2 | 210.8 | 269.1 | 221.1 |
| Retained worker memory after GC | 48.7 | 49.2 | 48.8 | 49.2 |
| Retained textures + worker memory | 1403.4 | 1384.1 | 1431.4 | 1412.1 |

GPU values count allocated textures, excluding driver overhead. Worker values
use CDP Runtime.getHeapUsage usedSize + backingStorageSize + embedderHeapUsedSize,
summed within each sample before taking its peak. Retained totals are measured
after collection while the finished model remains loaded.
These counters are not total phone RAM or whole-browser RSS. Memory scopes differ
between Chrome and Safari, so compare before/after within one device.

The additional sampled run (temperature 0.1, top-k 50) measured
121.45 prompt tokens/s and 12.30
generated tokens/s, with 60.08/60.07 prefill/generation FPS.

Greedy replies matched by trial: short [true,true,true],
long [true,true].

| Numerical check | Maximum absolute difference | Relative RMS | KL divergence | Same top token |
| --- | ---: | ---: | ---: | --- |
| fused-prefill-inputs-vs-original-full-model | 0.000e+0 | 0.000e+0 | 0.000e+0 | true |
| prepared-vs-incremental-cache-full-model | 0.000e+0 | 0.000e+0 | 0.000e+0 | true |
| Upload byte checks (1212416 bytes, 0 mismatches) | 0 | 0 | — | — |
| expanded-unsplit-vs-split-full-vocabulary | 3.815e-6 | 1.279e-7 | 3.395e-14 | true |
| native-vs-expanded-real-hidden-state | 1.907e-6 | 1.036e-7 | 5.006e-14 | true |
| streamed-vs-original-full-model | 8.736e-3 | 6.483e-4 | 1.366e-6 | true |
The streamed-kernel numerical row is an experimental path kept off by default; it is not the primary generation path.

Additional visible-motion validation used equal Android screenrecord settings in this same tab, separately from primary throughput. The encoder/compositor adds work. A green-ball centroid changing by 0.25 px counts as motion; 350 ms is excluded at phase boundaries. Near a natural turning point the ball can remain within that threshold for about 50 ms even at idle. These are motion events, not an exact display FPS counter.

| Recorded window | Before motion events/s | After motion events/s | Before longest position hold, ms | After longest position hold, ms |
| --- | ---: | ---: | ---: | ---: |
| idle | 58.33 | 58.35 | 51.08 | 49.77 |
| prefill | 12.25 | 57.00 | 393.03 | 66.63 |
| decode | 38.53 | 57.97 | 199.66 | 67.82 |

Canvas-only prefill clips: [before](results/presentation-before.mp4), [after](results/presentation-after.mp4). [Motion method and counts](results/presentation-motion-summary.json) and [per-frame centroids](results/presentation-motion-frames.jsonl) preserve the diagnostic evidence.


Diagnostic controls are kept separate from the final build.

| Earlier control | Prompt tokens/s range | Generated tokens/s range | Interpretation |
| --- | ---: | ---: | --- |
| Timer-free polling | 113.47–121.53 | 10.77–12.36 | Rejected: the second long run fell below the 12 generated tokens/s target. |

[Machine-readable metrics and selected run IDs](results/pixel-summary.json).

## iPhone 16 Pro Max

iOS 27.2 (24B119), as reported by CoreDevice; Safari 27.2; Apple GPU.

Optimized checkpoint: `70d3ea0`, engine fingerprint `446196241`.

**Version coverage:** Physical iPhone results use checkpoint 70d3ea0 (engine 446196241). Subsequent startup storage selection and fast-workload pacing refinements were tested on Pixel and Mac after the iPhone was returned; final branch needs an iPhone recheck.

Safari's “Prefer Page Rendering Updates near 60fps” preference was disabled by the user,
then Safari was reopened without closing tabs. A 120 Hz idle cadence was verified.
The engine follows observed rendering cadence and does not impose a 60 FPS ceiling.
This browser setting, rather than an engine change, enabled the 120 Hz test.
[WebKit describes the preference](https://bugs.webkit.org/show_bug.cgi?id=290671#c4).

| Metric | Before, short | After, short | Before, long | After, long |
| --- | ---: | ---: | ---: | ---: |
| Input tokens | 548 | 548 | 2085 | 2085 |
| Trials | 3 | 3 | 2 | 2 |
| Overall prompt tokens/s | 298.81 | 300.85 | 285.13 | 285.87 |
| Median prompt tokens/s | 305.77 | 300.13 | 285.15 | 285.88 |
| Overall generated tokens/s | 26.43 | 28.76 | 26.01 | 28.13 |
| Median generated tokens/s | 26.45 | 28.68 | 26.01 | 28.13 |
| Prefill animation FPS (rAF) | 120.23 | 120.46 | 120.06 | 120.03 |
| Generation animation FPS (rAF) | 120.21 | 120.16 | 120.18 | 120.25 |
| Worst prefill frame gap, ms | 9.42 | 9.44 | 12.54 | 16.86 |
| Worst generation frame gap, ms | 9.08 | 8.92 | 8.86 | 8.88 |
| Frames over 100 ms during inference | 0 | 0 | 0 | 0 |

| Throughput range | Before, short | After, short | Before, long | After, long |
| --- | ---: | ---: | ---: | ---: |
| Prompt tokens/s | 285.66–305.93 | 300.00–302.42 | 282.72–287.58 | 284.16–287.60 |
| Generated tokens/s | 26.15–26.70 | 28.66–28.96 | 25.95–26.08 | 28.08–28.18 |

Loading is cached GGUF files to ready, including worker startup and a painted frame
after ready. Fresh calibration removes derived GPU notes; it does not empty the OS
or shader-driver cache. Values are median seconds, with the full observed range. Trial counts are shown.

| Startup metric | Before | After |
| --- | ---: | ---: |
| Fresh / saved calibration trials | 3 / 3 | 3 / 3 |
| Fresh calibration: first ready, s | 3.54 (3.51–3.68) | 6.78 (6.76–6.82) |
| Saved calibration: ready, s | 2.23 (2.22–2.41) | 2.14 (2.13–2.28) |
| Fresh calibration: tuner completion, s | 12.63 | 6.78 |
| Fresh-calibration animation FPS (rAF) | 119.87 | 120.04 |
| Saved-calibration animation FPS (rAF) | 120.33 | 120.38 |
| Worst fresh-calibration frame gap, ms | 19.02 | 19.78 |
| Worst saved-calibration frame gap, ms | 9.68 | 10.22 |

The original signals ready before its background tuner completes; the updated
worker resolves its selections before ready. The two startup milestones are kept
separate. The sub-10-second target concerns cached-file startup, not transfer of
the roughly 1.05 GB model over the network.

On the tested iPhone checkpoint, fresh-calibration first ready increased from 3.54 to 6.78 seconds. Full calibration finished sooner (12.63 to 6.78 seconds), and saved-calibration loading changed from 2.23 to 2.14 seconds. The earlier ready milestone is therefore a tradeoff, not an across-the-board loading improvement.

Memory was profiled separately from throughput, with completed calibration and
one prompt per worker. MiB means 2^20 bytes.

| Memory, MiB | Before, short | After, short | Before, long | After, long |
| --- | ---: | ---: | ---: | ---: |
| Peak allocated GPU textures | 1310.2 | 1295.6 | 1338.8 | 1323.5 |
| GPU textures after generation | 1310.2 | 1295.6 | 1338.8 | 1323.5 |
| Peak WebKit resource categories | 521.5 | 565.2 | 566.3 | 568.9 |
| Retained WebKit resource categories after GC | 396.8 | 401.2 | 402.7 | 408.3 |
| Resource categories after model unload and GC | 299.6 | 301.4 | 300.2 | 303.5 |
| Loaded minus post-unload retained categories | 97.2 | 99.8 | 102.6 | 104.8 |

GPU values count allocated textures, excluding driver overhead. Safari values use
the [WebKit Memory protocol](https://raw.githubusercontent.com/WebKit/WebKit/main/Source/JavaScriptCore/inspector/protocol/Memory.json),
summing categories within each sample before taking its peak. The
[WebKit collector](https://raw.githubusercontent.com/WebKit/WebKit/main/Source/WebCore/page/cocoa/ResourceUsageThreadCocoa.mm)
uses shared WebContent-process allocation categories and the main VM; it is not a
dedicated-worker heap counter. Adjacent original/optimized captures include a
post-unload baseline so accumulated browser state is visible. Collection is requested
in the page and available worker inspector targets. The page counter and GPU allocation
counter are shown separately; their independent peaks are not added together.
These counters are not total phone RAM or whole-browser RSS. Memory scopes differ
between Chrome and Safari, so compare before/after within one device.

The WebKit resource peaks do not establish a CPU-memory reduction: short-prompt peak categories increased, while the long-prompt peaks were close. Loaded-minus-post-unload retained categories rose by about 2–3 MiB; allocated GPU textures fell by about 15 MiB. These separately sampled counters do not establish a combined RSS saving.

The additional sampled run (temperature 0.1, top-k 50) measured
293.80 prompt tokens/s and 28.38
generated tokens/s, with 120.45/120.28 prefill/generation FPS.

Greedy replies matched by trial: short [false,true,true],
long [true,true].

| Numerical check | Maximum absolute difference | Relative RMS | KL divergence | Same top token |
| --- | ---: | ---: | ---: | --- |
| fused-prefill-inputs-vs-original-full-model | 0.000e+0 | 0.000e+0 | 0.000e+0 | true |
| prepared-vs-incremental-cache-full-model | 0.000e+0 | 0.000e+0 | 0.000e+0 | true |
| native-vs-expanded-real-hidden-state | 1.907e-6 | 1.017e-7 | 6.162e-14 | true |
| streamed-vs-original-full-model | 0.000e+0 | 0.000e+0 | 0.000e+0 | true |
4/5 complete greedy replies matched the original. One iPhone short-trial reply differed; its cause was not isolated. The numerical rows above compare specific alternate paths and are not a claim that every original/optimized reply is bit-identical.

The streamed-kernel numerical row is an experimental path kept off by default; it is not the primary generation path.


Diagnostic controls are kept separate from the final build.

| Earlier control | Prompt tokens/s range | Generated tokens/s range | Interpretation |
| --- | ---: | ---: | --- |
| Original, paired control | 280.03–287.52 | 23.98–25.97 | Same-session reference on an earlier build; shows run-to-run variation. |
| Optimized, normal polling | 276.85–280.56 | 27.92–28.04 | Earlier 24-frame drain ceiling; superseded by the cadence-independent ceiling. |
| Optimized, timer-free polling | 278.12–280.18 | 27.86–27.96 | No consistent paired gain; periodic backoff retained. |

[Machine-readable metrics and selected run IDs](results/iphone-summary.json).

## Mac M3 Ultra

macOS 27.2; Chrome 156.0.8078.80; ANGLE (Apple, ANGLE Metal Renderer: Apple M3 Ultra, Unspecified Version).

Apple M3 Ultra, Mac15,14, 512 GiB OS-reported RAM.

Optimized checkpoint: `51b0400`, engine fingerprint `3321799229`.

| Metric | Before, short | After, short | Before, long | After, long |
| --- | ---: | ---: | ---: | ---: |
| Input tokens | 548 | 548 | 2085 | 2085 |
| Trials | 5 | 5 | 3 | 3 |
| Overall prompt tokens/s | 2837.80 | 2802.08 | 3091.65 | 3090.59 |
| Median prompt tokens/s | 2799.49 | 2764.05 | 3071.01 | 3105.27 |
| Overall generated tokens/s | 85.34 | 88.62 | 83.67 | 87.26 |
| Median generated tokens/s | 85.22 | 88.75 | 83.64 | 87.07 |
| Prefill animation FPS (rAF) | 62.26 | 62.08 | 60.72 | 60.60 |
| Generation animation FPS (rAF) | 60.21 | 60.22 | 60.18 | 60.20 |
| Worst prefill frame gap, ms | 18.58 | 18.28 | 18.65 | 18.57 |
| Worst generation frame gap, ms | 18.59 | 18.56 | 18.46 | 18.57 |
| Frames over 100 ms during inference | 0 | 0 | 0 | 0 |

| Throughput range | Before, short | After, short | Before, long | After, long |
| --- | ---: | ---: | ---: | ---: |
| Prompt tokens/s | 2723.79–3165.43 | 2711.73–3029.21 | 3047.76–3158.37 | 3055.84–3111.27 |
| Generated tokens/s | 85.08–85.75 | 87.62–89.76 | 83.16–84.20 | 86.57–88.15 |

Loading is cached GGUF files to ready, including worker startup and a painted frame
after ready. Fresh calibration removes derived GPU notes; it does not empty the OS
or shader-driver cache. Values are median seconds, with the full observed range. Trial counts are shown.

| Startup metric | Before | After |
| --- | ---: | ---: |
| Fresh / saved calibration trials | 3 / 3 | 3 / 3 |
| Fresh calibration: first ready, s | 1.86 (1.85–2.16) | 1.29 (1.29–1.33) |
| Saved calibration: ready, s | 1.43 (1.42–1.49) | 0.97 (0.94–0.97) |
| Fresh calibration: tuner completion, s | 2.68 | 1.29 |
| Fresh-calibration animation FPS (rAF) | 60.21 | 60.33 |
| Saved-calibration animation FPS (rAF) | 60.39 | 60.53 |
| Worst fresh-calibration frame gap, ms | 18.63 | 18.59 |
| Worst saved-calibration frame gap, ms | 18.63 | 18.66 |

The original signals ready before its background tuner completes; the updated
worker resolves its selections before ready. The two startup milestones are kept
separate. The sub-10-second target concerns cached-file startup, not transfer of
the roughly 1.05 GB model over the network.

Memory was profiled separately from throughput, with completed calibration and
one prompt per worker. MiB means 2^20 bytes.

| Memory, MiB | Before, short | After, short | Before, long | After, long |
| --- | ---: | ---: | ---: | ---: |
| Peak allocated GPU textures | 1338.2 | 1323.6 | 1366.8 | 1351.5 |
| GPU textures after generation | 1338.2 | 1323.6 | 1366.8 | 1351.5 |
| Retained page + dedicated worker after GC | 51.8 | 52.6 | 51.9 | 52.6 |

GPU values count allocated textures, excluding driver overhead. The browser
[measureUserAgentSpecificMemory API](https://web.dev/articles/monitor-total-page-memory-usage)
includes this page and its dedicated worker and measures after garbage collection.
These are retained measurements; natural CPU peaks were not captured in this existing
Chrome profile. The GPU texture peak is recorded independently.
These counters are not total phone RAM or whole-browser RSS. Memory scopes differ
between Chrome and Safari, so compare before/after within one device.

The additional sampled run (temperature 0.1, top-k 50) measured
3019.45 prompt tokens/s and 86.59
generated tokens/s, with 60.69/60.18 prefill/generation FPS.

Greedy replies matched by trial: short [true,true,true,true,true],
long [true,true,true].

| Numerical check | Maximum absolute difference | Relative RMS | KL divergence | Same top token |
| --- | ---: | ---: | ---: | --- |
| fused-prefill-inputs-vs-original-full-model | 0.000e+0 | 0.000e+0 | 0.000e+0 | true |
| prepared-vs-incremental-cache-full-model | 0.000e+0 | 0.000e+0 | 0.000e+0 | true |
| Upload byte checks (1212416 bytes, 0 mismatches) | 0 | 0 | — | — |
| expanded-unsplit-vs-split-full-vocabulary | 3.815e-6 | 1.278e-7 | 4.061e-14 | true |
| native-vs-expanded-real-hidden-state | 1.907e-6 | 1.034e-7 | 5.017e-14 | true |
| streamed-vs-original-full-model | 1.292e-2 | 1.205e-3 | 2.192e-6 | true |
Measured overall prefill changed by -1.26% short and -0.03% long. Individual trial ranges overlap; these measurements support roughly flat prefill, not a proof of exactly zero overhead.

The streamed-kernel numerical row is an experimental path kept off by default; it is not the primary generation path.


Diagnostic controls are kept separate from the final build.

| Earlier control | Prompt tokens/s range | Generated tokens/s range | Interpretation |
| --- | ---: | ---: | --- |
| Earlier original-engine control | 2878.98–3268.23 | 85.58–87.10 | Earlier three-trial baseline; final five-trial baseline is contemporaneous with the revised controller. |
| Rejected initial storage/controller selection | 2263.62–2531.00 | 70.47–71.28 | Native RGBA4 and over-conservative submission pacing regressed whole-model speed; rejected. |
| Submission pacing disabled (diagnostic only) | 2803.86–3300.71 | 89.30–89.88 | Restored speed and identified fence overhead. Final runs use automatic pacing with no overrides. |

[Machine-readable metrics and selected run IDs](results/mac-summary.json).

## Engine changes and validation

The engine uses measured frame cadence, fence latency, workload shape and memory
budgets. There are no GPU-vendor, phone-name or user-agent pacing gates. Pacing is
automatically active through the normal worker path. Fixed tuning values belong
to isolated benchmark experiments, with no new product or developer setting.

- Bounded, tile-aligned GPU submissions preserve the full prompt batch. Queue drains
  adapt to real rendering deadlines, and startup shader/tuner history is discarded
  before user inference. Larger overlap is tested only after full-batch shader warm-up and requires
  compositor progress. Confirmed failures keep the conservative quota; isolated
  callback jitter does not permanently revoke an otherwise smooth queue. Isolated missed frames preserve efficient matrix
  bands; repeated stalls can reduce them, and healthy work restores queue overlap.
  Fast complete pipelines retain their tested presentation window. Head work charges
  stay constant as the quota grows, and small projections avoid crossing chunk boundaries.
  Known prompt capacity is reserved once, avoiding repeated cache allocation/copies
  between prefill batches without increasing the final cache capacity.
- Program, uniform, texture, sampler and framebuffer state caching removes redundant
  WebGL calls, with invalidation after uploads/deletes. Fence checks use zero-timeout
  clientWaitSync, and short fences avoid unnecessary timer sleeps.
- Direct FP16 norm/activation outputs remove intermediate packing passes. Real
  dependent prompt pipelines select matrix layouts using reversed timing pairs.
  Native-versus-dequantized comparisons retain the real model operator mixture.
  RMS loops compile with fixed bounds and keep the original 64-part reduction;
  extra RMS calibration was removed because its gain was inconsistent.
- A losslessly expanded Q6_K byte head is permitted only when other GPU scratch
  savings fund its complete allocation. Its upload reuses existing integer staging.
  Small residual/MRT buffers and smaller score working sets fund the faster layout.
- Exact word copies, whole-word nibble packing, bounded contiguous file reads and
  direct tokenizer vocabulary sizing reduce startup work. Exclusively owned GGUF
  staging arrays and upload temporaries are released after WebGL consumes them.
  Dependent storage probes consume every output group; a layout ruled out by decode
  skips its redundant full-batch prefill probe.
  Platforms without ArrayBuffer.transfer retain conservative loading buffers.
- Startup correctness checks cover real inputs on every distinct attention/cache
  variant and both model ends, against CPU attention and the alternate GPU paths.
  Recurrent models keep their full sequence/state checks. No correctness check is
  replaced by a successful speed result.

Build, typecheck, test and lint results are recorded in
[validation.txt](results/validation.txt). Lint warnings are reported separately
from errors. Host work included compilation, mathematical tests, serving and device
control, followed by the user-authorized M3 Ultra Chrome model benchmark.

Rejected experiments included smaller prompt batches, draining every small band,
extra activation buffers, unrolled/transposed float kernels, mixed-precision dots,
synchronous FileReaderSync loading and large fixed submission budgets. Automatic
byte-activation streaming was also rejected after Mac full-model timing disagreed
with the isolated matvec probe. A storage-probe candidate picked native RGBA4 on
Pixel and caused a 20-second cold load; the final guarded selection was checked
in repeated fresh-calibration loads. The earlier Mac native-layout/controller
combination also regressed and is retained as a rejected control. A broad
eight-layer scratch-layout search took 26.25 seconds on iPhone and was rejected;
the retained packed comparison avoids those alternate scratch allocations. Intermediate
results are not combined with the final build. A user-confirmed tab-closing
interruption produced a roughly 30-second pause and is excluded from controlled
metrics. Earlier 60/80 Hz iPhone measurements are superseded by the 120 Hz comparison.

The FPS metric is requestAnimationFrame while repainting the canvas and animating
the CSS spinner, not a system-wide presentation trace. Device heat and OS scheduling
can change throughput; sustained stress measurements must not be compared with a
rested run as though only engine code changed. Older iPhones and macOS Safari were not measured; the same engine follows the
observed display cadence, with unit coverage for both 60 and 120 FPS. A final client-side
guard for whole-millisecond frame timestamps was added after the device timing runs
and verified with synthetic 60/120 Hz sequences. The reported device runs used
fractional-millisecond timestamps, whose estimator is unchanged.
[WebKit time precision implementation](https://github.com/WebKit/WebKit/blob/main/Source/WebCore/page/Performance.cpp)
provides the reason to cover coarse clocks; the rounded-clock branch was not
physically retested on the iPhone.

[Selected raw phase records](results/webgl-performance-2026-11-03.jsonl) and
[source/build provenance](results/source-manifest.json) preserve the evidence.
The dedicated worktree contains the changes; publishing or pushing is not part of
this measurement report.
