# Short shared-prefix caching in WebGL

Implemented on `checkers`, with baseline engine `aaf9992`. Native tests used Chrome
on an Apple M3 Ultra, as reported by WebGL. The user confirmed the background game
was closed before the final throughput runs. No browser other than Chrome was used.

## What changed

The 512-token prefill batch remains unchanged (and still scales to 1,024 for the
350M model). It is no longer a mandatory boundary for a reusable shared prefix.

- The worker compares actual encoded tokens, as before. When repeated requests
  reveal a useful shorter shared span, it captures that state inside the existing
  batch: LFM convolutions, sliding-window KV, and Qwen convolution/recurrent state.
- Qwen can extract a partial chunk state from the already computed chunk deltas.
  It does not repartition the model's normal recurrence chunks. At an aligned chunk
  endpoint it copies the existing result instead of recomputing it.
- A one-off cold request takes the original forward batches and GPU draws. There
  is no eager short-prefix snapshot, extra full model pass, or smaller batch size.
- Admission is conservative when an existing natural checkpoint could be displaced:
  preserve it on an initial profile-switch miss; learn a smaller content prefix
  after repeated matching evidence. Framing-only matches do not replace it.
- Small schema variations retain a safer previously learned fallback instead of
  continually promoting it by a few tokens. An ordinary already-saved point can be
  protected by metadata only. Exact retries do not promote the full endpoint over
  the reusable shorter prefix.
- Existing snapshot limits stay at eight for LFM and two for Qwen. Space is reserved
  before in-flight extraction; partial textures are released on a failed forward.

This is lazy caching: the engine cannot predict the exact future divergence of a
recurrent model without storing more states eagerly. Exact retries keep their existing
behavior; changed requests may need an initial learning request, and confirmation
when replacing a normal checkpoint. Consumers add no new options, boundaries, prompt
padding, or app/provider conditions. The standard TanStack API is unchanged.

No adapter, protocol, tokenizer vocabulary, shader, sampler, context/output allowance,
reasoning budget, model selection, dependency, or game-strategy changes were made.
Framing probes are tokenizer-only, memoized, and verified against the actual input;
their synthetic text is never sent to a model. The original instruction/schema layout
and actual native generation requests remain unchanged.

## Native correctness and memory

Two reversed before/after rounds tested LFM3-1.7B GGUF, Qwen3.5-2B GGUF, and
LFM2.5-350M ONNX. Inputs exercise 67-, 256-, and 2,048-token cold requests, exact
endpoints, and changing longer inputs sharing 21, 256, 389, or 733 leading tokens.
The ONNX model kept its normal 1,024-token batch; the others kept 512.

- All cold logits were bit-identical before/after, with the same GPU draw counts.
- Every admitted short boundary was restored, including points inside a batch or
  Qwen recurrence chunk. All warm top-1 logits matched their reference.
- Warm arithmetic can differ because a shorter tail uses different batch geometry:
  maximum observed absolute logit difference was 0.0184; maximum KL was 1.07e-5.
  Warm results are not claimed bit-identical in every geometry.
- Snapshot counts stayed within the unchanged limits. Reports include retained
  texture bytes and peak tracked GPU allocations. These are allocator measurements,
  not a measurement of total Chrome/driver physical memory.

Raw native records: [LFM3](lfm3-numerical.json.gz), [Qwen](qwen-numerical.json.gz),
[ONNX](onnx-numerical.json.gz).

## Standard SDK integration and long-prefix control

Final SDK tests use the same `chat()` consumer calls and direct adapter streaming
contract before/after, with model-default sampling and reasoning off. Standard
`promptCache: 'none'` isolates cold controls only; ordinary warm calls need no cache
options. The original native requests match exactly, excluding their random request
IDs: zero parity errors on both models.

Each model ran 120 cases across two reversed rounds: cold structured/plain requests,
changing schemas, seven recorded chess prompts, four long-profile controls, natural
and required tool calls, pre/during cancellation, and recovery.

| Model | Baseline | Candidate | Changed long-profile cached tokens, both variants |
|---|---:|---:|---:|
| LFM3 GGUF | 60/60 | 60/60 | 1024 |
| Qwen3.5 GGUF | 56/60 | 59/60 | 1024 |

The original long-profile counterexample is preserved: switching to a new long
system profile, retrying, then changing the user document still reuses 1,024 tokens.
A short observed match must not make that control fall back to zero.

Failures remain explicit. Qwen baseline had one wrong allowed value, two matcher
`No token can complete...` failures, and one structured-output truncation. Candidate
had one matcher dead end on a required London weather call. This error class also
occurs in baseline; this change does not claim to fix the existing structured-output
matcher issue. No missing value was guessed, no tags stripped, no schema weakened,
and no retry used to convert these failures into passes. Full errors/requests remain
in [Qwen SDK raw results](qwen-sdk.json.gz).

Pre-aborted TanStack structured finalization in this installed version reports the
generic `structured output finalization produced no result`, with no native request.
The control accepts that observed behavior only with an aborted signal and zero
inference; it does not require the SDK to preserve an AbortError class. During-run
cancellation and subsequent inference were checked through the real adapter. An
additional focused LFM cancellation run passed 12/12.

Raw SDK results: [LFM3](lfm3-sdk.json.gz), [Qwen](qwen-sdk.json.gz),
[focused cancellation](lfm3-cancellation.json.gz).

## Throughput and its limits

Cold prefill retains the original batch partition, shaders, and GPU work. Native
cold GPU draw counts match before/after at every tested length on all three models.
There is no performance tradeoff from reducing batches to manufacture checkpoints.

The normal SDK long-text runs give these aggregate generation rates (tokens after
the first divided by native decode time, six samples per variant):

| Model | Before tok/s | After tok/s |
|---|---:|---:|
| LFM3 GGUF | 89.75 | 89.75 |
| Qwen3.5 GGUF | 66.94 | 67.07 |

No consistent cold prefill throughput loss was established; normal wall-clock
samples were noisy, including large browser fence/timer scheduling spikes. Raw timing
explorations are retained under [diagnostics](diagnostics/), including reversed fresh
contexts, alternating same-context methods, and a benchmark-only control forcing the
GPU wrapper's existing blocking-readback fallback. Prototype-switch timing is a
methodology diagnostic, not a claim about normal SDK throughput. The blocking-wait
control is not shipped as a production setting. A temporary-allocation issue in the
benchmark's logit comparison was also removed; it did not change production code.

These observations support unchanged cold compute and steady decode throughput on
this Mac/Chrome. They are not a promise of exactly 0.0% wall-time difference on every
device. Model loading, downloads, first shader compilation, and retuning are separate
from warmed throughput. Full SDK first-call totals remain in the raw records rather
than being silently removed. Cached-tail pp/s is not directly comparable to a full
cold batch because it processes fewer new tokens; warm latency/cache counts are
reported separately. No sampler settings were overridden to stabilize the tests.

## Real Checkers integration

The existing lean prompt and consumer code were not changed to enable this feature.
All 37 unchanged production-order cases passed legality and all 27 scored objectives
passed. The forced single-move case still bypasses inference. Subsequent cached
requests typically reuse 410 input tokens; the pre-feature lean layout reused zero.
The seven-position recorded-game replay returns 28/28 legal results across both
historical/current layouts. Exact repeats retain full prompt reuse on LFM3.

See [quality](checkers-quality.json.gz), [cache replay](checkers-replay.json.gz),
and [Chrome proof](checkers-chrome.png). Timing differences across earlier app reports
are not claimed as a controlled performance improvement.

## Automated validation and reproduction

All 185 WebGL tests, 146 Checkers tests, typechecks, lint (zero errors; repository
warnings remain), builds/prerendering, and the independent Checkers suite audit pass.
The 16 new regressions fail on the pre-change engine: nine fail, seven pass. Tests
cover cold batching, short and fluctuating prefixes, preserved long-profile hits,
existing-point promotion, exact-retry retention, cache-disable/one-snapshot paths,
bounded simultaneous anchors, and interruption cleanup.

[Before regression output](before-regression.log), [Nx output](nx-checks.log),
[audit](checkers-audit.log), [source/dependency manifest](source-manifest.json).
Some numerical/diagnostic records were taken while admission policy was being
refined; final SDK/profile checks exercise the completed policy. Final Checkers
records were captured after the last metadata-retention guards and build checks.

```sh
node packages/webgl/bench/prepare-short-prefix.mjs aaf9992
# Reuse the Checkers dev server. In Chrome, open:
# http://localhost:3002/__short-prefix-qa.html
# http://localhost:3002/__short-prefix-sdk-qa.html
# Select a model and explicitly Run. No inference starts during preparation.
node packages/webgl/bench/prepare-short-prefix.mjs --clean
NX_DAEMON=false NX_TUI=false NX_LOAD_DOT_ENV_FILES=false pnpm nx run-many -p webgl checkers -t test typecheck lint build --outputStyle=static
```

The preparation script freezes only baseline TypeScript source, records checksums,
and creates owned temporary pages. Cleanup removes only those generated files and
never touches browser model storage. No environment files are read.
