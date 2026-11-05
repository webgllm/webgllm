# Adapter/engine review follow-ups

Baseline: `7c5f856`. Consumer inputs, schema, sampling, output allowance, context,
ordinary cancellation and standard TanStack cache API remain unchanged. No prompt
tuning or consumer adapter-specific switches were introduced.

## Prompt API pool fix

The desired-limit regression failed before the fix: six concurrent distinct cold
threads retained six idle bases instead of four. `trimIdleSessions()` now runs on
insertion and when a retained lease becomes idle. It skips active entries and evicts
the oldest eligible idle entry. Independent working clones survive parent eviction.
A superseded cold same-key streaming lease is destroyed on release instead of being
left alive outside the map; its replacement is preserved.

98 shared tests pass, covering concurrent cold threads, working-clone cancellation,
active leases, LRU eviction and same-key cold races. Live Chrome: six successes with
four retained bases; cancelling one of six clones leaves five successes, four bases,
zero working clones; explicit cleanup leaves zero sessions. Raw record destruction
flags in the original native diagnostic reflect post-cleanup state; `liveBases` and
`during` counters were captured before cleanup.

## Short-prefix benchmark decision

**Do not ship either experimental strategy globally.** Production batching and
prefix handling remain unchanged by this follow-up.

The three-way experiment ran 156 requests in two reversed rounds: normal natural
checkpoints, exact caller-instruction boundary, and a conservative tokenization
memo. All were legal; all 16 scored quality requests per variant passed. Actual
worker requests and full input token arrays matched across all variants (zero parity
errors). The initial instruction-only candidate retained 210 tokens; a separate
26-request confirmation reproduced the former patch's 222-token boundary by also
including its unchanged fixed JSON-format instruction. All 26 were legal and all
8 scored cases passed; its requests/token arrays also match the baseline exactly.
The first replay input is 2,214 tokens, matching the committed chess report.

Round-two replay changed-turn observations (milliseconds):

| Variant | Total median | Prefill median | Prefix discovery median | Reused tokens |
| --- | ---: | ---: | ---: | ---: |
| Baseline | 1009.6 | 800.5 | 0.5 | 0 |
| Exact instruction boundary (210) | 956.3 | 747.9 | 0.0 | 210 |
| Memo candidate | 997.9 | 790.3 | 0.0 | 0 |

Full input encoding still costs about 2.5–2.9 ms in these Chrome runs. The previously
reported aggregate 16 ms was not reproduced here; it includes mandatory input
encoding as well as discovery. The memo candidate saves the second discovery pass,
but **loses reuse on the long profile-switch control**: baseline reused 1,024 tokens,
memo reused zero (both rounds). Reject it despite the micro-optimization.

The exact boundary improves warm reuse, but it changes batch segmentation. On the
67-token cold single-prompt control, baseline/memo execute `[67]`, while exact splits
`[21,46]`. Cold total times were noisy (baseline 1325/1247 ms; exact 1157/1260 ms;
memo 1249/1312 ms), so the extra pass is established structurally, not inferred from
those totals. The first full replay cold totals were comparable (~1090–1120 ms),
but that does not establish a universal single-prompt benefit.

Timing limitations: Heroes of the Storm was running/paused during this session;
a pause was confirmed before an aborted IAB attempt, and a fresh quiet-state
confirmation was not obtained after switching to Chrome. Generation/host latency
varied substantially even across reversed rounds. Treat totals as observations,
not a reliable hardware speed guarantee or evidence of a 3x improvement. The later
222-token confirmation recorded ~1423 ms changed-turn total/~839 ms prefill in a
separate potentially contended window; do not compare that directly with the
round-two table. Preserve the raw per-round results. No sampling defaults changed.

## Numerical and memory checks

Real Chrome GPU checks use actual recorded chess inputs with shared prefixes of
278 and 319 tokens. Baseline/memo restored zero; exact confirmation restored 222.
All top-1 logits matched. Cross-engine baseline-cold versus exact-warm max absolute
differences were ~0.023–0.027; within each engine cold/warm comparisons were identical,
and both nonzero matched-segmentation copy controls were bit-identical with zero
snapshot writes during the disturbance. Across-engine drift also appears in the
memo variant with unchanged batching, so do not attribute all drift solely to the
split. Snapshot limit stayed 8; observed snapshot counts remained below it. No
bitwise-equivalence guarantee is claimed across fresh GPU contexts/layout choices.

## LFM3 formatting investigation

For the original35-token Paris trigger, accepted IDs were:

`41176 (Paris), 124902 (</think>), 41176 (Paris), 124900 (EOS)`.

The template already ended in `<think></think>`; thinking was false, no synthetic
opening or thinking budget was active. Streaming pieces and full decoding both
produce `Paris</think>Paris`. All four capital-city traces agree: this is generated
token behavior, not duplicated decoding/SDK text assembly. It does not by itself
identify whether model/runtime/reference inference differences cause that sampling
choice. No token mask, temperature change, deduplication, or blanket string stripping
was added.

Literal-preservation parser regressions pass. Native Chrome's ordinary enum JSON
control also passes. However the exact ASCII constant `Paris</think>Paris` fails in
the constraint path with `No token can complete the requested output schema`, after
accepted pieces include `{"`, `value`, `":"`, `Paris`, then an escaped quote and an
incomplete byte. This is a separate **unresolved matcher/tokenization investigation**;
we did not return an altered string, weaken the const assertion, or hide the failure.
A prior full UTF literal diagnostic was interrupted when its IAB worker appeared
stalled; that closure is cleanup, not proof of standard cancellation. No successful
Unicode-tag-literal native result is claimed.

## Reproduction

`node packages/webgl/bench/review-cache-prepare.mjs` prepares frozen baseline/exact/
memo variants from7c5f856 plus fixtures fromorigin/chess. Run `/__review-prefix-qa.html`
in Chrome. `node packages/webgl/bench/review-trace-setup.mjs`, followed by
`node packages/webgl/bench/review-trace-progress.mjs`, prepares the trace page.
The fixture provenance includes the original chess consumer and move-schema source
hashes. Generated source copies and app HTML pages are temporary and removed before
final Nx checks. Raw JSON is gzip-compressed here; e.g. Python `gzip.open` reads it.


## Final validation

All 508 automated tests pass (98 shared/native,159 WebGL,111 Transformers,140
Checkers); typechecks and builds passed across all seven local/shared/app projects.
The initial combined run caught a diagnostic-script empty catch; it was fixed and
WebGL lint reran successfully with only its48 existing warnings. New trace-runner
non-null assertions were also removed. No failing native literal diagnostic is
counted as an automated pass. The historical-boundary source generator was
smoke-tested and its generated worker bundled successfully. Temporary app pages
and both generated source trees were removed.

The current preparer reproduces the historical222 boundary (caller instructions
plus fixed JSON-format instruction). The first three-way dataset used210 (caller
instructions only); its saved `prepared-210.json` identifies those source hashes.
The numerical HTML template is `../review-numeric-template.html` and includes its
setup command. No production prefix or sampling strategy was changed after these
measurements.

A subsequent combined Nx confirmation passed all test/typecheck/lint/build targets; see `final-checks.log`.

Upstream matcher bug filed as https://github.com/nico-martin/llguidance-js/issues/3. A subsequent fail-closed audit added adapter regression coverage and rejects failed native generations before executable tool-call events. See `packages/structured-output/bench/review-fail-closed.md`.
