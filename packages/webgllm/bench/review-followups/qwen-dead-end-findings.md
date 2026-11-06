# Qwen finite-string matcher dead end

Clean native Chrome evidence uses the unchanged strict city schema and model defaults. No third-party patch, dependency change, schema weakening, guessed output, repair pass, retry, or sampling change was introduced.

| Run | Outcome |
| --- | --- |
| Original masks, after engine, ordinary cache policy | 20/20 |
| Original masks, before engine, retention:none | 11/12, then captured dead end |
| Isolated canonical guard, before/after x default/none | 48/48 |
| Production guard only, alternating required London/Berlin | 8/8; six requests restored 89 tokens, prefix89 and anchor99 retained |
| Production native cancellation before/during/recovery | 3/3 |

The clean failure is required Berlin with cachePrompt:false, cachedTokens:0, cacheMatchedTokens:0 and zero shared matcher-mask cache hits. The live matcher accepted an impossible escaped property name, then partial UTF-8 F0 9F 8C. A speculative clone detected dead_end on its initial mask before any proposed draft. CPU replay matches all 25 native accepted-token mask hashes and reproduces dead_end. The canonical guard blocks the second live token, 11595 (bytes quote + backslash), before the impossible key. Full selected IDs, bytes, state and previous mask are in qwen-clean-native-replay.json.gz; full native reports are compressed alongside it. Downloaded tokenizer contents are not vendored.

The independent oracle compared all 248,070 Qwen tokens at all 40 city-document byte prefixes: 9,922,800 comparisons, zero mismatches. Additional optional/multi-key/tag controls cover 75,301 byte-token comparisons with zero mismatches. Unit regressions repeat finite-language prefix checks against the integrated wrapper and cover clones/EOS, tag literals, Unicode/escaped-value/unknown-schema bypass, exact ByteLevel decoder checks, caller-hint bypass, empty pieces, byte overrides and odd vocabularies that can represent a valid value only through escapes.

The first-party guard recognizes only plain strict objects of finite printable-ASCII string enum/const properties with no literal quote/backslash/control characters or extra assertions. It preserves every permitted value through canonical raw spellings and intentionally excludes escaped spellings for this profile. Caller x-guidance, Unicode, arbitrary strings, nesting, other assertions, unknown/transforming decoders and insufficient normal printable-byte coverage use the existing backend. This does not fix all llguidance JSON Schema prefix pruning. Normal unconstrained generation and cancellation remain unchanged. The mask is cached by tokenizer/vocabulary/EOS; each fresh grammar mask adds one intersection. Native tracing is diagnostic, not an unbiased throughput benchmark; no speed guarantee is claimed.

The historical saved tool-London failure has no accepted-token trace. Its exact cause cannot be reconstructed; the fresh cache-disabled reproduction proves this failure mechanism exists independently of engine restoration, not the historical run's precise path. Earlier invalid diagnostic-harness outcomes are excluded.

CPU replay: download the public tokenizer matching qwen-guard-manifest.json's SHA256, then run node packages/webgllm/bench/review-followups/qwen-dead-end-replay.mjs /absolute/path/to/tokenizer.json. The raw matcher reproducer intentionally bypasses the first-party wrapper.
