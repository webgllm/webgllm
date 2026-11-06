# LFM2.5 GGUF support and validation

Three official models are registered in the WebGL package and Lofi Agent's WebGL group.
Existing model ids, default selection, caller overrides, and generation limits are unchanged.

| WebGL model | Official file | Temperature | Top-k | Repetition penalty |
|---|---|---:|---:|---:|
| `LiquidAI/LFM2.5-350M-GGUF` | `LFM2.5-350M-QAD-Q4_0.gguf` | 0.1 | 50 | 1.05 |
| `LiquidAI/LFM2.5-1.2B-Instruct-GGUF` | `LFM2.5-1.2B-Instruct-QAD-Q4_0.gguf` | 0.1 | 50 | 1.05 |
| `LiquidAI/LFM2.5-1.2B-Thinking-GGUF` | `LFM2.5-1.2B-Thinking-Q4_0.gguf` | 0.05 | 50 | 1.05 |

Recommendations are from the official [350M card](https://huggingface.co/LiquidAI/LFM2.5-350M-GGUF),
[Instruct card](https://huggingface.co/LiquidAI/LFM2.5-1.2B-Instruct-GGUF), and
[Thinking card](https://huggingface.co/LiquidAI/LFM2.5-1.2B-Thinking-GGUF).
Top-p and min-p are not specified in those card recommendations, so they inherit the
existing sampler defaults of 1 and 0. Every explicit caller sampling override still wins.
The three GGUFs declare 128,000-token context; no default maximum output or thinking
budget was introduced. Lofi's existing iOS context policy is unchanged.

Thinking is always enabled for the Thinking model, with `reasoning: {level:"high",
budgetTokens:...}` supported. The other two models do not advertise a thinking mode.
QAD metadata/tokenizers/templates come from each official GGUF repository's `qad/`
folder. Thinking uses its original official model tokenizer/template. Every usable
HF token matches GGUF, chat templates match exactly, and native tokenization agrees.
The GGUF vocabulary includes 1,134 unused padding tokens beyond the 64,402 HF tokens;
these are not missing usable tokens. Full inventory is saved in the tokenizer report.

## Loader changes

LFM2 uses `token_embd_norm.weight` as the final norm, separate operator/FFN norms,
no sandwich post-norms, and no attention output-gate tensor. Its short convolution,
NEOX full-head RoPE, q/k norms, Q4_0 projections and Q6_K tied embedding/head reuse the
existing WebGL kernels. GGUF metadata supplies the actual FFN dimensions (4,608 for
350M; 8,192 for 1.2B), rather than reading nominal HF intermediate sizes. Untied LFM2
heads and unknown architectures fail explicitly. No GPU kernel or third-party runtime
patch was required.

## Native numerical evidence

Chrome on Apple M3 Ultra ran the actual WebGL loader/engine against full-vocabulary
llama.cpp CPU distributions (reference commit `19cce69`). Each model used two prompts,
each followed by three teacher-forced continuation steps: eight distributions/model.

| Model | Top-token matches | Maximum KL vs quantized CPU | Cache restored/continued logits |
|---|---:|---:|---|
| 350M QAD | 8/8 | 0.006895 | Bit-identical |
| 1.2B Instruct QAD | 8/8 | 0.035475 | Bit-identical |
| 1.2B Thinking | 8/8 | 0.000149 | Bit-identical |

The original strict KL < 0.01 diagnostic failed one Instruct distribution. That failure
is retained in `instruct-numerical.json`; the tolerance was not widened to hide it.
A separate control dequantized the same Q4_0 model to F16 using llama.cpp's standard
`llama_model_quantize` API. The Q4 CPU versus F16 CPU difference reached KL 0.036132
on the same step. WebGL continued using the original Q4_0 file; against this F16 CPU
control it matched all eight top tokens, with maximum KL **0.000375**.

The matching reference source explains the precision-path distinction: Q4_0 CPU dot
products use Q8_0 activations (`ggml_vec_dot_q4_0_q8_0`), while the F16 control uses
F16 activations. These observations support a backend arithmetic difference rather
than an LFM2 loader/layout fault. They do not claim bit-identical CPU/GPU distributions.
Production inference never switches to a float model or CPU backend.

The 4,096-token context, four-token continuation length and comparison tolerances belong
to this diagnostic only. Actual adapter calls below retained model context/default
sampling. Model hashes, source hashes and all native results are archived.

## Actual adapter and dropdown checks

All three actual catalog models passed, in Chrome through the WebGL adapter:

- Four exact-schema city responses each, including a repeat and two changing prompts:
  **12/12 final responses correct**, no malformed JSON or schema dead ends.
- Exact retries restored all 78 input tokens; the fourth changed-prompt call restored
  65 shared tokens for every model using standard `threadId`/`promptCache` options.
- Required and automatic `get_weather` calls: **6/6**, coherent tool events with the
  expected Tokyo argument. Automatic calls used native LFM2 tool syntax.
- Lofi's actual picker exposed and selected all three new entries under **WebGL**.

The final Thinking test explicitly used a 512-token reasoning budget and a 1,024-token
output allowance. Observed reasoning counts were 418/512/512/362 on the four schema
calls, confirming native budget enforcement and natural early closure. Neither limit
is a new model default. A prior 128-token test produced valid JSON but selected Paris
instead of London once on a changing prompt (no cached input reused on that call).
That short-budget semantic failure is retained separately; it was not repaired or
silently retried. The same fixture passed at 512. This is not proof every future short
budget call fails, nor a guarantee of semantic correctness for arbitrary requests.

WebGL **197 tests** and Checkers **170 tests** pass; all available affected Nx tests,
typechecks, lint and builds pass for WebGL, Lofi Agent and Checkers. Existing lint
warnings remain; no new production warnings were introduced.

## Reproduction

Keep model weights outside the repository. Prepare matching tokenizer inputs:

```sh
packages/webgllm/node_modules/.bin/esbuild packages/webgllm/bench/prepare-lfm2-gguf.ts --bundle --platform=node --format=esm --outfile=/tmp/prepare-lfm2-gguf.mjs
node /tmp/prepare-lfm2-gguf.mjs /path/to/models
python3 packages/webgllm/bench/lfm2-gguf-reference.py /path/to/models /path/to/llama-server
MODELS_DIR=/path/to/models node packages/webgllm/bench/server.mjs 8797
```

Open `http://localhost:8797/lfm2-gguf.html` in Chrome and explicitly run each model.
To prepare the optional dequantized file, compile `dequant-lfm2-reference.cpp` against
your matching llama.cpp headers/libllama and run it with input/output GGUF paths.
The optional F16 control expects `instruct/refs-f16.json`, generated with the reference
script's `--key instruct --f16-control /path/to/dequantized.gguf` option. It always
compares the original quantized GPU weights with that separate CPU control.

The adapter checks (structured outputs, prompt caching and tool calls through the
TanStack AI adapter) live with the adapter in tanstack-ai-browser.

Evidence: summary (raw run archived locally),
manifest (raw run archived locally),
CPU precision control (raw run archived locally),
GPU/control comparison (raw run archived locally),
Thinking short-budget result (raw run archived locally),
tokenizer inventory (raw run archived locally),
Nx output (raw run archived locally),
Lofi dropdown (raw run archived locally).

Only [the current compact summary](latest.json) is tracked. Full transcripts, screenshots,
logs and intermediate precision controls are local artifacts under the repository retention policy.
