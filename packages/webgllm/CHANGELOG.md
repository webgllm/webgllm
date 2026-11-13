## 0.3.3 (2026-11-13)

### 🚀 Features

- **webgllm:** image input for LFM3-VL and LFM2.5-VL (GGUF and ONNX), and Qwen3.5's GGUF ([#12](https://github.com/webgllm/webgllm/pull/12))

### 🔥 Performance

- **webgllm:** Firefox on Android no longer stuck writing at 2 tokens/s, and reads prompts twice as fast ([41e907d](https://github.com/webgllm/webgllm/commit/41e907d))

### ❤️ Thank You

- daveycodez

## 0.3.2 (2026-11-12)

### 🔥 Performance

- **webgllm:** reading a prompt, waits on the GPU sleep instead of polling ([e46db93](https://github.com/webgllm/webgllm/commit/e46db93))

### ❤️ Thank You

- daveycodez

## 0.3.1 (2026-11-11)

### 🩹 Fixes

- **webgllm:** a worker of your own that fails no longer leaves requests waiting ([#11](https://github.com/webgllm/webgllm/pull/11))

### ❤️ Thank You

- daveycodez

## 0.3.0 (2026-11-11)

### 🚀 Features

- ⚠️  **webgllm:** a worker handler at the root, a worker of your own, tokens and logits ([#10](https://github.com/webgllm/webgllm/pull/10))

### ⚠️  Breaking Changes

- **webgllm:** a worker handler at the root, a worker of your own, tokens and logits  ([#10](https://github.com/webgllm/webgllm/pull/10))
  the worker option is a Worker, not () => Worker; the webgllm/advanced entry and the engine's exports are removed."
  M	apps/web/content/docs/guides/chat-and-streaming.mdx
  M	apps/web/content/docs/guides/loading-models.mdx
  D	apps/web/content/docs/reference/advanced.mdx
  M	apps/web/content/docs/reference/how-it-works.mdx
  M	apps/web/content/docs/reference/meta.json
  M	apps/web/content/docs/reference/webgllm.mdx
  M	apps/web/content/docs/requirements.mdx
  M	apps/web/src/components/demo/model.ts
  M	packages/webgllm/README.md
  M	packages/webgllm/bench/README.md
  M	packages/webgllm/bench/check-worker.ts
  A	packages/webgllm/bench/logits-worker.ts
  A	packages/webgllm/bench/logits.html
  A	packages/webgllm/bench/logits.ts
  M	packages/webgllm/bench/mem-worker.ts
  A	packages/webgllm/bench/own-worker.html
  A	packages/webgllm/bench/own-worker.ts
  M	packages/webgllm/bench/webgl-perf.ts
  M	packages/webgllm/package.json
  D	packages/webgllm/src/advanced.ts
  M	packages/webgllm/src/chat-completions.test.ts
  M	packages/webgllm/src/chat-completions.ts
  M	packages/webgllm/src/client.test.ts
  M	packages/webgllm/src/client.ts
  M	packages/webgllm/src/engine.ts
  M	packages/webgllm/src/fake-worker.test.ts
  A	packages/webgllm/src/handler.ts
  M	packages/webgllm/src/index.ts
  M	packages/webgllm/src/manifest.test.ts
  M	packages/webgllm/src/model-files.test.ts
  M	packages/webgllm/src/openai.ts
  M	packages/webgllm/src/pool.test.ts
  M	packages/webgllm/src/pool.ts
  M	packages/webgllm/src/prompt-images.ts
  M	packages/webgllm/src/prompt.ts
  M	packages/webgllm/src/protocol.ts
  M	packages/webgllm/src/sampler.test.ts
  M	packages/webgllm/src/sampler.ts
  M	packages/webgllm/src/shared-download.test.ts
  M	packages/webgllm/src/shared-download.ts
  M	packages/webgllm/src/storage.ts
  M	packages/webgllm/src/tokenizer-vocab.test.ts
  M	packages/webgllm/src/tokenizer.ts
  M	packages/webgllm/src/webgllm.test.ts
  M	packages/webgllm/src/webgllm.ts
  M	packages/webgllm/src/worker-cancellation.test.ts
  A	packages/webgllm/src/worker-download.test.ts
  M	packages/webgllm/src/worker-generation.test.ts
  M	packages/webgllm/src/worker-load.test.ts
  M	packages/webgllm/src/worker.ts

### ❤️ Thank You

- daveycodez

## 0.2.3 (2026-11-10)

### 🚀 Features

- **webgllm:** image input for Qwen3.5 ([#9](https://github.com/webgllm/webgllm/pull/9))

### ❤️ Thank You

- daveycodez

## 0.2.2 (2026-11-10)

### 🚀 Features

- **webgllm:** models by URL ([#8](https://github.com/webgllm/webgllm/pull/8))

### ❤️ Thank You

- daveycodez

## 0.2.1 (2026-11-09)

### 🚀 Features

- ⚠️  **webgllm:** OpenRouter's reasoning object for the effort and thinking budget ([#7](https://github.com/webgllm/webgllm/pull/7))

### ⚠️  Breaking Changes

- **webgllm:** OpenRouter's reasoning object for the effort and thinking budget  ([#7](https://github.com/webgllm/webgllm/pull/7))

### ❤️ Thank You

- daveycodez

## 0.2.0 (2026-11-09)

### 🚀 Features

- ⚠️  **webgllm:** an OpenAI-shaped client, and reasoning levels as reasoning_effort values ([#6](https://github.com/webgllm/webgllm/pull/6))

### ⚠️  Breaking Changes

- **webgllm:** an OpenAI-shaped client, and reasoning levels as reasoning_effort values  ([#6](https://github.com/webgllm/webgllm/pull/6))
  a model's reasoning levels hold the reasoning_effort value
  to send for each, { off: "none", high: "high" }, instead of the chat
  template's enable_thinking values ("false", "true"), which webgllm never
  read. They now describe webgllm's own API.
  * fix(webgllm): manifests stored before 0.2.0 read with today's reasoning levels; baseURL's comment says what it covers
  * chore: update every dependency to its latest version
  @huggingface/jinja 0.5.11, Nx 24.0.5, assistant-ui, fumadocs, TanStack
  Start, Tailwind and the rest. fumadocs-mdx stays at 15.5.1: 15.5.2 hands
  Vite its meta.json files as JavaScript, and the docs site's build fails.
  * fix(webgllm): a manifest from before 0.2.0 is deleted and its model resolves again, instead of being converted"
  M	apps/web/package.json
  M	package.json
  M	packages/webgllm/package.json
  M	packages/webgllm/src/lfm2-gguf.test.ts
  M	packages/webgllm/src/manifest.test.ts
  M	packages/webgllm/src/model-meta-types.test.ts
  M	packages/webgllm/src/model-meta.ts
  M	packages/webgllm/src/resolve-model-files.test.ts
  M	packages/webgllm/src/resolve-model.ts
  M	packages/webgllm/src/storage.ts
  M	packages/webgllm/src/webgllm.test.ts
  M	packages/webgllm/src/webgllm.ts
  M	pnpm-lock.yaml

### ❤️ Thank You

- daveycodez

## 0.1.9 (2026-11-09)

### 🩹 Fixes

- **webgllm:** count the prompt's tokens in the repetition penalty, as Transformers.js does ([#5](https://github.com/webgllm/webgllm/pull/5))

### ❤️ Thank You

- daveycodez

## 0.1.8 (2026-11-09)

### 🩹 Fixes

- **webgllm:** load ONNX exports by graph structure, and float16 tied embeddings ([#4](https://github.com/webgllm/webgllm/pull/4))

### ❤️ Thank You

- daveycodez

## 0.1.7 (2026-11-09)

### 🚀 Features

- **webgllm:** load any Hugging Face model by name, the catalog as known defaults ([#3](https://github.com/webgllm/webgllm/pull/3))

### ❤️ Thank You

- daveycodez

## 0.1.6 (2026-11-08)

### 🚀 Features

- **webgllm:** add models.downloaded to check storage without network requests ([296251f](https://github.com/webgllm/webgllm/commit/296251f))

### ❤️ Thank You

- daveycodez

## 0.1.5 (2026-11-06)

### 🩹 Fixes

- **webgllm:** close and merge allOf objects so they're fast to constrain ([5122079](https://github.com/webgllm/webgllm/commit/5122079))

### ❤️ Thank You

- daveycodez

## 0.1.4 (2026-11-06)

### 🩹 Fixes

- **webgllm:** stop a load still waiting when its model is unloaded ([01d2b9a](https://github.com/webgllm/webgllm/commit/01d2b9a))

### ❤️ Thank You

- daveycodez

## 0.1.3 (2026-11-06)

### 🚀 Features

- **webgllm:** add onMemoryClaim so a host can count webgllm's memory in its own budget ([a3ed68e](https://github.com/webgllm/webgllm/commit/a3ed68e))

### ❤️ Thank You

- daveycodez

## 0.1.2 (2026-11-06)

### 🩹 Fixes

- **webgllm:** close objects only where it narrows the schema, scope refs by name ([b191958](https://github.com/webgllm/webgllm/commit/b191958))
- **webgllm:** cancel a queued resize without the reply, settle downloads on stored files ([13d0d13](https://github.com/webgllm/webgllm/commit/13d0d13))
- **bench:** count a frame gap across the first token, wait for uploads ([2cd750d](https://github.com/webgllm/webgllm/commit/2cd750d))
- **webgllm:** cap only formatting whitespace, not spaces inside strings ([50b67f8](https://github.com/webgllm/webgllm/commit/50b67f8))
- **webgllm:** self-check the q|k|v|gate projection the GPU computed ([3d93afc](https://github.com/webgllm/webgllm/commit/3d93afc))
- **webgllm:** keep a finished file whole under a second download, cap no-version retries ([426212e](https://github.com/webgllm/webgllm/commit/426212e))
- **webgllm:** end a stream's waiting read on return, size GGUFs without HEAD ([ff4455f](https://github.com/webgllm/webgllm/commit/ff4455f))
- **webgllm:** penalize only the reply's tokens, and keep thinking off when asked ([8179148](https://github.com/webgllm/webgllm/commit/8179148))
- **webgllm:** close response_format objects as tool schemas are ([5cafb26](https://github.com/webgllm/webgllm/commit/5cafb26))
- **webgllm:** let constrained JSON take the whitespace models write ([560279d](https://github.com/webgllm/webgllm/commit/560279d))
- **webgllm:** close a spent thinking budget the way the template does ([0138f5a](https://github.com/webgllm/webgllm/commit/0138f5a))

### ❤️ Thank You

- daveycodez

## 0.1.1 (2026-11-06)

### 🚀 Features

- **webgllm:** model lifecycle events and exact catalog types ([f338c9b](https://github.com/webgllm/webgllm/commit/f338c9b))

### 🩹 Fixes

- **webgllm:** serialize loads, fit startup probes, free failed readbacks ([b5f8530](https://github.com/webgllm/webgllm/commit/b5f8530))
- **bench:** start decode timing at the first token, survive bad requests ([b3bced0](https://github.com/webgllm/webgllm/commit/b3bced0))
- **webgllm:** serialize stream reads, scope tool schemas, refuse a zero penalty ([c724bfa](https://github.com/webgllm/webgllm/commit/c724bfa))
- **webgllm:** share file downloads and drop canceled evictions ([695dc9d](https://github.com/webgllm/webgllm/commit/695dc9d))
- **webgllm:** only resume downloads that match the stored version ([ccff39a](https://github.com/webgllm/webgllm/commit/ccff39a))

### 🔥 Performance

- **webgllm:** sample large top_k and nucleus without full scans ([c74f7e8](https://github.com/webgllm/webgllm/commit/c74f7e8))

### ❤️ Thank You

- daveycodez

## 0.1.0 (2026-11-06)

### 🚀 Features

- **webgllm:** repair automatic tool calls that fail their tools ([8a16dc0](https://github.com/webgllm/webgllm/commit/8a16dc0))
- **webgllm:** add the WebGLLM class with an OpenAI-compatible API ([88341fd](https://github.com/webgllm/webgllm/commit/88341fd))
- **webgllm:** add stop sequences, seeds, tool choice and downloads to the worker ([7208b59](https://github.com/webgllm/webgllm/commit/7208b59))
- enforce structured output across browser adapters ([7117df0](https://github.com/webgllm/webgllm/commit/7117df0))
- **checkers:** bound AI thinking and play forced moves immediately ([3b0a7cc](https://github.com/webgllm/webgllm/commit/3b0a7cc))
- **checkers:** add local AI game with model selection ([8d40ddd](https://github.com/webgllm/webgllm/commit/8d40ddd))
- **webgl:** setPromptBatch lets the caller trade prompt speed for memory; no f16 scratch without 4-token matmuls ([ad06f7f](https://github.com/webgllm/webgllm/commit/ad06f7f))
- **lofi-agent:** LFM3's GGUF on the WebGL engine is the default model ([751b9d7](https://github.com/webgllm/webgllm/commit/751b9d7))
- **webgl:** Qwen3.5-2B on the WebGL engine, from unsloth's Q4_0 GGUF and onnx-community's q4f16 ONNX ([2db28e0](https://github.com/webgllm/webgllm/commit/2db28e0))
- runtimes share a memory budget: loading a model unloads other runtimes' least recently used models until it fits ([ac80a85](https://github.com/webgllm/webgllm/commit/ac80a85))
- models download their latest version (main), and resume only from the same version ([1aa697d](https://github.com/webgllm/webgllm/commit/1aa697d))
- every adapter defaults to the model's own context window, read from the model; contextLength on all three ([7003626](https://github.com/webgllm/webgllm/commit/7003626))
- **webgl:** grow the cache to the model's own context length ([763e2e9](https://github.com/webgllm/webgllm/commit/763e2e9))
- **webgl:** run the LFM2.5 ONNX models, warm shaders up at load ([b6a2e75](https://github.com/webgllm/webgllm/commit/b6a2e75))
- **webgl:** speculative decoding by prompt lookup ([cc5a697](https://github.com/webgllm/webgllm/commit/cc5a697))
- **webgl:** an LFM3 inference engine and adapter on WebGL 2 ([2d78423](https://github.com/webgllm/webgllm/commit/2d78423))

### 🩹 Fixes

- **webgllm:** fail fast on values the constraint library can't prune ([c0e16e9](https://github.com/webgllm/webgllm/commit/c0e16e9))
- **webgl:** address pacing and calibration review findings ([2a52337](https://github.com/webgllm/webgllm/commit/2a52337))
- **webgl:** estimate cadence correctly with rounded timestamps ([9b40e53](https://github.com/webgllm/webgllm/commit/9b40e53))
- **webgl:** honor caller output limits and budgets for all thinking models ([7a713f5](https://github.com/webgllm/webgllm/commit/7a713f5))
- **webgl:** use recommended sampling defaults and remove game retries ([23f2491](https://github.com/webgllm/webgllm/commit/23f2491))
- **webgl:** the worker claims the textures it measured from the memory budget, not 1.1x the file ([57ac750](https://github.com/webgllm/webgllm/commit/57ac750))
- **webgl:** cleaning up old pinned copies no longer deletes a freshly downloaded tokenizer_config.json ([f7ce03f](https://github.com/webgllm/webgllm/commit/f7ce03f))
- **webgl:** loader checker passes on today's loaders: chunked embeddings, native Q6_K ([a5ab936](https://github.com/webgllm/webgllm/commit/a5ab936))
- **webgl:** closing the adapter stream stops generation; bench server path checks; loader checker reads raw weights; persistence only on downloads ([33ca06f](https://github.com/webgllm/webgllm/commit/33ca06f))
- **webgl:** review fixes: cut-down batches guard their rows, robust draft timing, background tuning that gives way ([e927621](https://github.com/webgllm/webgllm/commit/e927621))
- **webgl:** resume downloads after a closed handle, hidden-page waits, work-sized draws ([1fe4349](https://github.com/webgllm/webgllm/commit/1fe4349))
- **webgl:** free a model's GPU memory on switch, one batch size, portability fixes ([131dd88](https://github.com/webgllm/webgllm/commit/131dd88))
- **webgl:** attention masking with whole-vector ops ([685c70a](https://github.com/webgllm/webgllm/commit/685c70a))
- **webgl:** ask the GPU which multi-target draws it can do ([b61ad2d](https://github.com/webgllm/webgllm/commit/b61ad2d))
- **webgl:** code review fixes; drafting tries again mid-reply ([c04c116](https://github.com/webgllm/webgllm/commit/c04c116))
- **webgl:** pace long prompts so the browser keeps drawing ([d1838eb](https://github.com/webgllm/webgllm/commit/d1838eb))
- **webgl:** code review fixes ([c45fb50](https://github.com/webgllm/webgllm/commit/c45fb50))
- **webgl:** thinking, stops, lost contexts and downloads ([7c3c2db](https://github.com/webgllm/webgllm/commit/7c3c2db))

### 🔥 Performance

- **webgl:** improve slow fence polling and benchmark rendering ([24f02c2](https://github.com/webgllm/webgllm/commit/24f02c2))
- **webgl:** adapt pacing to slow GPU fences ([b9d9101](https://github.com/webgllm/webgllm/commit/b9d9101))
- **webgl:** recover prefill and reduce attention texture memory ([781cdd1](https://github.com/webgllm/webgllm/commit/781cdd1))
- **webgl:** shorten cold calibration and retire consumed model reads ([20316e5](https://github.com/webgllm/webgllm/commit/20316e5))
- **webgl:** skip redundant startup layout work ([f4d51e1](https://github.com/webgllm/webgllm/commit/f4d51e1))
- **webgl:** scale pacing from full workloads and stabilize startup selection ([8fe0a5a](https://github.com/webgllm/webgllm/commit/8fe0a5a))
- **webgl:** pace inference and reduce model loading memory ([4d7fef3](https://github.com/webgllm/webgllm/commit/4d7fef3))
- **webgl:** checkpoint stable structured-output prefixes ([26e1f67](https://github.com/webgllm/webgllm/commit/26e1f67))
- **webgl:** writing's attention panels two across, timed per GPU at load: 15-33% less attention time on an iPhone from 2k-8k keys ([bada4b5](https://github.com/webgllm/webgllm/commit/bada4b5))
- **webgl:** the tuner times the f16 matmul's choices on whole prompt batches, as prompts run them ([5c346e6](https://github.com/webgllm/webgllm/commit/5c346e6))
- **webgl:** writing's attention in short chunks laid out in panels: 2-2.5x less attention time from a 7,900-token cache on an M3 ([330c7f9](https://github.com/webgllm/webgllm/commit/330c7f9))
- **webgl:** f16 h2 reads by default, RGBA4 through the f16 matmul where the tuner times it faster, in 32 MiB chunks ([36655ce](https://github.com/webgllm/webgllm/commit/36655ce))
- **webgl:** foldEach off by default: it wrote Qwen3.5 GGUF 1.6% slower on an iPhone; fuseLinear and qInCache confirmed there ([4ba37a8](https://github.com/webgllm/webgllm/commit/4ba37a8))
- **webgl:** the f16 prompt matmul reads h2 texels: 4-11% faster prompts on byte weights, a pass and 15 MiB fewer ([f5b2319](https://github.com/webgllm/webgllm/commit/f5b2319))
- **webgl:** writing's queries go in free KV cache slots, drawn with the keys: one render pass fewer per attention layer ([1c1f097](https://github.com/webgllm/webgllm/commit/1c1f097))
- **webgl:** writing's Gated DeltaNet block in 6 render passes, not 10: Qwen3.5 writes 13-18% faster on an M3 ([dd4ab5e](https://github.com/webgllm/webgllm/commit/dd4ab5e))
- **webgl:** Qwen3.5 GGUF folds its norms into writing's matmuls: 34 render passes fewer a token ([2aee8fa](https://github.com/webgllm/webgllm/commit/2aee8fa))
- **webgl:** RGBA4 weights read in place in prompt matmuls: up to 6% faster prompts and 61-77 MiB less ([f6ddb7e](https://github.com/webgllm/webgllm/commit/f6ddb7e))
- **webgl:** the attention cache grows 512 positions at a time: 90 MiB less just past 8k tokens ([b87940b](https://github.com/webgllm/webgllm/commit/b87940b))
- **webgl:** Qwen3.5's follow-ups read 159 tokens in 90 ms, not 687 in 274: an anchor snapshot where they stop sharing the chat ([2e5668c](https://github.com/webgllm/webgllm/commit/2e5668c))
- **webgl:** Qwen3.5 takes 49-63 MiB less: no conv buffers, logits for one row ([f47c1a4](https://github.com/webgllm/webgllm/commit/f47c1a4))
- **webgl:** a first load takes 29-36 MiB less: the self-check frees its scratch ([4ba93dd](https://github.com/webgllm/webgllm/commit/4ba93dd))
- **webgl:** Qwen3.5 reads prompts 3.4-4.7x faster: the Gated DeltaNet recurrence runs in chunks of 32 tokens ([6143693](https://github.com/webgllm/webgllm/commit/6143693))
- **webgl:** GGUF writes as fast as ONNX: a Q6_K output head in word texels, split along K ([823428d](https://github.com/webgllm/webgllm/commit/823428d))
- **webgl:** faster loads: warm-up through one layer of each kind, tuning when idle ([24a033f](https://github.com/webgllm/webgllm/commit/24a033f))
- **webgl:** tuner times a whole layer at 512 rows, bar scales with noise; draft counts per reply ([914a78b](https://github.com/webgllm/webgllm/commit/914a78b))
- **webgl:** load-time tuning of how prompt matmuls split K, remembered per GPU ([0243bea](https://github.com/webgllm/webgllm/commit/0243bea))
- **webgl:** drafts by measured cost per token; guessing can come back; prompts skip the last layer's unused rows ([2800af0](https://github.com/webgllm/webgllm/commit/2800af0))
- **webgl:** retries resume from the prompt's saved end; greedy reads back only token ids; GGUF heads pick all rows at once ([39ab8e5](https://github.com/webgllm/webgllm/commit/39ab8e5))
- **webgl:** ready once the weights are in; shaders compile after ([c782d61](https://github.com/webgllm/webgllm/commit/c782d61))
- **webgl:** faster loads from storage: remembered checks, raw ONNX quants into RGBA4 ([82aca74](https://github.com/webgllm/webgllm/commit/82aca74))
- **webgl:** RGBA4 layer weights' scales and zero points in one texel too ([7471cd7](https://github.com/webgllm/webgllm/commit/7471cd7))
- **webgl:** pick the weight layout per GPU by timing both at load ([e117d81](https://github.com/webgllm/webgllm/commit/e117d81))
- **webgl:** 4-bit layer weights in byte texels; scales and zero points in one texel ([834ea35](https://github.com/webgllm/webgllm/commit/834ea35))
- **webgl:** split matmuls' partial sums as halves ([ebb43a9](https://github.com/webgllm/webgllm/commit/ebb43a9))
- **webgl:** pick's argmax from the chunk maxima's positions, constant-bound pick loops ([43c34a9](https://github.com/webgllm/webgllm/commit/43c34a9))
- **webgl:** up to 7 drafts per batch; review fixes ([bf7b127](https://github.com/webgllm/webgllm/commit/bf7b127))
- **webgl:** wllama-style drafts (4-token matches, longer drafts for longer matches, backoff), blocking reads with nothing to overlap ([2fdcfea](https://github.com/webgllm/webgllm/commit/2fdcfea))
- **webgl:** scissor each pass to its rect (render-area hint for tile-based GPUs) ([fe0f9a2](https://github.com/webgllm/webgllm/commit/fe0f9a2))
- **webgl:** prompt matmuls on f16-dequantized weights, 1.3-1.6x faster prompts on PowerVR ([5f13744](https://github.com/webgllm/webgllm/commit/5f13744))
- **webgl:** flush every layer while writing; bandwidth probe, chained matmul bench ([3d69724](https://github.com/webgllm/webgllm/commit/3d69724))
- **webgl:** long-context attention in 3 passes, work-based pacing ([e978612](https://github.com/webgllm/webgllm/commit/e978612))
- **webgl:** few-row conv blocks draw their output and state in one pass ([d56f963](https://github.com/webgllm/webgllm/commit/d56f963))
- **webgl:** attention fragments take two heads of a KV head together ([bdb0a5c](https://github.com/webgllm/webgllm/commit/bdb0a5c))
- **webgl:** the top-50 pick adds the head's slices itself ([4015681](https://github.com/webgllm/webgllm/commit/4015681))
- **webgl:** chunk writing's attention only past 1,024 keys ([31a10f5](https://github.com/webgllm/webgllm/commit/31a10f5))
- **webgl:** fold layer 0's and the head's norms when writing ([dede47c](https://github.com/webgllm/webgllm/commit/dede47c))
- **webgl:** guess from the chunk maxima, 128-query attention slices ([e2a602d](https://github.com/webgllm/webgllm/commit/e2a602d))
- **webgl:** bigger prompt batches for smaller models ([339d62c](https://github.com/webgllm/webgllm/commit/339d62c))
- **webgl:** chunked attention for long key ranges ([9715318](https://github.com/webgllm/webgllm/commit/9715318))
- **webgl:** 64-query attention slices on desktop, 4-slice cap ([c4485f7](https://github.com/webgllm/webgllm/commit/c4485f7))
- **webgl:** four tokens per fragment in prompt matmuls ([cbfb53c](https://github.com/webgllm/webgllm/commit/cbfb53c))
- **webgl:** check a draft batch in one head pass and one readback ([43d50e7](https://github.com/webgllm/webgllm/commit/43d50e7))
- **webgl:** constant split counts, one K/V cache texture, faster attend ([f1adec7](https://github.com/webgllm/webgllm/commit/f1adec7))
- **webgl:** drop the norm pass when writing, memoize kernel sources ([e38846e](https://github.com/webgllm/webgllm/commit/e38846e))
- **webgl:** pipeline writing, faster output head and small passes ([a31e0ff](https://github.com/webgllm/webgllm/commit/a31e0ff))
- **webgl:** spread row sums over 64 fragments, A/B timing on the bench ([634231b](https://github.com/webgllm/webgllm/commit/634231b))
- **webgl:** K-split narrow prompt matmuls, fewer render passes ([5e61dfe](https://github.com/webgllm/webgllm/commit/5e61dfe))
- **webgl:** RGBA4 weights and two-token prompt matmuls ([cb4c6b3](https://github.com/webgllm/webgllm/commit/cb4c6b3))
- **webgl:** share the Q6_K head's decoded activations across a fragment's rows ([d5ba3d1](https://github.com/webgllm/webgllm/commit/d5ba3d1))
- **webgl:** less memory, and the GGUF's Q6_K head read natively ([c09e941](https://github.com/webgllm/webgllm/commit/c09e941))

### ❤️ Thank You

- daveycodez