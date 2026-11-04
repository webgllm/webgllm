# Bench

Raw run artifacts are no longer tracked. Keep only [the latest compact summary](latest.json);
regenerate raw files locally as needed. Fixed regression fixtures remain separate.


Pages for checking and timing the engine, served cross-origin isolated by `server.mjs`, which also bundles the package's TypeScript on request (`/bundle/<path>.js` is `<path>.ts`).

```sh
MODELS_DIR=… REF_DIR=… REF_ONNX_DIR=… ONNX_DIR=… HF_DIR=… node packages/webgl/bench/server.mjs   # :8787
adb reverse tcp:8787 tcp:8787        # a phone over USB: localhost is a secure context
```

- `MODELS_DIR`: `LFM3-1.7B-QAD-Q4_0.gguf` (and `LFM3-1.7B-Q4_K_M.gguf` for the CPU baseline)
- `ONNX_DIR`: a copy of `daveycodez/LFM3-1.7B-Web-ONNX` (config, tokenizer, `onnx/model_q4f16.onnx*`)
- `HF_DIR`: `LiquidAI/LFM3-1.7B`'s `tokenizer.json`, `tokenizer_config.json`, `chat_template.jinja`
- `REF_DIR`, `REF_ONNX_DIR`: reference tensors from llama.cpp and onnxruntime (`manifest.json`, `tokens.json`, `logits.bin`, `gen_logits.bin`, `l_out-N.bin`, …)

Results print in the server's terminal.

## check.html

- `?format=gguf|onnx&batch=128&precision=highp|mediump&layers=1`: runs the reference prompt and compares every layer and the logits with the reference, then the reference's greedy continuation one token at a time (the decode path, and the GPU's candidate pick against the full logits), then times the prompt.
- `?smoke=1&len=512&format=…&q=…`: a chat prompt padded to about `len` tokens, read and answered as the worker does it (no reference needed); prints the reply and the speed.
- `&skip=matmul,norm,…`: times the prompt and single-token steps with each kind of pass left out, to see where the time goes.
- `&linstate=16,32` (smoke): each Gated DeltaNet layer's recurrent state after a prompt batch and after the whole prompt, the recurrence in chunks of each size against a token at a time (max abs and relative error, by layer). `&small=1`: short batches after a 64-token prefix both ways, where chunks start to pay. `&compare=linChunk:0,32` compares the logits; `&abwrite=1` times writing after a prompt read each way, rounds alternating.
- `&anchor=1` (smoke, Qwen3.5): a snapshot saved from inside a prompt batch, 7 tokens before its end (a follow-up's anchor: see prompt.ts), against reading the prompt plainly: the batch's own logits, then the rest read again from the snapshot (KL and top token for each).
- `&compare=knob:a,b` (smoke): the prompt, 16 writing steps and batches of 3 and 8 tokens (a draft batch's size: writing's attention path) with each value, logits compared row by row (worst KL, same top token). `&abDecode=knob:a,b`: writing timed both ways, rounds alternating (wall time with each token read back, then GPU time of 16 queued steps).
- `&attnab=knob:a,b&decodeAt=7900` (smoke): attention alone. Reads `decodeAt` tokens, then times one writing token's attention passes in every attention layer (scores, chunks, combine; nothing else of the step), 32 queued, at 64, 512, 1,024, 2,048 and `decodeAt` cached tokens (`&attnAt=512,7900` for others), values alternating over `&rounds=12`; prints the median and best ms. Several knobs at once: `attnab=default|flashDecode:1,flashChunkTexels:64|…`; `skip:attend` leaves out passes of a name. On an M3 a pass costs tens of µs whatever it does, so whole writing steps there are mostly passes; this shows what attention's work costs.
- Every smoke run ends with `LOGITS HASH`: the bits of the prompt's logits and of 8 writing steps', hashed. Two builds or settings with the same hash computed the same numbers. Fix the weight layout (`&tune=layerRgba4:1`) when comparing: an M3's timing probe picks RGBA4 or bytes from load to load, and they round differently.

## compare.html

`?run=webgl&model=…&len=512&speculative=0&tune=knob:v,…`: the worker as the app runs it (downloading the model to the page's storage the first time): a short warm-up reply, then three runs of a `len`-token chat prompt (each with its own first line, so none reuses the last), 64 greedy tokens each; prints each run and the median prompt reading, writing and first-token time, then a retry of the last prompt. `tune` sets TUNING knobs in the worker before the load (`promptBatch:256` for another prompt batch, `layerRgba4:1` to fix the weight layout). `&followup=1` instead times a follow-up: each run a first message, its reply, then an ~80-token question with the reply in the chat, as the app sends it; prints how many of the follow-up's prompt tokens were read again and how long that took.

## mem.html: memory and speed over a chat (and the iPhone page)

Runs a chat in a worker the way `worker.ts` does (cache reuse, `readPrompt`'s batches and snapshots, the one-token guess), on one of the four builds, and at each checkpoint (after the load, the warm-up and self-check, and each turn) logs the live textures by category (weights, scratch, KV cache, recurrent and conv state, snapshots) and the most there were since the last checkpoint. The header shows the live and peak textures and the last turn's speed. The model's files download to the page's storage the first time (as the app's worker does); where there's no Origin Private File System (a page that isn't a secure context) they stream from Hugging Face on every load.

- `model=qwen-onnx|qwen-gguf|lfm-onnx|lfm-gguf` (or a model's full name).
- `len=512`: the first message's length in tokens; `follow=0`: what each follow-up adds (0: a short question); `reply=128`: tokens per reply; `turns=3`, or `target=8192`: turns until the cache holds that many tokens; `ctx`: the context length (the model's own when left out); `think=1`: thinking on.
- `batch`: the prompt batch (left out: the worker's, 512); `tune=knob:v,…`: TUNING knobs; `detail=1`: every texture group at each checkpoint, not just the categories.
- `mode=ab&variants=default|f16Matmul:2|packZeros:0|batch:256|batch:128&rounds=2`: each variant in a load of its own (TUNING knobs as `knob:v,…`, `batch:N` for the prompt batch), timed on a `len`-token first message and its reply, rounds in alternating order; prints each run, then the medians: prompt reading, first token, writing, textures at load and at the peak.
- `cdp=1` waits at each checkpoint for a driver to measure the JS heaps and the GPU process over the DevTools protocol.

On an iPhone: start the server on the Mac (`node packages/webgl/bench/server.mjs 8787`; the page needs none of the model directories), then either

- tunnel it over HTTPS (`cloudflared tunnel --url http://localhost:8787`, or `ngrok http 8787`) and open `https://<tunnel>/mem.html?…` in Safari: a secure context, so the model downloads once (about 1.1 GB) and loads from the page's storage after that, and the log also prints in the server's terminal; or
- serve the network (`HOST=0.0.0.0 node packages/webgl/bench/server.mjs 8787`) and open `http://<the Mac's address>:8787/mem.html?…` on the same Wi-Fi: no storage there, so each load streams the model from Hugging Face again.

For example, an 8k chat on Qwen3.5's GGUF, with live memory and speed: `mem.html?model=qwen-gguf&len=2048&follow=1024&reply=128&target=8192&ctx=8192` (`&batch=128` for the smallest batch's memory). The timings a phone's defaults would need: `mem.html?model=qwen-gguf&mode=ab&len=535&reply=64&rounds=3&variants=default|f16Matmul:2|packZeros:0|batch:512|batch:256|batch:128|batch:128,quadMinBatch:128` (`packZeros` only matters for the ONNX builds; run it on each build). A tab that reloads by itself ran out of memory.

## tools.html: does the reply start with a tool call?

Prompts that need a tool (weather, web search, calendar, run Python), with thinking off, as token ids from `REF_DIR/<set>/tool_prompts.json`. For each prompt it prints the probability that the reply starts with `<tool_call>`, the top token, and KL against each CPU reference in the set at that position, then writes a 48-token greedy reply to `RESULTS_FILE`.

- `?set=lfm3&runtime=webgl&format=onnx&onnx=onnx-community`: this engine on the ONNX build in `ONNX_COMMUNITY_DIR` (`&graph=` for another graph path).
- `?set=lfm3&runtime=webgl&format=gguf&model=LFM3-1.7B-QAD-Q4_0.gguf`: this engine on a GGUF in `MODELS_DIR`.
- `?set=lfm3&runtime=tjs&onnx=onnx-community`: Transformers.js 4.3.2 on WebGPU (q4f16), with the app's ONNX Runtime.

`tool-ref/` makes the sets: `make_lfm3_ref.py` (prompts and onnxruntime CPU logits), `add_lfm3_ref.py` (another CPU reference, such as exact 4-bit math), `make_qwen_ref.py` (the same prompts in Qwen3.5's template, no logits) and `llama_ref.py` (a running llama-server's first-token probabilities and replies, for GGUFs).

LFM3 1.7B, 20 prompts, Apple M3 Ultra (Nov 1):

| | Starts with a tool call | Greedy calls first | No call in 48 tokens |
|---|---|---|---|
| fp16 (onnxruntime CPU) | 0.966 | 20/20 | |
| LiquidAI's QAD Q4_0 GGUF, this engine | 0.960 | 20/20 | 0 |
| the same GGUF, llama.cpp CPU | 0.965 | 20/20 | 0 |
| onnx-community's q4f16, this engine | 0.634 | 16/20 | 4 |
| the same, onnxruntime CPU, exact 4-bit math | 0.634 | 16/20 | 4 |
| the same, onnxruntime CPU as shipped | 0.607 | 13/20 | 5 |
| the same, Transformers.js (WebGPU) | 0.576 | 14/20 | 4 |

The engine matches exact math (mean KL 1.1e-4, same top token at 20 of 20) and llama.cpp; Transformers.js is 200× further from exact math (KL 2.1e-2). onnx-community's graph sets `accuracy_level` 4 on its 4-bit matmuls, so onnxruntime's CPU quantizes their activations to 8 bits: an exact-math reference needs a copy with 0. The q4f16 build's misses are in its weights: on three calendar prompts and one weather prompt it reasons in text (it doesn't know today's date) where fp16 and the GGUF call the tool.

Qwen3.5 2B, the same prompts with thinking off, 96-token replies (`?set=qwen&steps=96`; its template allows reasoning before a call):

| | Starts with a tool call | Greedy calls first | No call in 96 tokens |
|---|---|---|---|
| unsloth's Q4_0 GGUF, this engine | 0.967 | 20/20 | 0 |
| the same GGUF, llama.cpp CPU | 0.969 | 20/20 | 0 |
| onnx-community's q4f16, this engine (`&graph=onnx/decoder_model_merged_q4f16.onnx`) | 0.741 | 18/20 | 1 |

The q4f16 build talks before one call, and answers "sort [5, 3, 9, 1] with Python" with a code block (that doesn't sort) instead of a call.

## tjs-mem.html: the Transformers.js worker's memory

`?model=onnx-community/Qwen3.5-2B-ONNX&len=2048&max=128&ctx=8192`: the transformers-js package's worker (debug on) loads the model, reads a chat prompt padded to about `len` tokens (counted with the model's own tokenizer from Hugging Face) and writes `max` tokens greedily. It prints the worker's memory log, then a summary: GPU buffers at the load peak, after the prompt and at the reply's peak and end, WebAssembly memory, and the prompt and writing speeds. `&wait=` ms between load and prompt, `&runs=` prompts in a row.

## index.html: the go/no-go benchmark

The first step, before the engine: one Q4_0 matmul kernel on every LFM3 projection shape at 128 tokens (`?run=gpu`), against wllama's relaxed-SIMD build reading the same number of tokens (`?run=cpu`). On a Pixel 10 Pro in Firefox, LFM3's matmuls for 128 tokens took 1.49 s (86 tokens/s) against 24.5 prompt tokens/s for wllama's whole model, so the engine was built.

Pitfalls it found: PowerVR merges and skips full-screen draws whose output is overwritten later, even across framebuffer switches, so a timed draw must read the previous draw's output; `unpackUnorm2x16` returns wrong values on that driver; a readback costs 10 to 35 ms of fixed latency there.

## bandwidth.html

How fast the GPU streams a 64 MiB texture through texelFetch, by texel size (RGBA4, RGBA8, RGBA8UI, RGBA16UI, RGBA32UI) and by how many texels each fragment reads. Each draw reads the previous draw's output, so no driver can skip one. On the Pixel 10 Pro (Chrome) the best layout reads ~23 GB/s, and RGBA4 tops out at ~19 GB/s (~9 G fetches/s); 16-byte texels are slowest. LFM3's writing matmuls already read their weights at ~16–17 GB/s there.

## mm2.html

`?batch=512&only=LFM3`: matmul layouts on the model's shapes (random weights; each timed draw's output gets read, so PowerVR can't skip it). On the Pixel the engine's RGBA4, 4-tokens-per-fragment prompt layout is fastest (~285 GFLOP/s); byte and word texels unpacked in the shader are 10–25% slower.

`&pick=<regex>` runs only the variants whose label matches (from `w=…`, `tile …` or `PURE …`), three passes over the list so a shared GPU slows each alike: e.g. `pick=^(w=rgba4 a=h2 T=4|PURE 4x4 acts (f16|h2) w (f16|h2))$`.

## Prompt matmul experiments on Apple (M3 Ultra, Nov 1)

Where a prompt's time goes (`check.html?smoke=1&len=1024&batch=512&skip=…&alternate=1`, LFM3 GGUF on RGBA4): matmuls 81%, attention (attend, scores) 6%, the adds 4%, conv 4%, norms 6%, heads 3%. Then `mm2.html?batch=512&only=LFM3&pick=…` on the 4-token RGBA4 kernel (gate|up as 2 K-slices 2.6–2.7 ms, down as 4 K-slices 1.7 ms):

- It isn't bound by its arithmetic: no activation sums (wrong results, timing only) is 3% faster, centered weights, mediump and each h2 texel fetched once for both halves are the same speed.
- Nor by weight fetches alone or activation fetches alone: one weight texel used for all 4 rows (a quarter of the weight fetches) or one token's activations for all 4 tokens are no faster; both at once are 1.6× faster. The transposed output (fragment x a token group) is the same speed.
- So bigger tiles would pay, but every kernel with more than 4 rows × 4 tokens' live state falls off a cliff on this compiler: 4 rows × 6 tokens (3 outputs) 7× slower, 4×8 and 8×4 8–10×, the same with their sums added into 2 outputs (so it isn't the 64 bytes of outputs), with the scale folded into each weight as it's read (no per-block sums) 8×, and with the K steps in a loop that can't unroll still 3–4×. 8 rows × 2 tokens: 1.4× slower. The 4×4 tile is the one that fits.
- The block's activation sums read from a texture rather than added up: 2× slower. Byte weights (RGBA8, split in the shader) with fewer values live: still 6× slower.

What does pay is the f16 matmul reading h2 texels (TUNING.f16Pack 3): its dequantized weights written as h2 (8 halves a texel, half the fragments in the dequantization) and its activations read straight from their h2 tensor (no RGBA16F copy, a pass fewer a matmul): half the fetches for the same bytes. `PURE 4x4 acts h2 w h2` against `acts f16 w f16`: gate|up 2.56 vs 3.37 ms, down 1.61 vs 1.84; against the in-place RGBA4 kernel 2.75 and 1.73. In the engine, the same logits (`LOGITS HASH` unchanged on GGUF weights; ONNX KL ≤1.5e-5, rounding of the zero points), and on byte weights (`tune=layerRgba4:0`, 5 rounds alternating, medians):

| | 535-token prompt, f16Pack 0 → 3 | 2,048 | 40 |
|---|---|---|---|
| LFM3 GGUF | 203.8 → 188.9 ms (−7%) | 717 → 693 (−3%) | 73.9 → 61.9 |
| LFM3 ONNX | 181.8 → 165.7 (−9%) | 1,209 → 1,136 (−6%) | 60.9 → 53.1 |
| Qwen3.5 ONNX | 204.6 → 195.7 (−4%) | 1,007 → 900 (−11%) | 66.2 → 59.3 |
| Qwen3.5 GGUF | 260.3 → 249.7 (−4%) | 1,956 → 1,821 (−7%) | 101.1 → 96.1 |

Writing doesn't use it. On an iPhone 16 Pro Max (Safari, LFM3 GGUF, 546 tokens) it pays more: byte weights 3,069 → 2,398 ms (−22%), and RGBA4 weights through the f16 matmul (`f16Matmul:2`) with h2 reads beat the in-place kernel, 3,768 → 3,243 ms (−14%); the M3: 163 vs 179 ms (−9%). So f16Pack 3 is the default, and Engine.tune times 0 against it wherever prompts take the f16 matmul (PowerVR's 16-byte texels stream slowest and its unpackHalf2x16 costs ALU, so the Pixel chooses by timing): on the M3, 0 is 8–16% slower.

RGBA4 weights through the f16 matmul need its scratch, which ae8aceb removed (60 MiB for LFM3, 48 for Qwen3.5). `f16ScratchMiB` (32) bounds it: a matrix bigger than that is dequantized and multiplied a group of its K-slices at a time (rows at a time if one slice doesn't fit: `Engine.f16Plan`), the scratch shaped for the fewest chunks. The logits are bit for bit the same. On the M3 the chunks cost what the f16 matmul saved (each waits on the last; within one page, alternating):

| 535-token prompt (M3) | in place | f16, whole matrix | f16, 32 MiB chunks |
|---|---|---|---|
| LFM3 GGUF | 203 ms | 196 | 216 |
| LFM3 GGUF, 2,048 tokens | 688 | 652 | 724–736 |
| LFM3 ONNX | 176 | 164 | 177–178 |
| Qwen3.5 GGUF | 318 | 308 | 326–333 |
| Qwen3.5 ONNX | 204 | 187 | 200–201 |

Smaller chunks cost more: LFM3 at 16 MiB 210 ms and 24 MiB 213, against 179 at 32 and 165 whole (row chunks rather than K-slices: 242 at 16 MiB); two scratch textures in turn (a chunk's dequantization not waiting on the last chunk's matmul) made no difference. So RGBA4 weights stay in place by default (TUNING.f16Matmul 1) and Engine.tune times the chunked f16 matmul against them, keeping it where it's faster.

It first timed a layer's matmuls back to back, which missed what prompts gain: on an iPhone 16 Pro Max whole prompts read 16% faster through the chunked f16 matmul (3,148 → 2,645 ms, LFM3 GGUF, 546 tokens) while the tuner measured −0.3 to +2.4% (noise 2–4%), and on the M3 it put the chunked f16 matmul 13–19% behind where whole prompts are 0–7% behind. Back to back, each dequantization waits on the matmul before it and adds its whole time; in a prompt it has no inputs from the passes before it, so it fills the gaps the norms, adds, conv and attention leave (dependent passes with few fragments). So the f16 choices are now timed on full prompt batches (512 synthetic tokens, the cache reset) through the first `tuneLayers` (8) layers, every pass included (`check.html?…&tuneab=tuneLayers:8,0` runs the tuner 3 times with each value). On the M3: chunked, f16Matmul gain −3.0 to −4.0% at noise 1.1–2.6% (8 layers and all 24 alike; whole prompts in the same page 177.5 vs 178.6 ms): it keeps in place. With the whole matrix (f16ScratchMiB 0), +3 to +10% (whole prompts 176.4 vs 167.1 ms): it picked f16 once in the six. Bytes: f16Pack 0 timed 11–35% slower, so 3 stays. The tuner then takes ~0.7 s on the M3. A forward needs the cache, so the worker tunes those two choices only while it holds no chat (`keepCache`), else leaves them for the next load (the note keeps no f16 picks). Byte weights, which have no in-place kernel, keep the whole matrix (`f16ScratchBytesMiB` 0; 32 MiB chunks: 180 vs 165 ms on the M3).

Textures (M3, a 535-token prompt; LFM3 GGUF / Qwen3.5 GGUF, at load and the peak):

| | LFM3 | Qwen3.5 |
|---|---|---|
| RGBA4 in place | 1,239 / 1,307 MiB | 1,487 / 1,542 |
| RGBA4, f16 matmul in 32 MiB chunks | 1,271 / 1,339 | 1,519 / 1,574 |
| RGBA4, f16 matmul, whole matrix | 1,299 / 1,367 | 1,535 / 1,590 |
| bytes, f16Pack 0 (before) | 1,316 / 1,384 | 1,549 / 1,604 |
| bytes, f16Pack 3 | 1,299 / 1,367 | 1,535 / 1,590 |

## Writing experiments that lost (Pixel 10 Pro, Oct 31)

- gate|up and SwiGLU in one pass with no K-split (576 fragments for the 350M, each over all of K): 3× slower on the Pixel and 1.8× on an M3. Writing passes need thousands of fragments; a pass fewer doesn't pay for that.
- Fewer RMS partial sums (rmsParts 16): no change; 4: 14% slower.
- Output head in 1 or 2 K-slices instead of 4: no change.
- Transform feedback for writing's chain (tf.html): a vertex shader per step writing a buffer the next step reads as a uniform block beat fragment passes for 1024-wide matrices on the Pixel (162 vs 267 µs a step), not 2048-wide; copying its buffer back into a texture costs 1–4 ms on the Pixel, and uniform blocks are 16 KB on some GPUs, so it isn't used.
- 8 rows a writing-matmul fragment (two row groups sharing a block's activations): 3× slower on the Pixel and on an M3.

## Weight texels (Oct 31)

4-bit layer weights now live in byte texels (RGBA8, normalized), not RGBA4: half the fetches per block, which a fetch-bound GPU (PowerVR) turns into writing speed (LFM3 12.8 → 16.6 tok/s, 350M +17%). Apple GPUs split RGBA4's nibbles for free in the texture unit and pay for the byte split in ALU; scales and zero points share one texel and the split takes a floor and two dots per texel, leaving the M3 2–4% and an iPhone 16 Pro Max 15–40% behind RGBA4. So the engine times both at load (Engine.chooseLayout: the first layer's gate|up shape, random weights, the writing matmul on each) and keeps the faster: bytes on the Pixel (0.405 vs 0.496 ms), RGBA4 on the iPhone (0.276 vs 0.306) and the M3. TUNING.layerRgba4 forces one.

## Apple writing experiments (350M, M3 Ultra and iPhone 16 Pro Max, Oct 31)

- What a pass costs (`mm2.html?batch=0` on the iPhone): every change of render target costs ~21 µs, even for a 1-pixel draw into a 1×1 texture, and scissor or invalidate hints don't change it; more draws into the same target cost ~1.2 µs each. A step that reads the last step's output needs its own render pass, so this is a floor per dependent step.
- A 350M writing step on the iPhone (`check.html?…&onnx=lfm25-350m&skip=attend`): 148 passes, 6.5 ms of GPU time. About 3 ms of that is the per-pass floor and ~2.3 ms is streaming the ~140 MB of 4-bit layer weights; most single matmuls take the same ~46 µs (iPhone) or ~0.13 ms (M3, Chromium) whatever their shape (`mm2.html?batch=1&only=350M`). Flushing every layer or only at the readback: no difference.
- Slab reads (`bandwidth.html`, each step of all fragments reading one contiguous block, like a blit): no faster on the iPhone (RGBA8 53 vs 49 GB/s, RGBA4 slower) and far slower on the M3. What does matter on Apple is fragments per pass: 4k fragments stream ~35 GB/s on the iPhone, 16k ~44, 64k+ ~49.
- One or two weight rows per fragment (4× or 2× the fragments, each streaming a quarter or half of a row group; `rowKernel` in mm2.ts): slower than four rows on both (each fragment still fetches every activation), e.g. the 350M down 0.067 vs 0.053 ms on the iPhone.
- The 350M's zero points aren't all 8 (73% differ), so the kernels without zero points don't apply to it.

## Review follow-ups that didn't pay (Nov 1)

- Short prompt batches on byte texels (the Pixel's layout) without f16 dequantization: the direct 2-token byte kernel read a 59-token 350M prompt in 390 ms against 174 ms through f16; a 4-token byte kernel (`mm2.html?batch=64`, gate|up 9216×1024 on the Pixel) takes 6.3 ms against RGBA4's 5.3 and plain f16's 3.2 (plus one dequant per matrix). Unpacking bytes in the shader costs more than the fetches it saves.
- Attention's softmax shared across a head's value fragments: at 8,192 prompt tokens on an M3 the value pass is 12% of the prompt (scores 4%, matmuls 70%); sharing would save about a fifth of that pass's fetches.
- Keeping dequantized f16 weights across prompt batches: dequantization is 5% of a 2,048-token LFM3 prompt on an M3, for hundreds of MB more per layer kept.
- Ranking chunks only up to k, or scattering chunk ranks with points: the four pick passes take ~0.2 ms a token on an iPhone (of ~2.3 ms for head, pick and readback); the head and the sync are the rest.
- JS-side pass state caching: issuing a 350M token takes 0.34 ms of JS on an iPhone, under 6.5 ms of GPU work it overlaps.

## Load-time tuning (Nov 1)

`check.html?…&sweep=knob:v1,v2;knob2:…` times each value on the prompt, its first 40 tokens and writing, rounds interleaved. On the 350M and LFM3 the writing knobs (splitFragments, maxSplit, headSplit) and f16Half were flat on an M3, an iPhone 16 Pro Max and the Pixel (within noise; the phones heat up during a sweep, so only values of one knob compare). How prompt matmuls split K wasn't: 4 slices for short batches on the M3 (LFM3 40 tokens 55 vs 62 ms with 2) but 2 on the iPhone (241 vs 276 ms; 350M 52 vs 61); K-slices of 256 values for full batches on the M3's 350M (44.6 vs 51.1 ms for 516 tokens), the same as 512 elsewhere, 1024 never better. Engine.tune times the short-batch choice (2 or 4 slices) on a layer's prompt matmuls when the worker is first idle after a load (not during it), each sample repeated to ~25 ms, the alternative warmed first, and keeps it only where it beats TUNING's by 4% and by twice the spread of its own rounds: 2 on the iPhone (11% faster), 4 elsewhere (2 is 30% slower on the M3). The full-batch slice size isn't tuned: timed alone, a layer's matmuls show no steady difference on the M3 (−2% to +16% from run to run), though a whole 516-token 350M prompt reads 7–13% faster with 256-value slices; timing whole prompts would take seconds on a phone.

## First-load time (Nov 1)

`check.html?…&programs=1` loads with salted shader sources (nothing reused from an earlier compile), then prints each kernel's compile time, grouped. On the iPhone 16 Pro Max (iOS 18.7) a cold load compiles ~60 programs in ~150 ms of compile proper: programs recompiled on an idle GPU take 1–6 ms, and the few that seemed to take 100–450 ms were waiting on warm-up passes still running. The ~4 s cold compile seen earlier is gone; what a load spends is GPU work. So the warm-up reads through the first and last layers and one layer of each other kind only (`TUNING.warmSubset`), not the whole model, and now also runs a 64-token batch (the f16 path full prompts take) and the greedy argmax, which no request then compiles: LFM3 warm-up 2.2 s → 0.65 s on the Pixel, ~0.35 → 0.25 s on the iPhone. The tuner runs when the worker is idle, not during the load (a request stops it). iPhone LFM3 first load after an update: ~2.4 s, of which the weights 1.7 s and the self-check 0.4 s (first load only).

## Greedy tokens chained on the GPU: nothing to gain (Nov 1)

`check.html?…&chain=1` times greedy writing the worker's way (each token read back while the next is read forward from the GPU's own pick) against the same forwards and output heads queued with no readback at all: 350M 7.8–8.3 vs 8.0–9.4 ms a token on the iPhone, 20.0 vs 19.8 on the Pixel, 5.5 vs 5.5 on an M3; LFM3 33–36 vs 31–33 (iPhone, heat-noisy) and 55.1 vs 55.0 (Pixel). The one-token lookahead already keeps the GPU busy, so chaining several tokens between readbacks (the review's item 10) isn't built.

## Qwen3.5's recurrence in chunks (Nov 1)

Where a Qwen3.5 2B prompt's time went (`skip=recurrence`, ONNX, M3 Ultra, Chrome): with the Gated DeltaNet recurrence run a token at a time (two dependent passes a token in each of 18 layers), 627 of 964 ms on a 512-token prompt (65%) and 2,543 of 3,935 ms on 2,048 tokens (65%), at 128-token batches. Prompt batches now run it in chunks (`TUNING.linChunk`; `delta-rule.ts` has the math and a CPU reference the tests check against the token-at-a-time rule): six passes over the whole batch (the chunks' decays summed, the Gram matrices, the triangular solve, U|W and Oi|Q'), then two passes a chunk. The same profile: 382 ms in all, the recurrence ~6 ms.

- Chunk size (`sweep=linChunk:…`, a 529-token prompt): at 512-token batches (the worker's), 0 (a token at a time) 896 ms, 16: 201, 24: 193, 32: 195, 48: 203; on 2,069 tokens 3,635 against 789, 760, 770, 800. At 128-token batches 8 is 2% slower than 16–32 and 64 is 29% slower (545 vs 424 ms): the solve keeps a chunk's column of T in a per-fragment array, 64 vec4s at 64. 32 is kept: within 1.5% of 24 here, and fewer dependent passes where each costs more (phones).
- Short batches (`small=1`, after a 64-token prefix): a token at a time wins at 2 tokens (17.9 vs 20.0 ms), ties at 3, and chunks win from 4 (21.2 vs 21.7; 8: 24.7 vs 28.6; 24: 43.9 vs 58.4), so batches of 4 or more go in chunks (`TUNING.linChunkMin`).
- Accuracy (`linstate=…`): the first Gated DeltaNet layer's state, read both ways from the same inputs, agrees to 2e-6–1.5e-5 of its largest value; deeper layers to 1e-4–1.6e-3, as the layers before them differ slightly too. Logits over 2,069 tokens and 16 writing steps: worst KL 6e-6, the same top token throughout. The reference checks (both builds, 512-, 128- and 48-token batches, without two render targets, with mediump matmuls) match the token-at-a-time numbers.
- The worker's self-check compares the two ways on its 40-token prompt (the first layer's state) and, past 1e-2, reloads plain, which runs the recurrence a token at a time: ~0.2 s more on a first load on the M3.
- The chunk tensors take 22 MiB at 512-token batches (5.4 MiB at 128).


## Memory and follow-ups (Nov 1)

`mem.html` (see above) on an M3 Ultra in Chrome, the four builds at 512-token batches; `compare.html` for speed, the weight layout fixed to RGBA4 on both sides of every comparison (`tune=layerRgba4:1`: the M3's layout probe picks RGBA4 or bytes from load to load, and the worker remembers its first pick per code version, which made some comparisons compare layouts).

- Kept, each the same speed within the runs' noise (prompt ±2–4%, writing ±1.5%, 3 runs a side): the self-check frees its scratch (29–36 MiB on a first load); Qwen3.5 makes no conv buffers and logits for one row (it reads no draft batches: 49–63 MiB); the anchor snapshot (follow-ups read 159 tokens in 90 ms rather than 687 in 274 ms, ONNX; GGUF 146 ms against 391; 2 snapshots kept, saved after the oldest is freed: 39 MiB less at the peak); the cache growing 512 positions at a time (an 8,192-token prompt's `LOGITS HASH` the same with `kvStep:0`, Qwen3.5 ONNX); no f16 scratch where no 4-token matmuls run (small batches).
- The anchor first came as a batch of its own for the generation prompt's 7 tokens, as long as a writing step (~12 ms, 6% of a 535-token first message's first token): it's now saved from inside the batch, its recurrence chunks laid out so one ends there (`delta-rule.ts` checks that layout against a token at a time; `check.html?…&anchor=1` the logits both ways: KL 3e-6 and 1e-5).
- The feed-forward in parts along its intermediate dimension (gate|up and SwiGLU a part at a time, each weight still read once; `ffnParts`): the same bits as in one go, but prompts 1.5% (Qwen3.5) to 8% (LFM3) slower at 2 parts, 11% at 4 (each part waits for the last to finish with the f16 copy's scratch). It would have saved 30 MiB (Qwen3.5) and 48 (LFM3). Not kept.
- `scoreRows:64`: −16 MiB of scores (Qwen3.5), prompts 1.7–2.2% slower at 2k. `packZeros:0` (ONNX): −41 MiB; Qwen3.5 the same speed, LFM3 writing 1.4–2.7% slower. Not kept.
- Kept: RGBA4 weights read in place by their own 4-token kernel in prompt matmuls, with no f16 copy (`f16Matmul` 1; byte weights, which have no such kernel, still go through f16). With `f16Matmul:0`, 3 runs a side: 0.3–5.8% faster prompts on the M3 on all four builds, the same quality, and 61–77 MiB less (the scratch is never made). A spot check after the change, M3, Chrome, 512-token batches: Qwen3.5 GGUF 2,121 against 2,067 tokens/s on a 2,069-token prompt, 1,487 against 1,549 MiB at load, prompt KL 2.08e-3 and 31 of 32 against llama.cpp; ONNX on RGBA4 KL 8.94e-6 against onnxruntime, greedy identical, 1,404 MiB. The engine picks RGBA4 where chooseLayout's timing at load favours it (Apple GPUs; the Pixel takes bytes), so no new device check. `f16Matmul:2` brings back f16 for RGBA4, which `mem.html` can time on a phone.
- Prompt batches by model, by device or by measured memory: not kept, the batch is one value for every device and model (512; `setPromptBatch` for the caller to choose). Batches of 256 and 128 read LFM3's prompts 32% and 56% slower than 512 did before, and nothing WebGL reports tells a phone's memory from a desktop's (an M3 and an iPhone both report 16,384-texel textures).

## Fewer render passes per written token (Nov 1)

Each change of render target costs ~21 µs on an iPhone however small the draw (see the Apple experiments above), so a writing step's render passes are worth counting: `check.html?smoke=1&skip=attend&decodeAt=600&len=600` prints them by kind, and `abDecode=knob:0,1` prints each value's count. Three knobs, each bit-identical to off except `foldEach` (Qwen3.5 GGUF only). `fuseLinear` and `qInCache` are on; `foldEach` is off:

- `foldEach`: each of a layer's norms folds into its own split matmuls, and 8-bit split matmuls read the add's prescaled input too. Qwen3.5 GGUF's beta|alpha is Q8_0, so its Gated DeltaNet layers drew both norms in passes of their own: 379 → 345 passes, writing KL against llama.cpp 1.86e-3 → 1.85e-3. Off by default: on an iPhone 16 Pro Max (sweep alternated in one page) it wrote 1.6% slower (32.5 against 32.0 GPU ms a token) and read a 613-token prompt slower, though an M3 wrote 5% faster.
- `fuseLinear`: writing's Gated DeltaNet block in 6 passes, not 10 (beta|alpha beside in_proj in the partial buffer, the conv with its state, and `linDeltaFused` computing the gates and q̂|k̂ itself): Qwen3.5 GGUF 345 → 273, ONNX 342 → 270; writing GPU time −13% and −18% on an M3; on an iPhone 16 Pro Max (alternated in one page), Qwen3.5 GGUF 34.6 → 32.7 GPU ms a token, ONNX 336 → 264 passes and 36.1 → 34.2 ms (43.1 → 41.6 ms a token as the app writes).
- `qInCache`: the queries' f32 bits go in the cache slots just past the tokens being written, drawn with the keys and values (`cache-slots.ts`): one pass fewer per attention layer, LFM3 281 → 271 (11 layers), Qwen3.5 −5 to −7. On the iPhone: LFM3 GGUF 28.8 → 28.6 GPU ms a token from 600 tokens and 43.6 → 43.0 from 4,000; Qwen3.5 GGUF 32.9 → 32.7.

What's left is the chain of passes that each read the last one's output. LFM3 GGUF's step (273 with the output head and its slice sum) is that chain: a conv layer is in_proj, conv, out_proj, rms, add, gate|up, SwiGLU, down, rms, add (10), an attention layer qkvg, q|k|v, scores, attend (and combine past 2,048 keys), o, rms, add, gate|up, SwiGLU, down, rms, add (12–13). Qwen3.5 sits 18 passes above its chain: the recurrent state's update draws into its own texture, beside the gated norm. Going lower means computing something twice, and the ways tried on paper cost more than a pass:

- The sandwich norms' rms passes (48 for LFM3): the add needs y's whole-row RMS before it can normalize y, and y comes from a K-split matmul, so no pass before the add can draw it. Drawing it in the add means every add fragment re-reads all of y's slices (~3.4 M fetches a pass for LFM3, ~100 µs on an iPhone). Deferring the add (keeping h and y·w apart and letting the next matmul combine them, with the sums of h², h·y and y² drawn with the add) removes the rms passes, but gives in_proj, qkvg and gate|up a second activation vector: ~11% more fetches a block on RGBA4 and ~19% on the Pixel's bytes, on two thirds of the weights, against ~1 ms of passes on the iPhone.
- SwiGLU folded into down's reads: gate and up come from RGBA16F partial sums, 32 fetches a block instead of 4 h2 ones, recomputed by each of down's 4,096 fragments.
- The queries computed inside the scores pass: each key texel's fragment would normalize and rotate its heads' queries again, thousands of times at long contexts.

## Writing's attention over a long cache (Nov 1)

Where it went: on an iPhone 16 Pro Max, writing an LFM3 token from a 7,900-token cache spent 5.9 of 46 GPU ms in attention (attend 4.3, scores 1.6). `attnab` on an M3 (one writing token's attention passes, all 11 LFM3 attention layers) split the same way: of 2.6 ms at 7,900 tokens, the chunk pass ~1.5, scores ~0.5, the combine ~0.2, the five sliding layers ~0.3; at 64 tokens 0.83 ms, all of it the passes themselves. The chunk pass read as many texels as scores but took three times as long: 31 chunks of 64 texels, 128 fragments each (3,968 in all, 4–16 tiles of a 128×31 target), each walking its chunk twice (the max, then the weighted sum) with one value texel and two score texels a step. The 6 full-attention layers' cache is ~97 MB at 7,900 tokens; the M3 streams that in ~0.12 ms, so the old pass was waiting on its own reads, not memory.

What's kept: writing (≤ 8 rows) past 512 keys (`flashPast` 128 texels) can read its keys in chunks of 32 texels (`flashChunkTexels`; more where that would make over 128 chunks), the softmax kept online in one loop (each score read once), chunk k in panel k % 2 of row k / 2 (`flashPanels` 2: LFM3 at 7,900 tokens, 62 chunks on a 256×31 target), then a combine that walks 8 chunks a step with their reads issued together (it's a few hundred fragments: each waits out its reads). Scores are unchanged. `Engine.tune` times this against the old chunks on a scratch 4,096-key cache of the first full-attention layer's shape and keeps it where it's 4% faster and twice its noise (`tuned.flashDecode`; `TUNING.flashDecode` 1 forces it, 0 is the old path): the iPhone 16 Pro Max picks it, 14% faster (LFM3 and Qwen3.5, 4 tunes in a row, noise ≤ 1.8%); an M3 shared with other GPU work measured 13–32% faster but often too noisy to switch. The self-check compares each attention layer's panel output with the one-pass kernel's from the same queries and cache (2e-4–3.5e-4), whichever path the GPU writes with; the warm-up compiles the path in use.

The first version put every chunk side by side (a 7,936×1 target at 7,900 tokens): 2× faster on the M3, but on the iPhone 2.4× slower than the old chunks (`attnab`, LFM3, 7,900 tokens: 14.4 against 6.0 ms). The phone, `attnab` alternated in one page, ms per token (best of 8):

| LFM3, 7,900 tokens | old | all across | 16 across | 8 | 4 | 3 | 2 | 1 |
|---|---|---|---|---|---|---|---|---|
| run 1 | 6.00 | 14.43 | | 5.17 | | | | |
| run 2 | 5.97 | | 6.95 | 5.41 | 4.96 | | 4.61 | |
| run 3 | 7.53 | | | | | 4.90 | 5.07 | 5.08 |

Other knobs at 2 across on the phone: 64-texel chunks 3.41 against 3.50 ms at 7,900 tokens (Qwen3.5: 3.50 against 3.90, but slower at 512 and 2,048); the max first and then the sums (`flashOnline` 0) 5.31 against 5.07; 2 texels a step (`flashUnroll` 2) 4.99 against 5.07. At 2 across, iPhone 16 Pro Max (Safari), old → new, best of 8:

| ms per token, attention passes only | 512 | 2,048 | 7,900 |
|---|---|---|---|
| LFM3 (11 attention layers) | 1.06 → 0.82 | 2.11 → 1.79 | 7.53 → 5.07 |
| Qwen3.5 2B (6) | 0.52 → 0.41 | 0.91 → 0.71 | 4.91 → 3.90 |

Whole writing steps from 7,900 tokens on the phone (`abDecode`, GPU ms a token, median of 6 alternated rounds): LFM3 GGUF 50.1 → 48.9 (best 49.2 → 47.6), Qwen3.5 GGUF 46.0 → 45.3 (best 45.0 → 44.2).

M3 Ultra, Chrome, `attnab=flashDecode:0,1`, best of 16–20 alternated rounds (ms per writing token, attention passes only):

| | 512 | 2,048 | 7,900 |
|---|---|---|---|
| LFM3, all across | 1.15 → 0.99 | 1.22 → 1.04 | 2.57 → 1.26 |
| LFM3, 2 across | 1.16 → 1.00 | 1.22 → 0.97 | 2.55 → 1.64 |
| Qwen3.5 2B, all across | 0.81 → 0.67 | 0.83 → 0.68 | 2.18 → 0.86 |

Tried on the M3 and not kept: 16-texel chunks (124 at 7,900 tokens): the combine doubles and Qwen3.5 slows (1.42 against 0.86); 2 or 4 texels per step of the chunk loop flat. The two-loop chunk pass with the combine still a chunk at a time cost as much in the combine as in the chunks at 124 chunks. On Apple GPUs `skip=` of a kind of pass can mislead: passes left drawing into one target back to back get elided (skipping attend left the scores passes at 0.13 ms for all 11 layers on the phone).

## Short shared-prefix caching

See [native Chrome results and reproduction](short-prefix-results/FINDINGS.md) for
inside-batch state extraction, preserved cold batches and long-prefix behavior,
SDK tool/cancellation checks, bounded memory, and explicit remaining matcher failures.
