# Bench

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
- Every smoke run ends with `LOGITS HASH`: the bits of the prompt's logits and of 8 writing steps', hashed. Two builds or settings with the same hash computed the same numbers. Fix the weight layout (`&tune=layerRgba4:1`) when comparing: an M3's timing probe picks RGBA4 or bytes from load to load, and they round differently.

## compare.html

`?run=webgl&model=…&len=512&speculative=0&tune=knob:v,…`: the worker as the app runs it (downloading the model to the page's storage the first time): a short warm-up reply, then three runs of a `len`-token chat prompt (each with its own first line, so none reuses the last), 64 greedy tokens each; prints each run and the median prompt reading, writing and first-token time, then a retry of the last prompt. `tune` sets TUNING knobs in the worker before the load (`promptBatch:256` for another prompt batch, `layerRgba4:1` to fix the weight layout). `&followup=1` instead times a follow-up: each run a first message, its reply, then an ~80-token question with the reply in the chat, as the app sends it; prints how many of the follow-up's prompt tokens were read again and how long that took.

## mem.html: memory and speed over a chat (and the iPhone page)

Runs a chat in a worker the way `worker.ts` does (cache reuse, `readPrompt`'s batches and snapshots, the one-token guess), on one of the four builds, and at each checkpoint (after the load, the warm-up and self-check, and each turn) logs the live textures by category (weights, scratch, KV cache, recurrent and conv state, snapshots) and the most there were since the last checkpoint. The header shows the live and peak textures and the last turn's speed. The model's files download to the page's storage the first time (as the app's worker does); where there's no Origin Private File System (a page that isn't a secure context) they stream from Hugging Face on every load.

- `model=qwen-onnx|qwen-gguf|lfm-onnx|lfm-gguf` (or a model's full name).
- `len=512`: the first message's length in tokens; `follow=0`: what each follow-up adds (0: a short question); `reply=128`: tokens per reply; `turns=3`, or `target=8192`: turns until the cache holds that many tokens; `ctx`: the context length (the model's own when left out); `think=1`: thinking on.
- `batch`: the prompt batch (left out: the worker's, 512); `tune=knob:v,…`: TUNING knobs; `detail=1`: every texture group at each checkpoint, not just the categories.
- `mode=ab&variants=default|f16Matmul:0|packZeros:0|batch:256|batch:128&rounds=2`: each variant in a load of its own (TUNING knobs as `knob:v,…`, `batch:N` for the prompt batch), timed on a `len`-token first message and its reply, rounds in alternating order; prints each run, then the medians: prompt reading, first token, writing, textures at load and at the peak.
- `cdp=1` waits at each checkpoint for a driver to measure the JS heaps and the GPU process over the DevTools protocol.

On an iPhone: start the server on the Mac (`node packages/webgl/bench/server.mjs 8787`; the page needs none of the model directories), then either

- tunnel it over HTTPS (`cloudflared tunnel --url http://localhost:8787`, or `ngrok http 8787`) and open `https://<tunnel>/mem.html?…` in Safari: a secure context, so the model downloads once (about 1.1 GB) and loads from the page's storage after that, and the log also prints in the server's terminal; or
- serve the network (`HOST=0.0.0.0 node packages/webgl/bench/server.mjs 8787`) and open `http://<the Mac's address>:8787/mem.html?…` on the same Wi-Fi: no storage there, so each load streams the model from Hugging Face again.

For example, an 8k chat on Qwen3.5's GGUF, with live memory and speed: `mem.html?model=qwen-gguf&len=2048&follow=1024&reply=128&target=8192&ctx=8192` (`&batch=128` for the smallest batch's memory). The timings a phone's defaults would need: `mem.html?model=qwen-gguf&mode=ab&len=535&reply=64&rounds=3&variants=default|f16Matmul:0|packZeros:0|batch:512|batch:256|batch:128|batch:128,quadMinBatch:128` (`packZeros` only matters for the ONNX builds; run it on each build). A tab that reloads by itself ran out of memory.

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

## index.html: the go/no-go benchmark

The first step, before the engine: one Q4_0 matmul kernel on every LFM3 projection shape at 128 tokens (`?run=gpu`), against wllama's relaxed-SIMD build reading the same number of tokens (`?run=cpu`). On a Pixel 10 Pro in Firefox, LFM3's matmuls for 128 tokens took 1.49 s (86 tokens/s) against 24.5 prompt tokens/s for wllama's whole model, so the engine was built.

Pitfalls it found: PowerVR merges and skips full-screen draws whose output is overwritten later, even across framebuffer switches, so a timed draw must read the previous draw's output; `unpackUnorm2x16` returns wrong values on that driver; a readback costs 10 to 35 ms of fixed latency there.

## bandwidth.html

How fast the GPU streams a 64 MiB texture through texelFetch, by texel size (RGBA4, RGBA8, RGBA8UI, RGBA16UI, RGBA32UI) and by how many texels each fragment reads. Each draw reads the previous draw's output, so no driver can skip one. On the Pixel 10 Pro (Chrome) the best layout reads ~23 GB/s, and RGBA4 tops out at ~19 GB/s (~9 G fetches/s); 16-byte texels are slowest. LFM3's writing matmuls already read their weights at ~16–17 GB/s there.

## mm2.html

`?batch=512&only=LFM3`: matmul layouts on the model's shapes (random weights; each timed draw's output gets read, so PowerVR can't skip it). On the Pixel the engine's RGBA4, 4-tokens-per-fragment prompt layout is fastest (~285 GFLOP/s); byte and word texels unpacked in the shader are 10–25% slower.

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

