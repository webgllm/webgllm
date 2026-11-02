# @tanstack-ai-browser/webgl

A [TanStack AI](https://tanstack.com/ai) adapter for LFM3 1.7B, LFM2.5 and Qwen3.5 2B running in the browser on its own WebGL 2 inference engine, in a Web Worker on an OffscreenCanvas. It runs with or without WebGPU (the demo app starts on it everywhere, iPhone Safari included); where there's no WebGPU, Firefox on Android among them, it reads prompts on the GPU about three times as fast as llama.cpp on the CPU.

```sh
npm install @tanstack-ai-browser/webgl @tanstack/ai
```

```ts
import { chat } from "@tanstack/ai"
import { webglText } from "@tanstack-ai-browser/webgl"

const stream = chat({
  adapter: webglText("LiquidAI/LFM3-1.7B-GGUF"),
  messages,
  threadId
})
```

It needs WebGL 2 with float render targets (`EXT_color_buffer_float`) and OffscreenCanvas in workers, which Firefox, Chrome and Safari have; `hasWebGl2()` checks.

## Models

`WEBGL_TEXT_MODELS` lists the models, and `WEBGL_MODEL_META` has each one's files (each repository's latest version, main), context, capabilities, reasoning levels and the sampling from its card (LFM3: temperature 0.1, top-k 50, repetition penalty 1.1; Qwen3.5: temperature 0.6, top-k 20, top-p 0.95).

| Model                            | Weights                                                       | Download |
| -------------------------------- | ------------------------------------------------------------- | -------- |
| `LiquidAI/LFM3-1.7B-GGUF`        | Liquid AI's QAD Q4_0 GGUF (quantization-aware trained)        | 1.05 GB  |
| `daveycodez/LFM3-1.7B-Web-ONNX`  | 4-bit ONNX build for the web, with a 49k-token vocabulary     | 900 MB   |
| `onnx-community/Qwen3.5-2B-ONNX` | onnx-community's q4f16 decoder (its lm_head is the embedding) | 1.09 GB  |
| `unsloth/Qwen3.5-2B-GGUF`        | unsloth's Q4_0 GGUF (its few Q4_1 and Q5_K matrices as 8-bit) | 1.21 GB  |

LFM3 thinks by default, Qwen3.5 doesn't; `chat({ reasoning })` switches it through the chat template's `enable_thinking`. **Tools**, **usage** (with `promptTokensDetails.cachedTokens`, and timings in `usage.providerUsageDetails`), `modelOptions` sampling and `chat({ outputSchema })` work as with the package's other adapters. `loadModel(model)` downloads and loads a model ahead of time, and `modelStatus(model)` is a store of its progress.

### Speed

Pixel 10 Pro (PowerVR DXT-48-1536), Firefox 159, a 515-token prompt:

|                                         | Prompt reading      | Writing       | First token |
| --------------------------------------- | ------------------- | ------------- | ----------- |
| `daveycodez/LFM3-1.7B-Web-ONNX`         | 75.6 tokens/s       | 9.2 tokens/s  | 6.8 s       |
| `LiquidAI/LFM3-1.7B-GGUF`               | 71.9 tokens/s       | 7.9 tokens/s  | 7.2 s       |
| wllama (CPU, relaxed SIMD), LFM3 Q4_K_M | 24.5 to 30 tokens/s | 11.6 tokens/s | 18.2 s      |

Writing is still slower than on the CPU: each token pays a GPU round trip and about 400 small draws. The Web build writes faster than the GGUF one, its output head being 49k rows of 4 bits rather than 128k of 8.

Qwen3.5 2B on a Mac (M3 Ultra, Chrome), a 535-token prompt, median of 3:

|                                                | Prompt reading | Writing       | First token |
| ---------------------------------------------- | -------------- | ------------- | ----------- |
| `onnx-community/Qwen3.5-2B-ONNX` (WebGL)       | 2,777 tokens/s | 81.2 tokens/s | 0.20 s      |
| `unsloth/Qwen3.5-2B-GGUF` Q4_0 (WebGL)         | 2,138 tokens/s | 70.0 tokens/s | 0.26 s      |
| Transformers.js (WebGPU), the same ONNX        | 17.7 tokens/s  | 19.7 tokens/s | 30.3 s      |
| wllama (WebGPU), unsloth's Q4_K_M              | 595 tokens/s   | 20.4 tokens/s | 0.90 s      |

The WebGL rows read prompts with the Gated DeltaNet recurrence in chunks (see How it runs), 512 tokens at a time, the weight layout fixed to RGBA4, medians of 18–21 runs; a token at a time they read 653 and 607 tokens/s, writing the same. A follow-up question (~80 tokens, after a 64-token reply) reads 159 tokens in 90 ms (ONNX) and 146 ms (GGUF). The other rows are from an earlier run.

### Correctness

Both builds were checked tensor by tensor against a reference:

- **GGUF against llama.cpp** at the commit wllama builds (`19cce69`), on a 345-token prompt (past the 256-token sliding window): mean KL 1.1e-3 between the logits, below the 3.0e-3 between llama.cpp's own CPU and Metal backends; the same top token at 31 of 32 single-token decode steps. Every layer's output has a cosine similarity of at least 0.9992. The same holds on the Pixel with mediump matmuls.
- **ONNX against onnxruntime** (CPU, exact 4-bit math): mean KL 2.2e-5, every top token the same, the greedy continuation identical over 32 tokens.
- **Qwen3.5 2B**, on a 204-token chat prompt and 32 greedy tokens: the ONNX build against onnxruntime, mean KL 8e-6 (prompt) and 1.1e-5 (writing), the greedy continuation identical; the GGUF build against llama.cpp (CPU), mean KL 2.1e-3 and 1.9e-3, the same top token at 31 of 32 steps, every layer's output at a cosine similarity of at least 0.9989. Most of that gap is llama.cpp's: its CPU matmuls quantize their inputs to 8 bits, and the Gated DeltaNet's output projection takes heavy-tailed ones (peak 27× the RMS), which costs it 3.4% there; with that quantization applied to its own input, the engine's dequantized Q5_K weights give llama.cpp's output to 2e-7.

`bench/check.html` runs these checks (see `bench/README.md`); `docs/llama-cpp-lfm3.md` is LFM3's graph as llama.cpp computes it, which the engine follows.

## How it runs

`chat()` runs tools and the agent loop; each model call is one generation in the worker. The worker downloads the files to the Origin Private File System (resuming a dropped download), then streams the weights from there into textures, one layer at a time.

- **Loaders:** `loadGguf` and `loadOnnx` read either format into the same weights (`weights.ts`): 4-bit blocks of 32 in llama.cpp's Q4_0 order, with zero points where the ONNX build has them. The GGUF's Q6_K embedding (also the output head) is kept as it is and read natively in the shader, so its weights are exactly llama.cpp's. The embedding goes to the GPU in chunks of rows, so it's never all in JavaScript memory. Only tensors are read from the ONNX graph; its ops aren't run.
- **Kernels:** fragment shaders, one output texel per fragment (`gpu/kernels.ts`). Matmuls read weights from byte textures and activations as packed halves, eight per texel. A one-token matmul splits its rows' blocks across fragments, so the GPU has enough work. Every op of LFM3's graph is a pass: RMSNorm, the gated short convolution with its state, QK-norm and RoPE, attention over a half-precision KV cache (sliding-window layers see their last 256 tokens), the sigmoid output gate, SwiGLU, and the sandwich norms.
- **Prompts** are read 512 tokens at a time on every device (`setPromptBatch` for fewer: see Memory); big passes are split into bands of rows, under mobile GPUs' watchdogs.
- **Sampling:** with top-k (LFM3's 50), the GPU applies the repetition penalty and picks the 50 chunks of 256 logits with the largest logits, which hold the top 50 tokens for sure, so writing a token reads back ~50 KB rather than every logit. Readbacks go through a pixel buffer and a fence.
- **Qwen3.5's Gated DeltaNet** (18 of its 24 layers) is a pass each for the gates (beta, decay), the causal convolution with its state, and q and k's L2 norms, then the recurrence over each head's 128×128 fp32 state, and the gated RMSNorm. Writing runs the recurrence as two passes a token (the delta and output, then the update). Prompt batches run it in chunks of 32 tokens, as llama.cpp and flash-linear-attention do (`delta-rule.ts` has the math): the decays summed within a chunk, the chunk's Gram matrices, a triangular solve and the products that don't read the state, for every chunk at once, then two passes a chunk that do (its deltas and outputs, then the state after it). Its full-attention layers gate their output per dim.
- **Cache reuse:** a follow-up reads only what's new. The full-attention cache rewinds to any position; the convolutions' state, Qwen3.5's recurrent state and the sliding-window layers' last window are saved (snapshots) to resume from: LFM3's after each prompt batch and reply (the latest 8, ~3 MB each), Qwen3.5's (~20 MB each) after the reply and at the anchor, where the next question's prompt stops sharing the chat, 2 in all. Qwen3.5's template writes a past reply back without the `<think>` tags its generation prompt opened, so a follow-up differs from `<|im_start|>assistant` on; the anchor is saved there, from inside the prompt's last batch (its recurrence chunks laid out so one ends there), and a follow-up reads only the reply and the new message (159 tokens rather than 687 in `bench/compare.html?followup=1`). `src/prompt.ts` finds where a template's follow-ups stop sharing, once per template (LFM3's write the reply back as it was: the anchor is the prompt's end). A tool's result keeps the last question's anchor, the reply before it being written back as it was. Qwen3.5 writes without speculative drafts (keeping only part of a draft batch would need the recurrent state after each of its tokens).
- **Context:** the model's own (LFM3: 262,144 tokens; LFM2.5: 128,000). Full-attention caches start at 1,024 tokens and grow 512 at a time as a chat grows (doubling left up to half of it unused), wrapping into side-by-side blocks past the GPU's texture height; sliding-window layers keep only their window plus a batch, in a ring. Past 8,192 keys, attention reads them in segments and combines them.

### Memory

GPU textures over a chat on an M3 Ultra in Chrome (`bench/mem.html`: a first message of 64, 2,048 or 8,192 tokens, a 64-token reply, then a short follow-up), at the 512-token batch every device takes, the most held at once, MiB. "Page" adds what Chrome's GPU process and the renderer took on top of the textures (both measured against their state before the load), as an estimate of a page's total; Safari's own overhead isn't measured here.

| Model                            | 64 tokens | 2,048 tokens | 8,192 tokens | Page at 8,192 | Before (64 / 2,048 / 8,192; page) |
| -------------------------------- | --------- | ------------ | ------------ | ------------- | --------------------------------- |
| `onnx-community/Qwen3.5-2B-ONNX` | 1,532     | 1,550        | 1,622        | ~2,210        | 1,603 / 1,677 / 1,821; ~2,340     |
| `unsloth/Qwen3.5-2B-GGUF`        | 1,613     | 1,631        | 1,703        | ~2,140        | 1,670 / 1,745 / 1,889; ~2,320     |
| `daveycodez/LFM3-1.7B-Web-ONNX`  | 1,247     | 1,275        | 1,365        | ~1,810        | 1,283 / 1,329 / 1,492; ~1,930     |
| `LiquidAI/LFM3-1.7B-GGUF`        | 1,361     | 1,389        | 1,480        | ~1,920        | 1,397 / 1,443 / 1,607; ~2,070     |

A first load also holds the self-check's scratch for a moment (LFM3: up to 1,270 and 1,384 MiB). What went: the self-check's scratch once it's done (29–36 MiB), Qwen3.5's unused conv buffers and the logits and head scratch for draft batches it never reads (49–63 MiB), its third snapshot and the one made before the oldest went (39 MiB at the peak), and the cache's doubling (it grows 512 positions at a time: 18 MiB at 2k, 90 MiB just past 8k). Qwen3.5's chunked recurrence added 21 MiB.

Phones take the same 512-token batch: nothing a GPU reports tells an iPhone's ~2 GB page from a desktop's, and smaller batches read LFM3's prompts slower than it read them before (a third slower at 256). `setPromptBatch(128)` before a load trades prompt speed for memory on any model:

| Prompt batch | Qwen3.5 ONNX: 8k textures, prompt | Qwen3.5 GGUF | LFM3 Web ONNX | LFM3 GGUF |
| ------------ | --------------------------------- | ------------ | ------------- | --------- |
| 512          | 1,622 MiB, 2,750–2,815 tokens/s   | 1,703, 2,110–2,140 | 1,365, 2,930–2,960 | 1,480, 2,920–2,950 |
| 256          | 1,485, 1,860–1,920                | 1,566, 1,340 | 1,240, 1,990  | 1,354, 2,000 |
| 128          | 1,353, 1,235–1,280                | 1,433, 770–800 | 1,099, 1,230–1,280 | 1,214, 1,230–1,280 |
| Before, 512  | 1,821, 650                        | 1,889, 610   | 1,492, 2,900–2,940 | 1,607, 2,920–2,950 |

(M3 Ultra, Chrome; prompt reading on 535- and 2,075-token first messages, `bench/compare.html`. 128-token batches take no 4-token matmuls; with them, `TUNING.quadMinBatch` 128, they read 9–23% faster for 62–64 MiB more.) Qwen3.5 at 128 reads prompts about twice as fast as at 512 before, and its 8k chat's page comes to an estimated ~1,910 MiB in Chrome; whether that fits an iPhone's page, and how fast it reads there, `bench/mem.html` measures on the phone (see `bench/README.md`).

The most JavaScript memory a load takes is ~110 MiB: it streams the weights, a layer or a chunk of the embedding at a time. The worker claims the textures it measured from the page's memory budget (`setMemoryBudget`; before a load, an estimate: 1.4× the weight files), and claims again as the cache grows; if the GPU runs out of memory growing the cache, the reply stops with an error asking for a new session.

**Status:** early (0.x); the API may change. See the [repository README](https://github.com/daveycodez/tanstack-ai-browser#readme).
