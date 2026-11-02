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

### Thinking budget

LFM3 GGUF accepts an optional thinking-token ceiling through the shared reasoning option:

```ts
const stream = chat({
  adapter: webglText("LiquidAI/LFM3-1.7B-GGUF"),
  messages,
  threadId,
  reasoning: { level: "high", budgetTokens: 256 },
  modelOptions: { max_new_tokens: 512 }
})
```

The worker counts emitted thinking content tokens, forces `</think>` at the ceiling, and continues the answer in the same generation. Opening and closing markers do not count toward the thinking budget; they do count toward `max_new_tokens`, which still limits the entire output. Leave enough total output tokens for the final answer. A zero budget closes thinking immediately. Omitting `budgetTokens` preserves unlimited thinking within the total output limit; `level: "off"` disables thinking and ignores a valid budget. Negative, fractional, or unsafe budgets are rejected. Only LFM3 GGUF currently advertises budget support.

The low-level worker/client generation option is `thinkingBudgetTokens`. Budgeted runs report `thinkingTokens`, `thinkingBudgetTokens`, and `thinkingBudgetExhausted` in `usage.providerUsageDetails`.

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
| `onnx-community/Qwen3.5-2B-ONNX` (WebGL)       | 628 tokens/s   | 74.4 tokens/s | 0.90 s      |
| `unsloth/Qwen3.5-2B-GGUF` Q4_0 (WebGL)         | 576 tokens/s   | 60.6 tokens/s | 0.99 s      |
| Transformers.js (WebGPU), the same ONNX        | 17.7 tokens/s  | 19.7 tokens/s | 30.3 s      |
| wllama (WebGPU), unsloth's Q4_K_M              | 595 tokens/s   | 20.4 tokens/s | 0.90 s      |

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
- **Prompts** are read 128 tokens at a time; big passes are split into bands of rows, under mobile GPUs' watchdogs.
- **Sampling:** with top-k (LFM3's 50), the GPU applies the repetition penalty and picks the 50 chunks of 256 logits with the largest logits, which hold the top 50 tokens for sure, so writing a token reads back ~50 KB rather than every logit. Readbacks go through a pixel buffer and a fence.
- **Qwen3.5's Gated DeltaNet** (18 of its 24 layers) is a pass each for the gates (beta, decay), the causal convolution with its state, and q and k's L2 norms, then two passes a token over each head's 128×128 fp32 state (the delta and output, then the update), and the gated RMSNorm. Prompts run the recurrence a token at a time. Its full-attention layers gate their output per dim.
- **Cache reuse:** a follow-up reads only what's new. The full-attention cache rewinds to any position; the convolutions' state, Qwen3.5's recurrent state and the sliding-window layers' last window are saved after each prompt batch and reply (the latest 8; 3 for Qwen3.5, ~20 MB each), to resume from. Qwen3.5 writes without speculative drafts (keeping only part of a draft batch would need the recurrent state after each of its tokens).
- **Context:** the model's own (LFM3: 262,144 tokens; LFM2.5: 128,000). Full-attention caches start at 1,024 tokens and double as a chat grows, wrapping into side-by-side blocks past the GPU's texture height; sliding-window layers keep only their window plus a batch, in a ring. Past 8,192 keys, attention reads them in segments and combines them.

### Memory

GPU textures at load (phone-size batches, the 1,024-token cache), and the most JavaScript memory a load takes (it streams the weights, a layer or a chunk of the embedding at a time):

| Model                           | GPU textures | Peak JavaScript memory while loading |
| ------------------------------- | ------------ | ------------------------------------ |
| `daveycodez/LFM3-1.7B-Web-ONNX` | 932 MiB      | ~110 MiB                             |
| `onnx-community/LFM3-1.7B-ONNX` | 1,025 MiB    | ~110 MiB                             |
| `LiquidAI/LFM3-1.7B-GGUF`       | 1,062 MiB    | ~110 MiB                             |

Qwen3.5 2B takes 1,269 MiB of textures (ONNX) and 1,352 MiB (GGUF) at desktop batches, more than an iPhone page has room for.

That's for iPhone Safari's ~2 GB limit on a page, which the tokenizer and the page itself also count against. The cache grows from there with the chat; if the GPU runs out of memory growing it, the reply stops with an error asking for a new session.

**Status:** early (0.x); the API may change. See the [repository README](https://github.com/daveycodez/tanstack-ai-browser#readme).
