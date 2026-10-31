# @tanstack-ai-browser/webgl

A [TanStack AI](https://tanstack.com/ai) adapter for LFM3 1.7B running in the browser on its own WebGL 2 inference engine, in a Web Worker on an OffscreenCanvas. It's for browsers without WebGPU, Firefox on Android among them: there, it reads prompts on the GPU about three times as fast as llama.cpp on the CPU.

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

`WEBGL_TEXT_MODELS` lists the models, and `WEBGL_MODEL_META` has each one's files (pinned revisions), context, capabilities, reasoning levels and the sampling from Liquid AI's card (temperature 0.1, top-k 50, repetition penalty 1.1).

| Model                      | Weights                                                   | Download |
| -------------------------- | --------------------------------------------------------- | -------- |
| `LiquidAI/LFM3-1.7B-GGUF`  | Liquid AI's QAD Q4_0 GGUF (quantization-aware trained)    | 1.05 GB  |
| `daveycodez/LFM3-1.7B-Web` | 4-bit ONNX build for the web, with a 49k-token vocabulary | 900 MB   |

Both think by default; `chat({ reasoning: "off" })` turns it off through the chat template's `enable_thinking`. **Tools**, **usage** (with `promptTokensDetails.cachedTokens`, and timings in `usage.providerUsageDetails`), `modelOptions` sampling and `chat({ outputSchema })` work as with the package's other adapters. `loadModel(model)` downloads and loads a model ahead of time, and `modelStatus(model)` is a store of its progress.

### Speed

Pixel 10 Pro (PowerVR DXT-48-1536), Firefox 159, a 515-token prompt:

|                                         | Prompt reading      | Writing       | First token |
| --------------------------------------- | ------------------- | ------------- | ----------- |
| `daveycodez/LFM3-1.7B-Web`              | 75.6 tokens/s       | 9.2 tokens/s  | 6.8 s       |
| `LiquidAI/LFM3-1.7B-GGUF`               | 71.9 tokens/s       | 7.9 tokens/s  | 7.2 s       |
| wllama (CPU, relaxed SIMD), LFM3 Q4_K_M | 24.5 to 30 tokens/s | 11.6 tokens/s | 18.2 s      |

Writing is still slower than on the CPU: each token pays a GPU round trip and about 400 small draws. The Web build writes faster than the GGUF one, its output head being 49k rows of 4 bits rather than 128k of 8.

### Correctness

Both builds were checked tensor by tensor against a reference:

- **GGUF against llama.cpp** at the commit wllama builds (`19cce69`), on a 345-token prompt (past the 256-token sliding window): mean KL 1.1e-3 between the logits, below the 3.0e-3 between llama.cpp's own CPU and Metal backends; the same top token at 31 of 32 single-token decode steps. Every layer's output has a cosine similarity of at least 0.9992. The same holds on the Pixel with mediump matmuls.
- **ONNX against onnxruntime** (CPU, exact 4-bit math): mean KL 2.2e-5, every top token the same, the greedy continuation identical over 32 tokens.

`bench/check.html` runs these checks (see `bench/README.md`); `docs/llama-cpp-lfm3.md` is LFM3's graph as llama.cpp computes it, which the engine follows.

## How it runs

`chat()` runs tools and the agent loop; each model call is one generation in the worker. The worker downloads the files to the Origin Private File System (resuming a dropped download), then streams the weights from there into textures, one layer at a time.

- **Loaders:** `loadGguf` and `loadOnnx` read either format into the same weights (`weights.ts`): 4-bit blocks of 32 in llama.cpp's Q4_0 order, zero points where the ONNX build has them, the GGUF's Q6_K embedding as 8-bit blocks. Only tensors are read from the ONNX graph; its ops aren't run.
- **Kernels:** fragment shaders, one output texel per fragment (`gpu/kernels.ts`). Matmuls read weights from byte textures and activations as packed halves, eight per texel. A one-token matmul splits its rows' blocks across fragments, so the GPU has enough work. Every op of LFM3's graph is a pass: RMSNorm, the gated short convolution with its state, QK-norm and RoPE, attention over a half-precision KV cache (sliding-window layers see their last 256 tokens), the sigmoid output gate, SwiGLU, and the sandwich norms.
- **Prompts** are read 128 tokens at a time; big passes are split into bands of rows, under mobile GPUs' watchdogs.
- **Sampling:** with top-k (LFM3's 50), the GPU applies the repetition penalty and picks the 50 chunks of 256 logits with the largest logits, which hold the top 50 tokens for sure, so writing a token reads back ~50 KB rather than every logit. Readbacks go through a pixel buffer and a fence.
- **Cache reuse:** a follow-up reads only what's new. The attention cache rewinds to any position; the convolutions' state is saved after each prompt batch and reply, to resume from.
- **Context:** 8,192 tokens.

**Status:** early (0.x); the API may change. See the [repository README](https://github.com/daveycodez/tanstack-ai-browser#readme).
