# webgllm

LLM inference in the browser on WebGL 2, with an OpenAI-compatible API. It runs LFM3, LFM2.5 and Qwen3.5 on its own WebGL 2 engine in a Web Worker, so it works where WebGPU doesn't: Firefox on Android, older iPhones, and everywhere else.

```sh
npm install webgllm
```

```ts
import { WebGLLM } from "webgllm"

const webgllm = new WebGLLM()

const completion = await webgllm.chat.completions.create({
  model: "LiquidAI/LFM3-1.7B-GGUF",
  messages: [{ role: "user", content: "What is the capital of France?" }]
})

console.log(completion.choices[0].message.content)
```

The first call downloads the model into the browser's storage (the Origin Private File System) and loads it; later calls and page loads reuse it. Pages don't need cross-origin isolation.

**Status:** early (0.x); the API may change. Docs: [webgllm.dev](https://webgllm.dev).

## Requirements

WebGL 2 with float render targets (`EXT_color_buffer_float`) and OffscreenCanvas in workers, which current Chrome, Firefox and Safari have. `WebGLLM.isSupported()` checks.

## Usage

### Loading with progress

```ts
await webgllm.loadModel("unsloth/Qwen3.5-2B-GGUF", {
  contextLength: 8192,
  onProgress: ({ phase, progress }) => console.log(phase, Math.round(progress * 100))
})
```

`loadModel` is optional: `create()` loads its model on demand, and takes `{ onProgress }` in its second argument for that case. `downloadModel(model)` fetches the files without loading them.

### Streaming and cancelling

```ts
const controller = new AbortController()
const stream = await webgllm.chat.completions.create(
  { model, messages, stream: true, stream_options: { include_usage: true } },
  { signal: controller.signal }
)
for await (const chunk of stream) {
  output.textContent += chunk.choices[0]?.delta.content ?? ""
}
```

Aborting rejects with a `DOMException` named `AbortError` and stops the generation.

### Thinking

```ts
const stream = await webgllm.chat.completions.create({
  model: "unsloth/Qwen3.5-2B-GGUF",
  messages,
  reasoning_effort: "high",        // "none" turns it off
  thinking_budget_tokens: 512,     // optional cap
  stream: true
})
for await (const chunk of stream) {
  const { reasoning_content, content } = chunk.choices[0].delta
}
```

### Tools

OpenAI's `tools` and `tool_choice` (`"auto"`, `"none"`, `"required"` or a named function). Calls come back as `message.tool_calls` with `arguments` as a JSON string; send results back as `tool` messages. A forced call (`"required"` or a name) is constrained to the tool's JSON Schema; objects that don't set `additionalProperties` are treated as closed, as in OpenAI's strict mode.

### Structured output

```ts
const reply = await webgllm.chat.completions.create({
  model,
  messages: [{ role: "user", content: "Extract as JSON: Ada Lovelace, born 1815 in London." }],
  response_format: { type: "json_schema", json_schema: { name: "person", schema } }
})
const person = JSON.parse(reply.choices[0].message.content!)
```

Answer tokens are constrained to the schema while they're picked, so the answer is always valid against it. Say in the prompt that you want JSON: small models constrained to a schema they weren't asked for write poor values.

### Several models

Several models can be loaded at once, one worker each, up to a memory budget (`new WebGLLM({ memoryBudget })`, picked from the device by default). Loading past it unloads the least recently used model, unless it was loaded with `keepLoaded: true`. Generations run one at a time across models.

```ts
await webgllm.models.list()      // the supported models, with downloaded / loaded
await webgllm.models.delete(id)  // remove a model's files from storage
await webgllm.unloadModel(id)    // free its GPU memory (no argument: all)
```

### Other request fields

`max_tokens` (or `max_completion_tokens`), `temperature`, `top_p`, `presence_penalty`, `stop`, `seed`, plus `top_k`, `min_p`, `repetition_penalty` and `cache_prompt`. Unset sampling fields take the model card's values. Every response has OpenAI's `usage` (with cached prompt tokens and reasoning tokens) and a `timings` object (first-token time, prompt and decode speed). Fields the engine can't honour (`n > 1`, `logprobs`, `logit_bias`, `frequency_penalty`, image or audio content) throw a `WebGLLMError` with code `"unsupported-parameter"` rather than being ignored.

### Errors

`WebGLLMError` has a `code`: `webgl2-unavailable`, `model-not-supported`, `model-not-loaded`, `download-failed`, `storage-full`, `out-of-memory`, `gpu-context-lost`, `context-length-exceeded`, `invalid-schema` or `unsupported-parameter`.

## Models

| Model                                | Weights                                                  |
| ------------------------------------ | -------------------------------------------------------- |
| `LiquidAI/LFM3-1.7B-GGUF`            | Liquid AI's QAD Q4_0 GGUF (quantization-aware trained)   |
| `onnx-community/LFM3-1.7B-ONNX`      | onnx-community's q4f16 export                            |
| `unsloth/Qwen3.5-2B-GGUF`            | unsloth's Q4_0 GGUF                                      |
| `onnx-community/Qwen3.5-2B-ONNX`     | onnx-community's q4f16 decoder                           |
| `LiquidAI/LFM2.5-1.2B-Instruct-GGUF` | official QAD Q4_0                                        |
| `LiquidAI/LFM2.5-1.2B-Thinking-GGUF` | official Q4_0; always thinks                             |
| `LiquidAI/LFM2.5-350M-GGUF`          | official QAD Q4_0                                        |
| `LiquidAI/LFM2.5-1.2B-Instruct-ONNX`, `LiquidAI/LFM2.5-1.2B-Thinking-ONNX`, `LiquidAI/LFM2.5-350M-ONNX` | Liquid AI's 4-bit ONNX exports |

LFM3 thinks by default and Qwen3.5 doesn't; `reasoning_effort` switches it through the chat template's `enable_thinking`. Sampling defaults come from each model card. See [LFM2.5 GGUF validation](bench/LFM2-GGUF.md).

## Advanced

`webgllm/advanced` exports the engine underneath (`Engine`, `loadGguf`, `loadOnnx`, `loadTokenizer`, byte sources, `setPromptBatch`) for experiments. It's not covered by semver while the package is 0.x.

## How it runs

Each loaded model runs in its own Web Worker on an OffscreenCanvas, and each `chat.completions.create` call is one generation there. The worker downloads the files to the Origin Private File System (resuming a dropped download), then streams the weights from there into textures, one layer at a time.

- **Loaders:** `loadGguf` and `loadOnnx` read either format into the same weights (`weights.ts`): 4-bit blocks of 32 in llama.cpp's Q4_0 order, with zero points where the ONNX build has them. The GGUF's Q6_K embedding (also the output head) is kept as it is and read natively in the shader, so its weights are exactly llama.cpp's. The embedding goes to the GPU in chunks of rows, so it's never all in JavaScript memory. Only tensors are read from the ONNX graph; its ops aren't run.
- **Kernels:** fragment shaders, one output texel per fragment (`gpu/kernels.ts`). Matmuls read weights from byte textures and activations as packed halves, eight per texel. A one-token matmul splits its rows' blocks across fragments, so the GPU has enough work. Every op of LFM3's graph is a pass: RMSNorm, the gated short convolution with its state, QK-norm and RoPE, attention over a half-precision KV cache (sliding-window layers see their last 256 tokens), the sigmoid output gate, SwiGLU, and the sandwich norms.
- **Prompts** are read 512 tokens at a time on every device (`setPromptBatch` in `webgllm/advanced` for fewer); big passes are split into bands of rows, under mobile GPUs' watchdogs.
- **Sampling:** with top-k (LFM3's 50), the GPU applies the repetition penalty and picks the 50 chunks of 256 logits with the largest logits, which hold the top 50 tokens for sure, so writing a token reads back ~50 KB rather than every logit. Readbacks go through a pixel buffer and a fence.
- **Qwen3.5's Gated DeltaNet** (18 of its 24 layers) is a pass each for the gates (beta, decay), the causal convolution with its state, and q and k's L2 norms, then the recurrence over each head's 128×128 fp32 state, and the gated RMSNorm. Writing runs the recurrence as two passes a token (the delta and output, then the update). Prompt batches run it in chunks of 32 tokens, as llama.cpp and flash-linear-attention do (`delta-rule.ts` has the math): the decays summed within a chunk, the chunk's Gram matrices, a triangular solve and the products that don't read the state, for every chunk at once, then two passes a chunk that do (its deltas and outputs, then the state after it). Its full-attention layers gate their output per dim.
- **Cache reuse:** a follow-up reads only what's new. The full-attention cache rewinds to any position; the convolutions' state, Qwen3.5's recurrent state and the sliding-window layers' last window are saved (snapshots) to resume from: LFM3's after each prompt batch and reply (the latest 8, ~3 MB each), Qwen3.5's (~20 MB each) after the reply and at the anchor, where the next question's prompt stops sharing the chat, 2 in all. Qwen3.5's template writes a past reply back without the `<think>` tags its generation prompt opened, so a follow-up differs from `<|im_start|>assistant` on; the anchor is saved there, from inside the prompt's last batch (its recurrence chunks laid out so one ends there), and a follow-up reads only the reply and the new message. `src/prompt.ts` finds where a template's follow-ups stop sharing, once per template (LFM3's write the reply back as it was: the anchor is the prompt's end). A tool's result keeps the last question's anchor, the reply before it being written back as it was. Qwen3.5 writes without speculative drafts (keeping only part of a draft batch would need the recurrent state after each of its tokens).
- **Context:** the model's own (LFM3: 262,144 tokens; LFM2.5: 128,000). Full-attention caches start at 1,024 tokens and grow 512 at a time as a chat grows (doubling left up to half of it unused), wrapping into side-by-side blocks past the GPU's texture height; sliding-window layers keep only their window plus a batch, in a ring. Past 8,192 keys, attention reads them in segments and combines them.

Memory and speed measurements are in [bench/WEBGL-PERFORMANCE.md](bench/WEBGL-PERFORMANCE.md) and the bench pages ([bench/README.md](bench/README.md)).

## License

Apache 2.0
