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

### Events

```ts
webgllm.addEventListener("progress", (p) => setProgress(p.model, p.progress))
webgllm.addEventListener("loadstart", ({ model }) => setStatus(model, "loading"))
webgllm.addEventListener("load", ({ model, contextLength }) => setStatus(model, "ready"))
webgllm.addEventListener("error", ({ model, error }) => setStatus(model, "error", error.message))
webgllm.addEventListener("unload", ({ model, reason }) => setStatus(model, "idle"))
```

They fire whatever started them: `loadModel`, a completion loading its model on demand, or room made for another model. `unload`'s `reason` is `"unloaded"`, `"evicted"`, `"aborted"`, `"failed"` (after an `error`) or `"crashed"` (after an `error`).

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
  reasoning: { effort: "high", max_tokens: 512 },  // "none" turns it off; max_tokens is an optional cap
  stream: true
})
for await (const chunk of stream) {
  const { reasoning_content, content } = chunk.choices[0].delta
}
```

### Tools

OpenAI's `tools` and `tool_choice` (`"auto"`, `"none"`, `"required"` or a named function). Calls come back as `message.tool_calls` with `arguments` as a JSON string; send results back as `tool` messages. A forced call (`"required"` or a name) is constrained to the tool's JSON Schema; objects that don't set `additionalProperties` are treated as closed, as in OpenAI's strict mode, where that keeps every declared property allowed (objects under `not`, `if`, `oneOf` or `contains`, `allOf` branches, and objects with `const`, `enum` or a `$ref` beside their own keywords stay as written). Automatic calls are checked against their tools: a call to an unknown tool, arguments that fail the schema, or call markup that doesn't parse is generated again, with the model's failed attempt in the chat and the call constrained to the schema (its tokens count in `usage`). `repair_tool_calls: false` returns calls as the model wrote them.

### Structured output

```ts
const reply = await webgllm.chat.completions.create({
  model,
  messages: [{ role: "user", content: "Extract as JSON: Ada Lovelace, born 1815 in London." }],
  response_format: { type: "json_schema", json_schema: { name: "person", schema } }
})
const person = JSON.parse(reply.choices[0].message.content!)
```

Answer tokens are constrained to the schema while they're picked, so the answer is always valid against it. Objects that don't set `additionalProperties` are treated as closed, as in OpenAI's strict mode and as tools' are: a free-form key is slow to constrain ([llguidance-js#5](https://github.com/nico-martin/llguidance-js/issues/5)) and lets a model invent keys; set `additionalProperties: true` for an open object. Objects composed with `not`, `if`, `oneOf`, `contains`, `allOf`, `const`, `enum` or a `$ref` beside their own keywords stay as written. `json_object` (OpenAI's JSON mode) still takes any object. `stop` sequences don't apply inside it (the schema decides where it ends), as they don't to a forced tool call's arguments. Say in the prompt that you want JSON: small models constrained to a schema they weren't asked for write poor values. `pattern`, `format`, numeric bounds and numeric enums are checked as each value ends rather than while it's written (a limitation of the constraint library, [llguidance-js#4](https://github.com/nico-martin/llguidance-js/issues/4)): an answer that can't satisfy them fails fast with `invalid-output`.

### Images

Models with a vision encoder (`models.list()` marks them with `supports.images`) read OpenAI `image_url` parts in user messages, as http(s) or base64 data URLs: Liquid AI's LFM3-VL and LFM2.5-VL in both their GGUF and ONNX builds (such as `LiquidAI/LFM2.5-VL-450M-GGUF`), and Qwen3.5's ONNX builds (such as `onnx-community/Qwen3.5-2B-ONNX` and `onnx-community/Qwen3.5-0.8B-ONNX`):

```ts
const reply = await webgllm.chat.completions.create({
  model: "onnx-community/Qwen3.5-2B-ONNX",
  messages: [
    {
      role: "user",
      content: [
        { type: "image_url", image_url: { url: "data:image/png;base64,..." } },
        { type: "text", text: "What's in this picture?" }
      ]
    }
  ]
})
```

The image is resized the way the model's processor does it, and `detail` sets how much of it the model reads. Qwen3.5 and LFM3-VL read an image whole (in whole 32-pixel blocks), within a pixel budget: `"low"` 256 × 256, `"auto"` (the default) 640 × 640, `"high"` 1024 × 1024, which is 64, 400 or 1,024 prompt tokens for a square image. LFM2.5-VL reads an image of up to about 512 × 512 pixels whole, as 64 to 256 tokens; a larger one as its processor cuts it, a grid of 512-pixel tiles (up to 10, 256 tokens each) and the whole image again as a thumbnail (about 1,800 tokens for a photo), and `"low"` reads any image whole, as one view. The vision encoder (62 MB for Qwen3.5 0.8B, 197 MB for 2B; 103 to 583 MB for Liquid AI's) downloads the first time a request has an image, loads on the GPU for the request's new images and is freed after them, so text chats never pay for it in memory or speed. Each image's embedding rows are kept for a few recent images, and the prompt cache tells images apart by content, so a follow-up about an image in the history reads only the new turn. `timings.images_ms` says how long a request's images took (fetching, resizing and encoding, the encoder's load included), before `prompt_ms`. On a GPU whose textures are shorter than an image's count of 16-pixel patches (4,096 at `"high"`), the image is made smaller to fit.

### Several models

Several models can be loaded at once, one worker each, up to a memory budget (`new WebGLLM({ memoryBudget })`, picked from the device by default; a [worker of your own](#a-worker-of-your-own) holds one model). Loading past it unloads the least recently used model, unless it was loaded with `keepLoaded: true`. Generations run one at a time across models.

```ts
await webgllm.models.list()      // the catalog, downloaded models and models used on the page
await webgllm.models.downloaded(id) // whether it loads without a download (storage only, no network)
await webgllm.models.delete(id)  // remove a model's files from storage
await webgllm.unloadModel(id)    // free its GPU memory (no argument: all)
```

### A worker of your own

webgllm starts a Web Worker for each model it loads, with `new Worker(new URL("./worker.js", import.meta.url))`, which Vite and webpack 5 bundle for you. To run models in a worker you start (bundled your way, or shared with your own code), write the worker with the package's handler and give it to `WebGLLM`:

```ts
// worker.ts
import { WebGLLMWorkerHandler } from "webgllm"

const handler = new WebGLLMWorkerHandler()
self.onmessage = (event) => handler.onmessage(event)
```

```ts
const worker = new Worker(new URL("./worker.ts", import.meta.url), { type: "module" })
const webgllm = new WebGLLM({ worker })
```

The worker holds one model: loading another replaces the one there once its running generation ends, whatever `keepLoaded` says (the replaced model's `unload` event has the reason `"evicted"`). `unloadModel` stops the model's generation and frees its GPU memory, and leaves the worker running; webgllm never ends a worker it didn't start. It resolves, after the model's `unload` event, once the worker has freed the memory. Completions take turns there, each from its load to its last token, so two for different models don't replace each other midway. Downloads run in the same worker, beside the loaded model. Every message webgllm sends or takes carries its own mark (`webgllm: true`), so your own messages can share the worker whatever their `type`. For several models loaded at once, make a `WebGLLM` for each worker, or leave `worker` out.

If the worker fails, what waits on it (a load, a completion, a token forward, a download, an unload) rejects with a `WebGLLMError` whose code is `worker-failed`, and the model's `error` and `unload` (`"crashed"`) events fire. A worker whose script failed to load never runs, so every later request rejects the same way, at once; a worker whose own code threw is still running, and the next request loads the model again. webgllm hears a failure only from `new WebGLLM({ worker })` on, so make the `WebGLLM` right after the worker: one that failed to load earlier looks like one that hasn't answered yet, and requests to it wait.

### Sharing memory with other runtimes

An app that also runs models in other runtimes (Transformers.js, wllama) can count webgllm's GPU memory in its own budget with `onMemoryClaim`. Each time a model's worker claims memory (an estimate before loading, the measured textures after, and each time its attention cache grows), the hook gets the bytes the model will hold in all, replacing its previous claim. The worker allocates only after the hook's promise resolves, so the app can make room first, for example by unloading another runtime's model (or another webgllm model with `unloadModel`). Rejecting fails the load, or the reply whose cache grew, with an `out-of-memory` error. `signal` aborts if the model is unloaded meanwhile. The model's memory is free again at its `unload` event.

```ts
const webgllm = new WebGLLM({
  memoryBudget: registry.limit, // the same limit: webgllm still orders and evicts its own models
  onMemoryClaim: ({ model, bytes, signal }) => registry.claim(`webgllm:${model}`, bytes, { signal })
})
webgllm.addEventListener("unload", ({ model }) => registry.release(`webgllm:${model}`))
```

### Tokens and logits

What OpenAI's API, WebLLM and wllama offer below the chat level:

```ts
// Each written token's log probability, with the 3 likeliest at its position.
const completion = await webgllm.chat.completions.create({ model, messages, logprobs: true, top_logprobs: 3 })
completion.choices[0].logprobs.content // [{ token, logprob, bytes, top_logprobs }, …]

// Token ids to a bias from -100 to 100, added to their logits before sampling.
await webgllm.chat.completions.create({ model, messages, logit_bias: { "5242": -100 } })

// A raw prompt, read as it is with no chat template.
const text = await webgllm.completions.create({ model, prompt: "The capital of France is", max_tokens: 8 })
text.choices[0].text
```

`logprobs` covers every token written, the reply's thinking and tool-call tokens included, and none of the text a `stop` sequence cut: a token the stop sequence cuts into is left out whole, so the last of the reply's text can have no entry; they are the model's probabilities before temperature, top-k and the penalties (after a `logit_bias`). A token's `bytes` are its own, so the tokens of a character split across them add up to it. `completions.create` takes a prompt that isn't empty, the sampling fields of chat completions, `echo`, `stop`, `seed`, `logprobs` with `top_logprobs` (a boolean and a count, as WebLLM has them), `logit_bias` and `stream`.

As in WebLLM, token ids go in with `forwardTokensAndSample`, and every logit comes out through a `LogitProcessor` on the worker's handler:

```ts
// worker.ts
import { WebGLLMWorkerHandler, type LogitProcessor } from "webgllm"

const processor: LogitProcessor = {
  processLogits: (logits) => logits,   // every logit after each forward pass; return the ones to sample from
  processSampledToken: (token) => {},  // each token chosen
  resetState: () => {}                 // resetChat() was called
}
const handler = new WebGLLMWorkerHandler()
handler.setLogitProcessorRegistry(new Map([["LiquidAI/LFM2.5-350M-GGUF", processor]]))
self.onmessage = (event) => handler.onmessage(event)
```

```ts
const webgllm = new WebGLLM({ worker })
const next = await webgllm.forwardTokensAndSample(inputIds, true) // reads the ids after what the cache holds; the next token
await webgllm.forwardTokensAndSample([next], false)                // pass it back in to go on
await webgllm.resetChat()                                          // forget the cache
```

`forwardTokensAndSample(inputIds, isPrefill, modelId?)` samples with the model card's settings (its repetition penalty counts the tokens the cache holds), from the processor's logits when the model has one; `modelId` is needed only when several models are loaded. A model's processor also shapes its chat and text completions. webgllm tokenizes inside its worker and exports no tokenizer, like WebLLM: token ids come from the model's `tokenizer.json`, with a tokenizer of your own.

The GPU normally picks each token itself and reads back only the candidates. `logprobs`, `logit_bias` and a `LogitProcessor` read every logit back for each token instead, so those replies are written slower.

### Other request fields

`max_tokens` (or `max_completion_tokens`), `temperature`, `top_p`, `presence_penalty`, `stop`, `seed`, plus `top_k`, `min_p`, `repetition_penalty` and `cache_prompt`. Unset sampling fields take the model card's values. Every response has OpenAI's `usage` (with cached prompt tokens and reasoning tokens) and a `timings` object (first-token time, prompt and decode speed). Fields the engine can't honour (`n > 1`, `frequency_penalty`, audio content, and images for a model without a vision encoder) throw a `WebGLLMError` with code `"unsupported-parameter"` rather than being ignored.

### Errors

`WebGLLMError` has a `code`: `webgl2-unavailable`, `model-not-supported`, `model-not-loaded`, `download-failed`, `storage-full`, `out-of-memory`, `gpu-context-lost`, `context-length-exceeded`, `invalid-schema`, `invalid-output`, `unsupported-parameter` or `worker-failed`.

## Models

Any Hugging Face model whose architecture webgllm runs loads by its repository name (`owner/name`): LFM2 and LFM2.5, LFM3 and Qwen3.5, as a GGUF or an ONNX export. A model hosted elsewhere loads by the URL of its `.gguf` file or `.onnx` graph, with its `tokenizer.json` and `tokenizer_config.json` (and `chat_template.jinja`, `generation_config.json`, and for ONNX `config.json` and the graph's data file, where there are) in the same folder.

```ts
await webgllm.chat.completions.create({ model: "LiquidAI/LFM2-350M-GGUF", messages })
```

webgllm reads the repository's file list from the Hub and picks, by one rule each:

- **Weights:** a GGUF file (Q4_0 first, a quantization-aware "QAD" Q4_0 before a plain one, then Q8_0, then the fewest bits among Q4_1, Q5_K, Q6_K and F16), else an ONNX graph under `onnx/` that the loader takes, checked by reading the graph before its weights download (4-bit `MatMulNBits` in blocks of 32, with 8-bit ones where the export keeps a matrix at 8 bits; its weights in the external data files the graph names; q4f16 tried first, then q4).
- **Tokenizer and chat template:** the repository's own, else those in a folder named in the weights file's name (Liquid AI's `qad/`), else the base model's (the card's `base_model`).
- **Thinking and tool calls:** from the chat template (`enable_thinking` makes thinking switchable; a template that opens `<think>`, or a model named "Thinking" or "Reasoning", always thinks). `<tool_call>` means the XML call format, otherwise LFM2's Pythonic calls.
- **Sampling:** the repository's `generation_config.json` as Transformers reads it: its values when it sets `do_sample`, else greedy decoding.
- **Images:** an ONNX build reads images when its `config.json` has a `vision_config` of a type the vision encoder runs (Qwen3.5's, SigLIP2 for LFM2.5-VL, LFM3-VL's), and beside the decoder graph is a `vision_encoder` (or `embed_images`) graph the encoder loads, with its data file and an image processor config (`preprocessor_config.json`, or `processor_config.json`): the 8-bit one first (`vision_encoder_q8.onnx`), then the one with the decoder's suffix (`vision_encoder_q4f16.onnx` for `decoder_model_merged_q4f16.onnx`), then the 4-bit ones. A GGUF build reads images when its repository has a vision projector (`mmproj-*.gguf`: Q8_0 first, then a float one) of a type the encoder runs (LFM2.5-VL's `lfm2`, LFM3-VL's `lfm3vl`); its image processor config comes from the repository its tokenizer does.

Another architecture, or a repository without weights it can load, fails with `model-not-supported` before anything downloads. A downloaded model keeps what was resolved beside its files, so after a reload it loads (and `models.downloaded` answers) with no request to the Hub, offline too.

The catalog holds known defaults for the models webgllm is tuned and tested on, such as the sampling from their model cards, which the Hub doesn't say. `models.list()` returns it, plus any other model downloaded or used on the page:

| Model                                | Weights                                                  |
| ------------------------------------ | -------------------------------------------------------- |
| `LiquidAI/LFM3-1.7B-GGUF`            | Liquid AI's QAD Q4_0 GGUF (quantization-aware trained)   |
| `onnx-community/LFM3-1.7B-ONNX`      | onnx-community's q4f16 export                            |
| `unsloth/Qwen3.5-2B-GGUF`            | unsloth's Q4_0 GGUF                                      |
| `onnx-community/Qwen3.5-2B-ONNX`     | onnx-community's q4f16 decoder and vision encoder (reads images) |
| `LiquidAI/LFM2.5-1.2B-Instruct-GGUF` | official QAD Q4_0                                        |
| `LiquidAI/LFM2.5-1.2B-Thinking-GGUF` | official Q4_0; always thinks                             |
| `LiquidAI/LFM2.5-350M-GGUF`          | official QAD Q4_0                                        |
| `LiquidAI/LFM2.5-1.2B-Instruct-ONNX`, `LiquidAI/LFM2.5-1.2B-Thinking-ONNX`, `LiquidAI/LFM2.5-350M-ONNX` | Liquid AI's 4-bit ONNX exports |
| `LiquidAI/LFM2-1.2B-Tool-GGUF`       | official Q4_0; tuned for tool calls, greedy decoding     |
| `onnx-community/LFM2-1.2B-Tool-ONNX` | onnx-community's q4f16 export                            |
| `LiquidAI/LFM3-VL-2B-GGUF`, `LiquidAI/LFM2.5-VL-3B-GGUF`, `LiquidAI/LFM2.5-VL-1.6B-GGUF`, `LiquidAI/LFM2.5-VL-450M-GGUF` | official Q4_0 (QAD for LFM3-VL) and Q8_0 vision projector (read images) |
| `LiquidAI/LFM3-VL-2B-ONNX`, `LiquidAI/LFM2.5-VL-3B-ONNX`, `LiquidAI/LFM2.5-VL-1.6B-ONNX`, `LiquidAI/LFM2.5-VL-450M-ONNX` | Liquid AI's 4-bit decoders and 8-bit vision encoders (read images) |

LFM3 thinks by default and Qwen3.5 doesn't; `reasoning.effort` (or OpenAI's `reasoning_effort`) switches it through the chat template's `enable_thinking`. Sampling defaults come from each model card. See [LFM2.5 GGUF validation](bench/LFM2-GGUF.md).

## How it runs

Each loaded model runs in its own Web Worker on an OffscreenCanvas, and each `chat.completions.create` call is one generation there. The worker downloads the files to the Origin Private File System (resuming a dropped download), then streams the weights from there into textures, one layer at a time.

- **Loaders:** `loadGguf` and `loadOnnx` read either format into the same weights (`weights.ts`): 4-bit blocks of 32 in llama.cpp's Q4_0 order, with zero points where the ONNX build has them. The GGUF's Q6_K embedding (also the output head) is kept as it is and read natively in the shader, so its weights are exactly llama.cpp's. The embedding goes to the GPU in chunks of rows, so it's never all in JavaScript memory. An ONNX export's float embedding, and a matrix it keeps at 8 bits with zero points (LFM3-VL's), are stored at 8 bits per block of 32. Only tensors are read from the ONNX graph; its ops aren't run.
- **Kernels:** fragment shaders, one output texel per fragment (`gpu/kernels.ts`). Matmuls read weights from byte textures and activations as packed halves, eight per texel. A one-token matmul splits its rows' blocks across fragments, so the GPU has enough work. Every op of LFM3's graph is a pass: RMSNorm, the gated short convolution with its state, QK-norm and RoPE, attention over a half-precision KV cache (sliding-window layers see their last 256 tokens), the sigmoid output gate, SwiGLU, and the sandwich norms.
- **Prompts** are read 512 tokens at a time on every device; big passes are split into bands of rows, under mobile GPUs' watchdogs.
- **Sampling:** with top-k (LFM3's 50), the GPU applies the repetition penalty and picks the 50 chunks of 256 logits with the largest logits, which hold the top 50 tokens for sure, so writing a token reads back ~50 KB rather than every logit. Readbacks go through a pixel buffer and a fence.
- **Qwen3.5's Gated DeltaNet** (18 of its 24 layers) is a pass each for the gates (beta, decay), the causal convolution with its state, and q and k's L2 norms, then the recurrence over each head's 128×128 fp32 state, and the gated RMSNorm. Writing runs the recurrence as two passes a token (the delta and output, then the update). Prompt batches run it in chunks of 32 tokens, as llama.cpp and flash-linear-attention do (`delta-rule.ts` has the math): the decays summed within a chunk, the chunk's Gram matrices, a triangular solve and the products that don't read the state, for every chunk at once, then two passes a chunk that do (its deltas and outputs, then the state after it). Its full-attention layers gate their output per dim.
- **Cache reuse:** a follow-up reads only what's new. The full-attention cache rewinds to any position; the convolutions' state, Qwen3.5's recurrent state and the sliding-window layers' last window are saved (snapshots) to resume from: LFM3's after each prompt batch and reply (the latest 8, ~3 MB each), Qwen3.5's (~20 MB each) after the reply and at the anchor, where the next question's prompt stops sharing the chat, 2 in all. Qwen3.5's template writes a past reply back without the `<think>` tags its generation prompt opened, so a follow-up differs from `<|im_start|>assistant` on; the anchor is saved there, from inside the prompt's last batch (its recurrence chunks laid out so one ends there), and a follow-up reads only the reply and the new message. `src/prompt.ts` finds where a template's follow-ups stop sharing, once per template (LFM3's write the reply back as it was: the anchor is the prompt's end). A tool's result keeps the last question's anchor, the reply before it being written back as it was. Qwen3.5 writes without speculative drafts (keeping only part of a draft batch would need the recurrent state after each of its tokens).
- **Images:** the vision encoder (`vision.ts`, `gpu/vision-kernels.ts`) runs once per image view, on plain kernels of its own: f32 activations, 4- and 8-bit matmuls, LayerNorm and RMS norm, 2D RoPE, attention in three passes and a 3 × 3 depthwise convolution. It runs Qwen3.5's ViT, SigLIP2 with LFM2.5-VL's projector and LFM3-VL's conv and attention tower, loaded from ONNX graphs or GGUF projectors. Its output rows are uploaded over the embedding pass's output for the image's tokens, which take Qwen's 3D rotary positions (Qwen3.5) or their own places in the sequence (LFM); text batches run the same programs as before. [docs/vision.md](docs/vision.md) has the details and the checks.
- **Context:** the model's own (LFM3: 262,144 tokens; LFM2.5: 128,000). Full-attention caches start at 1,024 tokens and grow 512 at a time as a chat grows (doubling left up to half of it unused), wrapping into side-by-side blocks past the GPU's texture height; sliding-window layers keep only their window plus a batch, in a ring. Past 8,192 keys, attention reads them in segments and combines them.

Memory and speed measurements are in [bench/WEBGL-PERFORMANCE.md](bench/WEBGL-PERFORMANCE.md) and the bench pages ([bench/README.md](bench/README.md)).

## License

Apache 2.0
