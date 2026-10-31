# Bench

Pages for checking and timing the engine, served cross-origin isolated by `server.mjs`, which also bundles the package's TypeScript on request (`/bundle/<path>.js` is `<path>.ts`).

```sh
MODELS_DIR=… REF_DIR=… REF_ONNX_DIR=… ONNX_DIR=… HF_DIR=… node packages/webgl/bench/server.mjs   # :8787
adb reverse tcp:8787 tcp:8787        # a phone over USB: localhost is a secure context
```

- `MODELS_DIR`: `LFM3-1.7B-QAD-Q4_0.gguf` (and `LFM3-1.7B-Q4_K_M.gguf` for the CPU baseline)
- `ONNX_DIR`: a copy of `daveycodez/LFM3-1.7B-Web` (config, tokenizer, `onnx/model_q4f16.onnx*`)
- `HF_DIR`: `LiquidAI/LFM3-1.7B`'s `tokenizer.json`, `tokenizer_config.json`, `chat_template.jinja`
- `REF_DIR`, `REF_ONNX_DIR`: reference tensors from llama.cpp and onnxruntime (`manifest.json`, `tokens.json`, `logits.bin`, `gen_logits.bin`, `l_out-N.bin`, …)

Results print in the server's terminal.

## check.html

- `?format=gguf|onnx&batch=128&precision=highp|mediump&layers=1`: runs the reference prompt and compares every layer and the logits with the reference, then the reference's greedy continuation one token at a time (the decode path, and the GPU's candidate pick against the full logits), then times the prompt.
- `?smoke=1&len=512&format=…&q=…`: a chat prompt padded to about `len` tokens, read and answered as the worker does it (no reference needed); prints the reply and the speed.
- `&skip=matmul,norm,…`: times the prompt and single-token steps with each kind of pass left out, to see where the time goes.

## index.html: the go/no-go benchmark

The first step, before the engine: one Q4_0 matmul kernel on every LFM3 projection shape at 128 tokens (`?run=gpu`), against wllama's relaxed-SIMD build reading the same number of tokens (`?run=cpu`). On a Pixel 10 Pro in Firefox, LFM3's matmuls for 128 tokens took 1.49 s (86 tokens/s) against 24.5 prompt tokens/s for wllama's whole model, so the engine was built.

Pitfalls it found: PowerVR merges and skips full-screen draws whose output is overwritten later, even across framebuffer switches, so a timed draw must read the previous draw's output; `unpackUnorm2x16` returns wrong values on that driver; a readback costs 10 to 35 ms of fixed latency there.
