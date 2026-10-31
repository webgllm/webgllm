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
