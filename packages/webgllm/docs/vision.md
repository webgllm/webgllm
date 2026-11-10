# Image input (Qwen3.5)

How webgllm reads images, starting with Qwen3.5's ONNX builds
(`onnx-community/Qwen3.5-0.8B-ONNX`, `onnx-community/Qwen3.5-2B-ONNX`).

## The model's parts

`onnx/vision_encoder_q4f16.onnx` (62 MB for 0.8B, 197 MB for 2B) takes
`pixel_values` (patches × 1536) and `image_grid_thw`, and returns
`image_features` (merged tokens × the text model's hidden size). Read from
the graph:

- Patch embedding: a MatMulNBits (K = 3 · 2 · 16 · 16 = 1536) plus bias.
- Position embedding: a 48 × 48 learned table (4-bit GatherBlockQuantized),
  bilinearly interpolated to the image's patch grid, added.
- `depth` blocks (0.8B: 12 × 768 wide, 12 heads; 2B: 24 × 1024, 16 heads;
  head dim 64): LayerNorm → qkv (+bias) → 2D RoPE on q and k (dims 0..15 of
  each half by the patch's row, 16..31 by its column, rotate-half over the
  64 dims) → bidirectional attention over the image's patches → proj
  (+bias) → residual; LayerNorm → fc1 (+bias) → tanh GELU → fc2 (+bias) →
  residual.
- Merger: LayerNorm, each 4 consecutive patches (a 2 × 2 block: the
  preprocessor orders patches so) as one 4·width row, linear1 (+bias), exact
  GELU, linear2 (+bias) → the text model's hidden size.

The text model takes the features in place of the embedding rows of the
`<|image_pad|>` tokens. Positions are Qwen2-VL's 3D RoPE (`mrope_section`
[11, 11, 10], interleaved): rotary frequency j uses the token's t, h or w
position for j mod 3 = 0, 1, 2. A text token's three positions are equal.
An image's tokens (merged grid gh × gw, row-major) at text position s get
t = s, h = s + row, w = s + column, and the text after it continues at
s + max(gh, gw). So past an image a token's rotary position is its cache
index plus a delta (max(gh, gw) − gh · gw per image before it).

## How it runs

1. **The encoder on WebGL 2 (`vision.ts`, `gpu/vision-kernels.ts`).** It runs
   once per image, so it has plain kernels of its own rather than the text
   path's tuned ones (whose buffers, splits and layouts are sized for the
   text model's batches): f32 activations (4 values a texel); a 4-bit matmul
   that reads MatMulNBits rows as stored (one RGBA32UI texel per row and
   block, scales and zero points per 4 rows), with the bias, GELU (tanh or
   exact) and residual add in its output; LayerNorm; a RoPE pass; and
   attention in three passes (scores, each row's max and sum, the weighted
   values) over chunks of query rows whose scores fit 32 MB. It shares the
   engine's `Gpu` and its paced draws (`runAsync`), so the page keeps
   painting. The interpolated position embedding and the RoPE angles are
   computed on the CPU (a few MB). `load-vision.ts` finds the weights by
   the graph's structure, as `load-onnx.ts` does the text model's.
2. **Loaded only for images.** The first request with an image downloads
   the encoder (and `preprocessor_config.json`); a request whose prompt
   reads an image's rows loads it, encodes the images it hasn't encoded
   before, reads the rows back (tokens × hidden floats) and frees the
   encoder. Text chats never touch any of it. The rows are kept for 8
   recent images per model.
3. **Into the prompt.** The chat template writes `<|vision_start|>
   <|image_pad|><|vision_end|>` per image part; the worker expands each
   `<|image_pad|>` to the image's gh · gw tokens. `Engine.setImages` takes
   the prompt's images (start, grid, rows). A batch with image rows (a)
   uploads their rows over the embedding pass's output (no new pass), and
   (b) runs the q/k head pass's 3D variant, which reads each row's (t, h, w)
   from a small texture. Text batches run the same programs as before (their
   shader source is byte for byte the same); only the head pass's `uPos0`
   uniform becomes the cache position plus the images' shift (0 without
   images).
4. **The prompt cache** compares keys: token ids, and for an image's rows a
   key hashed from its resized pixels and size, so another image in the same
   place doesn't match, and a follow-up about the same image reads only the
   new turn.
5. **Catalog.** One rule from the model's files (`visionFilesFor` in
   `resolve-model.ts`): an ONNX build whose config has a `vision_config` of a
   type the encoder runs, with `vision_encoder<the decoder's suffix>.onnx`
   beside the decoder (a graph `load-vision.ts` takes, one data file) and a
   `preprocessor_config.json`, declares image input. The catalog's Qwen3.5
   2B ONNX entry lists those files, and a test checks the rule agrees.
6. **Image size.** Qwen's processor rounds the image to whole 32-pixel
   blocks between `min_pixels` and `max_pixels` (its own maximum is 16.7 M
   pixels, 16k tokens). webgllm caps it by OpenAI's `detail`: `low`
   256 × 256 (64 tokens), `auto` 640 × 640 (400), `high` 1024 × 1024
   (1,024), and by the GPU's texture height (one texel row a patch).

## Checks (M3 Ultra, the desktop app's browser)

- **Encoder against onnxruntime** (`bench/vision-check.html`, the same
  patches through onnxruntime-node): against the float32-arithmetic `q4`
  builds (the same 4-bit weights), relative error 0.21% (0.8B, 640 × 384),
  0.27% (0.8B, 1024 × 1024) and 0.39% (2B, 640 × 384), worst token cosine
  0.9991 to 0.9996. onnxruntime's own `q4f16` build is 0.9% to 1.9% from
  those references (it computes in fp16).
- **Answers through `chat.completions.create`** (`bench/vision.html`): a
  red circle on white ("Circle, red" on 0.8B, "circle; red" on 2B), a blue
  square, a follow-up about the background ("white"), two images ("first
  red, second blue" on 2B), and a text question after them ("Paris"). On
  the same PNGs and prompts as Transformers.js 4.3.2 (onnxruntime-node
  CPU): the same prompt token counts, the same answers to the shape and
  left/right questions, and a long description that matches for its first
  10 tokens. (Transformers.js 4.3.2 can't take two images in one prompt: it
  reads them as frames of one video.)
- **Encoding time** (warm): 0.8B 55 ms at 256 × 256, about 0.2 s at
  640 × 384, 2.9 s at 1024 × 1024; 2B about 0.4 s at 640 × 384. Loading the
  encoder from storage: 0.1 to 0.2 s.
- **Text speed and memory unchanged** (`webgl-perf.html?worker=`, the two
  builds alternated, each tuning from scratch, layout fixed): Qwen3.5 2B
  prompt 1904 / 1983 tok/s before, 1960 / 1998 after, writing 90.1 / 90.8
  before, 90.3 / 90.6 after; 0.8B prompt 4045 / 4180 before, 3896 / 4103 after,
  writing 74.9 / 77.3 before, 76.8 / 76.7 after; painted frames the same.
  `mem.html` over an 8k chat: the same textures at every checkpoint (0.8B:
  866.0 MiB loaded to 1016.4 MiB at 8,120 tokens; nothing is allocated for
  images until a prompt has one).

## Left

- The encoder's matmul and attention are plain (one patch per fragment):
  1024 × 1024 takes seconds on a Mac, more on a phone. Tiling (several
  patches per fragment, as the text path's 4-token matmuls) is the next
  step if `high` matters.
- GGUF vision projectors (`mmproj`, unsloth's Qwen3.5 GGUFs) aren't read.
- Video parts.
- iPhone memory with the 2B encoder loaded (its 187 MiB of textures, freed
  after each request's images) hasn't been measured on the device.
