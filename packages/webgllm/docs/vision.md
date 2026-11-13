# Image input

How webgllm reads images. It started with Qwen3.5's ONNX builds
(`onnx-community/Qwen3.5-0.8B-ONNX`, `onnx-community/Qwen3.5-2B-ONNX`),
described first; Liquid AI's vision-language models (LFM2.5-VL 450M, 1.6B
and 3B, and LFM3-VL 2B, each in its GGUF and its ONNX build) run on the
same encoder code and are described [after](#liquid-ais-vision-language-models).

# Qwen3.5

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
   type the encoder runs, with a `vision_encoder` (or `embed_images`) graph
   beside the decoder (a graph `load-vision.ts` takes, one data file) and an
   image processor config, declares image input. The 8-bit graph is taken
   first, then the one with the decoder's suffix, then the 4-bit ones
   (Qwen3.5's repositories have no 8-bit one: its `q4f16`). The catalog's
   entries list those files, and a test checks the rule agrees.
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

## Qwen3.5's GGUF build

`unsloth/Qwen3.5-2B-GGUF` reads images too, from the repository's vision
projector (`mmproj-F16.gguf`, type `qwen3vl_merger`; `load-vision-gguf.ts`).
It holds the same encoder as the ONNX export, under llama.cpp's names: a
patch kernel per frame with a bias, the 48 × 48 position table, blocks
with a fused `attn_qkv`, `v.post_ln` (the merger's LayerNorm) and the
merger's `mm.0` and `mm.2`. The loader refuses a file with a pre-norm or
Qwen3-VL's deepstack layers, which the encoder has no pass for. RoPE's
frequencies aren't in the file: 1e4^(-j/16), the ONNX graph's `inv_freq`
(and llama.cpp's base). The repository has only float projectors (F16 668
MB, BF16, F32), so every projection is quantized to 8 bits per block of 32
as it loads: about twice the ONNX build's `q4f16` encoder in textures, and
3.4 times its download. The image processor's config comes from
`Qwen/Qwen3.5-2B`, where the tokenizer does.

# Liquid AI's vision-language models

`LiquidAI/LFM2.5-VL-450M`, `-1.6B` and `-3B` and `LiquidAI/LFM3-VL-2B`,
each in its `-GGUF` and `-ONNX` repository, as released.

## The models' parts

**LFM2.5-VL** is an LFM2.5 text model with SigLIP2 (NaFlex) in front: the
base encoder for 450M (768 wide, 12 blocks, 12 heads of 64), so400m for
1.6B and 3B (1,152 wide, 27 blocks, 16 heads of 72, an MLP 4,304 wide).

- Patches of 16 × 16 pixels (a row is y, x, channel) through a projection
  with a bias, plus a learned 16 × 16 position table resized bilinearly to
  the image's patch grid.
- Pre-norm blocks: LayerNorm, separate q, k and v projections with
  biases, attention over the view's patches, an output projection;
  LayerNorm, a projection, tanh GELU, a projection.
- A last LayerNorm, then the projector: each 2 × 2 block of patches as one
  row, a projection, exact GELU, a projection into the text model's width.
- The processor (`Lfm2VlImageProcessor`) resizes an image to whole
  32-pixel blocks within 64 to 256 tokens when it fits one view (up to
  twice 512 × 512 pixels). A larger image becomes a grid of 512-pixel
  tiles, 2 to 10 of them (1 to 10 for 3B), the grid nearest its aspect
  ratio, each tile 256 tokens, and the whole image again as a thumbnail.
- The chat template writes `<image>`; the processor replaces it with
  `<|image_start|>`, the image's tokens and `<|image_end|>`, a tiled
  image's tiles each after `<|img_row_R_col_C|>` and its thumbnail after
  `<|img_thumbnail|>`. Image rows take their place in the sequence like any
  token (LFM2's 1D RoPE).

**LFM3-VL** is LFM3 with a vision tower of its own kind: 14 blocks, 1,024
wide, RMS norms, no biases.

- Patches as Qwen's (a row is channel, frame, y, x: 1,536 values, the image
  as both frames) through a projection and an RMS norm. No position table.
- Eight conv blocks: RMS norm, a projection to B | C | x, a 3 × 3
  depthwise convolution of B · x over the patch grid, times C, an output
  projection. Six attention blocks: RMS norm, q (16 heads of 64) and k and
  v (8 heads: grouped-query), an RMS norm per head on q and k, 2D RoPE
  (frequencies 1e6^(-j/16); the first 16 pairs of a head turn by the
  patch's row, the next 16 by its column), an output projection. Each
  block ends in RMS norm and a SwiGLU MLP (gate | up, then down).
- The projector: LayerNorm, each 2 × 2 block of patches as one row, a
  projection (4,096 wide), exact GELU, a projection.
- The processor is Qwen2-VL's resize (whole 32-pixel blocks, 65,536 to
  16.7 M pixels). The chat template writes `<|vision_start|><|image|>
  <|vision_end|>`, and `<|image|>` becomes the image's tokens, in sequence
  positions.

## The files

| | Text weights | Vision encoder |
|---|---|---|
| GGUF | `…-Q4_0.gguf` (LFM3-VL: `…-QAD-Q4_0.gguf`), read as LFM2 and LFM3 GGUFs are | `mmproj-…-Q8_0.gguf` (103 MB for 450M, 583 MB for 1.6B and 3B, 203 MB for LFM3-VL) |
| ONNX | `decoder…_q4.onnx` (LFM3-VL: `decoder_q4f16.onnx`) | `vision_encoder_q8.onnx` (1.6B: `embed_images_q8.onnx`): 110, 495, 495 and 196 MB |

- **The ONNX encoder is the 8-bit one.** Liquid AI's `q4` encoder is far
  from its float one: on a 512 × 512 image, LFM2.5-VL-450M's `q4` features
  are 51% off `fp16`'s (relative L2; worst token cosine 0.45), its `q8`
  3.5% (0.993). The cards pair the `q4` decoder with the `fp16` encoder for
  that reason; `q8` is half its size.
- **GGUF projectors** (`load-vision-gguf.ts`): Q8_0 projections are taken
  as they are; float tensors (the patch kernels, and so400m's 4,304-wide
  MLP rows, which Q8_0's blocks of 32 can't hold) are quantized to 8 bits
  per block of 32, as the text loaders do a float matrix. The patch
  kernels are a convolution's ([out][channel][y][x], LFM3-VL's one per
  frame): their columns are reordered to the processor's patch rows.
  LFM3-VL's RoPE base isn't in the file (llama.cpp sets 1e6 for the type).
  The image processor's config comes from the original repository, where
  the tokenizer does.
- **ONNX decoders** (`load-onnx.ts`). LFM2.5-VL's take embeddings
  (`inputs_embeds`), with float32 scales and a float32 head that is the
  embedding transposed: the head is read as the tied embedding and stored
  at 8 bits (the separate `embed_tokens` file isn't downloaded), and scales
  round to float16. 3B's weights are in four data files
  (`WebGlModelMeta.files.moreData`). LFM3-VL's keeps 28 projections and
  the embedding at 8 bits with zero points (asymmetric): those are
  dequantized and stored at 8 bits per block of 32, symmetric, as the GGUF
  loader stores a matrix that isn't Q4_0 or Q8_0. A matrix fused from
  parts of both widths (q, k, v at 8 bits with a 4-bit gate) is 8-bit. It
  rotates q and k with `RotaryEmbedding` nodes over Hugging Face's head
  layout (dims j and j + 64; full-attention layers turn only the first 32
  pairs), so each head's rotated dims are put first, as the GGUF loader
  does (`headPermutation`); the loader checks the layout against the
  graph's sin cache.

## How it runs

One encoder class runs all three towers (`vision.ts`), from pieces both
loaders yield (`vision-weights.ts`): a block is attention or conv, its
norms LayerNorm or RMS, its MLP GELU or SwiGLU; q, k and v are one stacked
projection. Kernels added for these models: an 8-bit matmul (two RGBA32UI
texels per row and block of 32, zero points where the file has them), RMS
norm, the q and k pass with per-head norms and any head width, scores and
values with grouped-query heads, the 3 × 3 convolution over patches in
merge order, and SwiGLU's product.

An image is a list of views (`image.ts` `planViews`): one, or LFM2.5-VL's
tiles and thumbnail. Each view is encoded on its own, and its rows go to
the engine as an image of its own. `detail: "low"` reads any image as one
view. Position tables are resized on pixel centers, as SigLIP2 does (the
GGUF path with the wider filter Transformers uses where a grid is smaller
than the table; the ONNX graph resizes without it).

## Checks (M3 Ultra, the desktop app's browser)

- **Encoders against onnxruntime** (`bench/vision-check.html`, the same
  patches through onnxruntime 1.31 with `accuracy_level` 0, exact float
  arithmetic on the same weights): relative L2 1e-5 to 3e-5, every token's
  cosine 1.000000, for LFM2.5-VL-450M (`q4` and `q8`, 512 × 512 and
  384 × 512), 1.6B (`q8`, 512 × 512 and 320 × 448: a short last block),
  3B (`q8`) and LFM3-VL (`q8`, 512 × 512 and 384 × 640). (With the graphs'
  own `accuracy_level` 4, onnxruntime's `q4` is 13% from its exact result:
  it quantizes activations too.)
- **GGUF projectors** against the ONNX references: 450M 3.3% from `fp16`
  (worst token cosine 0.994), 1.6B 4.6% and 3B 3.3% from `q8`, LFM3-VL 2.8%
  from `fp16` (0.996): what two 8-bit quantizations of one encoder differ
  by (onnxruntime's `q8` is 3.5% to 3.7% from `fp16`).
- **Answers through `chat.completions.create`** (`bench/vision.html`), on
  all eight builds: a red circle on white, a blue square, a follow-up about
  the background, two images in one message, and a text question after
  them. On a 1024 × 768 scene with text (6 tiles and a thumbnail, 1,807
  prompt tokens) LFM2.5-VL-450M reads the text and names the colours as
  llama.cpp's `llama-mtmd-cli` does with the same files.
- **Speed and memory** (`bench/vl.html?model=…&ctx=8192&rounds=3`, each
  build in a load of its own, files in storage, the pane on screen; a game
  was open on the Mac, so speeds are lower bounds). Textures are the
  engine's own count (its memory claim): once loaded, and the most while an
  image is read (the encoder's weights and work beside the model's). The
  process column is how much the browser's GPU process grew (macOS
  `footprint`) from before the load to after an image was read.

  | Build | Load | Textures loaded | While reading an image | GPU process | Prompt (≈520 tokens) | Writing | An image, one view | 1024 × 768 |
  |---|---|---|---|---|---|---|---|---|
  | LFM2.5-VL-450M GGUF | 0.4 s | 446 MiB | 603 to 618 MiB | +0.58 GB | 9,200 tok/s | 167 tok/s | 0.31 s | 1.3 s (1,807 tokens) |
  | LFM2.5-VL-450M ONNX | 0.9 s | 479 MiB | 640 to 654 MiB | +0.64 GB | 3,400 tok/s | 139 tok/s | 0.29 s | 1.5 s |
  | LFM2.5-VL-1.6B GGUF | 0.7 s | 973 MiB | 1,507 to 1,521 MiB | +1.07 GB | 1,770 tok/s | 127 tok/s | 1.7 s | 5.8 s |
  | LFM2.5-VL-1.6B ONNX | 1.9 s | 1,064 MiB | 1,611 to 1,625 MiB | +1.24 GB | 2,870 tok/s | 124 tok/s | 0.9 s | 4.7 s |
  | LFM3-VL-2B GGUF | 1.0 s | 1,285 MiB | 1,569 to 1,619 MiB | +1.42 GB | 1,650 tok/s | 73 tok/s | 0.54 s | 0.7 s (423 tokens) |
  | LFM3-VL-2B ONNX | 4.5 s | 1,549 MiB | 1,833 to 1,883 MiB | +1.71 GB | 510 tok/s | 70 tok/s | 0.36 s | 0.6 s |
  | LFM2.5-VL-3B GGUF | 1.2 s | 1,886 MiB | 2,420 to 2,434 MiB | +1.99 GB | 670 tok/s | 49 tok/s | 1.8 s | 6.1 s |
  | LFM2.5-VL-3B ONNX | 4.2 s | 1,990 MiB | 2,537 to 2,551 MiB | +2.13 GB | 1,390 tok/s | 58 tok/s | 0.9 s | 4.9 s |

  An image's time is mostly loading the encoder for it (a GGUF projector's
  float rows are quantized as they load: so400m's take about a second
  more than the ONNX encoder's); a warm 512 × 512 view encodes in 0.16 s
  (450M, LFM3-VL) to 0.5 s (so400m). Against an iPhone's 2 GB a tab (not
  measured on one): 450M is far under; 1.6B is under while loaded and
  within 1.5 to 1.6 GiB of textures while it reads an image; LFM3-VL's GGUF
  loads like LFM3 1.7B (1.3 GiB) and reaches 1.6 GiB reading an image, its
  ONNX build 1.9 GiB; 3B is at 1.9 to 2.0 GiB before any image.

## Left

- The encoder's matmul and attention are plain (one patch per fragment):
  1024 × 1024 takes seconds on a Mac, more on a phone. Tiling (several
  patches per fragment, as the text path's 4-token matmuls) is the next
  step if `high` matters.
- Float (`fp16`) ONNX encoders aren't read, only MatMulNBits ones.
- The encoder is loaded for each request with a new image and freed after
  it. For so400m that load is most of an image's time (and 0.5 GiB of
  textures): keeping it loaded where memory allows would make a second
  image several times quicker.
- LFM3-VL's ONNX decoder reads prompts at a third of its GGUF's speed: its
  8-bit projections take the 8-bit kernels.
- Video parts.
- iPhone memory with the 2B encoder loaded (its 187 MiB of textures, freed
  after each request's images) hasn't been measured on the device.
