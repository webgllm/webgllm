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
2. **Read only for images, a piece at a time.** The first request with an
   image downloads the encoder (and `preprocessor_config.json`); a request
   whose prompt reads an image's rows encodes the images it hasn't encoded
   before and reads the rows back (tokens × hidden floats). Text chats
   never touch any of it. The rows are kept for 8 recent images per model.
   The encoder is never on the GPU whole (`VisionEncoder.encodeStream`):
   its file is read in the tower's order (the patch projection, each
   block, the merger), and each piece's weights are uploaded, every view
   of the request is passed through that piece, and the weights are freed
   before the next piece is read. Each view keeps its residual stream
   between pieces (4 MB a megapixel-sized view); the blocks' tensors are
   one set, sized for the largest view. So the most held beside the model
   is one block's weights and a view's work (`visionStreamBytes`, which
   the memory claim is made from and a test holds to the byte), not the
   tower: 204 MiB of weights for LFM3-VL's, 354 for Qwen3.5's GGUF
   projector, 459 for LFM2.5-VL's so400m. The bytes read and the draws
   are the ones loading the encoder and then encoding made, in the same
   order for a view, so the rows are bit for bit the same
   (`vision-check.html` compares them: 0 of 524,288 values differ for
   each of the three towers) and the time is the same or less (LFM3-VL
   409 against 478 ms, so400m 4.1 against 4.9 s, Qwen3.5's float
   projector 3.1 s either way, load included, alternated in one page).
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

Measured on an M3 Ultra (8,192-token context, `vl.html`): 1,494 MiB of
textures loaded, 1,608 MiB while it reads a 512 × 384 image and 1,648 MiB
for 1024 × 768 (1,948 and 1,989 MiB when the whole encoder, 354 MiB of
weights, was held). A new image takes 2 to 3 s, nearly all of it reading
the encoder's weights (the encode is 0.4 s for 256 tokens): the 334 M
float weights are quantized again for each image, where a Q8_0 projector
is taken as it is.

What the 8 bits cost, at 512 × 512 against onnxruntime's fp16 encoder
(`vision-check.html?encoder=Qwen3.5-2B-GGUF/mmproj-F16.gguf`): relative
L2 0.093, worst token's cosine 0.67. The same encoder in float32 numpy,
from the same file, is at 0.012 (worst 0.98), and with its matrices
quantized as the loader does at 0.094 (worst 0.67, the same token): the
structure is right and the rest is the quantization. For scale, the ONNX
build's `q4` encoder is 0.64 from float. Answers agree with llama.cpp's
on the same two files (`llama-mtmd-cli`, temperature 0) except on a
borderline glyph: on the 512 × 384 house scene the engine reads "HELLO
2023" for "HELLO 2026", and so does llama.cpp once its projector holds
the loader's 8-bit values (in float it reads 2026). More bits per weight
don't settle that digit (9 bits reads 2026, 10 bits 2023, 12 bits 2026);
only the float weights do, which would take twice the textures.

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
  image is read (a piece of the encoder and the views' work beside the
  model's; in brackets, what it was when the whole encoder was held).

  | Build | Load | Textures loaded | While reading an image | Prompt (≈520 tokens) | Writing | An image, one view | 1024 × 768 |
  |---|---|---|---|---|---|---|---|
  | LFM2.5-VL-450M GGUF | 0.4 s | 446 MiB | 510 to 545 MiB (603 to 618) | 9,200 tok/s | 167 tok/s | 0.25 s | 1.1 s (1,807 tokens) |
  | LFM2.5-VL-450M ONNX | 0.9 s | 479 MiB | 543 to 579 MiB (640 to 654) | 3,400 tok/s | 139 tok/s | 0.28 s | 1.5 s |
  | LFM2.5-VL-1.6B GGUF | 0.7 s | 973 MiB | 1,065 to 1,110 MiB (1,507 to 1,521) | 1,770 tok/s | 127 tok/s | 1.7 s | 5.4 s |
  | LFM2.5-VL-1.6B ONNX | 1.9 s | 1,064 MiB | 1,156 to 1,201 MiB (1,611 to 1,625) | 2,870 tok/s | 124 tok/s | 1.2 s | 5.5 s |
  | LFM3-VL-2B GGUF | 1.0 s | 1,285 MiB | 1,377 to 1,427 MiB (1,569 to 1,619) | 1,650 tok/s | 73 tok/s | 0.54 s | 0.7 s (423 tokens) |
  | LFM3-VL-2B ONNX | 4.5 s | 1,549 MiB | 1,641 to 1,691 MiB (1,833 to 1,883) | 510 tok/s | 70 tok/s | 0.6 s | 0.9 s |
  | LFM2.5-VL-3B GGUF | 1.2 s | 1,886 MiB | 1,977 to 2,022 MiB (2,420 to 2,434) | 670 tok/s | 49 tok/s | 1.8 s | 6.1 s |
  | LFM2.5-VL-3B ONNX | 4.2 s | 1,990 MiB | 2,082 to 2,128 MiB (2,537 to 2,551) | 1,390 tok/s | 58 tok/s | 0.8 s | 4.1 s |
  | Qwen3.5-2B GGUF | 2.5 s | 1,494 MiB | 1,608 to 1,648 MiB (1,948 to 1,989) | 740 tok/s | 66 tok/s | 2.1 s | 2.8 s |

  An image's time is mostly reading the encoder's weights for it (a float
  GGUF projector's rows are quantized as they are read: Qwen3.5's take
  about 2.5 s of its 3.1); a warm 512 × 512 view encodes in 0.16 s (450M,
  LFM3-VL) to 0.5 s (so400m).

- **On an iPhone** (16 Pro Max, Safari, the GGUF builds, the same page with
  `len=512&reply=32&rounds=1`, one after another without a rest, each
  model just downloaded). iOS ends a tab at 2 GiB (`per-process-limit` in
  the device's log, 2,097,666 KB when it took one). In brackets, the
  textures while an image was read when the whole encoder was held:

  | Build | Result | Load | Textures loaded | While reading an image | Prompt (≈520 tokens) | Writing | An image, one view (first, another) | 1024 × 768 |
  |---|---|---|---|---|---|---|---|---|
  | LFM2.5-VL-450M GGUF | answers text and images | 8 s | 436 MiB | 500 to 535 MiB (594 to 608) | 1,240 to 1,530 tok/s | 101 to 120 tok/s | 4.0 s, 0.75 s | 6.6 s |
  | LFM2.5-VL-1.6B GGUF | answers text and images | 22 s | 974 MiB | 1,065 to 1,110 MiB (1,507 to 1,522) | 460 tok/s | 44 tok/s | 20 s, 3.6 s | 33 s |
  | LFM3-VL-2B GGUF | answers text and images | 31 s | 1,285 MiB | 1,377 to 1,427 MiB (1,570 to 1,619) | 310 tok/s | 32 tok/s | 7.3 s, 1.2 s | 2.4 s |
  | Qwen3.5-2B GGUF | answers text and images | 39 s | 1,510 MiB | 1,613 to 1,654 MiB (killed reading its first image) | 240 tok/s | 27 tok/s | 22 s, 3.7 s | 7.2 s |
  | LFM2.5-VL-3B GGUF | loads, then iOS ends the tab at its first prompt | 51 s | 1,874 MiB | | | | | |

  (A first image's time has the projector's download in it.) The Liquid
  builds read "HELLO 2026" and the colours right in all three images, the
  same replies as with the whole encoder held; Qwen3.5 reads the 1024 ×
  768 one right and misreads the small text ("WEBLZ" for "WEBGL 77"), as
  on the Mac. Before the encoder's weights went through a piece at a
  time, Qwen3.5's image took the tab past 2 GiB (its 354 MiB projector
  beside 1.5 GiB of model); the 3B is over with its text alone.

## Left

- The encoder's matmul and attention are plain (one patch per fragment):
  1024 × 1024 takes seconds on a Mac, more on a phone. Tiling (several
  patches per fragment, as the text path's 4-token matmuls) is the next
  step if `high` matters.
- Float (`fp16`) ONNX encoders aren't read, only MatMulNBits ones.
- The encoder's weights are read again for each request with a new image.
  For so400m that is most of an image's time: keeping the tower loaded
  where memory allows (0.5 GiB of textures) would make a second image
  several times quicker.
- LFM3-VL's ONNX decoder reads prompts at a third of its GGUF's speed: its
  8-bit projections take the 8-bit kernels.
- Video parts.
- Qwen3.5 2B's ONNX build on an iPhone with its encoder loaded (187 MiB
  of textures, freed after each request's images) hasn't been measured on
  the device; its GGUF build's image doesn't fit there (above).
- A GGUF projector in float (Qwen3.5's) is quantized again for every new
  image, 2 to 3 s on a Mac. Keeping the quantized rows between images
  would remove most of it.
