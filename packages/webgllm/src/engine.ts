/**
 * LFM3, LFM2 and Qwen3.5 on WebGL 2: uploads the weights a loader hands over, and runs the
 * forward pass a batch of tokens at a time, keeping the attention cache and
 * the convolutions' state on the GPU between calls.
 */
import { expandQ6, fitsExpandedHead, headScratchBytes, fundedF16Scratch } from "./q6-expanded.js"
import { SubmissionBudget, calibratedReadQuota, calibratedWindow, pipelineSpeed } from "./gpu/submission-budget.js"
import { releaseOwnedBuffer } from "./upload-staging.js"
import { attentionError } from "./attention-check.js"
import { queriesFit, slotRuns } from "./cache-slots.js"
import { checkpointsToDrop, type SnapshotRole } from "./checkpoint-policy.js"
import { GL, Gpu, GpuError, type Pass, type Program } from "./gpu/gl.js"
import type { NativeSpec, WebGpu } from "./gpu/webgpu.js"
import { attentionPrep, bufferDequant, bufferMatvec, bufferPairMatmul, bufferPairs, chunkArgmax, linChunkLoop, linRecurrence, promptConv, promptMatmul, promptMatmulBuffer, promptPrep, chunkMax, expandedHead, FLASH_KEYS, flashChunks, flashCombine, flashFits, fusedAdd, q4WordHead, wordHead } from "./gpu/native.js"
import type { PendingRead, Texture } from "./gpu/gl.js"
import * as K from "./gpu/kernels.js"
import { panelLayout } from "./panels.js"
import { MAX_SEEN_ROWS, type SeenRow } from "./penalty-window.js"
import { weightHeadRegions, weightPanelSource } from "./weight-panels.js"
import { matrixRow, quantsFromRaw, rawRgba4, q4WordsToRgba4 } from "./repack.js"
import type {
  LayerType,
  LayerWeights,
  ModelConfig,
  QuantMatrix,
  WeightPiece
} from "./weights.js"

/** A quantized matrix on the GPU. */
type GpuMatrix = {
  N: number
  K: number
  bits: 4 | 6 | 8
  /** Quants in byte textures rather than RGBA32UI. */
  bytes: boolean
  /** 4-bit byte quants in RGBA8 (normalized) rather than RGBA8UI. */
  unorm: boolean
  /**
   * Scales and zero points packed: `scales` is RGBA32UI, each texel the
   * block's 4 scales then its 4 zero points as halves (one fetch for both);
   * `zeros` is null.
   */
  packed: boolean
  /**
   * 4-bit quants in an RGBA4 texture: texel (column · rowTexels + kb · 32 +
   * r · 8 + j, row) holds row r's weights 32kb + 4j .. + 3 of the group, so
   * the texture unit splits and converts them (layer weights).
   */
  rgba4: boolean
  blocks: number
  /** Row groups (N / 4). */
  groups: number
  /** Row groups per column of the textures. */
  rowsPerColumn: number
  /** Quant texels per row group in one column. */
  rowTexels: number
  quants: Texture
  /** The quants in a storage buffer for the native buffer matvec (see TUNING.bufferMatvec); null where they can't be. */
  stored?: Texture | null
  /** Q6_K embedding texture panels, including quants as the first. */
  quantPanels?: Array<Texture>
  q6Expanded?: boolean
  /** Q6_K word texels in the native head's order (kernels.ts's ORDER_Q6_WORDS). */
  q6Ordered?: boolean
  /** An expanded Q6_K head in RGBA32UI texels for the native expanded head (kernels.ts's EXPAND_Q6_WORDS). */
  q6ExpandedWords?: boolean
  scales: Texture
  zeros: Texture | null
}

type GpuLayer = {
  type: LayerType
  operatorNorm: Texture
  /** LFM3's sandwich norms; LFM2 has none. */
  postOperatorNorm?: Texture
  ffnNorm: Texture
  postFfnNorm?: Texture
  conv?: {
    inProj: GpuMatrix
    outProj: GpuMatrix
    kernel: Texture
    state: [Texture, Texture]
    /** The state after each prefix of the last few-token batch (see accept). */
    prefixes: Texture
  }
  attention?: {
    qkvg: GpuMatrix
    o: GpuMatrix
    qNorm: Texture
    kNorm: Texture
    /** The key and value cache: keys, then values from column kvDims / 8. */
    kv: Texture
    /**
     * Rows in the cache when it's a ring (sliding-window layers: the window
     * plus a batch, position p at row p % ring); 0 for a full cache.
     */
    ring: number
  }
  /** Qwen3.5's Gated DeltaNet (see LinearAttentionWeights). */
  linear?: {
    inProj: GpuMatrix
    ba: GpuMatrix
    outProj: GpuMatrix
    /** Taps: texel (c / 4, k). */
    kernel: Texture
    /** The last convKernel - 1 rows of raw q|k|v, and the next one (swapped each batch). */
    convState: [Texture, Texture]
    /** The recurrent state (see kernels.ts), and the next one (swapped each token). */
    state: [Texture, Texture]
    a: Texture
    dtBias: Texture
    norm: Texture
  }
  gateUp: GpuMatrix
  down: GpuMatrix
}

/**
 * Knobs for A/B timing on the bench page (bench/check.html?ab=...); the
 * defaults are what the engine uses.
 */
export const TUNING = {
  /** 1: run on WebGPU where the browser has it, else WebGL 2 (see gpu/webgpu.ts); 0: WebGL 2. */
  webgpu: 1,
  /**
   * WebGPU: a small pass with a K loop (a token's matmuls) shares each
   * texel's loop among a workgroup's threads, about this many threads in
   * all (0: one thread a texel; negative: never, untimed). An M3 Ultra
   * wrote LFM3 31% faster at 32,768; an iPad Pro M4 and a Pixel 10 Pro
   * slower: tune() times it.
   */
  lanes: 0,
  /**
   * WebGPU: a residual add (y's RMS, the residual, its sums, the next input)
   * in one native pass: 1 a token's, 2 a prompt's too. An M3 Ultra read
   * LFM3 1.7B's prompts at 5,060 tokens/s with the prompt's fused where
   * 4,920 (its sandwich norm made the add three passes), LFM2.5 350M's at
   * 21,950 where 21,570, 1.2B's at 7,410 where 7,350, Qwen3.5 2B's the same.
   * 3: a token's, and a prompt's where tune's lanes won (see workgroupFusions).
   */
  fusedAdd: 3,
  /** WebGPU: the Q6_K word head as a native pass, its activations in workgroup memory. */
  nativeHead: 1,
  /** ...in this many K-slices (its threads share their activations already). */
  nativeHeadSplit: 1,
  /** ...its texels uploaded in the order it reads them (kernels.ts's ORDER_Q6_WORDS). */
  q6Order: 1,
  /** WebGPU: an expanded Q6_K head in RGBA32UI texels, read by a native pass (gpu/native.ts's expandedHead)... */
  nativeExpanded: 1,
  /** ...this many threads a workgroup. */
  nativeExpandedThreads: 32,
  /** WebGPU: a token's queries, keys and values in one native pass (gpu/native.ts's attentionPrep). */
  nativePrep: 1,
  /** WebGPU: the pick's chunk maxima and argmax as native passes, a workgroup a chunk and a row (gpu/native.ts's chunkMax). */
  nativePick: 1,
  /**
   * WebGPU: a few tokens' 4-bit byte matvecs from storage buffer copies of
   * their weights, made as the weights load (see storeQuants; prompts
   * dequantize from the buffers)...
   */
  bufferMatvec: 1,
  /**
   * ...as bufferPairs (1: a slice a lane, its lanes over K about
   * bufferBlocks blocks each) or bufferMatvec (0: a workgroup a row group,
   * bufferMatvecThreads lanes, one slice); -1: whichever tune timed faster
   * where the workgroups' lanes share K loops (tuned.matvec), else
   * bufferPairs. An
   * M3 Ultra (lanes) wrote LFM2.5 350M's tokens at 570/s with bufferMatvec,
   * 509 with bufferPairs; a Pixel 10 Pro (no lanes) at 62 with bufferPairs,
   * 13 with bufferMatvec.
   */
  bufferPairs: -1,
  bufferMatvecThreads: 64,
  bufferBlocks: 16,
  /**
   * WebGPU, f16Pack 3: the dequantized weights in a storage buffer, read by
   * a native matmul (gpu/native.ts's promptMatmulBuffer). LFM3 1.7B read
   * prompts at 4,800 tokens/s on an M3 Ultra this way where 4,450, 167 on
   * a Pixel 10 Pro (whose probe then took f16Pack 3) where 152, and 350 on
   * an iPhone 16 Pro Max where 340.
   */
  bufferGemm: 1,
  /**
   * bufferGemm for the 8-bit matrices among 4-bit ones too (unsloth's
   * Qwen3.5 2B has 21: Q5_K and Q4_1 tensors the loader keeps as 8-bit):
   * dequantized into the same scratch, through the same matmul, where the
   * native pair matmul ran them at 2.7 TFLOP/s (the f16 matmul: 17 to 19).
   * An M3 Ultra read
   * Qwen3.5 2B's prompts at 4,285 tokens/s this way where 3,504, with the
   * same replies.
   */
  bufferGemm8: 1,
  /**
   * bufferGemm's matmul: row groups a thread (see gpu/native.ts's
   * promptMatmulBuffer); 0: one where tune's lanes won (threads sharing K
   * loops: a GPU that wants more threads), else two. Two read LFM3's
   * prompts 1.42x as fast on a Pixel 10 Pro (bench/gemm-exact.html), 3%
   * faster on an iPhone 16 Pro Max, and LFM2.5 350M's 5% slower on an M3
   * Ultra, whose lanes win.
   */
  gemmRowGroups: 0,
  /**
   * 1: bufferGemm's matmuls are paced in bands of columns, every token in
   * each, not of rows (see Pass.bandColumns): a Pixel 10 Pro paced LFM3's
   * prompt in bands of 128 tokens, at 0.6 TFLOP/s where 1.0 for 512. It
   * read LFM2.5 350M's prompts at 996 tokens/s this way where 724, 1.2B's
   * at 320 where 254, LFM3's at 200 where 170 and Qwen3.5 2B's at 122
   * where 109, at 60 frames a second; an M3 Ultra no faster, an iPhone 16
   * Pro Max 0 to 2% slower.
   */
  gemmColumnBands: 1,
  /** 1: WebGL's prompt matmuls are paced in bands of 32 columns, every token in each, as gemmColumnBands. */
  glColumnBands: 0,
  /**
   * WebGPU, f16Pack 0: prompts' f16 matmuls as the native pass, 8 rows × 4
   * tokens an invocation (gpu/native.ts's promptMatmul). The load's packing
   * probe times it against f16Pack 3's: a Pixel 10 Pro took it and read
   * LFM2.5 1.2B's prompts at 230 tokens/s where the GLSL pass read 194, LFM3
   * 1.7B's at 155 where 127; an M3 Ultra and an iPhone 16 Pro Max kept 3.
   */
  nativePromptMatmul: 1,
  /**
   * WebGPU: writing's Gated DeltaNet delta, output and state in one native
   * pass, the state read once (gpu/native.ts's linRecurrence). Qwen3.5 2B
   * wrote 175 tokens/s on an M3 Ultra where the two passes wrote 165, 30.5
   * on an iPhone 16 Pro Max where 29.6, and 11.9 on a Pixel 10 Pro either
   * way...
   */
  nativeRecurrence: 1,
  /**
   * WebGPU: a prompt batch's queries, keys and values in one native pass
   * (gpu/native.ts's promptPrep: a workgroup a head of a token, its sum of
   * squares through workgroup memory), where three passes each summed
   * the head's squares again for every texel (0). The same bits. An M3
   * Ultra ran a layer's in 0.04 ms where 0.39, and read LFM3 1.7B's
   * prompts at 5,300 tokens/s where 5,155, Qwen3.5 2B's at 4,880 where
   * 4,765, LFM2.5 1.2B's at 7,680 where 7,610, 350M's the same. -1: where
   * tune's lanes won (see workgroupFusions).
   */
  promptPrep: -1,
  /**
   * WebGPU: a prompt batch's short convolution and its state in one native
   * pass (gpu/native.ts's promptConv: B·x once a token, through workgroup
   * memory), where two passes multiplied it again for each tap of every
   * output texel (0). The same bits. An M3 Ultra ran LFM3's (7 taps) in
   * 0.07 ms a layer where 0.20, and read its prompts at 5,410 tokens/s
   * where 5,330, LFM2.5 1.2B's at 7,750 where 7,690, 350M's at 22,030
   * where 21,870. -1: where tune's lanes won (see workgroupFusions).
   */
  promptConv: -1,
  /**
   * WebGPU: a prompt batch's recurrence chunks one after another in one
   * native dispatch (gpu/native.ts's linChunkLoop: a workgroup a value
   * texel, its column of the state in workgroup memory from chunk to
   * chunk), where each chunk is two dependent passes (0). The same sums in
   * the same order. An M3 Ultra ran a layer's 17 chunks of a 531-token
   * prompt in 0.38 ms this way where 0.75, and read Qwen3.5 2B's prompts at
   * 4,590 tokens/s where 4,440, with the same replies.
   */
  chunkLoop: 1,
  /** ...a thread this many key rows of the state (4, 8 or 16). */
  recurrenceRows: 16,
  /** WebGPU: a token's attention as two native passes, its scores in workgroup memory (gpu/native.ts's flashChunks)... */
  nativeAttend: 1,
  /** ...in chunks of this many keys (a workgroup's threads). */
  flashKeys: FLASH_KEYS,
  /** Prompt matmuls with at most this many row groups split K in two. */
  sliceGroups: 1024,
  /** Few-token matmuls split K until they have about this many fragments... */
  splitFragments: 4096,
  /** Experimental activation streaming; full-model timing keeps the standard kernel. */
  streamMatvec: 0,
  /** ...into at most this many slices (the partial buffer holds 32). */
  maxSplit: 16,
  /** Pixels per row summing its squares (at most K.RMS_PARTS). */
  rmsParts: 64,
  /**
   * Attention over more than twice this many score texels (4 keys each)
   * splits each query's keys into chunks of at least this many, summed by
   * another pass, so many fragments share the work (0: never).
   */
  attendChunkTexels: 64,
  /**
   * For a few rows (writing), chunk only past this many score texels: the
   * extra combine pass costs more than one long loop until then.
   */
  attendChunkTexelsDecode: 256,
  /**
   * Score texels per query row (4 keys each; the scores buffer's width at
   * load): longer key ranges are read in segments of this many (see attend).
   */
  scoreTexels: 2048,
  /** 1: prompt matmuls take four tokens per fragment (where the engine has room). */
  quad: 1,
  /**
   * How those matmuls read 4-bit weights. 1: byte-texel weights are
   * dequantized to f16 first, then multiplied by a matmul of nothing but
   * dots (see MATMUL_F16), and RGBA4 weights are read in place (their own
   * 4-token kernel); 2: RGBA4 weights through f16 too, in chunks of
   * f16ScratchMiB (with f16Pack's h2 reads: 14% faster prompts on an
   * iPhone 16 Pro Max, the same as in place on an M3 once chunked).
   * Engine.tune times 2 against 1 and keeps it where it's faster. 0:
   * every 4-bit matmul reads in place.
   */
  f16Matmul: 1,
  /**
   * ...for batches of at least this many tokens: dequantizing every matrix
   * costs about as much as the f16 matmul saves on ~40 tokens (LFM3 on a
   * PowerVR GPU), so short batches read the weights in place.
   */
  f16MinRows: 64,
  /** 1: reads of candidates nothing runs behind (no guess) block rather than poll a fence. */
  blockingReads: 1,
  /**
   * 1: split matmuls' partial sums as halves (RGBA16F), half the bytes
   * their consumers read (set before the model loads).
   */
  halfPartials: 1,
  /**
   * 1: 4-bit byte weights' scales and zero points in one RGBA32UI texel a
   * block (one fetch, not two; set before the model loads).
   */
  packZeros: 1,
  /** 1: 4-bit byte weights in RGBA8 (normalized) textures (set before the model loads). */
  unormBytes: 1,
  /** 1: raw MatMulNBits rows go into 4-bit byte texels through a GPU pass, not repacked on the CPU (see writeMatrix). */
  rawUploadGpu: 1,
  /**
   * WebGPU: 4-bit embeddings (an ONNX model's) in word texels, the head a
   * native pass (gpu/native.ts's q4WordHead), not RGBA4 (which WebGPU holds
   * as r16uint: 2 bytes a fetch). LFM3 1.7B ONNX decoded at 20.6 tokens/s
   * on a Pixel 10 Pro where 18.4 (its GGUF's 21.2), the same on an M3 Ultra
   * and an iPhone 16 Pro Max (set before the model loads). WebGL keeps
   * RGBA4: byte texels were 5% faster on the Pixel's Firefox, 3% slower on
   * the M3 Ultra and the same on the iPhone; word texels slower on all.
   */
  wordEmbedding: 1,
  /**
   * 4-bit layer weights in RGBA4 textures (1) or byte textures (0); -1:
   * whichever this GPU writes faster (timed at load; see chooseLayout).
   */
  layerRgba4: -1,
  /** 1: the pick's argmax from the chunk maxima's positions, its loops with constant bounds. */
  fastPick: 1,
  /** Render-area hints for every pass (see Gpu.hints): 1 scissors to the pass's rect. */
  renderHints: 1,
  /** Avoid resubmitting unchanged programs, texture bindings, and samplers. */
  cacheState: 1,
  /**
   * 1: the f16 matmul's dots in mediump (faster where that's half
   * precision, less exact); options.matmulPrecision "mediump" turns it on.
   */
  f16Half: 0,
  /** Write prefill norms/activations directly into existing FP16 input scratch. */
  fusePromptInputs: 1,
  /**
   * How the f16 matmul reads (see matmulF16's `packed`), as bits: 1 the
   * activations straight from their h2 tensor, 8 halves a fetch, with no
   * RGBA16F copy (one pass and its scratch fewer); 2 the dequantized
   * weights as h2 too (the same bytes in half the fetches); 0: both as
   * RGBA16F, 4 values a fetch. Engine.tune times 0 against 3 on a GPU
   * whose prompts take the f16 matmul, and keeps it where it's faster.
   */
  f16Pack: -1,
  /**
   * RGBA4 weights' f16 matmul (f16Matmul 2) dequantizes at most this many
   * MiB at a time: a bigger matrix goes in chunks of K-slices (or of rows,
   * if one slice doesn't fit), each dequantized then multiplied, each
   * chunk two passes more (0: the whole matrix, the scratch sized for the
   * biggest: 64 MiB for LFM3). It's an alternative to their in-place
   * kernel, so its memory is bounded.
   */
  f16ScratchMiB: 32,
  /**
   * Engine.tune times the f16 matmul's choices on full prompt batches
   * through this many layers (0: all of them).
   */
  tuneLayers: 8,
  /**
   * The same for byte weights, whose prompts have no in-place kernel:
   * -1 funds the fastest scratch from the previous automatic-tuner's
   * working-set ceiling; 0 holds a whole matrix; positive values cap MiB.
   */
  f16ScratchBytesMiB: -1,
  /** Ordered draws can reuse one activation conversion texture. */
  f16ActivationBuffers: 1,
  /** Their K-slices: about this many K values each (twice for over 8192 rows)... */
  quadSliceK: 512,
  /** ...and at most this many (readers add them up). */
  quadMaxSplit: 4,
  /** K-slices of the RGBA4 output head (more than 1 adds a pass to sum them). */
  headSplit: 4,
  /** K-slices of a Q6_K output head (in super-blocks; more than 1 adds a pass to sum them). */
  q6HeadSplit: 2,
  /**
   * 1: a Q6_K embedding in RGBA32UI word texels (13 a super-block row, the
   * bits split with integer shifts, its head a loop: see
   * matmulQ6KHeadWords); 0: byte texels (set before the model loads).
   */
  q6Words: 1,
  /** Exact six-bit integers expanded to normalized bytes for simpler heads. */
  q6Expanded: -1,
  /** Repack the lossless head on GPU using existing activation storage where it fits. */
  q6UploadGpu: 1,
  /** GGUF embedding rows per piece (0: derive a bounded read budget). */
  ggufEmbeddingRows: 0,
  /** Concurrent file slices, still bounded by ReadPlan's 24 MiB cap. */
  ggufReadAhead: 6,
  /** 1: stored weights are read through synchronous access handles in a worker (storedSources); 0: as Blobs. */
  syncReads: 1,
  /**
   * 1: those reads run in a worker of their own, beside the repacking and
   * uploads (storedSources). Faster loads on an iPhone 16 Pro Max (Safari),
   * a Mac (Chrome) and a Pixel 10 Pro (Chrome, Firefox): reading 20-60%
   * shorter.
   */
  readerWorker: 1,
  /**
   * ONNX reads in flight ahead of the one in use, and their most MiB, where
   * the reader worker reads (about a layer ahead: most of 128 MiB's gain).
   */
  readerAhead: 6,
  readerAheadMiB: 48,
  /** 1: the ONNX loader frees each read's buffer once used (LoadOnnxOptions.releaseReadBuffers). */
  releaseOnnxReads: 1,

  /** Uploaded model pieces allowed ahead of the GPU, with periodic presentation drains. */
  uploadAhead: 1,
  /** Packing probe rows: 0 uses the actual prompt batch; positive values are diagnostics. */
  packingProbeRows: 0,
  /** 1: a word-texel Q6_K head reads the last add's scaled input (no norm pass; see headScaled). */
  q6FoldNorm: 1,
  /**
   * Few-token matmuls apply the RMS norm as they read, with no norm pass:
   * 1 multiplies by its weights there, 2 reads h2(x · weights) the add drew.
   */
  foldNorm: 2,
  /**
   * With foldNorm 2: each norm folds on its own (the feed-forward's even
   * where the operator's matmuls can't), and 8-bit split matmuls fold the
   * prescaled input too (Qwen3.5 GGUF's Q8_0 beta|alpha): no norm passes.
   * 0: a layer folds both norms or neither, 4-bit matmuls only. Off: on
   * an iPhone 16 Pro Max (Qwen3.5 GGUF, sweep alternated in one page) it
   * wrote 1.6% slower (32.5 against 32.0 GPU ms a token) and read prompts
   * slower, though an M3 wrote 5% faster.
   */
  foldEach: 0,
  /** 1: writing sizes its attention chunks by the key count (see attend). */
  attendDecodeChunking: 1,
  /**
   * 1: writing's attention in short chunks laid out in panels (see
   * attendPanel); 0: the old chunks. Engine.tune times both on a long
   * cache and keeps panels where they're clearly faster (tuned.flashDecode).
   */
  flashDecode: 0,
  /** ...of this many score texels (4 keys each)... */
  flashChunkTexels: 32,
  /** ...past this many score texels (-1: where the old path chunks). */
  flashPast: 128,
  /**
   * ...at most this many panels across (0: as many as a texture's width
   * holds): an iPhone 16 Pro Max took 14.4 ms over LFM3's 7,900 keys with
   * every chunk side by side (a 7,936×1 target), 5.2-5.4 with 8 across and
   * 4.6 with 2, the old path 6.0; an M3 is 15-25% slower at 2 than at 4 or all, still well ahead of the old path.
   */
  flashPanels: 2,
  /** Score texels attendPanel reads per step of its loop. */
  flashUnroll: 1,
  /** 1: attendPanel keeps the softmax online (one loop); 0: the max first. */
  flashOnline: 1,
  /** Key segments per scores pass (0: as many as the scores buffer holds). */
  segmentGroup: 0,
  /**
   * 1: writing's queries go into free slots of their layer's KV cache
   * texture (past the tokens being written, as f32 bits), drawn in the
   * same render pass as the new keys and values, and the scores read them
   * there: one render pass fewer per attention layer (see queryInCache).
   */
  qInCache: 1,
  /**
   * The worker's speculative decoding (see handler.ts): drafts follow an
   * earlier occurrence of at least draftMatch last tokens, as many as
   * pay by measured cost (see DraftPolicy); with
   * guessDrafts, a draft batch also guesses the token after its last row.
   */
  draftMatch: 2,
  guessDrafts: 0,
  /** Queries per scores slice (0: from the batch size; see scoreRows). */
  scoreRows: 0,
  /** 1: few-row conv blocks draw their output and new state in one pass. */
  fuseConv: 1,
  /**
   * 1: writing's Gated DeltaNet block in 6 render passes, not 10: beta|alpha's
   * and in_proj's split matmuls into one target, the convolution with its
   * new state (two targets), and the gates and q̂|k̂ computed by the
   * recurrence's first pass (see linDeltaFused).
   */
  fuseLinear: 1,
  /**
   * Prompt batches run at most paceAhead chunks ahead of the GPU, each
   * about this many full-batch layers of matmul work (attention counted
   * by paceAttention; 0: unpaced), so the browser's GPU process never
   * holds seconds of work and pages keep drawing (see Gpu.pace).
   */
  paceLayers: 4,
  paceAhead: 2,
  /**
   * Prompt passes are fenced in bands of this many estimated reads,
   * retaining the full prompt batch. Zero disables submission pacing.
   */
  submissionReads: 1 << 28,
  /** Target milliseconds per submitted work chunk, adjusted from GPU fences. */
  submissionMs: 4,
  /** Message polls between 1 ms sleeps while waiting for GPU work (0: no periodic sleep). */
  fencePolls: 32,
  /** Reading a prompt, how long a wait polls before it sleeps on timers (see Gpu.promptSpinMs; -1: as writing). */
  promptFenceSpinMs: 0.25,
  /** 1: fence polls wait inside the call where the browser allows it (see Gpu.timedPolls); 0: never. */
  timedFencePolls: 1,
  /** The frame rate a prompt's pacing defends where fences are slow to report (see Gpu.slowFencePromptFps). */
  slowFencePromptFps: 40,
  /** A full compute-queue drain after this many display frames (0 disables). */
  presentationFrames: 4,
  /** Decode layers per fenced submission (0 disables). */
  decodePaceLayers: -1,
  /** Rows per Q6_K embedding allocation (0: one texture). */
  embeddingPanelRows: 2048,
  /** Vocabulary rows per paced head submission. */
  headPaceRows: 16,
  /** What an attention multiply-add costs against a matmul one, for pacing. */
  paceAttention: 4,
  /**
   * Writing flushes queued passes every this many layers (0: only at the
   * readback), so the GPU starts on a token while the rest is still issued.
   */
  flushLayers: 1,
  /**
   * 1: a prompt batch's last layer runs its feed-forward block for the
   * last row only (no later layer reads the others, and only the last
   * row's logits are asked for).
   */
  lastRowOnly: 1,
  /** 1: the warm-up reads through one layer of each kind (see warmUp); 0: all. */
  warmSubset: 1,
  /** WebGPU: the warm-up compiles pipelines in the background without running them (see warmUp). */
  deferWarmUp: 1,
  /**
   * Gated DeltaNet prompt batches run the recurrence in chunks of this
   * many tokens (a multiple of 4; see delta-rule.ts), two dependent passes
   * a chunk rather than two a token; 0: a token at a time.
   */
  linChunk: 32,
  /** ...for batches of at least this many tokens (fewer go a token at a time). */
  linChunkMin: 4,
  /**
   * Full-attention caches grow by this many positions at a time (copied
   * into a bigger texture: see reserve), not doubling (0): at most a
   * step's worth unused, not up to half the cache.
   */
  kvStep: 512,
  /** The worker's prompt batch, for benchmarks (0: its PROMPT_BATCH). */
  promptBatch: 0,
  /** Prompt matmuls take four tokens per fragment from batches this big (EngineOptions.quadMatmuls). */
  quadMinBatch: 256,
  /**
   * Resume points kept for a model with Gated DeltaNet layers (~20 MB each
   * for Qwen3.5 2B; see snapshot): the anchor, where the next user turn's
   * prompt stops sharing this chat's, and the end of the last reply.
   */
  linearSnapshots: 2
}

/**
 * Where the output head left the logits: a texture of logitRows rows per
 * token, or `split` K-slices of it, `stride` rows apart, to be added up.
 */
type HeadOut = { logits: Texture; split: number; stride: number }

/** The time the load's packing probe may spend on timing pairs past its first two, and the most pairs it takes (see choosePromptPacking). */
/**
 * How much faster, in every timed round, another way must be to replace a
 * choice this GPU made before (Engine.incumbent): 5%, or more where the
 * timer is noisier (keepMargin). Ways that close on a fast GPU swap places
 * with the noise of one timing (Chrome on a Mac timed f16Pack 0 at
 * 0.87-1.17x of 3), and a re-pick changes the kernels' rounding, so the
 * same model on the same device writes differently. At 10% Firefox on a
 * Pixel 10 Pro kept f16Pack 3 where 0 read prompts 10% faster (466 against
 * 518 ms a probe).
 */
export const KEEP_MARGIN = 0.95
/** A fresh head choice takes the native expanded Q6_K head only below this fraction of the compact one's time (see chooseHead). */
export const EXPANDED_MARGIN = 0.8

/**
 * Whether `challenger` (a candidate's timed rounds) replaces `standing`
 * (the standing choice's, in the same order): faster in every round, by
 * `margin` (a share of the standing's time).
 */
export const replaces = (challenger: ReadonlyArray<number>, standing: ReadonlyArray<number>, margin: number) =>
  challenger.length > 0 && challenger.length === standing.length &&
  challenger.every((ms, round) => ms < standing[round] * margin)

/**
 * The margin a way needs to replace a standing choice timed with these
 * rounds: KEEP_MARGIN, or more where the timer is noisier than that (how
 * far one way's own rounds differ, the largest of the ways'). Firefox on a
 * Pixel times the same kernel up to 30% apart.
 */
export const keepMargin = (rounds: ReadonlyArray<ReadonlyArray<number>>) => {
  let noise = 0
  for (const r of rounds) if (r.length > 1) noise = Math.max(noise, (Math.max(...r) - Math.min(...r)) / Math.min(...r))
  return Math.min(KEEP_MARGIN, 1 / (1 + noise))
}

/**
 * Timed rounds without their first, where a way's first round was a cold
 * one (more than COLD_ROUND times its quickest later round): kept, it
 * failed the faster way's every-round test and widened keepMargin's noise
 * so far that nothing could replace a standing choice. Chrome on an M3
 * Ultra timed f16Pack 3 at [16.9, 10.6, 10.5, 10.8] ms against 0's
 * [14.8, 14.8, 14.9, 14.9]: once 0 stood, prompts stayed 30% slower in
 * every later session. Each way loses the same round, so rounds still pair
 * up; a way timed once keeps it.
 */
export const warmRounds = (rounds: ReadonlyArray<ReadonlyArray<number>>) =>
  rounds.every((r) => r.length > 1) && rounds.some((r) => r[0] > COLD_ROUND * Math.min(...r.slice(1)))
    ? rounds.map((r) => r.slice(1))
    : rounds
export const COLD_ROUND = 1.25

const PACKING_PROBE_MS = 250
const PACKING_PROBE_PAIRS = 4

/** See Engine.tuned. */
export type Tuned = {
  shortSplit: number | null
  streamMatvec?: boolean
  submissionReads?: number
  /** Measured overlap; zero records that larger submissions missed rendering deadlines. */
  submissionOverlap?: number
  /** Startup validated a larger presentation window on a fast complete pipeline. */
  submissionWindow?: number
  /** How long the load timed its pipeline at, for those: what a second timing is compared with (see retimedTuning). */
  submissionMs?: number
  /** 1: that timing was past two frames, where a GPU busy with something else puts a fast one too: the next load times it once more and keeps the better (see retimedTuning). */
  submissionRetime?: number
  rmsParts?: number
  f16Matmul?: number | null
  f16Pack?: number | null
  /** Writing's attention in panels (TUNING.flashDecode) or not. */
  flashDecode?: number | null
  /** Context length at which the attention variant was actually measured. */
  flashDecodeFrom?: number
  /** WebGPU: threads a small pass's K loop is spread over (the GPU's coopThreads), or TUNING.lanes's. */
  lanes?: number | null
  /** The buffer matvecs (see TUNING.bufferPairs): 1 bufferMatvec, else bufferPairs. */
  matvec?: number | null
}

/**
 * The stored tuning as a load that times the prompt pipeline once more
 * starts from: without what that timing sets, so it runs again. One marked
 * for it (Tuned.submissionRetime) keeps the mark and its kernels: the probe
 * then times those alone (see choosePromptPacking). One stored before
 * timings were kept has its packing chosen again too: it was two pairs'
 * pick on a fast GPU.
 */
export function tuningToRetime(stored: Tuned): Tuned {
  const { submissionReads: _reads, submissionOverlap: _overlap, submissionWindow: _window, submissionMs: _ms, submissionRetime: _retime, ...rest } = stored
  return stored.submissionMs === undefined ? { ...rest, f16Pack: null } : { ...rest, submissionRetime: 1 }
}

/**
 * Whether a stored tuning's prompt timing is taken once more at this load:
 * one marked for it (Tuned.submissionRetime), or one stored before
 * timings were kept (no submissionMs), whose pacing may be from the step
 * at two frames and whose packing from two pairs on a fast GPU.
 */
/** The pick shaders' uSeenRows and uSeenRowCount (see K.PENALIZED). */
function seenRowUniforms(rows: ReadonlyArray<SeenRow>) {
  if (rows.length > MAX_SEEN_ROWS) throw new RangeError(`A batch's penalty rows can't pass ${MAX_SEEN_ROWS}.`)
  const flat = new Array<number>(MAX_SEEN_ROWS * 4).fill(-1)
  rows.forEach((row, i) => flat.splice(i * 4, 4, ...row))
  return { uSeenRows: { ivec4s: flat }, uSeenRowCount: { int: rows.length } }
}

export function timesAgain(stored: Tuned): boolean {
  return !!stored.submissionRetime || (stored.submissionMs === undefined && stored.submissionReads !== undefined)
}

/** Whether two tunings run prompts through the same kernels, so that a timing of one is a timing of the other. */
function sameKernels(a: Tuned, b: Tuned) {
  return (a.f16Pack ?? null) === (b.f16Pack ?? null) && (a.f16Matmul ?? null) === (b.f16Matmul ?? null)
}

/**
 * The tuning to keep after a second timing: `now` is what the load just
 * measured, `previous` what was stored. The kernels are `now`'s, which are
 * the stored ones where the stored tuning was marked for this (the probe
 * timed those alone), and the pacing is a timing of those kernels, never
 * of others: the faster of the two timings where both are of the same
 * kernels (the same pipeline, timed the same way: the slower one was held
 * up), unless the new one found larger submissions miss frames, which
 * stands. Otherwise (a tuning stored before timings were kept, or kernels
 * chosen afresh) it is the new timing's. It isn't timed a third time.
 */
export function retimedTuning(previous: Tuned, now: Tuned): Tuned {
  const stored =
    sameKernels(previous, now) &&
    now.submissionOverlap !== 0 &&
    previous.submissionMs !== undefined &&
    previous.submissionMs <= (now.submissionMs ?? Infinity)
  const pacing = stored ? previous : now
  return {
    ...now,
    rmsParts: previous.rmsParts ?? now.rmsParts,
    submissionReads: pacing.submissionReads,
    submissionOverlap: pacing.submissionOverlap,
    submissionWindow: pacing.submissionWindow,
    submissionMs: pacing.submissionMs,
    submissionRetime: 0
  }
}

/** What resuming at a cache position needs (see Engine.snapshots). */
type Snapshot = {
  conv: Array<Texture>
  windows: Array<Texture>
  /** Each Gated DeltaNet layer's conv state, then its recurrent state. */
  linear: Array<Texture>
  hidden?: Texture
  /** The anchor: dropped for room only once a newer anchor is saved (see snapshot). */
  anchor?: boolean
  /** Stable system/tool prefix, retained alongside the conversational anchor. */
  prefix?: boolean
}

/**
 * A prompt prefix's state, kept apart from the live history (see
 * prefix-store.ts): what a snapshot at `position` holds, plus a copy of
 * each full-attention layer's keys and values for positions [0, position),
 * which the live cache loses once another prompt overwrites them.
 */
/** A saved prefix's texture, read back (see Engine.exportPrefix). */
export type PrefixTexture = { internal: number; width: number; height: number; bytes: Uint8Array }

export type SavedPrefix = {
  position: number
  state: Snapshot
  /** Each full-attention layer's cache rows, in blocks as the live cache's (see kvTexture). */
  kv: Array<Texture>
}

/**
 * A matmul's input: h2 activations, or the f32 residual and the norm to
 * apply (its weights, or "prescaled": already applied by the add).
 */
type MatIn = { a: Texture; norm?: Texture | "prescaled" }

export type EngineOptions = {
  /** The GPU to run on: WebGL 2 on the canvas when left out. */
  gpu?: Gpu | WebGpu
  /**
   * Scale the prompt batch up for smaller models, keeping about the same
   * buffer memory (default true).
   */
  scaleBatch?: boolean
  /**
   * Four tokens per fragment in prompt matmuls (a quarter fewer weight
   * reads per multiply; ~40 MB more at a 512-token batch). Default: for
   * batches of 256 or more.
   */
  quadMatmuls?: boolean
  /**
   * Sliding-window layers keep only their window (plus a batch) of keys and
   * values, in a ring: ~75 MB less for LFM3. On by default.
   */
  ringCache?: boolean
  /** Pass kinds not to draw, for profiling (outputs are wrong). */
  skip?: Array<string>
  /**
   * Most tokens the cache may hold, prompt and reply together: the model's
   * own context length when left out. The cache grows to it as it fills.
   * At least MIN_CONTEXT_LENGTH.
   */
  contextLength?: number
  /** Most tokens per forward batch. */
  batchSize?: number
  /** Multi-target draws to treat as unsupported, for testing the fallbacks. */
  noMrt?: Array<"scaled" | "quad" | "conv" | "chunks" | "lin">
  /**
   * false: Gated DeltaNet prompt batches run the recurrence a token at a
   * time, never in chunks (see TUNING.linChunk; the worker's plain retry).
   */
  chunkedRecurrence?: boolean
  /**
   * Rows per block of a full-attention cache (a power of two): the texture
   * height when left out. Smaller exercises the cache's wrapping, for tests.
   */
  kvBlockRows?: number
  /** Precision of the matmuls' per-block sums. */
  matmulPrecision?: "highp" | "mediump"
  /** Weights in byte textures (faster on the GPUs tried) or in RGBA32UI texels. */
  weightTexels?: "bytes" | "words"
}

/** A layer's intermediate tensors, read back for checking against a reference. */
/** The self-check's tokens: spread over the vocabulary. */
const checkToken = (i: number, vocabSize: number) => (i * 7919 + 13) % vocabSize
/** The self-check's first tokens whose embedding rows and logits headError checks. */
const HEAD_CHECK_TOKENS = 8

export type LayerProbe = (
  layer: number,
  name: string,
  /** The tensor: `rows` tokens from cache position `offset`, `dims` features each. */
  read: () => Promise<{
    data: Float32Array
    rows: number
    dims: number
    offset: number
  }>
) => void | Promise<void>

/**
 * An image's rows in the cache (see Engine.setImages): its merged-token grid
 * in row-major order from cache position `start`.
 */
export type EngineImage = {
  start: number
  /** Rows and columns of tokens: height · width rows in the cache. */
  height: number
  width: number
  /**
   * Its embedding rows (height · width rows of the model's hidden size),
   * read in place of the image tokens' own; null where the cache already
   * holds them (a read of those rows then fails).
   */
  features: Float32Array | null
}

const OUT_WIDTH = 1024
/** Resume points kept (see `snapshot`): ~3 MB each for LFM3 (Qwen3.5: TUNING.linearSnapshots). */
const MAX_SNAPSHOTS = 8
/**
 * Queries of each head per attention slice (see attend), a quarter of a
 * batch in this range: more is fewer passes and more fragments per pass,
 * for a bigger scores buffer.
 */
const SCORE_ROWS = 32
const SCORE_ROWS_LARGE = 128
/** Positions a full-attention layer's cache holds at first: it doubles as needed. */
const KV_START = 1024
/** Most key chunks per query in chunked attention (see attendChunk). */
const ATTEND_CHUNKS = 16
/** About the most key chunks a writing step's attention combines (see attendPanel). */
const FLASH_CHUNKS = 128
/** Logits per chunk when picking candidates on the GPU. */
const CHUNK = 256
/** Most chunks read back as candidates: exact for top-k up to this. */
export const MAX_CHUNKS = 64
/** Half-float bits of the integers 0 .. 255 (zero points packed as halves). */
const HALF_INT = Uint16Array.from({ length: 256 }, (_, n) => {
  if (n === 0) return 0
  const e = Math.floor(Math.log2(n))
  // n = 1.m · 2^e: 10 mantissa bits (exact for n < 2048).
  return ((e + 15) << 10) | (((n / 2 ** e - 1) * 1024) & 1023)
})

/** Batches this small split their matmuls along K (see TUNING). */
const SPLIT_ROWS = 8

/**
 * The fewest tokens a cache may hold: the load's self-check reads a prompt
 * of two tokens and one token after it.
 */
export const MIN_CONTEXT_LENGTH = 3

/**
 * expandQ6's bytes (RGBA8 texels, a row group's row: per super-block, each
 * row's 64) rearranged as EXPAND_Q6_WORDS lays them out: 16-byte runs, run
 * (row, q, half) of a super-block at q · 8 + half · 4 + row.
 */
function expandedWordsOrder(bytes: Uint8Array, rows: number, blocks: number) {
  const src = new Uint32Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 4)
  const out = new Uint32Array(src.length)
  for (let y = 0; y < rows; y++)
    for (let sb = 0; sb < blocks; sb++) {
      const base = (y * blocks + sb) * 256
      for (let r = 0; r < 4; r++)
        for (let k = 0; k < 16; k++) {
          const from = base + (r * 16 + k) * 4, to = base + ((k >> 1) * 8 + (k & 1) * 4 + r) * 4
          out[to] = src[from]; out[to + 1] = src[from + 1]; out[to + 2] = src[from + 2]; out[to + 3] = src[from + 3]
        }
    }
  return out
}

export class Engine {
  /**
   * How a matrix of N rows, K in `slices` slices of kbSlice blocks, goes
   * through an f16 scratch of `width` texels (4 K values each) by columns of
   * hd rows: `group` whole slices of every row at a time if one fits, else
   * one slice at a time, `rows` (a multiple of 4) at a time; chunks evened out.
   */
  static f16Plan(
    f: { hd: number; width: number },
    N: number,
    kbSlice: number,
    slices: number
  ) {
    const sliceKq = kbSlice * 8
    const fit = Math.floor(f.width / (Math.ceil(N / f.hd) * sliceKq))
    if (fit >= 1) {
      const groups = Math.ceil(slices / Math.min(fit, slices))
      return { group: Math.ceil(slices / groups), rows: N }
    }
    const most = Math.max(4, Math.floor(f.width / sliceKq) * f.hd)
    const chunks = Math.ceil(N / most)
    return { group: 1, rows: Math.min(most, Math.ceil(N / chunks / 4) * 4) }
  }

  readonly gpu: Gpu | WebGpu
  config!: ModelConfig
  /** Most tokens the cache holds: the model's, or options.contextLength (see configure). */
  contextLength = 0
  private readonly contextCap: number
  /**
   * Full-attention caches wrap into blocks of 2^kvShift rows, side by side
   * (see kvAt in kernels.ts), so they grow past a texture's height.
   */
  private kvShift = 0
  /** Positions the full-attention caches hold now (see reserve). */
  private kvPositions = 0
  /** Chunked attention's partial results, made when first needed (see attend). */
  private chunkBufs = new Map<number, Array<Texture>>()
  private readonly kvBlockRows: number
  /** Tokens per prompt batch: options.batchSize, scaled to the model (see configure). */
  batchSize: number
  /** options.batchSize: whether the engine runs with desktop-size batches. */
  private readonly baseBatch: number
  /**
   * Rows the batch buffers hold: a batch, plus room for a short tail of the
   * prompt to join the last batch rather than cost a pass of its own.
   */
  capacity: number
  private precision: "highp" | "mediump"
  private bytes: boolean
  private ringCache: boolean
  private layers: Array<GpuLayer> = []
  private embedding!: GpuMatrix
  private quadMatmuls: boolean
  /**
   * Draws into several textures at once that this GPU can do (asked of it
   * at load, see Gpu.canRender); the engine falls back where it can't.
   */
  private mrt = {
    scaled: true,
    quad: true,
    conv: true,
    chunks: true,
    lin: true,
    linFused: true
  }
  private readonly noMrt: Set<string>
  /** See EngineOptions.chunkedRecurrence. */
  private readonly chunkedRecurrence: boolean
  private readonly scaleBatch: boolean
  /** A pair buffer's second texture, for 4-token matmuls (see matmulQuad). */
  private quads = new Map<number, Texture>()
  private finalNorm!: Texture
  private ropeFull!: Texture
  private ropeSliding!: Texture
  /** Tokens in the cache. */
  position = 0
  /** The prompt's images, by cache position (see setImages). */
  private images: Array<EngineImage> = []
  /** Whether image rows take Qwen2-VL's 3D positions ("grid") or their cache positions like any token. */
  private imagePositions: "grid" | "sequence" = "grid"
  /**
   * The batch being read's rotary positions: from `pos0` on (text), or per
   * row in `positions` (rows t, h, w; a batch with image rows).
   */
  private batchRope: { pos0: number } | { positions: Texture } = { pos0: 0 }
  /** Rows' t, h, w positions for batches with image rows, made when first needed. */
  private ropePositions: Texture | null = null
  private buffers!: ReturnType<Engine["allocate"]>

  constructor(
    canvas: OffscreenCanvas | HTMLCanvasElement,
    options: EngineOptions = {}
  ) {
    const cap = options.contextLength
    if (cap !== undefined && !(Number.isInteger(cap) && cap >= MIN_CONTEXT_LENGTH))
      throw new Error(`A context length is a whole number of tokens, at least ${MIN_CONTEXT_LENGTH}; got ${cap}.`)
    this.gpu = options.gpu ?? new Gpu(canvas)
    this.contextCap = cap ?? Infinity
    this.noMrt = new Set(options.noMrt ?? [])
    this.chunkedRecurrence = options.chunkedRecurrence ?? true
    this.kvBlockRows = options.kvBlockRows ?? Infinity
    this.baseBatch = options.batchSize ?? 128
    this.batchSize = this.baseBatch
    this.capacity = this.batchSize + this.batchSize / 8
    this.scaleBatch = options.scaleBatch ?? true
    this.precision = options.matmulPrecision ?? "highp"
    this.gpu.hints = TUNING.renderHints
    this.gpu.cacheState = TUNING.cacheState > 0
    this.gpu.coopThreads = Math.max(0, this.tuned.lanes ?? TUNING.lanes)
    this.gpu.presentationFrames = TUNING.presentationFrames
    this.gpu.configurePacing(this.tuned.submissionReads ?? TUNING.submissionReads, TUNING.submissionMs)
    this.gpu.pollsBeforeSleep = TUNING.fencePolls
    this.gpu.promptSpinMs = TUNING.promptFenceSpinMs
    this.gpu.timedPolls = TUNING.timedFencePolls > 0
    this.gpu.slowFencePromptFps = TUNING.slowFencePromptFps
    this.gpu.presentationFrames = TUNING.presentationFrames
    this.bytes = (options.weightTexels ?? "bytes") === "bytes"
    this.ringCache = options.ringCache ?? true
    this.quadMatmuls = options.quadMatmuls ?? this.baseBatch >= TUNING.quadMinBatch
    for (const name of options.skip ?? []) this.gpu.skip.add(name)
  }

  /** Uploads the model, piece by piece, as the loader yields it. */
  async load(
    pieces: AsyncIterable<WeightPiece>,
    onProgress?: (piece: WeightPiece) => void,
    signal?: AbortSignal
  ) {
    let mark = performance.now()
    this.loadMs = {
      reading: 0,
      uploading: 0,
      pacing: 0,
      convert: 0,
      scratch: 0,
      packing: 0,
      matvec: 0
    }
    // The GPU takes a piece's uploads while the next piece is read: each
    // piece is flushed, not waited for, and the wait (pacing the queue as
    // before) comes just before the next piece's uploads.
    let unpaced = false
    for await (const piece of pieces) {
      signal?.throwIfAborted()
      const got = performance.now()
      this.loadMs.reading += got - mark
      if (unpaced) {
        unpaced = false
        await this.gpu.pace(Math.max(0, TUNING.uploadAhead), true)
        this.loadMs.pacing += performance.now() - got
      }
      if (piece.kind === "config") this.configure(piece.config)
      else if (piece.kind === "embedding") {
        // Whole, or in chunks of rows: allocated with the first.
        const { value, row0 = 0, totalRows = value.N } = piece
        if (row0 === 0) this.embeddingRows = totalRows
        if (row0 === 0 && value.bits === 6 && TUNING.q6Expanded < 0) {
          if (!this.canExpandHead(value.K))
            this.headChoice = { expanded: false, split: this.q6HeadSplit }
          else if (!this.headChoice) await this.chooseHead(value)
        }
        if (row0 === 0) this.resizeScoreBuffers(value.bits)
        if (row0 === 0) {
          // On WebGPU, word texels for the native head (TUNING.wordEmbedding).
          const words = value.bits === 4 && TUNING.wordEmbedding > 0 && this.nativeHeadOn
          this.embedding = this.allocateMatrix(
            totalRows,
            value.K,
            value.bits,
            !!value.zeros,
            // RGBA4 where it fits this GPU's textures; else byte texels
            // (half as wide: Qwen3.5's 248,320 × 2048 on an 8,192-texel GPU).
            value.bits === 4 && !words && this.fitsRgba4(totalRows, value.K),
            false,
            words
          )
        }
        // Written in chunks: zeroed on the GPU first, or the browser
        // zero-fills it on the first chunk's write, far more slowly (430 ms
        // for Qwen3.5 2B's on an iPhone 16 Pro Max, against a 60 ms clear).
        if (row0 === 0 && value.N < totalRows)
          for (const t of [this.embedding.quants, this.embedding.scales, this.embedding.zeros])
            if (t) this.gpu.zero(t)
        this.writeMatrix(this.embedding, value, row0)
        // The self-check's tokens' rows, as the loader gave them (headError).
        if (row0 === 0) this.embeddingSamples.clear()
        for (let i = 0; i < HEAD_CHECK_TOKENS; i++) {
          const id = checkToken(i, this.config.vocabSize)
          const row = id >= row0 && id < row0 + value.N ? matrixRow(value, id - row0) : null
          if (row) this.embeddingSamples.set(id, row)
        }
      } else if (piece.kind === "finalNorm") {
        this.finalNorm = this.vector(piece.value)
        this.finalNormValues = piece.value.slice()
      }
      else {
        // The weight layout this GPU writes fastest, once, before the
        // first layer's textures (see chooseLayout).
        if (!this.layoutChoice) await this.chooseLayout(piece.value.gateUp)
        this.layers[piece.layer] = this.layer(piece.value)
      }
      if (this.gpu.readsPerSlice > 0) {
        this.gpu.flush()
        unpaced = true
      }
      onProgress?.(piece)
      if (this.gpu.isLost)
        throw new GpuError("The GPU context was lost while loading.")
      mark = performance.now()
      this.loadMs.uploading += mark - got
    }
    if (
      !this.embedding ||
      !this.finalNorm ||
      this.layers.length !== this.config.numLayers
    ) {
      throw new Error("The model's weights are incomplete.")
    }
    // The f16 prompt matmul's scratch now, not in the middle of a prompt
    // (only 4-token matmuls on weights that take it run it: none, no
    // scratch).
    let started = performance.now()
    if (this.quadMatmuls && this.usesF16()) this.f16Scratch()
    await this.gpu.finish()
    this.loadMs.scratch = performance.now() - started
    this.expandedUpload = new Uint8Array(0)
    this.repackUpload = new Uint32Array(0)
    // (The upload staging textures, done with.)
    for (const stage of [this.rawStage, this.orderStage]) if (stage) this.gpu.delete(stage)
    this.rawStage = this.orderStage = null
    this.uploads = new Uint32Array(0)
    started = performance.now()
    await this.choosePromptPacking(signal)
    this.loadMs.packing = performance.now() - started
    started = performance.now()
    signal?.throwIfAborted()
    await this.chooseMatvec()
    this.loadMs.matvec = performance.now() - started
  }

  private configure(config: ModelConfig) {
    this.config = config
    this.legacyF16 = null
    if (this.scaleBatch) {
      // Smaller models take bigger batches (fewer, fuller passes) in about
      // the same buffer memory: per token it goes with hidden + FFN width,
      // against LFM3's 2048 + 7680, in powers of two (LFM2.5 350M: 2×).
      const ratio = (2048 + 7680) / (config.hiddenSize + config.intermediateSize)
      const scale = 2 ** Math.max(0, Math.min(2, Math.round(Math.log2(ratio))))
      this.batchSize = this.baseBatch * scale
      this.capacity = this.batchSize + this.batchSize / 8
    }
    // The cache's blocks of rows, side by side, fit a texture's width.
    const max = this.gpu.limits.maxTextureSize
    const kvWidth = this.kvWidth()
    this.kvShift = Math.min(
      Math.floor(Math.log2(Math.min(max, this.kvBlockRows))),
      Math.ceil(Math.log2(Math.max(2, config.contextLength)))
    )
    this.contextLength = Math.min(
      config.contextLength,
      this.contextCap,
      Math.floor(max / kvWidth) * 2 ** this.kvShift
    )
    this.kvPositions = Math.min(KV_START, 2 ** this.kvShift)
    const can = (name: string, formats: Array<number>) =>
      !this.noMrt.has(name) && this.gpu.canRender(formats)
    this.mrt = {
      // The add and the embedding with the next matmuls' scaled input.
      scaled: can("scaled", [GL.RGBA32F, GL.RGBA32UI]),
      // 4-token matmuls.
      quad: can("quad", [GL.RGBA32UI, GL.RGBA32UI]),
      // The conv block's output and state (convFused).
      conv: can("conv", [GL.RGBA32UI, GL.RGBA32F]),
      // Chunked attention, one head a fragment (headGroup tries more).
      chunks: can("chunks", [GL.RGBA32F, GL.RGBA32F, GL.RGBA32F]),
      // Gated DeltaNet's delta and output in one pass.
      lin: can("lin", [GL.RGBA32F, GL.RGBA32F]),
      // ...and writing's fused block (see fuseLinear).
      linFused: can("lin", [GL.RGBA32F, GL.RGBA32F, GL.RGBA32F, GL.RGBA32F])
    }
    this.quadMatmuls &&= this.mrt.quad
    if (!this.mrt.chunks) {
      // Without chunked attention, no key segments either: the context
      // is what one row of scores holds.
      this.contextLength = Math.min(this.contextLength, 4 * TUNING.scoreTexels)
    }
    this.buffers = this.allocate()
    if (this.linDims() && this.chunkedRecurrence && TUNING.linChunk > 0)
      this.linChunkBuffers(TUNING.linChunk)
    if (this.quadMatmuls) {
      const b = this.buffers
      // (Not for a 1×1 stand-in: see allocate.)
      for (const t of [b.opOut, b.bcx, b.qkvg, b.gateUp, b.ffnOut, b.lin?.inProj])
        if (t && t.width * t.height > 1)
          this.quads.set(t.id, this.gpu.texture(GL.RGBA32UI, t.width, t.height))
    }
    this.ropeFull = this.ropeTable(config.ropeThetaFull, config.ropeDimsFull)
    // (Only models with sliding-window layers read the second table.)
    this.ropeSliding = config.layerTypes.includes("sliding_attention")
      ? this.ropeTable(config.ropeThetaSliding, config.ropeDimsSliding)
      : this.ropeFull
  }

  /**
   * cos and sin of position × frequency, computed in doubles: texels 0..15
   * cos, 16..31 sin of frequencies 0..63. Rows 0..ROPE_LOW - 1 are those
   * positions, the rows after them multiples of ROPE_LOW: headPrep adds a
   * position's two parts' angles, so the table stays small at any context
   * length (~1,300 rows at 262,144). Frequencies past the rotated dims are
   * 0 (no rotation).
   */
  private ropeTable(theta: number, rotatedDims: number) {
    const { headDim } = this.config
    const freqDims = this.config.ropeFreqDims ?? headDim
    const half = headDim / 2
    const rows = K.ROPE_LOW + Math.ceil(this.contextLength / K.ROPE_LOW)
    const table = new Float32Array(rows * half * 2)
    for (let r = 0; r < rows; r++) {
      const pos = r < K.ROPE_LOW ? r : (r - K.ROPE_LOW) * K.ROPE_LOW
      for (let i = 0; i < half; i++) {
        const freq = i < rotatedDims / 2 ? theta ** ((-2 * i) / freqDims) : 0
        const angle = pos * freq
        table[r * headDim + i] = Math.cos(angle)
        table[r * headDim + half + i] = Math.sin(angle)
      }
    }
    return this.gpu.texture(GL.RGBA32F, headDim / 4, rows, {
      format: GL.RGBA,
      type: GL.FLOAT,
      pixels: table
    })
  }

  private f32(dims: number, rows: number) {
    return this.gpu.texture(GL.RGBA32F, dims / 4, rows)
  }

  private h2(dims: number, rows: number) {
    return this.gpu.texture(GL.RGBA32UI, dims / 8, rows)
  }

  /** A pair tensor: texel (x, y) holds features 4x.. of tokens 2y, 2y + 1 as halves. */
  private pair(dims: number, rows: number) {
    return this.gpu.texture(GL.RGBA32UI, dims / 4, Math.ceil(rows / 2))
  }

  /** Rows of the q|k|v|gate projection (see AttentionWeights). */
  private qkvgDims() {
    const c = this.config
    return (
      (c.numHeads + 2 * c.numKvHeads) * c.headDim +
      (c.gatePerDim ? c.numHeads * c.headDim : c.numHeads)
    )
  }

  /** Gated DeltaNet's dims: keys, values, q|k|v channels, value heads. */
  private linDims() {
    const l = this.config.linear
    if (!l) return null
    const K = l.numKeyHeads * l.keyHeadDim
    const V = l.numValueHeads * l.valueHeadDim
    return { K, V, C: 2 * K + V, H: l.numValueHeads, l }
  }

  /** The chunked recurrence's tensors (see linGram in kernels.ts), for chunks of `chunk`. */
  private linChunks: {
    chunk: number
    textures: { cum: Texture; a: Texture; t: Texture; p: Texture; u: Texture; w: Texture; oi: Texture; qp: Texture }
  } | null = null

  /** The chunked recurrence's tensors for chunks of `chunk`, made at load or when the chunk size changes. */
  private linChunkBuffers(chunk: number) {
    const lin = this.linDims()
    if (!lin) throw new Error("The model's config has no Gated DeltaNet shapes.")
    if (this.linChunks?.chunk === chunk) return this.linChunks.textures
    for (const tex of Object.values(this.linChunks?.textures ?? {})) this.gpu.delete(tex)
    const M = this.capacity
    const textures = {
      cum: this.f32(lin.H, M),
      a: this.f32(lin.H * chunk, M),
      t: this.f32(lin.H * chunk, M),
      p: this.f32(lin.H * chunk, M),
      u: this.f32(lin.V, M),
      w: this.f32(lin.K, M),
      oi: this.f32(lin.V, M),
      qp: this.f32(lin.K, M)
    }
    this.linChunks = { chunk, textures }
    return textures
  }

  private allocate() {
    const c = this.config
    const M = this.capacity
    const D = c.hiddenSize
    const qkvg = this.qkvgDims()
    const lin = this.linDims()
    const qDims = c.numHeads * c.headDim
    const I = c.intermediateSize
    // The conv block's tensors only for models with conv layers (1×1
    // stand-ins otherwise: Qwen3.5 has none).
    const hasConv = c.layerTypes.includes("conv")
    const conv = (make: () => Texture) =>
      hasConv ? make() : this.gpu.texture(GL.RGBA32UI, 1, 1)
    // Logits for one row at a time where no draft batch is read (see
    // canDraft: Gated DeltaNet models).
    const headRows = c.layerTypes.includes("linear_attention") ? 1 : SPLIT_ROWS
    const residualWidth = D / 4 + K.RMS_PARTS + D / 8
    const residualLarge = [0, 1].map(() =>
      this.gpu.texture(GL.RGBA32F, residualWidth, M)
    )
    // The scaled MRT is only used for at most SPLIT_ROWS rows. A matching
    // small residual target avoids reserving a full-batch integer MRT.
    // WebGL requires attachments to have matching dimensions.
    const smallRows = M > 2 * SPLIT_ROWS ? SPLIT_ROWS : M
    const residualSmall = smallRows < M
      ? [0, 1].map(() => this.gpu.texture(GL.RGBA32F, residualWidth, smallRows))
      : residualLarge
    // Each tensor has its own texture: sharing one between tensors that are
    // never alive at once saves ~25 MB at 512 rows, but a pass that writes a
    // texture the pass before read must wait for it, and prompts read ~12%
    // slower. The narrow matmuls' outputs are K-split, two slices tall.
    return {
      ids: this.gpu.texture(GL.R32UI, 1, M),
      // The hidden state, and its rows' partial sums of squares in the last
      // RMS_PARTS columns (drawn with the add that writes it).
      residual: residualLarge,
      residualLarge,
      residualSmall,
      /**
       * For a few rows, h2(residual · the next norm's weights) from column
       * D/4 + RMS_PARTS, drawn with the add (its second output; see
       * addNormed): one per residual texture.
       */
      scaled: [0, 1].map(() =>
        // (Same size as the residual: a framebuffer's textures must match.)
        this.gpu.texture(GL.RGBA32UI, residualWidth, smallRows)
      ),
      rms: this.gpu.texture(GL.RGBA32F, K.RMS_PARTS, M),
      normed: this.h2(D, M),
      opOut: this.pair(D, 2 * M),
      // (Room for more K-slices of 4-token matmuls: see matmulQuad.)
      bcx: conv(() => this.pair(3 * D, this.quadMatmuls ? 2 * M : M)),
      convOut: conv(() => this.h2(D, M)),
      /**
       * A few rows' convolution output, drawn with the new state in one
       * pass (see convFused): the state's size, h2 in its first D / 8 texels.
       */
      convSmall: conv(() => this.gpu.texture(GL.RGBA32UI, D / 4, this.convRows())),
      qkvg: this.pair(qkvg, 2 * M),
      q: this.f32(c.numHeads * c.headDim, M),
      /**
       * One per head of a group (see headGroup), member i's scores: rows
       * for scoreRows queries of every head, between them.
       */
      scores: lin ? [] as Texture[] : Array.from({ length: this.headGroup() }, () =>
        this.gpu.texture(
          GL.RGBA32F,
          Math.min(TUNING.scoreTexels, Math.ceil(this.contextLength / 4)),
          (c.numHeads / this.headGroup()) * Math.min(M, this.scoreRows())
        )
      ),
      attnOut: this.h2(qDims, M),
      gateUp: this.pair(2 * I, M),
      act: this.h2(I, M),
      ffnOut: this.pair(D, 2 * M),
      lastNormed: this.h2(D, SPLIT_ROWS),
      /** Logits: logitRows rows per token, for up to SPLIT_ROWS tokens (or one: headRows). */
      logits: this.gpu.texture(
        GL.RGBA32F,
        OUT_WIDTH,
        Math.ceil(c.vocabSize / 4 / OUT_WIDTH) * headRows
      ),
      /** Per token: 0 unseen, 1 prompt, 2 never drawn, 3 generated. Same rows as the logits. */
      mask: this.gpu.texture(
        GL.R8UI,
        OUT_WIDTH * 4,
        Math.ceil(c.vocabSize / 4 / OUT_WIDTH)
      ),
      /** Per token (row): each chunk's largest logit, and its rank. */
      chunkMax: this.gpu.texture(
        GL.RGBA32F,
        Math.ceil(c.vocabSize / CHUNK),
        SPLIT_ROWS
      ),
      chunkRank: this.gpu.texture(
        GL.RGBA32F,
        Math.ceil(c.vocabSize / CHUNK),
        SPLIT_ROWS
      ),
      /** Pixel (r, row): the chunk ranked r. */
      chunkByRank: this.gpu.texture(GL.RGBA32F, MAX_CHUNKS, SPLIT_ROWS),
      /** Per chunk: its logits, its index, and the guessed token (row 0). */
      candidates: this.gpu.texture(
        GL.RGBA32F,
        CHUNK / 4 + 2,
        MAX_CHUNKS * SPLIT_ROWS
      ),
      /** Split matmuls' partial sums: widest output × (rows × slices). */
      partial: this.gpu.texture(
        TUNING.halfPartials > 0 && this.gpu.canRender([GL.RGBA16F])
          ? GL.RGBA16F
          : GL.RGBA32F,
        Math.max(2 * c.intermediateSize, qkvg, 3 * D, lin ? lin.C + lin.V : 0) / 4,
        SPLIT_ROWS * 32
      ),
      /** Gated DeltaNet's tensors (see kernels.ts). */
      lin: lin && {
        inProj: this.pair(lin.C + lin.V, 2 * M),
        ba: this.pair(2 * lin.H, 2 * M),
        gates: this.f32(2 * lin.H, M),
        qkv: this.f32(lin.C, M),
        qk: this.f32(2 * lin.K, M),
        delta: this.f32(lin.V, M),
        o: this.f32(lin.V, M),
        out: this.h2(lin.V, M),
        /**
         * Writing's fused block (see fuseLinear): the conv output (the conv
         * state's size: drawn with it), then delta, the output, k̂ and the
         * decay (drawn together).
         */
        small: {
          qkv: this.f32(lin.C, lin.l.convKernel - 1),
          delta: this.f32(lin.V, lin.l.convKernel - 1),
          o: this.f32(lin.V, lin.l.convKernel - 1),
          k: this.f32(lin.V, lin.l.convKernel - 1),
          decay: this.f32(lin.V, lin.l.convKernel - 1)
        }
      }
    }
  }

  /** Rows of the conv state textures (and convSmall): a few rows' batch or the state. */
  private convRows() {
    return Math.max(SPLIT_ROWS, this.config.convKernel - 1)
  }

  /** Texels per cache row: a layer's keys, then its values (h2). */
  private kvWidth() {
    const c = this.config
    return (2 * c.numKvHeads * c.headDim) / 8
  }

  /** A full-attention cache for `positions` tokens, in blocks of 2^kvShift rows. */
  private kvTexture(positions: number) {
    const rows = 2 ** this.kvShift
    return this.gpu.texture(
      GL.RGBA32UI,
      this.kvWidth() * Math.ceil(positions / rows),
      Math.min(positions, rows)
    )
  }

  /** Queries of each head per attention slice: a quarter of a batch, 32 to 128. */
  private scoreRows(expanded = TUNING.q6Expanded > 0 || this.headChoice?.expanded === true) {
    if (TUNING.scoreRows > 0) return TUNING.scoreRows
    const base = Math.min(SCORE_ROWS_LARGE, Math.max(SCORE_ROWS, this.baseBatch / 4))
    return expanded ? Math.max(SCORE_ROWS, base / 2) : base
  }

  private scoreMemorySavings(expanded: boolean) {
    const rows = Math.min(this.capacity, this.scoreRows(false)) - Math.min(this.capacity, this.scoreRows(expanded))
    const width = Math.min(TUNING.scoreTexels, Math.ceil(this.contextLength / 4))
    return rows * this.config.numHeads * width * 16
  }

  private resizeScoreBuffers(embeddingBits: number) {
    const first = this.buffers.scores[0]
    // Recurrent models back their attention panels when first needed.
    if (!first) return
    const count = this.headGroup()
    const height = (this.config.numHeads / count) * Math.min(this.capacity, this.scoreRows())
    const expanded = TUNING.q6Expanded >= 0
      ? TUNING.q6Expanded > 0
      : this.headChoice?.expanded === true
    // Keep the full attention tiles alongside a compact Q6 head. Only
    // the stored scores are rounded. Softmax and its sums remain f32.
    const format = embeddingBits === 6 && !expanded && this.mrt.chunks &&
      this.gpu.canRender(new Array(count).fill(GL.RGBA16F))
      ? GL.RGBA16F : GL.RGBA32F
    if (first.height === height && first.internal === format) return
    for (const texture of this.buffers.scores) this.gpu.delete(texture)
    this.buffers.scores = Array.from({ length: count }, () =>
      this.gpu.texture(format, first.width, height)
    )
  }

  /** Grow recurrent attention storage without changing its logical panel geometry. */
  private growScoreBuffers(width: number, height: number, count: number) {
    const current = this.buffers.scores
    const first = current[0]
    if (first && first.width >= width && first.height >= height) return current
    const max = this.gpu.limits.maxTextureSize
    const w = Math.max(width, Math.min(Math.ceil(width / 64) * 64, max), first?.width ?? 0)
    const h = Math.max(height, Math.min(Math.ceil(height / 64) * 64, max), first?.height ?? 0)
    for (const texture of current) this.gpu.delete(texture)
    return (this.buffers.scores = Array.from({ length: count }, () =>
      this.gpu.texture(GL.RGBA32F, w, h)
    ))
  }

  /**
   * Makes room in the full-attention caches for tokens up to `end`: twice
   * the positions (or as many as asked), copied into a bigger texture of the
   * same layout. Sliding layers' rings never grow.
   */
  private async reserve(end: number) {
    if (end <= this.kvPositions) return
    const block = 2 ** this.kvShift
    const step = TUNING.kvStep
    let positions = Math.min(
      this.contextLength,
      step > 0 ? end : Math.max(end, 2 * this.kvPositions)
    )
    // Whole blocks past the first; within it, whole steps (or a power of two).
    positions =
      positions > block
        ? Math.ceil(positions / block) * block
        : step > 0
          ? Math.min(block, Math.ceil(positions / step) * step)
          : 2 ** Math.ceil(Math.log2(positions))
    const phase = this.gpu.activePhase
    this.gpu.setPhase("load")
    // Reuse the measured load quota for allocation overlap too. Slow pipelines
    // retain two clears/copies in flight; fast ones can issue the original batch.
    const growthBatch = Math.max(1, Math.floor(this.gpu.readsPerSlice / Math.max(1, TUNING.submissionReads)))
    let growthPending = 0
    try {
      for (const layer of this.layers) {
        const a = layer.attention
        if (!a || a.ring) continue
        const kv = this.kvTexture(positions)
        try {
          this.gpu.run({
            name: "grow",
            program: this.gpu.program(K.COPY_POSITIONS),
            inputs: { uX: a.kv },
            uniforms: {
              uPos0: { int: 0 },
              uRow0: { int: 0 },
              uMod: { int: a.kv.height }
            },
            output: kv,
            rect: { x: 0, y: 0, width: a.kv.width, height: a.kv.height }
          })
        } catch (error) {
          this.gpu.delete(kv)
          throw error
        }
        this.gpu.delete(a.kv)
        a.kv = kv
        // Cache growth also initializes textures in the GPU process.
        // Keep at most two clears/copies in flight, rather than an
        // uninterrupted allocation queue or an empty queue per layer.
        if (this.gpu.readsPerSlice > 0 && ++growthPending >= growthBatch) {
          await this.gpu.pace(1, true)
          growthPending = 0
        }
      }
      if (this.gpu.readsPerSlice > 0) await this.gpu.pace(0, true)
      if (this.gpu.outOfMemory()) throw new GpuError("Out of memory.")
    } catch (error) {
      // Some layers may hold a cache that didn't copy: start over.
      if (!this.gpu.isLost) this.reset()
      throw new GpuError(
        `The GPU couldn't grow the cache to ${positions} tokens (${error instanceof Error ? error.message : String(error)}). Start a new session to keep going.`
      )
    } finally {
      this.gpu.setPhase(phase)
    }
    this.kvPositions = positions
  }

  /** Reserve the known prompt once, avoiding copies at each prefill batch. */
  async preparePrompt(end: number) {
    if (!Number.isInteger(end) || end < this.position || end > this.contextLength)
      throw new Error(`Invalid prompt cache end: ${end}.`)
    if (end - this.position > this.batchSize) await this.reserve(end)
  }

  /** Back to the starting cache size, for a new sequence. */
  private shrink() {
    const start = Math.min(KV_START, 2 ** this.kvShift)
    if (this.kvPositions <= start) return
    for (const layer of this.layers) {
      const a = layer.attention
      if (!a || a.ring) continue
      this.gpu.delete(a.kv)
      a.kv = this.kvTexture(start)
    }
    this.kvPositions = start
  }

  /**
   * Query heads of a KV head that an attention fragment takes together:
   * the most that divides the group, with the chunks' 3 textures per head
   * (see attendChunk) within the GPU's render targets.
   */
  private headGroup() {
    if (this.group) return this.group
    const c = this.config
    const group = c.numHeads / c.numKvHeads
    // (The combine reads 3 textures a member plus up to 4 of the q|k|v|gate
    // projection's: within the GPU's texture units.)
    let G = Math.min(
      group,
      Math.floor(this.gpu.limits.maxDrawBuffers / 3),
      Math.floor((this.gpu.limits.maxTextureUnits - 4) / 3)
    )
    while (
      G > 1 &&
      (!this.mrt.chunks ||
        group % G ||
        !this.gpu.canRender(new Array(3 * G).fill(GL.RGBA32F)))
    )
      G--
    this.group = G
    return G
  }
  private group = 0

  /**
   * Chunked attention's partial results, for `rows` (query, chunk) pairs
   * at least: per head of a group, weighted values (two textures) and
   * (max, sum).
   */
  private chunkBuffers(rows: number, G: number) {
    const have = this.chunkBufs.get(G)
    if (have && have[0].height >= rows) return have
    const c = this.config
    const height = c.linear
      ? Math.max(rows, Math.min(Math.ceil(rows / 256) * 256, this.gpu.limits.maxTextureSize))
      : Math.max(rows, this.scoreRows() * ATTEND_CHUNKS)
    have?.forEach((t) => this.gpu.delete(t))
    const bufs = Array.from({ length: 3 * G }, () =>
      this.gpu.texture(GL.RGBA32F, (c.numHeads * c.headDim) / 8 / G, height)
    )
    this.chunkBufs.set(G, bufs)
    return bufs
  }

  /** attendPanel's partial results, at least width × height (see chunkBuffers). */
  private panelBuffers(width: number, height: number, G: number) {
    const have = this.panelBufs.get(G)
    if (have && have[0].width >= width && have[0].height >= height) return have
    const w = Math.max(width, have?.[0].width ?? 0)
    const h = Math.max(height, have?.[0].height ?? 0)
    have?.forEach((t) => this.gpu.delete(t))
    const bufs = Array.from({ length: 3 * G }, () => this.gpu.texture(GL.RGBA32F, w, h))
    this.panelBufs.set(G, bufs)
    return bufs
  }
  private panelBufs = new Map<number, Array<Texture>>()

  private vector(values: Float32Array, rows = 1) {
    return this.gpu.texture(GL.RGBA32F, values.length / 4 / rows, rows, {
      format: GL.RGBA,
      type: GL.FLOAT,
      pixels: values
    })
  }

  /** A quantized matrix as textures, filled with `m`. */
  /**
   * A hash of the engine's GPU code (every kernel and the engine itself):
   * what the worker keys a GPU's remembered self-check and weight layout
   * by, so any change to the code checks again.
   */
  static readonly fingerprint: string = (() => {
    const text =
      Object.entries(K)
        .map(([name, v]) =>
          name +
          (typeof v === "function"
            ? ((v as { source?: string }).source ?? v.toString())
            : String(v))
        )
        .join("\n") + Engine.toString() +
      Gpu.toString() +
      weightPanelSource.toString() +
      weightHeadRegions.toString() + expandQ6.toString() + fitsExpandedHead.toString() + headScratchBytes.toString() + fundedF16Scratch.toString() + calibratedReadQuota.toString() + calibratedWindow.toString() + pipelineSpeed.toString() + `${PACKING_PROBE_MS},${PACKING_PROBE_PAIRS}` + SubmissionBudget.toString() + q4WordsToRgba4.toString()
    // FNV-1a.
    let h = 0x811c9dc5
    for (let i = 0; i < text.length; i++) {
      h ^= text.charCodeAt(i)
      h = Math.imul(h, 0x01000193)
    }
    return (h >>> 0).toString(16)
  })()
  /**
   * How this GPU splits prompt matmuls along K, where it differs from
   * TUNING's (null: TUNING's): `shortSplit`, the most slices for batches
   * under TUNING.f16MinRows (TUNING.quadMaxSplit; an iPhone reads short
   * prompts 11% faster in 2, an M3 30% slower). From tune, or remembered
   * from an earlier load. (Full batches' slice size isn't tuned: timed
   * alone, their matmuls show no steady difference, though a whole prompt
   * on an M3 does; timing whole prompts would take seconds on a phone.)
   * `f16Matmul`: whether RGBA4 weights' full prompt batches go through
   * the f16 matmul or the in-place kernel (TUNING.f16Matmul 2 or 1).
   * `f16Pack`: how the f16 matmul reads (TUNING.f16Pack), wherever
   * prompts take it.
   */
  tuned: Tuned = { shortSplit: null, f16Matmul: null, f16Pack: null, flashDecode: null }
  /**
   * This GPU's last checked choices for the model, from a note another
   * browser version, engine code or settings wrote (the handler's
   * incumbentGpuTuning): when tuned afresh, each stands unless another way
   * is faster in every round by KEEP_MARGIN. Null: first choices.
   */
  incumbent: { tuned?: Tuned; head?: { expanded: boolean; split: number } } | null = null
  /** What the last tune measured: % faster than TUNING's, % noise, per choice. */
  tuneReport: Record<string, number> = {}

  /**
   * Times the choices in `tuned` on this GPU (a layer's worth of prompt
   * matmuls on the loaded weights) and keeps a choice only where it beats
   * TUNING's by 4% and by twice the spread between its own rounds (phones'
   * timings drift with heat): Apple GPUs differ from
   * PowerVR, and big ones from small ones, in how many slices fill them.
   * After ~100 ms of work (an idle GPU clocks down), each candidate in
   * turn, the order reversed in the second round, best of 2. Call after warmUp; returns what it picked, or null (choices
   * unchanged) if `stop` said to stop first.
   */
  async tune(
    stop: () => boolean = () => false,
    options: { keepCache?: boolean; alreadyBusy?: boolean } = {}
  ): Promise<Tuned | null> {
    const before = { ...this.tuned }
    this.tuned = {
      shortSplit: null,
      f16Matmul: before.f16Matmul ?? null,
      f16Pack: before.f16Pack ?? null,
      flashDecode: null,
      lanes: null,
      matvec: null,
      streamMatvec: before.streamMatvec,
      submissionReads: before.submissionReads,
      submissionOverlap: before.submissionOverlap,
      submissionMs: before.submissionMs,
      submissionRetime: before.submissionRetime
    }
    const b = this.buffers
    // A layer's prompt matmuls as the forward runs them: a conv block's,
    // an attention block's (the first of each kind), one feed-forward's.
    const conv = this.layers.find((l) => l.conv)?.conv
    const attention = this.layers.find((l) => l.attention)?.attention
    const ffn = this.layers[0]
    const run = async (rows: number) => {
      if (conv) {
        await this.matmul(b.normed, conv.inProj, b.bcx, rows)
        await this.matmul(b.convOut, conv.outProj, b.opOut, rows)
      }
      if (attention) {
        await this.matmul(b.normed, attention.qkvg, b.qkvg, rows)
        await this.matmul(b.attnOut, attention.o, b.opOut, rows)
      }
      await this.matmul(b.normed, ffn.gateUp, b.gateUp, rows)
      await this.matmul(b.act, ffn.down, b.ffnOut, rows)
    }
    // A full prompt batch through the first TUNING.tuneLayers layers, every
    // pass as a prompt runs them (synthetic tokens, the cache reset
    // before each): what the f16 matmul's choices change isn't only the
    // matmuls (see the f16 choices below).
    const tokens = Array.from(
      { length: Math.min(this.batchSize, this.capacity, this.contextLength) },
      (_, i) => 1000 + (i % 5000)
    )
    let forwards = false
    const forward = async () => {
      forwards = true
      const all = this.layers
      const work = this.work
      this.work = null
      if (TUNING.tuneLayers > 0) this.layers = all.slice(0, TUNING.tuneLayers)
      try {
        this.reset()
        await this.forwardHidden(tokens)
      } finally {
        this.layers = all
        this.work = work
      }
    }
    class Stopped extends Error {}
    const choose = async (
      key: "shortSplit" | "f16Matmul" | "f16Pack" | "flashDecode" | "lanes" | "matvec",
      candidates: Array<number>,
      sample: () => unknown
    ): Promise<number | null> => {
      // The alternatives once untimed (first use costs more; the warm-up
      // already ran TUNING's), and TUNING's timed once: each sample repeats
      // the matmuls to about 25 ms, so a fast GPU's few milliseconds
      // aren't lost in the wait for it.
      if (stop()) throw new Stopped()
      for (let c = 1; c < candidates.length; c++) {
        this.tuned[key] = candidates[c]
        await sample()
      }
      this.tuned[key] = candidates[0]
      await this.gpu.finish()
      if (stop()) throw new Stopped()
      const t0 = performance.now()
      await sample()
      await this.gpu.finish()
      const reps = Math.max(
        1,
        Math.min(16, Math.ceil(25 / (performance.now() - t0)))
      )
      // An idle GPU clocks down: busy it ~100 ms first, so the clock has
      // come up before anything is timed.
      for (
        const start = performance.now();
        !options.alreadyBusy && performance.now() - start < 100;
      ) {
        if (stop()) throw new Stopped()
        for (let r = 0; r < reps; r++) await sample()
        await this.gpu.finish()
      }
      const rounds = candidates.map((): Array<number> => [])
      for (let round = 0; round < 2; round++) {
        // Order reversed in the second round: steady drift cancels.
        for (let i = 0; i < candidates.length; i++) {
          const c = round === 0 ? i : candidates.length - 1 - i
          if (stop()) throw new Stopped()
          this.tuned[key] = candidates[c]
          await this.gpu.finish()
          const t = performance.now()
          for (let r = 0; r < reps; r++) await sample()
          await this.gpu.finish()
          rounds[c].push(performance.now() - t)
        }
      }
      this.tuned[key] = null
      const times = rounds.map((r) => Math.min(...r))
      // How far a candidate's two rounds differ: the noise to beat.
      const noise = Math.max(
        ...rounds.map((r) => (Math.max(...r) - Math.min(...r)) / Math.min(...r))
      )
      const best = times.indexOf(Math.min(...times))
      this.tuneReport[`${key}Gain`] =
        Math.round((1 - times[1] / times[0]) * 1000) / 10
      this.tuneReport[`${key}Noise`] = Math.round(noise * 1000) / 10
      // candidates[0] is TUNING's value, the standing choice unless this
      // GPU chose another before (incumbent), which then needs KEEP_MARGIN
      // to lose. The reversed pairs must independently show a useful gain.
      // A shared clock drift between rounds is not evidence against a
      // candidate that wins both within-round comparisons.
      // (An incumbent that kept TUNING's value stored null: that's its
      // choice too, as hard to replace as any other.)
      const prior = this.incumbent?.tuned
      const chose = !!prior && key in prior
      const earlier = prior?.[key]
      const standing = earlier == null ? 0 : Math.max(0, candidates.indexOf(earlier))
      const kept = best === standing || !replaces(rounds[best], rounds[standing], chose ? keepMargin(rounds) : 0.96)
      const pick = candidates[kept ? standing : best]
      return pick === candidates[0] ? null : pick
    }
    let picked: Tuned
    try {
      const short = Math.min(40, this.capacity)
      // Byte-weight f16 matmuls already use bounded weight chunks. Keep
      // their existing split instead of timing another full-model prompt.
      // Native RGBA4 matmuls retain the original split calibration.
      const shortSplit = this.layerMats().some((m) => m.rgba4)
        ? await choose(
            "shortSplit",
            [TUNING.quadMaxSplit, TUNING.quadMaxSplit === 2 ? 4 : 2],
            () => run(short)
          )
        : (before.shortSplit ?? null)
      this.tuned.shortSplit = shortSplit
      // Then, with the picks so far in place, RGBA4 weights through the
      // f16 matmul or in place, then how the f16 matmul reads (wherever
      // prompts take it), each timed on full prompt batches through the
      // first layers, not a layer's matmuls alone: an iPhone 16 Pro Max
      // reads whole prompts 16% faster through the f16 matmul, but its
      // matmuls timed back to back come out even, as each dequantization
      // then waits on the matmul before it rather than filling the gaps
      // between a prompt's small dependent passes. Forwards need the cache:
      // if the caller keeps something in it (`keepCache`), these choices
      // are left for another time (undefined).
      const quad = this.quadMatmuls && TUNING.quad > 0 && tokens.length >= TUNING.f16MinRows
      const free = !options.keepCache
      let f16Matmul: number | null | undefined = this.promptMatmulMeasured
        ? before.f16Matmul
        : null
      let f16Pack: number | null | undefined = null
      if (
        !this.promptMatmulMeasured &&
        quad &&
        TUNING.f16Matmul > 0 &&
        this.f16Mats().some((m) => m.rgba4) &&
        this.f16Sized()
      )
        f16Matmul = free
          ? await choose(
              "f16Matmul",
              [TUNING.f16Matmul, TUNING.f16Matmul > 1 ? 1 : 2],
              forward
            )
          : undefined
      this.tuned.f16Matmul = f16Matmul ?? null
      if (this.packingMeasured) f16Pack = before.f16Pack
      else if (quad && this.usesF16() && this.f16Scratch())
        f16Pack = free
          ? await choose(
              "f16Pack",
              [
                TUNING.f16Pack & 3,
                (TUNING.f16Pack & 3) === 3
                  ? this.activationCopyFits()
                    ? 0
                    : 1
                  : 3
              ],
              forward
            )
          : undefined
      this.tuned.f16Pack = f16Pack ?? null
      if (forwards) this.reset()
      // Writing's attention from a long cache (a synthetic one: 4,096
      // keys in a scratch cache of the first full-attention layer's
      // shape), in panels or the old way: Apple's phone and desktop GPUs
      // disagreed on panel layouts, and other GPUs are untimed. (The
      // scratch cache leaves the model's own alone.)
      const full = this.layers.find(
        (l) => l.attention && l.type !== "sliding_attention"
      )
      const keys = Math.min(4096, this.contextLength)
      let flashDecode: number | null = null
      if (
        full?.attention &&
        this.mrt.chunks &&
        TUNING.attendChunkTexels > 0 &&
        keys > 4 * TUNING.flashPast
      ) {
        const kv = this.kvTexture(keys)
        const layer = { ...full.attention, kv }
        try {
          flashDecode = await choose(
            "flashDecode",
            [TUNING.flashDecode, TUNING.flashDecode > 0 ? 0 : 1],
            () =>
              this.attention(
                layer,
                full.type,
                1,
                keys - 1,
                0,
                this.buffers.qkvg
              )
          )
        } finally {
          this.gpu.delete(kv)
        }
      }
      // WebGPU: a token's passes with their K loops shared among threads,
      // or a thread a texel (see TUNING.lanes), timed on a token's forward
      // through every layer (a layer's matmuls alone came out even on a Mac
      // whose whole token ran 31% faster). Forwards need the cache.
      const lanes = !this.gpu.sharesLoops || TUNING.lanes < 0
        ? null
        : free
          ? await choose("lanes", [TUNING.lanes, TUNING.lanes > 0 ? 0 : 32768], async () => {
              this.gpu.coopThreads = Math.max(0, this.tuned.lanes ?? TUNING.lanes)
              // (Shared K loops with bufferMatvec, else bufferPairs: an M3
              // Ultra's lanes won only so; see matvec below.)
              this.tuned.matvec = this.gpu.coopThreads > 0 && this.layers.some((l) => l.gateUp.stored) ? 1 : 0
              forwards = true
              this.reset()
              await this.forwardHidden([1000])
            })
          : (before.lanes ?? null)
      this.gpu.coopThreads = Math.max(0, lanes ?? TUNING.lanes)
      // Where the lanes won, bufferMatvec against bufferPairs on the same
      // forward, the lanes kept: a Pixel 10 Pro whose lanes won by noise
      // wrote LFM3's tokens at 4.5/s with bufferMatvec where 21.6 with
      // bufferPairs. Elsewhere bufferPairs.
      const stores = this.layers.some((l) => l.gateUp.stored)
      // (choose's null is its first candidate kept: bufferMatvec standing,
      // as the lanes' winning arm ran it.)
      const matvec = TUNING.bufferPairs >= 0 || !stores
        ? null
        : (lanes ?? TUNING.lanes) <= 0
          ? 0
          : free
            ? (await choose("matvec", [1, 0], async () => {
                forwards = true
                this.reset()
                await this.forwardHidden([1000])
              })) ?? 1
            : (before.matvec ?? 0)
      if (forwards) this.reset()
      picked = {
        matvec,
        shortSplit,
        f16Matmul,
        f16Pack,
        flashDecode,
        lanes,
        flashDecodeFrom: flashDecode == null ? undefined : keys,
        streamMatvec: before.streamMatvec,
        submissionReads: before.submissionReads,
        submissionOverlap: before.submissionOverlap,
        submissionWindow: before.submissionWindow,
        submissionMs: before.submissionMs,
        submissionRetime: before.submissionRetime,
        rmsParts: before.rmsParts
      }
    } catch (error) {
      this.tuned = before
      // (A forward stopped part way leaves its batch in the cache.)
      if (forwards) this.reset()
      this.f16Trim()
      if (error instanceof Stopped) return null
      throw error
    }
    this.tuned = picked
    this.f16Trim()
    return picked
  }

  /** The weight layout to use without timing (a remembered choice; see chooseLayout). */
  preferLayout: "rgba4" | "bytes" | null = null
  /**
   * The last load's milliseconds waiting on the weight pieces (reading and
   * repacking them) and turning them into textures.
   */
  loadMs = { reading: 0, uploading: 0, pacing: 0, convert: 0, scratch: 0, packing: 0, matvec: 0 }
  /** 4-bit layer weights in RGBA4 (else byte texels): see chooseLayout. */
  private rgba4Layers = true
  /** The layout chooseLayout picked and its timings (ms per writing matmul). */
  layoutChoice: {
    layout: "rgba4" | "bytes"
    rgba4Ms: number
    bytesMs: number
    /** How long choosing took. */
    ms?: number
    rgba4PrefillMs?: number
    bytesPrefillMs?: number
  } | null = null

  /** Choose storage from real weights, dependent matvecs and a full-batch projection. */
  private async chooseLayout(m: QuantMatrix) {
    // An isolated FFN projection misses sequence operators and the input
    // conversions they share with adjacent matmuls. The byte path can
    // calibrate its packing on those real operators after loading.
    const sequenceQuadPrefill =
      (this.config.linear !== undefined || this.config.layerTypes.includes("conv")) &&
      this.quadMatmuls && TUNING.quad > 0 && TUNING.f16Matmul > 0 &&
      this.contextLength >= TUNING.f16MinRows
    if (m.bits !== 4 || TUNING.layerRgba4 >= 0 || this.preferLayout || sequenceQuadPrefill) {
      this.rgba4Layers =
        TUNING.layerRgba4 >= 0
          ? TUNING.layerRgba4 !== 0
          : this.preferLayout
            ? this.preferLayout === "rgba4"
            : m.bits !== 4 || !sequenceQuadPrefill
      this.layoutChoice = {
        layout: this.rgba4Layers ? "rgba4" : "bytes",
        rgba4Ms: NaN,
        bytesMs: NaN
      }
      return
    }
    const started = performance.now()
    const made = [true, false].map((rgba4) => {
      const gm = this.allocateMatrix(m.N, m.K, 4, !!m.zeros, rgba4, true)
      this.writeMatrix(gm, m, 0)
      return gm
    })
    // Nonzero activations and a dependency through every output prevent
    // tiled/deferred GPUs from timing an unused or overwritten result.
    this.gpu.write(this.buffers.normed, { x: 0, y: 0, width: m.K / 8, height: 1 }, {
      format: GL.RGBA_INTEGER, type: GL.UNSIGNED_INT,
      pixels: new Uint32Array(m.K / 2).fill(0x3c003800)
    })
    const width = Math.ceil(m.N / 4 / 64)
    const height = Math.max(1, Math.min(TUNING.maxSplit, m.K / 32))
    const scratch = Array.from({ length: 2 }, () => this.gpu.texture(GL.RGBA32F, width, height))
    const checksum = this.gpu.program(`#version 300 es
precision highp float;precision highp int;uniform sampler2D uX,uPrev;uniform int uWidth;out vec4 o;
void main(){ivec2 p=ivec2(gl_FragCoord.xy);vec4 s=texelFetch(uPrev,p,0)*0.5;for(int i=0;i<64;i++){int x=p.x*64+i;if(x<uWidth)s+=texelFetch(uX,ivec2(x,p.y),0);}o=s;}`)
    let slices = 1
    const step = async (gm: GpuMatrix) => {
      slices = this.splitPartials(this.buffers.normed, gm, 1)
      this.gpu.run({ program: checksum, inputs: { uX: this.buffers.partial, uPrev: scratch[0] },
        uniforms: { uWidth: { int: gm.groups } }, output: scratch[1],
        rect: { x: 0, y: 0, width, height: slices }, work: 64 })
      scratch.reverse()
      await this.gpu.pace(1, true)
    }
    const consume = () => this.gpu.read(scratch[0], { x: 0, y: 0, width, height: slices })
    const best = [Infinity, Infinity]
    const layoutSamples = [[], []] as number[][]
    let prefill: { rgba4: number; bytes: number } | undefined
    try {
      for (const gm of made) await step(gm)
      await consume()
      for (let round = 0; round < 2; round++) {
        for (const i of round % 2 ? [1, 0] : [0, 1]) {
          const t = performance.now()
          for (let n = 0; n < 8; n++) await step(made[i])
          await consume()
          const ms = (performance.now() - t) / 8
          layoutSamples[i].push(ms)
          best[i] = Math.min(best[i], ms)
        }
      }
      // The final rule requires a clear decode win as well. Do not run
      // full-batch candidates after decode has already ruled native storage out.
      if (layoutSamples[0].every((ms, round) => ms < layoutSamples[1][round] * 0.95))
        prefill = await this.layoutPrefill(made)
    } finally {
      for (const gm of made)
        for (const texture of [gm.quants, gm.scales, gm.zeros, gm.stored]) if (texture) this.gpu.delete(texture)
      for (const texture of scratch) this.gpu.delete(texture)
    }
    // Native storage must win in both phases and both decode rounds.
    // Prefer bytes when results disagree: a single projection cannot establish
    // a whole-model win, and native storage adds conversion work at load.
    this.rgba4Layers = !!prefill && prefill.rgba4 < prefill.bytes * 0.98 &&
      layoutSamples[0].every((ms, round) => ms < layoutSamples[1][round] * 0.95)
    this.layoutChoice = {
      layout: this.rgba4Layers ? "rgba4" : "bytes",
      rgba4Ms: best[0],
      bytesMs: best[1],
      rgba4PrefillMs: prefill?.rgba4,
      bytesPrefillMs: prefill?.bytes,
      ms: performance.now() - started
    }
  }

  /** Storage must be competitive for prefill, not just a one-token matvec. */
  private async layoutPrefill(made: GpuMatrix[]) {
    const rows = Math.min(this.batchSize, this.contextLength)
    if (!this.quadMatmuls || rows < TUNING.f16MinRows || !this.quads.has(this.buffers.gateUp.id))
      return { rgba4: 1, bytes: 1 }
    const matrix = made[0]
    const probe = Object.create(Engine.prototype) as Engine
    Object.assign(probe, this, {
      layers: [{ gateUp: matrix, down: { ...matrix, N: matrix.K, K: matrix.N / 2,
        blocks: matrix.N / 2 / 32, groups: matrix.K / 4 } }] as unknown as GpuLayer[],
      tuned: { ...this.tuned }, f16: null, f16Kept: new Map(), legacyF16: null
    })
    this.gpu.write(this.buffers.normed, { x: 0, y: 0, width: matrix.K / 8, height: rows }, {
      format: GL.RGBA_INTEGER, type: GL.UNSIGNED_INT,
      pixels: new Uint32Array(matrix.K / 2 * rows).fill(0x3c003800)
    })
    const choices = [
      { layout: 1, mode: 1, pack: 3 },
      ...(this.gpu.canRender([GL.RGBA16F]) ? [{ layout: 1, mode: 1, pack: 0 }] : []),
      { layout: 0, mode: 1, pack: 3 }, { layout: 0, mode: 2, pack: 3 }
    ]
    const samples = choices.map(() => [] as number[])
    const run = async (index: number, count = rows) => {
      const choice = choices[index], weight = made[choice.layout]
      probe.layers[0].gateUp = weight
      probe.rgba4Layers = choice.layout === 0
      probe.tuned.f16Matmul = choice.mode
      probe.tuned.f16Pack = choice.pack
      const output = this.buffers.gateUp
      await probe.matmul(this.buffers.normed, weight, output, count)
      this.gpu.run({ name: "layout-checksum", program: this.gpu.program(probe.readPair(K.rmsPartial("pair"), "uX", output)),
        inputs: { uX: output, ...probe.quadInputs("uX", output) },
        uniforms: { uTexels: { int: weight.N / 4 }, uChunk: { int: Math.ceil(weight.N / 4 / K.RMS_PARTS) },
          uSplitIn: { int: 0 }, ...probe.pairIn(output) },
        output: this.buffers.rms, rect: { x: 0, y: 0, width: K.RMS_PARTS, height: count } })
      await this.gpu.read(this.buffers.rms, { x: 0, y: 0, width: K.RMS_PARTS, height: count })
    }
    try {
      for (let i = 0; i < choices.length; i++) await run(i, Math.min(rows, Math.max(64, TUNING.f16MinRows)))
      for (let index = 0; index < choices.length; index++) {
        const began = performance.now()
        await run(index)
        samples[index].push(performance.now() - began)
      }
      const best = (layout: number) => Math.min(...choices.flatMap((choice, i) => choice.layout === layout ? samples[i] : []))
      return { rgba4: best(0), bytes: best(1) }
    } finally {
      const textures = new Set<Texture>()
      for (const scratch of [probe.f16, ...probe.f16Kept.values()]) {
        if (!scratch) continue
        if (scratch.w) textures.add(scratch.w)
        if (scratch.wh2) textures.add(scratch.wh2)
        for (const texture of scratch.a ?? []) textures.add(texture)
      }
      for (const texture of textures) this.gpu.delete(texture)
    }
  }

  /** `source` (reading w's scales and zeros) as w stores them. */
  private scalesOf(w: GpuMatrix, source: string) {
    return w.packed ? K.withPackedScales(source) : source
  }

  private matrix(m: QuantMatrix): GpuMatrix {
    const gm = this.allocateMatrix(
      m.N,
      m.K,
      m.bits,
      !!m.zeros,
      m.bits === 4 && this.rgba4Layers,
      true
    )
    this.writeMatrix(gm, m, 0)
    this.storeQuants(gm)
    return gm
  }

  /** Whether an N × K 4-bit matrix fits this GPU's textures in RGBA4 (see allocateMatrix). */
  private fitsRgba4(N: number, K: number) {
    const max = this.gpu.limits.maxTextureSize
    return Math.ceil(N / 4 / max) * (K / 32) * 32 <= max
  }

  /**
   * Textures for an N × K matrix, in columns if it has more row groups than
   * a texture is tall.
   */
  private allocateMatrix(
    N: number,
    K: number,
    bits: 4 | 6 | 8,
    withZeros: boolean,
    rgba4 = false,
    /** A layer matrix: scales and zero points may be packed (see GpuMatrix.packed). */
    layer = false,
    /** 4- or 8-bit weights in RGBA32UI word texels, not byte texels. */
    words = false
  ): GpuMatrix {
    const max = this.gpu.limits.maxTextureSize
    if (rgba4) {
      const packed = layer && withZeros && TUNING.packZeros > 0
      const blocks = K / 32
      const groups = N / 4
      const rowTexels = blocks * 32
      const columns = Math.ceil(groups / max)
      const rowsPerColumn = Math.ceil(groups / columns)
      if (columns * rowTexels > max) {
        throw new GpuError(
          `A ${N}×${K} matrix doesn't fit in this GPU's textures.`
        )
      }
      return {
        N,
        K,
        bits,
        bytes: false,
        unorm: false,
        packed,
        rgba4: true,
        blocks,
        groups,
        rowsPerColumn,
        rowTexels,
        quants: this.gpu.texture(GL.RGBA4, columns * rowTexels, rowsPerColumn),
        scales: this.gpu.texture(
          packed ? GL.RGBA32UI : GL.RGBA16F,
          columns * blocks,
          rowsPerColumn
        ),
        zeros:
          withZeros && !packed
            ? this.gpu.texture(GL.RGBA8UI, columns * blocks, rowsPerColumn)
            : null
      }
    }
    // A block row: 4-bit 16 bytes, 8-bit 32, as RGBA32UI texels (1, 2) or
    // byte texels (4, 8); a Q6_K super-block row, 208 bytes: 52 byte texels
    // (their kernels split them with float math, faster than integer
    // shifts on mobile GPUs) or 13 word texels (TUNING.q6Words).
    const blocks = K / (bits === 6 ? 256 : 32)
    const groups = N / 4
    const columns = Math.ceil(groups / max)
    // Byte texels are 4× as wide: words where a row group wouldn't fit.
    const byteTexels = bits === 6 ? 52 : bits === 4 ? 4 : 8
    const bytes =
      bits === 6
        ? TUNING.q6Words <= 0
        : !words && this.bytes && columns * blocks * 4 * byteTexels <= max
    const expanded =
      bits === 6 &&
      (TUNING.q6Expanded > 0 ||
        (TUNING.q6Expanded < 0 && this.headChoice?.expanded)) &&
      columns * K <= max
    // (An expanded head the native pass reads: 16 bytes a texel.)
    const expandedWords = expanded && TUNING.nativeExpanded > 0 && this.nativeHeadOn && !this.gpu.slowZeroFill
    const texelsPerBlockRow = expanded
      ? expandedWords ? 16 : 64
      : bytes
        ? byteTexels
        : byteTexels / 4
    const rowTexels = blocks * 4 * texelsPerBlockRow
    // (Layer matrices only: the head's and embedding's kernels read integer texels.)
    const unorm = layer && bytes && bits === 4 && TUNING.unormBytes > 0
    const packed =
      layer && bytes && bits === 4 && withZeros && TUNING.packZeros > 0
    const quantFormat = expanded
      ? expandedWords ? GL.RGBA32UI : GL.RGBA8
      : !bytes
        ? GL.RGBA32UI
        : bits === 8
          ? GL.RGBA8I
          : unorm
            ? GL.RGBA8
            : GL.RGBA8UI
    const rowsPerColumn = Math.ceil(groups / columns)
    // A Q6_K head the native word head reads: its texels in that kernel's order.
    const ordered = bits === 6 && !bytes && !expanded && TUNING.q6Order > 0 && this.nativeHeadOn && !this.gpu.slowZeroFill
    if (columns * rowTexels > max) {
      throw new GpuError(
        `A ${N}×${K} matrix doesn't fit in this GPU's textures.`
      )
    }
    // A single large allocation can stall the browser GPU service while
    // it zero-initializes the embedding. Panels keep that work bounded.
    const panelRows =
      !layer && bits === 6 && this.gpu.readsPerSlice > 0 && this.gpu.slowZeroFill
        ? TUNING.embeddingPanelRows
        : 0
    const panelCount = panelRows > 0 ? Math.ceil(rowsPerColumn / panelRows) : 1
    // Allocate later panels only when their file chunk is uploaded. This
    // spreads initialization across the existing paced upload boundaries.
    const quantPanels = panelCount > 1 && panelCount + 4 <= this.gpu.limits.maxTextureUnits
      ? new Array<Texture>(panelCount) : undefined
    if (quantPanels) quantPanels[0] = this.gpu.texture(
      quantFormat, columns * rowTexels, Math.min(panelRows, rowsPerColumn)
    )
    return {
      N,
      K,
      bits,
      bytes,
      unorm,
      packed,
      rgba4: false,
      blocks,
      groups,
      rowsPerColumn,
      rowTexels,
      quants:
        quantPanels?.[0] ??
        this.gpu.texture(quantFormat, columns * rowTexels, rowsPerColumn),
      ...(quantPanels && { quantPanels }),
      ...(expanded && { q6Expanded: true }),
      ...(ordered && { q6Ordered: true }),
      ...(expandedWords && { q6ExpandedWords: true }),
      scales: this.gpu.texture(
        packed ? GL.RGBA32UI : GL.RGBA16F,
        columns * blocks,
        rowsPerColumn
      ),
      zeros:
        withZeros && !packed
          ? this.gpu.texture(GL.RGBA8UI, columns * blocks, rowsPerColumn)
          : null
    }
  }

  matvecReport: Array<{stream:boolean;ms:number}> = []

  private async chooseMatvec() {
    if (TUNING.streamMatvec >= 0 || this.tuned.streamMatvec !== undefined)
      return
    const layer = this.layers[0]
    if (layer.gateUp.rgba4 || !layer.gateUp.bytes) {
      this.tuned.streamMatvec = false
      return
    }
    const b = this.buffers,
      before = this.tuned.streamMatvec
    const scratch = this.gpu.texture(
      GL.RGBA32F,
      Math.ceil(b.partial.width / 64),
      b.partial.height
    )
    const checksum = this.gpu.program(`#version 300 es
precision highp float;precision highp int;uniform sampler2D uX;uniform int uWidth;out vec4 o;
void main(){ivec2 p=ivec2(gl_FragCoord.xy);vec4 a=vec4(0);for(int i=0;i<64;i++){int x=p.x*64+i;if(x<uWidth)a+=texelFetch(uX,ivec2(x,p.y),0);}o=a;}`)
    const samples = [[] as number[], [] as number[]]
    const sample = async (index: number) => {
      this.tuned.streamMatvec = index === 1
      for (const w of [layer.gateUp, layer.down]) {
        const slices = this.splitPartials(
          w === layer.down ? b.act : b.normed,
          w,
          1
        )
        const rect = {
          x: 0,
          y: 0,
          width: Math.ceil(w.groups / 64),
          height: slices
        }
        this.gpu.run({
          program: checksum,
          inputs: { uX: b.partial },
          uniforms: { uWidth: { int: w.groups } },
          output: scratch,
          rect
        })
        await this.gpu.read(scratch, rect)
      }
    }
    try {
      await sample(0)
      await sample(1)
      for (let round = 0; round < 2; round++)
        for (let j = 0; j < 2; j++) {
          const i = round ? 1 - j : j,
            start = performance.now()
          for (let n = 0; n < 2; n++) await sample(i)
          samples[i].push((performance.now() - start) / 2)
        }
      this.matvecReport = samples.map((times, i) => ({
        stream: i === 1,
        ms: Math.min(...times)
      }))
      // This GPU's earlier choice stands unless the other is faster in
      // every round by KEEP_MARGIN (see incumbent).
      const earlier = this.incumbent?.tuned?.streamMatvec
      this.tuned.streamMatvec = earlier == null
        ? this.matvecReport[1].ms < this.matvecReport[0].ms * 0.94
        : earlier
          ? !replaces(samples[0], samples[1], keepMargin(samples))
          : replaces(samples[1], samples[0], keepMargin(samples))
    } catch (error) {
      this.tuned.streamMatvec = before
      throw error
    } finally {
      this.gpu.delete(scratch)
    }
  }

  private expandedUpload = new Uint8Array(0)
  /** Embedding rows (by token) and the final norm's weights, for headError. */
  private embeddingSamples = new Map<number, Float32Array>()
  private finalNormValues: Float32Array | null = null
  /** Upload scratch for raw MatMulNBits rows repacked to the bytes layout (quantsFromRaw). */
  private repackUpload = new Uint32Array(0)
  /**
   * Upload scratch for texels converted on the CPU (rawRgba4's, packed
   * scales and zero points): one reused buffer, not one per matrix, whose
   * freed memory the allocator would keep resident for a while (iOS counts
   * it against a tab's 2 GB).
   */
  private uploads = new Uint32Array(0)
  private uploadScratch(words: number) {
    if (this.uploads.length < words) this.uploads = new Uint32Array(words)
    return this.uploads.subarray(0, words)
  }
  private embeddingRows = 0

  rmsReport: Array<{ parts: number; ms: number }> = []
  private packingMeasured = false
  private promptMatmulMeasured = false
  packingReport: Array<{pack:number;rows:number;ms:number;rounds?:Array<number>;matmul?:number;verificationRows?:number;verificationMs?:number}> = []

  private legacyF16: { bytes: number; kMax: number; half: boolean } | null = null

  private legacyF16Layout() {
    if (this.legacyF16) return this.legacyF16
    const mats = this.f16Mats(),
      max = this.gpu.limits.maxTextureSize
    if (!mats.length) return { bytes: 0, kMax: 0, half: false }
    let bytes = Infinity
    for (let h = max & ~3; h >= 64; h = (h >> 1) & ~3) {
      const w = Math.max(...mats.map((m) => Math.ceil(m.N / h) * (m.K / 4)))
      if (w <= max) bytes = Math.min(bytes, w * h * 8)
    }
    return (this.legacyF16 = {
      bytes,
      kMax: Math.max(0, ...mats.map((m) => m.K)),
      half: this.gpu.canRender([GL.RGBA16F])
    })
  }

  private headGrowthBytes(K = this.config.hiddenSize) {
    const columns = Math.ceil(
      (this.embeddingRows || this.config.vocabSize) /
        4 /
        this.gpu.limits.maxTextureSize
    )
    const rows =
      columns *
      Math.ceil((this.embeddingRows || this.config.vocabSize) / 4 / columns) *
      4
    return rows * (K / 256) * 48
  }

  private byteF16Budget(
    expanded = TUNING.q6Expanded >= 0
      ? TUNING.q6Expanded > 0
      : this.headChoice?.expanded === true
  ) {
    const legacy = this.legacyF16Layout()
    const copy = legacy.half && (TUNING.f16Pack < 0 || !(TUNING.f16Pack & 1))
    const one = (legacy.kMax / 4) * this.capacity * 8
    // The former automatic tuner could retain two conversion textures.
    const previous =
      legacy.bytes +
      (copy ? 2 * one : 0) +
      this.residualMemorySavings() +
      this.scoreMemorySavings(expanded)
    const activation = copy ? Math.max(1, TUNING.f16ActivationBuffers) * one : 0
    const extra = expanded ? this.headGrowthBytes() : 0
    const next =
      TUNING.f16ScratchBytesMiB < 0
        ? fundedF16Scratch(
            legacy.bytes,
            previous - legacy.bytes,
            activation,
            extra,
            legacy.kMax * 128
          )
        : TUNING.f16ScratchBytesMiB > 0
          ? TUNING.f16ScratchBytesMiB * 2 ** 20
          : legacy.bytes
    return { previous, activation, next, extra }
  }

  private residualMemorySavings() {
    if (this.capacity <= 2 * SPLIT_ROWS) return 0
    const width = this.config.hiddenSize / 4 + K.RMS_PARTS + this.config.hiddenSize / 8
    return 2 * width * (this.capacity - 2 * SPLIT_ROWS) * 16
  }

  private activationCopyFits() {
    const f = this.f16Sized()
    if (!f) return false
    const max = this.gpu.limits.maxTextureSize
    const previous = this.rgba4Layers
      ? TUNING.f16ScratchMiB * 2 ** 20 + this.residualMemorySavings()
      : this.byteF16Budget().previous
    const columns = Math.ceil(this.embedding.N / 4 / max)
    const rows = columns * Math.ceil(this.embedding.N / 4 / columns) * 4
    const weightGrowth = this.embedding.q6Expanded
      ? rows * (this.embedding.K / 256) * 48
      : 0
    const baseSplit = Math.max(
      1,
      Math.min(this.q6HeadSplit, 4, this.embedding.blocks)
    )
    const nextSplit = Math.max(
      1,
      Math.min(
        this.headChoice?.split ?? this.q6HeadSplit,
        4,
        this.embedding.blocks
      )
    )
    const head =
      weightGrowth +
      Math.max(
        0,
        headScratchBytes(
          nextSplit,
          this.canDraft ? SPLIT_ROWS : 1,
          this.logitRows,
          OUT_WIDTH
        ) -
          headScratchBytes(
            baseSplit,
            this.canDraft ? SPLIT_ROWS : 1,
            this.logitRows,
            OUT_WIDTH
          )
      )
    const activation =
      Math.max(1, TUNING.f16ActivationBuffers) *
      (f.kMax / 4) *
      this.capacity *
      8
    return fitsExpandedHead(head, previous, f.width * f.hd * 8, activation)
  }

  private async choosePromptPacking(signal?: AbortSignal) {
    signal?.throwIfAborted()
    if (!this.quadMatmuls) return
    const native = this.layerMats().some((m) => m.rgba4)
    // A stored tuning timed once more (Tuned.submissionRetime) keeps its
    // kernels: they alone are timed, so the pacing stored from this is of
    // the pipeline prompts run, not of another that timed faster here.
    const retained =
      !!this.tuned.submissionRetime &&
      this.tuned.f16Pack != null &&
      (!native || this.tuned.f16Matmul != null)
    if (
      !retained &&
      this.tuned.f16Pack != null &&
      (!native || this.tuned.f16Matmul !== undefined)
    ) {
      this.packingMeasured = true
      this.promptMatmulMeasured = native
      return
    }
    if (!native && TUNING.f16Pack >= 0) {
      this.tuned.f16Pack = TUNING.f16Pack & 3
      this.packingMeasured = true
      return
    }
    if (!native && !this.usesF16()) return
    const probeRows = Math.min(
      this.contextLength,
      this.batchSize,
      TUNING.packingProbeRows > 0
        ? Math.max(TUNING.f16MinRows, TUNING.packingProbeRows)
        : Infinity
    )
    if (probeRows < Math.max(SPLIT_ROWS + 1, TUNING.f16MinRows)) return
    const options: Array<{ pack: number; matmul?: number }> = retained
      ? [{ pack: this.tuned.f16Pack!, ...(native && { matmul: this.tuned.f16Matmul! }) }]
      : native
      ? [
          {
            pack: TUNING.f16Pack >= 0 ? TUNING.f16Pack & 3 : 3,
            matmul: TUNING.f16Matmul
          }
        ]
      : this.activationCopyFits()
        ? [{ pack: 3 }, { pack: 0 }]
        : [{ pack: 3 }, { pack: 1 }]
    if (!retained && native && TUNING.f16Matmul > 0) {
      const alternate = TUNING.f16Matmul === 2 ? 1 : 2
      options.push({ pack: options[0].pack, matmul: alternate })
    }
    if (options.length === 1 && !retained) return
    const all = this.layers,
      previous = this.tuned.f16Pack,
      previousMatmul = this.tuned.f16Matmul,
      previousParts = this.tuned.rmsParts,
      phase = this.gpu.activePhase,
      quota = this.tuned.submissionReads ?? TUNING.submissionReads
    let overlap = Math.max(quota, 2 ** 29), allowOverlap = true, trial = false, overlapFailures = 0
    const samples = options.map(() => [] as number[]),
      work = options.map(() => [] as number[])
    // Isolated matmuls mispredict layouts because their conversions wait
    // on each other. Use each operator kind's real norm/operator/FFN chain.
    // A GPU reduction consumes every hidden value before the readback.
    const kinds = new Set<string>()
    this.layers = native
      // In-place versus dequantized matmuls depend on the model's actual
      // operator mixture. One layer of each kind can badly skew that mix.
      ? all.slice(0, TUNING.tuneLayers > 0 ? TUNING.tuneLayers : all.length)
      : all.filter((layer) => {
          const kind = layer.conv ? "conv" : layer.linear ? "linear" : "attention"
          if (kinds.has(kind)) return false
          kinds.add(kind)
          return true
        })
    this.work = null
    this.gpu.configurePacing(quota, TUNING.submissionMs)
    // Only the option's own scratch held (its layout's weights, activation
    // copies): both packs' at once took an iPhone 16 Pro Max's GPU memory
    // 60 MB past LFM3's loaded 1,254 MB (48 past Qwen3.5 2B's 1,498).
    const use = (index: number) => {
      this.tuned.f16Pack = options[index].pack
      this.f16Trim()
      this.f16Scratch()
    }
    const sample = async (index: number, rows = probeRows) => {
      signal?.throwIfAborted()
      use(index)
      if (native) this.tuned.f16Matmul = options[index].matmul
      this.gpu.resetPacing(true)
      this.gpu.primePresentation(trial && allowOverlap ? this.gpu.presentationAdvice.cap : TUNING.presentationFrames, trial ? overlap : quota, trial && allowOverlap)
      this.gpu.setPhase("load")
      this.reset()
      const tokens = Array.from(
        { length: rows },
        (_, i) => (i * 7919 + 13) % this.config.vocabSize
      )
      await this.forwardHidden(tokens, { fullRows: true })
      signal?.throwIfAborted()
      const width = Math.ceil(this.config.hiddenSize / 4 / 64)
      await this.gpu.runAsync({
        program: this.gpu.program(`#version 300 es
precision highp float;precision highp int;uniform sampler2D uX;uniform int uWidth;out vec4 o;
void main(){ivec2 p=ivec2(gl_FragCoord.xy);vec4 a=vec4(0);for(int i=0;i<64;i++){int x=p.x*64+i;if(x<uWidth)a+=texelFetch(uX,ivec2(x,p.y),0);}o=a;}`),
        inputs: { uX: this.buffers.residual[0] },
        uniforms: { uWidth: { int: this.config.hiddenSize / 4 } },
        output: this.buffers.rms,
        rect: { x: 0, y: 0, width, height: rows },
        work: 64
      })
      await this.gpu.read(this.buffers.rms, { x: 0, y: 0, width, height: rows })
      signal?.throwIfAborted()
      // The normal frame controller rolls the trial back immediately if
      // this queue prevents presentation. Never retry it in later variants.
      if (trial && (this.gpu.readsPerSlice < overlap || !this.gpu.presentationAdvice.relaxation || this.gpu.presentationAdvice.window < this.gpu.presentationAdvice.cap)) {
        overlapFailures++
        if (overlapFailures >= 2 || !this.gpu.presentationAdvice.relaxation) {
          overlap = quota
          allowOverlap = false
        }
      }
    }
    try {
      for (let i = 0; i < options.length; i++)
        await sample(i, Math.min(probeRows, Math.max(64, TUNING.f16MinRows)))
      const verification = options.map(() => NaN)
      const verificationRows = Math.min(probeRows, Math.max(64, TUNING.f16MinRows))
      const timed = async (i: number) => {
        // (The scratch made before the clock starts.)
        use(i)
        const start = performance.now(), beforeWork = this.gpu.submittedWork,
          beforeCompile = this.gpu.compiled?.count ?? 0
        await sample(i)
        samples[i].push(performance.now() - start)
        work[i].push(this.gpu.submittedWork - beforeWork)
        return (this.gpu.compiled?.count ?? 0) !== beforeCompile
      }
      // The short warm-up compiles the same kernels and touches their full
      // weight scratch. Observe compilation rather than always repeating
      // every full-batch candidate before measuring it.
      trial = true
      let compiled = false
      for (let i = 0; i < options.length; i++) compiled = await timed(i) || compiled
      let verifiedWinner = -1
      if (compiled) {
        // An unanticipated full-batch specialization gets the conservative
        // complete warm-up and two reversed full-size pairs.
        trial = false
        for (let i = 0; i < options.length; i++) await sample(i)
        overlap = Math.max(quota, 2 ** 29)
        allowOverlap = true
        overlapFailures = 0
        for (const row of samples) row.length = 0
        for (const row of work) row.length = 0
        trial = true
        for (let i = 0; i < options.length; i++) await timed(i)
      } else if (options.length === 2 && verificationRows < probeRows) {
        const winner = samples[1][0] < samples[0][0] ? 1 : 0, other = 1 - winner
        if (samples[winner][0] < samples[other][0] * 0.8) {
          let shortCompiled = false
          // Verify a large full-batch win at the shorter size, in reverse
          // order. A disagreement or a close result retains both full pairs.
          for (let i = options.length - 1; i >= 0; i--) {
            const start = performance.now(), before = this.gpu.compiled?.count ?? 0
            await sample(i, verificationRows)
            verification[i] = performance.now() - start
            shortCompiled ||= (this.gpu.compiled?.count ?? 0) !== before
          }
          if (!shortCompiled && verification[winner] < verification[other] * 0.9)
            verifiedWinner = winner
        }
      }
      if (verifiedWinner < 0) {
        for (let i = options.length - 1; i >= 0; i--) await timed(i)
        // Two pairs settle it where a timing is long (a phone's pipeline is
        // half a second, its two ways a tenth apart). On a fast GPU it is
        // 26 ms, timed to within a few ms, and the ways are a millisecond
        // apart: winning both pairs was a coin's toss there (1 fresh
        // tuning in 4 on an M3 Ultra), kept for every later session, and
        // real prompts read 9% slower with the one they are slower with.
        // So more pairs, each in the reverse order of the last, while the
        // timings so far and one more pair fit PACKING_PROBE_MS, up to
        // PACKING_PROBE_PAIRS: the other way must win them all.
        const spent = () => samples.reduce((sum, row) => sum + row.reduce((a, ms) => a + ms, 0), 0)
        const pair = () => samples.reduce((sum, row) => sum + row[row.length - 1], 0)
        while (samples[0].length < PACKING_PROBE_PAIRS && spent() + pair() <= PACKING_PROBE_MS) {
          const forward = samples[0].length % 2 === 0
          for (let k = 0; k < options.length; k++) await timed(forward ? k : options.length - 1 - k)
        }
      }
      this.packingReport = options.map((setting, i) => ({
        ...setting,
        rows: probeRows,
        ms: Math.min(...samples[i]),
        rounds: samples[i].map((ms) => Math.round(ms * 10) / 10),
        ...(Number.isFinite(verification[i]) && { verificationRows, verificationMs: verification[i] })
      }))
      // A clock or compiler transition must not pick the next session's
      // kernel from one favorable sample. Require reversed verification;
      // close results retain both full-size pairs.
      // The standing way: the first, or this GPU's earlier choice
      // (incumbent), which another replaces only by KEEP_MARGIN.
      const earlier = this.incumbent?.tuned
      const standing = earlier?.f16Pack == null ? 0 : Math.max(0, options.findIndex((o) =>
        o.pack === earlier.f16Pack && (!native || o.matmul === earlier.f16Matmul)))
      // (A cold first round counts neither as a loss nor as timer noise: see warmRounds.)
      const warm = warmRounds(samples)
      const margin = earlier?.f16Pack == null ? 0.98 : keepMargin(warm)
      const eligible = this.packingReport.filter((_, i) =>
        i === standing || (verifiedWinner >= 0
          ? i === verifiedWinner && (earlier?.f16Pack == null || replaces(warm[i], warm[standing], margin))
          : replaces(warm[i], warm[standing], margin))
      )
      const best = eligible.reduce((a, b) => (b.ms < a.ms ? b : a))
      const selected = this.packingReport.indexOf(best)
      this.tuned.f16Pack = options[selected].pack
      if (native) this.tuned.f16Matmul = options[selected].matmul
      // Keep the original reduction shape. The alternate did not win
      // consistently, and repeating four complete pipelines delayed ready.
      this.tuned.rmsParts = this.parts
      this.tuned.submissionOverlap = allowOverlap ? overlap : 0
      this.tuned.submissionWindow = allowOverlap
        ? calibratedWindow(TUNING.presentationFrames, this.gpu.presentationAdvice.cap, this.packingReport[selected].ms, this.gpu.expectedFrameMs)
        : TUNING.presentationFrames
      this.tuned.submissionReads = calibratedReadQuota(
        TUNING.submissionReads,
        Math.min(...work[selected]),
        this.packingReport[selected].ms,
        this.gpu.expectedFrameMs
      )
      // A timing past two frames may be a fast GPU held up by other work
      // (a game beside the page): the next load times it once more.
      this.tuned.submissionMs = this.packingReport[selected].ms
      this.tuned.submissionRetime = allowOverlap && this.packingReport[selected].ms > this.gpu.expectedFrameMs * 2 ? 1 : 0
      this.packingMeasured = true
      this.promptMatmulMeasured = native
    } catch (error) {
      this.tuned.f16Pack = previous
      this.tuned.f16Matmul = previousMatmul
      this.tuned.rmsParts = previousParts
      throw error
    } finally {
      this.layers = all
      this.work = null
      this.reset()
      this.gpu.configurePacing(
        this.tuned.submissionReads ?? quota,
        TUNING.submissionMs
      )
      this.gpu.setPhase(phase)
      this.f16Trim()
    }
  }

  /** Measured head layout, never selected by vendor or device name. */
  headChoice: { expanded: boolean; split: number } | null = null
  headReport: Array<{expanded:boolean;split:number;ms:number}> = []

  private canExpandHead(
    K: number,
    split = this.q6HeadSplit,
    expanded = true
  ) {
    // A device using the in-place layout has no large dequantization
    // allocation to trade for an expanded head. Preserve its memory use.
    if (this.rgba4Layers || !this.quadMatmuls || !this.usesF16()) return false
    const max = this.gpu.limits.maxTextureSize,
      mats = this.f16Mats()
    const columns = Math.ceil(
      (this.embeddingRows || this.config.vocabSize) / 4 / max
    )
    if (columns * K > max || !mats.length) return false
    const budget = this.byteF16Budget(expanded)
    const { previous, activation, next } = budget
    const extra = expanded ? this.headGrowthBytes(K) : 0
    const baselineSplit = Math.max(1, Math.min(this.q6HeadSplit, 4, K / 256))
    const nextSplit = Math.max(1, Math.min(split, 4, K / 256))
    const extraScratch = Math.max(
      0,
      headScratchBytes(
        nextSplit,
        this.canDraft ? SPLIT_ROWS : 1,
        this.logitRows,
        OUT_WIDTH
      ) -
        headScratchBytes(
          baselineSplit,
          this.canDraft ? SPLIT_ROWS : 1,
          this.logitRows,
          OUT_WIDTH
        )
    )
    return fitsExpandedHead(extra + extraScratch, previous, next, activation)
  }

  private async chooseHead(piece: QuantMatrix) {
    if (
      Math.ceil(
        (this.embeddingRows || this.config.vocabSize) /
          4 /
          this.gpu.limits.maxTextureSize
      ) *
        piece.K >
      this.gpu.limits.maxTextureSize
    ) {
      this.headChoice = { expanded: false, split: this.q6HeadSplit }
      return
    }
    const count = Math.max(1, Math.ceil(8192 / piece.N))
    const matrix: QuantMatrix = { ...piece, N: piece.N * count }
    const original = this.embedding,
      originalScratch = this.headScratch,
      vocab = this.config.vocabSize,
      scaled = this.headScaled
    const made: GpuMatrix[] = []
    // Compare storage formats at the existing split. Probing larger split
    // buffers increased startup work and memory without a measured gain.
    const split = Math.max(1, Math.min(this.q6HeadSplit, 4, piece.K / 256))
    const settings = [
      { expanded: false, split },
      { expanded: true, split }
    ].filter(
      (choice) =>
        !choice.expanded || this.canExpandHead(piece.K, choice.split, true)
    )
    const samples = settings.map(() => [] as number[])
    const phase = this.gpu.activePhase
    const quota = this.tuned.submissionReads ?? TUNING.submissionReads
    try {
      // Measure the heads' GPU work: unpaced, three back to back between
      // waits (a pace's or a read's fixed wait, 0.4 to 3.6 ms, swamped the
      // difference on an M3 Ultra and a Pixel 10 Pro), with nonzero hidden
      // values (clear-color inputs can bias mobile GPU timings).
      this.gpu.setPhase("decode")
      this.gpu.configurePacing(0, TUNING.submissionMs)
      const input = this.buffers.lastNormed,
        seed = new Uint32Array(input.width * 4)
      for (let i = 0; i < seed.length; i++)
        seed[i] =
          0x3000 |
          ((i * 17) & 1023) |
          ((0x3000 | ((i * 31) & 1023) | (i & 1 ? 0x8000 : 0)) << 16)
      this.gpu.write(
        input,
        { x: 0, y: 0, width: input.width, height: 1 },
        { format: GL.RGBA_INTEGER, type: GL.UNSIGNED_INT, pixels: seed }
      )
      for (const expanded of [false, true]) {
        this.headChoice = { expanded, split }
        const gm = this.allocateMatrix(matrix.N, matrix.K, 6, false)
        for (let i = 0; i < count; i++) this.writeMatrix(gm, piece, i * piece.N)
        // Match the panel shader used for the complete embedding (where
        // it has panels: not on WebGPU, whose probe in panels timed one-row
        // bands no real head runs, and picked the head 3× slower on a Pixel).
        if (this.gpu.slowZeroFill) gm.quantPanels ??= [gm.quants]
        made.push(gm)
        await this.gpu.pace(0, true)
      }
      this.config.vocabSize = matrix.N
      this.headScaled = false
      this.setSeen([], matrix.N)
      const sample = async (index: number) => {
        this.headChoice = settings[index]
        this.gpu.setPhase("decode")
        this.embedding = made[settings[index].expanded ? 1 : 0]
        await this.head(0, 1)
      }
      for (let i = 0; i < settings.length; i++) await sample(i)
      for (let round = 0; round < (this.incumbent?.head ? 3 : 2); round++)
        for (let j = 0; j < settings.length; j++) {
          const i = round ? settings.length - 1 - j : j
          await this.gpu.finish()
          const start = performance.now()
          for (let n = 0; n < 3; n++) await sample(i)
          await this.gpu.finish()
          samples[i].push((performance.now() - start) / 3)
        }
      this.headReport = settings.map((setting, i) => ({
        ...setting,
        ms: Math.min(...samples[i])
      }))
      const best = this.headReport.reduce((a, b) => (b.ms < a.ms ? b : a))
      const chosen = best
      const base = this.headReport[0]
      // This GPU's earlier head stands unless the other is faster in every
      // round by KEEP_MARGIN (see incumbent).
      const earlier = this.incumbent?.head
      const standing = earlier ? settings.findIndex((s) => s.expanded === earlier.expanded && s.split === earlier.split) : -1
      if (standing >= 0) {
        const other = this.headReport.indexOf(best)
        const keep = other === standing || !replaces(samples[other], samples[standing], keepMargin(samples))
        const pick = settings[keep ? standing : other]
        this.headChoice = { expanded: pick.expanded, split: pick.split }
      } else
      // The native expanded head costs a fifth more memory: taken only when
      // it's clearly faster (a Pixel 10 Pro's: 43%; an M-series Mac's timed
      // 12% faster on the probe's slice but ran 1.5% slower whole).
      this.headChoice =
        chosen.ms < base.ms * (chosen.expanded && this.nativeHeadOn && TUNING.nativeExpanded > 0 ? EXPANDED_MARGIN : 0.94)
          ? { expanded: chosen.expanded, split: chosen.split }
          : { expanded: false, split }
    } finally {
      this.gpu.configurePacing(quota, TUNING.submissionMs)
      this.gpu.setPhase(phase)
      this.embedding = original
      if (this.headScratch !== originalScratch) {
        if (this.headScratch) this.gpu.delete(this.headScratch)
        this.headScratch = originalScratch
      }
      this.config.vocabSize = vocab
      this.headScaled = scaled
      for (const gm of made)
        for (const t of [
          ...(gm.quantPanels ?? [gm.quants]),
          gm.scales,
          gm.zeros
        ])
          if (t) this.gpu.delete(t)
    }
  }

  private weightSource(w: GpuMatrix, source: string) {
    return w.quantPanels
      ? weightPanelSource(
          source,
          w.quantPanels[0].height,
          w.quantPanels.length,
          !!w.q6Expanded
        )
      : source
  }

  private headWeightSource(w: GpuMatrix, source: string) {
    if (!w.quantPanels) return source
    if (w.q6Expanded)
      source = source
        .replace("texelFetch(s,p,0)", "texelFetch(s,p-ivec2(0,uWeightRow0),0)")
        .replace("vec4 q6bytes(", "uniform int uWeightRow0;\nvec4 q6bytes(")
    const read = `\n${w.q6Expanded ? "" : "uniform int uWeightRow0;"}\n${w.q6Expanded ? "vec4" : "uvec4"} weightTexel(ivec2 p, int level) { return texelFetch(uW, p - ivec2(0, uWeightRow0), level); }\n`
    return source
      .replaceAll("texelFetch(uW,", "weightTexel(")
      .replace(
        w.q6Expanded ? "uniform sampler2D uW;" : "uniform usampler2D uW;",
        `${w.q6Expanded ? "uniform sampler2D uW;" : "uniform usampler2D uW;"}${read}`
      )
  }

  /** Draw each vocabulary region with just its own Q6_K panel bound. */
  private async weightHead(
    w: GpuMatrix,
    pass: Pass,
    rows: number,
    split: number
  ) {
    if (!w.quantPanels)
      return this.gpu.runAsync(pass, Math.max(1, this.logitRows / 4))
    for (const { panel, row0, rect } of weightHeadRegions({
      groups: w.groups,
      rowsPerColumn: w.rowsPerColumn,
      panelRows: w.quantPanels[0].height,
      heights: w.quantPanels.map((t) => t.height),
      planes: split * rows,
      logitRows: this.logitRows,
      width: OUT_WIDTH
    })) {
      await this.gpu.runAsync({
        ...pass,
        inputs: { ...pass.inputs, uW: w.quantPanels[panel] },
        uniforms: { ...pass.uniforms, uWeightRow0: { int: row0 } },
        rect,
        // Stable work cost across panel boundaries: raising a measured
        // GPU quota must also increase head work between fences.
        work:
          Math.max(1, TUNING.submissionReads) /
          (Math.max(1, TUNING.headPaceRows) * OUT_WIDTH)
      })
    }
  }

  private weightInputs(w: GpuMatrix): Record<string, Texture> {
    return Object.fromEntries(
      (w.quantPanels ?? [w.quants]).map((t, i) => [i ? `uWP${i}` : "uW", t])
    )
  }

  /** Writes `m`'s rows into rows [row0, row0 + m.N) of `gm` (row0 a multiple of 4). */
  private writeMatrix(gm: GpuMatrix, m: QuantMatrix, row0: number) {
    // Raw MatMulNBits rows go into RGBA4 as they are (16-byte runs moved
    // into place; see rawRgba4), and into 4-bit byte texels through a GPU
    // pass (K.rawQ4Bytes: an ONNX model's ~0.8 GB repacked on the CPU took
    // LFM3 1.7B's load 300 ms on an M3 Ultra); other layouts get the rows
    // repacked here.
    const gpuRaw = !!m.raw && !gm.rgba4 && m.bits === 4 && !gm.quantPanels && TUNING.rawUploadGpu > 0
    if (m.raw && !gm.rgba4 && !gpuRaw) {
      const words = (m.N * m.K) / 8
      if (this.repackUpload.length < words) this.repackUpload = new Uint32Array(words)
      m = { ...m, quants: quantsFromRaw(m.raw, m.N, m.K, this.repackUpload), raw: undefined }
    }
    const { blocks, rowTexels, rowsPerColumn } = gm
    const g0 = row0 / 4
    const groups = m.N / 4
    // Words of one row group: its quant texels' bytes over 4.
    const words = gpuRaw ? blocks * 16 : m.quants.length / groups
    const stage = this.buffers?.act
    const gpuExpand =
      gm.q6Expanded &&
      TUNING.q6UploadGpu > 0 &&
      stage?.internal === GL.RGBA32UI &&
      stage.width >= blocks * 52
    for (let g = g0; g < g0 + groups;) {
      const column = Math.floor(g / rowsPerColumn)
      const y = g - column * rowsPerColumn
      const panelRows = gm.quantPanels?.[0].height ?? rowsPerColumn
      const panel = Math.floor(y / panelRows)
      if (gm.quantPanels && !gm.quantPanels[panel])
        gm.quantPanels[panel] = this.gpu.texture(
          gm.quants.internal, gm.quants.width,
          Math.min(panelRows, rowsPerColumn - panel * panelRows)
        )
      const run = Math.min(
        g0 + groups - g,
        rowsPerColumn - y,
        panelRows - (y % panelRows),
        // Keep CPU upload scratch small even for a wide embedding chunk.
        gpuExpand
          ? stage.height
          : gm.q6Expanded
            ? Math.max(1, Math.floor((128 << 10) / (blocks * 4 * 256)))
            : Infinity
      )
      const local = g - g0
      const quants = gpuRaw ? new Uint32Array(0) : m.quants.subarray(local * words, (local + run) * words)
      if (gpuRaw) {
        // Through a staging texture of raw rows (a texel a row's block), a
        // band at a time, rearranged into place on the GPU.
        const raw = m.raw!
        const rowBytes = m.K / 2
        if (this.rawStage && this.rawStage.width !== blocks) {
          this.gpu.delete(this.rawStage)
          this.rawStage = null
        }
        const stage = (this.rawStage ??= this.gpu.texture(GL.RGBA32UI, blocks, Math.min(1024, this.gpu.limits.maxTextureSize & ~3)))
        const band = stage.height / 4
        for (let r0 = 0; r0 < run; r0 += band) {
          const h = Math.min(band, run - r0)
          const start = raw.byteOffset + (local + r0) * 4 * rowBytes
          const bytes = h * 4 * rowBytes
          const pixels = start & 3
            ? new Uint32Array(raw.slice((local + r0) * 4 * rowBytes, (local + r0) * 4 * rowBytes + bytes).buffer)
            : new Uint32Array(raw.buffer, start, bytes / 4)
          this.gpu.write(
            stage,
            { x: 0, y: 0, width: blocks, height: 4 * h },
            { format: GL.RGBA_INTEGER, type: GL.UNSIGNED_INT, pixels }
          )
          const rect = { x: column * rowTexels, y: y + r0, width: rowTexels, height: h }
          this.gpu.run({
            name: "upload",
            program: this.gpu.program(K.rawQ4Bytes(gm.unorm, !gm.bytes)),
            inputs: { uRaw: stage },
            uniforms: { uX0: { int: rect.x }, uY0: { int: rect.y } },
            output: gm.quants,
            rect
          })
        }
      } else if (gpuExpand) {
        this.gpu.write(
          stage,
          { x: 0, y: 0, width: blocks * 52, height: run },
          {
            format: GL.RGBA_INTEGER,
            type: GL.UNSIGNED_INT,
            pixels: quants
          }
        )
        const rect = {
          x: column * rowTexels,
          y: gm.quantPanels ? y % panelRows : y,
          width: rowTexels,
          height: run
        }
        this.gpu.run({
          name: "upload",
          program: this.gpu.program(gm.q6ExpandedWords ? K.EXPAND_Q6_WORDS : K.EXPAND_Q6_BYTES),
          inputs: { uNative: stage },
          uniforms: {
            uX0: { int: rect.x },
            uY0: { int: rect.y }
          },
          output: gm.quantPanels?.[panel] ?? gm.quants,
          rect
        })
      } else if (gm.q6Ordered) {
        // Through a staging texture, rearranged into place on the GPU.
        const stage = (this.orderStage = this.orderStage?.width === rowTexels
          ? this.orderStage
          : this.gpu.texture(GL.RGBA32UI, rowTexels, Math.min(256, rowsPerColumn)))
        for (let r0 = 0; r0 < run; r0 += stage.height) {
          const h = Math.min(stage.height, run - r0)
          this.gpu.write(
            stage,
            { x: 0, y: 0, width: rowTexels, height: h },
            { format: GL.RGBA_INTEGER, type: GL.UNSIGNED_INT, pixels: quants.subarray(r0 * words, (r0 + h) * words) }
          )
          const rect = { x: column * rowTexels, y: y + r0, width: rowTexels, height: h }
          this.gpu.run({
            name: "upload",
            program: this.gpu.program(K.ORDER_Q6_WORDS),
            inputs: { uNative: stage },
            uniforms: { uX0: { int: rect.x }, uY0: { int: rect.y } },
            output: gm.quants,
            rect
          })
          // Sent now: the next chunk's write then goes straight to the queue
          // (behind an unsent pass reading the stage, it's copied twice).
          this.gpu.flush()
        }
      } else if (gm.q6Expanded) {
        const needed = run * blocks * 4 * 256
        if (this.expandedUpload.length < needed)
          this.expandedUpload = new Uint8Array(needed)
        const data = expandQ6(m, local, run, this.expandedUpload)
        this.gpu.write(
          gm.quantPanels?.[panel] ?? gm.quants,
          {
            x: column * rowTexels,
            y: gm.quantPanels ? y % panelRows : y,
            width: rowTexels,
            height: run
          },
          gm.q6ExpandedWords
            ? { format: GL.RGBA_INTEGER, type: GL.UNSIGNED_INT, pixels: expandedWordsOrder(data.quants, run, blocks) }
            : { format: GL.RGBA, type: GL.UNSIGNED_BYTE, pixels: data.quants }
        )
      } else if (gm.rgba4) {
        const t = performance.now()
        const pixels = m.raw
          ? rawRgba4(m.raw, m.K, local * 4, run * 4, this.uploadScratch(run * 4 * blocks * 4))
          : q4WordsToRgba4(quants)
        this.loadMs.convert += performance.now() - t
        try {
          this.gpu.write(
            gm.quants,
            { x: column * rowTexels, y, width: rowTexels, height: run },
            { format: GL.RGBA, type: GL.UNSIGNED_SHORT_4_4_4_4, pixels }
          )
        } finally {
          // A private upload temporary, never caller data: q4WordsToRgba4's
          // freed now (rawRgba4's is the reused upload scratch).
          if (!m.raw) releaseOwnedBuffer(pixels.buffer)
        }
      } else
        this.gpu.write(
          gm.quantPanels?.[panel] ?? gm.quants,
          {
            x: column * rowTexels,
            y: gm.quantPanels ? y % panelRows : y,
            width: rowTexels,
            height: run
          },
          {
            format: gm.unorm ? GL.RGBA : GL.RGBA_INTEGER,
            type: !gm.bytes
              ? GL.UNSIGNED_INT
              : m.bits === 8
                ? GL.BYTE
                : GL.UNSIGNED_BYTE,
            pixels: !gm.bytes
              ? quants
              : m.bits !== 8
                ? new Uint8Array(
                    quants.buffer,
                    quants.byteOffset,
                    quants.byteLength
                  )
                : new Int8Array(
                    quants.buffer,
                    quants.byteOffset,
                    quants.byteLength
                  )
          }
        )
      if (gm.packed && m.zeros) {
        // Each texel: 4 scales then 4 zero points, as halves.
        const packStart = performance.now()
        const n = run * blocks * 4
        const at = local * blocks * 4
        const texels = this.uploadScratch(n)
        const scales = m.scales
        const zeros = m.zeros
        for (let t = 0, a = at; t < n; t += 4, a += 4) {
          texels[t] = scales[a] | (scales[a + 1] << 16)
          texels[t + 1] = scales[a + 2] | (scales[a + 3] << 16)
          texels[t + 2] = HALF_INT[zeros[a]] | (HALF_INT[zeros[a + 1]] << 16)
          texels[t + 3] = HALF_INT[zeros[a + 2]] | (HALF_INT[zeros[a + 3]] << 16)
        }
        this.loadMs.convert += performance.now() - packStart
        this.gpu.write(
          gm.scales,
          { x: column * blocks, y, width: blocks, height: run },
          { format: GL.RGBA_INTEGER, type: GL.UNSIGNED_INT, pixels: texels }
        )
      } else
        this.gpu.write(
          gm.scales,
          { x: column * blocks, y, width: blocks, height: run },
          {
            format: GL.RGBA,
            type: GL.HALF_FLOAT,
            pixels: m.scales.subarray(
              local * blocks * 4,
              (local + run) * blocks * 4
            )
          }
        )
      if (gm.zeros && m.zeros) {
        this.gpu.write(
          gm.zeros,
          { x: column * blocks, y, width: blocks, height: run },
          {
            format: GL.RGBA_INTEGER,
            type: GL.UNSIGNED_BYTE,
            pixels: m.zeros.subarray(
              local * blocks * 4,
              (local + run) * blocks * 4
            )
          }
        )
      }
      g += run
    }
  }

  private layer(w: LayerWeights): GpuLayer {
    const c = this.config
    const layer: GpuLayer = {
      type: w.type,
      operatorNorm: this.vector(w.operatorNorm),
      ...(w.postOperatorNorm && {
        postOperatorNorm: this.vector(w.postOperatorNorm)
      }),
      ffnNorm: this.vector(w.ffnNorm),
      ...(w.postFfnNorm && { postFfnNorm: this.vector(w.postFfnNorm) }),
      gateUp: this.matrix(w.gateUp),
      down: this.matrix(w.down)
    }
    if (w.conv) {
      // Channel-major taps → texel (c / 4, k).
      const width = c.convKernel
      const D = c.hiddenSize
      const taps = new Float32Array(width * D)
      for (let ch = 0; ch < D; ch++) {
        for (let k = 0; k < width; k++)
          taps[k * D + ch] = w.conv.kernel[ch * width + k]
      }
      layer.conv = {
        inProj: this.matrix(w.conv.inProj),
        outProj: this.matrix(w.conv.outProj),
        kernel: this.vector(taps, width),
        // (As tall as convSmall: see convFused.)
        state: [0, 1].map(() => this.f32(D, this.convRows())) as [Texture, Texture],
        prefixes: this.f32(D, (width - 1) * SPLIT_ROWS)
      }
    }
    if (w.linear) {
      const lin = this.linDims()
      if (!lin) throw new Error("The model's config has no Gated DeltaNet shapes.")
      const width = lin.l.convKernel
      const C = lin.C
      // Channel-major taps → texel (c / 4, k).
      const taps = new Float32Array(width * C)
      for (let ch = 0; ch < C; ch++)
        for (let k = 0; k < width; k++)
          taps[k * C + ch] = w.linear.conv[ch * width + k]
      layer.linear = {
        inProj: this.matrix(w.linear.inProj),
        ba: this.matrix(w.linear.ba),
        outProj: this.matrix(w.linear.outProj),
        kernel: this.vector(taps, width),
        convState: [0, 1].map(() => this.f32(C, width - 1)) as [Texture, Texture],
        state: [0, 1].map(() =>
          this.f32(lin.V, lin.l.keyHeadDim)
        ) as [Texture, Texture],
        a: this.vector(w.linear.a),
        dtBias: this.vector(w.linear.dtBias),
        norm: this.vector(w.linear.norm)
      }
    }
    if (w.attention) {
      const kvDims = c.numKvHeads * c.headDim
      // A sliding-window layer only ever reads its window back: it keeps the
      // window plus one batch (written before the batch's queries read).
      const window = c.slidingWindow + this.capacity
      const ring =
        this.ringCache &&
        w.type === "sliding_attention" &&
        window < this.contextLength
          ? window
          : 0
      layer.attention = {
        qkvg: this.matrix(w.attention.qkvg),
        o: this.matrix(w.attention.o),
        qNorm: this.vector(w.attention.qNorm),
        kNorm: this.vector(w.attention.kNorm),
        kv: ring ? this.h2(2 * kvDims, ring) : this.kvTexture(this.kvPositions),
        ring
      }
    }
    return layer
  }

  /**
   * What a later prompt needs to resume at a cache position: the
   * convolutions' state, and the sliding-window layers' last window of keys
   * and values (their ring may have moved on since). Full-attention layers
   * keep every position, so they need nothing saved. At the end of a
   * prompt, also the last token's hidden row: the same prompt again then
   * needs only the output head (see resume).
   */
  private snapshots = new Map<number, Snapshot>()

  private freeSnapshot(saved: Snapshot) {
    saved.conv.forEach((t) => this.gpu.delete(t))
    saved.linear.forEach((t) => this.gpu.delete(t))
    saved.windows.forEach((kv) => this.gpu.delete(kv))
    if (saved.hidden) this.gpu.delete(saved.hidden)
  }

  /** Row `row` of residual-shaped texture `from` into row 0 of `to` (whole rows). */
  private copyRow(from: Texture, row: number, to: Texture) {
    this.gpu.run({
      program: this.gpu.program(K.COPY_ROWS),
      inputs: { uX: from },
      uniforms: { uOff: { int: row } },
      output: to,
      rect: { x: 0, y: 0, width: from.width, height: 1 }
    })
  }

  /** A copy of row `row` of the last batch's hidden state (its whole residual row). */
  private copyHidden(row: number) {
    if (this.lastRowOnly && row !== 0)
      throw new Error("The last batch kept only row 0 (see TUNING.lastRowOnly).")
    const r = this.buffers.residual[0]
    const copy = this.gpu.texture(GL.RGBA32F, r.width, 1)
    this.gpu.run({
      program: this.gpu.program(K.COPY_ROWS),
      inputs: { uX: r },
      uniforms: { uOff: { int: row } },
      output: copy
    })
    return copy
  }

  /**
   * Copies cache positions [from, to) between a ring of `fromRows` rows and
   * one of `toRows` (position p at row p % rows), in runs that don't wrap
   * in the target.
   */
  private copyPositions(
    source: Texture,
    fromRows: number,
    target: Texture,
    toRows: number,
    from: number,
    to: number
  ) {
    for (let a = from; a < to;) {
      const row0 = a % toRows
      const run = Math.min(to - a, toRows - row0)
      this.gpu.run({
        program: this.gpu.program(K.COPY_POSITIONS),
        inputs: { uX: source },
        uniforms: {
          uPos0: { int: a },
          uRow0: { int: row0 },
          uMod: { int: fromRows }
        },
        output: target,
        rect: { x: 0, y: row0, width: target.width, height: run }
      })
      a += run
    }
  }

  /** Most snapshots kept: MAX_SNAPSHOTS, or TUNING.linearSnapshots for Gated DeltaNet models. */
  get snapshotLimit() {
    return this.canDraft ? MAX_SNAPSHOTS : Math.max(1, TUNING.linearSnapshots)
  }

  /** Retains the reusable prefix and conversational anchor within the existing limit. */
  private dropSnapshots(keep: number, replacing?: SnapshotRole) {
    const points = [...this.snapshots].map(([position, saved]) => ({ position, anchor: saved.anchor, prefix: saved.prefix }))
    for (const position of checkpointsToDrop(points, keep, replacing)) {
      const saved = this.snapshots.get(position)
      if (saved) this.freeSnapshot(saved)
      this.snapshots.delete(position)
    }
  }

  /**
   * Saves what's needed to resume at the current position (see
   * `snapshots`), so a later prompt that shares the tokens so far reads only
   * what's new. Keeps the latest snapshotLimit, the oldest freed before a
   * new one is made (so no more than that are ever held). The stable prefix
   * and latest conversational anchor have priority over discretionary batch
   * points; with room for only one, conversational behavior is unchanged.
   * With `hiddenRow` (the row of the last batch
   * holding the last token), keeps that token's hidden state too, so
   * `resume` can draw the next token's logits without reading it again.
   */
  snapshot(hiddenRow?: number, anchor: boolean | "prefix" = false) {
    if (anchor === "prefix" && this.snapshotLimit < 2) return
    if (this.guessAt !== null)
      throw new Error("Snapshot while a guessed token is pending.")
    const at = this.position
    const had = this.snapshots.get(at)
    if (had) {
      if (hiddenRow !== undefined && !had.hidden) had.hidden = this.copyHidden(hiddenRow)
      if (anchor) this.keepSnapshot(at, had, anchor)
      return
    }
    // Do not evict either useful anchor just to save a discretionary point.
    if (!anchor && [...this.snapshots.values()].filter(saved => saved.anchor || saved.prefix).length >= this.snapshotLimit) return
    this.dropSnapshots(this.snapshotLimit - 1, anchor === true ? "anchor" : anchor || undefined)
    const W = this.config.slidingWindow
    const saved: Snapshot = {
      conv: [] as Array<Texture>,
      windows: [] as Array<Texture>,
      linear: [] as Array<Texture>,
      ...(hiddenRow !== undefined && { hidden: this.copyHidden(hiddenRow) })
    }
    for (const layer of this.layers) {
      if (layer.conv) {
        const state = layer.conv.state[0]
        const copy = this.gpu.texture(GL.RGBA32F, state.width, state.height)
        this.gpu.run({
          program: this.gpu.program(K.COPY),
          inputs: { uX: state },
          output: copy
        })
        saved.conv.push(copy)
      }
      if (layer.linear) {
        for (const state of [layer.linear.convState[0], layer.linear.state[0]]) {
          const copy = this.gpu.texture(GL.RGBA32F, state.width, state.height)
          this.gpu.run({
            program: this.gpu.program(K.COPY),
            inputs: { uX: state },
            output: copy
          })
          saved.linear.push(copy)
        }
      }
      const a = layer.attention
      if (a?.ring) {
        const kv = this.gpu.texture(GL.RGBA32UI, a.kv.width, W)
        this.copyPositions(a.kv, a.ring, kv, W, Math.max(0, at - W), at)
        saved.windows.push(kv)
      }
    }
    this.keepSnapshot(at, saved, anchor)
  }

  /** Adds `saved` as the snapshot at `at` (the anchor, with `anchor`), then drops the oldest past the limit. */
  private keepSnapshot(at: number, saved: Snapshot, anchor: boolean | "prefix") {
    this.snapshots.set(at, saved)
    if (anchor) {
      const role = anchor === true ? "anchor" : "prefix"
      for (const s of this.snapshots.values()) s[role] = false
      saved[role] = true
    }
    // (An anchor kept past the limit goes once it's no longer the anchor.)
    this.dropSnapshots(this.snapshotLimit)
  }

  /** The selected retained checkpoint, for cache diagnostics and native correctness checks. */
  snapshotPosition(role: SnapshotRole): number | undefined {
    for (const [position, saved] of this.snapshots) if (saved[role]) return position
    return undefined
  }

  /** A copy of an f32 texture. */
  private copyOf(from: Texture) {
    const copy = this.gpu.texture(GL.RGBA32F, from.width, from.height)
    try {
      this.gpu.run({ program: this.gpu.program(K.COPY), inputs: { uX: from }, output: copy })
      return copy
    } catch (error) {
      // The caller cannot own this texture until the copy has returned.
      this.gpu.delete(copy)
      throw error
    }
  }

  /**
   * Whether a batch can save a snapshot partway (see forwardHidden's
   * saveAt): only Gated DeltaNet layers' state is needed there (no short
   * convolutions, no sliding windows: Qwen3.5).
   */
  get canSaveInside() {
    return this.layers.every((l) => !l.conv && !l.attention?.ring)
  }

  /** What a batch saving a snapshot partway has copied so far (see forwardBatch). */
  private saving: Array<Texture> | null = null

  /** A learned shared boundary, extracted without splitting the forward batch. */
  private prefixSaving: { row: number; state: Snapshot } | null = null

  /** Whether `resume(position)` would work: a snapshot there kept its last hidden row. */
  canResume(position: number) {
    return !!this.snapshots.get(position)?.hidden
  }

  /**
   * Rewinds to the snapshot at `position` (see canResume) and puts its last
   * token's hidden state back as row 0 of the last batch, so `candidates`,
   * `pick` and `logits` on row 0 give the next token's logits, as after the
   * prompt that saved it.
   */
  resume(position: number): number {
    const at = this.rewind(position)
    const hidden = this.snapshots.get(at)?.hidden
    if (at !== position || !hidden)
      throw new Error(`No snapshot to resume at ${position}.`)
    this.gpu.run({
      program: this.gpu.program(K.COPY_ROWS),
      inputs: { uX: hidden },
      uniforms: { uOff: { int: 0 } },
      output: this.buffers.residual[0],
      rect: { x: 0, y: 0, width: hidden.width, height: 1 }
    })
    // The head normalizes the residual itself (the copy has its sums of squares).
    this.headScaled = false
    // (Only row 0 holds a hidden state.)
    this.lastRowOnly = true
    return at
  }

  /** Where `rewind(position)` would leave the cache: the latest snapshot at or before it (0: none). */
  rewindPosition(position: number): number {
    return Math.max(0, ...[...this.snapshots.keys()].filter((p) => p <= position))
  }

  /**
   * Rewinds the cache to the latest snapshot at or before `position`, and
   * returns where it is now: tokens before it are still in the cache.
   */
  rewind(position: number): number {
    this.draft = null
    this.guessAt = null
    const at = Math.max(
      0,
      ...[...this.snapshots.keys()].filter((p) => p <= position)
    )
    for (const [p, saved] of this.snapshots) {
      if (p <= at) continue
      this.freeSnapshot(saved)
      this.snapshots.delete(p)
    }
    const saved = this.snapshots.get(at)
    if (!saved) {
      this.reset()
      return 0
    }
    this.restoreState(saved, at)
    this.position = at
    return at
  }

  /** Puts a snapshot's state, saved at `at`, back in the layers (see rewind). */
  private restoreState(saved: Snapshot, at: number) {
    const W = this.config.slidingWindow
    let c = 0
    let w = 0
    let s = 0
    for (const layer of this.layers) {
      if (layer.conv) {
        this.gpu.run({
          program: this.gpu.program(K.COPY),
          inputs: { uX: saved.conv[c++] },
          output: layer.conv.state[0]
        })
      }
      if (layer.linear) {
        for (const state of [layer.linear.convState[0], layer.linear.state[0]]) {
          this.gpu.run({
            program: this.gpu.program(K.COPY),
            inputs: { uX: saved.linear[s++] },
            output: state
          })
        }
      }
      const a = layer.attention
      if (a?.ring) {
        const kv = saved.windows[w++]
        this.copyPositions(kv, W, a.kv, a.ring, Math.max(0, at - W), at)
      }
    }
  }

  /** A copy of a u32 texture (a window's saved keys and values, a cache block). */
  private copyWhole(from: Texture, to = this.gpu.texture(from.internal, from.width, from.height)) {
    try {
      this.gpu.run({
        program: this.gpu.program(K.COPY_POSITIONS),
        inputs: { uX: from },
        uniforms: { uPos0: { int: 0 }, uRow0: { int: 0 }, uMod: { int: from.height } },
        output: to,
        rect: { x: 0, y: 0, width: Math.min(from.width, to.width), height: Math.min(from.height, to.height) }
      })
      return to
    } catch (error) {
      this.gpu.delete(to)
      throw error
    }
  }

  /** A copy of a snapshot's state (no hidden row, no roles). */
  private cloneSnapshot(saved: Snapshot): Snapshot {
    const copy: Snapshot = { conv: [], windows: [], linear: [] }
    try {
      for (const t of saved.conv) copy.conv.push(this.copyOf(t))
      for (const t of saved.linear) copy.linear.push(this.copyOf(t))
      for (const t of saved.windows) copy.windows.push(this.copyWhole(t))
      return copy
    } catch (error) {
      this.freeSnapshot(copy)
      throw error
    }
  }

  /** GPU bytes a saved prefix would hold at `position` (its snapshot's textures, its cache rows). */
  savedPrefixBytes(position: number) {
    const saved = this.snapshots.get(position)
    if (!saved) return 0
    let bytes = 0
    for (const t of [...saved.conv, ...saved.linear, ...saved.windows]) bytes += t.bytes
    const rows = 2 ** this.kvShift
    const kvBytes = this.kvWidth() * Math.ceil(position / rows) * Math.min(position, rows) * 16
    for (const layer of this.layers) if (layer.attention && !layer.attention.ring) bytes += kvBytes
    return bytes
  }

  /**
   * A copy of what resuming at `position` needs, for another history to
   * restore (see restorePrefix): the snapshot there, and the full-attention
   * cache rows before it (still the prompt's: later tokens only write past
   * them). Undefined without a snapshot at `position`.
   */
  savePrefix(position: number): SavedPrefix | undefined {
    const saved = this.snapshots.get(position)
    if (!saved || position <= 0 || position > this.position) return undefined
    const state = this.cloneSnapshot(saved)
    const kv: Array<Texture> = []
    try {
      for (const layer of this.layers) {
        const a = layer.attention
        if (!a || a.ring) continue
        kv.push(this.copyWhole(a.kv, this.kvTexture(position)))
      }
    } catch (error) {
      this.freeSnapshot(state)
      kv.forEach((t) => this.gpu.delete(t))
      throw error
    }
    return { position, state, kv }
  }

  /** How many textures of each kind a saved prefix holds, and its full-attention caches' shape at `position`. */
  private prefixLayout(position: number) {
    const rows = 2 ** this.kvShift
    return {
      conv: this.layers.filter((l) => l.conv).length,
      linear: 2 * this.layers.filter((l) => l.linear).length,
      windows: this.layers.filter((l) => l.attention?.ring).length,
      kv: this.layers.filter((l) => l.attention && !l.attention.ring).length,
      kvShape: { internal: GL.RGBA32UI, width: this.kvWidth() * Math.ceil(position / rows), height: Math.min(position, rows) }
    }
  }

  /** A saved prefix's textures, in order: conv states, Gated DeltaNet states, windows, full-attention caches. */
  private prefixTextures(saved: SavedPrefix) {
    return [...saved.state.conv, ...saved.state.linear, ...saved.state.windows, ...saved.kv]
  }

  /** A saved prefix's textures' shapes and contents, read back from the GPU (see importPrefix). */
  async exportPrefix(saved: SavedPrefix): Promise<Array<PrefixTexture>> {
    const out: Array<PrefixTexture> = []
    for (const t of this.prefixTextures(saved))
      out.push({ internal: t.internal, width: t.width, height: t.height, bytes: await this.gpu.readBytes(t) })
    return out
  }

  /**
   * A saved prefix at `position` made from exported textures (see
   * exportPrefix), if they fit this engine: as many of each kind, in f32
   * and u32 as saved, the full-attention caches of their exact shape.
   * Undefined if they don't.
   */
  importPrefix(position: number, textures: Array<PrefixTexture>): SavedPrefix | undefined {
    if (!Number.isInteger(position) || position <= 0 || position >= this.contextLength) return undefined
    const layout = this.prefixLayout(position)
    const states = layout.conv + layout.linear
    if (textures.length !== states + layout.windows + layout.kv) return undefined
    const fits = textures.every((t, i) =>
      t.bytes.byteLength === t.width * t.height * 16 && t.width > 0 && t.height > 0 &&
      (i < states
        ? t.internal === GL.RGBA32F
        : i < states + layout.windows
          ? t.internal === GL.RGBA32UI && t.height === this.config.slidingWindow
          : t.internal === layout.kvShape.internal && t.width === layout.kvShape.width && t.height === layout.kvShape.height))
    if (!fits) return undefined
    const made: Array<Texture> = []
    try {
      for (const t of textures) {
        const integer = t.internal === GL.RGBA32UI
        const words = t.bytes.byteLength / 4
        const pixels = t.bytes.byteOffset % 4 === 0
          ? integer ? new Uint32Array(t.bytes.buffer, t.bytes.byteOffset, words) : new Float32Array(t.bytes.buffer, t.bytes.byteOffset, words)
          : integer ? new Uint32Array(t.bytes.slice().buffer) : new Float32Array(t.bytes.slice().buffer)
        made.push(this.gpu.texture(t.internal, t.width, t.height, {
          format: integer ? GL.RGBA_INTEGER : GL.RGBA,
          type: integer ? GL.UNSIGNED_INT : GL.FLOAT,
          pixels
        }))
      }
    } catch (error) {
      made.forEach((t) => this.gpu.delete(t))
      throw error
    }
    const take = (n: number) => made.splice(0, n)
    const conv = take(layout.conv)
    const linear = take(layout.linear)
    const windows = take(layout.windows)
    return { position, state: { conv, windows, linear }, kv: made }
  }

  /** Frees a saved prefix's textures. */
  freePrefix(saved: SavedPrefix) {
    this.freeSnapshot(saved.state)
    saved.kv.forEach((t) => this.gpu.delete(t))
  }

  /**
   * Makes `saved` the cache: its full-attention rows and its snapshot's
   * state put back, at its position. The live history's snapshots are
   * another prompt's, so they go; a copy of `saved`'s snapshot takes their
   * place, as the prefix (rewinds along the new history stop there).
   */
  async restorePrefix(saved: SavedPrefix): Promise<number> {
    this.draft = null
    this.guessAt = null
    for (const s of this.snapshots.values()) this.freeSnapshot(s)
    this.snapshots.clear()
    const at = saved.position
    await this.reserve(at)
    let k = 0
    for (const layer of this.layers) {
      const a = layer.attention
      if (!a || a.ring) continue
      this.copyWhole(saved.kv[k++], a.kv)
    }
    this.restoreState(saved.state, at)
    this.position = at
    // A live copy for rewinds along the new history: optional, so a failed one (out of memory) only skips it.
    if (this.snapshotLimit >= 2) {
      try {
        this.keepSnapshot(at, this.cloneSnapshot(saved.state), "prefix")
      } catch (error) {
        if (this.gpu.isLost) throw error
      }
    }
    return at
  }

  private debug: Texture | null = null
  private kvDebug: Texture | null = null

  /** A scratch tensor for probes, made the first time one asks. */
  private debugTexture() {
    const c = this.config
    const qkvg = this.qkvgDims()
    this.debug ??= this.f32(
      Math.max(2 * c.intermediateSize, qkvg),
      this.capacity
    )
    return this.debug
  }

  /**
   * The prompt's images (in place of their placeholder tokens), in order,
   * not overlapping. Their rows read their features instead of the tokens'
   * embeddings. With `positions` "grid" (Qwen3.5) they take Qwen2-VL's 3D
   * positions (see ropeAt), and the text after an image continues from its
   * largest position; with "sequence" (LFM2-VL, LFM3-VL) every row keeps
   * its cache position, like any token. Set before reading a prompt that
   * has images, and again (empty) for one that has none.
   */
  setImages(images: Array<EngineImage>, positions: "grid" | "sequence" = "grid") {
    this.imagePositions = positions
    let end = 0
    for (const image of images) {
      const rows = image.height * image.width
      if (!(image.start >= end && rows > 0)) throw new RangeError("Images must be in order and not overlap.")
      if (image.features && image.features.length !== rows * this.config.hiddenSize)
        throw new RangeError(`An image of ${rows} tokens needs ${rows * this.config.hiddenSize} feature values.`)
      end = image.start + rows
    }
    this.images = images
  }

  /**
   * The rotary position of cache position `pos`: itself, less what the
   * images before it saved (an image of h × w tokens advances positions by
   * max(h, w), not h · w). For an image's row, its t, h and w positions.
   */
  ropeAt(pos: number): [number, number, number] {
    if (this.imagePositions === "sequence") return [pos, pos, pos]
    let shift = 0
    for (const image of this.images) {
      const rows = image.height * image.width
      if (pos >= image.start + rows) {
        shift += Math.max(image.height, image.width) - rows
        continue
      }
      if (pos >= image.start) {
        const t = image.start + shift
        const r = pos - image.start
        return [t, t + Math.floor(r / image.width), t + (r % image.width)]
      }
      break
    }
    return [pos + shift, pos + shift, pos + shift]
  }

  /** Forgets the cache (and the images): the next forward starts a new sequence. */
  reset() {
    this.images = []
    this.draft = null
    this.guessAt = null
    this.position = 0
    this.shrink()
    for (const saved of this.snapshots.values()) this.freeSnapshot(saved)
    this.snapshots.clear()
    const zero = new Float32Array(
      this.config.hiddenSize * this.convRows()
    )
    for (const layer of this.layers) {
      for (const state of layer.conv?.state ?? []) {
        this.gpu.write(
          state,
          { x: 0, y: 0, width: state.width, height: state.height },
          {
            format: GL.RGBA,
            type: GL.FLOAT,
            pixels: zero
          }
        )
      }
      const l = layer.linear
      if (l) {
        for (const state of [...l.convState, ...l.state]) {
          this.gpu.write(
            state,
            { x: 0, y: 0, width: state.width, height: state.height },
            {
              format: GL.RGBA,
              type: GL.FLOAT,
              pixels: new Float32Array(state.width * state.height * 4)
            }
          )
        }
      }
    }
  }

  /**
   * Whether draft batches (speculative decoding) can be read: a Gated
   * DeltaNet layer would need its recurrent state after each prefix of the
   * batch to keep only some of it, so models with them read one token at
   * a time (the one-token guess still works: see dropGuess).
   */
  get canDraft() {
    return !this.layers.some((l) => l.linear)
  }

  /**
   * Runs every kind of pass once (a prompt batch, a draft batch, a single
   * token, both ways of sampling, a snapshot) so shaders compile, and the
   * driver builds its pipelines, at load rather than inside the first
   * reply's timings. Then forgets it all.
   */
  /** Programs each warm-up step compiled (load profiling). */
  warmCounts: Record<string, number> = {}
  async warmUp(options: { shortOnly?: boolean } = {}) {
    // (`shortOnly`: just a short batch, for programs a new Engine.tune pick
    // reads with; it resets the cache like the rest.)
    // The f16 scratch as a remembered tuned f16Pack reads it (the load made
    // TUNING's).
    if (this.quadMatmuls && this.usesF16()) this.f16Scratch()
    this.f16Trim()
    // Every program a layer compiles is shared by the layers of its kind:
    // the warm-up reads through the first and last layers (their own
    // variants: folded embedding, last-row tail) and one of each other
    // kind only, a fraction of the GPU work of the whole model.
    const all = this.layers
    this.work = null
    const kinds = new Set<string>()
    if (TUNING.warmSubset > 0) this.layers = all.filter((l, i) => {
      const kind = l.conv ? "conv" : `${l.type}:${l.attention?.ring ? "ring" : ""}`
      const keep = i === 0 || i === all.length - 1 || !kinds.has(kind)
      kinds.add(kind)
      return keep
    })
    // On WebGPU nothing need run: the pipelines compile in the background,
    // several at once (see WebGpu.deferPipelines), and are waited for.
    const defer = !!this.gpu && "deferPipelines" in this.gpu && TUNING.deferWarmUp > 0
    const steps = async () => {
      if (options.shortOnly) await this.forwardHidden(new Array(Math.min(SPLIT_ROWS + 1, this.contextLength)).fill(0))
      else await this.warmUpLayers()
    }
    // (Running them too once compiled, for the first prompt's sake, cost a
    // Pixel 10 Pro 380 ms more load and saved an M-series Mac 13 ms of a
    // 548-token prompt against its 25.)
    try {
      if (defer) {
        this.gpu.deferPipelines = true
        try {
          await steps()
        } finally {
          this.gpu.deferPipelines = false
          await this.gpu.settlePipelines()
        }
        // Full draft batches' pipelines go on compiling after the load (the
        // next forward waits for any still compiling): awaited here they
        // cost a Pixel 10 Pro 330 ms more load; left to its first draft
        // batch, 9 programs compiled mid-reply.
        if (!options.shortOnly && this.canDraft) {
          this.reset()
          this.gpu.deferPipelines = true
          try {
            await this.warmDraftBatches()
          } finally {
            this.gpu.deferPipelines = false
          }
        }
      } else await steps()
    } finally {
      this.layers = all
      // (Per-layer work for pacing, counted over the subset.)
      this.work = null
      this.reset()
    }
  }

  /** A full draft batch, plain and with chunked attention, its rows read both ways with a guess (see warmUp). */
  private async warmDraftBatches() {
    const rows = Math.min(SPLIT_ROWS, this.contextLength - 1)
    await this.forwardHidden(new Array<number>(rows).fill(0), { draft: true })
    // With a row seeing a token otherwise than the mask too, as drafts'
    // rows do: the pick's GLSL passes then (see nativePicks).
    for (const seen of [[], [[0, 0, 0, rows]]] as Array<Array<SeenRow>>) {
      await this.pick(0, rows, 1, 1, seen, true)
      this.dropGuess()
      await this.argmax(0, rows, 1, seen, true)
      this.dropGuess()
    }
    this.accept(1)
    if (TUNING.attendChunkTexels > 0) {
      const saved = [TUNING.attendChunkTexels, TUNING.attendChunkTexelsDecode, TUNING.flashPast]
      TUNING.attendChunkTexels = TUNING.attendChunkTexelsDecode = 1
      TUNING.flashPast = 0
      try {
        this.reset()
        await this.forwardHidden(new Array<number>(rows).fill(0), { draft: true })
        this.accept(1)
      } finally {
        ;[TUNING.attendChunkTexels, TUNING.attendChunkTexelsDecode, TUNING.flashPast] = saved
      }
    }
  }

  private async warmUpLayers() {
    let seen = this.gpu.compiled.count
    const step = (name: string) => {
      this.warmCounts[name] = this.gpu.compiled.count - seen
      seen = this.gpu.compiled.count
    }
    // Each probe fits the cache: none reads more tokens than it holds, and
    // one that won't fit after the last starts from an empty cache.
    const fit = (rows: number) => {
      rows = Math.min(rows, this.contextLength)
      if (this.position + rows > this.contextLength) this.reset()
      return new Array<number>(rows).fill(0)
    }
    const row = await this.forwardHidden(fit(SPLIT_ROWS + 1))
    step("batch9")
    await this.candidates(row, 1, 1, true)
    step("pick")
    this.dropGuess()
    await this.logits(row)
    step("logits")
    await this.argmax(row, 1, 1, [])
    step("argmax")
    // A full batch: the f16 matmul path prompts take, at the size they run
    // (a pass's WebGPU variant depends on its size: warmed at f16MinRows
    // rows, a cached load compiled 6 programs mid-reply for its first
    // prompt; tune, which a cached load skips, forwards full batches).
    // (A cache too small for it never takes that path.)
    const f16Rows = Math.max(TUNING.f16MinRows, SPLIT_ROWS + 1, Math.min(this.batchSize, this.capacity))
    if (TUNING.f16MinRows <= this.capacity && f16Rows <= this.contextLength) {
      await this.forwardHidden(fit(f16Rows))
      step("batchF16")
    }
    if (this.canDraft) {
      await this.forwardHidden(fit(2), { draft: true })
      this.accept(1)
      step("draft")
    }
    await this.logits(await this.forwardHidden(fit(1)))
    step("token")
    this.snapshot()
    this.reset()
    if (TUNING.attendChunkTexels > 0) {
      // Chunked attention (long key ranges) too, forced on for a short
      // batch and a token.
      // (Writing's panels too, where this GPU writes with them: see attendPanel.)
      const saved = TUNING.attendChunkTexels
      const savedDecode = TUNING.attendChunkTexelsDecode
      const savedPast = TUNING.flashPast
      TUNING.attendChunkTexels = 1
      TUNING.attendChunkTexelsDecode = 1
      TUNING.flashPast = 0
      try {
        await this.forwardHidden(fit(Math.min(12, this.contextLength - 1)))
        await this.logits(await this.forwardHidden(fit(1)))
        step("chunked")
      } finally {
        TUNING.attendChunkTexels = saved
        TUNING.attendChunkTexelsDecode = savedDecode
        TUNING.flashPast = savedPast
        this.reset()
      }
    }
  }

  /**
   * The chunked recurrence against a token at a time on this GPU (its
   * compiler could get the solve's per-fragment array wrong): the first
   * Gated DeltaNet layer's state after `tokens` (one batch, read both ways
   * from the same inputs), as max |difference| over max |state| (right:
   * ~1e-6; Infinity if any value isn't finite). 0 without such layers or
   * chunks. Leaves the cache empty.
   */
  private async recurrenceError(tokens: Array<number>) {
    const first = this.layers.findIndex((l) => l.linear)
    const chunk = TUNING.linChunk
    if (first < 0 || !this.chunkedRecurrence || chunk <= 0) return 0
    const state = async (c: number) => {
      TUNING.linChunk = c
      try {
        this.reset()
        await this.forward(tokens)
        const layer = this.layers[first].linear
        if (!layer) throw new Error("The model's config has no Gated DeltaNet shapes.")
        return await this.gpu.readNow(layer.state[0])
      } finally {
        TUNING.linChunk = chunk
      }
    }
    const want = await state(0)
    const got = await state(chunk)
    this.reset()
    let diff = 0
    let scale = 0
    for (let i = 0; i < want.length; i++) {
      if (!Number.isFinite(got[i])) return Infinity
      diff = Math.max(diff, Math.abs(got[i] - want[i]))
      scale = Math.max(scale, Math.abs(want[i]))
    }
    return scale > 0 ? diff / scale : diff
  }

  /**
   * Attention on this GPU against the CPU's (see attentionError): a short
   * prompt read as usual, then with chunked attention forced on; and the
   * chunked Gated DeltaNet recurrence against a token at a time (see
   * recurrenceError). Returns the largest relative error (about 2e-4
   * where the GPU gets it right).
   */
  async selfCheck() {
    // (The panel check reads one token after them: room for it too.)
    const tokens = Array.from(
      { length: Math.min(40, this.contextLength - 1) },
      (_, i) => checkToken(i, this.config.vocabSize)
    )
    let worst = 0
    const all = this.layers,
      config = this.config
    // Dense attention's shaders are shared by layer type and cache layout.
    // Validate real inputs on each distinct variant plus both model ends;
    // recurrent models keep their full state/sequence validation below.
    if (!config.linear) {
      const kinds = new Set<string>()
      this.layers = all.filter((layer, i) => {
        const kind = `${layer.type}:${layer.attention?.ring ? "ring" : "plain"}`
        const keep = i === 0 || i === all.length - 1 || !kinds.has(kind)
        kinds.add(kind)
        return keep
      })
      this.config = { ...config, layerTypes: this.layers.map((l) => l.type) }
      this.work = null
    }
    try {
      if (!this.config.linear && TUNING.attendChunkTexels > 0) {
        // Validate chunked attention from the same real queries while the
        // reference forward is paused, then reuse its cache for the panel
        // check. No second/third full-model prompt is needed.
        this.chunkErrors = []
        try {
          const referenceError = await attentionError(this, tokens, true)
          const chunkError = Math.max(0, ...this.chunkErrors)
          // The chunk comparison is relative to its GPU output. Convert
          // the two checks into a conservative CPU-relative error bound.
          worst =
            chunkError < 1
              ? (referenceError + chunkError) / (1 - chunkError)
              : Infinity
        } finally {
          this.chunkErrors = null
        }
        try {
          worst = Math.max(worst, await this.panelError(tokens, true))
        } finally {
          this.reset()
        }
      } else {
        worst = await attentionError(this, tokens)
        worst = Math.max(worst, await this.recurrenceError(tokens))
        if (TUNING.attendChunkTexels > 0) {
          const saved = TUNING.attendChunkTexels
          const savedDecode = TUNING.attendChunkTexelsDecode
          TUNING.attendChunkTexels = 1
          TUNING.attendChunkTexelsDecode = 1
          try {
            worst = Math.max(
              worst,
              await attentionError(this, tokens.slice(0, 24))
            )
          } finally {
            TUNING.attendChunkTexels = saved
            TUNING.attendChunkTexelsDecode = savedDecode
          }
          worst = Math.max(worst, await this.panelError(tokens))
        }
      }
      worst = Math.max(worst, await this.headError())
      // The probes' scratch isn't read again (~29 MB for Qwen3.5 2B at a
      // 512-token batch).
      for (const t of [this.debug, this.kvDebug]) if (t) this.gpu.delete(t)
      this.debug = this.kvDebug = null
      return worst
    } finally {
      this.layers = all
      this.config = config
      this.work = null
      this.reset()
    }
  }

  /**
   * The embedding lookup and the output head against the CPU: the rows the
   * GPU looked up for the self-check's first tokens against those the loader
   * gave (embeddingSamples), and the head's logits for those tokens against
   * the CPU's from the last layer's output, normed there. A weight layout
   * the kernels read wrong (texel format, nibble order) fails here instead
   * of writing gibberish. The largest relative error; 0 without samples
   * (6-bit embeddings).
   */
  private async headError(): Promise<number> {
    // (Fewer in a context too small for them all.)
    const ids = [...this.embeddingSamples.keys()].slice(0, Math.max(0, this.contextLength - 1))
    const weights = this.finalNormValues
    if (!ids.length || !weights) return 0
    const D = this.config.hiddenSize
    const last = this.layers.length - 1
    let embedded: Float32Array | undefined
    let out: { data: Float32Array; rows: number } | undefined
    this.reset()
    let logits: Float32Array
    try {
      logits = await this.forward(ids, async (layer, name, read) => {
        if (layer === -1 && name === "embedding") embedded = (await read()).data
        else if (layer === last && name === "l_out") out = await read()
      })
    } finally {
      this.reset()
    }
    if (!embedded || !out) return 0
    let worst = 0
    ids.forEach((id, r) => {
      const want = this.embeddingSamples.get(id) as Float32Array
      let diff = 0
      let scale = 0
      for (let k = 0; k < D; k++) {
        diff = Math.max(diff, Math.abs((embedded as Float32Array)[r * D + k] - want[k]))
        scale = Math.max(scale, Math.abs(want[k]))
      }
      worst = Math.max(worst, scale > 0 ? diff / scale : diff)
    })
    // The head reads the last token's output, final-normed (NORM_TO_H2).
    const x = out.data.subarray((out.rows - 1) * D, out.rows * D)
    let squares = 0
    for (let k = 0; k < D; k++) squares += x[k] * x[k]
    const r = 1 / Math.sqrt(squares / D + this.config.normEps)
    const want = ids.map((id) => {
      const row = this.embeddingSamples.get(id) as Float32Array
      let dot = 0
      for (let k = 0; k < D; k++) dot += row[k] * x[k] * r * weights[k]
      return dot
    })
    const scale = Math.max(...want.map(Math.abs))
    const diff = Math.max(...ids.map((id, i) => Math.abs(logits[id] - want[i])))
    return Math.max(worst, scale > 0 ? diff / scale : diff)
  }

  /**
   * Writing's attention in panels (attendPanel and its combine) against
   * the one-pass kernel (which attentionError checks against the CPU),
   * each attention layer's both ways from the same queries and cache: the
   * last of `tokens` after the rest, in chunks of 4 keys (enough to take
   * both of the combine's loops). The largest relative error, as
   * attentionError's. Leaves the cache empty.
   */
  private async panelError(tokens: Array<number>, reuseCache = false) {
    if (!this.mrt.chunks) return 0
    const saved = [
      TUNING.flashPast,
      TUNING.flashChunkTexels,
      TUNING.flashDecode
    ]
    const tuned = this.tuned.flashDecode
    this.tuned.flashDecode = null
    try {
      // (Whether or not this GPU writes that way: the tuner may pick it.)
      TUNING.flashPast = 0
      TUNING.flashChunkTexels = 1
      TUNING.flashDecode = 1
      if (!reuseCache) {
        this.reset()
        await this.forwardHidden(tokens.slice(0, -1))
      }
      this.panelErrors = []
      await this.forwardHidden(tokens.slice(-1))
      await this.gpu.finish()
      return this.panelErrors.length ? Math.max(...this.panelErrors) : Infinity
    } finally {
      ;[TUNING.flashPast, TUNING.flashChunkTexels, TUNING.flashDecode] = saved
      this.tuned.flashDecode = tuned
      this.panelErrors = null
      this.reset()
    }
  }
  /** While panelError runs: each attention layer's error. */
  private panelErrors: Array<number> | null = null
  /** Startup chunk comparisons on the reference forward's real inputs. */
  private chunkErrors: Array<number> | null = null

  /**
   * a · wᵀ for `rows` tokens: a few rows leave a split matmul's partial
   * sums in the partial buffer and return the number of slices (consumers
   * read them through pf, no reduce pass); more rows write the pair tensor
   * `out` and return 0.
   */
  private async project(
    a: Texture,
    w: GpuMatrix,
    out: Texture,
    rows: number,
    norm?: Texture | "prescaled"
  ) {
    if (rows <= SPLIT_ROWS) return this.splitPartials(a, w, rows, norm)
    if (norm) throw new Error("Only split matmuls fold the norm in.")
    await this.matmul(a, w, out, rows)
    return 0
  }

  /** The last split matmul's partial sums. */
  private partials() {
    return this.buffers.partial
  }

  /** `source`, reading `names` from the last split matmul's partials. */
  private fromSplit(source: string, names: Array<string>, split: number) {
    return K.withSplit(K.fromPartials(source, names), split)
  }

  /** How each pair tensor was last written: K-slices stacked, rows apart. */
  private slicing = new Map<
    number,
    { split: number; stride: number; quad?: boolean }
  >()

  /** Pair tensor t's second texture if a 4-token matmul wrote it last. */
  private quadOf(t: Texture) {
    return this.slicing.get(t.id)?.quad ? this.quads.get(t.id) : undefined
  }

  /** The second texture of pair tensor t as input `name`_B, if it has one. */
  private quadInputs(name: string, t: Texture): Record<string, Texture> {
    const partner = this.quadOf(t)
    return partner ? { [`${name}_B`]: partner } : {}
  }

  /** `source` reading pair tensor input `name` (t) as it was written. */
  private readPair(source: string, name: string, t: Texture) {
    if (!this.quadOf(t)) return source
    // The slice count as a constant: the reads unroll.
    const split = this.slicing.get(t.id)?.split ?? 1
    return K.withPairSplit(K.fromQuad(source, [name]), split)
  }

  /** The uniforms pf needs to read pair tensor t (its K-slices). */
  private pairIn(t: Texture) {
    const p = this.slicing.get(t.id) ?? { split: 1, stride: 0 }
    return { uPairSplit: { int: p.split }, uSliceStride: { int: p.stride } }
  }

  /**
   * The f16 prompt matmul's scratch: one matrix's dequantized weights (rows
   * wrapped into columns uHd tall: RGBA16F, or h2 with TUNING.f16Pack's
   * bit 2) and, unless it reads them as h2 (bit 1), the activations as
   * RGBA16F. Ordered draws reuse one conversion texture by default.
   * Each texture is made when first
   * needed (a sweep can change f16Pack). Null until sized; false where
   * there are no such matrices or no column height fits.
   */
  private f16:
      | {
        hd: number
        /** The weights' width in RGBA16F texels (h2: half). */
        width: number
        kMax: number
        /** The budget it was sized for (TUNING.f16ScratchMiB or f16ScratchBytesMiB). */
        mib: number
        w?: Texture
        wh2?: Texture
        /** wh2's h2 texels in a storage buffer, row r at r · width (see bufferGemm). */
        wbuf?: Texture
        a?: Array<Texture>
        /** Whether RGBA16F can be drawn into (asked when first needed). */
        halfRenders?: boolean
        next: number
      }
    | null
    | false = null

  /** Scratch made for another budget (sweeps only). */
  private f16Kept = new Map<number, NonNullable<Exclude<Engine["f16"], false>>>()

  /** Whether 4-bit matrix `w` goes through the f16 prompt matmul (see TUNING.f16Matmul). */
  private takesF16(w: GpuMatrix) {
    const mode = this.tuned.f16Matmul ?? TUNING.f16Matmul
    return w.bits === 4 && mode > 0 && (!w.rgba4 || mode > 1)
  }

  /**
   * Whether 8-bit matrix `w` (its quants in a storage buffer) goes through
   * the f16 scratch buffer to the native prompt matmul, as the 4-bit ones
   * beside it do (bufferGemm): through their scratch, which is at least
   * as wide as the widest of their K, so only a matrix no wider.
   */
  private eightTakesF16(w: GpuMatrix, f: { kMax: number; wbuf?: Texture }) {
    return TUNING.bufferGemm8 > 0 && w.bits === 8 && !!w.stored && !!f.wbuf && w.K <= f.kMax
  }

  /** The f16 scratch 8-bit matrix `w` goes through into `out` (a 4-token matmul's output: it has a second texture), or false. */
  private eightScratch(w: GpuMatrix, out: Texture) {
    if (w.bits !== 8 || !w.stored || !this.quads.has(out.id) || TUNING.quad <= 0 || !this.usesF16() || !(this.f16Pack() & 2)) return false
    const f = this.f16Scratch()
    return f && this.eightTakesF16(w, f) ? f : false
  }

  /** The layer matrices prompt matmuls read. */
  private layerMats() {
    return this.layers
      .flatMap((l) => [
        l.conv?.inProj,
        l.conv?.outProj,
        l.attention?.qkvg,
        l.attention?.o,
        l.linear?.inProj,
        l.linear?.ba,
        l.linear?.outProj,
        l.gateUp,
        l.down
      ])
      .filter((m): m is GpuMatrix => !!m)
  }

  /**
   * The matrices the f16 scratch is sized for: every 4-bit layer matrix,
   * so a tuned f16Matmul can switch RGBA4 weights to it.
   */
  private f16Mats() {
    return this.layerMats().filter((m) => m.bits === 4)
  }

  /** Whether any prompt matmul goes through the f16 matmul. */
  private usesF16() {
    return this.layerMats().some((m) => this.takesF16(m))
  }

  /** How the f16 matmul reads: this GPU's tuned choice, else TUNING.f16Pack. */
  private f16Pack() {
    return (this.tuned.f16Pack ?? TUNING.f16Pack) & 3
  }

  /** Frees the f16 scratch textures the current f16Matmul and f16Pack don't use. */
  private f16Trim() {
    const f = this.f16
    if (!f) return
    // (None of it where nothing takes the f16 matmul.)
    const none = !this.usesF16()
    const pack = this.f16Pack()
    if (f.w && (none || pack & 2)) {
      this.gpu.delete(f.w)
      f.w = undefined
    }
    if (f.wh2 && (none || !(pack & 2))) {
      this.gpu.delete(f.wh2)
      f.wh2 = undefined
    }
    if (f.wbuf && (none || !(pack & 2))) {
      this.gpu.delete(f.wbuf)
      f.wbuf = undefined
    }
    if (f.a && (none || pack & 1)) {
      for (const t of f.a) this.gpu.delete(t)
      f.a = undefined
    }
  }

  /**
   * The f16 scratch, its textures for f16Pack made; false if it needs
   * RGBA16F where that can't be drawn into.
   */
  private f16Scratch() {
    const f = this.f16Sized()
    if (!f) return false
    const pack = this.f16Pack()
    // (canRender draws and reads back: asked once, not every matmul.)
    if (
      (pack & 3) !== 3 &&
      !(f.halfRenders ??= this.gpu.canRender([GL.RGBA16F]))
    )
      return false
    if (pack & 2 && this.bufferGemm()) f.wbuf ??= (this.gpu as unknown as WebGpu).storage((f.width / 2) * f.hd * 16, f.width / 2, f.hd)
    else if (pack & 2) f.wh2 ??= this.gpu.texture(GL.RGBA32UI, f.width / 2, f.hd)
    else f.w ??= this.gpu.texture(GL.RGBA16F, f.width, f.hd)
    if (!(pack & 1))
      f.a ??= Array.from(
        { length: Math.max(1, TUNING.f16ActivationBuffers) },
        () => this.gpu.texture(GL.RGBA16F, f.kMax / 4, this.capacity)
      )
    return f
  }

  private f16Sized() {
    const requested = this.rgba4Layers
      ? Math.max(0, TUNING.f16ScratchMiB)
      : TUNING.f16ScratchBytesMiB
    const mib = requested < 0 ? this.byteF16Budget().next / 2 ** 20 : requested
    if (this.f16 && this.f16.mib !== mib) {
      // (A sweep changed the size: the other size's scratch kept for when
      // it comes back, so no timed prompt pays for making it again.)
      this.f16Kept.set(this.f16.mib, this.f16)
      this.f16 = this.f16Kept.get(mib) ?? null
      this.f16Kept.delete(mib)
    }
    if (this.f16 !== null) return this.f16
    const mats = this.f16Mats()
    if (mats.length === 0) return (this.f16 = false)
    const max = this.gpu.limits.maxTextureSize
    const kMax = Math.max(...mats.map((m) => m.K))
    let best: { hd: number; width: number } | null = null
    if (mib > 0) {
      // The budget's texels in the shape that splits the layer matrices
      // into the fewest chunks at a full batch (each chunk costs a
      // dequantization and a matmul pass, each waiting on the last; see
      // f16Plan), at least as wide as the widest K.
      const texels = (mib * 2 ** 20) / 8
      let fewest = Infinity
      for (let hd = max & ~3; hd >= 64; hd = (hd >> 1) & ~3) {
        const needed = this.rgba4Layers
          ? Infinity
          : Math.max(...mats.map((m) => Math.ceil(m.N / hd) * (m.K / 4)))
        const width = Math.min(max, Math.floor(texels / hd), needed) & ~1
        if (width < kMax / 4) continue
        const chunks = mats.reduce((n, m) => {
          const { kbSlice, slices } = this.quadSlices(m, TUNING.quadMaxSplit)
          const plan = Engine.f16Plan({ hd, width }, m.N, kbSlice, slices)
          return n + Math.ceil(slices / plan.group) * Math.ceil(m.N / plan.rows)
        }, 0)
        if (
          chunks < fewest ||
          (!this.rgba4Layers &&
            chunks === fewest &&
            chunks === mats.length &&
            best &&
            width * hd < best.width * best.hd)
        ) {
          fewest = chunks
          best = { hd, width }
        }
      }
    } else {
      // The column height that fits every matrix in the least texture.
      for (let hd = max & ~3; hd >= 64; hd = (hd >> 1) & ~3) {
        const width = Math.max(
          ...mats.map((m) => Math.ceil(m.N / hd) * (m.K / 4))
        )
        if (width <= max && (!best || width * hd < best.width * best.hd))
          best = { hd, width }
      }
    }
    if (!best) return (this.f16 = false)
    // (Every K is whole blocks, so widths are even: h2 halves them exactly.)
    this.f16 = { hd: best.hd, width: best.width, kMax, mib, next: 0 }
    return this.f16
  }

  /**
   * How a 4-token matmul splits `w`'s K: slices of kbSlice blocks, at
   * most `most` of them (TUNING.quadSliceK's size, twice for wide
   * matrices).
   */
  private quadSlices(w: GpuMatrix, most: number) {
    const sliceK = TUNING.quadSliceK * (w.groups > 2048 ? 2 : 1)
    const kbSlice = Math.ceil(
      w.blocks / Math.max(1, Math.min(most, Math.round(w.K / sliceK)))
    )
    return { kbSlice, slices: Math.ceil(w.blocks / kbSlice) }
  }

  /** out (a pair tensor) = a · wᵀ for `rows` tokens of h2 activations. */
  private async matmul(a: Texture, w: GpuMatrix, out: Texture, rows: number) {
    const bits = w.bits
    if (bits === 6)
      throw new GpuError("Q6_K weights are only read as the output head.")
    const weights = {
      uA: a,
      uW: w.quants,
      uS: w.scales,
      ...(w.zeros && { uZ: w.zeros })
    }
    const shape = {
      uKb: { int: w.blocks },
      uH: { int: w.rowsPerColumn },
      uRowTexels: { int: w.rowTexels },
      uG: { int: w.groups }
    }
    const partner = this.quads.get(out.id)
    // (An 8-bit matrix among 4-bit ones that take the buffer scratch: through it too.)
    const scratch8 = this.eightScratch(w, out)
    const eight = !!scratch8
    if ((w.rgba4 || w.bits === 4 || eight) && partner && TUNING.quad > 0) {
      // Four tokens per fragment, into two textures; K in slices of about
      // quadSliceK values (twice that for wide matrices), stacked in rows.
      const quadRows = Math.ceil(rows / 4)
      // (This GPU's own choice for short batches, if tune made one.)
      const short = rows < TUNING.f16MinRows
      // The most slices the output holds at a full batch: the same for
      // every batch, so the warm-up compiles the programs prompts use.
      const most = Math.min(
        short ? (this.tuned.shortSplit ?? TUNING.quadMaxSplit) : TUNING.quadMaxSplit,
        Math.floor(out.height / Math.ceil(this.capacity / 4))
      )
      const { kbSlice, slices } = this.quadSlices(w, most)
      // (Byte texels have no matmulQuad: the f16 path at any size. RGBA4
      // takes it only with f16Matmul 2, and not for short batches.)
      const f = eight
        ? scratch8
        : this.takesF16(w) && (!w.rgba4 || rows >= TUNING.f16MinRows)
          ? this.f16Scratch()
          : false
      if (f) {
        const kq = w.K / 4
        const pack = this.f16Pack()
        const h2W = (pack & 2) !== 0
        const wOut = h2W ? f.wbuf ?? f.wh2 : f.w
        if (!wOut) throw new GpuError("The f16 matmul's scratch is missing.")
        // The activations as RGBA16F, unless the matmul reads their h2.
        let fa = a
        if (!(pack & 1) && a.internal !== GL.RGBA16F) {
          if (!f.a) throw new GpuError("The f16 matmul's scratch is missing.")
          fa = f.a[f.next]
          f.next = (f.next + 1) % f.a.length
          await this.gpu.runAsync({
            name: "dequant",
            program: this.gpu.program(K.H2_TO_F16),
            inputs: { uA: a },
            output: fa,
            rect: {
              x: 0,
              y: 0,
              width: kq,
              height: Math.min(quadRows * 4, fa.height)
            }
          })
        }
        // (Weights copied to a storage buffer: the native dequant reads them there; see storedQuants.)
        const stored = this.storedQuants(w)
        const toBuffer = !!f.wbuf && h2W
        if (toBuffer && !stored) throw new GpuError("The f16 matmul's buffer scratch needs the weights in a buffer.")
        const storedDequant = stored
          ? (this.gpu as unknown as { nativeProgram(spec: NativeSpec): Program }).nativeProgram(this.bufferDequantSpec(h2W, toBuffer, eight ? 8 : 4, w.packed))
          : null
        const dequant = stored ? null : this.gpu.program(
          w.rgba4
            ? this.scalesOf(
                w,
                K.dequantRgba4({ zeros: !!w.zeros || w.packed, h2: h2W })
              )
            : K.dequantBytes({
                zeros: !!w.zeros,
                words: !w.bytes,
                unorm: w.unorm,
                packed: w.packed,
                h2: h2W
              })
        )
        const multiply = this.gpu.program(
          K.matmulF16({
            half: TUNING.f16Half > 0 || this.precision === "mediump",
            packed: pack
          })
        )
        // WebGPU, RGBA16F weights and activations (f16Pack 0) in f32 math:
        // the native prompt matmul (TUNING.nativePromptMatmul).
        const nativeMultiply = toBuffer && "nativeProgram" in this.gpu
          // The h2 weights in the buffer scratch: the native matmul reading them there (see bufferGemm).
          ? this.gpu.nativeProgram(this.promptBufferSpec(w.blocks % kbSlice === 0 ? kbSlice * 4 : 0))
          : TUNING.nativePromptMatmul > 0 && pack === 0 && !(TUNING.f16Half > 0 || this.precision === "mediump") && "nativeProgram" in this.gpu
          ? this.gpu.nativeProgram((this.promptSpec ??= promptMatmul()))
          : null
        const kqSlice = kbSlice * 8
        // The matrix through the scratch in chunks (the whole of it if it
        // fits): `group` K-slices at a time, and within those, rows r0 ..
        // r0 + n at a time; each chunk dequantized, then multiplied into
        // its slices' output rows.
        const plan = Engine.f16Plan(f, w.N, kbSlice, slices)
        for (let s0 = 0; s0 < slices; s0 += plan.group) {
          const s1 = Math.min(slices, s0 + plan.group)
          const kq0 = s0 * kqSlice
          const kqW = Math.min(kq, s1 * kqSlice) - kq0
          for (let r0 = 0; r0 < w.N; r0 += plan.rows) {
            const n = Math.min(plan.rows, w.N - r0)
            await this.gpu.runAsync({
              name: "dequant",
              program: (storedDequant ?? dequant)!,
              inputs: stored
                ? { uW: stored, uS: w.scales }
                : {
                    uW: w.quants,
                    uS: w.scales,
                    ...(w.zeros && { uZ: w.zeros })
                  },
              uniforms: {
                uKb: { int: w.blocks },
                uH: { int: w.rowsPerColumn },
                uRowTexels: { int: w.rowTexels },
                uWidth: { int: w.quants.width },
                uKq: { int: kqW },
                uKq0: { int: kq0 },
                uHd: { int: f.hd },
                uN: { int: n },
                uRow0: { int: r0 },
                uStride: { int: wOut.width }
              },
              output: wOut,
              rect: {
                x: 0,
                y: 0,
                width: (Math.ceil(n / f.hd) * kqW) / (h2W ? 2 : 1),
                height: Math.min(f.hd, n)
              },
              work: h2W ? 6 : 3
            })
            await this.gpu.runAsync({
              name: "matmul",
              program: nativeMultiply ?? multiply,
              inputs: { uW: wOut, uA: fa },
              uniforms: {
                uKq: { int: kq },
                uKqW: { int: kqW },
                uKq0: { int: kq0 },
                uHd: { int: f.hd },
                uSliceRows: { int: quadRows },
                uKqSlice: { int: kqSlice },
                uRow0: { int: r0 },
                uStride: { int: wOut.width }
              },
              output: [out, partner],
              rect: {
                x: r0 / 4,
                y: s0 * quadRows,
                width: Math.ceil(n / 4),
                height: quadRows * (s1 - s0)
              },
              // (Per K texel: 4 rows' weights and 4 tokens' activations.)
              work: kqSlice * 8,
              ...(nativeMultiply ? TUNING.gemmColumnBands > 0 && { bandColumns: 16 } : this.glColumnBands())
            })
          }
        }
        this.slicing.set(out.id, { split: slices, stride: quadRows, quad: true })
        return
      }
      if (w.rgba4) {
        await this.gpu.runAsync({
          name: "matmul",
          program: this.gpu.program(
            this.scalesOf(w, K.matmulQuad({ zeros: !!w.zeros || w.packed }))
          ),
          inputs: weights,
          uniforms: {
            ...shape,
            uSliceRows: { int: quadRows },
            uKbSlice: { int: kbSlice }
          },
          output: [out, partner],
          rect: { x: 0, y: 0, width: w.groups, height: quadRows * slices },
          // (Per block: 4 rows' weights and 4 tokens' activations.)
          work: kbSlice * 40,
          ...this.glColumnBands()
        })
        this.slicing.set(out.id, { split: slices, stride: quadRows, quad: true })
        return
      }
    }
    if (w.rgba4) {
      // Narrow matrices split K in two, stacked in the output, so there are
      // enough fragments to fill the GPU (consumers add the slices).
      const pairRows = Math.ceil(rows / 2)
      // Two slices only where the output has the rows for both (the narrow
      // matmuls' outputs are sized for two; LFM2.5 350M's in_proj is narrow
      // enough to qualify, but its output isn't).
      const slices =
        w.groups <= TUNING.sliceGroups && out.height >= 2 * pairRows ? 2 : 1
      const width = w.groups
      await this.gpu.runAsync({
        name: "matmul",
        program: this.gpu.program(
          this.scalesOf(w, K.matmulPair({ zeros: !!w.zeros || w.packed }))
        ),
        inputs: weights,
        uniforms: {
          ...shape,
          uSliceRows: { int: pairRows },
          uKbSlice: { int: Math.ceil(w.blocks / slices) }
        },
        output: out,
        rect: { x: 0, y: 0, width, height: pairRows * slices },
        work: Math.ceil(w.blocks / slices) * 24,
        ...this.glColumnBands()
      })
      this.slicing.set(out.id, { split: slices, stride: pairRows })
      return
    }
    this.slicing.set(out.id, { split: 1, stride: 0 })
    // 8-bit weights in a storage buffer (see storeQuants): the native pair matmul.
    const stored8 = bits === 8 ? this.storedQuants(w) : null
    if (stored8) {
      await this.gpu.runAsync({
        name: "matmul",
        program: (this.gpu as unknown as { nativeProgram(spec: NativeSpec): Program }).nativeProgram((this.pairSpec8 ??= bufferPairMatmul())),
        inputs: { uA: a, uS: w.scales, uW: stored8 },
        uniforms: { ...shape, uWidth: { int: w.quants.width } },
        output: out,
        rect: { x: 0, y: 0, width: w.groups, height: Math.ceil(rows / 2) },
        work: w.blocks * 24,
        ...this.glColumnBands()
      })
      return
    }
    await this.gpu.runAsync({
      name: "matmul",
      program: this.gpu.program(
        K.matmul({
          bits,
          zeros: !!w.zeros,
          wrap: false,
          precision: this.precision,
          bytes: w.bytes,
          unorm: w.unorm,
          packed: w.packed,
          pair: true
        })
      ),
      inputs: weights,
      uniforms: shape,
      output: out,
      rect: { x: 0, y: 0, width: w.groups, height: Math.ceil(rows / 2) },
      work: w.blocks * 24,
      ...this.glColumnBands()
    })
  }

  /**
   * A few rows' matmul split along K, so the GPU has enough fragments to
   * keep busy: partial sums into the partial buffer (row m · slices + s).
   * Returns the number of slices.
   */
  private splitPartials(
    a: Texture,
    w: GpuMatrix,
    rows: number,
    norm?: Texture | "prescaled",
    /** The output's first texel column (another matmul's slices beside it: see linearAttend). */
    x0 = 0
  ) {
    // RGBA4 and 4-bit byte texels have the split kernel that folds norms;
    // the generic one folds the add's prescaled input (see foldEach).
    const own = w.rgba4 || (w.bits === 4 && w.bytes) || !!w.stored
    if (norm && !own && norm !== "prescaled")
      throw new Error("Only 4-bit RGBA4 or byte matmuls fold the norm weights in.")
    const bits = w.bits
    if (bits === 6)
      throw new GpuError("Q6_K weights are only read as the output head.")
    const weights = {
      uA: a,
      uW: w.quants,
      uS: w.scales,
      ...(w.zeros && { uZ: w.zeros })
    }
    const shape = {
      uKb: { int: w.blocks },
      uH: { int: w.rowsPerColumn },
      uRowTexels: { int: w.rowTexels },
      uG: { int: w.groups }
    }
    const stored = this.storedQuants(w)
    if (stored) {
      const gpu = this.gpu as unknown as { nativeProgram(spec: NativeSpec): unknown }
      const format = this.buffers.partial.internal === GL.RGBA16F ? "rgba16float" : "rgba32float"
      const normed = !norm ? false : norm === "prescaled" ? norm : "weights"
      // bufferPairs: lanes over K about TUNING.bufferBlocks blocks each (a
      // power of 2, at most maxSplit slices); else bufferMatvec, one slice.
      const pairs = TUNING.bufferPairs < 0 ? !((this.tuned.matvec ?? 0) > 0) : TUNING.bufferPairs > 0
      const lanes = pairs
        ? Math.max(2, Math.min(TUNING.maxSplit, 2 ** Math.floor(Math.log2(Math.max(1, w.blocks / TUNING.bufferBlocks)))))
        : 1
      const bits = w.bits === 8 ? 8 : 4
      const zeros = w.packed
      const key = `${normed} ${format} ${lanes} ${bits} ${zeros}`
      let spec = this.bufferSpecs.get(key)
      if (!spec)
        this.bufferSpecs.set(key, (spec = lanes > 1
          ? bufferPairs({ normed, lanes, format, bits, zeros })
          : bufferMatvec({ normed, wg: TUNING.bufferMatvecThreads, format, bits, zeros })))
      this.gpu.run({
        name: "matmul",
        program: gpu.nativeProgram(spec) as Program,
        inputs: !norm
          ? { uA: a, uS: w.scales, uW: stored }
          : norm === "prescaled"
            ? { uA: this.scaledOf(a), uS: w.scales, uR: a, uW: stored }
            : { uA: a, uS: w.scales, uNorm: norm, uR: a, uW: stored },
        uniforms: {
          uKb: { int: w.blocks },
          uH: { int: w.rowsPerColumn },
          uRowTexels: { int: w.rowTexels },
          uWidth: { int: w.quants.width },
          uG: { int: w.groups },
          uAOff: { int: norm === "prescaled" ? this.config.hiddenSize / 4 + K.RMS_PARTS : 0 },
          uROff: { int: this.config.hiddenSize / 4 },
          uTexels: { int: this.config.hiddenSize / 4 },
          uParts: { int: norm === "prescaled" && this.sumsInOne(a) ? 1 : this.parts },
          uEps: this.config.normEps
        },
        output: this.buffers.partial,
        rect: { x: x0, y: 0, width: w.groups, height: rows }
      })
      return lanes
    }
    {
      const split = Math.max(
        1,
        Math.min(
          TUNING.maxSplit,
          // Slices per token: a few tokens (a token and its drafts) take
          // as long as one, their fragments' chains of reads no longer.
          Math.ceil(TUNING.splitFragments / w.groups),
          w.blocks / 4
        )
      )
      const kbSlice = Math.ceil(w.blocks / split)
      const slices = Math.ceil(w.blocks / kbSlice)
      this.gpu.run({
        name: "matmul",
        program: this.gpu.program(
          K.withRms(own
            ? K.matmulSplitRgba4({
                zeros: !!w.zeros,
                bytes: !w.rgba4,
                unorm: w.unorm,
                packed: w.packed,
                normed: !norm ? false : norm === "prescaled" ? norm : "weights",
                offset: x0 > 0,
                stream: TUNING.streamMatvec >= 0 ? TUNING.streamMatvec > 0 : this.tuned.streamMatvec === true
              })
            : K.matmul({
                bits,
                zeros: !!w.zeros,
                wrap: false,
                precision: this.precision,
                bytes: w.bytes,
                unorm: w.unorm,
                packed: w.packed,
                split: true,
                normed: norm === "prescaled" ? norm : false,
                offset: x0 > 0
              }), norm === "prescaled" && this.sumsInOne(a) ? 1 : this.parts)
        ),
        inputs: !norm
          ? weights
          : norm === "prescaled"
            ? { ...weights, uA: this.scaledOf(a), uR: a }
            : { ...weights, uNorm: norm, uR: a },
        uniforms: {
          ...shape,
          uSplit: { int: slices },
          uKbSlice: { int: kbSlice },
          ...(x0 > 0 && { uGx0: { int: x0 } }),
          ...(norm && {
            uAOff: { int: this.config.hiddenSize / 4 + K.RMS_PARTS },
            uROff: { int: this.config.hiddenSize / 4 },
            uParts: { int: this.parts },
            uTexels: { int: this.config.hiddenSize / 4 },
            uEps: this.config.normEps
          })
        },
        output: this.buffers.partial,
        rect: { x: x0, y: 0, width: w.groups, height: rows * slices }
      })
      return slices
    }
  }

  /**
   * Rows [row0, row0 + rows) of x's partial sums of squares, into the rms
   * buffer; x is f32, a pair tensor (a matmul's output), or a split
   * matmul's partial sums over `split` slices.
   */
  private get parts() { return this.tuned.rmsParts ?? TUNING.rmsParts }

  /** The native fused adds' programs, which write a row's sum of squares in one column. */
  private fusedAddPrograms = new Set<unknown>()

  /**
   * Whether residual texture x's sums of squares are in one column (a fused
   * add wrote it last), so a norm reads that column alone, not all parts.
   */
  private sumsInOne(x: Texture) {
    return "lastWriter" in this.gpu && this.fusedAddPrograms.has(this.gpu.lastWriter(x))
  }

  /** K-slices of a Q6_K head: the native head's on WebGPU (see TUNING.nativeHead). */
  get q6HeadSplit() {
    return this.nativeHeadOn ? TUNING.nativeHeadSplit : TUNING.q6HeadSplit
  }

  /** Whether Q6_K word heads run as the native pass (see TUNING.nativeHead). */
  private get nativeHeadOn() {
    return TUNING.nativeHead > 0 && "nativeProgram" in this.gpu
  }

  /** The staging texture ordered Q6_K uploads go through (see writeMatrix). */
  private orderStage: Texture | null = null
  /** Raw MatMulNBits rows on their way into byte texels (see writeMatrix). */
  private rawStage: Texture | null = null

  /**
   * The pick's native programs (see TUNING.nativePick), or null: not on
   * WebGL, nor where rows see tokens otherwise than the mask (draft
   * batches' seenRows), which the GLSL passes handle.
   */
  private nativePicks(seenRows: ReadonlyArray<SeenRow>, chunks: number) {
    if (TUNING.nativePick <= 0 || seenRows.length > 0 || !("nativeProgram" in this.gpu)) return null
    const gpu = this.gpu
    const spec = (key: string, make: () => NativeSpec) => {
      let s = this.pickSpecs.get(key)
      if (!s) this.pickSpecs.set(key, (s = make()))
      return gpu.nativeProgram(s)
    }
    return {
      max: spec("max", chunkMax),
      argmax: spec(`argmax ${chunks}`, () => chunkArgmax({ chunks, rows: false })),
      rows: spec(`rows ${chunks}`, () => chunkArgmax({ chunks, rows: true }))
    }
  }
  private pickSpecs = new Map<string, NativeSpec>()

  /**
   * w's quants in a storage buffer for the buffer matvecs (copied on first
   * use, outside tune), or null: not on WebGL, nor for weights other than
   * 4-bit bytes with zero points of 8.
   */
  private storedQuants(w: GpuMatrix): Texture | null {
    return w.stored ?? null
  }

  /**
   * A layer matrix's quants to a storage buffer for the buffer matvecs
   * (TUNING.bufferMatvec) as soon as they're written, and the texture
   * deleted: before any pass binds it, as Safari keeps a destroyed
   * texture's memory while a bind group holds it (an iPhone 16 Pro Max's
   * tab, copying 1.4 GB of textures to buffers, lived when no pass had
   * bound them and hit its memory limit when one had). From then on every
   * pass reading these weights reads the buffer (splitPartials, matmul's
   * dequant). Not on WebGL, nor for weights other than 4-bit bytes with
   * zero points of 8.
   */
  private storeQuants(w: GpuMatrix) {
    if (TUNING.bufferMatvec <= 0 || !("toStorage" in this.gpu)) return
    // (Zero points packed with the scales are read by the buffer kernels too: an ONNX model's asymmetric matrices.)
    if (!((w.bits === 4 || w.bits === 8) && w.bytes && !w.rgba4 && !w.zeros && !(w.packed && w.bits !== 4) && !w.quantPanels)) return
    w.stored = this.gpu.toStorage(w.quants)
    if (w.stored) this.gpu.delete(w.quants)
  }
  /** WebGL's prompt matmuls (rows of tokens) paced in bands of columns, every token in each (TUNING.glColumnBands). */
  private glColumnBands() {
    return TUNING.glColumnBands > 0 && !("nativeProgram" in this.gpu) ? { bandColumns: 32 } : {}
  }
  /**
   * Whether a prompt's add, its queries' prep and its short convolution run
   * as single native passes, a row's reduction shared by a workgroup's
   * threads through workgroup memory (TUNING.fusedAdd 3, promptPrep and
   * promptConv -1): where tune found threads sharing a pass's K loops
   * faster (its lanes; see TUNING.lanes), the same cooperation. A Pixel 10
   * Pro, whose lanes lose by 130-900%, read LFM2.5 350M's prompts at
   * 841-875 tokens/s with the q/k/v pass apart where 720-773 fused, at
   * 837-838 with the convolution's apart, 800-817 with the add's; an M3
   * Ultra, whose lanes win, gains 1-3% from each.
   */
  private workgroupFusions() {
    return this.gpu.coopThreads > 0
  }
  private bufferSpecs = new Map<string, NativeSpec>()
  private promptSpec: NativeSpec | undefined
  private pairSpec8: NativeSpec | undefined
  private recurrenceSpecs = new Map<string, NativeSpec>()
  private recurrenceSpecOf(shape: { keyHeadDim: number; valueHeadDim: number }) {
    const key = `${shape.keyHeadDim} ${shape.valueHeadDim} ${TUNING.recurrenceRows}`
    let spec = this.recurrenceSpecs.get(key)
    if (!spec) this.recurrenceSpecs.set(key, (spec = linRecurrence({ ...shape, rows: TUNING.recurrenceRows })))
    return spec
  }
  /** The native chunk loop for a Gated DeltaNet shape (see recurrenceChunks), or null where it doesn't fit. */
  private chunkLoopSpec(shape: { keyHeadDim: number; valueHeadDim: number; chunk: number }) {
    const key = `chunks ${shape.keyHeadDim} ${shape.valueHeadDim} ${shape.chunk}`
    let spec = this.chunkLoops.get(key)
    if (spec === undefined) {
      try {
        spec = linChunkLoop(shape)
      } catch {
        spec = null
      }
      this.chunkLoops.set(key, spec)
    }
    return spec
  }
  private chunkLoops = new Map<string, NativeSpec | null>()
  private bufferDequantSpec(h2: boolean, toBuffer = false, bits: 4 | 8 = 4, zeros = false) {
    const key = `dequant ${h2} ${toBuffer} ${bits} ${zeros}`
    let spec = this.bufferSpecs.get(key)
    if (!spec) this.bufferSpecs.set(key, (spec = bufferDequant({ h2, toBuffer, bits, zeros })))
    return spec
  }
  private promptBufferSpec(trips: number) {
    const rowGroups = TUNING.gemmRowGroups > 0 ? TUNING.gemmRowGroups : this.gpu.coopThreads > 0 ? 1 : 2
    const key = `gemm ${trips} ${rowGroups}`
    let spec = this.bufferSpecs.get(key)
    if (!spec) this.bufferSpecs.set(key, (spec = promptMatmulBuffer({ trips, rowGroups })))
    return spec
  }
  /**
   * Whether f16Pack 3's scratch is a storage buffer, its dequant and matmul
   * native passes (TUNING.bufferGemm): every f16 matrix's weights in a
   * buffer (see storeQuants).
   */
  private bufferGemm() {
    return TUNING.bufferGemm > 0 && "storage" in this.gpu && this.f16Mats().every((m) => !!m.stored)
  }

  /** The native attention's partial sums (see flashChunks). */
  private flashPartial: Texture | null = null
  private flashSpecsMap = new Map<string, NativeSpec>()
  private flashSpecs(kvWidth: number, split: number) {
    const c = this.config
    const gpu = this.gpu as WebGpu
    const get = (key: string, make: () => NativeSpec) => {
      let sp = this.flashSpecsMap.get(key)
      if (!sp) this.flashSpecsMap.set(key, (sp = make()))
      return gpu.nativeProgram(sp)
    }
    return {
      chunks: get(`chunks ${kvWidth} ${TUNING.flashKeys}`, () => flashChunks({ headDim: c.headDim, heads: c.numHeads, kvHeads: c.numKvHeads, kvWidth, keys: TUNING.flashKeys })),
      combine: get(`combine ${split}`, () => flashCombine({ headDim: c.headDim, heads: c.numHeads, split, gate: this.gate() }))
    }
  }

  private attentionPreps = new Map<string, NativeSpec>()
  private attentionPrepSpec(split: number, kvWidth: number) {
    const c = this.config
    const key = `${split} ${kvWidth}`
    let spec = this.attentionPreps.get(key)
    if (!spec)
      this.attentionPreps.set(key, (spec = attentionPrep({ headDim: c.headDim, heads: c.numHeads, kvHeads: c.numKvHeads, kvWidth, split, ropeLow: K.ROPE_LOW })))
    return spec
  }

  /** The native prompt conv for an in_proj output read as `y` in `slices` K-slices, or null where it doesn't fit. */
  private promptConvSpec(y: "pair" | "quad", slices: number) {
    const key = `conv ${y} ${slices}`
    let spec = this.promptPreps.get(key)
    if (spec === undefined) {
      try {
        spec = promptConv({ width: this.config.convKernel, y, slices })
      } catch {
        spec = null
      }
      this.promptPreps.set(key, spec)
    }
    return spec
  }
  /** The native prompt prep for a q|k|v|gate projection read as `y` in `slices` K-slices, or null where it doesn't fit. */
  private promptPrepSpec(y: "pair" | "quad", slices: number) {
    const c = this.config
    const key = `batch ${y} ${slices}`
    let spec = this.promptPreps.get(key)
    if (spec === undefined) {
      try {
        spec = promptPrep({ headDim: c.headDim, heads: c.numHeads, kvHeads: c.numKvHeads, y, slices, ropeLow: K.ROPE_LOW })
      } catch {
        spec = null
      }
      this.promptPreps.set(key, spec)
    }
    return spec
  }
  private promptPreps = new Map<string, NativeSpec | null>()

  private nativeExpandedSpec(split: boolean, prescaled: boolean) {
    const wg = TUNING.nativeExpandedThreads
    const key = `expanded ${split} ${prescaled} ${wg}`
    let spec = this.nativeHeads.get(key)
    if (!spec) this.nativeHeads.set(key, (spec = expandedHead({ split, prescaled, wg })))
    return spec
  }

  private nativeHeads = new Map<string, NativeSpec>()
  private nativeHeadSpec(split: boolean, prescaled: boolean, kbSlice: number, ordered: boolean) {
    const key = `${split} ${prescaled} ${kbSlice} ${ordered}`
    let spec = this.nativeHeads.get(key)
    if (!spec) this.nativeHeads.set(key, (spec = wordHead({ split, prescaled, kbSlice, ordered })))
    return spec
  }

  /** Native fused adds by shape (see addNormed). */
  private fusedAdds = new Map<string, NativeSpec | null>()

  private rms(x: Texture, row0: number, rows: number, split = -1) {
    const texels = this.config.hiddenSize / 4
    this.gpu.run({
      name: "rms",
      program: this.gpu.program(
        split > 0
          ? K.withSplit(K.rmsPartial("split"), split)
          : split < 0
            ? K.rmsPartial("f32")
            : this.readPair(K.rmsPartial("pair"), "uX", x)
      ),
      inputs: { uX: x, ...(split === 0 && this.quadInputs("uX", x)) },
      uniforms: {
        uTexels: { int: texels },
        uChunk: { int: Math.ceil(texels / this.parts) },
        uSplitIn: { int: Math.max(split, 0) },
        ...this.pairIn(x)
      },
      output: this.buffers.rms,
      rect: { x: 0, y: row0, width: this.parts, height: rows }
    })
  }

  /**
   * h2 out = rms-normed x · norm. x's rows' sums of squares are in its last
   * columns (the add that wrote it drew them), or, with `ownSums` false,
   * are computed here into the rms buffer first.
   */
  private normToH2(
    x: Texture,
    norm: Texture,
    out: Texture,
    rows: number,
    rowIn = -1,
    ownSums = true
  ) {
    const { hiddenSize } = this.config
    if (!ownSums) this.rms(x, Math.max(rowIn, 0), rows)
    this.gpu.run({
      name: "norm",
      program: this.gpu.program(K.withRms(K.NORM_TO_H2, this.parts)),
      inputs: { uX: x, uR: ownSums ? x : this.buffers.rms, uNorm: norm },
      uniforms: {
        uRowIn: { int: rowIn },
        uROff: { int: ownSums ? hiddenSize / 4 : 0 },
        uParts: { int: this.parts },
        uTexels: { int: hiddenSize / 4 },
        uEps: this.config.normEps
      },
      output: out,
      rect: { x: 0, y: 0, width: hiddenSize / 8, height: rows }
    })
  }

  /**
   * The RGBA16F texture a norm or SwiGLU can draw its output into for
   * `uses` (each matrix with the texture its matmul writes), if each of
   * those matmuls reads it: each takes the 4-token f16 matmul, which needs
   * the output's partner texture (see matmul). Any other reads h2.
   */
  private floatInput(rows: number, uses: ReadonlyArray<readonly [GpuMatrix, Texture]>, allowed = true) {
    if (!allowed || TUNING.fusePromptInputs <= 0 || !this.quadMatmuls || TUNING.quad <= 0 || rows < TUNING.f16MinRows || (this.f16Pack() & 1) || !uses.length || !uses.every(([m, out]) => this.takesF16(m) && this.quads.has(out.id))) return null
    const f = this.f16Scratch()
    return f && f.a ? f.a[f.next] : null
  }

  private normInput(x: Texture, norm: Texture, rows: number, uses: ReadonlyArray<readonly [GpuMatrix, Texture]>, ownSums = true, allowed = true) {
    const out = this.floatInput(rows, uses, allowed)
    if (!out) {
      this.normToH2(x, norm, this.buffers.normed, rows, -1, ownSums)
      return this.buffers.normed
    }
    if (!ownSums) this.rms(x, 0, rows)
    const { hiddenSize } = this.config
    this.gpu.run({
      name: "norm",
      program: this.gpu.program(K.withRms(K.NORM_TO_F16, this.parts)),
      inputs: { uX: x, uR: ownSums ? x : this.buffers.rms, uNorm: norm },
      uniforms: {
        uRowIn: { int: -1 }, uROff: { int: ownSums ? hiddenSize / 4 : 0 },
        uParts: { int: this.parts }, uTexels: { int: hiddenSize / 4 }, uEps: this.config.normEps
      },
      output: out,
      rect: { x: 0, y: 0, width: hiddenSize / 4, height: rows }
    })
    return out
  }

  /**
   * out = h + post-norm(y) (or h + y without a norm: LFM2), with
   * out's rows' sums of squares drawn into its last columns in the same pass
   * (same output texture: no new render pass). y is a pair tensor, or a
   * split matmul's partial sums over `split` slices.
   */
  private addNormed(
    h: Texture,
    y: Texture,
    norm: Texture | undefined,
    out: Texture,
    rows: number,
    split = 0,
    next?: Texture
  ) {
    const { hiddenSize } = this.config
    const texels = hiddenSize / 4
    // On WebGPU, the add in one native dispatch (see gpu/native.ts): a
    // token's (y a split matmul's partials) and, with fusedAdd 2, a
    // prompt's (y a pair tensor).
    const source = split > 0 ? "split" : this.quadOf(y) ? "quad" : "pair"
    const slicing = this.slicing.get(y.id) ?? { split: 1, stride: 0 }
    const slices = split > 0 ? 0 : Math.max(slicing.split, 1)
    const fused = (split > 0 ? TUNING.fusedAdd > 0 : TUNING.fusedAdd === 2 || (TUNING.fusedAdd === 3 && this.workgroupFusions())) && "nativeProgram" in this.gpu
    const key = `${texels} ${!!norm} ${!!next} ${source} ${slices}`
    if (fused && !this.fusedAdds.has(key)) {
      // (Null for a shape it doesn't fit: the passes below.)
      let spec: NativeSpec | null = null
      try {
        spec = fusedAdd({ texels, normed: !!norm, next: !!next, y: source, slices })
      } catch {
        spec = null
      }
      this.fusedAdds.set(key, spec)
    }
    const spec = fused ? this.fusedAdds.get(key) : null
    if (spec && "nativeProgram" in this.gpu) {
      const program = this.gpu.nativeProgram(spec)
      this.fusedAddPrograms.add(program)
      const partner = this.quadOf(y)
      this.gpu.run({
        name: "add",
        program,
        inputs: { uH: h, uY: y, ...(source === "quad" && partner && { uY_B: partner }), ...(norm && { uNorm: norm }), ...(next && { uNext: next }) },
        uniforms: {
          uSplitIn: { int: split },
          uParts: { int: this.parts },
          uEps: this.config.normEps,
          uBase: { int: texels + K.RMS_PARTS },
          uSliceStride: { int: slicing.stride }
        },
        output: next ? [out, this.scaledOf(out)] : out,
        rect: { x: 0, y: 0, width: texels, height: rows }
      })
      return
    }
    // Through the sandwich norm (LFM3), y's sums of squares first.
    const normed = !!norm
    if (normed) this.rms(y, 0, rows, split)
    const read = (source: string) =>
      this.gpu.program(
        K.withRms(split
          ? this.fromSplit(source, ["uY"], split)
          : this.readPair(source, "uY", y), this.parts)
      )
    const common = {
      uTexels: { int: texels },
      uEps: this.config.normEps,
      uROff: { int: 0 },
      uParts: { int: this.parts },
      uSplitIn: { int: split },
      ...this.pairIn(y)
    }
    const inputs = {
      uH: h,
      uY: y,
      ...(!split && this.quadInputs("uY", y)),
      ...(norm && { uR: this.buffers.rms, uNorm: norm })
    }
    // With `next`, a second output: h2(x · next's weights), drawn in the
    // same pass, for the next split matmuls (no norm pass).
    const mrt = !!next
    const output = next ? [out, this.scaledOf(out)] : out
    this.gpu.run({
      name: "add",
      program: read(K.addNormed(normed, mrt)),
      inputs,
      uniforms: common,
      output,
      rect: { x: 0, y: 0, width: texels, height: rows }
    })
    this.gpu.run({
      name: "add",
      program: read(K.rmsOfAdd(normed, mrt)),
      inputs,
      uniforms: {
        ...common,
        uChunk: { int: Math.ceil(texels / this.parts) },
        uBase: { int: texels }
      },
      output,
      rect: { x: texels, y: 0, width: this.parts, height: rows }
    })
    if (next) {
      const base = texels + K.RMS_PARTS
      this.gpu.run({
        name: "add",
        program: read(K.addScaled(normed)),
        inputs: { ...inputs, uNext: next },
        uniforms: { ...common, uBase: { int: base } },
        output,
        rect: { x: base, y: 0, width: texels / 2, height: rows }
      })
    }
  }

  /** The conv block; returns out_proj's split (see `project`). */
  private async convolve(
    layer: NonNullable<GpuLayer["conv"]>,
    rows: number,
    input: MatIn
  ) {
    const c = this.config
    const b = this.buffers
    const dt = c.hiddenSize / 4
    const split = await this.project(input.a, layer.inProj, b.bcx, rows, input.norm)
    const bcx = split ? this.partials() : b.bcx
    const read = (source: string) =>
      this.gpu.program(
        split
          ? this.fromSplit(source, ["uBCx"], split)
          : this.readPair(source, "uBCx", bcx)
      )
    const bcxIn = { uBCx: bcx, ...(!split && this.quadInputs("uBCx", bcx)) }
    if (this.prefixSaving) {
      const copy = this.f32(c.hiddenSize, c.convKernel - 1)
      this.prefixSaving.state.conv.push(copy)
      this.gpu.run({
        name: "conv",
        program: read(K.convState(c.convKernel)),
        inputs: { ...bcxIn, uState: layer.state[0] },
        uniforms: { uDt: { int: dt }, uM: { int: this.prefixSaving.row }, uSplitIn: { int: split }, ...this.pairIn(bcx) },
        output: copy,
        rect: { x: 0, y: 0, width: dt, height: c.convKernel - 1 }
      })
    }
    if (rows <= SPLIT_ROWS && TUNING.fuseConv > 0 && this.mrt.conv) {
      // A few rows: the output and the new state in one pass.
      this.gpu.run({
        name: "conv",
        program: read(K.convFused(c.convKernel)),
        inputs: { ...bcxIn, uState: layer.state[0], uConv: layer.kernel },
        uniforms: {
          uDt: { int: dt },
          uM: { int: rows },
          uSplitIn: { int: split },
          ...this.pairIn(bcx)
        },
        output: [b.convSmall, layer.state[1]],
        rect: {
          x: 0,
          y: 0,
          width: dt,
          height: Math.max(rows, c.convKernel - 1)
        }
      })
      this.convPrefixes(layer, rows, bcxIn, read, split)
      layer.state.reverse()
      return this.project(b.convSmall, layer.outProj, b.opOut, rows)
    }
    // A prompt batch's (a pair tensor's rows): the convolution and the
    // state after it in one native pass (see gpu/native.ts's promptConv).
    const batchConv = !split && rows >= c.convKernel - 1 && (TUNING.promptConv > 0 || (TUNING.promptConv < 0 && this.workgroupFusions())) && "nativeProgram" in this.gpu
      ? this.promptConvSpec(this.quadOf(bcx) ? "quad" : "pair", this.slicing.get(bcx.id)?.split ?? 1)
      : null
    if (batchConv && "nativeProgram" in this.gpu) {
      this.gpu.run({
        name: "conv",
        program: this.gpu.nativeProgram(batchConv),
        inputs: { ...bcxIn, uState: layer.state[0], uConv: layer.kernel },
        uniforms: { uDt: { int: dt }, uSliceStride: { int: this.slicing.get(bcx.id)?.stride ?? 0 } },
        output: [b.convOut, layer.state[1]],
        rect: { x: 0, y: 0, width: c.hiddenSize / 8, height: rows }
      })
    } else {
      this.gpu.run({
        name: "conv",
        program: read(K.shortConv(c.convKernel)),
        inputs: { ...bcxIn, uState: layer.state[0], uConv: layer.kernel },
        uniforms: { uDt: { int: dt }, uSplitIn: { int: split }, ...this.pairIn(bcx) },
        output: b.convOut,
        rect: { x: 0, y: 0, width: c.hiddenSize / 8, height: rows }
      })
      this.gpu.run({
        name: "conv",
        program: read(K.convState(c.convKernel)),
        inputs: { ...bcxIn, uState: layer.state[0] },
        uniforms: {
          uDt: { int: dt },
          uM: { int: rows },
          uSplitIn: { int: split },
          ...this.pairIn(bcx)
        },
        output: layer.state[1],
        rect: { x: 0, y: 0, width: dt, height: c.convKernel - 1 }
      })
    }
    this.convPrefixes(layer, rows, bcxIn, read, split)
    layer.state.reverse()
    return this.project(b.convOut, layer.outProj, b.opOut, rows)
  }

  /** In a draft batch, the conv state after each shorter prefix too, for accept(). */
  private convPrefixes(
    layer: NonNullable<GpuLayer["conv"]>,
    rows: number,
    bcxIn: Record<string, Texture>,
    read: (source: string) => Program,
    split: number
  ) {
    if (!this.keepPrefixes || rows <= 1) return
    const c = this.config
    const dt = c.hiddenSize / 4
    this.gpu.run({
      name: "conv",
      program: read(K.convPrefixes(c.convKernel)),
      inputs: { ...bcxIn, uState: layer.state[0] },
      uniforms: { uDt: { int: dt }, uSplitIn: { int: split }, ...this.pairIn(bcxIn.uBCx) },
      output: layer.prefixes,
      rect: { x: 0, y: 0, width: dt, height: (c.convKernel - 1) * rows }
    })
  }

  /** Cache slots (rows) a token's queries take in queryInCache. */
  private querySlots() {
    const c = this.config
    return Math.ceil((c.numHeads * c.headDim) / 4 / this.kvWidth())
  }

  /**
   * Whether a `rows`-row batch at pos0 puts its queries in free slots of
   * the layer's cache (see TUNING.qInCache): slots past the batch's own,
   * which the cache holds (a ring: none its queries' windows reach), each
   * head's query texels within one cache row.
   */
  private queryInCache(
    layer: NonNullable<GpuLayer["attention"]>,
    rows: number,
    pos0: number
  ) {
    const c = this.config
    if (TUNING.qInCache <= 0 || rows > SPLIT_ROWS) return false
    if (this.kvWidth() % (c.headDim / 4) !== 0) return false
    return queriesFit({
      pos0,
      rows,
      slots: this.querySlots(),
      ring: layer.ring,
      window: c.slidingWindow,
      positions: this.kvPositions
    })
  }

  private async attend(
    layer: NonNullable<GpuLayer["attention"]>,
    type: LayerType,
    rows: number,
    pos0: number,
    input: MatIn,
    paced = false,
    /** Whether the queries may go in the cache (see queryInCache; not while probing). */
    queriesInCache = false,
    /** Sees the q|k|v|gate projection before the out projection reuses the partials. */
    projected?: (split: number) => Promise<void>
  ) {
    const c = this.config
    const b = this.buffers
    const sliding = type === "sliding_attention"
    const ropeTable = sliding ? this.ropeSliding : this.ropeFull
    // Rotary positions: from the batch's first (text), or each row's own
    // (a batch with image rows: the 3D variant of the head pass).
    const br = this.batchRope
    const mrope = "positions" in br
    const rope: { uniforms: { uPos0: { int: number } }; inputs: Record<string, Texture> } = "positions" in br
      ? { uniforms: { uPos0: { int: 0 } }, inputs: { uPositions: br.positions } }
      : { uniforms: { uPos0: { int: br.pos0 } }, inputs: {} }
    const pairTexels = (sliding ? c.ropeDimsSliding : c.ropeDimsFull) / 8
    const qDims = c.numHeads * c.headDim
    const kvDims = c.numKvHeads * c.headDim
    const split = await this.project(
      input.a,
      layer.qkvg,
      b.qkvg,
      rows,
      input.norm
    )
    await projected?.(split)
    const qkvgIn = split ? this.partials() : b.qkvg
    const read = (source: string) =>
      this.gpu.program(
        split
          ? this.fromSplit(source, ["uQKVG"], split)
          : this.readPair(source, "uQKVG", qkvgIn)
      )
    const qkvgInputs = {
      uQKVG: qkvgIn,
      ...(!split && this.quadInputs("uQKVG", qkvgIn))
    }
    const kvWidth = this.kvWidth()
    const kvShift = layer.ring ? 30 : this.kvShift
    const blockRows = 2 ** kvShift
    const inCache = queriesInCache && this.queryInCache(layer, rows, pos0)
    const qSlots = this.querySlots()
    const qSlot0 = pos0 + rows
    // Queries: normalized, rotated, scaled by 1/sqrt(head dim); into the
    // cache's slots from qSlot0 (in runs, as the keys below), or b.q.
    const cacheShape = { kvWidth, slots: qSlots, texels: qDims / 4 }
    // One token's, from split partial sums, its queries in one run of slots
    // and its key in one row: a native pass for all three (see attentionPrep).
    const qRuns = inCache ? slotRuns(qSlot0, qSlots * rows, blockRows, layer.ring) : []
    const kSlot = layer.ring ? pos0 % layer.ring : pos0
    const prep = rows === 1 && split > 0 && split <= 16 && !mrope && inCache && qRuns.length === 1 && TUNING.nativePrep > 0 &&
      c.headDim / 4 <= 64 && "nativeProgram" in this.gpu
      ? this.gpu.nativeProgram(this.attentionPrepSpec(split, kvWidth))
      : null
    // A prompt batch's (a pair tensor's rows, the queries in their own
    // texture): the queries, keys and values of every token in one native
    // pass (see gpu/native.ts's promptPrep).
    // (Not a batch that wraps around its ring: its later tokens' keys must land last.)
    const batchPrep = !prep && !split && !inCache && !mrope && (!layer.ring || rows <= layer.ring) && (TUNING.promptPrep > 0 || (TUNING.promptPrep < 0 && this.workgroupFusions())) && "nativeProgram" in this.gpu
      ? this.promptPrepSpec(this.quadOf(qkvgIn) ? "quad" : "pair", this.slicing.get(qkvgIn.id)?.split ?? 1)
      : null
    if (batchPrep && "nativeProgram" in this.gpu) {
      this.gpu.run({
        name: "heads",
        program: this.gpu.nativeProgram(batchPrep),
        inputs: { ...qkvgInputs, uQNorm: layer.qNorm, uKNorm: layer.kNorm, uRope: ropeTable },
        uniforms: {
          ...rope.uniforms,
          uPairTexels: { int: pairTexels },
          uSliceStride: { int: this.slicing.get(qkvgIn.id)?.stride ?? 0 },
          uRing: { int: layer.ring },
          uKvShift: { int: kvShift },
          uKvWidth: { int: kvWidth },
          uSlot0: { int: pos0 },
          uEps: c.normEps,
          uScale: 1 / Math.sqrt(c.headDim)
        },
        output: [b.q, layer.kv],
        rect: { x: 0, y: 0, width: qDims / 4, height: rows }
      })
    } else if (prep) {
      this.gpu.run({
        name: "heads",
        program: prep,
        inputs: { uQKVG: qkvgIn, uQNorm: layer.qNorm, uKNorm: layer.kNorm, uRope: ropeTable },
        uniforms: {
          ...rope.uniforms,
          uPairTexels: { int: pairTexels },
          uQX0: { int: qRuns[0].block * kvWidth },
          uQRow0: { int: qRuns[0].row0 },
          uKX0: { int: Math.floor(kSlot / blockRows) * kvWidth },
          uKRow0: { int: kSlot % blockRows },
          uEps: c.normEps,
          uScale: 1 / Math.sqrt(c.headDim)
        },
        output: layer.kv,
        rect: { x: 0, y: 0, width: 1, height: 1 }
      })
    } else {
    if (inCache)
      for (const { s0, row0, block, run } of slotRuns(
        qSlot0,
        qSlots * rows,
        blockRows,
        layer.ring
      )) {
        const x0 = block * kvWidth
        this.gpu.run({
          name: "heads",
          program: read(
            K.headPrep({
              out: "cache",
              rope: true,
              norm: true,
              headDim: c.headDim,
              cache: cacheShape,
              ...(mrope && { mrope })
            })
          ),
          inputs: { ...qkvgInputs, uNorm: layer.qNorm, uRope: ropeTable, ...rope.inputs },
          uniforms: {
            uSplitIn: { int: split },
            ...this.pairIn(qkvgIn),
            uBase: { int: 0 },
            uPairTexels: { int: pairTexels },
            ...rope.uniforms,
            uRowOffset: { int: row0 - s0 },
            uXOff: { int: x0 },
            uEps: c.normEps,
            uScale: 1 / Math.sqrt(c.headDim)
          },
          output: layer.kv,
          rect: { x: x0, y: row0, width: kvWidth, height: run }
        })
      }
    else
      this.gpu.run({
        name: "heads",
        program: read(
          K.headPrep({ out: "f32", rope: true, norm: true, headDim: c.headDim, ...(mrope && { mrope }) })
        ),
        inputs: { ...qkvgInputs, uNorm: layer.qNorm, uRope: ropeTable, ...rope.inputs },
        uniforms: {
          uSplitIn: { int: split },
          ...this.pairIn(qkvgIn),
          uBase: { int: 0 },
          uPairTexels: { int: pairTexels },
          ...rope.uniforms,
          uRowOffset: { int: 0 },
          uXOff: { int: 0 },
          uEps: c.normEps,
          uScale: 1 / Math.sqrt(c.headDim)
        },
        output: b.q,
        rect: { x: 0, y: 0, width: qDims / 4, height: rows }
      })
    // Keys and values into the cache at positions pos0..., in runs of rows
    // split where a ring cache wraps around or a block of rows ends. (A
    // ring is one block, however tall.)
    for (let m0 = 0; m0 < rows;) {
      const slot = layer.ring ? (pos0 + m0) % layer.ring : pos0 + m0
      const row0 = slot % blockRows
      const x0 = Math.floor(slot / blockRows) * kvWidth
      const run = Math.min(
        rows - m0,
        blockRows - row0,
        layer.ring ? layer.ring - slot : Infinity
      )
      // A fragment at row y reads token m = y - offset.
      const offset = row0 - m0
      this.gpu.run({
        name: "heads",
        program: read(
          K.headPrep({ out: "h2", rope: true, norm: true, headDim: c.headDim, ...(mrope && { mrope }) })
        ),
        inputs: { ...qkvgInputs, uNorm: layer.kNorm, uRope: ropeTable, ...rope.inputs },
        uniforms: {
          uSplitIn: { int: split },
          ...this.pairIn(qkvgIn),
          uBase: { int: qDims / 4 },
          uPairTexels: { int: pairTexels },
          ...rope.uniforms,
          uRowOffset: { int: offset },
          uXOff: { int: x0 },
          uEps: c.normEps,
          uScale: 1
        },
        output: layer.kv,
        rect: { x: x0, y: row0, width: kvDims / 8, height: run }
      })
      // Values beside the keys: same texture, so no new render pass.
      this.gpu.run({
        name: "heads",
        program: read(K.values(c.headDim)),
        inputs: qkvgInputs,
        uniforms: {
          uSplitIn: { int: split },
          ...this.pairIn(qkvgIn),
          uBase: { int: (qDims + kvDims) / 4 },
          uRowOffset: { int: offset },
          uXOff: { int: x0 + kvDims / 8 }
        },
        output: layer.kv,
        rect: { x: x0 + kvDims / 8, y: row0, width: kvDims / 8, height: run }
      })
      m0 += run
    }
    }
    await this.attention(layer, type, rows, pos0, split, qkvgIn, paced, inCache)
    if (this.panelErrors || this.chunkErrors) {
      // panelError: this layer's output again by the one-pass kernel,
      // from the same queries and cache (that output goes on).
      const qDims = c.numHeads * c.headDim
      const read = async () => {
        this.gpu.run({
          program: this.gpu.program(K.H2_TO_F32),
          inputs: { uX: b.attnOut },
          output: this.debugTexture(),
          rect: { x: 0, y: 0, width: qDims / 4, height: rows }
        })
        return this.gpu.read(this.debugTexture(), {
          x: 0,
          y: 0,
          width: qDims / 4,
          height: rows
        })
      }
      const got = await read()
      const saved = TUNING.attendChunkTexels,
        savedDecode = TUNING.attendChunkTexelsDecode
      TUNING.attendChunkTexels = this.chunkErrors ? 1 : 0
      if (this.chunkErrors) TUNING.attendChunkTexelsDecode = 1
      try {
        await this.attention(
          layer,
          type,
          rows,
          pos0,
          split,
          qkvgIn,
          false,
          inCache
        )
      } finally {
        TUNING.attendChunkTexels = saved
        TUNING.attendChunkTexelsDecode = savedDecode
      }
      const want = await read()
      let worst = 0
      for (let row = 0; row < rows; row++) {
        let err = 0,
          ref = 0,
          finite = true
        for (let i = row * qDims; i < (row + 1) * qDims; i++) {
          finite &&= Number.isFinite(got[i]) && Number.isFinite(want[i])
          err += (got[i] - want[i]) ** 2
          ref += want[i] ** 2
        }
        worst = Math.max(
          worst,
          finite ? Math.sqrt(err / Math.max(ref, 1e-30)) : Infinity
        )
      }
      ;(this.panelErrors ?? this.chunkErrors)?.push(worst)
      // The normal attention output continues through the reference model.
      if (this.chunkErrors)
        await this.attention(
          layer,
          type,
          rows,
          pos0,
          split,
          qkvgIn,
          false,
          inCache
        )
    }
    return this.project(b.attnOut, layer.o, b.opOut, rows)
  }

  /**
   * Attention proper, after attend has the queries in buffers.q (or the
   * cache's slots: inCache) and the keys and values in the cache: buffers.attnOut for rows from pos0, the
   * gate read from the q|k|v|gate projection (qkvgIn, or its split
   * partial sums).
   */
  private async attention(
    layer: NonNullable<GpuLayer["attention"]>,
    type: LayerType,
    rows: number,
    pos0: number,
    split: number,
    qkvgIn: Texture,
    paced = false,
    /** Whether attend put the queries in the cache's slots past the batch (see queryInCache). */
    inCache = false
  ) {
    const c = this.config
    const b = this.buffers
    const sliding = type === "sliding_attention"
    const qSlots = this.querySlots()
    const qSlot0 = pos0 + rows
    const qDims = c.numHeads * c.headDim
    const kvDims = c.numKvHeads * c.headDim
    const read = (source: string) =>
      this.gpu.program(
        split
          ? this.fromSplit(source, ["uQKVG"], split)
          : this.readPair(source, "uQKVG", qkvgIn)
      )
    const qkvgInputs = {
      uQKVG: qkvgIn,
      ...(!split && this.quadInputs("uQKVG", qkvgIn))
    }
    const kvWidth = this.kvWidth()
    const kvShift = layer.ring ? 30 : this.kvShift
    const kvAt = { uKvShift: { int: kvShift }, uKvWidth: { int: kvWidth } }
    // Queries in slices, so the scores buffer stays small: each slice sees
    // keys up to its last query (and, in sliding layers, from its first
    // query's window on). The buffer holds scoreRows queries of every head
    // at TUNING.scoreTexels's width; with fewer keys than that, panels of them
    // sit side by side and a slice holds more queries. More keys than that
    // go in segments: each segment's scores, then its chunks of weighted
    // values (attendChunk), and at the end all chunks combined.
    // A fragment takes G query heads (a row group) sharing a KV head:
    // scores of member i in texture i.
    const group = c.numHeads / c.numKvHeads
    const G = this.headGroup()
    const rowGroups = c.numHeads / G
    const growScores = !!c.linear
    let scoreTex = b.scores
    const halfScores = scoreTex[0]?.internal === GL.RGBA16F
    const scoreInputs = (textures: Texture[]) => Object.fromEntries(
      textures.map((t, i) => [i === 0 ? "uS" : `uS${i}`, t])
    )
    let sIn = scoreInputs(scoreTex)
    const panelRows = growScores ? rowGroups * Math.min(this.capacity, this.scoreRows()) : scoreTex[0].height
    const scoreWidth = growScores ? Math.min(TUNING.scoreTexels, Math.ceil(this.contextLength / 4)) : scoreTex[0].width
    // One token whose queries are in the cache's slots, from split partial
    // sums: two native passes (see flashChunks), the scores never written out.
    const U = c.headDim / 8
    if (
      rows === 1 && inCache && split > 0 && split <= 16 && TUNING.nativeAttend > 0 && "nativeProgram" in this.gpu &&
      flashFits({ headDim: c.headDim, heads: c.numHeads, kvHeads: c.numKvHeads, keys: TUNING.flashKeys })
    ) {
      const j0 = sliding ? Math.max(0, pos0 - c.slidingWindow + 1) : 0
      const keys = pos0 + 1 - j0
      const chunks = Math.ceil(keys / TUNING.flashKeys)
      const width = c.numHeads * (2 * U + 1)
      const height = Math.ceil(this.contextLength / TUNING.flashKeys)
      if (this.flashPartial?.width !== width || this.flashPartial.height < height) {
        if (this.flashPartial) this.gpu.delete(this.flashPartial)
        this.flashPartial = this.gpu.texture(GL.RGBA32F, width, height)
      }
      const fc = this.flashSpecs(kvWidth, split)
      this.gpu.run({
        name: "attend",
        program: fc.chunks,
        inputs: { uKV: layer.kv },
        uniforms: {
          uQSlot0: { int: qSlot0 },
          uRing: { int: layer.ring },
          uJ0: { int: j0 },
          uKeys: { int: keys },
          uQPos: { int: pos0 },
          uWindow: { int: sliding ? c.slidingWindow : 0 },
          uKvShift: { int: kvShift }
        },
        output: this.flashPartial,
        rect: { x: 0, y: 0, width: c.numKvHeads, height: chunks }
      })
      this.gpu.run({
        name: "attend",
        program: fc.combine,
        inputs: { uP: this.flashPartial, uQKVG: qkvgIn },
        uniforms: { uChunks: { int: chunks }, uGateTexel: { int: (qDims + 2 * kvDims) / 4 } },
        output: b.attnOut,
        rect: { x: 0, y: 0, width: c.numHeads, height: 1 }
      })
      return
    }
    for (let m0 = 0; m0 < rows;) {
      const j0 = sliding ? Math.max(0, pos0 + m0 - c.slidingWindow + 1) : 0
      const widest = Math.ceil((pos0 + rows - j0) / 4)
      // (Segments need chunked attention: without it, whole rows.)
      const segTexels = Math.min(
        widest,
        scoreWidth,
        this.mrt.chunks ? TUNING.scoreTexels : Infinity
      )
      const segmentsMax = Math.ceil(widest / segTexels)
      const panels = Math.max(1, Math.floor(scoreWidth / segTexels))
      let slice = Math.min(
        rows - m0,
        Math.max(1, Math.floor((panels * panelRows) / rowGroups))
      )
      // Chunks per segment: up to ATTEND_CHUNKS of at least
      // attendChunkTexels texels (fewer where segments are many).
      // (Segments are parallel work already: with many, fewer chunks each,
      // so a slice keeps more queries.)
      const perSegment = Math.max(
        1,
        Math.min(
          Math.max(2, Math.ceil(ATTEND_CHUNKS / segmentsMax)),
          Math.floor(this.gpu.limits.maxTextureSize / segmentsMax)
        )
      )
      const texelsPer = (t: number) =>
        Math.max(TUNING.attendChunkTexels, Math.ceil(t / perSegment))
      const chunkRows = this.scoreRows() * ATTEND_CHUNKS
      if (segmentsMax > 1) {
        const chunksMax =
          segmentsMax * Math.ceil(segTexels / texelsPer(segTexels))
        slice = Math.min(slice, Math.max(1, Math.floor(chunkRows / chunksMax)))
      }
      const keys = pos0 + m0 + slice - j0
      const kt = Math.ceil(keys / 4)
      const segments = Math.ceil(kt / segTexels)
      // Chunks a segment holds, the last one's padded with empty chunks.
      const segChunksOf = (t: number) => Math.ceil(Math.min(kt, segTexels) / t)
      const chunksOf = (t: number) => segments * segChunksOf(t)
      let chunkTexels = texelsPer(Math.min(kt, segTexels))
      if (rows <= SPLIT_ROWS && TUNING.attendDecodeChunking > 0) {
        // Writing: few queries, so the chunks are the parallel work. About
        // √(kt / 2) texels each balances the chunk pass (each fragment
        // walks a chunk) against the combine (each walks every chunk).
        const segKt = Math.min(kt, segTexels)
        let t = Math.min(
          segKt,
          Math.max(
            TUNING.attendChunkTexels,
            2 ** Math.round(Math.log2(Math.sqrt(kt / 2)))
          )
        )
        while (t < segKt && slice * chunksOf(t) > chunkRows) t *= 2
        chunkTexels = t
      }
      const total = rowGroups * slice
      const ring = {
        uRows: { int: slice },
        uM0: { int: m0 },
        uRing: { int: layer.ring },
        uKt: { int: Math.min(kt, segTexels) },
        uPanelRows: { int: panelRows },
        ...kvAt
      }
      // Attention's multiply-adds over `n` keys, in matmul terms.
      const attentionWork = (n: number) =>
        slice * n * c.numHeads * c.headDim * 2 * TUNING.paceAttention
      // Segments stack as rows: segment s of a pass at rows s · total on.
      const segRows = { uSegRows: { int: total }, uSegKeys: { int: 4 * segTexels } }
      const scores = async (sj0: number, sKeys: number, n: number) => {
        const width = Math.ceil((n * total) / panelRows) * Math.min(kt, segTexels)
        const height = Math.min(panelRows, n * total)
        if (growScores) {
          const next = this.growScoreBuffers(width, height, G)
          if (next !== scoreTex) {
            scoreTex = next
            sIn = scoreInputs(next)
          }
        }
        await this.gpu.runAsync({
          name: "scores",
          program: this.gpu.program(
            K.scores(c.headDim, G, inCache ? { kvWidth, slots: qSlots } : undefined, halfScores)
          ),
          inputs: inCache ? { uK: layer.kv } : { uQ: b.q, uK: layer.kv },
          uniforms: {
            ...ring,
            ...(inCache && { uQSlot0: { int: qSlot0 } }),
            ...segRows,
            uJ0: { int: sj0 },
            uKeys: { int: sKeys },
            uPos0: { int: pos0 },
            uWindow: { int: sliding ? c.slidingWindow : 0 },
            uGroup: { int: group },
            uTotalRows: { int: n * total }
          },
          output: G === 1 ? scoreTex[0] : scoreTex,
          rect: { x: 0, y: 0, width, height },
          work: (c.headDim / 8) * (4 + 2 * G)
        })
      }
      const chunks = chunksOf(chunkTexels)
      const chunkPast =
        rows <= SPLIT_ROWS
          ? Math.max(TUNING.attendChunkTexelsDecode, 2 * TUNING.attendChunkTexels)
          : 2 * TUNING.attendChunkTexels
      const fitSegments = Math.max(1, Math.floor((panels * panelRows) / total))
      if (
        rows <= SPLIT_ROWS &&
        // (The choice tune() timed at its long cache applies from there on;
        // on a GPU whose passes cost little (WebGPU), at any length.)
        ((this.tuned.flashDecode != null && (keys >= (this.tuned.flashDecodeFrom ?? 0) || this.gpu.cheapPasses) ? this.tuned.flashDecode : TUNING.flashDecode)) > 0 &&
        TUNING.attendChunkTexels > 0 &&
        this.mrt.chunks &&
        kt > (TUNING.flashPast >= 0 ? TUNING.flashPast : chunkPast) &&
        slice === rows - m0 &&
        segments <= fitSegments
      ) {
        // Writing: few queries, so many short chunks of keys are the
        // parallel work, in one pass after the scores, then one combine
        // (which walks every chunk: about FLASH_CHUNKS of them at most).
        const panelW = qDims / 8 / G
        const lay = panelLayout({
          kt,
          segTexels,
          segments,
          slice,
          panelW,
          maxWidth: this.gpu.limits.maxTextureSize,
          chunkTexels: TUNING.flashChunkTexels,
          maxChunks: FLASH_CHUNKS,
          maxPanels: TUNING.flashPanels
        })
        const bufs = this.panelBuffers(lay.width, lay.height, G)
        const outPanels = { uOutPanels: { int: lay.across }, uOutPanelW: { int: panelW } }
        await scores(j0, keys, segments)
        await this.gpu.runAsync({
          name: "attend",
          program: this.gpu.program(K.attendPanel(c.headDim, G, TUNING.flashUnroll, TUNING.flashOnline > 0, halfScores)),
          inputs: { ...sIn, uV: layer.kv },
          uniforms: {
            ...ring,
            ...segRows,
            ...outPanels,
            uSegChunks: { int: lay.segChunks },
            uJ0: { int: j0 },
            uKeys: { int: keys },
            uGroup: { int: group },
            uVOff: { int: kvDims / 8 },
            uSliceQ: { int: slice },
            uChunkTexels: { int: lay.chunkTexels }
          },
          output: bufs,
          rect: { x: 0, y: 0, width: lay.width, height: lay.height },
          work: lay.chunkTexels * (2 * G + 8)
        })
        await this.gpu.runAsync({
          name: "attend",
          program: read(K.attendPanelCombine(c.headDim, this.gate(), G)),
          inputs: {
            ...Object.fromEntries(bufs.map((t, k) => [`uP${k}`, t])),
            ...qkvgInputs
          },
          uniforms: {
            uSplitIn: { int: split },
            ...this.pairIn(qkvgIn),
            ...outPanels,
            uM0: { int: m0 },
            uSliceQ: { int: slice },
            uChunks: { int: lay.chunks },
            uGateTexel: { int: (qDims + 2 * kvDims) / 4 }
          },
          output: b.attnOut,
          rect: { x: 0, y: m0, width: qDims / 8, height: slice },
          work: lay.chunks * 3 * G + 2
        })
        m0 += slice
        continue
      }
      if (
        segments > 1 ||
        (this.mrt.chunks &&
          TUNING.attendChunkTexels > 0 &&
          kt > chunkPast &&
          slice * chunks <= chunkRows)
      ) {
        // Long key ranges: chunks of keys in parallel, then combined. As
        // many segments as the scores buffer holds go in one scores pass
        // and one chunk pass (writing reads every key in 3 passes).
        const chunkBufs = this.chunkBuffers(slice * chunks, G)
        const segChunks = segChunksOf(chunkTexels)
        const fit = Math.max(1, Math.floor((panels * panelRows) / total))
        const perPass = Math.min(segments, fit, TUNING.segmentGroup || Infinity)
        let chunk0 = 0
        for (let s0 = 0; s0 < segments; s0 += perPass) {
          const n = Math.min(perPass, segments - s0)
          const gj0 = j0 + 4 * s0 * segTexels
          const gKeys = Math.min(keys - 4 * s0 * segTexels, 4 * n * segTexels)
          await scores(gj0, gKeys, n)
          await this.gpu.runAsync({
            name: "attend",
            program: this.gpu.program(K.attendChunk(c.headDim, G, halfScores)),
            inputs: { ...sIn, uV: layer.kv },
            uniforms: {
              ...ring,
              ...segRows,
              uSegChunks: { int: segChunks },
              uJ0: { int: gj0 },
              uKeys: { int: gKeys },
              uGroup: { int: group },
              uVOff: { int: kvDims / 8 },
              uSliceQ: { int: slice },
              uChunkTexels: { int: chunkTexels },
              uChunk0: { int: chunk0 }
            },
            output: chunkBufs,
            rect: {
              x: 0,
              y: chunk0 * slice,
              width: qDims / 8 / G,
              height: slice * n * segChunks
            },
            work: chunkTexels * (2 * G + 8)
          })
          chunk0 += n * segChunks
          if (paced) await this.charge(attentionWork(gKeys))
        }
        await this.gpu.runAsync({
          name: "attend",
          program: read(K.attendCombine(c.headDim, this.gate(), G)),
          inputs: {
            ...Object.fromEntries(chunkBufs.map((t, k) => [`uP${k}`, t])),
            ...qkvgInputs
          },
          uniforms: {
            uSplitIn: { int: split },
            ...this.pairIn(qkvgIn),
            uM0: { int: m0 },
            uSliceQ: { int: slice },
            uChunks: { int: chunk0 },
            uGateTexel: { int: (qDims + 2 * kvDims) / 4 }
          },
          output: b.attnOut,
          rect: { x: 0, y: m0, width: qDims / 8, height: slice },
          work: chunk0 * 3 * G + 2
        })
        m0 += slice
        continue
      }
      if (paced) await this.charge(attentionWork(keys))
      await scores(j0, keys, 1)
      await this.gpu.runAsync({
        name: "attend",
        program: read(K.attend(c.headDim, this.gate(), G)),
        inputs: { ...sIn, uV: layer.kv, ...qkvgInputs },
        uniforms: {
          uSplitIn: { int: split },
          ...this.pairIn(qkvgIn),
          ...ring,
          uJ0: { int: j0 },
          uKeys: { int: keys },
          uGroup: { int: group },
          uGateTexel: { int: (qDims + 2 * kvDims) / 4 },
          uVOff: { int: kvDims / 8 }
        },
        output: b.attnOut,
        rect: { x: 0, y: m0, width: qDims / 8, height: slice },
        work: Math.ceil(keys / 4) * 10
      })
      m0 += slice
    }
  }

  /** Attention's output gate (see K.Gate). */
  private gate(): K.Gate {
    const c = this.config
    return c.attentionGate ? (c.gatePerDim ? "dim" : true) : false
  }

  /**
   * A Gated DeltaNet block (Qwen3.5; see LinearAttentionWeights): the
   * gates, in_proj, the convolution, q̂ and k̂, then the recurrence (a
   * prompt batch in chunks, see recurrenceChunks; otherwise a token at a
   * time, two passes each), the gated norm. Returns out_proj's split (see
   * `project`).
   */
  private async linearAttend(
    layer: NonNullable<GpuLayer["linear"]>,
    rows: number,
    input: MatIn,
    paced: boolean,
    /** A row to save the state after (into `saving`: its conv state, then its recurrent state). */
    saveAt?: number,
    /** Whether writing's fused block may run (see fuseLinear; not while probing). */
    fusable = false
  ) {
    const lin = this.linDims()
    const b = this.buffers
    const t = b.lin
    if (!lin || !t) throw new Error("The model's config has no Gated DeltaNet shapes.")
    if (fusable && saveAt === undefined && !this.prefixSaving && this.fusesLinear(layer, rows))
      return this.linearFused(layer, rows, input)
    const dk = lin.l.keyHeadDim
    const dv = lin.l.valueHeadDim
    const width = lin.l.convKernel
    const H4 = lin.H / 4
    const K4 = lin.K / 4
    const V4 = lin.V / 4
    const C4 = lin.C / 4
    /** `source` reading pair or split input `name` as `project` left it. */
    const reader = (name: string, tex: Texture, split: number) => (source: string) =>
      this.gpu.program(
        split ? this.fromSplit(source, [name], split) : this.readPair(source, name, tex)
      )
    const pairInputs = (name: string, tex: Texture, split: number) => ({
      [name]: tex,
      ...(!split && this.quadInputs(name, tex))
    })
    // Prompt batches run the recurrence in chunks (see recurrenceChunks).
    const chunk = TUNING.linChunk
    const chunked =
      this.chunkedRecurrence &&
      chunk > 0 &&
      chunk % 4 === 0 &&
      rows >= Math.max(2, TUNING.linChunkMin) &&
      lin.l.numKeyHeads === lin.l.numValueHeads
    // beta and decay (its log, for chunks).
    // (An 8-bit `ba` that goes through the scratch buffer writes a 4-token
    // matmul's two textures: the block's output's, free until out_proj.
    // Its 32 rows over all of K took a prompt 0.25 ms a layer as a pair
    // matmul on an M3 Ultra, 0.05 ms dequantized and in 4 K-slices.)
    const baOut = rows > SPLIT_ROWS && this.eightScratch(layer.ba, b.opOut) ? b.opOut : t.ba
    const baSplit = await this.project(input.a, layer.ba, baOut, rows, input.norm)
    const baIn = baSplit ? this.partials() : baOut
    this.gpu.run({
      name: "linear",
      program: reader("uBA", baIn, baSplit)(K.linGates(H4, chunked)),
      inputs: {
        ...pairInputs("uBA", baIn, baSplit),
        uA: layer.a,
        uDt: layer.dtBias
      },
      uniforms: { uSplitIn: { int: baSplit }, ...this.pairIn(baIn) },
      output: t.gates,
      rect: { x: 0, y: 0, width: 2 * H4, height: rows }
    })
    // q|k|v|z, then silu(conv(q|k|v)) and the conv state after the batch.
    const split = await this.project(input.a, layer.inProj, t.inProj, rows, input.norm)
    const inTex = split ? this.partials() : t.inProj
    const read = reader("uIn", inTex, split)
    const inInputs = pairInputs("uIn", inTex, split)
    const inUniforms = { uSplitIn: { int: split }, ...this.pairIn(inTex) }
    if (this.prefixSaving) {
      const copy = this.f32(lin.C, width - 1)
      this.prefixSaving.state.linear.push(copy)
      this.gpu.run({
        name: "linear",
        program: read(K.linConvState(width)),
        inputs: { ...inInputs, uState: layer.convState[0] },
        uniforms: { ...inUniforms, uM: { int: this.prefixSaving.row } },
        output: copy,
        rect: { x: 0, y: 0, width: C4, height: width - 1 }
      })
    }
    this.gpu.run({
      name: "linear",
      program: read(K.linConv(width)),
      inputs: { ...inInputs, uState: layer.convState[0], uConv: layer.kernel },
      uniforms: inUniforms,
      output: t.qkv,
      rect: { x: 0, y: 0, width: C4, height: rows }
    })
    for (const m of [rows, ...(saveAt !== undefined ? [saveAt] : [])]) {
      // Own the anchor copy before compiling or drawing it. Do not allocate it
      // ahead of the full-batch state pass, which can itself be interrupted.
      const out = m === saveAt ? this.f32(lin.C, width - 1) : layer.convState[1]
      if (m === saveAt) this.saving?.push(out)
      this.gpu.run({
        name: "linear",
        program: read(K.linConvState(width)),
        inputs: { ...inInputs, uState: layer.convState[0] },
        uniforms: { ...inUniforms, uM: { int: m } },
        output: out,
        rect: { x: 0, y: 0, width: C4, height: width - 1 }
      })
    }
    layer.convState.reverse()
    this.gpu.run({
      name: "linear",
      program: this.gpu.program(K.linPrep(dk)),
      inputs: { uQKV: t.qkv },
      uniforms: { uK4: { int: K4 } },
      output: t.qk,
      rect: { x: 0, y: 0, width: 2 * K4, height: rows }
    })
    // The recurrence: in chunks, or a token at a time.
    const shape = { keyHeadDim: dk, valueHeadDim: dv }
    const common = { uK4: { int: K4 }, uH4: { int: H4 } }
    if (chunked) await this.recurrenceChunks(layer, rows, chunk, paced, saveAt)
    else for (let r = 0; r < rows; r++) {
      const deltaIn = { uS: layer.state[0], uQK: t.qk, uQKV: t.qkv, uG: t.gates }
      const rect = { x: 0, y: r, width: V4, height: 1 }
      if (this.mrt.lin) {
        this.gpu.run({
          name: "recurrence",
          program: this.gpu.program(K.linDelta({ ...shape, out: "both" })),
          inputs: deltaIn,
          uniforms: common,
          output: [t.delta, t.o],
          rect
        })
      } else {
        for (const [out, tex] of [["delta", t.delta], ["o", t.o]] as const) {
          this.gpu.run({
            name: "recurrence",
            program: this.gpu.program(K.linDelta({ ...shape, out })),
            inputs: deltaIn,
            uniforms: common,
            output: tex,
            rect
          })
        }
      }
      this.gpu.run({
        name: "recurrence",
        program: this.gpu.program(K.linState(shape)),
        inputs: { uS: layer.state[0], uQK: t.qk, uDelta: t.delta, uG: t.gates },
        uniforms: { ...common, uRow: { int: r } },
        output: layer.state[1],
        rect: { x: 0, y: 0, width: V4, height: dk }
      })
      layer.state.reverse()
      if (r + 1 === saveAt) this.saving?.push(this.copyOf(layer.state[0]))
      if (r + 1 === this.prefixSaving?.row) this.prefixSaving.state.linear.push(this.copyOf(layer.state[0]))
      // (Each token reads and writes the whole state: ~4 multiply-adds an element.)
      if (paced) await this.charge(4 * lin.V * dk)
    }
    this.gpu.run({
      name: "linear",
      program: read(K.linNorm(dv)),
      inputs: { ...inInputs, uO: t.o, uNorm: layer.norm },
      uniforms: { ...inUniforms, uZ: { int: C4 }, uEps: this.config.normEps },
      output: t.out,
      rect: { x: 0, y: 0, width: V4 / 2, height: rows }
    })
    return this.project(t.out, layer.outProj, b.opOut, rows)
  }

  /** Whether a `rows`-row batch runs writing's fused Gated DeltaNet block (see fuseLinear). */
  private fusesLinear(layer: NonNullable<GpuLayer["linear"]>, rows: number) {
    const lin = this.linDims()
    if (!lin || TUNING.fuseLinear <= 0 || !this.mrt.linFused) return false
    const l = lin.l
    return (
      rows === 1 &&
      l.keyHeadDim === l.valueHeadDim &&
      l.numKeyHeads === l.numValueHeads &&
      // beta|alpha's slices beside in_proj's in the partial buffer.
      layer.inProj.groups + layer.ba.groups <= this.buffers.partial.width
    )
  }

  /**
   * Writing's Gated DeltaNet block (see fuseLinear): the same math as
   * linearAttend's token at a time, in 6 render passes: beta|alpha and
   * in_proj (one target), the convolution and its state, delta, the output,
   * k-hat and the decay (linDeltaFused), the state, the gated norm, out_proj.
   */
  private async linearFused(
    layer: NonNullable<GpuLayer["linear"]>,
    rows: number,
    input: MatIn
  ) {
    const lin = this.linDims()
    const b = this.buffers
    const t = b.lin
    if (!lin || !t) throw new Error("The model's config has no Gated DeltaNet shapes.")
    const small = t.small
    const dk = lin.l.keyHeadDim
    const dv = lin.l.valueHeadDim
    const width = lin.l.convKernel
    const H4 = lin.H / 4
    const K4 = lin.K / 4
    const V4 = lin.V / 4
    const C4 = lin.C / 4
    const partial = this.partials()
    // beta|alpha beside in_proj, drawn first: one render pass for both.
    const baX = layer.inProj.groups
    const baSplit = this.splitPartials(input.a, layer.ba, rows, input.norm, baX)
    const split = this.splitPartials(input.a, layer.inProj, rows, input.norm)
    const inUniforms = { uSplitIn: { int: split }, ...this.pairIn(partial) }
    const read = (source: string) =>
      this.gpu.program(this.fromSplit(source, ["uIn"], split))
    this.gpu.run({
      name: "linear",
      program: read(K.linConvFused(width)),
      inputs: { uIn: partial, uState: layer.convState[0], uConv: layer.kernel },
      uniforms: { ...inUniforms, uM: { int: rows } },
      output: [small.qkv, layer.convState[1]],
      rect: { x: 0, y: 0, width: C4, height: width - 1 }
    })
    layer.convState.reverse()
    const shape = { keyHeadDim: dk, valueHeadDim: dv }
    // WebGPU: delta, the output and the new state in one native pass (see TUNING.nativeRecurrence).
    if (TUNING.nativeRecurrence > 0 && "nativeProgram" in this.gpu && dk % TUNING.recurrenceRows === 0 && dk / TUNING.recurrenceRows <= 64) {
      this.gpu.run({
        name: "recurrence",
        program: this.gpu.nativeProgram(this.recurrenceSpecOf(shape)),
        inputs: { uS: layer.state[0], uQKV: small.qkv, uBA: partial, uA: layer.a, uDt: layer.dtBias },
        uniforms: { uK4: { int: K4 }, uH4: { int: H4 }, uBAx: { int: baX }, uSplitIn: { int: baSplit } },
        output: [small.o, layer.state[1]],
        rect: { x: 0, y: 0, width: V4, height: rows }
      })
      layer.state.reverse()
      this.gpu.run({
        name: "linear",
        program: read(K.linNorm(dv)),
        inputs: { uIn: partial, uO: small.o, uNorm: layer.norm },
        uniforms: { ...inUniforms, uZ: { int: C4 }, uEps: this.config.normEps },
        output: t.out,
        rect: { x: 0, y: 0, width: V4 / 2, height: rows }
      })
      return this.project(t.out, layer.outProj, b.opOut, rows)
    }
    this.gpu.run({
      name: "recurrence",
      program: this.gpu.program(K.linDeltaFused(shape)),
      inputs: {
        uS: layer.state[0],
        uQKV: small.qkv,
        uBA: partial,
        uA: layer.a,
        uDt: layer.dtBias
      },
      uniforms: {
        uK4: { int: K4 },
        uH4: { int: H4 },
        uBAx: { int: baX },
        uSplitIn: { int: baSplit }
      },
      output: [small.delta, small.o, small.k, small.decay],
      rect: { x: 0, y: 0, width: V4, height: rows }
    })
    this.gpu.run({
      name: "recurrence",
      program: this.gpu.program(K.linState({ ...shape, texelDecay: true })),
      inputs: { uS: layer.state[0], uQK: small.k, uDelta: small.delta, uG: small.decay },
      uniforms: { uK4: { int: 0 }, uH4: { int: H4 }, uRow: { int: 0 } },
      output: layer.state[1],
      rect: { x: 0, y: 0, width: V4, height: dk }
    })
    layer.state.reverse()
    this.gpu.run({
      name: "linear",
      program: read(K.linNorm(dv)),
      inputs: { uIn: partial, uO: small.o, uNorm: layer.norm },
      uniforms: { ...inUniforms, uZ: { int: C4 }, uEps: this.config.normEps },
      output: t.out,
      rect: { x: 0, y: 0, width: V4 / 2, height: rows }
    })
    return this.project(t.out, layer.outProj, b.opOut, rows)
  }

  /**
   * The recurrence over a prompt batch in chunks of `chunk` tokens from
   * row 0 (see delta-rule.ts, which these passes follow step for step):
   * the chunks' decays, Gram matrices, triangular solve and the tensors
   * that don't read the state, for every chunk at once, then two passes a
   * chunk that do (its delta and output, then the state after it).
   */
  private async recurrenceChunks(
    layer: NonNullable<GpuLayer["linear"]>,
    rows: number,
    chunk: number,
    paced: boolean,
    /** A row where a chunk ends, the state after it copied into `saving` (see forwardBatch). */
    saveAt?: number
  ) {
    const lin = this.linDims()
    const t = this.buffers.lin
    if (!lin || !t) throw new Error("The model's config has no Gated DeltaNet shapes.")
    const c = this.linChunkBuffers(chunk)
    const dk = lin.l.keyHeadDim
    const dv = lin.l.valueHeadDim
    const H4 = lin.H / 4
    const K4 = lin.K / 4
    const V4 = lin.V / 4
    const both = this.mrt.lin
    const wide = (lin.H * chunk) / 4
    // The first chunk shorter, so that one ends at saveAt.
    const first = saveAt ? saveAt % chunk || chunk : chunk
    const run = (
      program: string,
      inputs: Record<string, Texture>,
      output: Texture | Array<Texture>,
      width: number,
      uniforms: Record<string, { int: number }> = {},
      rect?: { x: number; y: number; width: number; height: number },
      work = 1
    ) =>
      this.gpu.run({
        name: "recurrence",
        program: this.gpu.program(program),
        inputs,
        uniforms: { uK4: { int: K4 }, uH4: { int: H4 }, uFirst: { int: first }, ...uniforms },
        output,
        rect: rect ?? { x: 0, y: 0, width, height: rows },
        work
      })
    run(K.linCum(chunk), { uG: t.gates }, c.cum, H4)
    const gram = { uQK: t.qk, uG: t.gates, uCum: c.cum }
    if (both) run(K.linGram({ keyHeadDim: dk, chunk, out: "both" }), gram, [c.a, c.p], wide)
    else
      for (const [out, tex] of [["a", c.a], ["p", c.p]] as const)
        run(K.linGram({ keyHeadDim: dk, chunk, out }), gram, tex, wide)
    run(K.linSolve(chunk), { uA: c.a }, c.t, wide, {}, undefined, chunk)
    const shape = { keyHeadDim: dk, valueHeadDim: dv, chunk }
    const uw = { uT: c.t, uQK: t.qk, uQKV: t.qkv, uG: t.gates, uCum: c.cum }
    const qo = { uP: c.p, uU: c.u, uW: c.w, uQK: t.qk, uCum: c.cum }
    if (both && dk === dv) {
      run(K.linUW({ ...shape, out: "both" }), uw, [c.u, c.w], V4, {}, undefined, chunk)
      run(K.linQO({ ...shape, out: "both" }), qo, [c.oi, c.qp], V4, {}, undefined, chunk)
    } else {
      run(K.linUW({ ...shape, out: "u" }), uw, c.u, V4, {}, undefined, chunk)
      run(K.linUW({ ...shape, out: "w" }), uw, c.w, K4, {}, undefined, chunk)
      run(K.linQO({ ...shape, out: "oi" }), qo, c.oi, V4, {}, undefined, chunk)
      run(K.linQO({ ...shape, out: "q" }), qo, c.qp, K4, {}, undefined, chunk)
    }
    const heads = { keyHeadDim: dk, valueHeadDim: dv }
    const prefixAt = this.prefixSaving?.row
    // WebGPU: the chunks one after another in one native dispatch (see
    // gpu/native.ts's linChunkLoop), up to a row whose state is saved; a
    // chunk a saved prefix ends inside takes the passes below (its state
    // from the chunk's first rows).
    const loop = TUNING.chunkLoop > 0 && both && "nativeProgram" in this.gpu ? this.chunkLoopSpec({ ...heads, chunk }) : null
    const end = (c0: number) => Math.min(c0 + (c0 === 0 ? first : chunk), rows)
    for (let c0 = 0; c0 < rows;) {
      if (loop && "nativeProgram" in this.gpu) {
        // Whole chunks from c0: to the batch's end, or the first row to save a state after.
        let to = c0
        while (to < rows && !(prefixAt !== undefined && to < prefixAt && prefixAt < end(to))) {
          to = end(to)
          if (to === saveAt || to === prefixAt) break
        }
        if (to > c0) {
          this.gpu.run({
            name: "recurrence",
            program: this.gpu.nativeProgram(loop),
            inputs: { uS: layer.state[0], uU: c.u, uW: c.w, uOi: c.oi, uQp: c.qp, uQK: t.qk, uCum: c.cum },
            uniforms: { uK4: { int: K4 }, uC0: { int: c0 }, uC1: { int: to }, uFirst: { int: first } },
            output: [t.o, layer.state[1]],
            rect: { x: 0, y: c0, width: V4, height: to - c0 },
            work: dk
          })
          layer.state.reverse()
          if (to === saveAt) this.saving?.push(this.copyOf(layer.state[0]))
          if (to === prefixAt) this.prefixSaving!.state.linear.push(this.copyOf(layer.state[0]))
          if (paced) await this.charge(3 * lin.V * dk * (to - c0))
          c0 = to
          continue
        }
      }
      const c1 = end(c0)
      const outIn = { uS: layer.state[0], uU: c.u, uW: c.w, uOi: c.oi, uQp: c.qp }
      const rect = { x: 0, y: c0, width: V4, height: c1 - c0 }
      if (both)
        run(K.linChunkOut({ ...heads, out: "both" }), outIn, [t.delta, t.o], V4, {}, rect, dk)
      else
        for (const [out, tex] of [["delta", t.delta], ["o", t.o]] as const)
          run(K.linChunkOut({ ...heads, out }), outIn, tex, V4, {}, rect, dk)
      const prefixRow = this.prefixSaving?.row
      if (prefixRow !== undefined && c0 < prefixRow && prefixRow < c1) {
        const copy = this.f32(lin.V, dk)
        this.prefixSaving!.state.linear.push(copy)
        run(
          K.linChunkState(heads),
          { uS: layer.state[0], uQK: t.qk, uDelta: t.delta, uCum: c.cum },
          copy, V4, { uC0: { int: c0 }, uC1: { int: prefixRow } },
          { x: 0, y: 0, width: V4, height: dk }, prefixRow - c0
        )
      }
      run(
        K.linChunkState(heads),
        { uS: layer.state[0], uQK: t.qk, uDelta: t.delta, uCum: c.cum },
        layer.state[1],
        V4,
        { uC0: { int: c0 }, uC1: { int: c1 } },
        { x: 0, y: 0, width: V4, height: dk },
        c1 - c0
      )
      layer.state.reverse()
      if (c1 === saveAt) this.saving?.push(this.copyOf(layer.state[0]))
      if (c1 === prefixRow) this.prefixSaving!.state.linear.push(this.copyOf(layer.state[0]))
      // (Reading and writing the state: ~3 multiply-adds an element a token.)
      if (paced) await this.charge(3 * lin.V * dk * (c1 - c0))
      c0 = c1
    }
  }

  /**
   * Reads the model's tokens into the cache, from `this.position` on, and
   * returns the logits after the last one. `probe` sees each layer's
   * intermediate tensors (slow: for checking against a reference).
   */
  async forward(
    tokens: ArrayLike<number>,
    probe?: LayerProbe
  ): Promise<Float32Array> {
    if (this.position + tokens.length > this.contextLength) {
      throw new Error(
        `The cache holds ${this.contextLength} tokens; this would need ${this.position + tokens.length}.`
      )
    }
    let last = 0
    let start = 0
    for (const size of this.batchSizes(tokens.length)) {
      const from = start
      start += size
      const batch = Array.from({ length: size }, (_, i) => tokens[from + i])
      last = await this.forwardBatch(batch, probe)
    }
    return this.logits(last)
  }

  /**
   * How `count` tokens split into forward batches: full batches, and a
   * last one that may run up to `capacity` so a short tail rides along.
   */
  batchSizes(count: number): Array<number> {
    const sizes: Array<number> = []
    let left = count
    while (left > this.capacity) {
      sizes.push(this.batchSize)
      left -= this.batchSize
    }
    if (left > 0) sizes.push(left)
    return sizes
  }

  /** Reads tokens into the cache without computing logits (for timing). */
  async forwardNoLogits(tokens: ArrayLike<number>) {
    if (this.position + tokens.length > this.contextLength) {
      throw new Error(
        `The cache holds ${this.contextLength} tokens; this would need ${this.position + tokens.length}.`
      )
    }
    await this.forwardBatch(Array.from(tokens), undefined)
  }

  /**
   * Reads tokens into the cache like `forward`, without reading logits
   * back; returns the row of the last token, for `candidates`.
   */
  async forwardHidden(
    tokens: ArrayLike<number>,
    options: {
      draft?: boolean
      /** Retain all final hidden rows for a calibration checksum. */
      fullRows?: boolean
      /**
       * Saves the anchor snapshot (see snapshot) after this many of the
       * tokens, from inside the batch: no batch of its own (one batch at
       * most; see canSaveInside).
       */
      saveAt?: number
      /** Observed shared prefix, copied inside this batch without another model pass. */
      prefixAt?: number
    } = {}
  ): Promise<number> {
    // Pipelines the warm-up left compiling (see warmUp): done before this reads.
    if ("compiling" in this.gpu && this.gpu.compiling && !this.gpu.deferPipelines) await this.gpu.settlePipelines()
    if (options.draft) return this.forwardDraft(Array.from(tokens))
    if (this.guessAt !== null)
      throw new Error("Forward while a guessed token is pending.")
    // A plain forward ends any draft batch: nothing left to accept.
    this.draft = null
    if (this.position + tokens.length > this.contextLength) {
      throw new Error(
        `The cache holds ${this.contextLength} tokens; this would need ${this.position + tokens.length}.`
      )
    }
    const { saveAt, prefixAt, fullRows } = options
    if (saveAt !== undefined || prefixAt !== undefined) {
      if (tokens.length > this.capacity ||
        (saveAt !== undefined && (!this.canSaveInside || !(saveAt > 0 && saveAt < tokens.length))) ||
        (prefixAt !== undefined && (this.snapshotLimit < 2 || !(prefixAt > 0 && prefixAt < tokens.length))))
        throw new Error(`Can't save an in-batch checkpoint for ${tokens.length} tokens.`)
      const same = prefixAt !== undefined && prefixAt === saveAt
      const count = Number(saveAt !== undefined) + Number(prefixAt !== undefined && !same)
      // Reserve all new state before allocating; partial copies count toward the limit.
      this.dropSnapshots(this.snapshotLimit - count, count === 1 ? saveAt !== undefined ? "anchor" : "prefix" : undefined)
      const pos0 = this.position
      const learned: Snapshot | undefined = prefixAt !== undefined && !same ? { conv: [], windows: [], linear: [] } : undefined
      this.saving = saveAt !== undefined ? [] : null
      this.prefixSaving = learned ? { row: prefixAt!, state: learned } : null
      try {
        const last = await this.forwardBatch(Array.from(tokens), undefined, false, saveAt, fullRows)
        if (saveAt !== undefined) {
          const saved: Snapshot = { conv: [], windows: [], linear: this.saving! }
          this.keepSnapshot(pos0 + saveAt, saved, true)
          if (same) this.keepSnapshot(pos0 + saveAt, saved, "prefix")
        }
        if (learned) this.keepSnapshot(pos0 + prefixAt!, learned, "prefix")
        return last
      } catch (error) {
        for (const t of this.saving ?? []) this.gpu.delete(t)
        if (learned) this.freeSnapshot(learned)
        throw error
      } finally {
        this.saving = null
        this.prefixSaving = null
      }
    }
    let last = 0
    let start = 0
    for (const size of this.batchSizes(tokens.length)) {
      const from = start
      start += size
      const batch = Array.from({ length: size }, (_, i) => tokens[from + i])
      last = await this.forwardBatch(batch, undefined, false, undefined, fullRows)
    }
    return last
  }

  /**
   * Whether the last batch kept only its last token's hidden state, as row
   * 0 (see TUNING.lastRowOnly): reading other rows' logits would be wrong.
   */
  private lastRowOnly = false

  /** Where the last draft batch started, and how many tokens it had. */
  private draft: { pos0: number; rows: number } | null = null
  /** Whether the batch being read keeps each prefix's conv state. */
  private keepPrefixes = false

  /**
   * Reads a few tokens (at most SPLIT_ROWS: a token and its drafted
   * successors, for speculative decoding) keeping what's needed to keep
   * only a prefix of them afterwards (see accept). Returns the last row.
   */
  private async forwardDraft(tokens: Array<number>) {
    if (!this.canDraft)
      throw new Error("This model's Gated DeltaNet layers can't read draft batches.")
    if (tokens.length > SPLIT_ROWS)
      throw new Error(`A draft batch holds ${SPLIT_ROWS} tokens at most.`)
    if (this.position + tokens.length > this.contextLength) {
      throw new Error(
        `The cache holds ${this.contextLength} tokens; this would need ${this.position + tokens.length}.`
      )
    }
    const pos0 = this.position
    this.keepPrefixes = true
    try {
      const last = await this.forwardBatch(tokens, undefined)
      this.draft = { pos0, rows: tokens.length }
      return last
    } finally {
      this.keepPrefixes = false
    }
  }

  /**
   * Keeps the first `count` tokens of the last draft batch and forgets the
   * rest: the cache continues after them. Keys and values past them are
   * never read before they're written again; the convolutions' state is
   * set back to the prefix's.
   */
  accept(count: number) {
    const d = this.draft
    if (!d) throw new Error("No draft batch to accept tokens of.")
    this.draft = null
    if (count < 1 || count > d.rows)
      throw new Error(`Can't keep ${count} of a ${d.rows}-token batch.`)
    // All kept: nothing to undo (a guess may have moved the cache on).
    if (count === d.rows) return
    if (count < d.rows) {
      const rows = this.config.convKernel - 1
      for (const layer of this.layers) {
        if (!layer.conv) continue
        this.gpu.run({
          name: "conv",
          program: this.gpu.program(K.COPY_ROWS),
          inputs: { uX: layer.conv.prefixes },
          uniforms: { uOff: { int: (count - 1) * rows } },
          output: layer.conv.state[0],
          rect: { x: 0, y: 0, width: layer.conv.state[0].width, height: rows }
        })
      }
    }
    this.position = d.pos0 + count
  }

  /**
   * Whether the last batch's final add drew the head's scaled input (a few
   * rows, folded norms): then the head reads that, with no norm pass.
   */
  private headScaled = false

  /** The token the last pick with a guess read forward, or -1. */
  guessedToken = -1

  /** Where a guessed token was read (see `forwardGuess`), while it's pending. */
  private guessAt: number | null = null

  /** Whether a guessed token's forward is pending (keepGuess or dropGuess). */
  get guessing() {
    return this.guessAt !== null
  }

  /**
   * Reads the token in the ids texture (CHUNK_ARGMAX's pick) as one
   * more token, without the CPU: it runs while the candidates are read back
   * and a token drawn from them. If that's the token, keepGuess, and the
   * GPU never waited; if not, dropGuess forgets it.
   */
  private async forwardGuess() {
    if (this.position + 1 > this.contextLength) return
    const at = this.position
    await this.forwardBatch([0], undefined, true)
    this.guessAt = at
    this.gpu.flush()
  }

  /** forwardGuess while `read` goes back: a guess that throws frees the read. */
  private async guessWhile(read: PendingRead) {
    try {
      await this.forwardGuess()
    } catch (error) {
      read.cancel()
      throw error
    }
  }

  /** The guessed token was the one drawn: it stays in the cache. */
  keepGuess() {
    this.guessAt = null
  }

  /**
   * The guessed token wasn't the one drawn: back to before it. Its keys and
   * values are written over by the next token; the convolutions' state
   * before it is still in their other state texture.
   */
  dropGuess() {
    if (this.guessAt === null) return
    for (const layer of this.layers) {
      layer.conv?.state.reverse()
      layer.linear?.convState.reverse()
      layer.linear?.state.reverse()
    }
    this.position = this.guessAt
    this.guessAt = null
  }

  private allowedTokens: Texture | null = null
  private allowedEnabled = false
  private allowedRows = 0
  private allowedStride = 0

  /** One allowed-token bitset broadcast across all rows of GPU selection. */
  setAllowedTokens(mask: Uint32Array | null) {
    this.setAllowedTokenRows(mask === null ? null : [mask])
  }

  /** Allowed masks after each speculative prefix, before GPU ranking/top-k. */
  setAllowedTokenRows(masks: readonly Uint32Array[] | null) {
    this.allowedEnabled = masks !== null
    this.allowedRows = masks?.length ?? 0
    if (masks === null) return
    if (masks.length === 0) throw new Error("GPU schema masks must have at least one row.")
    const words = Math.ceil(this.config.vocabSize / 32)
    const width = Math.min(256, words)
    this.allowedStride = Math.ceil(words / width)
    const height = this.allowedStride * masks.length
    if (!this.allowedTokens || this.allowedTokens.width !== width || this.allowedTokens.height < height) {
      if (this.allowedTokens) this.gpu.delete(this.allowedTokens)
      this.allowedTokens = this.gpu.texture(GL.R32UI, width, height)
    }
    const padded = new Uint32Array(width * height)
    for (let row = 0; row < masks.length; row++) padded.set(masks[row].subarray(0, words), row * width * this.allowedStride)
    this.gpu.write(this.allowedTokens, { x: 0, y: 0, width, height }, {
      format: GL.RED_INTEGER, type: GL.UNSIGNED_INT, pixels: padded
    })
  }

  /** Each token's seen state, as the GPU's mask has it (see setSeenState); padding isn't in it. */
  private maskStates: Uint8Array | null = null
  private maskLimit = 0
  private maskUpdate: Uint8Array | null = null

  /**
   * Sets which tokens the repetition penalty applies to (state 1: see
   * penalty-window.ts), and bans every id from `vocabLimit` on (padding rows
   * the tokenizer has no token for).
   */
  setSeen(tokens: Iterable<number>, vocabLimit: number) {
    const mask = this.buffers.mask
    const size = mask.width * mask.height
    this.maskLimit = Math.min(vocabLimit, size)
    const states = this.maskStates?.length === this.maskLimit ? this.maskStates : new Uint8Array(this.maskLimit)
    states.fill(0)
    const data = new Uint8Array(size)
    data.fill(2, this.maskLimit)
    for (const t of tokens) if (Number.isInteger(t) && t >= 0 && t < this.maskLimit) {
      data[t] = 1
      states[t] = 1
    }
    this.maskStates = states
    this.gpu.write(
      mask,
      { x: 0, y: 0, width: mask.width, height: mask.height },
      {
        format: GL.RED_INTEGER,
        type: GL.UNSIGNED_BYTE,
        pixels: data
      }
    )
    releaseOwnedBuffer(data.buffer)
  }

  /**
   * A token's state for the repetition and presence penalties: 0 not
   * penalized, 1 in the window from the prompt, 3 written in this reply.
   */
  setSeenState(token: number, state: 0 | 1 | 3) {
    const states = this.maskStates
    if (!states || !Number.isInteger(token) || token < 0 || token >= this.maskLimit || states[token] === state) return
    states[token] = state
    const update = this.maskUpdate ??= new Uint8Array(1)
    update[0] = state
    const width = this.buffers.mask.width
    this.gpu.write(
      this.buffers.mask,
      { x: token % width, y: Math.floor(token / width), width: 1, height: 1 },
      {
        format: GL.RED_INTEGER,
        type: GL.UNSIGNED_BYTE,
        pixels: update
      }
    )
  }

  /** Marks a generated token for repetition and presence penalties (with no window: see setSeenState). */
  markSeen(token: number) {
    this.setSeenState(token, 3)
  }

  /**
   * The likeliest tokens after `row` of the last batch, with the
   * repetition penalty applied (see `setSeen`), picked on the GPU: the
   * `chunks` 256-logit chunks with the largest logits, which hold the top
   * `chunks` tokens for sure. Reads back ~1 KB per chunk instead of every
   * logit.
   */
  async candidates(
    row: number,
    chunks: number,
    penalty: number,
    guess = false,
    presencePenalty = 0
  ): Promise<{ ids: Int32Array; logits: Float32Array }> {
    return (await this.pick(row, 1, chunks, penalty, [], guess, 0, presencePenalty))[0]
  }

  private greedyOut: Texture | null = null

  /**
   * Greedy decoding's `pick`: the likeliest token (penalized as `pick`
   * penalizes) after each of rows [row0, row0 + rows) of the last batch,
   * reading back only the token ids. Ties go to the lower id. With
   * `guess`, the last row's token is read forward meanwhile (and is the
   * token: keepGuess).
   */
  async argmax(
    row0: number,
    rows: number,
    penalty: number,
    seenRows: ReadonlyArray<SeenRow>,
    guess = false,
    presencePenalty = 0
  ): Promise<Int32Array> {
    const c = this.config
    const b = this.buffers
    const head = await this.head(row0, rows, false)
    const total = Math.ceil(c.vocabSize / CHUNK)
    const native = this.nativePicks(seenRows, total)
    const picking = {
      uOutW: { int: OUT_WIDTH },
      uPenalty: penalty,
      uAllowedEnabled: { int: this.allowedEnabled ? 1 : 0 },
      uAllowedWidth: { int: this.allowedTokens?.width ?? 1 },
      uAllowedRows: { int: this.allowedRows },
      uAllowedStride: { int: this.allowedStride },
      uPresencePenalty: presencePenalty,
      uLogitRows: { int: this.logitRows },
      uHeadSplit: { int: head.split },
      uHeadStride: { int: head.stride },
      ...seenRowUniforms(seenRows),
      uPenaltyRow0: { int: 0 }
    }
    this.gpu.run({
      name: "pick",
      program: native?.max ?? this.gpu.program(K.CHUNK_MAX_AT),
      inputs: { uLogits: head.logits, uMask: b.mask, uAllowed: this.allowedTokens ?? b.mask },
      uniforms: picking,
      output: b.chunkMax,
      rect: { x: 0, y: 0, width: total, height: rows }
    })
    if (guess) {
      this.gpu.run({
        name: "pick",
        program: native?.argmax ?? this.gpu.program(K.withChunks(K.CHUNK_ARGMAX_AT, total)),
        inputs: { uMax: b.chunkMax },
        uniforms: { uRow: { int: rows - 1 } },
        output: b.ids,
        rect: { x: 0, y: 0, width: 1, height: 1 }
      })
    }
    this.greedyOut ??= this.gpu.texture(GL.RGBA32F, 1, SPLIT_ROWS)
    this.gpu.run({
      name: "pick",
      program: native?.rows ?? this.gpu.program(K.withChunks(K.CHUNK_ARGMAX_ROWS, total)),
      inputs: { uMax: b.chunkMax },
      output: this.greedyOut,
      rect: { x: 0, y: 0, width: 1, height: rows }
    })
    const rect = { x: 0, y: 0, width: 1, height: rows }
    let data: Float32Array
    if (guess || TUNING.blockingReads <= 0) {
      const pending = this.gpu.readStart(this.greedyOut, rect)
      if (guess) await this.guessWhile(pending)
      data = await pending()
    } else {
      data = await this.gpu.readNow(this.greedyOut, rect)
    }
    const ids = Int32Array.from({ length: rows }, (_, r) => Math.round(data[r * 4]))
    this.guessedToken = guess ? ids[rows - 1] : -1
    return ids
  }

  /** Rows of logits per token in the logits texture. */
  private get logitRows() {
    return Math.ceil(this.config.vocabSize / 4 / OUT_WIDTH)
  }

  /**
   * `candidates` for rows [row0, row0 + rows) of the last batch at once:
   * one head pass, one pick, one readback. `seenRows`: where the batch's
   * rows see tokens otherwise than the penalty mask (the drafts before
   * them, and what those pushed out of the window: PenaltyWindow.ahead),
   * as drawing them one by one would.
   * With `guess`, the last row's likeliest token is read forward (see
   * forwardGuess).
   */
  async pick(
    row0: number,
    rows: number,
    chunks: number,
    penalty: number,
    seenRows: ReadonlyArray<SeenRow>,
    guess = false,
    penaltyRow0 = 0,
    presencePenalty = 0
  ): Promise<Array<{ ids: Int32Array; logits: Float32Array }>> {
    const c = this.config
    const b = this.buffers
    const head = await this.head(row0, rows, false)
    const total = Math.ceil(c.vocabSize / CHUNK)
    const count = Math.min(chunks, MAX_CHUNKS, total)
    const fast = TUNING.fastPick > 0
    const native = this.nativePicks(seenRows, total)
    const picking = {
      uOutW: { int: OUT_WIDTH },
      uPenalty: penalty,
      uAllowedEnabled: { int: this.allowedEnabled ? 1 : 0 },
      uAllowedWidth: { int: this.allowedTokens?.width ?? 1 },
      uAllowedRows: { int: this.allowedRows },
      uAllowedStride: { int: this.allowedStride },
      uPresencePenalty: presencePenalty,
      uLogitRows: { int: this.logitRows },
      uHeadSplit: { int: head.split },
      uHeadStride: { int: head.stride },
      ...seenRowUniforms(seenRows),
      uPenaltyRow0: { int: penaltyRow0 }
    }
    this.gpu.run({
      name: "pick",
      program: native?.max ?? this.gpu.program(fast ? K.CHUNK_MAX_AT : K.CHUNK_MAX),
      inputs: { uLogits: head.logits, uMask: b.mask, uAllowed: this.allowedTokens ?? b.mask },
      uniforms: picking,
      output: b.chunkMax,
      rect: { x: 0, y: 0, width: total, height: rows }
    })
    if (guess) {
      // The last row's likeliest token as the next forward's input, from
      // the chunk maxima: the forward waits on this, not on the gather.
      this.gpu.run({
        name: "pick",
        program: native?.argmax ?? this.gpu.program(
          fast ? K.withChunks(K.CHUNK_ARGMAX_AT, total) : K.CHUNK_ARGMAX
        ),
        inputs: { uMax: b.chunkMax, uLogits: head.logits, uMask: b.mask, uAllowed: this.allowedTokens ?? b.mask },
        uniforms: {
          ...picking,
          uChunks: { int: total },
          uRow: { int: rows - 1 }
        },
        output: b.ids,
        rect: { x: 0, y: 0, width: 1, height: 1 }
      })
    }
    this.gpu.run({
      name: "pick",
      program: this.gpu.program(
        fast ? K.withChunks(K.CHUNK_RANK, total) : K.CHUNK_RANK
      ),
      inputs: { uMax: b.chunkMax },
      uniforms: { uChunks: { int: total } },
      output: b.chunkRank,
      rect: { x: 0, y: 0, width: total, height: rows }
    })
    this.gpu.run({
      name: "pick",
      program: this.gpu.program(
        fast ? K.withChunks(K.CHUNK_BY_RANK, total) : K.CHUNK_BY_RANK
      ),
      inputs: { uRank: b.chunkRank },
      uniforms: { uChunks: { int: total } },
      output: b.chunkByRank,
      rect: { x: 0, y: 0, width: count, height: rows }
    })
    this.gpu.run({
      name: "pick",
      program: this.gpu.program(K.CHUNK_GATHER),
      inputs: {
        uLogits: head.logits,
        uMask: b.mask,
        uAllowed: this.allowedTokens ?? b.mask,
        uByRank: b.chunkByRank,
        uIds: b.ids
      },
      uniforms: { ...picking, uCount: { int: count } },
      output: b.candidates,
      rect: { x: 0, y: 0, width: CHUNK / 4 + 2, height: count * rows }
    })
    const texels = CHUNK / 4 + 2
    const rect = { x: 0, y: 0, width: texels, height: count * rows }
    let data: Float32Array
    if (guess || TUNING.blockingReads <= 0) {
      const pending = this.gpu.readStart(b.candidates, rect)
      // Read it forward while the candidates go back to the CPU (see guess).
      if (guess) await this.guessWhile(pending)
      data = await pending()
    } else {
      // Nothing to overlap: a blocking read returns as soon as the GPU is
      // done, where polling a fence adds milliseconds.
      data = await this.gpu.readNow(b.candidates, rect)
    }
    // The token the GPU guessed (and read forward), as it picked it.
    this.guessedToken = guess ? Math.round(data[(CHUNK / 4 + 1) * 4]) : -1
    const out = []
    for (let r = 0; r < rows; r++) {
      const ids = new Int32Array(count * CHUNK)
      const logits = new Float32Array(count * CHUNK)
      for (let k = 0; k < count; k++) {
        const at = (r * count + k) * texels
        const chunk = Math.round(data[(at + CHUNK / 4) * 4])
        for (let i = 0; i < CHUNK; i++) {
          ids[k * CHUNK + i] = chunk * CHUNK + i
          logits[k * CHUNK + i] = data[at * 4 + i]
        }
      }
      out.push({ ids, logits })
    }
    return out
  }

  /**
   * The output head on rows [row0, row0 + rows) of the last batch: logits
   * into the logits texture, logitRows rows per token (only RGBA4 heads take
   * more than one).
   */
  private async head(row0: number, rows = 1, sum = true): Promise<HeadOut> {
    if (this.lastRowOnly && (row0 !== 0 || rows !== 1))
      throw new Error(
        "The last batch kept only its last token's hidden state (row 0; see TUNING.lastRowOnly)."
      )
    const b = this.buffers
    const e = this.embedding
    if (this.headScaled) {
      // The last add drew h2(x · final norm weights): no norm pass.
      return e.rgba4
        ? this.headRgba4(e, rows, row0, sum)
        : e.bits === 4
          ? this.headQ4Words(e, rows, row0, sum)
          : this.headQ6K(e, rows, sum, row0)
    }
    this.normToH2(b.residual[0], this.finalNorm, b.lastNormed, rows, row0)
    if (e.rgba4) return this.headRgba4(e, rows, undefined, sum)
    if (e.bits === 6) return this.headQ6K(e, rows, sum)
    if (this.q4WordsHead(e)) return this.headQ4Words(e, rows, undefined, sum)
    this.gpu.run({
      name: "head",
      program: this.gpu.program(K.matmul({
        bits: e.bits, zeros: !!e.zeros, wrap: true,
        precision: this.precision, bytes: e.bytes
      })),
      inputs: {
        uA: b.lastNormed,
        uW: e.quants,
        uS: e.scales,
        ...(e.zeros && { uZ: e.zeros })
      },
      uniforms: {
        uKb: { int: e.blocks },
        uH: { int: e.rowsPerColumn },
        uRowTexels: { int: e.rowTexels },
        uG: { int: e.groups },
        uOutW: { int: OUT_WIDTH },
        uLogitRows: { int: this.logitRows }
      },
      output: b.logits,
      // Row r's logits from logits row r · logitRows (as headRgba4's).
      rect: { x: 0, y: 0, width: OUT_WIDTH, height: rows * this.logitRows }
    })
    return { logits: b.logits, split: 1, stride: 0 }
  }

  /** The scaled texture drawn with residual texture `r` (see buffers.scaled). */
  private scaledOf(r: Texture) {
    const b = this.buffers
    return r === b.residual[0] ? b.scaled[0] : b.scaled[1]
  }

  private headScratch: Texture | null = null

  /** The texture a head split in `split` K-slices of `rows` rows draws into. */
  private headTarget(split: number, rows: number) {
    const sliceRows = this.logitRows
    const height = split * rows * sliceRows
    if (split === 1) return this.buffers.logits
    // Its own texture (not the partial buffer), so the pick passes that
    // read it don't hold up the next forward's matmuls.
    if (!this.headScratch || this.headScratch.height < height) {
      if (this.headScratch) this.gpu.delete(this.headScratch)
      // (Room for a draft batch's rows where the model reads them.)
      this.headScratch = this.f32(
        OUT_WIDTH * 4,
        Math.max(height, split * (this.canDraft ? SPLIT_ROWS : 1) * sliceRows)
      )
    }
    return this.headScratch
  }

  /**
   * A split head's slices in `target`: summed into the logits texture by
   * one more pass, or (`sum` false) left for readers that add them.
   */
  private headSlices(
    target: Texture,
    split: number,
    rows: number,
    sum: boolean
  ): HeadOut {
    const b = this.buffers
    const sliceRows = this.logitRows
    if (split === 1) return { logits: b.logits, split: 1, stride: 0 }
    if (!sum) return { logits: target, split, stride: rows * sliceRows }
    this.gpu.run({
      name: "head",
      program: this.gpu.program(K.HEAD_SUM),
      inputs: { uP: target },
      uniforms: {
        uSplit: { int: split },
        uSliceRows: { int: sliceRows },
        uRows: { int: rows }
      },
      output: b.logits,
      rect: { x: 0, y: 0, width: OUT_WIDTH, height: rows * sliceRows }
    })
    return { logits: b.logits, split: 1, stride: 0 }
  }

  /**
   * The output head through a Q6_K embedding, split along K in
   * this.q6HeadSplit slices of super-blocks (as headRgba4's). With
   * `scaledRow0` (word texels only; see headScaled), reads the last add's
   * scaled input from that row on instead of lastNormed.
   */
  private async headQ6K(
    e: GpuMatrix,
    rows: number,
    sum: boolean,
    scaledRow0?: number
  ): Promise<HeadOut> {
    const b = this.buffers
    const split = Math.max(
      1,
      Math.min(
        (TUNING.q6Expanded < 0 ? this.headChoice?.split : undefined) ??
          this.q6HeadSplit,
        4,
        e.blocks
      )
    )
    const sliceRows = this.logitRows
    const target = this.headTarget(split, rows)
    const prescaled = scaledRow0 !== undefined
    if (prescaled && e.bytes)
      throw new Error("Only word-texel Q6_K heads read the scaled input.")
    const oneSums = prescaled && this.sumsInOne(b.residual[0])
    // On WebGPU, the word head as a native pass (activations in workgroup memory: see gpu/native.ts).
    const native = !("nativeProgram" in this.gpu)
      ? null
      : e.q6ExpandedWords
        ? this.gpu.nativeProgram(this.nativeExpandedSpec(split > 1, prescaled))
        : !e.q6Expanded && !e.bytes && !e.quantPanels && this.nativeHeadOn
          ? this.gpu.nativeProgram(this.nativeHeadSpec(split > 1, prescaled, Math.ceil(e.blocks / split), !!e.q6Ordered))
          : null
    if ((e.q6Ordered || e.q6ExpandedWords) && !native) throw new GpuError("Only the native heads read ordered Q6_K texels.")
    await this.weightHead(
      e,
      {
        name: "head",
        program: native ?? this.gpu.program(
          K.withRms(this.headWeightSource(
            e,
            e.q6Expanded
              ? K.q6ExpandedHead({ split: split > 1, prescaled })
              : !e.bytes
                ? K.matmulQ6KHeadWords({ split: split > 1, prescaled })
                : split > 1
                  ? K.MATMUL_Q6_K_HEAD_SPLIT
                  : K.MATMUL_Q6_K_HEAD
          ), oneSums ? 1 : this.parts)
        ),
        inputs: {
          ...(prescaled && { uR: b.residual[0] }),
          uA: prescaled ? b.scaled[0] : b.lastNormed,
          uW: e.quants,
          uS: e.scales
        },
        uniforms: {
          uKb: { int: e.blocks },
          uH: { int: e.rowsPerColumn },
          uRowTexels: { int: e.rowTexels },
          uG: { int: e.groups },
          uOutW: { int: OUT_WIDTH },
          uLogitRows: { int: sliceRows },
          uKbSlice: { int: Math.ceil(e.blocks / split) },
          uRows: { int: rows },
          ...(prescaled && {
            uRowA0: { int: scaledRow0 },
            uAOff: { int: this.config.hiddenSize / 4 + K.RMS_PARTS },
            uROff: { int: this.config.hiddenSize / 4 },
            uParts: { int: oneSums ? 1 : this.parts },
            uTexels: { int: this.config.hiddenSize / 4 },
            uEps: this.config.normEps
          })
        },
        output: target,
        rect: { x: 0, y: 0, width: OUT_WIDTH, height: split * rows * sliceRows }
      },
      rows,
      split
    )
    return this.headSlices(target, split, rows, sum)
  }

  /** Whether `e` is a 4-bit embedding in word texels the native head reads (see q4WordHead). */
  private q4WordsHead(e: GpuMatrix) {
    return e.bits === 4 && !e.rgba4 && !e.bytes && !e.quantPanels && this.nativeHeadOn
  }

  /**
   * The output head through a 4-bit embedding in word texels, as the
   * native pass (see q4WordHead), in TUNING.nativeHeadSplit K-slices.
   */
  private async headQ4Words(
    e: GpuMatrix,
    rows: number,
    scaledRow0?: number,
    sum = true
  ): Promise<HeadOut> {
    const b = this.buffers
    const split = Math.max(1, Math.min(TUNING.nativeHeadSplit, 4, e.blocks))
    const sliceRows = this.logitRows
    const target = this.headTarget(split, rows)
    const prescaled = scaledRow0 !== undefined
    const kbSlice = Math.ceil(e.blocks / split)
    const key = `q4 ${split > 1} ${prescaled} ${kbSlice} ${!!e.zeros}`
    let spec = this.nativeHeads.get(key)
    if (!spec) this.nativeHeads.set(key, (spec = q4WordHead({ split: split > 1, prescaled, kbSlice, zeros: !!e.zeros })))
    const oneSums = prescaled && this.sumsInOne(b.residual[0])
    await this.weightHead(
      e,
      {
        name: "head",
        program: (this.gpu as WebGpu).nativeProgram(spec),
        inputs: {
          ...(prescaled && { uR: b.residual[0] }),
          uA: prescaled ? b.scaled[0] : b.lastNormed,
          uW: e.quants,
          uS: e.scales,
          ...(e.zeros && { uZ: e.zeros })
        },
        uniforms: {
          uKb: { int: e.blocks },
          uH: { int: e.rowsPerColumn },
          uRowTexels: { int: e.rowTexels },
          uG: { int: e.groups },
          uOutW: { int: OUT_WIDTH },
          uLogitRows: { int: sliceRows },
          uKbSlice: { int: kbSlice },
          uRows: { int: rows },
          ...(prescaled && {
            uRowA0: { int: scaledRow0 },
            uAOff: { int: this.config.hiddenSize / 4 + K.RMS_PARTS },
            uROff: { int: this.config.hiddenSize / 4 },
            uParts: { int: oneSums ? 1 : this.parts },
            uTexels: { int: this.config.hiddenSize / 4 },
            uEps: this.config.normEps
          })
        },
        output: target,
        rect: { x: 0, y: 0, width: OUT_WIDTH, height: split * rows * sliceRows }
      },
      rows,
      split
    )
    return this.headSlices(target, split, rows, sum)
  }

  /**
   * The output head through RGBA4 weights, split along K in
   * TUNING.headSplit slices (added up by one more pass).
   */
  /**
   * With `scaledRow0`, reads the last add's scaled input from that row on
   * (see headScaled) instead of lastNormed.
   */
  private async headRgba4(
    e: GpuMatrix,
    rows: number,
    scaledRow0?: number,
    sum = true
  ): Promise<HeadOut> {
    const b = this.buffers
    // (At most 4: the pick passes add up to 4 slices themselves.)
    const split = Math.max(1, Math.min(TUNING.headSplit, 4, e.blocks))
    const sliceRows = this.logitRows
    const height = split * rows * sliceRows
    const target = this.headTarget(split, rows)
    // Bound the vocabulary projection by the measured GPU quota, like
    // other matrix work. Fixed row bands force needless fences on fast GPUs.
    await this.gpu.runAsync({
      name: "head",
      work: Math.ceil(e.blocks / split) * 24,
      program: this.gpu.program(
        K.withRms(K.headRgba4({ zeros: !!e.zeros, prescaled: scaledRow0 !== undefined }), scaledRow0 !== undefined && this.sumsInOne(b.residual[0]) ? 1 : this.parts)
      ),
      inputs: {
        ...(scaledRow0 !== undefined && { uR: b.residual[0] }),
        uA: scaledRow0 !== undefined ? b.scaled[0] : b.lastNormed,
        uW: e.quants,
        uS: e.scales,
        ...(e.zeros && { uZ: e.zeros })
      },
      uniforms: {
        uKb: { int: e.blocks },
        uH: { int: e.rowsPerColumn },
        uRowTexels: { int: e.rowTexels },
        uG: { int: e.groups },
        uOutW: { int: OUT_WIDTH },
        uSliceRows: { int: sliceRows },
        uKbSlice: { int: Math.ceil(e.blocks / split) },
        uRows: { int: rows },
        ...(scaledRow0 !== undefined && {
          uRowA0: { int: scaledRow0 },
          uAOff: { int: this.config.hiddenSize / 4 + K.RMS_PARTS },
          uROff: { int: this.config.hiddenSize / 4 },
          uParts: { int: this.parts },
          uTexels: { int: this.config.hiddenSize / 4 },
          uEps: this.config.normEps
        })
      },
      output: target,
      rect: { x: 0, y: 0, width: OUT_WIDTH, height }
    })
    return this.headSlices(target, split, rows, sum)
  }

  /** Logits for one row of the last batch's final residual. */
  async logits(row: number): Promise<Float32Array> {
    const c = this.config
    const b = this.buffers
    await this.head(row)
    const all = await this.gpu.read(b.logits, {
      x: 0,
      y: 0,
      width: OUT_WIDTH,
      height: this.logitRows
    })
    return all.subarray(0, c.vocabSize)
  }

  /** The images' rows among cache positions [pos0, pos0 + rows): batch row, image, its first row there, count. */
  private imageRuns(pos0: number, rows: number) {
    const runs: Array<{ row: number; image: EngineImage; from: number; count: number }> = []
    for (const image of this.images) {
      const start = Math.max(pos0, image.start)
      const end = Math.min(pos0 + rows, image.start + image.height * image.width)
      if (start < end) runs.push({ row: start - pos0, image, from: start - image.start, count: end - start })
    }
    return runs
  }

  /** Each row's t, h, w rotary positions (see ropeAt), into ropePositions. */
  private writeRopePositions(pos0: number, rows: number) {
    if (!this.ropePositions || this.ropePositions.height < rows) {
      if (this.ropePositions) this.gpu.delete(this.ropePositions)
      this.ropePositions = this.gpu.texture(GL.RGBA32F, 1, this.capacity)
    }
    const positions = new Float32Array(4 * rows)
    for (let m = 0; m < rows; m++) positions.set(this.ropeAt(pos0 + m), 4 * m)
    this.gpu.write(this.ropePositions, { x: 0, y: 0, width: 1, height: rows }, { format: GL.RGBA, type: GL.FLOAT, pixels: positions })
    return this.ropePositions
  }

  private async forwardBatch(
    batch: Array<number>,
    probe: LayerProbe | undefined,
    idsOnGpu = false,
    /** Gated DeltaNet layers copy their state after this many rows into `saving`. */
    saveAt?: number,
    fullRows = false
  ) {
    const c = this.config
    const b = this.buffers
    const rows = batch.length
    b.residual = rows <= SPLIT_ROWS ? b.residualSmall : b.residualLarge
    this.gpu.setPhase(rows <= SPLIT_ROWS ? "decode" : "prompt")
    if (rows <= SPLIT_ROWS) this.gpu.decodeWork()
    const decodeLayers =
      TUNING.decodePaceLayers < 0
        ? this.gpu.decodeLayers
        : TUNING.decodePaceLayers
    const pos0 = this.position
    this.gpu.hints = TUNING.renderHints
    this.gpu.cacheState = TUNING.cacheState > 0
    this.gpu.coopThreads = Math.max(0, this.tuned.lanes ?? TUNING.lanes)
    this.gpu.presentationFrames = TUNING.presentationFrames
    this.gpu.configurePacing(
      this.tuned.submissionReads ?? TUNING.submissionReads,
      TUNING.submissionMs
    )
    this.gpu.pollsBeforeSleep = TUNING.fencePolls
    this.gpu.promptSpinMs = TUNING.promptFenceSpinMs
    this.gpu.timedPolls = TUNING.timedFencePolls > 0
    this.gpu.slowFencePromptFps = TUNING.slowFencePromptFps
    this.gpu.presentationFrames = TUNING.presentationFrames
    // (Writing: room past the batch for its queries too, see queryInCache.)
    await this.reserve(
      rows <= SPLIT_ROWS && TUNING.qInCache > 0
        ? Math.min(this.contextLength, pos0 + rows * (1 + this.querySlots()))
        : pos0 + rows
    )
    if (!idsOnGpu) {
      this.gpu.write(
        b.ids,
        { x: 0, y: 0, width: 1, height: rows },
        {
          format: GL.RED_INTEGER,
          type: GL.UNSIGNED_INT,
          pixels: new Uint32Array(batch)
        }
      )
    }
    // Image rows in this batch: their features replace the embedding's
    // rows, and the head passes read each row's 3D position (see ropeAt).
    const imageRuns = this.imageRuns(pos0, rows)
    if (imageRuns.some(({ image }) => !image.features)) throw new Error("An image's rows are read again without its features.")
    this.batchRope =
      imageRuns.length && this.imagePositions === "grid" ? { positions: this.writeRopePositions(pos0, rows) } : { pos0: this.ropeAt(pos0)[0] }
    const e = this.embedding
    // A few rows: the split matmuls apply the norm to the residual as they
    // read it (no norm pass; see TUNING.foldNorm).
    const prescaled = TUNING.foldNorm === 2 && this.mrt.scaled
    // (foldEach: 8-bit matrices fold the prescaled input too, and each of a
    // layer's two norms folds where its own matmuls can.)
    const each = prescaled && TUNING.foldEach > 0
    const folds = (m?: GpuMatrix) =>
      !!m && (m.rgba4 || (m.bits === 4 && m.bytes) || !!m.stored || (each && m.bits === 8))
    const canFold = rows <= SPLIT_ROWS && TUNING.foldNorm > 0 && !probe
    const opFolds = (l: GpuLayer) =>
      folds(l.conv?.inProj ?? l.attention?.qkvg ?? l.linear?.inProj) &&
      (!l.linear || folds(l.linear.ba))
    /** Whether layer j's operator matmuls fold its norm. */
    const foldAt = (j: number) => {
      const l = this.layers[j]
      return canFold && opFolds(l) && (each || folds(l.gateUp))
    }
    /** Whether layer j's gate|up matmul folds its norm. */
    const foldFfn = (j: number) => {
      const l = this.layers[j]
      return canFold && folds(l.gateUp) && (each || opFolds(l))
    }
    // The embedding draws layer 0's sums of squares and scaled input too.
    const embedFolded = prescaled && e.rgba4 && foldAt(0) && !imageRuns.length
    // The last add draws the final norm's scaled input for the head.
    this.headScaled =
      prescaled &&
      (e.rgba4 || this.q4WordsHead(e) || (e.bits === 6 && !e.bytes && TUNING.q6FoldNorm > 0)) &&
      foldFfn(this.layers.length - 1)
    if (embedFolded) {
      const texels = c.hiddenSize / 4
      const embedInputs = {
        uIds: b.ids,
        uW: e.quants,
        uS: e.scales,
        ...(e.zeros && { uZ: e.zeros }),
        uNext: this.layers[0].operatorNorm
      }
      const shape = {
        uKb: { int: e.blocks },
        uH: { int: e.rowsPerColumn },
        uRowTexels: { int: e.rowTexels },
        uTexels: { int: texels },
        uChunk: { int: Math.ceil(texels / this.parts) }
      }
      const output = [b.residual[0], b.scaled[0]]
      const parts: Array<["values" | "rms" | "scaled", number, number]> = [
        ["values", 0, texels],
        ["rms", texels, this.parts],
        ["scaled", texels + K.RMS_PARTS, texels / 2]
      ]
      for (const [part, x, width] of parts) {
        this.gpu.run({
          program: this.gpu.program(K.embedFold({ zeros: !!e.zeros, part })),
          inputs: embedInputs,
          uniforms: { ...shape, uBase: { int: x } },
          output,
          rect: { x, y: 0, width, height: rows }
        })
      }
    } else
      this.gpu.run({
        program: this.gpu.program(
          this.weightSource(
            e,
            e.bits === 6
              ? e.q6Expanded
                ? e.q6ExpandedWords ? K.EMBED_Q6_EXPANDED_WORDS : K.EMBED_Q6_EXPANDED
                : e.bytes
                  ? K.EMBED_Q6_K
                  : e.q6Ordered
                    ? K.EMBED_Q6_K_WORDS_ORDERED
                    : K.EMBED_Q6_K_WORDS
              : K.embed({
                  bits: e.bits,
                  zeros: !!e.zeros,
                  bytes: e.bytes,
                  rgba4: e.rgba4
                })
          )
        ),
        inputs: {
          uIds: b.ids,
          ...this.weightInputs(e),
          uS: e.scales,
          ...(e.zeros && { uZ: e.zeros })
        },
        uniforms: {
          uKb: { int: e.blocks },
          uH: { int: e.rowsPerColumn },
          uRowTexels: { int: e.rowTexels }
        },
        output: b.residual[0],
        rect: { x: 0, y: 0, width: c.hiddenSize / 4, height: rows }
      })
    const look = async (
      layer: number,
      name: string,
      tex: Texture,
      dims: number,
      h2: boolean | "pair" = false,
      /** A pair tensor left as this many slices of the last split matmul. */
      split = 0
    ) => {
      if (!probe) return
      if (split) tex = this.partials()
      await probe(layer, name, async () => {
        let src = tex
        if (h2) {
          this.gpu.run({
            program: this.gpu.program(
              split
                ? this.fromSplit(K.PAIR_TO_F32, ["uX"], split)
                : h2 === "pair"
                  ? this.readPair(K.PAIR_TO_F32, "uX", tex)
                  : K.H2_TO_F32
            ),
            uniforms: split ? {} : this.pairIn(tex),
            inputs: {
              uX: tex,
              ...(h2 === "pair" && !split && this.quadInputs("uX", tex))
            },
            output: this.debugTexture(),
            rect: { x: 0, y: 0, width: dims / 4, height: rows }
          })
          src = this.debugTexture()
        }
        const data = await this.gpu.read(src, {
          x: 0,
          y: 0,
          width: dims / 4,
          height: rows
        })
        return { data, rows, dims, offset: pos0 }
      })
    }
    for (const { row, image, from, count } of imageRuns) {
      if (!image.features) continue
      this.gpu.write(
        b.residual[0],
        { x: 0, y: row, width: c.hiddenSize / 4, height: count },
        {
          format: GL.RGBA,
          type: GL.FLOAT,
          pixels: image.features.subarray(from * c.hiddenSize, (from + count) * c.hiddenSize)
        }
      )
    }
    await look(-1, "embedding", b.residual[0], c.hiddenSize)

    const paced = rows > SPLIT_ROWS
    // Only the last row's logits are asked for (see lastRowOnly).
    const lastRowOnly = TUNING.lastRowOnly > 0 && rows > SPLIT_ROWS && !probe && !fullRows
    this.lastRowOnly = lastRowOnly
    for (let i = 0; i < this.layers.length; i++) {
      const layer = this.layers[i]
      const groupEnd = i > 0 && decodeLayers > 0 &&
        this.gpu.readsPerSlice > 0 && i % decodeLayers === 0
      if (paced) await this.charge(0)
      else {
        if (groupEnd) await this.gpu.pace(rows > 1 ? 0 : 2, true, 0, decodeLayers)
        // Grouped decode's fence flushes its own layers.
        if ((!this.gpu.groupsDecode || decodeLayers <= 0 || this.gpu.readsPerSlice <= 0) &&
          TUNING.flushLayers > 0 && i > 0 && i % TUNING.flushLayers === 0)
          this.gpu.flush()
      }
      const [h, next] = b.residual
      const fold = foldAt(i)
      const ffnFold = foldFfn(i)
      // (Layer 0 folds only if the embedding drew its sums and input.)
      const opIn: MatIn =
        fold && (i > 0 || (embedFolded && prescaled))
          ? { a: h, norm: prescaled ? "prescaled" : layer.operatorNorm }
          : { a: b.normed }
      if (!opIn.norm) {
        const mats = ([
          [layer.conv?.inProj, b.bcx],
          [layer.attention?.qkvg, b.qkvg],
          [layer.linear?.inProj, b.lin?.inProj],
          [layer.linear?.ba, b.lin?.ba]
        ] as const).filter((use): use is readonly [GpuMatrix, Texture] => !!use[0] && !!use[1])
        opIn.a = this.normInput(
          h,
          layer.operatorNorm,
          rows,
          mats,
          i > 0,
          !probe
        )
        await look(i, "operator_norm", b.normed, c.hiddenSize, true)
      }
      let opSplit = 0
      if (layer.linear) {
        opSplit = await this.linearAttend(
          layer.linear,
          rows,
          opIn,
          paced,
          saveAt,
          rows <= SPLIT_ROWS && !probe
        )
        const lin = this.linDims()
        if (lin && b.lin) {
          await look(i, "lin_qkv", b.lin.qkv, lin.C)
          await look(i, "lin_o", b.lin.o, lin.V)
          await look(i, "lin_out", b.lin.out, lin.V, true)
        }
      } else if (layer.conv) {
        opSplit = await this.convolve(layer.conv, rows, opIn)
        await look(i, "in_proj", b.bcx, 3 * c.hiddenSize, "pair")
        await look(i, "conv", b.convOut, c.hiddenSize, true)
      } else if (layer.attention) {
        opSplit = await this.attend(
          layer.attention,
          layer.type,
          rows,
          pos0,
          opIn,
          paced,
          !probe,
          (split) => look(i, "qkvg", b.qkvg, this.qkvgDims(), "pair", split)
        )
        if (this.prefixSaving && layer.attention.ring) {
          const at = pos0 + this.prefixSaving.row
          const window = this.gpu.texture(GL.RGBA32UI, layer.attention.kv.width, c.slidingWindow)
          this.prefixSaving.state.windows.push(window)
          this.copyPositions(layer.attention.kv, layer.attention.ring, window, c.slidingWindow, Math.max(0, at - c.slidingWindow), at)
        }
        await look(i, "q", b.q, c.numHeads * c.headDim)
        await look(i, "attn_out", b.attnOut, c.numHeads * c.headDim, true)
        // The cache's keys then values for positions 0.. (one block, no
        // ring), as f32: for checking attention against the CPU.
        const kvDims = c.numKvHeads * c.headDim
        const keys = pos0 + rows
        const kvCache = layer.attention.kv
        if (probe && !layer.attention.ring && keys <= 2 ** this.kvShift) {
          await probe(i, "kv", async () => {
            const width = (2 * kvDims) / 4
            if (this.kvDebug?.width !== width || this.kvDebug.height < keys) {
              if (this.kvDebug) this.gpu.delete(this.kvDebug)
              this.kvDebug = this.f32(2 * kvDims, Math.max(keys, this.capacity))
            }
            this.gpu.run({
              program: this.gpu.program(K.H2_TO_F32),
              inputs: { uX: kvCache },
              output: this.kvDebug,
              rect: { x: 0, y: 0, width, height: keys }
            })
            const data = await this.gpu.read(this.kvDebug, {
              x: 0,
              y: 0,
              width,
              height: keys
            })
            return { data, rows: keys, dims: 2 * kvDims, offset: 0 }
          })
        }
      }
      await look(i, "op_out", b.opOut, c.hiddenSize, "pair", opSplit)
      this.addNormed(
        h,
        opSplit ? this.partials() : b.opOut,
        layer.postOperatorNorm,
        next,
        rows,
        opSplit,
        ffnFold && prescaled ? layer.ffnNorm : undefined
      )
      await look(i, "after_op", next, c.hiddenSize)
      // A prompt batch's last layer: only the last row goes on (its hidden
      // row moves to row 0 of h, the feed-forward block runs on it into
      // row 0 of next, and back to h for the head; see lastRowOnly).
      const tail = lastRowOnly && i === this.layers.length - 1
      const fr = tail ? 1 : rows
      const base = tail ? h : next
      const dest = tail ? next : h
      if (tail) this.copyRow(next, rows - 1, h)
      const ffnInput = !ffnFold
        ? this.normInput(base, layer.ffnNorm, fr, [[layer.gateUp, b.gateUp]], true, !probe)
        : b.normed
      let downInput = b.act
      if (fr <= SPLIT_ROWS) {
        // A few rows: the split matmul's slices are summed and through
        // SwiGLU in one pass.
        const slices = ffnFold
          ? this.splitPartials(
              next,
              layer.gateUp,
              rows,
              prescaled ? "prescaled" : layer.ffnNorm
            )
          : this.splitPartials(ffnInput, layer.gateUp, fr)
        this.gpu.run({
          name: "swiglu",
          program: this.gpu.program(K.withSplit(K.REDUCE_SWIGLU, slices)),
          inputs: { uP: this.partials() },
          uniforms: {
            uSplitIn: { int: slices },
            uHalf: { int: c.intermediateSize / 4 }
          },
          output: b.act,
          rect: { x: 0, y: 0, width: c.intermediateSize / 8, height: fr }
        })
      } else {
        await this.matmul(ffnInput, layer.gateUp, b.gateUp, rows)
        downInput = this.floatInput(fr, [[layer.down, b.ffnOut]], !probe) ?? b.act
        const floating = downInput.internal === GL.RGBA16F
        this.gpu.run({
          name: "swiglu",
          program: this.gpu.program(
            this.readPair(floating ? K.SWIGLU_F16 : K.SWIGLU, "uGU", b.gateUp)
          ),
          inputs: { uGU: b.gateUp, ...this.quadInputs("uGU", b.gateUp) },
          uniforms: {
            uHalf: { int: c.intermediateSize / 4 },
            ...this.pairIn(b.gateUp)
          },
          output: downInput,
          rect: {
            x: 0,
            y: 0,
            width: c.intermediateSize / (floating ? 4 : 8),
            height: rows
          }
        })
      }
      const downSplit = await this.project(downInput, layer.down, b.ffnOut, fr)
      await look(i, "ffn_out", b.ffnOut, c.hiddenSize, "pair", downSplit)
      this.addNormed(
        base,
        downSplit ? this.partials() : b.ffnOut,
        layer.postFfnNorm,
        dest,
        fr,
        downSplit,
        !prescaled
          ? undefined
          : i + 1 < this.layers.length
            ? foldAt(i + 1)
              ? this.layers[i + 1].operatorNorm
              : undefined
            : this.headScaled
              ? this.finalNorm
              : undefined
      )
      if (tail) this.copyRow(next, 0, h)
      await look(i, "l_out", h, c.hiddenSize)
      if (paced) this.unpaced += rows * this.layerWork(i)
    }
    this.position += rows
    return lastRowOnly ? 0 : rows - 1
  }

  /** Work queued on the GPU since the last pace (multiply-adds, roughly). */
  private unpaced = 0
  private work: Array<number> | null = null

  /** Layer i's matmul multiply-adds per token. */
  private layerWork(i: number) {
    this.work ??= this.layers.map((l) =>
      [
        l.conv?.inProj,
        l.conv?.outProj,
        l.attention?.qkvg,
        l.attention?.o,
        l.linear?.inProj,
        l.linear?.ba,
        l.linear?.outProj,
        l.gateUp,
        l.down
      ]
        .filter((m): m is GpuMatrix => !!m)
        .reduce((sum, m) => sum + m.N * m.K, 0)
    )
    return this.work[i]
  }

  /**
   * Counts `work` as queued, and paces (see Gpu.pace) once paceLayers
   * full-batch layers' worth is: long attention charges its own work, so
   * a single long layer is split too.
   */
  private async charge(work: number) {
    this.unpaced += work
    if (TUNING.paceLayers <= 0) return
    const mean =
      this.layers.reduce((sum, _, i) => sum + this.layerWork(i), 0) /
      this.layers.length
    if (this.unpaced < TUNING.paceLayers * this.batchSize * mean) return
    this.unpaced = 0
    await this.gpu.pace(TUNING.paceAhead)
  }
}
