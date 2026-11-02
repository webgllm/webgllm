/**
 * LFM3, LFM2 and Qwen3.5 on WebGL 2: uploads the weights a loader hands over, and runs the
 * forward pass a batch of tokens at a time, keeping the attention cache and
 * the convolutions' state on the GPU between calls.
 */
import { attentionError } from "./attention-check.js"
import { queriesFit, slotRuns } from "./cache-slots.js"
import { GL, Gpu, GpuError, type Program } from "./gpu/gl.js"
import type { Texture } from "./gpu/gl.js"
import * as K from "./gpu/kernels.js"
import { panelLayout } from "./panels.js"
import { quantsFromRaw, rawRgba4 } from "./repack.js"
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
 * Q4_0 block rows (16 bytes: weight i in byte i's low nibble, weight
 * 16 + i in its high nibble) as RGBA4 texels of 4 consecutive weights
 * (UNSIGNED_SHORT_4_4_4_4: the first weight in the top nibble): word w of a
 * block row gives texels w (low nibbles) and 4 + w (high nibbles).
 */
function toRgba4(words: Uint32Array) {
  const out = new Uint16Array(words.length * 2)
  // The first weight in the low nibble (kernels read .wzyx; see kernels.ts).
  const four = (x: number) =>
    (x & 15) |
    (((x >>> 8) & 15) << 4) |
    (((x >>> 16) & 15) << 8) |
    (((x >>> 24) & 15) << 12)
  for (let row = 0; row < words.length; row += 4) {
    const o = row * 2
    for (let w = 0; w < 4; w++) {
      const x = words[row + w]
      out[o + w] = four(x & 0x0f0f0f0f)
      out[o + 4 + w] = four((x >>> 4) & 0x0f0f0f0f)
    }
  }
  return out
}

/**
 * Knobs for A/B timing on the bench page (bench/check.html?ab=...); the
 * defaults are what the engine uses.
 */
export const TUNING = {
  /** Prompt matmuls with at most this many row groups split K in two. */
  sliceGroups: 1024,
  /** Few-token matmuls split K until they have about this many fragments... */
  splitFragments: 4096,
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
  /**
   * 4-bit layer weights in RGBA4 textures (1) or byte textures (0); -1:
   * whichever this GPU writes faster (timed at load; see chooseLayout).
   */
  layerRgba4: -1,
  /** 1: the pick's argmax from the chunk maxima's positions, its loops with constant bounds. */
  fastPick: 1,
  /** Render-area hints for every pass (see Gpu.hints): 1 scissors to the pass's rect. */
  renderHints: 1,
  /**
   * 1: the f16 matmul's dots in mediump (faster where that's half
   * precision, less exact); options.matmulPrecision "mediump" turns it on.
   */
  f16Half: 0,
  /**
   * How the f16 matmul reads (see matmulF16's `packed`), as bits: 1 the
   * activations straight from their h2 tensor, 8 halves a fetch, with no
   * RGBA16F copy (one pass and its scratch fewer); 2 the dequantized
   * weights as h2 too (the same bytes in half the fetches); 0: both as
   * RGBA16F, 4 values a fetch. Engine.tune times 0 against 3 on a GPU
   * whose prompts take the f16 matmul, and keeps it where it's faster.
   */
  f16Pack: 3,
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
   * 0, the whole matrix, as they always had.
   */
  f16ScratchBytesMiB: 0,
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
   * The worker's speculative decoding (see worker.ts): drafts follow an
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

/** See Engine.tuned. */
export type Tuned = {
  shortSplit: number | null
  f16Matmul?: number | null
  f16Pack?: number | null
  /** Writing's attention in panels (TUNING.flashDecode) or not. */
  flashDecode?: number | null
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
}

/**
 * A matmul's input: h2 activations, or the f32 residual and the norm to
 * apply (its weights, or "prescaled": already applied by the add).
 */
type MatIn = { a: Texture; norm?: Texture | "prescaled" }

export type EngineOptions = {
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

  readonly gpu: Gpu
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
  private buffers!: ReturnType<Engine["allocate"]>

  constructor(
    canvas: OffscreenCanvas | HTMLCanvasElement,
    options: EngineOptions = {}
  ) {
    this.gpu = new Gpu(canvas)
    this.contextCap = options.contextLength ?? Infinity
    this.noMrt = new Set(options.noMrt ?? [])
    this.chunkedRecurrence = options.chunkedRecurrence ?? true
    this.kvBlockRows = options.kvBlockRows ?? Infinity
    this.baseBatch = options.batchSize ?? 128
    this.batchSize = this.baseBatch
    this.capacity = this.batchSize + this.batchSize / 8
    this.scaleBatch = options.scaleBatch ?? true
    this.precision = options.matmulPrecision ?? "highp"
    this.gpu.hints = TUNING.renderHints
    this.bytes = (options.weightTexels ?? "bytes") === "bytes"
    this.ringCache = options.ringCache ?? true
    this.quadMatmuls = options.quadMatmuls ?? this.baseBatch >= TUNING.quadMinBatch
    for (const name of options.skip ?? []) this.gpu.skip.add(name)
  }

  /** Uploads the model, piece by piece, as the loader yields it. */
  async load(
    pieces: AsyncIterable<WeightPiece>,
    onProgress?: (piece: WeightPiece) => void
  ) {
    let mark = performance.now()
    this.loadMs = { reading: 0, uploading: 0, convert: 0 }
    for await (const piece of pieces) {
      const got = performance.now()
      this.loadMs.reading += got - mark
      if (piece.kind === "config") this.configure(piece.config)
      else if (piece.kind === "embedding") {
        // Whole, or in chunks of rows: allocated with the first.
        const { value, row0 = 0, totalRows = value.N } = piece
        if (row0 === 0)
          this.embedding = this.allocateMatrix(
            totalRows,
            value.K,
            value.bits,
            !!value.zeros,
            value.bits === 4
          )
        this.writeMatrix(this.embedding, value, row0)
      } else if (piece.kind === "finalNorm")
        this.finalNorm = this.vector(piece.value)
      else {
        // The weight layout this GPU writes fastest, once, before the
        // first layer's textures (see chooseLayout).
        if (!this.layoutChoice) await this.chooseLayout(piece.value.gateUp)
        this.layers[piece.layer] = this.layer(piece.value)
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
    if (this.quadMatmuls && this.usesF16()) this.f16Scratch()
    await this.gpu.finish()
  }

  private configure(config: ModelConfig) {
    this.config = config
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
    // Each tensor has its own texture: sharing one between tensors that are
    // never alive at once saves ~25 MB at 512 rows, but a pass that writes a
    // texture the pass before read must wait for it, and prompts read ~12%
    // slower. The narrow matmuls' outputs are K-split, two slices tall.
    return {
      ids: this.gpu.texture(GL.R32UI, 1, M),
      // The hidden state, and its rows' partial sums of squares in the last
      // RMS_PARTS columns (drawn with the add that writes it).
      residual: [0, 1].map(() =>
        // (Room for the scaled texture's columns: written alongside.)
        this.gpu.texture(GL.RGBA32F, D / 4 + K.RMS_PARTS + D / 8, M)
      ) as [Texture, Texture],
      /**
       * For a few rows, h2(residual · the next norm's weights) from column
       * D/4 + RMS_PARTS, drawn with the add (its second output; see
       * addNormed): one per residual texture.
       */
      scaled: [0, 1].map(() =>
        // (Same size as the residual: a framebuffer's textures must match.)
        this.gpu.texture(GL.RGBA32UI, D / 4 + K.RMS_PARTS + D / 8, M)
      ) as [Texture, Texture],
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
      scores: Array.from({ length: this.headGroup() }, () =>
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
      /** Per token: 0, 1 seen (repetition penalty), 2 never drawn. Same rows as the logits. */
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
  private scoreRows() {
    if (TUNING.scoreRows > 0) return TUNING.scoreRows
    return Math.min(SCORE_ROWS_LARGE, Math.max(SCORE_ROWS, this.baseBatch / 4))
  }

  /**
   * Makes room in the full-attention caches for tokens up to `end`: twice
   * the positions (or as many as asked), copied into a bigger texture of the
   * same layout. Sliding layers' rings never grow.
   */
  private reserve(end: number) {
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
      }
      if (this.gpu.outOfMemory()) throw new GpuError("Out of memory.")
    } catch (error) {
      // Some layers may hold a cache that didn't copy: start over.
      if (!this.gpu.isLost) this.reset()
      throw new GpuError(
        `The GPU couldn't grow the cache to ${positions} tokens (${error instanceof Error ? error.message : String(error)}). Start a new session to keep going.`
      )
    }
    this.kvPositions = positions
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
    const height = Math.max(rows, this.scoreRows() * ATTEND_CHUNKS)
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
        .join("\n") + Engine.toString()
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
    options: { keepCache?: boolean } = {}
  ): Promise<Tuned | null> {
    const before = { ...this.tuned }
    this.tuned = { shortSplit: null, f16Matmul: null, f16Pack: null, flashDecode: null }
    const b = this.buffers
    // A layer's prompt matmuls as the forward runs them: a conv block's,
    // an attention block's (the first of each kind), one feed-forward's.
    const conv = this.layers.find((l) => l.conv)?.conv
    const attention = this.layers.find((l) => l.attention)?.attention
    const ffn = this.layers[0]
    const run = (rows: number) => {
      if (conv) {
        this.matmul(b.normed, conv.inProj, b.bcx, rows)
        this.matmul(b.convOut, conv.outProj, b.opOut, rows)
      }
      if (attention) {
        this.matmul(b.normed, attention.qkvg, b.qkvg, rows)
        this.matmul(b.attnOut, attention.o, b.opOut, rows)
      }
      this.matmul(b.normed, ffn.gateUp, b.gateUp, rows)
      this.matmul(b.act, ffn.down, b.ffnOut, rows)
    }
    // A full prompt batch through the first TUNING.tuneLayers layers, every
    // pass as a prompt runs them (synthetic tokens, the cache reset
    // before each): what the f16 matmul's choices change isn't only the
    // matmuls (see the f16 choices below).
    const tokens = Array.from({ length: Math.min(this.batchSize, this.capacity) }, (_, i) => 1000 + (i % 5000))
    let forwards = false
    const forward = async () => {
      forwards = true
      const all = this.layers
      if (TUNING.tuneLayers > 0) this.layers = all.slice(0, TUNING.tuneLayers)
      try {
        this.reset()
        await this.forwardHidden(tokens)
      } finally {
        this.layers = all
      }
    }
    class Stopped extends Error {}
    const choose = async (
      key: keyof Tuned,
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
      const reps = Math.max(1, Math.min(16, Math.ceil(25 / (performance.now() - t0))))
      // An idle GPU clocks down: busy it ~100 ms first, so the clock has
      // come up before anything is timed.
      for (const start = performance.now(); performance.now() - start < 100; ) {
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
      const noise = Math.max(...rounds.map((r) => (Math.max(...r) - Math.min(...r)) / Math.min(...r)))
      const best = times.indexOf(Math.min(...times))
      this.tuneReport[`${key}Gain`] = Math.round((1 - times[1] / times[0]) * 1000) / 10
      this.tuneReport[`${key}Noise`] = Math.round(noise * 1000) / 10
      // candidates[0] is TUNING's value.
      return best > 0 && times[best] < (1 - Math.max(0.04, 2 * noise)) * times[0]
        ? candidates[best]
        : null
    }
    let picked: Tuned
    try {
      const short = Math.min(40, this.capacity)
      const shortSplit = await choose(
        "shortSplit",
        [TUNING.quadMaxSplit, TUNING.quadMaxSplit === 2 ? 4 : 2],
        () => run(short)
      )
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
      const quad = this.quadMatmuls && TUNING.quad > 0
      const free = !options.keepCache
      let f16Matmul: number | null | undefined = null
      let f16Pack: number | null | undefined = null
      if (quad && TUNING.f16Matmul > 0 && this.f16Mats().some((m) => m.rgba4) && this.f16Sized())
        f16Matmul = free
          ? await choose("f16Matmul", [TUNING.f16Matmul, TUNING.f16Matmul > 1 ? 1 : 2], forward)
          : undefined
      this.tuned.f16Matmul = f16Matmul ?? null
      if (quad && this.usesF16() && this.f16Scratch())
        f16Pack = free
          ? await choose("f16Pack", [TUNING.f16Pack & 3, (TUNING.f16Pack & 3) === 3 ? 0 : 3], forward)
          : undefined
      this.tuned.f16Pack = f16Pack ?? null
      if (forwards) this.reset()
      // Writing's attention from a long cache (a synthetic one: 4,096
      // keys in a scratch cache of the first full-attention layer's
      // shape), in panels or the old way: Apple's phone and desktop GPUs
      // disagreed on panel layouts, and other GPUs are untimed. (The
      // scratch cache leaves the model's own alone.)
      const full = this.layers.find((l) => l.attention && l.type !== "sliding_attention")
      const keys = Math.min(4096, this.contextLength)
      let flashDecode: number | null = null
      if (full?.attention && this.mrt.chunks && TUNING.attendChunkTexels > 0 && keys > 4 * TUNING.flashPast) {
        const kv = this.kvTexture(keys)
        const layer = { ...full.attention, kv }
        try {
          flashDecode = await choose(
            "flashDecode",
            [TUNING.flashDecode, TUNING.flashDecode > 0 ? 0 : 1],
            () => void this.attention(layer, full.type, 1, keys - 1, 0, this.buffers.qkvg)
          )
        } finally {
          this.gpu.delete(kv)
        }
      }
      picked = { shortSplit, f16Matmul, f16Pack, flashDecode }
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
  loadMs = { reading: 0, uploading: 0, convert: 0 }
  /** 4-bit layer weights in RGBA4 (else byte texels): see chooseLayout. */
  private rgba4Layers = true
  /** The layout chooseLayout picked and its timings (ms per writing matmul). */
  layoutChoice: {
    layout: "rgba4" | "bytes"
    rgba4Ms: number
    bytesMs: number
    /** How long choosing took. */
    ms?: number
  } | null = null

  /**
   * Picks how 4-bit layer weights are stored by timing both layouts on this
   * GPU: RGBA4 has the texture unit split the nibbles (free where weight
   * fetches are cheap, as on Apple GPUs), byte texels take half the
   * fetches and split them in the shader (faster where fetches are the
   * limit, as on PowerVR). A random matrix of `m`'s shape in each, the
   * writing matmul on each in turn, best of a few rounds.
   */
  private async chooseLayout(m: QuantMatrix) {
    if (m.bits !== 4 || TUNING.layerRgba4 >= 0 || this.preferLayout) {
      this.rgba4Layers =
        TUNING.layerRgba4 >= 0
          ? TUNING.layerRgba4 !== 0
          : this.preferLayout !== "bytes"
      this.layoutChoice = {
        layout: this.rgba4Layers ? "rgba4" : "bytes",
        rgba4Ms: NaN,
        bytesMs: NaN
      }
      return
    }
    const started = performance.now()
    let seed = 1
    const quants = new Uint32Array(m.N * (m.K / 32) * 4)
    for (let i = 0; i < quants.length; i++) {
      seed = (seed * 1664525 + 1013904223) >>> 0
      quants[i] = seed
    }
    const fake: QuantMatrix = {
      N: m.N,
      K: m.K,
      bits: 4,
      quants,
      // Small scales (half 2^-7) and zero points 0 .. 15.
      scales: new Uint16Array(m.scales.length).fill(0x2000),
      zeros: m.zeros
        ? Uint8Array.from({ length: m.zeros.length }, (_, i) => quants[i] & 15)
        : null
    }
    const made = [true, false].map((rgba4) => {
      const gm = this.allocateMatrix(m.N, m.K, 4, !!fake.zeros, rgba4, true)
      this.writeMatrix(gm, fake, 0)
      return gm
    })
    // Each matmul's output gets a texel read, so no GPU skips one.
    const scratch = this.gpu.texture(GL.RGBA32F, 1, 1)
    const step = (gm: GpuMatrix) => {
      this.splitPartials(this.buffers.normed, gm, 1)
      this.gpu.run({
        program: this.gpu.program(K.COPY),
        inputs: { uX: this.buffers.partial },
        output: scratch,
        rect: { x: 0, y: 0, width: 1, height: 1 }
      })
    }
    for (const gm of made) step(gm)
    await this.gpu.finish()
    const best = [Infinity, Infinity]
    for (let round = 0; round < 3; round++) {
      for (const i of round % 2 ? [1, 0] : [0, 1]) {
        const t = performance.now()
        for (let n = 0; n < 16; n++) step(made[i])
        await this.gpu.finish()
        best[i] = Math.min(best[i], (performance.now() - t) / 16)
      }
    }
    for (const gm of made)
      for (const t of [gm.quants, gm.scales, gm.zeros]) if (t) this.gpu.delete(t)
    this.gpu.delete(scratch)
    // Bytes only if clearly faster (timing noise shouldn't flip it).
    this.rgba4Layers = !(best[1] < best[0] * 0.95)
    this.layoutChoice = {
      layout: this.rgba4Layers ? "rgba4" : "bytes",
      rgba4Ms: best[0],
      bytesMs: best[1],
      ms: performance.now() - started
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
    return gm
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
    layer = false
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
        : this.bytes && columns * blocks * 4 * byteTexels <= max
    const texelsPerBlockRow = bytes ? byteTexels : byteTexels / 4
    const rowTexels = blocks * 4 * texelsPerBlockRow
    const unorm = bytes && bits === 4 && TUNING.unormBytes > 0
    const packed =
      layer && bytes && bits === 4 && withZeros && TUNING.packZeros > 0
    const quantFormat = !bytes
      ? GL.RGBA32UI
      : bits === 8
        ? GL.RGBA8I
        : unorm
          ? GL.RGBA8
          : GL.RGBA8UI
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
      bytes,
      unorm,
      packed,
      rgba4: false,
      blocks,
      groups,
      rowsPerColumn,
      rowTexels,
      quants: this.gpu.texture(quantFormat, columns * rowTexels, rowsPerColumn),
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

  /** Writes `m`'s rows into rows [row0, row0 + m.N) of `gm` (row0 a multiple of 4). */
  private writeMatrix(gm: GpuMatrix, m: QuantMatrix, row0: number) {
    // Raw MatMulNBits rows go into RGBA4 as they are (16-byte runs moved
    // into place; see rawRgba4); other layouts get the rows repacked.
    if (m.raw && !gm.rgba4)
      m = { ...m, quants: quantsFromRaw(m.raw, m.N, m.K), raw: undefined }
    const { blocks, rowTexels, rowsPerColumn } = gm
    const g0 = row0 / 4
    const groups = m.N / 4
    // Words of one row group: its quant texels' bytes over 4.
    const words = m.quants.length / groups
    for (let g = g0; g < g0 + groups;) {
      const column = Math.floor(g / rowsPerColumn)
      const y = g - column * rowsPerColumn
      const run = Math.min(g0 + groups - g, rowsPerColumn - y)
      const local = g - g0
      const quants = m.quants.subarray(local * words, (local + run) * words)
      if (gm.rgba4) {
        const t = performance.now()
        const pixels = m.raw
          ? rawRgba4(m.raw, m.K, local * 4, run * 4)
          : toRgba4(quants)
        this.loadMs.convert += performance.now() - t
        this.gpu.write(
          gm.quants,
          { x: column * rowTexels, y, width: rowTexels, height: run },
          {
            format: GL.RGBA,
            type: GL.UNSIGNED_SHORT_4_4_4_4,
            pixels
          }
        )
      } else
      this.gpu.write(
        gm.quants,
        { x: column * rowTexels, y, width: rowTexels, height: run },
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
        const n = run * blocks * 4
        const at = local * blocks * 4
        const texels = new Uint32Array(n)
        for (let t = 0; t < n; t += 4) {
          const d = m.scales.subarray(at + t, at + t + 4)
          const z = m.zeros.subarray(at + t, at + t + 4)
          texels[t] = d[0] | (d[1] << 16)
          texels[t + 1] = d[2] | (d[3] << 16)
          texels[t + 2] = (HALF_INT[z[0]] | (HALF_INT[z[1]] << 16)) >>> 0
          texels[t + 3] = (HALF_INT[z[2]] | (HALF_INT[z[3]] << 16)) >>> 0
        }
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

  /** Frees the oldest snapshots but the anchor until at most `keep` are left. */
  private dropSnapshots(keep: number) {
    for (const [p, old] of this.snapshots) {
      if (this.snapshots.size <= keep) break
      if (old.anchor) continue
      this.freeSnapshot(old)
      this.snapshots.delete(p)
    }
  }

  /**
   * Saves what's needed to resume at the current position (see
   * `snapshots`), so a later prompt that shares the tokens so far reads only
   * what's new. Keeps the latest snapshotLimit, the oldest freed before a
   * new one is made (so no more than that are ever held), except the
   * anchor: the latest snapshot saved with `anchor` (where the next user
   * turn's prompt stops sharing this chat's: see readPrompt) stays until a
   * newer anchor replaces it. With `hiddenRow` (the row of the last batch
   * holding the last token), keeps that token's hidden state too, so
   * `resume` can draw the next token's logits without reading it again.
   */
  snapshot(hiddenRow?: number, anchor = false) {
    if (this.guessAt !== null)
      throw new Error("Snapshot while a guessed token is pending.")
    const at = this.position
    const had = this.snapshots.get(at)
    if (had) {
      if (hiddenRow !== undefined && !had.hidden) had.hidden = this.copyHidden(hiddenRow)
      if (anchor) this.keepSnapshot(at, had, true)
      return
    }
    this.dropSnapshots(this.snapshotLimit - 1)
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
  private keepSnapshot(at: number, saved: Snapshot, anchor: boolean) {
    this.snapshots.set(at, saved)
    if (anchor) {
      for (const s of this.snapshots.values()) s.anchor = false
      saved.anchor = true
    }
    // (An anchor kept past the limit goes once it's no longer the anchor.)
    this.dropSnapshots(this.snapshotLimit)
  }

  /** A copy of an f32 texture. */
  private copyOf(from: Texture) {
    const copy = this.gpu.texture(GL.RGBA32F, from.width, from.height)
    this.gpu.run({ program: this.gpu.program(K.COPY), inputs: { uX: from }, output: copy })
    return copy
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
    this.position = at
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

  /** Forgets the cache: the next forward starts a new sequence. */
  reset() {
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
    const kinds = new Set<string>()
    if (TUNING.warmSubset > 0) this.layers = all.filter((l, i) => {
      const kind = l.conv ? "conv" : `${l.type}:${l.attention?.ring ? "ring" : ""}`
      const keep = i === 0 || i === all.length - 1 || !kinds.has(kind)
      kinds.add(kind)
      return keep
    })
    try {
      if (options.shortOnly) await this.forwardHidden(new Array(SPLIT_ROWS + 1).fill(0))
      else await this.warmUpLayers()
    } finally {
      this.layers = all
      // (Per-layer work for pacing, counted over the subset.)
      this.work = null
      this.reset()
    }
  }

  private async warmUpLayers() {
    let seen = this.gpu.compiled.count
    const step = (name: string) => {
      this.warmCounts[name] = this.gpu.compiled.count - seen
      seen = this.gpu.compiled.count
    }
    const row = await this.forwardHidden(new Array(SPLIT_ROWS + 1).fill(0))
    step("batch9")
    await this.candidates(row, 1, 1, true)
    step("pick")
    this.dropGuess()
    await this.logits(row)
    step("logits")
    await this.argmax(row, 1, 1, [])
    step("argmax")
    // A batch past f16MinRows: the f16 matmul path full prompts take.
    if (TUNING.f16MinRows <= this.capacity) {
      await this.forwardHidden(new Array(Math.max(TUNING.f16MinRows, SPLIT_ROWS + 1)).fill(0))
      step("batchF16")
    }
    if (this.canDraft) {
      await this.forwardHidden([0, 0], { draft: true })
      step("draft")
      this.accept(1)
    }
    await this.logits(await this.forwardHidden([0]))
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
        await this.forwardHidden(new Array(12).fill(0))
        await this.logits(await this.forwardHidden([0]))
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
        return this.gpu.readNow(layer.state[0])
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
    const tokens = Array.from(
      { length: 40 },
      (_, i) => (i * 7919 + 13) % this.config.vocabSize
    )
    let worst = await attentionError(this, tokens)
    worst = Math.max(worst, await this.recurrenceError(tokens))
    if (TUNING.attendChunkTexels > 0) {
      const saved = TUNING.attendChunkTexels
      const savedDecode = TUNING.attendChunkTexelsDecode
      TUNING.attendChunkTexels = 1
      TUNING.attendChunkTexelsDecode = 1
      try {
        worst = Math.max(worst, await attentionError(this, tokens.slice(0, 24)))
      } finally {
        TUNING.attendChunkTexels = saved
        TUNING.attendChunkTexelsDecode = savedDecode
      }
      worst = Math.max(worst, await this.panelError(tokens))
    }
    // The probes' scratch isn't read again (~29 MB for Qwen3.5 2B at a
    // 512-token batch).
    for (const t of [this.debug, this.kvDebug]) if (t) this.gpu.delete(t)
    this.debug = this.kvDebug = null
    return worst
  }

  /**
   * Writing's attention in panels (attendPanel and its combine) against
   * the one-pass kernel (which attentionError checks against the CPU),
   * each attention layer's both ways from the same queries and cache: the
   * last of `tokens` after the rest, in chunks of 4 keys (enough to take
   * both of the combine's loops). The largest relative error, as
   * attentionError's. Leaves the cache empty.
   */
  private async panelError(tokens: Array<number>) {
    if (!this.mrt.chunks) return 0
    const saved = [TUNING.flashPast, TUNING.flashChunkTexels, TUNING.flashDecode]
    const tuned = this.tuned.flashDecode
    this.tuned.flashDecode = null
    try {
      // (Whether or not this GPU writes that way: the tuner may pick it.)
      TUNING.flashPast = 0
      TUNING.flashChunkTexels = 1
      TUNING.flashDecode = 1
      this.reset()
      await this.forwardHidden(tokens.slice(0, -1))
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

  /**
   * a · wᵀ for `rows` tokens: a few rows leave a split matmul's partial
   * sums in the partial buffer and return the number of slices (consumers
   * read them through pf, no reduce pass); more rows write the pair tensor
   * `out` and return 0.
   */
  private project(
    a: Texture,
    w: GpuMatrix,
    out: Texture,
    rows: number,
    norm?: Texture | "prescaled"
  ) {
    if (rows <= SPLIT_ROWS) return this.splitPartials(a, w, rows, norm)
    if (norm) throw new Error("Only split matmuls fold the norm in.")
    this.matmul(a, w, out, rows)
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
   * RGBA16F, in two textures used in turn (a matmul's conversion needn't
   * wait for the last matmul's reads). Each texture is made when first
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
        a?: [Texture, Texture]
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
    if ((pack & 3) !== 3 && !(f.halfRenders ??= this.gpu.canRender([GL.RGBA16F])))
      return false
    if (pack & 2) f.wh2 ??= this.gpu.texture(GL.RGBA32UI, f.width / 2, f.hd)
    else f.w ??= this.gpu.texture(GL.RGBA16F, f.width, f.hd)
    if (!(pack & 1))
      f.a ??= [0, 1].map(() =>
        this.gpu.texture(GL.RGBA16F, f.kMax / 4, this.capacity)
      ) as [Texture, Texture]
    return f
  }

  private f16Sized() {
    const mib = Math.max(
      0,
      this.rgba4Layers ? TUNING.f16ScratchMiB : TUNING.f16ScratchBytesMiB
    )
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
        const width = Math.min(max, Math.floor(texels / hd)) & ~1
        if (width < kMax / 4) continue
        const chunks = mats.reduce((n, m) => {
          const { kbSlice, slices } = this.quadSlices(m, TUNING.quadMaxSplit)
          const plan = Engine.f16Plan({ hd, width }, m.N, kbSlice, slices)
          return n + Math.ceil(slices / plan.group) * Math.ceil(m.N / plan.rows)
        }, 0)
        if (chunks < fewest) {
          fewest = chunks
          best = { hd, width }
        }
      }
    } else {
      // The column height that fits every matrix in the least texture.
      for (let hd = max & ~3; hd >= 64; hd = (hd >> 1) & ~3) {
        const width = Math.max(...mats.map((m) => Math.ceil(m.N / hd) * (m.K / 4)))
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
  private matmul(a: Texture, w: GpuMatrix, out: Texture, rows: number) {
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
    if ((w.rgba4 || w.bits === 4) && partner && TUNING.quad > 0) {
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
      const f =
        this.takesF16(w) && (!w.rgba4 || rows >= TUNING.f16MinRows)
          ? this.f16Scratch()
          : false
      if (f) {
        const kq = w.K / 4
        const pack = this.f16Pack()
        const h2W = (pack & 2) !== 0
        const wOut = h2W ? f.wh2 : f.w
        if (!wOut) throw new GpuError("The f16 matmul's scratch is missing.")
        // The activations as RGBA16F, unless the matmul reads their h2.
        let fa = a
        if (!(pack & 1)) {
          if (!f.a) throw new GpuError("The f16 matmul's scratch is missing.")
          fa = f.a[f.next]
          f.next ^= 1
          this.gpu.run({
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
        const dequant = this.gpu.program(
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
            this.gpu.run({
              name: "dequant",
              program: dequant,
              inputs: {
                uW: w.quants,
                uS: w.scales,
                ...(w.zeros && { uZ: w.zeros })
              },
              uniforms: {
                uKb: { int: w.blocks },
                uH: { int: w.rowsPerColumn },
                uRowTexels: { int: w.rowTexels },
                uKq: { int: kqW },
                uKq0: { int: kq0 },
                uHd: { int: f.hd },
                uN: { int: n },
                uRow0: { int: r0 }
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
            this.gpu.run({
              name: "matmul",
              program: multiply,
              inputs: { uW: wOut, uA: fa },
              uniforms: {
                uKq: { int: kq },
                uKqW: { int: kqW },
                uKq0: { int: kq0 },
                uHd: { int: f.hd },
                uSliceRows: { int: quadRows },
                uKqSlice: { int: kqSlice },
                uRow0: { int: r0 }
              },
              output: [out, partner],
              rect: {
                x: r0 / 4,
                y: s0 * quadRows,
                width: Math.ceil(n / 4),
                height: quadRows * (s1 - s0)
              },
              // (Per K texel: 4 rows' weights and 4 tokens' activations.)
              work: kqSlice * 8
            })
          }
        }
        this.slicing.set(out.id, { split: slices, stride: quadRows, quad: true })
        return
      }
      if (w.rgba4) {
        this.gpu.run({
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
          work: kbSlice * 40
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
      this.gpu.run({
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
        work: Math.ceil(w.blocks / slices) * 24
      })
      this.slicing.set(out.id, { split: slices, stride: pairRows })
      return
    }
    this.slicing.set(out.id, { split: 1, stride: 0 })
    this.gpu.run({
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
      work: w.blocks * 24
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
    const own = w.rgba4 || (w.bits === 4 && w.bytes)
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
          own
            ? K.matmulSplitRgba4({
                zeros: !!w.zeros,
                bytes: !w.rgba4,
                unorm: w.unorm,
                packed: w.packed,
                normed: !norm ? false : norm === "prescaled" ? norm : "weights",
                offset: x0 > 0
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
              })
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
            uParts: { int: TUNING.rmsParts },
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
        uChunk: { int: Math.ceil(texels / TUNING.rmsParts) },
        uSplitIn: { int: Math.max(split, 0) },
        ...this.pairIn(x)
      },
      output: this.buffers.rms,
      rect: { x: 0, y: row0, width: TUNING.rmsParts, height: rows }
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
      program: this.gpu.program(K.NORM_TO_H2),
      inputs: { uX: x, uR: ownSums ? x : this.buffers.rms, uNorm: norm },
      uniforms: {
        uRowIn: { int: rowIn },
        uROff: { int: ownSums ? hiddenSize / 4 : 0 },
        uParts: { int: TUNING.rmsParts },
        uTexels: { int: hiddenSize / 4 },
        uEps: this.config.normEps
      },
      output: out,
      rect: { x: 0, y: 0, width: hiddenSize / 8, height: rows }
    })
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
    // Through the sandwich norm (LFM3), y's sums of squares first.
    const normed = !!norm
    if (normed) this.rms(y, 0, rows, split)
    const read = (source: string) =>
      this.gpu.program(
        split
          ? this.fromSplit(source, ["uY"], split)
          : this.readPair(source, "uY", y)
      )
    const common = {
      uTexels: { int: texels },
      uEps: this.config.normEps,
      uROff: { int: 0 },
      uParts: { int: TUNING.rmsParts },
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
        uChunk: { int: Math.ceil(texels / TUNING.rmsParts) },
        uBase: { int: texels }
      },
      output,
      rect: { x: texels, y: 0, width: TUNING.rmsParts, height: rows }
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
  private convolve(
    layer: NonNullable<GpuLayer["conv"]>,
    rows: number,
    input: MatIn
  ) {
    const c = this.config
    const b = this.buffers
    const dt = c.hiddenSize / 4
    const split = this.project(input.a, layer.inProj, b.bcx, rows, input.norm)
    const bcx = split ? this.partials() : b.bcx
    const read = (source: string) =>
      this.gpu.program(
        split
          ? this.fromSplit(source, ["uBCx"], split)
          : this.readPair(source, "uBCx", bcx)
      )
    const bcxIn = { uBCx: bcx, ...(!split && this.quadInputs("uBCx", bcx)) }
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
    queriesInCache = false
  ) {
    const c = this.config
    const b = this.buffers
    const sliding = type === "sliding_attention"
    const rope = sliding ? this.ropeSliding : this.ropeFull
    const pairTexels = (sliding ? c.ropeDimsSliding : c.ropeDimsFull) / 8
    const qDims = c.numHeads * c.headDim
    const kvDims = c.numKvHeads * c.headDim
    const split = this.project(input.a, layer.qkvg, b.qkvg, rows, input.norm)
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
    if (inCache)
      for (const { s0, row0, block, run } of slotRuns(qSlot0, qSlots * rows, blockRows, layer.ring)) {
        const x0 = block * kvWidth
        this.gpu.run({
          name: "heads",
          program: read(
            K.headPrep({
              out: "cache",
              rope: true,
              norm: true,
              headDim: c.headDim,
              cache: cacheShape
            })
          ),
          inputs: { ...qkvgInputs, uNorm: layer.qNorm, uRope: rope },
          uniforms: {
            uSplitIn: { int: split },
            ...this.pairIn(qkvgIn),
            uBase: { int: 0 },
            uPairTexels: { int: pairTexels },
            uPos0: { int: pos0 },
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
        K.headPrep({ out: "f32", rope: true, norm: true, headDim: c.headDim })
      ),
      inputs: { ...qkvgInputs, uNorm: layer.qNorm, uRope: rope },
      uniforms: {
        uSplitIn: { int: split },
        ...this.pairIn(qkvgIn),
        uBase: { int: 0 },
        uPairTexels: { int: pairTexels },
        uPos0: { int: pos0 },
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
          K.headPrep({ out: "h2", rope: true, norm: true, headDim: c.headDim })
        ),
        inputs: { ...qkvgInputs, uNorm: layer.kNorm, uRope: rope },
        uniforms: {
          uSplitIn: { int: split },
          ...this.pairIn(qkvgIn),
          uBase: { int: qDims / 4 },
          uPairTexels: { int: pairTexels },
          uPos0: { int: pos0 },
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
    await this.attention(layer, type, rows, pos0, split, qkvgIn, paced, inCache)
    if (this.panelErrors) {
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
        return this.gpu.read(this.debugTexture(), { x: 0, y: 0, width: qDims / 4, height: rows })
      }
      const got = await read()
      const saved = TUNING.attendChunkTexels
      TUNING.attendChunkTexels = 0
      try {
        await this.attention(layer, type, rows, pos0, split, qkvgIn, false, inCache)
      } finally {
        TUNING.attendChunkTexels = saved
      }
      const want = await read()
      let err = 0
      let ref = 0
      for (let i = 0; i < want.length; i++) {
        err += (got[i] - want[i]) ** 2
        ref += want[i] ** 2
      }
      this.panelErrors.push(
        got.every(Number.isFinite) ? Math.sqrt(err / Math.max(ref, 1e-30)) : Infinity
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
    const scoreTex = b.scores
    const sIn = Object.fromEntries(
      scoreTex.map((t, i) => [i === 0 ? "uS" : `uS${i}`, t])
    )
    const panelRows = scoreTex[0].height
    const scoreWidth = scoreTex[0].width
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
      const scores = (sj0: number, sKeys: number, n: number) =>
        this.gpu.run({
          name: "scores",
          program: this.gpu.program(
            K.scores(c.headDim, G, inCache ? { kvWidth, slots: qSlots } : undefined)
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
          rect: {
            x: 0,
            y: 0,
            width: Math.ceil((n * total) / panelRows) * Math.min(kt, segTexels),
            height: Math.min(panelRows, n * total)
          },
          work: (c.headDim / 8) * (4 + 2 * G)
        })
      const chunks = chunksOf(chunkTexels)
      const chunkPast =
        rows <= SPLIT_ROWS
          ? Math.max(TUNING.attendChunkTexelsDecode, 2 * TUNING.attendChunkTexels)
          : 2 * TUNING.attendChunkTexels
      const fitSegments = Math.max(1, Math.floor((panels * panelRows) / total))
      if (
        rows <= SPLIT_ROWS &&
        (this.tuned.flashDecode ?? TUNING.flashDecode) > 0 &&
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
        scores(j0, keys, segments)
        this.gpu.run({
          name: "attend",
          program: this.gpu.program(K.attendPanel(c.headDim, G, TUNING.flashUnroll, TUNING.flashOnline > 0)),
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
        this.gpu.run({
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
          scores(gj0, gKeys, n)
          this.gpu.run({
            name: "attend",
            program: this.gpu.program(K.attendChunk(c.headDim, G)),
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
        this.gpu.run({
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
      scores(j0, keys, 1)
      this.gpu.run({
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
    if (fusable && saveAt === undefined && this.fusesLinear(layer, rows))
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
    const baSplit = this.project(input.a, layer.ba, t.ba, rows, input.norm)
    const baIn = baSplit ? this.partials() : t.ba
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
    const split = this.project(input.a, layer.inProj, t.inProj, rows, input.norm)
    const inTex = split ? this.partials() : t.inProj
    const read = reader("uIn", inTex, split)
    const inInputs = pairInputs("uIn", inTex, split)
    const inUniforms = { uSplitIn: { int: split }, ...this.pairIn(inTex) }
    this.gpu.run({
      name: "linear",
      program: read(K.linConv(width)),
      inputs: { ...inInputs, uState: layer.convState[0], uConv: layer.kernel },
      uniforms: inUniforms,
      output: t.qkv,
      rect: { x: 0, y: 0, width: C4, height: rows }
    })
    for (const [m, out] of [
      [rows, layer.convState[1]],
      // (The conv state after saveAt's rows, for a snapshot there.)
      ...(saveAt ? [[saveAt, this.f32(lin.C, width - 1)] as const] : [])
    ] as const) {
      this.gpu.run({
        name: "linear",
        program: read(K.linConvState(width)),
        inputs: { ...inInputs, uState: layer.convState[0] },
        uniforms: { ...inUniforms, uM: { int: m } },
        output: out,
        rect: { x: 0, y: 0, width: C4, height: width - 1 }
      })
      if (m === saveAt) this.saving?.push(out)
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
  private linearFused(
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
    for (let c0 = 0; c0 < rows; c0 += c0 === 0 ? first : chunk) {
      const c1 = Math.min(c0 + (c0 === 0 ? first : chunk), rows)
      const outIn = { uS: layer.state[0], uU: c.u, uW: c.w, uOi: c.oi, uQp: c.qp }
      const rect = { x: 0, y: c0, width: V4, height: c1 - c0 }
      if (both)
        run(K.linChunkOut({ ...heads, out: "both" }), outIn, [t.delta, t.o], V4, {}, rect, dk)
      else
        for (const [out, tex] of [["delta", t.delta], ["o", t.o]] as const)
          run(K.linChunkOut({ ...heads, out }), outIn, tex, V4, {}, rect, dk)
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
      // (Reading and writing the state: ~3 multiply-adds an element a token.)
      if (paced) await this.charge(3 * lin.V * dk * (c1 - c0))
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
      /**
       * Saves the anchor snapshot (see snapshot) after this many of the
       * tokens, from inside the batch: no batch of its own (one batch at
       * most; see canSaveInside).
       */
      saveAt?: number
    } = {}
  ): Promise<number> {
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
    const { saveAt } = options
    if (saveAt !== undefined) {
      if (!this.canSaveInside || tokens.length > this.capacity || !(saveAt > 0 && saveAt < tokens.length))
        throw new Error(`Can't save a snapshot after ${saveAt} of a ${tokens.length}-token batch.`)
      // Room first, so no more than the limit are ever held.
      this.dropSnapshots(this.snapshotLimit - 1)
      const pos0 = this.position
      this.saving = []
      try {
        const last = await this.forwardBatch(Array.from(tokens), undefined, false, saveAt)
        this.keepSnapshot(pos0 + saveAt, { conv: [], windows: [], linear: this.saving }, true)
        return last
      } catch (error) {
        for (const t of this.saving) this.gpu.delete(t)
        throw error
      } finally {
        this.saving = null
      }
    }
    let last = 0
    let start = 0
    for (const size of this.batchSizes(tokens.length)) {
      const from = start
      start += size
      const batch = Array.from({ length: size }, (_, i) => tokens[from + i])
      last = await this.forwardBatch(batch, undefined)
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

  /** The penalty mask on the CPU side, mirrored to the GPU's. */
  private maskData: Uint8Array | null = null

  /**
   * Sets which tokens the repetition penalty applies to, and bans every id
   * from `vocabLimit` on (padding rows the tokenizer has no token for).
   */
  setSeen(tokens: Iterable<number>, vocabLimit: number) {
    const mask = this.buffers.mask
    const data = new Uint8Array(mask.width * mask.height)
    data.fill(2, Math.min(vocabLimit, data.length))
    for (const t of tokens) if (t < vocabLimit) data[t] = 1
    this.maskData = data
    this.gpu.write(
      mask,
      { x: 0, y: 0, width: mask.width, height: mask.height },
      {
        format: GL.RED_INTEGER,
        type: GL.UNSIGNED_BYTE,
        pixels: data
      }
    )
  }

  /** Adds one token to those the repetition penalty applies to. */
  markSeen(token: number) {
    const data = this.maskData
    if (!data || data[token] !== 0) return
    data[token] = 1
    const width = this.buffers.mask.width
    this.gpu.write(
      this.buffers.mask,
      { x: token % width, y: Math.floor(token / width), width: 1, height: 1 },
      {
        format: GL.RED_INTEGER,
        type: GL.UNSIGNED_BYTE,
        pixels: data.subarray(token, token + 1)
      }
    )
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
    guess = false
  ): Promise<{ ids: Int32Array; logits: Float32Array }> {
    return (await this.pick(row, 1, chunks, penalty, [], guess))[0]
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
    drafts: Array<number>,
    guess = false
  ): Promise<Int32Array> {
    const c = this.config
    const b = this.buffers
    const head = this.head(row0, rows, false)
    const total = Math.ceil(c.vocabSize / CHUNK)
    type Ivec4 = [number, number, number, number]
    const picking = {
      uOutW: { int: OUT_WIDTH },
      uPenalty: penalty,
      uLogitRows: { int: this.logitRows },
      uHeadSplit: { int: head.split },
      uHeadStride: { int: head.stride },
      uExtra: { ivec4: [0, 1, 2, 3].map((i) => drafts[i] ?? -1) as Ivec4 },
      uExtra2: { ivec4: [4, 5, 6, 7].map((i) => drafts[i] ?? -1) as Ivec4 },
      uPenaltyRow0: { int: 0 }
    }
    this.gpu.run({
      name: "pick",
      program: this.gpu.program(K.CHUNK_MAX_AT),
      inputs: { uLogits: head.logits, uMask: b.mask },
      uniforms: picking,
      output: b.chunkMax,
      rect: { x: 0, y: 0, width: total, height: rows }
    })
    if (guess) {
      this.gpu.run({
        name: "pick",
        program: this.gpu.program(K.withChunks(K.CHUNK_ARGMAX_AT, total)),
        inputs: { uMax: b.chunkMax },
        uniforms: { uRow: { int: rows - 1 } },
        output: b.ids,
        rect: { x: 0, y: 0, width: 1, height: 1 }
      })
    }
    this.greedyOut ??= this.gpu.texture(GL.RGBA32F, 1, SPLIT_ROWS)
    this.gpu.run({
      name: "pick",
      program: this.gpu.program(K.withChunks(K.CHUNK_ARGMAX_ROWS, total)),
      inputs: { uMax: b.chunkMax },
      output: this.greedyOut,
      rect: { x: 0, y: 0, width: 1, height: rows }
    })
    const rect = { x: 0, y: 0, width: 1, height: rows }
    let data: Float32Array
    if (guess || TUNING.blockingReads <= 0) {
      const pending = this.gpu.readStart(this.greedyOut, rect)
      if (guess) await this.forwardGuess()
      data = await pending()
    } else {
      data = this.gpu.readNow(this.greedyOut, rect)
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
   * one head pass, one pick, one readback. `drafts` are the batch's inputs
   * after its first (row i + 1's input is drafts[i]): row r's penalty
   * counts drafts before it as seen, as drawing them one by one would.
   * With `guess`, the last row's likeliest token is read forward (see
   * forwardGuess).
   */
  async pick(
    row0: number,
    rows: number,
    chunks: number,
    penalty: number,
    drafts: Array<number>,
    guess = false,
    penaltyRow0 = 0
  ): Promise<Array<{ ids: Int32Array; logits: Float32Array }>> {
    const c = this.config
    const b = this.buffers
    const head = this.head(row0, rows, false)
    const total = Math.ceil(c.vocabSize / CHUNK)
    const count = Math.min(chunks, MAX_CHUNKS, total)
    type Ivec4 = [number, number, number, number]
    const extra = [0, 1, 2, 3].map((i) => drafts[i] ?? -1) as Ivec4
    const extra2 = [4, 5, 6, 7].map((i) => drafts[i] ?? -1) as Ivec4
    const fast = TUNING.fastPick > 0
    const picking = {
      uOutW: { int: OUT_WIDTH },
      uPenalty: penalty,
      uLogitRows: { int: this.logitRows },
      uHeadSplit: { int: head.split },
      uHeadStride: { int: head.stride },
      uExtra: { ivec4: extra },
      uExtra2: { ivec4: extra2 },
      uPenaltyRow0: { int: penaltyRow0 }
    }
    this.gpu.run({
      name: "pick",
      program: this.gpu.program(fast ? K.CHUNK_MAX_AT : K.CHUNK_MAX),
      inputs: { uLogits: head.logits, uMask: b.mask },
      uniforms: picking,
      output: b.chunkMax,
      rect: { x: 0, y: 0, width: total, height: rows }
    })
    if (guess) {
      // The last row's likeliest token as the next forward's input, from
      // the chunk maxima: the forward waits on this, not on the gather.
      this.gpu.run({
        name: "pick",
        program: this.gpu.program(
          fast ? K.withChunks(K.CHUNK_ARGMAX_AT, total) : K.CHUNK_ARGMAX
        ),
        inputs: { uMax: b.chunkMax, uLogits: head.logits, uMask: b.mask },
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
      if (guess) await this.forwardGuess()
      data = await pending()
    } else {
      // Nothing to overlap: a blocking read returns as soon as the GPU is
      // done, where polling a fence adds milliseconds.
      data = this.gpu.readNow(b.candidates, rect)
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
  private head(row0: number, rows = 1, sum = true): HeadOut {
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
        : this.headQ6K(e, rows, sum, row0)
    }
    this.normToH2(b.residual[0], this.finalNorm, b.lastNormed, rows, row0)
    if (e.rgba4) return this.headRgba4(e, rows, undefined, sum)
    if (e.bits === 6 && (TUNING.q6HeadSplit > 1 || !e.bytes))
      return this.headQ6K(e, rows, sum)
    this.gpu.run({
      name: "head",
      program: this.gpu.program(
        e.bits === 6
          ? K.MATMUL_Q6_K_HEAD
          : K.matmul({
              bits: e.bits,
              zeros: !!e.zeros,
              wrap: true,
              precision: this.precision,
              bytes: e.bytes
            })
      ),
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
   * TUNING.q6HeadSplit slices of super-blocks (as headRgba4's). With
   * `scaledRow0` (word texels only; see headScaled), reads the last add's
   * scaled input from that row on instead of lastNormed.
   */
  private headQ6K(
    e: GpuMatrix,
    rows: number,
    sum: boolean,
    scaledRow0?: number
  ): HeadOut {
    const b = this.buffers
    const split = Math.max(1, Math.min(TUNING.q6HeadSplit, 4, e.blocks))
    const sliceRows = this.logitRows
    const target = this.headTarget(split, rows)
    const prescaled = scaledRow0 !== undefined
    if (prescaled && e.bytes)
      throw new Error("Only word-texel Q6_K heads read the scaled input.")
    this.gpu.run({
      name: "head",
      program: this.gpu.program(
        !e.bytes
          ? K.matmulQ6KHeadWords({ split: split > 1, prescaled })
          : split > 1
            ? K.MATMUL_Q6_K_HEAD_SPLIT
            : K.MATMUL_Q6_K_HEAD
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
          uParts: { int: TUNING.rmsParts },
          uTexels: { int: this.config.hiddenSize / 4 },
          uEps: this.config.normEps
        })
      },
      output: target,
      rect: { x: 0, y: 0, width: OUT_WIDTH, height: split * rows * sliceRows }
    })
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
  private headRgba4(
    e: GpuMatrix,
    rows: number,
    scaledRow0?: number,
    sum = true
  ): HeadOut {
    const b = this.buffers
    // (At most 4: the pick passes add up to 4 slices themselves.)
    const split = Math.max(1, Math.min(TUNING.headSplit, 4, e.blocks))
    const sliceRows = this.logitRows
    const height = split * rows * sliceRows
    const target = this.headTarget(split, rows)
    this.gpu.run({
      name: "head",
      program: this.gpu.program(
        K.headRgba4({ zeros: !!e.zeros, prescaled: scaledRow0 !== undefined })
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
          uParts: { int: TUNING.rmsParts },
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
    this.head(row)
    const all = await this.gpu.read(b.logits, {
      x: 0,
      y: 0,
      width: OUT_WIDTH,
      height: this.logitRows
    })
    return all.subarray(0, c.vocabSize)
  }

  private async forwardBatch(
    batch: Array<number>,
    probe: LayerProbe | undefined,
    idsOnGpu = false,
    /** Gated DeltaNet layers copy their state after this many rows into `saving`. */
    saveAt?: number
  ) {
    const c = this.config
    const b = this.buffers
    const rows = batch.length
    const pos0 = this.position
    this.gpu.hints = TUNING.renderHints
    // (Writing: room past the batch for its queries too, see queryInCache.)
    this.reserve(
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
    const e = this.embedding
    // A few rows: the split matmuls apply the norm to the residual as they
    // read it (no norm pass; see TUNING.foldNorm).
    const prescaled = TUNING.foldNorm === 2 && this.mrt.scaled
    // (foldEach: 8-bit matrices fold the prescaled input too, and each of a
    // layer's two norms folds where its own matmuls can.)
    const each = prescaled && TUNING.foldEach > 0
    const folds = (m?: GpuMatrix) =>
      !!m && (m.rgba4 || (m.bits === 4 && m.bytes) || (each && m.bits === 8))
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
    const embedFolded = prescaled && e.rgba4 && foldAt(0)
    // The last add draws the final norm's scaled input for the head.
    this.headScaled =
      prescaled &&
      (e.rgba4 || (e.bits === 6 && !e.bytes && TUNING.q6FoldNorm > 0)) &&
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
        uChunk: { int: Math.ceil(texels / TUNING.rmsParts) }
      }
      const output = [b.residual[0], b.scaled[0]]
      const parts: Array<["values" | "rms" | "scaled", number, number]> = [
        ["values", 0, texels],
        ["rms", texels, TUNING.rmsParts],
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
        e.bits === 6
          ? e.bytes
            ? K.EMBED_Q6_K
            : K.EMBED_Q6_K_WORDS
          : K.embed({
              bits: e.bits,
              zeros: !!e.zeros,
              bytes: e.bytes,
              rgba4: e.rgba4
            })
      ),
      inputs: {
        uIds: b.ids,
        uW: e.quants,
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
      h2: boolean | "pair" = false
    ) => {
      if (!probe) return
      await probe(layer, name, async () => {
        let src = tex
        if (h2) {
          this.gpu.run({
            program: this.gpu.program(
              h2 === "pair"
                ? this.readPair(K.PAIR_TO_F32, "uX", tex)
                : K.H2_TO_F32
            ),
            uniforms: this.pairIn(tex),
            inputs: { uX: tex, ...(h2 === "pair" && this.quadInputs("uX", tex)) },
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
    await look(-1, "embedding", b.residual[0], c.hiddenSize)

    const paced = rows > SPLIT_ROWS
    // Only the last row's logits are asked for (see lastRowOnly).
    const lastRowOnly = TUNING.lastRowOnly > 0 && rows > SPLIT_ROWS && !probe
    this.lastRowOnly = lastRowOnly
    for (let i = 0; i < this.layers.length; i++) {
      const layer = this.layers[i]
      // (From layer 0: batches follow each other with no wait between.)
      if (paced) await this.charge(0)
      else if (TUNING.flushLayers > 0 && i > 0 && i % TUNING.flushLayers === 0)
        this.gpu.flush()
      const [h, next] = b.residual
      const fold = foldAt(i)
      const ffnFold = foldFfn(i)
      // (Layer 0 folds only if the embedding drew its sums and input.)
      const opIn: MatIn =
        fold && (i > 0 || (embedFolded && prescaled))
          ? { a: h, norm: prescaled ? "prescaled" : layer.operatorNorm }
          : { a: b.normed }
      if (!opIn.norm) {
        this.normToH2(h, layer.operatorNorm, b.normed, rows, -1, i > 0)
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
        opSplit = this.convolve(layer.conv, rows, opIn)
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
          !probe
        )
        await look(i, "qkvg", b.qkvg, this.qkvgDims(), "pair")
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
      await look(i, "op_out", b.opOut, c.hiddenSize, "pair")
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
      if (!ffnFold) this.normToH2(base, layer.ffnNorm, b.normed, fr)
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
          : this.splitPartials(b.normed, layer.gateUp, fr)
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
        this.matmul(b.normed, layer.gateUp, b.gateUp, rows)
        this.gpu.run({
          name: "swiglu",
          program: this.gpu.program(this.readPair(K.SWIGLU, "uGU", b.gateUp)),
          inputs: { uGU: b.gateUp, ...this.quadInputs("uGU", b.gateUp) },
          uniforms: {
            uHalf: { int: c.intermediateSize / 4 },
            ...this.pairIn(b.gateUp)
          },
          output: b.act,
          rect: { x: 0, y: 0, width: c.intermediateSize / 8, height: rows }
        })
      }
      const downSplit = this.project(b.act, layer.down, b.ffnOut, fr)
      await look(i, "ffn_out", b.ffnOut, c.hiddenSize, "pair")
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
