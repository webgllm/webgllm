/**
 * LFM3 on WebGL 2: uploads the weights a loader hands over, and runs the
 * forward pass a batch of tokens at a time, keeping the attention cache and
 * the convolutions' state on the GPU between calls.
 */
import { GL, Gpu, GpuError } from "./gpu/gl.js"
import type { Texture } from "./gpu/gl.js"
import * as K from "./gpu/kernels.js"
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
  const four = (x: number) =>
    ((x & 15) << 12) |
    (((x >>> 8) & 15) << 8) |
    (((x >>> 16) & 15) << 4) |
    ((x >>> 24) & 15)
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
  /** 1: prompt matmuls take four tokens per fragment (where the engine has room). */
  quad: 1,
  /** Their K-slices: about this many K values each (twice for over 8192 rows)... */
  quadSliceK: 512,
  /** ...and at most this many (readers add them up). */
  quadMaxSplit: 4,
  /** K-slices of the RGBA4 output head (more than 1 adds a pass to sum them). */
  headSplit: 4,
  /**
   * Few-token matmuls apply the RMS norm as they read, with no norm pass:
   * 1 multiplies by its weights there, 2 reads h2(x · weights) the add drew.
   */
  foldNorm: 2
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
  /** Tokens the cache holds, prompt and reply together. */
  contextLength?: number
  /** Most tokens per forward batch. */
  batchSize?: number
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
/** Resume points kept (see `snapshot`): ~3 MB each for LFM3. */
const MAX_SNAPSHOTS = 8
/** Queries per attention slice: the scores buffer holds this many per head. */
/**
 * Queries per attention slice (see attend): more is fewer passes and more
 * fragments per pass, for a bigger scores buffer (64: ~33 MB at an 8192
 * context), so it's 64 only for desktop-size batches.
 */
const SCORE_ROWS = 32
const SCORE_ROWS_LARGE = 64
/** Most key chunks per query in chunked attention (see attendChunk). */
const ATTEND_CHUNKS = 16
/** Logits per chunk when picking candidates on the GPU. */
const CHUNK = 256
/** Most chunks read back as candidates: exact for top-k up to this. */
export const MAX_CHUNKS = 64
/** Batches this small split their matmuls along K (see TUNING). */
const SPLIT_ROWS = 4

export class Engine {
  readonly gpu: Gpu
  config!: ModelConfig
  readonly contextLength: number
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
  private readonly quadMatmuls: boolean
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
    this.contextLength = options.contextLength ?? 8192
    this.baseBatch = options.batchSize ?? 128
    this.batchSize = this.baseBatch
    this.capacity = this.batchSize + this.batchSize / 8
    this.scaleBatch = options.scaleBatch ?? true
    this.precision = options.matmulPrecision ?? "highp"
    this.bytes = (options.weightTexels ?? "bytes") === "bytes"
    this.ringCache = options.ringCache ?? true
    this.quadMatmuls = options.quadMatmuls ?? this.baseBatch >= 256
    for (const name of options.skip ?? []) this.gpu.skip.add(name)
  }

  /** Uploads the model, piece by piece, as the loader yields it. */
  async load(
    pieces: AsyncIterable<WeightPiece>,
    onProgress?: (piece: WeightPiece) => void
  ) {
    for await (const piece of pieces) {
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
      else this.layers[piece.layer] = this.layer(piece.value)
      onProgress?.(piece)
      if (this.gpu.isLost)
        throw new GpuError("The GPU context was lost while loading.")
    }
    if (
      !this.embedding ||
      !this.finalNorm ||
      this.layers.length !== this.config.numLayers
    ) {
      throw new Error("The model's weights are incomplete.")
    }
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
    this.buffers = this.allocate()
    if (this.quadMatmuls) {
      const b = this.buffers
      for (const t of [b.opOut, b.bcx, b.qkvg, b.gateUp, b.ffnOut])
        this.quads.set(t.id, this.gpu.texture(GL.RGBA32UI, t.width, t.height))
    }
    this.ropeFull = this.ropeTable(config.ropeThetaFull, config.ropeDimsFull)
    this.ropeSliding = this.ropeTable(
      config.ropeThetaSliding,
      config.ropeDimsSliding
    )
  }

  /**
   * cos and sin of every position × frequency, computed in doubles: texels
   * 0..15 cos, 16..31 sin of frequencies 0..63. Frequencies past the
   * rotated dims are 0 (no rotation).
   */
  private ropeTable(theta: number, rotatedDims: number) {
    const { headDim } = this.config
    const half = headDim / 2
    const table = new Float32Array(this.contextLength * half * 2)
    for (let pos = 0; pos < this.contextLength; pos++) {
      for (let i = 0; i < half; i++) {
        const freq = i < rotatedDims / 2 ? theta ** ((-2 * i) / headDim) : 0
        const angle = pos * freq
        table[pos * headDim + i] = Math.cos(angle)
        table[pos * headDim + half + i] = Math.sin(angle)
      }
    }
    return this.gpu.texture(GL.RGBA32F, headDim / 4, this.contextLength, {
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

  private allocate() {
    const c = this.config
    const M = this.capacity
    const D = c.hiddenSize
    const qkvg = (c.numHeads + 2 * c.numKvHeads) * c.headDim + c.numHeads
    const scoreTexels = Math.ceil(this.contextLength / 4)
    const qDims = c.numHeads * c.headDim
    const I = c.intermediateSize
    // Each tensor has its own texture: sharing one between tensors that are
    // never alive at once saves ~25 MB at 512 rows, but a pass that writes a
    // texture the pass before read must wait for it, and prompts read ~12%
    // slower. The narrow matmuls' outputs are K-split, two slices tall.
    return {
      ids: this.gpu.texture(GL.R32UI, 1, M),
      // The hidden state, and its rows' partial sums of squares in the last
      // RMS_PARTS columns (drawn with the add that writes it).
      residual: [0, 1].map(() =>
        // The residual, its rows' sums of squares, and room for the
        // scaled texture's columns (written alongside, unused here).
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
      bcx: this.pair(3 * D, this.quadMatmuls ? 2 * M : M),
      convOut: this.h2(D, M),
      qkvg: this.pair(qkvg, 2 * M),
      q: this.f32(c.numHeads * c.headDim, M),
      scores: this.gpu.texture(
        GL.RGBA32F,
        scoreTexels,
        c.numHeads *
          Math.min(M, this.baseBatch >= 256 ? SCORE_ROWS_LARGE : SCORE_ROWS)
      ),
      attnOut: this.h2(qDims, M),
      /**
       * Chunked attention's partial results (desktop batches): weighted
       * values (two textures) and (max, sum), per query and chunk.
       */
      attnChunks:
        this.baseBatch >= 256
          ? ([0, 1, 2].map(() =>
              this.gpu.texture(
                GL.RGBA32F,
                qDims / 8,
                SCORE_ROWS_LARGE * ATTEND_CHUNKS
              )
            ) as [Texture, Texture, Texture])
          : null,
      gateUp: this.pair(2 * I, M),
      act: this.h2(I, M),
      ffnOut: this.pair(D, 2 * M),
      lastNormed: this.h2(D, SPLIT_ROWS),
      /** Logits: logitRows rows per token, for up to SPLIT_ROWS tokens. */
      logits: this.gpu.texture(
        GL.RGBA32F,
        OUT_WIDTH,
        Math.ceil(c.vocabSize / 4 / OUT_WIDTH) * SPLIT_ROWS
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
      candidates: this.gpu.texture(
        GL.RGBA32F,
        CHUNK / 4 + 1,
        MAX_CHUNKS * SPLIT_ROWS
      ),
      /** Split matmuls' partial sums: widest output × (rows × slices). */
      partial: this.gpu.texture(
        GL.RGBA32F,
        Math.max(2 * c.intermediateSize, qkvg, 3 * D) / 4,
        SPLIT_ROWS * 32
      )
    }
  }

  private vector(values: Float32Array, rows = 1) {
    return this.gpu.texture(GL.RGBA32F, values.length / 4 / rows, rows, {
      format: GL.RGBA,
      type: GL.FLOAT,
      pixels: values
    })
  }

  /** A quantized matrix as textures, filled with `m`. */
  private matrix(m: QuantMatrix): GpuMatrix {
    const gm = this.allocateMatrix(m.N, m.K, m.bits, !!m.zeros, m.bits === 4)
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
    rgba4 = false
  ): GpuMatrix {
    const max = this.gpu.limits.maxTextureSize
    if (rgba4) {
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
        rgba4: true,
        blocks,
        groups,
        rowsPerColumn,
        rowTexels,
        quants: this.gpu.texture(GL.RGBA4, columns * rowTexels, rowsPerColumn),
        scales: this.gpu.texture(GL.RGBA16F, columns * blocks, rowsPerColumn),
        zeros: withZeros
          ? this.gpu.texture(GL.RGBA8UI, columns * blocks, rowsPerColumn)
          : null
      }
    }
    // A block row: 4-bit 16 bytes, 8-bit 32, as RGBA32UI texels (1, 2) or
    // byte texels (4, 8); a Q6_K super-block row, 208 bytes: 52 byte texels
    // (always bytes: its kernels split them with float math, faster than
    // integer shifts on mobile GPUs).
    const blocks = K / (bits === 6 ? 256 : 32)
    const groups = N / 4
    const columns = Math.ceil(groups / max)
    // Byte texels are 4× as wide: words where a row group wouldn't fit.
    const byteTexels = bits === 6 ? 52 : bits === 4 ? 4 : 8
    const bytes =
      bits === 6 || (this.bytes && columns * blocks * 4 * byteTexels <= max)
    const texelsPerBlockRow = bytes ? byteTexels : byteTexels / 4
    const rowTexels = blocks * 4 * texelsPerBlockRow
    const quantFormat = !bytes
      ? GL.RGBA32UI
      : bits === 8
        ? GL.RGBA8I
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
      rgba4: false,
      blocks,
      groups,
      rowsPerColumn,
      rowTexels,
      quants: this.gpu.texture(quantFormat, columns * rowTexels, rowsPerColumn),
      scales: this.gpu.texture(GL.RGBA16F, columns * blocks, rowsPerColumn),
      zeros: withZeros
        ? this.gpu.texture(GL.RGBA8UI, columns * blocks, rowsPerColumn)
        : null
    }
  }

  /** Writes `m`'s rows into rows [row0, row0 + m.N) of `gm` (row0 a multiple of 4). */
  private writeMatrix(gm: GpuMatrix, m: QuantMatrix, row0: number) {
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
        this.gpu.write(
          gm.quants,
          { x: column * rowTexels, y, width: rowTexels, height: run },
          {
            format: GL.RGBA,
            type: GL.UNSIGNED_SHORT_4_4_4_4,
            pixels: toRgba4(quants)
          }
        )
      } else
      this.gpu.write(
        gm.quants,
        { x: column * rowTexels, y, width: rowTexels, height: run },
        {
          format: GL.RGBA_INTEGER,
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
        state: [this.f32(D, width - 1), this.f32(D, width - 1)],
        prefixes: this.f32(D, (width - 1) * SPLIT_ROWS)
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
      const rows = ring || this.contextLength
      layer.attention = {
        qkvg: this.matrix(w.attention.qkvg),
        o: this.matrix(w.attention.o),
        qNorm: this.vector(w.attention.qNorm),
        kNorm: this.vector(w.attention.kNorm),
        kv: this.h2(2 * kvDims, rows),
        ring
      }
    }
    return layer
  }

  /**
   * What a later prompt needs to resume at a cache position: the
   * convolutions' state, and the sliding-window layers' last window of keys
   * and values (their ring may have moved on since). Full-attention layers
   * keep every position, so they need nothing saved.
   */
  private snapshots = new Map<
    number,
    { conv: Array<Texture>; windows: Array<Texture> }
  >()

  private freeSnapshot(saved: {
    conv: Array<Texture>
    windows: Array<Texture>
  }) {
    saved.conv.forEach((t) => this.gpu.delete(t))
    saved.windows.forEach((kv) => this.gpu.delete(kv))
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

  /**
   * Saves what's needed to resume at the current position (see
   * `snapshots`), so a later prompt that shares the tokens so far reads only
   * what's new. Keeps the latest MAX_SNAPSHOTS.
   */
  snapshot() {
    const at = this.position
    if (this.snapshots.has(at)) return
    const W = this.config.slidingWindow
    const saved = {
      conv: [] as Array<Texture>,
      windows: [] as Array<Texture>
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
      const a = layer.attention
      if (a?.ring) {
        const kv = this.gpu.texture(GL.RGBA32UI, a.kv.width, W)
        this.copyPositions(a.kv, a.ring, kv, W, Math.max(0, at - W), at)
        saved.windows.push(kv)
      }
    }
    this.snapshots.set(at, saved)
    for (const [p, old] of this.snapshots) {
      if (this.snapshots.size <= MAX_SNAPSHOTS) break
      this.freeSnapshot(old)
      this.snapshots.delete(p)
    }
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
    for (const layer of this.layers) {
      if (layer.conv) {
        this.gpu.run({
          program: this.gpu.program(K.COPY),
          inputs: { uX: saved.conv[c++] },
          output: layer.conv.state[0]
        })
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

  /** A scratch tensor for probes, made the first time one asks. */
  private debugTexture() {
    const c = this.config
    const qkvg = (c.numHeads + 2 * c.numKvHeads) * c.headDim + c.numHeads
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
    for (const saved of this.snapshots.values()) this.freeSnapshot(saved)
    this.snapshots.clear()
    const zero = new Float32Array(
      this.config.hiddenSize * (this.config.convKernel - 1)
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
    }
  }

  /**
   * Runs every kind of pass once (a prompt batch, a draft batch, a single
   * token, both ways of sampling, a snapshot) so shaders compile, and the
   * driver builds its pipelines, at load rather than inside the first
   * reply's timings. Then forgets it all.
   */
  async warmUp() {
    const row = await this.forwardHidden(new Array(SPLIT_ROWS + 1).fill(0))
    await this.candidates(row, 1, 1, true)
    this.dropGuess()
    await this.logits(row)
    await this.forwardHidden([0, 0], { draft: true })
    this.accept(1)
    await this.logits(await this.forwardHidden([0]))
    this.snapshot()
    this.reset()
  }

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
    if (rows <= SPLIT_ROWS) {
      const slices = this.splitPartials(a, w, rows)
      this.gpu.run({
        name: "matmul",
        program: this.gpu.program(K.REDUCE_PAIR),
        inputs: { uP: this.buffers.partial },
        uniforms: { uSplit: { int: slices }, uRows: { int: rows } },
        output: out,
        rect: { x: 0, y: 0, width: w.groups, height: Math.ceil(rows / 2) }
      })
      this.slicing.set(out.id, { split: 1, stride: 0 })
      return
    }
    const partner = this.quads.get(out.id)
    if (w.rgba4 && partner && TUNING.quad > 0) {
      // Four tokens per fragment, into two textures; K in slices of about
      // quadSliceK values (twice that for wide matrices), stacked in rows.
      const quadRows = Math.ceil(rows / 4)
      const sliceK = TUNING.quadSliceK * (w.groups > 2048 ? 2 : 1)
      // The most slices the output holds at a full batch: the same for
      // every batch, so the warm-up compiles the programs prompts use.
      const most = Math.min(
        TUNING.quadMaxSplit,
        Math.floor(out.height / Math.ceil(this.capacity / 4))
      )
      const kbSlice = Math.ceil(
        w.blocks / Math.max(1, Math.min(most, Math.round(w.K / sliceK)))
      )
      const slices = Math.ceil(w.blocks / kbSlice)
      this.gpu.run({
        name: "matmul",
        program: this.gpu.program(K.matmulQuad({ zeros: !!w.zeros })),
        inputs: weights,
        uniforms: {
          ...shape,
          uSliceRows: { int: quadRows },
          uKbSlice: { int: kbSlice }
        },
        output: [out, partner],
        rect: { x: 0, y: 0, width: w.groups, height: quadRows * slices }
      })
      this.slicing.set(out.id, { split: slices, stride: quadRows, quad: true })
      return
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
          K.matmulPair({ zeros: !!w.zeros })
        ),
        inputs: weights,
        uniforms: {
          ...shape,
          uSliceRows: { int: pairRows },
          uKbSlice: { int: Math.ceil(w.blocks / slices) }
        },
        output: out,
        rect: { x: 0, y: 0, width, height: pairRows * slices }
      })
      this.slicing.set(out.id, { split: slices, stride: pairRows })
      return
    }
    this.slicing.set(out.id, { split: 1, stride: 0 })
    this.gpu.run({
      name: "matmul",
      program: this.gpu.program(
        w.rgba4
          ? K.matmulPair({ zeros: !!w.zeros })
          : K.matmul({
              bits,
              zeros: !!w.zeros,
              wrap: false,
              precision: this.precision,
              bytes: w.bytes,
              pair: true
            })
      ),
      inputs: weights,
      uniforms: shape,
      output: out,
      rect: { x: 0, y: 0, width: w.groups, height: Math.ceil(rows / 2) }
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
    norm?: Texture | "prescaled"
  ) {
    if (norm && !w.rgba4)
      throw new Error("Only RGBA4 matmuls fold the norm in.")
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
          w.rgba4
            ? K.matmulSplitRgba4({
                zeros: !!w.zeros,
                normed: !norm ? false : norm === "prescaled" ? norm : "weights"
              })
            : K.matmul({
                bits,
                zeros: !!w.zeros,
                wrap: false,
                precision: this.precision,
                bytes: w.bytes,
                split: true
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
          ...(norm && {
            uAOff: { int: this.config.hiddenSize / 4 + K.RMS_PARTS },
            uROff: { int: this.config.hiddenSize / 4 },
            uParts: { int: TUNING.rmsParts },
            uTexels: { int: this.config.hiddenSize / 4 },
            uEps: this.config.normEps
          })
        },
        output: this.buffers.partial,
        rect: { x: 0, y: 0, width: w.groups, height: rows * slices }
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
    if (this.keepPrefixes && rows > 1) {
      // The state after each shorter prefix too, for accept().
      this.gpu.run({
        name: "conv",
        program: read(K.convPrefixes(c.convKernel)),
        inputs: { ...bcxIn, uState: layer.state[0] },
        uniforms: { uDt: { int: dt }, uSplitIn: { int: split }, ...this.pairIn(bcx) },
        output: layer.prefixes,
        rect: { x: 0, y: 0, width: dt, height: (c.convKernel - 1) * rows }
      })
    }
    layer.state.reverse()
    return this.project(b.convOut, layer.outProj, b.opOut, rows)
  }

  private attend(
    layer: NonNullable<GpuLayer["attention"]>,
    type: LayerType,
    rows: number,
    pos0: number,
    input: MatIn
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
    // Queries: normalized, rotated, scaled by 1/sqrt(head dim).
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
        uEps: c.normEps,
        uScale: 1 / Math.sqrt(c.headDim)
      },
      output: b.q,
      rect: { x: 0, y: 0, width: qDims / 4, height: rows }
    })
    // Keys and values into the cache at positions pos0..., in up to two
    // runs of rows where a ring cache wraps around.
    const length = layer.ring || this.contextLength
    for (let m0 = 0; m0 < rows;) {
      const row0 = (pos0 + m0) % length
      const run = Math.min(rows - m0, length - row0)
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
          uEps: c.normEps,
          uScale: 1
        },
        output: layer.kv,
        rect: { x: 0, y: row0, width: kvDims / 8, height: run }
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
          uXOff: { int: kvDims / 8 }
        },
        output: layer.kv,
        rect: { x: kvDims / 8, y: row0, width: kvDims / 8, height: run }
      })
      m0 += run
    }
    // Queries in slices, so the scores buffer stays small: each slice sees
    // keys up to its last query (and, in sliding layers, from its first
    // query's window on). The buffer holds SCORE_ROWS queries of every head
    // at the full context's width; with fewer keys than that, panels of
    // them sit side by side and a slice holds more queries.
    const group = c.numHeads / c.numKvHeads
    const panelRows = b.scores.height
    for (let m0 = 0; m0 < rows;) {
      const j0 = sliding ? Math.max(0, pos0 + m0 - c.slidingWindow + 1) : 0
      const widest = Math.ceil((pos0 + rows - j0) / 4)
      const panels = Math.max(1, Math.floor(b.scores.width / widest))
      const slice = Math.min(
        rows - m0,
        Math.max(1, Math.floor((panels * panelRows) / c.numHeads))
      )
      const keys = pos0 + m0 + slice - j0
      const kt = Math.ceil(keys / 4)
      const total = c.numHeads * slice
      const ring = {
        uRows: { int: slice },
        uM0: { int: m0 },
        uRing: { int: layer.ring },
        uKt: { int: kt },
        uPanelRows: { int: panelRows }
      }
      this.gpu.run({
        name: "scores",
        program: this.gpu.program(K.scores(c.headDim)),
        inputs: { uQ: b.q, uK: layer.kv },
        uniforms: {
          ...ring,
          uJ0: { int: j0 },
          uKeys: { int: keys },
          uPos0: { int: pos0 },
          uWindow: { int: sliding ? c.slidingWindow : 0 },
          uGroup: { int: group },
          uTotalRows: { int: total }
        },
        output: b.scores,
        rect: {
          x: 0,
          y: 0,
          width: Math.ceil(total / panelRows) * kt,
          height: Math.min(panelRows, total)
        }
      })
      const chunkBufs = b.attnChunks
      const chunkTexels = Math.max(
        TUNING.attendChunkTexels,
        Math.ceil(kt / ATTEND_CHUNKS)
      )
      const chunks = Math.ceil(kt / chunkTexels)
      if (
        chunkBufs &&
        TUNING.attendChunkTexels > 0 &&
        kt > 2 * TUNING.attendChunkTexels &&
        slice * chunks <= chunkBufs[0].height
      ) {
        // Long key ranges: chunks of keys in parallel, then combined.
        this.gpu.run({
          name: "attend",
          program: this.gpu.program(K.attendChunk(c.headDim)),
          inputs: { uS: b.scores, uV: layer.kv },
          uniforms: {
            ...ring,
            uJ0: { int: j0 },
            uKeys: { int: keys },
            uGroup: { int: group },
            uVOff: { int: kvDims / 8 },
            uSliceQ: { int: slice },
            uChunkTexels: { int: chunkTexels }
          },
          output: chunkBufs,
          rect: { x: 0, y: 0, width: qDims / 8, height: slice * chunks }
        })
        this.gpu.run({
          name: "attend",
          program: read(K.attendCombine(c.headDim, c.attentionGate)),
          inputs: {
            uP0: chunkBufs[0],
            uP1: chunkBufs[1],
            uP2: chunkBufs[2],
            ...qkvgInputs
          },
          uniforms: {
            uSplitIn: { int: split },
            ...this.pairIn(qkvgIn),
            uM0: { int: m0 },
            uSliceQ: { int: slice },
            uChunks: { int: chunks },
            uGateTexel: { int: (qDims + 2 * kvDims) / 4 }
          },
          output: b.attnOut,
          rect: { x: 0, y: m0, width: qDims / 8, height: slice }
        })
        m0 += slice
        continue
      }
      this.gpu.run({
        name: "attend",
        program: read(K.attend(c.headDim, c.attentionGate)),
        inputs: { uS: b.scores, uV: layer.kv, ...qkvgInputs },
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
        rect: { x: 0, y: m0, width: qDims / 8, height: slice }
      })
      m0 += slice
    }
    return this.project(b.attnOut, layer.o, b.opOut, rows)
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
    await this.forwardBatch(Array.from(tokens), undefined)
  }

  /**
   * Reads tokens into the cache like `forward`, without reading logits
   * back; returns the row of the last token, for `candidates`.
   */
  async forwardHidden(
    tokens: ArrayLike<number>,
    options: { draft?: boolean } = {}
  ): Promise<number> {
    if (options.draft) return this.forwardDraft(Array.from(tokens))
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
      last = await this.forwardBatch(batch, undefined)
    }
    return last
  }

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

  /** Where a guessed token was read (see `forwardGuess`), while it's pending. */
  private guessAt: number | null = null

  /** Whether a guessed token's forward is pending (keepGuess or dropGuess). */
  get guessing() {
    return this.guessAt !== null
  }

  /**
   * Reads the token in the ids texture (CANDIDATE_ARGMAX's pick) as one
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
    for (const layer of this.layers) layer.conv?.state.reverse()
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
    guess = false
  ): Promise<Array<{ ids: Int32Array; logits: Float32Array }>> {
    const c = this.config
    const b = this.buffers
    if (rows > 1 && !this.embedding.rgba4) {
      // The other output heads take one row at a time.
      const out = []
      for (let r = 0; r < rows; r++) {
        out.push(
          ...(await this.pick(
            row0 + r,
            1,
            chunks,
            penalty,
            [],
            guess && r === rows - 1
          ))
        )
      }
      return out
    }
    this.head(row0, rows)
    const total = Math.ceil(c.vocabSize / CHUNK)
    const count = Math.min(chunks, MAX_CHUNKS, total)
    const extra = [0, 1, 2, 3].map((i) => drafts[i] ?? -1) as [
      number,
      number,
      number,
      number
    ]
    const picking = {
      uOutW: { int: OUT_WIDTH },
      uPenalty: penalty,
      uLogitRows: { int: this.logitRows },
      uExtra: { ivec4: extra }
    }
    this.gpu.run({
      name: "pick",
      program: this.gpu.program(K.CHUNK_MAX),
      inputs: { uLogits: b.logits, uMask: b.mask },
      uniforms: picking,
      output: b.chunkMax,
      rect: { x: 0, y: 0, width: total, height: rows }
    })
    this.gpu.run({
      name: "pick",
      program: this.gpu.program(K.CHUNK_RANK),
      inputs: { uMax: b.chunkMax },
      uniforms: { uChunks: { int: total } },
      output: b.chunkRank,
      rect: { x: 0, y: 0, width: total, height: rows }
    })
    this.gpu.run({
      name: "pick",
      program: this.gpu.program(K.CHUNK_BY_RANK),
      inputs: { uRank: b.chunkRank },
      uniforms: { uChunks: { int: total } },
      output: b.chunkByRank,
      rect: { x: 0, y: 0, width: count, height: rows }
    })
    this.gpu.run({
      name: "pick",
      program: this.gpu.program(K.CHUNK_GATHER),
      inputs: { uLogits: b.logits, uMask: b.mask, uByRank: b.chunkByRank },
      uniforms: { ...picking, uCount: { int: count } },
      output: b.candidates,
      rect: { x: 0, y: 0, width: CHUNK / 4 + 1, height: count * rows }
    })
    const texels = CHUNK / 4 + 1
    if (guess) {
      // The last row's likeliest token as the next forward's input.
      this.gpu.run({
        name: "pick",
        program: this.gpu.program(K.CANDIDATE_ARGMAX),
        inputs: { uCand: b.candidates },
        uniforms: { uCandRow: { int: (rows - 1) * count } },
        output: b.ids,
        rect: { x: 0, y: 0, width: 1, height: 1 }
      })
    }
    const pending = this.gpu.readStart(b.candidates, {
      x: 0,
      y: 0,
      width: texels,
      height: count * rows
    })
    // Read it forward while the candidates go back to the CPU (see guess).
    if (guess) await this.forwardGuess()
    const data = await pending()
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
  private head(row0: number, rows = 1) {
    const b = this.buffers
    const e = this.embedding
    this.normToH2(b.residual[0], this.finalNorm, b.lastNormed, rows, row0)
    if (e.rgba4) {
      this.headRgba4(e, rows)
      return
    }
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
        uRow: { int: 0 }
      },
      output: b.logits
    })
  }

  /** The scaled texture drawn with residual texture `r` (see buffers.scaled). */
  private scaledOf(r: Texture) {
    const b = this.buffers
    return r === b.residual[0] ? b.scaled[0] : b.scaled[1]
  }

  private headScratch: Texture | null = null

  /**
   * The output head through RGBA4 weights, split along K in
   * TUNING.headSplit slices (added up by one more pass).
   */
  private headRgba4(e: GpuMatrix, rows: number) {
    const b = this.buffers
    const split = Math.max(1, Math.min(TUNING.headSplit, e.blocks))
    const sliceRows = this.logitRows
    const height = split * rows * sliceRows
    let target = b.logits
    if (split > 1) {
      const p = b.partial
      if (p.width >= OUT_WIDTH && p.height >= height) target = p
      else {
        if (!this.headScratch || this.headScratch.height < height) {
          if (this.headScratch) this.gpu.delete(this.headScratch)
          this.headScratch = this.f32(OUT_WIDTH * 4, height)
        }
        target = this.headScratch
      }
    }
    this.gpu.run({
      name: "head",
      program: this.gpu.program(K.headRgba4({ zeros: !!e.zeros })),
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
        uSliceRows: { int: sliceRows },
        uKbSlice: { int: Math.ceil(e.blocks / split) },
        uRows: { int: rows }
      },
      output: target,
      rect: { x: 0, y: 0, width: OUT_WIDTH, height }
    })
    if (split === 1) return
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
    idsOnGpu = false
  ) {
    const c = this.config
    const b = this.buffers
    const rows = batch.length
    const pos0 = this.position
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
    this.gpu.run({
      program: this.gpu.program(
        e.bits === 6
          ? K.EMBED_Q6_K
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

    for (let i = 0; i < this.layers.length; i++) {
      const layer = this.layers[i]
      const [h, next] = b.residual
      // A few rows: the split matmuls apply the norm to the residual as
      // they read it (no norm pass). The embedding has no sums of squares
      // for that, so layer 0 normalizes first.
      const foldAt = (j: number) => {
        const l = this.layers[j]
        return (
          rows <= SPLIT_ROWS &&
          TUNING.foldNorm > 0 &&
          !probe &&
          (l.conv?.inProj ?? l.attention?.qkvg)?.rgba4 === true &&
          l.gateUp.rgba4
        )
      }
      const fold = foldAt(i)
      const prescaled = TUNING.foldNorm === 2
      const opIn: MatIn =
        fold && i > 0
          ? { a: h, norm: prescaled ? "prescaled" : layer.operatorNorm }
          : { a: b.normed }
      if (!opIn.norm) {
        this.normToH2(h, layer.operatorNorm, b.normed, rows, -1, i > 0)
        await look(i, "operator_norm", b.normed, c.hiddenSize, true)
      }
      let opSplit = 0
      if (layer.conv) {
        opSplit = this.convolve(layer.conv, rows, opIn)
        await look(i, "in_proj", b.bcx, 3 * c.hiddenSize, "pair")
        await look(i, "conv", b.convOut, c.hiddenSize, true)
      } else if (layer.attention) {
        opSplit = this.attend(layer.attention, layer.type, rows, pos0, opIn)
        const qkvgDims =
          (c.numHeads + 2 * c.numKvHeads) * c.headDim + c.numHeads
        await look(i, "qkvg", b.qkvg, qkvgDims, "pair")
        await look(i, "q", b.q, c.numHeads * c.headDim)
        await look(i, "attn_out", b.attnOut, c.numHeads * c.headDim, true)
      }
      await look(i, "op_out", b.opOut, c.hiddenSize, "pair")
      this.addNormed(
        h,
        opSplit ? this.partials() : b.opOut,
        layer.postOperatorNorm,
        next,
        rows,
        opSplit,
        fold && prescaled ? layer.ffnNorm : undefined
      )
      await look(i, "after_op", next, c.hiddenSize)
      if (!fold) this.normToH2(next, layer.ffnNorm, b.normed, rows)
      if (rows <= SPLIT_ROWS) {
        // A few rows: the split matmul's slices are summed and through
        // SwiGLU in one pass.
        const slices = fold
          ? this.splitPartials(
              next,
              layer.gateUp,
              rows,
              prescaled ? "prescaled" : layer.ffnNorm
            )
          : this.splitPartials(b.normed, layer.gateUp, rows)
        this.gpu.run({
          name: "swiglu",
          program: this.gpu.program(K.withSplit(K.REDUCE_SWIGLU, slices)),
          inputs: { uP: this.partials() },
          uniforms: {
            uSplitIn: { int: slices },
            uHalf: { int: c.intermediateSize / 4 }
          },
          output: b.act,
          rect: { x: 0, y: 0, width: c.intermediateSize / 8, height: rows }
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
      const downSplit = this.project(b.act, layer.down, b.ffnOut, rows)
      await look(i, "ffn_out", b.ffnOut, c.hiddenSize, "pair")
      this.addNormed(
        next,
        downSplit ? this.partials() : b.ffnOut,
        layer.postFfnNorm,
        h,
        rows,
        downSplit,
        prescaled && i + 1 < this.layers.length && foldAt(i + 1)
          ? this.layers[i + 1].operatorNorm
          : undefined
      )
      await look(i, "l_out", h, c.hiddenSize)
    }
    this.position += rows
    return rows - 1
  }
}
