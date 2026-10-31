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
  bits: 4 | 8
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
  postOperatorNorm: Texture
  ffnNorm: Texture
  postFfnNorm: Texture
  conv?: {
    inProj: GpuMatrix
    outProj: GpuMatrix
    kernel: Texture
    state: [Texture, Texture]
  }
  attention?: {
    qkvg: GpuMatrix
    o: GpuMatrix
    qNorm: Texture
    kNorm: Texture
    keys: Texture
    values: Texture
  }
  gateUp: GpuMatrix
  down: GpuMatrix
}

export type EngineOptions = {
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
/** Logits per chunk when picking candidates on the GPU. */
const CHUNK = 256
/** Most chunks read back as candidates: exact for top-k up to this. */
export const MAX_CHUNKS = 64
/** Batches this small sum each norm's row inside the pass that uses it. */
const FUSED_NORM_ROWS = 8
/** Batches this small split their matmuls along K... */
const SPLIT_ROWS = 4
/** ...into up to this many slices... */
const MAX_SPLIT = 16
/** ...for about this many fragments per matmul. */
const SPLIT_FRAGMENTS = 4096

export class Engine {
  readonly gpu: Gpu
  config!: ModelConfig
  readonly contextLength: number
  readonly batchSize: number
  private precision: "highp" | "mediump"
  private bytes: boolean
  private layers: Array<GpuLayer> = []
  private embedding!: GpuMatrix
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
    this.batchSize = options.batchSize ?? 128
    this.precision = options.matmulPrecision ?? "highp"
    this.bytes = (options.weightTexels ?? "bytes") === "bytes"
    for (const name of options.skip ?? []) this.gpu.skip.add(name)
  }

  /** Uploads the model, piece by piece, as the loader yields it. */
  async load(
    pieces: AsyncIterable<WeightPiece>,
    onProgress?: (piece: WeightPiece) => void
  ) {
    for await (const piece of pieces) {
      if (piece.kind === "config") this.configure(piece.config)
      else if (piece.kind === "embedding")
        this.embedding = this.matrix(piece.value)
      else if (piece.kind === "finalNorm")
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
    this.buffers = this.allocate()
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

  private allocate() {
    const c = this.config
    const M = this.batchSize
    const D = c.hiddenSize
    const qkvg = (c.numHeads + 2 * c.numKvHeads) * c.headDim + c.numHeads
    const scoreTexels = Math.ceil(this.contextLength / 4)
    return {
      ids: this.gpu.texture(GL.R32UI, 1, M),
      residual: [this.f32(D, M), this.f32(D, M)] as [Texture, Texture],
      rms: this.gpu.texture(GL.RGBA32F, 1, M),
      normed: this.h2(D, M),
      opOut: this.f32(D, M),
      bcx: this.f32(3 * D, M),
      convOut: this.h2(D, M),
      qkvg: this.f32(qkvg, M),
      q: this.f32(c.numHeads * c.headDim, M),
      scores: this.gpu.texture(GL.RGBA32F, scoreTexels, c.numHeads * M),
      attnOut: this.h2(c.numHeads * c.headDim, M),
      gateUp: this.f32(2 * c.intermediateSize, M),
      act: this.h2(c.intermediateSize, M),
      ffnOut: this.f32(D, M),
      lastNormed: this.h2(D, 1),
      logits: this.gpu.texture(
        GL.RGBA32F,
        OUT_WIDTH,
        Math.ceil(c.vocabSize / 4 / OUT_WIDTH)
      ),
      /** Per token: 0, 1 seen (repetition penalty), 2 never drawn. Same rows as the logits. */
      mask: this.gpu.texture(
        GL.R8UI,
        OUT_WIDTH * 4,
        Math.ceil(c.vocabSize / 4 / OUT_WIDTH)
      ),
      chunkMax: this.gpu.texture(GL.RGBA32F, Math.ceil(c.vocabSize / CHUNK), 1),
      chunkRank: this.gpu.texture(
        GL.RGBA32F,
        Math.ceil(c.vocabSize / CHUNK),
        1
      ),
      candidates: this.gpu.texture(GL.RGBA32F, CHUNK / 4 + 1, MAX_CHUNKS),
      debug: this.f32(Math.max(2 * c.intermediateSize, qkvg), M),
      /** Split matmuls' partial sums: widest output × (rows × slices). */
      partial: this.gpu.texture(
        GL.RGBA32F,
        Math.max(2 * c.intermediateSize, qkvg, 3 * D) / 4,
        SPLIT_ROWS * MAX_SPLIT
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

  /** A quantized matrix as textures, in columns if it has more row groups than fit. */
  private matrix(m: QuantMatrix): GpuMatrix {
    const max = this.gpu.limits.maxTextureSize
    const blocks = m.K / 32
    const groups = m.N / 4
    // A block row is 16 bytes (4-bit) or 32 (8-bit): 1 or 2 RGBA32UI texels,
    // or 4 or 8 byte texels.
    const texelsPerBlockRow = (m.bits === 4 ? 1 : 2) * (this.bytes ? 4 : 1)
    const rowTexels = blocks * 4 * texelsPerBlockRow
    const quantFormat = !this.bytes
      ? GL.RGBA32UI
      : m.bits === 4
        ? GL.RGBA8UI
        : GL.RGBA8I
    const columns = Math.ceil(groups / max)
    const rowsPerColumn = Math.ceil(groups / columns)
    if (columns * rowTexels > max)
      throw new GpuError(
        `A ${m.N}×${m.K} matrix doesn't fit in this GPU's textures.`
      )
    const quants = this.gpu.texture(
      quantFormat,
      columns * rowTexels,
      rowsPerColumn
    )
    const scales = this.gpu.texture(GL.RGBA16F, columns * blocks, rowsPerColumn)
    const zeros = m.zeros
      ? this.gpu.texture(GL.RGBA8UI, columns * blocks, rowsPerColumn)
      : null
    for (let c = 0; c < columns; c++) {
      const g0 = c * rowsPerColumn
      const rows = Math.min(rowsPerColumn, groups - g0)
      const qWords = (rowTexels * 4) / (this.bytes ? 4 : 1)
      const words = m.quants.subarray(g0 * qWords, (g0 + rows) * qWords)
      this.gpu.write(
        quants,
        { x: c * rowTexels, y: 0, width: rowTexels, height: rows },
        {
          format: GL.RGBA_INTEGER,
          type: !this.bytes
            ? GL.UNSIGNED_INT
            : m.bits === 4
              ? GL.UNSIGNED_BYTE
              : GL.BYTE,
          pixels: !this.bytes
            ? words
            : m.bits === 4
              ? new Uint8Array(words.buffer, words.byteOffset, words.byteLength)
              : new Int8Array(words.buffer, words.byteOffset, words.byteLength)
        }
      )
      this.gpu.write(
        scales,
        { x: c * blocks, y: 0, width: blocks, height: rows },
        {
          format: GL.RGBA,
          type: GL.HALF_FLOAT,
          pixels: m.scales.subarray(g0 * blocks * 4, (g0 + rows) * blocks * 4)
        }
      )
      if (zeros && m.zeros) {
        this.gpu.write(
          zeros,
          { x: c * blocks, y: 0, width: blocks, height: rows },
          {
            format: GL.RGBA_INTEGER,
            type: GL.UNSIGNED_BYTE,
            pixels: m.zeros.subarray(g0 * blocks * 4, (g0 + rows) * blocks * 4)
          }
        )
      }
    }
    return {
      N: m.N,
      K: m.K,
      bits: m.bits,
      blocks,
      groups,
      rowsPerColumn,
      rowTexels,
      quants,
      scales,
      zeros
    }
  }

  private layer(w: LayerWeights): GpuLayer {
    const c = this.config
    const layer: GpuLayer = {
      type: w.type,
      operatorNorm: this.vector(w.operatorNorm),
      postOperatorNorm: this.vector(w.postOperatorNorm),
      ffnNorm: this.vector(w.ffnNorm),
      postFfnNorm: this.vector(w.postFfnNorm),
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
        state: [this.f32(D, width - 1), this.f32(D, width - 1)]
      }
    }
    if (w.attention) {
      const kvDims = c.numKvHeads * c.headDim
      layer.attention = {
        qkvg: this.matrix(w.attention.qkvg),
        o: this.matrix(w.attention.o),
        qNorm: this.vector(w.attention.qNorm),
        kNorm: this.vector(w.attention.kNorm),
        keys: this.h2(kvDims, this.contextLength),
        values: this.h2(kvDims, this.contextLength)
      }
    }
    return layer
  }

  /** Convolution states saved at cache positions, to resume from later. */
  private snapshots = new Map<number, Array<Texture>>()

  /**
   * Saves the convolutions' state at the current position, so a later
   * prompt that shares the tokens so far can resume here: the attention
   * cache can be rewound to any position, the convolutions' state can't.
   */
  snapshot() {
    if (this.snapshots.has(this.position)) return
    const saved: Array<Texture> = []
    for (const layer of this.layers) {
      if (!layer.conv) continue
      const state = layer.conv.state[0]
      const copy = this.gpu.texture(GL.RGBA32F, state.width, state.height)
      this.gpu.run({
        program: this.gpu.program(K.COPY),
        inputs: { uX: state },
        output: copy
      })
      saved.push(copy)
    }
    this.snapshots.set(this.position, saved)
  }

  /**
   * Rewinds the cache to the latest snapshot at or before `position`, and
   * returns where it is now: tokens before it are still in the cache.
   */
  rewind(position: number): number {
    const at = Math.max(
      0,
      ...[...this.snapshots.keys()].filter((p) => p <= position)
    )
    for (const [p, saved] of this.snapshots) {
      if (p <= at) continue
      saved.forEach((t) => this.gpu.delete(t))
      this.snapshots.delete(p)
    }
    const saved = this.snapshots.get(at)
    if (!saved) {
      this.reset()
      return 0
    }
    let i = 0
    for (const layer of this.layers) {
      if (!layer.conv) continue
      this.gpu.run({
        program: this.gpu.program(K.COPY),
        inputs: { uX: saved[i++] },
        output: layer.conv.state[0]
      })
    }
    this.position = at
    return at
  }

  /** Forgets the cache: the next forward starts a new sequence. */
  reset() {
    this.position = 0
    for (const saved of this.snapshots.values())
      saved.forEach((t) => this.gpu.delete(t))
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

  private matmul(a: Texture, w: GpuMatrix, out: Texture, rows: number) {
    // A few rows: split K so the GPU has enough fragments to keep busy.
    const split =
      rows <= SPLIT_ROWS
        ? Math.max(
            1,
            Math.min(
              MAX_SPLIT,
              Math.ceil(SPLIT_FRAGMENTS / (w.groups * rows)),
              w.blocks / 8
            )
          )
        : 1
    if (split > 1) {
      const kbSlice = Math.ceil(w.blocks / split)
      const slices = Math.ceil(w.blocks / kbSlice)
      this.gpu.run({
        name: "matmul",
        program: this.gpu.program(
          K.matmul({
            bits: w.bits,
            zeros: !!w.zeros,
            wrap: false,
            precision: this.precision,
            bytes: this.bytes,
            split: true
          })
        ),
        inputs: {
          uA: a,
          uW: w.quants,
          uS: w.scales,
          ...(w.zeros && { uZ: w.zeros })
        },
        uniforms: {
          uKb: { int: w.blocks },
          uH: { int: w.rowsPerColumn },
          uRowTexels: { int: w.rowTexels },
          uG: { int: w.groups },
          uSplit: { int: slices },
          uKbSlice: { int: kbSlice }
        },
        output: this.buffers.partial,
        rect: { x: 0, y: 0, width: w.groups, height: rows * slices }
      })
      this.gpu.run({
        name: "matmul",
        program: this.gpu.program(K.REDUCE_SLICES),
        inputs: { uP: this.buffers.partial },
        uniforms: { uSplit: { int: slices } },
        output: out,
        rect: { x: 0, y: 0, width: w.groups, height: rows }
      })
      return
    }
    const program = this.gpu.program(
      K.matmul({
        bits: w.bits,
        zeros: !!w.zeros,
        wrap: false,
        precision: this.precision,
        bytes: this.bytes
      })
    )
    this.gpu.run({
      name: "matmul",
      program,
      inputs: {
        uA: a,
        uW: w.quants,
        uS: w.scales,
        ...(w.zeros && { uZ: w.zeros })
      },
      uniforms: {
        uKb: { int: w.blocks },
        uH: { int: w.rowsPerColumn },
        uRowTexels: { int: w.rowTexels },
        uG: { int: w.groups }
      },
      output: out,
      rect: { x: 0, y: 0, width: w.groups, height: rows }
    })
  }

  private rms(x: Texture, out: Texture, rows: number, texels: number) {
    this.gpu.run({
      name: "rms",
      program: this.gpu.program(K.RMS),
      inputs: { uX: x },
      uniforms: { uTexels: { int: texels }, uEps: this.config.normEps },
      output: out,
      rect: { x: 0, y: 0, width: 1, height: rows }
    })
  }

  private normToH2(
    x: Texture,
    norm: Texture,
    out: Texture,
    rows: number,
    rowIn = -1
  ) {
    const { hiddenSize } = this.config
    if (rows <= FUSED_NORM_ROWS) {
      // A few rows: each fragment sums the row itself, one draw instead of two.
      this.gpu.run({
        name: "norm",
        program: this.gpu.program(K.NORM_TO_H2_FUSED),
        inputs: { uX: x, uNorm: norm },
        uniforms: {
          uRowIn: { int: rowIn },
          uTexels: { int: hiddenSize / 4 },
          uEps: this.config.normEps
        },
        output: out,
        rect: { x: 0, y: 0, width: hiddenSize / 8, height: rows }
      })
      return
    }
    this.rms(x, this.buffers.rms, rowIn < 0 ? rows : rowIn + 1, hiddenSize / 4)
    this.gpu.run({
      name: "norm",
      program: this.gpu.program(K.NORM_TO_H2),
      inputs: { uX: x, uR: this.buffers.rms, uNorm: norm },
      uniforms: { uRowIn: { int: rowIn } },
      output: out,
      rect: { x: 0, y: 0, width: hiddenSize / 8, height: rows }
    })
  }

  /** out = h + post-norm(y): the sandwich norm and the residual add. */
  private addNormed(
    h: Texture,
    y: Texture,
    norm: Texture,
    out: Texture,
    rows: number
  ) {
    const { hiddenSize } = this.config
    if (rows <= FUSED_NORM_ROWS) {
      this.gpu.run({
        name: "add",
        program: this.gpu.program(K.ADD_NORMED_FUSED),
        inputs: { uH: h, uY: y, uNorm: norm },
        uniforms: {
          uTexels: { int: hiddenSize / 4 },
          uEps: this.config.normEps
        },
        output: out,
        rect: { x: 0, y: 0, width: hiddenSize / 4, height: rows }
      })
      return
    }
    this.rms(y, this.buffers.rms, rows, hiddenSize / 4)
    this.gpu.run({
      name: "add",
      program: this.gpu.program(K.ADD_NORMED),
      inputs: { uH: h, uY: y, uR: this.buffers.rms, uNorm: norm },
      output: out,
      rect: { x: 0, y: 0, width: hiddenSize / 4, height: rows }
    })
  }

  private convolve(layer: NonNullable<GpuLayer["conv"]>, rows: number) {
    const c = this.config
    const b = this.buffers
    const dt = c.hiddenSize / 4
    this.matmul(b.normed, layer.inProj, b.bcx, rows)
    this.gpu.run({
      name: "conv",
      program: this.gpu.program(K.shortConv(c.convKernel)),
      inputs: { uBCx: b.bcx, uState: layer.state[0], uConv: layer.kernel },
      uniforms: { uDt: { int: dt } },
      output: b.convOut,
      rect: { x: 0, y: 0, width: c.hiddenSize / 8, height: rows }
    })
    this.gpu.run({
      name: "conv",
      program: this.gpu.program(K.convState(c.convKernel)),
      inputs: { uBCx: b.bcx, uState: layer.state[0] },
      uniforms: { uDt: { int: dt }, uM: { int: rows } },
      output: layer.state[1],
      rect: { x: 0, y: 0, width: dt, height: c.convKernel - 1 }
    })
    layer.state.reverse()
    this.matmul(b.convOut, layer.outProj, b.opOut, rows)
  }

  private attend(
    layer: NonNullable<GpuLayer["attention"]>,
    type: LayerType,
    rows: number,
    pos0: number
  ) {
    const c = this.config
    const b = this.buffers
    const sliding = type === "sliding_attention"
    const rope = sliding ? this.ropeSliding : this.ropeFull
    const pairTexels = (sliding ? c.ropeDimsSliding : c.ropeDimsFull) / 8
    const qDims = c.numHeads * c.headDim
    const kvDims = c.numKvHeads * c.headDim
    this.matmul(b.normed, layer.qkvg, b.qkvg, rows)
    // Queries: normalized, rotated, scaled by 1/sqrt(head dim).
    this.gpu.run({
      name: "heads",
      program: this.gpu.program(
        K.headPrep({ out: "f32", rope: true, norm: true })
      ),
      inputs: { uQKVG: b.qkvg, uNorm: layer.qNorm, uRope: rope },
      uniforms: {
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
    // Keys and values into the cache at positions pos0...
    this.gpu.run({
      name: "heads",
      program: this.gpu.program(
        K.headPrep({ out: "h2", rope: true, norm: true })
      ),
      inputs: { uQKVG: b.qkvg, uNorm: layer.kNorm, uRope: rope },
      uniforms: {
        uBase: { int: qDims / 4 },
        uPairTexels: { int: pairTexels },
        uPos0: { int: pos0 },
        uRowOffset: { int: pos0 },
        uEps: c.normEps,
        uScale: 1
      },
      output: layer.keys,
      rect: { x: 0, y: pos0, width: kvDims / 8, height: rows }
    })
    this.gpu.run({
      name: "heads",
      program: this.gpu.program(K.VALUES),
      inputs: { uQKVG: b.qkvg },
      uniforms: {
        uBase: { int: (qDims + kvDims) / 4 },
        uRowOffset: { int: pos0 }
      },
      output: layer.values,
      rect: { x: 0, y: pos0, width: kvDims / 8, height: rows }
    })
    const j0 = sliding ? Math.max(0, pos0 - c.slidingWindow + 1) : 0
    const keys = pos0 + rows - j0
    const group = c.numHeads / c.numKvHeads
    this.gpu.run({
      name: "scores",
      program: this.gpu.program(K.SCORES),
      inputs: { uQ: b.q, uK: layer.keys },
      uniforms: {
        uM: { int: rows },
        uJ0: { int: j0 },
        uKeys: { int: keys },
        uPos0: { int: pos0 },
        uWindow: { int: sliding ? c.slidingWindow : 0 },
        uGroup: { int: group }
      },
      output: b.scores,
      rect: {
        x: 0,
        y: 0,
        width: Math.ceil(keys / 4),
        height: c.numHeads * rows
      }
    })
    this.gpu.run({
      name: "attend",
      program: this.gpu.program(K.ATTEND),
      inputs: { uS: b.scores, uV: layer.values, uQKVG: b.qkvg },
      uniforms: {
        uM: { int: rows },
        uJ0: { int: j0 },
        uKeys: { int: keys },
        uGroup: { int: group },
        uGateTexel: { int: (qDims + 2 * kvDims) / 4 }
      },
      output: b.attnOut,
      rect: { x: 0, y: 0, width: qDims / 8, height: rows }
    })
    this.matmul(b.attnOut, layer.o, b.opOut, rows)
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
    for (let start = 0; start < tokens.length; start += this.batchSize) {
      const batch = Array.from(
        { length: Math.min(this.batchSize, tokens.length - start) },
        (_, i) => tokens[start + i]
      )
      last = await this.forwardBatch(batch, probe)
    }
    return this.logits(last)
  }

  /** Reads tokens into the cache without computing logits (for timing). */
  async forwardNoLogits(tokens: ArrayLike<number>) {
    await this.forwardBatch(Array.from(tokens), undefined)
  }

  /**
   * Reads tokens into the cache like `forward`, without reading logits
   * back; returns the row of the last token, for `candidates`.
   */
  async forwardHidden(tokens: ArrayLike<number>): Promise<number> {
    if (this.position + tokens.length > this.contextLength) {
      throw new Error(
        `The cache holds ${this.contextLength} tokens; this would need ${this.position + tokens.length}.`
      )
    }
    let last = 0
    for (let start = 0; start < tokens.length; start += this.batchSize) {
      const batch = Array.from(
        { length: Math.min(this.batchSize, tokens.length - start) },
        (_, i) => tokens[start + i]
      )
      last = await this.forwardBatch(batch, undefined)
    }
    return last
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
    penalty: number
  ): Promise<{ ids: Int32Array; logits: Float32Array }> {
    const c = this.config
    const b = this.buffers
    this.head(row)
    const total = Math.ceil(c.vocabSize / CHUNK)
    const count = Math.min(chunks, MAX_CHUNKS, total)
    this.gpu.run({
      name: "pick",
      program: this.gpu.program(K.CHUNK_MAX),
      inputs: { uLogits: b.logits, uMask: b.mask },
      uniforms: { uOutW: { int: OUT_WIDTH }, uPenalty: penalty },
      output: b.chunkMax,
      rect: { x: 0, y: 0, width: total, height: 1 }
    })
    this.gpu.run({
      name: "pick",
      program: this.gpu.program(K.CHUNK_RANK),
      inputs: { uMax: b.chunkMax },
      uniforms: { uChunks: { int: total } },
      output: b.chunkRank,
      rect: { x: 0, y: 0, width: total, height: 1 }
    })
    this.gpu.run({
      name: "pick",
      program: this.gpu.program(K.CHUNK_GATHER),
      inputs: { uLogits: b.logits, uMask: b.mask, uRank: b.chunkRank },
      uniforms: {
        uOutW: { int: OUT_WIDTH },
        uChunks: { int: total },
        uPenalty: penalty
      },
      output: b.candidates,
      rect: { x: 0, y: 0, width: CHUNK / 4 + 1, height: count }
    })
    const texels = CHUNK / 4 + 1
    const data = await this.gpu.read(b.candidates, {
      x: 0,
      y: 0,
      width: texels,
      height: count
    })
    const ids = new Int32Array(count * CHUNK)
    const logits = new Float32Array(count * CHUNK)
    for (let r = 0; r < count; r++) {
      const chunk = Math.round(data[(r * texels + CHUNK / 4) * 4])
      for (let i = 0; i < CHUNK; i++) {
        ids[r * CHUNK + i] = chunk * CHUNK + i
        logits[r * CHUNK + i] = data[r * texels * 4 + i]
      }
    }
    return { ids, logits }
  }

  /** The output head on one row of the last batch: logits into the logits texture. */
  private head(row: number) {
    const b = this.buffers
    const e = this.embedding
    this.normToH2(b.residual[0], this.finalNorm, b.lastNormed, 1, row)
    this.gpu.run({
      name: "head",
      program: this.gpu.program(
        K.matmul({
          bits: e.bits,
          zeros: !!e.zeros,
          wrap: true,
          precision: this.precision,
          bytes: this.bytes
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

  /** Logits for one row of the last batch's final residual. */
  async logits(row: number): Promise<Float32Array> {
    const c = this.config
    const b = this.buffers
    this.head(row)
    const all = await this.gpu.read(b.logits)
    return all.subarray(0, c.vocabSize)
  }

  private async forwardBatch(
    batch: Array<number>,
    probe: LayerProbe | undefined
  ) {
    const c = this.config
    const b = this.buffers
    const rows = batch.length
    const pos0 = this.position
    this.gpu.write(
      b.ids,
      { x: 0, y: 0, width: 1, height: rows },
      {
        format: GL.RED_INTEGER,
        type: GL.UNSIGNED_INT,
        pixels: new Uint32Array(batch)
      }
    )
    const e = this.embedding
    this.gpu.run({
      program: this.gpu.program(
        K.embed({ bits: e.bits, zeros: !!e.zeros, bytes: this.bytes })
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
      h2 = false
    ) => {
      if (!probe) return
      await probe(layer, name, async () => {
        let src = tex
        if (h2) {
          this.gpu.run({
            program: this.gpu.program(K.H2_TO_F32),
            inputs: { uX: tex },
            output: b.debug,
            rect: { x: 0, y: 0, width: dims / 4, height: rows }
          })
          src = b.debug
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
      this.normToH2(h, layer.operatorNorm, b.normed, rows)
      await look(i, "operator_norm", b.normed, c.hiddenSize, true)
      if (layer.conv) {
        this.convolve(layer.conv, rows)
        await look(i, "in_proj", b.bcx, 3 * c.hiddenSize)
        await look(i, "conv", b.convOut, c.hiddenSize, true)
      } else if (layer.attention) {
        this.attend(layer.attention, layer.type, rows, pos0)
        const qkvgDims =
          (c.numHeads + 2 * c.numKvHeads) * c.headDim + c.numHeads
        await look(i, "qkvg", b.qkvg, qkvgDims)
        await look(i, "q", b.q, c.numHeads * c.headDim)
        await look(i, "attn_out", b.attnOut, c.numHeads * c.headDim, true)
      }
      await look(i, "op_out", b.opOut, c.hiddenSize)
      this.addNormed(h, b.opOut, layer.postOperatorNorm, next, rows)
      await look(i, "after_op", next, c.hiddenSize)
      this.normToH2(next, layer.ffnNorm, b.normed, rows)
      this.matmul(b.normed, layer.gateUp, b.gateUp, rows)
      this.gpu.run({
        name: "swiglu",
        program: this.gpu.program(K.SWIGLU),
        inputs: { uGU: b.gateUp },
        uniforms: { uHalf: { int: c.intermediateSize / 4 } },
        output: b.act,
        rect: { x: 0, y: 0, width: c.intermediateSize / 8, height: rows }
      })
      this.matmul(b.act, layer.down, b.ffnOut, rows)
      await look(i, "ffn_out", b.ffnOut, c.hiddenSize)
      this.addNormed(next, b.ffnOut, layer.postFfnNorm, h, rows)
      await look(i, "l_out", h, c.hiddenSize)
    }
    this.position += rows
    return rows - 1
  }
}
