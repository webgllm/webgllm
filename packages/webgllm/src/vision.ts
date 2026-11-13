/**
 * The vision encoder on WebGL 2 (see docs/vision.md): uploads the pieces a
 * loader yields (load-vision.ts, load-vision-gguf.ts), and turns an image
 * view's patches into the text model's embedding rows, one row per merged
 * token. It shares the engine's GPU context, and is loaded only while a
 * request has images to read.
 */
import { GL, type Gpu, type Pass, type Texture } from "./gpu/gl.js"
import * as V from "./gpu/vision-kernels.js"
import { patchPlace, type ImagePatches } from "./image.js"
import {
  checkVisionConfig,
  visionBlocks,
  type VisionBlock,
  type VisionConfig,
  type VisionLinear,
  type VisionMatrix,
  type VisionNorm,
  type VisionPiece
} from "./vision-weights.js"

type GpuLinear = { N: number; K: number; bits: 4 | 8; quants: Texture; scales: Texture; zeros: Texture | null; bias: Texture }
type GpuNorm = { weight: Texture; bias: Texture | null }
type GpuBlock =
  | { type: "attention"; norm1: GpuNorm; qkv: GpuLinear; qNorm: Texture | null; kNorm: Texture | null; proj: GpuLinear; norm2: GpuNorm; fc1: GpuLinear; fc2: GpuLinear }
  | { type: "conv"; norm1: GpuNorm; inProj: GpuLinear; kernel: Texture; outProj: GpuLinear; norm2: GpuNorm; fc1: GpuLinear; fc2: GpuLinear }

/** Score bytes a chunk of query rows may take (see attend). */
const SCORE_BYTES = 32 << 20

/** Query rows per attention chunk: as many as fit SCORE_BYTES and a texture's height. */
function chunkRows(config: VisionConfig, count: number, maxTextureSize: number) {
  const keyTexels = Math.ceil(count / 4)
  return Math.max(1, Math.min(count, Math.floor(maxTextureSize / config.heads), Math.floor(SCORE_BYTES / (keyTexels * 16 * config.heads))))
}

/** Bytes of an f32 tensor of `dims` features by `rows` (RGBA32F, 4 a texel). */
const f32Bytes = (dims: number, rows: number) => (dims / 4) * rows * 16

/** Texels of a tensor a projection reads `K` values of: whole blocks of 32, so a short last block reads zeros. */
const readTexels = (K: number) => visionBlocks(K) * 8

/** The widths of an attention block's fused q | k | v projection, and of q | k. */
const attentionRows = (c: VisionConfig) => {
  const kv = c.kvHeads * c.headDim
  return { q: c.width, kv, qkv: c.width + 2 * kv, qk: c.width + kv }
}

/** The rows of a block's first MLP projection: SwiGLU's holds its gate and its up rows. */
const fc1Rows = (c: VisionConfig) => (c.mlp === "swiglu" ? 2 * c.ffn : c.ffn)

/**
 * The encoder's weight textures, in bytes, as VisionEncoder.load allocates
 * them: each projection's weights (RGBA32UI: one texel per row and block
 * for 4-bit weights, two for 8-bit), scales (RGBA32F, a texel per 4 rows
 * and block), zero points (RGBA8UI, laid out the same: 4-bit weights
 * always have them, 8-bit ones where `zeros` says the file does) and bias,
 * and the norms. `bits` and `zeros` are the projections' (a file has them
 * all alike).
 */
export function visionWeightBytes(c: VisionConfig, bits: 4 | 8 = 4, zeros = bits === 4): number {
  const linear = (N: number, K: number) => visionBlocks(K) * (N * 16 * (bits / 4) + (N / 4) * 16 + (zeros || bits === 4 ? (N / 4) * 4 : 0)) + f32Bytes(N, 1)
  const vector = f32Bytes(c.width, 1)
  const norm = c.norm === "layer" ? 2 * vector : vector
  const W = c.width
  const mlp = linear(fc1Rows(c), W) + linear(W, c.ffn)
  const attention = linear(attentionRows(c).qkv, W) + linear(W, W) + (c.qkNorm ? 2 * f32Bytes(c.headDim, 1) : 0)
  const conv = linear(3 * W, W) + linear(W, W) + 9 * vector
  const blocks = c.layers.reduce((sum, type) => sum + 2 * norm + mlp + (type === "conv" ? conv : attention), 0)
  // The patch projection (with an RMS norm of its own in an RMS-norm tower), and the merger behind a LayerNorm.
  const patch = linear(W, c.patchValues) + (c.norm === "rms" ? vector : 0)
  return patch + blocks + 2 * vector + linear(c.mergerWidth, c.merge * c.merge * W) + linear(c.outWidth, c.mergerWidth)
}

/**
 * The most bytes of work textures encoding a gridH × gridW view takes at
 * once, as VisionEncoder.encode allocates them: the patch projection's
 * inputs (freed after it), then the blocks' tensors, attention's scores and
 * the merger's outputs.
 */
export function visionWorkBytes(c: VisionConfig, gridH: number, gridW: number, maxTextureSize: number): number {
  const count = gridH * gridW
  const tokens = count / (c.merge * c.merge)
  const W = c.width
  const { qkv, qk } = attentionRows(c)
  const attends = c.layers.includes("attention")
  const convolves = c.layers.includes("conv")
  const texels = (n: number, rows: number) => n * rows * 16
  const kept = (c.rope ? f32Bytes(c.headDim, count) : 0) + 2 * f32Bytes(W, count)
  const projection = f32Bytes(c.patchValues, count) + (c.grid ? f32Bytes(W, count) : 0)
  const rows = chunkRows(c, count, maxTextureSize)
  const attention = attends ? texels(Math.ceil(count / 4), rows * c.heads) + texels(1, rows * c.heads) : 0
  const swiglu = c.mlp === "swiglu"
  const blocks =
    f32Bytes(W, count) +
    f32Bytes(Math.max(attends ? qkv : 0, convolves ? 3 * W : 0), count) +
    (attends ? f32Bytes(qk, count) : 0) +
    f32Bytes(W, count) +
    texels(swiglu ? fc1Rows(c) / 4 : readTexels(c.ffn), count) +
    (swiglu ? texels(readTexels(c.ffn), count) : 0)
  const merger = texels(readTexels(c.mergerWidth), tokens) + f32Bytes(c.outWidth, tokens)
  return kept + Math.max(projection, blocks + Math.max(attention, merger))
}

/** A resampling axis: for each of `to` outputs, the inputs it reads and their weights. */
function axisWeights(from: number, to: number, mode: VisionConfig["interpolation"]): Array<Array<[number, number]>> {
  const out: Array<Array<[number, number]>> = []
  for (let i = 0; i < to; i++) {
    if (mode === "corners") {
      const at = (i * (from - 1)) / Math.max(to - 1, 1)
      const lo = Math.floor(at)
      const hi = Math.min(lo + 1, from - 1)
      out.push([
        [lo, 1 - (at - lo)],
        [hi, at - lo]
      ])
      continue
    }
    // Pixel centers; a triangle filter as wide as the step where the grid is smaller (antialias).
    const scale = from / to
    const support = mode === "antialias" ? Math.max(scale, 1) : 1
    const center = (i + 0.5) * scale
    const lo = Math.max(0, Math.floor(center - support + 0.5))
    const hi = Math.min(from, Math.floor(center + support + 0.5))
    const taps: Array<[number, number]> = []
    let sum = 0
    for (let j = lo; j < hi; j++) {
      const weight = Math.max(0, 1 - Math.abs((j + 0.5 - center) / support))
      taps.push([j, weight])
      sum += weight
    }
    if (!(sum > 0)) taps.splice(0, taps.length, [Math.min(from - 1, Math.max(0, Math.floor(center))), 1]), (sum = 1)
    out.push(taps.map(([j, weight]) => [j, weight / sum]))
  }
  return out
}

/**
 * The patches' position embeddings: the grid × grid table interpolated to
 * the image's patch grid as the encoder does it (see
 * VisionConfig.interpolation), rows in merge order.
 */
export function positionEmbeddings(
  table: Float32Array,
  grid: number,
  width: number,
  gridH: number,
  gridW: number,
  merge: number,
  mode: VisionConfig["interpolation"] = "corners"
): Float32Array {
  const count = gridH * gridW
  const out = new Float32Array(count * width)
  const rows = axisWeights(grid, gridH, mode)
  const cols = axisWeights(grid, gridW, mode)
  for (let p = 0; p < count; p++) {
    const { row, col } = patchPlace(p, gridW, merge)
    const o = p * width
    for (const [h, hw] of rows[row]) {
      for (const [w, ww] of cols[col]) {
        const weight = hw * ww
        if (!weight) continue
        const r = (h * grid + w) * width
        for (let e = 0; e < width; e++) out[o + e] += weight * table[r + e]
      }
    }
  }
  return out
}

/**
 * Each patch's RoPE cos and sin: with n frequencies (headDim / 4),
 * frequency j < n by its row, n..2n − 1 by its column (headDim / 4 texels
 * a patch: half of cos, half of sin).
 */
export function ropeTable(invFreq: Float32Array, gridH: number, gridW: number, merge: number): Float32Array {
  const n = invFreq.length
  const count = gridH * gridW
  const out = new Float32Array(count * 4 * n)
  for (let p = 0; p < count; p++) {
    const { row, col } = patchPlace(p, gridW, merge)
    for (let j = 0; j < 2 * n; j++) {
      const angle = j < n ? row * invFreq[j] : col * invFreq[j - n]
      out[p * 4 * n + j] = Math.cos(angle)
      out[p * 4 * n + 2 * n + j] = Math.sin(angle)
    }
  }
  return out
}

export class VisionEncoder {
  config!: VisionConfig
  private patch!: GpuLinear
  private patchNorm: GpuNorm | null = null
  private positions: Float32Array | null = null
  private invFreq: Float32Array | null = null
  private blocks: Array<GpuBlock> = []
  private merger!: { norm: GpuNorm; linear1: GpuLinear; linear2: GpuLinear }
  private weights: Array<Texture> = []

  constructor(readonly gpu: Gpu) {}

  /** The most patches one view may have here: a texture's height. */
  get maxPatches() {
    return this.gpu.limits.maxTextureSize
  }

  async load(pieces: AsyncIterable<VisionPiece>, signal?: AbortSignal) {
    for await (const piece of pieces) {
      signal?.throwIfAborted()
      if (piece.kind === "config") this.config = checkVisionConfig(piece.config)
      else if (piece.kind === "patch") {
        this.patch = this.linear(piece.patch)
        this.positions = piece.positions ?? null
        this.invFreq = piece.invFreq ?? null
        this.patchNorm = piece.norm ? this.norm(piece.norm) : null
      } else if (piece.kind === "block") this.blocks[piece.index] = this.block(piece.value)
      else this.merger = { norm: this.norm(piece.norm), linear1: this.linear(piece.linear1), linear2: this.linear(piece.linear2) }
      await this.gpu.pace(1, true)
    }
    const c = this.config
    if (!c || !this.patch || !this.merger || this.blocks.length !== c.layers.length || this.blocks.some((b, i) => b?.type !== c.layers[i]))
      throw new Error("The vision encoder's weights are incomplete.")
    if (c.grid && this.positions?.length !== c.grid * c.grid * c.width) throw new Error("The vision encoder's position table is missing.")
    if (c.rope && this.invFreq?.length !== c.headDim / 4) throw new Error("The vision encoder's RoPE frequencies are missing.")
    const { qkv } = attentionRows(c)
    const shaped = (l: GpuLinear, N: number, K: number) => l.N === N && l.K === K
    const per = c.merge * c.merge
    const ok =
      shaped(this.patch, c.width, c.patchValues) &&
      this.merger.linear1.K === per * c.width &&
      shaped(this.merger.linear2, c.outWidth, this.merger.linear1.N) &&
      this.blocks.every(
        (b) =>
          shaped(b.fc1, fc1Rows(c), c.width) &&
          shaped(b.fc2, c.width, c.ffn) &&
          (b.type === "attention"
            ? shaped(b.qkv, qkv, c.width) && shaped(b.proj, c.width, c.width) && !!b.qNorm === c.qkNorm && !!b.kNorm === c.qkNorm
            : shaped(b.inProj, 3 * c.width, c.width) && shaped(b.outProj, c.width, c.width))
      )
    if (!ok) throw new Error("The vision encoder's weights don't have its config's shapes.")
  }

  /** GPU bytes of the weights. */
  get bytes() {
    return this.weights.reduce((sum, t) => sum + t.bytes, 0)
  }

  private keep(t: Texture) {
    this.weights.push(t)
    return t
  }

  private vector(values: Float32Array) {
    return this.keep(
      this.gpu.texture(GL.RGBA32F, values.length / 4, 1, { format: GL.RGBA, type: GL.FLOAT, pixels: values })
    )
  }

  private norm(n: VisionNorm): GpuNorm {
    return { weight: this.vector(n.weight), bias: n.bias ? this.vector(n.bias) : null }
  }

  private block(b: VisionBlock): GpuBlock {
    const mlp = { norm2: this.norm(b.norm2), fc1: this.linear(b.fc1), fc2: this.linear(b.fc2) }
    if (b.type === "conv") {
      // Taps as rows: texel (x, tap) holds channels 4x..4x+3.
      const W = this.config.width
      if (b.kernel.length !== W * 9) throw new Error("The vision encoder's convolution isn't 3 × 3 depthwise.")
      const taps = new Float32Array(9 * W)
      for (let ch = 0; ch < W; ch++) for (let t = 0; t < 9; t++) taps[t * W + ch] = b.kernel[ch * 9 + t]
      const kernel = this.keep(this.gpu.texture(GL.RGBA32F, W / 4, 9, { format: GL.RGBA, type: GL.FLOAT, pixels: taps }))
      return { type: "conv", norm1: this.norm(b.norm1), inProj: this.linear(b.inProj), kernel, outProj: this.linear(b.outProj), ...mlp }
    }
    return {
      type: "attention",
      norm1: this.norm(b.norm1),
      qkv: this.linear(b.qkv),
      qNorm: b.qNorm ? this.vector(b.qNorm) : null,
      kNorm: b.kNorm ? this.vector(b.kNorm) : null,
      proj: this.linear(b.proj),
      ...mlp
    }
  }

  private linear({ m, bias }: VisionLinear): GpuLinear {
    const blocks = visionBlocks(m.K)
    const words = new Uint32Array(m.data.buffer, m.data.byteOffset, m.data.byteLength / 4)
    return {
      N: m.N,
      K: m.K,
      bits: m.bits,
      quants: this.keep(
        this.gpu.texture(GL.RGBA32UI, blocks * (m.bits / 4), m.N, { format: GL.RGBA_INTEGER, type: GL.UNSIGNED_INT, pixels: words })
      ),
      scales: this.keep(this.gpu.texture(GL.RGBA32F, blocks, m.N / 4, { format: GL.RGBA, type: GL.FLOAT, pixels: m.scales })),
      zeros: this.zeros(m, blocks),
      bias: this.vector(bias ?? new Float32Array(m.N))
    }
  }

  /** A matrix's zero points: 4-bit weights always read them (8 without any); 8-bit ones only where the file has them. */
  private zeros(m: VisionMatrix, blocks: number) {
    const zeros = m.zeros ?? (m.bits === 4 ? new Uint8Array(m.N * blocks).fill(8) : null)
    return zeros && this.keep(this.gpu.texture(GL.RGBA8UI, blocks, m.N / 4, { format: GL.RGBA_INTEGER, type: GL.UNSIGNED_BYTE, pixels: zeros }))
  }

  /** Frees the encoder's textures. */
  dispose() {
    for (const t of this.weights) this.gpu.delete(t)
    this.weights = []
    this.blocks = []
  }

  /** The running encode's abort (see encode). */
  private signal: AbortSignal | undefined

  private async run(pass: Pass) {
    this.signal?.throwIfAborted()
    await this.gpu.runAsync({ name: "vision", ...pass })
  }

  private async matmul(
    a: Texture,
    w: GpuLinear,
    out: Texture,
    rows: number,
    o: { act?: V.VisionAct; residual?: Texture; group?: number } = {}
  ) {
    const group = (o.group ?? 1) > 1
    const blocks = visionBlocks(w.K)
    await this.run({
      program: this.gpu.program(V.visionMatmul({ bits: w.bits, zeros: !!w.zeros, act: o.act ?? "none", residual: !!o.residual, group })),
      inputs: {
        uA: a,
        uW: w.quants,
        uS: w.scales,
        ...(w.zeros && { uZ: w.zeros }),
        uBias: w.bias,
        ...(o.residual && { uR: o.residual })
      },
      uniforms: {
        uBlocks: { int: blocks },
        ...(group && { uInTexels: { int: w.K / 4 / o.group! }, uGroup: { int: o.group! } })
      },
      output: out,
      rect: { x: 0, y: 0, width: w.N / 4, height: rows },
      work: blocks * (w.bits === 4 ? 14 : 18)
    })
  }

  /** A block's norm over each row: LayerNorm or RMS (config.norm). */
  private async blockNorm(x: Texture, n: GpuNorm, out: Texture, rows: number) {
    if (this.config.norm === "rms") return this.rmsNorm(x, n, out, rows)
    return this.layerNorm(x, n, out, rows, this.config.eps)
  }

  private async layerNorm(x: Texture, n: GpuNorm, out: Texture, rows: number, eps: number) {
    if (!n.bias) throw new Error("A LayerNorm of the vision encoder has no bias.")
    const texels = this.config.width / 4
    await this.run({
      program: this.gpu.program(V.VISION_LAYER_NORM),
      inputs: { uX: x, uW: n.weight, uB: n.bias },
      uniforms: { uTexels: { int: texels }, uEps: eps },
      output: out,
      rect: { x: 0, y: 0, width: texels, height: rows },
      work: 2 * texels
    })
  }

  private async rmsNorm(x: Texture, n: GpuNorm, out: Texture, rows: number) {
    const texels = this.config.width / 4
    await this.run({
      program: this.gpu.program(V.VISION_RMS_NORM),
      inputs: { uX: x, uW: n.weight },
      uniforms: { uTexels: { int: texels }, uEps: this.config.eps },
      output: out,
      rect: { x: 0, y: 0, width: texels, height: rows },
      work: texels
    })
  }

  /**
   * Bidirectional attention over all `count` patches (one view): scores,
   * softmax statistics and the weighted values, by chunks of query rows
   * whose scores fit SCORE_BYTES.
   */
  private async attend(qkv: Texture, qk: Texture, out: Texture, count: number) {
    const { heads, kvHeads, headDim, width } = this.config
    const head = headDim / 4
    const rep = heads / kvHeads
    const keyTexels = Math.ceil(count / 4)
    const max = this.gpu.limits.maxTextureSize
    const rows = chunkRows(this.config, count, max)
    const scores = this.gpu.texture(GL.RGBA32F, keyTexels, rows * heads)
    const stats = this.gpu.texture(GL.RGBA32F, 1, rows * heads)
    try {
      for (let q0 = 0; q0 < count; q0 += rows) {
        const n = Math.min(rows, count - q0)
        // Rows of a short last chunk are where a full chunk's would be.
        for (let h = 0; h < heads; h++) {
          await this.run({
            program: this.gpu.program(V.visionScores(head)),
            inputs: { uQK: qk },
            uniforms: { uTexels: { int: width / 4 }, uRows: { int: rows }, uQ0: { int: q0 }, uKeys: { int: count }, uRep: { int: rep } },
            output: scores,
            rect: { x: 0, y: h * rows, width: keyTexels, height: n },
            work: 5 * head
          })
          await this.run({
            program: this.gpu.program(V.VISION_SOFTMAX_STATS),
            inputs: { uS: scores },
            uniforms: { uWidth: { int: keyTexels } },
            output: stats,
            rect: { x: 0, y: h * rows, width: 1, height: n },
            work: 2 * keyTexels
          })
        }
        await this.run({
          program: this.gpu.program(V.visionValues(head)),
          inputs: { uS: scores, uStats: stats, uQKV: qkv },
          uniforms: {
            uWidth: { int: keyTexels },
            uRows: { int: rows },
            uQ0: { int: q0 },
            uKeys: { int: count },
            uVBase: { int: attentionRows(this.config).qk / 4 },
            uRep: { int: rep }
          },
          output: out,
          rect: { x: 0, y: q0, width: width / 4, height: n },
          work: 5 * keyTexels
        })
      }
    } finally {
      this.gpu.delete(scores)
      this.gpu.delete(stats)
    }
  }

  /**
   * A view's embedding rows: (gridH / merge) · (gridW / merge) rows of
   * outWidth floats, merged tokens in row-major order.
   */
  async encode(image: ImagePatches, signal?: AbortSignal): Promise<Float32Array> {
    this.signal = signal
    try {
      return await this.encodeImage(image, signal)
    } finally {
      this.signal = undefined
    }
  }

  private async encodeImage(image: ImagePatches, signal?: AbortSignal): Promise<Float32Array> {
    const c = this.config
    const { gridH, gridW } = image
    const count = gridH * gridW
    if (count > this.maxPatches) throw new RangeError(`An image of ${count} patches is past this GPU's ${this.maxPatches}.`)
    if (gridH % c.merge || gridW % c.merge) throw new RangeError("The patch grid isn't in whole merge blocks.")
    if (image.pixels.length !== count * c.patchValues) throw new RangeError("The image's patches don't match the encoder.")
    const tokens = count / (c.merge * c.merge)
    const W = c.width
    const made: Array<Texture> = []
    // `texels` wider than the values, for a tensor a projection reads past its end (zeros: see readTexels).
    const f32 = (dims: number, rows: number, pixels?: Float32Array, texels = dims / 4) => {
      const t = this.gpu.texture(GL.RGBA32F, texels, rows, pixels && { format: GL.RGBA, type: GL.FLOAT, pixels })
      made.push(t)
      return t
    }
    const free = (...textures: Array<Texture>) => {
      for (const t of textures) {
        this.gpu.delete(t)
        made.splice(made.indexOf(t), 1)
      }
    }
    try {
      const cs = c.rope ? f32(c.headDim, count, ropeTable(this.invFreq!, gridH, gridW, c.merge)) : null
      let x = f32(W, count)
      let next = f32(W, count)
      {
        // The patch projection's inputs, freed once it has read them.
        const pixels = f32(c.patchValues, count, image.pixels)
        const pos = this.positions ? f32(W, count, positionEmbeddings(this.positions, c.grid, W, gridH, gridW, c.merge, c.interpolation)) : null
        await this.matmul(pixels, this.patch, x, count, pos ? { residual: pos } : {})
        free(pixels, ...(pos ? [pos] : []))
        if (this.patchNorm) {
          await this.rmsNorm(x, this.patchNorm, next, count)
          ;[x, next] = [next, x]
        }
      }
      const { qkv: qkvRows, qk: qkRows } = attentionRows(c)
      const swiglu = c.mlp === "swiglu"
      const normed = f32(W, count)
      // The operator's projection: q | k | v, or a conv block's B | C | x.
      const wide = f32(Math.max(c.layers.includes("attention") ? qkvRows : 0, c.layers.includes("conv") ? 3 * W : 0), count)
      const qk = c.layers.includes("attention") ? f32(qkRows, count) : null
      const mixed = f32(W, count)
      const inner = f32(fc1Rows(c), count, undefined, swiglu ? fc1Rows(c) / 4 : readTexels(c.ffn))
      const gated = swiglu ? f32(c.ffn, count, undefined, readTexels(c.ffn)) : null
      for (const b of this.blocks) {
        signal?.throwIfAborted()
        await this.blockNorm(x, b.norm1, normed, count)
        if (b.type === "attention") {
          await this.matmul(normed, b.qkv, wide, count)
          await this.run({
            program: this.gpu.program(V.visionQk({ head: c.headDim / 4, norm: c.qkNorm, rope: c.rope })),
            inputs: { uQKV: wide, ...(cs && { uCS: cs }), ...(b.qNorm && b.kNorm && { uQW: b.qNorm, uKW: b.kNorm }) },
            uniforms: { uQ: { int: W / 4 }, uScale: 1 / Math.sqrt(c.headDim), ...(c.qkNorm && { uEps: c.eps }) },
            output: qk!,
            rect: { x: 0, y: 0, width: qkRows / 4, height: count },
            work: c.qkNorm ? c.headDim / 4 : 2
          })
          await this.attend(wide, qk!, mixed, count)
          await this.matmul(mixed, b.proj, next, count, { residual: x })
        } else {
          await this.matmul(normed, b.inProj, wide, count)
          await this.run({
            program: this.gpu.program(V.VISION_CONV),
            inputs: { uX: wide, uK: b.kernel },
            uniforms: { uTexels: { int: W / 4 }, uGridH: { int: gridH }, uGridW: { int: gridW }, uMerge: { int: c.merge } },
            output: mixed,
            rect: { x: 0, y: 0, width: W / 4, height: count },
            work: 28
          })
          await this.matmul(mixed, b.outProj, next, count, { residual: x })
        }
        ;[x, next] = [next, x]
        await this.blockNorm(x, b.norm2, normed, count)
        if (swiglu) {
          await this.matmul(normed, b.fc1, inner, count)
          await this.run({
            program: this.gpu.program(V.VISION_SWIGLU),
            inputs: { uX: inner },
            uniforms: { uTexels: { int: c.ffn / 4 } },
            output: gated!,
            rect: { x: 0, y: 0, width: c.ffn / 4, height: count },
            work: 2
          })
          await this.matmul(gated!, b.fc2, next, count, { residual: x })
        } else {
          await this.matmul(normed, b.fc1, inner, count, { act: c.mlp as V.VisionAct })
          await this.matmul(inner, b.fc2, next, count, { residual: x })
        }
        ;[x, next] = [next, x]
      }
      const per = c.merge * c.merge
      await this.layerNorm(x, this.merger.norm, normed, count, c.mergerEps)
      const { linear1, linear2 } = this.merger
      const merged = f32(linear1.N, tokens, undefined, readTexels(linear1.N))
      await this.matmul(normed, linear1, merged, tokens, { act: c.mergerAct, group: per })
      const out = f32(c.outWidth, tokens)
      await this.matmul(merged, linear2, out, tokens)
      return await this.gpu.read(out, { x: 0, y: 0, width: c.outWidth / 4, height: tokens })
    } finally {
      for (const t of made) this.gpu.delete(t)
    }
  }
}
