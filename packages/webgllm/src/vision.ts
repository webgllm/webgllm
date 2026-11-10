/**
 * The vision encoder on WebGL 2 (see docs/vision.md): uploads the pieces
 * loadVision yields, and turns an image's patches into the text model's
 * embedding rows, one row per merged token. It shares the engine's GPU
 * context, and is loaded only while a request has images to read.
 */
import { GL, type Gpu, type Pass, type Texture } from "./gpu/gl.js"
import * as V from "./gpu/vision-kernels.js"
import { patchPlace, type ImagePatches } from "./image.js"
import type { VisionConfig, VisionLinear, VisionNorm, VisionPiece } from "./load-vision.js"

type GpuLinear = { N: number; K: number; quants: Texture; scales: Texture; zeros: Texture; bias: Texture }
type GpuNorm = { weight: Texture; bias: Texture }
type GpuBlock = { norm1: GpuNorm; qkv: GpuLinear; proj: GpuLinear; norm2: GpuNorm; fc1: GpuLinear; fc2: GpuLinear }

/** Score bytes a chunk of query rows may take (see attend). */
const SCORE_BYTES = 32 << 20

/**
 * The patch's position embedding: the 48 × 48 table bilinearly
 * interpolated to the grid (corners to corners), as the encoder's graph
 * computes it.
 */
export function positionEmbeddings(
  table: Float32Array,
  grid: number,
  width: number,
  gridH: number,
  gridW: number,
  merge: number
): Float32Array {
  const count = gridH * gridW
  const out = new Float32Array(count * width)
  const hScale = (grid - 1) / Math.max(gridH - 1, 1)
  const wScale = (grid - 1) / Math.max(gridW - 1, 1)
  for (let p = 0; p < count; p++) {
    const { row, col } = patchPlace(p, gridW, merge)
    const hc = row * hScale
    const wc = col * wScale
    const h0 = Math.floor(hc)
    const w0 = Math.floor(wc)
    const h1 = Math.min(h0 + 1, grid - 1)
    const w1 = Math.min(w0 + 1, grid - 1)
    const dh = hc - h0
    const dw = wc - w0
    const corners: Array<[number, number]> = [
      [h0 * grid + w0, (1 - dh) * (1 - dw)],
      [h0 * grid + w1, (1 - dh) * dw],
      [h1 * grid + w0, dh * (1 - dw)],
      [h1 * grid + w1, dh * dw]
    ]
    const o = p * width
    for (const [index, weight] of corners) {
      const r = index * width
      for (let e = 0; e < width; e++) out[o + e] += weight * table[r + e]
    }
  }
  return out
}

/**
 * Each patch's RoPE cos and sin: frequency j < 16 by its row, 16..31 by its
 * column (16 texels a patch: 8 of cos, 8 of sin).
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
  private positions!: Float32Array
  private invFreq!: Float32Array
  private blocks: Array<GpuBlock> = []
  private merger!: { norm: GpuNorm; linear1: GpuLinear; linear2: GpuLinear }
  private weights: Array<Texture> = []

  constructor(readonly gpu: Gpu) {}

  /** The most patches one image may have here: a texture's height. */
  get maxPatches() {
    return this.gpu.limits.maxTextureSize
  }

  async load(pieces: AsyncIterable<VisionPiece>, signal?: AbortSignal) {
    for await (const piece of pieces) {
      signal?.throwIfAborted()
      if (piece.kind === "config") this.config = piece.config
      else if (piece.kind === "patch") {
        this.patch = this.linear(piece.patch)
        this.positions = piece.positions
        this.invFreq = piece.invFreq
      } else if (piece.kind === "block") {
        const b = piece.value
        this.blocks[piece.index] = {
          norm1: this.norm(b.norm1),
          qkv: this.linear(b.qkv),
          proj: this.linear(b.proj),
          norm2: this.norm(b.norm2),
          fc1: this.linear(b.fc1),
          fc2: this.linear(b.fc2)
        }
      } else {
        this.merger = { norm: this.norm(piece.norm), linear1: this.linear(piece.linear1), linear2: this.linear(piece.linear2) }
      }
      await this.gpu.pace(1, true)
    }
    if (!this.config || !this.patch || !this.merger || this.blocks.length !== this.config.depth)
      throw new Error("The vision encoder's weights are incomplete.")
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
    return { weight: this.vector(n.weight), bias: this.vector(n.bias) }
  }

  private linear({ m, bias }: VisionLinear): GpuLinear {
    const blocks = m.K / 32
    const raw = m.raw
    if (!raw) throw new Error("The vision encoder takes MatMulNBits rows as stored.")
    const words = new Uint32Array(raw.buffer, raw.byteOffset, raw.byteLength / 4)
    const zeros = m.zeros ?? new Uint8Array(m.N * blocks).fill(8)
    return {
      N: m.N,
      K: m.K,
      quants: this.keep(
        this.gpu.texture(GL.RGBA32UI, blocks, m.N, { format: GL.RGBA_INTEGER, type: GL.UNSIGNED_INT, pixels: words })
      ),
      scales: this.keep(
        this.gpu.texture(GL.RGBA16F, blocks, m.N / 4, { format: GL.RGBA, type: GL.HALF_FLOAT, pixels: m.scales })
      ),
      zeros: this.keep(
        this.gpu.texture(GL.RGBA8UI, blocks, m.N / 4, { format: GL.RGBA_INTEGER, type: GL.UNSIGNED_BYTE, pixels: zeros })
      ),
      bias: this.vector(bias)
    }
  }

  /** Frees the encoder's textures. */
  dispose() {
    for (const t of this.weights) this.gpu.delete(t)
    this.weights = []
    this.blocks = []
  }

  private async run(pass: Pass) {
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
    await this.run({
      program: this.gpu.program(V.visionMatmul({ act: o.act ?? "none", residual: !!o.residual, group })),
      inputs: {
        uA: a,
        uW: w.quants,
        uS: w.scales,
        uZ: w.zeros,
        uBias: w.bias,
        ...(o.residual && { uR: o.residual })
      },
      uniforms: {
        uBlocks: { int: w.K / 32 },
        ...(group && { uInTexels: { int: w.K / 4 / o.group! }, uGroup: { int: o.group! } })
      },
      output: out,
      rect: { x: 0, y: 0, width: w.N / 4, height: rows },
      work: (w.K / 32) * 14
    })
  }

  private async layerNorm(x: Texture, n: GpuNorm, out: Texture, rows: number) {
    const texels = this.config.width / 4
    await this.run({
      program: this.gpu.program(V.VISION_LAYER_NORM),
      inputs: { uX: x, uW: n.weight, uB: n.bias },
      uniforms: { uTexels: { int: texels }, uEps: this.config.eps },
      output: out,
      rect: { x: 0, y: 0, width: texels, height: rows },
      work: 2 * texels
    })
  }

  /**
   * Bidirectional attention over all `count` patches (one image): scores,
   * softmax statistics and the weighted values, by chunks of query rows
   * whose scores fit SCORE_BYTES.
   */
  private async attend(qkv: Texture, qk: Texture, out: Texture, count: number) {
    const { heads, width } = this.config
    const keyTexels = Math.ceil(count / 4)
    const max = this.gpu.limits.maxTextureSize
    const rows = Math.max(1, Math.min(count, Math.floor(max / heads), Math.floor(SCORE_BYTES / (keyTexels * 16 * heads))))
    const scores = this.gpu.texture(GL.RGBA32F, keyTexels, rows * heads)
    const stats = this.gpu.texture(GL.RGBA32F, 1, rows * heads)
    try {
      for (let q0 = 0; q0 < count; q0 += rows) {
        const n = Math.min(rows, count - q0)
        // Rows of a short last chunk are where a full chunk's would be.
        for (let h = 0; h < heads; h++) {
          await this.run({
            program: this.gpu.program(V.VISION_SCORES),
            inputs: { uQK: qk },
            uniforms: { uTexels: { int: width / 4 }, uRows: { int: rows }, uQ0: { int: q0 }, uKeys: { int: count } },
            output: scores,
            rect: { x: 0, y: h * rows, width: keyTexels, height: n },
            work: 80
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
          program: this.gpu.program(V.VISION_VALUES),
          inputs: { uS: scores, uStats: stats, uQKV: qkv },
          uniforms: {
            uWidth: { int: keyTexels },
            uRows: { int: rows },
            uQ0: { int: q0 },
            uKeys: { int: count },
            uVBase: { int: (2 * width) / 4 }
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
   * An image's embedding rows: (gridH / merge) · (gridW / merge) rows of
   * outWidth floats, merged tokens in row-major order.
   */
  async encode(image: ImagePatches, signal?: AbortSignal): Promise<Float32Array> {
    const c = this.config
    const { gridH, gridW } = image
    const count = gridH * gridW
    if (count > this.maxPatches) throw new RangeError(`An image of ${count} patches is past this GPU's ${this.maxPatches}.`)
    if (gridH % c.merge || gridW % c.merge) throw new RangeError("The patch grid isn't in whole merge blocks.")
    if (image.pixels.length !== count * c.patchValues) throw new RangeError("The image's patches don't match the encoder.")
    const tokens = count / (c.merge * c.merge)
    const W = c.width
    const made: Array<Texture> = []
    const f32 = (dims: number, rows: number, pixels?: Float32Array) => {
      const t = this.gpu.texture(
        GL.RGBA32F,
        dims / 4,
        rows,
        pixels && { format: GL.RGBA, type: GL.FLOAT, pixels }
      )
      made.push(t)
      return t
    }
    try {
      const pixels = f32(c.patchValues, count, image.pixels)
      const pos = f32(W, count, positionEmbeddings(this.positions, c.grid, W, gridH, gridW, c.merge))
      const cs = f32(16 * 4, count, ropeTable(this.invFreq, gridH, gridW, c.merge))
      let x = f32(W, count)
      let next = f32(W, count)
      const normed = f32(W, count)
      const qkv = f32(3 * W, count)
      const qk = f32(2 * W, count)
      const attn = f32(W, count)
      const inner = f32(c.ffn, count)
      await this.matmul(pixels, this.patch, x, count, { residual: pos })
      for (const b of this.blocks) {
        signal?.throwIfAborted()
        await this.layerNorm(x, b.norm1, normed, count)
        await this.matmul(normed, b.qkv, qkv, count)
        await this.run({
          program: this.gpu.program(V.VISION_ROPE),
          inputs: { uQKV: qkv, uCS: cs },
          uniforms: { uTexels: { int: W / 4 }, uScale: 1 / Math.sqrt(c.headDim) },
          output: qk,
          rect: { x: 0, y: 0, width: (2 * W) / 4, height: count }
        })
        await this.attend(qkv, qk, attn, count)
        await this.matmul(attn, b.proj, next, count, { residual: x })
        ;[x, next] = [next, x]
        await this.layerNorm(x, b.norm2, normed, count)
        await this.matmul(normed, b.fc1, inner, count, { act: c.act })
        await this.matmul(inner, b.fc2, next, count, { residual: x })
        ;[x, next] = [next, x]
      }
      const per = c.merge * c.merge
      await this.layerNorm(x, this.merger.norm, normed, count)
      const merged = f32(per * W, tokens)
      await this.matmul(normed, this.merger.linear1, merged, tokens, { act: c.mergerAct, group: per })
      const out = f32(c.outWidth, tokens)
      await this.matmul(merged, this.merger.linear2, out, tokens)
      return await this.gpu.read(out, { x: 0, y: 0, width: c.outWidth / 4, height: tokens })
    } finally {
      for (const t of made) this.gpu.delete(t)
    }
  }
}
