/**
 * A vision encoder as the GPU takes it, whatever file it came from: what
 * the ONNX loader (load-vision.ts) and the GGUF one (load-vision-gguf.ts)
 * both produce, and vision.ts uploads.
 *
 * Three encoders share these pieces (see docs/vision.md):
 * - Qwen3.5's ViT: LayerNorm blocks of fused q|k|v attention with 2D RoPE
 *   and a GELU MLP, a learned position table;
 * - LFM2-VL's SigLIP2: the same blocks without RoPE;
 * - LFM3-VL's tower: RMS-norm blocks of either a gated 3 × 3 depthwise
 *   convolution over the patch grid or grouped-query attention with 2D
 *   RoPE and per-head q and k norms, each followed by a SwiGLU MLP.
 * All end in a merger: a LayerNorm, each merge × merge block of patches as
 * one row, and two projections into the text model's width.
 */
import { halfTable } from "./half.js"

export type VisionConfig = {
  width: number
  heads: number
  /** Key and value heads: heads / kvHeads query heads read each. */
  kvHeads: number
  headDim: number
  /** The MLP's inner width. */
  ffn: number
  /** The merger's inner width (its first projection's rows). */
  mergerWidth: number
  /** The merger's output: the text model's hidden size. */
  outWidth: number
  /** Values per patch row of the pixels: 3 · frames · patch². */
  patchValues: number
  /** Patches per merged token side (2). */
  merge: number
  /** The blocks, in order. */
  layers: Array<"attention" | "conv">
  /** The blocks' norms: LayerNorm (weight and bias) or RMS (weight). */
  norm: "layer" | "rms"
  eps: number
  /** The merger's LayerNorm epsilon. */
  mergerEps: number
  /** The blocks' MLP: a projection, GELU (by tanh, or exact) and a projection; or SwiGLU (gate | up, then down). */
  mlp: "tanh" | "erf" | "swiglu"
  mergerAct: "tanh" | "erf"
  /** The learned position table's side (0: none). */
  grid: number
  /**
   * How the table is interpolated to an image's grid: corner to corner
   * (Qwen3.5), or bilinear on pixel centers (SigLIP2), with a wider filter
   * where the grid is smaller than the table (`antialias`).
   */
  interpolation: "corners" | "half-pixel" | "antialias"
  /** 2D RoPE on q and k (its frequencies come with the patch piece). */
  rope: boolean
  /** q and k RMS-normalized per head. */
  qkNorm: boolean
  /**
   * How the text model places an image's rows: Qwen3.5's 3D positions
   * (see Engine.setImages), or one after another like any token.
   */
  positions: "grid" | "sequence"
}

/**
 * A projection's weights, in blocks of 32 along K (the last one short when
 * K isn't a multiple of 32: its missing weights are zero points).
 */
export type VisionMatrix = {
  /** Output rows; a multiple of 4. */
  N: number
  K: number
  bits: 4 | 8
  /**
   * 4-bit: rows as MatMulNBits stores them, [N][blocks][16], element 2i of
   * a block in byte i's low nibble. 8-bit: [N][blocks · 32] bytes, one a
   * weight.
   */
  data: Uint8Array
  /** One per row and block, at ((n >> 2) · blocks + b) · 4 + (n & 3). */
  scales: Float32Array
  /** A weight is scale · (q − zero); laid out as the scales. Null: 8 (4-bit) or 128 (8-bit) everywhere. */
  zeros: Uint8Array | null
}

export type VisionLinear = { m: VisionMatrix; bias: Float32Array | null }
export type VisionNorm = { weight: Float32Array; bias: Float32Array | null }

export type VisionBlock =
  | {
      type: "attention"
      norm1: VisionNorm
      /** Rows q (heads · headDim), k and v (kvHeads · headDim each). */
      qkv: VisionLinear
      /** Per-head RMS norm weights (headDim), with config.qkNorm. */
      qNorm?: Float32Array
      kNorm?: Float32Array
      proj: VisionLinear
      norm2: VisionNorm
      /** The MLP's first projection; SwiGLU's gate rows then its up rows. */
      fc1: VisionLinear
      fc2: VisionLinear
    }
  | {
      type: "conv"
      norm1: VisionNorm
      /** Rows B, C, x (width each): the block is out(C · conv(B · x)). */
      inProj: VisionLinear
      /** The depthwise 3 × 3 kernel: channel c's tap (ky, kx) at c · 9 + ky · 3 + kx. */
      kernel: Float32Array
      outProj: VisionLinear
      norm2: VisionNorm
      fc1: VisionLinear
      fc2: VisionLinear
    }

export type VisionPiece =
  | { kind: "config"; config: VisionConfig }
  | {
      kind: "patch"
      patch: VisionLinear
      /** The position table: grid² rows of `width` (config.grid). */
      positions?: Float32Array
      /** RoPE's frequencies (headDim / 4): rows use them, columns the same again. */
      invFreq?: Float32Array
      /** An RMS norm of the embedded patches (LFM3-VL). */
      norm?: VisionNorm
    }
  | { kind: "block"; index: number; value: VisionBlock }
  | { kind: "merger"; norm: VisionNorm; linear1: VisionLinear; linear2: VisionLinear }

export const visionBlocks = (K: number) => Math.ceil(K / 32)

const scaleIndex = (n: number, b: number, blocks: number) => ((n >> 2) * blocks + b) * 4 + (n & 3)

/** An empty matrix of the layout above. */
export function newVisionMatrix(N: number, K: number, bits: 4 | 8, withZeros: boolean): VisionMatrix {
  if (N % 4) throw new Error(`A projection of ${N} rows isn't in groups of 4.`)
  const blocks = visionBlocks(K)
  return {
    N,
    K,
    bits,
    data: new Uint8Array(N * blocks * (bits === 4 ? 16 : 32)).fill(bits === 4 ? 0x88 : 128),
    scales: new Float32Array(N * blocks),
    zeros: withZeros ? new Uint8Array(N * blocks).fill(bits === 4 ? 8 : 128) : null
  }
}

/**
 * A MatMulNBits weight as a matrix. `quant` is [N][ceil(K / blockSize)]
 * [blockSize · bits / 8]; `scales` one float per row and block (float32 or
 * float16 bytes); `zp` the zero points: 4-bit ones packed two a byte (low
 * nibble first), 8-bit ones a byte each. 4-bit weights are in blocks of
 * 32; 8-bit ones in blocks of any multiple of 32, each scale then serving
 * its block's 32-weight blocks.
 */
export function matMulNBitsMatrix(
  w: { N: number; K: number; bits: 4 | 8; blockSize: number; scaleBytes: 2 | 4 },
  quant: Uint8Array,
  scales: Uint8Array,
  zp: Uint8Array | null
): VisionMatrix {
  const { N, K, bits, blockSize } = w
  if (blockSize % 32 || (bits === 4 && blockSize !== 32)) throw new Error(`MatMulNBits blocks of ${blockSize} ${bits}-bit weights aren't read.`)
  const fileBlocks = Math.ceil(K / blockSize)
  const blocks = visionBlocks(K)
  const per = blockSize / 32
  const m = newVisionMatrix(N, K, bits, zp !== null)
  const half = halfTable()
  const view = new DataView(scales.buffer, scales.byteOffset, scales.byteLength)
  const scaleAt = (i: number) => (w.scaleBytes === 2 ? half[view.getUint16(i * 2, true)] : view.getFloat32(i * 4, true))
  if (scales.length < N * fileBlocks * w.scaleBytes) throw new Error("MatMulNBits scales are too short.")
  if (bits === 4) {
    if (quant.length < N * blocks * 16) throw new Error("MatMulNBits weights are too short.")
    m.data.set(quant.subarray(0, N * blocks * 16))
  } else {
    const rowBytes = fileBlocks * blockSize
    if (quant.length < N * rowBytes) throw new Error("MatMulNBits weights are too short.")
    for (let n = 0; n < N; n++) m.data.set(quant.subarray(n * rowBytes, n * rowBytes + K), n * blocks * 32)
  }
  const zpStride = bits === 4 ? Math.ceil(fileBlocks / 2) : fileBlocks
  if (zp && zp.length < N * zpStride) throw new Error("MatMulNBits zero points are too short.")
  for (let n = 0; n < N; n++) {
    for (let b = 0; b < blocks; b++) {
      const fb = Math.floor(b / per)
      const at = scaleIndex(n, b, blocks)
      m.scales[at] = scaleAt(n * fileBlocks + fb)
      if (zp && m.zeros) m.zeros[at] = bits === 4 ? (zp[n * zpStride + (fb >> 1)] >> ((fb & 1) << 2)) & 15 : zp[n * zpStride + fb]
    }
  }
  return m
}

/**
 * Float rows (N × K) as an 8-bit matrix: per block of 32, scale =
 * max|w| / 127 and q = round(w / scale), as the text loaders quantize a
 * float matrix (repack.ts quantizeQ8).
 */
export function floatMatrix(values: Float32Array, N: number, K: number): VisionMatrix {
  if (values.length !== N * K) throw new Error(`A ${N} × ${K} projection has ${values.length} values.`)
  const m = newVisionMatrix(N, K, 8, false)
  const blocks = visionBlocks(K)
  for (let n = 0; n < N; n++) {
    for (let b = 0; b < blocks; b++) {
      const from = n * K + b * 32
      const count = Math.min(32, K - b * 32)
      let max = 0
      for (let j = 0; j < count; j++) max = Math.max(max, Math.abs(values[from + j]))
      const scale = max / 127
      m.scales[scaleIndex(n, b, blocks)] = scale
      if (!(scale > 0)) continue
      const to = (n * blocks + b) * 32
      // q + 128, q = floor(w / scale + 0.5) in -127..127 (without Math.round: several times quicker over a float projector).
      for (let j = 0; j < count; j++) {
        const u = (values[from + j] / scale + 128.5) | 0
        m.data[to + j] = u < 1 ? 1 : u > 255 ? 255 : u
      }
    }
  }
  return m
}

/** llama.cpp Q8_0 rows (blocks of an fp16 scale and 32 int8) as an 8-bit matrix. */
export function q8Matrix(bytes: Uint8Array, N: number, K: number): VisionMatrix {
  if (K % 32) throw new Error(`Q8_0 rows of ${K} weights aren't whole blocks.`)
  const blocks = K / 32
  if (bytes.length < N * blocks * 34) throw new Error("Q8_0 data is too short.")
  const m = newVisionMatrix(N, K, 8, false)
  const half = halfTable()
  for (let n = 0; n < N; n++) {
    for (let b = 0; b < blocks; b++) {
      const p = (n * blocks + b) * 34
      m.scales[scaleIndex(n, b, blocks)] = half[bytes[p] | (bytes[p + 1] << 8)]
      const to = (n * blocks + b) * 32
      for (let j = 0; j < 32; j++) m.data[to + j] = bytes[p + 2 + j] ^ 0x80
    }
  }
  return m
}

/** Matrices of the same K and bits as one, their rows in order (each a multiple of 4 rows). */
export function stackMatrices(parts: Array<VisionMatrix>): VisionMatrix {
  if (parts.length === 1) return parts[0]
  const { K, bits } = parts[0]
  if (parts.some((p) => p.K !== K || p.bits !== bits)) throw new Error("Projections of different shapes or bits can't be read as one.")
  const N = parts.reduce((sum, p) => sum + p.N, 0)
  const zeros = parts.some((p) => p.zeros)
  const m = newVisionMatrix(N, K, bits, zeros)
  let data = 0
  let scale = 0
  for (const p of parts) {
    m.data.set(p.data, data)
    m.scales.set(p.scales, scale)
    if (m.zeros && p.zeros) m.zeros.set(p.zeros, scale)
    data += p.data.length
    scale += p.scales.length
  }
  return m
}

/** Projections stacked into one: their matrices' rows, and their biases (zeros where one has none; null if none has). */
export function stackLinears(parts: Array<VisionLinear>): VisionLinear {
  if (parts.length === 1) return parts[0]
  const m = stackMatrices(parts.map((p) => p.m))
  if (parts.every((p) => !p.bias)) return { m, bias: null }
  const bias = new Float32Array(m.N)
  let at = 0
  for (const p of parts) {
    if (p.bias) bias.set(p.bias, at)
    at += p.m.N
  }
  return { m, bias }
}

/** Checks a config's shapes are ones the shaders lay out (whole texels, whole heads). */
export function checkVisionConfig(c: VisionConfig): VisionConfig {
  const whole = (v: number, what: string) => {
    if (!Number.isInteger(v) || v <= 0) throw new Error(`The vision encoder's ${what} isn't one webgllm's encoder runs.`)
  }
  whole(c.width / 4, "width")
  whole(c.headDim / 4, "head size")
  whole(c.heads / c.kvHeads, "key and value heads")
  whole(c.ffn / 4, "MLP width")
  whole(c.outWidth / 4, "output width")
  whole(c.mergerWidth / 4, "projector width")
  // A row of the blocks' width is read in whole blocks of 32 (vision.ts readTexels).
  whole(c.width / 32, "width")
  whole(c.patchValues / 32, "patch size")
  whole(c.patchValues / 4, "patch size")
  whole(c.merge, "merge size")
  if (c.heads * c.headDim !== c.width) throw new Error("The vision encoder's heads don't fill its width.")
  if (c.rope && c.headDim % 8) throw new Error("The vision encoder's RoPE needs heads of a multiple of 8 dims.")
  if (!c.layers.length) throw new Error("The vision encoder has no blocks.")
  return c
}
