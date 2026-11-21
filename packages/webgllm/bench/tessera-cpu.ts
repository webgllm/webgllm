/**
 * Tessera on the CPU, in float32 with float64 sums: the denoiser as
 * Tessera's model.py writes it, and CLIP's text tower, on the weights
 * tessera-weights.ts reads. A reference for the GPU's (tessera-check.html),
 * itself checked against ONNX Runtime (tessera-reference.mjs). `trace`
 * gets each stage's output.
 */
import { TESSERA_CONFIG as C, type ClipWeights, type QLinear, type TesseraBlock, type TesseraWeights } from "../src/tessera-weights.js"

export type Trace = (name: string, values: Float32Array) => void

/** rows × N = X (rows × K) · Wᵀ + bias. */
export function linearRows(l: QLinear, x: Float32Array, rows: number): Float32Array {
  const out = new Float32Array(rows * l.N)
  const scaled = new Float32Array(l.K)
  for (let r = 0; r < rows; r++) {
    for (let k = 0; k < l.K; k++) scaled[k] = x[r * l.K + k]! * (l.axis === "in" ? l.scale[k]! : 1)
    for (let n = 0; n < l.N; n++) {
      let s = 0
      const w = n * l.K
      for (let k = 0; k < l.K; k++) s += scaled[k]! * l.q[w + k]!
      out[r * l.N + n] = s * (l.axis === "out" ? l.scale[n]! : 1) + (l.bias?.[n] ?? 0)
    }
  }
  return out
}

function rmsNorm(x: Float32Array, rows: number, dim: number, weight: Float32Array | null, eps = C.eps): Float32Array {
  const out = new Float32Array(rows * dim)
  for (let r = 0; r < rows; r++) {
    let s = 0
    for (let d = 0; d < dim; d++) s += x[r * dim + d]! ** 2
    const inv = 1 / Math.sqrt(s / dim + eps)
    for (let d = 0; d < dim; d++) out[r * dim + d] = x[r * dim + d]! * inv * (weight ? weight[d]! : 1)
  }
  return out
}

function layerNorm(x: Float32Array, rows: number, dim: number, n: { weight: Float32Array; bias: Float32Array }, eps: number): Float32Array {
  const out = new Float32Array(rows * dim)
  for (let r = 0; r < rows; r++) {
    let mean = 0
    for (let d = 0; d < dim; d++) mean += x[r * dim + d]!
    mean /= dim
    let v = 0
    for (let d = 0; d < dim; d++) v += (x[r * dim + d]! - mean) ** 2
    const inv = 1 / Math.sqrt(v / dim + eps)
    for (let d = 0; d < dim; d++) out[r * dim + d] = (x[r * dim + d]! - mean) * inv * n.weight[d]! + n.bias[d]!
  }
  return out
}

const silu = (x: number) => x / (1 + Math.exp(-x))

/** Fourier features of each value: sin of value · 2^i · π for i < n, then the cosines. */
function fourier(values: ArrayLike<number>, n: number): Float32Array {
  const out = new Float32Array(values.length * 2 * n)
  for (let v = 0; v < values.length; v++)
    for (let i = 0; i < n; i++) {
      const a = values[v]! * 2 ** i * Math.PI
      out[v * 2 * n + i] = Math.sin(a)
      out[v * 2 * n + n + i] = Math.cos(a)
    }
  return out
}

/** cos and sin of 2D axial RoPE: [prefix + grid²][headDim / 2], prefix rows at angle 0 (model.py's build_rope). */
export function ropeTables(prefix: number, grid: number, stride: number) {
  const half = C.headDim / 2
  const n = C.headDim / 4
  const cos = new Float32Array((prefix + grid * grid) * half).fill(1)
  const sin = new Float32Array((prefix + grid * grid) * half)
  const off = (stride - 1) / 2
  for (let i = 0; i < grid; i++)
    for (let j = 0; j < grid; j++) {
      const row = (prefix + i * grid + j) * half
      for (let f = 0; f < n; f++) {
        const freq = 1 / C.ropeTheta ** (f / n)
        const ar = (i * stride + off) * freq
        const ac = (j * stride + off) * freq
        cos[row + f] = Math.cos(ar)
        sin[row + f] = Math.sin(ar)
        cos[row + n + f] = Math.cos(ac)
        sin[row + n + f] = Math.sin(ac)
      }
    }
  return { cos, sin }
}

/** Softmax attention per batch row of `seq` tokens; q, k, v rows × heads·headDim. `causal` masks later keys. */
function attention(q: Float32Array, k: Float32Array, v: Float32Array, batch: number, seq: number, heads: number, causal = false) {
  const hd = C.headDim
  const width = heads * hd
  const out = new Float32Array(batch * seq * width)
  const scores = new Float64Array(seq)
  const scale = 1 / Math.sqrt(hd)
  for (let b = 0; b < batch; b++)
    for (let h = 0; h < heads; h++)
      for (let i = 0; i < seq; i++) {
        const qi = (b * seq + i) * width + h * hd
        let max = -Infinity
        const last = causal ? i : seq - 1
        for (let j = 0; j <= last; j++) {
          const kj = (b * seq + j) * width + h * hd
          let s = 0
          for (let d = 0; d < hd; d++) s += q[qi + d]! * k[kj + d]!
          scores[j] = s * scale
          if (scores[j]! > max) max = scores[j]!
        }
        let sum = 0
        for (let j = 0; j <= last; j++) sum += scores[j] = Math.exp(scores[j]! - max)
        for (let d = 0; d < hd; d++) {
          let s = 0
          for (let j = 0; j <= last; j++) s += scores[j]! * v[(b * seq + j) * width + h * hd + d]!
          out[qi + d] = s / sum
        }
      }
  return out
}

/** One Tessera block over `batch` rows of `seq` tokens (`prefix` of them not image tokens, the rest a grid × grid image). */
function block(
  blk: TesseraBlock,
  xs: Float32Array,
  mod: Float32Array,
  rope: { cos: Float32Array; sin: Float32Array },
  batch: number,
  seq: number,
  prefix: number,
  grid: number
) {
  const D = C.dim
  const rows = batch * seq
  const m = (b: number, j: number, d: number) => mod[b * 6 * D + j * D + d]! + blk.modBias[j * D + d]!
  const modulate = (x: Float32Array, shift: number, scale: number) => {
    const n = rmsNorm(x, rows, D, null)
    for (let r = 0; r < rows; r++) {
      const b = Math.floor(r / seq)
      for (let d = 0; d < D; d++) n[r * D + d] = n[r * D + d]! * (1 + m(b, scale, d)) + m(b, shift, d)
    }
    return n
  }
  // Attention.
  let h = modulate(xs, 0, 1)
  const qkv = linearRows(blk.qkv, h, rows)
  const q = new Float32Array(rows * D)
  const k = new Float32Array(rows * D)
  const v = new Float32Array(rows * D)
  const hd = C.headDim
  const half = hd / 2
  for (let r = 0; r < rows; r++) {
    const pos = r % seq
    for (let hh = 0; hh < C.heads; hh++) {
      for (const [src, dst, w] of [[0, q, blk.qNorm], [D, k, blk.kNorm]] as const) {
        const base = r * 3 * D + src + hh * hd
        let s = 0
        for (let d = 0; d < hd; d++) s += qkv[base + d]! ** 2
        const inv = 1 / Math.sqrt(s / hd + C.eps)
        const o = r * D + hh * hd
        for (let d = 0; d < half; d++) {
          const x1 = qkv[base + d]! * inv * w[d]!
          const x2 = qkv[base + half + d]! * inv * w[half + d]!
          const c = rope.cos[pos * half + d]!
          const sn = rope.sin[pos * half + d]!
          dst[o + d] = x1 * c - x2 * sn
          dst[o + half + d] = x1 * sn + x2 * c
        }
      }
      for (let d = 0; d < hd; d++) v[r * D + hh * hd + d] = qkv[r * 3 * D + 2 * D + hh * hd + d]!
    }
  }
  const attn = linearRows(blk.proj, attention(q, k, v, batch, seq, C.heads), rows)
  const x1 = new Float32Array(rows * D)
  for (let r = 0; r < rows; r++) {
    const b = Math.floor(r / seq)
    for (let d = 0; d < D; d++) x1[r * D + d] = xs[r * D + d]! + m(b, 2, d) * attn[r * D + d]!
  }
  // Conv-SwiGLU.
  h = modulate(x1, 3, 4)
  const H = C.hidden
  const w12 = linearRows(blk.w12, h, rows)
  const gated = new Float32Array(rows * H)
  for (let b = 0; b < batch; b++)
    for (let t = 0; t < seq; t++) {
      const r = b * seq + t
      for (let c = 0; c < H; c++) {
        let g = w12[r * 2 * H + H + c]!
        if (t >= prefix) {
          const i = Math.floor((t - prefix) / grid)
          const j = (t - prefix) % grid
          let conv = blk.dwBias[c]!
          for (let dy = -1; dy <= 1; dy++)
            for (let dx = -1; dx <= 1; dx++) {
              const y = i + dy
              const x = j + dx
              if (y < 0 || y >= grid || x < 0 || x >= grid) continue
              conv += blk.dwWeight[c * 9 + (dy + 1) * 3 + dx + 1]! * w12[(b * seq + prefix + y * grid + x) * 2 * H + H + c]!
            }
          g += conv
        }
        gated[r * H + c] = w12[r * 2 * H + c]! * silu(g)
      }
    }
  const mlp = linearRows(blk.w3, gated, rows)
  const out = new Float32Array(rows * D)
  for (let r = 0; r < rows; r++) {
    const b = Math.floor(r / seq)
    for (let d = 0; d < D; d++) out[r * D + d] = x1[r * D + d]! + m(b, 5, d) * mlp[r * D + d]!
  }
  return out
}

export type DenoiserInputs = {
  x: Int32Array
  textStates: Float32Array
  textPooled: Float32Array
  maskRatio: Float32Array
  scProbs: Float32Array
  dropText: Uint8Array
}

/** The denoiser's logits, [batch][256][vocab], as model.py's forward computes them. */
export function denoise(w: TesseraWeights, inputs: DenoiserInputs, batch: number, trace?: Trace): Float32Array {
  const D = C.dim
  const G = C.grid
  const P = G * G
  const L = C.textLen
  const pre = C.registers + L
  // Image tokens.
  const sc = linearRows(w.scProj, inputs.scProbs, batch * P)
  const img = new Float32Array(batch * P * D)
  for (let r = 0; r < batch * P; r++) {
    const tok = inputs.x[r]!
    const masked = tok === C.vocab
    for (let d = 0; d < D; d++) img[r * D + d] = w.embed[tok * D + d]! + w.pos[(r % P) * D + d]! + (masked ? sc[r * D + d]! : 0)
  }
  trace?.("embed", img)
  // Text tokens and the global condition.
  const t = linearRows(w.textProj, rmsNorm(inputs.textStates, batch * L, C.textDim, w.textNorm), batch * L)
  const pooled = rmsNorm(inputs.textPooled, batch, C.textDim, w.poolNorm)
  const c = linearRows(w.pool3, linearRows(w.pool1, pooled, batch).map(silu), batch)
  for (let b = 0; b < batch; b++) {
    if (!inputs.dropText[b]) continue
    t.set(w.nullText, b * L * D)
    c.set(w.nullCond, b * D)
  }
  const r = fourier(Array.from(inputs.maskRatio, (m) => m * 2 - 1), 8)
  const ratio = linearRows(w.ratio2, linearRows(w.ratio0, r, batch).map(silu), batch)
  const g = c.map((v, i) => v + ratio[i]!)
  trace?.("cond", g)
  const sg = g.map(silu)
  const mod = linearRows(w.tBlock, sg, batch)
  trace?.("mod", mod)
  // [registers | text | image], per batch row.
  const seqIn = pre + P
  let xs = new Float32Array(batch * seqIn * D)
  for (let b = 0; b < batch; b++) {
    xs.set(w.registers, b * seqIn * D)
    xs.set(t.subarray(b * L * D, (b + 1) * L * D), (b * seqIn + C.registers) * D)
    xs.set(img.subarray(b * P * D, (b + 1) * P * D), (b * seqIn + pre) * D)
  }
  const ropeIn = ropeTables(pre, G, 1)
  for (const [i, blk] of w.blocksIn.entries()) {
    xs = block(blk, xs, mod, ropeIn, batch, seqIn, pre, G)
    trace?.(`in.${i}`, xs)
  }
  // Patch level: 2 × 2 merge, the core, then the split back plus the skip.
  const p = C.patch
  const g2 = G / p
  const seqCore = pre + g2 * g2
  const merged = new Float32Array(batch * g2 * g2 * p * p * D)
  for (let b = 0; b < batch; b++)
    for (let i = 0; i < G; i++)
      for (let j = 0; j < G; j++) {
        const tok = (b * g2 + Math.floor(i / p)) * g2 + Math.floor(j / p)
        const slot = (i % p) * p + (j % p)
        merged.set(xs.subarray((b * seqIn + pre + i * G + j) * D, (b * seqIn + pre + i * G + j + 1) * D), (tok * p * p + slot) * D)
      }
  const down = linearRows(w.down, rmsNorm(merged, batch * g2 * g2, p * p * D, w.downNorm), batch * g2 * g2)
  let core = new Float32Array(batch * seqCore * D)
  for (let b = 0; b < batch; b++) {
    core.set(xs.subarray(b * seqIn * D, (b * seqIn + pre) * D), b * seqCore * D)
    core.set(down.subarray(b * g2 * g2 * D, (b + 1) * g2 * g2 * D), (b * seqCore + pre) * D)
  }
  trace?.("down", core)
  const ropeCore = ropeTables(pre, g2, p)
  for (const [i, blk] of w.blocksCore.entries()) {
    core = block(blk, core, mod, ropeCore, batch, seqCore, pre, g2)
    trace?.(`core.${i}`, core)
  }
  const coreImg = new Float32Array(batch * g2 * g2 * D)
  for (let b = 0; b < batch; b++) coreImg.set(core.subarray((b * seqCore + pre) * D, (b + 1) * seqCore * D), b * g2 * g2 * D)
  const up = linearRows(w.up, coreImg, batch * g2 * g2)
  const next = new Float32Array(batch * seqIn * D)
  for (let b = 0; b < batch; b++) {
    next.set(core.subarray(b * seqCore * D, (b * seqCore + pre) * D), b * seqIn * D)
    for (let i = 0; i < G; i++)
      for (let j = 0; j < G; j++) {
        const tok = (b * g2 + Math.floor(i / p)) * g2 + Math.floor(j / p)
        const slot = (i % p) * p + (j % p)
        const row = (b * seqIn + pre + i * G + j) * D
        for (let d = 0; d < D; d++) next[row + d] = xs[row + d]! + up[(tok * p * p + slot) * D + d]!
      }
  }
  xs = next
  trace?.("up", xs)
  for (const [i, blk] of w.blocksOut.entries()) {
    xs = block(blk, xs, mod, ropeIn, batch, seqIn, pre, G)
    trace?.(`out.${i}`, xs)
  }
  // The head.
  const fm = linearRows(w.finalMod, sg, batch)
  const imgOut = new Float32Array(batch * P * D)
  for (let b = 0; b < batch; b++) imgOut.set(xs.subarray((b * seqIn + pre) * D, (b + 1) * seqIn * D), b * P * D)
  const n = rmsNorm(imgOut, batch * P, D, null)
  for (let r2 = 0; r2 < batch * P; r2++) {
    const b = Math.floor(r2 / P)
    for (let d = 0; d < D; d++) n[r2 * D + d] = n[r2 * D + d]! * (1 + fm[b * 2 * D + D + d]!) + fm[b * 2 * D + d]!
  }
  trace?.("final", n)
  return linearRows(w.head, n, batch * P)
}

/** CLIP's text tower on 24 ids: last_hidden_state [24][width] and pooler_output [width]. */
export function encodeText(w: ClipWeights, ids: ArrayLike<number>, trace?: Trace) {
  const W = w.width
  const L = ids.length
  let x = new Float32Array(L * W)
  for (let t = 0; t < L; t++)
    for (let d = 0; d < W; d++) x[t * W + d] = w.tokens.q[ids[t]! * W + d]! * w.tokens.scale[ids[t]!]! + w.positions[t * W + d]!
  for (const [i, layer] of w.layers.entries()) {
    const h = layerNorm(x, L, W, layer.ln1, w.eps)
    const a = attention(linearRows(layer.q, h, L), linearRows(layer.k, h, L), linearRows(layer.v, h, L), 1, L, w.heads, true)
    const o = linearRows(layer.out, a, L)
    x = x.map((v, j) => v + o[j]!)
    const h2 = layerNorm(x, L, W, layer.ln2, w.eps)
    const f = linearRows(layer.fc1, h2, L).map((v) => v / (1 + Math.exp(-1.702 * v)))
    const f2 = linearRows(layer.fc2, f, L)
    x = x.map((v, j) => v + f2[j]!)
    trace?.(`clip.${i}`, x)
  }
  const last = layerNorm(x, L, W, w.final, w.eps)
  let eos = 0
  for (let t = 1; t < L; t++) if (ids[t]! > ids[eos]!) eos = t
  return { states: last, pooled: last.slice(eos * W, (eos + 1) * W) }
}
