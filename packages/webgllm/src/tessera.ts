/**
 * Tessera-small on the GPU (WebGPU or WebGL 2, through the engine's Gpu):
 * CLIP's text tower and the denoiser, on int8 weights read in the shaders
 * (gpu/tessera-kernels.ts). It turns 24 token ids into CLIP's states, and a
 * batch of canvases into the denoiser's logits; the sampler
 * (tessera-sampler.ts) drives the steps.
 */
import { GL, type Gpu, type Pass, type Texture } from "./gpu/gl.js"
import type { WebGpu } from "./gpu/webgpu.js"
import * as K from "./gpu/tessera-kernels.js"
import { TESSERA_CONFIG as C, type ClipWeights, type QLinear, type TesseraBlock, type TesseraWeights } from "./tessera-weights.js"

type GpuLinear = { N: number; K: number; axis: "out" | "in"; w: Texture; s: Texture; bias: Texture | null }
type GpuBlock = {
  qkv: GpuLinear
  qNorm: Texture
  kNorm: Texture
  proj: GpuLinear
  w12: GpuLinear
  w3: GpuLinear
  kernel: Texture
  convBias: Texture
  modBias: Texture
}
type GpuNorm = { weight: Texture; bias: Texture }
type GpuClipLayer = { ln1: GpuNorm; q: GpuLinear; k: GpuLinear; v: GpuLinear; out: GpuLinear; ln2: GpuNorm; fc1: GpuLinear; fc2: GpuLinear }

const D = C.dim
const DT = D / 4
const P = C.grid * C.grid
const PRE = C.registers + C.textLen
const SEQ = PRE + P
const CORE_GRID = C.grid / C.patch
const CORE_SEQ = PRE + CORE_GRID * CORE_GRID

/** cos and sin of 2D axial RoPE per token row: 8 texels of cos, then 8 of sin (model.py's build_rope). */
export function ropeTexels(prefix: number, grid: number, stride: number): Float32Array {
  const half = C.headDim / 2
  const n = C.headDim / 4
  const out = new Float32Array((prefix + grid * grid) * 2 * half)
  for (let r = 0; r < prefix + grid * grid; r++) out.fill(1, r * 2 * half, r * 2 * half + half)
  const off = (stride - 1) / 2
  for (let i = 0; i < grid; i++)
    for (let j = 0; j < grid; j++) {
      const row = (prefix + i * grid + j) * 2 * half
      for (let f = 0; f < n; f++) {
        const freq = 1 / C.ropeTheta ** (f / n)
        const ar = (i * stride + off) * freq
        const ac = (j * stride + off) * freq
        out[row + f] = Math.cos(ar)
        out[row + n + f] = Math.cos(ac)
        out[row + half + f] = Math.sin(ar)
        out[row + half + n + f] = Math.sin(ac)
      }
    }
  return out
}

/** Sin, then cos, of a value · 2^i · π for i < 8: the mask ratio's features (model.py's fourier_features). */
export function ratioFeatures(maskRatio: number): Float32Array {
  const out = new Float32Array(16)
  const r = maskRatio * 2 - 1
  for (let i = 0; i < 8; i++) {
    out[i] = Math.sin(r * 2 ** i * Math.PI)
    out[8 + i] = Math.cos(r * 2 ** i * Math.PI)
  }
  return out
}

/** GPU bytes of the weights tessera.ts uploads, by part. */
export function tesseraWeightBytes(w: TesseraWeights, clip: ClipWeights) {
  const linear = (l: QLinear) => l.N * l.K + (l.axis === "out" ? l.N : l.K) * 4 + (l.bias ? l.N * 4 : 0)
  const block = (b: TesseraBlock) => linear(b.qkv) + linear(b.proj) + linear(b.w12) + linear(b.w3) + (b.dwWeight.length + b.dwBias.length + b.modBias.length + 2 * C.headDim) * 4
  let denoiser = (w.embed.length + w.pos.length + w.nullText.length + w.registers.length + w.textNorm.length + w.poolNorm.length + w.downNorm.length) * 4
  for (const l of [w.scProj, w.textProj, w.pool1, w.pool3, w.ratio0, w.ratio2, w.tBlock, w.down, w.up, w.finalMod, w.head]) denoiser += linear(l)
  for (const b of [...w.blocksIn, ...w.blocksCore, ...w.blocksOut]) denoiser += block(b)
  let text = 0
  for (const l of clip.layers) text += [l.q, l.k, l.v, l.out, l.fc1, l.fc2].reduce((s, x) => s + linear(x), 0) + 4 * clip.width * 4
  return { denoiser, text }
}

export class TesseraGpu {
  private weights: Array<Texture> = []
  private work = new Map<string, Texture>()
  private den!: {
    embed: Texture
    pos: Texture
    scProj: GpuLinear
    textNorm: Texture
    textProj: GpuLinear
    nullText: Texture
    poolNorm: Texture
    pool1: GpuLinear
    pool3: GpuLinear
    nullCond: Float32Array
    ratio0: GpuLinear
    ratio2: GpuLinear
    registers: Texture
    tBlock: GpuLinear
    blocksIn: Array<GpuBlock>
    downNorm: Texture
    down: GpuLinear
    blocksCore: Array<GpuBlock>
    up: GpuLinear
    blocksOut: Array<GpuBlock>
    finalMod: GpuLinear
    head: GpuLinear
    ropeIn: Texture
    ropeCore: Texture
  }
  private clip!: { layers: Array<GpuClipLayer>; final: GpuNorm; tokens: ClipWeights["tokens"]; positions: Float32Array; width: number; eps: number }
  /** The text rows and the pooled condition of the prompt set last (setText). */
  private text: { rows: Texture; cond: Float32Array } | null = null
  private signal: AbortSignal | undefined

  constructor(readonly gpu: Gpu | WebGpu) {}

  /** GPU bytes of the weights, and of the work textures made so far. */
  get bytes() {
    let work = 0
    for (const t of this.work.values()) work += t.bytes
    return { weights: this.weights.reduce((s, t) => s + t.bytes, 0), work }
  }

  // ------------------------------------------------------------ uploads

  private keep(t: Texture) {
    this.weights.push(t)
    return t
  }

  /** Rows of floats as an RGBA32F texture: `rows` rows of values.length / rows. */
  private floats(values: Float32Array, rows = 1) {
    const width = values.length / rows / 4
    return this.keep(this.gpu.texture(GL.RGBA32F, width, rows, { format: GL.RGBA, type: GL.FLOAT, pixels: values }))
  }

  private linear(l: QLinear): GpuLinear {
    if (l.K % 16 || l.N % 4) throw new Error(`Tessera: a ${l.N} × ${l.K} projection isn't in groups of 16 inputs and 4 outputs.`)
    const words = new Uint32Array(l.q.buffer, l.q.byteOffset, l.q.byteLength / 4)
    return {
      N: l.N,
      K: l.K,
      axis: l.axis,
      w: this.keep(this.gpu.texture(GL.RGBA32UI, l.K / 16, l.N, { format: GL.RGBA_INTEGER, type: GL.UNSIGNED_INT, pixels: words })),
      s: this.floats(l.scale),
      bias: l.bias ? this.floats(l.bias) : null
    }
  }

  private uploadBlock(b: TesseraBlock): GpuBlock {
    // Taps as rows: texel (x, tap) holds channels 4x..4x+3.
    const H = C.hidden
    const taps = new Float32Array(9 * H)
    for (let ch = 0; ch < H; ch++) for (let t = 0; t < 9; t++) taps[t * H + ch] = b.dwWeight[ch * 9 + t]!
    return {
      qkv: this.linear(b.qkv),
      qNorm: this.floats(b.qNorm),
      kNorm: this.floats(b.kNorm),
      proj: this.linear(b.proj),
      w12: this.linear(b.w12),
      w3: this.linear(b.w3),
      kernel: this.floats(taps, 9),
      convBias: this.floats(b.dwBias),
      modBias: this.floats(b.modBias)
    }
  }

  /** Uploads both models' weights. */
  async load(w: TesseraWeights, clip: ClipWeights, signal?: AbortSignal) {
    const step = async () => {
      signal?.throwIfAborted()
      await this.gpu.pace(1, true)
    }
    const blocks = async (list: Array<TesseraBlock>) => {
      const out: Array<GpuBlock> = []
      for (const b of list) {
        out.push(this.uploadBlock(b))
        await step()
      }
      return out
    }
    this.den = {
      embed: this.floats(w.embed, C.vocab + 1),
      pos: this.floats(w.pos, P),
      scProj: this.linear(w.scProj),
      textNorm: this.floats(w.textNorm),
      textProj: this.linear(w.textProj),
      nullText: this.floats(w.nullText, C.textLen),
      poolNorm: this.floats(w.poolNorm),
      pool1: this.linear(w.pool1),
      pool3: this.linear(w.pool3),
      nullCond: w.nullCond,
      ratio0: this.linear(w.ratio0),
      ratio2: this.linear(w.ratio2),
      registers: this.floats(w.registers, C.registers),
      tBlock: this.linear(w.tBlock),
      blocksIn: await blocks(w.blocksIn),
      downNorm: this.floats(w.downNorm),
      down: this.linear(w.down),
      blocksCore: await blocks(w.blocksCore),
      up: this.linear(w.up),
      blocksOut: await blocks(w.blocksOut),
      finalMod: this.linear(w.finalMod),
      head: this.linear(w.head),
      ropeIn: this.floats(ropeTexels(PRE, C.grid, 1), SEQ),
      ropeCore: this.floats(ropeTexels(PRE, CORE_GRID, C.patch), CORE_SEQ)
    }
    const norm = (n: { weight: Float32Array; bias: Float32Array }) => ({ weight: this.floats(n.weight), bias: this.floats(n.bias) })
    const layers: Array<GpuClipLayer> = []
    for (const l of clip.layers) {
      layers.push({
        ln1: norm(l.ln1),
        q: this.linear(l.q),
        k: this.linear(l.k),
        v: this.linear(l.v),
        out: this.linear(l.out),
        ln2: norm(l.ln2),
        fc1: this.linear(l.fc1),
        fc2: this.linear(l.fc2)
      })
      await step()
    }
    if (clip.width !== C.textDim || clip.heads !== C.heads) throw new Error("Tessera: CLIP's width or heads aren't the denoiser's.")
    this.clip = { layers, final: norm(clip.final), tokens: clip.tokens, positions: clip.positions, width: clip.width, eps: clip.eps }
  }

  /** Frees every texture. */
  dispose() {
    for (const t of this.weights) this.gpu.delete(t)
    for (const t of this.work.values()) this.gpu.delete(t)
    this.weights = []
    this.work.clear()
    this.text = null
  }

  // ------------------------------------------------------------ passes

  /** A work texture by name, made again if it's too small. */
  private tex(name: string, width: number, height: number) {
    const t = this.work.get(name)
    if (t && t.width >= width && t.height >= height) return t
    if (t) this.gpu.delete(t)
    const made = this.gpu.texture(GL.RGBA32F, Math.max(width, t?.width ?? 0), Math.max(height, t?.height ?? 0))
    this.work.set(name, made)
    return made
  }

  private upload(name: string, values: Float32Array, rows: number) {
    const width = values.length / rows / 4
    const t = this.tex(name, width, rows)
    this.gpu.write(t, { x: 0, y: 0, width, height: rows }, { format: GL.RGBA, type: GL.FLOAT, pixels: values })
    return t
  }

  private async run(name: string, pass: Omit<Pass, "name">) {
    this.signal?.throwIfAborted()
    await this.gpu.runAsync({ name, ...pass })
  }

  private async mm(
    l: GpuLinear,
    a: Texture,
    out: Texture,
    rows: number,
    o: { act?: K.TesseraAct; residual?: Texture; gate?: { mod: Texture; bias: Texture; seq: number; chunk: number }; siluIn?: boolean; out0?: number } = {}
  ) {
    const residual = o.residual ?? null
    await this.run("tessera-matmul", {
      program: this.gpu.program(
        K.tesseraMatmul({ axis: l.axis, bias: !!l.bias, act: o.act ?? "none", residual: !!residual && !o.gate, gate: !!o.gate, siluIn: !!o.siluIn })
      ),
      inputs: {
        uA: a,
        uW: l.w,
        uS: l.s,
        ...(l.bias && { uBias: l.bias }),
        ...(residual && { uR: residual }),
        ...(o.gate && { uMod: o.gate.mod, uModBias: o.gate.bias })
      },
      uniforms: {
        uSteps: { int: l.K / 16 },
        uOut0: { int: o.out0 ?? 0 },
        ...(o.gate && { uSeq: { int: o.gate.seq }, uGate: { int: o.gate.chunk }, uDimTexels: { int: l.N / 4 } })
      },
      output: out,
      rect: { x: o.out0 ?? 0, y: 0, width: l.N / 4, height: rows },
      work: (l.K / 16) * 9
    })
  }

  private async rmsNorm(
    x: Texture,
    out: Texture,
    rows: number,
    texels: number,
    o: { weight?: Texture; mod?: { mod: Texture; bias?: Texture; seq: number; shift: number; scale: number } }
  ) {
    await this.run("tessera-norm", {
      program: this.gpu.program(K.tesseraRmsNorm({ weight: !!o.weight, modulate: !!o.mod, modBias: !!o.mod?.bias })),
      inputs: { uX: x, ...(o.weight && { uW: o.weight }), ...(o.mod && { uMod: o.mod.mod }), ...(o.mod?.bias && { uModBias: o.mod.bias }) },
      uniforms: {
        uTexels: { int: texels },
        uEps: C.eps,
        ...(o.mod && { uSeq: { int: o.mod.seq }, uShift: { int: o.mod.shift }, uScale: { int: o.mod.scale } })
      },
      output: out,
      rect: { x: 0, y: 0, width: texels, height: rows },
      work: texels
    })
  }

  private async layerNorm(x: Texture, n: GpuNorm, out: Texture, rows: number, eps: number) {
    await this.run("tessera-layer-norm", {
      program: this.gpu.program(K.TESSERA_LAYER_NORM),
      inputs: { uX: x, uW: n.weight, uB: n.bias },
      uniforms: { uTexels: { int: DT }, uEps: eps },
      output: out,
      rect: { x: 0, y: 0, width: DT, height: rows },
      work: 2 * DT
    })
  }

  /**
   * Softmax attention of every row over its batch row's `seq` tokens, in
   * chunks of query rows whose scores fit a texture's height.
   */
  private async attend(q: Texture, qBase: number, k: Texture, kBase: number, v: Texture, vBase: number, out: Texture, rows: number, seq: number, causal: boolean) {
    const heads = C.heads
    const width = Math.ceil(seq / 4)
    const chunk = Math.min(rows, Math.floor(this.gpu.limits.maxTextureSize / heads))
    const scores = this.tex("scores", width, chunk * heads)
    const stats = this.tex("stats", 1, chunk * heads)
    for (let q0 = 0; q0 < rows; q0 += chunk) {
      const n = Math.min(chunk, rows - q0)
      await this.run("tessera-scores", {
        program: this.gpu.program(K.tesseraScores({ causal })),
        inputs: { uQ: q, uK: k },
        uniforms: { uRows: { int: n }, uQ0: { int: q0 }, uSeq: { int: seq }, uQBase: { int: qBase }, uKBase: { int: kBase } },
        output: scores,
        rect: { x: 0, y: 0, width, height: n * heads },
        work: 4 * 16 + 16
      })
      await this.run("tessera-stats", {
        program: this.gpu.program(K.TESSERA_SOFTMAX_STATS),
        inputs: { uS: scores },
        uniforms: { uWidth: { int: width } },
        output: stats,
        rect: { x: 0, y: 0, width: 1, height: n * heads },
        work: 2 * width
      })
      await this.run("tessera-values", {
        program: this.gpu.program(K.TESSERA_VALUES),
        inputs: { uS: scores, uStats: stats, uV: v },
        uniforms: { uWidth: { int: width }, uRows: { int: n }, uQ0: { int: q0 }, uSeq: { int: seq }, uVBase: { int: vBase } },
        output: out,
        rect: { x: 0, y: q0, width: DT, height: n },
        work: width + seq
      })
    }
  }

  /** One block, in place on `xs` (rows of `seq` a batch row). */
  private async block(b: GpuBlock, xs: Texture, rows: number, seq: number, grid: number, rope: Texture, mod: Texture) {
    const h = this.tex("h", DT, rows)
    const x1 = this.tex(`x1-${seq}`, DT, rows)
    const qkv = this.tex("qkv", 3 * DT, rows)
    const qk = this.tex("qk", 2 * DT, rows)
    const att = this.tex("att", DT, rows)
    const w12 = this.tex("w12", (2 * C.hidden) / 4, rows)
    const gated = this.tex("gated", C.hidden / 4, rows)
    const modOf = (shift: number, scale: number) => ({ mod, bias: b.modBias, seq, shift, scale })
    await this.rmsNorm(xs, h, rows, DT, { mod: modOf(0, 1) })
    await this.mm(b.qkv, h, qkv, rows)
    await this.run("tessera-qk", {
      program: this.gpu.program(K.TESSERA_QK_NORM_ROPE),
      inputs: { uQKV: qkv, uQN: b.qNorm, uKN: b.kNorm, uRope: rope },
      uniforms: { uSeq: { int: seq }, uDimTexels: { int: DT }, uEps: C.eps },
      output: qk,
      rect: { x: 0, y: 0, width: 2 * DT, height: rows },
      work: 20
    })
    await this.attend(qk, 0, qk, DT, qkv, 2 * DT, att, rows, seq, false)
    await this.mm(b.proj, att, x1, rows, { residual: xs, gate: { mod, bias: b.modBias, seq, chunk: 2 } })
    await this.rmsNorm(x1, h, rows, DT, { mod: modOf(3, 4) })
    await this.mm(b.w12, h, w12, rows)
    await this.run("tessera-conv-swiglu", {
      program: this.gpu.program(K.TESSERA_CONV_SWIGLU),
      inputs: { uX: w12, uK: b.kernel, uB: b.convBias },
      uniforms: { uSeq: { int: seq }, uPrefix: { int: PRE }, uGrid: { int: grid }, uHidden: { int: C.hidden / 4 } },
      output: gated,
      rect: { x: 0, y: 0, width: C.hidden / 4, height: rows },
      work: 20
    })
    await this.mm(b.w3, gated, xs, rows, { residual: x1, gate: { mod, bias: b.modBias, seq, chunk: 5 } })
  }

  // ------------------------------------------------------------ CLIP

  /** CLIP's text tower on `ids` (24 of them): last_hidden_state [24][width] and pooler_output. */
  async encodeText(ids: ArrayLike<number>, signal?: AbortSignal) {
    this.signal = signal
    const { width, tokens, positions } = this.clip
    const L = ids.length
    const x0 = new Float32Array(L * width)
    for (let t = 0; t < L; t++) {
      const id = ids[t]!
      if (!(id >= 0 && id < tokens.vocab)) throw new Error(`Tessera: token ${id} is outside CLIP's vocabulary.`)
      for (let d = 0; d < width; d++) x0[t * width + d] = tokens.q[id * width + d]! * tokens.scale[id]! + positions[t * width + d]!
    }
    const x = this.upload("clip-x", x0, L)
    const x1 = this.tex("clip-x1", DT, L)
    const h = this.tex("clip-h", DT, L)
    const qkv = this.tex("clip-qkv", 3 * DT, L)
    const att = this.tex("clip-att", DT, L)
    const fc = this.tex("clip-fc", this.clip.layers[0]!.fc1.N / 4, L)
    for (const l of this.clip.layers) {
      await this.layerNorm(x, l.ln1, h, L, this.clip.eps)
      await this.mm(l.q, h, qkv, L, { out0: 0 })
      await this.mm(l.k, h, qkv, L, { out0: DT })
      await this.mm(l.v, h, qkv, L, { out0: 2 * DT })
      await this.attend(qkv, 0, qkv, DT, qkv, 2 * DT, att, L, L, true)
      await this.mm(l.out, att, x1, L, { residual: x })
      await this.layerNorm(x1, l.ln2, h, L, this.clip.eps)
      await this.mm(l.fc1, h, fc, L, { act: "quickGelu" })
      await this.mm(l.fc2, fc, x, L, { residual: x1 })
    }
    await this.layerNorm(x, this.clip.final, h, L, this.clip.eps)
    const states = await this.gpu.read(h, { x: 0, y: 0, width: DT, height: L })
    let eos = 0
    for (let t = 1; t < L; t++) if (ids[t]! > ids[eos]!) eos = t
    return { states, pooled: states.slice(eos * width, (eos + 1) * width) }
  }

  // ------------------------------------------------------------ the denoiser

  /** The prompt's text rows and pooled condition, for the steps that follow (text_states and text_pooled of one prompt). */
  async setText(states: Float32Array, pooled: Float32Array, signal?: AbortSignal) {
    this.signal = signal
    const L = C.textLen
    const s = this.upload("text-states", states, L)
    const n = this.tex("text-normed", DT, L)
    await this.rmsNorm(s, n, L, DT, { weight: this.den.textNorm })
    const rows = this.tex("text-rows", DT, L)
    await this.mm(this.den.textProj, n, rows, L)
    const p = this.upload("pooled", pooled, 1)
    const pn = this.tex("pooled-normed", DT, 1)
    await this.rmsNorm(p, pn, 1, DT, { weight: this.den.poolNorm })
    const ph = this.tex("pooled-hidden", DT, 1)
    await this.mm(this.den.pool1, pn, ph, 1, { act: "silu" })
    const c = this.tex("cond", DT, 1)
    await this.mm(this.den.pool3, ph, c, 1)
    this.text = { rows, cond: await this.gpu.read(c, { x: 0, y: 0, width: DT, height: 1 }) }
  }

  /**
   * The denoiser's logits, [batch][256][vocab], for `batch` canvases:
   * rows below `count` with the prompt set last, the rest without it.
   * `scProbs` is the last step's distribution ([batch][256][vocab]), or
   * null for none (zeros).
   */
  async denoise(
    tokens: Int32Array,
    maskRatio: Float32Array,
    scProbs: Float32Array | null,
    batch: number,
    count: number,
    signal?: AbortSignal
  ): Promise<Float32Array> {
    if (!this.text) throw new Error("Tessera: no prompt was set (setText).")
    this.signal = signal
    const d = this.den
    // The conditioning: g = c + ratio MLP(mask ratio's features); then adaLN's modulations.
    const feats = new Float32Array(batch * 16)
    const cMix = new Float32Array(batch * D)
    for (let b = 0; b < batch; b++) {
      feats.set(ratioFeatures(maskRatio[b]!), b * 16)
      cMix.set(b < count ? this.text.cond : d.nullCond, b * D)
    }
    const ratioIn = this.upload("ratio-in", feats, batch)
    const ratioH = this.tex("ratio-h", DT, batch)
    await this.mm(d.ratio0, ratioIn, ratioH, batch, { act: "silu" })
    const cond = this.upload("cond-mix", cMix, batch)
    const g = this.tex("g", DT, batch)
    await this.mm(d.ratio2, ratioH, g, batch, { residual: cond })
    const mod = this.tex("mod", (6 * D) / 4, batch)
    await this.mm(d.tBlock, g, mod, batch, { siluIn: true })
    const fm = this.tex("final-mod", (2 * D) / 4, batch)
    await this.mm(d.finalMod, g, fm, batch, { siluIn: true })
    // The image rows' self-conditioning.
    const scRows = batch * P
    const sc = this.tex("sc", DT, scRows)
    if (scProbs) {
      const scIn = this.upload("sc-in", scProbs, scRows)
      await this.mm(d.scProj, scIn, sc, scRows)
    } else this.gpu.zero(sc)
    // The step's input rows.
    const rows = batch * SEQ
    const tok = this.upload("tokens", Float32Array.from(tokens), batch)
    let xs = this.tex("xs", DT, rows)
    await this.run("tessera-assemble", {
      program: this.gpu.program(K.TESSERA_ASSEMBLE),
      inputs: { uReg: d.registers, uText: this.text.rows, uNullText: d.nullText, uEmbed: d.embed, uPos: d.pos, uSc: sc, uTok: tok },
      uniforms: {
        uSeq: { int: SEQ },
        uRegisters: { int: C.registers },
        uTextLen: { int: C.textLen },
        uCount: { int: count },
        uMask: { int: C.vocab },
        uPixels: { int: P }
      },
      output: xs,
      rect: { x: 0, y: 0, width: DT, height: rows },
      work: 3
    })
    // Pixel level.
    for (const b of d.blocksIn) await this.block(b, xs, rows, SEQ, C.grid, d.ropeIn, mod)
    // The 2 × 2 merge, the core, and the split back with the skip.
    const coreTokens = batch * CORE_GRID * CORE_GRID
    const wide = (C.patch * C.patch * D) / 4
    const merged = this.tex("merged", wide, coreTokens)
    await this.run("tessera-merge", {
      program: this.gpu.program(K.TESSERA_MERGE),
      inputs: { uX: xs },
      uniforms: { uSeq: { int: SEQ }, uFrom: { int: PRE }, uGrid: { int: C.grid }, uDimTexels: { int: DT } },
      output: merged,
      rect: { x: 0, y: 0, width: wide, height: coreTokens }
    })
    const mergedN = this.tex("merged-normed", wide, coreTokens)
    await this.rmsNorm(merged, mergedN, coreTokens, wide, { weight: d.downNorm })
    const down = this.tex("down", DT, coreTokens)
    await this.mm(d.down, mergedN, down, coreTokens)
    const coreRows = batch * CORE_SEQ
    const core = this.tex("core", DT, coreRows)
    await this.run("tessera-join", {
      program: this.gpu.program(K.TESSERA_JOIN),
      inputs: { uPre: xs, uImg: down },
      uniforms: { uSeq: { int: CORE_SEQ }, uPreSeq: { int: SEQ }, uPrefix: { int: PRE }, uN: { int: CORE_GRID * CORE_GRID } },
      output: core,
      rect: { x: 0, y: 0, width: DT, height: coreRows }
    })
    for (const b of d.blocksCore) await this.block(b, core, coreRows, CORE_SEQ, CORE_GRID, d.ropeCore, mod)
    const coreImg = this.tex("core-img", DT, coreTokens)
    await this.takeRows(core, coreImg, coreTokens, CORE_SEQ, PRE, CORE_GRID * CORE_GRID)
    const up = this.tex("up", wide, coreTokens)
    await this.mm(d.up, coreImg, up, coreTokens)
    const next = this.tex("xs-next", DT, rows)
    await this.run("tessera-unmerge", {
      program: this.gpu.program(K.TESSERA_UNMERGE),
      inputs: { uCore: core, uSkip: xs, uUp: up },
      uniforms: { uSeq: { int: SEQ }, uCoreSeq: { int: CORE_SEQ }, uPrefix: { int: PRE }, uGrid: { int: C.grid }, uDimTexels: { int: DT } },
      output: next,
      rect: { x: 0, y: 0, width: DT, height: rows }
    })
    // (The two swap: the next step's xs is this step's xs-next.)
    this.work.set("xs", next)
    this.work.set("xs-next", xs)
    xs = next
    for (const b of d.blocksOut) await this.block(b, xs, rows, SEQ, C.grid, d.ropeIn, mod)
    // The head, on the image rows.
    const img = this.tex("img", DT, scRows)
    await this.takeRows(xs, img, scRows, SEQ, PRE, P)
    const imgN = this.tex("img-normed", DT, scRows)
    await this.rmsNorm(img, imgN, scRows, DT, { mod: { mod: fm, seq: P, shift: 0, scale: 1 } })
    const logits = this.tex("logits", C.vocab / 4, scRows)
    await this.mm(d.head, imgN, logits, scRows)
    return this.gpu.read(logits, { x: 0, y: 0, width: C.vocab / 4, height: scRows })
  }

  private async takeRows(x: Texture, out: Texture, rows: number, seq: number, from: number, n: number) {
    await this.run("tessera-take", {
      program: this.gpu.program(K.TESSERA_TAKE_ROWS),
      inputs: { uX: x },
      uniforms: { uSeq: { int: seq }, uFrom: { int: from }, uN: { int: n } },
      output: out,
      rect: { x: 0, y: 0, width: DT, height: rows }
    })
  }
}
