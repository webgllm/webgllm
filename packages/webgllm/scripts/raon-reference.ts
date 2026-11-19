/**
 * Raon-1 nano on the CPU, a token at a time, straight from its GGUF: a
 * reference for the engine's passes, written from KRAFTON's PyTorch model
 * (modeling_raon1_nano.py) and the folded tensor layout their llama.cpp
 * converter writes (conversion/raon1_nano.py, src/models/raon1nano.cpp).
 * Every weight is dequantized as llama.cpp does; sums are in doubles.
 *
 *   node scripts/raon-reference.mjs <model.gguf> <tokenizer dir> [--tokens n] [--out file.json] [--prompt text]
 *
 * Prints the reply it writes greedily and, with --out, the prompt's ids,
 * each position's likeliest tokens and logits, and the last hidden state.
 */
import { closeSync, openSync, readFileSync, readSync, statSync, writeFileSync } from "node:fs"
import { join } from "node:path"
import { GgmlType, readGgufHeader, type GgufHeader, type GgufTensorInfo } from "../src/gguf.js"
import { bytesToFloats, halvesToFloats } from "../src/half.js"
import { dequantizeQ4_K, dequantizeQ6_K, dequantizeQ8_0 } from "../src/repack.js"
import { loadTokenizer } from "../src/tokenizer.js"
import type { ByteSource } from "../src/weights.js"

const sigmoid = (x: number) => 1 / (1 + Math.exp(-x))
const silu = (x: number) => x * sigmoid(x)
const geluTanh = (x: number) => 0.5 * x * (1 + Math.tanh(Math.sqrt(2 / Math.PI) * (x + 0.044715 * x * x * x)))

/** rows × K weights · x. */
function matvec(w: Float32Array, rows: number, K: number, x: ArrayLike<number>, x0 = 0): Float64Array {
  const out = new Float64Array(rows)
  for (let r = 0; r < rows; r++) {
    let acc = 0
    const p = r * K
    for (let k = 0; k < K; k++) acc += w[p + k] * x[x0 + k]
    out[r] = acc
  }
  return out
}

/** x / sqrt(mean(x²) + eps) · weight, over [x0, x0 + n). */
function rmsNorm(x: ArrayLike<number>, x0: number, n: number, eps: number, weight?: ArrayLike<number>): Float64Array {
  let ss = 0
  for (let i = 0; i < n; i++) ss += x[x0 + i] * x[x0 + i]
  const inv = 1 / Math.sqrt(ss / n + eps)
  const out = new Float64Array(n)
  for (let i = 0; i < n; i++) out[i] = x[x0 + i] * inv * (weight ? weight[i] : 1)
  return out
}

export class RaonReference {
  readonly meta: Map<string, unknown>
  private cache = new Map<string, Float32Array>()
  readonly D: number
  readonly layers: number
  readonly hc: number
  readonly ple: number
  readonly eps: number
  readonly epsLatent: number
  readonly hcEps: number
  readonly lowerBound: number
  readonly rdaHeads: number
  readonly rdaDim: number
  readonly rank: number
  readonly conv: number
  readonly kvRank: number
  readonly extra: number
  readonly nk: number
  readonly nv: number
  readonly vocab: number
  readonly isRda: Array<boolean>
  readonly heads: Array<number>
  /** Per RDA layer: the last conv-1 raw q|k|v rows, oldest first, and each head's state [v][k]. */
  private convState: Array<Array<Float64Array>> = []
  private state: Array<Float64Array | null> = []
  /** Per MLA layer: each position's latent key row (kvRank + extra). */
  private keys: Array<Array<Float64Array>> = []
  position = 0
  /** Set to see each layer's streams after it (layer, 4·D values). */
  probe?: (layer: number, name: string, values: Float64Array) => void

  constructor(
    private fd: number,
    private header: GgufHeader
  ) {
    const meta = header.metadata as Map<string, unknown>
    this.meta = meta
    const arch = meta.get("general.architecture")
    if (arch !== "raon1nano") throw new Error(`not a raon1nano GGUF: ${String(arch)}`)
    const num = (key: string) => Number(meta.get(`raon1nano.${key}`))
    this.D = num("embedding_length")
    this.layers = num("block_count")
    this.hc = num("hyper_connection.count")
    this.ple = num("embedding_length_per_layer_input")
    this.eps = Number(num("attention.layer_norm_rms_epsilon").toPrecision(7))
    this.epsLatent = Number(num("attention.latent_norm_rms_epsilon").toPrecision(7))
    this.hcEps = Number(num("hyper_connection.epsilon").toPrecision(7))
    this.lowerBound = num("kda.gate_lower_bound")
    this.rdaDim = num("kda.head_dim")
    this.rank = num("ssm.time_step_rank")
    this.conv = num("ssm.conv_kernel")
    this.kvRank = num("attention.kv_lora_rank")
    this.extra = num("rope.dimension_count")
    this.nk = num("attention.key_length_mla")
    this.nv = num("attention.value_length_mla")
    this.vocab = num("vocab_size")
    this.heads = (meta.get("raon1nano.attention.head_count") as Array<number>).slice(0, this.layers)
    this.isRda = (meta.get("raon1nano.attention.head_count_kv") as Array<number>).slice(0, this.layers).map((kv) => kv === 0)
    this.rdaHeads = this.heads[this.isRda.indexOf(true)]
    this.reset()
  }

  static async open(path: string) {
    const fd = openSync(path, "r")
    const size = statSync(path).size
    const source: ByteSource = {
      size,
      read: async (offset, length) => {
        const bytes = new Uint8Array(length)
        readSync(fd, bytes, 0, length, offset)
        return bytes
      }
    }
    return new RaonReference(fd, await readGgufHeader(source, { chunkSize: 8 << 20 }))
  }

  close() {
    closeSync(this.fd)
  }

  reset() {
    this.position = 0
    this.convState = Array.from({ length: this.layers }, () => [])
    this.state = Array.from({ length: this.layers }, () => null)
    this.keys = Array.from({ length: this.layers }, () => [])
  }

  private info(name: string): GgufTensorInfo {
    const info = this.header.tensors.get(name)
    if (!info) throw new Error(`missing tensor ${name}`)
    return info
  }

  private bytes(info: GgufTensorInfo, offset = 0, length = info.byteLength) {
    const bytes = new Uint8Array(length)
    readSync(this.fd, bytes, 0, length, info.offset + offset)
    return bytes
  }

  /** Rows [row0, row0 + rows) of a tensor (ggml dims[0] values a row), as floats. */
  rows(name: string, row0 = 0, rows?: number): Float32Array {
    const info = this.info(name)
    const K = info.dims[0]
    const total = info.dims.slice(1).reduce((a, b) => a * b, 1)
    const n = rows ?? total
    const rowBytes = info.byteLength / total
    const bytes = this.bytes(info, row0 * rowBytes, n * rowBytes)
    const out = new Float32Array(n * K)
    switch (info.type) {
      case GgmlType.F32: out.set(bytesToFloats(bytes)); break
      case GgmlType.F16: out.set(halvesToFloats(bytes)); break
      case GgmlType.Q8_0: dequantizeQ8_0(bytes, n, K, out); break
      case GgmlType.Q6_K: dequantizeQ6_K(bytes, n, K, out); break
      case GgmlType.Q4_K: dequantizeQ4_K(bytes, n, K, out); break
      default: throw new Error(`${name}: tensor type ${info.type} isn't read here`)
    }
    return out
  }

  /** A whole tensor, kept. */
  private w(name: string): Float32Array {
    let values = this.cache.get(name)
    if (!values) this.cache.set(name, (values = this.rows(name)))
    return values
  }

  /** An mHC site: the branch's input, and how its output goes back into the streams. */
  private hcPre(S: Float64Array, blk: string, site: "attn" | "ffn", norm: string) {
    const { D, hc } = this
    const nmix = 2 * hc + (hc - 1) * (hc - 1)
    const flat = rmsNorm(S, 0, hc * D, this.hcEps)
    const mixes = matvec(this.w(`${blk}hc_${site}_fn.weight`), nmix, hc * D, flat)
    const base = this.w(`${blk}hc_${site}_base.weight`)
    for (let i = 0; i < nmix; i++) mixes[i] += base[i]
    const chart = this.w(`${blk}hc_${site}_chart.weight`)
    const pre = Array.from({ length: hc }, (_, i) => sigmoid(mixes[i]))
    const post = Array.from({ length: hc }, (_, i) => sigmoid(mixes[hc + i]))
    const hc2 = hc * hc
    const e = new Float64Array(hc2)
    let rho = -Infinity
    for (let j = 0; j < hc2; j++) {
      let acc = 0
      for (let i = 0; i < nmix; i++) acc += chart[j * nmix + i] * mixes[i]
      e[j] = acc
      rho = Math.max(rho, acc * chart[nmix * hc2 + j])
    }
    rho += 1
    // comb[dst + hc · src]
    const comb = new Float64Array(hc2)
    for (let j = 0; j < hc2; j++) comb[j] = e[j] / rho + chart[nmix * hc2 + hc2 + j]
    const agg = new Float64Array(D)
    for (let i = 0; i < hc; i++) for (let c = 0; c < D; c++) agg[c] += pre[i] * S[i * D + c]
    const cur = rmsNorm(agg, 0, D, this.eps, this.w(`${blk}${norm}.weight`))
    return { cur, post, comb }
  }

  private hcPost(S: Float64Array, branch: Float64Array, post: Array<number>, comb: Float64Array) {
    const { D, hc } = this
    const out = new Float64Array(hc * D)
    for (let dst = 0; dst < hc; dst++) {
      for (let c = 0; c < D; c++) {
        let acc = post[dst] * branch[c]
        for (let src = 0; src < hc; src++) acc += comb[dst + hc * src] * S[src * D + c]
        out[dst * D + c] = acc
      }
    }
    return out
  }

  private rda(il: number, blk: string, cur: Float64Array) {
    const { D, rdaDim: hd, rank, conv } = this
    const H = this.heads[il]
    const inner = H * hd
    const qkvg = matvec(this.w(`${blk}attn_qkv.weight`), 4 * inner, D, cur)
    // The depthwise causal convolution over q|k|v, then silu.
    const taps = this.w(`${blk}ssm_conv1d.weight`)
    const past = this.convState[il]
    const raw = qkvg.slice(0, 3 * inner)
    const qkv = new Float64Array(3 * inner)
    for (let c = 0; c < 3 * inner; c++) {
      let acc = taps[c * conv + conv - 1] * raw[c]
      for (let j = 0; j < conv - 1; j++) {
        const row = past[past.length - (conv - 1) + j]
        if (row) acc += taps[c * conv + j] * row[c]
      }
      qkv[c] = silu(acc)
    }
    past.push(raw)
    if (past.length > conv - 1) past.shift()
    const fab = matvec(this.w(`${blk}ssm_f_a.weight`), rank + 2 * H, D, cur)
    const g1 = matvec(this.w(`${blk}ssm_f_b.weight`), inner, rank, fab)
    const dt = this.w(`${blk}ssm_dt.bias`)
    const state = (this.state[il] ??= new Float64Array(H * hd * hd))
    const out = new Float64Array(inner)
    for (let h = 0; h < H; h++) {
      const q = new Float64Array(hd), k = new Float64Array(hd)
      let qq = 0, kk = 0
      for (let i = 0; i < hd; i++) {
        qq += qkv[h * hd + i] ** 2
        kk += qkv[inner + h * hd + i] ** 2
      }
      const qn = 1 / Math.max(Math.sqrt(qq), 1e-6) / Math.sqrt(hd)
      const kn = 1 / Math.max(Math.sqrt(kk), 1e-6)
      for (let i = 0; i < hd; i++) {
        q[i] = qkv[h * hd + i] * qn
        k[i] = qkv[inner + h * hd + i] * kn
      }
      const beta = sigmoid(fab[rank + h])
      const ratio = sigmoid(fab[rank + H + h]) / beta
      const S = state.subarray(h * hd * hd, (h + 1) * hd * hd)
      const decay = new Float64Array(hd)
      for (let i = 0; i < hd; i++) decay[i] = Math.exp(this.lowerBound * sigmoid(g1[h * hd + i] + dt[h * hd + i]))
      for (let v = 0; v < hd; v++) {
        let pred = 0
        for (let i = 0; i < hd; i++) {
          S[v * hd + i] *= decay[i]
          pred += S[v * hd + i] * k[i]
        }
        const err = beta * (qkv[2 * inner + h * hd + v] * ratio - pred)
        let o = 0
        for (let i = 0; i < hd; i++) {
          S[v * hd + i] += err * k[i]
          o += S[v * hd + i] * q[i]
        }
        out[h * hd + v] = o
      }
    }
    const norm = this.w(`${blk}ssm_norm.weight`)
    const gated = new Float64Array(inner)
    for (let h = 0; h < H; h++) {
      const n = rmsNorm(out, h * hd, hd, this.eps, norm)
      for (let i = 0; i < hd; i++) gated[h * hd + i] = n[i] * sigmoid(qkvg[3 * inner + h * hd + i])
    }
    this.probe?.(il, "rda_gated", gated)
    return matvec(this.w(`${blk}attn_output.weight`), D, inner, gated)
  }

  private mla(il: number, blk: string, cur: Float64Array) {
    const { D, kvRank, extra, nk, nv } = this
    const H = this.heads[il]
    const nn = nk - extra
    const qRank = this.info(`${blk}attn_q_a.weight`).dims[1]
    const qa = rmsNorm(matvec(this.w(`${blk}attn_q_a.weight`), qRank, D, cur), 0, qRank, this.epsLatent, this.w(`${blk}attn_q_a_norm.weight`))
    const q = matvec(this.w(`${blk}attn_q_b.weight`), H * nk, qRank, qa)
    const kv = matvec(this.w(`${blk}attn_kv_a_mqa.weight`), kvRank + extra, D, cur)
    const latent = rmsNorm(kv, 0, kvRank, this.epsLatent, this.w(`${blk}attn_kv_a_norm.weight`))
    const key = new Float64Array(kvRank + extra)
    key.set(latent)
    for (let i = 0; i < extra; i++) key[kvRank + i] = kv[kvRank + i]
    const keys = this.keys[il]
    keys.push(key)
    const kb = this.w(`${blk}attn_k_b.weight`)
    const vb = this.w(`${blk}attn_v_b.weight`)
    const gate = matvec(this.w(`${blk}attn_gate.weight`), H * nv, D, cur)
    const out = new Float64Array(H * nv)
    const scale = 1 / Math.sqrt(nk)
    for (let h = 0; h < H; h++) {
      // The query absorbed into the latent: k_b[h] (kvRank × nn) · q_nope, then the shared dims.
      const qh = new Float64Array(kvRank + extra)
      qh.set(matvec(kb.subarray(h * kvRank * nn, (h + 1) * kvRank * nn), kvRank, nn, q, h * nk))
      for (let i = 0; i < extra; i++) qh[kvRank + i] = q[h * nk + nn + i]
      const scores = keys.map((k) => {
        let acc = 0
        for (let i = 0; i < kvRank + extra; i++) acc += qh[i] * k[i]
        return acc * scale
      })
      const max = Math.max(...scores)
      const weights = scores.map((s) => Math.exp(s - max))
      const sum = weights.reduce((a, b) => a + b, 0)
      const mix = new Float64Array(kvRank)
      keys.forEach((k, t) => {
        const p = weights[t] / sum
        for (let i = 0; i < kvRank; i++) mix[i] += p * k[i]
      })
      const v = matvec(vb.subarray(h * nv * kvRank, (h + 1) * nv * kvRank), nv, kvRank, mix)
      for (let i = 0; i < nv; i++) out[h * nv + i] = v[i] * sigmoid(gate[h * nv + i])
    }
    this.probe?.(il, "mla_gated", out)
    return matvec(this.w(`${blk}attn_output.weight`), D, H * nv, out)
  }

  /** Reads one token; returns the final-normed hidden state (the head's input). */
  hidden(token: number): Float64Array {
    const { D, hc, ple, layers } = this
    const e = this.rows("token_embd.weight", token, 1)
    // Per-layer inputs: (RMSNorm(model_proj · e) + sqrt(ple) · table[token]) / sqrt(2), a layer's `ple` values together.
    const table = this.rows("per_layer_token_embd.weight", token, 1)
    const proj = matvec(this.w("per_layer_model_proj.weight"), ple * layers, D, e)
    const projNorm = this.w("per_layer_proj_norm.weight")
    const inputs = new Float64Array(ple * layers)
    for (let l = 0; l < layers; l++) {
      const n = rmsNorm(proj, l * ple, ple, this.eps, projNorm)
      for (let i = 0; i < ple; i++) inputs[l * ple + i] = (n[i] + Math.sqrt(ple) * table[l * ple + i]) / Math.SQRT2
    }
    this.probe?.(-1, "per_layer_inputs", inputs)
    let S = new Float64Array(hc * D)
    for (let i = 0; i < hc; i++) S.set(e, i * D)
    for (let il = 0; il < layers; il++) {
      const blk = `blk.${il}.`
      let site = this.hcPre(S, blk, "attn", "attn_norm")
      this.probe?.(il, "attn_in", site.cur)
      const branch = this.isRda[il] ? this.rda(il, blk, site.cur) : this.mla(il, blk, site.cur)
      this.probe?.(il, "attn_out", branch)
      S = this.hcPost(S, branch, site.post, site.comb)
      this.probe?.(il, "after_attn", S)
      site = this.hcPre(S, blk, "ffn", "ffn_norm")
      const F = this.info(`${blk}ffn_down.weight`).dims[0]
      const gu = matvec(this.w(`${blk}ffn_up.weight`), 2 * F, D, site.cur)
      const act = new Float64Array(F)
      for (let i = 0; i < F; i++) act[i] = silu(gu[i]) * gu[F + i]
      const down = matvec(this.w(`${blk}ffn_down.weight`), D, F, act)
      this.probe?.(il, "ffn_out", down)
      S = this.hcPost(S, down, site.post, site.comb)
      this.probe?.(il, "after_ffn", S)
      // The layer's per-layer input: streams += RMSNorm(proj(gelu(gate · stream sum) · input)).
      const sum = new Float64Array(D)
      for (let i = 0; i < hc; i++) for (let c = 0; c < D; c++) sum[c] += S[i * D + c]
      const g = matvec(this.w(`${blk}inp_gate.weight`), ple, D, sum)
      for (let i = 0; i < ple; i++) g[i] = geluTanh(g[i]) * inputs[il * ple + i]
      const delta = rmsNorm(matvec(this.w(`${blk}proj.weight`), D, ple, g), 0, D, this.eps, this.w(`${blk}post_norm.weight`))
      for (let i = 0; i < hc; i++) for (let c = 0; c < D; c++) S[i * D + c] += delta[c]
      this.probe?.(il, "l_out", S)
    }
    const mean = new Float64Array(D)
    for (let i = 0; i < hc; i++) for (let c = 0; c < D; c++) mean[c] += S[i * D + c] / hc
    this.position++
    return rmsNorm(mean, 0, D, this.eps, this.w("output_norm.weight"))
  }

  /** The tied head: every token's logit for a hidden state. */
  logits(hidden: Float64Array): Float32Array {
    const { D, vocab } = this
    const out = new Float32Array(vocab)
    const chunk = 4096
    for (let r = 0; r < vocab; r += chunk) {
      const n = Math.min(chunk, vocab - r)
      out.set(matvec(this.rows("token_embd.weight", r, n), n, D, hidden), r)
    }
    return out
  }
}

function top(logits: Float32Array, n: number) {
  const best: Array<number> = []
  for (let i = 0; i < logits.length; i++) {
    if (best.length < n || logits[i] > logits[best[best.length - 1]]) {
      best.push(i)
      best.sort((a, b) => logits[b] - logits[a])
      if (best.length > n) best.pop()
    }
  }
  return best
}

async function main() {
  const args = process.argv.slice(2)
  const flag = (name: string) => {
    const at = args.indexOf(`--${name}`)
    return at < 0 ? undefined : args.splice(at, 2)[1]
  }
  const tokens = Number(flag("tokens") ?? 16)
  const outFile = flag("out")
  const text = flag("prompt") ?? "What is the capital of France? Answer in one short sentence."
  const thinking = flag("thinking") === "1"
  const flag2 = { layers: flag("layers"), logits: flag("logits") }
  const [modelPath, tokenizerDir] = args
  if (!modelPath || !tokenizerDir) throw new Error("usage: raon-reference <model.gguf> <tokenizer dir> [--tokens n] [--out file] [--prompt text]")
  const json = (name: string) => JSON.parse(readFileSync(join(tokenizerDir, name), "utf8"))
  const tokenizer = loadTokenizer({
    tokenizerJson: json("tokenizer.json"),
    tokenizerConfig: json("tokenizer_config.json"),
    chatTemplate: readFileSync(join(tokenizerDir, "chat_template.jinja"), "utf8")
  })
  const prompt = tokenizer.applyChatTemplate([{ role: "user", content: text }], { addGenerationPrompt: true, enableThinking: thinking } as never)
  const ids = tokenizer.encode(prompt)
  console.log(JSON.stringify(prompt))
  console.log(ids.join(" "))
  const model = await RaonReference.open(modelPath)
  const eos = new Set([Number(model.meta.get("tokenizer.ggml.eos_token_id")), Number(model.meta.get("tokenizer.ggml.padding_token_id"))])
  const record: { prompt: string; ids: Array<number>; steps: Array<{ token: number; top: Array<[number, number]> }>; written: Array<number>; hidden?: Array<number> } = {
    prompt, ids, steps: [], written: []
  }
  const started = performance.now()
  let hidden = new Float64Array(0)
  // --layers: the last prompt token's tensors a layer, by name, as float32 (see the engine's probe names).
  const layersFile = flag2.layers
  const seen: Record<string, Array<number>> = {}
  for (const [i, id] of ids.entries()) {
    if (layersFile && i === ids.length - 1)
      model.probe = (layer, name, values) => {
        ;(seen[name] ??= []).push(...values)
        void layer
      }
    hidden = model.hidden(id)
    if (i === 0) console.log(`first token ${((performance.now() - started) / 1000).toFixed(1)} s (weights dequantized)`)
  }
  model.probe = undefined
  if (layersFile) {
    const names = Object.keys(seen)
    const index: Record<string, { offset: number; length: number }> = {}
    let offset = 0
    for (const name of names) {
      index[name] = { offset, length: seen[name].length }
      offset += seen[name].length
    }
    const all = new Float32Array(offset)
    for (const name of names) all.set(seen[name], index[name].offset)
    writeFileSync(layersFile, new Uint8Array(all.buffer))
    writeFileSync(`${layersFile}.json`, JSON.stringify(index))
  }
  record.hidden = Array.from(hidden)
  let reply = ""
  for (let n = 0; n < tokens; n++) {
    const logits = model.logits(hidden)
    if (n === 0 && flag2.logits) writeFileSync(flag2.logits, new Uint8Array(logits.buffer))
    const best = top(logits, 5)
    record.steps.push({ token: best[0], top: best.map((t) => [t, logits[t]]) })
    record.written.push(best[0])
    reply = tokenizer.decode(record.written)
    console.log(n, best.map((t) => `${JSON.stringify(tokenizer.decode([t]))} ${logits[t].toFixed(3)}`).join("  "))
    if (eos.has(best[0])) break
    hidden = model.hidden(best[0])
  }
  console.log(`reply: ${JSON.stringify(reply)}  (${((performance.now() - started) / 1000).toFixed(1)} s)`)
  if (outFile) writeFileSync(outFile, JSON.stringify(record))
  model.close()
}

if (process.argv[1]?.includes("raon-reference")) await main()
