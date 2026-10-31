/**
 * Correctness check, in a worker: runs the engine over the reference
 * prompt and compares every layer's output and the logits with llama.cpp's
 * (served from /ref/, see bench/README.md).
 */
import { Engine, TUNING } from "../src/engine.js"
import * as K from "../src/gpu/kernels.js"
import type { LayerProbe } from "../src/engine.js"
import { httpSource } from "../src/byte-source.js"
import { headPermutation, loadGguf } from "../src/load-gguf.js"
import { loadOnnx } from "../src/load-onnx.js"
import { loadTokenizer } from "../src/tokenizer.js"
import { lookupDraft } from "../src/draft.js"

type Manifest = {
  files: Array<{ file: string; name?: string; shape: Array<number> }>
}

const log = (text: string) => postMessage({ type: "log", text })

/** Where the ONNX build's files are served: /onnx-web/ or /onnx-community/. */
let onnxDir = "/onnx-web"

/** Where the reference tensors are: /ref/ (llama.cpp, GGUF) or /ref-onnx/ (onnxruntime). */
let refBase = "/ref/"

/** The model's weights: the GGUF from /models/, or the ONNX build from /onnx-web/. */
async function modelPieces(params: {
  model: string
  format?: "gguf" | "onnx"
}) {
  if (params.format !== "onnx")
    return loadGguf(await httpSource(`/models/${params.model}`))
  const config = (await (
    await fetch(`${onnxDir}/config.json`)
  ).json()) as Record<string, unknown>
  return loadOnnx({
    config,
    graph: await httpSource(`${onnxDir}/onnx/model_q4f16.onnx`),
    data: await httpSource(`${onnxDir}/onnx/model_q4f16.onnx_data`)
  })
}

/** Rows [from, from + rows) of a float32 [n][dims] reference file. */
async function refRows(file: string, dims: number, from: number, rows: number) {
  const res = await fetch(`${refBase}${file}`, {
    headers: {
      Range: `bytes=${from * dims * 4}-${(from + rows) * dims * 4 - 1}`
    }
  })
  if (!res.ok) throw new Error(`/ref/${file}: ${res.status}`)
  return new Float32Array(await res.arrayBuffer())
}

/** How far `got` is from `want`: relative RMS error, max abs difference, cosine. */
function compare(got: Float32Array, want: Float32Array) {
  let dd = 0
  let ww = 0
  let gw = 0
  let gg = 0
  let max = 0
  let worst = 0
  for (let i = 0; i < want.length; i++) {
    const d = got[i] - want[i]
    dd += d * d
    ww += want[i] * want[i]
    gg += got[i] * got[i]
    gw += got[i] * want[i]
    if (Math.abs(d) > max || Number.isNaN(d)) {
      max = Math.abs(d)
      worst = i
    }
  }
  return {
    rel: Math.sqrt(dd / ww),
    max,
    worst,
    cos: gw / Math.sqrt(gg * ww)
  }
}

const fmt = (r: ReturnType<typeof compare>) =>
  `rel ${r.rel.toExponential(2)}  cos ${r.cos.toFixed(6)}  max ${r.max.toExponential(2)} @${r.worst}`

function softmax(logits: Float32Array) {
  let mx = -Infinity
  for (const v of logits) mx = Math.max(mx, v)
  let sum = 0
  const p = new Float64Array(logits.length)
  for (let i = 0; i < logits.length; i++) {
    p[i] = Math.exp(logits[i] - mx)
    sum += p[i]
  }
  for (let i = 0; i < p.length; i++) p[i] /= sum
  return p
}

function argmax(a: ArrayLike<number>) {
  let best = 0
  for (let i = 1; i < a.length; i++) if (a[i] > a[best]) best = i
  return best
}

/** Logit comparison: KL(ref || ours), top-1 agreement, top-5 overlap. */
function compareLogits(got: Float32Array, want: Float32Array) {
  const p = softmax(want)
  const q = softmax(got)
  let kl = 0
  for (let i = 0; i < p.length; i++)
    if (p[i] > 0) kl += p[i] * Math.log(p[i] / Math.max(q[i], 1e-30))
  const top = (a: ArrayLike<number>) =>
    Array.from(a, (v, i) => [v, i] as const)
      .sort((x, y) => y[0] - x[0])
      .slice(0, 5)
      .map(([, i]) => i)
  const tw = top(want)
  const tg = top(got)
  return {
    kl,
    top1: tw[0] === tg[0],
    top5: tg.filter((i) => tw.includes(i)).length,
    ...compare(got, want)
  }
}

async function run(params: {
  model: string
  format?: "gguf" | "onnx"
  ring?: boolean
  batch: number
  precision: "highp" | "mediump"
  layers: boolean
}) {
  refBase = params.format === "onnx" ? "/ref-onnx/" : "/ref/"
  const manifest = (await (
    await fetch(`${refBase}manifest.json`)
  ).json()) as Manifest
  const tokensFile = (await (await fetch(`${refBase}tokens.json`)).json()) as {
    token_ids?: Array<number>
    tokens?: Array<number>
    greedy_continuation?: Array<number> | { token_ids?: Array<number> }
  }
  const tokens = tokensFile.token_ids ?? tokensFile.tokens ?? []
  const files = new Set(manifest.files.map((f) => f.file))
  log(`${tokens.length} reference tokens, ${files.size} reference files`)

  const canvas = new OffscreenCanvas(1, 1)
  const engine = new Engine(canvas, {
    batchSize: params.batch,
    matmulPrecision: params.precision,
    ringCache: (params as { ring?: boolean }).ring !== false
  })
  log(
    `GPU: ${engine.gpu.limits.renderer}, max texture ${engine.gpu.limits.maxTextureSize}`
  )
  const t0 = performance.now()
  await engine.load(await modelPieces(params))
  log(
    `loaded in ${((performance.now() - t0) / 1000).toFixed(1)} s, ${(engine.gpu.bytes / 2 ** 20).toFixed(0)} MiB of textures`
  )
  // The per-batch buffers, largest first.
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const buffers = (engine as any).buffers as Record<string, unknown>
  // Shared textures are counted once, under all their names.
  const byId = new Map<number, { names: Array<string>; bytes: number }>()
  for (const [name, value] of Object.entries(buffers)) {
    if (!value) continue
    for (const t of (Array.isArray(value) ? value : [value]) as Array<{
      id: number
      bytes: number
    }>) {
      const entry = byId.get(t.id) ?? { names: [], bytes: t.bytes }
      if (!entry.names.includes(name)) entry.names.push(name)
      byId.set(t.id, entry)
    }
  }
  const merged = new Map<string, number>()
  for (const { names, bytes } of byId.values()) {
    const key = names.join("/")
    merged.set(key, (merged.get(key) ?? 0) + bytes)
  }
  const sizes = [...merged.entries()]
  sizes.sort((x, y) => y[1] - x[1])
  log(
    `buffers: ${sizes.map(([n, b]) => `${n} ${(b / 2 ** 20).toFixed(1)}`).join(", ")} MiB; total ${(sizes.reduce((s2, [, b]) => s2 + b, 0) / 2 ** 20).toFixed(0)} MiB`
  )

  const D = engine.config.hiddenSize
  const V = engine.config.vocabSize
  const worstByLayer = new Map<string, number>()
  const c = engine.config
  const qDims = c.numHeads * c.headDim
  const kvDims = c.numKvHeads * c.headDim
  const qPerm = headPermutation(c.numHeads, c.headDim, c.ropeDimsFull)
  const kPerm = headPermutation(c.numKvHeads, c.headDim, c.ropeDimsFull)
  /** Reference rows put in our layout: q/k dims reordered in full-attention layers. */
  const reorder = (
    want: Float32Array,
    dims: number,
    perm: Int32Array | null
  ) => {
    if (!perm) return want
    const out = new Float32Array(want.length)
    for (let r = 0; r < want.length / dims; r++) {
      for (let i = 0; i < dims; i++)
        out[r * dims + i] = want[r * dims + perm[i]]
    }
    return out
  }
  const check = async (
    key: string,
    got: Float32Array,
    file: string,
    dims: number,
    offset: number,
    rows: number,
    perm: Int32Array | null = null,
    scale = 1
  ) => {
    if (!files.has(file)) return
    const want = reorder(await refRows(file, dims, offset, rows), dims, perm)
    const g = scale === 1 ? got : got.map((v) => v * scale)
    const r = compare(g, want)
    worstByLayer.set(key, Math.max(worstByLayer.get(key) ?? 0, r.rel))
    log(`  ${key.padEnd(20)} tokens ${offset}..${offset + rows - 1}  ${fmt(r)}`)
  }
  /** Columns [from, from + width) of a rows × dims tensor. */
  const columns = (
    data: Float32Array,
    rows: number,
    dims: number,
    from: number,
    width: number
  ) => {
    const out = new Float32Array(rows * width)
    for (let r = 0; r < rows; r++)
      out.set(
        data.subarray(r * dims + from, r * dims + from + width),
        r * width
      )
    return out
  }
  // Our probe names → the reference's files.
  const probe: LayerProbe | undefined = params.layers
    ? async (layer, name, read) => {
        const type = c.layerTypes[layer]
        const full = type === "full_attention"
        const simple: Record<string, string> = {
          operator_norm: "operator_norm",
          in_proj: "conv_in_proj",
          conv: "conv_gated",
          op_out: type === "conv" ? "conv_out_proj" : "attn_out_proj",
          after_op: "resid_mid",
          ffn_out: "ffn_out",
          l_out: "l_out",
          attn_out: "attn_gated"
        }
        const wanted = [...Object.keys(simple), "qkvg", "q"].includes(name)
        if (!wanted) return
        const file = (n: string) => `${n}-${layer}.bin`
        const any =
          name === "qkvg"
            ? files.has(file("q_proj"))
            : name === "q"
              ? files.has(file("q_rope"))
              : files.has(file(simple[name]))
        if (!any) return
        const t = await read()
        const rows = t.rows
        const data = t.data.subarray(0, rows * t.dims)
        if (name === "qkvg") {
          await check(
            `q_proj-${layer}`,
            columns(data, rows, t.dims, 0, qDims),
            file("q_proj"),
            qDims,
            t.offset,
            rows,
            full ? qPerm : null
          )
          await check(
            `k_proj-${layer}`,
            columns(data, rows, t.dims, qDims, kvDims),
            file("k_proj"),
            kvDims,
            t.offset,
            rows,
            full ? kPerm : null
          )
          await check(
            `v_proj-${layer}`,
            columns(data, rows, t.dims, qDims + kvDims, kvDims),
            file("v_proj"),
            kvDims,
            t.offset,
            rows
          )
          // The reference's gate is after its sigmoid.
          const gate = columns(
            data,
            rows,
            t.dims,
            qDims + 2 * kvDims,
            c.numHeads
          ).map((v) => 1 / (1 + Math.exp(-v)))
          await check(
            `attn_gate-${layer}`,
            gate,
            file("attn_gate"),
            c.numHeads,
            t.offset,
            rows
          )
        } else if (name === "q") {
          await check(
            `q_rope-${layer}`,
            data,
            file("q_rope"),
            qDims,
            t.offset,
            rows,
            full ? qPerm : null,
            Math.sqrt(c.headDim)
          )
        } else {
          await check(
            `${simple[name]}-${layer}`,
            data,
            file(simple[name]),
            t.dims,
            t.offset,
            rows
          )
        }
      }
    : undefined

  // The prompt in batches, with logits checked at a few rows of each batch.
  engine.reset()
  let agree = 0
  let checked = 0
  let klSum = 0
  const forwardStart = performance.now()
  // Every position scored too, against the fp32 reference where there is
  // one and the main reference (as the Transformers.js quality page does).
  const hasFp32 = files.has("logits_fp32.bin")
  const all = { fp32: { kl: 0, top1: 0 }, main: { kl: 0, top1: 0 } }
  const checkedFp32 = { kl: 0, top1: 0, n: 0 }
  for (let start = 0; start < tokens.length; start += params.batch) {
    const batch = tokens.slice(start, start + params.batch)
    await engine.forward(batch, probe)
    engine.snapshot()
    const mid = [0, Math.floor(batch.length / 2), batch.length - 1]
    for (let row = 0; row < batch.length; row++) {
      const got = await engine.logits(row)
      const main = compareLogits(got, await refRows("logits.bin", V, start + row, 1))
      all.main.kl += main.kl
      if (main.top1) all.main.top1++
      if (!hasFp32) continue
      const f = compareLogits(got, await refRows("logits_fp32.bin", V, start + row, 1))
      all.fp32.kl += f.kl
      if (f.top1) all.fp32.top1++
      if (mid.includes(row)) {
        checkedFp32.kl += f.kl
        if (f.top1) checkedFp32.top1++
        checkedFp32.n++
      }
    }
    for (const row of [0, Math.floor(batch.length / 2), batch.length - 1]) {
      const got = await engine.logits(row)
      const want = await refRows("logits.bin", V, start + row, 1)
      const r = compareLogits(got, want)
      checked++
      if (r.top1) agree++
      klSum += r.kl
      log(
        `logits @${start + row}: KL ${r.kl.toExponential(2)}  top1 ${r.top1 ? "same" : "DIFFERENT"}  top5 ${r.top5}/5  ${fmt(r)}`
      )
    }
  }
  log(
    `forward ${((performance.now() - forwardStart) / 1000).toFixed(2)} s (with checks), ${engine.gpu.draws} draws`
  )

  // Writing: the reference's greedy continuation fed one token at a time,
  // each step's logits against llama.cpp's single-token decodes.
  const cont = tokensFile.greedy_continuation
  const continuation = Array.isArray(cont) ? cont : (cont?.token_ids ?? [])
  if (files.has("gen_logits.bin") && continuation.length) {
    let genAgree = 0
    let genKl = 0
    let ours = argmax(
      await engine.logits(
        tokens.length -
          1 -
          Math.floor((tokens.length - 1) / params.batch) * params.batch
      )
    )
    let sameAsRef = ours === continuation[0]
    // gen_logits row i: after feeding continuation token i. Each step also
    // checks the GPU's candidate pick (top 50 chunks, a 1.1 penalty on the
    // tokens so far) against the same pick from the full logits.
    let pickAgree = 0
    let genFp32Kl = 0
    let genFp32Top1 = 0
    const seen = new Set<number>(tokens)
    engine.setSeen(tokens, V)
    for (let i = 0; i < continuation.length; i++) {
      const got = await engine.forward([continuation[i]])
      seen.add(continuation[i])
      engine.markSeen(continuation[i])
      const picked = await engine.candidates(0, 50, 1.1)
      const penalized = got.slice()
      for (const t of seen)
        penalized[t] =
          penalized[t] < 0 ? penalized[t] * 1.1 : penalized[t] / 1.1
      const order = Array.from(penalized, (v, j) => [v, j] as const)
        .sort((a, b) => b[0] - a[0])
        .slice(0, 50)
      const pickedSet = new Map(
        Array.from(picked.ids, (id, j) => [id, picked.logits[j]] as const)
      )
      const exact = order.every(
        ([v, j]) =>
          pickedSet.has(j) &&
          Math.abs((pickedSet.get(j) ?? 0) - v) <
            1e-4 * Math.max(1, Math.abs(v))
      )
      if (exact) pickAgree++
      else log(`candidate pick differs at step ${i}`)
      const want = await refRows("gen_logits.bin", V, i, 1)
      const r = compareLogits(got, want)
      if (files.has("gen_logits_fp32.bin")) {
        const f = compareLogits(got, await refRows("gen_logits_fp32.bin", V, i, 1))
        genFp32Kl += f.kl
        if (f.top1) genFp32Top1++
      }
      if (r.top1) genAgree++
      genKl += r.kl
      ours = argmax(got)
      if (i + 1 < continuation.length && ours !== continuation[i + 1])
        sameAsRef = false
      if (i < 4 || !r.top1)
        log(
          `decode step ${i + 1}: KL ${r.kl.toExponential(2)}  top1 ${r.top1 ? "same" : "DIFFERENT"}  top5 ${r.top5}/5  ${fmt(r)}`
        )
    }
    const steps = continuation.length
    log(
      `DECODE SUMMARY: top-1 agreement ${genAgree}/${steps}, mean KL ${(genKl / steps).toExponential(2)}, greedy path ${sameAsRef ? "identical" : "diverges"} over ${continuation.length} tokens`
    )
    log(
      `CANDIDATES: GPU top-50 pick with penalty matches the full logits at ${pickAgree}/${steps} steps`
    )
    const n = tokens.length
    if (hasFp32) {
      log(
        `QUALITY WebGL vs fp32 reference: prompt (all ${n}) mean KL ${(all.fp32.kl / n).toExponential(2)}, top-1 ${all.fp32.top1}/${n}; checked ${checkedFp32.n} mean KL ${(checkedFp32.kl / checkedFp32.n).toExponential(2)}, top-1 ${checkedFp32.top1}/${checkedFp32.n}; writing (decode path) mean KL ${(genFp32Kl / steps).toExponential(2)}, top-1 ${genFp32Top1}/${steps}`
      )
    }
    log(
      `QUALITY WebGL vs main reference: prompt (all ${n}) mean KL ${(all.main.kl / n).toExponential(2)}, top-1 ${all.main.top1}/${n}; writing mean KL ${(genKl / steps).toExponential(2)}, top-1 ${genAgree}/${steps}`
    )
    // Rewind: back to the last snapshot inside the prompt (the decode steps
    // above have since moved the sliding layers' rings on), read the rest of
    // the prompt again, and check the last position's logits.
    const back = engine.rewind(tokens.length - 1)
    const again = await engine.forward(tokens.slice(back))
    const r = compareLogits(
      again,
      await refRows("logits.bin", V, tokens.length - 1, 1)
    )
    log(
      `REWIND to ${back}, ${tokens.length - back} tokens read again: KL ${r.kl.toExponential(2)}  top1 ${r.top1 ? "same" : "DIFFERENT"}  top5 ${r.top5}/5`
    )
  }
  log(
    `SUMMARY: top-1 agreement ${agree}/${checked}, mean KL ${(klSum / checked).toExponential(2)}`
  )
  if (worstByLayer.size) {
    const worst = [...worstByLayer.entries()]
      .sort((a, b) => b[1] - a[1])
      .slice(0, 5)
    log(
      `worst tensors: ${worst.map(([k, v]) => `${k} ${v.toExponential(2)}`).join(", ")}`
    )
  }

  // Speed: the whole prompt again from an empty cache, no checks.
  engine.reset()
  await engine.gpu.finish()
  const t1 = performance.now()
  const logits = await engine.forward(tokens)
  const ms = performance.now() - t1
  log(
    `SPEED: ${tokens.length} tokens in ${ms.toFixed(0)} ms = ${((tokens.length / ms) * 1000).toFixed(1)} tokens/s (batch ${params.batch}), next token ${argmax(logits)}`
  )
  void D
}

/** Without reference tensors: a chat prompt, and a greedy reply decoded. */
async function smoke(params: {
  model: string
  format?: "gguf" | "onnx"
  /** The ONNX build's directory on the server (default onnx-web). */
  onnx?: string
  batch: number
  precision: "highp" | "mediump"
  question?: string
  /** Pad the question to about this many prompt tokens, to time prompt reading. */
  length?: number
  skip?: Array<string>
  texels?: "bytes" | "words"
  ab?: string
  abDecode?: string
}) {
  if (params.onnx) onnxDir = `/${params.onnx}`
  const dir = params.format === "onnx" ? onnxDir : "/hf"
  const [tokenizerJson, tokenizerConfig, chatTemplate] = await Promise.all([
    fetch(`${dir}/tokenizer.json`).then((r) => r.json() as Promise<object>),
    fetch(`${dir}/tokenizer_config.json`).then(
      (r) => r.json() as Promise<Record<string, unknown>>
    ),
    fetch(`${dir}/chat_template.jinja`).then((r) =>
      r.ok ? r.text() : undefined
    )
  ])
  const tokenizer = loadTokenizer({
    tokenizerJson,
    tokenizerConfig,
    chatTemplate
  })
  const canvas = new OffscreenCanvas(1, 1)
  const engine = new Engine(canvas, {
    batchSize: params.batch,
    matmulPrecision: params.precision,
    weightTexels: params.texels
  })
  log(
    `GPU: ${engine.gpu.limits.renderer}, max texture ${engine.gpu.limits.maxTextureSize}`
  )
  const t0 = performance.now()
  await engine.load(await modelPieces(params))
  // Profile: the prompt with each kind of pass left out in turn.
  if (params.skip) {
    log(
      `textures: ${(engine.gpu.bytes / 2 ** 20).toFixed(0)} MiB (batch ${params.batch})`
    )
    const tokens = Array.from(
      { length: params.length ?? 512 },
      (_, i) => 1000 + (i % 5000)
    )
    engine.reset()
    await engine.forward(tokens.slice(0, 8))
    const time = async (label: string, skip: Array<string>) => {
      engine.gpu.skip = new Set(skip)
      // Best of 5 (the first also compiles anything new).
      let ms = Infinity
      for (let round = 0; round < 5; round++) {
        engine.reset()
        await engine.gpu.finish()
        const t = performance.now()
        await engine.forward(tokens)
        ms = Math.min(ms, performance.now() - t)
      }
      log(`PROFILE ${label.padEnd(16)} ${ms.toFixed(0)} ms`)
      return ms
    }
    const all = await time("everything", [])
    for (const name of params.skip) {
      const ms = await time(`without ${name}`, [name])
      log(
        `  ${name}: ${(all - ms).toFixed(0)} ms (${(((all - ms) / all) * 100).toFixed(0)}%)`
      )
    }
    await time("everything again", [])
    // Writing: single-token steps, each with its logits read back.
    const step = async (
      label: string,
      skip: Array<string>,
      readLogits = true
    ) => {
      engine.gpu.skip = new Set(skip)
      engine.reset()
      await engine.forward(tokens.slice(0, 64))
      const draws = engine.gpu.draws
      const t = performance.now()
      for (let i = 0; i < 16; i++) {
        if (readLogits) await engine.forward([tokens[i]])
        else {
          await engine.forwardNoLogits([tokens[i]])
          await engine.gpu.finish()
        }
      }
      const ms = (performance.now() - t) / 16
      // GPU throughput: queued, one wait (best of 3 rounds of 16).
      let gpuMs = Infinity
      for (let round = 0; round < 3; round++) {
        await engine.gpu.finish()
        const tg = performance.now()
        for (let i = 0; i < 16; i++) await engine.forwardNoLogits([tokens[i]])
        await engine.gpu.finish()
        gpuMs = Math.min(gpuMs, (performance.now() - tg) / 16)
      }
      log(
        `DECODE ${label.padEnd(22)} ${ms.toFixed(1)} ms/token, GPU ${gpuMs.toFixed(2)} ms/token, ${((engine.gpu.draws - draws) / 16).toFixed(0)} draws/token`
      )
      return gpuMs
    }
    const full = await step("everything", [])
    {
      // Each decode matmul alone: 64 passes back to back (no readback).
      type Internals = {
        buffers: { normed: unknown; partial: unknown; rms: unknown; residual: Array<unknown> }
        layers: Array<{
          conv?: { inProj: unknown; outProj: unknown }
          attention?: { qkvg: unknown; o: unknown }
          gateUp: unknown
          down: unknown
        }>
        splitPartials(a: unknown, w: unknown, rows: number): number
        rms(x: unknown, row0: number, rows: number, split?: number): void
      }
      const e = engine as unknown as Internals
      const convLayer = e.layers.find((l) => l.conv)
      const attnLayer = e.layers.find((l) => l.attention)
      const mats: Array<[string, unknown]> = [
        ["in_proj", convLayer?.conv?.inProj],
        ["out_proj", convLayer?.conv?.outProj],
        ["qkvg", attnLayer?.attention?.qkvg],
        ["o", attnLayer?.attention?.o],
        ["gate_up", e.layers[0].gateUp],
        ["down", e.layers[0].down]
      ]
      const per = async (run: () => void) => {
        run()
        await engine.gpu.finish()
        let best = Infinity
        for (let round = 0; round < 3; round++) {
          const t = performance.now()
          for (let i = 0; i < 64; i++) run()
          await engine.gpu.finish()
          best = Math.min(best, ((performance.now() - t) / 64) * 1000)
        }
        return best
      }
      // Each timed pass alternates with a 1-texel copy into another texture,
      // so every pass is its own render pass (repeated draws into one
      // target let the GPU skip all but the last: hidden surface removal).
      const scratch = engine.gpu.texture(WebGL2RenderingContext.RGBA32F, 1, 1)
      const copy = () =>
        engine.gpu.run({
          program: engine.gpu.program(K.COPY),
          inputs: { uX: e.buffers.residual[0] as typeof scratch },
          output: scratch,
          rect: { x: 0, y: 0, width: 1, height: 1 }
        })
      const floor = await per(() => {
        e.rms(e.buffers.residual[0], 0, 1)
        copy()
      })
      const parts: Array<string> = [`2 tiny passes ${floor.toFixed(0)}`]
      for (const [name, w] of mats) {
        if (!w) continue
        const t = await per(() => {
          e.splitPartials(e.buffers.normed, w, 1)
          copy()
        })
        parts.push(`${name} +${(t - floor / 2).toFixed(0)}`)
      }
      log(`MATMUL µs (each with its own pass): ${parts.join(", ")}`)
    }
    {
      // Fragment work per kind of pass: each drawn twice (same target, no
      // new render pass), GPU time added. Best of 3 rounds of 16 tokens.
      const gpuTime = async () => {
        let best = Infinity
        for (let round = 0; round < 5; round++) {
          await engine.gpu.finish()
          const t = performance.now()
          for (let i = 0; i < 16; i++) await engine.forwardNoLogits([tokens[i]])
          await engine.gpu.finish()
          best = Math.min(best, (performance.now() - t) / 16)
        }
        return best
      }
      engine.gpu.repeatTimes = 4
      const base = await gpuTime()
      const parts: Array<string> = []
      for (const name of ["matmul", "rms", "add", "swiglu", "heads", "conv", "scores", "attend"]) {
        engine.gpu.repeat = new Set([name])
        parts.push(`${name} ${((await gpuTime() - base) / 3).toFixed(2)}`)
      }
      engine.gpu.repeat = new Set()
      log(`WORK (ms per token, fragments only; base ${base.toFixed(2)}): ${parts.join(", ")}`)
    }
    {
      // JS issue time: forwards queued, timed before the wait.
      await engine.gpu.finish()
      const ti = performance.now()
      for (let i = 0; i < 16; i++) await engine.forwardNoLogits([tokens[i]])
      const issue = (performance.now() - ti) / 16
      await engine.gpu.finish()
      const total = (performance.now() - ti) / 16
      log(`ISSUE: ${issue.toFixed(2)} ms/token in JS; ${total.toFixed(2)} ms/token with the wait`)
    }
    {
      // Which passes a writing step starts render passes for.
      const trace: Array<string> = []
      engine.gpu.trace = trace
      const before = engine.gpu.passes
      await engine.forward([tokens[0]])
      engine.gpu.trace = null
      const counts = new Map<string, number>()
      for (const name of trace) counts.set(name, (counts.get(name) ?? 0) + 1)
      log(
        `PASSES per writing step: ${engine.gpu.passes - before}; ` +
          [...counts].sort((a, b) => b[1] - a[1]).map(([n, c]) => `${n} ${c}`).join(", ")
      )
    }
    await step("no logits readback", [], false)
    for (const name of params.skip) {
      const ms = await step(`without ${name}`, [name])
      log(`  ${name}: ${(full - ms).toFixed(1)} ms`)
    }
    engine.gpu.skip = new Set()
    const t = performance.now()
    for (let i = 0; i < 16; i++) await engine.logits(0)
    log(
      `DECODE output head + readback alone: ${((performance.now() - t) / 16).toFixed(1)} ms`
    )
    {
      const head = (engine as unknown as { head(row: number): void }).head.bind(engine)
      const savedHeadSplit = TUNING.headSplit
      for (const split of [1, 2, 4, 8, 1]) {
        TUNING.headSplit = split
        head(0)
        await engine.gpu.finish()
        const th = performance.now()
        for (let i = 0; i < 64; i++) head(0)
        await engine.gpu.finish()
        log(`HEAD split ${split}: ${((performance.now() - th) / 64).toFixed(2)} ms per head`)
      }
      TUNING.headSplit = savedHeadSplit
    }
    const tc = performance.now()
    for (let i = 0; i < 16; i++) await engine.candidates(0, 50, 1.05)
    log(
      `DECODE head + top-50 pick + readback alone: ${((performance.now() - tc) / 16).toFixed(1)} ms`
    )
    {
      engine.gpu.skip = new Set(["pick"])
      await engine.gpu.finish()
      const th = performance.now()
      for (let i = 0; i < 16; i++) {
        await engine.candidates(0, 50, 1.05)
      }
      log(
        `DECODE head only (pick skipped) + readback: ${((performance.now() - th) / 16).toFixed(1)} ms`
      )
      engine.gpu.skip = new Set()
    }
    const t2 = performance.now()
    for (let i = 0; i < 16; i++) await engine.gpu.finish()
    log(
      `DECODE empty sync alone: ${((performance.now() - t2) / 16).toFixed(1)} ms`
    )
    const probe = engine.gpu.texture(WebGL2RenderingContext.RGBA32F, 1, 1)
    const t3 = performance.now()
    for (let i = 0; i < 16; i++) engine.gpu.readNow(probe)
    log(
      `DECODE blocking 1-pixel read alone: ${((performance.now() - t3) / 16).toFixed(2)} ms`
    )
    const residual = (engine as unknown as { buffers: { residual: Array<typeof probe> } }).buffers.residual[0]
    const t4 = performance.now()
    for (let i = 0; i < 16; i++) {
      await engine.forwardNoLogits([tokens[i]])
      engine.gpu.readNow(residual, { x: 0, y: 0, width: 1, height: 1 })
    }
    await engine.gpu.finish()
    log(
      `DECODE step + blocking read: ${((performance.now() - t4) / 16).toFixed(1)} ms/token`
    )
    return
  }
  log(
    `loaded in ${((performance.now() - t0) / 1000).toFixed(1)} s, ${(engine.gpu.bytes / 2 ** 20).toFixed(0)} MiB of textures`
  )
  let question =
    params.question ?? "What is the capital of France? Answer in one sentence."
  const filler =
    "The river runs past the old mill, where the miller's daughter counts sacks of grain each morning before the market opens in the square. "
  const chat = () => [{ role: "user" as const, content: question }]
  const encode = () =>
    tokenizer.encodeChat(chat(), {
      addGenerationPrompt: true,
      enableThinking: false
    })
  if (params.length) {
    let pad = ""
    while (
      tokenizer.encodeChat([{ role: "user", content: pad + question }], {
        addGenerationPrompt: true
      }).length < params.length
    )
      pad += filler
    question = `${pad}\n\n${question}`
  }
  const prompt = encode()
  log(`prompt: ${prompt.length} tokens`)
  // Warm up: compile every shader on a short prompt first.
  engine.reset()
  await engine.forward(prompt.slice(0, 8))
  await engine.forward(prompt.slice(8, 9))
  engine.reset()
  // As the worker does it: candidates picked on the GPU (top 50, penalty
  // 1.1), the likeliest drawn (greedy, for a repeatable reply).
  const pick = async (row: number) => {
    const { ids, logits } = await engine.candidates(row, 50, 1.1)
    return ids[argmax(logits)]
  }
  // The prompt three times (median), each from an empty cache, with the
  // GPU's own time from a timer query where there is one.
  const gl = engine.gpu.gl
  const tq = gl.getExtension("EXT_disjoint_timer_query_webgl2") as {
    TIME_ELAPSED_EXT: number
  } | null
  const times: Array<number> = []
  const gpuTimes: Array<number> = []
  let token = 0
  for (let run = 0; run < 3; run++) {
    engine.reset()
    await engine.gpu.finish()
    const query = tq ? gl.createQuery() : null
    if (tq && query) gl.beginQuery(tq.TIME_ELAPSED_EXT, query)
    const t1 = performance.now()
    engine.setSeen(prompt, tokenizer.vocabSize)
    const row = await engine.forwardHidden(prompt)
    if (tq && query) gl.endQuery(tq.TIME_ELAPSED_EXT)
    token = await pick(row)
    times.push(performance.now() - t1)
    if (tq && query) {
      while (!gl.getQueryParameter(query, gl.QUERY_RESULT_AVAILABLE))
        await new Promise((r) => setTimeout(r, 2))
      gpuTimes.push((gl.getQueryParameter(query, gl.QUERY_RESULT) as number) / 1e6)
      gl.deleteQuery(query)
    }
  }
  // A/B: ?ab=knob:a,b runs the prompt alternating TUNING[knob] = a and b,
  // 5 times each, and compares the medians (drift on a shared machine
  // cancels out).
  if (params.ab) {
    const [knob, values] = params.ab.split(":")
    const [a, b] = values.split(",").map(Number)
    const runs: Record<string, Array<number>> = { [a]: [], [b]: [] }
    const original = (TUNING as Record<string, number>)[knob]
    // One untimed run of each first: new settings may compile programs.
    for (const v of [a, b]) {
      ;(TUNING as Record<string, number>)[knob] = v
      engine.reset()
      await pick(await engine.forwardHidden(prompt))
    }
    for (let i = 0; i < 10; i++) {
      const v = i % 2 === 0 ? a : b
      ;(TUNING as Record<string, number>)[knob] = v
      engine.reset()
      await engine.gpu.finish()
      const t = performance.now()
      await pick(await engine.forwardHidden(prompt))
      runs[v].push(performance.now() - t)
    }
    ;(TUNING as Record<string, number>)[knob] = original
    const med = (xs: Array<number>) => [...xs].sort((x, y) => x - y)[2]
    log(
      `AB ${knob}=${a}: ${med(runs[a]).toFixed(1)} ms (${runs[a].map((t) => t.toFixed(0)).join(" ")}); ${knob}=${b}: ${med(runs[b]).toFixed(1)} ms (${runs[b].map((t) => t.toFixed(0)).join(" ")})`
    )
  }
  // The same for writing: ?abDecode=knob:a,b, 16 steps per value, 4 rounds.
  if (params.abDecode) {
    const [knob, values] = params.abDecode.split(":")
    const [a, b] = values.split(",").map(Number)
    const perToken: Record<string, Array<number>> = { [a]: [], [b]: [] }
    const original = (TUNING as Record<string, number>)[knob]
    engine.reset()
    let next = await pick(await engine.forwardHidden(prompt))
    for (let i = 0; i < 8; i++) {
      const v = i % 2 === 0 ? a : b
      ;(TUNING as Record<string, number>)[knob] = v
      const t = performance.now()
      for (let s2 = 0; s2 < 16; s2++) next = await pick(await engine.forwardHidden([next]))
      perToken[v].push((performance.now() - t) / 16)
    }
    ;(TUNING as Record<string, number>)[knob] = original
    const med = (xs: Array<number>) => [...xs].sort((x, y) => x - y)[Math.floor(xs.length / 2)]
    log(
      `AB DECODE ${knob}=${a}: ${med(perToken[a]).toFixed(2)} ms/token (${perToken[a].map((t) => t.toFixed(1)).join(" ")}); ${knob}=${b}: ${med(perToken[b]).toFixed(2)} ms/token (${perToken[b].map((t) => t.toFixed(1)).join(" ")})`
    )
    // GPU throughput: 16 single-token forwards queued with no readback in
    // between, one wait at the end (CPU timing noise mostly drops out).
    const gpuTok: Record<string, Array<number>> = { [a]: [], [b]: [] }
    for (let i = 0; i < 12; i++) {
      const v = i % 2 === 0 ? a : b
      ;(TUNING as Record<string, number>)[knob] = v
      await engine.gpu.finish()
      const t = performance.now()
      for (let s2 = 0; s2 < 16; s2++) await engine.forwardNoLogits([next])
      await engine.gpu.finish()
      gpuTok[v].push((performance.now() - t) / 16)
    }
    ;(TUNING as Record<string, number>)[knob] = original
    log(
      `AB GPU ${knob}=${a}: ${med(gpuTok[a]).toFixed(2)} ms/token (${gpuTok[a].map((t) => t.toFixed(1)).join(" ")}); ${knob}=${b}: ${med(gpuTok[b]).toFixed(2)} ms/token (${gpuTok[b].map((t) => t.toFixed(1)).join(" ")})`
    )
  }
  if (prompt.length > 1000) {
    // Chunked attention (long key ranges) against the one-pass kernel: the
    // same prompt and 16 writing steps, logits compared step by step.
    const run = async (chunkTexels: number) => {
      const saved = TUNING.attendChunkTexels
      TUNING.attendChunkTexels = chunkTexels
      engine.reset()
      const out: Array<Float32Array> = [await engine.forward(prompt)]
      for (let i = 0; i < 16; i++) out.push(await engine.forward([1000 + i]))
      TUNING.attendChunkTexels = saved
      return out
    }
    const plain = await run(0)
    const chunked = await run(64)
    let worst = 0
    let same = 0
    for (let i = 0; i < plain.length; i++) {
      const r = compareLogits(chunked[i], plain[i])
      worst = Math.max(worst, r.kl)
      if (r.top1) same++
    }
    log(
      `CHUNK CHECK: chunked vs one-pass attention over ${prompt.length} tokens + 16 steps: worst KL ${worst.toExponential(2)}, top-1 same ${same}/${plain.length}`
    )
  }
  if (params.ab || params.abDecode) {
    // The A/B runs left the cache elsewhere: read the prompt again.
    engine.reset()
    engine.setSeen(prompt, tokenizer.vocabSize)
    token = await pick(await engine.forwardHidden(prompt))
  }
  const promptMs = [...times].sort((x, y) => x - y)[1]
  log(`prompt runs: ${times.map((t) => t.toFixed(0)).join(", ")} ms; GPU ${gpuTimes.map((t) => t.toFixed(0)).join(", ")} ms`)
  const out: Array<number> = []
  const t2 = performance.now()
  for (let i = 0; i < (params.length ? 64 : 48); i++) {
    if (token === tokenizer.specialTokens.eos) break
    out.push(token)
    engine.markSeen(token)
    token = await pick(await engine.forwardHidden([token]))
  }
  const decodeMs = performance.now() - t2
  log(`REPLY: ${JSON.stringify(tokenizer.decode(out))}`)
  {
    // Determinism: junk tokens, a reset, and the same reply again.
    for (let i = 0; i < 200; i++) await engine.forwardNoLogits([1000 + i])
    const again: Array<number> = []
    engine.reset()
    engine.setSeen(prompt, tokenizer.vocabSize)
    let t = await pick(await engine.forwardHidden(prompt))
    for (let i = 0; i < out.length; i++) {
      if (t === tokenizer.specialTokens.eos) break
      again.push(t)
      engine.markSeen(t)
      t = await pick(await engine.forwardHidden([t]))
    }
    const same = again.length === out.length && again.every((x, i) => x === out[i])
    log(
      same
        ? "RESET CHECK: same reply after junk and a reset"
        : `RESET CHECK: DIFFERENT after junk and a reset: ${JSON.stringify(tokenizer.decode(again))}`
    )
  }
  // How speculative decoding with prompt lookup would fare on this reply:
  // at each step, draft up to 3 tokens by matching the history's last
  // 3, 2 or 1 tokens earlier in the history, and count how many of the
  // drafts the (greedy) reply actually continues with.
  {
    const history = [...prompt]
    let steps = 0
    let i = 0
    while (i < out.length) {
      const draft = lookupDraft([...history, out[i]], 3)
      let accepted = 0
      while (accepted < draft.length && out[i + 1 + accepted] === draft[accepted]) accepted++
      const kept = 1 + accepted
      history.push(...out.slice(i, i + kept))
      i += kept
      steps++
    }
    log(
      `SPECULATIVE (prompt lookup, 3 drafts): ${out.length} tokens in ${steps} verify steps, ${(out.length / steps).toFixed(2)} tokens per step`
    )
  }
  log(
    `prompt ${promptMs.toFixed(0)} ms (${((prompt.length / promptMs) * 1000).toFixed(1)} tokens/s), ${out.length} tokens in ${decodeMs.toFixed(0)} ms (${((out.length / decodeMs) * 1000).toFixed(1)} tokens/s)`
  )
  // The same reply with speculative decoding, as the worker writes:
  // drafts by prompt lookup checked in one batch, greedy, so the tokens
  // must match the plain loop's exactly.
  // The worker's policy (src/worker.ts: DRAFT_MATCH, DRAFTS).
  for (const [minMatch, maxDrafts] of [[2, 3]]) {
    engine.reset()
    engine.setSeen(prompt, tokenizer.vocabSize)
    let tok = await pick(await engine.forwardHidden(prompt))
    const history = [...prompt]
    const spec: Array<number> = []
    let steps = 0
    const t3 = performance.now()
    const limit = params.length ? 64 : 48
    while (spec.length < limit && tok !== tokenizer.specialTokens.eos) {
      spec.push(tok)
      engine.markSeen(tok)
      history.push(tok)
      if (spec.length >= limit) break
      const drafts = lookupDraft(
        history,
        Math.min(maxDrafts, limit - spec.length - 1),
        minMatch
      )
      const batch = [tok, ...drafts]
      await engine.forwardHidden(batch, { draft: drafts.length > 0 })
      steps++
      let kept = 1
      let next = -1
      for (let i = 0; i < batch.length; i++) {
        const choice = await pick(i)
        if (i + 1 < batch.length && choice === batch[i + 1] && batch[i + 1] !== tokenizer.specialTokens.eos) {
          kept++
          spec.push(choice)
          engine.markSeen(choice)
          history.push(choice)
          continue
        }
        next = choice
        break
      }
      if (drafts.length) engine.accept(kept)
      if (next < 0) break
      tok = next
    }
    const specMs = performance.now() - t3
    const same = spec.length === out.length && spec.every((t, i) => t === out[i])
    const firstDiff = spec.findIndex((t, i) => t !== out[i])
    log(
      `SPECULATIVE match ${minMatch}+, ${maxDrafts} drafts: ${spec.length} tokens in ${steps} steps, ${specMs.toFixed(0)} ms (${((spec.length / specMs) * 1000).toFixed(1)} tokens/s); ${same ? "same tokens as plain decoding" : `DIFFERENT from plain decoding at token ${firstDiff}`}`
    )
  }
}

onmessage = ({ data }) => {
  // ?tune=knob:value,knob:value sets TUNING before anything runs.
  for (const pair of ((data as { tune?: string }).tune ?? "").split(",")) {
    const [knob, value] = pair.split(":")
    if (knob && value !== undefined)
      (TUNING as Record<string, number>)[knob] = Number(value)
  }
  ;(data.smoke ? smoke(data) : run(data)).then(
    () => postMessage({ type: "done" }),
    (error: unknown) =>
      postMessage({
        type: "error",
        message:
          error instanceof Error
            ? `${error.message}\n${error.stack}`
            : String(error)
      })
  )
}
