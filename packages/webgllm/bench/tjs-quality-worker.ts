/**
 * Transformers.js (WebGPU, q4f16) scored against the onnxruntime reference
 * tensors, as the check page scores this package's engine: the same local
 * weights (/onnx-web/, the model the reference was made from), the
 * reference's 350 prompt tokens and its 32-token greedy continuation in one
 * forward, every position's logits against logits_fp32.bin (fp32 math)
 * and logits.bin (the main reference).
 */
const log = (text: string) => postMessage({ type: "log", text })

const TRANSFORMERS =
  "https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.3.2/dist/transformers.web.min.js"
/** The ONNX Runtime the app's Transformers.js worker uses (see its worker.ts). */
const ORT = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.31.1/dist/ort.webgpu.bundle.min.mjs"

/** Transformers.js on ONNX Runtime 1.31.1, imported as the app imports it. */
async function importTransformers() {
  const [transformers, ort] = await Promise.all(
    [TRANSFORMERS, ORT].map(async (url) => (await fetch(url)).text())
  )
  const blobUrl = (code: string) =>
    URL.createObjectURL(new Blob([code], { type: "text/javascript" }))
  const ortModule = blobUrl(ort)
  const url = blobUrl(
    transformers.replace(
      /from\s*"onnxruntime-(?:web\/webgpu|common)"/g,
      `from"${ortModule}"`
    )
  )
  return import(/* @vite-ignore */ url) as Promise<unknown>
}

type Tensor = {
  type: string
  dims: Array<number>
  data: ArrayLike<number>
}

/** Rows [from, from + rows) of a float32 [n][dims] reference file. */
async function refRows(file: string, dims: number, from: number, rows: number) {
  const res = await fetch(`/ref-onnx/${file}`, {
    headers: {
      Range: `bytes=${from * dims * 4}-${(from + rows) * dims * 4 - 1}`
    }
  })
  if (!res.ok) throw new Error(`/ref-onnx/${file}: ${res.status}`)
  return new Float32Array(await res.arrayBuffer())
}

function softmax(logits: ArrayLike<number>) {
  let mx = -Infinity
  for (let i = 0; i < logits.length; i++) mx = Math.max(mx, logits[i])
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

/** KL(ref || got) and whether the top token agrees. */
function score(got: ArrayLike<number>, want: ArrayLike<number>) {
  const p = softmax(want)
  const q = softmax(got)
  let kl = 0
  for (let i = 0; i < p.length; i++)
    if (p[i] > 0) kl += p[i] * Math.log(p[i] / Math.max(q[i], 1e-30))
  return { kl, top1: argmax(got) === argmax(want) }
}

/** Half-float bits to a number. */
function halfToFloat(h: number) {
  const s = h & 0x8000 ? -1 : 1
  const e = (h >> 10) & 0x1f
  const f = h & 0x3ff
  if (e === 0) return s * 2 ** -14 * (f / 1024)
  if (e === 31) return f ? NaN : s * Infinity
  return s * 2 ** (e - 15) * (1 + f / 1024)
}

function asFloat32(t: Tensor): Float32Array {
  if (t.type === "float32") return t.data as Float32Array
  if (t.type === "float16") {
    const d = t.data
    // Float16Array where the browser has it; raw half bits otherwise.
    if (d instanceof Uint16Array) return Float32Array.from(d, halfToFloat)
    return Float32Array.from(d)
  }
  throw new Error(`logits of type ${t.type}`)
}

async function main() {
  const tf = (await importTransformers()) as {
    env: {
      allowLocalModels: boolean
      allowRemoteModels: boolean
      localModelPath: string
    }
    Tensor: new (type: string, data: unknown, dims: Array<number>) => Tensor
    AutoModelForCausalLM: {
      from_pretrained(
        id: string,
        options: Record<string, unknown>
      ): Promise<(inputs: Record<string, unknown>) => Promise<{ logits: Tensor }>>
    }
  }
  tf.env.allowLocalModels = true
  tf.env.allowRemoteModels = false
  tf.env.localModelPath = `${location.origin}/`
  const manifest = (await (await fetch("/ref-onnx/manifest.json")).json()) as {
    n_vocab: number
  }
  const V = manifest.n_vocab
  const tokensFile = (await (await fetch("/ref-onnx/tokens.json")).json()) as {
    token_ids: Array<number>
    greedy_continuation: { token_ids: Array<number> }
  }
  const prompt = tokensFile.token_ids
  const cont = tokensFile.greedy_continuation.token_ids
  const ids = [...prompt, ...cont]
  const n = ids.length
  const t0 = performance.now()
  const model = await tf.AutoModelForCausalLM.from_pretrained("onnx-web", {
    device: "webgpu",
    dtype: "q4f16",
    use_external_data_format: true
  })
  log(`Transformers.js 4.3.2 loaded /onnx-web/ q4f16 on WebGPU in ${((performance.now() - t0) / 1000).toFixed(1)} s`)
  const int64 = (values: Array<number>) =>
    BigInt64Array.from(values, (v) => BigInt(v))
  const out = await model({
    input_ids: new tf.Tensor("int64", int64(ids), [1, n]),
    attention_mask: new tf.Tensor("int64", int64(ids.map(() => 1)), [1, n]),
    num_logits_to_keep: new tf.Tensor("int64", int64([n]), [])
  })
  const logits = asFloat32(out.logits)
  const rows = out.logits.dims[1]
  const width = out.logits.dims[2]
  log(`logits: ${out.logits.type} [${out.logits.dims.join(", ")}]`)
  if (rows !== n) throw new Error(`expected ${n} rows of logits, got ${rows}`)
  const row = (r: number) => logits.subarray(r * width, r * width + V)
  const checked = [0, 64, 127, 128, 192, 255, 256, 303, 349]
  for (const [label, promptFile, genFile] of [
    ["fp32 reference", "logits_fp32.bin", "gen_logits_fp32.bin"],
    ["main reference", "logits.bin", "gen_logits.bin"]
  ] as const) {
    let kl = 0
    let top1 = 0
    let klChecked = 0
    let top1Checked = 0
    for (let p = 0; p < prompt.length; p++) {
      const r = score(row(p), await refRows(promptFile, V, p, 1))
      kl += r.kl
      if (r.top1) top1++
      if (checked.includes(p)) {
        klChecked += r.kl
        if (r.top1) top1Checked++
      }
    }
    let genKl = 0
    let genTop1 = 0
    for (let i = 0; i < cont.length; i++) {
      // gen row i: the logits after continuation token i (position 350 + i).
      const r = score(row(prompt.length + i), await refRows(genFile, V, i, 1))
      genKl += r.kl
      if (r.top1) genTop1++
    }
    log(
      `QUALITY Transformers.js vs ${label}: prompt (all ${prompt.length}) mean KL ${(kl / prompt.length).toExponential(2)}, top-1 ${top1}/${prompt.length}; checked 9 mean KL ${(klChecked / 9).toExponential(2)}, top-1 ${top1Checked}/9; continuation (prefill path) mean KL ${(genKl / cont.length).toExponential(2)}, top-1 ${genTop1}/${cont.length}`
    )
  }
}

main().then(
  () => postMessage({ type: "done" }),
  (error: unknown) =>
    postMessage({
      type: "error",
      message: error instanceof Error ? `${error.message}\n${error.stack}` : String(error)
    })
)
