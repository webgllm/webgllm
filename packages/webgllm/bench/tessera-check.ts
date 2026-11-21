/**
 * Tessera on the GPU against ONNX Runtime's tensors (tessera-reference.ts):
 * CLIP's states, then the denoiser's logits for each case (on ORT's CLIP
 * states, so each model is checked alone), and selftest.json's argmax.
 * Then the speed of 1 and 4 sprites' steps. ?engine=webgpu|webgl,
 * ?models=<dir of the ONNX files>, ?refs=<dir of the reference tensors>.
 */
import { Gpu } from "../src/gpu/gl.js"
import { WebGpu } from "../src/gpu/webgpu.js"
import { TesseraGpu } from "../src/tessera.js"
import { readClipWeights, readTesseraWeights } from "../src/tessera-weights.js"
import { seededInputs, type CaseSpec } from "./tessera-cases.js"

const params = new URLSearchParams(location.search)
const engine = params.get("engine") ?? "webgpu"
const models = params.get("models") ?? "/models/tessera"
const refs = params.get("refs") ?? "/models/tessera-refs-noopt"
const tag = params.get("tag") ?? engine
const out = document.getElementById("out")!

const log = (line: string) => {
  out.textContent += `${line}\n`
  return fetch("/log", { method: "POST", body: `TESSERA ${tag} ${line}` }).catch(() => undefined)
}
const bytes = async (url: string) => new Uint8Array(await (await fetch(url)).arrayBuffer())
const floats = async (url: string) => new Float32Array((await bytes(url)).buffer)

function compare(got: Float32Array, want: Float32Array) {
  let max = 0
  let sum = 0
  for (let i = 0; i < want.length; i++) {
    const d = Math.abs(got[i]! - want[i]!)
    if (!(d <= max)) max = Number.isNaN(d) ? Infinity : d
    sum += d
  }
  return `max |diff| ${max.toExponential(2)}, mean ${(sum / want.length).toExponential(2)}`
}

function argmaxes(logits: Float32Array, rows: number, vocab = 512) {
  const out = new Int32Array(rows)
  for (let r = 0; r < rows; r++) {
    let best = 0
    for (let k = 1; k < vocab; k++) if (logits[r * vocab + k]! > logits[r * vocab + best]!) best = k
    out[r] = best
  }
  return out
}
const agreement = (a: ArrayLike<number>, b: ArrayLike<number>) => {
  let n = 0
  for (let i = 0; i < a.length; i++) if (a[i] === b[i]) n++
  return `${((100 * n) / a.length).toFixed(1)}%`
}

try {
  const MB = (n: number) => `${(n / 2 ** 20).toFixed(1)} MB`
  let t = performance.now()
  const [denBytes, clipBytes] = await Promise.all([bytes(`${models}/tessera_w8.onnx`), bytes(`${models}/clip_text_w8.onnx`)])
  const w = readTesseraWeights(denBytes)
  const clipW = readClipWeights(clipBytes)
  await log(`read ${Math.round(performance.now() - t)} ms`)
  const gpu = engine === "webgpu" ? await WebGpu.create() : new Gpu(new OffscreenCanvas(1, 1))
  if (!gpu) throw new Error("No WebGPU here.")
  await log(`start ${gpu.limits.renderer}`)
  t = performance.now()
  const tessera = new TesseraGpu(gpu)
  await tessera.load(w, clipW)
  await gpu.finish()
  await log(`uploaded ${Math.round(performance.now() - t)} ms, weights ${MB(tessera.bytes.weights)}`)

  // CLIP.
  const ids = (await (await fetch(`${refs}/clip-ids.json`)).json()) as Array<number>
  const states = await floats(`${refs}/clip-states.bin`)
  const pooled = await floats(`${refs}/clip-pooled.bin`)
  t = performance.now()
  const text = await tessera.encodeText(ids)
  await log(`CHECK clip states ${compare(text.states, states)} (${Math.round(performance.now() - t)} ms)`)
  await log(`CHECK clip pooled ${compare(text.pooled, pooled)}`)

  // The denoiser.
  const selftest = (await (await fetch(`${models}/selftest.json`)).json()) as { argmax: Array<number> }
  for (const name of ["masked", "partial"]) {
    const spec = (await (await fetch(`${refs}/case-${name}.json`)).json()) as CaseSpec
    const want = await floats(`${refs}/case-${name}.bin`)
    const inputs = seededInputs(spec, states, pooled)
    const count = spec.drop.findIndex((d) => d) < 0 ? spec.batch : spec.drop.findIndex((d) => d)
    await tessera.setText(states, pooled)
    t = performance.now()
    const logits = await tessera.denoise(inputs.x, inputs.maskRatio, spec.sc ? inputs.scProbs : null, spec.batch, count)
    const ms = Math.round(performance.now() - t)
    await log(`CHECK case ${name} logits ${compare(logits, want)}, argmax vs ORT ${agreement(argmaxes(logits, spec.batch * 256), argmaxes(want, spec.batch * 256))} (${ms} ms)`)
    if (name === "masked") await log(`CHECK selftest argmax ${agreement(argmaxes(logits, 256), selftest.argmax)}`)
  }

  // Speed: a sprite's 16 steps are 16 passes of 2 rows (with and without the prompt).
  for (const sprites of [1, 4]) {
    const batch = 2 * sprites
    const x = new Int32Array(batch * 256).fill(512)
    const ratio = new Float32Array(batch).fill(1)
    const sc = new Float32Array(batch * 256 * 512)
    await tessera.denoise(x, ratio, sc, batch, sprites)
    t = performance.now()
    for (let step = 0; step < 16; step++) await tessera.denoise(x, ratio, sc, batch, sprites)
    const ms = performance.now() - t
    await log(`SPEED ${sprites} sprite(s): 16 steps ${Math.round(ms)} ms (${(ms / 16).toFixed(1)} ms a step), work ${MB(tessera.bytes.work)}`)
  }
  await log("ALL DONE")
} catch (error) {
  await log(`ERROR ${error instanceof Error ? error.stack ?? error.message : String(error)}`)
}
