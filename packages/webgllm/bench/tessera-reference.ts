/**
 * Tessera's reference tensors, from ONNX Runtime on the CPU in Node, and a
 * check of tessera-cpu.ts against them. Writes, into OUT_DIR, what
 * tessera-check.html compares the GPU with:
 *
 *   clip-ids.json, clip-states.bin, clip-pooled.bin   CLIP on selftest.json's ids
 *   case-<name>.json   a denoiser case's inputs (small ones inline, sc_probs as a seed)
 *   case-<name>.bin    its logits
 *
 * Run (bundled with esbuild, ORT_NODE an installed onnxruntime-node):
 *   TESSERA_DIR=<the ONNX files> OUT_DIR=<refs> ORT_NODE=<path> node tessera-reference.mjs
 */
import { readFileSync, writeFileSync, mkdirSync } from "node:fs"
import { createRequire } from "node:module"
import { join } from "node:path"
import { readClipWeights, readTesseraWeights } from "../src/tessera-weights.js"
import { denoise, encodeText, type DenoiserInputs } from "./tessera-cpu.js"
import { seededInputs, type CaseSpec } from "./tessera-cases.js"

const dir = process.env.TESSERA_DIR!
const out = process.env.OUT_DIR!
mkdirSync(out, { recursive: true })
const require = createRequire(join(process.env.ORT_NODE!, "package.json"))
const ort = require(process.env.ORT_NODE!) as typeof import("onnxruntime-node")

const selftest = JSON.parse(readFileSync(join(dir, "selftest.json"), "utf8")) as { ids: Array<number>; argmax: Array<number>; top: Array<number> }

function compare(name: string, got: Float32Array, want: Float32Array) {
  let maxAbs = 0
  let sumAbs = 0
  let maxRef = 0
  for (let i = 0; i < want.length; i++) {
    const d = Math.abs(got[i]! - want[i]!)
    maxAbs = Math.max(maxAbs, d)
    sumAbs += d
    maxRef = Math.max(maxRef, Math.abs(want[i]!))
  }
  console.log(`${name}: max |diff| ${maxAbs.toExponential(2)}, mean ${(sumAbs / want.length).toExponential(2)} (max |ref| ${maxRef.toFixed(2)})`)
}

function argmaxAgreement(logits: Float32Array, rows: number, vocab: number, want: (row: number) => number) {
  let agree = 0
  for (let r = 0; r < rows; r++) {
    let best = 0
    for (let k = 1; k < vocab; k++) if (logits[r * vocab + k]! > logits[r * vocab + best]!) best = k
    if (best === want(r)) agree++
  }
  return agree / rows
}

const t0 = performance.now()
const clipW = readClipWeights(new Uint8Array(readFileSync(join(dir, "clip_text_w8.onnx"))))
const denW = readTesseraWeights(new Uint8Array(readFileSync(join(dir, "tessera_w8.onnx"))))
console.log(`weights read in ${Math.round(performance.now() - t0)} ms: CLIP ${clipW.layers.length} layers, width ${clipW.width}`)

const sessionOptions = process.env.ORT_OPT ? { graphOptimizationLevel: process.env.ORT_OPT as "disabled" } : {}
const clip = await ort.InferenceSession.create(join(dir, "clip_text_w8.onnx"), sessionOptions)
const den = await ort.InferenceSession.create(join(dir, "tessera_w8.onnx"), sessionOptions)

// CLIP.
const c = await clip.run({ input_ids: new ort.Tensor("int64", BigInt64Array.from(selftest.ids, BigInt), [1, 24]) })
const states = c.last_hidden_state!.data as Float32Array
const pooled = c.pooler_output!.data as Float32Array
const cpuClip = encodeText(clipW, selftest.ids)
compare("CLIP last_hidden_state (CPU vs ORT)", cpuClip.states, states)
compare("CLIP pooler_output (CPU vs ORT)", cpuClip.pooled, pooled)
writeFileSync(join(out, "clip-ids.json"), JSON.stringify(selftest.ids))
writeFileSync(join(out, "clip-states.bin"), states)
writeFileSync(join(out, "clip-pooled.bin"), pooled)

// Denoiser cases.
const cases: Array<CaseSpec> = [
  // selftest.json's: a fully masked canvas, with the prompt.
  { name: "masked", batch: 1, painted: 0, seed: 1, drop: [0], sc: false },
  // A canvas a third painted, with self-conditioning; the second row without the prompt.
  { name: "partial", batch: 2, painted: 0.35, seed: 7, drop: [0, 1], sc: true }
]
for (const spec of cases) {
  const inputs: DenoiserInputs = seededInputs(spec, states, pooled)
  const B = spec.batch
  const o = await den.run({
    x: new ort.Tensor("int64", BigInt64Array.from(inputs.x, BigInt), [B, 256]),
    text_states: new ort.Tensor("float32", inputs.textStates, [B, 24, 512]),
    text_pooled: new ort.Tensor("float32", inputs.textPooled, [B, 512]),
    mask_ratio: new ort.Tensor("float32", inputs.maskRatio, [B]),
    sc_probs: new ort.Tensor("float32", inputs.scProbs, [B, 256, 512]),
    drop_text: new ort.Tensor("bool", inputs.dropText, [B])
  })
  const logits = o.logits!.data as Float32Array
  const t1 = performance.now()
  const cpu = denoise(denW, inputs, B)
  console.log(`case ${spec.name}: CPU forward ${Math.round(performance.now() - t1)} ms`)
  compare(`  logits (CPU vs ORT)`, cpu, logits)
  console.log(`  argmax agreement CPU vs ORT: ${(argmaxAgreement(cpu, B * 256, 512, (r) => {
    let best = 0
    for (let k = 1; k < 512; k++) if (logits[r * 512 + k]! > logits[r * 512 + best]!) best = k
    return best
  }) * 100).toFixed(1)}%`)
  if (spec.name === "masked") console.log(`  selftest argmax agreement: ORT ${(argmaxAgreement(logits, 256, 512, (r) => selftest.argmax[r]!) * 100).toFixed(1)}%, CPU ${(argmaxAgreement(cpu, 256, 512, (r) => selftest.argmax[r]!) * 100).toFixed(1)}%`)
  writeFileSync(join(out, `case-${spec.name}.json`), JSON.stringify(spec))
  writeFileSync(join(out, `case-${spec.name}.bin`), logits)
}
