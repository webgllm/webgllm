/**
 * The denoiser's check cases, made the same way in Node (tessera-reference.ts,
 * for ONNX Runtime) and in the browser (tessera-check.html): from a seed, so
 * only the seed and a few numbers travel between them.
 */
import type { DenoiserInputs } from "./tessera-cpu.js"

export type CaseSpec = {
  name: string
  batch: number
  /** Share of pixels painted (the rest MASK). */
  painted: number
  seed: number
  /** drop_text per row. */
  drop: Array<number>
  /** Whether sc_probs holds a distribution (else zeros). */
  sc: boolean
}

function random(seed: number) {
  let a = seed >>> 0
  return () => {
    a = (a + 0x6d2b79f5) >>> 0
    let t = a
    t = Math.imul(t ^ (t >>> 15), t | 1)
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61)
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}

/** A case's inputs: every row with the same text, its pixels painted at random, sc_probs a random softmax per pixel. */
export function seededInputs(spec: CaseSpec, states: Float32Array, pooled: Float32Array): DenoiserInputs {
  const B = spec.batch
  const P = 256
  const K = 512
  const r = random(spec.seed)
  const x = new Int32Array(B * P)
  const maskRatio = new Float32Array(B)
  for (let b = 0; b < B; b++) {
    let masked = 0
    for (let i = 0; i < P; i++) {
      const paint = r() < spec.painted
      x[b * P + i] = paint ? Math.floor(r() * K) : K
      if (!paint) masked++
    }
    maskRatio[b] = masked / P
  }
  const scProbs = new Float32Array(B * P * K)
  if (spec.sc)
    for (let row = 0; row < B * P; row++) {
      let sum = 0
      for (let k = 0; k < K; k++) sum += scProbs[row * K + k] = Math.exp(4 * r())
      for (let k = 0; k < K; k++) scProbs[row * K + k] = scProbs[row * K + k]! / sum
    }
  const textStates = new Float32Array(B * 24 * 512)
  const textPooled = new Float32Array(B * 512)
  for (let b = 0; b < B; b++) {
    textStates.set(states, b * 24 * 512)
    textPooled.set(pooled, b * 512)
  }
  return { x, textStates, textPooled, maskRatio, scProbs, dropText: Uint8Array.from(spec.drop) }
}
