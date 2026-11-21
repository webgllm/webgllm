/**
 * Tessera's masked-diffusion sampler, as tanstack-ai-browser's onnx adapter
 * (packages/onnx/src/tessera.ts) runs it on ONNX Runtime, here on any
 * denoiser: it starts from a blank canvas and commits pixels over a fixed
 * number of steps, each step one pass of the denoiser over the whole
 * canvas.
 *
 * It follows Tessera's published sampler: classifier-free guidance (each
 * image runs with and without the prompt), a cosine schedule of how many
 * pixels stay blank, confidence-ordered commits with Gumbel noise that
 * fades over the steps, a spatial non-maximum suppression that spreads each
 * step's commits over the canvas, and self-conditioning (each step's
 * distribution goes back into the next).
 */

/** The canvas: GRID×GRID pixels, each one of PALETTE palette tokens. */
export const GRID = 16
export const PIXELS = GRID * GRID
export const PALETTE = 512
/** The token of a pixel not painted yet. */
export const MASK = PALETTE
export const TEXT_LEN = 24

/** The sampler's settings. */
export type SamplerSettings = {
  /** Denoiser passes; each commits a share of the blank pixels. */
  steps: number
  /** Classifier-free guidance: how far each pixel is pushed toward the prompt. */
  guidanceScale: number
  /** The Gumbel noise on the commit order at the first step (it fades to none). */
  choiceTemperature: number
  /** A pixel at least this certain is committed without competing with its neighbours. */
  certainty: number
  /** The random source's seed: the same seed, prompt and settings draw the same images. */
  seed: number
}

export const DEFAULT_SAMPLER: Omit<SamplerSettings, "seed"> = {
  steps: 16,
  guidanceScale: 3,
  choiceTemperature: 2,
  certainty: 0.9
}

/**
 * The denoiser's logits ([batch][PIXELS][PALETTE]) for `batch` canvases of
 * tokens: rows below `count` with the prompt, the rest without it.
 * `scProbs` is the last step's distribution, or null on the first step.
 */
export type Denoise = (tokens: Int32Array, maskRatio: Float32Array, scProbs: Float32Array | null, batch: number, count: number) => Promise<Float32Array>

/** Mulberry32: a small seeded random source, uniform on [0, 1). */
export function seededRandom(seed: number) {
  let a = seed >>> 0
  return () => {
    a = (a + 0x6d2b79f5) >>> 0
    let t = a
    t = Math.imul(t ^ (t >>> 15), t | 1)
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61)
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}

/**
 * The text encoder's input: the prompt's ids with its start and end tokens,
 * padded with the end token to TEXT_LEN, or cut to it with the end token
 * kept last.
 */
export function fixedLengthIds(ids: ReadonlyArray<number>, endToken: number): Array<number> {
  const kept = ids.length > TEXT_LEN ? [...ids.slice(0, TEXT_LEN - 1), endToken] : [...ids]
  while (kept.length < TEXT_LEN) kept.push(endToken)
  return kept
}

/** Blank pixels left after step `step` of `steps` (cosine schedule; none after the last). */
export function blankAfter(step: number, steps: number) {
  return step === steps - 1 ? 0 : Math.floor(PIXELS * Math.cos((Math.PI / 2) * ((step + 1) / steps)))
}

/**
 * The order to commit this step's blank pixels in, best first. Pixels at
 * least `certainty` sure go first; of the others, a pixel goes ahead only if
 * it scores at least as well as every uncertain blank neighbour, so a step's
 * commits spread out instead of clumping.
 */
export function commitOrder(blank: Uint8Array, confidence: Float32Array, score: Float32Array, certainty: number): Array<number> {
  const ranked: Array<[number, number]> = []
  for (let i = 0; i < PIXELS; i++) {
    if (!blank[i]) continue
    let ahead = confidence[i]! >= certainty
    if (!ahead) {
      const row = Math.floor(i / GRID)
      const col = i % GRID
      let best = -Infinity
      for (let r = Math.max(row - 1, 0); r <= Math.min(row + 1, GRID - 1); r++) {
        for (let c = Math.max(col - 1, 0); c <= Math.min(col + 1, GRID - 1); c++) {
          const j = r * GRID + c
          if (blank[j] && confidence[j]! < certainty) best = Math.max(best, score[j]!)
        }
      }
      ahead = score[i]! >= best
    }
    ranked.push([score[i]! + (ahead ? 1e4 : 0), i])
  }
  ranked.sort((a, b) => b[0] - a[0])
  return ranked.map(([, i]) => i)
}

/**
 * Draws `count` sprites with `denoise` (its prompt set). Returns each
 * sprite's palette tokens, row by row. `signal` stops it between steps;
 * `onStep` gets the tokens after each step.
 */
export async function drawSprites(
  denoise: Denoise,
  count: number,
  settings: SamplerSettings,
  signal?: AbortSignal,
  onStep?: (step: number, tokens: ReadonlyArray<Int32Array>) => void
): Promise<Array<Int32Array>> {
  const { steps, guidanceScale, choiceTemperature, certainty } = settings
  const random = seededRandom(settings.seed)
  // Rows [0, count) with the prompt, [count, 2·count) without it.
  const rows = count * 2
  const tokens = Array.from({ length: count }, () => new Int32Array(PIXELS).fill(MASK))
  const blank = Array.from({ length: count }, () => new Uint8Array(PIXELS).fill(1))
  const selfCond = new Float32Array(rows * PIXELS * PALETTE)
  const probs = new Float32Array(PIXELS * PALETTE)
  const picked = new Int32Array(PIXELS)
  const confidence = new Float32Array(PIXELS)
  const score = new Float32Array(PIXELS)

  for (let step = 0; step < steps; step++) {
    signal?.throwIfAborted()
    const progress = (step + 1) / steps
    const x = new Int32Array(rows * PIXELS)
    const maskRatio = new Float32Array(rows)
    for (let b = 0; b < count; b++) {
      let blanks = 0
      for (let i = 0; i < PIXELS; i++) {
        x[b * PIXELS + i] = tokens[b]![i]!
        x[(b + count) * PIXELS + i] = tokens[b]![i]!
        blanks += blank[b]![i]!
      }
      maskRatio[b] = maskRatio[b + count] = blanks / PIXELS
    }
    // (The first step's self-conditioning is all zeros.)
    const logits = await denoise(x, maskRatio, step === 0 ? null : selfCond, rows, count)
    signal?.throwIfAborted()

    for (let b = 0; b < count; b++) {
      const withText = b * PIXELS * PALETTE
      const withoutText = (b + count) * PIXELS * PALETTE
      // Guided distribution per pixel, and one token drawn from it.
      for (let i = 0; i < PIXELS; i++) {
        const row = i * PALETTE
        let max = -Infinity
        for (let k = 0; k < PALETTE; k++) {
          const without = logits[withoutText + row + k]!
          const guided = without + guidanceScale * (logits[withText + row + k]! - without)
          if (!Number.isFinite(guided)) throw new Error("The model returned a non-finite value: this device's GPU may not run it correctly")
          probs[row + k] = guided
          if (guided > max) max = guided
        }
        let sum = 0
        for (let k = 0; k < PALETTE; k++) {
          const e = Math.exp(probs[row + k]! - max)
          probs[row + k] = e
          sum += e
        }
        let u = random() * sum
        let pick = PALETTE - 1
        for (let k = 0; k < PALETTE; k++) {
          u -= probs[row + k]!
          if (u <= 0) {
            pick = k
            break
          }
        }
        for (let k = 0; k < PALETTE; k++) probs[row + k] = probs[row + k]! / sum
        picked[i] = pick
        confidence[i] = probs[row + pick]!
      }

      let blanks = 0
      for (let i = 0; i < PIXELS; i++) blanks += blank[b]![i]!
      const commits = Math.min(Math.max(blanks - blankAfter(step, steps), 1), blanks)

      // Confidence with fading Gumbel noise.
      for (let i = 0; i < PIXELS; i++) {
        const u = Math.min(Math.max(random(), 1e-10), 1 - 1e-10)
        score[i] = Math.log(Math.max(confidence[i]!, 1e-12)) + choiceTemperature * (1 - progress) * -Math.log(-Math.log(u))
      }
      for (const i of commitOrder(blank[b]!, confidence, score, certainty).slice(0, commits)) {
        tokens[b]![i] = picked[i]!
        blank[b]![i] = 0
      }

      // Self-conditioning: this step's distribution, for both rows of the image.
      selfCond.set(probs, withText)
      selfCond.set(probs, withoutText)
    }
    onStep?.(step, tokens)
  }
  return tokens
}

/** A sprite's RGBA pixels, from its tokens and the palette (`palette_rgba`). */
export function spritePixels(tokens: Int32Array, palette: ReadonlyArray<ReadonlyArray<number>>): Uint8Array {
  const rgba = new Uint8Array(PIXELS * 4)
  for (let i = 0; i < PIXELS; i++) rgba.set(palette[tokens[i]!] ?? [0, 0, 0, 0], i * 4)
  return rgba
}
