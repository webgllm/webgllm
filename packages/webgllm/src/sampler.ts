/** Picking the next token from logits, as the package's other runtimes do. */

/** Sampling settings, from the model card unless the caller overrides them. */
export type Sampling = {
  temperature: number
  top_k: number
  repetition_penalty: number
  /** Nucleus sampling: the likeliest tokens up to this probability. */
  top_p?: number
}

/**
 * A sampler for one reply. It applies the repetition penalty to every token
 * seen (prompt and reply, as Transformers.js does), keeps the top_k
 * likeliest tokens, and draws one at the temperature (greedy at 0).
 */
export function createSampler({
  temperature,
  top_k,
  repetition_penalty,
  top_p = 1
}: Sampling) {
  const seen = new Set<number>()
  return {
    /** Tokens to penalize from now on: the prompt, then each one drawn. */
    see(tokens: Iterable<number>) {
      for (const t of tokens) seen.add(t)
    },
    /** The next token. Changes `logits`. */
    sample(logits: Float32Array, random: () => number = Math.random): number {
      if (repetition_penalty !== 1) {
        for (const token of seen) {
          const x = logits[token]
          logits[token] =
            x < 0 ? x * repetition_penalty : x / repetition_penalty
        }
      }
      // The k largest, in no order: replace the smallest kept when beaten.
      const k = Math.max(
        1,
        Math.min(top_k > 0 ? top_k : logits.length, logits.length)
      )
      const keptIds = new Int32Array(k)
      const kept = new Float32Array(k).fill(-Infinity)
      let smallest = 0
      for (let i = 0; i < logits.length; i++) {
        const x = logits[i]
        if (!(x > kept[smallest])) continue
        kept[smallest] = x
        keptIds[smallest] = i
        smallest = 0
        for (let j = 1; j < k; j++) if (kept[j] < kept[smallest]) smallest = j
      }
      let best = 0
      for (let j = 1; j < k; j++) if (kept[j] > kept[best]) best = j
      if (temperature <= 0) return keptIds[best]
      const max = kept[best]
      let total = 0
      for (let j = 0; j < k; j++) {
        kept[j] = Math.exp((kept[j] - max) / temperature)
        total += kept[j]
      }
      if (top_p < 1) {
        const order = Array.from({ length: k }, (_, j) => j).sort(
          (a, b) => kept[b] - kept[a]
        )
        let share = 0
        let cut = false
        for (const j of order) {
          if (cut) {
            total -= kept[j]
            kept[j] = 0
          } else {
            share += kept[j] / total
            cut = share >= top_p
          }
        }
      }
      let draw = random() * total
      for (let j = 0; j < k; j++) {
        draw -= kept[j]
        if (draw <= 0) return keptIds[j]
      }
      return keptIds[best]
    }
  }
}
