/** Picking the next token from logits, as the package's other runtimes do. */

/** Sampling settings, from the model card unless the caller overrides them. */
export type Sampling = {
  temperature: number
  top_k: number
  repetition_penalty: number
  /** Nucleus sampling: the likeliest tokens up to this probability. */
  top_p?: number
  /** Minimum probability relative to the most likely token; zero disables it. */
  min_p?: number
  /** Subtract once for tokens generated in this reply (not prompt-only tokens). */
  presence_penalty?: number
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
  top_p = 1,
  min_p = 0,
  presence_penalty = 0
}: Sampling) {
  const seen = new Set<number>()
  const generated = new Set<number>()
  return {
    /** Tokens to penalize from now on: the prompt, then each one drawn. */
    see(tokens: Iterable<number>, output = false) {
      for (const t of tokens) {
        seen.add(t)
        if (output) generated.add(t)
      }
    },
    /**
     * The distribution the next token is drawn from: the kept tokens
     * (indices into `logits`) and their probabilities, after the penalty,
     * top-k, temperature and top-p (one token at probability 1 when
     * greedy). Changes `logits`.
     */
    distribution(logits: Float32Array): { ids: Int32Array; p: Float32Array } {
      if (repetition_penalty !== 1) {
        for (const token of seen) {
          const x = logits[token]
          logits[token] =
            x < 0 ? x * repetition_penalty : x / repetition_penalty
        }
      }
      if (presence_penalty !== 0) {
        for (const token of generated) logits[token] -= presence_penalty
      }
      const k = Math.max(
        1,
        Math.min(top_k > 0 ? top_k : logits.length, logits.length)
      )
      let keptIds: Int32Array
      let kept: Float32Array
      if (k >= logits.length) {
        // Every token kept (no top-k): no selection to make.
        keptIds = new Int32Array(k)
        for (let i = 0; i < k; i++) keptIds[i] = i
        kept = logits.slice()
      } else {
        // The k largest, in no order: replace the smallest kept when beaten.
        // A min-heap of slots finds it, ties to the lowest slot.
        keptIds = new Int32Array(k)
        kept = new Float32Array(k).fill(-Infinity)
        const heap = new Int32Array(k)
        for (let j = 0; j < k; j++) heap[j] = j
        for (let i = 0; i < logits.length; i++) {
          const x = logits[i]
          const slot = heap[0]
          if (!(x > kept[slot])) continue
          kept[slot] = x
          keptIds[slot] = i
          // Sift the slot down to its place.
          let at = 0
          for (;;) {
            let child = 2 * at + 1
            if (child >= k) break
            const right = child + 1
            if (right < k && below(kept, heap[right], heap[child])) child = right
            if (!below(kept, heap[child], slot)) break
            heap[at] = heap[child]
            at = child
          }
          heap[at] = slot
        }
      }
      let best = 0
      for (let j = 1; j < k; j++) if (kept[j] > kept[best]) best = j
      if (temperature <= 0) {
        return { ids: Int32Array.of(keptIds[best]), p: Float32Array.of(1) }
      }
      const max = kept[best]
      let total = 0
      for (let j = 0; j < k; j++) {
        kept[j] = Math.exp((kept[j] - max) / temperature)
        total += kept[j]
      }
      if (min_p > 0) {
        // Softmax weights are relative to the largest logit, whose weight is 1.
        for (let j = 0; j < k; j++) {
          if (kept[j] < min_p) {
            total -= kept[j]
            kept[j] = 0
          }
        }
      }
      if (top_p < 1) {
        // Zero weights change neither the share nor the total: skipped.
        const order = descending(kept)
        let share = 0
        let cut = false
        for (let o = 0; o < order.length; o++) {
          const j = order[o]
          if (cut) {
            total -= kept[j]
            kept[j] = 0
          } else {
            share += kept[j] / total
            cut = share >= top_p
          }
        }
      }
      for (let j = 0; j < k; j++) kept[j] /= total
      return { ids: keptIds, p: kept }
    },
    /** The next token. Changes `logits`. */
    sample(logits: Float32Array, random: () => number = Math.random): number {
      return draw(this.distribution(logits), random)
    }
  }
}

/** Whether slot `a` holds less than slot `b`, ties to the lower slot. */
const below = (kept: Float32Array, a: number, b: number) =>
  kept[a] < kept[b] || (kept[a] === kept[b] && a < b)

/**
 * Indices of `weights`' nonzero entries (the weights are nonnegative), from
 * largest to smallest, ties in index order, as a stable sort gives. A
 * radix sort on the weights' bits: nonnegative floats order as their bits
 * do, and it's much faster than a comparator over a whole vocabulary.
 */
function descending(weights: Float32Array): Int32Array {
  const n = weights.length
  const bits = new Uint32Array(weights.buffer, weights.byteOffset, n)
  let order = new Int32Array(n)
  let keys = new Uint32Array(n)
  let m = 0
  for (let j = 0; j < n; j++) {
    if (!(weights[j] > 0)) continue
    order[m] = j
    keys[m++] = 0x7fffffff - bits[j]
  }
  let nextOrder = new Int32Array(m)
  let nextKeys = new Uint32Array(m)
  const count = new Int32Array(2048)
  for (let shift = 0; shift < 32; shift += 11) {
    count.fill(0)
    for (let i = 0; i < m; i++) count[(keys[i] >>> shift) & 2047]++
    for (let d = 0, sum = 0; d < 2048; d++) {
      const c = count[d]
      count[d] = sum
      sum += c
    }
    for (let i = 0; i < m; i++) {
      const at = count[(keys[i] >>> shift) & 2047]++
      nextOrder[at] = order[i]
      nextKeys[at] = keys[i]
    }
    ;[order, nextOrder] = [nextOrder, order]
    ;[keys, nextKeys] = [nextKeys, keys]
  }
  return order.subarray(0, m)
}

/** A token drawn from a distribution (see `distribution`). */
export function draw(
  d: { ids: Int32Array; p: Float32Array },
  random: () => number = Math.random
): number {
  let left = random()
  let best = 0
  for (let j = 0; j < d.p.length; j++) {
    if (d.p[j] > d.p[best]) best = j
    left -= d.p[j]
    if (left <= 0 && d.p[j] > 0) return d.ids[j]
  }
  return d.ids[best]
}

/**
 * Speculative sampling's step for a drafted token (the draft proposes one
 * token for sure): keep it with the probability `d` gives it; otherwise draw
 * from `d` without it. The tokens kept or drawn are distributed exactly as
 * draws from `d` would be. Returns whether the draft was kept, and if not,
 * the token drawn instead.
 */
export function verify(
  d: { ids: Int32Array; p: Float32Array },
  drafted: number,
  random: () => number = Math.random
): { kept: true } | { kept: false; token: number } {
  let j = -1
  for (let i = 0; i < d.ids.length; i++) if (d.ids[i] === drafted) j = i
  const pd = j < 0 ? 0 : d.p[j]
  if (pd >= 1 || (pd > 0 && random() < pd)) return { kept: true }
  if (j < 0) return { kept: false, token: draw(d, random) }
  const rest = d.p.slice()
  rest[j] = 0
  const total = 1 - pd
  for (let i = 0; i < rest.length; i++) rest[i] /= total
  return { kept: false, token: draw({ ids: d.ids, p: rest }, random) }
}

/**
 * The token at `u` (in [0, 1)) of `d`'s cumulative distribution, with its
 * tokens taken in ascending id order: the same token for the same
 * distribution, whatever order its candidates were read back in (seeded
 * sampling; see seeded-random.ts).
 */
export function drawAt(d: { ids: Int32Array; p: Float32Array }, u: number): number {
  const n = d.ids.length
  let ordered = true
  for (let j = 1; j < n && ordered; j++) ordered = d.ids[j] > d.ids[j - 1]
  const order = ordered ? null : Array.from({ length: n }, (_, j) => j).sort((a, b) => d.ids[a] - d.ids[b])
  let left = u
  let best = order ? order[0] : 0
  for (let k = 0; k < n; k++) {
    const j = order ? order[k] : k
    if (d.p[j] > d.p[best]) best = j
    left -= d.p[j]
    if (left < 0 && d.p[j] > 0) return d.ids[j]
  }
  return d.ids[best]
}

/**
 * verify for seeded sampling: the token at `u` is drawn (drawAt), and the
 * draft is kept exactly when it's that token. Kept with the probability
 * `d` gives it, and replaced by a draw from `d` without it otherwise, as
 * verify does; and a reply's tokens don't depend on which were drafted.
 */
export function verifyAt(
  d: { ids: Int32Array; p: Float32Array },
  drafted: number,
  u: number
): { kept: true } | { kept: false; token: number } {
  const token = drawAt(d, u)
  return token === drafted ? { kept: true } : { kept: false, token }
}
