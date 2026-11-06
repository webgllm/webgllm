/**
 * Drafts for speculative decoding by prompt lookup: guesses at the next
 * tokens, taken from the text so far. The engine checks a token and its
 * drafts in one batch, as fast as one token alone (writing is bound by the
 * number of GPU passes, not their width), and keeps the drafts the model
 * agrees with.
 */

/**
 * Tokens that may follow `tokens`: those that followed the latest earlier
 * occurrence of its last 4 tokens (else 3, ... down to `minMatch`), or
 * none. A longer match is likelier to go on, so it drafts more: up to
 * 2^(n−1) tokens after an n-token match, and at most `max`.
 */
export function lookupDraft(
  tokens: ArrayLike<number>,
  max: number,
  minMatch = 1
): Array<number> {
  if (max <= 0) return []
  const end = tokens.length
  for (let n = 4; n >= minMatch; n--) {
    if (end <= n) continue
    for (let start = end - n - 1; start >= 0; start--) {
      let match = true
      for (let j = 0; j < n; j++) {
        if (tokens[start + j] !== tokens[end - n + j]) {
          match = false
          break
        }
      }
      if (!match) continue
      const most = Math.min(max, 2 ** (n - 1))
      const out: Array<number> = []
      for (let i = start + n; i < end && out.length < most; i++) out.push(tokens[i])
      return out
    }
  }
  return []
}

/**
 * How many drafts to check with each written token, from what they've
 * cost and earned so far in this reply: the count that writes the most
 * tokens per millisecond, expected tokens 1 + a₁ + a₁a₂ + … over the
 * measured time of a batch that size, against the measured time of a
 * token alone (0 drafts). aᵢ is how often draft i was right given the
 * ones before it were. Drafts not checked are still scored against the
 * tokens then drawn (a draft is kept with exactly the probability the
 * sampler draws it), so a policy that stopped drafting can start again.
 */
export class DraftPolicy {
  /** Decayed counts per draft position: kept, and checked. */
  private kept: Array<number>
  private checked: Array<number>
  /** Decayed average ms of a token alone, and of a batch per draft count. */
  private single: number | null = null
  private batch: Array<number | null>
  /**
   * Drafts found, scored as tokens are drawn from position `from` on (the
   * ones not checked; those before it a batch checks itself).
   */
  private shadow: { drafts: Array<number>; at: number; from: number } | null = null

  constructor(
    readonly max: number,
    /** Before any outcome: how often a draft is taken to be right. */
    private prior = 0.5,
    /** Weight of the prior, in outcomes. */
    private priorWeight = 0.5,
    /** Each outcome counts this much less with each newer one (~1 / (1 − decay) remembered). */
    private decay = 0.95,
    /** The same for times. */
    private timeDecay = 0.8
  ) {
    this.kept = new Array(max + 1).fill(0)
    this.checked = new Array(max + 1).fill(0)
    this.batch = new Array(max + 1).fill(null)
  }

  /**
   * How often draft i (1-based) is right when drafts before it were: its
   * own outcomes, leaning on draft i − 1's rate (the prior for draft 1)
   * while it has few.
   */
  rate(i: number): number {
    const lean = i === 1 ? this.prior : this.rate(i - 1)
    return (
      (this.kept[i] + lean * this.priorWeight) /
      (this.checked[i] + this.priorWeight)
    )
  }

  /** Expected ms of a batch of a token and d drafts (d ≥ 1). */
  private batchMs(d: number): number {
    const own = this.batch[d]
    if (own !== null) return own
    // Unmeasured: the nearest measured count's time, 3% more per draft
    // further; with none measured, a token alone's (so drafts get tried).
    for (let k = 1; k <= this.max; k++) {
      for (const j of [d - k, d + k]) {
        const t = j >= 1 && j <= this.max ? this.batch[j] : null
        if (t !== null) return t * (1 + 0.03 * (d - j))
      }
    }
    return this.single ?? 0
  }

  /**
   * Of `available` drafts (from lookupDraft), how many to check now.
   * Unchecked ones are remembered and scored against the next tokens.
   */
  choose(available: Array<number>): number {
    const best = this.peek(available)
    // The ones past `best` too: if the batch keeps all it checked, the
    // tokens drawn after it score them (positions a short batch never
    // reaches would otherwise keep their old rates).
    this.shadow =
      available.length > best
        ? { drafts: available.slice(0, this.max), at: 0, from: best }
        : null
    return best
  }

  /** What `choose` would answer, remembering nothing. */
  peek(available: Array<number>): number {
    const n = Math.min(available.length, this.max)
    if (n === 0) return 0
    // A token alone first, to have its time to beat.
    if (this.single === null) return 0
    let best = 0
    let bestRate = 1 / this.single
    let expect = 1
    let run = 1
    for (let d = 1; d <= n; d++) {
      run *= this.rate(d)
      expect += run
      const ms = this.batchMs(d)
      const rate = ms > 0 ? expect / ms : Infinity
      if (rate > bestRate) {
        best = d
        bestRate = rate
      }
    }
    return best
  }

  /** A token was written (any way): scores remembered unchecked drafts. */
  drawn(token: number) {
    const s = this.shadow
    if (!s) return
    const i = s.at + 1
    const right = s.drafts[s.at] === token
    // (Drafts a batch checked are scored by checkedBatch.)
    if (s.at >= s.from) this.outcome(i, right)
    s.at++
    if (!right || s.at >= s.drafts.length) this.shadow = null
  }

  /**
   * A batch of a token and `drafts` drafts took `ms`, and its first
   * `kept` drafts were kept, of the first `judged` checked (fewer than
   * `drafts` if the reply ended first).
   */
  checkedBatch(drafts: number, kept: number, ms: number, judged = drafts) {
    // (The shadow, if any, goes on: the batch's kept drafts were drawn
    // already, and the token drawn after them scores the next draft.)
    for (let i = 1; i <= Math.min(judged, kept + 1); i++) this.outcome(i, i <= kept)
    const old = this.batch[drafts]
    this.batch[drafts] =
      old === null
        ? Math.min(ms, 3 * this.batchMs(drafts) || ms)
        : this.timeDecay * old + (1 - this.timeDecay) * Math.min(ms, 3 * old)
  }

  /** A token alone (no drafts) took `ms`, waits it overlapped included. */
  alone(ms: number) {
    this.single =
      this.single === null
        ? ms
        : this.timeDecay * this.single + (1 - this.timeDecay) * Math.min(ms, 3 * this.single)
  }

  private outcome(i: number, right: boolean) {
    this.kept[i] = this.decay * this.kept[i] + (right ? 1 : 0)
    this.checked[i] = this.decay * this.checked[i] + 1
  }
}
