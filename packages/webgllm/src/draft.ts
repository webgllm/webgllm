/**
 * Drafts for speculative decoding by prompt lookup: guesses at the next
 * tokens, taken from the text so far. The engine checks a token and its
 * drafts in one batch, as fast as one token alone (writing is bound by the
 * number of GPU passes, not their width), and keeps the drafts the model
 * agrees with.
 */

/**
 * Tokens that may follow `tokens`: those that followed the earlier
 * occurrence matching the most last tokens (up to `longest`; the latest of
 * equally long ones), or none below `minMatch`. In repeated code, the latest
 * few matching tokens are often another copy's boilerplate. A longer match
 * is likelier to go on, so it drafts more: up to 2^(n−1) tokens after an
 * n-token match, and at most `max`.
 */
export function lookupDraft(
  tokens: ArrayLike<number>,
  max: number,
  minMatch = 1,
  longest = 32
): Array<number> {
  const end = tokens.length
  if (max <= 0 || end < 2) return []
  const last = tokens[end - 1]
  let best = 0
  let from = -1
  for (let next = end - 1; next >= 1; next--) {
    if (tokens[next - 1] !== last) continue
    let n = 1
    while (n < longest && next - 1 - n >= 0 && tokens[next - 1 - n] === tokens[end - 1 - n]) n++
    if (n > best) {
      best = n
      from = next
      if (n === longest) break
    }
  }
  if (best < Math.max(1, minMatch)) return []
  const most = Math.min(max, 2 ** Math.min(best - 1, 30))
  const out: Array<number> = []
  for (let i = from; i < end && out.length < most; i++) out.push(tokens[i])
  return out
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
/**
 * Decayed average ms of a token alone, and of a batch per draft count: what
 * drafts cost on this device, which a policy can share with the next
 * reply's (see DraftPolicy's `times`).
 */
export type DraftTimes = {
  single: number | null
  batch: Array<number | null>
  /** Batches timed so far. */
  samples?: number
}

/** No times measured yet, for drafts up to `max`. */
export function draftTimes(max: number): DraftTimes {
  return { single: null, batch: new Array(max + 1).fill(null) }
}

export class DraftPolicy {
  /** Decayed counts per draft position: kept, and checked. */
  private kept: Array<number>
  private checked: Array<number>
  /**
   * Drafts found, scored as tokens are drawn from position `from` on (the
   * ones not checked; those before it a batch checks itself).
   */
  private shadow: { drafts: Array<number>; at: number; from: number } | null = null
  /**
   * Drafts that must come true before the first batch is timed: one where a
   * batch is ready to run, two where it would first have shaders to compile.
   */
  startAfter = 2

  constructor(
    readonly max: number,
    /** Before any outcome: how often a draft is taken to be right. */
    private prior = 0.5,
    /** Weight of the prior, in outcomes. */
    private priorWeight = 0.5,
    /** Each outcome counts this much less with each newer one (~1 / (1 − decay) remembered). */
    private decay = 0.95,
    /** The same for times. */
    private timeDecay = 0.8,
    /**
     * The times measured (updated in place): shared from reply to reply,
     * a device whose batches cost more than drafts earn doesn't pay for a
     * batch again each reply to find out.
     */
    private times: DraftTimes = draftTimes(max)
  ) {
    this.kept = new Array(max + 1).fill(0)
    this.checked = new Array(max + 1).fill(0)
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
    const own = this.times.batch[d]
    if (own !== null) return own
    // Unmeasured: on the line through the two measured counts around it or
    // just below it (a batch has a fixed cost, so one count's time per row
    // overprices a bigger one), at least 3% more per draft. With one count
    // measured, 3% more per draft, so the next size gets tried and timed;
    // with none, a token alone's (so drafts get tried).
    let below = d - 1
    while (below >= 1 && this.times.batch[below] === null) below--
    let above = d + 1
    while (above <= this.max && this.times.batch[above] === null) above++
    const low = below >= 1 ? this.times.batch[below] : null
    const high = above <= this.max ? this.times.batch[above] : null
    if (low !== null && high !== null) return low + ((high - low) * (d - below)) / (above - below)
    if (low !== null) {
      let previous = below - 1
      while (previous >= 1 && this.times.batch[previous] === null) previous--
      const before = previous >= 1 ? this.times.batch[previous] : null
      const slope = before !== null ? (low - before) / (below - previous) : 0
      return low + Math.max(0.03 * low, slope) * (d - below)
    }
    if (high !== null) return high * (1 + 0.03 * (d - above))
    return this.times.single ?? 0
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
    if (this.times.single === null) return 0
    // Until a batch is timed: none before `startAfter` drafts have come true
    // (a reply that repeats nothing never pays for one), then one draft, to
    // find out what batches cost here for less than a full one would where
    // they don't pay. Twice if the first looks slow, so that one stall
    // doesn't end drafting.
    const samples = this.times.samples ?? 0
    const cheap = samples === 1 && (this.times.batch[1] ?? Infinity) <= 1.5 * this.times.single
    if (samples < 2 && !cheap) return this.kept[1] < this.startAfter - 0.5 ? 0 : 1
    // A second size next, at most 3 drafts: two sizes price the bigger ones.
    const sizes = this.times.batch.filter((t) => t !== null).length
    const most = sizes < 2 ? Math.min(n, 3) : n
    let best = 0
    let bestRate = 1 / this.times.single
    let expect = 1
    let run = 1
    for (let d = 1; d <= most; d++) {
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

  /**
   * Drafts found while no batch would be read: scored against the next
   * tokens like the ones `choose` leaves unchecked, so a reply whose guesses
   * keep hitting (and so never reaches `choose`) can still learn to draft.
   */
  watch(available: Array<number>) {
    if (!this.shadow && available.length)
      this.shadow = { drafts: available.slice(0, this.max), at: 0, from: 0 }
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
   * `drafts` if the reply ended first). `compiled`: shaders compiled during
   * it, so its time says nothing about what the next one will cost.
   */
  checkedBatch(drafts: number, kept: number, ms: number, judged = drafts, compiled = false) {
    // (The shadow, if any, goes on: the batch's kept drafts were drawn
    // already, and the token drawn after them scores the next draft.)
    for (let i = 1; i <= Math.min(judged, kept + 1); i++) this.outcome(i, i <= kept)
    if (compiled) return
    const samples = this.times.samples ?? 0
    this.times.samples = samples + 1
    const old = this.times.batch[drafts]
    this.times.batch[drafts] =
      old === null
        ? Math.min(ms, 3 * this.batchMs(drafts) || ms)
        : samples === 1
          ? Math.min(old, ms) // (The first two: the lesser, in case one stalled.)
          : this.timeDecay * old + (1 - this.timeDecay) * Math.min(ms, 3 * old)
  }

  /** A token alone (no drafts) took `ms`, waits it overlapped included. */
  alone(ms: number) {
    this.times.single =
      this.times.single === null
        ? ms
        : this.timeDecay * this.times.single + (1 - this.timeDecay) * Math.min(ms, 3 * this.times.single)
  }

  private outcome(i: number, right: boolean) {
    this.kept[i] = this.decay * this.kept[i] + (right ? 1 : 0)
    this.checked[i] = this.decay * this.checked[i] + 1
  }
}
