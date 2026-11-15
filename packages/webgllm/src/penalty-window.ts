/**
 * The tokens the repetition and presence penalties apply to, as llama.cpp
 * keeps them: the last `size` tokens read or written (its `repeat_last_n`,
 * 64 by default), the prompt's included, each penalized once however often
 * it appears there. Penalties apply inside a structured answer too, as they
 * do in llama.cpp, whose grammar holds the answer to its schema meanwhile.
 *
 * `push` returns the tokens whose state changed, for the GPU's copy of the
 * set (see Engine.setSeenState): 1 in the window from the prompt only, 3
 * written in this reply (the presence penalty's too), 0 out of it.
 */
export const DEFAULT_REPEAT_LAST_N = 64

export type SeenState = 0 | 1 | 3

export class PenaltyWindow {
  private readonly ring: Array<{ token: number; generated: boolean }> = []
  private start = 0
  /** Each token in the window: how many times, and how many of those written in this reply. */
  private readonly counts = new Map<number, { all: number; generated: number }>()

  /** `size` tokens: 0 penalizes none, a negative size all of them. */
  constructor(readonly size = DEFAULT_REPEAT_LAST_N) {}

  private stateOf(token: number): SeenState {
    const count = this.counts.get(token)
    return !count ? 0 : count.generated > 0 ? 3 : 1
  }

  /** Takes tokens in order (the prompt's, then each written); returns those whose state changed, with the state. */
  push(tokens: Iterable<number>, generated = false): Array<[number, SeenState]> {
    if (this.size === 0) return []
    const before = new Map<number, SeenState>()
    const note = (token: number) => { if (!before.has(token)) before.set(token, this.stateOf(token)) }
    for (const token of tokens) {
      note(token)
      this.ring.push({ token, generated })
      const count = this.counts.get(token) ?? { all: 0, generated: 0 }
      count.all++
      if (generated) count.generated++
      this.counts.set(token, count)
      if (this.size > 0 && this.ring.length - this.start > this.size) {
        const old = this.ring[this.start++]
        note(old.token)
        const c = this.counts.get(old.token)!
        c.all--
        if (old.generated) c.generated--
        if (c.all === 0) this.counts.delete(old.token)
      }
    }
    // (Drop what has left the ring now and then, so it doesn't grow with the reply.)
    if (this.start > 4096) {
      this.ring.splice(0, this.start)
      this.start = 0
    }
    const changes: Array<[number, SeenState]> = []
    for (const [token, was] of before) {
      const now = this.stateOf(token)
      if (now !== was) changes.push([token, now])
    }
    return changes
  }

  /** The tokens in the window, each once, with its state. */
  entries(): Array<[number, SeenState]> {
    return [...this.counts.keys()].map((token) => [token, this.stateOf(token)])
  }
}
