/** A thinking-only ceiling; forced delimiters still consume the total output limit. */
export class ThinkingBudget {
  private inThinking: boolean
  private tokens = 0
  private exhausted: boolean
  private closedAtBudget = false
  private closingMask: Uint32Array | undefined

  constructor(
    readonly limit: number,
    private readonly think: number,
    private readonly thinkEnd: number,
    startsInThinking: boolean
  ) {
    this.inThinking = startsInThinking
    this.exhausted = startsInThinking && limit === 0
  }

  /** Consult before sampling or accepting a speculative token. */
  get forcedToken(): number | null {
    return this.inThinking && this.tokens >= this.limit ? this.thinkEnd : null
  }

  /** Only emitted tokens count; rejected drafts and pipeline guesses never do. */
  observe(token: number) {
    if (token === this.think) this.inThinking = true
    else if (token === this.thinkEnd) {
      if (this.inThinking && this.tokens >= this.limit) this.closedAtBudget = true
      this.inThinking = false
    }
    else if (this.inThinking) this.tokens++
    if (this.inThinking && this.tokens >= this.limit) this.exhausted = true
  }

  /** After the explicitly budgeted span closes, reserve its control markers. */
  closedControlMask(vocabSize: number): Uint32Array | undefined {
    if (!this.closedAtBudget) return undefined
    if (!this.closingMask) {
      this.closingMask = new Uint32Array(Math.ceil(vocabSize / 32)).fill(0xffffffff)
      this.closingMask[this.think >>> 5] &= ~(1 << (this.think & 31))
      this.closingMask[this.thinkEnd >>> 5] &= ~(1 << (this.thinkEnd & 31))
    }
    return this.closingMask
  }

  get stats() {
    return {
      thinkingTokens: this.tokens,
      thinkingBudgetTokens: this.limit,
      thinkingBudgetExhausted: this.exhausted
    }
  }
}

/** With no budget, generation keeps its usual sampling and template behavior. */
export function createThinkingBudget(
  limit: number | undefined,
  markers: { think: number | null; thinkEnd: number | null },
  startsInThinking: boolean
): ThinkingBudget | null {
  if (limit === undefined) return null
  if (!Number.isSafeInteger(limit) || limit < 0) {
    throw new Error("thinkingBudgetTokens must be a nonnegative safe integer.")
  }
  const { think, thinkEnd } = markers
  if (
    think === null || thinkEnd === null || think === thinkEnd ||
    !Number.isSafeInteger(think) || !Number.isSafeInteger(thinkEnd) ||
    think < 0 || thinkEnd < 0
  ) {
    throw new Error("A thinking budget requires distinct single-token <think> and </think> markers.")
  }
  return new ThinkingBudget(limit, think, thinkEnd, startsInThinking)
}
