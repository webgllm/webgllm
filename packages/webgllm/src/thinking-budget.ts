/** A thinking-only ceiling; forced delimiters still consume the total output limit. */
export class ThinkingBudget {
  private inThinking: boolean
  private tokens = 0
  private exhausted: boolean
  private closedAtBudget = false
  private closingMask: Uint32Array | undefined
  /** How much of the close has been written once the budget ran out (-1: not closing). */
  private step = -1
  /** The close being written: the wrap-up and `closing`, or `closing` alone after no thinking. */
  private active: readonly number[]
  private readonly wrappedClosing: readonly number[]

  constructor(
    readonly limit: number,
    private readonly think: number,
    private readonly thinkEnd: number,
    startsInThinking: boolean,
    /**
     * What closes thinking at the budget, as the chat template writes it
     * between a reply's reasoning and its answer (`templateClosing`): the end
     * marker alone for LFM3, "\n</think>\n\n" for Qwen3.5. A close cut down to
     * the marker leaves Qwen3.5 mid-sentence and it carries on reasoning in
     * its answer.
     */
    private readonly closing: readonly number[] = [thinkEnd],
    /**
     * Written into the thinking before `closing` (`budgetWrapUp`): cut off
     * mid-thought, LFM2.5 Thinking carries on reasoning in its answer even
     * after the template's close. Left out when no thinking was written.
     */
    wrapUp: readonly number[] = []
  ) {
    this.inThinking = startsInThinking
    this.exhausted = startsInThinking && limit === 0
    this.wrappedClosing = [...wrapUp, ...closing]
    this.active = closing
  }

  /** The close a budget running out now would write. */
  private get nextClose() {
    return this.tokens > 0 ? this.wrappedClosing : this.closing
  }

  /** Consult before sampling or accepting a speculative token. */
  get forcedToken(): number | null {
    if (this.step >= 0) return this.step < this.active.length ? this.active[this.step] : null
    return this.inThinking && this.tokens >= this.limit ? this.nextClose[0] : null
  }

  /** Only emitted tokens count; rejected drafts and pipeline guesses never do. */
  observe(token: number) {
    const forced = this.forcedToken
    const closingStep = forced !== null && token === forced
    if (closingStep) {
      if (this.step < 0) this.active = this.nextClose
      this.step = Math.max(this.step, 0) + 1
    }
    if (token === this.think) {
      this.inThinking = true
      this.step = -1
    } else if (token === this.thinkEnd) {
      if (this.inThinking && this.tokens >= this.limit) this.closedAtBudget = true
      this.inThinking = false
    }
    // The close's own text isn't thinking.
    else if (this.inThinking && !closingStep) this.tokens++
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
  startsInThinking: boolean,
  closing?: readonly number[],
  wrapUp?: readonly number[]
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
  if (closing && (closing.filter((id) => id === thinkEnd).length !== 1 || closing.includes(think))) {
    throw new Error("A thinking budget's closing must hold the </think> marker exactly once, and no <think>.")
  }
  if (wrapUp && (wrapUp.includes(think) || wrapUp.includes(thinkEnd))) {
    throw new Error("A thinking budget's wrap-up can't hold a <think> or </think> marker.")
  }
  return new ThinkingBudget(limit, think, thinkEnd, startsInThinking, closing, wrapUp)
}

type ClosingTokenizer = {
  specialTokens: { think: number | null; thinkEnd: number | null }
  applyChatTemplate(messages: Array<{ role: "user" | "assistant"; content: string; reasoning_content?: string }>, options?: { enableThinking?: boolean }): string
  encode(text: string, options?: { addSpecialTokens?: boolean }): Array<number>
}

const closings = new WeakMap<object, readonly number[]>()

/**
 * The tokens the chat template writes between a reply's reasoning and its
 * answer, read by writing one back: a budget closes thinking the same way.
 * The end marker alone when the template shows no single close.
 */
export function templateClosing(tokenizer: ClosingTokenizer): readonly number[] {
  const cached = closings.get(tokenizer)
  if (cached) return cached
  const { think, thinkEnd } = tokenizer.specialTokens
  let closing: readonly number[] = thinkEnd === null ? [] : [thinkEnd]
  if (thinkEnd !== null) {
    try {
      const reasoning = "⁣reasoning⁣", answer = "⁣answer⁣"
      const text = tokenizer.applyChatTemplate(
        [{ role: "user", content: "?" }, { role: "assistant", content: answer, reasoning_content: reasoning }],
        { enableThinking: true }
      )
      const from = text.indexOf(reasoning), to = text.indexOf(answer)
      if (from >= 0 && to > from) {
        const ids = tokenizer.encode(text.slice(from + reasoning.length, to), { addSpecialTokens: false })
        if (ids.filter((id) => id === thinkEnd).length === 1 && (think === null || !ids.includes(think))) closing = ids
      }
    } catch {
      // A template that can't write reasoning back: the marker alone.
    }
  }
  closings.set(tokenizer, closing)
  return closing
}

/**
 * What a budget writes into the thinking before closing it, so the model
 * knows to answer from what it has rather than go on reasoning (Qwen's
 * guidance for thinking budgets does the same).
 */
export const BUDGET_WRAP_UP = "\n\nI'm out of thinking time, so I'll answer now from what I've worked out."

const wrapUps = new WeakMap<object, readonly number[]>()

/** `BUDGET_WRAP_UP`'s tokens. */
export function budgetWrapUp(tokenizer: Pick<ClosingTokenizer, "encode">): readonly number[] {
  let ids = wrapUps.get(tokenizer)
  if (!ids) wrapUps.set(tokenizer, ids = tokenizer.encode(BUDGET_WRAP_UP, { addSpecialTokens: false }))
  return ids
}
