/** Model configs declare no output cap: EOS or remaining context is the default. */
export function outputTokenLimit(remainingContext: number, requested?: number): number {
  if (requested === undefined) return remainingContext
  if (!Number.isSafeInteger(requested) || requested < 0) {
    throw new RangeError("maxTokens must be a nonnegative safe integer.")
  }
  return Math.min(requested, remainingContext)
}
