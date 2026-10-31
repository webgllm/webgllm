/**
 * Drafts for speculative decoding by prompt lookup: guesses at the next
 * tokens, taken from the text so far. The engine checks a token and its
 * drafts in one batch, as fast as one token alone (writing is bound by the
 * number of GPU passes, not their width), and keeps the drafts the model
 * agrees with.
 */

/**
 * Up to `max` tokens that may follow `tokens`: those that followed the
 * latest earlier occurrence of its last 3 tokens (else 2, ... down to
 * `minMatch`), or none.
 */
export function lookupDraft(
  tokens: ArrayLike<number>,
  max: number,
  minMatch = 1
): Array<number> {
  if (max <= 0) return []
  const end = tokens.length
  for (let n = 3; n >= minMatch; n--) {
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
      const out: Array<number> = []
      for (let i = start + n; i < end && out.length < max; i++) out.push(tokens[i])
      return out
    }
  }
  return []
}
