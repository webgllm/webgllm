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
