/** Owned inference state only; uploaded weights and run-local autoregressive KV stay available. */
export type PromptCacheState = {
  engine: { reset: () => void }
  cached: Array<number>
}

export function clearPromptCache(state: PromptCacheState) {
  try {
    state.engine.reset()
  } finally {
    state.cached.length = 0
  }
}

/** Explicit cachePrompt:false prevents cross-call reuse/retention, not intra-call attention. */
export async function withPromptCachePolicy<T>(state: PromptCacheState, cachePrompt: boolean | undefined, run: () => Promise<T>): Promise<T> {
  if (cachePrompt !== false) return run()
  clearPromptCache(state)
  try {
    return await run()
  } finally {
    clearPromptCache(state)
  }
}
