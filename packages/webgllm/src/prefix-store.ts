/**
 * Saved prompt prefixes: the engine's state at the end of a prompt's stable
 * prefix (its system prompt and tools), kept after its history moves on to
 * another prompt, so a later request starting with the same tokens reads
 * only what comes after them.
 *
 * The engine keeps one history, and rewinding along it reuses what the
 * cache already holds. A request with another system prompt overwrites
 * that history, so going back to the first (A, B, A) read it all again.
 * Each entry here holds what that needs: its tokens (the keys the cache
 * compares) and a copy of the state (see Engine.savePrefix). Entries are
 * freed least recently used first once their bytes pass the budget.
 */
export const DEFAULT_PROMPT_CACHE_BYTES = 128 * 2 ** 20

export type PrefixEntry<S> = {
  /** The prefix's keys (token ids, an image's rows by its key). */
  keys: Array<number>
  /** GPU bytes the state holds. */
  bytes: number
  state: S
}

export class PrefixStore<S> {
  /** Least recently used first. */
  private entries: Array<PrefixEntry<S>> = []

  constructor(
    private readonly free: (state: S) => void,
    private limit = DEFAULT_PROMPT_CACHE_BYTES
  ) {}

  /** Bytes the entries hold together. */
  get bytes() {
    return this.entries.reduce((sum, entry) => sum + entry.bytes, 0)
  }

  get size() {
    return this.entries.length
  }

  get budget() {
    return this.limit
  }

  /** A new budget: entries past it are freed now, least recently used first. */
  setBudget(bytes: number) {
    if (!Number.isFinite(bytes) || bytes < 0) throw new RangeError("promptCacheBytes must be a nonnegative number.")
    this.limit = bytes
    this.evict(0)
  }

  /**
   * The longest entry whose keys begin `keys`, no longer than `most` (it
   * becomes the most recently used).
   */
  match(keys: ArrayLike<number>, most = keys.length): PrefixEntry<S> | undefined {
    let best: PrefixEntry<S> | undefined
    for (const entry of this.entries) {
      const n = entry.keys.length
      if (n > most || n > keys.length || (best && n <= best.keys.length)) continue
      if (startsWith(keys, entry.keys)) best = entry
    }
    if (best) this.touch(best)
    return best
  }

  /** Whether an entry holds exactly `keys` (it becomes the most recently used). */
  has(keys: ArrayLike<number>) {
    const entry = this.entries.find((e) => e.keys.length === keys.length && startsWith(keys, e.keys))
    if (entry) this.touch(entry)
    return !!entry
  }

  /**
   * Frees the least recently used entries until `bytes` more fit the
   * budget, before they're allocated (so the store never holds more than
   * its budget, even for a moment). False if `bytes` can't fit at all.
   */
  makeRoom(bytes: number) {
    if (!this.fits(bytes)) return false
    this.evict(bytes)
    return true
  }

  /** Whether an entry holds `state` (not freed). */
  holds(state: S) {
    return this.entries.some((e) => e.state === state)
  }

  /** Whether an entry of `bytes` would fit the budget at all. */
  fits(bytes: number) {
    return bytes > 0 && bytes <= this.limit
  }

  /**
   * Keeps `state` for `keys`, freeing the least recently used entries past
   * the budget. A state that can't fit at all is freed, and false returned.
   */
  add(keys: Array<number>, bytes: number, state: S) {
    if (!this.fits(bytes)) {
      this.free(state)
      return false
    }
    const same = this.entries.findIndex((e) => e.keys.length === keys.length && startsWith(keys, e.keys))
    if (same >= 0) this.free(this.entries.splice(same, 1)[0].state)
    this.evict(bytes)
    this.entries.push({ keys: keys.slice(), bytes, state })
    return true
  }

  /** Frees every entry. */
  clear() {
    const entries = this.entries
    this.entries = []
    for (const entry of entries) this.free(entry.state)
  }

  /** Frees the least recently used entries until `room` more bytes fit. */
  private evict(room: number) {
    let total = this.bytes
    while (this.entries.length && total + room > this.limit) {
      const old = this.entries.shift()!
      total -= old.bytes
      this.free(old.state)
    }
  }

  private touch(entry: PrefixEntry<S>) {
    const at = this.entries.indexOf(entry)
    if (at >= 0 && at < this.entries.length - 1) {
      this.entries.splice(at, 1)
      this.entries.push(entry)
    }
  }
}

function startsWith(keys: ArrayLike<number>, prefix: ArrayLike<number>) {
  if (prefix.length > keys.length) return false
  for (let i = 0; i < prefix.length; i++) if (keys[i] !== prefix[i]) return false
  return true
}

/** What savePromptPrefix needs of the engine (see Engine). */
export type PrefixEngine<S> = {
  snapshotPosition(role: "prefix"): number | undefined
  savedPrefixBytes(position: number): number
  savePrefix(position: number): S | undefined
}

/** How savePromptPrefix copes with a failed copy: whether the GPU is gone (then it throws), and where to say so otherwise. */
export type SaveFailure = { lost?: () => boolean; log?: (text: string) => void }

/**
 * Keeps the state at the end of the prompt's system prompt and tools
 * (`prefix` tokens in) in `store` if it lacks it, from the engine's prefix
 * snapshot there (the cache holding `cached` of the prompt's `keys`). Not
 * a prefix with an image in it, nor one past the budget. Room is made
 * before the copy. The copy is optional: if it fails (out of memory, say)
 * with the GPU still there, nothing is kept and the request goes on.
 * Returns what it kept, if anything (for the disk: see prefix-disk.ts).
 */
export function savePromptPrefix<S>(
  engine: PrefixEngine<S>,
  store: PrefixStore<S>,
  cached: number,
  keys: Array<number>,
  images: Array<{ start: number }>,
  prefix: number | undefined,
  failure: SaveFailure = {}
) {
  if (prefix === undefined || prefix <= 0 || prefix > cached) return
  if (engine.snapshotPosition("prefix") !== prefix) return
  if (images.some((image) => image.start < prefix)) return
  const tokens = keys.slice(0, prefix)
  if (store.has(tokens)) return
  const bytes = engine.savedPrefixBytes(prefix)
  if (!store.makeRoom(bytes)) return
  let saved: S | undefined
  try {
    saved = engine.savePrefix(prefix)
  } catch (error) {
    // (savePrefix frees what it made before throwing.)
    if (failure.lost?.()) throw error
    failure.log?.(`Couldn't save a prompt prefix: ${error instanceof Error ? error.message : String(error)}`)
    return undefined
  }
  if (saved && store.add(tokens, bytes, saved)) return { keys: tokens, saved }
  return undefined
}
