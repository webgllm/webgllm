/**
 * The memory a WebGLLM's loaded models take together. Each model loads in
 * its own worker; before it loads (and when its cache grows) the worker
 * claims its bytes, and the least recently used other models are unloaded
 * until everything fits the budget. A phone can't hold two 1 GB models (an
 * iPhone Safari tab has 2 GB), and small models still stay loaded side by
 * side.
 */
import { throwIfAborted } from "./errors.js"

/** A loaded model, as the budget knows it. */
type Held = {
  model: string
  /** Its memory, estimated: weights and cache. */
  bytes: number
  lastUsed: number
  /** Never unloaded to make room for another model. */
  keepLoaded: boolean
}

/**
 * The default budget: 1.5 GB on iOS (a Safari tab gets 2 GB, the page and
 * Safari take the rest), half the device's memory where the browser says
 * (Chrome: up to 8 GB), else 4 GB.
 */
export function defaultMemoryBudget(): number {
  if (typeof navigator === "undefined") return Number.POSITIVE_INFINITY
  const { userAgent, maxTouchPoints } = navigator
  const ios =
    /iPhone|iPad|iPod/.test(userAgent) ||
    (/Macintosh/.test(userAgent) && maxTouchPoints > 1)
  if (ios) return 1.5e9
  const gigabytes = (navigator as { deviceMemory?: number }).deviceMemory
  return gigabytes ? gigabytes * 0.5e9 : 4e9
}

export class MemoryBudget {
  private readonly held = new Map<string, Held>()
  /** Claims one at a time, so two loads don't both count on the same room. */
  private claims: Promise<void> = Promise.resolve()
  /** Models a claim is waiting to unload, and what each claimed since (see claim). */
  private evicting = new Map<string, number>()
  /** A counter, not the time: two uses in one millisecond still order. */
  private clock = 0

  constructor(
    /** Bytes all models may take together. */
    private readonly limit: () => number,
    /**
     * Unloads a model to make room (it may wait for the model's generation
     * to end); false when `signal` aborted first and it stays loaded.
     */
    private readonly unload: (model: string, signal?: AbortSignal) => Promise<boolean | void> | boolean | void
  ) {}

  /**
   * Makes room for `model`'s `bytes` (in place of what it held before),
   * unloading other models, least recently used first, until it fits
   * (models kept loaded stay, even past the budget), then records it.
   * Aborting `signal` (the claimant went away) stops it: models not yet
   * unloaded stay, and nothing is recorded for `model`.
   */
  claim(model: string, bytes: number, keepLoaded: boolean, signal?: AbortSignal): Promise<void> {
    // A claim waiting to unload `model` holds the queue until `model`'s
    // generation ends; that generation's own claim (its vision encoder's
    // textures) can't wait behind it, or neither ends. It's admitted now:
    // the claim waiting on it has counted `model` out already, and `model`
    // unloads when its generation ends (or keeps these bytes if that claim
    // gives up).
    if (this.evicting.has(model)) {
      this.evicting.set(model, bytes)
      return Promise.resolve()
    }
    const run = this.claims.then(async () => {
      throwIfAborted(signal)
      const limit = this.limit()
      const own = this.held.get(model)
      this.held.delete(model)
      const others = [...this.held.values()].sort((a, b) => a.lastUsed - b.lastUsed)
      let total = others.reduce((sum, h) => sum + h.bytes, bytes)
      for (const other of others) {
        if (total <= limit) break
        if (other.keepLoaded) continue
        this.held.delete(other.model)
        total -= other.bytes
        this.evicting.set(other.model, other.bytes)
        let unloaded: boolean | void = undefined
        try {
          unloaded = await this.unload(other.model, signal)
        } finally {
          // (What it claimed while it waited, if it stays.)
          const latest = this.evicting.get(other.model) ?? other.bytes
          this.evicting.delete(other.model)
          if (unloaded === false) this.held.set(other.model, { ...other, bytes: latest })
        }
        throwIfAborted(signal)
      }
      this.held.set(model, {
        model,
        bytes,
        // A claim isn't a use: a growing cache keeps its model's place.
        lastUsed: own?.lastUsed ?? ++this.clock,
        keepLoaded
      })
    })
    this.claims = run.catch(() => undefined)
    return run
  }

  /** Marks `model` as just used (for least recently used order). */
  touch(model: string) {
    const held = this.held.get(model)
    if (held) held.lastUsed = ++this.clock
  }

  setKeepLoaded(model: string, keep: boolean) {
    const held = this.held.get(model)
    if (held) held.keepLoaded = keep
  }

  /** Forgets `model`, once it's unloaded or failed to load. */
  release(model: string) {
    this.held.delete(model)
  }

  /** Bytes held by every model together. */
  get total(): number {
    let sum = 0
    for (const h of this.held.values()) sum += h.bytes
    return sum
  }

  models(): Array<string> {
    return [...this.held.keys()]
  }
}
