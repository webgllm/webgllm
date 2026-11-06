/**
 * The memory a WebGLLM's loaded models take together. Each model loads in
 * its own worker; before it loads (and when its cache grows) the worker
 * claims its bytes, and the least recently used other models are unloaded
 * until everything fits the budget. A phone can't hold two 1 GB models (an
 * iPhone Safari tab has 2 GB), and small models still stay loaded side by
 * side.
 */

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
  /** A counter, not the time: two uses in one millisecond still order. */
  private clock = 0

  constructor(
    /** Bytes all models may take together. */
    private readonly limit: () => number,
    /** Unloads a model to make room (it may wait for the model's generation to end). */
    private readonly unload: (model: string) => Promise<void> | void
  ) {}

  /**
   * Makes room for `model`'s `bytes` (in place of what it held before),
   * unloading other models, least recently used first, until it fits
   * (models kept loaded stay, even past the budget), then records it.
   */
  claim(model: string, bytes: number, keepLoaded: boolean): Promise<void> {
    const run = this.claims.then(async () => {
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
        await this.unload(other.model)
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
