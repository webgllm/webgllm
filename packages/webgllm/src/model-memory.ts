/**
 * The memory the page's models take, shared by this project's runtimes
 * (Transformers.js, WebGL, wllama), each of which loads models in its own
 * worker. Before a runtime loads a model it claims the memory: other
 * runtimes' least recently used models are unloaded until everything fits
 * the budget. A phone can't hold two 1 GB models (an iPhone Safari tab has
 * 2 GB), and small models still stay loaded side by side.
 *
 * The packages each carry this file and share one registry through
 * globalThis, so they find each other even when bundled apart. Keep the
 * copies identical (packages/{transformers-js,webgl,wllama}/src).
 */

/** A runtime's loaded model, as the registry knows it. */
type Held = {
  /** The runtime ("transformers-js", "webgl" or "wllama"). */
  owner: string
  model: string
  /** Its memory, estimated: weights and cache. */
  bytes: number
  lastUsed: number
  /** Never unloaded to make room for another runtime's model. */
  keepLoaded: boolean
  unload: () => unknown
}

type Registry = {
  version: 1
  held: Map<string, Held>
  /** Bytes all models may take together; the device's default if unset. */
  budget?: number
}

const KEY = Symbol.for("webgllm/model-memory")

function registry(): Registry {
  const scope = globalThis as unknown as Record<symbol, Registry | undefined>
  return (scope[KEY] ??= { version: 1, held: new Map() })
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

/**
 * Sets how many bytes the page's models may take together, across this
 * project's runtimes, from the next load on; undefined for the default.
 */
export function setMemoryBudget(bytes: number | undefined) {
  registry().budget = bytes
}

/**
 * Makes room for `owner`'s model of about `bytes`, unloading other runtimes'
 * models, least recently used first, until it fits the budget (models kept
 * loaded stay), then records it in place of `owner`'s previous one.
 */
export async function claimMemory(model: {
  owner: string
  model: string
  bytes: number
  keepLoaded: boolean
  unload: () => unknown
}) {
  const { held, budget } = registry()
  const limit = budget ?? defaultMemoryBudget()
  held.delete(model.owner)
  const others = [...held.values()].sort((a, b) => a.lastUsed - b.lastUsed)
  let total = others.reduce((sum, h) => sum + h.bytes, model.bytes)
  for (const other of others) {
    if (total <= limit) break
    if (other.keepLoaded) continue
    held.delete(other.owner)
    total -= other.bytes
    await other.unload()
  }
  held.set(model.owner, { ...model, lastUsed: Date.now() })
}

/** Marks `owner`'s model as just used (for least recently used order). */
export function touchMemory(owner: string) {
  const held = registry().held.get(owner)
  if (held) held.lastUsed = Date.now()
}

/** Forgets `owner`'s model, once it's unloaded or failed to load. */
export function releaseMemory(owner: string) {
  registry().held.delete(owner)
}
