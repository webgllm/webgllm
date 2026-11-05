/** Snapshot roles share the engine's existing count/memory allowance. */
export type SnapshotRole = "anchor" | "prefix"
export type Checkpoint = { position: number; anchor?: boolean; prefix?: boolean }

/** Old discretionary points go first; a role being replaced need not survive. */
export function checkpointsToDrop(points: readonly Checkpoint[], keep: number, replacing?: SnapshotRole): number[] {
  const priority = (point: Checkpoint) => replacing && point[replacing] && !(replacing === "anchor" ? point.prefix : point.anchor)
    ? -1 : point.prefix ? 2 : point.anchor ? 1 : 0
  return points.map((point, index) => ({ ...point, index }))
    .sort((a, b) => priority(a) - priority(b) || a.index - b.index)
    .slice(0, Math.max(0, points.length - Math.max(0, keep)))
    .map(point => point.position)
}


/**
 * Actual matching tokens bound every checkpoint. Keep a useful retained point
 * when a small schema edit merely extends it within the same binary size band;
 * promoting on every few-token extension would evict the safer shared fallback.
 * A larger working set can still promote it, and a mismatch always shortens it.
 */
export function reusablePrefixFor(shared: number, declared?: number, retained?: number): number | undefined {
  if (shared === 0) return declared
  const available = Math.min(shared, declared ?? shared)
  return retained !== undefined && retained <= available && Math.floor(Math.log2(retained)) === Math.floor(Math.log2(available))
    ? retained : available
}


/** Retain baseline long-prefix behavior until a shorter content prefix is demonstrated repeatedly. */
export function adaptivePrefixFor(o: {
  shared: number; declared?: number; retained?: number; previousShared?: number; natural?: number; framing: number
}): { prefix?: number; previousShared?: number } {
  if (o.shared === 0 || o.shared <= o.framing) return { prefix: o.declared }
  const agrees = o.previousShared !== undefined && Math.floor(Math.log2(o.previousShared)) === Math.floor(Math.log2(o.shared))
  const lower = agrees ? Math.min(o.shared, o.previousShared!) : o.shared
  // A profile switch must not replace a useful normal checkpoint on its first miss.
  if (o.natural !== undefined && o.shared < o.natural && !agrees)
    return { prefix: o.declared, previousShared: o.shared }
  const available = reusablePrefixFor(o.shared, o.declared, o.retained)
  return { prefix: available === undefined ? undefined : Math.min(available, lower), previousShared: lower }
}
