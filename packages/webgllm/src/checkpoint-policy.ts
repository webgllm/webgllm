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
