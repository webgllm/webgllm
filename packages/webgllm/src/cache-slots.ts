/**
 * Writing's queries in their layer's KV cache texture (see
 * TUNING.qInCache): the slots (cache rows) past the tokens being written
 * are free until the next tokens' keys and values overwrite them, so the
 * queries go there, drawn in the same render pass as the new keys and
 * values.
 */

/** A run of consecutive cache slots within one block of rows (and one lap of a ring). */
export type SlotRun = {
  /** The run's first slot, counted from the first one asked for. */
  s0: number
  /** Its texture row. */
  row0: number
  /** Its block of rows (blocks sit side by side, a cache row's width apart). */
  block: number
  /** Slots in the run. */
  run: number
}

/**
 * `count` slots from position slot0 in runs that end where a block of
 * `blockRows` rows ends or a ring of `ring` rows wraps (0: no ring).
 */
export function slotRuns(
  slot0: number,
  count: number,
  blockRows: number,
  ring: number
): Array<SlotRun> {
  const runs: Array<SlotRun> = []
  for (let s0 = 0; s0 < count;) {
    const slot = ring ? (slot0 + s0) % ring : slot0 + s0
    const row0 = slot % blockRows
    const run = Math.min(count - s0, blockRows - row0, ring ? ring - slot : Infinity)
    runs.push({ s0, row0, block: Math.floor(slot / blockRows), run })
    s0 += run
  }
  return runs
}

/**
 * Whether `rows` tokens written at pos0 can put `slots` slots each of
 * queries in the slots after their own: a full cache must hold them (it has
 * `positions` rows); a ring (`ring` rows, `window` keys read back) must not
 * reuse a slot whose key any of the tokens still reads.
 */
export function queriesFit(o: {
  pos0: number
  rows: number
  slots: number
  ring: number
  window: number
  positions: number
}) {
  const extra = o.rows * o.slots
  // A ring's slot p % ring holds position p - ring: the query slots reach
  // back to pos0 + rows + extra - 1 - ring, which must be older than the
  // first token's window (pos0 - window + 1).
  return o.ring
    ? o.rows + extra <= o.ring - o.window
    : o.pos0 + o.rows + extra <= o.positions
}
