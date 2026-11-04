/** Report paint cadence without doing any rendering or identifying the device. */
export function monitorFrames(
  send: (
    intervalMs: number,
    expectedMs: number,
    frames: number,
    startedAt?: number,
    meanMs?: number
  ) => void
) {
  if (typeof requestAnimationFrame === "undefined") return () => undefined
  const recent: number[] = []
  const batch: number[] = []
  let frame = 0,
    last = 0,
    expected = Infinity,
    count = 0,
    batchStart = 0
  const tick = (now: number) => {
    if (document.visibilityState !== "visible") {
      last = 0
      count = 0
      batch.length = 0
      recent.length = 0
      expected = Infinity
    } else {
      const elapsed = last ? now - last : 0
      last = now
      if (elapsed > 0 && elapsed < 1000) {
        // Short catch-up callbacks belong in the mean even though they
        // cannot establish a display's nominal refresh period.
        if (elapsed >= 4) {
          recent.push(elapsed)
          if (recent.length > 120) recent.shift()
          const ordered = recent.slice().sort((a, b) => a - b)
          const fast = ordered.filter(
            (value) => value <= ordered[Math.floor(ordered.length * 0.1)] * 1.25
          )
          // Whole-millisecond clocks alternate 8/9 or 16/17 ms at a
          // healthy cadence. Their median alone underestimates the period.
          const rounded = fast.length >= 12 && fast.every(
            value => Math.abs(value - Math.round(value)) < 1e-6
          )
          expected = rounded
            ? fast.reduce((sum, value) => sum + value, 0) / fast.length
            : fast[Math.floor(fast.length / 2)]
        }
        if (!Number.isFinite(expected)) {
          frame = requestAnimationFrame(tick)
          return
        }
        if (!batch.length) batchStart = performance.timeOrigin + now - elapsed
        batch.push(elapsed)
        count++
        const stalled = elapsed > expected * 1.8
        if (count >= 24 || stalled) {
          const p95 = batch.slice().sort((a, b) => a - b)[
            Math.floor(batch.length * 0.95)
          ]
          send(
            stalled ? elapsed : p95,
            expected,
            count,
            batchStart,
            batch.reduce((sum, value) => sum + value, 0) / batch.length
          )
          count = 0
          batch.length = 0
        }
      }
    }
    frame = requestAnimationFrame(tick)
  }
  frame = requestAnimationFrame(tick)
  return () => cancelAnimationFrame(frame)
}
