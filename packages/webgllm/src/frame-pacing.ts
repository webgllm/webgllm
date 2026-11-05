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
  if (typeof requestAnimationFrame === "undefined" || typeof document === "undefined")
    return () => undefined
  const recent: number[] = []
  const batch: number[] = []
  const slower: number[] = []
  let frame = 0,
    last: number | undefined,
    expected = Infinity,
    batchStart = 0,
    slowerStart = 0
  const reset = () => {
    last = undefined
    expected = Infinity
    batch.length = recent.length = slower.length = 0
  }
  const flush = () => {
    if (!batch.length) return
    const ordered = batch.slice().sort((a, b) => a - b)
    send(
      ordered[Math.floor(ordered.length * 0.95)],
      expected,
      batch.length,
      batchStart,
      batch.reduce((sum, value) => sum + value, 0) / batch.length
    )
    batch.length = 0
  }
  const report = (elapsed: number, start: number) => {
    if (elapsed > expected * 1.8) {
      flush()
      // A missed frame gets its own window for GPU attribution.
      send(elapsed, expected, 1, start, elapsed)
      return
    }
    if (!batch.length) batchStart = start
    batch.push(elapsed)
    if (batch.length >= 24) flush()
  }
  const cadence = (elapsed: number) => {
    // Short catch-up callbacks and long stalls cannot set refresh cadence.
    if (elapsed < 4 || elapsed > 50) return
    recent.push(elapsed)
    if (recent.length > 120) recent.shift()
    const ordered = recent.slice().sort((a, b) => a - b)
    const fast = ordered.filter(
      (value) => value <= ordered[Math.floor(ordered.length * 0.1)] * 1.25
    )
    // Whole-millisecond clocks alternate 8/9 or 16/17 ms at a healthy cadence.
    const rounded = fast.length >= 12 && fast.every(
      value => Math.abs(value - Math.round(value)) < 1e-6
    )
    expected = rounded
      ? fast.reduce((sum, value) => sum + value, 0) / fast.length
      : fast[Math.floor(fast.length / 2)]
  }
  const observe = (elapsed: number, start: number) => {
    // A few consecutive, near-identical intervals can be a refresh-rate
    // change. Hold them briefly so a change doesn't look like repeated stalls.
    const candidate = elapsed <= 50 && elapsed > expected * 1.25
    if (slower.length && (!candidate ||
      Math.abs(elapsed - slower[0]) > Math.max(1, slower[0] * 0.1))) {
      let at = slowerStart
      for (const interval of slower) {
        report(interval, at)
        at += interval
      }
      slower.length = 0
    }
    if (candidate) {
      if (!slower.length) slowerStart = start
      slower.push(elapsed)
      if (slower.length === 4) {
        flush()
        recent.length = 0
        recent.push(...slower)
        expected = slower.reduce((sum, value) => sum + value, 0) / slower.length
        send(Math.max(...slower), expected, slower.length, slowerStart, expected)
        slower.length = 0
      }
      return
    }
    cadence(elapsed)
    if (Number.isFinite(expected)) report(elapsed, start)
    else if (elapsed > 50) send(elapsed, 1000 / 60, 1, start, elapsed)
  }
  const tick = (now: number) => {
    if (document.visibilityState !== "visible") reset()
    else {
      if (last !== undefined && now > last)
        observe(now - last, performance.timeOrigin + last)
      last = now
    }
    frame = requestAnimationFrame(tick)
  }
  document.addEventListener("visibilitychange", reset)
  frame = requestAnimationFrame(tick)
  return () => {
    document.removeEventListener("visibilitychange", reset)
    cancelAnimationFrame(frame)
  }
}
