/** A device-independent work quota adjusted from completed GPU fence latency. */
export class SubmissionBudget {
  reads = 0
  ahead = 2
  layers = 1
  private decodeAt = -Infinity
  private decodeSmooth = 0
  private activeWorkAt = -Infinity
  private submissions = new Float64Array(128)
  private submissionIndex = 0
  private pendingSince = Infinity
  private smoothFrames = 0
  private initial = -1
  private target = -1
  private lastFrame = -Infinity
  private healthyFrame = false
  private growAfter = 0
  private frameMs = 16.7
  presentationWindow = 0
  presentationMax = 24
  private presentationCap = 24
  private presentationBase = -1
  private presentationSmooth = 0
  private relaxedReadsAllowed = true
  private computeFrames = 0
  private recoveryFrames = 0
  private missedTime = 0
  private recentMisses = 0
  recovery = false
  private lastMissAt = -Infinity
  private fastMatrixAt = -Infinity

  constructor(private readonly decoding = false) {}

  reset(presentation = false) {
    if (presentation) {
      this.presentationWindow = this.presentationBase
      this.presentationMax = Math.max(
        this.presentationBase,
        this.presentationCap
      )
      this.presentationSmooth = 0
      this.relaxedReadsAllowed = true
    }
    this.computeFrames = this.recoveryFrames = this.missedTime = this.recentMisses = 0
    this.lastMissAt = this.fastMatrixAt = -Infinity
    this.reads = Math.max(0, this.initial)
    this.healthyFrame = false
    this.ahead = 2
    this.layers = this.recovery && this.decoding ? 2 : 1
    this.activeWorkAt = this.decodeAt = this.lastFrame = -Infinity
    this.submissionIndex = 0
    this.pendingSince = Infinity
    this.smoothFrames = this.decodeSmooth = 0
    this.growAfter = 0
  }

  configure(initialReads: number, targetMs: number, baseWindow = 4) {
    if (baseWindow !== this.presentationBase) {
      this.presentationBase = baseWindow
      this.presentationWindow = baseWindow
      this.presentationMax = Math.max(baseWindow, this.presentationCap)
    }
    if (initialReads === this.initial && targetMs === this.target) return
    this.initial = initialReads
    this.target = targetMs
    this.reads = Math.max(0, initialReads)
  }

  get quotaCollapsed() {
    return this.initial > 0 && this.reads <= this.initial / 4
  }

  enableRecovery(startingReads?: number) {
    this.recovery = true
    this.reads = startingReads === undefined
      ? Math.max(this.reads, Math.min(this.initial, 3 * 2 ** 25))
      : Math.min(this.initial, startingReads)
    this.ahead = Math.max(this.ahead, 1)
    if (this.decoding) this.layers = Math.max(this.layers, 2)
  }

  heartbeat(now = performance.now(), healthy = true, frameMs = this.frameMs) {
    this.setCadence(frameMs)
    this.lastFrame = now
    this.healthyFrame = healthy
  }

  decode(now = performance.now()) {
    this.decodeAt = now
  }

  work(now = performance.now()) {
    this.activeWorkAt = now
  }

  submitted(start: number, end: number) {
    const index = (this.submissionIndex++ * 2) % this.submissions.length
    this.submissions[index] = start
    this.submissions[index + 1] = end
  }

  waiting(start: number) {
    this.pendingSince = start
  }

  completed(end: number) {
    this.submitted(this.pendingSince, end)
    this.pendingSince = Infinity
  }

  private overlaps(start: number, end: number) {
    if (this.pendingSince < end) return true
    const limit = Math.min(this.submissionIndex * 2, this.submissions.length)
    for (let i = 0; i < limit; i += 2)
      if (this.submissions[i] < end && this.submissions[i + 1] > start) return true
    return false
  }

  frame(
    intervalMs: number,
    expectedMs: number,
    frames: number,
    now = performance.now(),
    meanMs?: number,
    startedAt?: number
  ) {
    if (
      this.initial <= 0 ||
      this.target <= 0 ||
      !Number.isFinite(intervalMs) ||
      !Number.isFinite(expectedMs) ||
      expectedMs < 4 ||
      intervalMs < 4
    )
      return
    this.lastFrame = now
    this.setCadence(expectedMs)
    const average = meanMs ?? intervalMs
    // A missed refresh at 120 Hz is still about 17 ms. During decode,
    // keep the head and layer submissions intact through such jitter.
    const decodeRecovery = this.recovery && this.decoding
    const severeMs = decodeRecovery ? Math.max(32, expectedMs * 1.8) : expectedMs * 1.8
    const meanLimit = expectedMs * (decodeRecovery ? 1.12 : 1.03)
    this.healthyFrame = intervalMs < severeMs && average <= meanLimit
    if (intervalMs > severeMs || average > meanLimit) {
      const span = average * frames
      const start = startedAt ?? now - span
      if (!this.overlaps(start, start + span)) return
      const previousWindow = this.presentationWindow
      const severe = intervalMs > severeMs
      let sustained = false
      if (severe) {
        // Charge only the time beyond a normal frame. Isolated misses
        // drain away between samples, independent of the refresh rate.
        if (this.recovery) {
          const recent = Math.max(0, this.missedTime - (now - this.lastMissAt) * 0.18)
          this.missedTime = recent + Math.min(intervalMs - expectedMs, 2 * expectedMs)
          sustained = recent > 0 && this.missedTime >= Math.max(20, 2 * expectedMs)
        } else {
          this.recentMisses = now - this.lastMissAt < 500 ? this.recentMisses + 1 : 1
          sustained = this.recentMisses >= 3
        }
        this.lastMissAt = now
      }
      if (!this.recovery || sustained || intervalMs > 100) this.recoveryFrames = 0
      const wasRelaxed = this.reads > this.initial
      if (wasRelaxed && (sustained || intervalMs > 100)) {
        this.reads = this.initial
        this.relaxedReadsAllowed = false
      }
      if (sustained || intervalMs > 100) this.relaxedReadsAllowed = false
      this.presentationMax = Math.max(
        sustained || intervalMs > 100 ? 2 : this.presentationBase,
        Math.min(this.presentationMax, this.presentationWindow - 1)
      )
      this.presentationWindow = Math.min(
        this.presentationBase,
        this.presentationMax
      )
      this.presentationSmooth = 0
      // First give presentation a chance to run. Reducing matrix bands
      // for an isolated dropped frame can cross a large tile-size cliff.
      if (
        !wasRelaxed &&
        (intervalMs > 100 ||
          (!decodeRecovery && previousWindow <= this.presentationBase && sustained && (!this.recovery || now >= this.growAfter)))
      ) {
        this.reads = Math.max(
          Math.min(this.initial, this.recovery && intervalMs <= 100 ? 3 * 2 ** 25 : 2 ** 25),
          Math.round(this.reads * (intervalMs > 100 ? 0.5 : 0.8))
        )
        this.growAfter = now + 1000
      }
      if (decodeRecovery && !severe) {
        // A slow mean with no long frame says nothing against the queue's
        // depth, so it still earns the queue back. Else a page that paints
        // most of a high-refresh display's frames (Firefox on a Pixel 10
        // Pro: about 100 of 120) never does: one burst of long frames left
        // writing at a wait a layer with nothing queued, 2.4 tokens/s
        // where it writes 10, for the rest of the session.
        this.earnBack(frames, now)
      } else {
        this.smoothFrames = 0
        if (severe && (!decodeRecovery || sustained || intervalMs > 100)) {
          this.ahead = Math.max(0, this.ahead - 1)
          this.layers = Math.max(
            1,
            Math.floor(this.layers / (intervalMs > 100 ? 4 : 2))
          )
        }
        this.decodeSmooth = 0
      }
    } else if (intervalMs < severeMs) {
      this.earnBack(frames, now)
      if (now - this.activeWorkAt > 250) return
      this.computeFrames += frames
      this.recoveryFrames += frames
      if (this.recoveryFrames >= 96) {
        this.presentationMax = Math.min(
          this.presentationCap,
          this.presentationMax + 4
        )
        // Frame evidence can restore the starting tile size even when
        // fences on very small submissions are dominated by scheduling.
        if (this.recovery && now >= this.growAfter && this.reads < this.initial)
          this.reads = Math.min(this.initial, Math.round(this.reads * 1.25))
        this.recoveryFrames = 0
      }
      this.presentationSmooth += frames
      if (this.presentationSmooth >= 24) {
        this.presentationWindow = Math.min(
          this.presentationMax,
          this.presentationWindow + 2
        )
        if (
          this.relaxedReadsAllowed &&
          this.presentationWindow >= Math.min(24, this.presentationCap) &&
          this.presentationMax >= Math.min(24, this.presentationCap) &&
          this.computeFrames >= 192 &&
          (now - this.fastMatrixAt < 500 || this.reads > this.initial)
        )
          this.reads = Math.min(2 ** 34, Math.round(this.reads * 1.25))
        this.presentationSmooth = 0
      }
    }
  }

  /** Frames without a long one: after 96, writing's layers per wait and the chunks queued ahead come back a step. */
  private earnBack(frames: number, now: number) {
    if (now - this.decodeAt < 250) {
      this.decodeSmooth += frames
      if (this.decodeSmooth >= 96) {
        this.layers = Math.max(this.layers, 2)
        this.decodeSmooth = 0
      }
    }
    this.smoothFrames += frames
    if (this.smoothFrames >= 96) {
      this.ahead = Math.min(2, this.ahead + 1)
      this.smoothFrames = 0
    }
  }

  /** Reuse observed overlap while preserving the conservative rollback quota. */
  primePresentation(window: number, reads: number, allowRelaxation = true) {
    this.presentationWindow = Math.max(
      2,
      Math.min(this.presentationCap, window)
    )
    this.relaxedReadsAllowed = allowRelaxation
    this.reads = Math.max(this.initial, Math.min(2 ** 34, reads))
  }

  get maxPresentationFrames() {
    return this.presentationCap
  }

  private setCadence(frameMs: number) {
    if (!Number.isFinite(frameMs) || frameMs < 4) return
    const cap = Math.max(this.presentationBase, Math.round(400 / frameMs))
    if (this.presentationMax === this.presentationCap)
      this.presentationMax = cap
    else this.presentationMax = Math.min(this.presentationMax, cap)
    this.presentationCap = cap
    this.frameMs = frameMs
    this.presentationWindow = Math.min(this.presentationWindow, cap)
  }

  get relaxationAllowed() {
    return this.relaxedReadsAllowed
  }

  get validatedPresentation() {
    return this.relaxedReadsAllowed &&
      this.presentationWindow >= Math.min(24, this.presentationCap) &&
      this.presentationMax >= this.presentationWindow &&
      this.computeFrames >= 192
      ? {
          reads: this.reads,
          window: this.presentationWindow,
          frameMs: Math.round(this.frameMs * 10) / 10
        }
      : null
  }

  layerSample(layers: number, elapsedMs: number, queuedChunks: number) {
    if (
      this.target <= 0 ||
      !this.healthyFrame ||
      !Number.isFinite(elapsedMs) ||
      elapsedMs <= 0 ||
      elapsedMs >= this.target * Math.max(1, queuedChunks) * 0.35
    )
      return
    const desired = Math.floor(
      (layers * this.target * Math.max(1, queuedChunks) * 0.5) / elapsedMs
    )
    this.layers = Math.min(
      32,
      Math.max(this.layers, Math.min(layers * 2, desired))
    )
  }

  sample(
    reads: number,
    elapsedMs: number,
    queuedChunks: number,
    now = performance.now()
  ) {
    if (now - this.lastFrame < 1000) {
      // Only substantial matrix work may grow the quota; tiny memory,
      // head and empty-queue fences cannot establish GPU throughput.
      const target = Math.min(this.target, this.frameMs / 2)
      if (
        this.initial > 0 &&
        target > 0 &&
        this.healthyFrame &&
        reads >= this.reads / 2 &&
        elapsedMs >= 0.1 &&
        elapsedMs < target * Math.max(1, queuedChunks)
      )
        this.fastMatrixAt = now
      if (
        this.initial > 0 &&
        target > 0 &&
        this.healthyFrame &&
        now >= this.growAfter &&
        reads >= this.reads / 2 &&
        elapsedMs >= 0.1 &&
        elapsedMs < target * Math.max(1, queuedChunks) * 0.35
      )
        this.reads = Math.min(
          this.relaxedReadsAllowed &&
            this.presentationWindow >= Math.min(24, this.presentationCap)
            ? 2 ** 34
            : this.initial,
          Math.round(this.reads * 1.25)
        )
      return
    }
    if (
      this.initial <= 0 ||
      this.target <= 0 ||
      reads <= 0 ||
      !Number.isFinite(elapsedMs) ||
      elapsedMs < 0.1 ||
      elapsedMs > 250
    )
      return
    // A fence's age includes earlier queued work. Bound the whole queue,
    // rather than treating that age as an isolated kernel execution time.
    const desired =
      (reads * this.target * Math.max(1, queuedChunks)) / elapsedMs
    const adjusted = this.reads * 0.8 + desired * 0.2
    this.reads = Math.round(
      Math.max(
        1 << 25,
        Math.min(
          this.initial,
          Math.max(this.reads * 0.75, Math.min(this.reads * 1.25, adjusted))
        )
      )
    )
  }
}

/**
 * How fast a complete pipeline ran, from 1 (within two frames) down to 0
 * (four frames or more): what its stored quota and presentation window
 * are scaled by. It falls steadily between the two, in the pipeline's
 * time on a log scale, so a few milliseconds either side of two frames
 * store nearly the same, where a step there stored eight times the quota
 * for 26 ms and an eighth for 35 (the same GPU, quiet and beside a game).
 */
export function pipelineSpeed(elapsedMs: number, frameMs: number) {
  const frames = elapsedMs / frameMs
  if (!(frames > 2)) return 1
  if (frames >= 4) return 0
  return 1 - Math.log2(frames / 2)
}

/** Scale fast devices from a complete, observed matrix workload, not a tiny draw. */
export function calibratedReadQuota(
  base: number,
  work: number,
  elapsedMs: number,
  frameMs: number
) {
  // A complete pipeline that finishes within two frames can queue two frames
  // of work. One that takes four or more keeps the quarter-frame starting
  // budget, and between them the share falls from one to the other.
  const frameShare = 0.25 * 8 ** pipelineSpeed(elapsedMs, frameMs)
  const estimate = (work / elapsedMs) * frameMs * frameShare
  return work > 0 && elapsedMs > 0 && frameMs >= 4 && Number.isFinite(estimate)
    ? Math.max(base, Math.min(2 ** 34, Math.round(estimate)))
    : base
}

/**
 * The presentation window stored with that quota: all of `cap` frames
 * where the pipeline finished within two, `base` where it took four or
 * more, and between them as the quota's share falls.
 */
export function calibratedWindow(
  base: number,
  cap: number,
  elapsedMs: number,
  frameMs: number
) {
  if (!(elapsedMs > 0) || !Number.isFinite(elapsedMs) || !(frameMs >= 4) || cap <= base) return base
  return Math.round(base + (cap - base) * pipelineSpeed(elapsedMs, frameMs))
}
