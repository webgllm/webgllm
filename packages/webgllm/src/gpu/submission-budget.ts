/** A device-independent work quota adjusted from completed GPU fence latency. */
export class SubmissionBudget {
  reads = 0
  ahead = 2
  layers = 1
  private decodeAt = -Infinity
  private decodeSmooth = 0
  private activeWorkAt = -Infinity
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
  private recentMisses = 0
  private lastMissAt = -Infinity
  private fastMatrixAt = -Infinity

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
    this.computeFrames = this.recoveryFrames = this.recentMisses = 0
    this.lastMissAt = this.fastMatrixAt = -Infinity
    this.reads = Math.max(0, this.initial)
    this.healthyFrame = false
    this.ahead = 2
    this.layers = 1
    this.activeWorkAt = this.decodeAt = this.lastFrame = -Infinity
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

  frame(
    intervalMs: number,
    expectedMs: number,
    frames: number,
    now = performance.now(),
    meanMs?: number
  ) {
    if (
      this.initial <= 0 ||
      this.target <= 0 ||
      !Number.isFinite(intervalMs) ||
      !Number.isFinite(expectedMs) ||
      expectedMs < 4 ||
      intervalMs < 4 ||
      intervalMs >= 1000
    )
      return
    this.lastFrame = now
    this.setCadence(expectedMs)
    const average = meanMs ?? intervalMs
    this.healthyFrame =
      intervalMs < expectedMs * 1.8 && average <= expectedMs * 1.03
    if (intervalMs > expectedMs * 1.8 || average > expectedMs * 1.03) {
      const previousWindow = this.presentationWindow
      const severe = intervalMs > expectedMs * 1.8
      if (severe) {
        this.recentMisses =
          now - this.lastMissAt < 500 ? this.recentMisses + 1 : 1
        this.lastMissAt = now
      }
      const sustained = severe && this.recentMisses >= 3
      this.recoveryFrames = 0
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
          (previousWindow <= this.presentationBase && sustained))
      ) {
        this.reads = Math.max(
          Math.min(this.initial, 1 << 25),
          Math.round(this.reads * (intervalMs > 100 ? 0.5 : 0.8))
        )
        this.growAfter = now + 1000
      }
      this.smoothFrames = 0
      if (severe) this.ahead = Math.max(0, this.ahead - 1)
      if (severe)
        this.layers = Math.max(
          1,
          Math.floor(this.layers / (intervalMs > 100 ? 4 : 2))
        )
      this.decodeSmooth = 0
    } else if (intervalMs < expectedMs * 1.8) {
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
      if (now - this.activeWorkAt > 250) return
      this.computeFrames += frames
      this.recoveryFrames += frames
      if (this.recoveryFrames >= 96) {
        this.presentationMax = Math.min(
          this.presentationCap,
          this.presentationMax + 4
        )
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

/** Scale fast devices from a complete, observed matrix workload, not a tiny draw. */
export function calibratedReadQuota(
  base: number,
  work: number,
  elapsedMs: number,
  frameMs: number
) {
  const estimate = ((work / elapsedMs) * frameMs) / 4
  return work > 0 && elapsedMs > 0 && frameMs >= 4 && Number.isFinite(estimate)
    ? Math.max(base, Math.min(2 ** 34, Math.round(estimate)))
    : base
}
