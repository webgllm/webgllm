/**
 * What both GPU backends share: frame pacing. Work goes to the GPU in
 * chunks ended by fences, and the next chunk waits until few enough are
 * queued (see pace), sized by a feedback controller from frame timings
 * (SubmissionBudget). The backend supplies passes, fences and reads.
 */

import { SubmissionBudget } from "./submission-budget.js"
import type { Pass } from "./types.js"

export abstract class PacedGpu<F> {
  abstract get isLost(): boolean
  abstract run(pass: Pass): void
  /** Ends the chunk of work issued so far and sends it to the GPU, behind a fence (null if none could be made). */
  protected abstract fence(): F | null
  /** Whether the GPU is past a fence: one look, without waiting. */
  protected abstract fenceReady(fence: F, waiting?: boolean): boolean
  protected abstract deleteFence(fence: F): void
  /** Resolves once the GPU passes a fence. */
  protected abstract wait(fence: F | null, budget?: SubmissionBudget, at?: number): Promise<void>
  /** Blocks until the GPU is done, where timers are throttled (see pace). */
  protected abstract readBack(): void
  /** The fewest rows of a `width`-texel pass runAsync issues at once (a band's floor). */
  protected minBandRows(width: number) {
    void width
    return 1
  }

  /** Texel reads per async submission; zero keeps ordinary submission. */
  readsPerSlice = 0
  submittedWork = 0
  expectedFrameMs = 16.7
  /** Drain the compute queue periodically so rendering can obtain the GPU. */
  presentationFrames = 2
  protected lastPresentationDrain = -Infinity

  private workStartedAt: number | undefined

  protected markGpuWork() {
    this.workStartedAt ??= performance.now()
  }

  protected submitGpuWork() {
    if (this.workStartedAt === undefined) return
    this.submissionBudget.submitted(this.workStartedAt, performance.now())
    this.workStartedAt = undefined
  }

  private sliceReads = 0
  private sliceMatrixReads = 0
  private sliceCalibrates = true
  protected budgets = {
    load: new SubmissionBudget(),
    prompt: new SubmissionBudget(),
    decode: new SubmissionBudget(true)
  }
  protected submissionBudget = this.budgets.load
  protected phase: "load" | "prompt" | "decode" = "load"
  private phaseAt = performance.timeOrigin + performance.now()
  private lastObservedFrame = -Infinity
  private lastFrameHealthy = false
  private canRecoverPacing = false
  private sampledPromptPacing = false
  protected promptFrameBudgetMs = 0
  /**
   * The frame rate the pacing defends where an idle GPU's fence is slow to
   * report (see measurePromptPacing: Firefox on Android, about 7 ms).
   * There every slice of work costs frames, far more than in Chrome on the
   * same GPU: Firefox on a Pixel 10 Pro read 57 to 67 tokens/s holding 80
   * frames a second and 68 to 72 at 36, where Chrome reads 100 at its 60;
   * and its odd 50 ms frame while writing, taken for a failing queue,
   * halved its speed. So it defends 40, which keeps the page above 30,
   * not the display's rate (75 for prompts, before).
   */
  slowFencePromptFps = 40

  /** The same feedback controller is used on every GPU. Zero ms fixes the quota. */
  configurePacing(reads: number, targetMs: number) {
    for (const budget of Object.values(this.budgets))
      budget.configure(reads, targetMs, this.presentationFrames)
    this.readsPerSlice = this.submissionBudget.reads
  }

  get activePhase() { return this.phase }

  get groupsDecode() { return this.budgets.decode.recovery }

  allowPacingRecovery() {
    this.canRecoverPacing = true
  }

  /** A slow empty fence makes frequent pacing boundaries expensive. */
  calibratePromptPacing() {
    if (!this.canRecoverPacing || this.sampledPromptPacing ||
      this.expectedFrameMs > 12 || performance.now() - this.lastObservedFrame > 500)
      return
    this.sampledPromptPacing = true
    return this.measurePromptPacing()
  }

  private async measurePromptPacing() {
    let slow = 0
    for (let i = 0; i < 3; i++) {
      const start = performance.now()
      await this.finish()
      if (performance.now() - start >= 3) slow++
    }
    if (slow < 2) return
    // Budget for presentation during prefill on high-refresh displays.
    // A real long stall still triggers the controller's immediate backoff.
    this.promptFrameBudgetMs = 1000 / this.slowFencePromptFps
    this.budgets.prompt.enableRecovery(160 * 2 ** 20)
    this.budgets.decode.enableRecovery()
  }

  setPhase(phase: "load" | "prompt" | "decode") {
    if (this.phase === phase) return
    this.submitGpuWork()
    this.phase = phase
    this.phaseAt = performance.timeOrigin + performance.now()
    this.submissionBudget = this.budgets[phase]
    this.readsPerSlice = this.submissionBudget.reads
    this.sliceReads = this.sliceMatrixReads = 0
    this.sliceCalibrates = true
  }

  resetPacing(presentation = false) {
    for (const budget of Object.values(this.budgets)) {
      budget.reset(presentation)
      budget.heartbeat(this.lastObservedFrame, this.lastFrameHealthy)
    }
    this.submissionBudget = this.budgets[this.phase]
    this.readsPerSlice = this.submissionBudget.reads
    this.sliceReads = this.sliceMatrixReads = 0
    this.sliceCalibrates = true
  }

  get presentationAdvice() {
    return { window: this.budgets.prompt.presentationWindow, max: this.budgets.prompt.presentationMax, relaxation: this.budgets.prompt.relaxationAllowed, cap: this.budgets.prompt.maxPresentationFrames }
  }

  get validatedPresentation() { return this.budgets.prompt.validatedPresentation }

  primePresentation(window: number, reads: number, allowRelaxation = true) {
    this.budgets.prompt.primePresentation(window, reads, allowRelaxation)
    this.readsPerSlice = this.submissionBudget.reads
  }

  get decodeLayers() {
    return this.submissionBudget.layers
  }
  decodeWork() {
    this.submissionBudget.decode()
  }

  /** Main-thread frame feedback takes precedence over fence-age estimation. */
  observeFrames(
    intervalMs: number,
    expectedMs: number,
    frames: number,
    startedAt?: number,
    meanMs?: number
  ) {
    if (
      !Number.isFinite(intervalMs) ||
      !Number.isFinite(expectedMs) ||
      intervalMs < 4 ||
      expectedMs < 4 ||
      frames < 1
    )
      return
    const now = performance.now()
    this.lastObservedFrame = now
    // (Where fences are slow to report, reading and writing are both judged
    // against the frame rate the pacing defends there: see slowFencePromptFps.)
    if (this.phase !== "load")
      expectedMs = Math.max(expectedMs, this.promptFrameBudgetMs || 0)
    this.expectedFrameMs = expectedMs
    this.lastFrameHealthy =
      intervalMs < expectedMs * 1.8 &&
      (meanMs == null || meanMs <= expectedMs * 1.03)
    // Cadence remains fresh across phase changes, but only active work
    // adjusts its quota. An idle phase must not trigger fence fallback.
    for (const budget of Object.values(this.budgets))
      budget.heartbeat(now, this.lastFrameHealthy, expectedMs)
    // Delayed messages describe the original frame window, not the moment
    // the worker can finally receive them.
    if (startedAt == null || startedAt >= this.phaseAt)
      this.submissionBudget.frame(
        intervalMs, expectedMs, frames, now, meanMs,
        startedAt == null ? undefined : startedAt - performance.timeOrigin
      )
    // A high-refresh session can lose occasional callbacks yet have
    // ample presentation headroom. Recover from a proven quota collapse
    // during real inference without retuning healthy queues.
    if (this.canRecoverPacing && this.phase !== "load" &&
      !this.submissionBudget.recovery && this.submissionBudget.quotaCollapsed &&
      expectedMs <= 12 && intervalMs <= 100) {
      this.budgets.prompt.enableRecovery()
      this.budgets.decode.enableRecovery()
    }
    this.readsPerSlice = this.submissionBudget.reads
  }

  /** A compute pass in bounded bands, letting the compositor run between them. */
  async runAsync(pass: Pass, maxRows = Infinity) {
    const output = Array.isArray(pass.output) ? pass.output[0] : pass.output
    const rect = pass.rect ?? {
      x: 0,
      y: 0,
      width: output.width,
      height: output.height
    }
    this.submittedWork += rect.width * rect.height * Math.max(1, pass.work ?? 1)
    if (this.readsPerSlice <= 0) return this.run(pass)
    if (pass.name === "matmul" && rect.height > 8) this.submissionBudget?.work()
    const perRow = rect.width * Math.max(1, pass.work ?? 1)
    this.sliceCalibrates &&= pass.name !== "head" && maxRows === Infinity
    const endSlice = async () => {
      const work =
        this.sliceCalibrates &&
        this.sliceMatrixReads >= this.sliceReads * 0.75
          ? this.sliceReads
          : 0
      this.sliceReads = this.sliceMatrixReads = 0
      this.sliceCalibrates = true
      await this.pace(1, true, work)
    }
    // (Prompts only: a token's one-row passes keep the row path's pacing.)
    if (pass.bandColumns && maxRows === Infinity && rect.height >= 8) {
      const q = pass.bandColumns
      const perCol = rect.height * Math.max(1, pass.work ?? 1)
      // (At least a band's floor of texels, in whole multiples of q.)
      // (A column quantum at least, not a band's floor of texels: bands of
      // 240 columns left an iPhone 16 Pro Max's slices half-filled at each
      // seam, fences more often than rows, and read Qwen3.5 2B's prompts at
      // 323-358 tokens/s where 347-363.)
      const least = q
      for (let x = rect.x; x < rect.x + rect.width;) {
        const fits = Math.floor((this.readsPerSlice - this.sliceReads) / perCol / q) * q
        // (As rows: a small matmul starts a new chunk rather than splitting.)
        const small = x === rect.x && rect.width * perCol <= this.readsPerSlice / 8 && fits < rect.width
        if ((fits < least || small) && this.sliceReads > 0) {
          await endSlice()
          continue
        }
        const width = Math.min(Math.max(fits, least), rect.x + rect.width - x)
        this.run({ ...pass, rect: { ...rect, x, width } })
        this.sliceReads += width * perCol
        if (pass.name === "matmul") this.sliceMatrixReads += width * perCol
        x += width
        if (this.sliceReads + least * perCol > this.readsPerSlice) await endSlice()
      }
      return
    }
    for (let y = rect.y; y < rect.y + rect.height;) {
      const rows = Math.max(
        this.minBandRows(rect.width),
        Math.floor((this.readsPerSlice - this.sliceReads) / perRow)
      )
      // End paced matrix draws on raster-tile boundaries. Flushing
      // A small projection also starts a new chunk rather than splitting.
      // Arbitrary bands make the GPU store/reload the same boundary
      // tiles repeatedly, unlike the original uninterrupted render pass.
      const quantum = pass.name === "matmul" && rect.height >= 32 ? 32 : 1
      if (
        quantum > 1 &&
        (rows < quantum ||
          (y === rect.y && rect.height * perRow <= this.readsPerSlice / 8 && rows < rect.height)) &&
        this.sliceReads > 0 &&
        this.readsPerSlice >= perRow * quantum
      ) {
        const work =
          this.sliceCalibrates &&
          this.sliceMatrixReads >= this.sliceReads * 0.75
            ? this.sliceReads
            : 0
        this.sliceReads = this.sliceMatrixReads = 0
        this.sliceCalibrates = true
        await this.pace(1, true, work)
        continue
      }
      const bandRows =
        quantum > 1 && rows >= quantum
          ? Math.floor((y + rows) / quantum) * quantum - y
          : rows
      const height = Math.min(bandRows, Math.max(maxRows, this.minBandRows(rect.width)), rect.y + rect.height - y)
      this.run({ ...pass, rect: { ...rect, y, height } })
      this.sliceReads += height * perRow
      if (pass.name === "matmul" && rect.height > 8)
        this.sliceMatrixReads += height * perRow
      y += height
      if (
        this.sliceReads + perRow > this.readsPerSlice ||
        maxRows < Infinity ||
        (quantum > 1 && y < rect.y + rect.height)
      ) {
        const work =
          this.sliceCalibrates &&
          this.sliceMatrixReads >= this.sliceReads * 0.75
            ? this.sliceReads
            : 0
        this.sliceReads = this.sliceMatrixReads = 0
        this.sliceCalibrates = true
        await this.pace(1, true, work)
      }
    }
  }


  /** Fences after paced chunks of passes, oldest first (see pace). */
  protected paced: Array<{
    sync: F
    at: number
    reads: number
    chunks: number
    budget: SubmissionBudget
    layers: number
  }> = []

  /**
   * Ends a chunk of passes, and waits (without blocking) until at most
   * `ahead` chunks are queued on the GPU. Without it, a long prompt queues
   * seconds of work at once, and the browser's GPU process (which draws
   * every page, and the browser itself) stalls until the queue drains.
   */
  async pace(ahead: number, messages = false, reads = 0, layers = 0) {
    if (messages && reads === 0)
      ahead = Math.min(ahead, this.submissionBudget.ahead)
    if (
      messages &&
      (reads > 0 || this.phase === "load") &&
      this.presentationFrames > 0 &&
      performance.now() - this.lastPresentationDrain >=
        this.expectedFrameMs *
          (this.submissionBudget.presentationWindow || this.presentationFrames)
    ) {
      ahead = 0
      this.lastPresentationDrain = performance.now()
    }
    const sync = this.fence()
    this.submitGpuWork()
    if (sync)
      this.paced.push({
        sync,
        at: performance.now(),
        reads,
        chunks: ahead + 1,
        budget: this.submissionBudget,
        layers
      })
    while (this.paced.length > ahead && !this.isLost) {
      const oldest = this.paced[0]
      if (this.fenceReady(oldest.sync)) {
        this.deleteFence(oldest.sync)
        this.paced.shift()
        const elapsed = performance.now() - oldest.at
        oldest.budget.sample(oldest.reads, elapsed, oldest.chunks)
        if (oldest.layers > 0)
          oldest.budget.layerSample(oldest.layers, elapsed, oldest.chunks)
        this.readsPerSlice = this.submissionBudget.reads
        continue
      }
      if (messages) {
        await this.wait(oldest.sync, oldest.budget, oldest.at)
        continue
      }
      // A page the browser counts as hidden has its timers throttled (a
      // 1 ms wait can take a second, or a minute): then wait by reading a
      // pixel of the last pass's output instead, which blocks until the
      // GPU has done the queue. The queue stays bounded either way.
      if (performance.now() < this.throttledUntil) {
        this.readBack()
        for (const f of this.paced) this.deleteFence(f.sync)
        this.paced = []
        break
      }
      const t = performance.now()
      await new Promise((resolve) => setTimeout(resolve, 1))
      if (performance.now() - t > 50)
        this.throttledUntil = performance.now() + 2000
    }
  }

  /** Until when pace waits by reading back, not by timers (see pace). */
  protected throttledUntil = 0

  private yielder: MessageChannel | null = null
  private yielded: Array<() => void> = []

  /**
   * Gives the event loop a turn: a message to itself, which isn't clamped
   * to 4 ms like nested timeouts are.
   */
  protected yieldOnce() {
    if (!this.yielder) {
      this.yielder = new MessageChannel()
      this.yielder.port1.onmessage = () => this.yielded.shift()?.()
    }
    return new Promise<void>((resolve) => {
      this.yielded.push(resolve)
      this.yielder?.port2.postMessage(null)
    })
  }

  /** Bound active polling as well as GPU work, to avoid heating a CPU core. */
  pollsBeforeSleep = 32
  /**
   * Reading a prompt, a wait polls on messages this long, then sleeps on
   * 1 ms timers until the GPU is there (-1: polls as writing does, with a
   * sleep every pollsBeforeSleep polls). A warm iPhone 16 Pro Max read
   * prompts 8 to 10% faster sleeping, and missed no more frames (fewer in
   * one of two runs); cool, the same speed. The polling kept a CPU core
   * busy, heat the throttled GPU paid for.
   * Writing keeps polling: its chunks are a few ms, and timer sleeps cost
   * it 4 ms a token there.
   */
  promptSpinMs = 0.25
  protected polls = 0

  /**
   * Whether a prompt's waits poll their fence with a timeout where the
   * browser allows one (MAX_CLIENT_WAIT_TIMEOUT_WEBGL above zero: Firefox;
   * Chrome and Safari report 0). Firefox answers a zero-timeout poll from
   * a status it refreshes about every 7 ms, so no fence reports sooner
   * than that; with any timeout the poll asks the GPU's side, and returns
   * in a millisecond or two. Firefox on a Pixel 10 Pro read prompts 20 to
   * 60% faster for it, with fewer missed frames.
   */
  timedPolls = true

  /** Waits for every pass issued so far to finish. */
  async finish() {
    const sync = this.fence()
    this.submitGpuWork()
    try {
      await this.wait(sync)
    } finally {
      if (sync) this.deleteFence(sync)
    }
  }
}
