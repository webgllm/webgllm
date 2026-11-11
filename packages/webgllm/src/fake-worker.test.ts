/**
 * A scripted stand-in for handler.ts, for the page-side tests: it answers
 * the protocol the way the worker does (claims before "ready", tokens then
 * "done"), and lets a test hold generations open.
 */
import type { ModelStorage } from "./model-files.js"
import type { RunTimings, WorkerRequest, WorkerResponse } from "./protocol.js"

export const STATS: RunTimings = {
  inputTokens: 10,
  cachedTokens: 0,
  outputTokens: 3,
  firstTokenMs: 5,
  promptMs: 4,
  decodeMs: 20
}

export const tick = () => new Promise<void>((resolve) => setTimeout(resolve, 0))
export async function settle(times = 5) {
  for (let i = 0; i < times; i++) await tick()
}

type Generate = Extract<WorkerRequest, { type: "generate" }>

export type FakeOptions = {
  /** Bytes each model claims when it loads. */
  bytes?: (model: string) => number
  /** Fails the load with this error. */
  loadError?: (model: string) => { message: string; code?: Extract<WorkerResponse, { type: "load-error" }>["code"] } | undefined
  /** What a generation writes; held open (finish it with `finish`) when undefined. */
  reply?: (request: Generate) => { tokens: Array<string>; done?: Partial<Extract<WorkerResponse, { type: "done" }>> } | undefined
  /** A load has its files in storage (then the fake says "stored"). */
  stored?: (model: string) => void
  /** Holds a load after its files are stored until this resolves. */
  hold?: (model: string) => Promise<void>
  /** Holds a download until this resolves. */
  holdDownload?: (model: string) => Promise<void>
  /** Holds an "unload" (the model still in memory) until this resolves. */
  holdUnload?: (model: string) => Promise<void>
}

export class FakeWorker {
  private readonly listeners = { message: new Set<(event: never) => void>(), error: new Set<(event: never) => void>() }
  /** Models it was asked to free ("unload": a worker the caller owns). */
  readonly unloaded: Array<string> = []
  terminated = false
  readonly sent: Array<WorkerRequest> = []
  model: string | undefined
  private claimed = new Map<number, () => void>()
  private claims = 0
  /** Downloads stopped ("stop" with their id). */
  readonly stoppedDownloads: Array<string> = []
  readonly open = new Map<string, Generate>()
  private idlers: Array<() => void> = []

  constructor(private readonly options: FakeOptions) {}

  postMessage(message: WorkerRequest) {
    if (this.terminated) return
    this.sent.push(message)
    void this.handle(message)
  }

  terminate() {
    this.terminated = true
  }

  addEventListener(type: "message" | "error", listener: (event: never) => void) {
    this.listeners[type].add(listener)
  }

  removeEventListener(type: "message" | "error", listener: (event: never) => void) {
    this.listeners[type].delete(listener)
  }

  /** How many listeners the page has on it. */
  get listening(): number {
    return this.listeners.message.size + this.listeners.error.size
  }

  emit(data: WorkerResponse) {
    if (!this.terminated) for (const listener of [...this.listeners.message]) listener({ data } as never)
  }

  /** The worker fails, as an uncaught error in it does. */
  fail(message: string) {
    for (const listener of [...this.listeners.error]) listener({ message } as never)
  }

  /** Claims memory again (a cache growing during a reply); resolves once "claimed". */
  claim(bytes: number) {
    return this.claimFor(this.model!, bytes)
  }

  private claimFor(model: string, bytes: number) {
    return new Promise<void>((resolve) => {
      const id = ++this.claims
      this.claimed.set(id, resolve)
      this.emit({ type: "claim", id, model, bytes })
    })
  }

  /** Ends a held generation. */
  finish(id: string, tokens: Array<string> = [], done: Partial<Extract<WorkerResponse, { type: "done" }>> = {}) {
    this.open.delete(id)
    for (const text of tokens) this.emit({ type: "token", id, text })
    this.emit({ type: "done", id, stats: STATS, finishReason: "stop", ...done })
    if (this.open.size === 0) for (const resolve of this.idlers.splice(0)) resolve()
  }

  generations(): Array<Generate> {
    return this.sent.filter((m): m is Generate => m.type === "generate")
  }

  private async handle(message: WorkerRequest) {
    await Promise.resolve()
    if (message.type === "claimed") {
      this.claimed.get(message.id)?.()
      this.claimed.delete(message.id)
    }
    else if (message.type === "load") {
      // Like handler.ts: after the generations running.
      if (this.open.size) await new Promise<void>((resolve) => this.idlers.push(resolve))
      this.model = message.model
      this.emit({ type: "progress", model: message.model, progress: 0.5, cached: true })
      if (this.options.stored) {
        this.options.stored(message.model)
        this.emit({ type: "stored", model: message.model })
      }
      await this.options.hold?.(message.model)
      const error = this.options.loadError?.(message.model)
      if (error) return this.emit({ type: "load-error", model: message.model, loadId: message.loadId, ...error })
      await this.claimFor(message.model, this.options.bytes?.(message.model) ?? 10)
      this.emit({ type: "ready", model: message.model, loadId: message.loadId })
    } else if (message.type === "download") {
      this.model = message.model
      await this.options.holdDownload?.(message.model)
      if (this.stoppedDownloads.includes(message.id))
        return this.emit({ type: "download-error", model: message.model, id: message.id, message: "The download was stopped." })
      this.emit({ type: "progress", model: message.model, progress: 0.45, cached: false, loadedBytes: 50, totalBytes: 100 })
      this.emit({ type: "downloaded", model: message.model, id: message.id })
    } else if (message.type === "generate") {
      const reply = this.options.reply?.(message)
      if (!reply) {
        this.open.set(message.id, message)
        return
      }
      this.finish(message.id, reply.tokens, reply.done)
    } else if (message.type === "forward") {
      // The token after the last one read.
      this.emit({ type: "forwarded", id: message.id, token: message.inputIds.at(-1)! + 1 })
    } else if (message.type === "reset") {
      this.emit({ type: "reset-done", id: message.id })
    } else if (message.type === "stop") {
      if (this.open.has(message.id)) this.finish(message.id)
      else if (message.id.startsWith("dl_")) this.stoppedDownloads.push(message.id)
    } else if (message.type === "unload") {
      await this.options.holdUnload?.(message.model)
      this.unloaded.push(message.model)
      if (this.model === message.model) this.model = undefined
      this.emit({ type: "unloaded", id: message.id })
    }
  }
}

/** A worker factory that keeps every worker it made. */
export function fakeWorkers(options: FakeOptions = {}) {
  const workers: Array<FakeWorker> = []
  return {
    workers,
    create: () => {
      const worker = new FakeWorker(options)
      workers.push(worker)
      return worker as unknown as Worker
    },
    /** The live worker that loaded `model`. */
    of: (model: string) => workers.find((w) => w.model === model && !w.terminated)
  }
}

export function fakeStorage(downloaded = new Set<string>()): ModelStorage & { removed: Array<[string, Set<string>]> } {
  const removed: Array<[string, Set<string>]> = []
  return {
    removed,
    downloaded: async (model: string) => downloaded.has(model),
    size: async () => 1234,
    contextLength: async () => 32768,
    manifest: async () => undefined,
    manifests: async () => [],
    remove: async (model, keep) => {
      removed.push([model, keep])
      downloaded.delete(model)
    }
  }
}
