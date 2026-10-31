import type { ByteSource } from "./weights.js"

export type PlannedRead = {
  /** What the bytes are, to catch a loader taking them out of order. */
  key: string
  source: ByteSource
  offset: number
  length: number
  /** Bytes already in memory (an inline ONNX tensor). */
  bytes?: Uint8Array
}

/**
 * The reads a loader makes, in the order it uses them. A few are kept in
 * flight ahead of the one in use (bounded in count and bytes), so network
 * latency overlaps the repacking without holding more than a few tensors.
 */
export class ReadPlan {
  private reads: Array<PlannedRead> = []
  private pending: Array<{
    key: string
    promise: Promise<Uint8Array>
    length: number
  }> = []
  private next = 0
  private started = false

  constructor(
    private ahead = 3,
    private maxBytesAhead = 24 << 20
  ) {}

  add(read: PlannedRead): void {
    if (this.started)
      throw new Error("reads must be planned before the first take")
    this.reads.push(read)
  }

  private fill(): void {
    let bytes = 0
    for (const p of this.pending) bytes += p.length
    while (this.next < this.reads.length && this.pending.length < this.ahead) {
      const read = this.reads[this.next]
      if (this.pending.length > 0 && bytes + read.length > this.maxBytesAhead)
        break
      this.next++
      const promise = read.bytes
        ? Promise.resolve(read.bytes)
        : read.source.read(read.offset, read.length)
      // Rejections surface when taken; don't report them as unhandled before that.
      promise.catch(() => undefined)
      this.pending.push({ key: read.key, promise, length: read.length })
      bytes += read.length
    }
  }

  /** The next read's bytes; `key` must be the one planned next. */
  async take(key: string): Promise<Uint8Array> {
    this.started = true
    this.fill()
    const p = this.pending.shift()
    if (!p) throw new Error(`no read planned for ${key}`)
    if (p.key !== key)
      throw new Error(`loader took ${key} but ${p.key} was planned next`)
    this.fill()
    return p.promise
  }
}
