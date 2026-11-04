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
type ReadLease = { length: number; remaining: number; borrowed: number }

export class ReadPlan {
  private owned = new WeakMap<ArrayBufferLike, ReadLease>()
  private released = new WeakSet<Uint8Array>()
  private reads: Array<PlannedRead> = []
  private pending: Array<{
    key: string
    promise: Promise<Uint8Array>
    length: number
    lease: ReadLease
  }> = []
  private next = 0
  private started = false
  private largestRead = 0

  constructor(
    private ahead = 3,
    private maxBytesAhead = 24 << 20,
    private retireConsumed = false
  ) {}

  add(read: PlannedRead): void {
    if (this.started)
      throw new Error("reads must be planned before the first take")
    this.reads.push(read)
    this.largestRead = Math.max(this.largestRead, read.length)
  }

  private fill(): void {
    // Count a shared read's whole backing store until its last pending
    // slice is taken; otherwise coalescing could silently exceed the cap.
    const leases = new Set(this.pending.map(p => p.lease))
    let bytes = 0
    for (const lease of leases) bytes += lease.length
    while (this.next < this.reads.length && this.pending.length < this.ahead) {
      const first = this.reads[this.next]
      if (this.pending.length > 0 && bytes + first.length > this.maxBytesAhead) break
      const batch = [first]
      let length = first.length
      // Never make a backing store larger than an original planned read.
      // Together with the lease cap this preserves the previous peak bound:
      // read-ahead bytes plus at most one original current tensor.
      const limit = typeof (ArrayBuffer.prototype as ArrayBuffer & { transfer?: unknown }).transfer === "function"
        ? Math.min(8 << 20, this.largestRead) : 0
      if (!first.bytes) {
        for (let i = this.next + 1; i < this.reads.length && this.pending.length + batch.length < this.ahead; i++) {
          const next = this.reads[i]
          if (next.bytes || next.source !== first.source || next.offset !== first.offset + length || length + next.length > limit || bytes + length + next.length > this.maxBytesAhead) break
          batch.push(next)
          length += next.length
        }
      }
      this.next += batch.length
      const lease = { length, remaining: batch.length, borrowed: 0 }
      const incoming = first.bytes ? Promise.resolve(first.bytes) : first.source.read(first.offset, length)
      const read = this.retireConsumed ? incoming.then(data => {
        // Only a complete, newly owned read can be retired. Inline tensors
        // and slices of an external backing store retain their ownership.
        if (!first.bytes && data.byteOffset === 0 && data.byteLength === length && data.buffer.byteLength === length)
          this.owned.set(data.buffer, lease)
        return data
      }) : incoming
      let offset = 0
      for (const item of batch) {
        const from = offset
        const promise = batch.length === 1 ? read : read.then(data => data.subarray(from, from + item.length))
        promise.catch(() => undefined)
        this.pending.push({ key: item.key, promise, length: item.length, lease })
        offset += item.length
      }
      bytes += length
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
    if (!this.retireConsumed) return p.promise
    const bytes = await p.promise
    p.lease.remaining--
    p.lease.borrowed++
    return bytes
  }

  /** The converter has finished all uses of this returned view. */
  release(bytes: Uint8Array): void {
    if (!this.retireConsumed || this.released.has(bytes)) return
    const buffer = bytes.buffer, lease = this.owned.get(buffer)
    if (!lease) return
    this.released.add(bytes)
    lease.borrowed--
    if (lease.remaining || lease.borrowed) return
    this.owned.delete(buffer)
    ;(buffer as ArrayBuffer & { transfer?: (length: number) => ArrayBuffer }).transfer?.(0)
  }
}
