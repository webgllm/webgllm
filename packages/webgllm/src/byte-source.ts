import type { ByteSource } from "./weights.js"

export type HttpSourceOptions = {
  /** Extra request headers (an Authorization header, say). */
  headers?: Record<string, string>
  /** Cancels the size probe and every later read. */
  signal?: AbortSignal
  /** Attempts per read before giving up on network errors and 5xx (default 3). */
  retries?: number
}

/**
 * A file on a server that takes range requests, such as a Hugging Face
 * resolve URL (redirects are followed). Works in workers. The size comes
 * from a HEAD request, or from a one-byte range request when HEAD gives no
 * length.
 */
export async function httpSource(
  url: string,
  options: HttpSourceOptions = {}
): Promise<ByteSource> {
  const { headers = {}, signal, retries = 3 } = options

  const request = async (init: RequestInit): Promise<Response> => {
    for (let attempt = 1; ; attempt++) {
      try {
        const response = await fetch(url, { ...init, signal })
        if (response.status < 500 || attempt >= retries) return response
        await response.body?.cancel()
      } catch (error) {
        if (signal?.aborted || attempt >= retries) throw error
      }
      await new Promise((resolve) => setTimeout(resolve, 250 * 2 ** attempt))
    }
  }

  let size = Number.NaN
  const head = await request({ method: "HEAD", headers })
  if (head.ok && !head.headers.get("content-encoding")) {
    size = Number(head.headers.get("content-length") ?? Number.NaN)
  }
  if (!(size >= 0)) {
    const probe = await request({ headers: { ...headers, Range: "bytes=0-0" } })
    const range = probe.headers.get("content-range")
    await probe.body?.cancel()
    const total = range?.match(/\/(\d+)\s*$/)
    if (probe.status !== 206 || !total) {
      throw new Error(
        `${url}: the server does not report a size or take range requests (status ${probe.status})`
      )
    }
    size = Number(total[1])
  }

  return {
    size,
    read: async (offset, length) => {
      if (length <= 0) return new Uint8Array(0)
      if (offset < 0 || offset + length > size) {
        throw new RangeError(
          `read of ${length} bytes at ${offset} is outside ${url} (${size} bytes)`
        )
      }
      const response = await request({
        headers: { ...headers, Range: `bytes=${offset}-${offset + length - 1}` }
      })
      // A 200 is the whole file: only acceptable when that is what we asked for.
      if (
        response.status !== 206 &&
        !(response.status === 200 && offset === 0 && length === size)
      ) {
        await response.body?.cancel()
        throw new Error(
          `${url}: range request failed (status ${response.status})`
        )
      }
      const bytes = new Uint8Array(await response.arrayBuffer())
      if (bytes.length !== length) {
        throw new Error(
          `${url}: asked for ${length} bytes at ${offset}, got ${bytes.length}`
        )
      }
      return bytes
    }
  }
}

/** A Blob or File, such as an OPFS file from `getFile()`. */
export function blobSource(blob: Blob): ByteSource {
  return {
    size: blob.size,
    read: async (offset, length) => {
      if (offset < 0 || offset + length > blob.size) {
        throw new RangeError(
          `read of ${length} bytes at ${offset} is outside the blob (${blob.size} bytes)`
        )
      }
      return new Uint8Array(
        await blob.slice(offset, offset + length).arrayBuffer()
      )
    }
  }
}

/** A stored file's handle and folder (storage.ts), for reading it through a synchronous access handle. */
const fileHandles = new WeakMap<Blob, { handle: FileSystemFileHandle; folder: ReadonlyArray<string> }>()

/** `file`, remembered as `handle`'s, in the folder at `folder` from the origin's storage root (see storedSources). */
export function withHandle(file: File, handle: FileSystemFileHandle, folder: ReadonlyArray<string>): File {
  fileHandles.set(file, { handle, folder })
  return file
}

type SyncAccessHandle = {
  read: (into: Uint8Array, options: { at: number }) => number
  getSize: () => number
  close: () => void
}

type StoredSources = { sources: Array<ByteSource>; close: () => Promise<void>; reader?: boolean }

/**
 * Sources for a load's files. A stored file in a worker is read through a
 * synchronous access handle: faster than `Blob.slice().arrayBuffer()` (on an
 * iPhone 16 Pro Max, 1,090 MB of Qwen3.5 weights in ~100 ms from the page
 * cache, ~650 ms from flash, against 300–640 ms and more as Blobs). With
 * `reader`, the handles are opened in a worker of their own, so the reads
 * run beside the caller's repacking and uploads instead of blocking them
 * (where a worker can't start one, or it fails, the handles are opened
 * here). Any other file, or one whose handle can't be opened (another
 * reader holds it), is read as a Blob. `close()` releases the handles: call
 * it once the load has read everything.
 */
export async function storedSources(blobs: ReadonlyArray<Blob>, options: { reader?: boolean } = {}): Promise<StoredSources> {
  const inWorker = typeof (globalThis as { WorkerGlobalScope?: unknown }).WorkerGlobalScope !== "undefined"
  if (inWorker && options.reader && typeof Worker !== "undefined") {
    const read = await readerSources(blobs).catch(() => null)
    if (read) return read
  }
  const opened: Array<SyncAccessHandle> = []
  const pool = new BufferPool()
  const sources = await Promise.all(
    blobs.map(async (blob) => {
      const handle = fileHandles.get(blob)?.handle as (FileSystemFileHandle & { createSyncAccessHandle?: () => Promise<SyncAccessHandle> }) | undefined
      if (inWorker && handle?.createSyncAccessHandle) {
        try {
          const access = await handle.createSyncAccessHandle()
          opened.push(access)
          return syncSource(access, pool)
        } catch {
          // Held by another reader or writer: the Blob.
        }
      }
      return blobSource(blob)
    })
  )
  return {
    sources,
    close: async () => {
      pool.clear()
      for (const access of opened.splice(0)) {
        try {
          access.close()
        } catch {
          // Already closed.
        }
      }
    }
  }
}

/**
 * A stored file read now and then for as long as a model is loaded (a
 * token's row of Raon's per-layer table), through a synchronous access
 * handle kept open: a `Blob.slice().arrayBuffer()` of a few kilobytes took a
 * Pixel 10 Pro's Chrome about 10 ms, an iPad Pro M4's Safari 1.3 and an M3
 * Ultra's Chrome 0.7, before every token not read lately. The handle locks
 * the file, as a load's do (see storedSources: another reader of it gets
 * the Blob). A read the handle fails (iOS closes a suspended page's
 * handles) is read from the Blob, and the handle opened again for the
 * next. `close()` releases it: call it when the model is unloaded.
 */
export async function keptSource(blob: Blob): Promise<{ source: ByteSource; close: () => Promise<void> }> {
  const plain = blobSource(blob)
  let stored: StoredSources | null = await storedSources([blob])
  let closed = false
  return {
    source: {
      size: blob.size,
      recycle: (buffer) => stored?.sources[0].recycle?.(buffer),
      read: async (offset, length) => {
        const now = stored
        if (now) {
          try {
            return await now.sources[0].read(offset, length)
          } catch (error) {
            if (error instanceof RangeError) throw error
            if (stored === now) {
              stored = null
              await now.close()
              if (!closed) stored = await storedSources([blob]).catch(() => null)
              if (closed) await stored?.close()
            }
          }
        }
        return plain.read(offset, length)
      }
    },
    close: async () => {
      closed = true
      const now = stored
      stored = null
      await now?.close()
    }
  }
}

/**
 * The reader worker: opens files by path through synchronous access handles
 * and answers reads with the bytes, transferred. One read at a time, in the
 * order asked.
 */
const READER = `
const files = []
// Buffers sent back for reuse, by length, up to 64 MiB (see BufferPool).
const free = new Map()
let held = 0
onmessage = async ({ data }) => {
  const { id } = data
  try {
    if (data.type === "open") {
      const root = await navigator.storage.getDirectory()
      const sizes = []
      for (const path of data.paths) {
        if (!path) { files.push(null); sizes.push(-1); continue }
        let folder = root
        for (const name of path.slice(0, -1)) folder = await folder.getDirectoryHandle(name)
        try {
          const access = await (await folder.getFileHandle(path[path.length - 1])).createSyncAccessHandle()
          files.push(access)
          sizes.push(access.getSize())
        } catch {
          files.push(null)
          sizes.push(-1)
        }
      }
      postMessage({ id, sizes })
    } else if (data.type === "recycle") {
      if (held + data.buffer.byteLength > 64 * 1048576) return
      held += data.buffer.byteLength
      const list = free.get(data.buffer.byteLength) || []
      list.push(data.buffer)
      free.set(data.buffer.byteLength, list)
      return
    } else if (data.type === "read") {
      const reused = free.get(data.length)?.pop()
      if (reused) held -= reused.byteLength
      const bytes = reused ? new Uint8Array(reused) : new Uint8Array(data.length)
      const read = files[data.file].read(bytes, { at: data.offset })
      if (read !== data.length) throw new Error("read " + read + " of " + data.length + " bytes at " + data.offset)
      postMessage({ id, bytes }, [bytes.buffer])
    } else if (data.type === "close") {
      for (const access of files.splice(0)) if (access) access.close()
      free.clear()
      held = 0
      postMessage({ id })
    }
  } catch (error) {
    postMessage({ id, error: String((error && error.message) || error) })
  }
}
`

async function readerSources(blobs: ReadonlyArray<Blob>): Promise<StoredSources> {
  const url = URL.createObjectURL(new Blob([READER], { type: "text/javascript" }))
  let worker: Worker
  try {
    worker = new Worker(url)
  } finally {
    URL.revokeObjectURL(url)
  }
  const pending = new Map<number, { resolve: (data: Record<string, unknown>) => void; reject: (error: Error) => void }>()
  let next = 0
  let failed: Error | null = null
  const fail = (error: Error) => {
    failed ??= error
    for (const p of pending.values()) p.reject(error)
    pending.clear()
    worker.terminate()
  }
  worker.onmessage = ({ data }: MessageEvent<Record<string, unknown> & { id: number }>) => {
    const p = pending.get(data.id)
    if (!p) return
    pending.delete(data.id)
    if (typeof data.error === "string") p.reject(new Error(data.error))
    else p.resolve(data)
  }
  worker.onerror = (event) => {
    event.preventDefault()
    fail(new Error(`The file reader worker failed: ${event.message || "it couldn't start"}`))
  }
  const ask = (message: Record<string, unknown>, transfer: Array<Transferable> = []) =>
    new Promise<Record<string, unknown>>((resolve, reject) => {
      if (failed) return reject(failed)
      const id = next++
      pending.set(id, { resolve, reject })
      worker.postMessage({ ...message, id }, transfer)
    })
  const paths = blobs.map((blob) => {
    const stored = fileHandles.get(blob)
    return stored ? [...stored.folder, stored.handle.name] : null
  })
  let sizes: Array<number>
  try {
    sizes = (await ask({ type: "open", paths })).sizes as Array<number>
  } catch (error) {
    worker.terminate()
    throw error
  }
  const sources = blobs.map((blob, file): ByteSource => {
    const size = sizes[file]
    if (size < 0) return blobSource(blob)
    return {
      size,
      recycle: (buffer) => {
        if (!failed) worker.postMessage({ type: "recycle", buffer }, [buffer])
      },
      read: async (offset, length) => {
        if (offset < 0 || offset + length > size) {
          throw new RangeError(`read of ${length} bytes at ${offset} is outside the file (${size} bytes)`)
        }
        return (await ask({ type: "read", file, offset, length })).bytes as Uint8Array
      }
    }
  })
  return {
    sources,
    reader: true,
    close: async () => {
      try {
        await ask({ type: "close" })
      } catch {
        // Failed already: terminating closes its handles.
      }
      worker.terminate()
    }
  }
}
/**
 * Buffers given back for reuse (ByteSource.recycle), by length: a model's
 * layers read the same tensor lengths over and over, so a few buffers serve
 * a whole load instead of an allocation a read (freed buffers stay
 * resident in the allocator for a while; iOS counts them against a tab's
 * 2 GB).
 */
export class BufferPool {
  private free = new Map<number, Array<ArrayBuffer>>()
  private held = 0
  constructor(private readonly most = 64 << 20) {}
  take(length: number): Uint8Array {
    const buffer = this.free.get(length)?.pop()
    if (!buffer) return new Uint8Array(length)
    this.held -= length
    return new Uint8Array(buffer)
  }
  give(buffer: ArrayBuffer) {
    if (this.held + buffer.byteLength > this.most) return
    const list = this.free.get(buffer.byteLength) ?? []
    list.push(buffer)
    this.free.set(buffer.byteLength, list)
    this.held += buffer.byteLength
  }
  clear() {
    this.free.clear()
    this.held = 0
  }
}

function syncSource(access: SyncAccessHandle, pool: BufferPool): ByteSource {
  const size = access.getSize()
  return {
    size,
    recycle: (buffer) => pool.give(buffer),
    read: async (offset, length) => {
      if (offset < 0 || offset + length > size) {
        throw new RangeError(`read of ${length} bytes at ${offset} is outside the file (${size} bytes)`)
      }
      const bytes = pool.take(length)
      const read = access.read(bytes, { at: offset })
      if (read !== length) throw new Error(`read ${read} of ${length} bytes at ${offset}`)
      return bytes
    }
  }
}
