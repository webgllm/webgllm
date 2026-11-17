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

/** A stored file's handle (storage.ts), for reading it through a synchronous access handle. */
const fileHandles = new WeakMap<Blob, FileSystemFileHandle>()

/** `file`, remembered as `handle`'s (see storedSources). */
export function withHandle(file: File, handle: FileSystemFileHandle): File {
  fileHandles.set(file, handle)
  return file
}

type SyncAccessHandle = {
  read: (into: Uint8Array, options: { at: number }) => number
  getSize: () => number
  close: () => void
}

/**
 * Sources for a load's files. A stored file in a worker is read through a
 * synchronous access handle: on an iPhone 16 Pro Max (Safari, iOS 27.2) its
 * 1,090 MB of Qwen3.5 weights read in ~100 ms that way, against 300–640 ms
 * through `Blob.slice().arrayBuffer()`. Any other file, or one whose handle
 * can't be opened (another reader holds it), is read as a Blob. `close()`
 * releases the handles: call it once the load has read everything.
 */
export async function storedSources(blobs: ReadonlyArray<Blob>): Promise<{ sources: Array<ByteSource>; close: () => void }> {
  const opened: Array<SyncAccessHandle> = []
  const inWorker = typeof (globalThis as { WorkerGlobalScope?: unknown }).WorkerGlobalScope !== "undefined"
  const sources = await Promise.all(
    blobs.map(async (blob) => {
      const handle = fileHandles.get(blob) as (FileSystemFileHandle & { createSyncAccessHandle?: () => Promise<SyncAccessHandle> }) | undefined
      if (inWorker && handle?.createSyncAccessHandle) {
        try {
          const access = await handle.createSyncAccessHandle()
          opened.push(access)
          return syncSource(access)
        } catch {
          // Held by another reader or writer: the Blob.
        }
      }
      return blobSource(blob)
    })
  )
  return {
    sources,
    close: () => {
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

function syncSource(access: SyncAccessHandle): ByteSource {
  const size = access.getSize()
  return {
    size,
    read: async (offset, length) => {
      if (offset < 0 || offset + length > size) {
        throw new RangeError(`read of ${length} bytes at ${offset} is outside the file (${size} bytes)`)
      }
      const bytes = new Uint8Array(length)
      const read = access.read(bytes, { at: offset })
      if (read !== length) throw new Error(`read ${read} of ${length} bytes at ${offset}`)
      return bytes
    }
  }
}
