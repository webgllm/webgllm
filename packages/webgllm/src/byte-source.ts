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
