/**
 * Model files in the Origin Private File System: downloaded once (resuming
 * where a dropped connection left off), then read from disk, offline too.
 * Runs in the worker: files are written through sync access handles, which
 * every browser with OPFS has in workers (Safari before 26 has no
 * `createWritable`). A file counts as downloaded once its `.done` marker
 * exists, so an interrupted download is never mistaken for a whole one.
 */
import { fileUrl } from "./model-meta.js"
import type { ModelFile } from "./model-meta.js"

const DIR = "webgl-models"
const ATTEMPTS = 6

/** The file's name on disk: its repository, commit and path. */
const diskName = (file: ModelFile) =>
  `${file.repo}@${file.revision}/${file.path}`.replace(/[^\w.@-]+/g, "_")

const marker = (file: ModelFile) => `${diskName(file)}.done`

type SyncAccessHandle = {
  getSize: () => number
  truncate: (size: number) => void
  write: (data: Uint8Array, options: { at: number }) => number
  flush: () => void
  close: () => void
}

async function dir() {
  const root = await navigator.storage.getDirectory()
  return root.getDirectoryHandle(DIR, { create: true })
}

/** The downloaded file, or null if it isn't complete on disk. */
export async function storedFile(file: ModelFile): Promise<File | null> {
  try {
    const folder = await dir()
    await folder.getFileHandle(marker(file))
    const stored = await (await folder.getFileHandle(diskName(file))).getFile()
    return file.bytes === undefined || stored.size === file.bytes
      ? stored
      : null
  } catch {
    return null
  }
}

/**
 * Downloads `file` to disk unless it's there, reporting bytes as they come,
 * and returns it.
 */
export async function download(
  file: ModelFile,
  onProgress: (loaded: number, total: number) => void,
  signal?: AbortSignal
): Promise<File> {
  const done = await storedFile(file)
  if (done) return done
  const folder = await dir()
  const handle = await folder.getFileHandle(diskName(file), { create: true })
  const access = (await (
    handle as FileSystemFileHandle & {
      createSyncAccessHandle: () => Promise<SyncAccessHandle>
    }
  ).createSyncAccessHandle()) as SyncAccessHandle
  try {
    let loaded = access.getSize()
    if (file.bytes !== undefined && loaded > file.bytes) {
      access.truncate(0)
      loaded = 0
    }
    let total = file.bytes ?? 0
    for (let attempt = 0; ; attempt++) {
      // Already whole (the marker was all that was missing).
      if (file.bytes !== undefined && loaded === file.bytes) break
      try {
        const response = await fetch(fileUrl(file), {
          signal,
          headers: loaded > 0 ? { Range: `bytes=${loaded}-` } : {},
          referrerPolicy: "no-referrer"
        })
        // Nothing past the end: the file was whole (its size wasn't known).
        if (response.status === 416 && loaded > 0) break
        if (response.status === 200 && loaded > 0) {
          // The server sent the whole file instead of the rest.
          access.truncate(0)
          loaded = 0
        } else if (!response.ok) {
          throw new Error(`HTTP ${response.status} for ${file.path}`)
        }
        const length = Number(response.headers.get("Content-Length") ?? 0)
        if (!total) total = loaded + length
        const reader = response.body?.getReader()
        if (!reader) throw new Error("No response body.")
        for (;;) {
          const { done: end, value } = await reader.read()
          if (end) break
          access.write(value, { at: loaded })
          loaded += value.byteLength
          onProgress(loaded, total)
        }
        access.flush()
        break
      } catch (error) {
        if (signal?.aborted || attempt + 1 >= ATTEMPTS) throw error
        await new Promise((resolve) => setTimeout(resolve, 1000 * 2 ** attempt))
        loaded = access.getSize()
      }
    }
    if (file.bytes !== undefined && access.getSize() !== file.bytes) {
      const size = access.getSize()
      access.truncate(0)
      throw new Error(
        `${file.path} downloaded as ${size} bytes, expected ${file.bytes}.`
      )
    }
  } finally {
    access.close()
  }
  await folder.getFileHandle(marker(file), { create: true })
  return handle.getFile()
}
