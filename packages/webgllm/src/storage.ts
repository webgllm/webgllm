/**
 * Model files in the Origin Private File System: downloaded once (resuming
 * where a dropped connection left off), then read from disk, offline too.
 * Runs in the worker: files are written through sync access handles, which
 * every browser with OPFS has in workers (Safari before 26 has no
 * `createWritable`). A file counts as downloaded once its `.done` marker
 * exists, so an interrupted download is never mistaken for a whole one.
 */
import type { Tuned } from "./engine.js"
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
 * What a load learned about this GPU and model that the next load needn't
 * measure again (see worker.ts), keyed by GPU, browser, model and engine
 * code. Best effort: no storage means measuring each time.
 */
export type GpuNote = {
  layout: "rgba4" | "bytes"
  selfCheck: number
  /** Engine.tune's picks (older notes have none: tuned again). */
  tuned?: Tuned
}

const NOTES = "gpu-notes.json"

export async function readGpuNotes(): Promise<Record<string, GpuNote>> {
  return (await readNotes()) ?? {}
}

/** The notes; {} if there are none yet, null if they couldn't be read. */
async function readNotes(): Promise<Record<string, GpuNote> | null> {
  let file: File
  try {
    file = await (await (await dir()).getFileHandle(NOTES)).getFile()
  } catch (error) {
    return error instanceof DOMException && error.name === "NotFoundError" ? {} : null
  }
  try {
    return JSON.parse(await file.text()) as Record<string, GpuNote>
  } catch {
    // Unreadable or torn: written over.
    return {}
  }
}

/** Note writes, one at a time (a file's sync access handle is exclusive). */
let noteWrites: Promise<void> = Promise.resolve()

export function writeGpuNote(key: string, note: GpuNote): Promise<void> {
  noteWrites = noteWrites.then(() => writeNote(key, note))
  return noteWrites
}

async function writeNote(key: string, note: GpuNote) {
  try {
    // (Notes that exist but can't be read now, say another tab holds the
    // file, aren't written over with this one alone.)
    const notes = await readNotes()
    if (!notes) return
    // Notes for this GPU, browser and model from older engine code (the
    // key's last part) are stale: dropped.
    const prefix = key.slice(0, key.lastIndexOf("|") + 1)
    for (const k of Object.keys(notes)) if (k.startsWith(prefix)) delete notes[k]
    notes[key] = note
    const handle = await (await dir()).getFileHandle(NOTES, { create: true })
    const access = await openAccess(handle)
    try {
      const bytes = new TextEncoder().encode(JSON.stringify(notes))
      access.truncate(0)
      access.write(bytes, { at: 0 })
      access.flush()
    } finally {
      access.close()
    }
  } catch {
    // Remembering is optional.
  }
}

/** Waits `ms`, or rejects when `signal` aborts. */
const sleep = (ms: number, signal?: AbortSignal) =>
  new Promise<void>((resolve, reject) => {
    const abort = () => {
      clearTimeout(timer)
      reject(signal?.reason)
    }
    const timer = setTimeout(() => {
      signal?.removeEventListener("abort", abort)
      resolve()
    }, ms)
    signal?.addEventListener("abort", abort, { once: true })
  })

/**
 * The file's sync access handle. One file has one at a time: a download
 * of it that was just superseded may hold it until its fetch notices, so
 * a few tries.
 */
async function openAccess(
  handle: FileSystemFileHandle,
  signal?: AbortSignal
): Promise<SyncAccessHandle> {
  for (let attempt = 0; ; attempt++) {
    try {
      return (await (
        handle as FileSystemFileHandle & {
          createSyncAccessHandle: () => Promise<SyncAccessHandle>
        }
      ).createSyncAccessHandle()) as SyncAccessHandle
    } catch (error) {
      if (signal?.aborted || attempt >= 10) throw error
      await sleep(250 * (attempt + 1), signal)
    }
  }
}

/**
 * Downloads `file` to disk unless it's there, reporting bytes as they come,
 * and returns it. Survives dropped connections (resuming with a range
 * request) and a closed access handle (browsers close them, iOS Safari
 * for a page it suspends): a new handle, from the bytes on disk.
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
  let access = await openAccess(handle, signal)
  /** The bytes on disk, through a new handle if the browser closed this one. */
  const size = async () => {
    try {
      return access.getSize()
    } catch {
      try {
        access.close()
      } catch {
        // Already closed.
      }
      access = await openAccess(handle, signal)
      return access.getSize()
    }
  }
  try {
    let loaded = await size()
    if (file.bytes !== undefined && loaded > file.bytes) {
      access.truncate(0)
      loaded = 0
    }
    let total = file.bytes ?? 0
    // Failures in a row with no bytes gained: an interruption that made
    // progress (iOS suspending the page, a dropped connection) starts over.
    let attempt = 0
    for (;;) {
      // Already whole (the marker was all that was missing).
      if (file.bytes !== undefined && loaded === file.bytes) break
      const start = loaded
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
        try {
          for (;;) {
            const { done: end, value } = await reader.read()
            if (end) break
            access.write(value, { at: loaded })
            loaded += value.byteLength
            onProgress(loaded, total)
          }
          access.flush()
        } catch (error) {
          // (A write failed: stop the response rather than let it run on.)
          reader.cancel().catch(() => undefined)
          throw error
        }
        break
      } catch (error) {
        if (signal?.aborted) throw error
        loaded = await size()
        attempt = loaded > start ? 1 : attempt + 1
        if (attempt >= ATTEMPTS) throw error
        await sleep(1000 * 2 ** (attempt - 1), signal)
      }
    }
    const final = await size()
    if (file.bytes !== undefined && final !== file.bytes) {
      access.truncate(0)
      throw new Error(
        `${file.path} downloaded as ${final} bytes, expected ${file.bytes}.`
      )
    }
  } finally {
    try {
      access.close()
    } catch {
      // Already closed.
    }
  }
  await folder.getFileHandle(marker(file), { create: true })
  return handle.getFile()
}
