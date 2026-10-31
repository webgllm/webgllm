/**
 * Model files in the Origin Private File System: downloaded once (resuming
 * where a dropped connection left off), then read from disk, offline too.
 */
import { fileUrl } from "./model-meta.js"
import type { ModelFile } from "./model-meta.js"

const DIR = "webgl-models"
const ATTEMPTS = 6

/** The file's name on disk: its repository, commit and path. */
const diskName = (file: ModelFile) =>
  `${file.repo}@${file.revision}/${file.path}`.replace(/[^\w.@-]+/g, "_")

async function dir() {
  const root = await navigator.storage.getDirectory()
  return root.getDirectoryHandle(DIR, { create: true })
}

/** The downloaded file, or null if it isn't complete on disk. */
export async function storedFile(file: ModelFile): Promise<File | null> {
  try {
    const handle = await (await dir()).getFileHandle(diskName(file))
    const stored = await handle.getFile()
    // A partial download is kept under its own name until it's complete.
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
  const partName = `${diskName(file)}.part`
  const part = await folder.getFileHandle(partName, { create: true })
  let loaded = (await part.getFile()).size
  let total = file.bytes ?? 0
  for (let attempt = 0; ; attempt++) {
    try {
      const response = await fetch(fileUrl(file), {
        signal,
        headers: loaded > 0 ? { Range: `bytes=${loaded}-` } : {},
        referrerPolicy: "no-referrer"
      })
      if (response.status === 200 && loaded > 0) loaded = 0
      else if (!response.ok)
        throw new Error(`HTTP ${response.status} for ${file.path}`)
      const length = Number(response.headers.get("Content-Length") ?? 0)
      if (!total) total = loaded + length
      const writable = await part.createWritable({
        keepExistingData: loaded > 0
      })
      if (loaded > 0) await writable.seek(loaded)
      else await writable.truncate(0)
      const reader = response.body?.getReader()
      if (!reader) throw new Error("No response body.")
      try {
        for (;;) {
          const { done: end, value } = await reader.read()
          if (end) break
          await writable.write(value)
          loaded += value.byteLength
          onProgress(loaded, total)
        }
      } finally {
        await writable.close()
      }
      break
    } catch (error) {
      if (signal?.aborted || attempt + 1 >= ATTEMPTS) throw error
      await new Promise((resolve) => setTimeout(resolve, 1000 * 2 ** attempt))
      loaded = (await part.getFile()).size
    }
  }
  // Complete: give it its real name.
  const finished = await part.getFile()
  if (file.bytes !== undefined && finished.size !== file.bytes) {
    await folder.removeEntry(partName)
    throw new Error(
      `${file.path} downloaded as ${finished.size} bytes, expected ${file.bytes}.`
    )
  }
  const movable = part as FileSystemFileHandle & {
    move?: (name: string) => Promise<void>
  }
  if (movable.move) {
    try {
      await movable.move(diskName(file))
      return (await folder.getFileHandle(diskName(file))).getFile()
    } catch {
      // Not supported for this handle: copy instead.
    }
  }
  const target = await folder.getFileHandle(diskName(file), { create: true })
  const out = await target.createWritable()
  await finished.stream().pipeTo(out)
  await folder.removeEntry(partName)
  return target.getFile()
}
