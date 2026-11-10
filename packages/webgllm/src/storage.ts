/**
 * Model files in the Origin Private File System: downloaded once (resuming
 * where a dropped connection left off), then read from disk, offline too.
 * Runs in the worker: files are written through sync access handles, which
 * every browser with OPFS has in workers (Safari before 26 has no
 * `createWritable`). A file counts as downloaded once its `.done` marker
 * exists, so an interrupted download is never mistaken for a whole one.
 */
import type { Tuned } from "./engine.js"
import { fileUrl, isCatalogModel, WEBGL_MODEL_META } from "./model-meta.js"
import type { ModelFile, WebGlModelMeta } from "./model-meta.js"

const DIR = "webgl-models"
const ATTEMPTS = 6

const safe = (name: string) => name.replace(/[^\w.@-]+/g, "_")

/**
 * Every character but letters, digits and "-" as `_` and its UTF-8 bytes in
 * hex: reversible, so two strings never share a name (unlike `safe`, which
 * makes "a/b" and "a_b" both "a_b").
 */
export const escapeName = (text: string) =>
  [...new TextEncoder().encode(text)]
    .map((b) => (/[A-Za-z0-9-]/.test(String.fromCharCode(b)) ? String.fromCharCode(b) : `_${b.toString(16).padStart(2, "0")}`))
    .join("")

/** The catalog's files, which keep the names earlier versions saved them under. */
let catalogFiles: Set<string> | undefined
const isCatalogFile = (file: ModelFile) => {
  catalogFiles ??= new Set(
    Object.values(WEBGL_MODEL_META).flatMap((meta) => {
      const m = meta as WebGlModelMeta
      const weights = m.files.kind === "gguf" ? [m.files.model] : [m.files.graph, m.files.data, m.files.config]
      return [...weights, m.tokenizer.tokenizer, m.tokenizer.config, ...(m.tokenizer.chatTemplate ? [m.tokenizer.chatTemplate] : [])]
    }).map((f) => `${f.repo}\n${f.path}`)
  )
  return catalogFiles.has(`${file.repo}\n${file.path}`)
}

/**
 * The file's name on disk, at main. A catalog file keeps the name earlier
 * versions gave it (its repository and path made safe), so what's
 * downloaded is found again; the catalog's names don't collide. Any other
 * file's name is its repository and path escaped (escapeName) around
 * "@main~": no two paths share one, and none is a catalog name (those have
 * no "~"), a marker (those end in ".done" or ".etag") or a manifest.
 */
export const diskName = (file: ModelFile) =>
  isCatalogFile(file) ? safe(`${file.repo}@main/${file.path}`) : `${escapeName(file.repo)}@main~${escapeName(file.path)}`

const marker = (file: ModelFile) => `${diskName(file)}.done`

/** The version (ETag or Last-Modified) of a download in progress, to resume only from it. */
const etagName = (file: ModelFile) => `${diskName(file)}.etag`

type SyncAccessHandle = {
  getSize: () => number
  read: (buffer: Uint8Array, options: { at: number }) => number
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
    return await (await folder.getFileHandle(diskName(file))).getFile()
  } catch {
    return null
  }
}

/**
 * Manifests: the meta of each model outside the catalog whose files were
 * downloaded, as the page resolved it. After a reload the model resolves
 * from it with no request to the Hub, and deleting another model keeps the
 * files it still reads.
 */
const MANIFEST = "manifest~"
const manifestName = (model: string) => `${MANIFEST}${escapeName(model)}.json`

/** Whether `value` has a model meta's shape, for `model` (a manifest someone could have edited). */
function isMeta(value: unknown, model?: string): value is WebGlModelMeta {
  const m = value as WebGlModelMeta | null
  const file = (f: unknown) => !!f && typeof (f as ModelFile).repo === "string" && typeof (f as ModelFile).path === "string"
  return (
    !!m && typeof m === "object" && typeof m.name === "string" && (model === undefined || m.name === model) &&
    (m.files?.kind === "gguf" ? file(m.files.model) : m.files?.kind === "onnx" && file(m.files.graph) && file(m.files.data) && file(m.files.config)) &&
    file(m.tokenizer?.tokenizer) && file(m.tokenizer?.config) && !!m.supports && !!m.sampling
  )
}

/**
 * Whether a stored manifest is from before 0.2.0, whose reasoning levels held
 * the chat template's values ("false", "true"): it's deleted, and the model
 * resolves again (its downloaded files stay, and are reused).
 */
const isStale = (meta: WebGlModelMeta) => {
  const levels = meta.reasoning?.levels as Record<string, string> | undefined
  return !!levels && (levels.off === "false" || levels.high === "true")
}

/** `model`'s stored manifest, or undefined (none, unreadable, or no storage here). */
export async function readManifest(model: string): Promise<WebGlModelMeta | undefined> {
  try {
    const text = await (await (await (await dir()).getFileHandle(manifestName(model))).getFile()).text()
    const meta: unknown = JSON.parse(text)
    if (!isMeta(meta, model)) return undefined
    if (isStale(meta)) {
      await removeManifest(model)
      return undefined
    }
    return meta
  } catch {
    return undefined
  }
}

/** Every stored manifest. */
export async function storedManifests(): Promise<Array<WebGlModelMeta>> {
  const metas: Array<WebGlModelMeta> = []
  try {
    const folder = await dir()
    for await (const name of (folder as unknown as { keys: () => AsyncIterable<string> }).keys()) {
      if (!name.startsWith(MANIFEST)) continue
      try {
        const meta: unknown = JSON.parse(await (await (await folder.getFileHandle(name)).getFile()).text())
        if (!isMeta(meta) || manifestName(meta.name) !== name) continue
        if (isStale(meta)) await removeManifest(meta.name)
        else metas.push(meta)
      } catch {
        // Unreadable: skipped.
      }
    }
  } catch {
    // No storage here.
  }
  return metas
}

/** Saves a downloaded model's manifest (in the worker: sync access handles). Catalog models need none. */
export async function saveManifest(meta: WebGlModelMeta): Promise<void> {
  if (isCatalogModel(meta.name)) return
  try {
    await writeText(await dir(), manifestName(meta.name), JSON.stringify(meta))
  } catch {
    // Best effort: without it the model resolves from the Hub again.
  }
}

/** Removes `model`'s manifest. */
export async function removeManifest(model: string): Promise<void> {
  try {
    await (await dir()).removeEntry(manifestName(model))
  } catch {
    // None.
  }
}

/**
 * What a load learned about this GPU and model that the next load needn't
 * measure again (see handler.ts), keyed by GPU, browser, model and engine
 * code. Best effort: no storage means measuring each time.
 */
export type GpuNote = {
  layout: "rgba4" | "bytes"
  calibrated?: boolean
  /** Proven during real prompt processing, never inferred from startup probes. */
  pacing?: { reads: number; window: number; frameMs: number }
  selfCheck: number
  head?: { expanded: boolean; split: number }
  /** Engine.tune's picks (older notes have none: tuned again). */
  tuned?: Tuned
}

const NOTES = "gpu-notes.json"

/** The notes; {} if there are none or they can't be read. */
export async function readGpuNotes(): Promise<Record<string, GpuNote>> {
  try {
    const file = await (await (await dir()).getFileHandle(NOTES)).getFile()
    return JSON.parse(await file.text()) as Record<string, GpuNote>
  } catch {
    return {}
  }
}

/** Tuning choices from the same GPU, browser, model and settings. */
export function previousGpuTuning(
  notes: Record<string, GpuNote>,
  scope: string,
  settings: string,
  maxError: number
): GpuNote | undefined {
  return Object.entries(notes).find(([key, note]) =>
    key.startsWith(scope) && key.endsWith(`.${settings}`) &&
    note?.calibrated === true && note.tuned &&
    note.selfCheck >= 0 && note.selfCheck <= maxError
  )?.[1]
}

/**
 * Note writes, one at a time in this worker. Workers each have their own
 * queue, so a write reads the notes through the file's sync access handle,
 * which is exclusive across workers and tabs: none can write in between.
 */
let noteWrites: Promise<void> = Promise.resolve()

export function writeGpuNote(key: string, note: GpuNote): Promise<void> {
  noteWrites = noteWrites.then(() => writeNote(key, note))
  return noteWrites
}

async function writeNote(key: string, note: GpuNote) {
  try {
    const handle = await (await dir()).getFileHandle(NOTES, { create: true })
    const access = await openAccess(handle, undefined, 1)
    try {
      const old = new Uint8Array(access.getSize())
      access.read(old, { at: 0 })
      let notes: Record<string, GpuNote> = {}
      try {
        if (old.byteLength) notes = JSON.parse(new TextDecoder().decode(old)) as Record<string, GpuNote>
      } catch {
        // Unreadable or torn: written over.
      }
      // Notes for this GPU, browser and model from older engine code (the
      // key's last part) are stale: dropped.
      const prefix = key.slice(0, key.lastIndexOf("|") + 1)
      for (const k of Object.keys(notes))
        if (k.startsWith(prefix)) delete notes[k]
      notes[key] = note
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

/** The origin's storage quota ran out (a DOMException: not an Error in every engine). */
export const isQuotaError = (error: unknown) =>
  !!error && typeof error === "object" && (error as { name?: unknown }).name === "QuotaExceededError"

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
  signal?: AbortSignal,
  retries = 10
): Promise<SyncAccessHandle> {
  for (let attempt = 0; ; attempt++) {
    try {
      return (await (
        handle as FileSystemFileHandle & {
          createSyncAccessHandle: () => Promise<SyncAccessHandle>
        }
      ).createSyncAccessHandle()) as SyncAccessHandle
    } catch (error) {
      if (signal?.aborted || attempt >= retries) throw error
      await sleep(250 * (attempt + 1), signal)
    }
  }
}

/** A small text file's contents, or "" if there's none. */
async function readText(folder: FileSystemDirectoryHandle, name: string) {
  try {
    return await (await (await folder.getFileHandle(name)).getFile()).text()
  } catch {
    return ""
  }
}

async function writeText(
  folder: FileSystemDirectoryHandle,
  name: string,
  text: string,
  signal?: AbortSignal
) {
  const access = await openAccess(
    await folder.getFileHandle(name, { create: true }),
    signal
  )
  try {
    access.truncate(0)
    access.write(new TextEncoder().encode(text), { at: 0 })
    access.flush()
  } finally {
    access.close()
  }
}

/**
 * Whether `name` on disk (a file, or its `.done` or `.etag`) is `file` as
 * earlier versions of this package saved it, at a pinned commit:
 * "repo@<commit>/path" made safe. Matched whole, commit included: with "/"
 * made "_", other files' names can end the same (main's
 * tokenizer_config.json ends like config.json).
 */
export function isOldVersion(name: string, file: ModelFile) {
  const base = name.replace(/\.(done|etag)$/, "")
  const prefix = safe(`${file.repo}@`)
  const own = safe(`/${file.path}`)
  return (
    base.startsWith(prefix) &&
    base.endsWith(own) &&
    /^[0-9a-f]{40}$/.test(base.slice(prefix.length, base.length - own.length))
  )
}

/**
 * Removes the files earlier versions of this package saved for `file` at a
 * pinned commit, once it's downloaded at main.
 */
async function removeOldVersions(
  folder: FileSystemDirectoryHandle,
  file: ModelFile
) {
  const old: Array<string> = []
  for await (const name of (
    folder as unknown as { keys: () => AsyncIterable<string> }
  ).keys()) {
    if (isOldVersion(name, file)) old.push(name)
  }
  for (const name of old) await folder.removeEntry(name).catch(() => undefined)
}

/**
 * Downloads `file` to disk unless it's there, reporting bytes as they come,
 * and returns it. Survives dropped connections (resuming with a range
 * request) and a closed access handle (browsers close them, iOS Safari
 * for a page it suspends): a new handle, from the bytes on disk. Files are
 * at main, so one can change between tries: a resume asks for the rest only
 * of the same version (If-Range with its ETag or Last-Modified), and gets
 * the whole new file otherwise. Bytes with no version to check aren't resumed.
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
  // Another download may have finished it while this one waited for the lock.
  if (await storedFile(file)) {
    access.close()
    return handle.getFile()
  }
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
    // The version (ETag, or Last-Modified) the bytes on disk are from: a
    // resume without one could join an old prefix to a newer file.
    let version = loaded > 0 ? await readText(folder, etagName(file)) : ""
    if (loaded > 0 && !version) {
      access.truncate(0)
      loaded = 0
    }
    // The file's size, from the server once it answers.
    let total = 0
    // Failures in a row with no bytes gained: an interruption that made
    // progress (iOS suspending the page, a dropped connection) starts over.
    let attempt = 0
    for (;;) {
      const start = loaded
      try {
        const headers: Record<string, string> = {}
        if (loaded > 0) {
          headers.Range = `bytes=${loaded}-`
          headers["If-Range"] = version
        }
        const response = await fetch(fileUrl(file), {
          signal,
          headers,
          referrerPolicy: "no-referrer"
        })
        /** Drops the bytes on disk and asks for the whole file. */
        const restart = () => {
          void response.body?.cancel().catch(() => undefined)
          // (Already asking for the whole file: a failure, not a loop.)
          if (start === 0) throw new Error(`HTTP ${response.status} for ${file.path} without the whole file`)
          access.truncate(0)
          loaded = 0
          version = ""
        }
        if (response.status === 416 && loaded > 0) {
          // "bytes */size": nothing past the end, so the same version is
          // whole on disk, if the sizes agree. Otherwise the partial is stale.
          const whole = /^bytes \*\/(\d+)$/.exec(response.headers.get("Content-Range") ?? "")
          if (whole && Number(whole[1]) === loaded && (!total || total === loaded)) break
          restart()
          continue
        }
        if (!response.ok) {
          throw new Error(`HTTP ${response.status} for ${file.path}`)
        }
        if (response.status === 206) {
          // "bytes start-end/total": the rest, from where the file stops,
          // of the size it had. Anything else would misplace it: start over.
          const range = /^bytes (\d+)-\d+\/(\d+)$/.exec(
            response.headers.get("Content-Range") ?? ""
          )
          if (!range || Number(range[1]) !== loaded || (total && Number(range[2]) !== total)) {
            restart()
            continue
          }
          total = Number(range[2])
        } else {
          // The whole file: a first download, a server that ignores
          // ranges, or a file that changed since the last try.
          if (loaded > 0) access.truncate(0)
          loaded = 0
          version = response.headers.get("ETag") ?? response.headers.get("Last-Modified") ?? ""
          await writeText(folder, etagName(file), version, signal)
          total = Number(response.headers.get("Content-Length") ?? 0)
        }
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
        // A full disk stays full: no point trying again.
        if (signal?.aborted || isQuotaError(error)) throw error
        loaded = await size()
        if (loaded > 0 && !version) {
          // No version to resume from: the next try asks for the whole file.
          access.truncate(0)
          loaded = 0
        }
        // (Progress is only bytes kept for the next try.)
        attempt = loaded > start ? 1 : attempt + 1
        if (attempt >= ATTEMPTS) throw error
        await sleep(1000 * 2 ** (attempt - 1), signal)
      }
    }
    const final = await size()
    if (total && final !== total) {
      access.truncate(0)
      throw new Error(
        `${file.path} downloaded as ${final} bytes, expected ${total}.`
      )
    }
    // Marked done while still holding the lock: a download waiting for it
    // then sees the file finished instead of starting it over.
    await folder.getFileHandle(marker(file), { create: true })
    await folder.removeEntry(etagName(file)).catch(() => undefined)
  } finally {
    try {
      access.close()
    } catch {
      // Already closed.
    }
  }
  await removeOldVersions(folder, file)
  return handle.getFile()
}
