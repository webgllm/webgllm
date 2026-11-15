/**
 * Saved prompt prefixes on disk (the Origin Private File System), so a
 * page loaded again starts with the system prompts it read before: the
 * prefix store's entries (see prefix-store.ts), each read back from the GPU
 * once and written to a file of its own, then uploaded again from it.
 *
 * A file is `WGL2`, its header's length (u32), when it was last used (f64,
 * ms: rewritten on each use, so the least recently used go first after a
 * reload too), the header (JSON: the identity it was made for, its
 * position, its tokens' count and each texture's format and size), its
 * tokens (i32), then each texture's bytes.
 * Only a file made for the same identity (the model's weight files, the
 * engine's code and settings: see handler.ts) is used. Another identity's
 * (another webgllm bundle on the same origin, another version) is left for
 * it, never matched, and counts toward the budget like any file. Files cut
 * short (a write that never finished) are deleted when found. Files past the
 * byte budget go least recently used first, whoever made them.
 */
import type { PrefixTexture } from "./engine.js"
import { escapeName } from "./storage.js"

export const DEFAULT_PROMPT_CACHE_DISK_BYTES = 512 * 2 ** 20

const DIR = "webgl-prompt-cache"
const MAGIC = 0x324c4757 // "WGL2", little-endian
/** Where the header starts: after the magic, its length and the last use. */
const HEAD = 16
const SUFFIX = ".prefix"

type Header = {
  identity: string
  position: number
  keys: number
  textures: Array<{ internal: number; width: number; height: number }>
}

/** A file in the folder: its tokens (none for another identity's, which never match), its bytes, its last use. */
type Indexed = { name: string; keys: Int32Array; bytes: number; data: number; used: number; foreign?: boolean }

/** A prefix read from disk: its tokens, and what Engine.importPrefix takes. */
export type DiskPrefix = { keys: Array<number>; position: number; textures: Array<PrefixTexture> }

type SyncAccessHandle = {
  write: (data: Uint8Array, options: { at: number }) => number
  truncate: (size: number) => void
  flush: () => void
  close: () => void
}

/** The OPFS operations used here (a fake one in tests). */
export type PrefixFolder = {
  names(): Promise<Array<string>>
  read(name: string, from?: number, to?: number): Promise<{ size: number; bytes: ArrayBuffer }>
  write(name: string, parts: Array<Uint8Array>): Promise<void>
  /** Overwrites `bytes` at `at`, the rest of the file as it was. */
  writeAt(name: string, at: number, bytes: Uint8Array): Promise<void>
  remove(name: string): Promise<void>
}

/** The model's folder of prefixes in the Origin Private File System. */
export function opfsFolder(model: string): PrefixFolder {
  let folder: Promise<FileSystemDirectoryHandle> | undefined
  const open = () =>
    (folder ??= navigator.storage
      .getDirectory()
      .then((root) => root.getDirectoryHandle(DIR, { create: true }))
      .then((dir) => dir.getDirectoryHandle(escapeName(model), { create: true })))
  return {
    async names() {
      const names: Array<string> = []
      const dir = (await open()) as FileSystemDirectoryHandle & { keys(): AsyncIterable<string> }
      for await (const name of dir.keys()) names.push(name)
      return names
    },
    async read(name, from = 0, to?: number) {
      const file = await (await (await open()).getFileHandle(name)).getFile()
      return { size: file.size, bytes: await file.slice(from, to).arrayBuffer() }
    },
    async write(name, parts) {
      const handle = await (await open()).getFileHandle(name, { create: true })
      // Sync access handles: every browser with OPFS has them in workers (Safari before 26 has no createWritable).
      const access = (await (handle as FileSystemFileHandle & {
        createSyncAccessHandle: () => Promise<SyncAccessHandle>
      }).createSyncAccessHandle())
      try {
        access.truncate(0)
        let at = 0
        for (const part of parts) at += access.write(part, { at })
        access.flush()
      } finally {
        access.close()
      }
    },
    async writeAt(name, at, bytes) {
      const handle = await (await open()).getFileHandle(name)
      const access = (await (handle as FileSystemFileHandle & {
        createSyncAccessHandle: () => Promise<SyncAccessHandle>
      }).createSyncAccessHandle())
      try {
        access.write(bytes, { at })
        access.flush()
      } finally {
        access.close()
      }
    },
    async remove(name) {
      await (await open()).removeEntry(name).catch(() => undefined)
    }
  }
}

/**
 * `promise`, or undefined as soon as `signal` aborts (a stopped request
 * stops waiting; what it was waiting for finishes unused).
 */
export function untilAborted<T>(promise: Promise<T>, signal?: AbortSignal): Promise<T | undefined> {
  if (!signal) return promise
  if (signal.aborted) {
    promise.catch(() => undefined)
    return Promise.resolve(undefined)
  }
  return new Promise<T | undefined>((resolve, reject) => {
    const onAbort = () => resolve(undefined)
    signal.addEventListener("abort", onAbort, { once: true })
    promise.then(
      (value) => { signal.removeEventListener("abort", onAbort); resolve(value) },
      (error) => { signal.removeEventListener("abort", onAbort); reject(error) }
    )
  })
}

/** A file's name: its tokens' count and their hash (compared in full when read). */
function nameOf(keys: ArrayLike<number>) {
  let a = 0x811c9dc5
  let b = 0x01000193
  for (let i = 0; i < keys.length; i++) {
    a = Math.imul(a ^ keys[i], 0x01000193)
    b = Math.imul(b ^ keys[i], 0x5bd1e995) ^ (b >>> 15)
  }
  return `${keys.length}-${(a >>> 0).toString(16)}${(b >>> 0).toString(16)}${SUFFIX}`
}

/** A last-use time as the file holds it (f64, little-endian). */
function usedBytes(used: number) {
  const bytes = new Uint8Array(8)
  new DataView(bytes.buffer).setFloat64(0, used, true)
  return bytes
}

function startsWith(keys: ArrayLike<number>, prefix: ArrayLike<number>) {
  if (prefix.length > keys.length) return false
  for (let i = 0; i < prefix.length; i++) if (keys[i] !== prefix[i]) return false
  return true
}

export class PrefixDisk {
  private index: Promise<Map<string, Indexed>> | undefined
  /** The last use's time: later than every file's, and each use later than the one before. */
  private clock = 0

  private stamp() {
    return (this.clock = Math.max(Date.now(), this.clock + 1))
  }

  constructor(
    private readonly folder: PrefixFolder,
    /** What a file must have been made for (see the module's comment). */
    private readonly identity: string,
    private limit = DEFAULT_PROMPT_CACHE_DISK_BYTES
  ) {}

  get budget() {
    return this.limit
  }

  /** A new budget: files past it are deleted, least recently used first. */
  async setBudget(bytes: number) {
    if (!Number.isFinite(bytes) || bytes < 0) throw new RangeError("promptCacheDiskBytes must be a nonnegative number.")
    if (bytes === this.limit) return
    this.limit = bytes
    await this.evict(0)
  }

  /** The files on disk for this identity, read once: others and broken ones are deleted. */
  private entries() {
    return (this.index ??= (async () => {
      const index = new Map<string, Indexed>()
      for (const name of await this.folder.names()) {
        if (!name.endsWith(SUFFIX)) continue
        try {
          const found = await this.readHeader(name)
          if (found) {
            index.set(name, found.entry)
            this.clock = Math.max(this.clock, found.entry.used)
          } else await this.folder.remove(name)
        } catch {
          await this.folder.remove(name)
        }
      }
      return index
    })())
  }

  /** A file's header and tokens, if it's whole and for this identity. */
  private async readHeader(name: string) {
    const start = await this.folder.read(name, 0, HEAD)
    if (start.bytes.byteLength < HEAD) return undefined
    const view = new DataView(start.bytes)
    if (view.getUint32(0, true) !== MAGIC) return undefined
    const length = view.getUint32(4, true)
    const used = view.getFloat64(8, true)
    const head = await this.folder.read(name, HEAD, HEAD + length)
    const header = JSON.parse(new TextDecoder().decode(head.bytes)) as Header
    const keysAt = HEAD + length
    const dataAt = keysAt + 4 * header.keys
    const data = header.textures.reduce((sum, t) => sum + t.width * t.height * 16, 0)
    const total = dataAt + data
    if (start.size !== total || header.keys !== header.position) return undefined
    const lastUse = Number.isFinite(used) ? used : 0
    // Another identity's: whole, so left for it.
    if (header.identity !== this.identity) return { header, dataAt, entry: { name, keys: new Int32Array(0), bytes: total, data, used: lastUse, foreign: true } }
    const keys = new Int32Array((await this.folder.read(name, keysAt, dataAt)).bytes)
    return { header, dataAt, entry: { name, keys, bytes: total, data, used: lastUse } }
  }

  /**
   * The longest prefix on disk that begins `keys`, longer than `least`
   * tokens and no longer than `most`, whose textures take no more than
   * `maxBytes`: read whole, and the most recently used now (on disk too).
   * Undefined if there's none (or it couldn't be read).
   */
  async match(keys: ArrayLike<number>, most: number, least = 0, maxBytes = Infinity): Promise<DiskPrefix | undefined> {
    if (this.limit <= 0) return undefined
    const index = await this.entries()
    let best: Indexed | undefined
    for (const entry of index.values()) {
      const n = entry.keys.length
      if (entry.foreign || n <= least || n > most || entry.data > maxBytes || (best && n <= best.keys.length)) continue
      if (startsWith(keys, entry.keys)) best = entry
    }
    if (!best) return undefined
    try {
      const found = await this.readHeader(best.name)
      if (!found || found.entry.foreign) throw new Error("The file changed.")
      const { header, dataAt } = found
      const file = await this.folder.read(best.name, dataAt)
      const textures: Array<PrefixTexture> = []
      let at = 0
      for (const t of header.textures) {
        const size = t.width * t.height * 16
        textures.push({ ...t, bytes: new Uint8Array(file.bytes, at, size) })
        at += size
      }
      if (at !== file.bytes.byteLength) throw new Error("The file is cut short.")
      best.used = this.stamp()
      await this.folder.writeAt(best.name, 8, usedBytes(best.used)).catch(() => undefined)
      return { keys: Array.from(best.keys), position: header.position, textures }
    } catch {
      index.delete(best.name)
      await this.folder.remove(best.name)
      return undefined
    }
  }

  /** Marks the file holding exactly `keys`, if there's one, as used now (a hit in memory counts too). */
  async touch(keys: ArrayLike<number>) {
    const entry = (await this.entries()).get(nameOf(keys))
    if (!entry || entry.foreign || entry.keys.length !== keys.length || !startsWith(keys, entry.keys)) return
    entry.used = this.stamp()
    await this.folder.writeAt(entry.name, 8, usedBytes(entry.used)).catch(() => undefined)
  }

  /** Whether a file holds exactly `keys`. */
  async has(keys: ArrayLike<number>) {
    const entry = (await this.entries()).get(nameOf(keys))
    return !!entry && !entry.foreign && entry.keys.length === keys.length && startsWith(keys, entry.keys)
  }

  /**
   * Writes a prefix (its tokens and textures) to a file, deleting the least
   * recently used past the budget first. Nothing if it can't fit at all, or
   * a token isn't a 32-bit integer (an image's key).
   */
  async save(keys: Array<number>, textures: Array<PrefixTexture>) {
    const data = textures.reduce((sum, t) => sum + t.bytes.byteLength, 0)
    if (!keys.length || keys.some((k) => !Number.isInteger(k) || k < -(2 ** 31) || k >= 2 ** 31)) return
    const header: Header = {
      identity: this.identity,
      position: keys.length,
      keys: keys.length,
      textures: textures.map(({ internal, width, height }) => ({ internal, width, height }))
    }
    // (Padded to whole words, so the tokens and textures read as typed arrays where they lie.)
    let json = JSON.stringify(header)
    while (new TextEncoder().encode(json).byteLength % 4) json += " "
    const head = new TextEncoder().encode(json)
    const total = HEAD + head.byteLength + 4 * keys.length + data
    if (total > this.limit) return
    const index = await this.entries()
    const name = nameOf(keys)
    await this.evict(total, name)
    const used = this.stamp()
    const start = new Uint8Array(HEAD)
    const view = new DataView(start.buffer)
    view.setUint32(0, MAGIC, true)
    view.setUint32(4, head.byteLength, true)
    view.setFloat64(8, used, true)
    const tokens = Int32Array.from(keys)
    await this.folder.write(name, [start, head, new Uint8Array(tokens.buffer), ...textures.map((t) => t.bytes)])
    index.set(name, { name, keys: tokens, bytes: total, data, used })
  }

  /** Deletes every file of this identity's (another's stay for it). */
  async clear() {
    const index = await this.entries()
    for (const [name, entry] of [...index]) {
      if (entry.foreign) continue
      index.delete(name)
      await this.folder.remove(name)
    }
  }

  /** Deletes the least recently used files until `room` more bytes fit (`replacing` is rewritten: not counted). */
  private async evict(room: number, replacing?: string) {
    const index = await this.entries()
    const entries = [...index.values()].filter((e) => e.name !== replacing).sort((a, b) => a.used - b.used)
    let total = entries.reduce((sum, e) => sum + e.bytes, 0)
    for (const entry of entries) {
      if (total + room <= this.limit) break
      total -= entry.bytes
      index.delete(entry.name)
      await this.folder.remove(entry.name)
    }
  }
}

/** Deletes a model's saved prefixes from disk (see models.delete). */
export async function removePrefixFolder(model: string) {
  try {
    const dir = await (await navigator.storage.getDirectory()).getDirectoryHandle(DIR)
    await dir.removeEntry(escapeName(model), { recursive: true })
  } catch {
    // None saved, or no OPFS.
  }
}

/** What persistPrefix needs of the engine (see Engine). */
export type ExportEngine<S> = { exportPrefix(saved: S): Promise<Array<PrefixTexture>> }

/**
 * Writes a prefix just saved in the memory store to disk, if the store
 * still holds it (its textures aren't freed) and the disk lacks it. A
 * failure is only logged: the memory store has it anyway.
 */
export async function persistPrefix<S>(
  disk: PrefixDisk | undefined,
  held: (saved: S) => boolean,
  engine: ExportEngine<S>,
  fresh: { keys: Array<number>; saved: S } | undefined,
  log: (text: string) => void
) {
  if (!disk || !fresh || disk.budget <= 0 || !held(fresh.saved)) return
  try {
    if (await disk.has(fresh.keys)) return
    await disk.save(fresh.keys, await engine.exportPrefix(fresh.saved))
  } catch (error) {
    log(`Couldn't save a prompt prefix to disk: ${error instanceof Error ? error.message : String(error)}`)
  }
}
