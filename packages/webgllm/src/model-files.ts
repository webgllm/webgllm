/**
 * What the page can learn about a model's files without loading it: whether
 * they're in storage (the worker's OPFS folder, see storage.ts), their size,
 * the model's context length, and removing them. Facts that don't change
 * (sizes, context lengths) are looked up once per page.
 */
import { httpSource, blobSource } from "./byte-source.js"
import { fileUrl, modelMeta } from "./model-meta.js"
import type { ModelFile, WebGlModelMeta, WebGlTextModel } from "./model-meta.js"
import { diskName, isOldVersion, storedFile } from "./storage.js"
import type { ByteSource } from "./weights.js"

/** Every file a model loads: tokenizer, chat template and weights (as the worker fetches them). */
export function modelFiles(meta: WebGlModelMeta): Array<ModelFile> {
  const weights =
    meta.files.kind === "gguf" ? [meta.files.model] : [meta.files.config, meta.files.graph, meta.files.data]
  return [
    meta.tokenizer.tokenizer,
    meta.tokenizer.config,
    ...(meta.tokenizer.chatTemplate ? [meta.tokenizer.chatTemplate] : []),
    ...weights
  ]
}

/** The storage questions the pool asks (a fake one in tests). */
export type ModelStorage = {
  /** Whether all the model's files are in storage. */
  downloaded: (model: WebGlTextModel) => Promise<boolean>
  /** Bytes to download, all files together; 0 when unknown. */
  size: (model: WebGlTextModel) => Promise<number>
  /** The model's own context length; 0 when it can't be read. */
  contextLength: (model: WebGlTextModel) => Promise<number>
  /** Removes the model's files (not ones `keep` still uses). */
  remove: (model: WebGlTextModel, keep: Set<string>) => Promise<void>
}

const fileKey = (file: ModelFile) => `${file.repo}/${file.path}`
export { fileKey }


const sizes = new Map<string, Promise<number>>()
const contexts = new Map<string, Promise<number>>()

/** A file's bytes: from storage when it's there, else the server's (a HEAD request). */
function fileSize(file: ModelFile): Promise<number> {
  const key = fileKey(file)
  let size = sizes.get(key)
  if (!size) {
    size = (async () => {
      const stored = await storedFile(file)
      if (stored) return stored.size
      const head = await fetch(fileUrl(file), { method: "HEAD", referrerPolicy: "no-referrer" })
      const length = Number(head.headers.get("content-length") ?? Number.NaN)
      if (!head.ok || !(length >= 0)) throw new Error(`No size for ${file.path}`)
      return length
    })()
    // A failed lookup is tried again next time.
    size.catch(() => sizes.delete(key))
    sizes.set(key, size)
  }
  return size
}

async function source(file: ModelFile): Promise<ByteSource> {
  const stored = await storedFile(file)
  return stored ? blobSource(stored) : httpSource(fileUrl(file))
}

/** Bytes of each GGUF scalar type (gguf.ts's GgufValueType). */
const GGUF_SCALAR: Record<number, number> = { 0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8 }
const GGUF_STRING = 8
const GGUF_ARRAY = 9

/**
 * A GGUF's `<arch>.context_length`, reading its metadata in order and
 * stopping there: writers put the architecture's keys before the
 * tokenizer's megabytes of vocabulary, so this reads a chunk or two (a
 * range request when the file isn't downloaded). Undefined if absent.
 */
export async function ggufContextLength(source: ByteSource, chunkSize = 64 << 10): Promise<number | undefined> {
  let buf: Uint8Array = new Uint8Array(0)
  let start = 0
  let pos = 0
  const need = async (n: number) => {
    if (pos + n <= start + buf.length) return
    const length = Math.min(Math.max(n, chunkSize), source.size - pos)
    if (length < n) throw new Error("GGUF: unexpected end of file")
    buf = await source.read(pos, length)
    start = pos
  }
  const view = () => new DataView(buf.buffer, buf.byteOffset, buf.byteLength)
  const u32 = async () => {
    await need(4)
    const v = view().getUint32(pos - start, true)
    pos += 4
    return v
  }
  const u64 = async () => {
    await need(8)
    const v = view().getUint32(pos - start, true) + view().getUint32(pos - start + 4, true) * 2 ** 32
    pos += 8
    return v
  }
  const text = async () => {
    const n = await u64()
    await need(n)
    const s = new TextDecoder().decode(buf.subarray(pos - start, pos - start + n))
    pos += n
    return s
  }
  const skip = async (type: number): Promise<void> => {
    if (type === GGUF_STRING) {
      // (Not `pos += await u64()`: that adds to the position before the read.)
      const length = await u64()
      pos += length
    } else if (type === GGUF_ARRAY) {
      const element = await u32()
      const length = await u64()
      if (element === GGUF_STRING || element === GGUF_ARRAY) for (let i = 0; i < length; i++) await skip(element)
      else pos += length * (GGUF_SCALAR[element] ?? Number.NaN)
    } else pos += GGUF_SCALAR[type] ?? Number.NaN
    if (Number.isNaN(pos)) throw new Error(`GGUF: unknown value type ${type}`)
  }
  await need(24)
  if (new TextDecoder().decode(buf.subarray(0, 4)) !== "GGUF") throw new Error("not a GGUF file")
  pos = 8
  await u64() // tensors
  const count = await u64()
  let arch: string | undefined
  for (let i = 0; i < count; i++) {
    const key = await text()
    const type = await u32()
    if (key === "general.architecture" && type === GGUF_STRING) {
      arch = await text()
      continue
    }
    if (arch && key === `${arch}.context_length` && GGUF_SCALAR[type]) {
      await need(GGUF_SCALAR[type])
      const v = view()
      const p = pos - start
      return type === 10 || type === 11 ? v.getUint32(p, true) + v.getUint32(p + 4, true) * 2 ** 32 : v.getUint32(p, true)
    }
    await skip(type)
  }
  return undefined
}

/**
 * The model's context window, as the loaders read it: the GGUF's
 * `<arch>.context_length`, or config.json's max_position_embeddings
 * (Qwen3.5's in text_config); 8,192 where a model doesn't say, as there.
 */
async function readContextLength(meta: WebGlModelMeta): Promise<number> {
  if (meta.files.kind === "gguf") return (await ggufContextLength(await source(meta.files.model))) ?? 8192
  const file = meta.files.config
  const stored = await storedFile(file)
  const text = stored ? await stored.text() : await (await fetch(fileUrl(file), { referrerPolicy: "no-referrer" })).text()
  const outer = JSON.parse(text) as Record<string, unknown>
  const config = (typeof outer.text_config === "object" && outer.text_config !== null ? outer.text_config : outer) as Record<string, unknown>
  const value = config.max_position_embeddings
  return typeof value === "number" ? value : 8192
}

async function folder(): Promise<FileSystemDirectoryHandle> {
  const root = await navigator.storage.getDirectory()
  return root.getDirectoryHandle("webgl-models", { create: true })
}

/** The browser's storage (OPFS), read from the page. */
export const browserStorage: ModelStorage = {
  async downloaded(model) {
    const stored = await Promise.all(modelFiles(modelMeta(model)).map(storedFile))
    return stored.every(Boolean)
  },
  async size(model) {
    try {
      const all = await Promise.all(modelFiles(modelMeta(model)).map(fileSize))
      return all.reduce((a, b) => a + b, 0)
    } catch {
      return 0
    }
  },
  contextLength(model) {
    let length = contexts.get(model)
    if (!length) {
      length = readContextLength(modelMeta(model)).catch(() => {
        contexts.delete(model)
        return 0
      })
      contexts.set(model, length)
    }
    return length
  },
  async remove(model, keep) {
    let dir: FileSystemDirectoryHandle
    try {
      dir = await folder()
    } catch {
      return
    }
    const files = modelFiles(modelMeta(model)).filter((f) => !keep.has(fileKey(f)))
    const names = new Set<string>()
    for (const file of files) {
      const name = diskName(file)
      names.add(name).add(`${name}.done`).add(`${name}.etag`)
    }
    // And what earlier versions saved at a pinned commit.
    for await (const name of (dir as unknown as { keys: () => AsyncIterable<string> }).keys()) {
      if (files.some((f) => isOldVersion(name, f))) names.add(name)
    }
    // The marker first: a file without it never counts as downloaded.
    const ordered = [...names].sort((a, b) => Number(b.endsWith(".done")) - Number(a.endsWith(".done")))
    for (const name of ordered) await dir.removeEntry(name).catch(() => undefined)
    for (const file of files) sizes.delete(fileKey(file))
  }
}
