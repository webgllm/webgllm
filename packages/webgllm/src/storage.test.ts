import assert from "node:assert/strict"
import { describe, test } from "node:test"
import { diskName, download, isOldVersion, previousGpuTuning, readGpuNotes, writeGpuNote, type GpuNote } from "./storage.js"

const repo = "onnx-community/LFM3-1.7B-ONNX"
const commit = "ac2586a34f1a1c03329dbdcdefabd8b6478ddaa7"
const config = { repo, path: "config.json" }

describe("isOldVersion", () => {
  test("a file and its markers saved at a pinned commit", () => {
    const old = `onnx-community_LFM3-1.7B-ONNX@${commit}_config.json`
    assert.ok(isOldVersion(old, config))
    assert.ok(isOldVersion(`${old}.done`, config))
    assert.ok(isOldVersion(`${old}.etag`, config))
  })

  test("not the file at main", () => {
    assert.ok(!isOldVersion("onnx-community_LFM3-1.7B-ONNX@main_config.json", config))
  })

  test("not another file whose name ends the same", () => {
    // config.json's cleanup deleted a fresh tokenizer_config.json.
    for (const name of [
      "onnx-community_LFM3-1.7B-ONNX@main_tokenizer_config.json",
      "onnx-community_LFM3-1.7B-ONNX@main_tokenizer_config.json.done",
      `onnx-community_LFM3-1.7B-ONNX@${commit}_tokenizer_config.json`
    ]) {
      assert.ok(!isOldVersion(name, config), name)
    }
  })

  test("not another repo's file", () => {
    const other = `onnx-community_LFM3-1.7B-ONNX-Web@${commit}_config.json`
    assert.ok(!isOldVersion(other, config))
  })

  test("a file in a folder", () => {
    const data = { repo, path: "onnx/model_q4f16.onnx_data" }
    const graph = { repo, path: "onnx/model_q4f16.onnx" }
    const old = `onnx-community_LFM3-1.7B-ONNX@${commit}_onnx_model_q4f16.onnx_data`
    assert.ok(isOldVersion(old, data))
    assert.ok(!isOldVersion(old, graph))
  })
})

test("a new engine can reuse checked tuning from the same GPU, model and settings", () => {
  const scope = "gpu|browser|model|mrt|"
  const note: GpuNote = {
    layout: "bytes",
    calibrated: true,
    selfCheck: 0.001,
    tuned: { shortSplit: null, f16Pack: 0 }
  }
  const notes = {
    "gpu|browser|other|mrt|older.settings": note,
    [`${scope}older.other-settings`]: note,
    [`${scope}older.settings`]: { ...note, selfCheck: 0.05 },
    [`${scope}previous.settings`]: note
  }
  assert.equal(previousGpuTuning(notes, scope, "settings", 0.01), note)
  assert.equal(previousGpuTuning(notes, scope, "missing", 0.01), undefined)
  assert.equal(previousGpuTuning(notes, scope, "settings", 0.0001), undefined)
})

test("locked calibration writes stop after one retry and release the next queued write", async (t) => {
  const note: GpuNote = { layout: "bytes", selfCheck: 0 }
  const latest = { ...note, calibrated: true }
  const key = "gpu|browser|model|code"
  let opens = 0
  let saved = ""
  const handle = {
    getFile: async () => new Blob(["{}"]),
    async createSyncAccessHandle() {
      if (++opens <= 2) throw new DOMException("locked", "NoModificationAllowedError")
      return {
        getSize: () => 2,
        read: (buffer: Uint8Array) => { buffer.set(new TextEncoder().encode("{}")); return 2 },
        truncate: () => undefined,
        write: (bytes: Uint8Array) => { saved = new TextDecoder().decode(bytes) },
        flush: () => undefined,
        close: () => undefined
      }
    }
  }
  t.mock.getter(globalThis, "navigator", () => ({
    storage: {
      getDirectory: async () => ({
        getDirectoryHandle: async () => ({ getFileHandle: async () => handle })
      })
    }
  }))
  const abandoned = writeGpuNote(key, note)
  const following = writeGpuNote(key, latest)
  await Promise.all([abandoned, following])
  assert.equal(opens, 3)
  assert.deepEqual(JSON.parse(saved), { [key]: latest })
})

/** An OPFS folder in memory, with exclusive sync access handles. */
function fakeFolder() {
  const files = new Map<string, Uint8Array<ArrayBuffer>>()
  const locked = new Set<string>()
  let beforeLock: ((name: string) => void) | undefined
  const fileHandle = (name: string) => ({
    getFile: async () => new File([files.get(name) ?? new Uint8Array()], name),
    async createSyncAccessHandle() {
      beforeLock?.(name)
      if (locked.has(name)) throw new DOMException("locked", "NoModificationAllowedError")
      locked.add(name)
      const bytes = () => files.get(name) ?? new Uint8Array()
      return {
        getSize: () => bytes().byteLength,
        read: (buffer: Uint8Array, { at }: { at: number }) => {
          const part = bytes().subarray(at, at + buffer.byteLength)
          buffer.set(part)
          return part.byteLength
        },
        truncate: (size: number) => { files.set(name, bytes().slice(0, size)) },
        write: (data: Uint8Array, { at }: { at: number }) => {
          const old = bytes()
          const next = new Uint8Array(Math.max(old.byteLength, at + data.byteLength))
          next.set(old)
          next.set(data, at)
          files.set(name, next)
          return data.byteLength
        },
        flush: () => undefined,
        close: () => { locked.delete(name) }
      }
    }
  })
  const folder = {
    async getFileHandle(name: string, options?: { create?: boolean }) {
      if (!files.has(name)) {
        if (!options?.create) throw new DOMException("missing", "NotFoundError")
        files.set(name, new Uint8Array())
      }
      return fileHandle(name)
    },
    async removeEntry(name: string) { files.delete(name) },
    async *keys() { yield* [...files.keys()] }
  }
  return {
    files,
    folder,
    text: (name: string) => new TextDecoder().decode(files.get(name)),
    put: (name: string, text: string) => { files.set(name, new TextEncoder().encode(text)) },
    /** Runs `fn` each time someone asks for a file's access handle, just before it's granted. */
    onLock: (fn: (name: string) => void) => { beforeLock = fn }
  }
}

function useFolder(t: { mock: { getter: (o: object, k: string, f: () => unknown) => unknown } }) {
  const fake = fakeFolder()
  t.mock.getter(globalThis, "navigator", () => ({
    storage: { getDirectory: async () => ({ getDirectoryHandle: async () => fake.folder }) }
  }))
  return fake
}

type Served = { status: number; body?: string; headers?: Record<string, string | undefined> }

/** A fetch answering each request with `serve`, recording what was asked. */
function useFetch(t: { mock: { method: (o: object, k: string, f: unknown) => unknown } }, serve: (headers: Record<string, string>) => Served) {
  const asked: Array<Record<string, string>> = []
  t.mock.method(globalThis, "fetch", async (_url: string, init: { headers: Record<string, string> }) => {
    asked.push({ ...init.headers })
    const { status, body, headers } = serve(init.headers)
    return new Response(status === 416 ? null : body ?? "", { status, headers: headers as Record<string, string> })
  })
  return asked
}

/** A response body that drops after `text`. */
function dropsAfter(text: string) {
  let sent = false
  return new ReadableStream<Uint8Array>({
    pull(c) {
      if (sent) return c.error(new Error("dropped"))
      sent = true
      c.enqueue(new TextEncoder().encode(text))
    }
  })
}

describe("download", () => {
  const file = { repo, path: "model.gguf" }
  const name = diskName(file)
  const etag = `${name}.etag`

  test("a 416 for another size starts over instead of keeping the stale partial", async (t) => {
    const fake = useFolder(t)
    fake.put(name, "hello world")
    fake.put(etag, '"v1"')
    const asked = useFetch(t, (h) => h.Range
      ? { status: 416, headers: { "Content-Range": "bytes */5" } }
      : { status: 200, body: "abcde", headers: { ETag: '"v2"', "Content-Length": "5" } })
    const got = await download(file, () => undefined)
    assert.equal(await got.text(), "abcde")
    assert.equal(asked.length, 2)
    assert.equal(asked[1].Range, undefined)
  })

  test("a 416 for exactly the bytes on disk completes it", async (t) => {
    const fake = useFolder(t)
    fake.put(name, "abcde")
    fake.put(etag, '"v1"')
    const asked = useFetch(t, () => ({ status: 416, headers: { "Content-Range": "bytes */5" } }))
    const got = await download(file, () => undefined)
    assert.equal(await got.text(), "abcde")
    assert.equal(asked.length, 1)
    assert.deepEqual(asked[0], { Range: "bytes=5-", "If-Range": '"v1"' })
  })

  test("a partial with no version to check isn't joined to a newer file", async (t) => {
    const fake = useFolder(t)
    fake.put(name, "abc")
    // The file changed on the server; with no If-Range it serves the range anyway.
    useFetch(t, (h) => h.Range === "bytes=3-"
      ? { status: 206, body: "WV", headers: { "Content-Range": "bytes 3-4/5" } }
      : { status: 200, body: "XYZWV", headers: { "Content-Length": "5" } })
    const got = await download(file, () => undefined)
    assert.equal(await got.text(), "XYZWV")
  })

  test("a 206 without a Content-Range starts over", async (t) => {
    const fake = useFolder(t)
    fake.put(name, "abc")
    fake.put(etag, '"v1"')
    useFetch(t, (h) => h.Range
      ? { status: 206, body: "WV" }
      : { status: 200, body: "XYZWV", headers: { ETag: '"v1"', "Content-Length": "5" } })
    const got = await download(file, () => undefined)
    assert.equal(await got.text(), "XYZWV")
  })

  test("a 206 for another total starts over", async (t) => {
    useFolder(t)
    let first = true
    t.mock.method(globalThis, "fetch", async (_url: string, init: { headers: Record<string, string> }) => {
      if (first) {
        first = false
        // A full answer (5 bytes) that drops after 3.
        const body = dropsAfter("XYZ")
        return new Response(body, { status: 200, headers: { ETag: '"v1"', "Content-Length": "5" } })
      }
      return init.headers.Range
        ? new Response("WVUT", { status: 206, headers: { "Content-Range": "bytes 3-6/7" } })
        : new Response("XYZWV", { status: 200, headers: { ETag: '"v1"', "Content-Length": "5" } })
    })
    const got = await download(file, () => undefined)
    assert.equal(await got.text(), "XYZWV")
  })

  test("a drop with no version to resume from asks for the whole file again", async (t) => {
    useFolder(t)
    const asked: Array<Record<string, string>> = []
    t.mock.method(globalThis, "fetch", async (_url: string, init: { headers: Record<string, string> }) => {
      asked.push({ ...init.headers })
      return asked.length === 1
        ? new Response(dropsAfter("ab"), { status: 200, headers: { "Content-Length": "5" } })
        : new Response("abcde", { status: 200, headers: { "Content-Length": "5" } })
    })
    const got = await download(file, () => undefined)
    assert.equal(await got.text(), "abcde")
    assert.deepEqual(asked[1], {})
  })

  test("resumes with Last-Modified when the server sends no ETag", async (t) => {
    useFolder(t)
    const modified = "Wed, 21 Oct 2026 07:28:00 GMT"
    const asked: Array<Record<string, string>> = []
    t.mock.method(globalThis, "fetch", async (_url: string, init: { headers: Record<string, string> }) => {
      asked.push({ ...init.headers })
      if (asked.length === 1) {
        const body = dropsAfter("ab")
        return new Response(body, { status: 200, headers: { "Last-Modified": modified, "Content-Length": "5" } })
      }
      return new Response("cde", { status: 206, headers: { "Content-Range": "bytes 2-4/5" } })
    })
    const got = await download(file, () => undefined)
    assert.equal(await got.text(), "abcde")
    assert.deepEqual(asked[1], { Range: "bytes=2-", "If-Range": modified })
  })

  test("drops with no version to keep their bytes still run out of tries", async (t) => {
    useFolder(t)
    // No waiting between tries.
    t.mock.method(globalThis, "setTimeout", (fn: () => void) => { queueMicrotask(fn); return 0 })
    let asked = 0
    t.mock.method(globalThis, "fetch", async () => {
      // (A stop for the loop if it never gives up.)
      if (++asked > 20) throw new Error("refused")
      return new Response(dropsAfter("ab"), { status: 200, headers: { "Content-Length": "5" } })
    })
    await assert.rejects(download(file, () => undefined), /dropped/)
    assert.equal(asked, 6)
  })

  test("a second download that waited for the lock leaves the finished file whole", async (t) => {
    const fake = useFolder(t)
    let finish!: () => void
    let fetched!: () => void
    const streaming = new Promise<void>((resolve) => { fetched = resolve })
    let firstLocked!: () => void
    const locked = new Promise<void>((resolve) => { firstLocked = resolve })
    let locks = 0
    // The second download asks for the file's lock while the first holds it.
    fake.onLock((n) => { if (n === name && ++locks === 2) firstLocked() })
    const second = new AbortController()
    let asked = 0
    t.mock.method(globalThis, "fetch", async () => {
      if (++asked === 1) {
        const body = new ReadableStream<Uint8Array>({
          start(c) {
            finish = () => {
              c.enqueue(new TextEncoder().encode("abcde"))
              c.close()
            }
            fetched()
          }
        })
        return new Response(body, { status: 200, headers: { ETag: '"v1"', "Content-Length": "5" } })
      }
      // The second download fetching again: canceled.
      second.abort(new Error("canceled"))
      throw second.signal.reason
    })
    const a = download(file, () => undefined)
    const b = download(file, () => undefined, second.signal)
    await Promise.all([locked, streaming])
    finish()
    assert.equal(await (await a).text(), "abcde")
    assert.equal(await (await b).text(), "abcde")
    assert.equal(asked, 1)
    assert.equal(fake.text(name), "abcde")
  })
})

test("a note written by another worker during a write isn't lost", async (t) => {
  const fake = useFolder(t)
  const other: GpuNote = { layout: "rgba4", selfCheck: 0 }
  const mine: GpuNote = { layout: "bytes", selfCheck: 0 }
  let raced = false
  // Another worker (its own write queue) writes its note just before this
  // one gets the file's lock: it only can if the file isn't locked.
  fake.onLock((name) => {
    if (raced || name !== "gpu-notes.json") return
    raced = true
    const notes = JSON.parse(fake.text(name) || "{}")
    notes["gpu|browser|other-model|code"] = other
    fake.put(name, JSON.stringify(notes))
  })
  await writeGpuNote("gpu|browser|model|code", mine)
  assert.deepEqual(await readGpuNotes(), {
    "gpu|browser|other-model|code": other,
    "gpu|browser|model|code": mine
  })
})
