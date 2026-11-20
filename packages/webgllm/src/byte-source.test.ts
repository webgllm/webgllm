import assert from "node:assert/strict"
import { test } from "node:test"
import { keptSource, withHandle } from "./byte-source.js"

/**
 * A stored file whose synchronous access handles are exclusive and can be
 * closed under their reader, as iOS closes a suspended page's; `stale`
 * also makes the File taken before then unreadable, as Safari's is.
 */
function storedFile(bytes: Uint8Array<ArrayBuffer>) {
  const state = { opened: 0, open: 0, syncReads: 0, blobReads: 0, fetched: 0, stale: false, suspend: () => {} }
  const real = new File([bytes], "model.gguf")
  const handle = {
    getFile: async () => {
      state.fetched++
      return new File([bytes], "model.gguf")
    },
    createSyncAccessHandle: async () => {
      if (state.open > 0) throw new DOMException("locked", "NoModificationAllowedError")
      state.opened++
      state.open++
      let alive = true
      state.suspend = () => { alive = false }
      return {
        getSize: () => bytes.byteLength,
        read: (into: Uint8Array, { at }: { at: number }) => {
          if (!alive) throw new DOMException("closed", "InvalidStateError")
          state.syncReads++
          into.set(bytes.subarray(at, at + into.byteLength))
          return into.byteLength
        },
        close: () => {
          if (alive) alive = false
          state.open--
        }
      }
    }
  }
  // The File taken at the load: readable until it's stale.
  const taken = {
    size: real.size,
    slice: (start: number, end: number) => ({
      arrayBuffer: async () => {
        if (state.stale) throw new DOMException("The object can not be found here.", "NotFoundError")
        state.blobReads++
        return real.slice(start, end).arrayBuffer()
      }
    })
  } as unknown as File
  const file = withHandle(taken, handle as unknown as FileSystemFileHandle, ["models"])
  return { file, state }
}

const inWorker = async (run: () => Promise<void>) => {
  const scope = globalThis as { WorkerGlobalScope?: unknown }
  const before = scope.WorkerGlobalScope
  scope.WorkerGlobalScope = class {}
  try {
    await run()
  } finally {
    scope.WorkerGlobalScope = before
  }
}

test("a kept source reads through its handle, and through another opened in its place when the handle was closed under it", () => inWorker(async () => {
  const bytes = Uint8Array.from({ length: 64 }, (_, i) => i)
  const { file, state } = storedFile(bytes)
  const lost: Array<string> = []
  const kept = await keptSource(file, (error) => lost.push((error as Error).name))
  assert.deepEqual([...await kept.source.read(8, 4)], [8, 9, 10, 11])
  assert.deepEqual([state.opened, state.syncReads], [1, 1])
  state.suspend()
  // (The read that finds the handle closed is answered by one opened anew, and says so.)
  assert.deepEqual([...await kept.source.read(20, 3)], [20, 21, 22])
  assert.deepEqual([state.opened, state.open, state.syncReads], [2, 1, 2])
  assert.deepEqual(lost, ["InvalidStateError"])
  assert.deepEqual([...await kept.source.read(40, 2)], [40, 41])
  assert.equal(state.syncReads, 3)
  await assert.rejects(kept.source.read(60, 8), RangeError)
  assert.equal(state.opened, 2)
  await kept.close()
  assert.equal(state.open, 0)
  // (Closed: the file as stored, and no handle opened again.)
  assert.deepEqual([...await kept.source.read(0, 2)], [0, 1])
  assert.deepEqual([state.opened, state.open], [2, 0])
}))

test("a kept source whose page was suspended reads through a handle opened anew, the File taken at the load being stale too", () => inWorker(async () => {
  const bytes = Uint8Array.from({ length: 64 }, (_, i) => 200 - i)
  const { file, state } = storedFile(bytes)
  const lost: Array<string> = []
  const kept = await keptSource(file, (error) => lost.push((error as Error).name))
  assert.deepEqual([...await kept.source.read(0, 2)], [200, 199])
  state.suspend()
  state.stale = true
  // The same read is answered by the new handle: the stale File is never asked.
  assert.deepEqual([...await kept.source.read(10, 3)], [190, 189, 188])
  assert.deepEqual([state.opened, state.open, state.syncReads, state.blobReads, lost.length], [2, 1, 2, 0, 1])
  // No handle to be had (another reader holds the file) and the taken File stale: the file as stored now.
  state.suspend()
  state.open = 2
  assert.deepEqual([...await kept.source.read(20, 2)], [180, 179])
  assert.ok(state.fetched >= 1)
  assert.equal(state.blobReads, 0)
  await kept.close()
}))

test("a kept source whose file is no longer stored says so", () => inWorker(async () => {
  const { file, state } = storedFile(Uint8Array.from({ length: 16 }, (_, i) => i))
  const gone = () => { throw new DOMException("The object can not be found here.", "NotFoundError") }
  const handles = (await import("./byte-source.js")) as unknown as { withHandle: typeof withHandle }
  // Its handle opens no access handle and gives no file: the storage was deleted under the model.
  handles.withHandle(file, { createSyncAccessHandle: async () => gone(), getFile: async () => gone() } as unknown as FileSystemFileHandle, ["models"])
  state.stale = true
  const kept = await keptSource(file)
  await assert.rejects(kept.source.read(0, 4), /stored file can no longer be read \(The object can not be found here\.\): load the model again/)
  await kept.close()
}))

test("a kept source of a file another reader holds reads its Blob", () => inWorker(async () => {
  const bytes = Uint8Array.from({ length: 16 }, (_, i) => 100 + i)
  const { file, state } = storedFile(bytes)
  state.open = 1
  const kept = await keptSource(file)
  assert.deepEqual([...await kept.source.read(2, 3)], [102, 103, 104])
  assert.deepEqual([state.opened, state.syncReads], [0, 0])
  await kept.close()
}))
