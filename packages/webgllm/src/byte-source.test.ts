import assert from "node:assert/strict"
import { test } from "node:test"
import { keptSource, withHandle } from "./byte-source.js"

/** A stored file whose synchronous access handles are exclusive and can be closed under their reader, as iOS closes a suspended page's. */
function storedFile(bytes: Uint8Array<ArrayBuffer>) {
  const state = { opened: 0, open: 0, syncReads: 0, suspend: () => {} }
  const handle = {
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
  const file = withHandle(new File([bytes], "model.gguf"), handle as unknown as FileSystemFileHandle, ["models"])
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

test("a kept source reads through its handle, through the Blob when the handle was closed under it, and opens another", () => inWorker(async () => {
  const bytes = Uint8Array.from({ length: 64 }, (_, i) => i)
  const { file, state } = storedFile(bytes)
  const kept = await keptSource(file)
  assert.deepEqual([...await kept.source.read(8, 4)], [8, 9, 10, 11])
  assert.deepEqual([state.opened, state.syncReads], [1, 1])
  state.suspend()
  // (The read that finds the handle closed still answers, from the Blob.)
  assert.deepEqual([...await kept.source.read(20, 3)], [20, 21, 22])
  assert.deepEqual([state.opened, state.open, state.syncReads], [2, 1, 1])
  assert.deepEqual([...await kept.source.read(40, 2)], [40, 41])
  assert.equal(state.syncReads, 2)
  await assert.rejects(kept.source.read(60, 8), RangeError)
  assert.equal(state.opened, 2)
  await kept.close()
  assert.equal(state.open, 0)
  // (Closed: the Blob, and no handle opened again.)
  assert.deepEqual([...await kept.source.read(0, 2)], [0, 1])
  assert.deepEqual([state.opened, state.open], [2, 0])
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
