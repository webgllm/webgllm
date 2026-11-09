/** An Origin Private File System folder in memory, for tests of what's stored. */

/** A folder with exclusive sync access handles, as OPFS gives workers. */
export function fakeFolder() {
  const files = new Map<string, Uint8Array<ArrayBuffer>>()
  const locked = new Set<string>()
  const fileHandle = (name: string) => ({
    getFile: async () => new File([files.get(name) ?? new Uint8Array()], name),
    async createSyncAccessHandle() {
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
        truncate: (size: number) => {
          files.set(name, bytes().slice(0, size))
        },
        write: (data: Uint8Array, { at }: { at: number }) => {
          const old = bytes()
          const next = new Uint8Array(Math.max(old.byteLength, at + data.byteLength))
          next.set(old)
          next.set(data, at)
          files.set(name, next)
          return data.byteLength
        },
        flush: () => undefined,
        close: () => {
          locked.delete(name)
        }
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
    async removeEntry(name: string) {
      if (!files.delete(name)) throw new DOMException("missing", "NotFoundError")
    },
    async *keys() {
      yield* [...files.keys()]
    }
  }
  return {
    files,
    folder,
    put: (name: string, text: string) => {
      files.set(name, new TextEncoder().encode(text))
    }
  }
}

/** `navigator.storage` with a fake folder for the test (restored after it). */
export function useFolder(t: { mock: { getter: (o: object, k: string, f: () => unknown) => unknown } }) {
  const fake = fakeFolder()
  t.mock.getter(globalThis, "navigator", () => ({
    storage: { getDirectory: async () => ({ getDirectoryHandle: async () => fake.folder }) }
  }))
  return fake
}
