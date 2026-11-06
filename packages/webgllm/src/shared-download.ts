/**
 * One download of a file at a time per worker: a "download" request and a
 * load of the same model share it (two would contend for the file's
 * exclusive access handle). Each caller hears its progress, and it's
 * aborted only once every caller that joined it has aborted.
 */

type Progress = (loaded: number, total: number) => void
type Start<T> = (onProgress: Progress, signal: AbortSignal) => Promise<T>

type Running<T> = {
  promise: Promise<T>
  listeners: Set<Progress>
  abort: AbortController
  last?: [number, number]
}

export function createSharedDownloads<T>() {
  const running = new Map<string, Running<T>>()
  return function shared(key: string, start: Start<T>, onProgress: Progress, signal?: AbortSignal): Promise<T> {
    if (signal?.aborted) return Promise.reject(signal.reason as Error)
    let entry = running.get(key)
    if (!entry) {
      const abort = new AbortController()
      const listeners = new Set<Progress>()
      const created: Running<T> = {
        abort,
        listeners,
        promise: start((loaded, total) => {
          created.last = [loaded, total]
          for (const listener of listeners) listener(loaded, total)
        }, abort.signal)
      }
      entry = created
      running.set(key, created)
      const settle = () => {
        if (running.get(key) === created) running.delete(key)
      }
      created.promise.then(settle, settle)
    } else if (entry.last) onProgress(...entry.last)
    const joined = entry
    joined.listeners.add(onProgress)
    if (!signal) return joined.promise.finally(() => joined.listeners.delete(onProgress))
    return new Promise<T>((resolve, reject) => {
      const leave = () => {
        joined.listeners.delete(onProgress)
        signal.removeEventListener("abort", aborted)
      }
      const aborted = () => {
        leave()
        // The last caller gone: stop it, and let a new caller start afresh.
        if (!joined.listeners.size) {
          if (running.get(key) === joined) running.delete(key)
          joined.abort.abort(signal.reason)
        }
        reject(signal.reason)
      }
      signal.addEventListener("abort", aborted, { once: true })
      joined.promise.then(
        (value) => {
          leave()
          resolve(value)
        },
        (error: unknown) => {
          leave()
          reject(error)
        }
      )
    })
  }
}
