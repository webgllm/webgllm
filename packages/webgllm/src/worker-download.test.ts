// The worker's file downloads against a stop: fetchFiles (extracted, as
// worker-load.test.ts extracts startLoad) with the real shared downloads,
// on writers a test opens and closes. It settles only once every file
// operation of its own has ended, so the page may delete the files then.
import assert from "node:assert/strict"
import { after, test } from "node:test"
import { mkdtemp, readFile, rm } from "node:fs/promises"
import { tmpdir } from "node:os"
import { join } from "node:path"
import { pathToFileURL } from "node:url"
import { createRequire } from "node:module"

const require = createRequire(join(process.cwd(), "package.json"))
const { build } = await import(pathToFileURL(require.resolve("esbuild")).href) as typeof import("esbuild")
const temporary = await mkdtemp(join(tmpdir(), "webgl-worker-download-"))
after(() => rm(temporary, { recursive: true, force: true }))
const source = await readFile(join(process.cwd(), "src/handler.ts"), "utf8")
const start = source.indexOf("async function fetchFiles(")
const end = source.indexOf("\n/** Attention's largest relative error", start)
assert.ok(start >= 0 && end > start)
const outfile = join(temporary, "download.mjs")
await build({ stdin: { loader: "ts", resolveDir: join(process.cwd(), "src"), contents: `
  import { createSharedDownloads } from './shared-download.js';
  import { downloadError } from './worker-errors.js';
  export function makeFetchFiles(state) {
    const downloads = createSharedDownloads();
    const post = (m) => state.messages.push(m);
    const resolveModel = async () => state.meta, storedFile = async () => undefined, saveManifest = async () => undefined;
    const download = (file, onProgress, signal) => state.download(file, onProgress, signal);
    ${source.slice(start, end)}
    return fetchFiles;
  }
` }, outfile, bundle: true, format: "esm", platform: "node", logLevel: "error" })
const { makeFetchFiles } = await import(pathToFileURL(outfile).href)

const settle = async () => { for (let i = 0; i < 10; i++) await new Promise((r) => setTimeout(r, 0)) }
const file = (path: string) => ({ repo: "org/model", path })

/** Files whose writers a test finishes, fails and closes: an aborted writer rejects only once it's closed. */
function writers() {
  const open = new Map<string, { signal: AbortSignal; finish: () => void; fail: (error: Error) => void; close: () => void; closed: boolean }>()
  const state = {
    messages: [] as unknown[],
    meta: { files: { kind: "gguf", model: file("weights.gguf") }, tokenizer: { tokenizer: file("tokenizer.json"), config: file("tokenizer_config.json") } },
    download: (f: { path: string }, _onProgress: unknown, signal: AbortSignal) =>
      new Promise((resolve, reject) => {
        const writer = {
          signal,
          closed: false,
          finish: () => { writer.closed = true; resolve({ name: f.path }) },
          fail: (error: Error) => { writer.closed = true; reject(error) },
          // The writer's cleanup after its signal aborted.
          close: () => { writer.closed = true; reject(signal.reason) }
        }
        open.set(f.path, writer)
      })
  }
  const fetchFiles = makeFetchFiles(state) as (model: string, signal: AbortSignal, share?: number) => Promise<unknown>
  return { fetchFiles, open }
}

test("a stopped download settles only once every one of its files' writers has closed", async () => {
  const w = writers()
  const abort = new AbortController()
  let settled = false
  const fetching = w.fetchFiles("org/model", abort.signal).then(() => "done", (error: Error) => error).finally(() => { settled = true })
  await settle()
  assert.equal(w.open.size, 3)
  abort.abort(new Error("stopped"))
  await settle()
  assert.ok([...w.open.values()].every((writer) => writer.signal.aborted))
  // The tokenizer's writers close first: the weights' is still open.
  w.open.get("tokenizer.json")!.close()
  w.open.get("tokenizer_config.json")!.close()
  await settle()
  assert.equal(settled, false, "settled while the weights' writer was still open")
  w.open.get("weights.gguf")!.close()
  const error = await fetching
  assert.equal((error as Error).message, "stopped")
})

test("a file that fails stops the request's other files, and it settles once they've closed", async () => {
  const w = writers()
  let settled = false
  const fetching = w.fetchFiles("org/model", new AbortController().signal).then(() => "done", (error: Error & { code?: string }) => error).finally(() => { settled = true })
  await settle()
  w.open.get("tokenizer.json")!.fail(new Error("404"))
  await settle()
  assert.equal(w.open.get("weights.gguf")!.signal.aborted, true, "the weights went on downloading for nothing")
  assert.equal(settled, false, "settled while the other writers were still open")
  w.open.get("tokenizer_config.json")!.close()
  w.open.get("weights.gguf")!.close()
  const error = await fetching as Error & { code?: string }
  assert.equal(error.code, "download-failed")
  assert.match(error.message, /404/)
})

test("a download another request still needs goes on when one request stops", async () => {
  const w = writers()
  const first = new AbortController()
  const stopped = w.fetchFiles("org/model", first.signal).then(() => "done", (error: Error) => error.message)
  await settle()
  // A load of the same model joins the same three downloads.
  const other = w.fetchFiles("org/model", new AbortController().signal)
  await settle()
  first.abort(new Error("stopped"))
  assert.equal(await stopped, "stopped")
  assert.ok([...w.open.values()].every((writer) => !writer.signal.aborted), "the shared downloads were stopped")
  for (const writer of w.open.values()) writer.finish()
  const fetched = await other as { weightFiles: Array<{ name: string }> }
  assert.deepEqual(fetched.weightFiles.map((f) => f.name), ["weights.gguf"])
})
