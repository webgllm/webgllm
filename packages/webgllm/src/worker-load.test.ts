// The worker's loads against its generations: a new context length waits for
// the generation running, loads never overlap, and each "ready" names the
// load it answers. The worker's startLoad and message loop are extracted (as
// worker-generation.test.ts extracts generate), on a scripted load.
import assert from "node:assert/strict"
import { after, test } from "node:test"
import { mkdtemp, readFile, rm } from "node:fs/promises"
import { tmpdir } from "node:os"
import { join } from "node:path"
import { pathToFileURL } from "node:url"
import { createRequire } from "node:module"
import type { WorkerRequest, WorkerResponse } from "./protocol.js"

const require = createRequire(join(process.cwd(), "package.json"))
const { build } = await import(pathToFileURL(require.resolve("esbuild")).href) as typeof import("esbuild")
const temporary = await mkdtemp(join(tmpdir(), "webgl-worker-load-"))
after(() => rm(temporary, { recursive: true, force: true }))
const source = await readFile(join(process.cwd(), "src/worker.ts"), "utf8")
const loadStart = source.indexOf("function startLoad(")
const loadEnd = source.indexOf("\n/**\n * Downloads `model`'s files", loadStart)
const loopStart = source.indexOf("let queue: Promise<void>")
assert.ok(loadStart >= 0 && loadEnd > loadStart && loopStart > loadEnd)
const outfile = join(temporary, "loads.mjs")
await build({ stdin: { loader: "ts", resolveDir: join(process.cwd(), "src"), contents: `
  export function makeWorker(state) {
    let current = null, loading = null, pacingEngine = null
    const post = (m) => state.messages.push(m)
    const load = (model, signal, contextLength) => state.load(model, signal, contextLength)
    const release = (engine) => { engine.released = true }
    class Superseded extends Error { constructor() { super("Switched to another model.") } }
    const errorMessage = (e) => e instanceof Error ? e.message : String(e)
    const errorCode = () => undefined
    const stopped = new Set(), aborts = new Map(), claims = [], TUNING = {}
    const downloadModel = () => undefined
    async function generate(request) {
      const model = await startLoad(request.model, request.contextLength)
      await state.generate(model, request)
    }
    ${source.slice(loadStart, loadEnd)}
    ${source.slice(loopStart).replace("onmessage = (", "state.onmessage = (")}
  }
` }, outfile, bundle: true, format: "esm", platform: "node", logLevel: "error" })
const { makeWorker } = await import(pathToFileURL(outfile).href)

type Engine = { contextLength: number; released: boolean; gpu: { isLost: boolean } }
type Loaded = { model: string; contextLength: number | undefined; engine: Engine; cached: number[] }

function gate() {
  let release!: () => void
  const promise = new Promise<void>((resolve) => { release = resolve })
  return { promise, release }
}
const settle = async () => { for (let i = 0; i < 10; i++) await new Promise((r) => setTimeout(r, 0)) }

function worker() {
  const messages: WorkerResponse[] = []
  const loads: Array<{ contextLength: number | undefined; signal: AbortSignal; finish: () => void; engine: Engine }> = []
  const generations: Array<{ engine: Engine; finish: () => void; releasedMidway?: boolean }> = []
  let active = 0, most = 0
  const state = {
    messages,
    onmessage: null as unknown as (event: { data: WorkerRequest }) => void,
    async load(model: string, signal: AbortSignal, contextLength: number | undefined): Promise<Loaded> {
      active++
      most = Math.max(most, active)
      const done = gate()
      const engine: Engine = { contextLength: contextLength ?? 4096, released: false, gpu: { isLost: false } }
      loads.push({ contextLength, signal, finish: done.release, engine })
      try {
        await done.promise
      } finally {
        active--
      }
      return { model, contextLength, engine, cached: [] }
    },
    async generate(model: Loaded) {
      const done = gate()
      const entry = { engine: model.engine, finish: done.release, releasedMidway: false }
      generations.push(entry)
      await done.promise
      entry.releasedMidway = model.engine.released
    }
  }
  makeWorker(state)
  const send = (data: WorkerRequest) => state.onmessage({ data })
  return { send, messages, loads, generations, most: () => most }
}

const MODEL = "LiquidAI/LFM3-1.7B-GGUF"
const readies = (messages: WorkerResponse[]) => messages.filter((m) => m.type === "ready")
const generate = (id: string): WorkerRequest => ({
  type: "generate", id, model: MODEL, chat: [], tools: [], sampling: { temperature: 0, top_k: 1, repetition_penalty: 1 }
})

test("a new context length waits for the generation running before freeing its engine", async () => {
  const w = worker()
  w.send({ type: "load", model: MODEL, loadId: 1 })
  await settle()
  w.loads[0].finish()
  await settle()
  w.send(generate("g"))
  await settle()
  assert.equal(w.generations.length, 1)
  w.send({ type: "load", model: MODEL, contextLength: 64, loadId: 2 })
  await settle()
  assert.equal(w.loads[0].engine.released, false, "the reply's engine is freed mid-reply")
  assert.equal(w.loads.length, 1)
  w.generations[0].finish()
  await settle()
  assert.equal(w.generations[0].releasedMidway, false)
  assert.equal(w.loads.length, 2)
  w.loads[1].finish()
  await settle()
  assert.equal(w.loads[0].engine.released, true)
  assert.deepEqual(readies(w.messages).map((m) => m.type === "ready" && [m.loadId, m.contextLength]), [[1, 4096], [2, 64]])
})

test("loads never overlap: a superseded load ends before the next starts", async () => {
  const w = worker()
  w.send({ type: "load", model: MODEL, loadId: 1 })
  await settle()
  w.send({ type: "load", model: MODEL, contextLength: 64, loadId: 2 })
  await settle()
  assert.equal(w.loads[0].signal.aborted, true)
  assert.equal(w.most(), 1, "two loads (and their warm-ups) ran at once")
  w.loads[0].finish()
  await settle()
  assert.equal(w.loads[0].engine.released, true)
  assert.equal(w.loads.length, 2)
  w.loads[1].finish()
  await settle()
  assert.deepEqual(readies(w.messages).map((m) => m.type === "ready" && m.loadId), [2])
  assert.equal(w.most(), 1)
})

test("a load of what's loaded answers ready with its own id", async () => {
  const w = worker()
  w.send({ type: "load", model: MODEL, contextLength: 64, loadId: 1 })
  await settle()
  w.loads[0].finish()
  await settle()
  w.send({ type: "load", model: MODEL, contextLength: 64, loadId: 2 })
  await settle()
  assert.equal(w.loads.length, 1)
  assert.deepEqual(readies(w.messages).map((m) => m.type === "ready" && m.loadId), [1, 2])
})
