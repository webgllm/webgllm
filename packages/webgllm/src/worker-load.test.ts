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
const source = await readFile(join(process.cwd(), "src/handler.ts"), "utf8")
const loadStart = source.indexOf("function startLoad(")
const loadEnd = source.indexOf("\n/**\n * Downloads `model`'s files", loadStart)
const loopStart = source.indexOf("let queue: Promise<void>")
const routeStart = source.indexOf("const REQUESTS: ReadonlySet<string>")
const routeEnd = source.indexOf("/** Claims the page hasn't answered yet, by id. */")
assert.ok(routeStart >= 0 && routeEnd > routeStart)
const claimsStart = source.indexOf("/** Claims the page hasn't answered yet, by id. */")
const claimsEnd = source.indexOf("/** A claim again once the textures grow", claimsStart)
assert.ok(loadStart >= 0 && loadEnd > loadStart && loopStart > loadEnd && claimsStart >= 0 && claimsEnd > claimsStart)
const outfile = join(temporary, "loads.mjs")
await build({ stdin: { loader: "ts", resolveDir: join(process.cwd(), "src"), contents: `
  export function makeWorker(state) {
    let current = null, loading = null, stopping, pacingEngine = null
    const post = (m) => state.messages.push(m)
    const load = (model, signal, contextLength) => state.load(model, signal, contextLength, claimMemory)
    const release = (engine) => { engine.released = true }
    class Superseded extends Error { constructor() { super("Switched to another model.") } }
    const errorMessage = (e) => e instanceof Error ? e.message : String(e)
    const errorCode = () => undefined
    const stopped = new Set(), aborts = new Map(), TUNING = {}
    ${source.slice(claimsStart, claimsEnd)}
    const downloadModel = () => undefined
    async function generate(request) {
      const model = await startLoad(request.model, request.contextLength)
      const abort = aborts.get(request.id) ?? new AbortController()
      aborts.set(request.id, abort)
      await state.generate(model, request, claimMemory, abort.signal)
    }
    ${source.slice(loadStart, loadEnd)}
    ${source.slice(loopStart)}
    ${source.slice(routeStart, routeEnd)}
    state.onmessage = ({ data }) => { if (isRequest(data)) receive(data) }
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

test("unload frees the loaded model after the generation running, and the worker loads again", async () => {
  const w = worker()
  w.send({ type: "load", model: MODEL, loadId: 1 })
  await settle()
  w.loads[0].finish()
  await settle()
  w.send(generate("g"))
  await settle()
  w.send({ type: "unload", model: MODEL, id: 9 })
  await settle()
  assert.equal(w.loads[0].engine.released, false, "freed under the reply running")
  assert.equal(w.messages.some((m) => m.type === "unloaded"), false, "said it was freed while the reply held it")
  w.generations[0].finish()
  await settle()
  assert.equal(w.loads[0].engine.released, true)
  assert.deepEqual(w.messages.filter((m) => m.type === "unloaded"), [{ type: "unloaded", id: 9 }])
  // Still running: the next load starts afresh.
  w.send({ type: "load", model: MODEL, loadId: 2 })
  await settle()
  assert.equal(w.loads.length, 2)
  w.loads[1].finish()
  await settle()
  assert.deepEqual(readies(w.messages).map((m) => m.type === "ready" && m.loadId), [1, 2])
  assert.equal(w.loads[1].engine.released, false)
})

test("unload stops a load in progress, without an answer; the next load waits for it to end", async () => {
  const w = worker()
  w.send({ type: "load", model: MODEL, loadId: 1 })
  await settle()
  w.send({ type: "unload", model: MODEL, id: 9 })
  await settle()
  assert.equal(w.loads[0].signal.aborted, true)
  w.send({ type: "load", model: "unsloth/Qwen3.5-2B-GGUF", loadId: 2 })
  await settle()
  assert.equal(w.loads.length, 1, "two loads ran at once")
  assert.equal(w.messages.some((m) => m.type === "unloaded"), false, "said it was freed while its load still ran")
  w.loads[0].finish()
  await settle()
  assert.equal(w.loads[0].engine.released, true)
  assert.equal(w.messages.some((m) => m.type === "unloaded"), true)
  assert.equal(w.loads.length, 2)
  w.loads[1].finish()
  await settle()
  assert.equal(w.most(), 1)
  assert.deepEqual(w.messages.filter((m) => m.type === "ready" || m.type === "load-error").map((m) => m.type === "ready" && m.loadId), [2])
})

test("unload of another model leaves the loaded one", async () => {
  const w = worker()
  w.send({ type: "load", model: MODEL, loadId: 1 })
  await settle()
  w.loads[0].finish()
  await settle()
  w.send({ type: "unload", model: "unsloth/Qwen3.5-2B-GGUF", id: 9 })
  await settle()
  assert.equal(w.loads[0].engine.released, false)
})

// The page's memory claims against an unload (a worker of the caller's): each
// claim has an id and belongs to a load or a generation, so one made for a
// model that's leaving can't hold up the next model's.
type Claim = (model: string, bytes: number, signal?: AbortSignal) => Promise<void>
const OTHER = "unsloth/Qwen3.5-2B-GGUF"
const claimsOf = (messages: WorkerResponse[]) => messages.filter((m): m is Extract<WorkerResponse, { type: "claim" }> => m.type === "claim")

/** A worker whose loads fetch files (held by `files`), then claim memory as the real load does. */
function claiming() {
  const messages: WorkerResponse[] = []
  const files = new Map<string, ReturnType<typeof gate>>()
  const generations: Array<{ claimed: ReturnType<typeof gate>; cleaned: ReturnType<typeof gate> }> = []
  const state = {
    messages,
    onmessage: null as unknown as (event: { data: WorkerRequest }) => void,
    async load(model: string, signal: AbortSignal, contextLength: number | undefined, claimMemory: Claim): Promise<Loaded> {
      await files.get(model)?.promise
      await claimMemory(model, 10, signal)
      if (signal.aborted) throw new Error("stopped")
      return { model, contextLength, engine: { contextLength: 4096, released: false, gpu: { isLost: false } }, cached: [] }
    },
    // A generation with an image: a claim for the vision encoder, then one for the model alone once it's freed.
    async generate(model: Loaded, _request: unknown, claimMemory: Claim, signal: AbortSignal) {
      const entry = { claimed: gate(), cleaned: gate() }
      generations.push(entry)
      try {
        await claimMemory(model.model, 100, signal)
        entry.claimed.release()
        signal.throwIfAborted()
      } finally {
        void claimMemory(model.model, 10)
        entry.cleaned.release()
      }
    }
  }
  makeWorker(state)
  const send = (data: WorkerRequest) => state.onmessage({ data })
  /** The page's answer to the last claim made for `model`. */
  const answer = (model: string) => send({ type: "claimed", id: claimsOf(messages).filter((m) => m.model === model).at(-1)!.id })
  return { send, answer, messages, files, generations }
}

test("a load that returns from fetching files after its model was unloaded claims nothing, and the next load isn't held up", async () => {
  const w = claiming()
  w.files.set(MODEL, gate())
  w.send({ type: "load", model: MODEL, loadId: 1 })
  await settle()
  w.send({ type: "unload", model: MODEL, id: 2 })
  w.send({ type: "load", model: OTHER, loadId: 3 })
  await settle()
  // The stopped load comes back from its files only now.
  w.files.get(MODEL)!.release()
  await settle()
  assert.deepEqual(claimsOf(w.messages).map((m) => m.model), [OTHER], "the unloaded model claimed, or the next load never reached its claim")
  w.answer(OTHER)
  await settle()
  assert.deepEqual(readies(w.messages).map((m) => m.type === "ready" && m.loadId), [3])
  assert.equal(w.messages.some((m) => m.type === "unloaded" && m.id === 2), true)
})

test("an unload while a generation waits for its vision claim: the next model's answer reaches its own claim", async () => {
  const w = claiming()
  w.send({ type: "load", model: MODEL, loadId: 1 })
  await settle()
  w.answer(MODEL)
  await settle()
  w.send(generate("g"))
  await settle()
  assert.equal(claimsOf(w.messages).at(-1)!.bytes, 100)
  // The page stops the generation and unloads the model, then loads another.
  w.send({ type: "stop", id: "g" })
  w.send({ type: "unload", model: MODEL, id: 2 })
  w.send({ type: "load", model: OTHER, loadId: 3 })
  await w.generations[0].cleaned.promise
  await settle()
  assert.deepEqual(claimsOf(w.messages).map((m) => m.model), [MODEL, MODEL, OTHER], "the unloaded model's cleanup claimed again")
  w.answer(OTHER)
  await settle()
  assert.deepEqual(readies(w.messages).map((m) => m.type === "ready" && m.loadId), [1, 3], "the next model's answer went to another claim")
})

test("the handler takes only the page's requests: the worker's own messages change nothing", async () => {
  const w = worker()
  w.send({ type: "load", model: MODEL, loadId: 1 })
  await settle()
  w.loads[0].finish()
  await settle()
  for (const data of ["ping", 7, null, { type: "ping", model: MODEL }, { type: "generate", model: MODEL }, { type: "unload", model: MODEL }, { kind: "mine" }])
    w.send(data as never)
  await settle()
  assert.equal(w.loads[0].engine.released, false)
  assert.equal(w.generations.length, 0)
  assert.equal(w.loads.length, 1)
})
