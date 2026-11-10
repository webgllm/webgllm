import assert from "node:assert/strict"
import { after, test } from "node:test"
import { mkdtemp, readFile, rm } from "node:fs/promises"
import { tmpdir } from "node:os"
import { join } from "node:path"
import { pathToFileURL } from "node:url"
import { createRequire } from "node:module"
import type { WorkerRequest, WorkerResponse } from "./protocol.js"
import { readPrompt } from "./prompt.js"
import type { Engine } from "./engine.js"

const require = createRequire(join(process.cwd(), "package.json"))
const { build } = await import(pathToFileURL(require.resolve("esbuild")).href) as typeof import("esbuild")
const temporary = await mkdtemp(join(tmpdir(), "webgl-worker-cancel-"))
after(() => rm(temporary, { recursive: true, force: true }))
const source = await readFile(join(process.cwd(), "src/handler.ts"), "utf8")
const start = source.indexOf("async function generate(request: GenerateRequest)")
const end = source.indexOf("\n/**\n * Generations run one after another", start)
assert.ok(start >= 0 && end > start, "exercise the current complete worker generate function")
const outfile = join(temporary, "generate.mjs")
await build({ stdin: { loader: "ts", resolveDir: join(process.cwd(), "src"), contents: `
  import { adaptivePrefixFor } from './checkpoint-policy.js';
  import { modelMeta } from './model-meta.js';
  import { anchorFor, readPrompt, systemPrefixFor, templateFramingFor } from './prompt.js';
  import { createThinkingBudget } from './thinking-budget.js';
  import { outputTokenLimit } from './generation-limits.js';
  import { StructuredGeneration } from './structured-generation.js';
  import { withPromptCachePolicy } from './prompt-cache-policy.js';
  import { planToolChoice } from './tool-choice.js';
  import { createStopFilter } from './stop-sequences.js';
  import { seededUniforms } from './seeded-random.js';
  import { CodedError } from './worker-errors.js';
  export function makeGenerate(state) {
    const stopped=state.stopped,post=m=>state.messages.push(m);
    const startLoad=async()=>state.model,createConstraint=()=>state.createConstraint();
    // (Text prompts: the image steps pass the prompt through; image-input.test.ts covers them.)
    const promptImages=(...a)=>(state.promptImages??(async(_model,prompt)=>({prompt,keys:prompt,images:[]})))(...a),encodeImages=(...a)=>(state.encodeImages??(async()=>0))(...a);
    const abortFor=(id)=>{const a=new AbortController();if(stopped.has(id))a.abort();state.signals?.set(id,a);return a.signal};
    ${source.slice(start, end)}
    return generate;
  }
` }, outfile, bundle: true, format: "esm", platform: "node", conditions: ["webgllm/source"], logLevel: "error" })
const { makeGenerate } = await import(pathToFileURL(outfile).href)
function gate() {
  let release!: () => void
  const promise = new Promise<void>(resolve => { release = resolve })
  return { promise, release }
}
function probe() {
  const calls = { forwards: 0, rewinds: 0, disposed: 0, drops: 0, masks: 0, snapshots: 0 }
  const cached = [9, 9], stopped = new Set<string>(), messages: WorkerResponse[] = []
  const engine = {
    contextLength: 8192, snapshotLimit: 2, canSaveInside: false,
    gpu: { compiled: { count: 0 }, calibratePromptPacing: (): Promise<void> | undefined => undefined },
    setAllowedTokens() { calls.masks++ }, dropGuess() { calls.drops++ },
    reset() { cached.length = 0 }, canResume() { return false },
    rewind() { calls.rewinds++; return 0 }, snapshotPosition() { return undefined },
    snapshot() { calls.snapshots++; return undefined },
    batchSizes(count: number) { const sizes = []; for (; count; count -= Math.min(256, count)) sizes.push(Math.min(256, count)); return sizes },
    preparePrompt: async () => undefined,
    setImages() {},
    async forwardHidden() { calls.forwards++; return 0 }
  }
  const tokenizer = { encodeChat: () => Array.from({ length: 512 }, (_, i) => 10 + i % 30), vocabSize: 128,
    specialTokens: { think: null, thinkEnd: null, eos: 1, imStart: 2 }, constraintTokenizer: {} }
  return { calls, stopped, messages, model: { engine, tokenizer, cached },
    createConstraint: async () => ({ dispose() { calls.disposed++ } }) }
}
const request = (id: string, cachePrompt: boolean): Extract<WorkerRequest, { type: "generate" }> => ({
  type: "generate", id, model: "LiquidAI/LFM3-1.7B-GGUF", cachePrompt,
  chat: [{ role: "user", content: "Choose a move" }], tools: [], thinking: false,
  outputSchema: { type: "object", properties: { move: { const: "a" } } },
  sampling: { temperature: 0.1, top_k: 50, top_p: 0.1, min_p: 0, repetition_penalty: 1, presence_penalty: 0 }
})

for (const cache of [true, false]) test(`stop during constraint loading performs zero forwards and disposes the constraint (cache ${cache})`, async () => {
  const state = probe(), entered = gate(), pending = gate()
  const constraint = await state.createConstraint()
  state.createConstraint = async () => { entered.release(); await pending.promise; return constraint }
  const generate = makeGenerate(state)
  const job = generate(request("cancelled", cache))
  await entered.promise; state.stopped.add("cancelled"); pending.release(); await job
  assert.equal(state.calls.forwards, 0)
  assert.equal(state.calls.rewinds, 0, "cancellation must precede prompt-cache restoration too")
  assert.deepEqual(state.model.cached, cache ? [9, 9] : [])
  assert.equal(state.calls.disposed, 1); assert.equal(state.calls.drops, 1)
  assert.equal(state.calls.masks, 1, "the cleanup scope clears any allowed-token mask")
  assert.equal(state.messages.some(m => m.type === "token"), false)
  const done = state.messages.find(m => m.type === "done")
  assert.ok(done?.type === "done"); assert.equal(done.stats.inputTokens, 0); assert.equal(done.stats.outputTokens, 0)
  // The next queued request still uses the unchanged real prompt reader.
  const recovery = { ...request("next-request", cache), maxTokens: 0 }; delete recovery.outputSchema
  await generate(recovery)
  assert.equal(state.calls.forwards, 2)
  assert.deepEqual(state.model.cached, cache ? state.model.tokenizer.encodeChat() : [])
  assert.ok(state.messages.some(m => m.type === "done" && m.id === "next-request"))
})

test("readPrompt does not forward or promote checkpoints if already stopped", async () => {
  const state = probe(), cached = [...state.model.cached]
  const engine = state.model.engine as unknown as Engine
  await readPrompt(engine, [9, 9, 10, 11], 2, state.model.cached, { prefix: 2, stop: () => true })
  assert.equal(state.calls.forwards, 0); assert.equal(state.calls.snapshots, 0); assert.deepEqual(state.model.cached, cached)
})

test("readPrompt checks stop again between planning and the first forward", async () => {
  const state = probe(); let stopped = false
  state.model.engine.batchSizes = count => { stopped = true; return [count] }
  const before = [...state.model.cached]
  await readPrompt(state.model.engine as unknown as Engine, Array(512).fill(10), 0, state.model.cached, { checkpoints: false, stop: () => stopped })
  assert.equal(state.calls.forwards, 0); assert.deepEqual(state.model.cached, before)
})


test("stop during prompt reservation performs zero forwards", async () => {
  const state = probe(), entered = gate(), pending = gate()
  state.model.engine.preparePrompt = async () => {
    entered.release()
    await pending.promise
  }
  const job = makeGenerate(state)(request("cancelled", true))
  await entered.promise
  state.stopped.add("cancelled")
  pending.release()
  await job
  assert.equal(state.calls.forwards, 0)
  assert.equal(state.calls.snapshots, 0)
  assert.equal(state.calls.disposed, 1)
  assert.equal(state.messages.some(message => message.type === "token"), false)
  assert.ok(state.messages.some(message => message.type === "done"))
})

test("stop during pacing calibration performs zero forwards", async () => {
  const state = probe(), entered = gate(), pending = gate()
  state.model.engine.gpu.calibratePromptPacing = () => {
    entered.release()
    return pending.promise
  }
  const job = makeGenerate(state)(request("cancelled", true))
  await entered.promise
  state.stopped.add("cancelled")
  pending.release()
  await job
  assert.equal(state.calls.forwards, 0)
  assert.deepEqual(state.model.cached, [9, 9])
  assert.ok(state.messages.some(message => message.type === "done"))
})

for (const step of ["promptImages", "encodeImages"] as const)
  test(`a stop while the images are ${step === "promptImages" ? "fetched" : "encoded"} ends the request, and the next one runs`, async () => {
    const state = probe() as ReturnType<typeof probe> & Record<string, unknown>
    const entered = gate()
    const signals = new Map<string, AbortController>()
    state.signals = signals
    // The image step waits on its request's signal, as a stalled fetch, download or encode does.
    state[step] = async (...args: Array<unknown>) => {
      const signal = args.find((a): a is AbortSignal => a instanceof AbortSignal)!
      entered.release()
      await new Promise((_, reject) => signal.addEventListener("abort", () => reject(signal.reason)))
    }
    const generate = makeGenerate(state)
    const job = generate({ ...request("stalled", true), maxTokens: 0, outputSchema: undefined })
    await entered.promise
    state.stopped.add("stalled")
    signals.get("stalled")!.abort(new DOMException("stopped", "AbortError"))
    await job
    assert.ok(state.messages.some((m) => m.type === "done" && m.id === "stalled" && m.finishReason === "stop"))
    assert.equal(state.messages.some((m) => m.type === "error"), false)
    // The next request (text) runs.
    delete state[step]
    const before = state.calls.forwards
    await generate({ ...request("next", true), maxTokens: 0, outputSchema: undefined })
    assert.ok(state.calls.forwards > before)
    assert.ok(state.messages.some((m) => m.type === "done" && m.id === "next"))
  })
