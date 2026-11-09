// The worker's generate function on a scripted engine: finish reasons, stop
// sequences, forced tool calls and seeded sampling, end to end through the
// real decode loop (as worker-cancellation.test.ts extracts it).
import assert from "node:assert/strict"
import { after, test } from "node:test"
import { mkdtemp, readFile, rm } from "node:fs/promises"
import { tmpdir } from "node:os"
import { join } from "node:path"
import { pathToFileURL } from "node:url"
import { createRequire } from "node:module"
import type { WorkerRequest, WorkerResponse } from "./protocol.js"
import type { Constraint } from "./constraint/index.js"

const require = createRequire(join(process.cwd(), "package.json"))
const { build } = await import(pathToFileURL(require.resolve("esbuild")).href) as typeof import("esbuild")
const temporary = await mkdtemp(join(tmpdir(), "webgl-worker-generation-"))
after(() => rm(temporary, { recursive: true, force: true }))
const source = await readFile(join(process.cwd(), "src/worker.ts"), "utf8")
const start = source.indexOf("async function generate(request: GenerateRequest)")
const end = source.indexOf("\n/**\n * Generations run one after another", start)
assert.ok(start >= 0 && end > start)
const outfile = join(temporary, "generate.mjs")
await build({ stdin: { loader: "ts", resolveDir: join(process.cwd(), "src"), contents: `
  import { adaptivePrefixFor } from './checkpoint-policy.js';
  import { modelMeta } from './model-meta.js';
  import { anchorFor, readPrompt, systemPrefixFor, templateFramingFor } from './prompt.js';
  import { createThinkingBudget } from './thinking-budget.js';
  import { outputTokenLimit } from './generation-limits.js';
  import { StructuredGeneration, maskLogits } from './structured-generation.js';
  import { withPromptCachePolicy } from './prompt-cache-policy.js';
  import { planToolChoice } from './tool-choice.js';
  import { createStopFilter } from './stop-sequences.js';
  import { seededUniforms } from './seeded-random.js';
  import { CodedError } from './worker-errors.js';
  import { createSampler, draw, drawAt, verify, verifyAt } from './sampler.js';
  import { DraftPolicy, lookupDraft } from './draft.js';
  import { THINK_OPEN } from './reply.js';
  const MAX_CHUNKS = 64, TUNING = { draftMatch: 2, guessDrafts: 0 };
  const DRAFTS = 7, GUESS_MIN_HITS = 0.6, GUESS_WINDOW = 16, RECLAIM_BYTES = Infinity;
  const claimMemory = () => undefined, writeGpuNote = () => undefined;
  export function makeGenerate(state) {
    const stopped = state.stopped, post = m => state.messages.push(m);
    const startLoad = async () => state.model, createConstraint = async () => state.constraint();
    ${source.slice(start, end)}
    return generate;
  }
` }, outfile, bundle: true, format: "esm", platform: "node", conditions: ["webgllm/source"], logLevel: "error" })
const { makeGenerate } = await import(pathToFileURL(outfile).href)

/** Token ids 0–9 are control tokens; the rest are words of TEXT. */
const EOS = 1, THINK = 3, THINK_END = 4
const TEXT: Record<number, string> = { [THINK]: "<think>", [THINK_END]: "</think>" }
const words = ["Hello", " world", ".", " STOP", " more", "{", "\"city\"", ":", "\"Oslo\"", "}", " hm", "\n\n", "a", "b", "c", "d"]
words.forEach((w, i) => { TEXT[10 + i] = w })
const id = (w: string) => 10 + words.indexOf(w)

/**
 * A model whose greedy picks follow `script` (then EOS), and whose sampled
 * candidates are `candidates` (ids and logits) at every step.
 */
function scripted(script: number[], options: { candidates?: [number[], number[]]; constraint?: () => Constraint; contextLength?: number } = {}) {
  const queue = script.slice()
  const stopped = new Set<string>(), messages: WorkerResponse[] = []
  const masks: Array<Uint32Array | null> = []
  // What the penalties were given: the tokens set as seen, and each pick's penalties.
  const seen: number[][] = [], penalties: Array<[number, number]> = []
  let position = 0
  const engine = {
    contextLength: options.contextLength ?? 8192, snapshotLimit: 2, canSaveInside: false, canDraft: false, guessing: false, guessedToken: -1,
    get position() { return position },
    gpu: { compiled: { count: 0, times: new Map() }, bytes: 0, presentationAdvice: { relaxation: true }, validatedPresentation: undefined,
      calibratePromptPacing: () => undefined },
    tuned: { submissionOverlap: 0 },
    setAllowedTokens(mask: Uint32Array | null) { masks.push(mask) }, setAllowedTokenRows() {},
    dropGuess() {}, keepGuess() {}, accept() {}, reset() {}, canResume() { return false },
    rewind() { position = 0; return 0 }, snapshotPosition() { return undefined }, snapshot() { return undefined },
    batchSizes(count: number) { return [count] },
    preparePrompt: async () => undefined,
    async forwardHidden(batch: number[]) { position += batch.length; return batch.length - 1 },
    setSeen(tokens: Iterable<number>) { seen.push([...tokens]) }, markSeen() {},
    async argmax(_row: number, _rows: number, penalty: number, _drafts: number[], _guess: boolean, presence = 0) {
      penalties.push([penalty, presence])
      return Int32Array.of(queue.shift() ?? EOS)
    },
    async candidates() {
      const [ids, logits] = options.candidates!
      return { ids: Int32Array.from(ids), logits: Float32Array.from(logits) }
    },
    async pick(_row: number, rows: number) {
      return Promise.all(Array.from({ length: rows }, () => this.candidates()))
    }
  }
  const tokenizer = {
    encodeChat: () => [2, 20, 21, 22, 23, 24],
    vocabSize: 64,
    specialTokens: { think: THINK, thinkEnd: THINK_END, eos: EOS, imStart: 2 },
    constraintTokenizer: {},
    decode: (ids: number[]) => ids.map((t) => TEXT[t] ?? "").join(""),
    createStreamDecoder: () => ({ push: (t: number) => TEXT[t] ?? "", flush: () => "" })
  }
  const state = { stopped, messages, model: { engine, tokenizer, cached: [] as number[] }, constraint: options.constraint }
  return { generate: makeGenerate(state) as (r: Request) => Promise<void>, messages, masks, seen, penalties }
}

type Request = Extract<WorkerRequest, { type: "generate" }>
const request = (extra: Partial<Request> = {}): Request => ({
  type: "generate", id: "r", model: "LiquidAI/LFM3-1.7B-GGUF", chat: [{ role: "user", content: "Hi" }], tools: [], thinking: false,
  sampling: { temperature: 0, top_k: 1, repetition_penalty: 1 }, ...extra
})
const text = (messages: WorkerResponse[]) => messages.map((m) => m.type === "token" ? m.text : "").join("")
const done = (messages: WorkerResponse[]) => {
  const d = messages.find((m) => m.type === "done")
  assert.ok(d?.type === "done")
  return d
}

test("an end-of-turn token finishes with stop, maxTokens with length", async () => {
  const a = scripted([id("Hello"), id(" world")])
  await a.generate(request())
  assert.equal(text(a.messages), "Hello world")
  assert.equal(done(a.messages).finishReason, "stop")
  const b = scripted([id("Hello"), id(" world"), id(".")])
  await b.generate(request({ maxTokens: 2 }))
  assert.equal(text(b.messages), "Hello world")
  assert.equal(done(b.messages).finishReason, "length")
  // The end-of-turn token as the last allowed one is still a stop.
  const c = scripted([id("Hello")])
  await c.generate(request({ maxTokens: 2 }))
  assert.equal(done(c.messages).finishReason, "stop")
  const z = scripted([id("Hello")])
  await z.generate(request({ maxTokens: 0 }))
  assert.equal(done(z.messages).finishReason, "length")
})

test("a stop sequence ends the reply before it, and is never sent", async () => {
  const s = scripted([id("Hello"), id(" world"), id(" STOP"), id(" more")])
  await s.generate(request({ stop: [" ST"] }))
  assert.equal(text(s.messages), "Hello world")
  assert.equal(done(s.messages).finishReason, "stop")
  assert.equal(done(s.messages).stats.outputTokens, 3)
  // Not in the thinking.
  const t = scripted([THINK, id(" STOP"), THINK_END, id("\n\n"), id("Hello"), id(" STOP"), id(" more")])
  await t.generate(request({ stop: [" STOP"], thinking: true }))
  assert.equal(text(t.messages), "<think> STOP</think>\n\nHello")
  // A held-back partial match that doesn't complete is sent at the end.
  const u = scripted([id("Hello"), id(" STOP")])
  await u.generate(request({ stop: [" STOPPED"] }))
  assert.equal(text(u.messages), "Hello STOP")
})

test("greedy text is unchanged by a seed", async () => {
  const plain = scripted([id("a"), id("b"), id("c")])
  await plain.generate(request())
  const seeded = scripted([id("a"), id("b"), id("c")])
  await seeded.generate(request({ seed: 7 }))
  assert.equal(text(seeded.messages), text(plain.messages))
})

test("a seed reproduces a sampled reply", async () => {
  const candidates: [number[], number[]] = [[id("a"), id("b"), id("c"), id("d")], [1, 1, 1, 1]]
  const sampling = { temperature: 1, top_k: 4, repetition_penalty: 1 }
  const run = async (seed?: number) => {
    const s = scripted([], { candidates })
    await s.generate(request({ sampling, seed, maxTokens: 24 }))
    return text(s.messages)
  }
  const first = await run(1234)
  assert.equal(first.length, 24)
  assert.equal(await run(1234), first)
  const others = new Set([await run(1), await run(2), await run(3)])
  assert.ok(others.size > 1 || !others.has(first))
})

/** A constraint that takes any token and is complete once `last` is accepted. */
function jsonConstraint(last: number) {
  const accepted: number[] = []
  const constraint: Constraint = {
    vocabSize: 64, eosTokenId: EOS,
    get complete() { return accepted.includes(last) },
    computeMask: () => new Uint32Array(2).fill(0xffffffff),
    accept: (t) => { accepted.push(t) },
    clone: () => constraint,
    dispose: () => {}
  }
  return { constraint, accepted }
}

const weather = { type: "function" as const, function: { name: "get_weather", parameters: { type: "object", properties: { city: { type: "string" } }, required: ["city"] } } }
const json = [id("{"), id("\"city\""), id(":"), id("\"Oslo\""), id("}")]

test("a forced tool call is sent as toolCall on done, not as text", async () => {
  const j = jsonConstraint(id("}"))
  const s = scripted([...json], { constraint: () => j.constraint })
  await s.generate(request({ tools: [weather], toolChoice: { name: "get_weather" } }))
  assert.equal(text(s.messages), "")
  const d = done(s.messages)
  assert.deepEqual(d.toolCall, { name: "get_weather", arguments: { city: "Oslo" } })
  assert.equal(d.finishReason, "stop")
  assert.deepEqual(j.accepted, [...json, EOS])
})

test("thinking comes before a forced call, and is sent as text", async () => {
  const j = jsonConstraint(id("}"))
  const s = scripted([id(" hm"), THINK_END, ...json], { constraint: () => j.constraint })
  await s.generate(request({ tools: [weather], toolChoice: "required", thinking: true, stop: ["{"] }))
  // The template ended the prompt outside the thinking: it's opened first.
  assert.equal(text(s.messages), "<think> hm</think>")
  assert.deepEqual(done(s.messages).toolCall, { name: "get_weather", arguments: { city: "Oslo" } })
  assert.deepEqual(j.accepted, [...json, EOS])
})

test("toolChoice naming an unknown tool fails", async () => {
  const s = scripted([])
  await assert.rejects(s.generate(request({ tools: [weather], toolChoice: { name: "nope" } })), /nope/)
})

test("a session too long for the model fails with context-length-exceeded", async () => {
  const s = scripted([], { contextLength: 6 })
  await assert.rejects(s.generate(request()), (error: unknown) =>
    // (The extracted bundle has its own CodedError class: read the code.)
    (error as { code?: string }).code === "context-length-exceeded")
})

test("stop sequences don't apply inside a structured answer: it stays valid JSON", async () => {
  const schema = { type: "object", properties: { city: { type: "string" } }, required: ["city"] }
  for (const stop of [["{"], ["}"], ["Oslo"]]) {
    const j = jsonConstraint(id("}"))
    const s = scripted([...json], { constraint: () => j.constraint })
    await s.generate(request({ outputSchema: schema, stop }))
    assert.equal(text(s.messages), "{\"city\":\"Oslo\"}", `stop ${JSON.stringify(stop)}`)
    assert.equal(done(s.messages).finishReason, "stop")
  }
  // Thinking before it is still sent as written.
  const j = jsonConstraint(id("}"))
  const t = scripted([id(" hm"), THINK_END, ...json], { constraint: () => j.constraint })
  await t.generate(request({ outputSchema: schema, stop: ["{"], thinking: true }))
  assert.equal(text(t.messages), "<think> hm</think>{\"city\":\"Oslo\"}")
})

test("the repetition penalty counts the prompt's tokens, as Transformers.js does", async () => {
  const s = scripted([id("Oslo")])
  await s.generate(request({ sampling: { temperature: 0, top_k: 1, repetition_penalty: 1.1 } }))
  assert.equal(s.seen.length, 1)
  assert.ok(s.seen[0].length > 0, "the prompt's tokens are penalized")
})

test("no repetition or presence penalty inside a constrained answer", async () => {
  const schema = { type: "object", properties: { city: { type: "string" } }, required: ["city"] }
  const j = jsonConstraint(id("}"))
  const s = scripted([id(" hm"), THINK_END, ...json], { constraint: () => j.constraint })
  await s.generate(request({ outputSchema: schema, thinking: true, sampling: { temperature: 0, top_k: 1, repetition_penalty: 1.1, presence_penalty: 1.5 } }))
  assert.equal(text(s.messages), "<think> hm</think>{\"city\":\"Oslo\"}")
  // Thinking: the request's penalties. The answer: none (its quotes, keys and
  // copied values repeat by design).
  assert.deepEqual(s.penalties.slice(0, 2), [[1.1, 1.5], [1.1, 1.5]])
  assert.deepEqual(new Set(s.penalties.slice(2).map(String)), new Set(["1,0"]))
})

test("with thinking switched off, the reply can't open or close thinking", async () => {
  const s = scripted([id("Hello")])
  await s.generate(request({ thinking: false }))
  const mask = s.masks.find((m) => m !== null)
  assert.ok(mask, "a mask is applied")
  const allowed = (t: number) => (mask[t >>> 5] & (1 << (t & 31))) !== 0
  assert.equal(allowed(THINK), false)
  assert.equal(allowed(THINK_END), false)
  assert.equal(allowed(id("Hello")), true)
  // Thinking left to the template (or on) is untouched.
  const on = scripted([id("Hello")])
  await on.generate(request({ thinking: undefined }))
  assert.deepEqual(on.masks.filter((m) => m !== null), [])
})
