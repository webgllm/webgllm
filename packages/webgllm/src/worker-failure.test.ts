// A worker of the caller's that fails: its script never loads (an "error"
// event with nothing in it, and no answer to anything), or its code throws
// later. What waits on it rejects, and nothing waits on it after.
import assert from "node:assert/strict"
import { test } from "node:test"
import { FakeWorker, fakeStorage, settle } from "./fake-worker.test.js"
import type { FakeOptions } from "./fake-worker.test.js"
import { internals, WebGLLM } from "./webgllm.js"

const MODEL = "LiquidAI/LFM2.5-350M-GGUF"
const hi = [{ role: "user" as const, content: "hi" }]
internals.hasWebGl2 = () => true

function given(options: FakeOptions = {}) {
  internals.storage = fakeStorage()
  const worker = new FakeWorker(options)
  const llm = new WebGLLM({ worker: worker as unknown as Worker, memoryBudget: 1000 })
  const events: Array<string> = []
  llm.addEventListener("error", ({ model, error }) => events.push(`error ${model} ${(error as { code?: string }).code}`))
  llm.addEventListener("unload", ({ model, reason }) => events.push(`unload ${model} ${reason}`))
  return { llm, worker, events }
}

/** How `promise` ends within a moment: its rejection's code, "resolved", or "pending" if it's still waiting. */
async function outcome(promise: Promise<unknown>): Promise<string> {
  const ended = promise.then(() => "resolved", (error: { code?: string; name?: string }) => error.code ?? error.name ?? "rejected")
  return Promise.race([ended, settle(20).then(() => "pending")])
}

test("a worker whose script fails to load while a load waits on it: the load rejects", async () => {
  const { llm, worker, events } = given({ dead: true })
  const loading = llm.loadModel(MODEL)
  await settle()
  worker.failToLoad()
  assert.equal(await outcome(loading), "worker-failed")
  assert.deepEqual(events, [`error ${MODEL} worker-failed`, `unload ${MODEL} crashed`])
  assert.equal(worker.terminated, false)
})

test("a worker whose script failed to load before anything was asked of it: requests reject instead of waiting", async () => {
  const { llm, worker } = given({ dead: true })
  worker.failToLoad()
  assert.equal(await outcome(llm.loadModel(MODEL)), "worker-failed")
  assert.equal(await outcome(llm.chat.completions.create({ model: MODEL, messages: hi })), "worker-failed")
  assert.equal(await outcome(llm.completions.create({ model: MODEL, prompt: "x" })), "worker-failed")
  assert.equal(await outcome(llm.forwardTokensAndSample([1], true, MODEL)), "worker-failed")
  assert.equal(await outcome(llm.resetChat(false, MODEL)), "worker-failed")
  assert.equal(await outcome(llm.downloadModel(MODEL)), "worker-failed")
  // Nothing was sent to a worker known to be dead, and it isn't ended.
  assert.deepEqual(worker.sent, [])
  assert.equal(worker.terminated, false)
  // Nothing is loaded in it: unloading has nothing to wait for.
  assert.equal(await outcome(llm.unloadModel()), "resolved")
})

test("a worker whose script fails to load fails everything waiting on it: a completion, a forward, a download, an unload", async () => {
  // A completion mid-reply.
  const chatting = given()
  const chat = chatting.llm.chat.completions.create({ model: MODEL, messages: hi })
  await settle()
  assert.equal(chatting.worker.generations().length, 1)
  chatting.worker.failToLoad()
  assert.equal(await outcome(chat), "worker-failed")
  assert.deepEqual(chatting.events, [`error ${MODEL} worker-failed`, `unload ${MODEL} crashed`])

  // A token forward and a download, in a worker that answers nothing.
  const waiting = given({ dead: true })
  const forwarded = waiting.llm.forwardTokensAndSample([1], true, MODEL)
  const downloading = waiting.llm.downloadModel("unsloth/Qwen3.5-2B-GGUF")
  await settle()
  waiting.worker.failToLoad()
  assert.equal(await outcome(forwarded), "worker-failed")
  assert.equal(await outcome(downloading), "worker-failed")

  // An unload the worker never acknowledges.
  let hold!: () => void
  const unloading = given({ holdUnload: () => new Promise<void>((resolve) => { hold = resolve }) })
  await unloading.llm.loadModel(MODEL)
  const unload = unloading.llm.unloadModel()
  await settle()
  unloading.worker.failToLoad()
  assert.equal(await outcome(unload), "worker-failed")
  hold()
  assert.equal(unloading.worker.terminated, false)
})

test("a worker whose own code throws is still running: what waited fails, and the next request goes on", async () => {
  const { llm, worker, events } = given({ reply: () => ({ tokens: ["ok"] }) })
  await llm.loadModel(MODEL)
  worker.fail("ReferenceError: mine is not defined")
  await settle()
  assert.deepEqual(events, [`error ${MODEL} worker-failed`, `unload ${MODEL} crashed`])
  assert.equal((await llm.models.retrieve(MODEL)).loaded, false)
  // The handler in it is as it was: the model loads again and answers.
  const completion = await llm.chat.completions.create({ model: MODEL, messages: hi })
  assert.equal(completion.choices[0]!.message.content, "ok")
  assert.equal(worker.terminated, false)
})

test("webgllm's own worker that fails: the load rejects with worker-failed, and the next load starts another", async () => {
  internals.storage = fakeStorage()
  const made: Array<FakeWorker> = []
  internals.createWorker = () => {
    const worker = new FakeWorker(made.length ? {} : { dead: true })
    made.push(worker)
    return worker as unknown as Worker
  }
  const llm = new WebGLLM({ memoryBudget: 1000 })
  const loading = llm.loadModel(MODEL)
  await settle()
  made[0]!.failToLoad()
  assert.equal(await outcome(loading), "worker-failed")
  assert.equal(made[0]!.terminated, true)
  await llm.loadModel(MODEL)
  assert.equal(made.length, 2)
})
