import assert from "node:assert/strict"
import { test } from "node:test"
import { useFolder } from "./fake-opfs.test.js"
import { browserStorage, modelFiles } from "./model-files.js"
import { forgetResolvedModels } from "./model-meta.js"
import type { WebGlModelMeta } from "./model-meta.js"
import { hub, resolveModel } from "./resolve-model.js"
import { diskName, saveManifest, storedManifests } from "./storage.js"
import { internals, WebGLLM } from "./webgllm.js"

/** A GGUF model outside the catalog whose tokenizer is LFM2 350M's (as LiquidAI/LFM2-350M-GGUF's is). */
const ggufModel = (name: string): WebGlModelMeta => ({
  name,
  format: "gguf",
  files: { kind: "gguf", model: { repo: name, path: "model-Q4_0.gguf" } },
  tokenizer: {
    tokenizer: { repo: "LiquidAI/LFM2-350M", path: "tokenizer.json" },
    config: { repo: "LiquidAI/LFM2-350M", path: "tokenizer_config.json" },
    chatTemplate: { repo: "LiquidAI/LFM2-350M", path: "chat_template.jinja" }
  },
  supports: { input: ["text"], output: ["text"], capabilities: ["tools"] },
  size: "350m",
  sampling: { temperature: 0, top_k: 1, repetition_penalty: 1 },
  toolFormat: "lfm2"
})

const ONE = ggufModel("someone/One-GGUF")
const TWO = ggufModel("someone/Two-GGUF")

test("after a reload, deleting one downloaded model keeps the files another still reads, with no Hub request", async (t) => {
  const fake = useFolder(t)
  // Both downloaded on an earlier page: their files, and the meta the worker keeps beside them.
  for (const meta of [ONE, TWO]) {
    for (const file of modelFiles(meta)) {
      fake.put(diskName(file), "bytes")
      fake.put(`${diskName(file)}.done`, "")
    }
    await saveManifest(meta)
  }
  // A new page: nothing resolved, and no network.
  forgetResolvedModels()
  const asked: Array<string> = []
  hub.fetch = async (url: string) => {
    asked.push(url)
    throw new TypeError("offline")
  }
  t.mock.method(globalThis, "fetch", async () => {
    throw new TypeError("offline")
  })
  const saved = internals.storage
  internals.storage = browserStorage
  try {
    const llm = new WebGLLM({ memoryBudget: 1000 })
    await llm.models.delete(ONE.name)
    const stored = (file: { repo: string; path: string }) => fake.files.has(`${diskName(file)}.done`)
    assert.ok(!stored(ONE.files.kind === "gguf" ? ONE.files.model : ONE.files.graph), "its weights are gone")
    for (const file of modelFiles(TWO)) assert.ok(stored(file), `${file.repo}/${file.path} stays`)
    assert.equal(await llm.models.downloaded(TWO.name), true)
    assert.equal(await llm.models.downloaded(ONE.name), false)
    assert.ok((await llm.models.list()).some((m) => m.id === TWO.name && m.downloaded))
  } finally {
    internals.storage = saved
  }
  assert.deepEqual(asked, [], "no Hub request")
})

test("a downloaded model resolves from its stored meta after a reload, offline", async (t) => {
  useFolder(t)
  await saveManifest(TWO)
  forgetResolvedModels()
  let asked = 0
  hub.fetch = async () => {
    asked++
    throw new TypeError("offline")
  }
  assert.deepEqual(await resolveModel(TWO.name), TWO)
  assert.equal(asked, 0)
})

test("models.downloaded for a model with no stored meta reads storage only", async (t) => {
  useFolder(t)
  forgetResolvedModels()
  let asked = 0
  hub.fetch = async () => {
    asked++
    throw new TypeError("offline")
  }
  const saved = internals.storage
  internals.storage = browserStorage
  try {
    const llm = new WebGLLM({ memoryBudget: 1000 })
    assert.equal(await llm.models.downloaded("someone/Never-GGUF"), false)
  } finally {
    internals.storage = saved
  }
  assert.equal(asked, 0)
})

test("a manifest stored before 0.2.0 is deleted, and the model resolves again", async (t) => {
  useFolder(t)
  // 0.1.9 stored the chat template's enable_thinking values.
  const old = { ...ggufModel("someone/Thinks-GGUF"), reasoning: { levels: { off: "false", high: "true" }, budget: true } }
  await saveManifest(old as unknown as WebGlModelMeta)
  assert.equal((await storedManifests()).some((m) => m.name === old.name), false, "not listed, and deleted")
  await saveManifest(old as unknown as WebGlModelMeta)
  forgetResolvedModels()
  const asked: Array<string> = []
  hub.fetch = async (url: string) => {
    asked.push(url)
    throw new TypeError("offline")
  }
  await assert.rejects(resolveModel(old.name))
  assert.ok(asked.length > 0, "it resolves from the Hub again")
  assert.equal((await storedManifests()).some((m) => m.name === old.name), false)
})
