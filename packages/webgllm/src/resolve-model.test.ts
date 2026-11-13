import assert from "node:assert/strict"
import { test } from "node:test"
import { WebGLLMError } from "./errors.js"
import { api, asRepo, code, fakeHub, recorded } from "./fake-hub.test.js"
import { fileUrl, modelMeta, WEBGL_MODEL_META, WEBGL_TEXT_MODELS } from "./model-meta.js"
import type { WebGlModelMeta } from "./model-meta.js"
import { hub, hubModelMeta, onnxCandidates, pickGguf, resolveModel, samplingFrom, sizeFrom } from "./resolve-model.js"

test("the resolver agrees with the catalog on every catalog model", async () => {
  fakeHub()
  const oneFile = hub.checkGraph
  for (const name of WEBGL_TEXT_MODELS) {
    const catalog: WebGlModelMeta = WEBGL_MODEL_META[name]
    // The fake's graph check names one data file: a graph whose weights are in several (LFM2.5-VL 3B's) names its own.
    const several = catalog.files.kind === "onnx" && catalog.files.moreData ? [catalog.files.data, ...catalog.files.moreData] : undefined
    hub.checkGraph = several ? () => ({ dataFiles: several.map((file) => file.path.split("/").pop()!) }) : oneFile
    const meta = await hubModelMeta(name)
    assert.equal(meta.name, name)
    assert.equal(meta.format, catalog.format, name)
    assert.deepEqual(meta.files, catalog.files, name)
    assert.deepEqual(meta.tokenizer.tokenizer, catalog.tokenizer.tokenizer, name)
    assert.deepEqual(meta.tokenizer.config, catalog.tokenizer.config, name)
    if ("chatTemplate" in catalog.tokenizer) assert.deepEqual(meta.tokenizer.chatTemplate, catalog.tokenizer.chatTemplate, name)
    // The catalog's LFM3 ONNX predates its chat_template.jinja (it reads tokenizer_config.json's).
    else assert.deepEqual(meta.tokenizer.chatTemplate, { repo: catalog.tokenizer.config.repo, path: "chat_template.jinja" }, name)
    assert.deepEqual(meta.reasoning, catalog.reasoning, name)
    assert.equal(meta.toolFormat, catalog.toolFormat ?? "lfm3", name)
    assert.deepEqual(meta.supports, catalog.supports, name)
    assert.equal(meta.size, catalog.size, name)
  }
})

test("a catalog model resolves to its catalog entry without asking the Hub", async () => {
  const calls = fakeHub()
  assert.equal(await resolveModel("LiquidAI/LFM3-1.7B-GGUF"), WEBGL_MODEL_META["LiquidAI/LFM3-1.7B-GGUF"])
  assert.equal(await resolveModel("LiquidAI/LFM2-1.2B-Tool-GGUF"), WEBGL_MODEL_META["LiquidAI/LFM2-1.2B-Tool-GGUF"])
  assert.equal(calls.length, 0)
})

test("an uncataloged GGUF: its Q4_0 file, the base model's tokenizer, greedy sampling", async () => {
  const calls = fakeHub()
  const name = "LiquidAI/LFM2-350M-GGUF"
  assert.throws(() => modelMeta(name))
  const meta = await resolveModel(name)
  assert.deepEqual(meta.files, { kind: "gguf", model: { repo: name, path: "LFM2-350M-Q4_0.gguf" } })
  assert.deepEqual(meta.tokenizer, {
    tokenizer: { repo: "LiquidAI/LFM2-350M", path: "tokenizer.json" },
    config: { repo: "LiquidAI/LFM2-350M", path: "tokenizer_config.json" },
    chatTemplate: { repo: "LiquidAI/LFM2-350M", path: "chat_template.jinja" }
  })
  assert.equal(meta.reasoning, undefined)
  assert.equal(meta.toolFormat, "lfm2")
  assert.deepEqual(meta.supports.capabilities, ["tools"])
  assert.equal(meta.size, "350m")
  // Its generation_config.json sets no sampling: greedy, as Transformers decodes it.
  assert.deepEqual(meta.sampling, { temperature: 0, top_k: 1, repetition_penalty: 1 })
  // Kept: the sync lookup has it, and resolving again asks the Hub nothing.
  assert.equal(modelMeta(name), meta)
  const asked = calls.length
  assert.equal(await resolveModel(name), meta)
  assert.equal(calls.length, asked)
})

test("an uncataloged ONNX build: its q4f16 graph, one data file, the repository's tokenizer", async () => {
  fakeHub()
  const name = "onnx-community/LFM2-350M-ONNX"
  const meta = await resolveModel(name)
  assert.equal(meta.format, "onnx")
  assert.deepEqual(meta.files, {
    kind: "onnx",
    graph: { repo: name, path: "onnx/model_q4f16.onnx" },
    data: { repo: name, path: "onnx/model_q4f16.onnx_data" },
    config: { repo: name, path: "config.json" }
  })
  assert.equal(meta.tokenizer.tokenizer.repo, name)
  assert.equal(meta.toolFormat, "lfm2")
})

/** The recorded LFM2-350M tokenizer and config files, as served from `folder`. */
function servedAt(folder: string) {
  const out: Record<string, (typeof recorded)[string]> = {}
  for (const file of ["tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "generation_config.json"]) {
    const answer = recorded[`https://huggingface.co/LiquidAI/LFM2-350M/resolve/main/${file}`]
    if (answer) out[`${folder}/${file}`] = answer
  }
  return out
}

test("a model by URL: its .gguf file, and the tokenizer files beside it", async () => {
  const folder = "https://models.example.com/lfm2"
  const url = `${folder}/LFM2-350M-Q4_0.gguf`
  const calls = fakeHub(servedAt(folder))
  const meta = await resolveModel(url)
  assert.equal(meta.name, url)
  assert.deepEqual(meta.files, { kind: "gguf", model: { repo: folder, path: "LFM2-350M-Q4_0.gguf" } })
  assert.deepEqual(meta.tokenizer.tokenizer, { repo: folder, path: "tokenizer.json" })
  assert.equal(meta.toolFormat, "lfm2")
  assert.equal(meta.size, "350m")
  assert.ok(calls.every((call) => call.startsWith(folder)), "nothing asked of the Hub")
  assert.equal(fileUrl(meta.tokenizer.tokenizer), `${folder}/tokenizer.json`)
})

test("a model by URL: an .onnx graph with its config.json and data file beside it", async () => {
  const folder = "https://models.example.com/lfm2-onnx"
  fakeHub({ ...servedAt(folder), [`${folder}/config.json`]: { status: 200, body: { model_type: "lfm2" } } })
  const meta = await resolveModel(`${folder}/model_q4f16.onnx`)
  assert.deepEqual(meta.files, {
    kind: "onnx",
    graph: { repo: folder, path: "model_q4f16.onnx" },
    data: { repo: folder, path: "model_q4f16.onnx_data" },
    config: { repo: folder, path: "config.json" }
  })
})

test("a URL that isn't a .gguf or .onnx file isn't a model", async () => {
  fakeHub()
  assert.equal(await code(resolveModel("https://models.example.com/lfm2/")), "model-not-supported")
  assert.equal(await code(resolveModel("https://models.example.com/lfm2/tokenizer.json")), "model-not-supported")
})

test("concurrent resolutions of one model ask the Hub once", async () => {
  // A name that isn't in the catalog, answered as LFM2 1.2B Tool's GGUF repository is.
  const alias = "someone/LFM2-1.2B-Tool-GGUF"
  const answers = recorded
  const calls = fakeHub({ [api(alias)]: answers[api("LiquidAI/LFM2-1.2B-Tool-GGUF")]! })
  const [a, b] = await Promise.all([resolveModel(alias), resolveModel(alias)])
  assert.equal(a, b)
  assert.equal(calls.filter((url) => url === api(alias)).length, 1)
  assert.deepEqual(a.files, { kind: "gguf", model: { repo: alias, path: "LFM2-1.2B-Tool-Q4_0.gguf" } })
  assert.equal(a.tokenizer.tokenizer.repo, "LiquidAI/LFM2-1.2B-Tool")
})

test("an unsupported architecture fails before any download, naming it", async () => {
  fakeHub()
  for (const name of ["unsloth/Llama-3.2-1B-Instruct-GGUF", "onnx-community/Llama-3.2-1B-Instruct-ONNX"]) {
    await assert.rejects(resolveModel(name), (error: unknown) => {
      assert.ok(error instanceof WebGLLMError)
      assert.equal(error.code, "model-not-supported")
      assert.match(error.message, /"llama" model/)
      return true
    })
  }
})

test("a repository with no weights webgllm loads fails clearly", async () => {
  fakeHub({
    [api("someone/only-k-quants-GGUF")]: {
      status: 200,
      body: { siblings: [{ rfilename: "m-Q4_K_M.gguf" }, { rfilename: "m-IQ4_XS.gguf" }, { rfilename: "tokenizer.json" }], gguf: { architecture: "lfm2" } }
    },
    [api("someone/split-onnx")]: {
      status: 200,
      body: { siblings: ["config.json", "onnx/model.onnx", "onnx/model.onnx_data", "onnx/model.onnx_data_1"].map((rfilename) => ({ rfilename })), config: { model_type: "lfm2" } }
    }
  })
  await assert.rejects(resolveModel("someone/only-k-quants-GGUF"), (error: unknown) => {
    assert.ok(error instanceof WebGLLMError)
    assert.equal(error.code, "model-not-supported")
    assert.match(error.message, /no weights webgllm loads/)
    return true
  })
  // A graph whose weights are in a file the repository doesn't have.
  hub.checkGraph = () => ({ dataFiles: ["model.onnx_data", "model.onnx_data_2"] })
  await assert.rejects(resolveModel("someone/split-onnx"), (error: unknown) => {
    assert.ok(error instanceof WebGLLMError)
    assert.equal(error.code, "model-not-supported")
    assert.match(error.message, /onnx\/model\.onnx_data_2, which the repository doesn't have/)
    return true
  })
})

test("an ONNX graph whose weights are in several data files lists them all", async () => {
  const name = "someone/Split-350M-ONNX"
  const base = asRepo("LiquidAI/LFM2.5-350M-ONNX", name)
  const listing = base[api(name)]!.body as { siblings: Array<{ rfilename: string }> }
  fakeHub({
    ...base,
    [api(name)]: { status: 200, body: { ...listing, siblings: [...listing.siblings, { rfilename: "onnx/model_q4f16.onnx_data_1" }] } }
  })
  hub.checkGraph = (bytes: Uint8Array) => {
    const graph = new TextDecoder().decode(bytes)
    return { dataFiles: [`${graph}_data_1`, `${graph}_data`] }
  }
  const meta = await resolveModel(name)
  assert.ok(meta.files.kind === "onnx")
  assert.deepEqual(meta.files.data, { repo: name, path: "onnx/model_q4f16.onnx_data_1" })
  assert.deepEqual(meta.files.moreData, [{ repo: name, path: "onnx/model_q4f16.onnx_data" }])
  const { modelFiles } = await import("./model-files.js")
  assert.deepEqual(
    modelFiles(meta).map((f) => f.path).filter((path) => path.includes("onnx_data")),
    ["onnx/model_q4f16.onnx_data_1", "onnx/model_q4f16.onnx_data"]
  )
})

test("names that aren't Hub repositories, and an unreachable Hub", async () => {
  const calls = fakeHub()
  assert.equal(await code(resolveModel("gpt-4o")), "model-not-supported")
  assert.equal(calls.length, 0, "not an owner/name: no request")
  assert.equal(await code(resolveModel("nope/nope")), "model-not-supported")
  hub.fetch = async () => {
    throw new TypeError("Failed to fetch")
  }
  assert.equal(await code(resolveModel("offline/model")), "download-failed")
  // A failure isn't kept: the next try asks again.
  fakeHub()
  assert.equal(await code(resolveModel("offline/model")), "model-not-supported")
})

test("a repository without a tokenizer, here or in its base model, fails clearly", async () => {
  fakeHub({
    [api("someone/bare-GGUF")]: { status: 200, body: { siblings: [{ rfilename: "m-Q4_0.gguf" }], gguf: { architecture: "lfm2" } } }
  })
  await assert.rejects(resolveModel("someone/bare-GGUF"), /no tokenizer/)
})

test("GGUF files by preference: QAD Q4_0, Q4_0, Q8_0, then the fewest bits the loader takes", () => {
  const all = ["m-F16.gguf", "m-Q6_K.gguf", "m-Q5_K_M.gguf", "m-Q8_0.gguf", "m-Q4_K_M.gguf", "m-Q4_0.gguf", "m-QAD-Q4_0.gguf", "mmproj-F16.gguf"]
  assert.equal(pickGguf(all), "m-QAD-Q4_0.gguf")
  assert.equal(pickGguf(all.filter((f) => !f.includes("QAD"))), "m-Q4_0.gguf")
  assert.equal(pickGguf(["m-Q6_K.gguf", "m-Q8_0.gguf", "m-Q4_K_M.gguf"]), "m-Q8_0.gguf")
  assert.equal(pickGguf(["m-Q6_K.gguf", "m-Q5_K_M.gguf", "m-Q4_K_M.gguf"]), "m-Q5_K_M.gguf")
  assert.equal(pickGguf(["model.gguf", "m-Q4_K_M.gguf"]), "model.gguf")
  assert.equal(pickGguf(["m-Q4_K_M.gguf", "m-UD-Q4_K_XL.gguf", "m-BF16.gguf", "mmproj-F16.gguf"]), undefined)
  assert.equal(pickGguf(["m-Q4_0-00001-of-00002.gguf", "m-Q4_0-00002-of-00002.gguf"]), undefined)
})

test("ONNX graphs to try: q4f16, then q4, then the rest; decoders first", () => {
  assert.deepEqual(
    onnxCandidates(["onnx/model.onnx", "onnx/model.onnx_data", "onnx/model_q4.onnx", "onnx/model_fp16.onnx", "onnx/model_q4f16.onnx", "config.json"]),
    ["onnx/model_q4f16.onnx", "onnx/model_q4.onnx", "onnx/model_fp16.onnx", "onnx/model.onnx"]
  )
  // Qwen3.5's decoder before its embedding and vision graphs.
  assert.deepEqual(onnxCandidates(["onnx/embed_tokens_q4f16.onnx", "onnx/vision_encoder_q4f16.onnx", "onnx/decoder_model_merged_q4f16.onnx"]), [
    "onnx/decoder_model_merged_q4f16.onnx",
    "onnx/embed_tokens_q4f16.onnx",
    "onnx/vision_encoder_q4f16.onnx"
  ])
})

test("sampling as Transformers reads generation_config.json; size from the name, else the Hub", () => {
  assert.deepEqual(samplingFrom(undefined), { temperature: 0, top_k: 1, repetition_penalty: 1 })
  assert.deepEqual(samplingFrom({ temperature: 0.1, repetition_penalty: 1.1 }), { temperature: 0, top_k: 1, repetition_penalty: 1.1 })
  assert.deepEqual(samplingFrom({ do_sample: true, temperature: 0.6, top_k: 20, top_p: 0.95 }), { temperature: 0.6, top_k: 20, repetition_penalty: 1, top_p: 0.95 })
  assert.deepEqual(samplingFrom({ do_sample: true }), { temperature: 1, top_k: 50, repetition_penalty: 1 })
  assert.equal(sizeFrom("LiquidAI/LFM2-1.2B-Tool-GGUF", 1170340608), "1.2b")
  assert.equal(sizeFrom("Qwen/Qwen3.5-2B", 2274069824), "2b")
  assert.equal(sizeFrom("someone/tiny", 354483968), "354m")
  assert.equal(sizeFrom("someone/tiny", 1881825088), "1.9b")
  assert.equal(sizeFrom("someone/tiny", undefined), "")
})

test("a URL model resolves its text model when the vision look can't get an answer (a server refusing HEAD)", async () => {
  const folder = "https://models.example.com/qwen-head-405"
  fakeHub({
    ...servedAt(folder),
    [`${folder}/config.json`]: { status: 200, body: { model_type: "qwen3_5", vision_config: { model_type: "qwen3_5" } } },
    // (The first encoder graph the look asks for: the 8-bit one.)
    [`${folder}/vision_encoder_q8.onnx`]: { status: 405, body: "Method Not Allowed" }
  })
  const meta = await resolveModel(`${folder}/decoder_model_merged_q4f16.onnx`)
  assert.equal(meta.files.kind, "onnx")
  assert.deepEqual(meta.supports.input, ["text"])
  assert.equal(meta.visionChecked, false, "looked at again later")
})

test("a stored meta from before the vision look is looked at again online, and kept offline", async (t) => {
  const { useFolder } = await import("./fake-opfs.test.js")
  const { saveManifest, readManifest } = await import("./storage.js")
  const { asRepo } = await import("./fake-hub.test.js")
  useFolder(t)
  const legacy = (meta: Awaited<ReturnType<typeof hubModelMeta>>, name: string) => {
    const { visionChecked: _, ...rest } = meta
    const files = rest.files.kind === "onnx" ? (({ vision: __, ...f }) => f)(rest.files) : rest.files
    return { ...rest, name, files, supports: { ...rest.supports, input: ["text"] as const } }
  }
  const answers = (name: string) => ({
    ...asRepo("onnx-community/Qwen3.5-2B-ONNX", name),
    [`https://huggingface.co/${name}/resolve/main/config.json`]: { status: 200, body: { model_type: "qwen3_5", vision_config: { model_type: "qwen3_5" } } }
  })
  // Online: refreshed, and saved.
  const online = "someone/Qwen3.5-legacy-online"
  fakeHub(answers(online))
  await saveManifest(legacy(await hubModelMeta(online), online))
  const refreshed = await resolveModel(online)
  assert.deepEqual(refreshed.supports.input, ["text", "image"])
  const saved = (await readManifest(online))?.files
  assert.ok(saved?.kind === "onnx" && saved.vision, "the refreshed meta is saved")
  // Offline: the stored meta loads text as before.
  const offline = "someone/Qwen3.5-legacy-offline"
  fakeHub(answers(offline))
  await saveManifest(legacy(await hubModelMeta(offline), offline))
  hub.fetch = async () => {
    throw new TypeError("offline")
  }
  const kept = await resolveModel(offline)
  assert.deepEqual(kept.supports.input, ["text"])
})
