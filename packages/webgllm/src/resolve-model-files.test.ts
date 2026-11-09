import assert from "node:assert/strict"
import { readFileSync } from "node:fs"
import { join } from "node:path"
import { test } from "node:test"
import { WebGLLMError } from "./errors.js"
import { api, asRepo, code, fakeHub } from "./fake-hub.test.js"
import * as onnxLoader from "./load-onnx.js"
import { hub, resolveModel } from "./resolve-model.js"
import { loadTokenizer } from "./tokenizer.js"

// The files a resolution reads from the Hub besides its listing: ONNX
// graphs, chat templates, generation configs.

/** LFM2 350M's real ONNX graphs (no weights) and config.json. */
const fixture = (name: string) => new Uint8Array(readFileSync(join(process.cwd(), "src/test-fixtures/lfm2-350m-onnx", name)))

/** The ONNX loader's own graph check, in place of fakeHub's stand-in. */
const realGraphCheck = () => {
  hub.checkGraph = onnxLoader.checkOnnxGraph
}

test("an FP16-only ONNX export fails before its weights download", async () => {
  const name = "someone/LFM2-350M-fp16-ONNX"
  const calls = fakeHub({
    ...asRepo("onnx-community/LFM2-350M-ONNX", name),
    [api(name)]: {
      status: 200,
      body: {
        siblings: ["config.json", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "onnx/model_fp16.onnx", "onnx/model_fp16.onnx_data"].map((rfilename) => ({ rfilename })),
        config: { model_type: "lfm2" }
      }
    },
    [`https://huggingface.co/${name}/resolve/main/config.json`]: { status: 200, body: new TextDecoder().decode(fixture("config.json")) },
    [`https://huggingface.co/${name}/resolve/main/onnx/model_fp16.onnx`]: { status: 200, body: fixture("model_fp16.onnx") }
  })
  realGraphCheck()
  await assert.rejects(resolveModel(name), (error: unknown) => {
    assert.ok(error instanceof WebGLLMError)
    assert.equal(error.code, "model-not-supported")
    assert.match(error.message, /model_fp16\.onnx: ONNX: missing initializer .*_quant/)
    return true
  })
  assert.ok(!calls.some((url) => url.endsWith(".onnx_data")), "the weights weren't asked for")
})

test("an ONNX export is picked by its contents: an unfamiliar graph name, the data file the graph names", async () => {
  const name = "someone/LFM2-350M-renamed-ONNX"
  const calls = fakeHub({
    ...asRepo("onnx-community/LFM2-350M-ONNX", name),
    [api(name)]: {
      status: 200,
      body: {
        siblings: ["config.json", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "onnx/decoder-4bit.onnx", "onnx/model_q4f16.onnx_data"].map((rfilename) => ({ rfilename })),
        config: { model_type: "lfm2" }
      }
    },
    [`https://huggingface.co/${name}/resolve/main/config.json`]: { status: 200, body: new TextDecoder().decode(fixture("config.json")) },
    [`https://huggingface.co/${name}/resolve/main/onnx/decoder-4bit.onnx`]: { status: 200, body: fixture("model_q4f16.onnx") }
  })
  realGraphCheck()
  const meta = await resolveModel(name)
  assert.deepEqual(meta.files, {
    kind: "onnx",
    graph: { repo: name, path: "onnx/decoder-4bit.onnx" },
    data: { repo: name, path: "onnx/model_q4f16.onnx_data" },
    config: { repo: name, path: "config.json" }
  })
  assert.ok(!calls.some((url) => url.endsWith(".onnx_data")))
})

test("named chat templates: the resolver reads the one the tokenizer renders", async () => {
  const template = "{% for m in messages %}{{ m.content }}{% endfor %}{% if enable_thinking %}<think>{% endif %}{% if tools %}!{% endif %}"
  const named = [{ name: "tool_use", template: "{{ messages }}" }, { name: "default", template }]
  const name = "someone/LFM2-350M-named-template-GGUF"
  fakeHub({
    ...asRepo("LiquidAI/LFM2-350M-GGUF", name),
    [api("someone/named-template-base")]: {
      status: 200,
      body: { siblings: ["tokenizer.json", "tokenizer_config.json"].map((rfilename) => ({ rfilename })) }
    },
    [api(name)]: {
      status: 200,
      body: { siblings: [{ rfilename: "m-Q4_0.gguf" }], cardData: { base_model: "someone/named-template-base" }, gguf: { architecture: "lfm2" } }
    },
    ["https://huggingface.co/someone/named-template-base/resolve/main/tokenizer_config.json"]: { status: 200, body: { chat_template: named } }
  })
  const meta = await resolveModel(name)
  assert.equal(meta.tokenizer.chatTemplate, undefined)
  assert.deepEqual(meta.reasoning, { levels: { off: "false", high: "true" }, budget: true })
  assert.deepEqual(meta.supports.capabilities, ["tools", "thinking"])
  // The tokenizer renders the same template from the same tokenizer_config.json.
  const tokenizer = loadTokenizer({
    tokenizerJson: {
      model: { type: "BPE", vocab: { x: 0 }, merges: [], unk_token: "x" },
      normalizer: null,
      pre_tokenizer: null,
      post_processor: null,
      decoder: null,
      added_tokens: [
        { id: 9, content: "<|im_end|>", special: true, normalized: false },
        { id: 17, content: "<|pad|>", special: true, normalized: false }
      ]
    },
    tokenizerConfig: { chat_template: named, eos_token: "<|im_end|>", pad_token: "<|pad|>" }
  })
  assert.equal(tokenizer.applyChatTemplate([{ role: "user", content: "hi" }], { enableThinking: true, addGenerationPrompt: true }), "hi<think>")
})

test("a transient Hub error isn't kept: the next resolution reads generation_config.json again", async () => {
  const name = "someone/LFM2-350M-flaky-GGUF"
  let available = false
  const calls = fakeHub({
    ...asRepo("LiquidAI/LFM2-350M-GGUF", name),
    [api(name)]: {
      status: 200,
      body: {
        siblings: ["LFM2-350M-Q4_0.gguf", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "generation_config.json"].map((rfilename) => ({ rfilename })),
        gguf: { architecture: "lfm2" }
      }
    },
    [`https://huggingface.co/${name}/resolve/main/chat_template.jinja`]: { status: 200, body: "{{ tools }}" },
    [`https://huggingface.co/${name}/resolve/main/generation_config.json`]: () =>
      available ? { status: 200, body: { do_sample: true, temperature: 0.3, top_k: 40 } } : { status: 503, body: "busy" }
  })
  assert.equal(await code(resolveModel(name)), "download-failed")
  available = true
  const meta = await resolveModel(name)
  assert.deepEqual(meta.sampling, { temperature: 0.3, top_k: 40, repetition_penalty: 1 })
  assert.equal(calls.filter((url) => url.endsWith("generation_config.json")).length, 2)
})
