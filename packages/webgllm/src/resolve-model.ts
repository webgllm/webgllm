/**
 * Any Hugging Face model by name: a catalog model's meta as the catalog has
 * it, any other repository's built from the Hub (its file list, base model,
 * architecture, chat template and generation config), each by one general
 * rule. Resolved metas are kept for the thread's sync lookups (modelMeta).
 *
 * The rules:
 * - Weights: a `.gguf` file (not a vision projector, not a split file) makes
 *   a GGUF model: Q4_0 first (a quantization-aware "QAD" Q4_0 before a plain
 *   one), then Q8_0, then the fewest bits among the other types the loader
 *   takes (Q4_1, Q5_K, Q6_K, F16), then unlabeled files. Otherwise an ONNX
 *   decoder graph under `onnx/` (`model*.onnx` or `decoder_model_merged*.onnx`)
 *   with exactly one external data file (`<graph>_data`): q4f16 first, then
 *   q4, then the rest.
 * - Architecture: the Hub's GGUF architecture, or config.json's model_type,
 *   must be one the loaders run, else the model fails before any download.
 * - Tokenizer and chat template: tokenizer.json and tokenizer_config.json
 *   (and chat_template.jinja when there is one) at the repository's root;
 *   else in a folder named in the chosen weights file's name (Liquid AI's
 *   `qad/` for its QAD GGUFs); else at the root of the base model's
 *   repository (the first, if the card names several).
 * - Thinking: a template that reads `enable_thinking` can switch it off; one
 *   that opens `<think>` for the reply, or a model whose name, base model or
 *   tags say "thinking" or "reasoning", always thinks; else none.
 * - Tool calls: `<tool_call>` in the template means the XML format (LFM3's
 *   when the template reads a message's `thinking`, else Qwen3.5's); any
 *   other template takes LFM2's Pythonic calls. Tools are offered when the
 *   template reads `tools`.
 * - Sampling: what Transformers' generate() does with the repository's
 *   generation_config.json: sampling with its fields (and Transformers'
 *   defaults for the rest) when it sets do_sample, else greedy decoding.
 * - Size: the name's parameter count ("1.2B", "350M"), else the Hub's.
 */
import { WebGLLMError } from "./errors.js"
import { knownModelMeta, rememberModel } from "./model-meta.js"
import type { ModelFile, ModelReasoning, WebGlModelMeta } from "./model-meta.js"
import type { Sampling } from "./sampler.js"
import type { ToolFormat } from "./tool-calls.js"

const HUB = "https://huggingface.co"

/** GGUF architectures the GGUF loader runs (load-gguf.ts). */
export const GGUF_ARCHITECTURES = ["lfm2", "lfm3", "qwen35"] as const
/** config.json model types the ONNX loader runs (load-onnx.ts). */
export const ONNX_MODEL_TYPES = ["lfm2", "lfm3", "qwen3_5"] as const

/** What the tests swap: the fetch the Hub is asked with. */
export const hub = { fetch: (input: string, init?: RequestInit) => fetch(input, init) }

/** The fields of the Hub's `/api/models/{repo}` this reads. */
type HubModel = {
  id?: string
  siblings?: Array<{ rfilename: string }>
  tags?: Array<string>
  cardData?: { base_model?: string | Array<string> }
  gguf?: { architecture?: string; total?: number }
  config?: { model_type?: string }
  safetensors?: { total?: number }
}

const notSupported = (message: string) => new WebGLLMError("model-not-supported", message)

/** A Hugging Face model id: `owner/name`. */
const MODEL_ID = /^[\w.-]+\/[\w.-]+$/

const pending = new Map<string, Promise<WebGlModelMeta>>()

/**
 * `model`'s meta: the catalog's, one resolved before, or one built from the
 * Hub now (kept for modelMeta). Rejects with "model-not-supported" for a
 * name that isn't a Hugging Face repository webgllm can run, and
 * "download-failed" when the Hub can't be reached.
 */
export function resolveModel(model: unknown): Promise<WebGlModelMeta> {
  if (typeof model !== "string" || !model) return Promise.reject(new TypeError("model is required."))
  const known = knownModelMeta(model)
  if (known) return Promise.resolve(known)
  if (!MODEL_ID.test(model)) {
    return Promise.reject(notSupported(`${model} isn't a model webgllm runs: give a catalog model (see models.list()) or a Hugging Face repository as owner/name.`))
  }
  let promise = pending.get(model)
  if (!promise) {
    promise = hubModelMeta(model).then((meta) => {
      rememberModel(meta)
      return meta
    })
    pending.set(model, promise)
    // A failed resolution is tried again next time.
    promise.catch(() => pending.delete(model))
  }
  return promise
}

async function hubGet(url: string, what: string): Promise<Response> {
  let response: Response
  try {
    response = await hub.fetch(url, { referrerPolicy: "no-referrer" })
  } catch (error) {
    throw new WebGLLMError("download-failed", `Couldn't reach Hugging Face for ${what}: ${error instanceof Error ? error.message : String(error)}`, { cause: error })
  }
  return response
}

async function apiModel(repo: string): Promise<HubModel> {
  const response = await hubGet(`${HUB}/api/models/${repo}`, repo)
  if (response.status === 401 || response.status === 403 || response.status === 404) {
    throw notSupported(`${repo} isn't a model webgllm can download: there's no public Hugging Face repository by that name (or it's gated).`)
  }
  if (!response.ok) throw new WebGLLMError("download-failed", `Hugging Face answered ${response.status} for ${repo}.`)
  return (await response.json()) as HubModel
}

async function fileText(file: ModelFile): Promise<string | undefined> {
  const response = await hubGet(`${HUB}/${file.repo}/resolve/main/${file.path}`, `${file.repo}/${file.path}`)
  if (!response.ok) {
    await response.body?.cancel()
    return undefined
  }
  return response.text()
}

async function fileJson(file: ModelFile): Promise<Record<string, unknown> | undefined> {
  const text = await fileText(file)
  if (text === undefined) return undefined
  try {
    const value: unknown = JSON.parse(text)
    return typeof value === "object" && value !== null ? (value as Record<string, unknown>) : undefined
  } catch {
    return undefined
  }
}

const filesOf = (info: HubModel) => new Set((info.siblings ?? []).map((s) => s.rfilename))

/** GGUF types by preference (see the rules above); a file's type is the label ending its name. */
const GGUF_TYPES = ["Q4_0", "Q8_0", "Q4_1", "Q5_K_S", "Q5_K_M", "Q6_K", "F16"]

/** The GGUF file to load, or undefined when the repository has no loadable one. */
export function pickGguf(files: Iterable<string>): string | undefined {
  const ggufs = [...files].filter(
    (f) => f.toLowerCase().endsWith(".gguf") && !/mmproj/i.test(f) && !/-\d{5}-of-\d{5}\.gguf$/i.test(f)
  )
  const label = (name: string, pattern: string) => new RegExp(`(^|[-_.])(${pattern})($|[-_.])`, "i").test(name)
  const rank = (file: string) => {
    const name = file.split("/").pop()!.replace(/\.gguf$/i, "")
    const type = GGUF_TYPES.findIndex((t) => label(name, t))
    if (type < 0) {
      // A labeled type the loader can't take (Q4_K_M, IQ4_XS, Q5_0, BF16…) never loads.
      return label(name, "IQ\\d\\w*|Q\\d_K(_[A-Z]+)?|Q\\d_\\d|BF16") ? Infinity : GGUF_TYPES.length * 2
    }
    // A quantization-aware build of a type before the plain one.
    return type * 2 + (label(name, "QAD") ? 0 : 1)
  }
  const ranked = ggufs.map((f) => ({ f, r: rank(f) })).filter((x) => x.r < Infinity)
  ranked.sort((a, b) => a.r - b.r || a.f.localeCompare(b.f))
  return ranked[0]?.f
}

/** ONNX quantizations by preference; the graph name's suffix (`model_q4f16.onnx`). */
const ONNX_QUANTS = ["q4f16", "q4"]

/** The ONNX graph to load (with its one data file), or undefined. */
export function pickOnnx(files: Set<string>): { graph: string; data: string } | undefined {
  const graphs = [...files].filter(
    (f) =>
      /^onnx\/(model|decoder_model_merged)(_\w+)?\.onnx$/.test(f) &&
      files.has(`${f}_data`) &&
      !files.has(`${f}_data_1`)
  )
  const quant = (f: string) => /_(\w+)\.onnx$/.exec(f.replace(/^onnx\/decoder_model_merged/, "onnx/model"))?.[1] ?? ""
  const rank = (f: string) => {
    const i = ONNX_QUANTS.indexOf(quant(f))
    return i < 0 ? ONNX_QUANTS.length : i
  }
  graphs.sort((a, b) => rank(a) - rank(b) || a.localeCompare(b))
  const graph = graphs[0]
  return graph ? { graph, data: `${graph}_data` } : undefined
}

/** Where the tokenizer files are: a repository and a folder in it ("" or "qad/"). */
type TokenizerSource = { repo: string; prefix: string; files: Set<string> }

const hasTokenizer = (files: Set<string>, prefix: string) =>
  files.has(`${prefix}tokenizer.json`) && files.has(`${prefix}tokenizer_config.json`)

async function tokenizerSource(repo: string, info: HubModel, files: Set<string>, weights: string): Promise<TokenizerSource> {
  if (hasTokenizer(files, "")) return { repo, prefix: "", files }
  // A folder named in the weights file's name: `qad/` for `…-QAD-Q4_0.gguf`.
  const words = new Set(weights.split("/").pop()!.toLowerCase().split(/[-_.]/))
  for (const file of files) {
    const folder = /^([^/]+)\/tokenizer\.json$/.exec(file)?.[1]
    if (folder && words.has(folder.toLowerCase()) && hasTokenizer(files, `${folder}/`)) {
      return { repo, prefix: `${folder}/`, files }
    }
  }
  const base = info.cardData?.base_model
  const baseRepo = Array.isArray(base) ? base[0] : base
  if (typeof baseRepo === "string" && MODEL_ID.test(baseRepo)) {
    const baseFiles = filesOf(await apiModel(baseRepo))
    if (hasTokenizer(baseFiles, "")) return { repo: baseRepo, prefix: "", files: baseFiles }
  }
  throw notSupported(`${repo} has no tokenizer webgllm can read: no tokenizer.json and tokenizer_config.json in it or in its base model's repository.`)
}

const word = (pattern: string) => new RegExp(`(^|[^a-z])(${pattern})($|[^a-z])`, "i")

/** Whether the template opens `<think>` (left open) in its generation prompt. */
export function opensThinking(template: string): boolean {
  const at = template.lastIndexOf("add_generation_prompt")
  if (at < 0) return false
  const tail = template.slice(at)
  for (let i = tail.indexOf("<think>"); i >= 0; i = tail.indexOf("<think>", i + 1)) {
    const after = tail.slice(i + "<think>".length).replace(/^(\\n|\s)*/, "")
    if (!after.startsWith("</think>")) return true
  }
  return false
}

/** The model's thinking, from its chat template and its names (see the rules above). */
export function reasoningFrom(template: string, labels: Array<string>): ModelReasoning | undefined {
  if (/\benable_thinking\b/.test(template)) return { levels: { off: "false", high: "true" }, budget: true }
  if (opensThinking(template) || labels.some((l) => word("thinking|reasoning").test(l))) {
    return { levels: { high: "true" }, budget: true }
  }
  return undefined
}

/** How the model writes tool calls, from its chat template. */
export function toolFormatFrom(template: string): ToolFormat {
  if (!template.includes("<tool_call>")) return "lfm2"
  return /\bmessage\.thinking\b|message\[["']thinking["']\]/.test(template) ? "lfm3" : "qwen3.5"
}

/** Sampling as Transformers' generate() reads generation_config.json. */
export function samplingFrom(config: Record<string, unknown> | undefined): Sampling {
  const num = (key: string) => (typeof config?.[key] === "number" ? (config[key] as number) : undefined)
  const repetition_penalty = num("repetition_penalty") ?? 1
  if (config?.do_sample !== true) return { temperature: 0, top_k: 1, repetition_penalty }
  const top_p = num("top_p")
  const min_p = num("min_p")
  const presence_penalty = num("presence_penalty")
  return {
    temperature: num("temperature") ?? 1,
    top_k: num("top_k") ?? 50,
    repetition_penalty,
    ...(top_p !== undefined && { top_p }),
    ...(min_p !== undefined && { min_p }),
    ...(presence_penalty !== undefined && { presence_penalty })
  }
}

/** "1.2b" or "350m": the name's parameter count, else the Hub's; "" if neither says. */
export function sizeFrom(name: string, parameters: number | undefined): string {
  const named = /(?:^|[-_])(\d+(?:\.\d+)?)([bm])(?=$|[-_])/i.exec(name.split("/").pop() ?? "")
  if (named) return `${named[1]}${named[2].toLowerCase()}`
  if (!parameters) return ""
  if (parameters < 1e9) return `${Math.round(parameters / 1e6)}m`
  return `${Number((parameters / 1e9).toFixed(1))}b`
}

/** The chat template's text: chat_template.jinja's, else tokenizer_config.json's. */
async function templateText(tokenizer: WebGlModelMeta["tokenizer"]): Promise<string | undefined> {
  if (tokenizer.chatTemplate) {
    const text = await fileText(tokenizer.chatTemplate)
    if (text !== undefined) return text
  }
  const template = (await fileJson(tokenizer.config))?.chat_template
  if (typeof template === "string") return template
  // Several named templates: the default one, else the first.
  if (Array.isArray(template)) {
    const named = template as Array<{ name?: string; template?: string }>
    return (named.find((t) => t.name === "default") ?? named[0])?.template
  }
  return undefined
}

/**
 * `repo`'s meta built from the Hub, by the rules above (no catalog lookup:
 * resolveModel does that).
 */
export async function hubModelMeta(repo: string): Promise<WebGlModelMeta> {
  const info = await apiModel(repo)
  const files = filesOf(info)
  const gguf = pickGguf(files)
  let weights: WebGlModelMeta["files"]
  let weightsFile: string
  if (gguf) {
    const arch = info.gguf?.architecture
    if (arch && !(GGUF_ARCHITECTURES as ReadonlyArray<string>).includes(arch)) {
      throw notSupported(`${repo} is a "${arch}" model; webgllm runs ${GGUF_ARCHITECTURES.join(", ")} GGUFs.`)
    }
    weights = { kind: "gguf", model: { repo, path: gguf } }
    weightsFile = gguf
  } else {
    const onnx = pickOnnx(files)
    if (!onnx || !files.has("config.json")) {
      throw notSupported(
        `${repo} has no weights webgllm loads: no GGUF file of a type it reads (Q4_0, Q8_0, Q4_1, Q5_K, Q6_K, F16), and no ONNX graph under onnx/ with one external data file and a config.json.`
      )
    }
    const type = info.config?.model_type ?? (await fileJson({ repo, path: "config.json" }))?.model_type
    if (typeof type !== "string" || !(ONNX_MODEL_TYPES as ReadonlyArray<string>).includes(type)) {
      throw notSupported(`${repo} is a "${String(type)}" model; webgllm runs ${ONNX_MODEL_TYPES.join(", ")} ONNX models.`)
    }
    weights = {
      kind: "onnx",
      graph: { repo, path: onnx.graph },
      data: { repo, path: onnx.data },
      config: { repo, path: "config.json" }
    }
    weightsFile = onnx.graph
  }
  const source = await tokenizerSource(repo, info, files, weightsFile)
  const at = (path: string): ModelFile => ({ repo: source.repo, path: `${source.prefix}${path}` })
  const tokenizer: WebGlModelMeta["tokenizer"] = {
    tokenizer: at("tokenizer.json"),
    config: at("tokenizer_config.json"),
    ...(source.files.has(`${source.prefix}chat_template.jinja`) && { chatTemplate: at("chat_template.jinja") })
  }
  const generation = files.has("generation_config.json")
    ? { repo, path: "generation_config.json" }
    : source.files.has(`${source.prefix}generation_config.json`)
      ? at("generation_config.json")
      : undefined
  const [template, generationConfig] = await Promise.all([
    templateText(tokenizer),
    generation ? fileJson(generation) : Promise.resolve(undefined)
  ])
  if (template === undefined) throw notSupported(`${repo} has no chat template (chat_template.jinja or tokenizer_config.json's).`)
  const base = info.cardData?.base_model
  const labels = [repo, ...(Array.isArray(base) ? base : base ? [base] : []), ...(info.tags ?? [])]
  const reasoning = reasoningFrom(template, labels)
  const tools = /\btools\b/.test(template)
  const capabilities = [...(tools ? (["tools"] as const) : []), ...(reasoning ? (["thinking"] as const) : [])]
  return {
    name: repo,
    format: weights.kind,
    files: weights,
    tokenizer,
    supports: { input: ["text"], output: ["text"], capabilities },
    size: sizeFrom(repo, info.gguf?.total ?? info.safetensors?.total),
    ...(reasoning && { reasoning }),
    sampling: samplingFrom(generationConfig),
    toolFormat: toolFormatFrom(template)
  }
}
