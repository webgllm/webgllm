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
 *   graph under `onnx/` that the ONNX loader takes, read before anything
 *   else downloads: its own checks (4-bit MatMulNBits in blocks of 32, the
 *   layer layout, config.json's shapes) pass, and its weights are in one
 *   external data file, which the graph names. Graphs are tried q4f16 first,
 *   then q4, then the rest (decoders before other graphs).
 * - Architecture: the Hub's GGUF architecture, or config.json's model_type,
 *   must be one the loaders run, else the model fails before any download.
 * - A model whose files were downloaded resolves from its manifest in storage
 *   (storage.ts), with no request to the Hub.
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
 *
 * A model can also be given by URL: its `.gguf` file, or its ONNX graph
 * (`.onnx`, with config.json and its one external data file beside it), and
 * tokenizer.json and tokenizer_config.json (and chat_template.jinja and
 * generation_config.json, when there are) in the same folder. The same
 * rules apply to what's there.
 */
import { chatTemplateOf } from "./chat-template.js"
import { abortable, throwIfAborted, WebGLLMError } from "./errors.js"
import { checkOnnxGraph } from "./load-onnx.js"
import { planVision } from "./load-vision.js"
import { fileUrl, inputsOf, knownModelMeta, rememberModel } from "./model-meta.js"
import type { ModelFile, ModelReasoning, VisionFiles, WebGlModelMeta } from "./model-meta.js"
import type { Sampling } from "./sampler.js"
import { readManifest } from "./storage.js"
import type { ToolFormat } from "./tool-calls.js"

const HUB = "https://huggingface.co"

/** GGUF architectures the GGUF loader runs (load-gguf.ts). */
export const GGUF_ARCHITECTURES = ["lfm2", "lfm3", "qwen35"] as const
/** config.json model types the ONNX loader runs (load-onnx.ts). */
export const ONNX_MODEL_TYPES = ["lfm2", "lfm3", "qwen3_5"] as const

/** config.json `vision_config.model_type`s the vision encoder runs (load-vision.ts). */
export const VISION_MODEL_TYPES = ["qwen3_5"] as const

/** What the tests swap: the fetch the Hub is asked with, and the ONNX loaders' graph checks. */
export const hub = {
  fetch: (input: string, init?: RequestInit) => fetch(input, init),
  checkGraph: checkOnnxGraph,
  /** The vision encoder graph's external data files (throws for one load-vision.ts doesn't take). */
  checkVisionGraph: (bytes: Uint8Array): Array<string> => [
    ...new Set(planVision(bytes).reads.flatMap((r) => (r.tensor.external ? [r.tensor.external.location] : [])))
  ]
}

/**
 * A model's vision encoder, by one rule: its config has a vision_config of
 * a type the encoder runs, and beside the decoder graph (`<folder>/<name>
 * <suffix>.onnx`) is `<folder>/vision_encoder<suffix>.onnx`, a graph the
 * encoder loads, with its one data file, and a preprocessor_config.json.
 * `has` says whether a file is there (a repository listing; for a URL
 * model, whether it can be fetched).
 */
export async function visionFilesFor(
  at: (path: string) => ModelFile,
  graph: string,
  config: Record<string, unknown>,
  has: (path: string) => boolean | Promise<boolean>
): Promise<VisionFiles | undefined> {
  const type = (config.vision_config as { model_type?: unknown } | undefined)?.model_type
  if (typeof type !== "string" || !(VISION_MODEL_TYPES as ReadonlyArray<string>).includes(type)) return undefined
  const slash = graph.lastIndexOf("/") + 1
  const suffix = /(_[a-z0-9]+)?\.onnx$/.exec(graph.slice(slash))?.[1] ?? ""
  const visionGraph = `${graph.slice(0, slash)}vision_encoder${suffix}.onnx`
  const preprocessor = "preprocessor_config.json"
  if (!(await has(visionGraph)) || !(await has(preprocessor))) return undefined
  const bytes = await fileBytes(at(visionGraph))
  if (!bytes) return undefined
  let dataFiles: Array<string>
  try {
    dataFiles = hub.checkVisionGraph(bytes)
  } catch {
    return undefined
  }
  const data = `${graph.slice(0, slash)}${dataFiles[0]}`
  if (dataFiles.length !== 1 || !(await has(data))) return undefined
  return { graph: at(visionGraph), data: at(data), preprocessor: at(preprocessor) }
}

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
/** A model's weights by URL: a `.gguf` file or an `.onnx` graph. */
const MODEL_URL = /^https?:\/\/\S+\/[^/\s]+\.(?:gguf|onnx)$/i

const pending = new Map<string, Promise<WebGlModelMeta>>()

/** Whether `model` is a name the resolver can look up (a catalog model or `owner/name`). */
export const isModelId = (model: unknown): model is string =>
  typeof model === "string" && (!!knownModelMeta(model) || MODEL_ID.test(model) || MODEL_URL.test(model))

/**
 * `model`'s meta: the catalog's, one resolved before, a downloaded model's
 * manifest, or one built from the Hub now (kept for modelMeta). Rejects
 * with "model-not-supported" for a name that isn't a Hugging Face
 * repository webgllm can run, and "download-failed" when the Hub can't be
 * reached. Aborting `signal` stops this caller's wait (nothing is asked if
 * it's aborted already); a resolution other callers wait for goes on.
 */
export function resolveModel(model: unknown, signal?: AbortSignal): Promise<WebGlModelMeta> {
  try {
    throwIfAborted(signal)
  } catch (error) {
    return Promise.reject(error)
  }
  if (typeof model !== "string" || !model) return Promise.reject(new TypeError("model is required."))
  const known = knownModelMeta(model)
  if (known) return Promise.resolve(known)
  const url = MODEL_URL.test(model)
  if (!url && !MODEL_ID.test(model)) {
    return Promise.reject(
      notSupported(`${model} isn't a model webgllm runs: give a catalog model (see models.list()), a Hugging Face repository as owner/name, or a URL to a .gguf or .onnx file.`)
    )
  }
  let promise = pending.get(model)
  if (!promise) {
    promise = (async () => (await readManifest(model)) ?? (await (url ? urlModelMeta(model) : hubModelMeta(model))))().then((meta) => {
      rememberModel(meta)
      return meta
    })
    pending.set(model, promise)
    // A failed resolution is tried again next time.
    promise.catch(() => pending.delete(model))
  }
  return abortable(promise, signal)
}

async function hubGet(url: string, what: string, method?: "HEAD"): Promise<Response> {
  let response: Response
  try {
    response = await hub.fetch(url, { referrerPolicy: "no-referrer", ...(method && { method }) })
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

/**
 * A file from the Hub, or undefined when it isn't there (404, the Hub's
 * "entry not found"). Any other failure (a 5xx, a network error, another
 * status) throws "download-failed": not an answer, so it isn't kept and
 * the next resolution asks again.
 */
async function fileResponse(file: ModelFile, method?: "HEAD"): Promise<Response | undefined> {
  const url = fileUrl(file)
  const response = await hubGet(url, url, method)
  if (response.ok) return response
  await response.body?.cancel()
  if (response.status === 404) return undefined
  throw new WebGLLMError("download-failed", `The server answered ${response.status} for ${url}.`)
}

async function fileText(file: ModelFile): Promise<string | undefined> {
  return (await fileResponse(file))?.text()
}

async function fileBytes(file: ModelFile): Promise<Uint8Array | undefined> {
  const response = await fileResponse(file)
  return response ? new Uint8Array(await response.arrayBuffer()) : undefined
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

/**
 * The ONNX graphs to try, in order: q4f16, then q4, then the rest, and
 * within each, decoders (`model*`, `decoder_model_merged*`) before other
 * graphs. Which one loads is decided by reading it (onnxWeights).
 */
export function onnxCandidates(files: Iterable<string>): Array<string> {
  const graphs = [...files].filter((f) => /^onnx\/.+\.onnx$/.test(f))
  const quant = (f: string) => /_([a-z0-9]+)\.onnx$/.exec(f)?.[1] ?? ""
  const rank = (f: string) => {
    const q = ONNX_QUANTS.indexOf(quant(f))
    const decoder = /^onnx\/(model|decoder_model_merged)(_[a-z0-9]+)?\.onnx$/.test(f) ? 0 : 1
    return (q < 0 ? ONNX_QUANTS.length : q) * 2 + decoder
  }
  return graphs.sort((a, b) => rank(a) - rank(b) || a.localeCompare(b))
}

/**
 * The first graph the ONNX loader takes, and its one external data file,
 * reading only graphs (small) and config.json: never a data file.
 */
async function onnxWeights(repo: string, files: Set<string>, config: Record<string, unknown>): Promise<{ graph: string; data: string }> {
  let reason = "none under onnx/"
  for (const graph of onnxCandidates(files)) {
    const bytes = await fileBytes({ repo, path: graph })
    if (!bytes) continue
    let dataFiles: Array<string>
    try {
      dataFiles = hub.checkGraph(bytes, config).dataFiles
    } catch (error) {
      reason = `${graph}: ${error instanceof Error ? error.message : String(error)}`
      continue
    }
    const data = `${graph.slice(0, graph.lastIndexOf("/") + 1)}${dataFiles[0]}`
    if (dataFiles.length !== 1) reason = `${graph} keeps its weights in ${dataFiles.length} files; webgllm reads one`
    else if (!files.has(data)) reason = `${graph}'s weights are in ${data}, which the repository doesn't have`
    else return { graph, data }
  }
  throw notSupported(`${repo} has no ONNX graph webgllm loads (${reason}).`)
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
  if (/\benable_thinking\b/.test(template)) return { levels: { off: "none", high: "high" }, budget: true }
  if (opensThinking(template) || labels.some((l) => word("thinking|reasoning").test(l))) {
    return { levels: { high: "high" }, budget: true }
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

/**
 * The chat template the worker will render (chatTemplateOf, as the
 * tokenizer picks it), and the tokenizer files without a listed
 * chat_template.jinja that turned out not to be there.
 */
async function templateOf(tokenizer: WebGlModelMeta["tokenizer"]): Promise<{ template: string | undefined; tokenizer: WebGlModelMeta["tokenizer"] }> {
  if (tokenizer.chatTemplate) {
    const file = await fileText(tokenizer.chatTemplate)
    if (file !== undefined) return { template: file, tokenizer }
  }
  const { chatTemplate: _, ...rest } = tokenizer
  return { template: chatTemplateOf(await fileJson(tokenizer.config)), tokenizer: rest }
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
    const config = files.has("config.json") && onnxCandidates(files).length ? await fileJson({ repo, path: "config.json" }) : undefined
    if (!config) {
      throw notSupported(
        `${repo} has no weights webgllm loads: no GGUF file of a type it reads (Q4_0, Q8_0, Q4_1, Q5_K, Q6_K, F16), and no ONNX graph under onnx/ with a config.json.`
      )
    }
    const type = info.config?.model_type ?? config.model_type
    if (typeof type !== "string" || !(ONNX_MODEL_TYPES as ReadonlyArray<string>).includes(type)) {
      throw notSupported(`${repo} is a "${String(type)}" model; webgllm runs ${ONNX_MODEL_TYPES.join(", ")} ONNX models.`)
    }
    const onnx = await onnxWeights(repo, files, config)
    const vision = await visionFilesFor((path) => ({ repo, path }), onnx.graph, config, (path) => files.has(path))
    weights = {
      kind: "onnx",
      graph: { repo, path: onnx.graph },
      data: { repo, path: onnx.data },
      config: { repo, path: "config.json" },
      ...(vision && { vision })
    }
    weightsFile = onnx.graph
  }
  const source = await tokenizerSource(repo, info, files, weightsFile)
  const at = (path: string): ModelFile => ({ repo: source.repo, path: `${source.prefix}${path}` })
  const listed: WebGlModelMeta["tokenizer"] = {
    tokenizer: at("tokenizer.json"),
    config: at("tokenizer_config.json"),
    ...(source.files.has(`${source.prefix}chat_template.jinja`) && { chatTemplate: at("chat_template.jinja") })
  }
  const generation = files.has("generation_config.json")
    ? { repo, path: "generation_config.json" }
    : source.files.has(`${source.prefix}generation_config.json`)
      ? at("generation_config.json")
      : undefined
  const base = info.cardData?.base_model
  return metaFrom({
    name: repo,
    weights,
    listed,
    generation,
    labels: [repo, ...(Array.isArray(base) ? base : base ? [base] : []), ...(info.tags ?? [])],
    parameters: info.gguf?.total ?? info.safetensors?.total
  })
}

/**
 * The model given by URL (see the header): its weights file, and the
 * tokenizer and config files beside it, by the same rules.
 */
export async function urlModelMeta(url: string): Promise<WebGlModelMeta> {
  const slash = url.lastIndexOf("/")
  const file = url.slice(slash + 1)
  const here = (path: string): ModelFile => ({ repo: url.slice(0, slash), path })
  let weights: WebGlModelMeta["files"]
  if (/\.gguf$/i.test(file)) {
    weights = { kind: "gguf", model: here(file) }
  } else {
    const config = await fileJson(here("config.json"))
    if (!config) throw notSupported(`${url} has no config.json beside it.`)
    const type = config.model_type
    if (typeof type !== "string" || !(ONNX_MODEL_TYPES as ReadonlyArray<string>).includes(type)) {
      throw notSupported(`${url} is a "${String(type)}" model; webgllm runs ${ONNX_MODEL_TYPES.join(", ")} ONNX models.`)
    }
    const bytes = await fileBytes(here(file))
    if (!bytes) throw notSupported(`${url} isn't there.`)
    let dataFiles: Array<string>
    try {
      dataFiles = hub.checkGraph(bytes, config).dataFiles
    } catch (error) {
      throw notSupported(`${url} isn't a graph webgllm loads: ${error instanceof Error ? error.message : String(error)}`)
    }
    if (dataFiles.length !== 1) throw notSupported(`${url} keeps its weights in ${dataFiles.length} files; webgllm reads one.`)
    const vision = await visionFilesFor(here, file, config, async (path) => (await fileResponse(here(path), "HEAD")) !== undefined)
    weights = { kind: "onnx", graph: here(file), data: here(dataFiles[0]!), config: here("config.json"), ...(vision && { vision }) }
  }
  return metaFrom({
    name: url,
    weights,
    listed: { tokenizer: here("tokenizer.json"), config: here("tokenizer_config.json"), chatTemplate: here("chat_template.jinja") },
    generation: here("generation_config.json"),
    labels: [url],
    parameters: undefined
  })
}

/** A model's meta from its weights and tokenizer files: its template, thinking, tools and sampling. */
async function metaFrom(model: {
  name: string
  weights: WebGlModelMeta["files"]
  listed: WebGlModelMeta["tokenizer"]
  generation: ModelFile | undefined
  labels: Array<string>
  parameters: number | undefined
}): Promise<WebGlModelMeta> {
  const [{ template, tokenizer }, generationConfig] = await Promise.all([
    templateOf(model.listed),
    model.generation ? fileJson(model.generation) : Promise.resolve(undefined)
  ])
  if (template === undefined) throw notSupported(`${model.name} has no chat template (chat_template.jinja or tokenizer_config.json's).`)
  const reasoning = reasoningFrom(template, model.labels)
  const tools = /\btools\b/.test(template)
  const capabilities = [...(tools ? (["tools"] as const) : []), ...(reasoning ? (["thinking"] as const) : [])]
  return {
    name: model.name,
    format: model.weights.kind,
    files: model.weights,
    tokenizer,
    supports: { input: inputsOf(model.weights), output: ["text"], capabilities },
    size: sizeFrom(model.name, model.parameters),
    ...(reasoning && { reasoning }),
    sampling: samplingFrom(generationConfig),
    toolFormat: toolFormatFrom(template)
  }
}
