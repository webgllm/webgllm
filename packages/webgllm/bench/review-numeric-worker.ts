/** Bench-only logits/state-copy review. No model answer is sampled or generated. */
import { blobSource } from "../src/byte-source.js"
import { loadGguf } from "../src/load-gguf.js"
import { loadOnnx } from "../src/load-onnx.js"
import { fileUrl, modelMeta } from "../src/model-meta.js"
import type { WebGlTextModel } from "../src/model-meta.js"
import { download } from "../src/storage.js"
import { loadTokenizer } from "../src/tokenizer.js"
import type { ChatMessage, Tokenizer } from "../src/tokenizer.js"
import type { Engine } from "../src/engine.js"

type Variant = "baseline" | "exact" | "memo"
type CapturedRequest = {
  id: string
  originalSystem: string
  chat: ChatMessage[]
  outputSchema: object
  source: unknown
  recordedPromptTokens?: number
  schemaHash: string
  chatHash: string
}
type RunRequest = {
  type: "run"
  model: WebGlTextModel
  requests: CapturedRequest[]
  provenance: Record<string, unknown>
}
type Runtime = {
  Engine: new (canvas: OffscreenCanvas, options: { batchSize: number }) => Engine
  TUNING: Record<string, unknown>
  readPrompt: (engine: Engine, prompt: number[], start: number, cached: number[], options?: { prefix?: number; anchor?: number; onForward?: (count: number) => void }) => Promise<number>
  anchorFor: (tokenizer: Tokenizer, prompt: number[], chat: ChatMessage[], thinking: boolean) => number | undefined
  systemPrefixFor?: (tokenizer: Tokenizer, prompt: number[], chat: ChatMessage[], thinking: boolean, tools?: undefined, previous?: number[]) => number | undefined
}
type Snapshot = { conv?: { bytes: number }[]; windows?: { bytes: number }[]; linear?: { bytes: number }[]; hidden?: { bytes: number }; anchor?: boolean; prefix?: boolean }
type Tokenized = CapturedRequest & { ids: number[]; renderedHash: string; tokenHash: string }
const emit = (type: string, value: Record<string, unknown>) => postMessage({ type, ...value })
const progress = (message: string) => emit("status", { message })
const shared = (a: number[], b: number[]) => { let i = 0; while (i < a.length && i < b.length && a[i] === b[i]) i++; return i }
const hash = async (value: string | Uint8Array) => {
  const bytes = typeof value === "string" ? new TextEncoder().encode(value) : value
  const digest = await crypto.subtle.digest("SHA-256", bytes as BufferSource)
  return Array.from(new Uint8Array(digest), byte => byte.toString(16).padStart(2, "0")).join("")
}
function snapshotState(engine: Engine) {
  const snapshots = Reflect.get(engine, "snapshots") as Map<number, Snapshot>
  if (!(snapshots instanceof Map)) throw new Error("Cannot inspect this review variant's snapshot accounting.")
  const textures = new Set<{ bytes: number }>()
  const points = [...snapshots].map(([position, saved]) => {
    const owned = [...(saved.conv ?? []), ...(saved.windows ?? []), ...(saved.linear ?? []), ...(saved.hidden ? [saved.hidden] : [])]
    owned.forEach(texture => textures.add(texture))
    return { position, anchor: !!saved.anchor, prefix: !!saved.prefix, bytes: owned.reduce((sum, texture) => sum + texture.bytes, 0) }
  })
  return { count: snapshots.size, limit: engine.snapshotLimit, bytes: [...textures].reduce((sum, texture) => sum + texture.bytes, 0), allocatedBytes: engine.gpu.bytes, points }
}
function instrument(engine: Engine) {
  let peakBytes = engine.gpu.bytes, maxSnapshots = 0, snapshotWrites = 0
  const gpu = engine.gpu as unknown as { texture: (...args: unknown[]) => { bytes: number }; bytes: number }
  const allocate = gpu.texture.bind(gpu)
  gpu.texture = (...args) => { const result = allocate(...args); peakBytes = Math.max(peakBytes, gpu.bytes); return result }
  const save = engine.snapshot.bind(engine)
  engine.snapshot = (...args) => { save(...args); snapshotWrites++; maxSnapshots = Math.max(maxSnapshots, snapshotState(engine).count) }
  return () => ({ peakAllocatedBytes: peakBytes, maxSnapshots, snapshotWrites })
}
const top1 = (values: Float32Array) => { let at = 0; for (let i = 1; i < values.length; i++) if (values[i] > values[at]) at = i; return at }
function probabilities(values: Float32Array) {
  let maximum = -Infinity
  for (const value of values) maximum = Math.max(maximum, value)
  const output = Float64Array.from(values, value => Math.exp(value - maximum))
  const total = output.reduce((sum, value) => sum + value, 0)
  return output.map(value => value / total)
}
function compare(reference: Float32Array, tested: Float32Array) {
  if (reference.length !== tested.length) throw new Error("Logit vocabulary lengths differ.")
  let maxAbs = 0
  for (let i = 0; i < reference.length; i++) {
    if (reference[i] === tested[i]) continue
    maxAbs = Math.max(maxAbs, Math.abs(reference[i] - tested[i]))
  }
  const p = probabilities(reference), q = probabilities(tested)
  let kl = 0
  for (let i = 0; i < p.length; i++) if (p[i] > 0) kl += p[i] * Math.log(p[i] / Math.max(q[i], Number.MIN_VALUE))
  if (!Number.isFinite(maxAbs) || !Number.isFinite(kl)) throw new Error("Non-finite logits or numerical differences.")
  const a = new Uint32Array(reference.buffer, reference.byteOffset, reference.length)
  const b = new Uint32Array(tested.buffer, tested.byteOffset, tested.length)
  const bitwiseEqual = a.every((value, index) => value === b[index])
  const referenceTop1 = top1(reference), testedTop1 = top1(tested)
  return { maxAbs, kl, bitwiseEqual, referenceTop1, testedTop1, top1Equal: referenceTop1 === testedTop1 }
}
async function assets(model: WebGlTextModel) {
  const meta = modelMeta(model), tokenizerFiles = meta.tokenizer
  const weights = meta.files.kind === "gguf" ? [meta.files.model] : [meta.files.config, meta.files.graph, meta.files.data]
  const descriptors = [tokenizerFiles.tokenizer, tokenizerFiles.config, ...(tokenizerFiles.chatTemplate ? [tokenizerFiles.chatTemplate] : []), ...weights]
  const files = await Promise.all(descriptors.map(file => download(file, () => undefined)))
  const [json, config, ...rest] = files
  const template = tokenizerFiles.chatTemplate ? rest.shift() : undefined
  const tokenizer = loadTokenizer({ tokenizerJson: JSON.parse(await json.text()) as object, tokenizerConfig: JSON.parse(await config.text()) as Record<string, unknown>, ...(template ? { chatTemplate: await template.text() } : {}) })
  return { meta, tokenizer, weights: rest, fileProvenance: descriptors.map((file, index) => ({ ...file, url: fileUrl(file), cachedFileName: files[index].name, bytes: files[index].size, lastModified: files[index].lastModified })) }
}
async function runtime(variant: Variant): Promise<Runtime> {
  // Generated, frozen review copies; no production module is patched by this harness.
  const enginePath = "../.review-cache-qa/" + variant + "/engine.ts"
  const promptPath = "../.review-cache-qa/" + variant + "/prompt.ts"
  const engineUrl = new URL(enginePath, import.meta.url).href
  const promptUrl = new URL(promptPath, import.meta.url).href
  const engine = await import(/* @vite-ignore */ engineUrl)
  const prompt = await import(/* @vite-ignore */ promptUrl)
  return { ...engine, ...prompt } as Runtime
}
function prefixFor(module: Runtime, tokenizer: Tokenizer, request: Tokenized, variant: Variant, previous?: number[]) {
  if (!module.systemPrefixFor) return undefined
  // The exact-prefix experiment saves the normalized caller instructions before
  // the changing schema. Only prefix discovery sees this copy; actual chat/tokens
  // remain the exact SDK request supplied to every numerical variant.
  const chat = variant === "exact" ? request.chat.map((message, index) => index === 0 ? { ...message, content: request.originalSystem + "\n\nAnswer with only a JSON value matching this JSON Schema:\n" } : message) : request.chat
  return module.systemPrefixFor(tokenizer, request.ids, chat, false, undefined, previous)
}
/** No snapshots are written while this function disturbs or reconstructs state. */
async function feedWithoutSnapshots(engine: Engine, ids: number[], start: number) {
  let at = start, row = 0
  for (const size of engine.batchSizes(ids.length - start)) {
    row = await engine.forwardHidden(ids.slice(at, at + size))
    at += size
  }
  return row
}
async function matchedCopyControl(engine: Engine, first: number[], second: number[], boundary: number, measurements: () => { snapshotWrites: number }) {
  if (boundary === 0) return { skipped: true, reason: "Production selected restore 0; there is no nonzero checkpoint-copy path to isolate." }
  if (boundary > shared(first, second)) throw new Error("Control boundary exceeds token-verified common prefix.")
  engine.reset()
  await feedWithoutSnapshots(engine, second.slice(0, boundary), 0)
  engine.snapshot(undefined, true)
  const referenceRow = await feedWithoutSnapshots(engine, second, boundary)
  const reference = await engine.logits(referenceRow)
  engine.reset()
  await feedWithoutSnapshots(engine, first.slice(0, boundary), 0)
  engine.snapshot(undefined, true)
  const before = snapshotState(engine), writes = measurements().snapshotWrites
  await feedWithoutSnapshots(engine, first, boundary)
  const afterDisturb = snapshotState(engine), afterWrites = measurements().snapshotWrites
  const restored = engine.rewind(boundary)
  if (restored !== boundary) throw new Error(`Matched control lost checkpoint ${boundary}: restored ${restored}.`)
  const checkedRow = await feedWithoutSnapshots(engine, second, restored)
  const checked = await engine.logits(checkedRow)
  const difference = compare(reference, checked)
  return { skipped: false, boundary, restored, segmentation: "identical prefix and identical second tail; no snapshots during disturbance", snapshotWritesDuringDisturbance: afterWrites - writes, beforeDisturb: before, afterDisturb, ...difference, pass: difference.bitwiseEqual && difference.maxAbs === 0 && difference.top1Equal && afterWrites === writes }
}
async function pair(module: Runtime, engine: Engine, tokenizer: Tokenizer, variant: Variant, first: Tokenized, second: Tokenized, measurements: () => { snapshotWrites: number }, baselineCold: Map<string, Float32Array>) {
  const common = shared(first.ids, second.ids)
  const coldPrefix = prefixFor(module, tokenizer, second, variant, [])
  const firstAnchor = module.anchorFor(tokenizer, first.ids, first.chat, false)
  const secondAnchor = module.anchorFor(tokenizer, second.ids, second.chat, false)
  const coldBatches: number[] = [], firstBatches: number[] = [], warmBatches: number[] = []
  // A full system boundary may legitimately exceed common. Eligibility is decided
  // by actual saved checkpoint positions, never assumed from a prefix label.
  engine.reset()
  const coldStart = performance.now()
  const coldRow = await module.readPrompt(engine, second.ids, 0, [], { prefix: coldPrefix, anchor: secondAnchor, onForward: count => coldBatches.push(count) })
  const cold = await engine.logits(coldRow)
  const referenceKey = `${first.chatHash}:${second.chatHash}`
  if (variant === "baseline") baselineCold.set(referenceKey, cold.slice())
  const baselineReference = baselineCold.get(referenceKey)
  if (!baselineReference) throw new Error("Baseline cold reference is missing for this request pair.")
  const coldMs = performance.now() - coldStart, coldSnapshots = snapshotState(engine)
  engine.reset()
  const cached: number[] = []
  const firstPrefix = prefixFor(module, tokenizer, first, variant, [])
  await module.readPrompt(engine, first.ids, 0, cached, { prefix: firstPrefix, anchor: firstAnchor, onForward: count => firstBatches.push(count) })
  const secondPrefix = prefixFor(module, tokenizer, second, variant, first.ids)
  const firstSnapshots = snapshotState(engine)
  const expectedRestore = Math.max(0, ...firstSnapshots.points.map(point => point.position).filter(position => position <= Math.min(common, second.ids.length - 1)))
  const restored = engine.rewind(Math.min(common, second.ids.length - 1))
  if (restored !== expectedRestore) throw new Error(`Saved checkpoint selection mismatch: expected ${expectedRestore}, got ${restored}.`)
  cached.length = restored
  const afterRestore = snapshotState(engine), warmStart = performance.now()
  const warmRow = await module.readPrompt(engine, second.ids, restored, cached, { prefix: secondPrefix, anchor: secondAnchor, onForward: count => warmBatches.push(count) })
  const warm = await engine.logits(warmRow)
  const warmMs = performance.now() - warmStart, warmSnapshots = snapshotState(engine)
  const difference = compare(cold, warm)
  const crossVariantColdVsWarm = compare(baselineReference, warm)
  const crossVariantColdVsCold = compare(baselineReference, cold)
  const copyControl = await matchedCopyControl(engine, first.ids, second.ids, restored, measurements)
  engine.reset()
  return { firstId: first.id, secondId: second.id, firstSource: first.source, secondSource: second.source, commonTokens: common, firstTokens: first.ids.length, secondTokens: second.ids.length, coldPrefix, firstPrefix, secondPrefix, firstAnchor, secondAnchor, coldBatches, firstBatches, warmBatches, expectedRestore, restoredTokens: restored, coldMs, warmMs, timingScope: "prefill plus logits readback; not end-to-end generation latency", ...difference, crossVariantReference: "baseline cold second-request logits", crossVariantColdVsWarm, crossVariantColdVsCold, copyControl, coldSnapshots, firstSnapshots, afterRestore, warmSnapshots, afterReset: snapshotState(engine), top1Text: { cold: tokenizer.idToToken(difference.referenceTop1), warm: tokenizer.idToToken(difference.testedTop1) } }
}
let running = false
self.addEventListener("message", async ({ data }: MessageEvent<RunRequest>) => {
  if (data.type !== "run" || running) return
  running = true
  const report: Record<string, unknown> = { schemaVersion: 1, startedAt: new Date().toISOString(), model: data.model, provenance: data.provenance, settings: { generation: "none; logits only", sampling: "not used or overridden", batchSize: 512, contextLength: "model default", snapshotLimit: "unchanged engine default" }, variants: [], error: null }
  try {
    if (data.requests.length < 2) throw new Error("The replay fixture needs at least two requests.")
    progress("Loading cached tokenizer and model-file handles…")
    const got = await assets(data.model)
    report.files = got.fileProvenance
    const tokenized: Tokenized[] = []
    for (const request of data.requests) {
      const rendered = got.tokenizer.applyChatTemplate(request.chat, { addGenerationPrompt: true, enableThinking: false })
      const ids = got.tokenizer.encodeChat(request.chat, { addGenerationPrompt: true, enableThinking: false })
      tokenized.push({ ...request, ids, renderedHash: await hash(rendered), tokenHash: await hash(JSON.stringify(ids)) })
    }
    report.inputs = tokenized.map(request => ({ id: request.id, source: request.source, schemaHash: request.schemaHash, chatHash: request.chatHash, renderedHash: request.renderedHash, tokenHash: request.tokenHash, tokens: request.ids.length, recordedPromptTokens: request.recordedPromptTokens }))
    report.historicalSanity = { expectedFirstTokens: 2214, actualFirstTokens: tokenized[0].ids.length, matches: tokenized[0].ids.length === 2214, note: "Provenance check only; prompts are never padded or edited to force this number." }
    const candidates = tokenized.slice(1).map((request, index) => ({ first: index, second: index + 1, common: shared(tokenized[index].ids, request.ids) }))
    report.adjacentCommonPrefixes = candidates
    const selected = [candidates[0]]
    const later = candidates.find(candidate => candidate.first > 0 && candidate.common === 319)
    if (later) selected.push(later)
    report.selectedPairs = selected
    report.later319Control = later ? "selected" : "not present among adjacent fixture requests; first pair only"
    // CPU-only reference vectors stay local and are never serialized into the report.
    const baselineCold = new Map<string, Float32Array>()
    for (const variant of ["baseline", "exact", "memo"] as const) {
      progress(`${variant}: loading one engine with unchanged defaults…`)
      const module = await runtime(variant), tuningHash = await hash(JSON.stringify(module.TUNING))
      const engine = new module.Engine(new OffscreenCanvas(1, 1), { batchSize: 512 })
      const measurements = instrument(engine)
      const result: Record<string, unknown> = { variant, tuningHash, pairs: [], inputHashes: tokenized.map(request => ({ id: request.id, schemaHash: request.schemaHash, chatHash: request.chatHash, tokenHash: request.tokenHash })) }
      try {
        const loadStarted = performance.now()
        const pieces = got.meta.files.kind === "gguf" ? loadGguf(blobSource(got.weights[0])) : loadOnnx({ config: JSON.parse(await got.weights[0].text()) as Record<string, unknown>, graph: blobSource(got.weights[1]), data: blobSource(got.weights[2]) })
        await engine.load(pieces)
        await engine.warmUp()
        result.loadAndWarmupMs = performance.now() - loadStarted
        result.actualBatchSize = engine.batchSize
        result.actualContextLength = engine.contextLength
        result.snapshotLimit = engine.snapshotLimit
        result.loadedBytes = engine.gpu.bytes
        for (const selectedPair of selected) {
          progress(`${variant}: logits comparison ${selectedPair.first + 1} → ${selectedPair.second + 1}…`)
          const checked = await pair(module, engine, got.tokenizer, variant, tokenized[selectedPair.first], tokenized[selectedPair.second], measurements, baselineCold)
          ;(result.pairs as unknown[]).push(checked)
          emit("result", { variant, result: checked })
        }
        for (const request of tokenized) {
          if (await hash(JSON.stringify(request.chat)) !== request.chatHash || await hash(JSON.stringify(request.outputSchema)) !== request.schemaHash || await hash(JSON.stringify(request.ids)) !== request.tokenHash) throw new Error(`${variant} mutated an input request or token array.`)
        }
        result.inputHashesVerifiedAfterRun = true
        Object.assign(result, measurements())
      } finally {
        engine.gpu.dispose()
        result.bytesAfterDispose = engine.gpu.bytes
      }
      ;(report.variants as unknown[]).push(result)
    }
    report.finishedAt = new Date().toISOString()
    emit("done", { report })
  } catch (error) {
    report.error = error instanceof Error ? { message: error.message, stack: error.stack } : String(error)
    report.finishedAt = new Date().toISOString()
    emit("error", { report })
  } finally { running = false }
})
