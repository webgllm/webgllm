/**
 * Memory and speed of the transformers-js package's worker on one prompt:
 * loads a model with debug on, reads a chat prompt padded to about ?len=
 * tokens, writes ?max= tokens greedily, and prints the worker's memory log
 * with a summary per phase (GPU buffers at the load peak, after the prompt,
 * at the reply's peak and end; WebAssembly memory; speeds).
 *
 *   ?model=onnx-community/Qwen3.5-2B-ONNX&len=2048&max=128&ctx=8192&wait=0&runs=1
 *
 * The prompt is counted with the model's own tokenizer, fetched from
 * Hugging Face as the worker fetches its files.
 */
import { loadTokenizer } from "../src/tokenizer.js"

const params = new URLSearchParams(location.search)
const out = document.getElementById("out") as HTMLPreElement
const log = (text: string) => {
  out.textContent += `${text}\n`
  void fetch("/log", { method: "POST", body: text }).catch(() => undefined)
}

const FILLER =
  "The river runs past the old mill, where the miller's daughter counts sacks of grain each morning before the market opens in the square. "
const QUESTION = "Describe the river scene above in a short paragraph."

/** A user message of about `length` prompt tokens, chat template included. */
async function promptText(model: string, length: number) {
  const file = (name: string) =>
    fetch(`https://huggingface.co/${model}/resolve/main/${name}`, {
      referrerPolicy: "no-referrer"
    })
  const [tokenizerJson, tokenizerConfig, template] = await Promise.all([
    file("tokenizer.json").then((r) => r.json() as Promise<object>),
    file("tokenizer_config.json").then(
      (r) => r.json() as Promise<Record<string, unknown>>
    ),
    file("chat_template.jinja").then((r) => (r.ok ? r.text() : undefined))
  ])
  const tokenizer = loadTokenizer({
    tokenizerJson,
    tokenizerConfig,
    chatTemplate: template
  })
  const count = (pad: string) =>
    tokenizer.encodeChat([{ role: "user", content: pad + QUESTION }], {
      addGenerationPrompt: true,
      enableThinking: false
    }).length
  const base = count("")
  if (length <= base + 4) return QUESTION
  const per = count(FILLER) - base
  let pad = FILLER.repeat(Math.max(0, Math.floor((length - base) / per)))
  while (count(pad) < length) pad += FILLER
  return `${pad}\n\n${QUESTION}`
}

/** Peaks and levels from the worker's "[memory]" lines, by phase (MB). */
const summary = {
  loadPeak: 0,
  loaded: 0,
  promptPeak: 0,
  afterPrompt: 0,
  replyPeak: 0,
  end: 0,
  wasm: 0
}

function readMemoryLine(text: string) {
  const m =
    /^\[memory\] (.*?): wasm (\d+) MB, GPU buffers (\d+) MB \(.*?peak (\d+) MB since the last line/.exec(
      text
    )
  if (!m) return
  const [, step, wasm, live, peak] = m
  const mb = { live: Number(live), peak: Number(peak) }
  summary.wasm = Math.max(summary.wasm, Number(wasm))
  if (step.startsWith("loaded")) {
    summary.loadPeak = mb.peak
    summary.loaded = mb.live
  } else if (step.startsWith("prompt read")) {
    summary.promptPeak = Math.max(summary.promptPeak, mb.peak)
    summary.afterPrompt = mb.live
  } else if (step.startsWith("reply")) {
    summary.replyPeak = Math.max(summary.replyPeak, mb.peak)
    summary.end = mb.live
  }
}

async function main() {
  log(navigator.userAgent)
  const model = params.get("model") ?? "onnx-community/Qwen3.5-2B-ONNX"
  const len = Number(params.get("len") ?? 64)
  const max = Number(params.get("max") ?? 64)
  const contextLength = Number(params.get("ctx") ?? 8192)
  const text = await promptText(model, len)
  const worker = new Worker("/bundle-tjs/src/worker.js", {
    type: "module",
    name: "transformers-js:debug"
  })
  let reply = ""
  type Stats = Record<string, number | null>
  let ready: () => void = () => undefined
  let failed: (error: Error) => void = () => undefined
  let done: (stats: Stats) => void = () => undefined
  let lastPercent = -10
  worker.onerror = (e) => log(`ERROR worker: ${e.message}`)
  worker.onmessage = ({ data }) => {
    if (data.type === "claim") worker.postMessage({ type: "claimed" })
    else if (data.type === "progress") {
      const percent = Math.floor(data.progress * 100)
      if (percent >= lastPercent + 10 || (percent === 0 && lastPercent > 0)) {
        lastPercent = percent
        log(
          `loading ${percent}% (${data.cached ? "from storage" : "downloading"})`
        )
      }
    } else if (data.type === "ready") ready()
    else if (data.type === "load-error") failed(new Error(data.message))
    else if (data.type === "log") {
      log(`[worker] ${data.text}`)
      readMemoryLine(data.text)
    } else if (data.type === "token") reply += data.text
    else if (data.type === "done") done(data.stats)
    else if (data.type === "error") failed(new Error(data.message))
  }
  const began = performance.now()
  await new Promise<void>((resolve, reject) => {
    ready = resolve
    failed = reject
    worker.postMessage({
      type: "load",
      modelId: model,
      dtype: "q4f16",
      contextLength
    })
  })
  log(`LOADED in ${((performance.now() - began) / 1000).toFixed(1)} s`)
  const wait = Number(params.get("wait") ?? 0)
  if (wait > 0) await new Promise((r) => setTimeout(r, wait))
  const runs = Number(params.get("runs") ?? 1)
  for (let i = 0; i < runs; i++) {
    reply = ""
    const stats = await new Promise<Stats>((resolve, reject) => {
      done = resolve
      failed = reject
      worker.postMessage({
        type: "generate",
        id: `run${i}`,
        chat: [{ role: "user", content: i === 0 ? text : `Again. ${text}` }],
        tools: [],
        thinking: false,
        sampling: { temperature: 0, top_k: 1, repetition_penalty: 1 },
        maxTokens: max,
        contextLength,
        thinkingOpen: "<think>"
      })
    })
    const read = (stats.inputTokens ?? 0) - (stats.cachedTokens ?? 0)
    const promptRate = (read / (stats.promptMs ?? 1)) * 1000
    const writeRate =
      (((stats.outputTokens ?? 1) - 1) / (stats.decodeMs || 1)) * 1000
    log(
      `RESULT run ${i}: ${stats.inputTokens} prompt tokens (${read} read) at ${promptRate.toFixed(1)} tok/s; ${stats.outputTokens} tokens written at ${writeRate.toFixed(1)} tok/s`
    )
    log(`REPLY ${JSON.stringify(reply)}`)
  }
  const s = summary
  log(
    `SUMMARY GPU buffers (MB): load peak ${s.loadPeak}, loaded ${s.loaded}; prompt peak ${s.promptPeak}, after it ${s.afterPrompt}; reply peak ${s.replyPeak}, end ${s.end}. WebAssembly ${s.wasm} MB`
  )
  worker.terminate()
}

main().then(
  () => log("ALL DONE"),
  (error: unknown) =>
    log(
      `ERROR: ${error instanceof Error ? `${error.message}\n${error.stack}` : String(error)}\nALL DONE`
    )
)
