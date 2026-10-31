/**
 * The other runtimes on the same prompt as check.html's smoke test, for
 * comparison: Transformers.js on WebGPU (the transformers-js package's own
 * worker, driven through its messages) and wllama with every layer on
 * WebGPU, or each on the CPU in WebAssembly. `?run=tjs`, `tjs-cpu`, `wllama`
 * or `wllama-cpu`; results go to the server's log.
 */
import { loadTokenizer } from "../src/tokenizer.js"

const params = new URLSearchParams(location.search)
const out = document.getElementById("out") as HTMLPreElement
const log = (text: string) => {
  out.textContent += `${text}\n`
  void fetch("/log", { method: "POST", body: text }).catch(() => undefined)
}

const QUESTION = "Describe the river scene above in a short paragraph."
const FILLER =
  "The river runs past the old mill, where the miller's daughter counts sacks of grain each morning before the market opens in the square. "

/** The smoke test's prompt: the question after filler, about `length` tokens in all. */
async function promptText(length: number) {
  const [tokenizerJson, tokenizerConfig, chatTemplate] = await Promise.all([
    fetch("/hf/tokenizer.json").then((r) => r.json() as Promise<object>),
    fetch("/hf/tokenizer_config.json").then(
      (r) => r.json() as Promise<Record<string, unknown>>
    ),
    fetch("/hf/chat_template.jinja").then((r) => r.text())
  ])
  const tokenizer = loadTokenizer({
    tokenizerJson,
    tokenizerConfig,
    chatTemplate
  })
  let pad = ""
  while (
    tokenizer.encodeChat([{ role: "user", content: pad + QUESTION }], {
      addGenerationPrompt: true
    }).length < length
  ) {
    pad += FILLER
  }
  return `${pad}\n\n${QUESTION}`
}

const median = (values: Array<number>) =>
  [...values].sort((a, b) => a - b)[Math.floor(values.length / 2)]

function summarize(
  name: string,
  runs: Array<{ promptTps: number; decodeTps: number; ttftMs: number }>
) {
  log(
    `RESULT ${name}: prompt ${median(runs.map((r) => r.promptTps)).toFixed(1)} tokens/s, ` +
      `writing ${median(runs.map((r) => r.decodeTps)).toFixed(1)} tokens/s, ` +
      `first token ${(median(runs.map((r) => r.ttftMs)) / 1000).toFixed(2)} s (median of ${runs.length})`
  )
}

/** Transformers.js on WebGPU, through the transformers-js package's worker. */
async function runTjs(text: string, cpu = false) {
  const worker = new Worker(
    cpu ? "/bundle/bench/tjs-cpu-worker.js" : "/bundle-tjs/src/worker.js",
    { type: "module", name: "transformers-js" }
  )
  let lastPct = -10
  const pending = new Map<
    string,
    {
      tokens: number
      resolve: (stats: Record<string, number | null>) => void
      reject: (e: Error) => void
    }
  >()
  let ready: () => void = () => undefined
  let failed: (e: Error) => void = () => undefined
  worker.onmessage = ({ data }) => {
    if (data.type === "progress") {
      const pct = Math.floor(data.progress * 100)
      if (pct >= lastPct + 10) {
        lastPct = pct
        log(`loading ${pct}% (${data.cached ? "from storage" : "downloading"})`)
      }
    } else if (data.type === "ready") ready()
    else if (data.type === "load-error") failed(new Error(data.message))
    else if (data.type === "log") log(`[worker] ${data.text}`)
    else if (data.type === "token") {
      const p = pending.get(data.id)
      if (p) p.tokens++
    } else if (data.type === "done") pending.get(data.id)?.resolve(data.stats)
    else if (data.type === "error")
      pending.get(data.id)?.reject(new Error(data.message))
  }
  const model = params.get("model") ?? "onnx-community/LFM3-1.7B-ONNX"
  const t0 = performance.now()
  await new Promise<void>((resolve, reject) => {
    ready = resolve
    failed = reject
    worker.postMessage({ type: "load", modelId: model, dtype: cpu ? "q4" : "q4f16" })
  })
  log(
    `Transformers.js loaded ${model} in ${((performance.now() - t0) / 1000).toFixed(1)} s`
  )
  let n = 0
  const generate = (content: string, maxTokens: number) =>
    new Promise<Record<string, number | null>>((resolve, reject) => {
      const id = `run${n++}`
      pending.set(id, { tokens: 0, resolve, reject })
      worker.postMessage({
        type: "generate",
        id,
        chat: [{ role: "user", content }],
        tools: [],
        thinking: false,
        sampling: { temperature: 0, top_k: 1, repetition_penalty: 1.1 },
        maxTokens,
        contextLength: 8192,
        thinkingOpen: "<think>"
      })
    })
  await generate("Hi", 4)
  const runs = []
  for (let i = 0; i < 3; i++) {
    // A different first line each time, so no run reuses another's cache.
    const wall0 = performance.now()
    const s = await generate(`Run ${i + 1} of the benchmark.\n\n${text}`, 64)
    const wall = performance.now() - wall0
    const input = (s.inputTokens ?? 0) - (s.cachedTokens ?? 0)
    const promptTps = (input / (s.promptMs ?? 1)) * 1000
    const decodeTps = (((s.outputTokens ?? 1) - 1) / (s.decodeMs ?? 1)) * 1000
    log(
      `  stopwatch: whole reply ${wall.toFixed(0)} ms for ${s.outputTokens} tokens; minus the prompt, ${(((s.outputTokens ?? 1) - 1) / ((wall - (s.promptMs ?? 0)) / 1000)).toFixed(1)} tokens/s`
    )
    log(
      `Transformers.js ${cpu ? "WASM" : "WebGPU"} run ${i + 1}: ${input} prompt tokens in ${(s.promptMs ?? 0).toFixed(0)} ms (${promptTps.toFixed(1)} tokens/s), ${s.outputTokens} tokens at ${decodeTps.toFixed(1)} tokens/s`
    )
    runs.push({ promptTps, decodeTps, ttftMs: s.promptMs ?? 0 })
  }
  summarize(
    cpu ? "Transformers.js WASM (CPU), onnx-community ONNX q4" : "Transformers.js WebGPU, onnx-community ONNX q4f16",
    runs
  )
}

/**
 * This package's WebGL 2 engine, through its worker as the app runs it
 * (with speculative decoding unless ?speculative=0), on the same prompt.
 */
async function runWebgl(text: string) {
  const worker = new Worker("/bundle/src/worker.js", { type: "module" })
  const model = params.get("model") ?? "onnx-community/LFM3-1.7B-ONNX"
  const speculative = params.get("speculative") !== "0"
  // ?tune=knob:value,… sets engine TUNING knobs in the worker.
  const tune = Object.fromEntries(
    (params.get("tune") ?? "")
      .split(",")
      .filter(Boolean)
      .map((pair) => {
        const [knob, value] = pair.split(":")
        return [knob, Number(value)]
      })
  )
  if (Object.keys(tune).length) {
    worker.postMessage({ type: "tune", values: tune })
    log(`tuning: ${JSON.stringify(tune)}`)
  }
  let lastPct = -10
  let ready: () => void = () => undefined
  let failed: (e: Error) => void = () => undefined
  const pending = new Map<
    string,
    { resolve: (s: Record<string, number | null>) => void; reject: (e: Error) => void }
  >()
  let reply = ""
  worker.onmessage = ({ data }) => {
    if (data.type === "progress") {
      const pct = Math.floor(data.progress * 100)
      if (pct >= lastPct + 10) {
        lastPct = pct
        log(`loading ${pct}% (${data.cached ? "from storage" : "downloading"})`)
      }
    } else if (data.type === "ready") ready()
    else if (data.type === "load-error") failed(new Error(data.message))
    else if (data.type === "token") reply += data.text
    else if (data.type === "done") pending.get(data.id)?.resolve(data.stats)
    else if (data.type === "error") pending.get(data.id)?.reject(new Error(data.message))
  }
  const t0 = performance.now()
  await new Promise<void>((resolve, reject) => {
    ready = resolve
    failed = reject
    worker.postMessage({ type: "load", model })
  })
  log(`WebGL loaded ${model} in ${((performance.now() - t0) / 1000).toFixed(1)} s`)
  let n = 0
  const generate = (content: string, maxTokens: number) =>
    new Promise<Record<string, number | null>>((resolve, reject) => {
      const id = `run${n++}`
      pending.set(id, { resolve, reject })
      reply = ""
      worker.postMessage({
        type: "generate",
        id,
        model,
        chat: [{ role: "user", content }],
        tools: [],
        thinking: false,
        sampling: { temperature: 0, top_k: 1, repetition_penalty: 1.1 },
        maxTokens,
        speculative
      })
    })
  await generate("Hi", 4)
  const runs = []
  for (let i = 0; i < 3; i++) {
    const s = await generate(`Run ${i + 1} of the benchmark.\n\n${text}`, 64)
    const input = (s.inputTokens ?? 0) - (s.cachedTokens ?? 0)
    const promptTps = (input / (s.promptMs ?? 1)) * 1000
    const decodeTps = (((s.outputTokens ?? 1) - 1) / (s.decodeMs ?? 1)) * 1000
    log(
      `WebGL run ${i + 1}: ${input} prompt tokens in ${(s.promptMs ?? 0).toFixed(0)} ms (${promptTps.toFixed(1)} tokens/s), ${s.outputTokens} tokens at ${decodeTps.toFixed(1)} tokens/s`
    )
    if (i === 0) log(`reply: ${JSON.stringify(reply.slice(0, 200))}`)
    runs.push({ promptTps, decodeTps, ttftMs: s.firstTokenMs ?? 0 })
  }
  summarize(`WebGL${speculative ? " (speculative)" : ""}, ${model}`, runs)
  worker.terminate()
}

/** wllama (llama.cpp in WebAssembly) with every layer offloaded to WebGPU. */
async function runWllama(text: string, gpu = true) {
  const url = "/wllama/index.js"
  const { Wllama, LoggerWithoutDebug } = (await import(
    /* @vite-ignore */ url
  )) as {
    Wllama: new (
      paths: Record<string, string>,
      config: Record<string, unknown>
    ) => {
      loadModelFromUrl: (
        url: string,
        options: Record<string, unknown>
      ) => Promise<void>
      createChatCompletion: (
        options: Record<string, unknown>
      ) => Promise<{
        timings?: Record<string, number>
        choices: Array<{ message: { content?: string } }>
      }>
      exit: () => Promise<void>
    }
    LoggerWithoutDebug: unknown
  }
  const model = params.get("model") ?? "LFM3-1.7B-QAD-Q4_0.gguf"
  const wasm = params.get("wasm") ?? "wllama-relaxed.wasm"
  const debug = params.get("debug") === "1"
  const wllama = new Wllama(
    { default: `/wasm/${wasm}` },
    debug
      ? { logger: console }
      : { logger: LoggerWithoutDebug, suppressNativeLog: true }
  )
  const t0 = performance.now()
  await wllama.loadModelFromUrl(`/models/${model}`, {
    n_ctx: 2048,
    n_gpu_layers: gpu ? 999 : 0,
    swa_full: true
  })
  log(
    `wllama loaded ${model} with n_gpu_layers ${gpu ? 999 : 0} in ${((performance.now() - t0) / 1000).toFixed(1)} s (${wasm})`
  )
  const generate = (content: string, maxTokens: number) =>
    wllama.createChatCompletion({
      messages: [{ role: "user", content }],
      max_tokens: maxTokens,
      temperature: 0,
      cache_prompt: false,
      chat_template_kwargs: { enable_thinking: false }
    })
  await generate("Hi", 4)
  const runs = []
  for (let i = 0; i < 3; i++) {
    const res = await generate(`Run ${i + 1} of the benchmark.\n\n${text}`, 64)
    const t = res.timings ?? {}
    const promptTps = (t.prompt_n / t.prompt_ms) * 1000
    const decodeTps = (t.predicted_n / t.predicted_ms) * 1000
    log(
      `wllama ${gpu ? "WebGPU" : "WASM"} run ${i + 1}: ${t.prompt_n} prompt tokens in ${t.prompt_ms?.toFixed(0)} ms (${promptTps.toFixed(1)} tokens/s), ${t.predicted_n} tokens at ${decodeTps.toFixed(1)} tokens/s`
    )
    if (i === 0)
      log(
        `reply: ${JSON.stringify((res.choices[0]?.message.content ?? "").slice(0, 160))}`
      )
    runs.push({ promptTps, decodeTps, ttftMs: t.prompt_ms })
  }
  summarize(`wllama ${gpu ? "WebGPU" : "WASM (CPU)"}, ${model}`, runs)
  await wllama.exit()
}

async function main() {
  log(navigator.userAgent)
  const text = await promptText(Number(params.get("len") ?? 512))
  const run = params.get("run")
  if (run === "tjs") await runTjs(text)
  else if (run === "tjs-cpu") await runTjs(text, true)
  else if (run === "wllama") await runWllama(text)
  else if (run === "wllama-cpu") await runWllama(text, false)
  else if (run === "webgl") await runWebgl(text)
  else log("Pass ?run=tjs, tjs-cpu, wllama or wllama-cpu.")
}

main().then(
  () => log("ALL DONE"),
  (error: unknown) =>
    log(
      `ERROR: ${error instanceof Error ? `${error.message}\n${error.stack}` : String(error)}\nALL DONE`
    )
)
