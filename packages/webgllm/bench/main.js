/**
 * The go/no-go page: the WebGL 2 Q4_0 matmul (gpu-bench-worker.js) against
 * wllama's relaxed-SIMD CPU build reading the same number of tokens with
 * LFM3 1.7B Q4_K_M. `?run=gpu` or `?run=cpu` runs one on load; results go
 * to the page and to the dev server (POST /log), which prints them.
 */
const params = new URLSearchParams(location.search)
const out = document.getElementById("out")

function log(text) {
  out.textContent += `${text}\n`
  console.log(text)
  void fetch("/log", { method: "POST", body: text }).catch(() => undefined)
}

function report(data) {
  void fetch("/result", { method: "POST", body: JSON.stringify(data) }).catch(
    () => undefined
  )
}

let wakeLock = null
async function keepAwake() {
  try {
    wakeLock = await navigator.wakeLock?.request("screen")
  } catch {
    // Not offered: the phone may dim, which slows it down.
  }
}

function runGpu() {
  const M = Number(params.get("m") ?? 128)
  log(`--- GPU: WebGL 2 Q4_0 matmul, ${M} tokens, in a worker`)
  return new Promise((resolve) => {
    const worker = new Worker(
      new URL("./gpu-bench-worker.js", import.meta.url),
      {
        type: "module"
      }
    )
    worker.onmessage = ({ data }) => {
      if (data.type === "log") log(data.text)
      else if (data.type === "result") report(data.data)
      else if (data.type === "error") {
        log(`GPU error: ${data.message}`)
        worker.terminate()
        resolve()
      } else if (data.type === "done") {
        log("--- GPU done")
        worker.terminate()
        resolve()
      }
    }
    worker.onerror = (e) => {
      log(`GPU worker error: ${e.message}`)
      resolve()
    }
    worker.postMessage({ type: "run", M })
  })
}

/** About `n` tokens of plain English, different for each `seed`. */
function promptText(words, seed) {
  const vocab =
    "the quick brown fox jumps over a lazy dog while seven small birds sing about rivers mountains cities and long summer evenings near quiet harbors".split(
      " "
    )
  let s = seed * 7919
  const out = [`Note ${seed}:`]
  for (let i = 0; i < words; i++) {
    s = (s * 1103515245 + 12345) % 2147483648
    out.push(vocab[s % vocab.length])
  }
  return out.join(" ")
}

async function runCpu() {
  // Served by bench/server.mjs from the workspace's @wllama/wllama.
  const wllamaUrl = "/wllama/index.js"
  const { Wllama, LoggerWithoutDebug } = await import(wllamaUrl)
  const model = params.get("model") ?? "LFM3-1.7B-Q4_K_M.gguf"
  const wasm = params.get("wasm") ?? "wllama-relaxed.wasm"
  log(
    `--- CPU: wllama ${wasm}, ${model}, threads ${Math.floor((navigator.hardwareConcurrency || 1) / 2)} of ${navigator.hardwareConcurrency}, isolated ${crossOriginIsolated}`
  )
  const wllama = new Wllama(
    { default: `/wasm/${wasm}` },
    { logger: LoggerWithoutDebug, suppressNativeLog: true }
  )
  let lastPct = -10
  const t0 = performance.now()
  await wllama.loadModelFromUrl(`/models/${model}`, {
    n_ctx: 2048,
    n_gpu_layers: 0,
    swa_full: true,
    progressCallback: ({ loaded, total }) => {
      const pct = Math.floor((loaded / total) * 100)
      if (pct >= lastPct + 10) {
        lastPct = pct
        log(`loading ${pct}%`)
      }
    }
  })
  log(`loaded in ${((performance.now() - t0) / 1000).toFixed(1)} s`)

  const rows = []
  // Words per prompt: about 1.3 tokens each for this text.
  for (const [words, label] of [
    [96, "~128"],
    [96, "~128"],
    [96, "~128"],
    [390, "~512"]
  ]) {
    const seed = rows.length + 1
    const res = await wllama.createCompletion({
      prompt: promptText(words, seed),
      max_tokens: 1,
      temperature: 0,
      cache_prompt: false
    })
    const t = res.timings
    const tps = (t.prompt_n / t.prompt_ms) * 1000
    rows.push({ prompt_n: t.prompt_n, prompt_ms: t.prompt_ms, tps })
    log(
      `CPU prompt ${label}: ${t.prompt_n} tokens in ${t.prompt_ms.toFixed(0)} ms = ${tps.toFixed(1)} tokens/s`
    )
  }
  report({
    kind: "cpu",
    model,
    wasm,
    threads: Math.floor((navigator.hardwareConcurrency || 1) / 2),
    rows
  })
  await wllama.exit()
  log("--- CPU done")
}

document.getElementById("gpu").onclick = () => void runGpu()
document.getElementById("cpu").onclick = () => void runCpu()

log(`${navigator.userAgent}`)
log(`crossOriginIsolated: ${crossOriginIsolated}`)
const run = params.get("run")
if (run) {
  void keepAwake().then(async () => {
    try {
      if (run === "gpu" || run === "all") await runGpu()
      if (run === "cpu" || run === "all") await runCpu()
    } catch (error) {
      log(`error: ${error?.stack ?? error}`)
    }
    log("ALL DONE")
    void wakeLock?.release()
  })
}
