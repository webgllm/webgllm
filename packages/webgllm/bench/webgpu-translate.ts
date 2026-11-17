/** Translates every kernel in shaders.jsonl (the last corpus) and compiles it with WebGPU. */
import { coopLoop, parseGlsl, toWgsl } from "../src/gpu/wgsl.js"

const out = document.querySelector("#out")!
const log = (s: string) => {
  out.textContent += s + "\n"
  void fetch("/log", { method: "POST", body: s })
}
const FORMATS: Record<number, GPUTextureFormat> = {
  0x8d70: "rgba32uint", 0x8814: "rgba32float", 0x881a: "rgba16float", 0x8058: "rgba8unorm",
  0x822e: "r32float", 0x8236: "r32uint", 0x8d7c: "rgba8uint", 0x8232: "r8uint"
}
try {
  const text = await (await fetch("/shaders.jsonl")).text()
  const corpus = JSON.parse(text.trim().split("\n").reduce((a, b) => (b.length > a.length ? b : a))) as Array<{ source: string; names: string[]; inputs: Record<string, number>; outputs: number[] }>
  const adapter = await navigator.gpu.requestAdapter()
  const device = await adapter!.requestDevice({ requiredLimits: { maxStorageTexturesPerShaderStage: adapter!.limits.maxStorageTexturesPerShaderStage } })
  let ok = 0
  let total = 0, coops = 0
  for (const [i, k] of corpus.entries()) for (const lanes of [1, 8]) {
    try {
      const shader = parseGlsl(k.source)
      if (lanes > 1 && !coopLoop(shader)) continue
      total++
      if (lanes > 1) coops++
      const t = toWgsl(shader, {
        outputs: k.outputs.map((f) => FORMATS[f]),
        rgba4: Object.entries(k.inputs).filter(([, f]) => f === 0x8056).map(([n]) => n),
        workgroup: [64, 1],
        unroll: 64,
        lanes
      })
      const module = device.createShaderModule({ code: t.code })
      const info = await module.getCompilationInfo()
      const errors = info.messages.filter((m) => m.type === "error")
      if (errors.length) {
        const lines = t.code.split("\n")
        log(`#${i} ${k.names} lanes ${lanes} FAIL: ${errors.map((e) => `${e.lineNum}: ${e.message} | ${lines[e.lineNum - 1]?.trim()}`).join("\n   ")}`)
      } else ok++
    } catch (e) {
      log(`#${i} ${k.names} lanes ${lanes} THROW: ${(e as Error).message}`)
    }
  }
  log(`TRANSLATE ${ok}/${total} compile (${coops} with lanes)`)
} catch (e) {
  log(`ERROR: ${(e as Error).stack}`)
}
log("ALL DONE")
