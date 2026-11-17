/** How long the dumped kernels take to compile as compute pipelines: created one by one (as a load's first passes do) or all at once asynchronously. */
import { parseGlsl, toWgsl } from "../src/gpu/wgsl.js"

const out = document.querySelector("#out")!
const log = (s: string) => {
  out.textContent += s + "\n"
  void fetch("/log", { method: "POST", body: s })
}
const FORMATS: Record<number, GPUTextureFormat> = {
  0x8d70: "rgba32uint", 0x8814: "rgba32float", 0x881a: "rgba16float", 0x8058: "rgba8unorm", 0x822e: "r32float", 0x8236: "r32uint"
}
try {
  const text = await (await fetch("/shaders.jsonl")).text()
  const corpus = JSON.parse(text.trim().split("\n").reduce((a, b) => (b.length > a.length ? b : a))) as Array<{ source: string; outputs: number[] }>
  const adapter = await navigator.gpu.requestAdapter()
  const device = await adapter!.requestDevice({ requiredLimits: { maxStorageTexturesPerShaderStage: adapter!.limits.maxStorageTexturesPerShaderStage, maxSampledTexturesPerShaderStage: adapter!.limits.maxSampledTexturesPerShaderStage } })
  const salt = Math.random().toString(36).slice(2)
  const codes = (tag: string) => corpus.map((k) => toWgsl(parseGlsl(k.source), { outputs: k.outputs.map((f) => FORMATS[f]), workgroup: [16, 4] }).code + `\n// ${tag} ${salt}`)
  // One by one: each pipeline used by a dispatch right away, as a load's warm-up does.
  const t0 = performance.now()
  for (const code of codes("sync")) {
    const p = device.createComputePipeline({ layout: "auto", compute: { module: device.createShaderModule({ code }), entryPoint: "main" } })
    void p
  }
  await device.queue.onSubmittedWorkDone()
  const sync = performance.now() - t0
  const t1 = performance.now()
  await Promise.all(codes("async").map((code) => device.createComputePipelineAsync({ layout: "auto", compute: { module: device.createShaderModule({ code }), entryPoint: "main" } })))
  const parallel = performance.now() - t1
  // Again, the same code: whatever the browser keeps.
  const t2 = performance.now()
  await Promise.all(codes("async").map((code) => device.createComputePipelineAsync({ layout: "auto", compute: { module: device.createShaderModule({ code }), entryPoint: "main" } })))
  const again = performance.now() - t2
  log(`COMPILE ${corpus.length} kernels: one by one ${sync.toFixed(0)} ms, all at once async ${parallel.toFixed(0)} ms, the same again ${again.toFixed(0)} ms | ${navigator.userAgent.slice(0, 60)}`)
} catch (e) {
  log(`ERROR: ${(e as Error).message}`)
}
log("ALL DONE")
