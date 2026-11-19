/** A GPU's multiply-add rate through the engine's own GPU layer (WebGPU where the engine uses it, else WebGL 2): predictSpeed's quick test, measured. */
import { Engine, TUNING } from "../src/engine.js"
import { GL } from "../src/gpu/gl.js"
import { WebGpu } from "../src/gpu/webgpu.js"

const params = new URLSearchParams(location.search)
if (params.has("webgpu")) TUNING.webgpu = Number(params.get("webgpu"))
const iters = Number(params.get("iters") ?? 512)
const out = document.getElementById("out")!
const log = (text: string) => {
  out.textContent += `${text}\n`
  void fetch("/log", { method: "POST", body: text }).catch(() => undefined)
}
try {
  const t0 = performance.now()
  const webgpu = TUNING.webgpu > 0 ? await WebGpu.create().catch(() => null) : null
  const engine = new Engine(new OffscreenCanvas(1, 1), { ...(webgpu && { gpu: webgpu }), batchSize: 512 })
  const gpu = engine.gpu
  const tGpu = performance.now()
  const size = 512
  const input = gpu.texture(GL.RGBA32F, size, size)
  const output = gpu.texture(GL.RGBA32F, size, size)
  // Four independent chains of vec4 multiply-adds: 32 flops an iteration a texel.
  const program = gpu.program(`#version 300 es
precision highp float;uniform sampler2D uX;uniform int uIters;out vec4 o;
void main(){ivec2 p=ivec2(gl_FragCoord.xy);vec4 a=texelFetch(uX,p,0);vec4 x=texelFetch(uX,ivec2(p.y,p.x),0);vec4 b=a+0.25;vec4 c=a+0.5;vec4 d=a+0.75;vec4 m=0.9999+x*1e-7;vec4 n=0.0001+x.wzyx*1e-7;
for(int i=0;i<uIters;i++){a=a*m+n;b=b*m+n;c=c*m+n;d=d*m+n;}o=a+b+c+d;}`)
  const pass = { program, inputs: { uX: input }, uniforms: { uIters: { int: iters } }, output }
  const flops = size * size * iters * 32
  gpu.run(pass)
  await gpu.finish()
  const tCompile = performance.now()
  // Until ~30 ms of runs: a phone's clock comes up.
  let reps = 0
  const start = performance.now()
  while (performance.now() - start < 30) {
    for (let i = 0; i < 4; i++) gpu.run(pass)
    reps += 4
    await gpu.finish()
  }
  const ms = (performance.now() - start) / reps
  const t1 = performance.now()
  log(`MICRO ${navigator.userAgent.slice(0, 50)} backend ${webgpu ? "webgpu" : "webgl"} ${(flops / ms / 1e6).toFixed(0)} GFLOP/s (${ms.toFixed(3)} ms a pass, ${reps} passes) | gpu ${Math.round(tGpu - t0)} ms, compile ${Math.round(tCompile - tGpu)} ms, timing ${Math.round(t1 - tCompile)} ms, total ${Math.round(t1 - t0)} ms`)
  gpu.dispose()
  log("ALL DONE")
} catch (error) {
  log(`ERROR: ${error instanceof Error ? error.stack ?? error.message : String(error)}`)
}
