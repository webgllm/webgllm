/** The cost of a pass in a chain of dependent ones (each reads the last's output), by size, on each backend. */
import { GL, Gpu, type Texture } from "../src/gpu/gl.js"
import { WebGpu } from "../src/gpu/webgpu.js"

const out = document.querySelector("#out")!
const log = (s: string) => {
  out.textContent += s + "\n"
  void fetch("/log", { method: "POST", body: s })
}
const SRC = `#version 300 es
precision highp float;
uniform sampler2D uX;
uniform int uWork;
out vec4 o;
void main() {
  ivec2 p = ivec2(gl_FragCoord.xy);
  vec4 a = texelFetch(uX, p, 0);
  for (int i = 0; i < uWork; i++) a = a * 0.999 + 0.001;
  o = a + 1.0;
}`

async function chain(gpu: Gpu | WebGpu, width: number, height: number, n: number, work: number) {
  const a = gpu.texture(GL.RGBA32F, width, height, { format: GL.RGBA, type: GL.FLOAT, pixels: new Float32Array(width * height * 4) })
  const b = gpu.texture(GL.RGBA32F, width, height)
  const t: Array<Texture> = [a, b]
  const program = gpu.program(SRC)
  const go = () => {
    for (let i = 0; i < n; i++) gpu.run({ program, inputs: { uX: t[i & 1] }, uniforms: { uWork: { int: work } }, output: t[(i + 1) & 1] })
  }
  go()
  await gpu.read(t[n & 1], { x: 0, y: 0, width: 1, height: 1 })
  let best = Infinity
  for (let r = 0; r < 4; r++) {
    const t0 = performance.now()
    go()
    // (A read, not a fence: it waits for the last pass's output itself.)
    await gpu.read(t[n & 1], { x: 0, y: 0, width: 1, height: 1 })
    best = Math.min(best, performance.now() - t0)
  }
  gpu.delete(a)
  gpu.delete(b)
  return (best * 1000) / n
}

/** One small pass, then its output read back: the round trip a token waits on. */
async function roundTrip(gpu: Gpu | WebGpu, how: "read" | "now" | "finish") {
  const a = gpu.texture(GL.RGBA32F, 8, 1, { format: GL.RGBA, type: GL.FLOAT, pixels: new Float32Array(32) })
  const b = gpu.texture(GL.RGBA32F, 8, 1)
  const program = gpu.program(SRC)
  const times: Array<number> = []
  for (let i = 0; i < 40; i++) {
    const t0 = performance.now()
    gpu.run({ program, inputs: { uX: a }, uniforms: { uWork: { int: 0 } }, output: b })
    if (how === "finish") await gpu.finish()
    else if (how === "now") await gpu.readNow(b, { x: 0, y: 0, width: 8, height: 1 })
    else await gpu.read(b, { x: 0, y: 0, width: 8, height: 1 })
    times.push(performance.now() - t0)
  }
  gpu.delete(a)
  gpu.delete(b)
  times.sort((x, y) => x - y)
  return `median ${times[20].toFixed(2)} ms, p10 ${times[4].toFixed(2)}, p90 ${times[36].toFixed(2)}`
}

try {
  const gl = new Gpu(new OffscreenCanvas(1, 1))
  const wg = await WebGpu.create()
  if (!wg) throw new Error("No WebGPU here.")
  log(`CHAIN ${navigator.userAgent}`)
  log(`CHAIN round trip WebGL readNow: ${await roundTrip(gl, "now")}`)
  log(`CHAIN round trip WebGL read: ${await roundTrip(gl, "read")}`)
  log(`CHAIN round trip WebGPU read: ${await roundTrip(wg, "read")}`)
  log(`CHAIN round trip WebGL fence: ${await roundTrip(gl, "finish")}`)
  log(`CHAIN round trip WebGPU fence: ${await roundTrip(wg, "finish")}`)
  for (const [w, h, work] of [[64, 1, 0], [512, 1, 0], [512, 8, 0], [4096, 1, 0], [512, 8, 64], [4096, 4, 64]]) {
    const x = await chain(gl, w, h, 400, work)
    const y = await chain(wg, w, h, 400, work)
    log(`CHAIN ${w}x${h} work ${work}: WebGL ${x.toFixed(1)} µs a pass, WebGPU ${y.toFixed(1)} µs (${(y / x).toFixed(2)}x)`)
  }
} catch (e) {
  log(`ERROR: ${(e as Error).message} ${(e as Error).stack ?? ""}`)
}
log("ALL DONE")
