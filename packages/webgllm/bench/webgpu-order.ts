/**
 * Checks on the WebGPU backend, on real WebGPU, each with its control:
 * - a small decode pass waiting to be merged (WebGpu.queue) still runs
 *   before a later upload to, or deletion of, a texture it uses;
 * - deleting a texture a queued pass reads leaves no bind group or index
 *   entry for it (the caches stay bounded);
 * - a K loop shared by cooperating lanes (coopThreads) sums its terms onto
 *   the value its accumulator held before the loop once, not once a lane.
 * Logs `WEBGPU-ORDER ...` lines and `ALL DONE`. `?baseline=1` runs another
 * checkout's sources (see api.html).
 */
import { GL } from "../src/gpu/gl.js"
import type { WebGpu as WebGpuClass } from "../src/gpu/webgpu.js"

const baseline = new URLSearchParams(location.search).get("baseline") === "1"
// (A path in a variable: the bundler leaves the import to the browser.)
const sources = baseline ? "/bundle/bench/baseline-src/gpu/webgpu.js" : "/bundle/src/gpu/webgpu.js"
const { WebGpu } = (await import(sources)) as { WebGpu: typeof WebGpuClass }
type WebGpu = WebGpuClass

const out = document.querySelector("#out")!
const log = (s: string) => {
  out.textContent += s + "\n"
  void fetch("/log", { method: "POST", body: `WEBGPU-ORDER ${s}` })
}

const COPY = `#version 300 es
precision highp float;
uniform sampler2D uX;
out vec4 o;
void main() { o = texelFetch(uX, ivec2(gl_FragCoord.xy), 0); }`

// The accumulator starts at uInit's texel; each of uN iterations adds uStep.
const SUM = `#version 300 es
precision highp float;
uniform sampler2D uInit;
uniform int uN;
uniform float uStep;
out vec4 o;
void main() {
  vec4 acc = texelFetch(uInit, ivec2(gl_FragCoord.xy), 0);
  for (int k = 0; k < uN; k++) {
    acc += vec4(uStep);
  }
  o = acc;
}`

const pixel = (v: number) => ({ format: GL.RGBA, type: GL.FLOAT, pixels: new Float32Array([v, v, v, v]) })

async function order(wg: WebGpu, then: "write" | "delete", merge: boolean) {
  WebGpu.merge = merge
  wg.setPhase("decode")
  wg.errors.length = 0
  const a = wg.texture(GL.RGBA32F, 1, 1, pixel(1))
  const b = wg.texture(GL.RGBA32F, 1, 1)
  // (Makes sure a's upload has landed, and nothing is being built.)
  await wg.read(a)
  wg.run({ program: wg.program(COPY), inputs: { uX: a }, output: b })
  if (then === "write") wg.write(a, { x: 0, y: 0, width: 1, height: 1 }, pixel(2))
  else wg.delete(a)
  const got = (await wg.read(b))[0]
  await wg.finish()
  const errors = wg.errors.length ? ` errors: ${wg.errors.join(" | ")}` : ""
  log(`${then} after a ${merge ? "queued" : "dispatched"} copy: copy read ${got} (want 1) ${got === 1 && !errors ? "OK" : "FAIL"}${errors}`)
  if (then === "write") wg.delete(a)
  wg.delete(b)
  WebGpu.merge = true
}

/** Inputs made, read by a queued copy, deleted, four times: the bind group caches stay bounded. */
async function deletedCaches(wg: WebGpu) {
  WebGpu.merge = true
  wg.setPhase("decode")
  wg.errors.length = 0
  const program = wg.program(COPY)
  const b = wg.texture(GL.RGBA32F, 1, 1)
  const sizes: Array<string> = []
  for (let i = 0; i < 4; i++) {
    const a = wg.texture(GL.RGBA32F, 1, 1, pixel(i + 1))
    wg.run({ program, inputs: { uX: a }, output: b })
    wg.delete(a)
    await wg.finish()
    const index = Reflect.get(wg, "groupsByTexture") as Map<number, Set<string>>
    sizes.push(`${(Reflect.get(wg, "bindGroups") as Map<string, unknown>).size}/${index.has(a.id) ? "indexed" : "gone"}`)
  }
  const got = (await wg.read(b))[0]
  const bounded = sizes.every((x) => x === sizes[0]) && sizes.every((x) => x.endsWith("gone"))
  log(`deleted queued inputs: bind groups/index after each ${sizes.join(" ")}; copy read ${got} (want 4) ${bounded && got === 4 && !wg.errors.length ? "OK" : "FAIL"}`)
  wg.delete(b)
}

async function lanes(wg: WebGpu, threads: number, subgroups: boolean, step: number) {
  wg.coopThreads = threads
  Reflect.set(wg, "subgroups", subgroups)
  wg.setPhase("decode")
  wg.errors.length = 0
  const init = wg.texture(GL.RGBA32F, 1, 1, pixel(1))
  const o = wg.texture(GL.RGBA32F, 1, 1)
  // (A fresh program each time: variants are cached by program.)
  wg.run({ program: wg.program(SUM + `\n// ${threads} ${subgroups} ${step}`), inputs: { uInit: init }, uniforms: { uN: { int: 16 }, uStep: step }, output: o })
  const got = (await wg.read(o))[0]
  const want = 1 + 16 * step
  const errors = wg.errors.length ? ` errors: ${wg.errors.join(" | ")}` : ""
  log(`lanes ${threads ? `${threads} threads, ${subgroups ? "subgroup shuffles" : "workgroup memory"}` : "off"}, init 1, 16 × ${step}: ${got} (want ${want}) ${got === want && !errors ? "OK" : "FAIL"}${errors}`)
  wg.delete(init)
  wg.delete(o)
  wg.coopThreads = 0
}

try {
  const wg = await WebGpu.create()
  if (!wg) throw new Error("No WebGPU.")
  log(`${baseline ? "baseline sources; " : ""}device: ${wg.limits.renderer}; subgroups ${Reflect.get(wg, "subgroups")}`)
  for (const then of ["write", "delete"] as const) for (const merge of [false, true]) await order(wg, then, merge)
  await deletedCaches(wg)
  const subgroups = Reflect.get(wg, "subgroups") as boolean
  for (const step of [0, 1]) {
    await lanes(wg, 0, false, step)
    await lanes(wg, 64, false, step)
    if (subgroups) await lanes(wg, 64, true, step)
  }
} catch (error) {
  log(`ERROR ${(error as Error)?.stack ?? error}`)
}
log("ALL DONE")
