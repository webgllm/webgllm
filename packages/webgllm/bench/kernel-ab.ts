/** Replays dumped passes on WebGL 2 and on WebGPU, random inputs of the same sizes, and times each. */
import { GL, Gpu, type Pass, type Texture } from "../src/gpu/gl.js"
import { WebGpu } from "../src/gpu/webgpu.js"

const params = new URLSearchParams(location.search)
const out = document.querySelector("#out")!
const log = (s: string) => {
  out.textContent += s + "\n"
  void fetch("/log", { method: "POST", body: s })
}
type Shape = { internal: number; width: number; height: number }
type Replay = { uniforms: Pass["uniforms"]; inputs: Record<string, Shape>; outputs: Array<Shape>; rect: { x: number; y: number; width: number; height: number }; name?: string }
type Entry = { source: string; names: string[]; last?: Replay; biggest?: Replay }

const RED = 0x1903
const UPLOAD: Record<number, { format: number; type: number; channels: number; bytes: number }> = {
  [GL.RGBA32UI]: { format: GL.RGBA_INTEGER, type: GL.UNSIGNED_INT, channels: 4, bytes: 4 },
  [GL.RGBA32F]: { format: GL.RGBA, type: GL.FLOAT, channels: 4, bytes: 4 },
  [GL.RGBA16F]: { format: GL.RGBA, type: GL.HALF_FLOAT, channels: 4, bytes: 2 },
  [GL.RGBA8]: { format: GL.RGBA, type: GL.UNSIGNED_BYTE, channels: 4, bytes: 1 },
  [GL.RGBA8UI]: { format: GL.RGBA_INTEGER, type: GL.UNSIGNED_BYTE, channels: 4, bytes: 1 },
  [GL.RGBA8I]: { format: GL.RGBA_INTEGER, type: GL.BYTE, channels: 4, bytes: 1 },
  [GL.RGBA4]: { format: GL.RGBA, type: GL.UNSIGNED_SHORT_4_4_4_4, channels: 1, bytes: 2 },
  [GL.R8UI]: { format: GL.RED_INTEGER, type: GL.UNSIGNED_BYTE, channels: 1, bytes: 1 },
  [GL.R32UI]: { format: GL.RED_INTEGER, type: GL.UNSIGNED_INT, channels: 1, bytes: 4 },
  [GL.R32F]: { format: RED, type: GL.FLOAT, channels: 1, bytes: 4 }
}

/** Random contents for a texture: small floats, halves in range, any bits for integers (h2 words as two small halves). */
function pixels(s: Shape) {
  const u = UPLOAD[s.internal]
  const n = s.width * s.height * u.channels
  if (u.type === GL.FLOAT) return Float32Array.from({ length: n }, () => Math.random() * 2 - 1)
  if (u.type === GL.HALF_FLOAT) return Uint16Array.from({ length: n }, () => 0x3000 + ((Math.random() * 0x0bff) | 0) | (Math.random() < 0.5 ? 0x8000 : 0))
  if (u.type === GL.UNSIGNED_INT) {
    const half = () => 0x3000 + ((Math.random() * 0x0bff) | 0) | (Math.random() < 0.5 ? 0x8000 : 0)
    return Uint32Array.from({ length: n }, () => (half() | (half() << 16)) >>> 0)
  }
  if (u.bytes === 2) return Uint16Array.from({ length: n }, () => (Math.random() * 65536) | 0)
  const b = new Uint8Array(n)
  for (let i = 0; i < n; i += 65536) crypto.getRandomValues(b.subarray(i, Math.min(n, i + 65536)))
  return u.type === GL.BYTE ? new Int8Array(b.buffer) : b
}

/** An RGBA32UI h2 texture's rectangle as floats (8 halves a texel). */
function halves(bytes: Uint8Array, t: Texture, rect: Replay["rect"]) {
  const u16 = new Uint16Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 2)
  const out = new Float32Array(rect.width * rect.height * 8)
  const f = (h: number) => {
    const e = (h >> 10) & 31, m = h & 1023, s = h & 0x8000 ? -1 : 1
    return e === 0 ? s * m * 2 ** -24 : e === 31 ? (m ? NaN : s * Infinity) : s * (1 + m / 1024) * 2 ** (e - 15)
  }
  for (let y = 0; y < rect.height; y++)
    for (let x = 0; x < rect.width * 8; x++) out[y * rect.width * 8 + x] = f(u16[((rect.y + y) * t.width + rect.x) * 8 + x])
  return out
}

/** Largest difference relative to the largest magnitude. */
function diff(a: Float32Array | null, b: Float32Array | null) {
  if (!a || !b) return "-"
  let d = 0, m = 0
  for (let i = 0; i < a.length; i++) {
    if (!Number.isFinite(a[i]) || !Number.isFinite(b[i])) continue
    d = Math.max(d, Math.abs(a[i] - b[i]))
    m = Math.max(m, Math.abs(a[i]))
  }
  return m === 0 ? "zero" : (d / m).toExponential(1)
}

/** Waits until the GPU has done everything issued: WebGL by a blocking pixel read (its fences may report early). */
async function drain(gpu: Gpu | WebGpu, made: Array<Texture>) {
  if (gpu instanceof WebGpu) return gpu.finish()
  const t = gpu.texture(GL.RGBA32F, 1, 1)
  made.push(t)
  gpu.run({ program: gpu.program(`#version 300 es\nprecision highp float;\nout vec4 o;\nvoid main() { o = vec4(1.0); }`), inputs: {}, output: t })
  gpu.readNow(t)
}

async function time(gpu: Gpu | WebGpu, e: Entry, r: Replay, data: Map<string, ArrayBufferView>, reps: number) {
  const made: Array<Texture> = []
  try {
    const inputs: Record<string, Texture> = {}
    for (const [name, s] of Object.entries(r.inputs)) {
      const u = UPLOAD[s.internal]
      const t = gpu.texture(s.internal, s.width, s.height, { format: u.format, type: u.type, pixels: data.get(name)! })
      made.push(t)
      inputs[name] = t
    }
    // Two sets of outputs, alternated: a tile GPU skips what a later draw
    // into the same target overwrites, so repeats into one would cost one.
    const outputs = r.outputs.map((s) => gpu.texture(s.internal, s.width, s.height))
    const spare = r.outputs.map((s) => gpu.texture(s.internal, s.width, s.height))
    made.push(...outputs, ...spare)
    const one = (o: Array<Texture>) => (o.length === 1 ? o[0] : o)
    const pass: Pass = { name: r.name, program: gpu.program(e.source), inputs, uniforms: r.uniforms, output: one(outputs), rect: r.rect }
    const other: Pass = { ...pass, output: one(spare) }
    gpu.run(pass)
    const o = outputs[0]
    const got = o.internal === GL.RGBA32F ? await gpu.read(o, r.rect) : o.internal === GL.RGBA32UI ? halves(await gpu.readBytes(o), o, r.rect) : null
    for (let i = 0; i < 3; i++) gpu.run(pass)
    await drain(gpu, made)
    let best = Infinity
    for (let round = 0; round < 3; round++) {
      const t0 = performance.now()
      for (let i = 0; i < reps; i++) gpu.run(i & 1 ? other : pass)
      await drain(gpu, made)
      best = Math.min(best, (performance.now() - t0) / reps)
    }
    return { ms: best, got }
  } finally {
    for (const t of made) gpu.delete(t)
    await gpu.finish()
  }
}

try {
  const text = await (await fetch("/shaders.jsonl")).text()
  const lines = text.trim().split("\n").filter((l) => l.includes('"last"'))
  const corpus = JSON.parse(lines[params.has("line") ? Number(params.get("line")) : lines.length - 1 - Number(params.get("corpus") ?? 0)]) as Array<Entry>
  const which = (params.get("which") ?? "last") as "last" | "biggest"
  const name = params.get("name")
  const reps = Number(params.get("reps") ?? 20)
  const shapes = (params.get("wg") ?? "").split(",").filter(Boolean).map((w) => w.split("x").map(Number) as [number, number])
  const gl = new Gpu(new OffscreenCanvas(1, 1))
  const wg = await WebGpu.create()
  if (!wg) throw new Error("No WebGPU here.")
  if (params.has("lanes")) wg.coopThreads = Number(params.get("lanes"))
  log(`KAB ${navigator.userAgent} | ${gl.limits.renderer} | ${wg.limits.renderer} | features ${[...wg.device.features].join(",")} | subgroups ${Reflect.get(wg, "subgroups")} size ${Reflect.get(wg, "subgroupSize")}`)
  let glSum = 0, wgSum = 0
  for (const [i, e] of corpus.entries()) {
    const r = e[which]
    if (!r || (name && !e.names.includes(name)) || r.rect.width * r.rect.height < Number(params.get("min") ?? 2)) continue
    // (alt=<file in bench/alt>: that GLSL instead of the dumped source, for kernel experiments.)
    const alt = params.get("alt")
    if (alt) e.source = await (await fetch(`/alt/${alt}`)).text()
    const data = new Map(Object.entries(r.inputs).map(([k, s]) => [k, pixels(s)]))
    let a: number, b: number, check: string
    try {
      const x = await time(gl, e, r, data, reps)
      const y = await time(wg, e, r, data, reps)
      a = x.ms
      b = y.ms
      check = diff(x.got, y.got)
      for (const shape of shapes) {
        WebGpu.workgroup = shape
        const z = await time(wg, e, r, data, reps)
        log(`KAB #${i} ${e.names.join(",")} workgroup ${shape.join("x")}: ${(z.ms * 1000).toFixed(0)} µs, diff ${diff(x.got, z.got)}`)
      }
      WebGpu.workgroup = null
    } catch (error) {
      log(`KAB #${i} ${e.names.join(",")} skipped: ${(error as Error).message}`)
      continue
    }
    glSum += a
    wgSum += b
    log(`KAB #${i} ${e.names.join(",")} ${r.rect.width}x${r.rect.height}: WebGL ${(a * 1000).toFixed(0)} µs, WebGPU ${(b * 1000).toFixed(0)} µs (${(b / a).toFixed(2)}x), diff ${check}`)
  }
  log(`KAB total WebGL ${glSum.toFixed(2)} ms, WebGPU ${wgSum.toFixed(2)} ms${wg.errors.length ? ` | errors: ${wg.errors.join(" || ")}` : ""}`)
} catch (e) {
  log(`ERROR: ${(e as Error).message} ${(e as Error).stack ?? ""}`)
}
log("ALL DONE")
