// The native Q6_K word head (gpu/native.ts's wordHead) alone at LFM3 1.7B's head: 128,000 rows of 2,048, random
// weights, timed back to back. `v` lists variants as wordHead option sets, e.g. ?v=ordered,plain,ordered.f16:
// each dot-separated word is an option turned on (ordered, f16, direct, ...), "plain" none.
import { wordHead } from "../src/gpu/native.js"

const dev = /iPad/.test(navigator.userAgent) ? "ipad" : /Android/.test(navigator.userAgent) ? "pixel" : "mac"
const out = document.getElementById("out")!
const log = (t: string) => {
  t = `HEADK ${dev} ${t}`
  out.textContent += `${t}\n`
  void fetch("/log", { method: "POST", body: t }).catch(() => undefined)
}
try {
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" })
  if (!adapter) throw new Error("no adapter")
  const f16 = adapter.features.has("shader-f16")
  const device = await adapter.requestDevice({ requiredFeatures: f16 ? ["shader-f16"] : [], requiredLimits: { maxTextureDimension2D: adapter.limits.maxTextureDimension2D } })
  device.addEventListener("uncapturederror", (e) => log(`GPU ERROR: ${(e as GPUUncapturedErrorEvent).error.message}`))
  log(`${navigator.userAgent} f16 ${f16}`)
  const N = 128000, K = 2048, G = N / 4, kb = K / 256, max = device.limits.maxTextureDimension2D
  const columns = Math.ceil(G / max), H = Math.ceil(G / columns), rowTexels = kb * 52, outW = 1024, logitRows = Math.ceil(G / outW)
  const W = device.createTexture({ size: [columns * rowTexels, H], format: "rgba32uint", usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST })
  {
    const rows = 64, chunk = new Uint32Array(columns * rowTexels * 4 * rows)
    for (let i = 0; i < chunk.length; i += 16384) crypto.getRandomValues(chunk.subarray(i, i + 16384))
    for (let y = 0; y < H; y += rows) device.queue.writeTexture({ texture: W, origin: [0, y] }, chunk, { bytesPerRow: columns * rowTexels * 16 }, [columns * rowTexels, Math.min(rows, H - y)])
  }
  // The same layout transposed (row groups along x), for "transposed" variants (other random bytes: other logits).
  const WT = device.createTexture({ size: [H, columns * rowTexels], format: "rgba32uint", usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST })
  {
    const rows = 16, chunk = new Uint32Array(H * 4 * rows)
    for (let i = 0; i < chunk.length; i += 16384) crypto.getRandomValues(chunk.subarray(i, i + 16384))
    for (let y = 0; y < columns * rowTexels; y += rows) device.queue.writeTexture({ texture: WT, origin: [0, y] }, chunk, { bytesPerRow: H * 16 }, [H, Math.min(rows, columns * rowTexels - y)])
  }
  // xp: the expanded head's bytes (a weight a byte, each 16's first 4 bytes carrying its scale in their top bits)
  // as RGBA32UI, a 16-weight run a texel, in reading order (super-block, 32-block, half, row): a timing estimate.
  const XP = device.createTexture({ size: [columns * kb * 64, H], format: "rgba32uint", usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST })
  {
    const rows = 32, chunk = new Uint32Array(columns * kb * 64 * 4 * rows)
    for (let i = 0; i < chunk.length; i += 16384) crypto.getRandomValues(chunk.subarray(i, i + 16384))
    for (let y = 0; y < H; y += rows) device.queue.writeTexture({ texture: XP, origin: [0, y] }, chunk, { bytesPerRow: columns * kb * 64 * 16 }, [columns * kb * 64, Math.min(rows, H - y)])
  }
  const xpCode = (wg: number) => `
struct Uni { rect: vec4<u32>, kb: i32, h: i32, rowTexels: i32, g: i32, outW: i32, logitRows: i32, kbSlice: i32, rows: i32, aOff: i32, rowA0: i32, rOff: i32, parts: i32, texels: i32, eps: f32 }
@group(0) @binding(0) var<uniform> u: Uni;
@group(0) @binding(1) var tA: texture_2d<u32>;
@group(0) @binding(2) var tW: texture_2d<u32>;
@group(0) @binding(3) var tS: texture_2d<f32>;
@group(0) @binding(4) var out0: texture_storage_2d<rgba32float, write>;
fn lo(w: vec4<u32>) -> vec4<f32> { return vec4<f32>(unpack2x16float(w.x), unpack2x16float(w.y)); }
fn hi(w: vec4<u32>) -> vec4<f32> { return vec4<f32>(unpack2x16float(w.z), unpack2x16float(w.w)); }
@compute @workgroup_size(${wg})
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let x = i32(id.x); let y = i32(id.y);
  let g = y * u.outW + x;
  var o = vec4<f32>(0.0);
  if (g < u.g && x < u.outW) {
    let col = g / u.h; let gr = g - col * u.h;
    let wx = col * u.kb * 64; let sx = col * u.kb;
    var acc = vec4<f32>(0.0);
    for (var sb = 0; sb < u.kb; sb++) {
      let d = textureLoad(tS, vec2<i32>(sx + sb, gr), 0);
      var P = vec4<f32>(0.0);
      for (var q = 0; q < 8; q++) {
        for (var hf = 0; hf < 2; hf++) {
          let a0 = textureLoad(tA, vec2<i32>(sb * 32 + q * 4 + hf * 2, 0), 0);
          let a1 = textureLoad(tA, vec2<i32>(sb * 32 + q * 4 + hf * 2 + 1, 0), 0);
          let A0 = lo(a0); let A1 = hi(a0); let A2 = lo(a1); let A3 = hi(a1);
          let c = -32.0 * dot(A0 + A1 + A2 + A3, vec4<f32>(1.0));
          let B0 = A0 * 255.0; let B1 = A1 * 255.0; let B2 = A2 * 255.0; let B3 = A3 * 255.0;
          for (var r = 0; r < 4; r++) {
            let T = textureLoad(tW, vec2<i32>(wx + sb * 64 + q * 8 + hf * 4 + r, gr), 0);
            let sc = dot(unpack4x8unorm((T.x >> 6u) & 0x03030303u), vec4<f32>(255.0, 1020.0, 4080.0, 16320.0));
            let s = select(sc, sc - 256.0, sc >= 128.0);
            let p = dot(unpack4x8unorm(T.x & 0x3F3F3F3Fu), B0) + dot(unpack4x8unorm(T.y), B1) + dot(unpack4x8unorm(T.z), B2) + dot(unpack4x8unorm(T.w), B3) + c;
            P[r] += s * p;
          }
        }
      }
      acc += d * P;
    }
    o = acc;
  }
  if (x < u.outW) { textureStore(out0, vec2<i32>(x, y), o); }
}`
  const S = device.createTexture({ size: [columns * kb, H], format: "rgba32float", usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST })
  device.queue.writeTexture({ texture: S }, new Float32Array(columns * kb * H * 4).map(() => Math.random() * 1e-3), { bytesPerRow: columns * kb * 16 }, [columns * kb, H])
  const half = (x: number) => { if (Math.abs(x) < 2 ** -14) return 0; const f = new Float32Array([x]); const i = new Uint32Array(f.buffer)[0]; return ((i >> 16) & 0x8000) | ((((i >> 23) & 0xff) - 112) << 10) | ((i >> 13) & 0x3ff) }
  const A = device.createTexture({ size: [K / 8, 1], format: "rgba32uint", usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST })
  device.queue.writeTexture({ texture: A }, new Uint32Array(K / 2).map(() => half(Math.random() - 0.5) | (half(Math.random() - 0.5) << 16)), { bytesPerRow: K / 8 * 16 }, [K / 8, 1])
  const O = device.createTexture({ size: [outW, logitRows], format: "rgba32float", usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.COPY_SRC })
  const U = device.createBuffer({ size: 80, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST })
  {
    const u = new Uint32Array(20)
    u.set([0, 0, outW, logitRows, kb, H, rowTexels, G, outW, logitRows, kb, 1, 0, 0, 0, 1, K / 4])
    device.queue.writeBuffer(U, 0, u)
  }
  const bytes = G * kb * 52 * 16, xpBytes = G * kb * 64 * 16
  const variants = (new URLSearchParams(location.search).get("v") ?? "ordered,plain").split(",")
  let ref: string | null = null
  for (const v of variants) {
    const opts = v.startsWith("xp") ? {} : Object.fromEntries(v.split(".").filter((w) => w !== "plain").map((w) => /^wg\d+$/.test(w) ? ["wg", Number(w.slice(2))] : [w, true]))
    const xp = v.startsWith("xp"), xpWg = Number(v.slice(2)) || 64
    const spec = xp
      ? { code: xpCode(xpWg), groups: (r: { width: number; height: number }) => [Math.ceil(r.width / xpWg), r.height] }
      : wordHead({ split: false, prescaled: false, kbSlice: kb, ...opts } as Parameters<typeof wordHead>[0])
    const pipe = await device.createComputePipelineAsync({ layout: "auto", compute: { module: device.createShaderModule({ code: spec.code }), entryPoint: "main" } })
    const bg = device.createBindGroup({ layout: pipe.getBindGroupLayout(0), entries: [
      { binding: 0, resource: { buffer: U } }, { binding: 1, resource: A.createView() }, { binding: 2, resource: (v.startsWith("xp") ? XP : v.includes("transposed") ? WT : W).createView() },
      { binding: 3, resource: S.createView() }, { binding: 4, resource: O.createView() }] })
    const [gx, gy] = spec.groups({ x: 0, y: 0, width: outW, height: logitRows } as never)
    const run = async (n: number) => {
      const e = device.createCommandEncoder(), p = e.beginComputePass()
      p.setPipeline(pipe); p.setBindGroup(0, bg)
      for (let i = 0; i < n; i++) p.dispatchWorkgroups(gx, gy)
      p.end(); device.queue.submit([e.finish()]); await device.queue.onSubmittedWorkDone()
    }
    await run(2)
    let best = Infinity
    for (let i = 0; i < 5; i++) { const t = performance.now(); await run(5); best = Math.min(best, (performance.now() - t) / 5) }
    // A few logits, to see variants agree (on the same bytes: ordered and plain read them differently, so differ).
    const rb = device.createBuffer({ size: 256 * 16, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST })
    const e = device.createCommandEncoder(); e.copyTextureToBuffer({ texture: O }, { buffer: rb, bytesPerRow: 4096 }, [256, 1]); device.queue.submit([e.finish()])
    await rb.mapAsync(GPUMapMode.READ); const got = [...new Float32Array(rb.getMappedRange()).slice(0, 4)].map((x) => x.toPrecision(5)).join(","); rb.unmap()
    if (v.includes("ordered") && ref === null) ref = got
    log(`${v}: ${best.toFixed(2)} ms, ${((v.startsWith("xp") ? xpBytes : bytes) / best / 1e6).toFixed(1)} GB/s, logits ${got}${v.includes("ordered") && ref !== got ? " (DIFFERS from the first ordered)" : ""}`)
  }
  log("ALL DONE")
} catch (e) { log(`ERROR: ${(e as Error)?.stack ?? e}`) }
