/**
 * Prompt matmul variants against the engine's exact kernel on WebGPU: the
 * engine's own WebGpu backend, its kernels.ts matmulF16 (f16Pack 3) as
 * translated by wgsl.ts at run time, its native buffer dequant (and the GLSL
 * dequantBytes), with the engine's texture formats, uniforms, rectangles,
 * slices and phase. Variants are native WGSL passes on the same textures,
 * writing the same output layout, checked against the baseline's output.
 * Times are GPU timestamps per pass (WebGpu.profile), median over rounds,
 * variants alternated round by round.
 *
 * ?shapes=gate_up,down  ?v=base,d11,...  ?rounds=7&reps=6  ?m=512  ?dev=engine
 */
import { GL, type Pass, type Texture } from "../src/gpu/types.js"
import { WebGpu, type NativeSpec } from "../src/gpu/webgpu.js"
import * as K from "../src/gpu/kernels.js"
import { bufferDequant, promptMatmulBuffer } from "../src/gpu/native.js"

const params = new URLSearchParams(location.search)
const out = document.querySelector("#out")!
const log = (s: string) => {
  out.textContent += s + "\n"
  void fetch("/log", { method: "POST", body: `GEX ${s}` })
}

// ------------------------------------------------------------ halves

const f32 = new Float32Array(1), u32 = new Uint32Array(f32.buffer)
function toHalf(v: number) {
  f32[0] = v
  const x = u32[0]
  const s = (x >>> 16) & 0x8000
  let e = ((x >>> 23) & 0xff) - 127 + 15
  let m = x & 0x7fffff
  if (e <= 0) {
    if (e < -10) return s
    m = (m | 0x800000) >> (1 - e)
    const r = m & 0x1fff, h = m >> 13
    return s | (h + (r > 0x1000 || (r === 0x1000 && (h & 1)) ? 1 : 0))
  }
  if (e >= 31) return s | 0x7c00
  const r = m & 0x1fff
  let h = (e << 10) | (m >> 13)
  if (r > 0x1000 || (r === 0x1000 && (h & 1))) h++
  return s | h
}
function fromHalf(h: number) {
  const e = (h >> 10) & 31, m = h & 1023, s = h & 0x8000 ? -1 : 1
  return e === 0 ? s * m * 2 ** -24 : e === 31 ? (m ? NaN : s * Infinity) : s * (1 + m / 1024) * 2 ** (e - 15)
}

// ------------------------------------------------------------ the engine's plan

/** LFM3-1.7B's prompt matrices (rows N × K). */
const ALL_SHAPES = [
  { name: "in_proj", N: 6144, K: 2048 },
  { name: "gate_up", N: 15360, K: 2048 },
  { name: "down", N: 2048, K: 7680 },
  { name: "out", N: 2048, K: 2048 }
]
const M = Number(params.get("m") ?? 512)
const CAPACITY = 576
/** The f16 scratch the engine picks for LFM3 on this Mac: columns of 1,024 rows, 7,680 RGBA16F texels wide (WGPROFILE's "dequant 3840x1024"). */
const HD = 1024, SCRATCH_W = 7680
/** TUNING.quadSliceK, quadMaxSplit. */
const QUAD_SLICE_K = 512, QUAD_MAX_SPLIT = 4

/** Engine.quadSlices. */
function quadSlices(N: number, Kd: number, most: number) {
  const groups = N / 4, blocks = Kd / 32
  const sliceK = QUAD_SLICE_K * (groups > 2048 ? 2 : 1)
  const kbSlice = Math.ceil(blocks / Math.max(1, Math.min(most, Math.round(Kd / sliceK))))
  return { kbSlice, slices: Math.ceil(blocks / kbSlice) }
}
/** Engine.f16Plan. */
function f16Plan(f: { hd: number; width: number }, N: number, kbSlice: number, slices: number) {
  const sliceKq = kbSlice * 8
  const fit = Math.floor(f.width / (Math.ceil(N / f.hd) * sliceKq))
  if (fit >= 1) {
    const groups = Math.ceil(slices / Math.min(fit, slices))
    return { group: Math.ceil(slices / groups), rows: N }
  }
  const most = Math.max(4, Math.floor(f.width / sliceKq) * f.hd)
  const chunks = Math.ceil(N / most)
  return { group: 1, rows: Math.min(most, Math.ceil(N / chunks / 4) * 4) }
}

// ------------------------------------------------------------ variants

const UNI = `struct Uni { rect: vec4<u32>, kq: i32, kqW: i32, kq0: i32, hd: i32, sliceRows: i32, kqSlice: i32, row0: i32 }
@group(0) @binding(0) var<uniform> u: Uni;
@group(0) @binding(1) var tW: texture_2d<u32>;
@group(0) @binding(2) var tA: texture_2d<u32>;
@group(0) @binding(3) var out0: texture_storage_2d<rgba32uint, write>;
@group(0) @binding(4) var out1: texture_storage_2d<rgba32uint, write>;
fn lo(v: vec4<u32>) -> vec4<f32> { return vec4<f32>(unpack2x16float(v.x), unpack2x16float(v.y)); }
fn hi(v: vec4<u32>) -> vec4<f32> { return vec4<f32>(unpack2x16float(v.z), unpack2x16float(v.w)); }
fn packH2(a0: vec4<f32>, b0: vec4<f32>) -> vec4<u32> {
  let a = clamp(a0, vec4<f32>(-65504.0), vec4<f32>(65504.0));
  let b = clamp(b0, vec4<f32>(-65504.0), vec4<f32>(65504.0));
  return vec4<u32>(pack2x16float(a.xy), pack2x16float(a.zw), pack2x16float(b.xy), pack2x16float(b.zw));
}`
const UNIFORMS: NativeSpec["uniforms"] = ["uKq", "uKqW", "uKq0", "uHd", "uSliceRows", "uKqSlice", "uRow0"].map((name) => ({ name, type: "int" as const }))
/** The shape being run (a native spec's groups() reads it: slices aren't in the rectangle). */
const cur = { sliceRows: 1 }

const SW = 7680 / 2, SA = 7680 / 8
/** UNI with the weight scratch and activations as storage buffers (bindings 1, 2), rows SW and SA texels. */
const UNI_BUF = UNI.replace("@group(0) @binding(1) var tW: texture_2d<u32>;", "@group(0) @binding(1) var<storage, read> bW: array<vec4<u32>>;")
  .replace("@group(0) @binding(2) var tA: texture_2d<u32>;", "@group(0) @binding(2) var<storage, read> bA: array<vec4<u32>>;")

/**
 * Straight from the textures (or storage buffer copies: buf), as the
 * engine's kernel, TM row groups × TN token quads a thread, workgroups
 * WX × WY; swap: x indexes token tiles; u: h2 texels a loop step; trips: the
 * loop's count as a constant (the shape's kqSlice / 2).
 */
function direct(o: { tm: number; tn: number; wx: number; wy: number; swap?: boolean; buf?: boolean | "w" | "a"; split?: boolean; ptr?: boolean; u?: number; trips?: number; math?: "f32" | "nocvt" | "h4" | "h8" }): NativeSpec {
  const { tm, tn } = o
  const U = o.u ?? 1
  const math = o.math ?? "f32"
  const H = math === "h4" || math === "h8"
  const pre = math === "nocvt"
    ? `fn lo_(v: vec4<u32>) -> vec4<f32> { return bitcast<vec4<f32>>(v); }\nfn hi_(v: vec4<u32>) -> vec4<f32> { return bitcast<vec4<f32>>(v.zwxy); }\n`
    : H
      ? `fn lo_(v: vec4<u32>) -> vec4<f16> { return vec4<f16>(bitcast<vec2<f16>>(v.x), bitcast<vec2<f16>>(v.y)); }\nfn hi_(v: vec4<u32>) -> vec4<f16> { return vec4<f16>(bitcast<vec2<f16>>(v.z), bitcast<vec2<f16>>(v.w)); }\n`
      : `fn lo_(v: vec4<u32>) -> vec4<f32> { return lo(v); }\nfn hi_(v: vec4<u32>) -> vec4<f32> { return hi(v); }\n`
  const L: string[] = []
  const decl: string[] = []
  const wB = o.buf === true || o.buf === "w", aB = o.buf === true || o.buf === "a"
  const ld = (w: boolean, x: string, y: string) => (w ? wB : aB) ? `${w ? "bW" : "bA"}[(${y}) * ${w ? SW : SA} + ${x}]` : `textureLoad(${w ? "tW" : "tA"}, vec2<i32>(${x}, ${y}), 0)`
  const texs = [...(wB ? [] : ["W"]), ...(aB ? [] : ["A"])], bufs = [...(wB ? ["W"] : []), ...(aB ? ["A"] : [])]
  const head = UNI.replace(/@group\(0\) @binding\(1\) var tW: texture_2d<u32>;\n@group\(0\) @binding\(2\) var tA: texture_2d<u32>;/,
    [...texs.map((n, i) => `@group(0) @binding(${1 + i}) var t${n}: texture_2d<u32>;`), ...bufs.map((n, i) => `@group(0) @binding(${1 + texs.length + i}) var<storage, read> b${n}: array<${o.split ? "vec2<u32>" : "vec4<u32>"}>;`)].join("\n"))
  if (o.split && (!wB || !aB || U !== 1 || math !== "f32")) throw new Error("split: buffers, u 1, f32 only")
  for (let j = 0; j < tn * 4; j++) for (let i = 0; i < tm; i++) decl.push(`  var acc_${j}_${i} = vec4<f32>(0.0);`)
  if (o.split) for (let h = 0; h < 2; h++) {
    L.push(`    {`)
    for (let i = 0; i < tm; i++) for (let r = 0; r < 4; r++) L.push(`      let W${i}_${r} = bW[2 * ((wy + ${4 * i + r}) * ${SW} + tw) + ${h}];`)
    for (let jq = 0; jq < tn; jq++) for (let t = 0; t < 4; t++) L.push(`      let A${jq}_${t} = bA[2 * ((m0 + ${4 * jq + t}) * ${SA} + ta) + ${h}];`)
    for (let i = 0; i < tm; i++) for (let r = 0; r < 4; r++) L.push(`      let w${i}_${r} = vec4<f32>(unpack2x16float(W${i}_${r}.x), unpack2x16float(W${i}_${r}.y));`)
    for (let jq = 0; jq < tn; jq++) for (let t = 0; t < 4; t++) {
      L.push(`      { let a = vec4<f32>(unpack2x16float(A${jq}_${t}.x), unpack2x16float(A${jq}_${t}.y));`)
      for (let i = 0; i < tm; i++) L.push(`        acc_${4 * jq + t}_${i} += vec4<f32>(dot(w${i}_0, a), dot(w${i}_1, a), dot(w${i}_2, a), dot(w${i}_3, a));`)
      L.push(`      }`)
    }
    L.push(`    }`)
  }
  else for (let s = 0; s < U; s++) {
    for (let i = 0; i < tm; i++) for (let r = 0; r < 4; r++) L.push(o.ptr ? `    let W${s}_${i}_${r} = bW[wp + ${(4 * i + r) * SW + s}];` : `    let W${s}_${i}_${r} = ${ld(true, `tw + ${s}`, `wy + ${4 * i + r}`)};`)
    for (let jq = 0; jq < tn; jq++) for (let t = 0; t < 4; t++) L.push(`    let A${s}_${jq}_${t} = ${ld(false, `ta + ${s}`, `m0 + ${4 * jq + t}`)};`)
  }
  if (!o.split) for (let s = 0; s < U; s++)
    if (math === "h8") {
      L.push(`    {`)
      for (let i = 0; i < tm; i++) for (let r = 0; r < 4; r++) L.push(`      let w${i}_${r} = lo_(W${s}_${i}_${r}); let v${i}_${r} = hi_(W${s}_${i}_${r});`)
      for (let jq = 0; jq < tn; jq++) for (let t = 0; t < 4; t++) {
        L.push(`      { let a = lo_(A${s}_${jq}_${t}); let b = hi_(A${s}_${jq}_${t});`)
        for (let i = 0; i < tm; i++) L.push(`        acc_${4 * jq + t}_${i} += vec4<f32>(vec4<f16>(dot(w${i}_0, a) + dot(v${i}_0, b), dot(w${i}_1, a) + dot(v${i}_1, b), dot(w${i}_2, a) + dot(v${i}_2, b), dot(w${i}_3, a) + dot(v${i}_3, b)));`)
        L.push(`      }`)
      }
      L.push(`    }`)
    } else
    for (const h of ["lo_", "hi_"]) {
      L.push(`    {`)
      for (let i = 0; i < tm; i++) for (let r = 0; r < 4; r++) L.push(`      let w${i}_${r} = ${h}(W${s}_${i}_${r});`)
      for (let jq = 0; jq < tn; jq++) for (let t = 0; t < 4; t++) {
        L.push(`      { let a = ${h}(A${s}_${jq}_${t});`)
        for (let i = 0; i < tm; i++) L.push(H
          ? `        acc_${4 * jq + t}_${i} += vec4<f32>(vec4<f16>(dot(w${i}_0, a), dot(w${i}_1, a), dot(w${i}_2, a), dot(w${i}_3, a)));`
          : `        acc_${4 * jq + t}_${i} += vec4<f32>(dot(w${i}_0, a), dot(w${i}_1, a), dot(w${i}_2, a), dot(w${i}_3, a));`)
        L.push(`      }`)
      }
      L.push(`    }`)
    }
  const S: string[] = []
  for (let jq = 0; jq < tn; jq++) for (let i = 0; i < tm; i++)
    S.push(`  if (q0 + ${jq} < u.sliceRows) {
    let p = vec2<i32>(g0 + ${i}, slice * u.sliceRows + q0 + ${jq});
    textureStore(out0, p, packH2(acc_${4 * jq}_${i}, acc_${4 * jq + 1}_${i}));
    textureStore(out1, p, packH2(acc_${4 * jq + 2}_${i}, acc_${4 * jq + 3}_${i}));
  }`)
  const loop = o.trips
    ? `  for (var it = 0; it < ${o.trips / U}; it++) {
    let kq = kq0 + ${2 * U} * it;${o.ptr ? `\n    let wp = wb + ${U} * it;` : ""}`
    : `  for (var kq = kq0; kq < kq1; kq += ${2 * U}) {`
  const code = `${H ? "enable f16;\n" : ""}${head}
${pre}
@compute @workgroup_size(${o.wx}, ${o.wy})
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let rt = i32(id.${o.swap ? "y" : "x"});
  let yt = i32(id.${o.swap ? "x" : "y"});
  let qT = (u.sliceRows + ${tn - 1}) / ${tn};
  let slices = i32(u.rect.w) / u.sliceRows;
  if (rt * ${tm} >= i32(u.rect.z) || yt >= slices * qT) { return; }
  let slice = yt / qT;
  let q0 = (yt - slice * qT) * ${tn};
  let m0 = 4 * q0;
  let g0 = i32(u.rect.x) + rt * ${tm};
  let row = 4 * g0 - u.row0;
  let c = row / u.hd;
  let wx = c * u.kqW - u.kq0;
  let wy = row - c * u.hd;
  let kq0 = slice * u.kqSlice;
  let kq1 = min(u.kq, kq0 + u.kqSlice);
${decl.join("\n")}${o.ptr ? `\n  let wb = wy * ${SW} + ((wx + kq0) >> 1u);` : ""}
${loop}
    let tw = (wx + kq) >> 1u;
    let ta = kq >> 1u;
${L.join("\n")}
  }
${S.join("\n")}
}`
  return {
    code,
    inputs: texs.map((n) => ({ name: `u${n}`, kind: "u" as const })),
    buffers: bufs.map((n) => ({ name: `b${n}` })),
    uniforms: UNIFORMS,
    groups: (rect) => {
      const rows = Math.ceil(rect.width / tm)
      const ys = (rect.height / cur.sliceRows) * Math.ceil(cur.sliceRows / tn)
      return o.swap ? [Math.ceil(ys / o.wx), Math.ceil(rows / o.wy)] : [Math.ceil(rows / o.wx), Math.ceil(ys / o.wy)]
    }
  }
}

/**
 * Workgroup tiles: BMg row groups × BNq token quads a workgroup, K in steps
 * of BKt h2 texels (8 halves each) through workgroup memory (f32 vec4s, or
 * h2 texels as loaded with `h2`), TM row groups × TN token quads a thread;
 * prefetch: the next step's texels loaded before this step's math.
 */
function tiled(o: { bm: number; bn: number; tm: number; tn: number; bk: number; h2?: boolean; swap?: boolean; prefetch?: boolean; tokFast?: boolean }): NativeSpec {
  const { bm, bn, tm, tn, bk } = o
  const BM = bm * 4, BN = bn * 4
  const nx = bm / tm, ny = bn / tn, T = nx * ny
  if (T > 256 || T < 32) throw new Error(`bad thread count ${T}`)
  const wLoads = (BM * bk) / T, aLoads = (BN * bk) / T
  if (!Number.isInteger(wLoads) || !Number.isInteger(aLoads)) throw new Error("loads don't divide")
  const k4s = o.h2 ? bk : 2 * bk
  const elem = o.h2 ? "vec4<u32>" : "vec4<f32>"
  const L: string[] = []
  // Thread (lx, ly): row groups ig·nx + lx, quads jq·ny + ly (tokFast: lx indexes quads).
  const loadRegs = (indent: string, kq = "kq") => {
    const s: string[] = []
    for (let i = 0; i < wLoads; i++) s.push(`${indent}rw${i} = textureLoad(tW, vec2<i32>(((wxl${i} + ${kq}) >> 1u) + tl${i}, wyl${i}), 0);`)
    for (let i = 0; i < aLoads; i++) s.push(`${indent}ra${i} = textureLoad(tA, vec2<i32>((${kq} >> 1u) + tal${i}, mal${i}), 0);`)
    return s
  }
  const storeRegs = (indent: string) => {
    const s: string[] = []
    for (let i = 0; i < wLoads; i++)
      s.push(o.h2 ? `${indent}Ws[tl${i} * ${BM} + rl${i}] = rw${i};` : `${indent}Ws[(2 * tl${i}) * ${BM} + rl${i}] = lo(rw${i}); Ws[(2 * tl${i} + 1) * ${BM} + rl${i}] = hi(rw${i});`)
    for (let i = 0; i < aLoads; i++)
      s.push(o.h2 ? `${indent}As[tal${i} * ${BN} + kl${i}] = ra${i};` : `${indent}As[(2 * tal${i}) * ${BN} + kl${i}] = lo(ra${i}); As[(2 * tal${i} + 1) * ${BN} + kl${i}] = hi(ra${i});`)
    return s
  }
  const math: string[] = []
  for (let k = 0; k < k4s; k++) {
    math.push(`    {`)
    for (let i = 0; i < tm; i++) for (let r = 0; r < 4; r++) math.push(`      let W${i}_${r} = Ws[${k * BM} + 4 * (${i * nx} + lx) + ${r}];`)
    for (let jq = 0; jq < tn; jq++) for (let t = 0; t < 4; t++) math.push(`      let A${jq}_${t} = As[${k * BN} + 4 * (${jq * ny} + ly) + ${t}];`)
    const halves = o.h2 ? ["lo", "hi"] : [""]
    for (const h of halves) {
      math.push(`      {`)
      for (let i = 0; i < tm; i++) for (let r = 0; r < 4; r++) math.push(`        let w${i}_${r} = ${h}(W${i}_${r});`)
      for (let jq = 0; jq < tn; jq++) for (let t = 0; t < 4; t++) {
        math.push(`        { let a = ${h}(A${jq}_${t});`)
        for (let i = 0; i < tm; i++) math.push(`          acc_${4 * jq + t}_${i} += vec4<f32>(dot(w${i}_0, a), dot(w${i}_1, a), dot(w${i}_2, a), dot(w${i}_3, a));`)
        math.push(`        }`)
      }
      math.push(`      }`)
    }
    math.push(`    }`)
  }
  const decl: string[] = []
  for (let j = 0; j < tn * 4; j++) for (let i = 0; i < tm; i++) decl.push(`  var acc_${j}_${i} = vec4<f32>(0.0);`)
  for (let i = 0; i < wLoads; i++) decl.push(`  let ew${i} = ${i * T} + lid; let rl${i} = ew${i} / ${bk}; let tl${i} = ew${i} % ${bk};
  let rowl${i} = 4 * gBase + rl${i} - u.row0; let cl${i} = rowl${i} / u.hd; let wxl${i} = cl${i} * u.kqW - u.kq0; let wyl${i} = rowl${i} - cl${i} * u.hd;
  var rw${i}: vec4<u32>;`)
  for (let i = 0; i < aLoads; i++) decl.push(`  let ea${i} = ${i * T} + lid; let kl${i} = ea${i} / ${bk}; let tal${i} = ea${i} % ${bk}; let mal${i} = 4 * q0 + kl${i};
  var ra${i}: vec4<u32>;`)
  const S: string[] = []
  for (let jq = 0; jq < tn; jq++) for (let i = 0; i < tm; i++)
    S.push(`  { let q = q0 + ${jq * ny} + ly; let g = gBase + ${i * nx} + lx;
    if (q < u.sliceRows) {
      let p = vec2<i32>(g, slice * u.sliceRows + q);
      textureStore(out0, p, packH2(acc_${4 * jq}_${i}, acc_${4 * jq + 1}_${i}));
      textureStore(out1, p, packH2(acc_${4 * jq + 2}_${i}, acc_${4 * jq + 3}_${i}));
    } }`)
  const body = o.prefetch
    ? [
        ...loadRegs("  ", "kq0"),
        `  for (var kq = kq0; kq < kq1; kq += ${2 * bk}) {`,
        ...storeRegs("    "),
        `    workgroupBarrier();`,
        `    if (kq + ${2 * bk} < kq1) {`,
        `      let kn = kq + ${2 * bk};`,
        ...loadRegs("      ", "kn"),
        `    }`,
        ...math,
        `    workgroupBarrier();`,
        `  }`
      ]
    : [
        `  for (var kq = kq0; kq < kq1; kq += ${2 * bk}) {`,
        ...loadRegs("    "),
        ...storeRegs("    "),
        `    workgroupBarrier();`,
        ...math,
        `    workgroupBarrier();`,
        `  }`
      ]
  const code = `${UNI}
var<workgroup> Ws: array<${elem}, ${k4s * BM}>;
var<workgroup> As: array<${elem}, ${k4s * BN}>;
@compute @workgroup_size(${T})
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  let lid = i32(li);
  let lx = ${o.tokFast ? `lid / ${ny}` : `lid % ${nx}`};
  let ly = ${o.tokFast ? `lid % ${ny}` : `lid / ${nx}`};
  let rt = i32(wg.${o.swap ? "y" : "x"});
  let yt = i32(wg.${o.swap ? "x" : "y"});
  let qT = (u.sliceRows + ${bn - 1}) / ${bn};
  let slice = yt / qT;
  let q0 = (yt - slice * qT) * ${bn};
  let gBase = i32(u.rect.x) + rt * ${bm};
  let kq0 = slice * u.kqSlice;
  let kq1 = min(u.kq, kq0 + u.kqSlice);
${decl.join("\n")}
${body.join("\n")}
${S.join("\n")}
}`
  return {
    code,
    inputs: [{ name: "uW", kind: "u" }, { name: "uA", kind: "u" }],
    uniforms: UNIFORMS,
    groups: (rect) => {
      const rows = Math.ceil(rect.width / bm)
      const ys = (rect.height / cur.sliceRows) * Math.ceil(cur.sliceRows / bn)
      return o.swap ? [ys, rows] : [rows, ys]
    }
  }
}

/**
 * No dequantization: the 4-bit weights straight from their storage buffer
 * (16 bytes a row a block, as vec4<u32>), each block's sums scaled by the
 * rows' scales (binding uS) as they're added. TM row groups × 4 tokens a
 * thread, workgroups WX rect rows × WY threads' row groups, `trips` blocks a
 * slice. A block in 4 steps of one h2 texel of activations (8 values): the
 * low nibbles of two words, then the high nibbles.
 */
function quant(o: { tm: number; wx: number; wy: number; trips: number; late?: boolean }): NativeSpec {
  const I = Array.from({ length: o.tm }, (_, i) => i), T = [0, 1, 2, 3], R = [0, 1, 2, 3]
  const late = o.late !== false
  const step = (e: number) => {
    const nib = e < 2 ? "nlo" : "nhi", c = "xyzw"[2 * (e & 1)], c1 = "xyzw"[2 * (e & 1) + 1]
    return `    {
${T.map((t) => `      let A${t} = textureLoad(tA, vec2<i32>(4 * kb + ${e}, m0 + ${t}), 0);`).join("\n")}
${["lo", "hi"].map((h, k) => `      {
${I.map((i) => R.map((r) => `        let w${i}_${r} = ${nib}(Q${i}_${r}.${k ? c1 : c})${late ? "" : ` * d${i}.${"xyzw"[r]}`};`).join("\n")).join("\n")}
${T.map((t) => `        { let a = ${h}(A${t});
${I.map((i) => `          ${late ? "bd" : "acc"}${t}_${i} += vec4<f32>(dot(w${i}_0, a), dot(w${i}_1, a), dot(w${i}_2, a), dot(w${i}_3, a));`).join("\n")}
        }`).join("\n")}
      }`).join("\n")}
    }`
  }
  const code = `
struct Uni { rect: vec4<u32>, kq: i32, kqW: i32, kq0: i32, hd: i32, sliceRows: i32, kqSlice: i32, row0: i32, kb: i32, h: i32, rowTexels: i32, width: i32 }
@group(0) @binding(0) var<uniform> u: Uni;
@group(0) @binding(1) var tA: texture_2d<u32>;
@group(0) @binding(2) var tS: texture_2d<f32>;
@group(0) @binding(3) var<storage, read> bQ: array<vec4<u32>>;
@group(0) @binding(4) var out0: texture_storage_2d<rgba32uint, write>;
@group(0) @binding(5) var out1: texture_storage_2d<rgba32uint, write>;
fn lo(v: vec4<u32>) -> vec4<f32> { return vec4<f32>(unpack2x16float(v.x), unpack2x16float(v.y)); }
fn hi(v: vec4<u32>) -> vec4<f32> { return vec4<f32>(unpack2x16float(v.z), unpack2x16float(v.w)); }
/* A word's 4 low (high) nibbles less 8: each flipped at its top bit, then sign-extended. */
fn nlo(w: u32) -> vec4<f32> { return vec4<f32>(bitcast<vec4<i32>>(vec4<u32>(w ^ 0x88888888u) << vec4<u32>(28u, 20u, 12u, 4u)) >> vec4<u32>(28u)); }
fn nhi(w: u32) -> vec4<f32> { return vec4<f32>(bitcast<vec4<i32>>(vec4<u32>(w ^ 0x88888888u) << vec4<u32>(24u, 16u, 8u, 0u)) >> vec4<u32>(28u)); }
fn packH2(a0: vec4<f32>, b0: vec4<f32>) -> vec4<u32> {
  let a = clamp(a0, vec4<f32>(-65504.0), vec4<f32>(65504.0));
  let b = clamp(b0, vec4<f32>(-65504.0), vec4<f32>(65504.0));
  return vec4<u32>(pack2x16float(a.xy), pack2x16float(a.zw), pack2x16float(b.xy), pack2x16float(b.zw));
}
@compute @workgroup_size(${o.wx}, ${o.wy})
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  if (id.x >= u.rect.w || id.y * ${o.tm}u >= u.rect.z) { return; }
  let y = i32(id.x + u.rect.y);
  let g0 = i32(id.y * ${o.tm}u + u.rect.x);
  let slice = y / u.sliceRows;
  let m0 = 4 * (y - slice * u.sliceRows);
  let kb0 = slice * (u.kqSlice >> 3u);
${I.map((i) => `  let ok${i} = ${i === 0 ? "true" : `g0 + ${i} < i32(u.rect.x + u.rect.z)`};
  let gg${i} = ${i === 0 ? "g0" : `select(g0, g0 + ${i}, ok${i})`};
  let col${i} = gg${i} / u.h;
  let gr${i} = gg${i} - col${i} * u.h;
  let qb${i} = (gr${i} * u.width + col${i} * u.rowTexels) >> 2u;
  let sx${i} = col${i} * u.kb;
${T.map((t) => `  var acc${t}_${i} = vec4<f32>(0.0);`).join("\n")}`).join("\n")}
  for (var it = 0; it < ${o.trips}; it++) {
    let kb = kb0 + it;
${I.map((i) => `    let d${i} = textureLoad(tS, vec2<i32>(sx${i} + kb, gr${i}), 0);
${R.map((r) => `    let Q${i}_${r} = bQ[qb${i} + 4 * kb + ${r}];`).join("\n")}
${late ? T.map((t) => `    var bd${t}_${i} = vec4<f32>(0.0);`).join("\n") : ""}`).join("\n")}
${[0, 1, 2, 3].map(step).join("\n")}
${late ? I.map((i) => T.map((t) => `    acc${t}_${i} += d${i} * bd${t}_${i};`).join("\n")).join("\n") : ""}
  }
${I.map((i) => `  if (ok${i}) {
    textureStore(out0, vec2<i32>(g0 + ${i}, y), packH2(acc0_${i}, acc1_${i}));
    textureStore(out1, vec2<i32>(g0 + ${i}, y), packH2(acc2_${i}, acc3_${i}));
  }`).join("\n")}
}`
  return {
    code,
    inputs: [{ name: "uA", kind: "u" }, { name: "uS", kind: "f" }],
    buffers: [{ name: "bQ" }],
    uniforms: [...UNIFORMS, ...["uKb", "uH", "uRowTexels", "uWidth"].map((name) => ({ name, type: "int" as const }))],
    groups: (rect) => [Math.ceil(rect.height / o.wx), Math.ceil(rect.width / (o.wy * o.tm))]
  }
}

/**
 * The buffer dequant a thread an h2 texel, as the engine ran it before a
 * thread took a row's block (native.ts's bufferDequant): each texel fetches
 * its two words and the row's scale twice.
 */
function dequantTexels(): string {
  return `
struct Uni { rect: vec4<u32>, kb: i32, h: i32, rowTexels: i32, width: i32, kq: i32, kq0: i32, hd: i32, n: i32, row0: i32, stride: i32 }
@group(0) @binding(0) var<uniform> u: Uni;
@group(0) @binding(1) var tS: texture_2d<f32>;
@group(0) @binding(2) var<storage, read> bW: array<u32>;
@group(0) @binding(3) var<storage, read_write> out0: array<vec4<u32>>;
fn deq(x: i32, y: i32) -> vec4<f32> {
  let c = x / u.kq;
  let kq = x - c * u.kq + u.kq0;
  var row = c * u.hd + y;
  if (row >= u.n) { return vec4<f32>(0.0); }
  row += u.row0;
  let g = row / 4;
  let r = row - g * 4;
  let col = g / u.h;
  let gr = g - col * u.h;
  let kb = kq / 8;
  let j = kq - kb * 8;
  let word = bW[gr * u.width + col * u.rowTexels + kb * 16 + r * 4 + (j & 3)];
  let b = (vec4<u32>(word) >> vec4<u32>(0u, 8u, 16u, 24u)) & vec4<u32>(255u);
  let q = vec4<f32>(select(b >> vec4<u32>(4u), b & vec4<u32>(15u), j < 4));
  let d = textureLoad(tS, vec2<i32>(col * u.kb + kb, gr), 0)[r];
  return (q - 8.0) * d;
}
@compute @workgroup_size(64, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  if (id.x >= u.rect.z || id.y >= u.rect.w) { return; }
  let p = vec2<i32>(id.xy + u.rect.xy);
  let hk = u.kq >> 1u;
  let c = p.x / hk;
  let x = c * u.kq + 2 * (p.x - c * hk);
  let a = clamp(deq(x, p.y), vec4<f32>(-65504.0), vec4<f32>(65504.0));
  let b = clamp(deq(x + 1, p.y), vec4<f32>(-65504.0), vec4<f32>(65504.0));
  out0[p.y * u.stride + p.x] = vec4<u32>(pack2x16float(a.xy), pack2x16float(a.zw), pack2x16float(b.xy), pack2x16float(b.zw));
}`
}

/**
 * The buffer dequant with a thread a pair of words (`words` 2: their low
 * nibbles are one h2 texel and their high nibbles the texel two on) or,
 * with `rows` 4, a block of a row group (16 words, 16 texels). The engine's
 * is between them: a thread a row's block (4 words).
 */
function dequantPairs(words: 2 | 4, rows: 1 | 4 = 1): string {
  const P = words === 4 ? [0, 1] : [0]
  const R = rows === 4 ? [0, 1, 2, 3] : [0]
  return `
struct Uni { rect: vec4<u32>, kb: i32, h: i32, rowTexels: i32, width: i32, kq: i32, kq0: i32, hd: i32, n: i32, row0: i32, stride: i32 }
@group(0) @binding(0) var<uniform> u: Uni;
@group(0) @binding(1) var tS: texture_2d<f32>;
@group(0) @binding(2) var<storage, read> bW: array<u32>;
@group(0) @binding(3) var<storage, read_write> out0: array<vec4<u32>>;
fn nib(w: u32, s: vec4<u32>) -> vec4<f32> { return vec4<f32>((vec4<u32>(w) >> s) & vec4<u32>(15u)) - 8.0; }
fn h2(a0: vec4<f32>, b0: vec4<f32>) -> vec4<u32> {
  let a = clamp(a0, vec4<f32>(-65504.0), vec4<f32>(65504.0));
  let b = clamp(b0, vec4<f32>(-65504.0), vec4<f32>(65504.0));
  return vec4<u32>(pack2x16float(a.xy), pack2x16float(a.zw), pack2x16float(b.xy), pack2x16float(b.zw));
}
@compute @workgroup_size(64, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  // Threads a column: its h2 texels (kq / 2) over ${words}.
  let per = u.kq / ${2 * words};
  let c = i32(id.x) / per;
  let t = i32(id.x) - c * per;
  let y = ${rows} * i32(id.y);
  if (y >= i32(u.rect.w) || c * (u.kq >> 1u) >= i32(u.rect.z)) { return; }
  let kbl = t ${words === 4 ? "" : ">> 1u"};
  let at = y * u.stride + c * (u.kq >> 1u) + 4 * kbl${words === 4 ? "" : " + (t & 1)"};
  var row = c * u.hd + y;
  if (row >= u.n) {
${R.map((r) => P.map((p) => `    out0[at + ${r} * u.stride + ${p}] = vec4<u32>(0u); out0[at + ${r} * u.stride + ${p} + 2] = vec4<u32>(0u);`).join("\n")).join("\n")}
    return;
  }
  row += u.row0;
  let g = row / 4;
  let r = row - g * 4;
  let col = g / u.h;
  let gr = g - col * u.h;
  let kb = (u.kq0 >> 3u) + kbl;
  let ds = textureLoad(tS, vec2<i32>(col * u.kb + kb, gr), 0);
  let wb = gr * u.width + col * u.rowTexels + kb * 16 + r * 4${words === 4 ? "" : " + 2 * (t & 1)"};
${R.map((rr) => `  {
    let d = ds[${rows === 4 ? rr : "r"}];
${P.map((p) => `    {
      let w0 = bW[wb + ${4 * rr + 2 * p}];
      let w1 = bW[wb + ${4 * rr + 2 * p + 1}];
      out0[at + ${rr} * u.stride + ${p}] = h2(nib(w0, vec4<u32>(0u, 8u, 16u, 24u)) * d, nib(w1, vec4<u32>(0u, 8u, 16u, 24u)) * d);
      out0[at + ${rr} * u.stride + ${p} + 2] = h2(nib(w0, vec4<u32>(4u, 12u, 20u, 28u)) * d, nib(w1, vec4<u32>(4u, 12u, 20u, 28u)) * d);
    }`).join("\n")}
  }`).join("\n")}
}`
}

type Ctx = { kqSlice: number }
type Variant = { name: string; spec: (c: Ctx) => NativeSpec; buf?: boolean; engine?: boolean }
const VARIANTS: Variant[] = [
  { name: "d11", spec: () => direct({ tm: 1, tn: 1, wx: 8, wy: 8 }) },
  { name: "d11_cu2", spec: (c) => direct({ tm: 1, tn: 1, wx: 8, wy: 8, trips: c.kqSlice / 2, u: 2 }) },
  { name: "d11_bwc", spec: (c) => direct({ tm: 1, tn: 1, wx: 8, wy: 8, buf: "w", trips: c.kqSlice / 2 }) },
  { name: "d11_bwc_p", spec: (c) => direct({ tm: 1, tn: 1, wx: 8, wy: 8, buf: "w", ptr: true, trips: c.kqSlice / 2 }) },
  { name: "d11_bwc_ps", spec: (c) => direct({ tm: 1, tn: 1, wx: 8, wy: 8, buf: "w", ptr: true, swap: true, trips: c.kqSlice / 2 }) },
  { name: "d11_bwc_p4x16", spec: (c) => direct({ tm: 1, tn: 1, wx: 4, wy: 16, buf: "w", ptr: true, trips: c.kqSlice / 2 }) },
  { name: "d11_bwc_ps16x4", spec: (c) => direct({ tm: 1, tn: 1, wx: 16, wy: 4, buf: "w", ptr: true, swap: true, trips: c.kqSlice / 2 }) },
  { name: "d11_bc_p", spec: (c) => direct({ tm: 1, tn: 1, wx: 8, wy: 8, buf: true, ptr: true, trips: c.kqSlice / 2 }) },
  // The engine's own kernel (native.ts's promptMatmulBuffer), one and two row groups a thread.
  { name: "eng1", spec: (c) => promptMatmulBuffer({ trips: c.kqSlice / 2, rowGroups: 1 }), engine: true },
  { name: "eng2", spec: (c) => promptMatmulBuffer({ trips: c.kqSlice / 2, rowGroups: 2 }), engine: true },
  // Straight from the 4-bit weights (no dequantization pass).
  { name: "q1_16x4", spec: (c) => quant({ tm: 1, wx: 16, wy: 4, trips: c.kqSlice / 8 }) },
  { name: "q1_8x8", spec: (c) => quant({ tm: 1, wx: 8, wy: 8, trips: c.kqSlice / 8 }) },
  { name: "q2_8x8", spec: (c) => quant({ tm: 2, wx: 8, wy: 8, trips: c.kqSlice / 8 }) },
  { name: "q1e_16x4", spec: (c) => quant({ tm: 1, wx: 16, wy: 4, trips: c.kqSlice / 8, late: false }) }
]

// ------------------------------------------------------------ run

async function makeGpu() {
  if (params.get("dev") === "engine") return (await WebGpu.create())!
  const adapter = (await navigator.gpu.requestAdapter({ powerPreference: "high-performance" }))!
  const want = (name: keyof GPUSupportedLimits) => adapter.limits[name] as number
  const features = (["timestamp-query", "subgroups", "shader-f16", "chromium-experimental-subgroup-matrix"] as const).filter((f) => adapter.features.has(f as GPUFeatureName))
  const device = await adapter.requestDevice({
    requiredFeatures: features as unknown as GPUFeatureName[],
    requiredLimits: {
      maxTextureDimension2D: want("maxTextureDimension2D"),
      maxStorageTexturesPerShaderStage: want("maxStorageTexturesPerShaderStage"),
      maxSampledTexturesPerShaderStage: want("maxSampledTexturesPerShaderStage"),
      maxBufferSize: want("maxBufferSize"),
      maxComputeWorkgroupStorageSize: want("maxComputeWorkgroupStorageSize"),
      maxComputeInvocationsPerWorkgroup: want("maxComputeInvocationsPerWorkgroup"),
      maxComputeWorkgroupSizeX: want("maxComputeWorkgroupSizeX")
    }
  })
  const info = adapter.info
  log(`adapter features: ${[...adapter.features].join(",")}`)
  log(`adapter info: ${JSON.stringify({ vendor: info.vendor, architecture: info.architecture, device: info.device, description: info.description, subgroupMinSize: (info as unknown as Record<string, unknown>).subgroupMinSize, subgroupMaxSize: (info as unknown as Record<string, unknown>).subgroupMaxSize, subgroupMatrixConfigs: (info as unknown as Record<string, unknown>).subgroupMatrixConfigs })}`)
  log(`limits: wgStorage ${want("maxComputeWorkgroupStorageSize")} invocations ${want("maxComputeInvocationsPerWorkgroup")} tex ${want("maxTextureDimension2D")}`)
  return new WebGpu(device, `WebGPU ${[info.vendor, info.architecture, info.device, info.description].filter(Boolean).join(" ")}`, info)
}

type Run = { name: string; pass: Pass; key: string }

function halvesOf(bytes: Uint8Array) {
  const u16 = new Uint16Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 2)
  const f = new Float32Array(u16.length)
  for (let i = 0; i < u16.length; i++) f[i] = fromHalf(u16[i])
  return f
}
function diff(a: Float32Array, b: Float32Array) {
  let d = 0, m = 0, bad = 0
  for (let i = 0; i < a.length; i++) {
    if (!Number.isFinite(a[i]) || !Number.isFinite(b[i])) { if (!(Number.isNaN(a[i]) && Number.isNaN(b[i]))) bad++; continue }
    d = Math.max(d, Math.abs(a[i] - b[i]))
    m = Math.max(m, Math.abs(a[i]))
  }
  return { rel: m ? d / m : 0, bad }
}
const median = (xs: number[]) => { const s = [...xs].sort((a, b) => a - b); return s.length ? (s.length % 2 ? s[(s.length - 1) / 2] : (s[s.length / 2 - 1] + s[s.length / 2]) / 2) : NaN }

if (params.has("code")) {
  const v = VARIANTS.find((x) => x.name === params.get("code"))!
  for (const kqSlice of [128, 256, 480]) log(`CODE ${v.name} kqSlice ${kqSlice}\n${v.spec({ kqSlice }).code}`)
  log("GEX DONE")
  throw new Error("code only")
}
try {
  const gpu = await makeGpu()
  if (params.get("probe") === "1") throw new Error("probe only")
  gpu.setPhase("prompt")
  gpu.coopThreads = 32768
  log(`${navigator.userAgent} | ${gpu.limits.renderer} | device features ${[...gpu.device.features].join(",")} | M ${M}`)
  const shapes = params.get("shapes") ? ALL_SHAPES.filter((s) => params.get("shapes")!.split(",").includes(s.name)) : ALL_SHAPES
  const pick = params.get("v")?.split(",")
  // (Also "x<tm><tn>_<wx>x<wy>[s][h4|h8]": the engine kernel's shape family, buffer weights, constant trips.)
  const parsed = (pick ?? []).flatMap((name) => {
    const m = /^x(\d)(\d)_(\d+)x(\d+)(s?)(h4|h8)?$/.exec(name)
    if (m) return [{ name, spec: (c: Ctx) => direct({ tm: +m[1], tn: +m[2], wx: +m[3], wy: +m[4], swap: !!m[5], math: m[6] as "h4" | "h8" | undefined, buf: "w", ptr: true, trips: c.kqSlice / 2 }) }]
    // (Also "t<bm>_<bn>_<tm><tn>_<bk>[h][p][s][f]": tiled through workgroup memory; h2 tiles, prefetch, swap, token quads fastest.)
    const t = /^t(\d+)_(\d+)_(\d)(\d)_(\d+)(h?)(p?)(s?)(f?)$/.exec(name)
    return t ? [{ name, spec: () => tiled({ bm: +t[1], bn: +t[2], tm: +t[3], tn: +t[4], bk: +t[5], h2: !!t[6], prefetch: !!t[7], swap: !!t[8], tokFast: !!t[9] }) }] : []
  })
  const variants = [...VARIANTS.filter((v) => !pick || pick.includes(v.name)), ...parsed]
  const rounds = Number(params.get("rounds") ?? 7), reps = Number(params.get("reps") ?? 6)
  const rows = M, quadRows = Math.ceil(rows / 4)
  // The engine's programs.
  const glMatmul = gpu.program(K.matmulF16({ half: false, packed: 3 }))
  const glDequant = gpu.program(K.dequantBytes({ zeros: false, words: false, unorm: true, packed: false, h2: true }))
  const nativeDequant = gpu.nativeProgram(bufferDequant({ h2: true }))
  // Shared: activations (h2, capacity rows), the weight scratch (h2: SCRATCH_W / 2 × HD).
  const kMax = 7680
  const aWords = new Uint32Array((kMax / 8) * CAPACITY * 4)
  for (let i = 0; i < aWords.length; i++) aWords[i] = (toHalf(Math.random() * 2 - 1) | (toHalf(Math.random() * 2 - 1) << 16)) >>> 0
  const A = gpu.texture(GL.RGBA32UI, kMax / 8, CAPACITY, { format: GL.RGBA_INTEGER, type: GL.UNSIGNED_INT, pixels: aWords })
  const scratch = gpu.texture(GL.RGBA32UI, SCRATCH_W / 2, HD)
  const results: Record<string, Record<string, number[]>> = {}
  const checks: Record<string, Record<string, string>> = {}
  for (const shape of shapes) {
    const { N, K: Kd } = shape
    const groups = N / 4, blocks = Kd / 32, rowTexels = blocks * 16, kq = Kd / 4
    const { kbSlice, slices } = quadSlices(N, Kd, Math.min(QUAD_MAX_SPLIT, Math.floor(CAPACITY / Math.ceil(CAPACITY / 4))))
    const plan = f16Plan({ hd: HD, width: SCRATCH_W }, N, kbSlice, slices)
    if (plan.group !== slices || plan.rows !== N) throw new Error(`${shape.name}: the plan is chunked (${JSON.stringify(plan)}); not modelled`)
    const kqSlice = kbSlice * 8, kqW = kq
    // 4-bit weights as the engine holds them (RGBA8 unorm bytes, one column), scales RGBA16F.
    const qBytes = new Uint8Array(rowTexels * groups * 4)
    for (let i = 0; i < qBytes.length; i += 65536) crypto.getRandomValues(qBytes.subarray(i, Math.min(qBytes.length, i + 65536)))
    const sHalves = new Uint16Array(blocks * groups * 4)
    for (let i = 0; i < sHalves.length; i++) sHalves[i] = toHalf(0.002 + Math.random() * 0.02)
    const quants = gpu.texture(GL.RGBA8, rowTexels, groups, { format: GL.RGBA, type: GL.UNSIGNED_BYTE, pixels: qBytes })
    const scales = gpu.texture(GL.RGBA16F, blocks, groups, { format: GL.RGBA, type: GL.HALF_FLOAT, pixels: sHalves })
    const stored = gpu.toStorage(quants)!
    const outs = [gpu.texture(GL.RGBA32UI, groups, quadRows * slices), gpu.texture(GL.RGBA32UI, groups, quadRows * slices)]
    const dqUniforms = {
      uKb: { int: blocks }, uH: { int: groups }, uRowTexels: { int: rowTexels }, uWidth: { int: rowTexels },
      uKq: { int: kqW }, uKq0: { int: 0 }, uHd: { int: HD }, uN: { int: N }, uRow0: { int: 0 }
    }
    const dqRect = { x: 0, y: 0, width: (Math.ceil(N / HD) * kqW) / 2, height: Math.min(HD, N) }
    const mmUniforms = {
      uKq: { int: kq }, uKqW: { int: kqW }, uKq0: { int: 0 }, uHd: { int: HD },
      uSliceRows: { int: quadRows }, uKqSlice: { int: kqSlice }, uRow0: { int: 0 }
    }
    const mmRect = { x: 0, y: 0, width: groups, height: quadRows * slices }
    cur.sliceRows = quadRows
    const dqNative: Pass = { name: `dqn_${shape.name}`, program: nativeDequant, inputs: { uW: stored, uS: scales }, uniforms: dqUniforms, output: scratch, rect: dqRect, work: 6 }
    const dqGl: Pass = { name: `dqg_${shape.name}`, program: glDequant, inputs: { uW: quants, uS: scales }, uniforms: dqUniforms, output: scratch, rect: dqRect, work: 6 }
    const runs: Run[] = [
      { name: "base", key: "", pass: { name: `base_${shape.name}`, program: glMatmul, inputs: { uW: scratch, uA: A }, uniforms: mmUniforms, output: outs, rect: mmRect, work: kqSlice * 8 } }
    ]
    gpu.run(dqNative)
    const scratchBuf = gpu.toStorage(scratch)!, aBuf = gpu.toStorage(A)!
    // The native dequant writing a storage buffer (rows SW texels) rather than the scratch texture: raw WebGPU, its own timestamps.
    const dev = gpu.device
    const dqBufCode = bufferDequant({ h2: true, toBuffer: true }).code
    const uniBuf = dev.createBuffer({ size: 64, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST })
    const uniData = new Int32Array(16)
    uniData.set([0, 0, dqRect.width, dqRect.height, blocks, groups, rowTexels, rowTexels, kqW, 0, HD, N, 0, SW])
    dev.queue.writeBuffer(uniBuf, 0, uniData)
    const outB = dev.createBuffer({ size: SW * HD * 16, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST })
    const qs = dev.createQuerySet({ type: "timestamp", count: 2 })
    // (per: h2 texels a thread.)
    const rawDequant = (code: string, per: number, rowsPer = 1) => {
      const pipe = dev.createComputePipeline({ layout: "auto", compute: { module: dev.createShaderModule({ code }), entryPoint: "main" } })
      const group = dev.createBindGroup({ layout: pipe.getBindGroupLayout(0), entries: [
        { binding: 0, resource: { buffer: uniBuf } }, { binding: 1, resource: (scales.tex as GPUTexture).createView() },
        { binding: 2, resource: { buffer: stored.tex as GPUBuffer } }, { binding: 3, resource: { buffer: outB } }] })
      return async (n: number) => {
        gpu.flush()
        const enc = dev.createCommandEncoder()
        const cp = enc.beginComputePass({ timestampWrites: { querySet: qs, beginningOfPassWriteIndex: 0, endOfPassWriteIndex: 1 } })
        cp.setPipeline(pipe); cp.setBindGroup(0, group)
        for (let i = 0; i < n; i++) cp.dispatchWorkgroups(Math.ceil(dqRect.width / per / 64), Math.ceil(dqRect.height / rowsPer))
        cp.end()
        const res = dev.createBuffer({ size: 16, usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC })
        const rd = dev.createBuffer({ size: 16, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST })
        enc.resolveQuerySet(qs, 0, 2, res, 0); enc.copyBufferToBuffer(res, 0, rd, 0, 16)
        dev.queue.submit([enc.finish()])
        await rd.mapAsync(GPUMapMode.READ)
        const t = new BigUint64Array(rd.getMappedRange())
        const ms = Number(t[1] - t[0]) / 1e6 / n
        rd.unmap(); rd.destroy(); res.destroy()
        return ms
      }
    }
    const readOutB = async () => {
      const rd = dev.createBuffer({ size: SW * HD * 16, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST })
      const enc = dev.createCommandEncoder()
      enc.copyBufferToBuffer(outB, 0, rd, 0, SW * HD * 16)
      dev.queue.submit([enc.finish()])
      await rd.mapAsync(GPUMapMode.READ)
      const copy = new Uint32Array(rd.getMappedRange().slice(0))
      rd.unmap(); rd.destroy()
      return copy
    }
    // The other dequants against bufferDequant's buffer, texel for texel (where it writes: each column's kq / 2 texels of a row).
    const rawDq = rawDequant(dqBufCode, 4)
    const rawDqT = rawDequant(dequantTexels(), 1), rawDq2 = rawDequant(dequantPairs(2), 2), rawDq44 = rawDequant(dequantPairs(4, 4), 4, 4)
    await rawDq(1)
    const wanted = await readOutB()
    for (const [name, run] of [["dq_texels", rawDqT], ["dq_pairs2", rawDq2], ["dq_group4", rawDq44]] as const) {
      dev.queue.writeBuffer(outB, 0, new Uint32Array(SW * HD * 4))
      await run(1)
      const got = await readOutB()
      let bad = 0
      for (let y = 0; y < dqRect.height; y++) for (let x = 0; x < dqRect.width * 4; x++) if (got[y * SW * 4 + x] !== wanted[y * SW * 4 + x]) bad++
      log(`${shape.name}: ${name} vs bufferDequant's buffer: ${bad} words differ`)
    }
    await rawDq(1)
    await rawDq(1)
    const outBTex = { id: 1e9 + shapes.indexOf(shape), tex: outB, width: SW, height: HD, internal: -1, bytes: SW * HD * 16 } as unknown as Texture
    for (const v of variants) {
      let spec: NativeSpec
      try { spec = v.spec({ kqSlice }) } catch (e) { log(`${v.name}: ${(e as Error).message}`); continue }
      const program = gpu.nativeProgram({ ...spec, code: `// ${v.name} ${shape.name}\n${spec.code}` })
      runs.push({ name: v.name, key: "", pass: { name: `${v.name}_${shape.name}`, program, inputs: { bW: outBTex, bA: aBuf, uW: (v as Variant).engine ? outBTex : scratch, uA: A, uS: scales, bQ: stored }, uniforms: { ...mmUniforms, uStride: { int: SW }, uKb: { int: blocks }, uH: { int: groups }, uRowTexels: { int: rowTexels }, uWidth: { int: rowTexels } }, output: outs, rect: mmRect } })
    }
    log(`${shape.name}: N ${N} K ${Kd} slices ${slices} × kqSlice ${kqSlice}, rect ${mmRect.width}x${mmRect.height}, dequant rect ${dqRect.width}x${dqRect.height}`)
    // Checks: the GLSL dequant against the native one, each variant against the baseline, the baseline against the CPU at sample points.
    gpu.run(dqGl)
    const scrGl = await gpu.readBytes(scratch)
    gpu.run(dqNative)
    const scrNat = await gpu.readBytes(scratch)
    let dqSame = 0
    for (let i = 0; i < scrGl.length; i++) if (scrGl[i] !== scrNat[i]) dqSame++
    log(`${shape.name}: GLSL dequantBytes vs native bufferDequant: ${dqSame} bytes differ`)
    checks[shape.name] = {}
    let baseOut: [Float32Array, Float32Array] | null = null
    const errors0 = gpu.errors.length
    for (const r of runs) {
      gpu.zero(outs[0]); gpu.zero(outs[1])
      gpu.run(r.pass)
      const got: [Float32Array, Float32Array] = [halvesOf(await gpu.readBytes(outs[0])), halvesOf(await gpu.readBytes(outs[1]))]
      if (!baseOut) {
        baseOut = got
        // CPU reference at sample points.
        const u16a = new Uint16Array(aWords.buffer)
        let worst = 0, big = 0
        for (let s = 0; s < 48; s++) {
          const row = (Math.random() * N) | 0, tok = (Math.random() * rows) | 0, slice = (Math.random() * slices) | 0
          let ref = 0
          for (let k = slice * kqSlice * 4; k < Math.min(Kd, (slice + 1) * kqSlice * 4); k++) {
            const g = row >> 2, r4 = row & 3, kqi = k >> 2, i = k & 3, kb = kqi >> 3, j = kqi & 7
            const byte = qBytes[(g * rowTexels + kb * 16 + r4 * 4 + (j & 3)) * 4 + i]
            const nib = j < 4 ? byte & 15 : byte >> 4
            const d = fromHalf(sHalves[(g * blocks + kb) * 4 + r4])
            const w = fromHalf(toHalf((nib - 8) * d))
            const a = fromHalf(u16a[tok * (kMax / 8) * 8 + k])
            ref += w * a
          }
          const q = tok >> 2, t = tok & 3
          const o = t < 2 ? got[0] : got[1]
          const v = o[((slice * quadRows + q) * groups + (row >> 2)) * 8 + (t & 1) * 4 + (row & 3)]
          worst = Math.max(worst, Math.abs(v - ref)); big = Math.max(big, Math.abs(ref))
        }
        checks[shape.name].base = `cpu ${(worst / big).toExponential(1)}`
      } else {
        const d0 = diff(baseOut[0], got[0]), d1 = diff(baseOut[1], got[1])
        checks[shape.name][r.name] = `${Math.max(d0.rel, d1.rel).toExponential(1)}${d0.bad + d1.bad ? ` bad ${d0.bad + d1.bad}` : ""}`
      }
    }
    if (gpu.errors.length > errors0) log(`${shape.name} GPU errors: ${gpu.errors.slice(errors0).join(" || ").slice(0, 1500)}`)
    // Timing: rounds, variants rotated, reps each, GPU timestamps.
    const all: Array<{ name: string; pass?: Pass; raw?: (n: number) => Promise<number> }> = [{ name: "dq_native", pass: dqNative }, { name: "dq_glsl", pass: dqGl }, { name: "dq_tobuf", raw: rawDq }, { name: "dq_texels", raw: rawDqT }, { name: "dq_pairs2", raw: rawDq2 }, { name: "dq_group4", raw: rawDq44 }, ...runs.map((r) => ({ name: r.name, pass: r.pass }))]
    results[shape.name] = Object.fromEntries(all.map((r) => [r.name, [] as number[]]))
    for (const r of all) if (r.pass) gpu.run(r.pass)
    await gpu.finish()
    for (let round = 0; round < rounds; round++) {
      const order = all.map((_, i) => all[(i + round) % all.length])
      for (const r of order) {
        if (r.raw) { results[shape.name][r.name].push(await r.raw(reps)); continue }
        // (All reps in one compute pass between two timestamps: back to back, as a prompt's dispatches run.)
        gpu.profile = new Map()
        WebGpu.coarse = true
        // (?bands=n&by=rows|cols: each run as n bands of the rectangle, as the pacing cuts a prompt's matmuls; cols in 16s.)
        const bands = r.name.startsWith("dq_") || r.name === "base" ? 1 : Number(params.get("bands") ?? 1)
        const rect = r.pass!.rect!
        const cuts = Array.from({ length: bands + 1 }, (_, b) => params.get("by") === "cols"
          ? Math.min(rect.width, Math.ceil((rect.width * b) / bands / 16) * 16)
          : Math.round((rect.height * b) / bands))
        for (let i = 0; i < reps; i++)
          for (let b = 0; b < bands; b++)
            gpu.run({ ...r.pass!, rect: params.get("by") === "cols"
              ? { ...rect, x: rect.x + cuts[b], width: cuts[b + 1] - cuts[b] }
              : { ...rect, y: rect.y + cuts[b], height: cuts[b + 1] - cuts[b] } })
        const prof = await gpu.profiled()
        WebGpu.coarse = false
        gpu.profile = null
        const busy = prof!.get("busy")
        if (busy) results[shape.name][r.name].push(busy.ms / reps)
      }
    }
    const flop = 2 * rows * N * Kd
    const base = median(results[shape.name].base)
    for (const r of all) {
      const ms = median(results[shape.name][r.name])
      const tf = r.name.startsWith("dq_") ? "" : ` ${(flop / ms / 1e9).toFixed(2)} TFLOP/s, ${(base / ms).toFixed(3)}x base`
      log(`${shape.name} ${r.name}: ${(ms * 1000).toFixed(1)} µs${tf} | check ${checks[shape.name][r.name] ?? "-"} | rounds ${results[shape.name][r.name].map((x) => (x * 1000).toFixed(0)).join(",")}`)
    }
    for (const t of [quants, scales, stored, scratchBuf, aBuf, ...outs]) gpu.delete(t)
    outB.destroy(); uniBuf.destroy(); qs.destroy()
    await gpu.finish()
  }
  // Sum over the shapes (one each).
  const names = Object.keys(results[shapes[0].name])
  log(`SUM ${names.map((n) => `${n} ${(shapes.reduce((s, sh) => s + median(results[sh.name][n] ?? []), 0) * 1000).toFixed(0)}µs`).join(", ")}`)
  if (gpu.errors.length) log(`errors: ${gpu.errors.join(" || ").slice(0, 2000)}`)
} catch (e) {
  log(`ERROR: ${(e as Error).message} ${(e as Error).stack ?? ""}`)
}
log("GEX DONE")
