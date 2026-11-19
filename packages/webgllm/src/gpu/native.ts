/**
 * Kernels written for WebGPU itself, where a chain of the WebGL kernels
 * (see kernels.ts) costs more as dependent dispatches than one shader
 * that can share work across a workgroup. Each keeps the WebGL kernels'
 * inputs and outputs, so the passes around it don't change.
 */

import type { NativeSpec } from "./webgpu.js"

/** Matches kernels.ts's PRESCALE: the scaled input the next matmul reads is divided by it. */
const PRESCALE = 32

/**
 * A residual add, in one dispatch (the passes it replaces: Engine.addNormed's
 * rms of y, then the add, its sums of squares and its scaled copy, drawn as
 * regions). For token m:
 * - y = a matmul's output, as `y` says: "split", a split matmul's partial
 *   sums over uSplitIn slices (rows m·uSplitIn + k); "pair", a pair tensor
 *   (tokens 2q, 2q + 1 the halves of row q's h2 texels); "quad", a 4-token
 *   matmul's two textures (tokens 4q, 4q + 1 in uY, 4q + 2, 4q + 3 in
 *   uY_B, row q). The last two add up `slices` K-slices, uSliceStride rows
 *   apart (kernels.ts's pf and pfq);
 * - with `normed` (LFM3's sandwich norm), y · 1/rms(y) · uNorm;
 * - out = uH + that, into columns [0, texels) of the first output;
 * - its sum of squares into column texels (the next norm sums `parts`
 *   columns from there: the rest are zeros);
 * - with `next`, h2(out · uNext / PRESCALE) into the second output from
 *   column uBase (texel j: features 8j..8j+7).
 * A workgroup takes the tokens one row of y holds (1, 2 or 4), so each texel
 * of y is fetched once; each thread takes texel pairs; the sums go through
 * workgroup memory.
 */
export function fusedAdd(o: { texels: number; normed: boolean; next: boolean; y?: "split" | "pair" | "quad"; slices?: number }): NativeSpec {
  const source = o.y ?? "split"
  const T = source === "quad" ? 4 : source === "pair" ? 2 : 1
  const slices = Math.max(o.slices ?? 1, 1)
  const pairs = o.texels / 2
  if (!Number.isInteger(pairs) || pairs < 1) throw new Error(`No fused add for ${o.texels} texels.`)
  // A power of two (the sums halve their way down), each thread as many texel pairs.
  let threads = 256
  while (pairs % threads) threads >>= 1
  const per = pairs / threads
  const inputs: NativeSpec["inputs"] = [
    { name: "uH", kind: "f" },
    { name: "uY", kind: source === "split" ? "f" : "u" },
    ...(source === "quad" ? [{ name: "uY_B", kind: "u" as const }] : []),
    ...(o.normed ? [{ name: "uNorm", kind: "f" as const }] : []),
    ...(o.next ? [{ name: "uNext", kind: "f" as const }] : [])
  ]
  const binding = (name: string) => inputs.findIndex((i) => i.name === name) + 1
  const outBase = inputs.length + 1
  const p = Array.from({ length: per }, (_, i) => i)
  const tk = Array.from({ length: T }, (_, t) => t)
  // The tokens' sums: a float, or a vector of them.
  const R = T === 1 ? "f32" : `vec${T}<f32>`
  const of = (v: string, t: number) => (T === 1 ? v : `${v}[${t}]`)
  const squares = (v: string) =>
    `${R}(${tk.map((t) => p.map((i) => `dot(${v}a${i}_${t}, ${v}a${i}_${t}) + dot(${v}b${i}_${t}, ${v}b${i}_${t})`).join(" + ")).join(", ")})`
  const stores = (lines: (t: number) => string) =>
    tk.map((t) => (T === 1 ? lines(t) : `  if (m${t} < end) {\n${lines(t)}\n  }`)).join("\n")
  const code = `
struct U { rect: vec4<u32>, split: i32, parts: i32, eps: f32, base: i32, stride: i32 }
@group(0) @binding(0) var<uniform> u: U;
@group(0) @binding(${binding("uH")}) var tH: texture_2d<f32>;
@group(0) @binding(${binding("uY")}) var tY: texture_2d<${source === "split" ? "f32" : "u32"}>;
${source === "quad" ? `@group(0) @binding(${binding("uY_B")}) var tYB: texture_2d<u32>;` : ""}
${o.normed ? `@group(0) @binding(${binding("uNorm")}) var tNorm: texture_2d<f32>;` : ""}
${o.next ? `@group(0) @binding(${binding("uNext")}) var tNext: texture_2d<f32>;` : ""}
@group(0) @binding(${outBase}) var out0: texture_storage_2d<rgba32float, write>;
${o.next ? `@group(0) @binding(${outBase + 1}) var out1: texture_storage_2d<rgba32uint, write>;` : ""}
var<workgroup> red: array<${R}, ${threads}>;
fn total(v: ${R}, l: u32) -> ${R} {
  red[l] = v;
  workgroupBarrier();
  for (var s = ${threads >> 1}u; s > 0u; s >>= 1u) {
    if (l < s) { red[l] += red[l + s]; }
    workgroupBarrier();
  }
  let t = red[0];
  workgroupBarrier();
  return t;
}
fn lo(v: vec4<u32>) -> vec4<f32> { return vec4<f32>(unpack2x16float(v.x), unpack2x16float(v.y)); }
fn hi(v: vec4<u32>) -> vec4<f32> { return vec4<f32>(unpack2x16float(v.z), unpack2x16float(v.w)); }
/* Texel x of y for the tokens of its row q. */
struct Y { ${tk.map((t) => `t${t}: vec4<f32>`).join(", ")} }
fn y(x: i32, q: i32) -> Y {
  var r = Y();
${source === "split"
    ? `  for (var k = 0; k < u.split; k++) { r.t0 += textureLoad(tY, vec2<i32>(x, q * u.split + k), 0); }`
    : `  for (var k = 0; k < ${slices}; k++) {
    let at = vec2<i32>(x, q + k * u.stride);
    let a = textureLoad(tY, at, 0);
    r.t0 += lo(a);
    r.t1 += hi(a);${source === "quad" ? `
    let b = textureLoad(tYB, at, 0);
    r.t2 += lo(b);
    r.t3 += hi(b);` : ""}
  }`}
  return r;
}
fn h2(a: vec4<f32>, b: vec4<f32>) -> vec4<u32> {
  let x = clamp(a, vec4<f32>(-65504.0), vec4<f32>(65504.0));
  let z = clamp(b, vec4<f32>(-65504.0), vec4<f32>(65504.0));
  return vec4<u32>(pack2x16float(x.xy), pack2x16float(x.zw), pack2x16float(z.xy), pack2x16float(z.zw));
}
@compute @workgroup_size(${threads})
fn main(@builtin(local_invocation_index) l: u32, @builtin(workgroup_id) w: vec3<u32>) {
  let q = i32(u.rect.y) / ${T} + i32(w.x);
  let end = i32(u.rect.y + u.rect.w);
${tk.map((t) => `  let m${t} = ${T} * q + ${t};`).join("\n")}
  let j0 = i32(l) * ${per};
${p.map((i) => `  let xa${i} = 2 * (j0 + ${i});
  let xb${i} = xa${i} + 1;
  let Ya${i} = y(xa${i}, q);
  let Yb${i} = y(xb${i}, q);
${tk.map((t) => `  let ya${i}_${t} = Ya${i}.t${t};
  let yb${i}_${t} = Yb${i}.t${t};`).join("\n")}`).join("\n")}
${o.normed
    ? `  let rs = inverseSqrt(total(${squares("y")}, l) / ${o.texels * 4}.0 + u.eps);
${p.map((i) => `  let na${i} = textureLoad(tNorm, vec2<i32>(xa${i}, 0), 0);
  let nb${i} = textureLoad(tNorm, vec2<i32>(xb${i}, 0), 0);
${tk.map((t) => `  let va${i}_${t} = textureLoad(tH, vec2<i32>(xa${i}, m${t}), 0) + ya${i}_${t} * ${of("rs", t)} * na${i};
  let vb${i}_${t} = textureLoad(tH, vec2<i32>(xb${i}, m${t}), 0) + yb${i}_${t} * ${of("rs", t)} * nb${i};`).join("\n")}`).join("\n")}`
    : p.map((i) => tk.map((t) => `  let va${i}_${t} = textureLoad(tH, vec2<i32>(xa${i}, m${t}), 0) + ya${i}_${t};
  let vb${i}_${t} = textureLoad(tH, vec2<i32>(xb${i}, m${t}), 0) + yb${i}_${t};`).join("\n")).join("\n")}
${stores((t) => p.map((i) => `  textureStore(out0, vec2<i32>(xa${i}, m${t}), va${i}_${t});
  textureStore(out0, vec2<i32>(xb${i}, m${t}), vb${i}_${t});`).join("\n"))}
  let sum = total(${squares("v")}, l);
  // (Every column the next norm sums: fewer threads than those take several each.)
  for (var c = i32(l); c < u.parts; c += ${threads}) {
${stores((t) => `    textureStore(out0, vec2<i32>(${o.texels} + c, m${t}), select(vec4<f32>(0.0), vec4<f32>(${of("sum", t)}, 0.0, 0.0, 0.0), c == 0));`)}
  }
${o.next ? `${p.map((i) => `  let ta${i} = textureLoad(tNext, vec2<i32>(xa${i}, 0), 0) * ${1 / PRESCALE};
  let tb${i} = textureLoad(tNext, vec2<i32>(xb${i}, 0), 0) * ${1 / PRESCALE};`).join("\n")}
${stores((t) => p.map((i) => `  textureStore(out1, vec2<i32>(u.base + j0 + ${i}, m${t}), h2(va${i}_${t} * ta${i}, vb${i}_${t} * tb${i}));`).join("\n"))}` : ""}
}
`
  return {
    code,
    inputs,
    uniforms: [
      { name: "uSplitIn", type: "int" },
      { name: "uParts", type: "int" },
      { name: "uEps", type: "float" },
      { name: "uBase", type: "int" },
      { name: "uSliceStride", type: "int" }
    ],
    // A workgroup per row of y in the rectangle.
    groups: (rect) => [Math.ceil(rect.height / T), 1]
  }
}

/**
 * The output head through a Q6_K embedding in word texels (kernels.ts's
 * matmulQ6KHeadWords, the same math and layout), with the activations a
 * workgroup reads loaded into workgroup memory once: the 64 threads of a
 * workgroup are 64 neighbouring output texels of one row, so they share
 * their K-slice of the hidden state, which every thread of the WebGL
 * kernel fetches for itself (as many bytes as the weights, through the
 * texture units). `kbSlice`: super-blocks a slice (the whole row without
 * `split`).
 */
export function wordHead(o: { split: boolean; prescaled: boolean; kbSlice: number; ordered?: boolean }): NativeSpec {
  // ordered: the texels in the order this kernel reads them
  // (kernels.ts's ORDER_Q6_WORDS), which an iPad streams at 108 GB/s
  // against 65 for the usual order (bench/webgpu-headread.html).
  const inputs: NativeSpec["inputs"] = [
    { name: "uA", kind: "u" },
    { name: "uW", kind: "u" },
    { name: "uS", kind: "f" },
    ...(o.prescaled ? [{ name: "uR", kind: "f" as const }] : [])
  ]
  const C = "xyzw"
  const texels = 32 * o.kbSlice
  const rowBlock = (r: number) => `      {
${o.ordered ? `        let t = wx + sb * 52 + 4 + q * 12 + ${r * 3};
        let QA = textureLoad(tW, vec2<i32>(t, gr), 0);
        let QB = textureLoad(tW, vec2<i32>(t + 1, gr), 0);
        let QH = textureLoad(tW, vec2<i32>(t + 2, gr), 0);` : `        let t = wx + (sb * 4 + ${r}) * 13;
        let QA = textureLoad(tW, vec2<i32>(t + qa, gr), 0);
        let QB = textureLoad(tW, vec2<i32>(t + qa + 2, gr), 0);
        let QH = textureLoad(tW, vec2<i32>(t + qh, gr), 0);`}
        let SS = select(S${r}.zw, S${r}.xy, n == 0);
        var p = vec4<f32>(0.0);
${[0, 1, 2, 3].map((wi) => `        p += vec4<f32>(dot(q6v(q6g0(QA.${C[wi]}, QH.${C[wi]})), A${wi}0), dot(q6v(q6g1(QB.${C[wi]}, QH.${C[wi]})), A${wi}1), dot(q6v(q6g2(QA.${C[wi]}, QH.${C[wi]})), A${wi}2), dot(q6v(q6g3(QB.${C[wi]}, QH.${C[wi]})), A${wi}3));`).join("\n")}
        P.${C[r]} += dot(p, vec4<f32>(sc8w(SS.x, b0), sc8w(SS.x, b2), sc8w(SS.y, b0), sc8w(SS.y, b2)));
      }`
  const code = `
struct Uni { rect: vec4<u32>, kb: i32, h: i32, rowTexels: i32, g: i32, outW: i32, logitRows: i32, kbSlice: i32, rows: i32, aOff: i32, rowA0: i32, rOff: i32, parts: i32, texels: i32, eps: f32 }
@group(0) @binding(0) var<uniform> u: Uni;
@group(0) @binding(1) var tA: texture_2d<u32>;
@group(0) @binding(2) var tW: texture_2d<u32>;
@group(0) @binding(3) var tS: texture_2d<f32>;
${o.prescaled ? "@group(0) @binding(4) var tR: texture_2d<f32>;" : ""}
@group(0) @binding(${inputs.length + 1}) var out0: texture_storage_2d<rgba32float, write>;
var<workgroup> act: array<vec4<u32>, ${texels}>;
fn q6v(q: u32) -> vec4<f32> {
  let e = unpack2x16float((q & 0x00FF00FFu) | 0x64006400u);
  let f = unpack2x16float(((q >> 8u) & 0x00FF00FFu) | 0x64006400u);
  return vec4<f32>(e.x, f.x, e.y, f.y) - 1056.0;
}
fn q6g0(a: u32, h: u32) -> u32 { return (a & 0x0F0F0F0Fu) | ((h & 0x03030303u) << 4u); }
fn q6g1(b: u32, h: u32) -> u32 { return (b & 0x0F0F0F0Fu) | ((h & 0x0C0C0C0Cu) << 2u); }
fn q6g2(a: u32, h: u32) -> u32 { return ((a >> 4u) & 0x0F0F0F0Fu) | (h & 0x30303030u); }
fn q6g3(b: u32, h: u32) -> u32 { return ((b >> 4u) & 0x0F0F0F0Fu) | ((h >> 2u) & 0x30303030u); }
fn sc8w(w: u32, b: u32) -> f32 { return f32(bitcast<i32>(w << (24u - 8u * b)) >> 24u); }
fn lo(w: vec4<u32>) -> vec4<f32> { return vec4<f32>(unpack2x16float(w.x), unpack2x16float(w.y)); }
fn hi(w: vec4<u32>) -> vec4<f32> { return vec4<f32>(unpack2x16float(w.z), unpack2x16float(w.w)); }
@compute @workgroup_size(64)
fn main(@builtin(local_invocation_index) l: u32, @builtin(workgroup_id) w: vec3<u32>) {
  let x = i32(u.rect.x + w.x * 64u + l);
  var y = i32(u.rect.y + w.y);
  var m = y / u.logitRows;
  y -= m * u.logitRows;
${o.split ? `  let slice = m / u.rows;
  m -= slice * u.rows;
  let sb0 = slice * u.kbSlice;
  let sb1 = min(u.kb, sb0 + u.kbSlice);` : `  let sb0 = 0;
  let sb1 = u.kb;`}
  let row = ${o.prescaled ? "u.rowA0 + m" : "m"};
  // The slice's activations, once for the workgroup.
  let count = (sb1 - sb0) * 32;
  for (var i = i32(l); i < count; i += 64) { act[i] = textureLoad(tA, vec2<i32>(u.aOff + sb0 * 32 + i, row), 0); }
  workgroupBarrier();
  let g = y * u.outW + x;
  var o = vec4<f32>(0.0);
  if (g < u.g && x < i32(u.rect.x + u.rect.z)) {
    let col = g / u.h;
    let gr = g - col * u.h;
    let wx = col * u.rowTexels;
    let sx = col * u.kb;
    var acc = vec4<f32>(0.0);
    for (var sb = sb0; sb < sb1; sb++) {
      let d = textureLoad(tS, vec2<i32>(sx + sb, gr), 0);
      let S0 = textureLoad(tW, vec2<i32>(${o.ordered ? "wx + sb * 52 + 0" : "wx + (sb * 4) * 13 + 12"}, gr), 0);
      let S1 = textureLoad(tW, vec2<i32>(${o.ordered ? "wx + sb * 52 + 1" : "wx + (sb * 4 + 1) * 13 + 12"}, gr), 0);
      let S2 = textureLoad(tW, vec2<i32>(${o.ordered ? "wx + sb * 52 + 2" : "wx + (sb * 4 + 2) * 13 + 12"}, gr), 0);
      let S3 = textureLoad(tW, vec2<i32>(${o.ordered ? "wx + sb * 52 + 3" : "wx + (sb * 4 + 3) * 13 + 12"}, gr), 0);
      var P = vec4<f32>(0.0);
      for (var q = 0; q < 4; q++) {
        let n = (q >> 1u) & 1;
        let hw = q & 1;
        let a0 = (sb - sb0) * 32 + 16 * n + 2 * hw;
${[0, 1].flatMap((wp) => [0, 1, 2, 3].map((gg) => `        let u${wp}${gg} = act[a0 + ${4 * gg + wp}]; let A${2 * wp}${gg} = lo(u${wp}${gg}); let A${2 * wp + 1}${gg} = hi(u${wp}${gg});`)).join("\n")}
        let qa = 4 * n + hw;
        let qh = 8 + 2 * n + hw;
        let b0 = u32(hw);
        let b2 = u32(hw + 2);
${[0, 1, 2, 3].map(rowBlock).join("\n")}
      }
      acc += d * P;
    }
${o.prescaled ? `    var s = vec4<f32>(0.0);
    for (var p = 0; p < u.parts; p++) { s += textureLoad(tR, vec2<i32>(u.rOff + p, u.rowA0 + m), 0); }
    o = acc * (inverseSqrt(dot(s, vec4<f32>(1.0)) / f32(u.texels * 4) + u.eps) * 32.0);` : "    o = acc;"}
  }
  if (x < i32(u.rect.x + u.rect.z)) { textureStore(out0, vec2<i32>(x, i32(u.rect.y + w.y)), o); }
}
`
  return {
    code,
    inputs,
    uniforms: ["uKb", "uH", "uRowTexels", "uG", "uOutW", "uLogitRows", "uKbSlice", "uRows", "uAOff", "uRowA0", "uROff", "uParts", "uTexels"]
      .map((name): NativeSpec["uniforms"][number] => ({ name, type: "int" }))
      .concat([{ name: "uEps", type: "float" }]),
    groups: (rect) => [Math.ceil(rect.width / 64), rect.height]
  }
}

/**
 * The output head through an expanded Q6_K embedding in RGBA32UI texels
 * (kernels.ts's EXPAND_Q6_WORDS: a weight a byte, 16 a texel, in this
 * kernel's reading order), the math of kernels.ts's q6ExpandedHead: a
 * run's first word carries its signed 8-bit scale in its bytes' top 2
 * bits. unpack4x8unorm splits a word's 4 bytes; the 255 it divides by and
 * the -32 come out of the activations, once a run for the 4 rows. On a
 * Pixel 10 Pro (whose texture unit unpacks RGBA8 for free but fetches 4 of
 * them a texel), 16 bytes a fetch make the head 1.4× faster.
 */
export function expandedHead(o: { split: boolean; prescaled: boolean; wg: number }): NativeSpec {
  const inputs: NativeSpec["inputs"] = [
    { name: "uA", kind: "u" },
    { name: "uW", kind: "u" },
    { name: "uS", kind: "f" },
    ...(o.prescaled ? [{ name: "uR", kind: "f" as const }] : [])
  ]
  const WG = o.wg
  const code = `
struct Uni { rect: vec4<u32>, kb: i32, h: i32, rowTexels: i32, g: i32, outW: i32, logitRows: i32, kbSlice: i32, rows: i32, aOff: i32, rowA0: i32, rOff: i32, parts: i32, texels: i32, eps: f32 }
@group(0) @binding(0) var<uniform> u: Uni;
@group(0) @binding(1) var tA: texture_2d<u32>;
@group(0) @binding(2) var tW: texture_2d<u32>;
@group(0) @binding(3) var tS: texture_2d<f32>;
${o.prescaled ? "@group(0) @binding(4) var tR: texture_2d<f32>;" : ""}
@group(0) @binding(${inputs.length + 1}) var out0: texture_storage_2d<rgba32float, write>;
fn lo(w: vec4<u32>) -> vec4<f32> { return vec4<f32>(unpack2x16float(w.x), unpack2x16float(w.y)); }
fn hi(w: vec4<u32>) -> vec4<f32> { return vec4<f32>(unpack2x16float(w.z), unpack2x16float(w.w)); }
@compute @workgroup_size(${WG})
fn main(@builtin(local_invocation_index) l: u32, @builtin(workgroup_id) w: vec3<u32>) {
  let x = i32(u.rect.x + w.x * ${WG}u + l);
  var y = i32(u.rect.y + w.y);
  var m = y / u.logitRows;
  y -= m * u.logitRows;
${o.split ? `  let slice = m / u.rows;
  m -= slice * u.rows;
  let sb0 = slice * u.kbSlice;
  let sb1 = min(u.kb, sb0 + u.kbSlice);` : `  let sb0 = 0;
  let sb1 = u.kb;`}
  let row = ${o.prescaled ? "u.rowA0 + m" : "m"};
  let g = y * u.outW + x;
  var o = vec4<f32>(0.0);
  if (g < u.g && x < i32(u.rect.x + u.rect.z)) {
    let col = g / u.h;
    let gr = g - col * u.h;
    let wx = col * u.rowTexels;
    let sx = col * u.kb;
    var acc = vec4<f32>(0.0);
    for (var sb = sb0; sb < sb1; sb++) {
      let d = textureLoad(tS, vec2<i32>(sx + sb, gr), 0);
      var P = vec4<f32>(0.0);
      for (var q = 0; q < 8; q++) {
        for (var hf = 0; hf < 2; hf++) {
          let a = u.aOff + sb * 32 + q * 4 + hf * 2;
          let a0 = textureLoad(tA, vec2<i32>(a, row), 0);
          let a1 = textureLoad(tA, vec2<i32>(a + 1, row), 0);
          let A0 = lo(a0); let A1 = hi(a0); let A2 = lo(a1); let A3 = hi(a1);
          let c = -32.0 * dot(A0 + A1 + A2 + A3, vec4<f32>(1.0));
          let B0 = A0 * 255.0; let B1 = A1 * 255.0; let B2 = A2 * 255.0; let B3 = A3 * 255.0;
          let t = wx + sb * 64 + q * 8 + hf * 4;
${[0, 1, 2, 3].map((r) => `          {
            let T = textureLoad(tW, vec2<i32>(t + ${r}, gr), 0);
            let s = dot(unpack4x8unorm((T.x >> 6u) & 0x03030303u), vec4<f32>(255.0, 1020.0, 4080.0, 16320.0));
            P.${"xyzw"[r]} += select(s, s - 256.0, s >= 128.0) * (dot(unpack4x8unorm(T.x & 0x3F3F3F3Fu), B0) + dot(unpack4x8unorm(T.y), B1) + dot(unpack4x8unorm(T.z), B2) + dot(unpack4x8unorm(T.w), B3) + c);
          }`).join("\n")}
        }
      }
      acc += d * P;
    }
${o.prescaled ? `    var s = vec4<f32>(0.0);
    for (var p = 0; p < u.parts; p++) { s += textureLoad(tR, vec2<i32>(u.rOff + p, u.rowA0 + m), 0); }
    o = acc * (inverseSqrt(dot(s, vec4<f32>(1.0)) / f32(u.texels * 4) + u.eps) * 32.0);` : "    o = acc;"}
  }
  if (x < i32(u.rect.x + u.rect.z)) { textureStore(out0, vec2<i32>(x, i32(u.rect.y + w.y)), o); }
}
`
  return {
    code,
    inputs,
    uniforms: ["uKb", "uH", "uRowTexels", "uG", "uOutW", "uLogitRows", "uKbSlice", "uRows", "uAOff", "uRowA0", "uROff", "uParts", "uTexels"]
      .map((name): NativeSpec["uniforms"][number] => ({ name, type: "int" }))
      .concat([{ name: "uEps", type: "float" }]),
    groups: (rect) => [Math.ceil(rect.width / WG), rect.height]
  }
}

/**
 * A token's attention inputs in one pass (kernels.ts's headPrep for the
 * queries, into the cache's slots, and the keys, and values(): the same
 * math), from the q|k|v projection's `split` partial sums: a workgroup a
 * head, its threads a texel each. The head is read once and its sum of
 * squares added up in workgroup memory, where each output texel of the
 * GLSL passes read the whole head again (twice, for its RoPE partner).
 * Queries go to texel (uQX0 + q % kvWidth, uQRow0 + q / kvWidth), keys to
 * row uKRow0 from uKX0, values beside them.
 */
export function attentionPrep(o: { headDim: number; heads: number; kvHeads: number; kvWidth: number; split: number; ropeLow: number }): NativeSpec {
  const T = o.headDim / 4
  const inputs: NativeSpec["inputs"] = [
    { name: "uQKVG", kind: "f" },
    { name: "uQNorm", kind: "f" },
    { name: "uKNorm", kind: "f" },
    { name: "uRope", kind: "f" }
  ]
  const qT = o.heads * T, kvT = o.kvHeads * T
  const code = `
struct Uni { rect: vec4<u32>, pos: i32, pairTexels: i32, qx0: i32, qRow0: i32, kx0: i32, kRow0: i32, eps: f32, scale: f32 }
@group(0) @binding(0) var<uniform> u: Uni;
@group(0) @binding(1) var tP: texture_2d<f32>;
@group(0) @binding(2) var tQN: texture_2d<f32>;
@group(0) @binding(3) var tKN: texture_2d<f32>;
@group(0) @binding(4) var tRope: texture_2d<f32>;
@group(0) @binding(5) var out0: texture_storage_2d<rgba32uint, write>;
var<workgroup> sq: array<vec4<f32>, ${T}>;
var<workgroup> nv: array<vec4<f32>, ${T}>;
fn pf(x: i32) -> vec4<f32> {
  var acc = vec4<f32>(0.0);
  for (var k = 0; k < ${o.split}; k++) { acc += textureLoad(tP, vec2<i32>(x, k), 0); }
  return acc;
}
fn h2(a: vec4<f32>, b: vec4<f32>) -> vec4<u32> {
  let c = clamp(a, vec4<f32>(-65504.0), vec4<f32>(65504.0));
  let d = clamp(b, vec4<f32>(-65504.0), vec4<f32>(65504.0));
  return vec4<u32>(pack2x16float(c.xy), pack2x16float(c.zw), pack2x16float(d.xy), pack2x16float(d.zw));
}
@compute @workgroup_size(${T})
fn main(@builtin(local_invocation_index) li: u32, @builtin(workgroup_id) w: vec3<u32>) {
  let t = i32(li);
  let unit = i32(w.x);
  let isQ = unit < ${o.heads};
  let isK = !isQ && unit < ${o.heads + o.kvHeads};
  let head = select(select(unit - ${o.heads + o.kvHeads}, unit - ${o.heads}, isK), unit, isQ);
  let base = select(select(${qT + kvT}, ${qT}, isK), 0, isQ) + head * ${T};
  let v = pf(base + t);
  sq[t] = v * v;
  workgroupBarrier();
  // (Summed as the GLSL pass sums: texels in order, then the 4 lanes.)
  var s4 = vec4<f32>(0.0);
  for (var i = 0; i < ${T}; i++) { s4 += sq[i]; }
  let inv = inverseSqrt(dot(s4, vec4<f32>(1.0)) / ${o.headDim}.0 + u.eps);
  let n = v * inv * select(textureLoad(tKN, vec2<i32>(t, 0), 0), textureLoad(tQN, vec2<i32>(t, 0), 0), isQ);
  nv[t] = n;
  workgroupBarrier();
  var r = n;
  if (t < 2 * u.pairTexels) {
    let first = t < u.pairTexels;
    let tp = select(t - u.pairTexels, t + u.pairTexels, first);
    let f = select(tp, t, first);
    let p = nv[tp];
    let lo = u.pos & ${o.ropeLow - 1};
    let hi = ${o.ropeLow} + (u.pos >> ${Math.log2(o.ropeLow)}u);
    let c1 = textureLoad(tRope, vec2<i32>(f, lo), 0);
    let s1 = textureLoad(tRope, vec2<i32>(${T / 2} + f, lo), 0);
    let c2 = textureLoad(tRope, vec2<i32>(f, hi), 0);
    let s2 = textureLoad(tRope, vec2<i32>(${T / 2} + f, hi), 0);
    let cs = c1 * c2 - s1 * s2;
    let sn = s1 * c2 + c1 * s2;
    r = select(n * cs + p * sn, n * cs - p * sn, first);
  }
  workgroupBarrier();
  nv[t] = r;
  workgroupBarrier();
  if (isQ) {
    let q = head * ${T} + t;
    textureStore(out0, vec2<i32>(u.qx0 + q % ${o.kvWidth}, u.qRow0 + q / ${o.kvWidth}), bitcast<vec4<u32>>(r * u.scale));
  } else if (t < ${T / 2}) {
    if (isK) {
      textureStore(out0, vec2<i32>(u.kx0 + head * ${T / 2} + t, u.kRow0), h2(nv[2 * t], nv[2 * t + 1]));
    } else {
      textureStore(out0, vec2<i32>(u.kx0 + ${kvT / 2} + head * ${T / 2} + t, u.kRow0), h2(pf(base + 2 * t), pf(base + 2 * t + 1)));
    }
  }
}
`
  return {
    code,
    inputs,
    uniforms: [
      { name: "uPos0", type: "int" }, { name: "uPairTexels", type: "int" }, { name: "uQX0", type: "int" }, { name: "uQRow0", type: "int" },
      { name: "uKX0", type: "int" }, { name: "uKRow0", type: "int" }, { name: "uEps", type: "float" }, { name: "uScale", type: "float" }
    ],
    groups: () => [o.heads + 2 * o.kvHeads, 1]
  }
}

/**
 * A prompt batch's short convolution and its state in one dispatch
 * (kernels.ts's shortConv and convState, the same products and sums): from
 * in_proj's output B | C | x (uDt texels each; a pair tensor, `y` "pair", or
 * a 4-token matmul's two textures, "quad", in `slices` K-slices
 * uSliceStride rows apart), z = C · Σ_k w[k] · (B·x)[t - (width-1) + k]
 * as h2 into the first output, and the last width-1 tokens' B·x into the
 * second (the state: f32, oldest first). A workgroup takes an h2 texel of
 * the output and a block of tokens, a thread a row of the tensor (2 or 4
 * tokens): B·x once a token, through workgroup memory, where every output
 * texel of the GLSL pass multiplied its `width` tokens' again; tokens
 * before the batch come from the state (uState). For batches of at least
 * width-1 tokens.
 */
export function promptConv(o: { width: number; y: "pair" | "quad"; slices: number }): NativeSpec {
  const quad = o.y === "quad"
  // Tokens a row of the tensor, rows a workgroup, and the rows before a block that hold the width-1 tokens before it.
  const T = quad ? 4 : 2
  const L = 64
  const W = o.width
  const HR = Math.ceil((W - 1) / T)
  const H = HR * T
  if (W < 2 || HR > L) throw new Error(`No prompt conv for a kernel of ${W}.`)
  const tk = Array.from({ length: T }, (_, t) => t)
  const inputs: NativeSpec["inputs"] = [
    { name: "uBCx", kind: "u" },
    ...(quad ? [{ name: "uBCx_B", kind: "u" as const }] : []),
    { name: "uState", kind: "f" },
    { name: "uConv", kind: "f" }
  ]
  const binding = (name: string) => inputs.findIndex((i) => i.name === name) + 1
  const code = `
struct Uni { rect: vec4<u32>, dt: i32, stride: i32 }
@group(0) @binding(0) var<uniform> u: Uni;
@group(0) @binding(${binding("uBCx")}) var tP: texture_2d<u32>;
${quad ? `@group(0) @binding(${binding("uBCx_B")}) var tPB: texture_2d<u32>;` : ""}
@group(0) @binding(${binding("uState")}) var tState: texture_2d<f32>;
@group(0) @binding(${binding("uConv")}) var tConv: texture_2d<f32>;
@group(0) @binding(${inputs.length + 1}) var outZ: texture_storage_2d<rgba32uint, write>;
@group(0) @binding(${inputs.length + 2}) var outState: texture_storage_2d<rgba32float, write>;
/* B·x of the block's tokens and the ${H} before them: token (block's first - ${H} + i)'s two texels at 2i, 2i + 1. */
var<workgroup> Y: array<vec4<f32>, ${2 * (L * T + H)}>;
fn lo(v: vec4<u32>) -> vec4<f32> { return vec4<f32>(unpack2x16float(v.x), unpack2x16float(v.y)); }
fn hi(v: vec4<u32>) -> vec4<f32> { return vec4<f32>(unpack2x16float(v.z), unpack2x16float(v.w)); }
/* Texel x of row q's tokens (its K-slices added, as pf adds them). */
struct Row { ${tk.map((t) => `t${t}: vec4<f32>`).join(", ")} }
fn row(x: i32, q: i32) -> Row {
  var r = Row();
  for (var k = 0; k < ${Math.max(o.slices, 1)}; k++) {
    let at = vec2<i32>(x, q + k * u.stride);
    let a = textureLoad(tP, at, 0);
    r.t0 += lo(a);
    r.t1 += hi(a);${quad ? `
    let b = textureLoad(tPB, at, 0);
    r.t2 += lo(b);
    r.t3 += hi(b);` : ""}
  }
  return r;
}
fn h2(a: vec4<f32>, b: vec4<f32>) -> vec4<u32> {
  let c = clamp(a, vec4<f32>(-65504.0), vec4<f32>(65504.0));
  let d = clamp(b, vec4<f32>(-65504.0), vec4<f32>(65504.0));
  return vec4<u32>(pack2x16float(c.xy), pack2x16float(c.zw), pack2x16float(d.xy), pack2x16float(d.zw));
}
@compute @workgroup_size(${L})
fn main(@builtin(local_invocation_index) li: u32, @builtin(workgroup_id) w: vec3<u32>) {
  let l = i32(li);
  let x = i32(w.x);
  let c0 = 2 * x;
  let rows = i32(u.rect.w);
  // The block's first row of the tensor, and this thread's.
  let q0 = i32(w.y) * ${L};
  let q = q0 + l;
  let m0 = ${T} * q;
  // This row's B, x and C, texels c0 and c0 + 1.
  let B0 = row(c0, q);
  let B1 = row(c0 + 1, q);
  let X0 = row(2 * u.dt + c0, q);
  let X1 = row(2 * u.dt + c0 + 1, q);
  let C0 = row(u.dt + c0, q);
  let C1 = row(u.dt + c0 + 1, q);
${tk.map((t) => `  let y0_${t} = B0.t${t} * X0.t${t};
  let y1_${t} = B1.t${t} * X1.t${t};
  Y[2 * (${H} + ${T} * l + ${t})] = y0_${t};
  Y[2 * (${H} + ${T} * l + ${t}) + 1] = y1_${t};`).join("\n")}
  // The ${H} tokens before the block (its first threads take a row of them each): the batch's, or the state's.
  if (l < ${HR}) {
    let qh = q0 - ${HR} + l;
    if (qh >= 0) {
      let b0 = row(c0, qh);
      let b1 = row(c0 + 1, qh);
      let x0 = row(2 * u.dt + c0, qh);
      let x1 = row(2 * u.dt + c0 + 1, qh);
${tk.map((t) => `      Y[2 * (${T} * l + ${t})] = b0.t${t} * x0.t${t};
      Y[2 * (${T} * l + ${t}) + 1] = b1.t${t} * x1.t${t};`).join("\n")}
    } else {
${tk.map((t) => `      {
        // Token ${T} * qh + ${t} (before the batch): the state's row ${W - 1} + it, if it holds it.
        let s = ${W - 1} + ${T} * qh + ${t};
        Y[2 * (${T} * l + ${t})] = select(vec4<f32>(0.0), textureLoad(tState, vec2<i32>(c0, s), 0), s >= 0);
        Y[2 * (${T} * l + ${t}) + 1] = select(vec4<f32>(0.0), textureLoad(tState, vec2<i32>(c0 + 1, s), 0), s >= 0);
      }`).join("\n")}
    }
  }
  workgroupBarrier();
${Array.from({ length: W }, (_, k) => `  let w0_${k} = textureLoad(tConv, vec2<i32>(c0, ${k}), 0);
  let w1_${k} = textureLoad(tConv, vec2<i32>(c0 + 1, ${k}), 0);`).join("\n")}
${tk.map((t) => `  if (m0 + ${t} < rows) {
    // Token m0 + ${t}: taps from ${W - 1} tokens back to its own (Y's index of token m is ${H} + m - ${T} * q0).
    let at = 2 * (${H} + ${T} * l + ${t} - ${W - 1});
    var a0 = vec4<f32>(0.0);
    var a1 = vec4<f32>(0.0);
${Array.from({ length: W }, (_, k) => `    a0 = fma(w0_${k}, Y[at + ${2 * k}], a0);
    a1 = fma(w1_${k}, Y[at + ${2 * k + 1}], a1);`).join("\n")}
    textureStore(outZ, vec2<i32>(x, m0 + ${t}), h2(C0.t${t} * a0, C1.t${t} * a1));
    // The batch's last ${W - 1} tokens are the state after it.
    let i = m0 + ${t} - (rows - ${W - 1});
    if (i >= 0) {
      textureStore(outState, vec2<i32>(c0, i), y0_${t});
      textureStore(outState, vec2<i32>(c0 + 1, i), y1_${t});
    }
  }`).join("\n")}
}
`
  return {
    code,
    inputs,
    uniforms: [
      { name: "uDt", type: "int" },
      { name: "uSliceStride", type: "int" }
    ],
    // A workgroup an h2 texel of the output and a block of the tensor's rows.
    groups: (rect) => [rect.width, Math.ceil(rect.height / (L * T))]
  }
}

/**
 * A prompt batch's attention inputs in one dispatch (kernels.ts's headPrep
 * for the queries and for the keys, and its `values`: three passes, each
 * texel summing its head's squares for itself): a workgroup a head of a
 * token (the q heads, then the k, then the v), a thread a texel, the head's
 * sum of squares through workgroup memory (summed as the GLSL pass sums,
 * so the same bits: texels in order, then the 4 lanes). The q|k|v|gate projection is a pair
 * tensor (`y` "pair") or a 4-token matmul's two textures ("quad"), in
 * `slices` K-slices uSliceStride rows apart (kernels.ts's pf and pfq).
 * Queries (normalized, rotated, times uScale) go to the first output, f32
 * texel (head · T + t, m); keys (normalized, rotated) and values to the
 * second, the KV cache, as h2 at token m's slot: uSlot0 + m, wrapped at
 * uRing if there is one, in blocks of 2^uKvShift rows uKvWidth texels apart.
 */
export function promptPrep(o: { headDim: number; heads: number; kvHeads: number; y: "pair" | "quad"; slices: number; ropeLow: number }): NativeSpec {
  const T = o.headDim / 4
  if (T > 256 || T % 2) throw new Error(`No prompt prep for heads of ${o.headDim}.`)
  const quad = o.y === "quad"
  const inputs: NativeSpec["inputs"] = [
    { name: "uQKVG", kind: "u" },
    ...(quad ? [{ name: "uQKVG_B", kind: "u" as const }] : []),
    { name: "uQNorm", kind: "f" },
    { name: "uKNorm", kind: "f" },
    { name: "uRope", kind: "f" }
  ]
  const binding = (name: string) => inputs.findIndex((i) => i.name === name) + 1
  const qT = o.heads * T, kvT = o.kvHeads * T
  const code = `
struct Uni { rect: vec4<u32>, pos0: i32, pairTexels: i32, stride: i32, ring: i32, kvShift: i32, kvWidth: i32, slot0: i32, eps: f32, scale: f32 }
@group(0) @binding(0) var<uniform> u: Uni;
@group(0) @binding(${binding("uQKVG")}) var tP: texture_2d<u32>;
${quad ? `@group(0) @binding(${binding("uQKVG_B")}) var tPB: texture_2d<u32>;` : ""}
@group(0) @binding(${binding("uQNorm")}) var tQN: texture_2d<f32>;
@group(0) @binding(${binding("uKNorm")}) var tKN: texture_2d<f32>;
@group(0) @binding(${binding("uRope")}) var tRope: texture_2d<f32>;
@group(0) @binding(${inputs.length + 1}) var outQ: texture_storage_2d<rgba32float, write>;
@group(0) @binding(${inputs.length + 2}) var outKV: texture_storage_2d<rgba32uint, write>;
var<workgroup> sq: array<vec4<f32>, ${T}>;
var<workgroup> nv: array<vec4<f32>, ${T}>;
fn lo(v: vec4<u32>) -> vec4<f32> { return vec4<f32>(unpack2x16float(v.x), unpack2x16float(v.y)); }
fn hi(v: vec4<u32>) -> vec4<f32> { return vec4<f32>(unpack2x16float(v.z), unpack2x16float(v.w)); }
/* Texel x of token m. */
fn pf(x: i32, m: i32) -> vec4<f32> {
  var acc = vec4<f32>(0.0);
  for (var k = 0; k < ${Math.max(o.slices, 1)}; k++) {
    let at = vec2<i32>(x, (m >> ${quad ? 2 : 1}u) + k * u.stride);
${quad
    ? `    var t: vec4<u32>;
    if (((m >> 1u) & 1) == 1) { t = textureLoad(tPB, at, 0); } else { t = textureLoad(tP, at, 0); }`
    : `    let t = textureLoad(tP, at, 0);`}
    acc += select(lo(t), hi(t), (m & 1) == 1);
  }
  return acc;
}
fn h2(a: vec4<f32>, b: vec4<f32>) -> vec4<u32> {
  let c = clamp(a, vec4<f32>(-65504.0), vec4<f32>(65504.0));
  let d = clamp(b, vec4<f32>(-65504.0), vec4<f32>(65504.0));
  return vec4<u32>(pack2x16float(c.xy), pack2x16float(c.zw), pack2x16float(d.xy), pack2x16float(d.zw));
}
@compute @workgroup_size(${T})
fn main(@builtin(local_invocation_index) li: u32, @builtin(workgroup_id) w: vec3<u32>) {
  let t = i32(li);
  let unit = i32(w.x);
  let m = i32(w.y + u.rect.y);
  let isQ = unit < ${o.heads};
  let isK = !isQ && unit < ${o.heads + o.kvHeads};
  let head = select(select(unit - ${o.heads + o.kvHeads}, unit - ${o.heads}, isK), unit, isQ);
  let base = select(select(${qT + kvT}, ${qT}, isK), 0, isQ) + head * ${T};
  let v = pf(base + t, m);
  sq[t] = v;
  workgroupBarrier();
  // (Summed as the GLSL pass sums: texels in order, each a fused multiply-add, as its \`s += v * v\` compiles; then the 4 lanes.)
  var s4 = vec4<f32>(0.0);
  for (var i = 0; i < ${T}; i++) { s4 = fma(sq[i], sq[i], s4); }
  let inv = inverseSqrt(dot(s4, vec4<f32>(1.0)) / ${o.headDim}.0 + u.eps);
  let n = v * inv * select(textureLoad(tKN, vec2<i32>(t, 0), 0), textureLoad(tQN, vec2<i32>(t, 0), 0), isQ);
  nv[t] = n;
  workgroupBarrier();
  var r = n;
  if (t < 2 * u.pairTexels) {
    let first = t < u.pairTexels;
    let tp = select(t - u.pairTexels, t + u.pairTexels, first);
    let f = select(tp, t, first);
    let p = nv[tp];
    let pos = u.pos0 + m;
    let pl = pos & ${o.ropeLow - 1};
    let ph = ${o.ropeLow} + (pos >> ${Math.log2(o.ropeLow)}u);
    let c1 = textureLoad(tRope, vec2<i32>(f, pl), 0);
    let s1 = textureLoad(tRope, vec2<i32>(${T / 2} + f, pl), 0);
    let c2 = textureLoad(tRope, vec2<i32>(f, ph), 0);
    let s2 = textureLoad(tRope, vec2<i32>(${T / 2} + f, ph), 0);
    let cs = c1 * c2 - s1 * s2;
    let sn = s1 * c2 + c1 * s2;
    r = select(n * cs + p * sn, n * cs - p * sn, first);
  }
  workgroupBarrier();
  nv[t] = r;
  workgroupBarrier();
  if (isQ) {
    textureStore(outQ, vec2<i32>(head * ${T} + t, m), r * u.scale);
  } else if (t < ${T / 2}) {
    // Token m's slot of the cache.
    var slot = u.slot0 + m;
    if (u.ring > 0) { slot = slot % u.ring; }
    let kx = (slot >> u32(u.kvShift)) * u.kvWidth + head * ${T / 2} + t;
    let ky = slot & ((1 << u32(u.kvShift)) - 1);
    if (isK) {
      textureStore(outKV, vec2<i32>(kx, ky), h2(nv[2 * t], nv[2 * t + 1]));
    } else {
      textureStore(outKV, vec2<i32>(kx + ${kvT / 2}, ky), h2(pf(base + 2 * t, m), pf(base + 2 * t + 1, m)));
    }
  }
}
`
  return {
    code,
    inputs,
    uniforms: [
      { name: "uPos0", type: "int" }, { name: "uPairTexels", type: "int" }, { name: "uSliceStride", type: "int" }, { name: "uRing", type: "int" },
      { name: "uKvShift", type: "int" }, { name: "uKvWidth", type: "int" }, { name: "uSlot0", type: "int" }, { name: "uEps", type: "float" }, { name: "uScale", type: "float" }
    ],
    // A workgroup a head of a token: the q heads, the k heads, the v heads; the rectangle's rows are the tokens.
    groups: (rect) => [o.heads + 2 * o.kvHeads, rect.height]
  }
}

/**
 * kernels.ts's CHUNK_MAX_AT as a workgroup per 256-token chunk and row, a
 * thread a texel of 4 logits, the 64 reduced in workgroup memory (the
 * GLSL pass is a thread a chunk: 500 threads for a 128,000-word
 * vocabulary, each looping over 64 texels, at the end of every token).
 * The same penalties (repetition, presence, the allowed-token mask) and
 * the same answer: the chunk's largest logit and its position, ties to the
 * lower position. Without CHUNK_MAX_AT's per-row seen overrides (uSeenRows:
 * draft batches), which the GLSL pass keeps.
 */
export function chunkMax(): NativeSpec {
  const inputs: NativeSpec["inputs"] = [
    { name: "uLogits", kind: "f" },
    { name: "uMask", kind: "u" },
    { name: "uAllowed", kind: "u" }
  ]
  const code = `
struct Uni { rect: vec4<u32>, outW: i32, logitRows: i32, headSplit: i32, headStride: i32, penaltyRow0: i32, allowedEnabled: i32, allowedWidth: i32, allowedRows: i32, allowedStride: i32, penalty: f32, presence: f32 }
@group(0) @binding(0) var<uniform> u: Uni;
@group(0) @binding(1) var tLogits: texture_2d<f32>;
@group(0) @binding(2) var tMask: texture_2d<u32>;
@group(0) @binding(3) var tAllowed: texture_2d<u32>;
@group(0) @binding(4) var out0: texture_storage_2d<rgba32float, write>;
var<workgroup> bv: array<f32, 64>;
var<workgroup> bi: array<i32, 64>;
fn penalized(v0: f32, m: u32, id: i32, r: i32) -> f32 {
  var v = v0;
  if (u.allowedEnabled != 0) {
    let word = id >> 5u;
    let maskRow = select(r, 0, u.allowedRows == 1);
    if (maskRow >= u.allowedRows) { return -1e30; }
    let bits = textureLoad(tAllowed, vec2<i32>(word % u.allowedWidth, word / u.allowedWidth + maskRow * u.allowedStride), 0).x;
    if ((bits & (1u << u32(id & 31))) == 0u) { return -1e30; }
  }
  if (m == 2u) { return -1e30; }
  let seen = m == 1u || m == 3u;
  if (seen) { v = select(v / u.penalty, v * u.penalty, v < 0.0); }
  return select(v, v - u.presence, seen);
}
@compute @workgroup_size(64)
fn main(@builtin(local_invocation_index) l: u32, @builtin(workgroup_id) w: vec3<u32>) {
  let c = i32(w.x);
  let r = i32(w.y);
  let t = i32(l);
  let texel = c * 64 + t;
  let p = vec2<i32>(texel % u.outW, texel / u.outW);
  var v = vec4<f32>(0.0);
  for (var s = 0; s < 4; s++) {
    if (s >= u.headSplit) { break; }
    v += textureLoad(tLogits, vec2<i32>(p.x, p.y + r * u.logitRows + s * u.headStride), 0);
  }
  let base = 4 * p.x;
  let id = 4 * texel;
  let pr = r + u.penaltyRow0;
  let q = vec4<f32>(
    penalized(v.x, textureLoad(tMask, vec2<i32>(base, p.y), 0).x, id, pr),
    penalized(v.y, textureLoad(tMask, vec2<i32>(base + 1, p.y), 0).x, id + 1, pr),
    penalized(v.z, textureLoad(tMask, vec2<i32>(base + 2, p.y), 0).x, id + 2, pr),
    penalized(v.w, textureLoad(tMask, vec2<i32>(base + 3, p.y), 0).x, id + 3, pr));
  // (As the GLSL scan: strictly larger replaces, so the first of equals stays.)
  var mx = -3e38;
  var at = 0;
  if (q.x > mx) { mx = q.x; at = 4 * t; }
  if (q.y > mx) { mx = q.y; at = 4 * t + 1; }
  if (q.z > mx) { mx = q.z; at = 4 * t + 2; }
  if (q.w > mx) { mx = q.w; at = 4 * t + 3; }
  bv[l] = mx;
  bi[l] = at;
  workgroupBarrier();
  for (var s = 32u; s > 0u; s >>= 1u) {
    if (l < s) {
      let ov = bv[l + s];
      let oi = bi[l + s];
      if (ov > bv[l] || (ov == bv[l] && oi < bi[l])) { bv[l] = ov; bi[l] = oi; }
    }
    workgroupBarrier();
  }
  if (l == 0u) { textureStore(out0, vec2<i32>(c, r), vec4<f32>(bv[0], f32(bi[0]), 0.0, 0.0)); }
}
`
  return {
    code,
    inputs,
    uniforms: [
      { name: "uOutW", type: "int" }, { name: "uLogitRows", type: "int" }, { name: "uHeadSplit", type: "int" }, { name: "uHeadStride", type: "int" },
      { name: "uPenaltyRow0", type: "int" }, { name: "uAllowedEnabled", type: "int" }, { name: "uAllowedWidth", type: "int" }, { name: "uAllowedRows", type: "int" },
      { name: "uAllowedStride", type: "int" }, { name: "uPenalty", type: "float" }, { name: "uPresencePenalty", type: "float" }
    ],
    groups: (rect) => [rect.width, rect.height]
  }
}

/**
 * kernels.ts's CHUNK_ARGMAX_AT (`rows` false: row uRow's token id into an
 * R32UI texel) or CHUNK_ARGMAX_ROWS (`rows`: each row's id and logit as
 * floats), a 256-thread workgroup per row over `chunks` chunk maxima
 * (the GLSL passes are one thread scanning them all): the first chunk
 * with the largest maximum, then its position in it.
 */
export function chunkArgmax(o: { chunks: number; rows: boolean }): NativeSpec {
  const code = `
struct Uni { rect: vec4<u32>, row: i32 }
@group(0) @binding(0) var<uniform> u: Uni;
@group(0) @binding(1) var tMax: texture_2d<f32>;
@group(0) @binding(2) var out0: texture_storage_2d<${o.rows ? "rgba32float" : "r32uint"}, write>;
var<workgroup> bv: array<f32, 256>;
var<workgroup> bi: array<i32, 256>;
@compute @workgroup_size(256)
fn main(@builtin(local_invocation_index) l: u32, @builtin(workgroup_id) w: vec3<u32>) {
  let row = ${o.rows ? "i32(u.rect.y + w.y)" : "u.row"};
  var mx = -3e38;
  var at = 0;
  for (var i = i32(l); i < ${o.chunks}; i += 256) {
    let v = textureLoad(tMax, vec2<i32>(i, row), 0).x;
    if (v > mx) { mx = v; at = i; }
  }
  bv[l] = mx;
  bi[l] = at;
  workgroupBarrier();
  for (var s = 128u; s > 0u; s >>= 1u) {
    if (l < s) {
      let ov = bv[l + s];
      let oi = bi[l + s];
      if (ov > bv[l] || (ov == bv[l] && oi < bi[l])) { bv[l] = ov; bi[l] = oi; }
    }
    workgroupBarrier();
  }
  if (l == 0u) {
    let best = bi[0];
    let pos = i32(textureLoad(tMax, vec2<i32>(best, row), 0).y);
    ${o.rows
      ? "textureStore(out0, vec2<i32>(0, row), vec4<f32>(f32(best * 256 + pos), bv[0], 0.0, 0.0));"
      : "textureStore(out0, vec2<i32>(0, 0), vec4<u32>(u32(best * 256 + pos), 0u, 0u, 0u));"}
  }
}
`
  return {
    code,
    inputs: [{ name: "uMax", kind: "f" }],
    uniforms: [{ name: "uRow", type: "int" }],
    groups: (rect) => [1, o.rows ? rect.height : 1]
  }
}

/** Keys a native attention chunk takes (see flashChunks). */
export const FLASH_KEYS = 128

/**
 * A token's attention over its cache, part 1 (kernels.ts's scores,
 * attendPanel: the same math): a workgroup per KV head and FLASH_KEYS-key
 * chunk scores each key against the G query heads sharing that KV head
 * (queries from the cache's slots, as headPrep's "cache" left them), and
 * sums the chunk's softmax-weighted values, the scores in workgroup memory
 * rather than a texture. Texel h·(2U + 1) + 2u (+1) of row c: head h's
 * weighted values lo (hi) of its texel u; h·(2U + 1) + 2U: (max, sum).
 */
/**
 * Whether flashChunks has a kernel for this shape: whole groups of threads,
 * and its workgroup memory within WebGPU's guaranteed 16 KiB.
 */
export function flashFits(o: { headDim: number; heads: number; kvHeads: number; keys?: number }): boolean {
  const U = o.headDim / 8
  const G = o.heads / o.kvHeads
  const K = o.keys ?? FLASH_KEYS
  const T = 2 * K
  const groups = T / (G * U)
  const bytes = 16 * G * 2 * U + 4 * G * T + 2 * 4 * G * K + 16 * T
  return Number.isInteger(U) && U % 2 === 0 && Number.isInteger(G) && Number.isInteger(groups) && Number.isInteger(K / groups) && bytes <= 16384
}

export function flashChunks(o: { headDim: number; heads: number; kvHeads: number; kvWidth: number; keys?: number }): NativeSpec {
  const U = o.headDim / 8
  const G = o.heads / o.kvHeads
  const K = o.keys ?? FLASH_KEYS
  // Two threads a key for its scores (half of the head's texels each), then
  // every (member, texel) pair's weighted values over K / groups keys.
  const T = 2 * K
  const C = G * U
  const groups = T / C
  const per = K / groups
  if (!flashFits(o)) throw new Error(`No native attention for ${G} heads of ${o.headDim} at ${K} keys.`)
  const W = 2 * U + 1
  const Gs = Array.from({ length: G }, (_, g) => g)
  const code = `
struct Uni { rect: vec4<u32>, qSlot0: i32, ring: i32, j0: i32, keys: i32, qpos: i32, window: i32, kvShift: i32 }
@group(0) @binding(0) var<uniform> u: Uni;
@group(0) @binding(1) var tKV: texture_2d<u32>;
@group(0) @binding(2) var out0: texture_storage_2d<rgba32float, write>;
var<workgroup> q: array<vec4<f32>, ${G * 2 * U}>;
var<workgroup> part: array<f32, ${G * T}>;
var<workgroup> s: array<f32, ${G * K}>;
var<workgroup> red: array<f32, ${G * K}>;
var<workgroup> acc: array<vec4<f32>, ${T}>;
fn kvAt(x: i32, slot: i32) -> vec2<i32> {
  return vec2<i32>(x + (slot >> u32(u.kvShift)) * ${o.kvWidth}, slot & ((1 << u32(u.kvShift)) - 1));
}
fn ringed(j: i32) -> i32 { return select(j, j - (j / u.ring) * u.ring, u.ring > 0); }
fn lo(w: vec4<u32>) -> vec4<f32> { return vec4<f32>(unpack2x16float(w.x), unpack2x16float(w.y)); }
fn hi(w: vec4<u32>) -> vec4<f32> { return vec4<f32>(unpack2x16float(w.z), unpack2x16float(w.w)); }
@compute @workgroup_size(${T})
fn main(@builtin(local_invocation_index) l: u32, @builtin(workgroup_id) w: vec3<u32>) {
  let kh = i32(w.x);
  let c = i32(w.y);
  let t = i32(l);
  // The group's queries (pre-scaled f32 bits in the cache's slots).
  if (t < ${G * 2 * U}) {
    let g = t / ${2 * U};
    let qt = (kh * ${G} + g) * ${2 * U};
    let slot = ringed(u.qSlot0 + qt / ${o.kvWidth});
    let p = kvAt(qt - (qt / ${o.kvWidth}) * ${o.kvWidth}, slot);
    q[t] = bitcast<vec4<f32>>(textureLoad(tKV, vec2<i32>(p.x + (t - g * ${2 * U}), p.y), 0));
  }
  workgroupBarrier();
  // Half a key's scores: key t / 2, texels from (t % 2) · U / 2.
  let kk = t / 2;
  let h0 = (t % 2) * ${U / 2};
  let jk = u.j0 + c * ${K} + kk;
  let liveK = jk - u.j0 < u.keys && jk <= u.qpos && (u.window <= 0 || u.qpos - jk < u.window);
  let pk = kvAt(kh * ${U}, select(0, ringed(jk), liveK));
${Gs.map((g) => `  var a${g} = vec4<f32>(0.0);`).join("\n")}
  for (var x = h0; x < h0 + ${U / 2}; x++) {
    let k = textureLoad(tKV, vec2<i32>(pk.x + x, pk.y), 0);
    let kl = lo(k);
    let kv = hi(k);
${Gs.map((g) => `    a${g} += q[${g * 2 * U} + 2 * x] * kl + q[${g * 2 * U} + 2 * x + 1] * kv;`).join("\n")}
  }
${Gs.map((g) => `  part[${g * T} + t] = dot(a${g}, vec4<f32>(1.0));`).join("\n")}
  workgroupBarrier();
  // Key t's scores (masked: -1e30), every member's max at once.
  let live = t < ${K} && (u.j0 + c * ${K} + t) - u.j0 < u.keys && (u.j0 + c * ${K} + t) <= u.qpos && (u.window <= 0 || u.qpos - (u.j0 + c * ${K} + t) < u.window);
  if (t < ${K}) {
${Gs.map((g) => `    let sc${g} = select(-1e30, part[${g * T} + 2 * t] + part[${g * T} + 2 * t + 1], live);
    s[${g * K} + t] = sc${g};
    red[${g * K} + t] = sc${g};`).join("\n")}
  }
  workgroupBarrier();
  for (var st = ${K / 2}; st > 0; st >>= 1u) {
    if (t < st) {
${Gs.map((g) => `      red[${g * K} + t] = max(red[${g * K} + t], red[${g * K} + t + st]);`).join("\n")}
    }
    workgroupBarrier();
  }
${Gs.map((g) => `  let m${g} = red[${g * K}];`).join("\n")}
  workgroupBarrier();
  if (t < ${K}) {
${Gs.map((g) => `    let e${g} = select(0.0, exp(s[${g * K} + t] - m${g}), live && m${g} > -1e29);
    s[${g * K} + t] = e${g};
    red[${g * K} + t] = e${g};`).join("\n")}
  }
  workgroupBarrier();
  for (var st = ${K / 2}; st > 0; st >>= 1u) {
    if (t < st) {
${Gs.map((g) => `      red[${g * K} + t] = red[${g * K} + t] + red[${g * K} + t + st];`).join("\n")}
    }
    workgroupBarrier();
  }
${Gs.map((g) => `  let l${g} = red[${g * K}];`).join("\n")}
  // Weighted values: pair (member g, texel x) = t % ${C}, keys of group t / ${C}.
  let pair = t % ${C};
  let g = pair / ${U};
  let x = pair % ${U};
  let grp = t / ${C};
  var vlo = vec4<f32>(0.0);
  var vhi = vec4<f32>(0.0);
  for (var i = grp * ${per}; i < grp * ${per} + ${per}; i++) {
    let e = s[g * ${K} + i];
    if (e != 0.0) {
      let v = textureLoad(tKV, kvAt(${o.kvHeads * U} + kh * ${U} + x, ringed(u.j0 + c * ${K} + i)), 0);
      vlo += e * lo(v);
      vhi += e * hi(v);
    }
  }
  // The groups' sums, lo then hi through one buffer (16 KiB of workgroup memory is WebGPU's default limit).
  acc[t] = vlo;
  workgroupBarrier();
  var sl = vec4<f32>(0.0);
  if (t < ${C}) { for (var r = 0; r < ${groups}; r++) { sl += acc[r * ${C} + t]; } }
  workgroupBarrier();
  acc[t] = vhi;
  workgroupBarrier();
  if (t < ${C}) {
    var sh = vec4<f32>(0.0);
    for (var r = 0; r < ${groups}; r++) { sh += acc[r * ${C} + t]; }
    let h = kh * ${G} + g;
    textureStore(out0, vec2<i32>(h * ${W} + 2 * x, c), sl);
    textureStore(out0, vec2<i32>(h * ${W} + 2 * x + 1, c), sh);
  }
  if (t < ${G}) {
    let h = kh * ${G} + t;
${Gs.map((gg) => `    if (t == ${gg}) { textureStore(out0, vec2<i32>(h * ${W} + ${2 * U}, c), select(vec4<f32>(m${gg}, l${gg}, 0.0, 0.0), vec4<f32>(-1e30, 0.0, 0.0, 0.0), m${gg} <= -1e29)); }`).join("\n")}
  }
}
`
  return {
    code,
    inputs: [{ name: "uKV", kind: "u" }],
    uniforms: ["uQSlot0", "uRing", "uJ0", "uKeys", "uQPos", "uWindow", "uKvShift"].map((name): NativeSpec["uniforms"][number] => ({ name, type: "int" })),
    groups: (rect) => [rect.width, rect.height]
  }
}

/**
 * A token's attention, part 2 (kernels.ts's attendPanelCombine): head h's
 * chunks combined, the running max rescaling the sums, times its gate (one
 * sigmoid a head, or one a dim: `gate`), into attnOut's texel h·U + u (h2).
 */
export function flashCombine(o: { headDim: number; heads: number; split: number; gate: false | true | "dim" }): NativeSpec {
  const U = o.headDim / 8
  const W = 2 * U + 1
  const code = `
struct Uni { rect: vec4<u32>, chunks: i32, gateTexel: i32 }
@group(0) @binding(0) var<uniform> u: Uni;
@group(0) @binding(1) var tP: texture_2d<f32>;
@group(0) @binding(2) var tQKVG: texture_2d<f32>;
@group(0) @binding(3) var out0: texture_storage_2d<rgba32uint, write>;
fn pf(x: i32) -> vec4<f32> {
  var a = vec4<f32>(0.0);
  for (var k = 0; k < ${o.split}; k++) { a += textureLoad(tQKVG, vec2<i32>(x, k), 0); }
  return a;
}
fn h2(a: vec4<f32>, b: vec4<f32>) -> vec4<u32> {
  let c = clamp(a, vec4<f32>(-65504.0), vec4<f32>(65504.0));
  let d = clamp(b, vec4<f32>(-65504.0), vec4<f32>(65504.0));
  return vec4<u32>(pack2x16float(c.xy), pack2x16float(c.zw), pack2x16float(d.xy), pack2x16float(d.zw));
}
@compute @workgroup_size(${U})
fn main(@builtin(local_invocation_index) l: u32, @builtin(workgroup_id) w: vec3<u32>) {
  let h = i32(w.x);
  let x = i32(l);
  var mx = -1e30;
  var sum = 0.0;
  var vlo = vec4<f32>(0.0);
  var vhi = vec4<f32>(0.0);
  for (var c = 0; c < u.chunks; c++) {
    let ml = textureLoad(tP, vec2<i32>(h * ${W} + ${2 * U}, c), 0).xy;
    let mn = max(mx, ml.x);
    let scale = exp(mx - mn);
    let wgt = exp(ml.x - mn);
    sum = sum * scale + ml.y * wgt;
    vlo = vlo * scale + wgt * textureLoad(tP, vec2<i32>(h * ${W} + 2 * x, c), 0);
    vhi = vhi * scale + wgt * textureLoad(tP, vec2<i32>(h * ${W} + 2 * x + 1, c), 0);
    mx = mn;
  }
  let ox = h * ${U} + x;
${o.gate === "dim"
    ? `  let g0 = pf(u.gateTexel + 2 * ox);
  let g1 = pf(u.gateTexel + 2 * ox + 1);
  textureStore(out0, vec2<i32>(ox, 0), h2(vlo / (sum * (1.0 + exp(-g0))), vhi / (sum * (1.0 + exp(-g1)))));`
    : o.gate
      ? `  let gate = pf(u.gateTexel + h / 4)[h & 3];
  let scale = 1.0 / (sum * (1.0 + exp(-gate)));
  textureStore(out0, vec2<i32>(ox, 0), h2(vlo * scale, vhi * scale));`
      : `  let scale = 1.0 / sum;
  textureStore(out0, vec2<i32>(ox, 0), h2(vlo * scale, vhi * scale));`}
}
`
  return {
    code,
    inputs: [{ name: "uP", kind: "f" }, { name: "uQKVG", kind: "f" }],
    uniforms: [{ name: "uChunks", type: "int" }, { name: "uGateTexel", type: "int" }],
    groups: (rect) => [rect.width, 1]
  }
}

/**
 * A few tokens' 4-bit matvec from weights in a storage buffer (WebGpu.toStorage
 * of the byte texels: the texture's rows one after another), into a split
 * matmul's partial sums as one slice: a workgroup a row group (x) and token
 * (y), its threads striding over the K blocks so that neighbours read
 * neighbouring bytes, then the workgroup's sum. Texture reads top out far
 * below buffer reads on some GPUs (an M3 Ultra's byte texels ~300-390 GB/s,
 * its buffers past 600). Row r's block is 16 bytes: byte i of word w holds
 * weight 4w + i in its low nibble and 16 + 4w + i in its high one; zero
 * points are 8. `normed`: "weights" reads the f32 residual times the norm
 * weights (uNorm), "prescaled" the add's h2(x · weights / 32) at column
 * uAOff of uA; either way the sums are times the row's 1 / RMS, from uR's
 * partial sums of squares (kernels.ts's matmulSplitRgba4 alike).
 */
export function bufferMatvec(o: { normed: false | "weights" | "prescaled"; wg: number; format: "rgba16float" | "rgba32float"; bits?: 4 | 8; zeros?: boolean }): NativeSpec {
  const weights = o.normed === "weights"
  const rms = !!o.normed
  const A = weights
    ? Array.from({ length: 8 }, (_, j) => `    let A${j} = textureLoad(tA, vec2<i32>(kb * 8 + ${j}, m), 0) * textureLoad(tNorm, vec2<i32>(kb * 8 + ${j}, 0), 0);`).join("\n")
    : [0, 1, 2, 3].map((j) => `    let a${j} = textureLoad(tA, vec2<i32>(u.aOff + kb * 4 + ${j}, m), 0);
    let A${2 * j} = lo(a${j});
    let A${2 * j + 1} = hi(a${j});`).join("\n")
  const rows = o.bits === 8
    ? [0, 1, 2, 3].map((r) => `    {
      let q0 = bW[base + kb * 8 + ${2 * r}];
      let q1 = bW[base + kb * 8 + ${2 * r + 1}];
      b.${"xyzw"[r]} = dot(s8(q0.x), A0) + dot(s8(q0.y), A1) + dot(s8(q0.z), A2) + dot(s8(q0.w), A3) + dot(s8(q1.x), A4) + dot(s8(q1.y), A5) + dot(s8(q1.z), A6) + dot(s8(q1.w), A7);
    }`).join("\n")
    : [0, 1, 2, 3].map((r) => `    {
      let q = bW[base + kb * 4 + ${r}];
      b.${"xyzw"[r]} = 255.0 * (dot(unpack4x8unorm(q.x & NB), A0) + dot(unpack4x8unorm(q.y & NB), A1) + dot(unpack4x8unorm(q.z & NB), A2) + dot(unpack4x8unorm(q.w & NB), A3)
        + dot(unpack4x8unorm((q.x >> 4u) & NB), A4) + dot(unpack4x8unorm((q.y >> 4u) & NB), A5) + dot(unpack4x8unorm((q.z >> 4u) & NB), A6) + dot(unpack4x8unorm((q.w >> 4u) & NB), A7));
    }`).join("\n")
  const code = `
struct Uni { rect: vec4<u32>, kb: i32, h: i32, rowTexels: i32, width: i32, aOff: i32, rOff: i32, texels: i32, parts: i32, eps: f32 }
@group(0) @binding(0) var<uniform> u: Uni;
@group(0) @binding(1) var tA: texture_2d<${weights ? "f32" : "u32"}>;
@group(0) @binding(2) var tS: texture_2d<${o.zeros ? "u32" : "f32"}>;
${weights ? "@group(0) @binding(3) var tNorm: texture_2d<f32>;" : ""}
${rms ? `@group(0) @binding(${weights ? 4 : 3}) var tR: texture_2d<f32>;` : ""}
@group(0) @binding(${2 + (weights ? 1 : 0) + (rms ? 1 : 0) + 1}) var<storage, read> bW: array<vec4<u32>>;
@group(0) @binding(${2 + (weights ? 1 : 0) + (rms ? 1 : 0) + 2}) var out0: texture_storage_2d<${o.format}, write>;
var<workgroup> red: array<vec4<f32>, ${o.wg}>;
const NB = 0x0F0F0F0Fu;
fn lo(w: vec4<u32>) -> vec4<f32> { return vec4<f32>(unpack2x16float(w.x), unpack2x16float(w.y)); }
fn hi(w: vec4<u32>) -> vec4<f32> { return vec4<f32>(unpack2x16float(w.z), unpack2x16float(w.w)); }
fn s8(w: u32) -> vec4<f32> { return vec4<f32>(bitcast<vec4<i32>>(vec4<u32>(w) << vec4<u32>(24u, 16u, 8u, 0u)) >> vec4<u32>(24u)); }
@compute @workgroup_size(${o.wg})
fn main(@builtin(local_invocation_index) l: u32, @builtin(workgroup_id) w: vec3<u32>) {
  let g = i32(w.x);
  let m = i32(w.y);
  let col = g / u.h;
  let gr = g - col * u.h;
  let base = (gr * u.width + col * u.rowTexels) / 4;
  let sx = col * u.kb;
  var acc = vec4<f32>(0.0);
  for (var kb = i32(l); kb < u.kb; kb += ${o.wg}) {
${o.zeros
    ? `    // (Packed: the block's 4 scales, then its 4 zero points, as halves; see GpuMatrix.packed.)
    let sz = textureLoad(tS, vec2<i32>(sx + kb, gr), 0);
    let d = vec4<f32>(unpack2x16float(sz.x), unpack2x16float(sz.y));
    let z = vec4<f32>(unpack2x16float(sz.z), unpack2x16float(sz.w));`
    : "    let d = textureLoad(tS, vec2<i32>(sx + kb, gr), 0);"}
${A}
    let sa = dot(A0 + A1 + A2 + A3 + A4 + A5 + A6 + A7, vec4<f32>(1.0));
    var b = vec4<f32>(0.0);
${rows}
    acc += ${o.bits === 8 ? "d * b" : o.zeros ? "d * (b - z * sa)" : "d * (b - 8.0 * sa)"};
  }
  red[l] = acc;
  workgroupBarrier();
${Array.from({ length: Math.log2(o.wg) }, (_, i) => o.wg >> (i + 1)).map((st) => `  if (l < ${st}u) { red[l] += red[l + ${st}u]; }
  workgroupBarrier();`).join("\n")}
  if (l == 0u) {
    var r = red[0];
${rms ? `    var s = vec4<f32>(0.0);
    for (var p = 0; p < u.parts; p++) { s += textureLoad(tR, vec2<i32>(u.rOff + p, m), 0); }
    r *= inverseSqrt(dot(s, vec4<f32>(1.0)) / f32(u.texels * 4) + u.eps)${o.normed === "prescaled" ? " * 32.0" : ""};` : ""}
    textureStore(out0, vec2<i32>(g + i32(u.rect.x), m), r);
  }
}`
  return {
    code,
    inputs: [
      { name: "uA", kind: weights ? "f" : "u" },
      { name: "uS", kind: o.zeros ? "u" : "f" },
      ...(weights ? [{ name: "uNorm", kind: "f" as const }] : []),
      ...(rms ? [{ name: "uR", kind: "f" as const }] : [])
    ],
    buffers: [{ name: "uW" }],
    uniforms: [
      { name: "uKb", type: "int" },
      { name: "uH", type: "int" },
      { name: "uRowTexels", type: "int" },
      { name: "uWidth", type: "int" },
      { name: "uAOff", type: "int" },
      { name: "uROff", type: "int" },
      { name: "uTexels", type: "int" },
      { name: "uParts", type: "int" },
      { name: "uEps", type: "float" }
    ],
    groups: (rect) => [rect.width, rect.height]
  }
}

/**
 * bufferMatvec's work as split partial sums, a slice a lane (TUNING.bufferPairs): a thread a row pair (rows p, p + 2)
 * and block, `lanes` of them over K a row group, so that neighbouring threads
 * read neighbouring 16-byte words; a workgroup of 64 threads holds 32 / lanes
 * row groups (x) of one token (y). Neighbours on neighbouring words reads a
 * Pixel 10 Pro's buffers at ~30 GB/s where its byte texels gave ~19, and an M3
 * Ultra's past 800 where texels gave ~360; an iPhone's alike. Row r's block
 * is 16 bytes: byte i of word w holds weight 4w + i in its low nibble and
 * 16 + 4w + i in its high one (unpack4x8unorm splits them); zero points are
 * 8. `normed`: "weights" reads the f32 residual times the norm weights
 * (uNorm), "prescaled" the add's h2(x · weights / 32) at column uAOff of uA;
 * either way each slice is times the row's 1 / RMS, from uR's partial sums of
 * squares (kernels.ts's matmulSplitRgba4 alike).
 */
export function bufferPairs(o: { normed: false | "weights" | "prescaled"; lanes: number; format: "rgba16float" | "rgba32float"; bits?: 4 | 8; zeros?: boolean }): NativeSpec {
  const weights = o.normed === "weights"
  const rms = !!o.normed
  const T = 2 * o.lanes
  const G = 64 / T
  const A = weights
    ? Array.from({ length: 8 }, (_, j) => `      let A${j} = textureLoad(tA, vec2<i32>(kb * 8 + ${j}, m), 0) * textureLoad(tNorm, vec2<i32>(kb * 8 + ${j}, 0), 0);`).join("\n")
    : [0, 1, 2, 3].map((j) => `      let a${j} = textureLoad(tA, vec2<i32>(u.aOff + kb * 4 + ${j}, m), 0);
      let A${2 * j} = lo(a${j});
      let A${2 * j + 1} = hi(a${j});`).join("\n")
  const code = `
struct Uni { rect: vec4<u32>, kb: i32, h: i32, rowTexels: i32, width: i32, groups: i32, aOff: i32, rOff: i32, texels: i32, parts: i32, eps: f32 }
@group(0) @binding(0) var<uniform> u: Uni;
@group(0) @binding(1) var tA: texture_2d<${weights ? "f32" : "u32"}>;
@group(0) @binding(2) var tS: texture_2d<${o.zeros ? "u32" : "f32"}>;
${weights ? "@group(0) @binding(3) var tNorm: texture_2d<f32>;" : ""}
${rms ? `@group(0) @binding(${weights ? 4 : 3}) var tR: texture_2d<f32>;` : ""}
@group(0) @binding(${2 + (weights ? 1 : 0) + (rms ? 1 : 0) + 1}) var<storage, read> bW: array<vec4<u32>>;
@group(0) @binding(${2 + (weights ? 1 : 0) + (rms ? 1 : 0) + 2}) var out0: texture_storage_2d<${o.format}, write>;
var<workgroup> pairs: array<vec2<f32>, 64>;
fn lo(w: vec4<u32>) -> vec4<f32> { return vec4<f32>(unpack2x16float(w.x), unpack2x16float(w.y)); }
fn hi(w: vec4<u32>) -> vec4<f32> { return vec4<f32>(unpack2x16float(w.z), unpack2x16float(w.w)); }
fn rd(q: vec4<u32>, A0: vec4<f32>, A1: vec4<f32>, A2: vec4<f32>, A3: vec4<f32>, A4: vec4<f32>, A5: vec4<f32>, A6: vec4<f32>, A7: vec4<f32>) -> f32 {
  let n = 0x0F0F0F0Fu;
  return 255.0 * (dot(unpack4x8unorm(q.x & n), A0) + dot(unpack4x8unorm(q.y & n), A1) + dot(unpack4x8unorm(q.z & n), A2) + dot(unpack4x8unorm(q.w & n), A3)
    + dot(unpack4x8unorm((q.x >> 4u) & n), A4) + dot(unpack4x8unorm((q.y >> 4u) & n), A5) + dot(unpack4x8unorm((q.z >> 4u) & n), A6) + dot(unpack4x8unorm((q.w >> 4u) & n), A7));
}
fn s8(w: u32) -> vec4<f32> { return vec4<f32>(bitcast<vec4<i32>>(vec4<u32>(w) << vec4<u32>(24u, 16u, 8u, 0u)) >> vec4<u32>(24u)); }
fn rd8(q0: vec4<u32>, q1: vec4<u32>, A0: vec4<f32>, A1: vec4<f32>, A2: vec4<f32>, A3: vec4<f32>, A4: vec4<f32>, A5: vec4<f32>, A6: vec4<f32>, A7: vec4<f32>) -> f32 {
  return dot(s8(q0.x), A0) + dot(s8(q0.y), A1) + dot(s8(q0.z), A2) + dot(s8(q0.w), A3) + dot(s8(q1.x), A4) + dot(s8(q1.y), A5) + dot(s8(q1.z), A6) + dot(s8(q1.w), A7);
}
@compute @workgroup_size(64)
fn main(@builtin(local_invocation_index) l: u32, @builtin(workgroup_id) w: vec3<u32>) {
  let t = i32(l % ${T}u);
  let lane = t / 2;
  let p = t % 2;
  let g = i32(w.x) * ${G} + i32(l / ${T}u);
  let m = i32(w.y);
  let live = g < u.groups;
  let col = g / u.h;
  let gr = g - col * u.h;
  let base = (gr * u.width + col * u.rowTexels) / 4${o.bits === 8 ? "" : " + p"};
  let sx = col * u.kb;
  var acc = vec2<f32>(0.0);
  if (live) {
    for (var kb = lane; kb < u.kb; kb += ${o.lanes}) {
${A}
      let sa = dot(A0 + A1 + A2 + A3 + A4 + A5 + A6 + A7, vec4<f32>(1.0));
${o.zeros
    ? `      let sz = textureLoad(tS, vec2<i32>(sx + kb, gr), 0);
      let d4 = vec4<f32>(unpack2x16float(sz.x), unpack2x16float(sz.y));
      let z4 = vec4<f32>(unpack2x16float(sz.z), unpack2x16float(sz.w));
      let z = select(z4.xz, z4.yw, p == 1);`
    : "      let d4 = textureLoad(tS, vec2<i32>(sx + kb, gr), 0);"}
      let d = select(d4.xz, d4.yw, p == 1);
${o.bits === 8
    ? `      let r0 = base + kb * 8 + 2 * p;
      let r2 = r0 + 4;
      acc += d * vec2<f32>(rd8(bW[r0], bW[r0 + 1], A0, A1, A2, A3, A4, A5, A6, A7), rd8(bW[r2], bW[r2 + 1], A0, A1, A2, A3, A4, A5, A6, A7));`
    : `      acc += d * (vec2<f32>(rd(bW[base + kb * 4], A0, A1, A2, A3, A4, A5, A6, A7), rd(bW[base + kb * 4 + 2], A0, A1, A2, A3, A4, A5, A6, A7)) - ${o.zeros ? "z" : "8.0"} * sa);`}
    }
  }
${rms ? `  var s = vec4<f32>(0.0);
  for (var q = 0; q < u.parts; q++) { s += textureLoad(tR, vec2<i32>(u.rOff + q, m), 0); }
  acc *= inverseSqrt(dot(s, vec4<f32>(1.0)) / f32(u.texels * 4) + u.eps)${o.normed === "prescaled" ? " * 32.0" : ""};` : ""}
  pairs[l] = acc;
  workgroupBarrier();
  if (live && p == 0) {
    let b = pairs[l + 1u];
    textureStore(out0, vec2<i32>(g + i32(u.rect.x), m * ${o.lanes} + lane), vec4<f32>(acc.x, b.x, acc.y, b.y));
  }
}`
  return {
    code,
    inputs: [
      { name: "uA", kind: weights ? "f" : "u" },
      { name: "uS", kind: o.zeros ? "u" : "f" },
      ...(weights ? [{ name: "uNorm", kind: "f" as const }] : []),
      ...(rms ? [{ name: "uR", kind: "f" as const }] : [])
    ],
    buffers: [{ name: "uW" }],
    uniforms: [
      { name: "uKb", type: "int" },
      { name: "uH", type: "int" },
      { name: "uRowTexels", type: "int" },
      { name: "uWidth", type: "int" },
      { name: "uG", type: "int" },
      { name: "uAOff", type: "int" },
      { name: "uROff", type: "int" },
      { name: "uTexels", type: "int" },
      { name: "uParts", type: "int" },
      { name: "uEps", type: "float" }
    ],
    groups: (rect) => [Math.ceil(rect.width / G), rect.height]
  }
}

/**
 * kernels.ts's dequantBytes (no zero points: 8) from weights in a storage
 * buffer (see bufferMatvec): the f16 layout's texels, or with `h2` texels
 * 2x' and 2x' + 1 of each column as one h2 texel x' (into a texture, or
 * with `toBuffer` a storage buffer, row y at y · uStride). A thread takes a
 * row's block of 32 weights (x: the block, column after column; the
 * rectangle starts at x 0), so it fetches the block's words and scale once:
 * 4-bit, a word's low nibbles are one texel and its high nibbles the texel 4
 * on. An M3 Ultra dequantized LFM3's four prompt matrices in 156 µs this
 * way where 266 a thread an h2 texel, the same bits
 * (bench/gemm-exact.html). `bits` 8: bufferMatvec's 8-bit layout (a word a
 * texel).
 */
export function bufferDequant(o: { h2: boolean; toBuffer?: boolean; bits?: 4 | 8; zeros?: boolean }): NativeSpec {
  if (o.toBuffer && !o.h2) throw new Error("A dequant into a buffer writes h2 texels.")
  const bits = o.bits ?? 4
  // Texels a block: 8 of the f16 layout, 4 of h2.
  const per = o.h2 ? 4 : 8
  const J = [0, 1, 2, 3, 4, 5, 6, 7]
  const words = bits === 8 ? J : [0, 1, 2, 3]
  const store = (i: number, v: string) => o.toBuffer ? `  out0[at + ${i}] = ${v};` : `  textureStore(out0, vec2<i32>(x + ${i}, y), ${v});`
  const stores = (v: (j: number) => string) =>
    (o.h2 ? [0, 1, 2, 3].map((i) => store(i, `h2(${v(2 * i)}, ${v(2 * i + 1)})`)) : J.map((j) => store(j, v(j)))).join("\n")
  const code = `
struct Uni { rect: vec4<u32>, kb: i32, h: i32, rowTexels: i32, width: i32, kq: i32, kq0: i32, hd: i32, n: i32, row0: i32, stride: i32 }
@group(0) @binding(0) var<uniform> u: Uni;
@group(0) @binding(1) var tS: texture_2d<${o.zeros ? "u32" : "f32"}>;
@group(0) @binding(2) var<storage, read> bW: array<u32>;
${o.toBuffer ? "@group(0) @binding(3) var<storage, read_write> out0: array<vec4<u32>>;" : `@group(0) @binding(3) var out0: texture_storage_2d<${o.h2 ? "rgba32uint" : "rgba16float"}, write>;`}
${bits === 8
    ? "fn q(w: u32) -> vec4<f32> { return vec4<f32>(bitcast<vec4<i32>>(vec4<u32>(w) << vec4<u32>(24u, 16u, 8u, 0u)) >> vec4<u32>(24u)); }"
    : `fn q(w: u32, s: vec4<u32>, z: f32) -> vec4<f32> { return vec4<f32>((vec4<u32>(w) >> s) & vec4<u32>(15u)) - z; }`}
${o.h2 ? `fn h2(a0: vec4<f32>, b0: vec4<f32>) -> vec4<u32> {
  let a = clamp(a0, vec4<f32>(-65504.0), vec4<f32>(65504.0));
  let b = clamp(b0, vec4<f32>(-65504.0), vec4<f32>(65504.0));
  return vec4<u32>(pack2x16float(a.xy), pack2x16float(a.zw), pack2x16float(b.xy), pack2x16float(b.zw));
}` : ""}
@compute @workgroup_size(64, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  // Blocks a column, and this thread's: block kbl of column c.
  let kbs = u.kq >> 3u;
  let c = i32(id.x) / kbs;
  let kbl = i32(id.x) - c * kbs;
  let x = ${o.h2 ? "c * (u.kq >> 1u) + 4 * kbl" : "c * u.kq + 8 * kbl"};
  if (id.y >= u.rect.w || x >= i32(u.rect.z)) { return; }
  let y = i32(id.y + u.rect.y);
  ${o.toBuffer ? "let at = y * u.stride + x;" : ""}
  var row = c * u.hd + y;
  if (row >= u.n) {
${stores(() => "vec4<f32>(0.0)")}
    return;
  }
  row += u.row0;
  let g = row / 4;
  let r = row - g * 4;
  let col = g / u.h;
  let gr = g - col * u.h;
  let kb = (u.kq0 >> 3u) + kbl;
${o.zeros
    ? `  // (Packed: the block's 4 scales, then its 4 zero points, as halves; see GpuMatrix.packed.)
  let sz = textureLoad(tS, vec2<i32>(col * u.kb + kb, gr), 0);
  let d = vec4<f32>(unpack2x16float(sz.x), unpack2x16float(sz.y))[r];
  let z = vec4<f32>(unpack2x16float(sz.z), unpack2x16float(sz.w))[r];`
    : `  let d = textureLoad(tS, vec2<i32>(col * u.kb + kb, gr), 0)[r];${bits === 8 ? "" : "\n  let z = 8.0;"}`}
  let wb = gr * u.width + col * u.rowTexels + kb * ${4 * words.length} + r * ${words.length};
${words.map((j) => `  let w${j} = bW[wb + ${j}];`).join("\n")}
${J.map((j) => `  let v${j} = ${bits === 8 ? `q(w${j})` : `q(w${j & 3}, vec4<u32>(${j < 4 ? "0u, 8u, 16u, 24u" : "4u, 12u, 20u, 28u"}), z)`} * d;`).join("\n")}
${stores((j) => `v${j}`)}
}`
  return {
    code,
    inputs: [{ name: "uS", kind: o.zeros ? "u" : "f" }],
    buffers: [{ name: "uW" }],
    uniforms: [
      { name: "uKb", type: "int" },
      { name: "uH", type: "int" },
      { name: "uRowTexels", type: "int" },
      { name: "uWidth", type: "int" },
      { name: "uKq", type: "int" },
      { name: "uKq0", type: "int" },
      { name: "uHd", type: "int" },
      { name: "uN", type: "int" },
      { name: "uRow0", type: "int" },
      { name: "uStride", type: "int" }
    ],
    groups: (rect) => [Math.ceil(rect.width / per / 64), rect.height]
  }
}

/**
 * kernels.ts's matmulF16 (RGBA16F weights and activations: its `packed` 0)
 * as 8 rows (two row groups) × 4 tokens an invocation, workgroups of 8 row
 * pairs × 16 token quads, the quads varying fastest from workgroup to
 * workgroup, so that one block of rows meets every token block in turn.
 * Outputs as matmulF16's: row group g's texel at row y (slice · uSliceRows
 * + q) of o0 holds tokens 4q, 4q + 1 and of o1 4q + 2, 4q + 3, as h2. A
 * Pixel 10 Pro ran LFM3's gate|up at 581 GFLOP/s this way where 4 × 4 ran
 * 442, its down at 544 where 4 × 4 ran 452 (bench/webgpu-prompt-mm.html).
 */
export function promptMatmul(): NativeSpec {
  const code = `
struct Uni { rect: vec4<u32>, kq: i32, kqW: i32, kq0: i32, hd: i32, sliceRows: i32, kqSlice: i32, row0: i32 }
@group(0) @binding(0) var<uniform> u: Uni;
@group(0) @binding(1) var tW: texture_2d<f32>;
@group(0) @binding(2) var tA: texture_2d<f32>;
@group(0) @binding(3) var out0: texture_storage_2d<rgba32uint, write>;
@group(0) @binding(4) var out1: texture_storage_2d<rgba32uint, write>;
fn h2(a: vec4<f32>, b: vec4<f32>) -> vec4<u32> {
  let c = clamp(a, vec4<f32>(-65504.0), vec4<f32>(65504.0));
  let d = clamp(b, vec4<f32>(-65504.0), vec4<f32>(65504.0));
  return vec4<u32>(pack2x16float(c.xy), pack2x16float(c.zw), pack2x16float(d.xy), pack2x16float(d.zw));
}
@compute @workgroup_size(8, 16)
fn main(@builtin(local_invocation_id) l: vec3<u32>, @builtin(workgroup_id) w: vec3<u32>) {
  // Row y of the rectangle (absolute: the pacing may cut it into bands), row groups g and g + 1.
  let yr = w.x * 16u + l.y;
  let gp = 2u * (w.y * 8u + l.x);
  if (yr >= u.rect.w || gp >= u.rect.z) { return; }
  let y = i32(yr + u.rect.y);
  let g = i32(gp + u.rect.x);
  let two = gp + 1u < u.rect.z;
  let slice = y / u.sliceRows;
  let m0 = 4 * (y - slice * u.sliceRows);
  let row = 4 * g - u.row0;
  let c = row / u.hd;
  let wx = c * u.kqW - u.kq0;
  let wy = row - c * u.hd;
  let c2 = (row + 4) / u.hd;
  let wx2 = c2 * u.kqW - u.kq0;
  let wy2 = row + 4 - c2 * u.hd;
  let kq0 = slice * u.kqSlice;
  let kq1 = min(u.kq, kq0 + u.kqSlice);
  var a0 = vec4<f32>(0.0); var a1 = vec4<f32>(0.0); var a2 = vec4<f32>(0.0); var a3 = vec4<f32>(0.0);
  var b0 = vec4<f32>(0.0); var b1 = vec4<f32>(0.0); var b2 = vec4<f32>(0.0); var b3 = vec4<f32>(0.0);
  for (var kq = kq0; kq < kq1; kq++) {
    let w0 = textureLoad(tW, vec2<i32>(wx + kq, wy), 0);
    let w1 = textureLoad(tW, vec2<i32>(wx + kq, wy + 1), 0);
    let w2 = textureLoad(tW, vec2<i32>(wx + kq, wy + 2), 0);
    let w3 = textureLoad(tW, vec2<i32>(wx + kq, wy + 3), 0);
    let w4 = textureLoad(tW, vec2<i32>(wx2 + kq, wy2), 0);
    let w5 = textureLoad(tW, vec2<i32>(wx2 + kq, wy2 + 1), 0);
    let w6 = textureLoad(tW, vec2<i32>(wx2 + kq, wy2 + 2), 0);
    let w7 = textureLoad(tW, vec2<i32>(wx2 + kq, wy2 + 3), 0);
${[0, 1, 2, 3].map((t) => `    { let a = textureLoad(tA, vec2<i32>(kq, m0 + ${t}), 0);
      a${t} += vec4<f32>(dot(w0, a), dot(w1, a), dot(w2, a), dot(w3, a));
      b${t} += vec4<f32>(dot(w4, a), dot(w5, a), dot(w6, a), dot(w7, a)); }`).join("\n")}
  }
  textureStore(out0, vec2<i32>(g, y), h2(a0, a1));
  textureStore(out1, vec2<i32>(g, y), h2(a2, a3));
  if (two) {
    textureStore(out0, vec2<i32>(g + 1, y), h2(b0, b1));
    textureStore(out1, vec2<i32>(g + 1, y), h2(b2, b3));
  }
}`
  return {
    code,
    inputs: [{ name: "uW", kind: "f" }, { name: "uA", kind: "f" }],
    uniforms: [
      { name: "uKq", type: "int" },
      { name: "uKqW", type: "int" },
      { name: "uKq0", type: "int" },
      { name: "uHd", type: "int" },
      { name: "uSliceRows", type: "int" },
      { name: "uKqSlice", type: "int" },
      { name: "uRow0", type: "int" }
    ],
    // (Token rows across workgroups x, row-group pairs across y.)
    groups: (rect) => [Math.ceil(rect.height / 16), Math.ceil(rect.width / 16)]
  }
}

/**
 * Writing's Gated DeltaNet recurrence in one pass: kernels.ts's
 * linDeltaFused and linState (a head's q and k over their L2 norms, beta
 * and the decay from the b|a projection's split partial sums at texel uBAx,
 * then the output and S = decay·S + k̂ deltaᵀ), the state read once. A
 * workgroup takes 4 value texels of one head (y), keyHeadDim / rows
 * threads each (x), a thread `rows` key rows (4, 8 or 16): their partial sums of S·k and S·q meet in workgroup
 * memory, then each thread writes its rows of the new state from the
 * values it holds. linState's pass read the whole state again after
 * linDeltaFused had (Qwen3.5 2B: 1 MiB a layer each, 18 layers a token).
 * Heads with as many key as value dims, as many key heads as value heads.
 */
export function linRecurrence(o: { keyHeadDim: number; valueHeadDim: number; rows?: number }): NativeSpec {
  const DK4 = o.keyHeadDim / 4
  const DV4 = o.valueHeadDim / 4
  const R = o.rows ?? 8
  const R4 = R / 4
  const T = o.keyHeadDim / R
  if (DK4 !== DV4 || ![4, 8, 16].includes(R) || o.keyHeadDim % R || DV4 % 4 || 4 * T > 256) throw new Error(`No native recurrence for heads of ${o.keyHeadDim}.`)
  const scale = (1 / Math.sqrt(o.keyHeadDim)).toFixed(9)
  const J = Array.from({ length: R4 }, (_, j) => j)
  const S = Array.from({ length: R }, (_, j) => j)
  const kc = (r: number) => `k${r >> 2}.${"xyzw"[r & 3]}`
  const qc = (r: number) => `q${r >> 2}.${"xyzw"[r & 3]}`
  const code = `
struct Uni { rect: vec4<u32>, k4: i32, h4: i32, bax: i32, splitIn: i32 }
@group(0) @binding(0) var<uniform> u: Uni;
@group(0) @binding(1) var tS: texture_2d<f32>;
@group(0) @binding(2) var tQKV: texture_2d<f32>;
@group(0) @binding(3) var tBA: texture_2d<f32>;
@group(0) @binding(4) var tA: texture_2d<f32>;
@group(0) @binding(5) var tDt: texture_2d<f32>;
@group(0) @binding(6) var out0: texture_storage_2d<rgba32float, write>;
@group(0) @binding(7) var out1: texture_storage_2d<rgba32float, write>;
var<workgroup> nq: array<f32, ${4 * T}>;
var<workgroup> nk: array<f32, ${4 * T}>;
var<workgroup> rkv: array<vec4<f32>, ${4 * T}>;
var<workgroup> rqs: array<vec4<f32>, ${4 * T}>;
var<workgroup> rqk: array<f32, ${4 * T}>;
fn pf(x: i32) -> vec4<f32> {
  var a = vec4<f32>(0.0);
  for (var k = 0; k < 16; k++) { if (k >= u.splitIn) { break; } a += textureLoad(tBA, vec2<i32>(x, k), 0); }
  return a;
}
@compute @workgroup_size(${T}, 4)
fn main(@builtin(local_invocation_id) l: vec3<u32>, @builtin(workgroup_id) w: vec3<u32>) {
  let c = i32(l.x);
  let slot = i32(l.y) * ${T} + c;
  let x = i32(w.x) * 4 + i32(l.y);
  let h = x / ${DV4};
  let kb = h * ${DK4};
  // The head's q and k norms, then this thread's ${R} keys.
${J.map((j) => `  let qr${j} = textureLoad(tQKV, vec2<i32>(kb + ${R4} * c + ${j}, 0), 0);
  let kr${j} = textureLoad(tQKV, vec2<i32>(u.k4 + kb + ${R4} * c + ${j}, 0), 0);`).join("\n")}
  nq[slot] = ${J.map((j) => `dot(qr${j}, qr${j})`).join(" + ")};
  nk[slot] = ${J.map((j) => `dot(kr${j}, kr${j})`).join(" + ")};
  workgroupBarrier();
  var sq = 0.0;
  var sk = 0.0;
  for (var i = 0; i < ${T}; i++) { sq += nq[i32(l.y) * ${T} + i]; sk += nk[i32(l.y) * ${T} + i]; }
  let invq = inverseSqrt(sq + 1e-6);
  let invk = inverseSqrt(sk + 1e-6);
${J.map((j) => `  let q${j} = qr${j} * invq * ${scale};
  let k${j} = kr${j} * invk;`).join("\n")}
  let r0 = ${R} * c;
${S.map((j) => `  let s${j} = textureLoad(tS, vec2<i32>(x, r0 + ${j}), 0);`).join("\n")}
  rkv[slot] = ${S.map((j) => `s${j} * ${kc(j)}`).join(" + ")};
  rqs[slot] = ${S.map((j) => `s${j} * ${qc(j)}`).join(" + ")};
  rqk[slot] = ${J.map((j) => `dot(q${j}, k${j})`).join(" + ")};
  workgroupBarrier();
  var kv = vec4<f32>(0.0);
  var qs = vec4<f32>(0.0);
  var qk = 0.0;
  for (var i = 0; i < ${T}; i++) { let j = i32(l.y) * ${T} + i; kv += rkv[j]; qs += rqs[j]; qk += rqk[j]; }
  let g = h >> 2u;
  let beta = (1.0 / (1.0 + exp(-pf(u.bax + g))))[h & 3];
  let t = pf(u.bax + u.h4 + g) + textureLoad(tDt, vec2<i32>(g, 0), 0);
  let softplus = max(t, vec4<f32>(0.0)) + log(1.0 + exp(-abs(t)));
  let decay = exp(textureLoad(tA, vec2<i32>(g, 0), 0) * softplus)[h & 3];
  let v = textureLoad(tQKV, vec2<i32>(2 * u.k4 + x, 0), 0);
  let d = beta * (v - decay * kv);
  if (c == 0) { textureStore(out0, vec2<i32>(x, 0), decay * qs + qk * d); }
${S.map((j) => `  textureStore(out1, vec2<i32>(x, r0 + ${j}), decay * s${j} + ${kc(j)} * d);`).join("\n")}
}`
  return {
    code,
    inputs: [
      { name: "uS", kind: "f" },
      { name: "uQKV", kind: "f" },
      { name: "uBA", kind: "f" },
      { name: "uA", kind: "f" },
      { name: "uDt", kind: "f" }
    ],
    uniforms: [
      { name: "uK4", type: "int" },
      { name: "uH4", type: "int" },
      { name: "uBAx", type: "int" },
      { name: "uSplitIn", type: "int" }
    ],
    groups: (rect) => [rect.width / 4, 1]
  }
}

/**
 * A prompt batch's chunked recurrence (kernels.ts's linChunkOut and
 * linChunkState, the same sums in the same order), chunk after chunk in
 * one dispatch where those are two dependent passes a chunk: rows
 * [uC0, uC1) of the batch, whole chunks (of `chunk` rows after a first one
 * of uFirst, as chunkStart has them). A workgroup takes a value texel x and
 * holds its column of the state (a thread a key row k) in workgroup memory
 * from chunk to chunk:
 * - a thread a token of the chunk: delta D = U - W S0 into workgroup
 *   memory, and the output O = Oi + Q' S0 into the first output;
 * - a thread a key row: S = exp(G_last) S0 + Σ_s exp(G_last - G_s) k̂_s D_s;
 * and the last chunk's state goes to the second output (texel (x, k)).
 */
export function linChunkLoop(o: { keyHeadDim: number; valueHeadDim: number; chunk: number }): NativeSpec {
  const dk = o.keyHeadDim
  const DK4 = dk / 4
  const DV4 = o.valueHeadDim / 4
  if (dk > 256 || o.chunk > dk || dk % 4 || o.valueHeadDim % 4) throw new Error(`No chunk loop for ${dk} key dims and chunks of ${o.chunk}.`)
  const code = `
struct Uni { rect: vec4<u32>, k4: i32, c0: i32, c1: i32, first: i32 }
@group(0) @binding(0) var<uniform> u: Uni;
@group(0) @binding(1) var tS: texture_2d<f32>;
@group(0) @binding(2) var tU: texture_2d<f32>;
@group(0) @binding(3) var tW: texture_2d<f32>;
@group(0) @binding(4) var tOi: texture_2d<f32>;
@group(0) @binding(5) var tQp: texture_2d<f32>;
@group(0) @binding(6) var tQK: texture_2d<f32>;
@group(0) @binding(7) var tCum: texture_2d<f32>;
@group(0) @binding(8) var outO: texture_storage_2d<rgba32float, write>;
@group(0) @binding(9) var outS: texture_storage_2d<rgba32float, write>;
var<workgroup> S: array<vec4<f32>, ${dk}>;
var<workgroup> D: array<vec4<f32>, ${o.chunk}>;
fn G(h: i32, r: i32) -> f32 { return textureLoad(tCum, vec2<i32>(h >> 2u, r), 0)[h & 3]; }
@compute @workgroup_size(${dk})
fn main(@builtin(local_invocation_index) l: u32, @builtin(workgroup_id) w: vec3<u32>) {
  let x = i32(w.x + u.rect.x);
  let k = i32(l);
  let h = x / ${DV4};
  let kb = h * ${DK4};
  let kc = h * ${dk} + k;
  S[k] = textureLoad(tS, vec2<i32>(x, k), 0);
  workgroupBarrier();
  var c0 = u.c0;
  while (c0 < u.c1) {
    let c1 = min(c0 + select(${o.chunk}, u.first, c0 == 0), u.c1);
    if (k < c1 - c0) {
      let t = c0 + k;
      var ws = vec4<f32>(0.0);
      var qs = vec4<f32>(0.0);
      for (var i = 0; i < ${DK4}; i++) {
        let wv = textureLoad(tW, vec2<i32>(kb + i, t), 0);
        let q = textureLoad(tQp, vec2<i32>(kb + i, t), 0);
        let s0 = S[4 * i];
        let s1 = S[4 * i + 1];
        let s2 = S[4 * i + 2];
        let s3 = S[4 * i + 3];
        ws += s0 * wv.x + s1 * wv.y + s2 * wv.z + s3 * wv.w;
        qs += s0 * q.x + s1 * q.y + s2 * q.z + s3 * q.w;
      }
      D[k] = textureLoad(tU, vec2<i32>(x, t), 0) - ws;
      textureStore(outO, vec2<i32>(x, t), textureLoad(tOi, vec2<i32>(x, t), 0) + qs);
    }
    workgroupBarrier();
    let last = G(h, c1 - 1);
    var acc = exp(last) * S[k];
    for (var s = c0; s < c1; s++) {
      let kk = textureLoad(tQK, vec2<i32>(u.k4 + (kc >> 2u), s), 0)[kc & 3];
      acc += exp(last - G(h, s)) * kk * D[s - c0];
    }
    // (Its own row: the others read theirs alone here, and all of them only after the barrier.)
    S[k] = acc;
    workgroupBarrier();
    c0 = c1;
  }
  textureStore(outS, vec2<i32>(x, k), S[k]);
}
`
  return {
    code,
    inputs: [
      { name: "uS", kind: "f" },
      { name: "uU", kind: "f" },
      { name: "uW", kind: "f" },
      { name: "uOi", kind: "f" },
      { name: "uQp", kind: "f" },
      { name: "uQK", kind: "f" },
      { name: "uCum", kind: "f" }
    ],
    uniforms: [
      { name: "uK4", type: "int" },
      { name: "uC0", type: "int" },
      { name: "uC1", type: "int" },
      { name: "uFirst", type: "int" }
    ],
    // A workgroup a value texel of the rectangle's width.
    groups: (rect) => [rect.width, 1]
  }
}

/**
 * kernels.ts's 8-bit pair matmul (`matmul` with `pair`) from weights in a
 * storage buffer (see bufferMatvec's 8-bit layout): invocation (x, y) takes
 * row group x for tokens 2y and 2y + 1 (rows of h2 activations, 8 values a
 * texel) and writes their sums to the pair tensor's texel (x, y) as h2, so
 * prompts read the same buffer the token matvecs do and the texture can go.
 */
export function bufferPairMatmul(): NativeSpec {
  const code = `
struct Uni { rect: vec4<u32>, kb: i32, h: i32, rowTexels: i32, width: i32 }
@group(0) @binding(0) var<uniform> u: Uni;
@group(0) @binding(1) var tA: texture_2d<u32>;
@group(0) @binding(2) var tS: texture_2d<f32>;
@group(0) @binding(3) var<storage, read> bW: array<vec4<u32>>;
@group(0) @binding(4) var out0: texture_storage_2d<rgba32uint, write>;
fn lo(w: vec4<u32>) -> vec4<f32> { return vec4<f32>(unpack2x16float(w.x), unpack2x16float(w.y)); }
fn hi(w: vec4<u32>) -> vec4<f32> { return vec4<f32>(unpack2x16float(w.z), unpack2x16float(w.w)); }
fn s8(w: u32) -> vec4<f32> { return vec4<f32>(bitcast<vec4<i32>>(vec4<u32>(w) << vec4<u32>(24u, 16u, 8u, 0u)) >> vec4<u32>(24u)); }
fn row(q0: vec4<u32>, q1: vec4<u32>, A0: vec4<f32>, A1: vec4<f32>, A2: vec4<f32>, A3: vec4<f32>, A4: vec4<f32>, A5: vec4<f32>, A6: vec4<f32>, A7: vec4<f32>) -> f32 {
  return dot(s8(q0.x), A0) + dot(s8(q0.y), A1) + dot(s8(q0.z), A2) + dot(s8(q0.w), A3) + dot(s8(q1.x), A4) + dot(s8(q1.y), A5) + dot(s8(q1.z), A6) + dot(s8(q1.w), A7);
}
fn h2(a: vec4<f32>, b: vec4<f32>) -> vec4<u32> {
  let c = clamp(a, vec4<f32>(-65504.0), vec4<f32>(65504.0));
  let d = clamp(b, vec4<f32>(-65504.0), vec4<f32>(65504.0));
  return vec4<u32>(pack2x16float(c.xy), pack2x16float(c.zw), pack2x16float(d.xy), pack2x16float(d.zw));
}
@compute @workgroup_size(64, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  if (id.x >= u.rect.z || id.y >= u.rect.w) { return; }
  let g = i32(id.x + u.rect.x);
  let y = i32(id.y + u.rect.y);
  let col = g / u.h;
  let gr = g - col * u.h;
  let base = (gr * u.width + col * u.rowTexels) / 4;
  let sx = col * u.kb;
  var acc = array<vec4<f32>, 2>(vec4<f32>(0.0), vec4<f32>(0.0));
  for (var kb = 0; kb < u.kb; kb++) {
    let d = textureLoad(tS, vec2<i32>(sx + kb, gr), 0);
${[0, 1, 2, 3, 4, 5, 6, 7].map((i) => `    let q${i} = bW[base + kb * 8 + ${i}];`).join("\n")}
    for (var tt = 0; tt < 2; tt++) {
      let m = 2 * y + tt;
${[0, 1, 2, 3].map((j) => `      let a${j} = textureLoad(tA, vec2<i32>(kb * 4 + ${j}, m), 0);`).join("\n")}
      let A0 = lo(a0); let A1 = hi(a0); let A2 = lo(a1); let A3 = hi(a1); let A4 = lo(a2); let A5 = hi(a2); let A6 = lo(a3); let A7 = hi(a3);
      acc[tt] += d * vec4<f32>(row(q0, q1, A0, A1, A2, A3, A4, A5, A6, A7), row(q2, q3, A0, A1, A2, A3, A4, A5, A6, A7), row(q4, q5, A0, A1, A2, A3, A4, A5, A6, A7), row(q6, q7, A0, A1, A2, A3, A4, A5, A6, A7));
    }
  }
  textureStore(out0, vec2<i32>(g, y), h2(acc[0], acc[1]));
}`
  return {
    code,
    inputs: [{ name: "uA", kind: "u" }, { name: "uS", kind: "f" }],
    buffers: [{ name: "uW" }],
    uniforms: [
      { name: "uKb", type: "int" },
      { name: "uH", type: "int" },
      { name: "uRowTexels", type: "int" },
      { name: "uWidth", type: "int" }
    ],
    groups: (rect) => [Math.ceil(rect.width / 64), rect.height]
  }
}

/**
 * kernels.ts's matmulF16 with both inputs in h2 texels (its `packed` 3),
 * the dequantized weights from a storage buffer (bufferDequant's
 * `toBuffer`: row r at r · uStride h2 texels) and `trips` steps of two K
 * texels a slice (uKqSlice = 2 · trips: a pipeline a slice length; 0: as
 * many as the slice has, for a last slice cut short). An M3
 * Ultra ran LFM3's prompt matmuls 1.14x as fast this way as the GLSL
 * pass, with the same output (bench/gemm-exact.html).
 */
export function promptMatmulBuffer(o: { trips: number; rowGroups?: number }): NativeSpec {
  // rowGroups a thread (adjacent row groups: 4 rows each), workgroups of
  // wx rect rows by wy threads' row groups.
  const tm = o.rowGroups ?? 1
  const [wx, wy] = tm > 1 ? [8, 8] : [16, 4]
  const I = Array.from({ length: tm }, (_, i) => i)
  const T = [0, 1, 2, 3]
  const half = (h: "lo" | "hi") => `    {
${I.map((i) => `      let w${i}_0 = ${h}(W${i}_0); let w${i}_1 = ${h}(W${i}_1); let w${i}_2 = ${h}(W${i}_2); let w${i}_3 = ${h}(W${i}_3);`).join("\n")}
${T.map((t) => `      { let a = ${h}(A${t});
${I.map((i) => `        acc${t}_${i} += vec4<f32>(dot(w${i}_0, a), dot(w${i}_1, a), dot(w${i}_2, a), dot(w${i}_3, a));`).join("\n")}
      }`).join("\n")}
    }`
  const code = `
struct Uni { rect: vec4<u32>, kq: i32, kqW: i32, kq0: i32, hd: i32, sliceRows: i32, kqSlice: i32, row0: i32, stride: i32 }
@group(0) @binding(0) var<uniform> u: Uni;
@group(0) @binding(1) var tA: texture_2d<u32>;
@group(0) @binding(2) var<storage, read> bW: array<vec4<u32>>;
@group(0) @binding(3) var out0: texture_storage_2d<rgba32uint, write>;
@group(0) @binding(4) var out1: texture_storage_2d<rgba32uint, write>;
fn lo(v: vec4<u32>) -> vec4<f32> { return vec4<f32>(unpack2x16float(v.x), unpack2x16float(v.y)); }
fn hi(v: vec4<u32>) -> vec4<f32> { return vec4<f32>(unpack2x16float(v.z), unpack2x16float(v.w)); }
fn packH2(a0: vec4<f32>, b0: vec4<f32>) -> vec4<u32> {
  let a = clamp(a0, vec4<f32>(-65504.0), vec4<f32>(65504.0));
  let b = clamp(b0, vec4<f32>(-65504.0), vec4<f32>(65504.0));
  return vec4<u32>(pack2x16float(a.xy), pack2x16float(a.zw), pack2x16float(b.xy), pack2x16float(b.zw));
}
@compute @workgroup_size(${wx}, ${wy})
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  // x: the rectangle's rows (absolute: the pacing may cut it into bands), y: its row groups, ${tm} a thread.
  if (id.x >= u.rect.w || id.y * ${tm}u >= u.rect.z) { return; }
  let y = i32(id.x + u.rect.y);
  let g0 = i32(id.y * ${tm}u + u.rect.x);
  let slice = y / u.sliceRows;
  let m0 = 4 * (y - slice * u.sliceRows);
  let kq0 = slice * u.kqSlice;
${I.map((i) => `  let ok${i} = ${i === 0 ? "true" : `g0 + ${i} < i32(u.rect.x + u.rect.z)`};
  let row${i} = 4 * ${i === 0 ? "g0" : `select(g0, g0 + ${i}, ok${i})`} - u.row0;
  let c${i} = row${i} / u.hd;
  let wb${i} = (row${i} - c${i} * u.hd) * u.stride + ((c${i} * u.kqW - u.kq0 + kq0) >> 1u);
${T.map((t) => `  var acc${t}_${i} = vec4<f32>(0.0);`).join("\n")}`).join("\n")}
  ${o.trips > 0 ? "" : "let trips = (min(u.kq, kq0 + u.kqSlice) - kq0) >> 1u;"}
  for (var it = 0; it < ${o.trips > 0 ? o.trips : "trips"}; it++) {
    let ta = (kq0 >> 1u) + it;
${I.map((i) => `    let W${i}_0 = bW[wb${i} + it]; let W${i}_1 = bW[wb${i} + it + u.stride]; let W${i}_2 = bW[wb${i} + it + 2 * u.stride]; let W${i}_3 = bW[wb${i} + it + 3 * u.stride];`).join("\n")}
${T.map((t) => `    let A${t} = textureLoad(tA, vec2<i32>(ta, m0 + ${t}), 0);`).join("\n")}
${half("lo")}
${half("hi")}
  }
${I.map((i) => `  if (ok${i}) {
    textureStore(out0, vec2<i32>(g0 + ${i}, y), packH2(acc0_${i}, acc1_${i}));
    textureStore(out1, vec2<i32>(g0 + ${i}, y), packH2(acc2_${i}, acc3_${i}));
  }`).join("\n")}
}`
  return {
    code,
    inputs: [{ name: "uA", kind: "u" }],
    buffers: [{ name: "uW" }],
    uniforms: [
      { name: "uKq", type: "int" },
      { name: "uKqW", type: "int" },
      { name: "uKq0", type: "int" },
      { name: "uHd", type: "int" },
      { name: "uSliceRows", type: "int" },
      { name: "uKqSlice", type: "int" },
      { name: "uRow0", type: "int" },
      { name: "uStride", type: "int" }
    ],
    groups: (rect) => [Math.ceil(rect.height / wx), Math.ceil(rect.width / (wy * tm))]
  }
}
