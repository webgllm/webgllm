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
 * A token's add, in one dispatch, a workgroup per row (the passes it
 * replaces: Engine.addNormed's rms of y, then the add, its sums of squares
 * and its scaled copy, drawn as regions). For row m:
 * - y = the split matmul's partial sums over uSplit slices (rows m·uSplit + k);
 * - with `normed` (LFM3's sandwich norm), y · 1/rms(y) · uNorm;
 * - out = uH + that, into columns [0, texels) of the first output;
 * - its sum of squares into column texels (the next norm sums `parts`
 *   columns from there: the rest are zeros);
 * - with `next`, h2(out · uNext / PRESCALE) into the second output from
 *   column uBase (texel j: features 8j..8j+7).
 * Each thread takes texel pairs; the two sums go through workgroup memory.
 */
export function fusedAdd(o: { texels: number; normed: boolean; next: boolean }): NativeSpec {
  const pairs = o.texels / 2
  const threads = Math.min(256, pairs)
  const per = pairs / threads
  if (!Number.isInteger(per) || o.texels % 2) throw new Error(`No fused add for ${o.texels} texels.`)
  const inputs: NativeSpec["inputs"] = [
    { name: "uH", kind: "f" },
    { name: "uY", kind: "f" },
    ...(o.normed ? [{ name: "uNorm", kind: "f" as const }] : []),
    ...(o.next ? [{ name: "uNext", kind: "f" as const }] : [])
  ]
  const binding = (name: string) => inputs.findIndex((i) => i.name === name) + 1
  const outBase = inputs.length + 1
  const p = Array.from({ length: per }, (_, i) => i)
  const code = `
struct U { rect: vec4<u32>, split: i32, parts: i32, eps: f32, base: i32 }
@group(0) @binding(0) var<uniform> u: U;
@group(0) @binding(${binding("uH")}) var tH: texture_2d<f32>;
@group(0) @binding(${binding("uY")}) var tY: texture_2d<f32>;
${o.normed ? `@group(0) @binding(${binding("uNorm")}) var tNorm: texture_2d<f32>;` : ""}
${o.next ? `@group(0) @binding(${binding("uNext")}) var tNext: texture_2d<f32>;` : ""}
@group(0) @binding(${outBase}) var out0: texture_storage_2d<rgba32float, write>;
${o.next ? `@group(0) @binding(${outBase + 1}) var out1: texture_storage_2d<rgba32uint, write>;` : ""}
var<workgroup> red: array<f32, ${threads}>;
fn total(v: f32, l: u32) -> f32 {
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
fn y(x: i32, m: i32) -> vec4<f32> {
  var a = vec4<f32>(0.0);
  for (var k = 0; k < u.split; k++) { a += textureLoad(tY, vec2<i32>(x, m * u.split + k), 0); }
  return a;
}
fn h2(a: vec4<f32>, b: vec4<f32>) -> vec4<u32> {
  let x = clamp(a, vec4<f32>(-65504.0), vec4<f32>(65504.0));
  let z = clamp(b, vec4<f32>(-65504.0), vec4<f32>(65504.0));
  return vec4<u32>(pack2x16float(x.xy), pack2x16float(x.zw), pack2x16float(z.xy), pack2x16float(z.zw));
}
@compute @workgroup_size(${threads})
fn main(@builtin(local_invocation_index) l: u32, @builtin(workgroup_id) w: vec3<u32>) {
  let m = i32(u.rect.y + w.x);
  let j0 = i32(l) * ${per};
${p.map((i) => `  let ya${i} = y(2 * (j0 + ${i}), m);
  let yb${i} = y(2 * (j0 + ${i}) + 1, m);`).join("\n")}
${o.normed
    ? `  let rs = inverseSqrt(total(${p.map((i) => `dot(ya${i}, ya${i}) + dot(yb${i}, yb${i})`).join(" + ")}, l) / ${o.texels * 4}.0 + u.eps);
${p.map((i) => `  let va${i} = textureLoad(tH, vec2<i32>(2 * (j0 + ${i}), m), 0) + ya${i} * rs * textureLoad(tNorm, vec2<i32>(2 * (j0 + ${i}), 0), 0);
  let vb${i} = textureLoad(tH, vec2<i32>(2 * (j0 + ${i}) + 1, m), 0) + yb${i} * rs * textureLoad(tNorm, vec2<i32>(2 * (j0 + ${i}) + 1, 0), 0);`).join("\n")}`
    : p.map((i) => `  let va${i} = textureLoad(tH, vec2<i32>(2 * (j0 + ${i}), m), 0) + ya${i};
  let vb${i} = textureLoad(tH, vec2<i32>(2 * (j0 + ${i}) + 1, m), 0) + yb${i};`).join("\n")}
${p.map((i) => `  textureStore(out0, vec2<i32>(2 * (j0 + ${i}), m), va${i});
  textureStore(out0, vec2<i32>(2 * (j0 + ${i}) + 1, m), vb${i});`).join("\n")}
  let sum = total(${p.map((i) => `dot(va${i}, va${i}) + dot(vb${i}, vb${i})`).join(" + ")}, l);
  if (i32(l) < u.parts) {
    textureStore(out0, vec2<i32>(${o.texels} + i32(l), m), select(vec4<f32>(0.0), vec4<f32>(sum, 0.0, 0.0, 0.0), l == 0u));
  }
${o.next ? p.map((i) => `  textureStore(out1, vec2<i32>(u.base + j0 + ${i}, m), h2(
    va${i} * textureLoad(tNext, vec2<i32>(2 * (j0 + ${i}), 0), 0) * ${1 / PRESCALE},
    vb${i} * textureLoad(tNext, vec2<i32>(2 * (j0 + ${i}) + 1, 0), 0) * ${1 / PRESCALE}));`).join("\n") : ""}
}
`
  return {
    code,
    inputs,
    uniforms: [
      { name: "uSplitIn", type: "int" },
      { name: "uParts", type: "int" },
      { name: "uEps", type: "float" },
      { name: "uBase", type: "int" }
    ],
    // A workgroup per row of the rectangle.
    groups: (rect) => [rect.height, 1]
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
