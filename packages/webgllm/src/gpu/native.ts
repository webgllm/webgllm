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
export function wordHead(o: { split: boolean; prescaled: boolean; kbSlice: number }): NativeSpec {
  const inputs: NativeSpec["inputs"] = [
    { name: "uA", kind: "u" },
    { name: "uW", kind: "u" },
    { name: "uS", kind: "f" },
    ...(o.prescaled ? [{ name: "uR", kind: "f" as const }] : [])
  ]
  const C = "xyzw"
  const texels = 32 * o.kbSlice
  const rowBlock = (r: number) => `      {
        let t = wx + (sb * 4 + ${r}) * 13;
        let QA = textureLoad(tW, vec2<i32>(t + qa, gr), 0);
        let QB = textureLoad(tW, vec2<i32>(t + qa + 2, gr), 0);
        let QH = textureLoad(tW, vec2<i32>(t + qh, gr), 0);
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
      let S0 = textureLoad(tW, vec2<i32>(wx + (sb * 4) * 13 + 12, gr), 0);
      let S1 = textureLoad(tW, vec2<i32>(wx + (sb * 4 + 1) * 13 + 12, gr), 0);
      let S2 = textureLoad(tW, vec2<i32>(wx + (sb * 4 + 2) * 13 + 12, gr), 0);
      let S3 = textureLoad(tW, vec2<i32>(wx + (sb * 4 + 3) * 13 + 12, gr), 0);
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
