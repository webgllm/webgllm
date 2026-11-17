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
