/**
 * Matmul layouts compared on synthetic data, no model load: 4-bit weights
 * as RGBA4 (the texture unit splits nibbles) or RGBA8UI (split in the
 * shader), activations as RGBA16F (the texture unit converts halves) or
 * RGBA32UI h2 (unpacked in the shader), and tokens per fragment. Each
 * fragment computes 4 weight rows; weights stream past, as `matmul` does.
 */
import { Gpu } from "../src/gpu/gl.js"
import type { Texture } from "../src/gpu/gl.js"

const log = (text: string) => postMessage({ type: "log", text })

const HEADER = `#version 300 es
precision highp float;
precision highp int;
precision highp sampler2D;
precision highp usampler2D;
vec4 h2lo(uvec4 u) { return vec4(unpackHalf2x16(u.x), unpackHalf2x16(u.y)); }
vec4 h2hi(uvec4 u) { return vec4(unpackHalf2x16(u.z), unpackHalf2x16(u.w)); }
uvec4 packH2(vec4 a, vec4 b) {
  return uvec4(packHalf2x16(a.xy), packHalf2x16(a.zw), packHalf2x16(b.xy), packHalf2x16(b.zw));
}
`

type Variant = {
  weights: "rgba4" | "bytes" | "words" | "unorm8"
  acts: "f16" | "h2"
  tokens: 1 | 2 | 4 | 8
  /** Weights, activations and block sums as mediump (half precision on Metal?). */
  half?: boolean
  /** Timing only: skip the activation sums (wrong results). */
  noSum?: boolean
  /** Fetch each h2 activation texel once for both its halves. */
  paired?: boolean
  /** Scale folded into each step (no per-block sums held per token). */
  fold?: boolean
}

function kernel(v: Variant) {
  const T = v.tokens
  const L: Array<string> = []
  const line = (s: string) => L.push(`    ${s}`)
  line(`vec4 d = texelFetch(uS, ivec2(kb, g), 0);`)
  for (let t = 0; t < T; t++) line(`vec4 b${t} = vec4(0.0); float s${t} = 0.0;`)
  // Activation k = 4j..4j+3 of token t as a vec4.
  const act = (j: string, t: number) =>
    v.acts === "f16"
      ? `texelFetch(uA, ivec2(kb * 8 + ${j}, m0 + ${t}), 0)`
      : `h2pick(texelFetch(uA, ivec2(kb * 4 + (${j}) / 2, m0 + ${t}), 0), (${j}) & 1)`
  if (v.weights === "rgba4" && v.fold) {
    // Per block: the scale times 15 up front; each step adds straight into
    // the accumulators, so no per-token block sums are held.
    {
      line(`vec4 d15 = 15.0 * texelFetch(uS, ivec2(kb, g), 0);`)
      for (let t = 0; t < T; t++) line(`float s${t} = 0.0;`)
      for (let j = 0; j < 8; j++) {
        line(`{`)
        for (const r of [0, 1, 2, 3])
          line(`  vec4 w${r} = texelFetch(uW, ivec2(kb * 32 + ${r * 8 + j}, g), 0);`)
        for (let t = 0; t < T; t++) {
          line(`  { vec4 a = ${j & 1 ? "h2hi" : "h2lo"}(texelFetch(uA, ivec2(kb * 4 + ${j >> 1}, m0 + ${t}), 0));`)
          line(`    s${t} += dot(a, vec4(1.0));`)
          line(`    acc${t} += d15 * vec4(dot(w0, a), dot(w1, a), dot(w2, a), dot(w3, a)); }`)
        }
        line(`}`)
      }
      for (let t = 0; t < T; t++) line(`acc${t} -= (8.0 / 15.0) * d15 * s${t};`)
    }
    // Drop the shared preamble (d and the per-token block sums).
    L.splice(0, 1 + T)
  } else if (v.weights === "rgba4" && v.paired) {
    // Each h2 texel jj fetched once: features 8jj..8jj+3 (lo), +4..+7 (hi).
    for (let jj = 0; jj < 4; jj++) {
      line(`{`)
      for (let t = 0; t < T; t++)
        line(`  uvec4 u${t} = texelFetch(uA, ivec2(kb * 4 + ${jj}, m0 + ${t}), 0);`)
      for (const half of [0, 1]) {
        const j = 2 * jj + half
        line(`  {`)
        for (const r of [0, 1, 2, 3])
          line(`    vec4 w${r} = texelFetch(uW, ivec2(kb * 32 + ${r * 8 + j}, g), 0);`)
        for (let t = 0; t < T; t++) {
          line(`    { vec4 a = ${half ? "h2hi" : "h2lo"}(u${t});`)
          line(`      s${t} += dot(a, vec4(1.0));`)
          line(`      b${t} += vec4(dot(w0, a), dot(w1, a), dot(w2, a), dot(w3, a)); }`)
        }
        line(`  }`)
      }
      line(`}`)
    }
    for (let t = 0; t < T; t++) line(`acc${t} += d * (15.0 * b${t} - 8.0 * s${t});`)
  } else if (v.weights === "rgba4") {
    // Texel (kb * 32 + r * 8 + j, g): row r, weights 4j..4j+3 (q / 15).
    for (let j = 0; j < 8; j++) {
      line(`{`)
      for (const r of [0, 1, 2, 3])
        line(`  vec4 w${r} = texelFetch(uW, ivec2(kb * 32 + ${r * 8 + j}, g), 0);`)
      for (let t = 0; t < T; t++) {
        line(`  { vec4 a = ${act(String(j), t)};`)
        line(`    s${t} += dot(a, vec4(1.0));`)
        line(`    b${t} += vec4(dot(w0, a), dot(w1, a), dot(w2, a), dot(w3, a)); }`)
      }
      line(`}`)
    }
    for (let t = 0; t < T; t++) line(`acc${t} += d * (15.0 * b${t} - 8.0 * s${t});`)
  } else if (v.weights === "unorm8") {
    // RGBA8 texel (kb * 16 + r * 4 + w, g), read as floats: bytes of weights
    // 4w.. (low nibbles) and 16+4w.. (high), split with float math.
    for (let w = 0; w < 4; w++) {
      line(`{`)
      for (const r of [0, 1, 2, 3]) {
        line(`  vec4 B${r} = floor(texelFetch(uW, ivec2(kb * 16 + ${r * 4 + w}, g), 0) * 255.0 + 0.5);`)
        line(`  vec4 H${r} = floor(B${r} * 0.0625);`)
        line(`  vec4 L${r} = B${r} - 16.0 * H${r};`)
      }
      for (let t = 0; t < T; t++) {
        line(`  { uvec4 u = texelFetch(uA, ivec2(kb * 4 + ${w >> 1}, m0 + ${t}), 0);`)
        line(`    uvec4 u2 = texelFetch(uA, ivec2(kb * 4 + ${(w >> 1) + 2}, m0 + ${t}), 0);`)
        line(`    vec4 x = ${w & 1 ? "h2hi" : "h2lo"}(u); vec4 y = ${w & 1 ? "h2hi" : "h2lo"}(u2);`)
        line(`    s${t} += dot(x + y, vec4(1.0));`)
        line(
          `    b${t} += vec4(dot(L0, x) + dot(H0, y), dot(L1, x) + dot(H1, y), dot(L2, x) + dot(H2, y), dot(L3, x) + dot(H3, y)); }`
        )
      }
      line(`}`)
    }
    for (let t = 0; t < T; t++) line(`acc${t} += d * (b${t} - 8.0 * s${t});`)
  } else if (v.weights === "words") {
    // RGBA32UI texel (kb * 4 + r, g): row r's whole block, uint k holding
    // weights 8k..8k+7 (nibble i at bits 4i), split with integer shifts.
    for (const r of [0, 1, 2, 3])
      line(`uvec4 W${r} = texelFetch(uW, ivec2(kb * 4 + ${r}, g), 0);`)
    for (let k = 0; k < 4; k++) {
      line(`{`)
      for (const r of [0, 1, 2, 3]) {
        line(`  vec4 L${r} = vec4((uvec4(W${r}[${k}]) >> uvec4(0u, 4u, 8u, 12u)) & 15u);`)
        line(`  vec4 H${r} = vec4((uvec4(W${r}[${k}]) >> uvec4(16u, 20u, 24u, 28u)) & 15u);`)
      }
      for (let t = 0; t < T; t++) {
        line(`  { uvec4 u = texelFetch(uA, ivec2(kb * 4 + ${k}, m0 + ${t}), 0);`)
        line(`    vec4 x = h2lo(u); vec4 y = h2hi(u);`)
        line(`    s${t} += dot(x + y, vec4(1.0));`)
        line(
          `    b${t} += vec4(dot(L0, x) + dot(H0, y), dot(L1, x) + dot(H1, y), dot(L2, x) + dot(H2, y), dot(L3, x) + dot(H3, y)); }`
        )
      }
      line(`}`)
    }
    for (let t = 0; t < T; t++) line(`acc${t} += d * (b${t} - 8.0 * s${t});`)
  } else {
    // Byte texel (kb * 16 + r * 4 + w, g): weights 4w.. (lo), 16+4w.. (hi).
    for (let w = 0; w < 4; w++) {
      line(`{`)
      for (const r of [0, 1, 2, 3]) {
        line(`  vec4 B${r} = vec4(texelFetch(uW, ivec2(kb * 16 + ${r * 4 + w}, g), 0));`)
        line(`  vec4 H${r} = floor(B${r} * 0.0625);`)
        line(`  vec4 L${r} = B${r} - 16.0 * H${r};`)
      }
      for (let t = 0; t < T; t++) {
        line(`  { vec4 x = ${act(String(w), t)}; vec4 y = ${act(String(w + 4), t)};`)
        line(`    s${t} += dot(x + y, vec4(1.0));`)
        line(
          `    b${t} += vec4(dot(L0, x) + dot(H0, y), dot(L1, x) + dot(H1, y), dot(L2, x) + dot(H2, y), dot(L3, x) + dot(H3, y)); }`
        )
      }
      line(`}`)
    }
    for (let t = 0; t < T; t++) line(`acc${t} += d * (b${t} - 8.0 * s${t});`)
  }
  const outs =
    T === 1
      ? "out vec4 o0;"
      : T === 2
        ? "out uvec4 o0;"
        : Array.from({ length: T / 2 }, (_, k) => `layout(location = ${k}) out uvec4 o${k};`).join("\n")
  const writes =
    T === 1
      ? "o0 = acc0;"
      : Array.from({ length: T / 2 }, (_, k) => `o${k} = packH2(acc${2 * k}, acc${2 * k + 1});`).join("\n  ")
  if (v.noSum) {
    for (let i = 0; i < L.length; i++)
      L[i] = L[i].replace(/s\d \+= dot\([^;]*\);/g, "")
  }
  if (v.half) {
    for (let i = 0; i < L.length; i++)
      L[i] = L[i]
        .replace(/\bvec4 (w\d|a|b\d) /g, "mediump vec4 $1 ")
        .replace(/\bfloat (s\d) /g, "mediump float $1 ")
  }
  return `${HEADER}
vec4 h2pick(uvec4 u, int hi) { return hi == 1 ? h2hi(u) : h2lo(u); }
uniform ${v.acts === "f16" ? "sampler2D" : "usampler2D"} uA;
uniform ${v.weights === "rgba4" || v.weights === "unorm8" ? "sampler2D" : "usampler2D"} uW;
uniform sampler2D uS;
uniform int uKb;
${outs}
void main() {
  int g = int(gl_FragCoord.x);
  int m0 = int(gl_FragCoord.y) * ${T};
  ${Array.from({ length: T }, (_, t) => `vec4 acc${t} = vec4(0.0);`).join(" ")}
  for (int kb = 0; kb < uKb; kb++) {
${L.join("\n")}
  }
  ${writes}
}`
}

/**
 * RGBA4 weights, h2 activations, `rows` × `tokens` = 8 per fragment, one
 * RGBA32UI output of 8 halves (rows × tokens, token-major).
 */
function shapeKernel(rows: number, T: number) {
  const L: Array<string> = []
  const line = (s: string) => L.push(`    ${s}`)
  const groups = Math.ceil(rows / 4)
  for (let G = 0; G < groups; G++) line(`vec4 d${G} = texelFetch(uS, ivec2(kb, g + ${G}), 0);`)
  for (let t = 0; t < T; t++) line(`vec4 b${t}[${groups}]; float s${t} = 0.0;`)
  for (let t = 0; t < T; t++) for (let G = 0; G < groups; G++) line(`b${t}[${G}] = vec4(0.0);`)
  for (let j = 0; j < 8; j++) {
    line(`{`)
    for (let r = 0; r < rows; r++)
      line(
        `  vec4 w${r} = texelFetch(uW, ivec2(kb * 32 + ${(r % 4) * 8 + j}, g + ${r >> 2}), 0);`
      )
    const half = j & 1 ? "h2hi" : "h2lo"
    for (let t = 0; t < T; t++) {
      line(`  { vec4 a = ${half}(texelFetch(uA, ivec2(kb * 4 + ${j >> 1}, m0 + ${t}), 0));`)
      line(`    s${t} += dot(a, vec4(1.0));`)
      for (let G = 0; G < groups; G++) {
        const comps = [0, 1, 2, 3].map((c) => {
          const r = G * 4 + c
          return r < rows ? `dot(w${r}, a)` : "0.0"
        })
        line(`    b${t}[${G}] += vec4(${comps.join(", ")});`)
      }
      line(`  }`)
    }
    line(`}`)
  }
  for (let t = 0; t < T; t++)
    for (let G = 0; G < groups; G++)
      line(`acc${t}_${G} += d${G} * (15.0 * b${t}[${G}] - 8.0 * s${t});`)
  const accs: Array<string> = []
  for (let t = 0; t < T; t++) for (let G = 0; G < groups; G++) accs.push(`vec4 acc${t}_${G} = vec4(0.0);`)
  // 8 output halves: rows 8 → acc0_0, acc0_1; 4×2 → acc0_0, acc1_0;
  // 2×4 → xy of acc0..3.
  const write =
    rows === 8
      ? "o = packH2(acc0_0, acc0_1);"
      : rows === 4
        ? "o = packH2(acc0_0, acc1_0);"
        : "o = packH2(vec4(acc0_0.xy, acc1_0.xy), vec4(acc2_0.xy, acc3_0.xy));"
  return `${HEADER}
uniform usampler2D uA;
uniform sampler2D uW;
uniform sampler2D uS;
uniform int uKb;
out uvec4 o;
void main() {
  int g = int(gl_FragCoord.x) * ${groups};
  int m0 = int(gl_FragCoord.y) * ${T};
  ${accs.join(" ")}
  for (int kb = 0; kb < uKb; kb++) {
${L.join("\n")}
  }
  ${write}
}`
}

/**
 * gate|up fused with SwiGLU: each fragment computes gate rows 4g.. and up
 * rows 4g.. (row group g + uHalfG) for 2 tokens, 8 rows sharing each
 * activation fetch, and writes silu(gate) · up: 8 halves.
 */
function swigluKernel() {
  const L: Array<string> = []
  const line = (s: string) => L.push(`    ${s}`)
  line(`vec4 dg = texelFetch(uS, ivec2(kb, g), 0);`)
  line(`vec4 du = texelFetch(uS, ivec2(kb, g + uHalfG), 0);`)
  line(`vec4 bg0 = vec4(0.0); vec4 bg1 = vec4(0.0); vec4 bu0 = vec4(0.0); vec4 bu1 = vec4(0.0); float s0 = 0.0; float s1 = 0.0;`)
  for (let j = 0; j < 8; j++) {
    line(`{`)
    for (const r of [0, 1, 2, 3]) {
      line(`  vec4 g${r} = texelFetch(uW, ivec2(kb * 32 + ${r * 8 + j}, g), 0);`)
      line(`  vec4 u${r} = texelFetch(uW, ivec2(kb * 32 + ${r * 8 + j}, g + uHalfG), 0);`)
    }
    const half = j & 1 ? "h2hi" : "h2lo"
    for (const t of [0, 1]) {
      line(`  { vec4 a = ${half}(texelFetch(uA, ivec2(kb * 4 + ${j >> 1}, m0 + ${t}), 0));`)
      line(`    s${t} += dot(a, vec4(1.0));`)
      line(`    bg${t} += vec4(dot(g0, a), dot(g1, a), dot(g2, a), dot(g3, a));`)
      line(`    bu${t} += vec4(dot(u0, a), dot(u1, a), dot(u2, a), dot(u3, a)); }`)
    }
    line(`}`)
  }
  for (const t of [0, 1]) {
    line(`ag${t} += dg * (15.0 * bg${t} - 8.0 * s${t});`)
    line(`au${t} += du * (15.0 * bu${t} - 8.0 * s${t});`)
  }
  return `${HEADER}
uniform usampler2D uA;
uniform sampler2D uW;
uniform sampler2D uS;
uniform int uKb;
uniform int uHalfG;
out uvec4 o;
vec4 silu(vec4 x) { return x / (1.0 + exp(-x)); }
void main() {
  int g = int(gl_FragCoord.x);
  int m0 = int(gl_FragCoord.y) * 2;
  vec4 ag0 = vec4(0.0); vec4 ag1 = vec4(0.0); vec4 au0 = vec4(0.0); vec4 au1 = vec4(0.0);
  for (int kb = 0; kb < uKb; kb++) {
${L.join("\n")}
  }
  o = packH2(silu(ag0) * au0, silu(ag1) * au1);
}`
}

/**
 * The same matmul in a vertex shader, captured with transform feedback:
 * vertex i = y · G + g computes rows 4g..4g+3 of tokens T·y..T·y+T-1 into
 * T vec4 varyings, with no per-pixel output budget. RGBA4 weights, h2
 * activations.
 */
function vertexKernel(T: number) {
  const L: Array<string> = []
  const line = (s: string) => L.push(`    ${s}`)
  line(`vec4 d = texelFetch(uS, ivec2(kb, g), 0);`)
  for (let t = 0; t < T; t++) line(`vec4 b${t} = vec4(0.0); float s${t} = 0.0;`)
  for (let j = 0; j < 8; j++) {
    line(`{`)
    for (const r of [0, 1, 2, 3])
      line(`  vec4 w${r} = texelFetch(uW, ivec2(kb * 32 + ${r * 8 + j}, g), 0);`)
    const half = j & 1 ? "h2hi" : "h2lo"
    for (let t = 0; t < T; t++) {
      line(`  { vec4 a = ${half}(texelFetch(uA, ivec2(kb * 4 + ${j >> 1}, m0 + ${t}), 0));`)
      line(`    s${t} += dot(a, vec4(1.0));`)
      line(`    b${t} += vec4(dot(w0, a), dot(w1, a), dot(w2, a), dot(w3, a)); }`)
    }
    line(`}`)
  }
  for (let t = 0; t < T; t++) line(`acc${t} += d * (15.0 * b${t} - 8.0 * s${t});`)
  return `${HEADER}
uniform usampler2D uA;
uniform sampler2D uW;
uniform sampler2D uS;
uniform int uKb;
uniform int uG;
${Array.from({ length: T }, (_, t) => `out vec4 v${t};`).join("\n")}
void main() {
  int g = gl_VertexID % uG;
  int m0 = (gl_VertexID / uG) * ${T};
  ${Array.from({ length: T }, (_, t) => `vec4 acc${t} = vec4(0.0);`).join(" ")}
  for (int kb = 0; kb < uKb; kb++) {
${L.join("\n")}
  }
  ${Array.from({ length: T }, (_, t) => `v${t} = acc${t};`).join("\n  ")}
  gl_Position = vec4(0.0);
  gl_PointSize = 1.0;
}`
}

async function transformFeedback(
  gpu: Gpu,
  label: string,
  N: number,
  K: number,
  M: number,
  w: Texture,
  a: Texture,
  scales: Texture
) {
  const gl = gpu.gl
  const G = N / 4
  for (const T of [4, 8]) {
    const vs = gl.createShader(gl.VERTEX_SHADER)!
    gl.shaderSource(vs, vertexKernel(T).replace(/precision highp float;/, "precision highp float;"))
    gl.compileShader(vs)
    if (!gl.getShaderParameter(vs, gl.COMPILE_STATUS))
      throw new Error(`vertex shader: ${gl.getShaderInfoLog(vs)}`)
    const fs = gl.createShader(gl.FRAGMENT_SHADER)!
    gl.shaderSource(fs, `#version 300 es
precision highp float;
out vec4 o;
void main() { o = vec4(0.0); }`)
    gl.compileShader(fs)
    const program = gl.createProgram()!
    gl.attachShader(program, vs)
    gl.attachShader(program, fs)
    gl.transformFeedbackVaryings(
      program,
      Array.from({ length: T }, (_, t) => `v${t}`),
      gl.INTERLEAVED_ATTRIBS
    )
    gl.linkProgram(program)
    if (!gl.getProgramParameter(program, gl.LINK_STATUS))
      throw new Error(`link: ${gl.getProgramInfoLog(program)}`)
    const vertices = G * (M / T)
    const buffers = [0, 1].map(() => {
      const b = gl.createBuffer()!
      gl.bindBuffer(gl.TRANSFORM_FEEDBACK_BUFFER, b)
      gl.bufferData(gl.TRANSFORM_FEEDBACK_BUFFER, vertices * T * 16, gl.DYNAMIC_COPY)
      return b
    })
    gl.bindBuffer(gl.TRANSFORM_FEEDBACK_BUFFER, null)
    const tf = gl.createTransformFeedback()
    gl.useProgram(program)
    const units: Array<[string, Texture]> = [
      ["uA", a],
      ["uW", w],
      ["uS", scales]
    ]
    units.forEach(([name, tex], i) => {
      gl.activeTexture(gl.TEXTURE0 + i)
      gl.bindTexture(gl.TEXTURE_2D, tex.tex)
      gl.uniform1i(gl.getUniformLocation(program, name), i)
    })
    gl.uniform1i(gl.getUniformLocation(program, "uKb"), K / 32)
    gl.uniform1i(gl.getUniformLocation(program, "uG"), G)
    const run = (i: number) => {
      gl.useProgram(program)
      gl.enable(gl.RASTERIZER_DISCARD)
      gl.bindTransformFeedback(gl.TRANSFORM_FEEDBACK, tf)
      gl.bindBufferBase(gl.TRANSFORM_FEEDBACK_BUFFER, 0, buffers[i % 2])
      gl.beginTransformFeedback(gl.POINTS)
      gl.drawArrays(gl.POINTS, 0, vertices)
      gl.endTransformFeedback()
      gl.bindBufferBase(gl.TRANSFORM_FEEDBACK_BUFFER, 0, null)
      gl.bindTransformFeedback(gl.TRANSFORM_FEEDBACK, null)
      gl.disable(gl.RASTERIZER_DISCARD)
    }
    run(0)
    run(1)
    await gpu.finish()
    let best = Infinity
    for (let round = 0; round < 3; round++) {
      const t0 = performance.now()
      for (let i = 0; i < 10; i++) run(i)
      await gpu.finish()
      best = Math.min(best, (performance.now() - t0) / 10)
    }
    log(
      `MM2 ${label} ${N}x${K} M=${M} vertex+feedback rgba4 h2 T=${T}: ${best.toFixed(3)} ms, ${((2 * N * K * M) / best / 1e6).toFixed(0)} GFLOP/s`
    )
    for (const b of buffers) gl.deleteBuffer(b)
    gl.deleteTransformFeedback(tf)
    gl.deleteProgram(program)
  }
}

async function main(params: { batch?: number }) {
  const gpu = new Gpu(new OffscreenCanvas(1, 1))
  const gl = gpu.gl
  log(`GPU: ${gpu.limits.renderer}`)
  if (params.batch) return shapes(gpu, params.batch)
  // Bare pass cost: tiny passes alternating between two outputs (each a new
  // framebuffer, as in a forward), and the same into one output.
  {
    const tiny = gpu.program(`${HEADER}
uniform sampler2D uX;
out vec4 o;
void main() { o = texelFetch(uX, ivec2(0), 0) + 1.0; }`)
    const t = [0, 1, 2].map(() => gpu.texture(gl.RGBA32F, 64, 64))
    for (const [label, width] of [
      ["1 pixel", 1],
      ["64x64 pixels", 64]
    ] as const) {
      const run = (i: number) =>
        gpu.run({
          name: "x",
          program: tiny,
          inputs: { uX: t[i % 3] },
          output: t[(i + 1) % 3],
          rect: { x: 0, y: 0, width, height: width }
        })
      for (let i = 0; i < 50; i++) run(i)
      await gpu.finish()
      let best = Infinity
      for (let round = 0; round < 3; round++) {
        const t0 = performance.now()
        for (let i = 0; i < 1000; i++) run(i)
        await gpu.finish()
        best = Math.min(best, (performance.now() - t0) / 1000)
      }
      log(`PASS ${label}, chained: ${(best * 1000).toFixed(1)} µs per pass`)
    }
    // The same 1-pixel pass, into big textures (4096 × 512 RGBA32F).
    const big = [0, 1, 2].map(() => gpu.texture(gl.RGBA32F, 4096, 512))
    const runBig = (i: number) =>
      gpu.run({
        name: "x",
        program: tiny,
        inputs: { uX: big[i % 3] },
        output: big[(i + 1) % 3],
        rect: { x: 0, y: 0, width: 1, height: 1 }
      })
    for (let i = 0; i < 50; i++) runBig(i)
    await gpu.finish()
    let bestBig = Infinity
    for (let round = 0; round < 3; round++) {
      const t0 = performance.now()
      for (let i = 0; i < 300; i++) runBig(i)
      await gpu.finish()
      bestBig = Math.min(bestBig, (performance.now() - t0) / 300)
    }
    log(`PASS 1 pixel into 4096x512 f32, chained: ${(bestBig * 1000).toFixed(1)} µs per pass`)
    // Same output texture every pass (an atlas: each pass draws its own
    // 1-pixel spot), reading textures that aren't attached: no framebuffer
    // switch between passes.
    {
      const atlas = gpu.texture(gl.RGBA32F, 64, 64)
      const runSame = (i: number) =>
        gpu.run({
          name: "x",
          program: tiny,
          inputs: { uX: t[i % 3] },
          output: atlas,
          rect: { x: i % 64, y: (i >> 6) % 64, width: 1, height: 1 }
        })
      for (let i = 0; i < 50; i++) runSame(i)
      await gpu.finish()
      let best = Infinity
      for (let round = 0; round < 3; round++) {
        const t0 = performance.now()
        for (let i = 0; i < 1000; i++) runSame(i)
        await gpu.finish()
        best = Math.min(best, (performance.now() - t0) / 1000)
      }
      log(`PASS same output texture, new spot each pass: ${(best * 1000).toFixed(1)} µs per pass`)
      // And alternating between two programs into the same texture.
      const tiny2 = gpu.program(`${HEADER}
uniform sampler2D uX;
out vec4 o;
void main() { o = texelFetch(uX, ivec2(0), 0) * 2.0; }`)
      const runTwo = (i: number) =>
        gpu.run({
          name: "x",
          program: i & 1 ? tiny : tiny2,
          inputs: { uX: t[i % 3] },
          output: atlas,
          rect: { x: i % 64, y: (i >> 6) % 64, width: 1, height: 1 }
        })
      best = Infinity
      for (let round = 0; round < 3; round++) {
        const t0 = performance.now()
        for (let i = 0; i < 1000; i++) runTwo(i)
        await gpu.finish()
        best = Math.min(best, (performance.now() - t0) / 1000)
      }
      log(`PASS same output texture, two programs alternating: ${(best * 1000).toFixed(1)} µs per pass`)
      gpu.delete(atlas)
    }
    // Output switches every pass but nothing reads what was just written.
    {
      const src = gpu.texture(gl.RGBA32F, 64, 64)
      const outs = [0, 1].map(() => gpu.texture(gl.RGBA32F, 64, 64))
      const runAlt = (i: number) =>
        gpu.run({
          name: "x",
          program: tiny,
          inputs: { uX: src },
          output: outs[i % 2],
          rect: { x: 0, y: 0, width: 1, height: 1 }
        })
      let best = Infinity
      for (let round = 0; round < 3; round++) {
        const t0 = performance.now()
        for (let i = 0; i < 1000; i++) runAlt(i)
        await gpu.finish()
        best = Math.min(best, (performance.now() - t0) / 1000)
      }
      log(`PASS output switches, no reads of fresh output: ${(best * 1000).toFixed(1)} µs per pass`)
      // One pass per FBO but two draws in it: A→B then (same FBO B) a draw
      // reading C, to see if extra draws in a pass are cheap.
      gpu.delete(src)
      for (const o of outs) gpu.delete(o)
    }
    // One framebuffer, two attachments, the draw target chosen per draw with
    // drawBuffers (raw GL: the Gpu class binds one framebuffer per output set).
    {
      const src = gpu.texture(gl.RGBA32F, 64, 64)
      const a0 = gpu.texture(gl.RGBA32F, 64, 64)
      const a1 = gpu.texture(gl.RGBA32F, 64, 64)
      const fb = gl.createFramebuffer()
      gl.bindFramebuffer(gl.FRAMEBUFFER, fb)
      gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, a0.tex, 0)
      gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT1, gl.TEXTURE_2D, a1.tex, 0)
      const one = gpu.program(`${HEADER}
uniform sampler2D uX;
layout(location = 0) out vec4 o0;
layout(location = 1) out vec4 o1;
void main() { vec4 v = texelFetch(uX, ivec2(0), 0) + 1.0; o0 = v; o1 = v; }`)
      gl.useProgram(one.program)
      gl.activeTexture(gl.TEXTURE0)
      gl.bindTexture(gl.TEXTURE_2D, src.tex)
      gl.uniform1i(gl.getUniformLocation(one.program, "uX"), 0)
      const targets = [
        [gl.COLOR_ATTACHMENT0, gl.NONE],
        [gl.NONE, gl.COLOR_ATTACHMENT1]
      ]
      const draw = (i: number) => {
        gl.drawBuffers(targets[i % 2])
        gl.viewport(i % 64, 0, 1, 1)
        gl.drawArrays(gl.TRIANGLES, 0, 3)
      }
      for (let i = 0; i < 50; i++) draw(i)
      await gpu.finish()
      let best = Infinity
      for (let round = 0; round < 3; round++) {
        const t0 = performance.now()
        for (let i = 0; i < 1000; i++) draw(i)
        await gpu.finish()
        best = Math.min(best, (performance.now() - t0) / 1000)
      }
      log(`PASS one framebuffer, drawBuffers switching between 2 targets: ${(best * 1000).toFixed(1)} µs per draw`)
      gl.bindFramebuffer(gl.FRAMEBUFFER, null)
      gl.deleteFramebuffer(fb)
      for (const x of [src, a0, a1]) gpu.delete(x)
    }
    // Additive blending: 16 instances summed into one RGBA32F output in a
    // single draw (a split-K matmul without its reduce pass), chained.
    {
      const blendOk = !!gl.getExtension("EXT_float_blend")
      log(`EXT_float_blend: ${blendOk ? "yes" : "no"}`)
      const vs = gl.createShader(gl.VERTEX_SHADER)!
      gl.shaderSource(vs, `#version 300 es
flat out int vSlice;
void main() {
  vec2 p = vec2(float((gl_VertexID << 1) & 2), float(gl_VertexID & 2));
  vSlice = gl_InstanceID;
  gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
}`)
      gl.compileShader(vs)
      const fs = gl.createShader(gl.FRAGMENT_SHADER)!
      gl.shaderSource(fs, `${HEADER}
flat in int vSlice;
uniform sampler2D uX;
out vec4 o;
void main() { o = texelFetch(uX, ivec2(0), 0) * 0.0 + float(vSlice); }`)
      gl.compileShader(fs)
      if (!gl.getShaderParameter(fs, gl.COMPILE_STATUS)) log(`fs: ${gl.getShaderInfoLog(fs)}`)
      const prog = gl.createProgram()!
      gl.attachShader(prog, vs)
      gl.attachShader(prog, fs)
      gl.linkProgram(prog)
      const outs = [0, 1, 2].map(() => gpu.texture(gl.RGBA32F, 64, 64))
      const fbs = outs.map((o) => {
        const fb = gl.createFramebuffer()!
        gl.bindFramebuffer(gl.FRAMEBUFFER, fb)
        gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, o.tex, 0)
        return fb
      })
      gl.useProgram(prog)
      const loc = gl.getUniformLocation(prog, "uX")
      let zeroDraw = false
      const step = (i: number) => {
        gl.bindFramebuffer(gl.FRAMEBUFFER, fbs[(i + 1) % 3])
        gl.activeTexture(gl.TEXTURE0)
        gl.bindTexture(gl.TEXTURE_2D, outs[i % 3].tex)
        gl.uniform1i(loc, 0)
        gl.viewport(0, 0, 4, 1)
        if (zeroDraw) {
          // Zero the output with a draw (blend factors ZERO, ZERO), then sum.
          gl.enable(gl.BLEND)
          gl.blendFunc(gl.ZERO, gl.ZERO)
          gl.drawArrays(gl.TRIANGLES, 0, 3)
          gl.blendFunc(gl.ONE, gl.ONE)
          gl.drawArraysInstanced(gl.TRIANGLES, 0, 3, 16)
          gl.disable(gl.BLEND)
        } else {
          gl.enable(gl.SCISSOR_TEST)
          gl.scissor(0, 0, 4, 1)
          gl.clearBufferfv(gl.COLOR, 0, [0, 0, 0, 0])
          gl.disable(gl.SCISSOR_TEST)
          gl.enable(gl.BLEND)
          gl.blendFunc(gl.ONE, gl.ONE)
          gl.drawArraysInstanced(gl.TRIANGLES, 0, 3, 16)
          gl.disable(gl.BLEND)
        }
      }
      for (let i = 0; i < 30; i++) step(i)
      await gpu.finish()
      let best = Infinity
      for (let round = 0; round < 3; round++) {
        const t0 = performance.now()
        for (let i = 0; i < 300; i++) step(i)
        await gpu.finish()
        best = Math.min(best, (performance.now() - t0) / 300)
      }
      gl.bindFramebuffer(gl.FRAMEBUFFER, fbs[1])
      const px = new Float32Array(4)
      gl.readPixels(0, 0, 1, 1, gl.RGBA, gl.FLOAT, px)
      log(`BLEND clear + 16 blended instances, chained: ${(best * 1000).toFixed(1)} µs per step, sum ${px[0]} (expect 120)`)
      zeroDraw = true
      for (let i = 0; i < 30; i++) step(i)
      await gpu.finish()
      best = Infinity
      for (let round = 0; round < 3; round++) {
        const t0 = performance.now()
        for (let i = 0; i < 300; i++) step(i)
        await gpu.finish()
        best = Math.min(best, (performance.now() - t0) / 300)
      }
      gl.bindFramebuffer(gl.FRAMEBUFFER, fbs[1])
      gl.readPixels(0, 0, 1, 1, gl.RGBA, gl.FLOAT, px)
      log(`BLEND zeroing draw + 16 blended instances, chained: ${(best * 1000).toFixed(1)} µs per step, sum ${px[0]} (expect 120)`)
      gl.bindFramebuffer(gl.FRAMEBUFFER, null)
      for (const fb of fbs) gl.deleteFramebuffer(fb)
      for (const o of outs) gpu.delete(o)
    }
    // Timer queries: GPU time of 200 chained passes.
    const tq = gl.getExtension("EXT_disjoint_timer_query_webgl2") as {
      TIME_ELAPSED_EXT: number
    } | null
    log(`timer query extension: ${tq ? "yes" : "no"}`)
    if (tq) {
      const q = gl.createQuery()!
      gl.beginQuery(tq.TIME_ELAPSED_EXT, q)
      const t0 = performance.now()
      for (let i = 0; i < 200; i++)
        gpu.run({ name: "x", program: tiny, inputs: { uX: t[i % 3] }, output: t[(i + 1) % 3], rect: { x: 0, y: 0, width: 1, height: 1 } })
      gl.endQuery(tq.TIME_ELAPSED_EXT)
      await gpu.finish()
      const wall = performance.now() - t0
      for (let i = 0; i < 200; i++) {
        if (gl.getQueryParameter(q, gl.QUERY_RESULT_AVAILABLE)) break
        await new Promise((r) => setTimeout(r, 5))
      }
      const ns = gl.getQueryParameter(q, gl.QUERY_RESULT) as number
      log(`TIMER 200 chained passes: GPU ${(ns / 1e3 / 200).toFixed(1)} µs per pass, wall ${((wall * 1000) / 200).toFixed(1)} µs per pass`)
    }
    // CPU side alone: issue 1000 passes without waiting.
    const t1 = performance.now()
    for (let i = 0; i < 1000; i++)
      gpu.run({ name: "x", program: tiny, inputs: { uX: t[i % 3] }, output: t[(i + 1) % 3], rect: { x: 0, y: 0, width: 1, height: 1 } })
    const issue = (performance.now() - t1) / 1000
    await gpu.finish()
    log(`PASS issue time on the JS side: ${(issue * 1000).toFixed(1)} µs per pass`)
    for (const x of big) gpu.delete(x)
    for (const x of t) gpu.delete(x)
  }
  if (params.batch === 0) return
}

async function shapes(gpu: Gpu, batch: number) {
  const gl = gpu.gl
  const params = { batch }
  {
    // gate|up 15360 × 2048 fused with SwiGLU: 3840 fragments wide.
    const N = 15360
    const K = 2048
    const M = batch
    const blocks = K / 32
    const G = N / 4
    const w = gpu.texture(gl.RGBA4, blocks * 32, G)
    const sc = gpu.texture(gl.RGBA16F, blocks, G)
    const a = gpu.texture(gl.RGBA32UI, K / 8, M)
    const outs = [0, 1].map(() => gpu.texture(gl.RGBA32UI, G / 2, M / 2))
    const program = gpu.program(swigluKernel())
    const run = (i: number) =>
      gpu.run({
        name: "mm",
        program,
        inputs: { uA: a, uW: w, uS: sc },
        uniforms: { uKb: { int: blocks }, uHalfG: { int: G / 2 } },
        output: outs[i % 2]
      })
    run(0)
    run(1)
    await gpu.finish()
    let best = Infinity
    for (let round = 0; round < 3; round++) {
      const t0 = performance.now()
      for (let i = 0; i < 10; i++) run(i)
      await gpu.finish()
      best = Math.min(best, (performance.now() - t0) / 10)
    }
    log(
      `MM2 gate+up fused with SwiGLU ${N}x${K} M=${M}: ${best.toFixed(3)} ms, ${((2 * N * K * M) / best / 1e6).toFixed(0)} GFLOP/s`
    )
    for (const x of [w, sc, a, ...outs]) gpu.delete(x)
  }
  for (const [label, N, K] of [
    ["350M in_proj", 3072, 1024],
    ["350M in_proj as 2 K-slices", 6144, 512],
    ["350M in_proj as 4 K-slices", 12288, 256],
    ["350M qkvg", 2048, 1024],
    ["350M qkvg as 2 K-slices", 4096, 512],
    ["350M qkvg as 4 K-slices", 8192, 256],
    ["350M o", 1024, 1024],
    ["350M o as 2 K-slices", 2048, 512],
    ["350M o as 4 K-slices", 4096, 256],
    ["350M gate+up", 9216, 1024],
    ["350M gate+up as 2 K-slices", 18432, 512],
    ["350M down as 2 K-slices", 2048, 2304],
    ["350M down as 4 K-slices", 4096, 1152],
    ["350M down as 8 K-slices", 8192, 576]
] as const) {
    const blocks = K / 32
    const G = N / 4
    const wRgba4 = gpu.texture(gl.RGBA4, blocks * 32, G)
    const wBytes = gpu.texture(gl.RGBA8UI, blocks * 16, G)
    const wWords = G <= 16384 ? gpu.texture(gl.RGBA32UI, blocks * 4, G) : null
    const wUnorm8 = gpu.texture(gl.RGBA8, blocks * 16, G)
    const scales = gpu.texture(gl.RGBA16F, blocks, G)
    for (const M of [params.batch ?? 512]) {
      const aF16 = gpu.texture(gl.RGBA16F, K / 4, M)
      const aH2 = gpu.texture(gl.RGBA32UI, K / 8, M)
      const flops = 2 * N * K * M
      const variants: Array<Variant> =
        M === 1
          ? [
              { weights: "bytes", acts: "h2", tokens: 1 },
              { weights: "bytes", acts: "f16", tokens: 1 },
              { weights: "rgba4", acts: "f16", tokens: 1 }
            ]
          : [
              { weights: "rgba4", acts: "h2", tokens: 4 },
              { weights: "rgba4", acts: "h2", tokens: 2 }
            ]
      for (const v of variants) {
        const T = v.tokens
        const rows = Math.max(1, M / T)
        const outs = [0, 1].map((): Array<Texture> =>
          T === 1
            ? [gpu.texture(gl.RGBA32F, G, rows)]
            : Array.from({ length: T / 2 }, () => gpu.texture(gl.RGBA32UI, G, rows))
        )
        const program = gpu.program(kernel(v))
        const run = (i: number) =>
          gpu.run({
            name: "mm",
            program,
            inputs: {
              uA: v.acts === "f16" ? aF16 : aH2,
              uW:
                v.weights === "rgba4"
                  ? wRgba4
                  : v.weights === "unorm8"
                    ? wUnorm8
                    : v.weights === "words"
                      ? (wWords ?? wBytes)
                      : wBytes,
              uS: scales
            },
            uniforms: { uKb: { int: blocks } },
            output: outs[i % 2].length === 1 ? outs[i % 2][0] : outs[i % 2]
          })
        run(0)
        run(1)
        await gpu.finish()
        const n = M === 1 ? 50 : 10
        // Best of 3 rounds: the machine may be shared.
        let best = Infinity
        for (let round = 0; round < 3; round++) {
          const t0 = performance.now()
          for (let i = 0; i < n; i++) run(i)
          await gpu.finish()
          best = Math.min(best, (performance.now() - t0) / n)
        }
        log(
          `MM2 ${label} ${N}x${K} M=${M} w=${v.weights} a=${v.acts} T=${T}${v.half ? " half" : ""}${v.noSum ? " no sums" : ""}${v.paired ? " paired" : ""}${v.fold ? " fold" : ""}: ${best.toFixed(3)} ms, ${(flops / best / 1e6).toFixed(0)} GFLOP/s, ${((N * K) / 2 / best / 1e6).toFixed(0)} GB/s of weights`
        )
        for (const set of outs) for (const t of set) gpu.delete(t)
      }
      for (const [rows, T] of [] as Array<[number, number]>) {
        // 2 rows: half a row group per fragment, so fragments cover N / 2.
        const width = (N / rows)
        const outs = [0, 1].map(() => gpu.texture(gl.RGBA32UI, width, M / T))
        const program = gpu.program(
          rows === 2
            ? shapeKernel(2, T).replace("int g = int(gl_FragCoord.x) * 1;", "int g = int(gl_FragCoord.x) / 2;")
            : shapeKernel(rows, T)
        )
        const run = (i: number) =>
          gpu.run({
            name: "mm",
            program,
            inputs: { uA: aH2, uW: wRgba4, uS: scales },
            uniforms: { uKb: { int: blocks } },
            output: outs[i % 2]
          })
        run(0)
        run(1)
        await gpu.finish()
        let best = Infinity
        for (let round = 0; round < 3; round++) {
          const t0 = performance.now()
          for (let i = 0; i < 10; i++) run(i)
          await gpu.finish()
          best = Math.min(best, (performance.now() - t0) / 10)
        }
        log(
          `MM2 ${label} ${N}x${K} M=${M} shape ${rows}x${T} rgba4 h2: ${best.toFixed(3)} ms, ${(flops / best / 1e6).toFixed(0)} GFLOP/s`
        )
        for (const t of outs) gpu.delete(t)
      }
      gpu.delete(aF16)
      gpu.delete(aH2)
    }
    gpu.delete(wRgba4)
    gpu.delete(wBytes)
    if (wWords) gpu.delete(wWords)
    gpu.delete(wUnorm8)
    gpu.delete(scales)
  }
}

onmessage = ({ data }) => {
  main(data).then(
    () => postMessage({ type: "done" }),
    (error: unknown) =>
      postMessage({
        type: "error",
        message: error instanceof Error ? `${error.message}\n${error.stack}` : String(error)
      })
  )
}
