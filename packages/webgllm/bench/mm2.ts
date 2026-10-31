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
  weights: "rgba4" | "bytes"
  acts: "f16" | "h2"
  tokens: 1 | 2 | 4
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
  if (v.weights === "rgba4") {
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
        : "layout(location = 0) out uvec4 o0;\nlayout(location = 1) out uvec4 o1;"
  const writes =
    T === 1
      ? "o0 = acc0;"
      : T === 2
        ? "o0 = packH2(acc0, acc1);"
        : "o0 = packH2(acc0, acc1);\n  o1 = packH2(acc2, acc3);"
  return `${HEADER}
vec4 h2pick(uvec4 u, int hi) { return hi == 1 ? h2hi(u) : h2lo(u); }
uniform ${v.acts === "f16" ? "sampler2D" : "usampler2D"} uA;
uniform ${v.weights === "rgba4" ? "sampler2D" : "usampler2D"} uW;
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
  for (const [label, N, K] of [
    ["gate+up", 15360, 2048],
    ["in_proj", 6144, 2048],
    ["down", 2048, 7680]
  ] as const) {
    const blocks = K / 32
    const G = N / 4
    const wRgba4 = gpu.texture(gl.RGBA4, blocks * 32, G)
    const wBytes = gpu.texture(gl.RGBA8UI, blocks * 16, G)
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
              { weights: "bytes", acts: "h2", tokens: 1 },
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
              uW: v.weights === "rgba4" ? wRgba4 : wBytes,
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
          `MM2 ${label} ${N}x${K} M=${M} w=${v.weights} a=${v.acts} T=${T}: ${best.toFixed(3)} ms, ${(flops / best / 1e6).toFixed(0)} GFLOP/s, ${((N * K) / 2 / best / 1e6).toFixed(0)} GB/s of weights`
        )
        for (const set of outs) for (const t of set) gpu.delete(t)
      }
      for (const [rows, T] of [
        [8, 1],
        [4, 2],
        [2, 4]
      ]) {
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
