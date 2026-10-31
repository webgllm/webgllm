/**
 * Chains of dependent matrix-vector steps, each reading the last one's
 * output, as fragment passes (the engine's way: textures through
 * framebuffers) against transform feedback (a vertex shader per output
 * group, writing a buffer the next step reads as a uniform block, no
 * framebuffer). Both whole-K and split along K (the consumer adding the
 * slices, as the engine's split matmuls do). 4-bit weights as RGBA4.
 */
import { Gpu } from "../src/gpu/gl.js"
import type { Texture } from "../src/gpu/gl.js"

const log = (text: string) => postMessage({ type: "log", text })

const STEPS = 200

async function main() {
  const gpu = new Gpu(new OffscreenCanvas(1, 1))
  const gl = gpu.gl
  gpu.hints = 0
  const maxBlock = gl.getParameter(gl.MAX_UNIFORM_BLOCK_SIZE) as number
  log(`GPU: ${gpu.limits.renderer}`)
  log(
    `MAX_UNIFORM_BLOCK_SIZE ${maxBlock}, MAX_VERTEX_TEXTURE_IMAGE_UNITS ${gl.getParameter(gl.MAX_VERTEX_TEXTURE_IMAGE_UNITS)}`
  )
  let seed = 1
  const noise16 = (n: number) => {
    const h = new Uint16Array(n)
    for (let i = 0; i < n; i++) {
      seed = (seed * 1664525 + 1013904223) >>> 0
      h[i] = seed >>> 16
    }
    return h
  }

  // Transform feedback programs: a vertex shader writing `o`.
  const compile = (type: number, src: string) => {
    const sh = gl.createShader(type)
    if (!sh) throw new Error("shader")
    gl.shaderSource(sh, src)
    gl.compileShader(sh)
    if (!gl.getShaderParameter(sh, gl.COMPILE_STATUS))
      throw new Error(`${gl.getShaderInfoLog(sh)}\n${src}`)
    return sh
  }
  const tfProgram = (vs: string) => {
    const prog = gl.createProgram()
    if (!prog) throw new Error("program")
    gl.attachShader(prog, compile(gl.VERTEX_SHADER, vs))
    gl.attachShader(
      prog,
      compile(
        gl.FRAGMENT_SHADER,
        "#version 300 es\nprecision highp float;\nout vec4 c;\nvoid main() { c = vec4(0.0); }"
      )
    )
    gl.transformFeedbackVaryings(prog, ["o"], gl.SEPARATE_ATTRIBS)
    gl.linkProgram(prog)
    if (!gl.getProgramParameter(prog, gl.LINK_STATUS))
      throw new Error(gl.getProgramInfoLog(prog) ?? "link")
    const block = gl.getUniformBlockIndex(prog, "In")
    if (block !== gl.INVALID_INDEX) gl.uniformBlockBinding(prog, block, 0)
    return prog
  }
  const tf = gl.createTransformFeedback()

  const time = async (label: string, step: (i: number) => void) => {
    for (let i = 0; i < 20; i++) step(i)
    await gpu.finish()
    let best = Infinity
    for (let round = 0; round < 3; round++) {
      const t0 = performance.now()
      for (let i = 0; i < STEPS; i++) step(i)
      await gpu.finish()
      best = Math.min(best, (performance.now() - t0) / STEPS)
    }
    log(`TF ${label}: ${(best * 1000).toFixed(1)} µs per step`)
  }

  for (const D of [1024, 2048]) {
    const KQ = D / 4
    const G = D / 4
    const w = gpu.texture(gl.RGBA4, KQ, D, {
      format: gl.RGBA,
      type: gl.UNSIGNED_SHORT_4_4_4_4,
      pixels: noise16(KQ * D)
    })
    // Weights centred on 0 and the sums scaled to keep values near 1, so
    // the activations stay random (no all-zero textures to compress).
    const dot4 = (a: string) =>
      `vec4(${[0, 1, 2, 3].map((r) => `dot(texelFetch(uW, ivec2(kq, 4 * g + ${r}), 0) - 0.5, ${a})`).join(", ")})`
    const scale = (3.46 / Math.sqrt(D)).toFixed(6)
    const init = new Float32Array(D * 16)
    for (let i = 0; i < init.length; i++) {
      seed = (seed * 1664525 + 1013904223) >>> 0
      init[i] = (seed / 2 ** 32) * 2 - 1
    }
    const state = (S: number) =>
      gpu.texture(gl.RGBA32F, G, S, {
        format: gl.RGBA,
        type: gl.FLOAT,
        pixels: init.subarray(0, G * S * 4)
      })
    const header = "#version 300 es\nprecision highp float;\nprecision highp int;\nprecision highp sampler2D;\n"
    for (const S of [1, 4, 8, 16]) {
      const span = KQ / S
      let fragmentResult = new Float32Array(0)
      // Fragment passes: (g, s) over K-slice s; the next step adds the slices.
      {
        const program = gpu.program(`${header}
uniform sampler2D uW;
uniform sampler2D uP;
out vec4 o;
vec4 act(int kq) {
  vec4 a = vec4(0.0);
  for (int s = 0; s < ${S}; s++) a += texelFetch(uP, ivec2(kq, s), 0);
  return a;
}
void main() {
  int g = int(gl_FragCoord.x);
  int s = int(gl_FragCoord.y);
  vec4 acc = vec4(0.0);
  for (int kq = s * ${span}; kq < (s + 1) * ${span}; kq++) {
    vec4 a = act(kq);
    acc += ${dot4("a")};
  }
  o = acc * ${scale};
}`)
        let outs: Array<Texture> = [0, 1].map(() => state(S))
        const step = (i: number) =>
          gpu.run({
            name: "mv",
            program,
            inputs: { uW: w, uP: outs[(i + 1) % 2] },
            output: outs[i % 2]
          })
        await time(`fragment D=${D} split ${S} (${G * S} fragments)`, step)
        for (const t of outs) gpu.delete(t)
        // 5 steps from the same start, for checking the other chain.
        outs = [0, 1].map(() => state(S))
        for (let i = 0; i < 5; i++) step(i)
        fragmentResult = gpu.readNow(outs[0])
        for (const t of outs) gpu.delete(t)
      }
      // Transform feedback reading textures, its buffer copied into a
      // texture: a drop-in for the fragment pass, other passes unchanged.
      // Vertex v writes texel (v % G, v / G): group g = v % G, slice v / G.
      {
        const prog = tfProgram(`${header}
uniform sampler2D uW;
uniform sampler2D uP;
out vec4 o;
vec4 act(int kq) {
  vec4 a = vec4(0.0);
  for (int s = 0; s < ${S}; s++) a += texelFetch(uP, ivec2(kq, s), 0);
  return a;
}
void main() {
  int s = gl_VertexID / ${G};
  int g = gl_VertexID - s * ${G};
  vec4 acc = vec4(0.0);
  for (int kq = s * ${span}; kq < (s + 1) * ${span}; kq++) {
    vec4 a = act(kq);
    acc += ${dot4("a")};
  }
  o = acc * ${scale};
}`)
        const buf = gl.createBuffer()
        gl.bindBuffer(gl.ARRAY_BUFFER, buf)
        gl.bufferData(gl.ARRAY_BUFFER, G * S * 16, gl.DYNAMIC_COPY)
        gl.bindBuffer(gl.ARRAY_BUFFER, null)
        let texs: Array<Texture> = [0, 1].map(() => state(S))
        const uWl = gl.getUniformLocation(prog, "uW")
        const uPl = gl.getUniformLocation(prog, "uP")
        const step = (i: number) => {
          const src = texs[(i + 1) % 2]
          const dst = texs[i % 2]
          // (A framebuffer left bound with a texture this step reads would
          // be a feedback loop, rasterizer discard or not.)
          gl.bindFramebuffer(gl.FRAMEBUFFER, null)
          gl.useProgram(prog)
          gl.activeTexture(gl.TEXTURE0)
          gl.bindTexture(gl.TEXTURE_2D, w.tex)
          gl.uniform1i(uWl, 0)
          gl.activeTexture(gl.TEXTURE1)
          gl.bindTexture(gl.TEXTURE_2D, src.tex)
          gl.uniform1i(uPl, 1)
          gl.enable(gl.RASTERIZER_DISCARD)
          gl.bindTransformFeedback(gl.TRANSFORM_FEEDBACK, tf)
          gl.bindBufferBase(gl.TRANSFORM_FEEDBACK_BUFFER, 0, buf)
          gl.beginTransformFeedback(gl.POINTS)
          gl.drawArrays(gl.POINTS, 0, G * S)
          gl.endTransformFeedback()
          gl.bindBufferBase(gl.TRANSFORM_FEEDBACK_BUFFER, 0, null)
          gl.bindTransformFeedback(gl.TRANSFORM_FEEDBACK, null)
          gl.disable(gl.RASTERIZER_DISCARD)
          gl.bindBuffer(gl.PIXEL_UNPACK_BUFFER, buf)
          gl.bindTexture(gl.TEXTURE_2D, dst.tex)
          gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, G, S, gl.RGBA, gl.FLOAT, 0)
          gl.bindBuffer(gl.PIXEL_UNPACK_BUFFER, null)
        }
        await time(`transform feedback + copy D=${D} split ${S} (${G * S} vertices)`, step)
        for (const t of texs) gpu.delete(t)
        texs = [0, 1].map(() => state(S))
        while (gl.getError() !== gl.NO_ERROR) {
          // Clear earlier errors.
        }
        for (let i = 0; i < 5; i++) step(i)
        const error = gl.getError()
        if (error !== gl.NO_ERROR) log(`TF   GL error 0x${error.toString(16)} in the transform feedback chain`)
        const got = gpu.readNow(texs[0])
        let err = 0
        let ref = 0
        for (let i = 0; i < got.length; i++) {
          err += (got[i] - fragmentResult[i]) ** 2
          ref += fragmentResult[i] ** 2
        }
        log(`TF   check after 5 steps: relative difference ${Math.sqrt(err / ref).toExponential(2)} (rms value ${Math.sqrt(ref / got.length).toFixed(3)})`)
        for (const t of texs) gpu.delete(t)
        gl.deleteBuffer(buf)
        gl.deleteProgram(prog)
      }
      // Transform feedback: vertex g · S + s; the next step reads the
      // slices from the uniform block.
      const bytes = G * S * 16
      if (bytes > maxBlock) {
        log(`TF transform feedback D=${D} split ${S}: skipped (${bytes} B uniform block)`)
        continue
      }
      const prog = tfProgram(`${header}
layout(std140) uniform In { vec4 p[${G * S}]; };
uniform sampler2D uW;
out vec4 o;
vec4 act(int kq) {
  vec4 a = vec4(0.0);
  for (int s = 0; s < ${S}; s++) a += p[kq * ${S} + s];
  return a;
}
void main() {
  int g = gl_VertexID / ${S};
  int s = gl_VertexID - g * ${S};
  vec4 acc = vec4(0.0);
  for (int kq = s * ${span}; kq < (s + 1) * ${span}; kq++) {
    vec4 a = act(kq);
    acc += ${dot4("a")};
  }
  o = acc * ${scale};
}`)
      const bufs = [0, 1].map(() => {
        const b = gl.createBuffer()
        gl.bindBuffer(gl.UNIFORM_BUFFER, b)
        gl.bufferData(gl.UNIFORM_BUFFER, bytes, gl.DYNAMIC_COPY)
        return b
      })
      gl.bindBuffer(gl.UNIFORM_BUFFER, null)
      const uW = gl.getUniformLocation(prog, "uW")
      const step = (i: number) => {
        const src = bufs[(i + 1) % 2]
        const dst = bufs[i % 2]
        gl.useProgram(prog)
        gl.activeTexture(gl.TEXTURE0)
        gl.bindTexture(gl.TEXTURE_2D, w.tex)
        gl.uniform1i(uW, 0)
        gl.enable(gl.RASTERIZER_DISCARD)
        gl.bindBufferBase(gl.UNIFORM_BUFFER, 0, src)
        gl.bindTransformFeedback(gl.TRANSFORM_FEEDBACK, tf)
        gl.bindBufferBase(gl.TRANSFORM_FEEDBACK_BUFFER, 0, dst)
        gl.beginTransformFeedback(gl.POINTS)
        gl.drawArrays(gl.POINTS, 0, G * S)
        gl.endTransformFeedback()
        gl.bindBufferBase(gl.TRANSFORM_FEEDBACK_BUFFER, 0, null)
        gl.bindTransformFeedback(gl.TRANSFORM_FEEDBACK, null)
        gl.bindBufferBase(gl.UNIFORM_BUFFER, 0, null)
        gl.disable(gl.RASTERIZER_DISCARD)
      }
      await time(`transform feedback D=${D} split ${S} (${G * S} vertices)`, step)
      for (const b of bufs) gl.deleteBuffer(b)
      gl.deleteProgram(prog)
    }
    gpu.delete(w)
  }
}

main().then(
  () => postMessage({ type: "done" }),
  (e: unknown) => postMessage({ type: "error", message: String(e) })
)
