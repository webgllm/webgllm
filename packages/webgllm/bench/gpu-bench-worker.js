/**
 * Benchmarks the Q4_0 matmul kernel variants in WebGL 2, in a worker on an
 * OffscreenCanvas, on LFM3's projection shapes, after checking each
 * variant's output against a JavaScript reference.
 */
import {
  fromHalf,
  matmulShader,
  packQ4_0,
  toHalf,
  VERTEX_SHADER
} from "./q4-matmul.js"

const log = (text) => postMessage({ type: "log", text })
const result = (data) => postMessage({ type: "result", data })

/** A deterministic random generator (mulberry32), so runs compare. */
function rng(seed) {
  let a = seed >>> 0
  return () => {
    a = (a + 0x6d2b79f5) >>> 0
    let t = a
    t = Math.imul(t ^ (t >>> 15), t | 1)
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61)
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}

/** Random Q4_0 blocks for an N × K matrix, scales like a real model's. */
function randomQ4_0(N, K, rand) {
  const blocks = new Uint8Array(N * (K / 32) * 18)
  const view = new DataView(blocks.buffer)
  for (let b = 0; b < N * (K / 32); b++) {
    view.setUint16(b * 18, toHalf(0.004 + rand() * 0.02), true)
    for (let j = 0; j < 16; j++) blocks[b * 18 + 2 + j] = (rand() * 256) | 0
  }
  return blocks
}

/** One output of the reference: Σ_k W[n][k] · A[m][k], in doubles. */
function reference(blocks, act, K, n, m) {
  const kb = K / 32
  const view = new DataView(blocks.buffer)
  let sum = 0
  for (let b = 0; b < kb; b++) {
    const src = (n * kb + b) * 18
    const d = fromHalf(view.getUint16(src, true))
    for (let j = 0; j < 16; j++) {
      const q = blocks[src + 2 + j]
      const k = b * 32 + j
      sum += d * ((q & 15) - 8) * act[m * K + k]
      sum += d * ((q >> 4) - 8) * act[m * K + k + 16]
    }
  }
  return sum
}

function setup() {
  const canvas = new OffscreenCanvas(1, 1)
  let lost = false
  canvas.addEventListener("webglcontextlost", (e) => {
    e.preventDefault()
    lost = true
    log("WebGL context lost")
  })
  const gl = canvas.getContext("webgl2", {
    antialias: false,
    depth: false,
    stencil: false,
    alpha: true,
    powerPreference: "high-performance"
  })
  if (!gl) throw new Error("No WebGL 2 here.")
  const floatTargets = gl.getExtension("EXT_color_buffer_float")
  const info = {
    vendor: gl.getParameter(gl.VENDOR),
    renderer: gl.getParameter(gl.RENDERER),
    version: gl.getParameter(gl.VERSION),
    glsl: gl.getParameter(gl.SHADING_LANGUAGE_VERSION),
    maxTextureSize: gl.getParameter(gl.MAX_TEXTURE_SIZE),
    maxArrayLayers: gl.getParameter(gl.MAX_ARRAY_TEXTURE_LAYERS),
    maxDrawBuffers: gl.getParameter(gl.MAX_DRAW_BUFFERS),
    maxColorAttachments: gl.getParameter(gl.MAX_COLOR_ATTACHMENTS),
    maxViewport: Array.from(gl.getParameter(gl.MAX_VIEWPORT_DIMS)),
    colorBufferFloat: !!floatTargets,
    timerQuery: !!gl.getExtension("EXT_disjoint_timer_query_webgl2"),
    debugRenderer: (() => {
      const ext = gl.getExtension("WEBGL_debug_renderer_info")
      return ext ? gl.getParameter(ext.UNMASKED_RENDERER_WEBGL) : null
    })()
  }
  if (!floatTargets)
    throw new Error(
      "No EXT_color_buffer_float: can't render to float textures."
    )
  return { gl, info, isLost: () => lost }
}

function compile(gl, vsSource, fsSource) {
  const shader = (type, source) => {
    const s = gl.createShader(type)
    gl.shaderSource(s, source)
    gl.compileShader(s)
    if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) {
      throw new Error(gl.getShaderInfoLog(s) ?? "compile failed")
    }
    return s
  }
  const program = gl.createProgram()
  gl.attachShader(program, shader(gl.VERTEX_SHADER, vsSource))
  gl.attachShader(program, shader(gl.FRAGMENT_SHADER, fsSource))
  gl.linkProgram(program)
  if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
    throw new Error(gl.getProgramInfoLog(program) ?? "link failed")
  }
  return program
}

function texture2D(gl, internal, width, height, format, type, data) {
  const tex = gl.createTexture()
  gl.bindTexture(gl.TEXTURE_2D, tex)
  gl.texStorage2D(gl.TEXTURE_2D, 1, internal, width, height)
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST)
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST)
  if (data)
    gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, width, height, format, type, data)
  return tex
}

function textureArray(gl, internal, width, height, layers, format, type, data) {
  const tex = gl.createTexture()
  gl.bindTexture(gl.TEXTURE_2D_ARRAY, tex)
  gl.texStorage3D(gl.TEXTURE_2D_ARRAY, 1, internal, width, height, layers)
  gl.texParameteri(gl.TEXTURE_2D_ARRAY, gl.TEXTURE_MIN_FILTER, gl.NEAREST)
  gl.texParameteri(gl.TEXTURE_2D_ARRAY, gl.TEXTURE_MAG_FILTER, gl.NEAREST)
  if (data) {
    gl.texSubImage3D(
      gl.TEXTURE_2D_ARRAY,
      0,
      0,
      0,
      0,
      width,
      height,
      layers,
      format,
      type,
      data
    )
  }
  return tex
}

/** Activations A (M × K) in the layout for S and T: f32, f16 or packed halves (h2). */
function layoutActivations(act, M, K, S, T, kind) {
  if (kind === "h2") {
    const width = K / 8
    const height = M / T
    const out = new Uint32Array(width * height * T * 4)
    for (let m = 0; m < M; m++) {
      for (let i = 0; i < K / 2; i++) {
        const x = i >> 2
        const y = Math.floor(m / T)
        const layer = m % T
        const dst = ((layer * height + y) * width + x) * 4 + (i & 3)
        const lo = toHalf(act[m * K + 2 * i])
        const hi = toHalf(act[m * K + 2 * i + 1])
        out[dst] = (hi << 16) | lo
      }
    }
    return { data: out, width, height, layers: T }
  }
  const half = kind === "f16"
  const width = K / 4 / S
  const height = M / T
  const layers = T * S
  const out = half
    ? new Uint16Array(width * height * layers * 4)
    : new Float32Array(width * height * layers * 4)
  for (let m = 0; m < M; m++) {
    for (let i = 0; i < K / 4; i++) {
      const x = Math.floor(i / S)
      const y = Math.floor(m / T)
      const layer = (m % T) * S + (i % S)
      const dst = ((layer * height + y) * width + x) * 4
      for (let c = 0; c < 4; c++) {
        const value = act[m * K + i * 4 + c]
        out[dst + c] = half ? toHalf(value) : value
      }
    }
  }
  return { data: out, width, height, layers }
}

function runVariant(gl, shape, variant, weights, act, blocks, M, rand) {
  const { K, N } = shape
  const { S, T } = variant
  // Outputs are f16 only with f16 activations; f32 otherwise.
  const half = variant.act === "f16"
  const fs = matmulShader({ ...variant, dep: true })
  const program = compile(gl, VERTEX_SHADER, fs)
  const a = layoutActivations(act, M, K, S, T, variant.act)
  const internal = half ? gl.RGBA16F : gl.RGBA32F
  const type = half ? gl.HALF_FLOAT : gl.FLOAT
  const aTex =
    variant.act === "h2"
      ? textureArray(
          gl,
          gl.RGBA32UI,
          a.width,
          a.height,
          a.layers,
          gl.RGBA_INTEGER,
          gl.UNSIGNED_INT,
          a.data
        )
      : textureArray(
          gl,
          internal,
          a.width,
          a.height,
          a.layers,
          gl.RGBA,
          type,
          a.data
        )
  const outWidth = N / 4 / S
  const outHeight = M / T
  // Two outputs, alternated, so every draw is its own pass: a tile-based
  // GPU skips a full-screen draw that the next draw overwrites.
  const targets = [0, 1].map(() => {
    const tex = textureArray(
      gl,
      internal,
      outWidth,
      outHeight,
      T * S,
      gl.RGBA,
      type,
      null
    )
    const fb = gl.createFramebuffer()
    gl.bindFramebuffer(gl.FRAMEBUFFER, fb)
    const buffers = []
    for (let l = 0; l < T * S; l++) {
      gl.framebufferTextureLayer(
        gl.FRAMEBUFFER,
        gl.COLOR_ATTACHMENT0 + l,
        tex,
        0,
        l
      )
      buffers.push(gl.COLOR_ATTACHMENT0 + l)
    }
    gl.drawBuffers(buffers)
    const status = gl.checkFramebufferStatus(gl.FRAMEBUFFER)
    if (status !== gl.FRAMEBUFFER_COMPLETE) {
      throw new Error(`framebuffer incomplete: 0x${status.toString(16)}`)
    }
    return { tex, fb }
  })

  gl.useProgram(program)
  // One unit per sampler: samplers of different types can't share one.
  const h2 = variant.act === "h2"
  gl.activeTexture(gl.TEXTURE0)
  gl.bindTexture(gl.TEXTURE_2D, weights.quants)
  gl.activeTexture(gl.TEXTURE1)
  gl.bindTexture(gl.TEXTURE_2D, weights.scales)
  gl.activeTexture(h2 ? gl.TEXTURE5 : gl.TEXTURE2)
  gl.bindTexture(gl.TEXTURE_2D_ARRAY, aTex)
  gl.activeTexture(gl.TEXTURE4)
  gl.bindTexture(gl.TEXTURE_2D, weights.bytes)
  gl.uniform1i(gl.getUniformLocation(program, "uW"), 0)
  gl.uniform1i(gl.getUniformLocation(program, "uS"), 1)
  gl.uniform1i(gl.getUniformLocation(program, "uA"), 2)
  gl.uniform1i(gl.getUniformLocation(program, "uW8"), 4)
  gl.uniform1i(gl.getUniformLocation(program, "uAh"), 5)
  gl.uniform1i(gl.getUniformLocation(program, "uKb"), K / 32)
  gl.uniform1i(gl.getUniformLocation(program, "uPrev"), 3)
  gl.uniform1f(gl.getUniformLocation(program, "uZero"), 0)
  gl.viewport(0, 0, outWidth, outHeight)

  let calls = 0
  // Each draw reads the previous draw's output (see `dep`).
  const draw = () => {
    const i = calls++
    gl.activeTexture(gl.TEXTURE3)
    gl.bindTexture(gl.TEXTURE_2D_ARRAY, targets[(i + 1) & 1].tex)
    gl.bindFramebuffer(gl.FRAMEBUFFER, targets[i & 1].fb)
    gl.drawArrays(gl.TRIANGLES, 0, 3)
  }
  const px = half ? new Uint16Array(4) : new Float32Array(4)
  const sync = () => {
    gl.readBuffer(gl.COLOR_ATTACHMENT0)
    gl.readPixels(0, 0, 1, 1, gl.RGBA, type, px)
  }

  // Correctness: 64 random outputs against the reference.
  draw()
  gl.bindFramebuffer(gl.FRAMEBUFFER, targets[0].fb)
  const layerData = []
  for (let l = 0; l < T * S; l++) {
    const data = half
      ? new Uint16Array(outWidth * outHeight * 4)
      : new Float32Array(outWidth * outHeight * 4)
    gl.readBuffer(gl.COLOR_ATTACHMENT0 + l)
    gl.readPixels(0, 0, outWidth, outHeight, gl.RGBA, type, data)
    layerData.push(data)
  }
  let maxErr = 0
  let maxRef = 0
  for (let check = 0; check < 64; check++) {
    const m = Math.floor(rand() * M)
    const n = Math.floor(rand() * N)
    const i = n >> 2
    const x = Math.floor(i / S)
    const y = Math.floor(m / T)
    const layer = (m % T) * S + (i % S)
    const raw = layerData[layer][(y * outWidth + x) * 4 + (n & 3)]
    const got = half ? fromHalf(raw) : raw
    const want = reference(blocks, act, K, n, m)
    maxErr = Math.max(maxErr, Math.abs(got - want))
    maxRef = Math.max(maxRef, Math.abs(want))
  }
  const relErr = maxErr / maxRef

  // Timing: warm up, then two batches of draws. Each ends in a readback,
  // whose fixed cost (10-25 ms on Android) the difference cancels.
  for (let i = 0; i < 3; i++) draw()
  sync()
  const batch = (n) => {
    const t0 = performance.now()
    for (let i = 0; i < n; i++) draw()
    sync()
    return performance.now() - t0
  }
  const single = batch(1)
  const few = Math.max(4, Math.min(50, Math.round(150 / Math.max(single, 1))))
  const many = few * 4
  const tFew = batch(few)
  const tMany = batch(many)
  const ms = (tMany - tFew) / (many - few)
  const iterations = many
  const overhead = tFew - ms * few

  for (const { tex, fb } of targets) {
    gl.deleteFramebuffer(fb)
    gl.deleteTexture(tex)
  }
  gl.deleteTexture(aTex)
  gl.deleteProgram(program)
  const macs = M * N * K
  return {
    ms,
    single,
    iterations,
    overhead,
    gmacs: macs / ms / 1e6,
    relErr,
    maxRef
  }
}

/** LFM3's projections: [K, N, count in the model]. */
const SHAPES = [
  { name: "ffn gate / up", K: 2048, N: 7680, count: 48 },
  { name: "ffn down", K: 7680, N: 2048, count: 24 },
  { name: "conv in_proj", K: 2048, N: 6144, count: 13 },
  { name: "conv out_proj / attn q / o", K: 2048, N: 2048, count: 13 + 22 },
  { name: "attn k / v", K: 2048, N: 512, count: 22 }
]

const VARIANTS = [
  { S: 1, T: 2, unpack: "shift", act: "f32" },
  { S: 1, T: 2, unpack: "shift", act: "h2" },
  { S: 1, T: 2, unpack: "shift", act: "h2", prec: "mediump" },
  { S: 1, T: 2, unpack: "float", act: "f32" },
  { S: 1, T: 2, unpack: "float", act: "h2", prec: "mediump" },
  { S: 1, T: 4, unpack: "shift", act: "h2", prec: "mediump" },
  { S: 1, T: 4, unpack: "float", act: "h2", prec: "mediump" },
  { S: 1, T: 1, unpack: "float", act: "h2", prec: "mediump" }
]

const variantName = (v) =>
  `S${v.S} T${v.T} ${v.unpack} ${v.act}${v.prec === "mediump" ? " mp" : ""}`

async function run({ M = 128, variants = VARIANTS, shapes = SHAPES }) {
  const { gl, info, isLost } = setup()
  log(`GPU: ${JSON.stringify(info)}`)
  result({ kind: "info", info })
  const rand = rng(1234)
  const rows = []
  for (const shape of shapes) {
    const { K, N } = shape
    const blocks = randomQ4_0(N, K, rand)
    const { quants, scales } = packQ4_0(blocks, N, K)
    const weights = {
      quants: texture2D(
        gl,
        gl.RGBA32UI,
        (K / 32) * 4,
        N / 4,
        gl.RGBA_INTEGER,
        gl.UNSIGNED_INT,
        quants
      ),
      scales: texture2D(
        gl,
        gl.RGBA16F,
        K / 32,
        N / 4,
        gl.RGBA,
        gl.HALF_FLOAT,
        scales
      ),
      // The same quant bytes as RGBA8, for the float-unpack variants.
      bytes: texture2D(
        gl,
        gl.RGBA8,
        (K / 32) * 16,
        N / 4,
        gl.RGBA,
        gl.UNSIGNED_BYTE,
        new Uint8Array(quants.buffer)
      )
    }
    const act = new Float32Array(M * K)
    for (let i = 0; i < act.length; i++) act[i] = rand() * 2 - 1
    for (const variant of variants) {
      if (variant.S * variant.T > info.maxDrawBuffers || M % variant.T !== 0)
        continue
      if (isLost()) throw new Error("WebGL context lost")
      try {
        const r = runVariant(gl, shape, variant, weights, act, blocks, M, rand)
        const row = {
          shape: shape.name,
          K,
          N,
          variant: variantName(variant),
          ...r
        }
        rows.push(row)
        log(
          `${shape.name.padEnd(28)} ${K}x${N}  ${variantName(variant).padEnd(20)} ` +
            `${r.ms.toFixed(2)} ms  ${r.gmacs.toFixed(1)} GMAC/s  relErr ${r.relErr.toExponential(1)}  ` +
            `(sync ${r.overhead.toFixed(1)} ms, x${r.iterations})`
        )
      } catch (error) {
        log(`${shape.name} ${variantName(variant)}: ${error.message}`)
      }
      // Let the worker breathe between variants.
      await new Promise((resolve) => setTimeout(resolve, 0))
    }
    gl.deleteTexture(weights.quants)
    gl.deleteTexture(weights.scales)
    gl.deleteTexture(weights.bytes)
  }

  // The model's matmul time for M tokens with each variant, and with the
  // best variant per shape.
  const summary = {}
  for (const variant of variants.map(variantName)) {
    let ms = 0
    let complete = true
    for (const shape of shapes) {
      const row = rows.find(
        (r) => r.shape === shape.name && r.variant === variant
      )
      if (!row || row.relErr > 1e-2) complete = false
      else ms += row.ms * shape.count
    }
    if (complete && ms > 0)
      summary[variant] = { ms, tokensPerSecond: (M / ms) * 1000 }
  }
  let bestMs = 0
  const best = {}
  for (const shape of shapes) {
    const candidates = rows.filter(
      (r) => r.shape === shape.name && r.relErr <= 1e-2
    )
    const top = candidates.sort((a, b) => a.ms - b.ms)[0]
    if (top) {
      best[shape.name] = top.variant
      bestMs += top.ms * shape.count
    }
  }
  summary["best per shape"] = {
    ms: bestMs,
    tokensPerSecond: (M / bestMs) * 1000,
    best
  }
  for (const [name, s] of Object.entries(summary)) {
    log(
      `model matmuls, ${M} tokens, ${name}: ${s.ms.toFixed(0)} ms = ${s.tokensPerSecond.toFixed(1)} tokens/s`
    )
  }
  result({ kind: "gpu", M, rows, summary })
}

onmessage = ({ data }) => {
  if (data.type !== "run") return
  run(data).then(
    () => postMessage({ type: "done" }),
    (error) =>
      postMessage({ type: "error", message: String(error?.stack ?? error) })
  )
}
