import test from "node:test"
import assert from "node:assert/strict"
import { Gpu, GL, type Texture, type Pass } from "./gpu/gl.js"

test("cached bindings are reused and a texture upload invalidates its active sampler binding", () => {
  const calls: string[] = []
  const gl = {
    TEXTURE0: 100,
    TEXTURE_2D: 1,
    FRAMEBUFFER: 2,
    SCISSOR_TEST: 3,
    TRIANGLES: 4,
    UNPACK_ALIGNMENT: 5,
    isContextLost: () => false,
    useProgram: () => calls.push("program"),
    activeTexture: (unit: number) => calls.push(`active:${unit}`),
    bindTexture: (_target: number, t: { name: string }) =>
      calls.push(`texture:${t.name}`),
    uniform1i: () => calls.push("sampler"),
    uniform1f: () => calls.push("float"),
    uniform2f: () => calls.push("vector"),
    uniform4i: () => calls.push("integer-vector"),
    bindFramebuffer: () => calls.push("framebuffer"),
    disable: () => undefined,
    viewport: () => undefined,
    drawArrays: () => undefined,
    pixelStorei: () => undefined,
    texSubImage2D: () => undefined
  }
  const gpu = Object.create(Gpu.prototype) as Gpu
  const fb = {},
    shader = {}
  Object.assign(gpu, {
    gl,
    lost: false,
    limits: { maxTextureUnits: 16 },
    skip: new Set(),
    repeat: new Set(),
    cacheState: true,
    boundProgram: null,
    textureUnit: 0,
    boundTextures: new Map(),
    boundFb: null,
    hints: 0,
    maxReadsPerDraw: 2 ** 28,
    maxFragmentsPerDraw: 2 ** 22,
    passes: 0,
    draws: 0,
    framebuffer: () => fb,
    uniform: () => ({})
  })
  const texture = (name: string) =>
    ({
      tex: { name },
      width: 1,
      height: 1,
      internal: GL.RGBA32F
    }) as unknown as Texture
  const a = texture("a"),
    w = texture("w"),
    out = texture("out"),
    other = texture("upload")
  const pass = {
    program: { program: shader, samplers: ["uA", "uW"], uniforms: new Map() },
    inputs: { uA: a, uW: w },
    output: out
  } as unknown as Pass
  gpu.run(pass)
  const first = calls.slice()
  gpu.run(pass)
  assert.deepEqual(calls, first)
  assert.equal(calls.filter((c) => c === "program").length, 1)
  assert.equal(calls.filter((c) => c === "framebuffer").length, 1)
  assert.equal(calls.filter((c) => c === "sampler").length, 2)
  gpu.write(
    other,
    { x: 0, y: 0, width: 1, height: 1 },
    { format: GL.RGBA, type: GL.FLOAT, pixels: new Float32Array(4) }
  )
  gpu.run(pass)
  assert.deepEqual(calls.slice(-2), ["texture:upload", "texture:w"])
  const offset: [number, number] = [1, 2],
    indices: [number, number, number, number] = [1, 2, 3, 4]
  pass.uniforms = {
    uScalar: 3,
    uShape: { int: 7 },
    uOffset: offset,
    uIndices: { ivec4: indices }
  }
  gpu.run(pass)
  const unchanged = calls.slice()
  gpu.run(pass)
  assert.deepEqual(calls, unchanged)
  offset[0] = 4
  gpu.run(pass)
  assert.deepEqual(calls.slice(unchanged.length), ["vector"])
  indices[2] = 9
  gpu.run(pass)
  assert.equal(calls.at(-1), "integer-vector")
  pass.uniforms.uScalar = -0
  gpu.run(pass)
  const negativeZero = calls.length
  pass.uniforms.uScalar = 0
  gpu.run(pass)
  assert.equal(calls.length, negativeZero + 1)
  gpu.cacheState = false
  gpu.run(pass)
  gpu.cacheState = true
  const enabled = calls.length
  gpu.run(pass)
  assert.deepEqual(calls.slice(enabled), [
    "float",
    "sampler",
    "vector",
    "integer-vector"
  ])
})
