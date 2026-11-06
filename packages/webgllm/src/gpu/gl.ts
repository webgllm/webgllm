/**
 * A thin layer over WebGL 2 for compute in fragment shaders: textures as
 * tensors, one full-viewport triangle per pass, and async readback.
 *
 * Every pass draws into a texture through a framebuffer, so outputs must be
 * color-renderable: float targets need EXT_color_buffer_float, checked up
 * front along with the texture size the model needs.
 */

import { SubmissionBudget } from "./submission-budget.js"

export type GpuLimits = {
  maxTextureSize: number
  maxDrawBuffers: number
  /** Textures a pass can read (MAX_TEXTURE_IMAGE_UNITS: at least 16). */
  maxTextureUnits: number
  renderer: string
}

export class GpuError extends Error {}

/**
 * The WebGL 2 enums used here, as numbers: `WebGL2RenderingContext` isn't
 * defined where the module may also be imported (a server rendering pages).
 */
export const GL = {
  BYTE: 0x1400,
  UNSIGNED_BYTE: 0x1401,
  UNSIGNED_INT: 0x1405,
  FLOAT: 0x1406,
  HALF_FLOAT: 0x140b,
  RGBA: 0x1908,
  RGBA32F: 0x8814,
  RGBA4: 0x8056,
  RGBA8: 0x8058,
  UNSIGNED_SHORT_4_4_4_4: 0x8033,
  RGBA16F: 0x881a,
  R32F: 0x822e,
  R32UI: 0x8236,
  R8UI: 0x8232,
  RGBA32UI: 0x8d70,
  RGBA8UI: 0x8d7c,
  RGBA8I: 0x8d8e,
  RED_INTEGER: 0x8d94,
  RGBA_INTEGER: 0x8d99
} as const

/** A 2D texture used as a tensor. */
export type Texture = {
  /** Unique per texture, for caching framebuffers over several outputs. */
  id: number
  tex: WebGLTexture
  width: number
  height: number
  internal: number
  /** Bytes it holds on the GPU, for the memory count. */
  bytes: number
}

export type Program = {
  program: WebGLProgram
  uniforms: Map<string, WebGLUniformLocation | null>
  samplers: Array<string>
  samplersBound?: boolean
  uniformValues?: Map<string, NonNullable<Pass["uniforms"]>[string]>
}

export type Uniform =
  | number
  | [number, number]
  | [number, number, number]
  | [number, number, number, number]

export type Pass = {
  /** What kind of pass it is, for profiling (see `skip`). */
  name?: string
  program: Program
  /** Sampler uniform name → texture. */
  inputs: Record<string, Texture>
  /** Integer uniforms go as `{ int: n }`; plain numbers are floats. */
  uniforms?: Record<
    string,
    Uniform | { int: number } | { ivec4: [number, number, number, number] }
  >
  /**
   * The texture drawn into, or several of one size (multiple render
   * targets: the shader's output `location` i writes texture i).
   */
  output: Texture | Array<Texture>
  /** The rectangle to draw, in texels; the whole output by default. */
  rect?: { x: number; y: number; width: number; height: number }
  /**
   * About how many texels a fragment reads (1 by default): draws are split
   * by fragments times this, so a heavy pass never runs long in one draw.
   */
  work?: number
}

const VERTEX_SHADER = `#version 300 es
void main() {
  vec2 p = vec2(float((gl_VertexID << 1) & 2), float(gl_VertexID & 2));
  gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
}`

/** Bytes per texel for the internal formats used here. */
const TEXEL_BYTES: Record<number, number> = {
  [0x8d70]: 16, // RGBA32UI
  [0x8814]: 16, // RGBA32F
  [0x881a]: 8, // RGBA16F
  [0x8d7c]: 4, // RGBA8UI
  [0x8056]: 2, // RGBA4
  [0x8d8e]: 4, // RGBA8I
  [0x8058]: 4, // RGBA8
  [0x822e]: 4, // R32F
  [0x8236]: 4, // R32UI
  [0x8232]: 1 // R8UI
}

export class Gpu {
  readonly gl: WebGL2RenderingContext
  readonly limits: GpuLimits
  private vertexShader: WebGLShader
  private programs = new Map<string, Program>()
  private framebuffers = new Map<string, WebGLFramebuffer>()
  private nextId = 1
  private lost = false
  /** Bytes in textures created and not deleted. */
  bytes = 0
  /** Draw calls issued, for profiling. */
  draws = 0
  /** Render passes started (each change of output texture), for profiling. */
  passes = 0
  /** When set, collects the name of each pass that starts a render pass. */
  trace: Array<string> | null = null
  /** With `trace`: draws per pass name. */
  drawTrace: Map<string, number> | null = null
  private boundFb: WebGLFramebuffer | null = null
  /** Pass names not to draw: for timing what the rest costs (results are wrong). */
  skip = new Set<string>()
  /**
   * Render-area hints for tile-based GPUs, which otherwise may load and
   * store a whole output texture for a pass that draws a corner of it:
   * 1 scissors each pass to its rect, 2 invalidates the rect first (it's
   * overwritten), 3 both. Scissoring cut a 1-pixel pass into a 4096×512
   * texture from 151 to 60 µs on a PowerVR GPU (invalidating, to 76; both,
   * 146).
   */
  hints = 1
  /**
   * Pass names to draw twice, into the same target (no new render pass):
   * for timing what their fragments cost beyond the pass itself.
   */
  repeat = new Set<string>()
  /** How many times `repeat` passes are drawn. */
  repeatTimes = 2
  /** Fragments per draw before a pass is split into bands (GPU watchdog). */
  maxFragmentsPerDraw = 1 << 22
  /**
   * Texel reads (fragments × Pass.work) per draw before a pass is split:
   * tens of milliseconds on a slow integrated GPU, far inside a watchdog
   * (Windows resets a GPU after 2 s).
   */
  maxReadsPerDraw = 1 << 28

  cacheState = false
  private boundProgram: WebGLProgram | null = null
  private textureUnit = 0
  private boundTextures = new Map<number, WebGLTexture>()

  constructor(canvas: OffscreenCanvas | HTMLCanvasElement) {
    canvas.addEventListener("webglcontextlost", (event) => {
      event.preventDefault()
      this.lost = true
    })
    const gl = canvas.getContext("webgl2", {
      antialias: false,
      depth: false,
      stencil: false,
      alpha: true,
      premultipliedAlpha: false,
      preserveDrawingBuffer: false,
      powerPreference: "high-performance"
    }) as WebGL2RenderingContext | null
    if (!gl) throw new GpuError("This browser has no WebGL 2.")
    if (!gl.getExtension("EXT_color_buffer_float")) {
      throw new GpuError(
        "This GPU can't render to float textures (EXT_color_buffer_float)."
      )
    }
    this.gl = gl
    const debug = gl.getExtension("WEBGL_debug_renderer_info")
    this.limits = {
      maxTextureSize: gl.getParameter(gl.MAX_TEXTURE_SIZE) as number,
      maxDrawBuffers: gl.getParameter(gl.MAX_DRAW_BUFFERS) as number,
      maxTextureUnits: gl.getParameter(gl.MAX_TEXTURE_IMAGE_UNITS) as number,
      renderer: String(
        debug
          ? gl.getParameter(debug.UNMASKED_RENDERER_WEBGL)
          : gl.getParameter(gl.RENDERER)
      )
    }
    this.vertexShader = this.shader(gl.VERTEX_SHADER, VERTEX_SHADER)
    gl.disable(gl.DITHER)
    gl.disable(gl.BLEND)
    gl.disable(gl.DEPTH_TEST)
  }

  get isLost() {
    return this.lost || this.gl.isContextLost()
  }

  private shader(type: number, source: string) {
    const gl = this.gl
    const shader = gl.createShader(type)
    if (!shader) throw new GpuError("Couldn't create a shader.")
    gl.shaderSource(shader, source)
    gl.compileShader(shader)
    if (
      !gl.getShaderParameter(shader, gl.COMPILE_STATUS) &&
      !gl.isContextLost()
    ) {
      const log = gl.getShaderInfoLog(shader)
      const numbered = source
        .split("\n")
        .map((line, i) => `${String(i + 1).padStart(4)} ${line}`)
        .join("\n")
      throw new GpuError(`Shader didn't compile: ${log}\n${numbered}`)
    }
    return shader
  }

  /** Programs compiled, and the slowest ones' milliseconds by a source prefix (load profiling). */
  compiled = {
    count: 0,
    ms: 0,
    slowest: [] as Array<[number, string]>,
    /** Each source's milliseconds. */
    times: new Map<string, number>()
  }
  /**
   * Appended to every fragment shader compiled (profiling: a fresh salt
   * keeps the browser from reusing compiles from an earlier load).
   */
  salt = ""

  /** A fragment shader's program, compiled once per source. */
  program(source: string): Program {
    const cached = this.programs.get(source)
    if (cached) return cached
    const started = performance.now()
    try {
      return this.compile(source)
    } finally {
      const ms = performance.now() - started
      this.compiled.count++
      this.compiled.ms += ms
      this.compiled.times.set(source, ms)
      // A source's first line after the header, to tell kernels apart.
      const tag = source.split("\n").find((l) => l.startsWith("uniform")) ?? ""
      this.compiled.slowest.push([Math.round(ms), `${tag.slice(0, 40)} ${source.length}`])
      this.compiled.slowest.sort((a, b) => b[0] - a[0])
      this.compiled.slowest.length = Math.min(this.compiled.slowest.length, 8)
    }
  }

  private compile(source: string): Program {
    const gl = this.gl
    const program = gl.createProgram()
    if (!program) throw new GpuError("Couldn't create a program.")
    gl.attachShader(program, this.vertexShader)
    const fragment = this.shader(gl.FRAGMENT_SHADER, source + this.salt)
    gl.attachShader(program, fragment)
    gl.linkProgram(program)
    // (Flagged for deletion: freed with the program.)
    gl.deleteShader(fragment)
    const entry = { program, uniforms: new Map(), samplers: this.linked(program) }
    this.programs.set(source, entry)
    return entry
  }

  /** A linked program's samplers (throws if it didn't link). */
  private linked(program: WebGLProgram): Array<string> {
    const gl = this.gl
    if (
      !gl.getProgramParameter(program, gl.LINK_STATUS) &&
      !gl.isContextLost()
    ) {
      throw new GpuError(
        `Program didn't link: ${gl.getProgramInfoLog(program)}`
      )
    }
    const samplers: Array<string> = []
    const count = gl.getProgramParameter(program, gl.ACTIVE_UNIFORMS) as number
    for (let i = 0; i < count; i++) {
      const info = gl.getActiveUniform(program, i)
      if (!info) continue
      const t = info.type
      if (
        t === gl.SAMPLER_2D ||
        t === gl.INT_SAMPLER_2D ||
        t === gl.UNSIGNED_INT_SAMPLER_2D
      ) {
        samplers.push(info.name)
      }
    }
    return samplers
  }

  private uniform(p: Program, name: string) {
    if (!p.uniforms.has(name)) {
      p.uniforms.set(name, this.gl.getUniformLocation(p.program, name))
    }
    return p.uniforms.get(name) ?? null
  }

  /**
   * A texture of `width` × `height` texels. `format`/`type` describe `data`
   * when given (RGBA_INTEGER/UNSIGNED_INT for RGBA32UI, RGBA/FLOAT for
   * RGBA32F, RGBA/HALF_FLOAT for RGBA16F, …).
   */
  texture(
    internal: number,
    width: number,
    height: number,
    data?: { format: number; type: number; pixels: ArrayBufferView }
  ): Texture {
    const gl = this.gl
    const max = this.limits.maxTextureSize
    if (width > max || height > max || width < 1 || height < 1) {
      throw new GpuError(
        `A ${width}×${height} texture is past this GPU's limit of ${max}.`
      )
    }
    const tex = gl.createTexture()
    if (!tex) throw new GpuError("Couldn't create a texture.")
    gl.bindTexture(gl.TEXTURE_2D, tex)
    this.boundTextures.set(this.textureUnit, tex)
    gl.texStorage2D(gl.TEXTURE_2D, 1, internal, width, height)
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST)
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST)
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE)
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE)
    if (data) {
      gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1)
      this.markGpuWork()
      gl.texSubImage2D(
        gl.TEXTURE_2D,
        0,
        0,
        0,
        width,
        height,
        data.format,
        data.type,
        data.pixels
      )
    }
    const bytes = width * height * (TEXEL_BYTES[internal] ?? 16)
    this.bytes += bytes
    const texture = { id: this.nextId++, tex, width, height, internal, bytes }
    this.live.add(texture)
    return texture
  }

  /** Textures made and not yet deleted (for dispose). */
  private live = new Set<Texture>()

  /**
   * Frees everything this context holds, now: every texture, framebuffer
   * and program deleted and the deletes finished, then the context lost.
   * (Losing the context alone leaves the memory to the browser's own
   * timing: a page that loads another model at once can hold both.)
   */
  dispose() {
    const gl = this.gl
    if (!gl.isContextLost()) {
      for (const t of [...this.live]) this.delete(t)
      for (const fb of this.framebuffers.values()) gl.deleteFramebuffer(fb)
      this.framebuffers.clear()
      for (const p of this.programs.values()) gl.deleteProgram(p.program)
      this.programs.clear()
      gl.deleteShader(this.vertexShader)
      for (const fence of this.paced) gl.deleteSync(fence.sync)
      gl.finish()
    }
    this.lastOutput = null
    this.paced.length = 0
    gl.getExtension("WEBGL_lose_context")?.loseContext()
  }

  /** Release an idle context in bounded batches so other rendering can run. */
  async disposePaced() {
    try {
      let bytes = 0
      for (const texture of this.live) {
        if (this.isLost) break
        this.delete(texture)
        bytes += texture.bytes
        if (bytes >= 64 * 2 ** 20) {
          await this.finish()
          bytes = 0
        }
      }
    } catch (error) {
      if (!this.isLost) throw error
    } finally {
      this.dispose()
    }
  }

  /** Writes `pixels` into a rectangle of a texture. */
  write(
    target: Texture,
    rect: { x: number; y: number; width: number; height: number },
    data: { format: number; type: number; pixels: ArrayBufferView }
  ) {
    const gl = this.gl
    gl.bindTexture(gl.TEXTURE_2D, target.tex)
    this.boundTextures.set(this.textureUnit, target.tex)
    gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1)
    gl.texSubImage2D(
      gl.TEXTURE_2D,
      0,
      rect.x,
      rect.y,
      rect.width,
      rect.height,
      data.format,
      data.type,
      data.pixels
    )
  }

  /**
   * Whether the GPU can draw into textures of these formats at once (one
   * per target). Asked of the GPU, not assumed: some cap the bits a pixel
   * writes (ANGLE on Apple's mobile GPUs: 512, so 6 RGBA32F targets fail).
   */
  canRender(internals: Array<number>) {
    const gl = this.gl
    const count = internals.length
    if (count > this.limits.maxDrawBuffers) return false
    if (count > (gl.getParameter(gl.MAX_COLOR_ATTACHMENTS) as number)) return false
    const integer = (internal: number) =>
      internal === GL.RGBA32UI || internal === GL.R32UI || internal === GL.R8UI
    const textures = internals.map((internal) => this.texture(internal, 1, 1))
    try {
      // Drawn and read back, not just "complete": some drivers accept a
      // framebuffer they then draw wrong. Target i gets (i + 1, 2, 3, 4).
      const outs = internals
        .map((internal, i) =>
          integer(internal)
            ? `layout(location = ${i}) out uvec4 o${i};`
            : `layout(location = ${i}) out vec4 o${i};`
        )
        .join("\n")
      const sets = internals
        .map((internal, i) =>
          integer(internal)
            ? `o${i} = uvec4(${i + 1}u, 2u, 3u, 4u);`
            : `o${i} = vec4(${i + 1}.0, 2.0, 3.0, 4.0);`
        )
        .join(" ")
      const program = this.program(
        `#version 300 es\nprecision highp float;\n${outs}\nvoid main() { ${sets} }`
      )
      this.run({
        program,
        inputs: {},
        output: count === 1 ? textures[0] : textures,
        rect: { x: 0, y: 0, width: 1, height: 1 }
      })
      gl.bindFramebuffer(gl.FRAMEBUFFER, this.framebuffer(count === 1 ? textures[0] : textures))
      let ok = true
      internals.forEach((internal, i) => {
        gl.readBuffer(gl.COLOR_ATTACHMENT0 + i)
        const got = integer(internal) ? new Uint32Array(4) : new Float32Array(4)
        if (integer(internal))
          gl.readPixels(0, 0, 1, 1, gl.RGBA_INTEGER, gl.UNSIGNED_INT, got)
        else gl.readPixels(0, 0, 1, 1, gl.RGBA, gl.FLOAT, got)
        // (Single-channel formats read back their first channel only.)
        const channels = internal === GL.R32UI || internal === GL.R8UI ? 1 : 4
        for (let c = 0; c < channels; c++)
          if (got[c] !== (c === 0 ? i + 1 : c + 1)) ok = false
      })
      gl.readBuffer(gl.COLOR_ATTACHMENT0)
      return ok && gl.getError() === gl.NO_ERROR
    } catch {
      return false
    } finally {
      gl.bindFramebuffer(gl.FRAMEBUFFER, null)
      this.boundFb = null
      textures.forEach((t) => this.delete(t))
    }
  }

  /** Whether the GPU has run out of memory (a texture it couldn't make). */
  outOfMemory() {
    const gl = this.gl
    let out = gl.isContextLost()
    for (let e = gl.getError(); e !== gl.NO_ERROR && !out; e = gl.getError())
      out = e === gl.OUT_OF_MEMORY || e === gl.CONTEXT_LOST_WEBGL
    return out
  }

  delete(texture: Texture) {
    for (const [key, fb] of this.framebuffers) {
      if (key.split(",").includes(String(texture.id))) {
        this.gl.deleteFramebuffer(fb)
        this.framebuffers.delete(key)
      }
    }
    this.gl.deleteTexture(texture.tex)
    for (const [unit, bound] of this.boundTextures)
      if (bound === texture.tex) this.boundTextures.delete(unit)
    this.live.delete(texture)
    this.bytes -= texture.bytes
    if (this.lastOutput === texture) this.lastOutput = null
  }

  private framebuffer(output: Texture | Array<Texture>) {
    const textures = Array.isArray(output) ? output : [output]
    const key = textures.map((t) => t.id).join(",")
    let fb = this.framebuffers.get(key)
    if (!fb) {
      const gl = this.gl
      if (textures.length > this.limits.maxDrawBuffers) {
        throw new GpuError(
          `This GPU draws into ${this.limits.maxDrawBuffers} textures at once; a pass needs ${textures.length}.`
        )
      }
      const created = gl.createFramebuffer()
      if (!created) throw new GpuError("Couldn't create a framebuffer.")
      fb = created
      gl.bindFramebuffer(gl.FRAMEBUFFER, fb)
      textures.forEach((texture, i) =>
        gl.framebufferTexture2D(
          gl.FRAMEBUFFER,
          gl.COLOR_ATTACHMENT0 + i,
          gl.TEXTURE_2D,
          texture.tex,
          0
        )
      )
      if (textures.length > 1)
        gl.drawBuffers(textures.map((_, i) => gl.COLOR_ATTACHMENT0 + i))
      const status = gl.checkFramebufferStatus(gl.FRAMEBUFFER)
      if (status !== gl.FRAMEBUFFER_COMPLETE && !gl.isContextLost()) {
        throw new GpuError(
          `Can't render to this texture (framebuffer status 0x${status.toString(16)}).`
        )
      }
      this.framebuffers.set(key, fb)
    }
    return fb
  }

  /** Runs a pass: one triangle over the output rectangle, in bands if it's big. */
  run(pass: Pass) {
    if (this.isLost) throw new GpuError("The GPU context was lost.")
    if (pass.name && this.skip.has(pass.name)) return
    const gl = this.gl
    const { program } = pass
    const outputs = Array.isArray(pass.output) ? pass.output : [pass.output]
    if (program.samplers.length > this.limits.maxTextureUnits) {
      throw new GpuError(
        `This GPU reads ${this.limits.maxTextureUnits} textures in a pass; one needs ${program.samplers.length}.`
      )
    }
    this.lastOutput = outputs[0]
    if (!this.cacheState || this.boundProgram !== program.program) {
      gl.useProgram(program.program)
      this.boundProgram = program.program
    }
    let unit = 0
    for (const name of program.samplers) {
      const input = pass.inputs[name]
      if (!input) throw new GpuError(`Pass is missing input ${name}.`)
      if (outputs.some((o) => o.tex === input.tex))
        throw new GpuError(`Pass reads and writes ${name}.`)
      if (!this.cacheState || this.boundTextures.get(unit) !== input.tex) {
        if (!this.cacheState || this.textureUnit !== unit)
          gl.activeTexture(gl.TEXTURE0 + unit)
        this.textureUnit = unit
        gl.bindTexture(gl.TEXTURE_2D, input.tex)
        this.boundTextures.set(unit, input.tex)
      }
      if (!this.cacheState || !program.samplersBound)
        gl.uniform1i(this.uniform(program, name), unit)
      unit++
    }
    program.samplersBound = true
    if (!this.cacheState) program.uniformValues?.clear()
    for (const [name, value] of Object.entries(pass.uniforms ?? {})) {
      const loc = this.uniform(program, name)
      if (loc === null) continue
      if (this.cacheState && !this.uniformChanged(program, name, value))
        continue
      if (typeof value === "number") gl.uniform1f(loc, value)
      else if ("int" in value) gl.uniform1i(loc, value.int)
      else if ("ivec4" in value) gl.uniform4i(loc, ...value.ivec4)
      else if (value.length === 2) gl.uniform2f(loc, value[0], value[1])
      else if (value.length === 3)
        gl.uniform3f(loc, value[0], value[1], value[2])
      else gl.uniform4f(loc, value[0], value[1], value[2], value[3])
    }
    const fb = this.framebuffer(pass.output)
    if (fb !== this.boundFb) {
      this.passes++
      this.trace?.push(`${pass.name ?? ""}(${program.samplers.join(",")})`)
      this.boundFb = fb
      gl.bindFramebuffer(gl.FRAMEBUFFER, fb)
    } else if (!this.cacheState) gl.bindFramebuffer(gl.FRAMEBUFFER, fb)
    const rect = pass.rect ?? {
      x: 0,
      y: 0,
      width: outputs[0].width,
      height: outputs[0].height
    }
    if (this.hints & 1) {
      gl.enable(gl.SCISSOR_TEST)
      gl.scissor(rect.x, rect.y, rect.width, rect.height)
    } else gl.disable(gl.SCISSOR_TEST)
    if (this.hints & 2) {
      gl.invalidateSubFramebuffer(
        gl.FRAMEBUFFER,
        outputs.map((_, i) => gl.COLOR_ATTACHMENT0 + i),
        rect.x,
        rect.y,
        rect.width,
        rect.height
      )
    }
    // Bands of rows, each its own draw, so no single draw runs long enough
    // to trip the GPU watchdog: by fragments, and by the reads they make.
    const rowsPerBand = Math.max(
      1,
      Math.floor(
        Math.min(
          this.maxFragmentsPerDraw,
          this.maxReadsPerDraw / Math.max(1, pass.work ?? 1)
        ) / rect.width
      )
    )
    const times = pass.name && this.repeat.has(pass.name) ? this.repeatTimes : 1
    for (let n = 0; n < times; n++) {
      for (let y = rect.y; y < rect.y + rect.height; y += rowsPerBand) {
        const height = Math.min(rowsPerBand, rect.y + rect.height - y)
        gl.viewport(rect.x, y, rect.width, height)
        this.markGpuWork()
        gl.drawArrays(gl.TRIANGLES, 0, 3)
        this.draws++
        if (this.drawTrace) {
          const key = `${pass.name ?? ""}(${program.samplers.join(",")})`
          this.drawTrace.set(key, (this.drawTrace.get(key) ?? 0) + 1)
        }
      }
    }
  }

  private sameUniformArray(a: ReadonlyArray<number>, b: ReadonlyArray<number>) {
    if (a.length !== b.length) return false
    for (let i = 0; i < a.length; i++) if (!Object.is(a[i], b[i])) return false
    return true
  }

  private uniformChanged(
    program: Program,
    name: string,
    value: NonNullable<Pass["uniforms"]>[string]
  ) {
    const values = (program.uniformValues ??= new Map())
    const old = values.get(name)
    if (typeof value === "number") {
      if (typeof old === "number" && Object.is(old, value)) return false
      values.set(name, value)
    } else if ("int" in value) {
      if (
        old &&
        typeof old === "object" &&
        "int" in old &&
        old.int === value.int
      )
        return false
      values.set(name, { int: value.int })
    } else if ("ivec4" in value) {
      if (
        old &&
        typeof old === "object" &&
        "ivec4" in old &&
        this.sameUniformArray(old.ivec4, value.ivec4)
      )
        return false
      values.set(name, { ivec4: [...value.ivec4] })
    } else {
      if (Array.isArray(old) && this.sameUniformArray(old, value)) return false
      values.set(name, [...value] as Uniform)
    }
    return true
  }

  /** Texel reads per async submission; zero keeps ordinary submission. */
  readsPerSlice = 0
  submittedWork = 0
  expectedFrameMs = 16.7
  /** Drain the compute queue periodically so rendering can obtain the GPU. */
  presentationFrames = 2
  private lastPresentationDrain = -Infinity

  private workStartedAt: number | undefined

  private markGpuWork() {
    this.workStartedAt ??= performance.now()
  }

  private submitGpuWork() {
    if (this.workStartedAt === undefined) return
    this.submissionBudget.submitted(this.workStartedAt, performance.now())
    this.workStartedAt = undefined
  }

  private sliceReads = 0
  private sliceMatrixReads = 0
  private sliceCalibrates = true
  private budgets = {
    load: new SubmissionBudget(),
    prompt: new SubmissionBudget(),
    decode: new SubmissionBudget(true)
  }
  private submissionBudget = this.budgets.load
  private phase: "load" | "prompt" | "decode" = "load"
  private phaseAt = performance.timeOrigin + performance.now()
  private lastObservedFrame = -Infinity
  private lastFrameHealthy = false
  private canRecoverPacing = false
  private sampledPromptPacing = false
  private promptFrameBudgetMs = 0

  /** The same feedback controller is used on every GPU. Zero ms fixes the quota. */
  configurePacing(reads: number, targetMs: number) {
    for (const budget of Object.values(this.budgets))
      budget.configure(reads, targetMs, this.presentationFrames)
    this.readsPerSlice = this.submissionBudget.reads
  }

  get activePhase() { return this.phase }

  get groupsDecode() { return this.budgets.decode.recovery }

  allowPacingRecovery() {
    this.canRecoverPacing = true
  }

  /** A slow empty fence makes frequent pacing boundaries expensive. */
  calibratePromptPacing() {
    if (!this.canRecoverPacing || this.sampledPromptPacing ||
      this.expectedFrameMs > 12 || performance.now() - this.lastObservedFrame > 500)
      return
    this.sampledPromptPacing = true
    return this.measurePromptPacing()
  }

  private async measurePromptPacing() {
    let slow = 0
    for (let i = 0; i < 3; i++) {
      const start = performance.now()
      await this.finish()
      if (performance.now() - start >= 3) slow++
    }
    if (slow < 2) return
    // Budget for presentation during prefill on high-refresh displays.
    // A real long stall still triggers the controller's immediate backoff.
    this.promptFrameBudgetMs = 1000 / 75
    this.budgets.prompt.enableRecovery(160 * 2 ** 20)
    this.budgets.decode.enableRecovery()
  }

  setPhase(phase: "load" | "prompt" | "decode") {
    if (this.phase === phase) return
    this.submitGpuWork()
    this.phase = phase
    this.phaseAt = performance.timeOrigin + performance.now()
    this.submissionBudget = this.budgets[phase]
    this.readsPerSlice = this.submissionBudget.reads
    this.sliceReads = this.sliceMatrixReads = 0
    this.sliceCalibrates = true
  }

  resetPacing(presentation = false) {
    for (const budget of Object.values(this.budgets)) {
      budget.reset(presentation)
      budget.heartbeat(this.lastObservedFrame, this.lastFrameHealthy)
    }
    this.submissionBudget = this.budgets[this.phase]
    this.readsPerSlice = this.submissionBudget.reads
    this.sliceReads = this.sliceMatrixReads = 0
    this.sliceCalibrates = true
  }

  get presentationAdvice() {
    return { window: this.budgets.prompt.presentationWindow, max: this.budgets.prompt.presentationMax, relaxation: this.budgets.prompt.relaxationAllowed, cap: this.budgets.prompt.maxPresentationFrames }
  }

  get validatedPresentation() { return this.budgets.prompt.validatedPresentation }

  primePresentation(window: number, reads: number, allowRelaxation = true) {
    this.budgets.prompt.primePresentation(window, reads, allowRelaxation)
    this.readsPerSlice = this.submissionBudget.reads
  }

  get decodeLayers() {
    return this.submissionBudget.layers
  }
  decodeWork() {
    this.submissionBudget.decode()
  }

  /** Main-thread frame feedback takes precedence over fence-age estimation. */
  observeFrames(
    intervalMs: number,
    expectedMs: number,
    frames: number,
    startedAt?: number,
    meanMs?: number
  ) {
    if (
      !Number.isFinite(intervalMs) ||
      !Number.isFinite(expectedMs) ||
      intervalMs < 4 ||
      expectedMs < 4 ||
      frames < 1
    )
      return
    const now = performance.now()
    this.lastObservedFrame = now
    if (this.phase === "prompt")
      expectedMs = Math.max(expectedMs, this.promptFrameBudgetMs || 0)
    this.expectedFrameMs = expectedMs
    this.lastFrameHealthy =
      intervalMs < expectedMs * 1.8 &&
      (meanMs == null || meanMs <= expectedMs * 1.03)
    // Cadence remains fresh across phase changes, but only active work
    // adjusts its quota. An idle phase must not trigger fence fallback.
    for (const budget of Object.values(this.budgets))
      budget.heartbeat(now, this.lastFrameHealthy, expectedMs)
    // Delayed messages describe the original frame window, not the moment
    // the worker can finally receive them.
    if (startedAt == null || startedAt >= this.phaseAt)
      this.submissionBudget.frame(
        intervalMs, expectedMs, frames, now, meanMs,
        startedAt == null ? undefined : startedAt - performance.timeOrigin
      )
    // A high-refresh session can lose occasional callbacks yet have
    // ample presentation headroom. Recover from a proven quota collapse
    // during real inference without retuning healthy queues.
    if (this.canRecoverPacing && this.phase !== "load" &&
      !this.submissionBudget.recovery && this.submissionBudget.quotaCollapsed &&
      expectedMs <= 12 && intervalMs <= 100) {
      this.budgets.prompt.enableRecovery()
      this.budgets.decode.enableRecovery()
    }
    this.readsPerSlice = this.submissionBudget.reads
  }

  /** A compute pass in bounded bands, letting the compositor run between them. */
  async runAsync(pass: Pass, maxRows = Infinity) {
    const output = Array.isArray(pass.output) ? pass.output[0] : pass.output
    const rect = pass.rect ?? {
      x: 0,
      y: 0,
      width: output.width,
      height: output.height
    }
    this.submittedWork += rect.width * rect.height * Math.max(1, pass.work ?? 1)
    if (this.readsPerSlice <= 0) return this.run(pass)
    if (pass.name === "matmul" && rect.height > 8) this.submissionBudget?.work()
    const perRow = rect.width * Math.max(1, pass.work ?? 1)
    this.sliceCalibrates &&= pass.name !== "head" && maxRows === Infinity
    for (let y = rect.y; y < rect.y + rect.height;) {
      const rows = Math.max(
        1,
        Math.floor((this.readsPerSlice - this.sliceReads) / perRow)
      )
      // End paced matrix draws on raster-tile boundaries. Flushing
      // A small projection also starts a new chunk rather than splitting.
      // Arbitrary bands make the GPU store/reload the same boundary
      // tiles repeatedly, unlike the original uninterrupted render pass.
      const quantum = pass.name === "matmul" && rect.height >= 32 ? 32 : 1
      if (
        quantum > 1 &&
        (rows < quantum ||
          (y === rect.y && rect.height * perRow <= this.readsPerSlice / 8 && rows < rect.height)) &&
        this.sliceReads > 0 &&
        this.readsPerSlice >= perRow * quantum
      ) {
        const work =
          this.sliceCalibrates &&
          this.sliceMatrixReads >= this.sliceReads * 0.75
            ? this.sliceReads
            : 0
        this.sliceReads = this.sliceMatrixReads = 0
        this.sliceCalibrates = true
        await this.pace(1, true, work)
        continue
      }
      const bandRows =
        quantum > 1 && rows >= quantum
          ? Math.floor((y + rows) / quantum) * quantum - y
          : rows
      const height = Math.min(bandRows, maxRows, rect.y + rect.height - y)
      this.run({ ...pass, rect: { ...rect, y, height } })
      this.sliceReads += height * perRow
      if (pass.name === "matmul" && rect.height > 8)
        this.sliceMatrixReads += height * perRow
      y += height
      if (
        this.sliceReads + perRow > this.readsPerSlice ||
        maxRows < Infinity ||
        (quantum > 1 && y < rect.y + rect.height)
      ) {
        const work =
          this.sliceCalibrates &&
          this.sliceMatrixReads >= this.sliceReads * 0.75
            ? this.sliceReads
            : 0
        this.sliceReads = this.sliceMatrixReads = 0
        this.sliceCalibrates = true
        await this.pace(1, true, work)
      }
    }
  }

  /**
   * Reads a rectangle of an RGBA32F texture back without stalling the GPU:
   * into a pixel buffer, then waits on a fence before copying it out.
   */
  read(
    texture: Texture,
    rect: { x: number; y: number; width: number; height: number } = {
      x: 0,
      y: 0,
      width: texture.width,
      height: texture.height
    }
  ): Promise<Float32Array> {
    return this.readStart(texture, rect)()
  }

  /** Fences after paced chunks of passes, oldest first (see pace). */
  private paced: Array<{
    sync: WebGLSync
    at: number
    reads: number
    chunks: number
    budget: SubmissionBudget
    layers: number
  }> = []

  /**
   * Ends a chunk of passes, and waits (without blocking) until at most
   * `ahead` chunks are queued on the GPU. Without it, a long prompt queues
   * seconds of work at once, and the browser's GPU process (which draws
   * every page, and the browser itself) stalls until the queue drains.
   */
  async pace(ahead: number, messages = false, reads = 0, layers = 0) {
    const gl = this.gl
    if (messages && reads === 0)
      ahead = Math.min(ahead, this.submissionBudget.ahead)
    if (
      messages &&
      (reads > 0 || this.phase === "load") &&
      this.presentationFrames > 0 &&
      performance.now() - this.lastPresentationDrain >=
        this.expectedFrameMs *
          (this.submissionBudget.presentationWindow || this.presentationFrames)
    ) {
      ahead = 0
      this.lastPresentationDrain = performance.now()
    }
    const sync = gl.fenceSync(gl.SYNC_GPU_COMMANDS_COMPLETE, 0)
    gl.flush()
    this.submitGpuWork()
    if (sync)
      this.paced.push({
        sync,
        at: performance.now(),
        reads,
        chunks: ahead + 1,
        budget: this.submissionBudget,
        layers
      })
    while (this.paced.length > ahead && !this.isLost) {
      const oldest = this.paced[0]
      if (this.fenceReady(oldest.sync)) {
        gl.deleteSync(oldest.sync)
        this.paced.shift()
        const elapsed = performance.now() - oldest.at
        oldest.budget.sample(oldest.reads, elapsed, oldest.chunks)
        if (oldest.layers > 0)
          oldest.budget.layerSample(oldest.layers, elapsed, oldest.chunks)
        this.readsPerSlice = this.submissionBudget.reads
        continue
      }
      if (messages) {
        await this.wait(oldest.sync, oldest.budget, oldest.at)
        continue
      }
      // A page the browser counts as hidden has its timers throttled (a
      // 1 ms wait can take a second, or a minute): then wait by reading a
      // pixel of the last pass's output instead, which blocks until the
      // GPU has done the queue. The queue stays bounded either way.
      if (performance.now() < this.throttledUntil) {
        this.readBack()
        for (const f of this.paced) gl.deleteSync(f.sync)
        this.paced = []
        break
      }
      const t = performance.now()
      await new Promise((resolve) => setTimeout(resolve, 1))
      if (performance.now() - t > 50)
        this.throttledUntil = performance.now() + 2000
    }
  }

  /** Until when pace waits by reading back, not by timers (see pace). */
  private throttledUntil = 0
  /** The last pass's first output, for readBack. */
  private lastOutput: Texture | null = null

  /** Reads a pixel of the last pass's output: returns once the GPU has done it. */
  private readBack() {
    const t = this.lastOutput
    if (this.isLost) return
    const gl = this.gl
    if (!t) {
      gl.finish()
      return
    }
    const float = t.internal === gl.RGBA32F || t.internal === gl.RGBA16F
    const fb = this.framebuffer(t)
    gl.bindFramebuffer(gl.FRAMEBUFFER, fb)
    this.boundFb = fb
    if (float) gl.readPixels(0, 0, 1, 1, gl.RGBA, gl.FLOAT, new Float32Array(4))
    else if (t.internal === gl.RGBA32UI)
      gl.readPixels(0, 0, 1, 1, gl.RGBA_INTEGER, gl.UNSIGNED_INT, new Uint32Array(4))
    else gl.finish()
  }

  /** Flushes issued passes to the GPU, so it starts on them. */
  flush() {
    this.gl.flush()
    this.submitGpuWork()
  }

  /**
   * Issues a read like `read`'s, and returns a function that waits for it:
   * passes issued in between run after the read on the GPU, without the
   * read waiting for them.
   */
  readStart(
    texture: Texture,
    rect: { x: number; y: number; width: number; height: number } = {
      x: 0,
      y: 0,
      width: texture.width,
      height: texture.height
    }
  ): () => Promise<Float32Array> {
    const gl = this.gl
    const out = new Float32Array(rect.width * rect.height * 4)
    const buffer = gl.createBuffer()
    if (!buffer) throw new GpuError("Couldn't create a pixel buffer.")
    this.boundFb = this.framebuffer(texture)
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.boundFb)
    gl.bindBuffer(gl.PIXEL_PACK_BUFFER, buffer)
    gl.bufferData(gl.PIXEL_PACK_BUFFER, out.byteLength, gl.STREAM_READ)
    gl.readPixels(rect.x, rect.y, rect.width, rect.height, gl.RGBA, gl.FLOAT, 0)
    gl.bindBuffer(gl.PIXEL_PACK_BUFFER, null)
    const sync = gl.fenceSync(gl.SYNC_GPU_COMMANDS_COMPLETE, 0)
    gl.flush()
    this.submitGpuWork()
    const budget = this.submissionBudget, at = performance.now()
    return async () => {
      try {
        await this.wait(sync, budget, at)
        gl.bindBuffer(gl.PIXEL_PACK_BUFFER, buffer)
        gl.getBufferSubData(gl.PIXEL_PACK_BUFFER, 0, out)
        gl.bindBuffer(gl.PIXEL_PACK_BUFFER, null)
      } finally {
        if (sync) gl.deleteSync(sync)
        gl.deleteBuffer(buffer)
      }
      return out
    }
  }

  /**
   * Reads a rectangle of an RGBA32F texture back, blocking until the GPU
   * gets there: no fence polling, so less latency (fine in a worker).
   */
  readNow(
    texture: Texture,
    rect = { x: 0, y: 0, width: texture.width, height: texture.height }
  ): Float32Array {
    const gl = this.gl
    const out = new Float32Array(rect.width * rect.height * 4)
    this.boundFb = this.framebuffer(texture)
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.boundFb)
    gl.readPixels(rect.x, rect.y, rect.width, rect.height, gl.RGBA, gl.FLOAT, out)
    this.submitGpuWork()
    return out
  }

  private yielder: MessageChannel | null = null
  private yielded: Array<() => void> = []

  /**
   * Gives the event loop a turn: a message to itself, which isn't clamped
   * to 4 ms like nested timeouts are.
   */
  private yieldOnce() {
    if (!this.yielder) {
      this.yielder = new MessageChannel()
      this.yielder.port1.onmessage = () => this.yielded.shift()?.()
    }
    return new Promise<void>((resolve) => {
      this.yielded.push(resolve)
      this.yielder?.port2.postMessage(null)
    })
  }

  /** Bound active polling as well as GPU work, to avoid heating a CPU core. */
  pollsBeforeSleep = 32
  private polls = 0

  /** A zero-timeout fence check avoids a synchronous parameter-query round trip. */
  private fenceReady(sync: WebGLSync) {
    const gl = this.gl, status = gl.clientWaitSync(sync, 0, 0)
    if (status === gl.WAIT_FAILED) throw new GpuError("Waiting on the GPU failed.")
    return status === gl.ALREADY_SIGNALED || status === gl.CONDITION_SATISFIED
  }

  /** Resolves once the GPU passes `sync`, polling without blocking. */
  private async wait(
    sync: WebGLSync | null,
    budget = this.submissionBudget,
    at = performance.now()
  ) {
    if (this.isLost) throw new GpuError("The GPU context was lost.")
    if (!sync || this.fenceReady(sync)) return
    budget.waiting(at)
    const began = performance.now()
    // Measured slow fences get one high-refresh interval to signal before
    // we add the latency of a timer wake-up.
    const pollGraceMs = this.promptFrameBudgetMs > 0 ? 8 : 2
    try {
      for (let spins = 0; ; spins++) {
        // Spin on messages at first; back off to timers on long waits. With
        // timers throttled (a hidden page: a 1 ms wait can take a second),
        // block on a pixel of the last pass instead, as pace does.
        const sleep =
          this.phase !== "load" &&
          this.pollsBeforeSleep > 0 &&
          performance.now() - began >= pollGraceMs &&
          ++this.polls % this.pollsBeforeSleep === 0
        if (spins < 2000 && !sleep) await this.yieldOnce()
        else if (performance.now() < this.throttledUntil) {
          this.readBack()
          return
        } else {
          const t = performance.now()
          await new Promise((resolve) => setTimeout(resolve, 1))
          if (performance.now() - t > 50)
            this.throttledUntil = performance.now() + 2000
        }
        if (this.isLost) throw new GpuError("The GPU context was lost.")
        if (this.fenceReady(sync)) return
      }
    } finally {
      budget.completed(performance.now())
    }
  }

  /** Waits for every pass issued so far to finish. */
  async finish() {
    const gl = this.gl
    const sync = gl.fenceSync(gl.SYNC_GPU_COMMANDS_COMPLETE, 0)
    gl.flush()
    this.submitGpuWork()
    try {
      await this.wait(sync)
    } finally {
      if (sync) gl.deleteSync(sync)
    }
  }
}
