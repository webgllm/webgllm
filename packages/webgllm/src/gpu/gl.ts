/**
 * A thin layer over WebGL 2 for compute in fragment shaders: textures as
 * tensors, one full-viewport triangle per pass, and async readback.
 *
 * Every pass draws into a texture through a framebuffer, so outputs must be
 * color-renderable: float targets need EXT_color_buffer_float, checked up
 * front along with the texture size the model needs.
 */

export type GpuLimits = {
  maxTextureSize: number
  maxDrawBuffers: number
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
   * Pass names to draw twice, into the same target (no new render pass):
   * for timing what their fragments cost beyond the pass itself.
   */
  repeat = new Set<string>()
  /** How many times `repeat` passes are drawn. */
  repeatTimes = 2
  /** Fragments per draw before a pass is split into bands (GPU watchdog). */
  maxFragmentsPerDraw = 1 << 22

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

  /** A fragment shader's program, compiled once per source. */
  program(source: string): Program {
    const cached = this.programs.get(source)
    if (cached) return cached
    const gl = this.gl
    const program = gl.createProgram()
    if (!program) throw new GpuError("Couldn't create a program.")
    gl.attachShader(program, this.vertexShader)
    gl.attachShader(program, this.shader(gl.FRAGMENT_SHADER, source))
    gl.linkProgram(program)
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
    const entry = { program, uniforms: new Map(), samplers }
    this.programs.set(source, entry)
    return entry
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
    gl.texStorage2D(gl.TEXTURE_2D, 1, internal, width, height)
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST)
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST)
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE)
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE)
    if (data) {
      gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1)
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
    return { id: this.nextId++, tex, width, height, internal, bytes }
  }

  /** Writes `pixels` into a rectangle of a texture. */
  write(
    target: Texture,
    rect: { x: number; y: number; width: number; height: number },
    data: { format: number; type: number; pixels: ArrayBufferView }
  ) {
    const gl = this.gl
    gl.bindTexture(gl.TEXTURE_2D, target.tex)
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
    this.bytes -= texture.bytes
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
    gl.useProgram(program.program)
    let unit = 0
    for (const name of program.samplers) {
      const input = pass.inputs[name]
      if (!input) throw new GpuError(`Pass is missing input ${name}.`)
      if (outputs.some((o) => o.tex === input.tex))
        throw new GpuError(`Pass reads and writes ${name}.`)
      gl.activeTexture(gl.TEXTURE0 + unit)
      gl.bindTexture(gl.TEXTURE_2D, input.tex)
      gl.uniform1i(this.uniform(program, name), unit)
      unit++
    }
    for (const [name, value] of Object.entries(pass.uniforms ?? {})) {
      const loc = this.uniform(program, name)
      if (loc === null) continue
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
    }
    gl.bindFramebuffer(gl.FRAMEBUFFER, fb)
    const rect = pass.rect ?? {
      x: 0,
      y: 0,
      width: outputs[0].width,
      height: outputs[0].height
    }
    // Bands of rows, each its own draw, so no single draw runs long enough
    // to trip the GPU watchdog.
    const rowsPerBand = Math.max(
      1,
      Math.floor(this.maxFragmentsPerDraw / rect.width)
    )
    const times = pass.name && this.repeat.has(pass.name) ? this.repeatTimes : 1
    for (let n = 0; n < times; n++) {
      for (let y = rect.y; y < rect.y + rect.height; y += rowsPerBand) {
        const height = Math.min(rowsPerBand, rect.y + rect.height - y)
        gl.viewport(rect.x, y, rect.width, height)
        gl.drawArrays(gl.TRIANGLES, 0, 3)
        this.draws++
        if (this.drawTrace) {
          const key = `${pass.name ?? ""}(${program.samplers.join(",")})`
          this.drawTrace.set(key, (this.drawTrace.get(key) ?? 0) + 1)
        }
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
  private paced: Array<WebGLSync> = []

  /**
   * Ends a chunk of passes, and waits (without blocking) until at most
   * `ahead` chunks are queued on the GPU. Without it, a long prompt queues
   * seconds of work at once, and the browser's GPU process (which draws
   * every page, and the browser itself) stalls until the queue drains.
   */
  async pace(ahead: number) {
    const gl = this.gl
    const sync = gl.fenceSync(gl.SYNC_GPU_COMMANDS_COMPLETE, 0)
    gl.flush()
    if (sync) this.paced.push(sync)
    while (this.paced.length > ahead && !this.isLost) {
      const oldest = this.paced[0]
      if (gl.getSyncParameter(oldest, gl.SYNC_STATUS) === gl.SIGNALED) {
        gl.deleteSync(oldest)
        this.paced.shift()
      } else await new Promise((resolve) => setTimeout(resolve, 1))
    }
  }

  /** Flushes issued passes to the GPU, so it starts on them. */
  flush() {
    this.gl.flush()
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
    return async () => {
      try {
        await this.wait(sync)
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

  /** Resolves once the GPU passes `sync`, polling without blocking. */
  private async wait(sync: WebGLSync | null) {
    const gl = this.gl
    for (let spins = 0; ; spins++) {
      if (this.isLost) throw new GpuError("The GPU context was lost.")
      const status = sync ? gl.clientWaitSync(sync, 0, 0) : gl.ALREADY_SIGNALED
      if (status === gl.WAIT_FAILED)
        throw new GpuError("Waiting on the GPU failed.")
      if (status !== gl.TIMEOUT_EXPIRED) return
      // Spin on messages at first; back off to timers on long waits.
      if (spins < 2000) await this.yieldOnce()
      else await new Promise((resolve) => setTimeout(resolve, 1))
    }
  }

  /** Waits for every pass issued so far to finish. */
  async finish() {
    const gl = this.gl
    const sync = gl.fenceSync(gl.SYNC_GPU_COMMANDS_COMPLETE, 0)
    gl.flush()
    try {
      await this.wait(sync)
    } finally {
      if (sync) gl.deleteSync(sync)
    }
  }
}
