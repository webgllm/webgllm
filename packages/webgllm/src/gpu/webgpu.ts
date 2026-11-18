/**
 * The engine's GPU on WebGPU: the same textures-as-tensors and passes as
 * gl.ts, with each pass's fragment shader run as a compute shader (see
 * wgsl.ts) that stores into storage textures. Passes go into one compute
 * pass per submission; uniforms into one buffer, at a dynamic offset per
 * dispatch; reads through mapped buffers.
 *
 * Formats: as gl.ts's, with RGBA4 held as r16uint (the same 16 bits; the
 * shaders unpack them). Outputs must be storage formats: canRender says
 * which.
 */

import { PacedGpu } from "./paced.js"
import type { SubmissionBudget } from "./submission-budget.js"
import { GL, GpuError, type GpuLimits, type Pass, type PendingRead, type Program, type Texture } from "./types.js"
import { coopLoop, parseGlsl, toWgsl, type Shader, type Translation } from "./wgsl.js"

type Value = NonNullable<Pass["uniforms"]>[string]

type Variant = {
  t: Translation
  formats: Array<GPUTextureFormat>
  rgba4: Array<string>
  unroll: number
  pipeline: GPUComputePipeline
  layout: GPUBindGroupLayout
  workgroup: [number, number]
  id: number
  /** What made the pipeline: made again, at once, by a pass that can't wait for it (see deferPipelines). */
  descriptor?: GPUComputePipelineDescriptor
}

/**
 * A hand-written compute shader in the pass model (see nativeProgram):
 * binding 0 its uniforms (the pass's rectangle, then `uniforms` in order,
 * 4 bytes each), then `inputs` as textures, then the outputs as storage
 * textures; `groups` sizes the dispatch from the rectangle.
 */
export type NativeSpec = {
  code: string
  inputs: Array<{ name: string; kind: "f" | "u" }>
  uniforms: Array<{ name: string; type: "int" | "float" }>
  groups: (rect: { width: number; height: number }) => [number, number]
}

type NativeProgram = Program & {
  native: NativeSpec
  pipelines: Map<string, { pipeline: GPUComputePipeline; layout: GPUBindGroupLayout; id: number; descriptor?: GPUComputePipelineDescriptor }>
  values: Map<string, Value>
}

type WgProgram = Program & {
  shader: Shader
  /** Whether main has a loop the lanes of a workgroup can share (see wgsl.ts's coopLoop). */
  coop: boolean
  source: string
  variants: Map<string, Variant>
  /** Uniform values: they persist between passes, as a GL program's do. */
  values: Map<string, Value>
}

/** A pass waiting to be merged (see WebGpu.queue). */
type Queued = { v: Variant; inputs: Array<Texture>; outputs: Array<Texture>; block: Uint8Array; rect: { x: number; y: number; width: number; height: number } }

/** A fence: done once the GPU has finished the work submitted before it. */
type Fence = { done: boolean; promise: Promise<void> }

const FORMATS: Record<number, { format: GPUTextureFormat; bytes: number; storage: boolean }> = {
  [GL.RGBA32UI]: { format: "rgba32uint", bytes: 16, storage: true },
  [GL.RGBA32F]: { format: "rgba32float", bytes: 16, storage: true },
  [GL.RGBA16F]: { format: "rgba16float", bytes: 8, storage: true },
  [GL.RGBA8UI]: { format: "rgba8uint", bytes: 4, storage: true },
  [GL.RGBA8I]: { format: "rgba8sint", bytes: 4, storage: true },
  [GL.RGBA8]: { format: "rgba8unorm", bytes: 4, storage: true },
  [GL.RGBA4]: { format: "r16uint", bytes: 2, storage: false },
  [GL.R32F]: { format: "r32float", bytes: 4, storage: true },
  [GL.R32UI]: { format: "r32uint", bytes: 4, storage: true },
  [GL.R8UI]: { format: "r8uint", bytes: 1, storage: false }
}

function formatOf(internal: number) {
  const f = FORMATS[internal]
  if (!f) throw new GpuError(`WebGPU has no format for GL format 0x${internal.toString(16)}.`)
  return f
}

/** WebGPU's flag values (the global namespaces aren't in TypeScript's DOM types, nor in Node). */
const BUFFER = { MAP_READ: 1, COPY_SRC: 4, COPY_DST: 8, UNIFORM: 64, QUERY_RESOLVE: 512 }
const TEXTURE = { COPY_SRC: 1, COPY_DST: 2, TEXTURE_BINDING: 4, STORAGE_BINDING: 8 }
const COMPUTE = 4
const MAP_READ = 1

const align = (n: number, a: number) => Math.ceil(n / a) * a

/** Float32 to IEEE half bits, rounded to nearest (uploads of RGBA/FLOAT data into RGBA16F). */
function toHalves(src: Float32Array): Uint16Array {
  const out = new Uint16Array(src.length)
  const f = new Float32Array(1)
  const u = new Uint32Array(f.buffer)
  for (let i = 0; i < src.length; i++) {
    f[0] = src[i]
    const x = u[0]
    const sign = (x >>> 16) & 0x8000
    const exp = (x >>> 23) & 0xff
    let mant = x & 0x7fffff
    if (exp === 0xff) out[i] = sign | 0x7c00 | (mant ? 0x200 : 0)
    else {
      const e = exp - 127 + 15
      if (e >= 31) out[i] = sign | 0x7c00
      else if (e <= 0) {
        if (e < -10) out[i] = sign
        else {
          mant |= 0x800000
          const shift = 14 - e
          let h = mant >>> shift
          const rest = mant & ((1 << shift) - 1)
          const half = 1 << (shift - 1)
          if (rest > half || (rest === half && h & 1)) h++
          out[i] = sign | h
        }
      } else {
        let h = (e << 10) | (mant >>> 13)
        const rest = mant & 0x1fff
        if (rest > 0x1000 || (rest === 0x1000 && h & 1)) h++
        out[i] = sign | h
      }
    }
  }
  return out
}

export class WebGpu extends PacedGpu<Fence> {
  readonly limits: GpuLimits
  /** Bytes in textures created and not deleted. */
  bytes = 0
  /** Dispatches issued, for profiling. */
  draws = 0
  /** Passes issued, for profiling. */
  passes = 0
  trace: Array<string> | null = null
  drawTrace: Map<string, number> | null = null
  /** Pass names not to run: for timing what the rest costs (results are wrong). */
  skip = new Set<string>()
  /** Kept for the engine's settings; render-area hints mean nothing here. */
  hints = 1
  cacheState = false
  /** (WebGPU clears a texture on the GPU, before its first partial write.) */
  readonly slowZeroFill = false
  repeat = new Set<string>()
  repeatTimes = 2
  /**
   * Texels and texel reads per dispatch before a pass is split into bands
   * (GPU watchdog: see gl.ts). Far higher than a draw's: a dispatch ends
   * where the next begins (Metal runs a compute pass's dispatches one after
   * another), so bands leave the GPU idle at each seam; LFM3's output head
   * in 74 bands took 5 ms a token on an M-series Mac.
   */
  maxFragmentsPerDraw = 1 << 24
  maxReadsPerDraw = 2 ** 32
  salt = ""
  compiled = {
    count: 0,
    ms: 0,
    slowest: [] as Array<[number, string]>,
    times: new Map<string, number>()
  }

  /**
   * When set (and the device has timestamp queries), each pass runs in its
   * own compute pass between timestamps, and its GPU milliseconds add up
   * here by pass name: for profiling (it slows the work down).
   */
  profile: Map<string, { ms: number; n: number }> | null = null
  private querySet: GPUQuerySet | null = null
  private queryNames: Array<string> = []
  /** Profiled submissions not yet read back. */
  private profiling: Array<Promise<void>> = []

  /** The first errors the device reported (validation, memory), for diagnosis. */
  errors: Array<string> = []

  private programs = new Map<string, WgProgram>()
  private nextId = 1
  private nextVariant = 1
  private lost = false
  private oom = false
  private live = new Set<Texture>()
  private encoder: GPUCommandEncoder | null = null
  private computePass: GPUComputePassEncoder | null = null
  /** Buffers and textures to destroy once the submission that uses them is sent. */
  private transient: Array<GPUBuffer> = []
  private doomed: Array<GPUTexture> = []
  /** Uniform blocks of the submission being built (copied to `uniformBuffer` at submit). */
  private ring = new Uint8Array(1 << 20)
  private ringUsed = 0
  private uniformBuffer: GPUBuffer
  private bindGroups = new Map<string, GPUBindGroup>()
  private groupsByTexture = new Map<number, Set<string>>()
  private readonly uniformAlign: number

  /** A WebGPU device with the limits the engine needs, or null where there's no WebGPU. */
  static async create(): Promise<WebGpu | null> {
    const gpu = (globalThis.navigator as Navigator | undefined)?.gpu
    if (!gpu) return null
    const adapter = await gpu.requestAdapter({ powerPreference: "high-performance" })
    if (!adapter) return null
    const want = (name: keyof GPUSupportedLimits) => adapter.limits[name] as number
    const device = await adapter.requestDevice({
      // (Timestamps only for profiling: see `profile`.)
      requiredFeatures: (["timestamp-query", "subgroups"] as const).filter((f) => adapter.features.has(f) && (f !== "subgroups" || WebGpu.useSubgroups)),
      requiredLimits: {
        maxTextureDimension2D: want("maxTextureDimension2D"),
        maxStorageTexturesPerShaderStage: want("maxStorageTexturesPerShaderStage"),
        maxSampledTexturesPerShaderStage: want("maxSampledTexturesPerShaderStage"),
        maxBufferSize: want("maxBufferSize")
      }
    })
    const info = adapter.info
    return new WebGpu(device, `WebGPU ${[info.vendor, info.architecture, info.device, info.description].filter(Boolean).join(" ")}`, info)
  }

  /** Lanes sum by subgroup shuffles where the device can (experiments: false keeps workgroup memory). */
  static useSubgroups = true

  constructor(readonly device: GPUDevice, renderer: string, info?: GPUAdapterInfo) {
    super()
    this.limits = {
      maxTextureSize: device.limits.maxTextureDimension2D,
      maxDrawBuffers: device.limits.maxStorageTexturesPerShaderStage,
      maxTextureUnits: device.limits.maxSampledTexturesPerShaderStage,
      renderer
    }
    this.uniformAlign = device.limits.minUniformBufferOffsetAlignment
    const min = (info as GPUAdapterInfo & { subgroupMinSize?: number } | undefined)?.subgroupMinSize ?? 0
    this.subgroups = WebGpu.useSubgroups && device.features.has("subgroups") && min >= 2
    this.subgroupSize = Math.min(32, 2 ** Math.floor(Math.log2(Math.max(1, min))))
    this.uniformBuffer = device.createBuffer({
      size: this.ring.byteLength,
      usage: BUFFER.UNIFORM | BUFFER.COPY_DST
    })
    void device.lost.then(() => {
      this.lost = true
    })
    device.addEventListener("uncapturederror", (event) => {
      const error = (event as GPUUncapturedErrorEvent).error
      if (error instanceof GPUOutOfMemoryError) this.oom = true
      if (this.errors.length < 8) this.errors.push(error.message)
      console.error(`WebGPU: ${error.message}`)
    })
  }

  get isLost() {
    return this.lost
  }

  // ------------------------------------------------------------ programs

  program(source: string): WgProgram {
    const cached = this.programs.get(source)
    if (cached) return cached
    const started = performance.now()
    const shader = parseGlsl(source)
    const program: WgProgram = {
      shader,
      coop: coopLoop(shader) !== null,
      source,
      variants: new Map(),
      values: new Map(),
      // (Every sampler; a variant binds the ones main reads.)
      samplers: shader.samplers.map((s) => s.name)
    }
    this.programs.set(source, program)
    this.count(source, performance.now() - started)
    return program
  }

  private count(source: string, ms: number) {
    this.compiled.count++
    this.compiled.ms += ms
    this.compiled.times.set(source, ms)
    const tag = source.split("\n").find((l) => l.startsWith("uniform")) ?? ""
    this.compiled.slowest.push([Math.round(ms), `${tag.slice(0, 40)} ${source.length}`])
    this.compiled.slowest.sort((a, b) => b[0] - a[0])
    this.compiled.slowest.length = Math.min(this.compiled.slowest.length, 8)
  }

  /**
   * Threads a small pass with a K loop is spread over (0: a thread a
   * texel): see wgsl.ts's coopLoop, and TUNING.lanes.
   */
  coopThreads = 0
  /** Whether passes can share a texel's K loop among threads (coopThreads). */
  readonly sharesLoops = true
  /**
   * Passes cost little: a dependent dispatch about 4 µs on an iPad Pro M4,
   * against a WebGL render pass's 38 (chain.html), so chunked attention
   * pays at short contexts too (see Engine.attention).
   */
  readonly cheapPasses = true
  /** Whether lanes sum by subgroup shuffles (see wgsl.ts's Variant.subgroups), and the fewest lanes a subgroup has. */
  private subgroups: boolean
  private subgroupSize: number
  /** Profile whole compute passes (GPU busy time) rather than each pass in its own (see profile). */
  static coarse = false
  /** What a pass whose pipeline is still compiling dispatches to (see deferPipelines). */
  private static noPass = { setBindGroup() {}, dispatchWorkgroups() {} } as unknown as GPUComputePassEncoder

  /**
   * While set, new pipelines compile in the background
   * (createComputePipelineAsync), several at once, and the passes that
   * need them don't run: for a warm-up, whose results are thrown away,
   * where each pipeline compiled when first used, one after another (a
   * Pixel 10 Pro's took 2.2 s). settlePipelines waits for them.
   */
  deferPipelines = false
  private pendingPipelines: Array<Promise<unknown>> = []

  /** Whether pipelines deferPipelines asked for are still compiling. */
  get compiling() {
    return this.pendingPipelines.length > 0
  }

  /**
   * `x`'s pipeline: made at once if it's still compiling in the background
   * and a pass needs it now (outside deferPipelines, where it can't wait).
   */
  private pipelineOf(x: { pipeline: GPUComputePipeline; descriptor?: GPUComputePipelineDescriptor }): GPUComputePipeline | undefined {
    if (!x.pipeline && !this.deferPipelines && x.descriptor) x.pipeline = this.device.createComputePipeline(x.descriptor)
    return x.pipeline
  }

  /** Waits for the pipelines deferPipelines left compiling. */
  async settlePipelines() {
    const pending = this.pendingPipelines
    this.pendingPipelines = []
    await Promise.all(pending)
  }

  /** A compute pipeline: made now, or (deferPipelines) in the background, handed to `set` when it's done. */
  private makePipeline(descriptor: GPUComputePipelineDescriptor, set: (p: GPUComputePipeline) => void): GPUComputePipeline | undefined {
    if (!this.deferPipelines) return this.device.createComputePipeline(descriptor)
    this.pendingPipelines.push(
      this.device.createComputePipelineAsync(descriptor).then(set, (error: Error) => {
        this.errors.push(`pipeline: ${error.message}`)
        set(this.device.createComputePipeline(descriptor))
      })
    )
    return undefined
  }
  /** A workgroup shape for every pass (experiments). */
  static workgroup: [number, number] | null = null

  private variant(program: WgProgram, outputs: Array<Texture>, inputs: Record<string, Texture>, width: number, height: number): Variant {
    const formats = outputs.map((o) => formatOf(o.internal).format)
    const rgba4 = program.samplers.filter((s) => inputs[s]?.internal === GL.RGBA4)
    // Rows of 64 for a few rows (a token's), 16 × 4 above that, and 8 × 8
    // for a prompt's tall passes: an iPad Pro M4 read LFM3 prompts 7%
    // faster than at 16 × 4, a Pixel 10 Pro 35%; but the Pixel's output
    // head (a token's, 64 rows) took 11 ms more a token at 8 × 8.
    const workgroup: [number, number] = WebGpu.workgroup ??
      (height < 4 ? [64, 1] : height >= 32 && this.phase === "prompt" ? [8, 8] : [16, 4])
    // A small pass with a K loop shares each texel's loop among lanes, for
    // about coopThreads threads in all.
    const texels = width * height
    const lanes = program.coop && this.coopThreads > 0 && texels * 2 <= this.coopThreads
      ? Math.min(this.subgroups ? this.subgroupSize : 32, 2 ** Math.floor(Math.log2(this.coopThreads / texels)))
      : 1
    const unroll = this.phase === "prompt" ? 0 : WebGpu.unroll
    const key = `${formats.join(",")}|${rgba4.join(",")}|${workgroup.join("x")}|${lanes}|${unroll}`
    let v = program.variants.get(key)
    if (v) return v
    const started = performance.now()
    const t = toWgsl(program.shader, { outputs: formats, rgba4, workgroup, lanes, subgroups: this.subgroups, unroll })
    const device = this.device
    const entries: Array<GPUBindGroupLayoutEntry> = [
      { binding: 0, visibility: COMPUTE, buffer: { type: "uniform", hasDynamicOffset: true, minBindingSize: t.uniformBytes } }
    ]
    t.samplers.forEach((s, i) =>
      entries.push({
        binding: i + 1,
        visibility: COMPUTE,
        texture: { sampleType: s.kind === "u" ? "uint" : s.kind === "i" ? "sint" : "unfilterable-float" }
      })
    )
    formats.forEach((format, i) =>
      entries.push({
        binding: t.samplers.length + 1 + i,
        visibility: COMPUTE,
        storageTexture: { access: "write-only", format }
      })
    )
    const layout = device.createBindGroupLayout({ entries })
    const module = device.createShaderModule({ code: t.code })
    const descriptor = {
      layout: device.createPipelineLayout({ bindGroupLayouts: [layout] }),
      compute: { module, entryPoint: "main" }
    }
    const made: Variant = { t, formats, rgba4, unroll, pipeline: undefined as unknown as GPUComputePipeline, layout, workgroup: t.workgroup, id: this.nextVariant++, descriptor }
    const pipeline = this.makePipeline(descriptor, (p) => { made.pipeline ??= p })
    if (pipeline) made.pipeline = pipeline
    v = made
    program.variants.set(key, v)
    this.count(`${key}\n${program.source}`, performance.now() - started)
    return v
  }

  /** A hand-written WGSL pass (see NativeSpec), cached by its code. */
  nativeProgram(spec: NativeSpec): NativeProgram {
    const cached = this.natives.get(spec.code)
    if (cached) return cached
    const program: NativeProgram = { native: spec, pipelines: new Map(), values: new Map(), samplers: spec.inputs.map((i) => i.name) }
    this.natives.set(spec.code, program)
    return program
  }

  private natives = new Map<string, NativeProgram>()

  /** The program that last wrote each texture (a write from the CPU clears it). */
  private writers = new Map<number, Program>()

  /** The program whose pass last wrote `texture`, if a pass did since its last upload. */
  lastWriter(texture: Texture): Program | undefined {
    return this.writers.get(texture.id)
  }

  /** Runs a native pass: one dispatch, its groups from the rectangle. */
  private runNative(pass: Pass, program: NativeProgram) {
    this.flushQueue()
    const spec = program.native
    const outputs = Array.isArray(pass.output) ? pass.output : [pass.output]
    const rect = pass.rect ?? { x: 0, y: 0, width: outputs[0].width, height: outputs[0].height }
    const formats = outputs.map((o) => formatOf(o.internal).format)
    const key = formats.join(",")
    let p = program.pipelines.get(key)
    if (!p) {
      const started = performance.now()
      const entries: Array<GPUBindGroupLayoutEntry> = [
        { binding: 0, visibility: COMPUTE, buffer: { type: "uniform", hasDynamicOffset: true, minBindingSize: this.nativeBytes(spec) } }
      ]
      spec.inputs.forEach((s, i) =>
        entries.push({ binding: i + 1, visibility: COMPUTE, texture: { sampleType: s.kind === "u" ? "uint" : "unfilterable-float" } }))
      formats.forEach((format, i) =>
        entries.push({ binding: spec.inputs.length + 1 + i, visibility: COMPUTE, storageTexture: { access: "write-only", format } }))
      const layout = this.device.createBindGroupLayout({ entries })
      const descriptor = {
        layout: this.device.createPipelineLayout({ bindGroupLayouts: [layout] }),
        compute: { module: this.device.createShaderModule({ code: spec.code }), entryPoint: "main" }
      }
      const made = { pipeline: undefined as unknown as GPUComputePipeline, layout, id: this.nextVariant++, descriptor }
      const pipeline = this.makePipeline(descriptor, (q) => { made.pipeline ??= q })
      if (pipeline) made.pipeline = pipeline
      p = made
      program.pipelines.set(key, p)
      this.count(`native ${spec.code.slice(0, 40)}`, performance.now() - started)
    }
    const inputs = spec.inputs.map(({ name }) => {
      const t = pass.inputs[name]
      if (!t) throw new GpuError(`Pass is missing input ${name}.`)
      return t
    })
    for (const [name, value] of Object.entries(pass.uniforms ?? {})) program.values.set(name, value)
    for (const t of inputs) this.touched.set(t.id, this.generation)
    for (const t of outputs) {
      this.touched.set(t.id, this.generation)
      this.writers.set(t.id, program)
    }
    const key2 = `n${p.id}|${inputs.map((t) => t.id).join(",")}|${outputs.map((t) => t.id).join(",")}`
    let group = this.bindGroups.get(key2)
    if (!group) {
      const entries: Array<GPUBindGroupEntry> = [{ binding: 0, resource: { buffer: this.uniformBuffer, size: this.nativeBytes(spec) } }]
      inputs.forEach((t, i) => entries.push({ binding: i + 1, resource: (t.tex as GPUTexture).createView() }))
      outputs.forEach((t, i) => entries.push({ binding: inputs.length + 1 + i, resource: (t.tex as GPUTexture).createView() }))
      group = this.device.createBindGroup({ layout: p.layout, entries })
      this.bindGroups.set(key2, group)
      for (const t of [...inputs, ...outputs]) {
        let keys = this.groupsByTexture.get(t.id)
        if (!keys) this.groupsByTexture.set(t.id, (keys = new Set()))
        keys.add(key2)
      }
    }
    const block = new ArrayBuffer(this.nativeBytes(spec))
    const i32 = new Int32Array(block), f32 = new Float32Array(block)
    spec.uniforms.forEach(({ name, type }, i) => {
      const v = program.values.get(name)
      const x = typeof v === "number" ? v : v && "int" in v ? v.int : 0
      if (type === "int") i32[4 + i] = x
      else f32[4 + i] = x
    })
    const offset = this.pushUniforms(new Uint8Array(block), rect.x, rect.y, rect.width, rect.height)
    if (this.profile && !WebGpu.coarse && this.device.features.has("timestamp-query")) this.timed(`native ${pass.name ?? ""} ${rect.width}x${rect.height}`)
    const cpass = this.computePassFor(this.pipelineOf(p))
    cpass.setBindGroup(0, group, [offset])
    this.markGpuWork()
    cpass.dispatchWorkgroups(...spec.groups(rect))
    this.passes++
    this.draws++
  }

  private nativeBytes(spec: NativeSpec) {
    return align(16 + 4 * spec.uniforms.length, 16)
  }

  // ------------------------------------------------------------ textures

  texture(
    internal: number,
    width: number,
    height: number,
    data?: { format: number; type: number; pixels: ArrayBufferView }
  ): Texture {
    const max = this.limits.maxTextureSize
    if (width > max || height > max || width < 1 || height < 1)
      throw new GpuError(`A ${width}×${height} texture is past this GPU's limit of ${max}.`)
    const f = formatOf(internal)
    this.device.pushErrorScope("out-of-memory")
    const tex = this.device.createTexture({
      size: { width, height },
      format: f.format,
      usage:
        TEXTURE.TEXTURE_BINDING |
        TEXTURE.COPY_DST |
        TEXTURE.COPY_SRC |
        (f.storage ? TEXTURE.STORAGE_BINDING : 0)
    })
    void this.device.popErrorScope().then((e) => {
      if (e) this.oom = true
    })
    const bytes = width * height * f.bytes
    this.bytes += bytes
    const texture: Texture = { id: this.nextId++, tex, width, height, internal, bytes }
    this.live.add(texture)
    // (A new texture: nothing issued reads it, so the queue's write may go first.)
    if (data) this.upload(texture, { x: 0, y: 0, width, height }, data, true)
    return texture
  }

  /** Writes `pixels` into a rectangle of a texture, after the passes issued so far. */
  write(
    target: Texture,
    rect: { x: number; y: number; width: number; height: number },
    data: { format: number; type: number; pixels: ArrayBufferView }
  ) {
    // The queue's write lands before the commands not yet submitted: fine
    // unless one of them uses the texture (it would see the new contents).
    // Otherwise staged behind them, which ends the compute pass.
    this.upload(target, rect, data, !this.encoder || this.touched.get(target.id) !== this.generation)
  }

  /** Submissions sent: the commands being built are generation `generation`. */
  private generation = 0
  /** The generation each texture was last read or written by a command in. */
  private touched = new Map<number, number>()

  private upload(
    target: Texture,
    rect: { x: number; y: number; width: number; height: number },
    data: { format: number; type: number; pixels: ArrayBufferView },
    direct: boolean
  ) {
    this.writers.delete(target.id)
    const f = formatOf(target.internal)
    let pixels = data.pixels
    if (target.internal === GL.RGBA16F && data.type === GL.FLOAT)
      pixels = toHalves(new Float32Array(pixels.buffer, pixels.byteOffset, rect.width * rect.height * 4))
    const row = rect.width * f.bytes
    const src = new Uint8Array(pixels.buffer, pixels.byteOffset, row * rect.height)
    const tex = target.tex as GPUTexture
    if (direct) {
      this.device.queue.writeTexture(
        { texture: tex, origin: { x: rect.x, y: rect.y } },
        src,
        { bytesPerRow: row, rowsPerImage: rect.height },
        { width: rect.width, height: rect.height }
      )
      return
    }
    // Behind the passes issued: staged in a buffer, copied in the same submission.
    const padded = align(row, 256)
    const buffer = this.device.createBuffer({
      size: padded * rect.height,
      usage: BUFFER.COPY_SRC,
      mappedAtCreation: true
    })
    const mapped = new Uint8Array(buffer.getMappedRange())
    if (padded === row) mapped.set(src)
    else for (let y = 0; y < rect.height; y++) mapped.set(src.subarray(y * row, (y + 1) * row), y * padded)
    buffer.unmap()
    this.endComputePass()
    this.touched.set(target.id, this.generation)
    this.commands().copyBufferToTexture(
      { buffer, bytesPerRow: padded, rowsPerImage: rect.height },
      { texture: tex, origin: { x: rect.x, y: rect.y } },
      { width: rect.width, height: rect.height }
    )
    this.transient.push(buffer)
  }

  delete(texture: Texture) {
    for (const key of this.groupsByTexture.get(texture.id) ?? []) this.bindGroups.delete(key)
    this.touched.delete(texture.id)
    this.writers.delete(texture.id)
    this.groupsByTexture.delete(texture.id)
    // (Passes issued and not yet submitted may read it: destroyed after they're sent.)
    if (this.encoder) this.doomed.push(texture.tex as GPUTexture)
    else (texture.tex as GPUTexture).destroy()
    if (this.live.delete(texture)) this.bytes -= texture.bytes
  }

  /** Whether passes can write textures of these formats at once (one output each). */
  canRender(internals: Array<number>) {
    return (
      internals.length <= this.limits.maxDrawBuffers &&
      internals.every((i) => FORMATS[i]?.storage === true)
    )
  }

  outOfMemory() {
    return this.oom || this.lost
  }

  /** Fills `texture` with zeros on the GPU (a fresh texture already is). */
  zero(texture: Texture) {
    if (!FORMATS[texture.internal]?.storage) return
    const integer = /uint|sint/.test(formatOf(texture.internal).format)
    const type = integer ? (texture.internal === GL.RGBA8I ? "ivec4" : "uvec4") : "vec4"
    const program = this.program(`#version 300 es\nprecision highp float;\nout ${type} o;\nvoid main() { o = ${type}(0); }`)
    this.run({ program, inputs: {}, output: texture })
  }

  // ------------------------------------------------------------ passes

  private commands() {
    return (this.encoder ??= this.device.createCommandEncoder())
  }

  private endComputePass() {
    this.flushQueue()
    this.computePass?.end()
    this.computePass = null
    this.boundPipeline = null
  }

  private boundPipeline: GPUComputePipeline | null = null

  /** Starts a compute pass of its own between two timestamps, for `name` (see profile). */
  private timed(name: string) {
    this.endComputePass()
    if (this.queryNames.length >= 2048) this.submit()
    this.querySet ??= this.device.createQuerySet({ type: "timestamp", count: 4096 })
    const k = this.queryNames.length
    this.queryNames.push(name)
    this.computePass = this.commands().beginComputePass({
      timestampWrites: { querySet: this.querySet, beginningOfPassWriteIndex: 2 * k, endOfPassWriteIndex: 2 * k + 1 }
    })
  }

  /** Resolves the submission's timestamps (before it's submitted): what reads them into `profile` once it is. */
  private resolveTimestamps(): (() => void) | undefined {
    const n = this.queryNames.length
    if (!n || !this.querySet || !this.encoder) return
    const names = this.queryNames
    this.queryNames = []
    const resolved = this.device.createBuffer({ size: 16 * n, usage: BUFFER.QUERY_RESOLVE | BUFFER.COPY_SRC })
    const read = this.device.createBuffer({ size: 16 * n, usage: BUFFER.MAP_READ | BUFFER.COPY_DST })
    this.encoder.resolveQuerySet(this.querySet, 0, 2 * n, resolved, 0)
    this.encoder.copyBufferToBuffer(resolved, 0, read, 0, 16 * n)
    this.transient.push(resolved)
    return () => this.profiling.push(
      read.mapAsync(MAP_READ).then(() => {
        const t = new BigUint64Array(read.getMappedRange())
        names.forEach((name, k) => {
          const e = this.profile?.get(name) ?? { ms: 0, n: 0 }
          e.ms += Number(t[2 * k + 1] - t[2 * k]) / 1e6
          e.n++
          this.profile?.set(name, e)
        })
        read.destroy()
      })
    )
  }

  /** Waits for the profiled submissions' timestamps to arrive in `profile`. */
  async profiled() {
    this.flush()
    // (Bounded: a browser that never maps some of them still reports the rest.)
    const pending = this.profiling.length
    let settled = 0
    this.profiling.forEach((p) => void p.then(() => settled++, () => settled++))
    await Promise.race([Promise.allSettled(this.profiling), new Promise((r) => setTimeout(r, 3000))])
    if (settled < pending) this.errors.push(`${pending - settled} of ${pending} profiled submissions never reported`)
    this.profiling = []
    return this.profile
  }

  /** The open compute pass, with this pipeline set. */
  private computePassFor(pipeline: GPUComputePipeline | undefined) {
    // (A pipeline still compiling while pipelines are deferred: nothing runs.)
    if (!pipeline) return WebGpu.noPass
    if (!this.computePass && this.profile && WebGpu.coarse && this.device.features.has("timestamp-query")) this.timed("busy")
    const cpass = (this.computePass ??= this.commands().beginComputePass())
    if (this.boundPipeline !== pipeline) {
      cpass.setPipeline(pipeline)
      this.boundPipeline = pipeline
    }
    return cpass
  }

  /** Runs a pass: a dispatch over the output rectangle, in bands if it's big. */
  run(pass: Pass) {
    if (this.lost) throw new GpuError("The GPU device was lost.")
    if (pass.name && this.skip.has(pass.name)) return
    if ("native" in pass.program) return this.runNative(pass, pass.program as NativeProgram)
    const program = pass.program as WgProgram
    const outputs = Array.isArray(pass.output) ? pass.output : [pass.output]
    const rect = pass.rect ?? { x: 0, y: 0, width: outputs[0].width, height: outputs[0].height }
    if (rect.width <= 0 || rect.height <= 0) return
    const v = this.variant(program, outputs, pass.inputs, rect.width, rect.height)
    if (v.t.samplers.length > this.limits.maxTextureUnits)
      throw new GpuError(`This GPU reads ${this.limits.maxTextureUnits} textures in a pass; one needs ${v.t.samplers.length}.`)
    const inputs = v.t.samplers.map(({ name }) => {
      const input = pass.inputs[name]
      if (!input) throw new GpuError(`Pass is missing input ${name}.`)
      if (outputs.some((o) => o.tex === input.tex)) throw new GpuError(`Pass reads and writes ${name}.`)
      return input
    })
    for (const [name, value] of Object.entries(pass.uniforms ?? {})) program.values.set(name, value)
    for (const t of inputs) this.touched.set(t.id, this.generation)
    for (const t of outputs) {
      this.touched.set(t.id, this.generation)
      this.writers.set(t.id, program)
    }
    const block = this.uniformBlock(program, v)
    this.passes++
    this.trace?.push(`${pass.name ?? ""}(${v.t.samplers.map((s) => s.name).join(",")})`)
    const rowsPerBand = Math.max(
      1,
      Math.floor(Math.min(this.maxFragmentsPerDraw, this.maxReadsPerDraw / Math.max(1, pass.work ?? 1)) / rect.width)
    )
    const times = pass.name && this.repeat.has(pass.name) ? this.repeatTimes : 1
    // A small pass waits to be merged with the next ones (see queue).
    if (
      WebGpu.merge && this.phase === "decode" && times === 1 && rowsPerBand >= rect.height && v.t.lanes === 1 &&
      rect.width * rect.height <= 65536 && !(this.profile && !WebGpu.coarse)
    ) {
      this.queue({ v, inputs, outputs, block, rect })
      return
    }
    this.flushQueue()
    const group = this.bindGroup(v, inputs, outputs)
    const [wx, wy] = v.workgroup
    for (let n = 0; n < times; n++) {
      for (let y = rect.y; y < rect.y + rect.height; y += rowsPerBand) {
        const height = Math.min(rowsPerBand, rect.y + rect.height - y)
        const offset = this.pushUniforms(block, rect.x, y, rect.width, height)
        if (this.profile && !WebGpu.coarse && this.device.features.has("timestamp-query")) this.timed(`${pass.name ?? ""} ${rect.width}x${height}`)
        const cpass = this.computePassFor(this.pipelineOf(v))
        cpass.setBindGroup(0, group, [offset])
        this.markGpuWork()
        if (v.t.lanes > 1) cpass.dispatchWorkgroups(Math.ceil(rect.width / wy), height)
        else cpass.dispatchWorkgroups(Math.ceil(rect.width / wx), Math.ceil(height / wy))
        this.draws++
        if (this.drawTrace) {
          const key = `${pass.name ?? ""}`
          this.drawTrace.set(key, (this.drawTrace.get(key) ?? 0) + 1)
        }
      }
    }
  }

  /** Merge small passes into one dispatch (see queue). */
  static merge = true
  /**
   * A token's passes have constant loops unrolled up to this many
   * iterations (see wgsl.ts's Variant.unroll). Off: on an iPad Pro M4 it
   * made no difference to writing, and read prompts 5% slower.
   */
  static unroll = 0
  /** Small passes not yet dispatched: they write disjoint regions of the same outputs and read none of them. */
  private queued: Array<Queued> = []
  /** Merged shaders by their members' variants: null while one compiles (see merged). */
  private mergedPipelines = new Map<string, { pipeline: GPUComputePipeline; layout: GPUBindGroupLayout; id: number } | null>()

  /**
   * Queues a small pass: consecutive ones that write disjoint rectangles of
   * the same outputs, and read none of them, go to the GPU as one dispatch
   * (a shader running each one's main over its rectangle). The kernels
   * draw an add's sums, residual and scaled input as three such regions:
   * three dependent dispatches where one does, and dispatches here don't
   * overlap the way a WebGL frame's draws do.
   */
  private queue(q: Queued) {
    const g = this.queued
    if (g.length) {
      const first = g[0]
      const same = first.outputs.length === q.outputs.length && first.outputs.every((t, i) => t === q.outputs[i])
      const reads = same && q.inputs.some((t) => q.outputs.includes(t))
      const overlaps = g.some((x) =>
        x.rect.x < q.rect.x + q.rect.width && q.rect.x < x.rect.x + x.rect.width &&
        x.rect.y < q.rect.y + q.rect.height && q.rect.y < x.rect.y + x.rect.height)
      const textures = g.reduce((n, x) => n + x.inputs.length, q.inputs.length)
      if (!same || reads || overlaps || g.length >= 8 || textures > this.limits.maxTextureUnits) this.flushQueue()
    }
    this.queued.push(q)
  }

  /** Dispatches the queued passes. */
  private flushQueue() {
    const g = this.queued
    if (!g.length) return
    this.queued = []
    const bytes = g.reduce((n, q) => n + align(q.block.byteLength, this.uniformAlign), 0)
    if (align(this.ringUsed, this.uniformAlign) + bytes > this.ring.byteLength) this.submit()
    let merged: { pipeline: GPUComputePipeline; layout: GPUBindGroupLayout; id: number } | null = null
    if (g.length > 1)
      try {
        merged = this.merged(g)
      } catch (error) {
        this.errors.push(`merge: ${(error as Error).message}`)
      }
    if (!merged) {
      for (const q of g) {
        const offset = this.pushUniforms(q.block, q.rect.x, q.rect.y, q.rect.width, q.rect.height)
        const cpass = this.computePassFor(this.pipelineOf(q.v))
        cpass.setBindGroup(0, this.bindGroup(q.v, q.inputs, q.outputs), [offset])
        this.markGpuWork()
        const [wx, wy] = q.v.workgroup
        cpass.dispatchWorkgroups(Math.ceil(q.rect.width / wx), Math.ceil(q.rect.height / wy))
        this.draws++
      }
      return
    }
    const offsets = g.map((q) => this.pushUniforms(q.block, q.rect.x, q.rect.y, q.rect.width, q.rect.height))
    const key = `m${merged.id}|${g.map((q) => q.inputs.map((t) => t.id).join(",")).join("|")}|${g[0].outputs.map((t) => t.id).join(",")}`
    let group = this.bindGroups.get(key)
    if (!group) {
      const entries: Array<GPUBindGroupEntry> = []
      let binding = 0
      for (const q of g) {
        entries.push({ binding: binding++, resource: { buffer: this.uniformBuffer, size: q.v.t.uniformBytes } })
        for (const t of q.inputs) entries.push({ binding: binding++, resource: (t.tex as GPUTexture).createView() })
      }
      for (const t of g[0].outputs) entries.push({ binding: binding++, resource: (t.tex as GPUTexture).createView() })
      group = this.device.createBindGroup({ layout: merged.layout, entries })
      this.bindGroups.set(key, group)
      for (const t of [...g.flatMap((q) => q.inputs), ...g[0].outputs]) {
        let keys = this.groupsByTexture.get(t.id)
        if (!keys) this.groupsByTexture.set(t.id, (keys = new Set()))
        keys.add(key)
      }
    }
    const cpass = this.computePassFor(merged.pipeline)
    cpass.setBindGroup(0, group, offsets)
    this.markGpuWork()
    cpass.dispatchWorkgroups(g.reduce((n, q) => n + Math.ceil((q.rect.width * q.rect.height) / 64), 0))
    this.draws++
  }

  /**
   * The shader running each queued pass's main over its rectangle, in turn
   * by workgroup; null until it has compiled, in the background: the
   * passes go one by one meanwhile (compiled when first needed, a Pixel 10
   * Pro's load took 1.4 s longer).
   */
  private merged(g: Array<Queued>) {
    const key = g.map((q) => q.v.id).join(",")
    const m = this.mergedPipelines.get(key)
    if (m !== undefined) return m
    // (Null stays where it can't be merged: its passes go one by one.)
    this.mergedPipelines.set(key, null)
    const started = performance.now()
    const lines: Array<string> = []
    const entries: Array<GPUBindGroupLayoutEntry> = []
    const calls: Array<string> = []
    let binding = 0
    g.forEach((q, i) => {
      const program = this.programOf(q.v)
      const t = toWgsl(program.shader, { outputs: q.v.formats, rgba4: q.v.rgba4, workgroup: [64, 1], unroll: q.v.unroll, embed: { prefix: `m${i}_`, binding } })
      const e = t.embedded!
      lines.push(t.code)
      entries.push({ binding, visibility: COMPUTE, buffer: { type: "uniform", hasDynamicOffset: true, minBindingSize: t.uniformBytes } })
      t.samplers.forEach((s, j) =>
        entries.push({ binding: binding + 1 + j, visibility: COMPUTE, texture: { sampleType: s.kind === "u" ? "uint" : s.kind === "i" ? "sint" : "unfilterable-float" } }))
      binding += 1 + t.samplers.length
      const U = e.uniforms
      calls.push(
        `  {`,
        `    let n = ${U}.rect.z * ${U}.rect.w;`,
        `    let groups = (n + 63u) / 64u;`,
        `    if (g < groups) {`,
        `      let i = g * 64u + l;`,
        `      if (i >= n) { return; }`,
        `      let p = vec2<u32>(${U}.rect.x + i % ${U}.rect.z, ${U}.rect.y + i / ${U}.rect.z);`,
        `      fc = vec4<f32>(vec2<f32>(p) + 0.5, 0.0, 1.0);`,
        `      ${e.main}();`,
        ...e.outputs.map((o, k) => `      textureStore(out${k}, p, ${o.name});`),
        `      return;`,
        `    }`,
        `    g -= groups;`,
        `  }`
      )
    })
    g[0].v.formats.forEach((format, k) => {
      lines.push(`@group(0) @binding(${binding + k}) var out${k}: texture_storage_2d<${format}, write>;`)
      entries.push({ binding: binding + k, visibility: COMPUTE, storageTexture: { access: "write-only", format } })
    })
    lines.push(
      `var<private> fc: vec4<f32>;`,
      `@compute @workgroup_size(64)`,
      `fn main(@builtin(workgroup_id) w: vec3<u32>, @builtin(local_invocation_index) l: u32) {`,
      `  var g = w.x;`,
      ...calls,
      `}`
    )
    const layout = this.device.createBindGroupLayout({ entries })
    const id = this.nextVariant++
    void this.device
      .createComputePipelineAsync({
        layout: this.device.createPipelineLayout({ bindGroupLayouts: [layout] }),
        compute: { module: this.device.createShaderModule({ code: lines.join("\n") }), entryPoint: "main" }
      })
      .then(
        (pipeline) => { if (!this.lost) this.mergedPipelines.set(key, { pipeline, layout, id }) },
        (error: Error) => this.errors.push(`merge: ${error.message}`)
      )
    this.count(`merged ${key}`, performance.now() - started)
    return null
  }

  /** The program a variant was made for. */
  private programOf(v: Variant) {
    for (const p of this.programs.values()) for (const x of p.variants.values()) if (x === v) return p
    throw new Error("A variant without its program.")
  }

  private bindGroup(v: Variant, inputs: Array<Texture>, outputs: Array<Texture>) {
    const key = `${v.id}|${inputs.map((t) => t.id).join(",")}|${outputs.map((t) => t.id).join(",")}`
    let group = this.bindGroups.get(key)
    if (group) return group
    const entries: Array<GPUBindGroupEntry> = [
      { binding: 0, resource: { buffer: this.uniformBuffer, size: v.t.uniformBytes } }
    ]
    inputs.forEach((t, i) => entries.push({ binding: i + 1, resource: (t.tex as GPUTexture).createView() }))
    outputs.forEach((t, i) =>
      entries.push({ binding: inputs.length + 1 + i, resource: (t.tex as GPUTexture).createView() })
    )
    group = this.device.createBindGroup({ layout: v.layout, entries })
    this.bindGroups.set(key, group)
    for (const t of [...inputs, ...outputs]) {
      let keys = this.groupsByTexture.get(t.id)
      if (!keys) this.groupsByTexture.set(t.id, (keys = new Set()))
      keys.add(key)
    }
    return group
  }

  /** The program's uniform values in its layout (the rectangle's 16 bytes left for each band). */
  private uniformBlock(program: WgProgram, v: Variant) {
    const block = new ArrayBuffer(v.t.uniformBytes)
    const i32 = new Int32Array(block)
    const f32 = new Float32Array(block)
    const u32 = new Uint32Array(block)
    for (const [name, { type, offset }] of v.t.uniforms) {
      const value = program.values.get(name)
      if (value === undefined) continue
      const at = offset / 4
      const base = type.startsWith("i") ? "i" : type.startsWith("u") ? "u" : "f"
      const put = (k: number, x: number) => {
        if (base === "i") i32[at + k] = x
        else if (base === "u") u32[at + k] = x
        else f32[at + k] = x
      }
      if (typeof value === "number") put(0, value)
      else if (Array.isArray(value)) value.forEach((x, k) => put(k, x))
      else if ("int" in value) put(0, value.int)
      else if ("ivec4" in value) value.ivec4.forEach((x, k) => put(k, x))
      else Array.from(value.ivec4s).forEach((x, k) => put(k, x))
    }
    return new Uint8Array(block)
  }

  /** Appends a dispatch's uniforms (block plus its band) to the submission's: their offset. */
  private pushUniforms(block: Uint8Array, x: number, y: number, width: number, height: number) {
    let offset = align(this.ringUsed, this.uniformAlign)
    if (offset + block.byteLength > this.ring.byteLength) {
      this.submit()
      offset = 0
    }
    this.ring.set(block, offset)
    const rect = new Uint32Array(this.ring.buffer, offset, 4)
    rect[0] = x
    rect[1] = y
    rect[2] = width
    rect[3] = height
    this.ringUsed = offset + block.byteLength
    return offset
  }

  /** Sends the commands issued so far to the GPU. */
  private submit() {
    this.endComputePass()
    if (!this.encoder) return
    const timestamps = this.resolveTimestamps()
    if (this.ringUsed > 0) this.device.queue.writeBuffer(this.uniformBuffer, 0, this.ring, 0, align(this.ringUsed, 4))
    this.device.queue.submit([this.encoder.finish()])
    timestamps?.()
    this.encoder = null
    this.generation++
    this.ringUsed = 0
    for (const b of this.transient) b.destroy()
    this.transient = []
    for (const t of this.doomed) t.destroy()
    this.doomed = []
  }

  flush() {
    this.submit()
    this.submitGpuWork()
  }

  /**
   * Bands of at least 64Ki texels: a WebGPU dispatch doesn't overlap the
   * next, so small ones leave the GPU idle. LFM3's output head paced a row
   * of 1,024 texels at a time took 5 ms a token on an M-series Mac, where
   * the whole of it is a fraction of one.
   */
  protected override minBandRows(width: number) {
    return Math.ceil(65536 / width)
  }

  // ------------------------------------------------------------ fences

  protected fence(): Fence {
    this.submit()
    const fence: Fence = { done: false, promise: Promise.resolve() }
    fence.promise = this.device.queue.onSubmittedWorkDone().then(() => {
      fence.done = true
    })
    return fence
  }

  protected fenceReady(fence: Fence) {
    return fence.done
  }

  protected deleteFence() {}

  async pace(ahead: number, messages = false, reads = 0, layers = 0) {
    if (this.deferPipelines) return
    return super.pace(ahead, messages, reads, layers)
  }

  protected async wait(fence: Fence | null, budget: SubmissionBudget = this.submissionBudget, at = performance.now()) {
    if (this.deferPipelines) return
    if (this.lost) throw new GpuError("The GPU device was lost.")
    if (!fence || fence.done) return
    budget.waiting(at)
    try {
      await fence.promise
    } finally {
      budget.completed(performance.now())
    }
    if (this.lost) throw new GpuError("The GPU device was lost.")
  }

  /** (WebGPU can't block: a throttled page's waits stay on the fences.) */
  protected readBack() {}

  // ------------------------------------------------------------ reads

  read(
    texture: Texture,
    rect = { x: 0, y: 0, width: texture.width, height: texture.height }
  ): Promise<Float32Array> {
    return this.readStart(texture, rect)()
  }

  /** As read, now: WebGPU has no blocking read. */
  readNow(texture: Texture, rect = { x: 0, y: 0, width: texture.width, height: texture.height }) {
    return this.read(texture, rect)
  }

  /** Copies a rectangle into a mappable buffer, behind the passes issued: the bytes, unpadded, once there. */
  private copyOut(texture: Texture, rect: { x: number; y: number; width: number; height: number }) {
    const f = formatOf(texture.internal)
    const row = rect.width * f.bytes
    const padded = align(row, 256)
    const buffer = this.device.createBuffer({
      size: padded * rect.height,
      usage: BUFFER.MAP_READ | BUFFER.COPY_DST
    })
    this.endComputePass()
    this.commands().copyTextureToBuffer(
      { texture: texture.tex as GPUTexture, origin: { x: rect.x, y: rect.y } },
      { buffer, bytesPerRow: padded, rowsPerImage: rect.height },
      { width: rect.width, height: rect.height }
    )
    this.submit()
    this.submitGpuWork()
    const mapped = buffer.mapAsync(MAP_READ)
    // (Unhandled until awaited: a cancelled read's rejection is expected.)
    mapped.catch(() => {})
    const bytes = async (into: Uint8Array) => {
      await mapped
      const data = new Uint8Array(buffer.getMappedRange())
      if (padded === row) into.set(data.subarray(0, into.byteLength))
      else for (let y = 0; y < rect.height; y++) into.set(data.subarray(y * padded, y * padded + row), y * row)
      buffer.unmap()
    }
    return { buffer, bytes }
  }

  readStart(
    texture: Texture,
    rect = { x: 0, y: 0, width: texture.width, height: texture.height }
  ): PendingRead {
    if (texture.internal !== GL.RGBA32F)
      throw new Error(`Can't read back a texture of format ${texture.internal} as floats.`)
    const out = new Float32Array(rect.width * rect.height * 4)
    // (Nothing ran while pipelines are deferred: nothing to wait for.)
    if (this.deferPipelines) return Object.assign(async () => out, { cancel: () => undefined })
    const { buffer, bytes } = this.copyOut(texture, rect)
    const budget = this.submissionBudget, at = performance.now()
    let freed = false
    const free = () => {
      if (freed) return
      freed = true
      buffer.destroy()
    }
    return Object.assign(
      async () => {
        if (freed) throw new Error("This read was cancelled.")
        budget.waiting(at)
        try {
          await bytes(new Uint8Array(out.buffer))
        } finally {
          budget.completed(performance.now())
          free()
        }
        if (this.lost) throw new GpuError("The GPU device was lost.")
        return out
      },
      { cancel: free }
    )
  }

  /** A whole RGBA32F or RGBA32UI texture's bytes (see gl.ts). */
  async readBytes(texture: Texture): Promise<Uint8Array> {
    if (this.deferPipelines) return new Uint8Array(texture.width * texture.height * 16)
    if (texture.internal !== GL.RGBA32UI && texture.internal !== GL.RGBA32F)
      throw new Error(`Can't read back a texture of format ${texture.internal}.`)
    const out = new Uint8Array(texture.width * texture.height * 16)
    const { buffer, bytes } = this.copyOut(texture, { x: 0, y: 0, width: texture.width, height: texture.height })
    try {
      await bytes(out)
    } finally {
      buffer.destroy()
    }
    return out
  }

  // ------------------------------------------------------------ teardown

  dispose() {
    for (const t of [...this.live]) this.delete(t)
    this.programs.clear()
    this.bindGroups.clear()
    this.uniformBuffer.destroy()
    this.paced.length = 0
    this.device.destroy()
    this.lost = true
  }

  async disposePaced() {
    try {
      let bytes = 0
      for (const texture of [...this.live]) {
        if (this.lost) break
        this.delete(texture)
        bytes += texture.bytes
        if (bytes >= 64 * 2 ** 20) {
          await this.finish()
          bytes = 0
        }
      }
    } finally {
      this.dispose()
    }
  }
}
