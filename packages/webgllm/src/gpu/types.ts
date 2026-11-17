/** Types and constants both GPU backends share. */

export type GpuLimits = {
  maxTextureSize: number
  maxDrawBuffers: number
  /** Textures a pass can read (MAX_TEXTURE_IMAGE_UNITS: at least 16). */
  maxTextureUnits: number
  renderer: string
}

/** A read issued by readStart: call it to wait for the data, or cancel it. */
export type PendingRead = (() => Promise<Float32Array>) & { cancel: () => void }

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
  /** The backend's texture: a WebGLTexture or a GPUTexture. */
  tex: unknown
  width: number
  height: number
  internal: number
  /** Bytes it holds on the GPU, for the memory count. */
  bytes: number
}

/** A compiled shader: each backend keeps its own handles alongside the samplers it reads. */
export type Program = {
  /** The sampler uniforms the shader reads, which a pass must give. */
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
    Uniform | { int: number } | { ivec4: [number, number, number, number] } | { ivec4s: ArrayLike<number> }
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

