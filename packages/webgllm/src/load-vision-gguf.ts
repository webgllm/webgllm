/**
 * Loads a vision encoder from a llama.cpp projector file (`mmproj-*.gguf`,
 * architecture `clip`) as the pieces vision.ts runs (vision-weights.ts).
 * Three projector types are read:
 *
 * - `lfm2` (LiquidAI/LFM2.5-VL-*-GGUF): SigLIP2 NaFlex and LFM2-VL's
 *   projector. `v.patch_embd` is a convolution's kernel (out, channel, y,
 *   x): its columns are reordered to the processor's patch rows (y, x,
 *   channel). `v.position_embd` is the 16 × 16 position table.
 * - `lfm3vl` (LiquidAI/LFM3-VL-*-GGUF): LFM3-VL's tower. `v.patch_embd`
 *   has a kernel per frame of a patch (`.weight`, `.weight.1`), which
 *   become the patch row's two frames (channel, frame, y, x).
 * - `qwen3vl_merger` (unsloth/Qwen3.5-*-GGUF): Qwen3.5's ViT, as its ONNX
 *   export has it (load-vision.ts): a kernel per frame, a 48 × 48 position
 *   table interpolated corner to corner, a fused q|k|v projection, and the
 *   merger behind the last LayerNorm.
 *
 * RoPE's base isn't in the files: 1e6 for LFM3-VL and 1e4 for Qwen3.5, as
 * llama.cpp sets them for the types (frequency j of headDim / 4 is
 * base^(-j / (headDim / 4))).
 *
 * Q8_0 projections are taken as they are; float ones (F32, F16: the patch
 * kernels, rows Q8_0 can't hold such as SigLIP2's 4,304-wide MLP, and all
 * of a float projector, which is all there is of Qwen3.5's) are quantized
 * to 8 bits per block of 32, as the text loaders do a float matrix.
 */
import { GgmlType, readGgufHeader, type GgufHeader, type GgufTensorInfo } from "./gguf.js"
import { bytesToFloats, halvesToFloats } from "./half.js"
import { ReadPlan } from "./read-plan.js"
import {
  checkVisionConfig,
  floatMatrix,
  q8Matrix,
  stackLinears,
  type VisionBlock,
  type VisionConfig,
  type VisionLinear,
  type VisionNorm,
  type VisionPiece
} from "./vision-weights.js"
import type { ByteSource } from "./weights.js"

/** Projector types this reads (`clip.projector_type`, or `clip.vision.projector_type`). */
export const GGUF_PROJECTOR_TYPES = ["lfm2", "lfm3vl", "qwen3vl_merger"] as const
type ProjectorType = (typeof GGUF_PROJECTOR_TYPES)[number]

/** What differs between the projector types: the tower's shape, and where its tensors are. */
type Projector = {
  /** The key of the patches-per-token side. */
  merge: string
  /** Patch rows: y, x, channel (one frame), or channel, frame, y, x (a kernel per frame). */
  channelLast: boolean
  /** A learned position table (`v.position_embd`), and how it's interpolated. */
  positions: VisionConfig["interpolation"] | null
  /** RoPE's base where q and k are rotated. */
  ropeTheta: number | null
  /** LFM3-VL's tower: RMS norms (one after the patch projection, per-head ones on q and k), conv blocks, SwiGLU. */
  tower: boolean
  /** The attention projections: q, k and v, or one fused. */
  qkv: Array<string>
  /** The merger's LayerNorm and its two projections. */
  mergerNorm: string
  merger: [string, string]
  /** How the text model places an image's rows. */
  place: VisionConfig["positions"]
}

const PROJECTORS: Record<ProjectorType, Projector> = {
  lfm2: {
    merge: "clip.vision.projector.scale_factor",
    channelLast: true,
    positions: "antialias",
    ropeTheta: null,
    tower: false,
    qkv: ["attn_q", "attn_k", "attn_v"],
    mergerNorm: "v.post_ln",
    merger: ["mm.1", "mm.2"],
    place: "sequence"
  },
  lfm3vl: {
    merge: "clip.vision.spatial_merge_size",
    channelLast: false,
    positions: null,
    ropeTheta: 1e6,
    tower: true,
    qkv: ["attn_q", "attn_k", "attn_v"],
    mergerNorm: "mm.input_norm",
    merger: ["mm.0", "mm.2"],
    place: "sequence"
  },
  qwen3vl_merger: {
    merge: "clip.vision.spatial_merge_size",
    channelLast: false,
    positions: "corners",
    ropeTheta: 1e4,
    tower: false,
    qkv: ["attn_qkv"],
    mergerNorm: "v.post_ln",
    merger: ["mm.0", "mm.2"],
    place: "grid"
  }
}

/** The file's header, and the encoder's config from it. */
export async function planVisionGguf(source: ByteSource): Promise<{ header: GgufHeader; config: VisionConfig; type: ProjectorType; bits: 8; zeros: false }> {
  const header = await readGgufHeader(source)
  const meta = header.metadata
  if (meta.get("general.architecture") !== "clip") throw new Error("GGUF: not a vision projector (general.architecture isn't clip)")
  const type = meta.get("clip.vision.projector_type") ?? meta.get("clip.projector_type")
  if (typeof type !== "string" || !(GGUF_PROJECTOR_TYPES as ReadonlyArray<string>).includes(type))
    throw new Error(`GGUF: a "${String(type)}" projector; webgllm runs ${GGUF_PROJECTOR_TYPES.join(", ")}`)
  const kind = PROJECTORS[type as ProjectorType]
  const num = (key: string, fallback?: number) => {
    const value = meta.get(key) ?? fallback
    if (typeof value !== "number") throw new Error(`GGUF: ${key} is missing`)
    return value
  }
  const tensor = (name: string) => {
    const info = header.tensors.get(name)
    if (!info) throw new Error(`GGUF: no tensor ${name}`)
    return info
  }
  // The types the loader reads, checked with the header: before any weights download.
  for (const info of header.tensors.values())
    if (info.type !== GgmlType.F32 && info.type !== GgmlType.F16 && info.type !== GgmlType.Q8_0)
      throw new Error(`GGUF: ${info.name} has tensor type ${info.type}, which the vision loader doesn't take (Q8_0, F16, F32 do)`)
  const width = num("clip.vision.embedding_length")
  const heads = num("clip.vision.attention.head_count")
  const depth = num("clip.vision.block_count")
  const patch = num("clip.vision.patch_size")
  const eps = num("clip.vision.attention.layer_norm_epsilon")
  const layers: VisionConfig["layers"] = []
  for (let i = 0; i < depth; i++) layers.push(header.tensors.has(`v.blk.${i}.conv2d_in_proj.weight`) ? "conv" : "attention")
  // What the encoder has no pass for: a projector's own input norm where the type has none, a norm before the blocks, Qwen3-VL's deepstack features.
  if (header.tensors.has("mm.input_norm.weight") !== (kind.mergerNorm === "mm.input_norm"))
    throw new Error("GGUF: this projector's input norm isn't where webgllm's encoder applies one")
  if (header.tensors.has("v.pre_ln.weight") || [...header.tensors.keys()].some((name) => name.includes("deepstack")))
    throw new Error("GGUF: this projector has a pre-norm or deepstack layers, which webgllm's encoder doesn't run")
  const frames = kind.channelLast ? 1 : header.tensors.has("v.patch_embd.weight.1") ? 2 : 1
  const positions = kind.positions ? tensor("v.position_embd.weight") : null
  const grid = positions ? Math.round(Math.sqrt(positions.dims[1])) : 0
  if (positions && (grid * grid !== positions.dims[1] || positions.dims[0] !== width))
    throw new Error(`GGUF: the position table is ${positions.dims.join("×")}, expected ${width} by a square grid`)
  const config = checkVisionConfig({
    width,
    heads,
    kvHeads: kind.tower ? num("clip.vision.attention.head_count_kv", heads) : heads,
    headDim: width / heads,
    ffn: num("clip.vision.feed_forward_length"),
    mergerWidth: tensor(`${kind.merger[0]}.weight`).dims[1],
    outWidth: num("clip.vision.projection_dim"),
    patchValues: 3 * frames * patch * patch,
    merge: num(kind.merge),
    layers,
    norm: kind.tower ? "rms" : "layer",
    eps,
    // The projector's own LayerNorm (LFM3-VL: 1e-6, as llama.cpp), or the encoder's last one.
    mergerEps: kind.tower ? 1e-6 : eps,
    mlp: kind.tower ? "swiglu" : "tanh",
    mergerAct: "erf",
    grid,
    interpolation: kind.positions ?? "corners",
    rope: kind.ropeTheta !== null,
    qkNorm: kind.tower,
    positions: kind.place
  })
  return { header, config, type: type as ProjectorType, bits: 8, zeros: false }
}

/** A tensor's values as floats (F32, F16). */
function floats(info: GgufTensorInfo, bytes: Uint8Array): Float32Array {
  if (info.type === GgmlType.F32) return bytesToFloats(bytes)
  if (info.type === GgmlType.F16) return halvesToFloats(bytes)
  throw new Error(`GGUF: ${info.name} has tensor type ${info.type}, expected F32 or F16`)
}

/** A [K, N] tensor as a projection's matrix. */
function matrix(info: GgufTensorInfo, bytes: Uint8Array) {
  const [K, N] = info.dims
  if (info.dims.length !== 2) throw new Error(`GGUF: ${info.name} is [${info.dims}], expected a matrix`)
  if (info.type === GgmlType.Q8_0) return q8Matrix(bytes, N, K)
  return floatMatrix(floats(info, bytes), N, K)
}

/** Yields the encoder's pieces: config, patch embedding and positions, each block, the merger. */
export async function* loadVisionGguf(source: ByteSource): AsyncGenerator<VisionPiece, void, undefined> {
  const { header, config, type } = await planVisionGguf(source)
  const kind = PROJECTORS[type]
  const lfm3 = kind.tower
  const tensor = (name: string) => {
    const info = header.tensors.get(name)
    if (!info) throw new Error(`GGUF: no tensor ${name}`)
    if (!Number.isFinite(info.byteLength)) throw new Error(`GGUF: ${name} has tensor type ${info.type}, which the vision loader doesn't take`)
    return info
  }
  // Every tensor the load takes, in the order it takes them.
  const plan = new ReadPlan()
  const want = (name: string) => {
    const info = tensor(name)
    plan.add({ key: name, source, offset: info.offset, length: info.byteLength })
  }
  const optional = (name: string) => header.tensors.has(name)
  const linearNames = (name: string) => [`${name}.weight`, ...(optional(`${name}.bias`) ? [`${name}.bias`] : [])]
  const normNames = linearNames
  const side = Number(header.metadata.get("clip.vision.patch_size"))
  const frames = config.patchValues / (3 * side * side)
  const patchKernels = ["v.patch_embd.weight", ...(frames === 2 ? ["v.patch_embd.weight.1"] : [])]
  for (const name of patchKernels) want(name)
  if (optional("v.patch_embd.bias")) want("v.patch_embd.bias")
  if (config.grid) want("v.position_embd.weight")
  if (lfm3) want("v.embedding_norm.weight")
  const blockNames = config.layers.map((layer, i) => {
    const p = `v.blk.${i}.`
    const mlp = lfm3 ? ["ffn_gate", "ffn_up", "ffn_down"] : ["ffn_up", "ffn_down"]
    const operator = layer === "conv" ? ["conv2d_in_proj", "conv2d_out_proj"] : [...kind.qkv, "attn_out"]
    const extra = layer === "conv" ? [`${p}conv2d_spatial.weight`] : lfm3 ? [`${p}attn_q_norm.weight`, `${p}attn_k_norm.weight`] : []
    return [...normNames(`${p}ln1`), ...operator.flatMap((n) => linearNames(p + n)), ...extra, ...normNames(`${p}ln2`), ...mlp.flatMap((n) => linearNames(p + n))]
  })
  for (const names of blockNames) names.forEach(want)
  const { mergerNorm } = kind
  const [first, second] = kind.merger
  for (const name of [...normNames(mergerNorm), ...linearNames(first), ...linearNames(second)]) want(name)

  const take = async (name: string) => ({ info: tensor(name), bytes: await plan.take(name) })
  const vector = async (name: string) => {
    const { info, bytes } = await take(name)
    return floats(info, bytes)
  }
  const linear = async (name: string): Promise<VisionLinear> => {
    const { info, bytes } = await take(`${name}.weight`)
    return { m: matrix(info, bytes), bias: optional(`${name}.bias`) ? await vector(`${name}.bias`) : null }
  }
  const norm = async (name: string): Promise<VisionNorm> => ({
    weight: await vector(`${name}.weight`),
    bias: optional(`${name}.bias`) ? await vector(`${name}.bias`) : null
  })

  yield { kind: "config", config }
  // The patch kernels ([x, y, channel, out] in ggml's order: out, channel, y, x) as rows of the processor's layout.
  const W = config.width
  const patchValues = new Float32Array(W * config.patchValues)
  for (const [frame, name] of patchKernels.entries()) {
    const { info, bytes } = await take(name)
    if (info.dims.join() !== `${side},${side},3,${W}`) throw new Error(`GGUF: ${name} is [${info.dims}], expected [${side}, ${side}, 3, ${W}]`)
    const kernel = floats(info, bytes)
    for (let n = 0; n < W; n++)
      for (let c = 0; c < 3; c++)
        for (let y = 0; y < side; y++)
          for (let x = 0; x < side; x++) {
            const k = kind.channelLast ? (y * side + x) * 3 + c : ((c * frames + frame) * side + y) * side + x
            patchValues[n * config.patchValues + k] = kernel[((n * 3 + c) * side + y) * side + x]
          }
  }
  const patch: VisionLinear = { m: floatMatrix(patchValues, W, config.patchValues), bias: optional("v.patch_embd.bias") ? await vector("v.patch_embd.bias") : null }
  let positions: Float32Array | undefined
  if (config.grid) positions = await vector("v.position_embd.weight")
  const patchNorm = lfm3 ? { weight: await vector("v.embedding_norm.weight"), bias: null } : undefined
  let invFreq: Float32Array | undefined
  if (kind.ropeTheta !== null) {
    const n = config.headDim / 4
    const theta = kind.ropeTheta
    invFreq = Float32Array.from({ length: n }, (_, j) => theta ** (-j / n))
  }
  yield { kind: "patch", patch, ...(positions && { positions }), ...(invFreq && { invFreq }), ...(patchNorm && { norm: patchNorm }) }

  for (const [index, layer] of config.layers.entries()) {
    const p = `v.blk.${index}.`
    const norm1 = await norm(`${p}ln1`)
    let value: VisionBlock
    const mlp = async () => {
      const norm2 = await norm(`${p}ln2`)
      if (lfm3) {
        const fc1 = stackLinears([await linear(`${p}ffn_gate`), await linear(`${p}ffn_up`)])
        return { norm2, fc1, fc2: await linear(`${p}ffn_down`) }
      }
      // The projection that reads the block's width is the first, whatever the file calls it.
      const up = await linear(`${p}ffn_up`)
      const down = await linear(`${p}ffn_down`)
      return up.m.K === W ? { norm2, fc1: up, fc2: down } : { norm2, fc1: down, fc2: up }
    }
    if (layer === "conv") {
      const inProj = await linear(`${p}conv2d_in_proj`)
      const outProj = await linear(`${p}conv2d_out_proj`)
      const spatial = await take(`${p}conv2d_spatial.weight`)
      if (spatial.info.dims.join() !== `3,3,1,${W}`) throw new Error(`GGUF: ${spatial.info.name} is [${spatial.info.dims}], expected a 3 × 3 depthwise kernel`)
      value = { type: "conv", norm1, inProj, kernel: floats(spatial.info, spatial.bytes), outProj, ...(await mlp()) }
    } else {
      const parts: Array<VisionLinear> = []
      for (const name of kind.qkv) parts.push(await linear(p + name))
      const qkv = stackLinears(parts)
      const proj = await linear(`${p}attn_out`)
      const qNorm = lfm3 ? await vector(`${p}attn_q_norm.weight`) : undefined
      const kNorm = lfm3 ? await vector(`${p}attn_k_norm.weight`) : undefined
      value = { type: "attention", norm1, qkv, ...(qNorm && { qNorm }), ...(kNorm && { kNorm }), proj, ...(await mlp()) }
    }
    yield { kind: "block", index, value }
  }
  yield { kind: "merger", norm: await norm(mergerNorm), linear1: await linear(first), linear2: await linear(second) }
}
