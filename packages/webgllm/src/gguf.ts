/**
 * A GGUF (v2, v3) header reader: metadata and tensor infos, read in chunks
 * from a ByteSource. Large arrays (the tokenizer's tokens and merges) are
 * skipped, not decoded; they are kept as their file offset and length for
 * `readGgufArray`.
 */
import type { ByteSource } from "./weights.js"

export const GgufValueType = {
  UINT8: 0,
  INT8: 1,
  UINT16: 2,
  INT16: 3,
  UINT32: 4,
  INT32: 5,
  FLOAT32: 6,
  BOOL: 7,
  STRING: 8,
  ARRAY: 9,
  UINT64: 10,
  INT64: 11,
  FLOAT64: 12
} as const

/** ggml tensor types the loaders know about. */
export const GgmlType = {
  F32: 0,
  F16: 1,
  Q4_0: 2,
  Q4_1: 3,
  Q5_0: 6,
  Q5_1: 7,
  Q8_0: 8,
  Q8_1: 9,
  Q2_K: 10,
  Q3_K: 11,
  Q4_K: 12,
  Q5_K: 13,
  Q6_K: 14,
  Q8_K: 15,
  I8: 24,
  I16: 25,
  I32: 26,
  I64: 27,
  F64: 28,
  BF16: 30
} as const

/** [weights per block, bytes per block] */
const TYPE_LAYOUT: Record<number, [number, number]> = {
  0: [1, 4],
  1: [1, 2],
  2: [32, 18],
  3: [32, 20],
  6: [32, 22],
  7: [32, 24],
  8: [32, 34],
  9: [32, 36],
  10: [256, 84],
  11: [256, 110],
  12: [256, 144],
  13: [256, 176],
  14: [256, 210],
  15: [256, 292],
  24: [1, 1],
  25: [1, 2],
  26: [1, 4],
  27: [1, 8],
  28: [1, 8],
  30: [1, 2]
}

/** An array too long to decode with the header: where it is in the file. */
export type GgufLazyArray = {
  lazy: true
  /** Element type (a GgufValueType). */
  type: number
  length: number
  /** File offset of the first element. */
  offset: number
}

export type GgufValue =
  number | boolean | string | Array<number | boolean | string> | GgufLazyArray

export type GgufTensorInfo = {
  name: string
  /** ggml dims: dims[0] (ne0) is the contiguous one. */
  dims: Array<number>
  /** A GgmlType. */
  type: number
  /** Absolute file offset of the tensor's data. */
  offset: number
  /** Bytes of data (NaN for an unknown type). */
  byteLength: number
}

export type GgufHeader = {
  version: number
  metadata: Map<string, GgufValue>
  tensors: Map<string, GgufTensorInfo>
  /** Absolute file offset of the tensor data section. */
  dataOffset: number
  /** Where the header ends (before alignment padding). */
  headerLength: number
}

export type ReadGgufOptions = {
  /** Arrays with more elements than this stay lazy (default 1024). */
  maxArrayLength?: number
  /** Bytes per read (default 2 MB). */
  chunkSize?: number
}

const SCALAR_SIZE: Record<number, number> = {
  0: 1,
  1: 1,
  2: 2,
  3: 2,
  4: 4,
  5: 4,
  6: 4,
  7: 1,
  10: 8,
  11: 8,
  12: 8
}

const utf8 = new TextDecoder()

/** Sequential little-endian reads over a ByteSource, a chunk at a time. */
class Cursor {
  private buf: Uint8Array = new Uint8Array(0)
  private view: DataView = new DataView(this.buf.buffer)
  private start = 0
  pos: number

  constructor(
    private source: ByteSource,
    private chunkSize: number,
    pos = 0
  ) {
    this.pos = pos
  }

  /** Whether the next n bytes are in memory. */
  has(n: number): boolean {
    return (
      this.pos >= this.start && this.pos + n <= this.start + this.buf.length
    )
  }

  /** Makes the next n bytes available. */
  async fill(n: number): Promise<void> {
    if (this.has(n)) return
    const length = Math.min(
      Math.max(n, this.chunkSize),
      this.source.size - this.pos
    )
    if (length < n)
      throw new Error(`GGUF: unexpected end of file at ${this.pos}`)
    this.buf = await this.source.read(this.pos, length)
    this.view = new DataView(
      this.buf.buffer,
      this.buf.byteOffset,
      this.buf.byteLength
    )
    this.start = this.pos
  }

  private at(n: number): number {
    const p = this.pos - this.start
    this.pos += n
    return p
  }

  u8(): number {
    return this.view.getUint8(this.at(1))
  }
  u32(): number {
    return this.view.getUint32(this.at(4), true)
  }
  u64(): number {
    const p = this.at(8)
    const hi = this.view.getUint32(p + 4, true)
    if (hi >= 0x200000) throw new Error("GGUF: 64-bit value beyond 2^53")
    return hi * 2 ** 32 + this.view.getUint32(p, true)
  }
  scalar(type: number): number | boolean {
    const v = this.view
    switch (type) {
      case 0:
        return v.getUint8(this.at(1))
      case 1:
        return v.getInt8(this.at(1))
      case 2:
        return v.getUint16(this.at(2), true)
      case 3:
        return v.getInt16(this.at(2), true)
      case 4:
        return v.getUint32(this.at(4), true)
      case 5:
        return v.getInt32(this.at(4), true)
      case 6:
        return v.getFloat32(this.at(4), true)
      case 7:
        return v.getUint8(this.at(1)) !== 0
      case 10:
        return this.u64()
      case 11:
        return Number(v.getBigInt64(this.at(8), true))
      case 12:
        return v.getFloat64(this.at(8), true)
      default:
        throw new Error(`GGUF: unknown value type ${type}`)
    }
  }
  bytes(n: number): Uint8Array {
    const p = this.at(n)
    return this.buf.subarray(p, p + n)
  }
  skip(n: number): void {
    this.pos += n
  }

  async string(): Promise<string> {
    if (!this.has(8)) await this.fill(8)
    const n = this.u64()
    if (!this.has(n)) await this.fill(n)
    return utf8.decode(this.bytes(n))
  }
}

async function readValue(
  c: Cursor,
  type: number,
  maxArrayLength: number
): Promise<GgufValue> {
  if (type === GgufValueType.STRING) return c.string()
  if (type !== GgufValueType.ARRAY) {
    const size = SCALAR_SIZE[type]
    if (size === undefined) throw new Error(`GGUF: unknown value type ${type}`)
    if (!c.has(size)) await c.fill(size)
    return c.scalar(type)
  }
  if (!c.has(12)) await c.fill(12)
  const elementType = c.u32()
  const length = c.u64()
  const offset = c.pos
  if (length > maxArrayLength) {
    if (elementType === GgufValueType.STRING) {
      // Walk the length prefixes without decoding.
      for (let i = 0; i < length; i++) {
        if (!c.has(8)) await c.fill(8)
        c.skip(c.u64())
      }
    } else {
      const size = SCALAR_SIZE[elementType]
      if (size === undefined)
        throw new Error(`GGUF: can't skip an array of type ${elementType}`)
      c.skip(length * size)
    }
    return { lazy: true, type: elementType, length, offset }
  }
  const values: Array<number | boolean | string> = []
  for (let i = 0; i < length; i++) {
    const value = await readValue(c, elementType, maxArrayLength)
    if (typeof value === "object")
      throw new Error("GGUF: nested arrays are not supported")
    values.push(value)
  }
  return values
}

/** Reads a GGUF file's header: metadata and tensor infos. */
export async function readGgufHeader(
  source: ByteSource,
  options: ReadGgufOptions = {}
): Promise<GgufHeader> {
  const { maxArrayLength = 1024, chunkSize = 2 << 20 } = options
  const c = new Cursor(source, chunkSize)
  await c.fill(24)
  const magic = utf8.decode(c.bytes(4))
  if (magic !== "GGUF") throw new Error("not a GGUF file")
  const version = c.u32()
  if (version !== 2 && version !== 3)
    throw new Error(`GGUF version ${version} is not supported`)
  const tensorCount = c.u64()
  const kvCount = c.u64()

  const metadata = new Map<string, GgufValue>()
  for (let i = 0; i < kvCount; i++) {
    const key = await c.string()
    if (!c.has(4)) await c.fill(4)
    const type = c.u32()
    metadata.set(key, await readValue(c, type, maxArrayLength))
  }

  const infos: Array<
    Omit<GgufTensorInfo, "offset" | "byteLength"> & { relative: number }
  > = []
  for (let i = 0; i < tensorCount; i++) {
    const name = await c.string()
    if (!c.has(4)) await c.fill(4)
    const nDims = c.u32()
    if (!c.has(nDims * 8 + 12)) await c.fill(nDims * 8 + 12)
    const dims: Array<number> = []
    for (let d = 0; d < nDims; d++) dims.push(c.u64())
    const type = c.u32()
    const relative = c.u64()
    infos.push({ name, dims, type, relative })
  }

  const headerLength = c.pos
  const alignment = Number(metadata.get("general.alignment") ?? 32)
  const dataOffset = Math.ceil(headerLength / alignment) * alignment
  const tensors = new Map<string, GgufTensorInfo>()
  for (const { name, dims, type, relative } of infos) {
    tensors.set(name, {
      name,
      dims,
      type,
      offset: dataOffset + relative,
      byteLength: tensorByteLength(type, dims)
    })
  }
  return { version, metadata, tensors, dataOffset, headerLength }
}

/** Bytes of a tensor of this type and ggml dims (NaN for an unknown type). */
export function tensorByteLength(type: number, dims: Array<number>): number {
  const layout = TYPE_LAYOUT[type]
  if (!layout) return Number.NaN
  const [blockWeights, blockBytes] = layout
  let rows = 1
  for (let d = 1; d < dims.length; d++) rows *= dims[d]
  return Math.ceil((dims[0] ?? 1) / blockWeights) * blockBytes * rows
}

/** Decodes a lazy array (the tokenizer's tokens, say). */
export async function readGgufArray(
  source: ByteSource,
  array: GgufLazyArray,
  chunkSize = 4 << 20
): Promise<Array<number | boolean | string>> {
  const c = new Cursor(source, chunkSize, array.offset)
  const values: Array<number | boolean | string> = new Array(array.length)
  for (let i = 0; i < array.length; i++) {
    const value = await readValue(c, array.type, Infinity)
    if (typeof value === "object")
      throw new Error("GGUF: nested arrays are not supported")
    values[i] = value
  }
  return values
}
