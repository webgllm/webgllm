/**
 * A minimal ONNX ModelProto reader: the graph's initializers (where their
 * bytes are: inline, or in the external data file) and nodes. Enough to
 * find weights and check how the graph uses them; it does not run anything.
 */

/** TensorProto.DataType values. */
export const OnnxDataType = {
  FLOAT: 1,
  UINT8: 2,
  INT8: 3,
  UINT16: 4,
  INT16: 5,
  INT32: 6,
  INT64: 7,
  STRING: 8,
  BOOL: 9,
  FLOAT16: 10,
  DOUBLE: 11,
  UINT32: 12,
  UINT64: 13,
  BFLOAT16: 16,
  UINT4: 21,
  INT4: 22
} as const

export type OnnxTensor = {
  name: string
  dims: Array<number>
  /** An OnnxDataType. */
  dataType: number
  /** Inline bytes (a view into the model file's buffer). */
  raw?: Uint8Array
  /** Bytes in the external data file. */
  external?: { location: string; offset: number; length: number }
  /** Typed fields some exporters use for small constants. */
  int64Data?: Array<number>
  floatData?: Array<number>
}

export type OnnxAttribute = {
  name: string
  /** AttributeProto.AttributeType: 1 float, 2 int, 3 string, 4 tensor, 6 floats, 7 ints, 8 strings. */
  type: number
  f?: number
  i?: number
  s?: string
  t?: OnnxTensor
  floats?: Array<number>
  ints?: Array<number>
  strings?: Array<string>
}

export type OnnxNode = {
  name: string
  opType: string
  domain: string
  inputs: Array<string>
  outputs: Array<string>
  attributes: Map<string, OnnxAttribute>
}

export type OnnxGraph = {
  name: string
  nodes: Array<OnnxNode>
  initializers: Map<string, OnnxTensor>
  inputs: Array<string>
  outputs: Array<string>
}

const utf8 = new TextDecoder()

/** Protobuf wire-format reads over one message's bytes. */
class Proto {
  pos: number
  constructor(
    readonly buf: Uint8Array,
    start = 0,
    readonly end = buf.length
  ) {
    this.pos = start
  }

  done(): boolean {
    return this.pos >= this.end
  }

  /** A varint as a number, int64 sign included (exact below 2^53). */
  varint(): number {
    const b = this.buf
    let lo = 0
    let byte: number
    for (let shift = 0; shift < 28; shift += 7) {
      byte = b[this.pos++]
      lo |= (byte & 0x7f) << shift
      if (byte < 0x80) return lo >>> 0
    }
    byte = b[this.pos++]
    lo |= (byte & 0x0f) << 28
    let hi = (byte & 0x7f) >> 4
    if (byte >= 0x80) {
      for (let shift = 3; shift < 32; shift += 7) {
        byte = b[this.pos++]
        hi |= (byte & 0x7f) << shift
        if (byte < 0x80) break
      }
    }
    return (hi | 0) * 2 ** 32 + (lo >>> 0)
  }

  /** The next field's number and wire type, or null at the end. */
  tag(): [number, number] | null {
    if (this.pos >= this.end) return null
    const key = this.varint()
    return [Math.floor(key / 8), key & 7]
  }

  /** A length-delimited field as a sub-reader. */
  sub(): Proto {
    const n = this.varint()
    const start = this.pos
    this.pos += n
    if (this.pos > this.end) throw new Error("ONNX: truncated message")
    return new Proto(this.buf, start, start + n)
  }

  bytes(): Uint8Array {
    const n = this.varint()
    const start = this.pos
    this.pos += n
    if (this.pos > this.end) throw new Error("ONNX: truncated message")
    return this.buf.subarray(start, start + n)
  }

  string(): string {
    return utf8.decode(this.bytes())
  }

  float(): number {
    const b = this.buf
    const p = this.pos
    this.pos += 4
    return new DataView(b.buffer, b.byteOffset + p, 4).getFloat32(0, true)
  }

  skip(wire: number): void {
    switch (wire) {
      case 0:
        this.varint()
        break
      case 1:
        this.pos += 8
        break
      case 2: {
        const n = this.varint()
        this.pos += n
        break
      }
      case 5:
        this.pos += 4
        break
      default:
        throw new Error(`ONNX: unsupported wire type ${wire}`)
    }
  }

  /** A repeated varint field, packed or not. */
  varints(wire: number, out: Array<number>): void {
    if (wire === 2) {
      const p = this.sub()
      while (!p.done()) out.push(p.varint())
    } else out.push(this.varint())
  }

  floats(wire: number, out: Array<number>): void {
    if (wire === 2) {
      const p = this.sub()
      while (!p.done()) out.push(p.float())
    } else out.push(this.float())
  }
}

function readTensor(p: Proto): OnnxTensor {
  const tensor: OnnxTensor = { name: "", dims: [], dataType: 0 }
  let external: Record<string, string> | undefined
  for (let tag = p.tag(); tag; tag = p.tag()) {
    const [field, wire] = tag
    if (field === 1) p.varints(wire, tensor.dims)
    else if (field === 2) tensor.dataType = p.varint()
    else if (field === 8) tensor.name = p.string()
    else if (field === 9) tensor.raw = p.bytes()
    else if (field === 4) p.floats(wire, (tensor.floatData ??= []))
    else if (field === 7) p.varints(wire, (tensor.int64Data ??= []))
    else if (field === 13) {
      const entry = p.sub()
      let key = ""
      let value = ""
      for (let t = entry.tag(); t; t = entry.tag()) {
        if (t[0] === 1) key = entry.string()
        else if (t[0] === 2) value = entry.string()
        else entry.skip(t[1])
      }
      ;(external ??= {})[key] = value
    } else p.skip(wire)
  }
  if (external) {
    tensor.external = {
      location: external.location ?? "",
      offset: Number(external.offset ?? 0),
      length: Number(external.length ?? Number.NaN)
    }
  }
  return tensor
}

function readAttribute(p: Proto): OnnxAttribute {
  const a: OnnxAttribute = { name: "", type: 0 }
  for (let tag = p.tag(); tag; tag = p.tag()) {
    const [field, wire] = tag
    if (field === 1) a.name = p.string()
    else if (field === 20) a.type = p.varint()
    else if (field === 2) a.f = p.float()
    else if (field === 3) a.i = p.varint()
    else if (field === 4) a.s = p.string()
    else if (field === 5) a.t = readTensor(p.sub())
    else if (field === 7) p.floats(wire, (a.floats ??= []))
    else if (field === 8) p.varints(wire, (a.ints ??= []))
    else if (field === 9) (a.strings ??= []).push(p.string())
    else p.skip(wire)
  }
  return a
}

function readNode(p: Proto): OnnxNode {
  const node: OnnxNode = {
    name: "",
    opType: "",
    domain: "",
    inputs: [],
    outputs: [],
    attributes: new Map()
  }
  for (let tag = p.tag(); tag; tag = p.tag()) {
    const [field, wire] = tag
    if (field === 1) node.inputs.push(p.string())
    else if (field === 2) node.outputs.push(p.string())
    else if (field === 3) node.name = p.string()
    else if (field === 4) node.opType = p.string()
    else if (field === 7) node.domain = p.string()
    else if (field === 5) {
      const a = readAttribute(p.sub())
      node.attributes.set(a.name, a)
    } else p.skip(wire)
  }
  return node
}

/** ValueInfoProto: just the name. */
function readValueInfoName(p: Proto): string {
  let name = ""
  for (let tag = p.tag(); tag; tag = p.tag()) {
    if (tag[0] === 1) name = p.string()
    else p.skip(tag[1])
  }
  return name
}

/** Parses a ModelProto's graph. Inline tensor bytes stay views into `model`. */
export function parseOnnxModel(model: Uint8Array): OnnxGraph {
  const top = new Proto(model)
  for (let tag = top.tag(); tag; tag = top.tag()) {
    if (tag[0] !== 7 || tag[1] !== 2) {
      top.skip(tag[1])
      continue
    }
    const p = top.sub()
    const graph: OnnxGraph = {
      name: "",
      nodes: [],
      initializers: new Map(),
      inputs: [],
      outputs: []
    }
    for (let t = p.tag(); t; t = p.tag()) {
      const [field, wire] = t
      if (field === 1) graph.nodes.push(readNode(p.sub()))
      else if (field === 2) graph.name = p.string()
      else if (field === 5) {
        const tensor = readTensor(p.sub())
        graph.initializers.set(tensor.name, tensor)
      } else if (field === 11) graph.inputs.push(readValueInfoName(p.sub()))
      else if (field === 12) graph.outputs.push(readValueInfoName(p.sub()))
      else p.skip(wire)
    }
    return graph
  }
  throw new Error("ONNX: no graph in the model")
}

/** Bytes per element, for the fixed-size types (4-bit types: 0.5). */
export function onnxElementSize(dataType: number): number {
  switch (dataType) {
    case 1:
    case 6:
    case 12:
      return 4
    case 2:
    case 3:
    case 9:
      return 1
    case 4:
    case 5:
    case 10:
    case 16:
      return 2
    case 7:
    case 11:
    case 13:
      return 8
    case 21:
    case 22:
      return 0.5
    default:
      return Number.NaN
  }
}

/** An initializer's int64 values (from raw bytes or int64_data). */
export function onnxInt64s(tensor: OnnxTensor): Array<number> {
  if (tensor.int64Data) return tensor.int64Data
  if (tensor.dataType !== OnnxDataType.INT64 || !tensor.raw) {
    throw new Error(`ONNX: ${tensor.name} has no inline int64 data`)
  }
  const raw = tensor.raw
  const view = new DataView(raw.buffer, raw.byteOffset, raw.byteLength)
  const out: Array<number> = []
  for (let p = 0; p + 8 <= raw.length; p += 8)
    out.push(Number(view.getBigInt64(p, true)))
  return out
}
