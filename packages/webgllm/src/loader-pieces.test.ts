import assert from "node:assert/strict"
import { readFileSync } from "node:fs"
import { join } from "node:path"
import test from "node:test"
import { GgmlType, type GgufHeader, type GgufTensorInfo, type GgufValue } from "./gguf.js"
import { loadGguf } from "./load-gguf.js"
import { loadOnnx } from "./load-onnx.js"
import type { ByteSource } from "./weights.js"
import type { WeightPiece } from "./weights.js"

/** Every buffer a piece's matrices hold (quants, scales, zero points, raw rows), by where it is in the piece. */
function buffers(piece: WeightPiece): Array<ArrayBufferLike> {
  const found: Array<ArrayBufferLike> = []
  const walk = (v: unknown, depth: number) => {
    if (!v || typeof v !== "object" || depth > 4) return
    const m = v as { quants?: Uint32Array; scales?: ArrayBufferView; zeros?: ArrayBufferView | null; raw?: ArrayBufferView }
    if (m.quants instanceof Uint32Array && m.scales) {
      for (const view of [m.quants, m.scales, m.zeros, m.raw]) if (view && view.byteLength) found.push(view.buffer)
      return
    }
    for (const x of Object.values(v)) walk(x, depth + 1)
  }
  walk("value" in piece ? piece.value : piece, 0)
  return found
}

/** Whether pieces yielded earlier share a buffer with a later one (and so change under a consumer that keeps them). */
async function shared(pieces: AsyncIterable<WeightPiece>) {
  const seen = new Set<ArrayBufferLike>()
  let sharing = false
  for await (const piece of pieces) {
    const mine = new Set(buffers(piece))
    for (const b of mine) if (seen.has(b)) sharing = true
    for (const b of mine) seen.add(b)
  }
  return sharing
}

// Two LFM2 layers whose feed-forward matrices have one shape, and an embedding of 8 rows (two chunks of 4).
function ggufFixture() {
  const tensors = new Map<string, GgufTensorInfo>(), chunks: Array<Uint8Array> = []
  let offset = 0
  const add = (name: string, dims: Array<number>, type: number = GgmlType.Q4_0) => {
    const count = dims.reduce((a, b) => a * b, 1)
    const bytes = new Uint8Array(type === GgmlType.F32 ? count * 4 : (count / 32) * 18)
    if (type === GgmlType.F32) new Float32Array(bytes.buffer).fill(1.25)
    else for (let i = 0; i < bytes.length; i += 18) { bytes[i + 1] = 60; bytes.fill(136 + (offset & 7), i + 2, i + 18) }
    tensors.set(name, { name, dims, type, offset, byteLength: bytes.length })
    offset += bytes.length
    chunks.push(bytes)
  }
  for (let i = 0; i < 2; i++) {
    const p = `blk.${i}.`
    for (const norm of ["attn_norm", "ffn_norm"]) add(p + norm + ".weight", [32], GgmlType.F32)
    add(p + "shortconv.in_proj.weight", [32, 96])
    add(p + "shortconv.out_proj.weight", [32, 32])
    add(p + "shortconv.conv.weight", [3, 32], GgmlType.F32)
    for (const name of ["ffn_gate", "ffn_up", "ffn_down"]) add(p + name + ".weight", [32, 32])
  }
  add("token_embd_norm.weight", [32], GgmlType.F32)
  add("token_embd.weight", [32, 8])
  const metadata = new Map<string, GgufValue>([
    ["general.architecture", "lfm2"], ["lfm2.block_count", 2], ["lfm2.embedding_length", 32], ["lfm2.attention.head_count", 8],
    ["lfm2.attention.head_count_kv", [0, 0]], ["lfm2.feed_forward_length", 32], ["lfm2.context_length", 128000],
    ["lfm2.rope.freq_base", 1000000], ["lfm2.attention.layer_norm_rms_epsilon", 1e-5], ["lfm2.shortconv.l_cache", 3]
  ])
  const bytes = new Uint8Array(offset)
  let at = 0
  for (const chunk of chunks) { bytes.set(chunk, at); at += chunk.length }
  const header: GgufHeader = { version: 3, metadata, tensors, dataOffset: 0, headerLength: 0 }
  const source: ByteSource = { size: bytes.length, read: async (o: number, length: number) => bytes.slice(o, o + length) }
  return { header, source }
}

test("the GGUF loader's pieces are each their own unless the consumer takes them one at a time (reusePieces)", async () => {
  const f = ggufFixture()
  assert.equal(await shared(loadGguf(f.source, { header: f.header, embeddingChunkRows: 4 })), false)
  assert.equal(await shared(loadGguf(f.source, { header: f.header, embeddingChunkRows: 4, reusePieces: true })), true)
})

test("the ONNX loader's pieces are each their own unless the consumer takes them one at a time (reusePieces)", async () => {
  const dir = join(process.cwd(), "src/test-fixtures/lfm2-350m-onnx")
  const graph = new Uint8Array(readFileSync(join(dir, "model_q4f16.onnx")))
  const config = JSON.parse(readFileSync(join(dir, "config.json"), "utf8")) as unknown
  const zeros: ByteSource = { size: 2 ** 31, read: async (_, length) => new Uint8Array(length) }
  assert.equal(await shared(loadOnnx({ graph, data: zeros, config, embeddingChunkRows: 16384 })), false)
  assert.equal(await shared(loadOnnx({ graph, data: zeros, config, embeddingChunkRows: 16384, reusePieces: true })), true)
})

test("with read buffers retired (releaseReadBuffers) and reuse off, no piece the ONNX loader yielded is detached", async () => {
  const dir = join(process.cwd(), "src/test-fixtures/lfm2-350m-onnx")
  const graph = new Uint8Array(readFileSync(join(dir, "model_q4f16.onnx")))
  const config = JSON.parse(readFileSync(join(dir, "config.json"), "utf8")) as unknown
  const zeros: ByteSource = { size: 2 ** 31, read: async (_, length) => new Uint8Array(length) }
  const kept: Array<{ kind: string; views: Array<ArrayBufferView> }> = []
  for await (const piece of loadOnnx({ graph, data: zeros, config, embeddingChunkRows: 16384, releaseReadBuffers: true })) {
    const views: Array<ArrayBufferView> = []
    const walk = (v: unknown, depth: number) => {
      if (!v || typeof v !== "object" || depth > 4) return
      const m = v as { quants?: Uint32Array; scales?: ArrayBufferView; zeros?: ArrayBufferView | null; raw?: ArrayBufferView }
      if (m.quants instanceof Uint32Array && m.scales) {
        for (const view of [m.scales, m.zeros, m.raw]) if (view && view.byteLength) views.push(view)
        return
      }
      for (const x of Object.values(v)) walk(x, depth + 1)
    }
    walk("value" in piece ? piece.value : piece, 0)
    kept.push({ kind: piece.kind, views })
  }
  const detached = kept.flatMap(({ kind, views }) => views.filter((v) => v.buffer.byteLength === 0).map(() => kind))
  assert.deepEqual([...new Set(detached)], [], "pieces whose buffers were detached")
})

test("with read buffers retired (releaseReadBuffers) and reuse off, no piece the GGUF loader yielded is detached", async () => {
  const f = ggufFixture()
  const kept: Array<WeightPiece> = []
  for await (const piece of loadGguf(f.source, { header: f.header, embeddingChunkRows: 4, releaseReadBuffers: true })) kept.push(piece)
  const detached = kept.filter((piece) => buffers(piece).some((b) => b.byteLength === 0)).map((piece) => piece.kind)
  assert.deepEqual(detached, [])
})
