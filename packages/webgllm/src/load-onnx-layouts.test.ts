import assert from "node:assert/strict"
import { readFileSync } from "node:fs"
import { join } from "node:path"
import { test } from "node:test"
import { halfTable } from "./half.js"
import { checkOnnxGraph, loadOnnx } from "./load-onnx.js"
import { parseOnnxModel } from "./onnx.js"
import type { QuantMatrix } from "./weights.js"

// onnx-community/LFM2-1.2B-Tool-ONNX's real q4f16 graph (no weights) and
// config.json. Its export differs from the onnxruntime-genai ones: the
// MatMulNBits weights are named `model.layers.0.mlp.gate_proj.MatMul.weight_Q4`
// (no zero points), the conv's split sizes come from a Constant node, and
// the embedding is float16, read by a plain Gather, with the LM head a
// MatMul over the same tensor transposed.
const dir = join(process.cwd(), "src/test-fixtures/lfm2-1.2b-tool-onnx")
const graph = new Uint8Array(readFileSync(join(dir, "model_q4f16.onnx")))
const config = JSON.parse(readFileSync(join(dir, "config.json"), "utf8"))

test("an export with differently named 4-bit weights and a float16 embedding passes the graph check", () => {
  assert.deepEqual(checkOnnxGraph(graph, config), { dataFiles: ["model_q4f16.onnx_data"] })
})

test("a float16 tied embedding loads quantized to 8 bits, in chunks of rows", async () => {
  const emb = parseOnnxModel(graph).initializers.get("model.embed_tokens.weight")
  assert.ok(emb?.external)
  const [vocab, D] = emb.dims
  const ext = emb.external
  // Embedding value of row r, column k (exact in float16).
  const value = (r: number, k: number) => ((r * 7 + k) % 33) - 16
  const half = (v: number) => {
    const table = halfTable()
    for (let h = 0; h < 65536; h++) if (table[h] === v) return h
    throw new Error(`no half for ${v}`)
  }
  const halves = new Map<number, number>()
  for (let v = -16; v <= 16; v++) halves.set(v, half(v))
  // Zeros everywhere but the embedding.
  const data = {
    size: ext.offset + ext.length,
    read: async (offset: number, length: number) => {
      const out = new Uint8Array(length)
      const from = Math.max(offset, ext.offset)
      const to = Math.min(offset + length, ext.offset + ext.length)
      for (let p = from - (from - ext.offset) % 2; p < to; p += 2) {
        const i = (p - ext.offset) / 2
        const h = halves.get(value(Math.floor(i / D), i % D)) as number
        if (p >= offset) out[p - offset] = h & 255
        if (p + 1 < offset + length) out[p + 1 - offset] = h >> 8
      }
      return out
    }
  }
  const chunkRows = 4096
  let layers = 0
  let rows = 0
  let first: QuantMatrix | undefined
  for await (const piece of loadOnnx({ graph, data, config, embeddingChunkRows: chunkRows })) {
    if (piece.kind === "config") {
      assert.equal(piece.config.vocabSize, vocab)
      assert.equal(piece.config.hiddenSize, D)
    }
    if (piece.kind === "layer") layers++
    if (piece.kind !== "embedding") continue
    assert.equal(piece.row0, rows)
    assert.equal(piece.totalRows, vocab)
    assert.equal(piece.value.bits, 8)
    first ??= piece.value
    rows += piece.value.N
  }
  assert.equal(layers, config.num_hidden_layers)
  assert.equal(rows, vocab)
  // 8-bit layout: rows in groups of 4, 32 values per block (repack.ts).
  const m = first as QuantMatrix
  const q = new Int8Array(m.quants.buffer, m.quants.byteOffset, m.quants.byteLength)
  const table = halfTable()
  const blocks = D / 32
  for (const r of [0, 1, 5, chunkRows - 1]) {
    for (const k of [0, 31, 32, 1000, D - 1]) {
      const si = (r >> 2) * blocks * 4 + (r & 3) + 4 * (k >> 5)
      const got = q[si * 32 + (k & 31)] * table[m.scales[si]]
      assert.ok(Math.abs(got - value(r, k)) < 0.07, `row ${r} col ${k}: ${got} vs ${value(r, k)}`)
    }
  }
})
