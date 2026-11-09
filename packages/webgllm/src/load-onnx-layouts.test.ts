import assert from "node:assert/strict"
import { readFileSync } from "node:fs"
import { join } from "node:path"
import { test } from "node:test"
import { halfTable } from "./half.js"
import { checkOnnxGraph, checkOnnxModel, loadOnnx } from "./load-onnx.js"
import { OnnxDataType, parseOnnxModel, type OnnxGraph } from "./onnx.js"
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

// Edited graphs (review scenarios): a copy whose nodes and initializers can
// be changed without touching the parsed fixture.
const parse = (path: string) => parseOnnxModel(new Uint8Array(readFileSync(path)))
const copy = (g: OnnxGraph): OnnxGraph => ({
  ...g,
  nodes: g.nodes.map((n) => ({ ...n, inputs: [...n.inputs], outputs: [...n.outputs], attributes: new Map(n.attributes) })),
  initializers: new Map(g.initializers)
})
const small = join(process.cwd(), "src/test-fixtures/lfm2-350m-onnx")
const smallConfig = JSON.parse(readFileSync(join(small, "config.json"), "utf8"))

test("a plain MatMul head with its own weights is refused, whatever the embedding", () => {
  const g = copy(parse(join(small, "model_q4f16.onnx")))
  const i = g.nodes.findIndex((n) => n.name === "/lm_head/MatMul_Quant")
  const head = g.nodes[i]
  const D = smallConfig.hidden_size
  const vocab = smallConfig.vocab_size
  g.initializers.set("lm_head.weight", {
    name: "lm_head.weight",
    dims: [D, vocab],
    dataType: OnnxDataType.FLOAT16,
    external: { location: "model_q4f16.onnx_data", offset: 0, length: D * vocab * 2 }
  })
  g.nodes[i] = { ...head, name: "/lm_head/MatMul", opType: "MatMul", inputs: [head.inputs[0], "lm_head.weight"], attributes: new Map() }
  assert.throws(() => checkOnnxModel(g, smallConfig), /LM head/)
})

test("a projection is found by the weights it reads when its node has another name", () => {
  const g = copy(parse(join(small, "model_q4f16.onnx")))
  const node = g.nodes.find((n) => n.name === "/model/layers.0/mlp/gate_proj/MatMul_Quant")
  assert.ok(node)
  node.name = "/other/gate/MatMul_Quant"
  assert.deepEqual(checkOnnxModel(g, smallConfig), { dataFiles: ["model_q4f16.onnx_data"] })
})

test("a float embedding whose declared bytes don't cover its rows is refused before any read", () => {
  const g = copy(parse(join(dir, "model_q4f16.onnx")))
  const emb = g.initializers.get("model.embed_tokens.weight")
  assert.ok(emb?.external)
  g.initializers.set(emb.name, { ...emb, external: { ...emb.external, length: emb.external.length / 2 } })
  assert.throws(() => checkOnnxModel(g, config), /model\.embed_tokens\.weight has \d+ bytes, expected/)
})

test("an embedding whose rows aren't a multiple of 4 is refused before any read", () => {
  const g = copy(parse(join(dir, "model_q4f16.onnx")))
  const emb = g.initializers.get("model.embed_tokens.weight")
  assert.ok(emb?.external)
  const [, D] = emb.dims
  const rows = emb.dims[0] - 1
  g.initializers.set(emb.name, { ...emb, dims: [rows, D], external: { ...emb.external, length: rows * D * 2 } })
  assert.throws(() => checkOnnxModel(g, { ...config, vocab_size: rows }), /65535 rows, not a multiple of 4/)
})
