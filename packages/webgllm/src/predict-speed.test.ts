import assert from "node:assert/strict"
import { describe, test } from "node:test"
import { predictHere, speedFromTimes, speedKey, type SpeedDeps, type SpeedNote } from "./predict-speed.js"
import { matrixShapes, middleStack, probeConfig, PROBE_VOCAB, PROMPT_TOKENS, type ProbeTimes, type SpeedShape } from "./speed-probe.js"
import { PROMPT_BATCH } from "./handler.js"
import type { LayerType, ModelConfig } from "./weights.js"

const times: ProbeTimes = {
  promptBase: 20,
  // Six middle layers took 30 ms more than the first and last alone.
  promptStack: 50,
  stackLayers: 6,
  decodeLayers: 9,
  headAt: 0.5,
  headPerRow: 0.0001,
  vocab: 65536,
  promptTokens: 512
}

describe("speedFromTimes", () => {
  test("the stack's middle layers scale to the model's", () => {
    // 24 layers: 22 in the middle, 5 ms each.
    const types = Array.from({ length: 24 }, (): LayerType => "conv")
    const { prompt } = speedFromTimes(times, types, 65536)
    assert.equal(prompt, (1000 * 512) / (20 + 22 * 5 + 0.5))
  })

  test("a writing step is every layer, then the head", () => {
    assert.equal(speedFromTimes(times, ["conv", "conv"], 65536).decode, 1000 / 9.5)
  })

  test("the head's rows past the probe's add to both", () => {
    const types: Array<LayerType> = ["conv", "full_attention"]
    const head = 0.5 + (128000 - 65536) * 0.0001
    const { prompt, decode } = speedFromTimes(times, types, 128000)
    assert.ok(Math.abs(prompt - (1000 * 512) / (20 + head)) < 1e-9)
    assert.ok(Math.abs(decode! - 1000 / (9 + head)) < 1e-9)
  })

  test("a vocabulary no bigger than the probe's adds nothing", () => {
    assert.equal(speedFromTimes(times, ["conv"], 4096).decode, 1000 / 9.5)
  })

  test("a model of one or two layers has no middle", () => {
    const { prompt } = speedFromTimes({ ...times, stackLayers: 0, promptStack: 20 }, ["conv"], 65536)
    assert.equal(prompt, (1000 * 512) / 20.5)
  })

  test("a stack faster than the base (noise) adds nothing rather than less", () => {
    const { prompt } = speedFromTimes({ ...times, promptStack: 19 }, ["conv", "conv", "conv"], 65536)
    assert.equal(prompt, (1000 * 512) / 20.5)
  })
})

describe("middleStack", () => {
  // LFM3 1.7B's 24 layers.
  const lfm3 = [..."ccfcscfcscfcscfcscfcscfc"].map((c): LayerType => (c === "c" ? "conv" : c === "f" ? "full_attention" : "sliding_attention"))
  const count = (types: Array<LayerType>) => Object.fromEntries(["conv", "full_attention", "sliding_attention"].map((t) => [t, types.filter((x) => x === t).length]))

  test("the model's kinds in proportion: 11 conv, 6 full, 5 sliding in the middle → 3, 2 (1.6), 1", () => {
    const stack = middleStack(lfm3, 6)
    assert.equal(stack.length, 6)
    assert.deepEqual(count(stack), { conv: 3, full_attention: 2, sliding_attention: 1 })
  })

  test("in the model's order, each kind's picks spread over its layers", () => {
    // Middle conv layers 2, 10, 18; full 5, 17; sliding 11.
    assert.deepEqual(middleStack(lfm3, 6), ["conv", "full_attention", "conv", "sliding_attention", "full_attention", "conv"])
  })

  test("a model with few middle layers runs them all", () => {
    assert.deepEqual(middleStack(["conv", "full_attention", "conv", "conv"], 6), ["full_attention", "conv"])
    assert.deepEqual(middleStack(["conv"], 6), [])
  })

  test("Qwen3.5: 17 Gated DeltaNet and 5 full-attention layers in the middle → 5 and 1", () => {
    const qwen = Array.from({ length: 24 }, (_, i): LayerType => ((i + 1) % 4 === 0 ? "full_attention" : "linear_attention"))
    const stack = middleStack(qwen, 6)
    assert.equal(stack.filter((t) => t === "linear_attention").length, 5)
    assert.equal(stack.filter((t) => t === "full_attention").length, 1)
  })
})

const config: ModelConfig = {
  format: "gguf",
  hiddenSize: 2048,
  intermediateSize: 7680,
  numLayers: 4,
  numHeads: 32,
  numKvHeads: 8,
  headDim: 64,
  vocabSize: 65536,
  layerTypes: ["conv", "full_attention", "conv", "sliding_attention"],
  slidingWindow: 512,
  contextLength: 128000,
  ropeThetaFull: 1e6,
  ropeThetaSliding: 1e4,
  ropeDimsFull: 32,
  ropeDimsSliding: 64,
  normEps: 1e-5,
  convKernel: 7,
  attentionGate: true,
  bosTokenId: 1,
  eosTokenId: 2,
  padTokenId: 0
}
const q4 = { bits: 4 as const, zeros: false, raw: false }
const shape: SpeedShape = {
  config,
  kinds: [
    { type: "conv", matrices: { inProj: q4, outProj: q4, gateUp: q4, down: q4 } },
    { type: "full_attention", matrices: { qkvg: q4, o: q4, gateUp: q4, down: q4 } },
    { type: "sliding_attention", matrices: { qkvg: q4, o: q4, gateUp: q4, down: q4 } }
  ],
  embedding: { bits: 6, zeros: false, raw: false }
}

describe("the probe model", () => {
  test("one layer of each kind, the head cut to PROBE_VOCAB rows, the rest the model's", () => {
    const probe = probeConfig({ ...shape, config: { ...config, vocabSize: 128000 } })
    assert.equal(probe.numLayers, 3)
    assert.deepEqual(probe.layerTypes, ["conv", "full_attention", "sliding_attention"])
    assert.equal(probe.vocabSize, PROBE_VOCAB)
    assert.equal(probe.hiddenSize, config.hiddenSize)
    assert.equal(probe.intermediateSize, config.intermediateSize)
  })

  test("a smaller vocabulary is kept", () => {
    assert.equal(probeConfig({ ...shape, config: { ...config, vocabSize: 4096 } }).vocabSize, 4096)
  })

  test("matrix shapes as the loaders build them", () => {
    assert.deepEqual(matrixShapes(config, "conv").inProj, { N: 3 * 2048, K: 2048 })
    // q (32 · 64) + k, v (8 · 64 each) + one gate row per head.
    assert.deepEqual(matrixShapes(config, "full_attention").qkvg, { N: 2048 + 1024 + 32, K: 2048 })
    assert.deepEqual(matrixShapes(config, "sliding_attention").gateUp, { N: 2 * 7680, K: 2048 })
    const linear = { ...config, gatePerDim: true, linear: { numKeyHeads: 16, numValueHeads: 16, keyHeadDim: 128, valueHeadDim: 128, convKernel: 4 } }
    assert.deepEqual(matrixShapes(linear, "linear_attention").inProj, { N: 2 * 2048 + 2 * 2048, K: 2048 })
    assert.deepEqual(matrixShapes(linear, "linear_attention").ba, { N: 32, K: 2048 })
    assert.deepEqual(matrixShapes(linear, "full_attention").qkvg, { N: 2 * 2048 + 1024, K: 2048 })
  })

  test("the prompt is one of the worker's prompt batches", () => {
    assert.ok(PROMPT_TOKENS <= PROMPT_BATCH)
  })
})

describe("speedKey", () => {
  test("GPU, browser, model and engine code", () => {
    const key = speedKey("Apple M3", "UA/1", "LiquidAI/LFM3-1.7B-GGUF", "abc.def")
    assert.equal(key, "Apple M3|UA/1|LiquidAI/LFM3-1.7B-GGUF|abc.def")
    assert.notEqual(speedKey("Apple M3", "UA/1", "m"), speedKey("Apple M3", "UA/2", "m"))
    assert.notEqual(speedKey("Apple M3", "UA/1", "m"), speedKey("Mali", "UA/1", "m"))
  })
})

/** predictHere's dependencies, measuring with `times` and counting it. */
function fakeDeps(notes = new Map<string, SpeedNote>(), renderer = "Apple M3") {
  let measured = 0
  const deps: SpeedDeps = {
    renderer: () => renderer,
    userAgent: "UA/1",
    store: {
      read: async (key) => notes.get(key),
      write: async (key, note) => void notes.set(key, note)
    },
    measure: async () => {
      measured++
      await new Promise((r) => setTimeout(r, 5))
      return { shape, report: { times, shapeMs: 1, measureMs: 2, loadMs: 1, loadParts: {}, programs: 0, peakBytes: 0, phases: { prompt: 0, decode: 0, head: 0 } } }
    }
  }
  return { deps, notes, measured: () => measured }
}

describe("predictHere's memory", () => {
  test("measures once, then answers from the note", async () => {
    const f = fakeDeps()
    const first = await predictHere("m", {}, f.deps)
    assert.equal(first.report.cached, false)
    assert.equal(f.measured(), 1)
    const second = await predictHere("m", {}, f.deps)
    assert.equal(second.report.cached, true)
    assert.deepEqual(second.prediction, first.prediction)
    assert.equal(f.measured(), 1)
    const [key] = f.notes.keys()
    assert.ok(key.startsWith("Apple M3|UA/1|m|"))
  })

  test("force measures again and replaces the note", async () => {
    const f = fakeDeps()
    await predictHere("m", {}, f.deps)
    f.notes.set([...f.notes.keys()][0], { prompt: 1, decode: 1, at: 0 })
    const again = await predictHere("m", { force: true }, f.deps)
    assert.equal(again.report.cached, false)
    assert.equal(f.measured(), 2)
    assert.ok([...f.notes.values()][0].prompt > 1)
  })

  test("another GPU or model measures for itself", async () => {
    const notes = new Map<string, SpeedNote>()
    const a = fakeDeps(notes, "Apple M3")
    const b = fakeDeps(notes, "PowerVR")
    await predictHere("m", {}, a.deps)
    await predictHere("m", {}, b.deps)
    await predictHere("n", {}, a.deps)
    assert.equal(a.measured() + b.measured(), 3)
    assert.equal(notes.size, 3)
  })

  test("calls at once share one measurement", async () => {
    const f = fakeDeps()
    const [x, y] = await Promise.all([predictHere("m", {}, f.deps), predictHere("m", {}, f.deps)])
    assert.equal(f.measured(), 1)
    assert.deepEqual(x.prediction, y.prediction)
  })

  test("a failed measurement isn't remembered", async () => {
    const f = fakeDeps()
    const failing: SpeedDeps = { ...f.deps, measure: async () => { throw new Error("no GPU") } }
    await assert.rejects(predictHere("m", {}, failing), /no GPU/)
    assert.equal(f.notes.size, 0)
    await predictHere("m", {}, f.deps)
    assert.equal(f.measured(), 1)
  })
})

describe("shapes without weights", () => {
  test("an ONNX model's from its graph and config: each kind's storage, as loadOnnx stores it", async () => {
    const { readFileSync } = await import("node:fs")
    const { join } = await import("node:path")
    const { onnxSpeedShape } = await import("./speed-probe.js")
    const dir = join(process.cwd(), "src/test-fixtures/lfm2-350m-onnx")
    const shape = onnxSpeedShape(new Uint8Array(readFileSync(join(dir, "model_q4f16.onnx"))), JSON.parse(readFileSync(join(dir, "config.json"), "utf8")))
    assert.equal(shape.config.format, "onnx")
    assert.deepEqual(shape.kinds.map((k) => k.type), [...new Set(shape.config.layerTypes)])
    const conv = shape.kinds.find((k) => k.type === "conv")!
    assert.deepEqual(Object.keys(conv.matrices).sort(), ["down", "gateUp", "inProj", "outProj"])
    const attention = shape.kinds.find((k) => k.type !== "conv")!
    assert.deepEqual(Object.keys(attention.matrices).sort(), ["down", "gateUp", "o", "qkvg"])
    // 4-bit MatMulNBits rows, handed to the engine as they are.
    assert.deepEqual(conv.matrices.gateUp, { bits: 4, zeros: conv.matrices.gateUp.zeros, raw: true })
    assert.equal(shape.embedding.bits === 4 ? shape.embedding.raw : !shape.embedding.raw, true)
    for (const k of shape.kinds)
      for (const [role, { N, K }] of Object.entries(matrixShapes(shape.config, k.type))) {
        assert.ok(k.matrices[role], `${k.type} ${role}`)
        assert.equal(N % 4, 0)
        assert.equal(K % 32, 0)
      }
  })
})

/** A GGUF header's bytes (version 3) with the given scalar metadata and tensors, and no tensor data. */
function ggufHeaderBytes(metadata: Record<string, string | number>, tensors: Array<{ name: string; dims: Array<number>; type: number }>): Uint8Array {
  const bytes: Array<number> = []
  const u32 = (v: number) => bytes.push(v & 255, (v >> 8) & 255, (v >> 16) & 255, (v >>> 24) & 255)
  const u64 = (v: number) => (u32(v), u32(0))
  const text = (s: string) => {
    const encoded = new TextEncoder().encode(s)
    u64(encoded.length)
    bytes.push(...encoded)
  }
  bytes.push(...new TextEncoder().encode("GGUF"))
  u32(3)
  u64(tensors.length)
  u64(Object.keys(metadata).length)
  for (const [key, value] of Object.entries(metadata)) {
    text(key)
    if (typeof value === "string") (u32(8), text(value))
    else if (Number.isInteger(value)) (u32(4), u32(value))
    else (u32(6), bytes.push(...new Uint8Array(new Float32Array([value]).buffer)))
  }
  for (const t of tensors) {
    text(t.name)
    u32(t.dims.length)
    for (const d of t.dims) u64(d)
    u32(t.type)
    u64(0)
  }
  while (bytes.length % 32) bytes.push(0)
  return Uint8Array.from(bytes)
}

test("a GGUF's shape from its header alone: each kind's tensor types, the embedding's", async () => {
  const { ggufSpeedShape } = await import("./speed-probe.js")
  const { GgmlType } = await import("./gguf.js")
  const layer = (i: number) => [
    { name: `blk.${i}.attn_q.weight`, dims: [64, 64], type: GgmlType.Q4_0 },
    { name: `blk.${i}.attn_output.weight`, dims: [64, 64], type: GgmlType.Q8_0 },
    { name: `blk.${i}.ffn_gate.weight`, dims: [64, 128], type: GgmlType.Q4_0 },
    { name: `blk.${i}.ffn_down.weight`, dims: [128, 64], type: GgmlType.Q4_0 }
  ]
  const bytes = ggufHeaderBytes(
    {
      "general.architecture": "lfm2",
      "lfm2.block_count": 2,
      "lfm2.embedding_length": 64,
      "lfm2.attention.head_count": 2,
      "lfm2.attention.head_count_kv": 1,
      "lfm2.attention.key_length": 32,
      "lfm2.rope.freq_base": 1000000,
      "lfm2.attention.layer_norm_rms_epsilon": 0.00001,
      "lfm2.shortconv.l_cache": 3,
      "lfm2.feed_forward_length": 128,
      "lfm2.context_length": 1024
    },
    [{ name: "token_embd.weight", dims: [64, 256], type: GgmlType.Q8_0 }, ...layer(0), ...layer(1)]
  )
  let read = 0
  const shape = await ggufSpeedShape({
    size: bytes.length + (1 << 20),
    read: async (offset, length) => {
      read += length
      const out = new Uint8Array(length)
      out.set(bytes.subarray(offset, offset + length))
      return out
    }
  })
  assert.deepEqual(shape.config.layerTypes, ["full_attention", "full_attention"])
  assert.equal(shape.config.vocabSize, 256)
  assert.deepEqual(shape.kinds, [
    {
      type: "full_attention",
      matrices: {
        gateUp: { bits: 4, zeros: false, raw: false },
        down: { bits: 4, zeros: false, raw: false },
        qkvg: { bits: 4, zeros: false, raw: false },
        o: { bits: 8, zeros: false, raw: false }
      }
    }
  ])
  assert.deepEqual(shape.embedding, { bits: 8, zeros: false, raw: false })
  // The header's bytes only, never a weight's.
  assert.ok(read <= bytes.length + (2 << 20))
})
