import assert from "node:assert/strict"
import { test } from "node:test"
import { toChatMessages, userContent } from "./chat-completions.js"
import { Engine, type EngineImage } from "./engine.js"
import { asRepo, fakeHub } from "./fake-hub.test.js"
import { imageProcessorConfig, patchPlace, pixelBudget, smartResize, toPatches } from "./image.js"
import { dequantizeTable } from "./load-vision.js"
import { chatImages, countImageSlots, expandImageTokens, fetchImage, ImageCache, imageKey, type PreparedImage } from "./prompt-images.js"
import { hubModelMeta } from "./resolve-model.js"
import { positionEmbeddings, ropeTable, VisionEncoder, visionWeightBytes, visionWorkBytes } from "./vision.js"
import type { VisionConfig, VisionLinear, VisionPiece } from "./load-vision.js"
import type { Gpu, Texture } from "./gpu/gl.js"
import { storedModelFiles } from "./model-files.js"

/** Qwen3.5's preprocessor_config.json. */
const QWEN = imageProcessorConfig({
  size: { longest_edge: 16777216, shortest_edge: 65536 },
  patch_size: 16,
  temporal_patch_size: 2,
  merge_size: 2,
  image_mean: [0.5, 0.5, 0.5],
  image_std: [0.5, 0.5, 0.5]
})

test("images resize to whole 32-pixel blocks within the pixel budget, as Qwen2-VL's smart_resize", () => {
  assert.deepEqual(smartResize(400, 400, 32, 65536, 640 * 640), { height: 416, width: 416 })
  assert.deepEqual(smartResize(320, 512, 32, 65536, 640 * 640), { height: 320, width: 512 })
  // Too small: scaled up past the minimum.
  assert.deepEqual(smartResize(100, 100, 32, 65536, 640 * 640), { height: 256, width: 256 })
  // Too large: scaled down under the maximum, keeping the aspect ratio.
  const big = smartResize(3000, 4000, 32, 65536, 640 * 640)
  assert.ok(big.height * big.width <= 640 * 640 && big.height % 32 === 0 && big.width % 32 === 0)
  assert.ok(Math.abs(big.width / big.height - 4 / 3) < 0.05)
  assert.throws(() => smartResize(10, 3000, 32, 65536, 640 * 640), RangeError)
})

test("detail sets the pixel budget, inside the processor's own", () => {
  assert.deepEqual(pixelBudget("low", QWEN), { minPixels: 65536, maxPixels: 65536 })
  assert.deepEqual(pixelBudget(undefined, QWEN), { minPixels: 65536, maxPixels: 640 * 640 })
  assert.deepEqual(pixelBudget("high", QWEN), { minPixels: 65536, maxPixels: 1024 * 1024 })
})

test("patches come in merge order, channel, frame, y, x: Transformers.js's reshape and permute", () => {
  const height = 64
  const width = 96
  const rgba = Uint8Array.from({ length: height * width * 4 }, (_, i) => (i * 7919) % 251)
  const got = toPatches(rgba, height, width, QWEN)
  assert.equal(got.gridH, 4)
  assert.equal(got.gridW, 6)
  // The reference: pixel (c, y, x) normalized, the frame repeated, viewed as
  // (t, T, C, H/m, m, p, W/m, m, p) and permuted (0, 3, 6, 4, 7, 2, 1, 5, 8).
  const p = 16, m = 2, T = 2
  const value = (c: number, y: number, x: number) => (rgba[(y * width + x) * 4 + c] / 255 - 0.5) / 0.5
  const want: Array<number> = []
  for (let bh = 0; bh < height / p / m; bh++)
    for (let bw = 0; bw < width / p / m; bw++)
      for (let mh = 0; mh < m; mh++)
        for (let mw = 0; mw < m; mw++)
          for (let c = 0; c < 3; c++)
            for (let f = 0; f < T; f++)
              for (let py = 0; py < p; py++)
                for (let px = 0; px < p; px++)
                  want.push(value(c, (bh * m + mh) * p + py, (bw * m + mw) * p + px))
  assert.equal(got.pixels.length, want.length)
  for (let i = 0; i < want.length; i++) assert.ok(Math.abs(got.pixels[i] - want[i]) < 1e-6, `value ${i}`)
  // Patch 5 of a 6-wide grid: block 1 (columns 2..3), its second patch.
  assert.deepEqual(patchPlace(5, 6, 2), { row: 0, col: 3 })
  assert.deepEqual(patchPlace(13, 6, 2), { row: 2, col: 1 })
})

/**
 * Transformers.js's Qwen2-VL get_rope_index for one sequence: t, h, w
 * positions of each token, images given as merged grids in order.
 */
function referencePositions(ids: Array<number>, imageToken: number, grids: Array<[number, number]>) {
  const out: Array<[number, number, number]> = []
  let st = 0
  let next = 0
  let start = 0
  while (st < ids.length) {
    const ed = ids.indexOf(imageToken, st)
    const end = ed < 0 ? ids.length : ed
    for (let i = st; i < end; i++) out.push([start + i - st, start + i - st, start + i - st])
    start += end - st
    if (ed < 0) break
    const [h, w] = grids[next++]
    for (let i = 0; i < h * w; i++) out.push([start, start + Math.floor(i / w), start + (i % w)])
    start += Math.max(h, w)
    st = ed + h * w
  }
  return out
}

test("the engine's rotary positions are Qwen2-VL's: images in 3D, the text after them shifted", () => {
  const IMG = 9
  const images = [
    { key: -1, height: 3, width: 5 },
    { key: -2, height: 4, width: 2 }
  ]
  const prompt = [1, 2, 7, IMG, 8, 4, 5, 7, IMG, 8, 6]
  const { tokens, starts } = expandImageTokens(prompt, { start: 7, pad: IMG, end: 8 }, images)
  const engine = Object.create(Engine.prototype) as Engine
  Object.assign(engine, { config: { hiddenSize: 4 }, images: [] })
  engine.setImages(
    starts.map((start, i): EngineImage => ({ start, height: images[i].height, width: images[i].width, features: null }))
  )
  const want = referencePositions(tokens, IMG, images.map((g) => [g.height, g.width]))
  // Past the prompt too: writing continues from the text's positions.
  for (let i = 0; i < 3; i++) {
    const last = want[want.length - 1][0] + 1
    want.push([last, last, last])
  }
  assert.deepEqual(
    want.map((_, pos) => engine.ropeAt(pos)),
    want
  )
  // Without images, positions are the cache's.
  engine.setImages([])
  assert.deepEqual(engine.ropeAt(17), [17, 17, 17])
  assert.throws(() => engine.setImages([{ start: 4, height: 2, width: 2, features: null }, { start: 6, height: 1, width: 1, features: null }]), RangeError)
  assert.throws(() => engine.setImages([{ start: 0, height: 2, width: 2, features: new Float32Array(3) }]), RangeError)
})

test("image tokens expand to their images' tokens, and image rows key by content", () => {
  const images = [
    { key: imageKey(new Uint8Array([1, 2, 3]), 32, 32), height: 1, width: 2 },
    { key: imageKey(new Uint8Array([1, 2, 4]), 32, 32), height: 2, width: 1 }
  ]
  const slot = { start: 7, pad: 9, end: 8 }
  const { tokens, keys, starts } = expandImageTokens([5, 7, 9, 8, 6, 7, 9, 8], slot, images)
  assert.deepEqual(tokens, [5, 7, 9, 9, 8, 6, 7, 9, 9, 8])
  assert.deepEqual(starts, [2, 7])
  assert.deepEqual(keys, [5, 7, images[0].key, images[0].key, 8, 6, 7, images[1].key, images[1].key, 8])
  for (const { key } of images) assert.ok(Number.isSafeInteger(key) && key < 0)
  assert.notEqual(images[0].key, images[1].key)
  // The same pixels at another size are another image.
  assert.notEqual(imageKey(new Uint8Array([1, 2, 3]), 32, 64), images[0].key)
  assert.throws(() => expandImageTokens([7, 9, 8, 7, 9, 8], slot, images.slice(0, 1)))
  assert.throws(() => expandImageTokens([7, 9, 8], slot, images))
  // The pad token written in a message's text (the literal "<|image_pad|>")
  // isn't a slot: it stays one token, and the image still finds its own.
  const literal = [5, 9, 6, 7, 9, 8]
  assert.equal(countImageSlots(literal, slot), 1)
  assert.deepEqual(expandImageTokens(literal, slot, images.slice(0, 1)).tokens, [5, 9, 6, 7, 9, 9, 8])
})

test("a chat's images in template order, and the cache of prepared ones", () => {
  const chat = toChatMessages(
    [
      { role: "system", content: "Be brief." },
      { role: "user", content: [{ type: "image_url", image_url: { url: "data:image/png;base64,AA" } }, { type: "text", text: "What?" }] },
      { role: "assistant", content: "A cat." },
      {
        role: "user",
        content: [
          { type: "text", text: "And " },
          { type: "text", text: "these?" },
          { type: "image_url", image_url: { url: "https://example.com/a.png", detail: "low" } },
          { type: "image_url", image_url: { url: "https://example.com/b.png", detail: "auto" } }
        ]
      }
    ],
    "qwen3.5",
    true
  )
  assert.deepEqual(chat[3].content, [
    { type: "text", text: "And these?" },
    { type: "image", image: "https://example.com/a.png", detail: "low" },
    { type: "image", image: "https://example.com/b.png" }
  ])
  assert.deepEqual(chatImages(chat), [
    { url: "data:image/png;base64,AA" },
    { url: "https://example.com/a.png", detail: "low" },
    { url: "https://example.com/b.png" }
  ])
  // By content; data URLs (whose bytes never change) by URL too, http URLs never.
  const cache = new ImageCache(2)
  const made = (key: number): PreparedImage => ({ key, height: 1, width: 1, pixels: null, features: null })
  cache.set({ url: "data:image/png;base64,AA" }, "auto|1", made(-1))
  cache.set({ url: "https://example.com/b.png" }, "auto|2", made(-2))
  assert.equal(cache.getUrl({ url: "data:image/png;base64,AA" })?.key, -1)
  assert.equal(cache.getUrl({ url: "https://example.com/b.png" }), undefined, "an http URL is fetched again")
  assert.equal(cache.get("auto|2")?.key, -2)
  cache.set({ url: "https://example.com/c.png" }, "auto|3", made(-3))
  assert.equal(cache.get("auto|1"), undefined, "the least recent goes")
  assert.equal(cache.getUrl({ url: "data:image/png;base64,AA" }), undefined, "and its URL with it")
  assert.equal(cache.getUrl({ url: "data:image/png;base64,AA", detail: "low" }), undefined)
})

test("image parts only for models that read images; text parts still join", () => {
  const parts = [{ type: "image_url" as const, image_url: { url: "data:image/png;base64,AA" } }]
  assert.throws(() => userContent(parts, false), { code: "unsupported-parameter" })
  assert.throws(() => userContent([{ type: "image_url", image_url: { url: "file:///x.png" } }], true), TypeError)
  assert.throws(() => userContent([{ type: "image_url", image_url: { url: "data:,", detail: "max" as "low" } }], true), TypeError)
  assert.equal(userContent([{ type: "text", text: "a" }, { type: "text", text: "b" }], false), "ab")
  assert.throws(() => userContent([{ type: "input_audio" } as never], true), { code: "unsupported-parameter" })
})

test("image input is declared by the rule: a vision encoder beside the decoder, for its quantization", async () => {
  fakeHub()
  const qwen = await hubModelMeta("onnx-community/Qwen3.5-2B-ONNX")
  assert.deepEqual(qwen.supports.input, ["text", "image"])
  assert.ok(qwen.files.kind === "onnx")
  assert.deepEqual(qwen.files.vision, {
    graph: { repo: "onnx-community/Qwen3.5-2B-ONNX", path: "onnx/vision_encoder_q4f16.onnx" },
    data: { repo: "onnx-community/Qwen3.5-2B-ONNX", path: "onnx/vision_encoder_q4f16.onnx_data" },
    preprocessor: { repo: "onnx-community/Qwen3.5-2B-ONNX", path: "preprocessor_config.json" }
  })
  // A config without a vision_config the encoder runs: text only.
  fakeHub({
    ...asRepo("onnx-community/Qwen3.5-2B-ONNX", "someone/Qwen3.5-2B-text"),
    "https://huggingface.co/someone/Qwen3.5-2B-text/resolve/main/config.json": { status: 200, body: { model_type: "qwen3_5" } }
  })
  assert.deepEqual((await hubModelMeta("someone/Qwen3.5-2B-text")).supports.input, ["text"])
  // GGUF and LFM builds: text only.
  fakeHub()
  for (const name of ["unsloth/Qwen3.5-2B-GGUF", "onnx-community/LFM3-1.7B-ONNX"])
    assert.deepEqual((await hubModelMeta(name)).supports.input, ["text"], name)
})

test("the position table dequantizes nibble by nibble, low first, minus its block's zero point", () => {
  // One row of 64: block 0 all 9 (zero point 8, scale 0.5), block 1 all 3 (zero 4, scale 2).
  const quant = new Uint8Array(32)
  quant.fill(0x99, 0, 16)
  quant.fill(0x33, 16, 32)
  const out = dequantizeTable(quant, Float32Array.from([0.5, 2]), Uint8Array.from([0x48]), 1, 64)
  assert.deepEqual([out[0], out[31], out[32], out[63]], [0.5, 0.5, -2, -2])
})

test("position embeddings: the table itself at its own size; RoPE angles by row, then column", () => {
  const grid = 4
  const width = 4
  const table = Float32Array.from({ length: grid * grid * width }, (_, i) => i)
  const same = positionEmbeddings(table, grid, width, grid, grid, 2)
  for (let p = 0; p < grid * grid; p++) {
    const { row, col } = patchPlace(p, grid, 2)
    assert.deepEqual(Array.from(same.subarray(p * width, (p + 1) * width)), Array.from(table.subarray((row * grid + col) * width, (row * grid + col + 1) * width)))
  }
  // Half the grid: corners to corners, so patch (1, 1) is the table's (3, 3).
  const half = positionEmbeddings(table, grid, width, 2, 2, 2)
  assert.deepEqual(Array.from(half.subarray(3 * width, 4 * width)), Array.from(table.subarray(15 * width, 16 * width)))
  const inv = Float32Array.from([1, 0.5])
  const cs = ropeTable(inv, 2, 2, 2)
  // Patch 3 (row 1, column 1): angles [1, 0.5, 1, 0.5] (rows, then columns).
  const row = cs.subarray(3 * 8, 4 * 8)
  assert.deepEqual(Array.from(row.subarray(0, 4)), [1, 0.5, 1, 0.5].map(Math.cos).map((v) => Math.fround(v)))
  assert.deepEqual(Array.from(row.subarray(4, 8)), [1, 0.5, 1, 0.5].map(Math.sin).map((v) => Math.fround(v)))
})

test("the pixel cap holds after rounding to whole blocks: portrait, landscape, strips and the GPU's limit", () => {
  const low = 256 * 256
  for (const [h, w] of [[100, 150], [150, 100], [32, 4800], [4800, 32], [33, 33], [1, 1], [255, 257]]) {
    const size = smartResize(h, w, 32, low, low)
    assert.ok(size.height * size.width <= low, `${h}×${w} → ${size.height}×${size.width}`)
    assert.ok(size.height % 32 === 0 && size.width % 32 === 0 && size.height >= 32 && size.width >= 32)
  }
  // A GPU whose textures hold 1,024 patches: at most 1,024 · 256 pixels.
  for (const [h, w] of [[1000, 1000], [600, 2000], [40, 7000]]) {
    const size = smartResize(h, w, 32, 65536, 1024 * 256)
    assert.ok(size.height * size.width <= 1024 * 256, `${h}×${w} → ${size.height}×${size.width}`)
  }
  // Inside the budget nothing changes.
  assert.deepEqual(smartResize(400, 400, 32, 65536, 640 * 640), { height: 416, width: 416 })
})

test("deleting a model removes its vision encoder's files too", () => {
  fakeHub()
  return hubModelMeta("onnx-community/Qwen3.5-2B-ONNX").then((meta) => {
    const paths = storedModelFiles(meta).map((f) => f.path)
    for (const path of ["onnx/vision_encoder_q4f16.onnx", "onnx/vision_encoder_q4f16.onnx_data", "preprocessor_config.json"])
      assert.ok(paths.includes(path), path)
  })
})

test("a stopped image fetch rejects with the stop, not a fetch error", async (t) => {
  t.mock.method(globalThis, "fetch", (_url: string, init?: RequestInit) =>
    new Promise((_, reject) => init?.signal?.addEventListener("abort", () => reject(new DOMException("aborted", "AbortError"))))
  )
  const stop = new AbortController()
  const fetching = fetchImage("https://example.com/slow.png", stop.signal)
  stop.abort(new DOMException("stopped", "AbortError"))
  await assert.rejects(fetching, (error: Error) => error.name === "AbortError" && error.message === "stopped")
})

/** A GPU that only counts texture bytes (live and peak), for the encoder's allocations. */
function countingGpu() {
  let live = 0
  let peak = 0
  let id = 0
  const bytesPer: Record<number, number> = { 0x8d70: 16, 0x8814: 16, 0x881a: 8, 0x8d7c: 4 }
  const textures = new Set<Texture>()
  const gpu = {
    limits: { maxTextureSize: 16384 },
    texture(internal: number, width: number, height: number) {
      const t = { id: id++, internal, width, height, bytes: width * height * bytesPer[internal]! } as Texture
      textures.add(t)
      live += t.bytes
      peak = Math.max(peak, live)
      return t
    },
    delete(t: Texture) {
      if (textures.delete(t)) live -= t.bytes
    },
    program: (source: string) => ({ source }),
    run: () => undefined,
    runAsync: async () => undefined,
    pace: async () => undefined,
    read: async (t: Texture) => new Float32Array(t.width * t.height * 4)
  }
  return { gpu: gpu as unknown as Gpu, live: () => live, peak: () => peak, resetPeak: () => (peak = live) }
}

/** A tiny encoder's pieces (64-wide, one head, one block). */
function tinyEncoder(): Array<VisionPiece> {
  const config: VisionConfig = {
    width: 64, heads: 1, headDim: 64, ffn: 128, depth: 1, outWidth: 128, patchSize: 16, temporal: 2, merge: 2,
    patchValues: 1536, grid: 4, eps: 1e-6, act: "tanh", mergerAct: "erf"
  }
  const linear = (N: number, K: number): VisionLinear => ({
    m: { N, K, bits: 4, quants: new Uint32Array(0), scales: new Uint16Array(N * (K / 32)), zeros: null, raw: new Uint8Array(N * (K / 32) * 16) },
    bias: new Float32Array(N)
  })
  const norm = { weight: new Float32Array(64), bias: new Float32Array(64) }
  return [
    { kind: "config", config },
    { kind: "patch", patch: linear(64, 1536), positions: new Float32Array(16 * 64), invFreq: new Float32Array(16) },
    { kind: "block", index: 0, value: { norm1: norm, qkv: linear(192, 64), proj: linear(64, 64), norm2: norm, fc1: linear(128, 64), fc2: linear(64, 128) } },
    { kind: "merger", norm, linear1: linear(256, 256), linear2: linear(128, 256) }
  ]
}

async function* piecesOf(pieces: Array<VisionPiece>) {
  yield* pieces
}

test("the vision claim is what the encoder allocates: its weights, and its largest image's work at once", async () => {
  const counter = countingGpu()
  const encoder = new VisionEncoder(counter.gpu)
  const pieces = tinyEncoder()
  await encoder.load(piecesOf(pieces))
  const config = (pieces[0] as { config: VisionConfig }).config
  assert.equal(counter.live(), visionWeightBytes(config))
  for (const [gridH, gridW] of [[4, 4], [8, 12], [40, 40]]) {
    counter.resetPeak()
    const weights = counter.live()
    await encoder.encode({ pixels: new Float32Array(gridH * gridW * 1536), gridH, gridW })
    assert.equal(counter.peak() - weights, visionWorkBytes(config, gridH, gridW, 16384), `${gridH}×${gridW}`)
    assert.equal(counter.live(), weights, "its work textures are freed")
  }
  encoder.dispose()
  assert.equal(counter.live(), 0)
})

test("a stopped encode rejects, freeing its work textures", async () => {
  const counter = countingGpu()
  const encoder = new VisionEncoder(counter.gpu)
  await encoder.load(piecesOf(tinyEncoder()))
  const weights = counter.live()
  const stop = new AbortController()
  let passes = 0
  ;(counter.gpu as unknown as { runAsync: () => Promise<void> }).runAsync = async () => {
    if (++passes === 5) stop.abort(new DOMException("stopped", "AbortError"))
  }
  await assert.rejects(encoder.encode({ pixels: new Float32Array(16 * 1536), gridH: 4, gridW: 4 }, stop.signal), { name: "AbortError" })
  assert.equal(counter.live(), weights)
  assert.ok(passes < 10, "it stops at the next pass")
})
