import assert from "node:assert/strict"
import { test } from "node:test"
import { toChatMessages, userContent } from "./chat-completions.js"
import type { ChatMessage } from "./tokenizer.js"
import { Engine, type EngineImage } from "./engine.js"
import { asRepo, fakeHub } from "./fake-hub.test.js"
import { imageProcessorConfig, patchPlace, pixelBudget, planViews, smartResize, tileGrid, tileResize, toPatches } from "./image.js"
import { dequantizeTable } from "./load-vision.js"
import { chatImages, countImageSlots, expandImageTokens, fetchImage, ImageCache, imageKey, imagePlaceholders, imageTokenCount, type PreparedImage } from "./prompt-images.js"
import { hubModelMeta } from "./resolve-model.js"
import { positionEmbeddings, ropeTable, VisionEncoder, visionStreamBytes, visionWeightBytes, visionWorkBytes } from "./vision.js"
import { newVisionMatrix, type VisionConfig, type VisionLinear, type VisionPiece } from "./vision-weights.js"

/** An image read as one view of height × width tokens. */
const oneView = (image: { key: number; height: number; width: number }) => ({ key: image.key, views: [{ height: image.height, width: image.width }] })
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
  const { tokens, starts } = expandImageTokens(prompt, { kind: "wrapped", start: 7, pad: IMG, end: 8 }, images.map(oneView))
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
  // LFM2-VL and LFM3-VL: an image's rows keep their cache positions, like any token.
  engine.setImages(
    starts.map((start, i): EngineImage => ({ start, height: images[i].height, width: images[i].width, features: null })),
    "sequence"
  )
  for (const pos of [0, starts[0], starts[0] + 7, starts[1] + 3, tokens.length]) assert.deepEqual(engine.ropeAt(pos), [pos, pos, pos])
  assert.throws(() => engine.setImages([{ start: 4, height: 2, width: 2, features: null }, { start: 6, height: 1, width: 1, features: null }]), RangeError)
  assert.throws(() => engine.setImages([{ start: 0, height: 2, width: 2, features: new Float32Array(3) }]), RangeError)
})

test("image tokens expand to their images' tokens, and image rows key by content", () => {
  const images = [
    { key: imageKey(new Uint8Array([1, 2, 3]), 32, 32), height: 1, width: 2 },
    { key: imageKey(new Uint8Array([1, 2, 4]), 32, 32), height: 2, width: 1 }
  ]
  const slot = { kind: "wrapped", start: 7, pad: 9, end: 8 } as const
  const views = images.map(oneView)
  const { tokens, keys, starts } = expandImageTokens([5, 7, 9, 8, 6, 7, 9, 8], slot, views)
  assert.deepEqual(tokens, [5, 7, 9, 9, 8, 6, 7, 9, 9, 8])
  assert.deepEqual(starts, [2, 7])
  assert.deepEqual(keys, [5, 7, images[0].key, images[0].key, 8, 6, 7, images[1].key, images[1].key, 8])
  for (const { key } of images) assert.ok(Number.isSafeInteger(key) && key < 0)
  assert.notEqual(images[0].key, images[1].key)
  // The same pixels at another size are another image.
  assert.notEqual(imageKey(new Uint8Array([1, 2, 3]), 32, 64), images[0].key)
  assert.throws(() => expandImageTokens([7, 9, 8, 7, 9, 8], slot, views.slice(0, 1)))
  assert.throws(() => expandImageTokens([7, 9, 8], slot, views))
  // The pad token written in a message's text (the literal "<|image_pad|>")
  // isn't a slot: it stays one token, and the image still finds its own.
  const literal = [5, 9, 6, 7, 9, 8]
  assert.equal(countImageSlots(literal, slot), 1)
  assert.deepEqual(expandImageTokens(literal, slot, views.slice(0, 1)).tokens, [5, 9, 6, 7, 9, 9, 8])
})

test("LFM2-VL's bare image token becomes its start, each tile after its place, the thumbnail and its end", () => {
  const IMG = 9, START = 20, END = 21, THUMB = 22
  const slot = { kind: "bare", pad: IMG, start: START, end: END, thumbnail: THUMB, tile: (row: number, col: number) => 100 + row * 10 + col } as const
  // One view: no place tokens.
  const single = { key: -5, views: [{ height: 1, width: 2 }] }
  const one = expandImageTokens([1, IMG, 2], slot, [single])
  assert.deepEqual(one.tokens, [1, START, IMG, IMG, END, 2])
  assert.deepEqual(one.keys, [1, START, -5, -5, END, 2])
  assert.deepEqual(one.starts, [2])
  assert.equal(imageTokenCount(single, slot), 4)
  // Tiles (1 × 2 here), row by row, then the thumbnail.
  const tiled = {
    key: -6,
    views: [
      { height: 1, width: 1, place: { row: 1, col: 1 } },
      { height: 1, width: 1, place: { row: 1, col: 2 } },
      { height: 1, width: 2, place: "thumbnail" as const }
    ]
  }
  const many = expandImageTokens([IMG, 3, IMG], slot, [tiled, single])
  assert.deepEqual(many.tokens, [START, 111, IMG, 112, IMG, THUMB, IMG, IMG, END, 3, START, IMG, IMG, END])
  assert.deepEqual(many.keys, [START, 111, -6, 112, -6, THUMB, -6, -6, END, 3, START, -5, -5, END])
  assert.deepEqual(many.starts, [2, 4, 6, 11])
  assert.equal(imageTokenCount(tiled, slot), 9)
  assert.equal(countImageSlots([IMG, 3, IMG], slot), 2)
  assert.throws(() => expandImageTokens([IMG], slot, []))
})

test("LFM2-VL's processor: one view up to 256 tokens, then a grid of 512-pixel tiles and a thumbnail", () => {
  const LFM2 = imageProcessorConfig({
    image_processor: {
      do_image_splitting: true, downsample_factor: 2, encoder_patch_size: 16, image_mean: [0.5, 0.5, 0.5], image_std: [0.5, 0.5, 0.5],
      max_image_tokens: 256, max_pixels_tolerance: 2.0, max_tiles: 10, min_image_tokens: 64, min_tiles: 2, tile_size: 512, use_thumbnail: true
    }
  })
  assert.equal(LFM2.layout, "channel-last")
  assert.equal(LFM2.temporalPatchSize, 1)
  assert.deepEqual(LFM2.tiles, { size: 512, min: 2, max: 10, tolerance: 2, thumbnail: true })
  // Transformers' smart_resize: whole 32-pixel blocks, 64 to 256 tokens.
  assert.deepEqual(tileResize(400, 400, LFM2), { height: 416, width: 416 })
  assert.deepEqual(tileResize(100, 100, LFM2), { height: 256, width: 256 })
  assert.deepEqual(tileResize(600, 800, LFM2), { height: 416, width: 576 })
  // An image within twice the pixel limit is one view.
  assert.deepEqual(planViews(400, 400, undefined, LFM2, 16384), [{ height: 416, width: 416 }])
  assert.deepEqual(planViews(700, 700, undefined, LFM2, 16384), [{ height: 512, width: 512 }])
  // A larger one: the grid nearest its aspect ratio, of two as near the larger when it has the pixels.
  assert.deepEqual(tileGrid(768, 1024, LFM2.tiles!), { cols: 3, rows: 2 })
  assert.deepEqual(tileGrid(1024, 1024, LFM2.tiles!), { cols: 2, rows: 2 })
  assert.deepEqual(tileGrid(2000, 2000, LFM2.tiles!), { cols: 3, rows: 3 })
  assert.deepEqual(tileGrid(3024, 4032, LFM2.tiles!), { cols: 3, rows: 2 })
  assert.deepEqual(tileGrid(512, 4096, LFM2.tiles!), { cols: 8, rows: 1 })
  const views = planViews(768, 1024, undefined, LFM2, 16384)
  assert.equal(views.length, 7)
  assert.deepEqual(views[0], { height: 512, width: 512, from: { height: 1024, width: 1536, x: 0, y: 0 }, place: { row: 1, col: 1 } })
  assert.deepEqual(views[4], { height: 512, width: 512, from: { height: 1024, width: 1536, x: 512, y: 512 }, place: { row: 2, col: 2 } })
  assert.deepEqual(views[6], { height: 416, width: 576, place: "thumbnail" })
  // detail "low": the whole image as one view, however large.
  assert.deepEqual(planViews(768, 1024, "low", LFM2, 16384), [{ height: 416, width: 576 }])
  // Splitting off (or one tile at most): one view.
  const whole = imageProcessorConfig({ encoder_patch_size: 16, downsample_factor: 2, min_image_tokens: 64, max_image_tokens: 256, do_image_splitting: false, image_mean: [0.5, 0.5, 0.5], image_std: [0.5, 0.5, 0.5] })
  assert.equal(whole.tiles, undefined)
  assert.deepEqual(planViews(3000, 4000, undefined, whole, 16384), [{ height: 416, width: 576 }])
  // Its patch rows are y, x, channel.
  const rgba = Uint8Array.from({ length: 32 * 32 * 4 }, (_, i) => (i * 31) % 251)
  const got = toPatches(rgba, 32, 32, LFM2)
  assert.equal(got.pixels.length, 4 * 768)
  const value = (c: number, y: number, x: number) => (rgba[(y * 32 + x) * 4 + c] / 255 - 0.5) / 0.5
  // Patch 3 (merge order: row 1, column 1), pixel (y 5, x 7), channel 2.
  assert.ok(Math.abs(got.pixels[3 * 768 + (5 * 16 + 7) * 3 + 2] - value(2, 16 + 5, 16 + 7)) < 1e-6)
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
  const made = (key: number): PreparedImage => ({ key, views: [{ height: 1, width: 1, pixels: null, features: null }] })
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
  // A GGUF build: the repository's projector (the float16 one where there's no Q8_0), and the processor config where its tokenizer is.
  fakeHub()
  const gguf = await hubModelMeta("unsloth/Qwen3.5-2B-GGUF")
  assert.deepEqual(gguf.supports.input, ["text", "image"])
  assert.deepEqual(gguf.files.vision, {
    projector: { repo: "unsloth/Qwen3.5-2B-GGUF", path: "mmproj-F16.gguf" },
    preprocessor: { repo: "Qwen/Qwen3.5-2B", path: "preprocessor_config.json" }
  })
  assert.equal(gguf.projectorChecked, true)
  // A projector of a type the encoder doesn't run, and a build without one: text only.
  fakeHub(asRepo("unsloth/Qwen3.5-2B-GGUF", "someone/Other-Qwen3.5-2B-GGUF"))
  const other = await hubModelMeta("someone/Other-Qwen3.5-2B-GGUF")
  assert.deepEqual([other.supports.input, other.files.vision, other.projectorChecked], [["text"], undefined, true])
  fakeHub()
  assert.deepEqual((await hubModelMeta("onnx-community/LFM3-1.7B-ONNX")).supports.input, ["text"])
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
  const draws: Array<string> = []
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
    runAsync: async (pass: { program: { source: string }; rect: unknown; uniforms: unknown }) => {
      draws.push(JSON.stringify([pass.program.source.length, pass.rect, pass.uniforms]))
    },
    pace: async () => undefined,
    read: async (t: Texture) => new Float32Array(t.width * t.height * 4)
  }
  return { gpu: gpu as unknown as Gpu, live: () => live, peak: () => peak, resetPeak: () => (peak = live), draws }
}

const linear = (N: number, K: number, bits: 4 | 8 = 4, bias = true): VisionLinear => ({ m: newVisionMatrix(N, K, bits, false), bias: bias ? new Float32Array(N) : null })

/** A tiny Qwen-style encoder's pieces (64-wide, one head, one block; 4-bit). */
function tinyEncoder(): Array<VisionPiece> {
  const config: VisionConfig = {
    width: 64, heads: 1, kvHeads: 1, headDim: 64, ffn: 128, mergerWidth: 256, outWidth: 128, merge: 2, patchValues: 1536,
    layers: ["attention"], norm: "layer", eps: 1e-6, mergerEps: 1e-6, mlp: "tanh", mergerAct: "erf",
    grid: 4, interpolation: "corners", rope: true, qkNorm: false, positions: "grid"
  }
  const norm = { weight: new Float32Array(64), bias: new Float32Array(64) }
  return [
    { kind: "config", config },
    { kind: "patch", patch: linear(64, 1536), positions: new Float32Array(16 * 64), invFreq: new Float32Array(16) },
    { kind: "block", index: 0, value: { type: "attention", norm1: norm, qkv: linear(192, 64), proj: linear(64, 64), norm2: norm, fc1: linear(128, 64), fc2: linear(64, 128) } },
    { kind: "merger", norm, linear1: linear(256, 256), linear2: linear(128, 256) }
  ]
}

/** A tiny LFM3-VL-style tower's pieces (64-wide: a conv block and a grouped-query attention block; 8-bit, an MLP width that isn't whole blocks). */
function tinyTower(): Array<VisionPiece> {
  const config: VisionConfig = {
    width: 64, heads: 4, kvHeads: 2, headDim: 16, ffn: 80, mergerWidth: 256, outWidth: 128, merge: 2, patchValues: 1536,
    layers: ["conv", "attention"], norm: "rms", eps: 1e-5, mergerEps: 1e-6, mlp: "swiglu", mergerAct: "erf",
    grid: 0, interpolation: "corners", rope: true, qkNorm: true, positions: "sequence"
  }
  const rms = { weight: new Float32Array(64), bias: null }
  const mlp = { norm2: rms, fc1: linear(160, 64, 8, false), fc2: linear(64, 80, 8, false) }
  return [
    { kind: "config", config },
    { kind: "patch", patch: linear(64, 1536, 8, false), invFreq: new Float32Array(4), norm: rms },
    { kind: "block", index: 0, value: { type: "conv", norm1: rms, inProj: linear(192, 64, 8, false), kernel: new Float32Array(64 * 9), outProj: linear(64, 64, 8, false), ...mlp } },
    {
      kind: "block",
      index: 1,
      value: { type: "attention", norm1: rms, qkv: linear(128, 64, 8, false), qNorm: new Float32Array(16), kNorm: new Float32Array(16), proj: linear(64, 64, 8, false), ...mlp }
    },
    { kind: "merger", norm: { weight: new Float32Array(64), bias: new Float32Array(64) }, linear1: linear(256, 256, 8), linear2: linear(128, 256, 8) }
  ]
}

async function* piecesOf(pieces: Array<VisionPiece>) {
  yield* pieces
}

test("a literal <image> in a message's text isn't an attachment: only the placeholders the template writes for image parts are slots", async () => {
  const { loadTokenizer } = await import("./tokenizer.js")
  const { readFileSync } = await import("node:fs")
  // Liquid AI's chat template (LiquidAI/LFM2.5-VL-450M) and its special tokens' names, over a vocabulary of one text token.
  const specials = ["<|startoftext|>", "<|im_start|>", "<|im_end|>", "<|pad|>", "<image>", "<|image_start|>", "<|image_end|>", "<|img_thumbnail|>"]
  const tokenizer = loadTokenizer({
    tokenizerJson: {
      model: { type: "BPE", vocab: { x: 0 }, merges: [], unk_token: "x" },
      normalizer: null,
      pre_tokenizer: null,
      post_processor: null,
      decoder: null,
      added_tokens: specials.map((content, i) => ({ id: 10 + i, content, special: true, normalized: false }))
    },
    tokenizerConfig: {
      chat_template: readFileSync("src/test-fixtures/lfm2.5-vl-450m-onnx/chat_template.jinja", "utf8"),
      bos_token: "<|startoftext|>",
      eos_token: "<|im_end|>",
      pad_token: "<|pad|>"
    }
  })
  const id = (token: string) => tokenizer.tokenToId(token)!
  const IMG = id("<image>")
  const slot = { kind: "bare" as const, pad: IMG, start: id("<|image_start|>"), end: id("<|image_end|>"), thumbnail: id("<|img_thumbnail|>"), tile: () => 99 }
  const picture = { type: "image" as const, image: "data:image/png;base64,AAAA" }
  const text = (t: string) => ({ type: "text" as const, text: t })
  const view = { key: -5, views: [{ height: 1, width: 2 }] }
  const render = (input: { chat: Array<ChatMessage> }) => tokenizer.applyChatTemplate(input.chat, { addGenerationPrompt: true })
  const cases: Array<[string, Array<ChatMessage>, Array<boolean>]> = [
    ["no literal", [{ role: "user", content: [picture, text("what is this?")] }], [true]],
    ["a literal after the image, in the same message", [{ role: "user", content: [picture, text("explain the literal <image> marker")] }], [true, false]],
    ["a literal right after the image", [{ role: "user", content: [picture, text("<image>")] }], [true, false]],
    ["a literal before the image", [{ role: "user", content: [text("the <image> marker, then:"), picture] }], [false, true]],
    [
      "a literal earlier in the conversation",
      [
        { role: "user", content: "what does <image> mean in a prompt?" },
        { role: "assistant", content: "It marks where <image> rows go." },
        { role: "user", content: [text("and this one?"), picture] }
      ],
      [false, false, true]
    ],
    ["two images around a literal", [{ role: "user", content: [picture, text("<image><image>"), picture] }], [true, false, false, true]]
  ]
  for (const [name, chat, expected] of cases) {
    const prompt = tokenizer.encodeChat(chat, { addGenerationPrompt: true })
    const written = imagePlaceholders({ chat }, "<image>", render)
    assert.deepEqual(written, expected, name)
    // One placeholder token per placeholder in the rendered text, in its order.
    assert.equal(prompt.filter((t) => t === IMG).length, expected.length, name)
    const images = expected.filter(Boolean).length
    assert.equal(chatImages(chat).length, images, name)
    assert.equal(countImageSlots(prompt, slot, written), images, name)
    // Without the template's word for it, every placeholder token counts: what failed before.
    assert.equal(countImageSlots(prompt, slot), expected.length, name)
    const { tokens, keys, starts } = expandImageTokens(prompt, slot, Array.from({ length: images }, () => view), written)
    assert.equal(starts.length, images, name)
    // Each image part's placeholder became start, its rows, end; each literal one is still that one token, keyed as text.
    let at = 0
    let image = 0
    for (let i = 0; i < prompt.length; i++) {
      if (prompt[i] !== IMG) {
        assert.equal(tokens[at++], prompt[i], name)
        continue
      }
      if (expected.shift()) {
        assert.deepEqual(tokens.slice(at, at + 4), [slot.start, IMG, IMG, slot.end], name)
        assert.equal(starts[image++], at + 1, name)
        assert.deepEqual(keys.slice(at, at + 4), [slot.start, -5, -5, slot.end], name)
        at += 4
      } else {
        assert.deepEqual([tokens[at], keys[at]], [IMG, IMG], name)
        at++
      }
    }
    assert.equal(at, tokens.length, name)
  }
  // A template that writes a wrapped placeholder: a literal pad token between literal start and end tokens is text too.
  const wrapped = { kind: "wrapped" as const, start: 7, pad: 9, end: 8 }
  assert.equal(countImageSlots([7, 9, 8, 5, 7, 9, 8], wrapped), 2)
  assert.equal(countImageSlots([7, 9, 8, 5, 7, 9, 8], wrapped, [false, true]), 1)
  assert.deepEqual(expandImageTokens([7, 9, 8, 5, 7, 9, 8], wrapped, [view], [false, true]).tokens, [7, 9, 8, 5, 7, 9, 9, 8])
  // Placeholders in a tool's description and a tool call's arguments are text as well.
  const withTools = imagePlaceholders(
    { chat: [{ role: "user", content: [picture] }] as Array<ChatMessage>, tools: "<image>" },
    "<image>",
    (input) => `${input.tools}|${tokenizer.applyChatTemplate(input.chat, {})}`
  )
  assert.deepEqual(withTools, [false, true])
})

test("a streamed encode holds one piece of the encoder at a time: its claim is what it allocates, and its draws are the loaded encoder's", async () => {
  for (const [name, pieces, bits] of [["a ViT, 4-bit", tinyEncoder(), 4], ["a conv and attention tower, 8-bit", tinyTower(), 8]] as const) {
    const config = (pieces[0] as { config: VisionConfig }).config
    const view = (gridH: number, gridW: number) => ({ pixels: new Float32Array(gridH * gridW * 1536), gridH, gridW })
    for (const grids of [[[4, 4]], [[40, 40]], [[8, 12], [8, 12], [4, 4]], [[4, 4], [40, 40]]] as Array<Array<[number, number]>>) {
      const counter = countingGpu()
      // Each view's patches are asked for once, as its own patch projection runs: after the draws of the views before it.
      const asked: Array<[number, number]> = []
      const sources = grids.map(([gridH, gridW], i) => ({
        gridH,
        gridW,
        pixels: () => {
          asked.push([i, counter.draws.length])
          return view(gridH, gridW).pixels
        }
      }))
      const rows = await new VisionEncoder(counter.gpu).encodeStream(piecesOf(pieces), sources)
      assert.deepEqual(asked.map(([i]) => i), grids.map((_, i) => i))
      for (let i = 1; i < asked.length; i++) assert.ok(asked[i][1] > asked[i - 1][1], "a view's patches are made after the last view's projection drew")
      assert.ok(asked[asked.length - 1][1] < counter.draws.length)
      const label = `${name}, ${grids.map((g) => g.join("×")).join(" ")}`
      assert.deepEqual(rows.map((r) => r.length), grids.map(([h, w]) => ((h * w) / 4) * config.outWidth), label)
      assert.equal(counter.peak(), visionStreamBytes(config, bits, bits === 4, grids, 16384), label)
      assert.equal(counter.live(), 0, "nothing is left on the GPU")
      // Less than the loaded encoder and its largest view's work.
      const loaded = visionWeightBytes(config, bits) + Math.max(...grids.map(([h, w]) => visionWorkBytes(config, h, w, 16384)))
      if (grids.length === 1) assert.ok(counter.peak() < loaded, label)
      if (grids.length > 1) continue
      // One view: the same draws, in the same order, as the encoder loaded whole.
      const whole = countingGpu()
      const encoder = new VisionEncoder(whole.gpu)
      await encoder.load(piecesOf(pieces))
      await encoder.encode(view(grids[0][0], grids[0][1]))
      assert.deepEqual(counter.draws, whole.draws, label)
      assert.ok(whole.draws.length > 0)
    }
  }
})

test("a streamed encode refuses pieces out of the encoder's order, and frees what it made", async () => {
  const pieces = tinyTower()
  const view = { pixels: new Float32Array(16 * 1536), gridH: 4, gridW: 4 }
  for (const [order, message] of [
    [[0, 2, 1, 3, 4], /incomplete|out of order/],
    [[0, 1, 2, 4], /incomplete/],
    [[0, 1, 2, 3], /incomplete/],
    [[1, 2, 3, 4], /before its config/]
  ] as const) {
    const counter = countingGpu()
    await assert.rejects(new VisionEncoder(counter.gpu).encodeStream(piecesOf(order.map((i) => pieces[i])), [{ ...view, pixels: () => view.pixels }]), message)
    assert.equal(counter.live(), 0)
  }
})

test("the vision claim is what the encoder allocates: its weights, and its largest image's work at once", async () => {
  for (const [name, pieces, bits] of [["a ViT, 4-bit", tinyEncoder(), 4], ["a conv and attention tower, 8-bit", tinyTower(), 8]] as const) {
    const counter = countingGpu()
    const encoder = new VisionEncoder(counter.gpu)
    await encoder.load(piecesOf(pieces))
    const config = (pieces[0] as { config: VisionConfig }).config
    assert.equal(counter.live(), visionWeightBytes(config, bits), name)
    for (const [gridH, gridW] of [[4, 4], [8, 12], [40, 40]]) {
      counter.resetPeak()
      const weights = counter.live()
      await encoder.encode({ pixels: new Float32Array(gridH * gridW * 1536), gridH, gridW })
      assert.equal(counter.peak() - weights, visionWorkBytes(config, gridH, gridW, 16384), `${name}, ${gridH}×${gridW}`)
      assert.equal(counter.live(), weights, "its work textures are freed")
    }
    encoder.dispose()
    assert.equal(counter.live(), 0)
  }
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
