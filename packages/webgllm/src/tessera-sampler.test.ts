import assert from "node:assert/strict"
import { test } from "node:test"

import { blankAfter, commitOrder, drawSprites, fixedLengthIds, GRID, MASK, PALETTE, PIXELS, seededRandom, spritePixels, type Denoise } from "./tessera-sampler.js"
import { ratioFeatures, ropeTexels } from "./tessera.js"

test("the text encoder's ids are padded with the end token to 24", () => {
  // "a red apple", as Tessera's own self-test encodes it.
  assert.deepEqual(fixedLengthIds([49406, 320, 736, 3055, 49407], 49407), [49406, 320, 736, 3055, ...Array<number>(20).fill(49407)])
})

test("a prompt over 24 tokens is cut with the end token kept last", () => {
  const ids = fixedLengthIds([49406, ...Array<number>(30).fill(7), 49407], 49407)
  assert.equal(ids.length, 24)
  assert.equal(ids[0], 49406)
  assert.equal(ids[22], 7)
  assert.equal(ids[23], 49407)
})

test("the cosine schedule leaves fewer blank pixels each step, and none after the last", () => {
  const blanks = Array.from({ length: 16 }, (_, step) => blankAfter(step, 16))
  for (let i = 1; i < blanks.length; i++) assert.ok(blanks[i]! <= blanks[i - 1]!)
  assert.ok(blanks[0]! < PIXELS)
  assert.equal(blanks[15], 0)
})

test("the seeded random source repeats for a seed and stays in [0, 1)", () => {
  const a = seededRandom(42)
  const b = seededRandom(42)
  for (let i = 0; i < 100; i++) {
    const x = a()
    assert.equal(x, b())
    assert.ok(x >= 0 && x < 1)
  }
  assert.notEqual(seededRandom(1)(), seededRandom(2)())
})

test("a blank pixel goes ahead if it's certain or beats its uncertain blank neighbours, then by score", () => {
  const blank = new Uint8Array(PIXELS).fill(1)
  const confidence = new Float32Array(PIXELS).fill(0.1)
  const score = new Float32Array(PIXELS).fill(-5)
  score[0] = -1
  score[1] = -2
  confidence[200] = 0.95
  score[200] = -4
  score[201] = -3
  const order = commitOrder(blank, confidence, score, 0.9)
  assert.equal(order.length, PIXELS)
  assert.deepEqual(order.slice(0, 3), [0, 201, 200])
  assert.ok(order.indexOf(1) > order.indexOf(GRID * 8 + 8))
})

test("committed pixels are left out of the order", () => {
  const blank = new Uint8Array(PIXELS)
  blank[5] = 1
  blank[77] = 1
  const order = commitOrder(blank, new Float32Array(PIXELS), new Float32Array(PIXELS), 0.9)
  assert.deepEqual([...order].sort((a, b) => a - b), [5, 77])
})

test("a sprite's pixels are its tokens' palette colors", () => {
  const tokens = new Int32Array(PIXELS)
  tokens[1] = 2
  const rgba = spritePixels(tokens, [
    [0, 0, 0, 0],
    [1, 1, 1, 255],
    [10, 20, 30, 255]
  ])
  assert.deepEqual([...rgba.subarray(0, 8)], [0, 0, 0, 0, 10, 20, 30, 255])
})

/** A denoiser sure of one token per pixel and image (`want`), the same with the prompt and without; it records its calls. */
function fakeDenoiser(want: (image: number, pixel: number) => number) {
  const calls: Array<{ tokens: Int32Array; maskRatio: Float32Array; sc: Float32Array | null; batch: number; count: number }> = []
  const denoise: Denoise = async (tokens, maskRatio, sc, batch, count) => {
    calls.push({ tokens: tokens.slice(), maskRatio: maskRatio.slice(), sc: sc && sc.slice(0, PALETTE), batch, count })
    const logits = new Float32Array(batch * PIXELS * PALETTE)
    for (let b = 0; b < batch; b++)
      for (let i = 0; i < PIXELS; i++) logits[(b * PIXELS + i) * PALETTE + want(b % count, i)] = 30
    return logits
  }
  return { denoise, calls }
}

test("drawing paints every pixel over the steps, two rows an image, self-conditioning from the second step", async () => {
  const { denoise, calls } = fakeDenoiser((image, pixel) => (image * 7 + pixel) % PALETTE)
  const steps = 8
  const sprites = await drawSprites(denoise, 2, { steps, guidanceScale: 3, choiceTemperature: 2, certainty: 0.9, seed: 1 })
  assert.equal(calls.length, steps)
  for (const call of calls) assert.deepEqual([call.batch, call.count], [4, 2])
  // The first step sees a blank canvas and no self-conditioning; later ones, the last step's distribution.
  assert.ok(calls[0]!.tokens.every((t) => t === MASK))
  assert.equal(calls[0]!.sc, null)
  assert.ok(calls[1]!.sc && calls[1]!.sc[(0 * 7 + 0) % PALETTE]! > 0.99)
  // Each image's two rows are the same canvas, and fewer pixels stay blank each step.
  for (const call of calls) {
    assert.deepEqual(call.tokens.subarray(0, PIXELS), call.tokens.subarray(2 * PIXELS, 3 * PIXELS))
    assert.equal(call.maskRatio[0], call.maskRatio[2])
  }
  for (let i = 1; i < calls.length; i++) assert.ok(calls[i]!.maskRatio[0]! < calls[i - 1]!.maskRatio[0]!)
  for (const [image, sprite] of sprites.entries()) for (let i = 0; i < PIXELS; i++) assert.equal(sprite[i], (image * 7 + i) % PALETTE)
})

test("the same seed draws the same sprites; aborting stops between steps", async () => {
  // A flat distribution: the draw is all chance.
  const flat: Denoise = async (_t, _r, _s, batch) => new Float32Array(batch * PIXELS * PALETTE)
  const settings = { steps: 4, guidanceScale: 3, choiceTemperature: 2, certainty: 0.9 }
  const a = await drawSprites(flat, 1, { ...settings, seed: 9 })
  const b = await drawSprites(flat, 1, { ...settings, seed: 9 })
  const c = await drawSprites(flat, 1, { ...settings, seed: 10 })
  assert.deepEqual(a, b)
  assert.notDeepEqual(a, c)
  const abort = new AbortController()
  let steps = 0
  await assert.rejects(
    drawSprites(flat, 1, { ...settings, seed: 1 }, abort.signal, () => {
      if (++steps === 2) abort.abort()
    }),
    { name: "AbortError" }
  )
  assert.equal(steps, 2)
})

test("RoPE's table: prefix rows unrotated, each pixel at its row and column, the core's at its 2 × 2 patch's centre", () => {
  const prefix = 28
  const table = ropeTexels(prefix, 16, 1)
  // 32 values a row: 16 cos (rows' angles, then columns'), 16 sin.
  assert.equal(table.length, (prefix + 256) * 64)
  for (let i = 0; i < 32; i++) {
    assert.equal(table[i], 1)
    assert.equal(table[32 + i], 0)
  }
  // Pixel (1, 2): angle row · θ^(−f/16) for the first 16, column · θ^(−f/16) for the next.
  const row = (prefix + 1 * 16 + 2) * 64
  assert.ok(Math.abs(table[row + 0]! - Math.cos(1)) < 1e-6)
  assert.ok(Math.abs(table[row + 16]! - Math.cos(2)) < 1e-6)
  assert.ok(Math.abs(table[row + 32 + 1]! - Math.sin(1 / 100 ** (1 / 16))) < 1e-6)
  const core = ropeTexels(prefix, 8, 2)
  assert.ok(Math.abs(core[(prefix + 0) * 64]! - Math.cos(0.5)) < 1e-6)
})

test("the mask ratio's features: sines then cosines of (2r − 1) · 2^i · π", () => {
  const f = ratioFeatures(0.75)
  for (let i = 0; i < 8; i++) {
    assert.ok(Math.abs(f[i]! - Math.sin(0.5 * 2 ** i * Math.PI)) < 1e-6)
    assert.ok(Math.abs(f[8 + i]! - Math.cos(0.5 * 2 ** i * Math.PI)) < 1e-6)
  }
})
