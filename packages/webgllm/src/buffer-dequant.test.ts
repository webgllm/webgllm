import assert from "node:assert/strict"
import { describe, test } from "node:test"
import { Engine } from "./engine.js"
import { bufferDequant } from "./gpu/native.js"

describe("the buffer dequant, a thread a row's block (gpu/native.ts's bufferDequant)", () => {
  test("into the scratch buffer: 4 h2 texels a thread, each word's low nibbles and its high nibbles 2 texels on", () => {
    const spec = bufferDequant({ h2: true, toBuffer: true })
    // A workgroup 64 blocks of a row: a block is 4 h2 texels.
    assert.deepEqual(spec.groups({ width: 3840, height: 1024 }), [15, 1024])
    assert.deepEqual(spec.groups({ width: 4, height: 7 }), [1, 7])
    assert.match(spec.code, /let x = c \* \(u\.kq >> 1u\) \+ 4 \* kbl;/)
    assert.match(spec.code, /let at = y \* u\.stride \+ x;/)
    // One fetch of each of the row's 4 words, and one of its scale.
    for (const j of [0, 1, 2, 3]) assert.equal(spec.code.split(`bW[wb + ${j}]`).length, 2)
    assert.equal(spec.code.split("textureLoad(tS").length, 2)
    assert.match(spec.code, /let wb = gr \* u\.width \+ col \* u\.rowTexels \+ kb \* 16 \+ r \* 4;/)
    // Texels 4j..4j+3 of the block (f16 layout): the low nibbles of word j, then the high nibbles of word j - 4.
    assert.match(spec.code, /let v1 = q\(w1, vec4<u32>\(0u, 8u, 16u, 24u\), z\) \* d;/)
    assert.match(spec.code, /let v6 = q\(w2, vec4<u32>\(4u, 12u, 20u, 28u\), z\) \* d;/)
    assert.match(spec.code, /out0\[at \+ 0\] = h2\(v0, v1\);/)
    assert.match(spec.code, /out0\[at \+ 3\] = h2\(v6, v7\);/)
    // Rows past the matrix's: zeros.
    assert.match(spec.code, /if \(row >= u\.n\) \{\n {2}out0\[at \+ 0\] = h2\(vec4<f32>\(0\.0\), vec4<f32>\(0\.0\)\);/)
  })

  test("into textures: h2 texels, or the f16 layout's 8 texels a block", () => {
    const h2 = bufferDequant({ h2: true })
    assert.match(h2.code, /texture_storage_2d<rgba32uint, write>/)
    assert.match(h2.code, /textureStore\(out0, vec2<i32>\(x \+ 2, y\), h2\(v4, v5\)\);/)
    const f16 = bufferDequant({ h2: false })
    assert.match(f16.code, /texture_storage_2d<rgba16float, write>/)
    assert.match(f16.code, /let x = c \* u\.kq \+ 8 \* kbl;/)
    assert.match(f16.code, /textureStore\(out0, vec2<i32>\(x \+ 7, y\), v7\);/)
    assert.deepEqual(f16.groups({ width: 7680, height: 1024 }), [15, 1024])
    assert.throws(() => bufferDequant({ h2: false, toBuffer: true }))
  })

  test("8-bit weights: a word a texel, 8 words a row's block", () => {
    const spec = bufferDequant({ h2: true, toBuffer: true, bits: 8 })
    assert.match(spec.code, /let wb = gr \* u\.width \+ col \* u\.rowTexels \+ kb \* 32 \+ r \* 8;/)
    for (const j of [0, 1, 2, 3, 4, 5, 6, 7]) assert.equal(spec.code.split(`bW[wb + ${j}]`).length, 2)
    assert.match(spec.code, /let v5 = q\(w5\) \* d;/)
    assert.match(spec.code, /out0\[at \+ 2\] = h2\(v4, v5\);/)
  })
})

test("an 8-bit matrix takes the buffer scratch only where the 4-bit ones' holds its K", () => {
  const takes = Reflect.get(Engine.prototype, "eightTakesF16") as (w: unknown, f: unknown) => boolean
  const wbuf = {}
  const stored = {}
  assert.equal(takes({ bits: 8, stored, K: 2048 }, { kMax: 6144, wbuf }), true)
  assert.equal(takes({ bits: 8, stored, K: 6144 }, { kMax: 6144, wbuf }), true)
  // Wider than the scratch was sized for, weights not in a buffer, no buffer scratch, or 4-bit: as before.
  assert.equal(takes({ bits: 8, stored, K: 8192 }, { kMax: 6144, wbuf }), false)
  assert.equal(takes({ bits: 8, K: 2048 }, { kMax: 6144, wbuf }), false)
  assert.equal(takes({ bits: 8, stored, K: 2048 }, { kMax: 6144 }), false)
  assert.equal(takes({ bits: 4, stored, K: 2048 }, { kMax: 6144, wbuf }), false)
})

test("an 8-bit matrix goes through the scratch only into an output with a second texture", () => {
  const scratch = Reflect.get(Engine.prototype, "eightScratch") as (this: unknown, w: unknown, out: unknown) => unknown
  const f = { kMax: 6144, wbuf: {} }
  const state = {
    quads: new Map([[1, {}]]),
    usesF16: () => true,
    f16Pack: () => 3,
    f16Scratch: () => f,
    eightTakesF16: Reflect.get(Engine.prototype, "eightTakesF16")
  }
  const w = { bits: 8, stored: {}, K: 2048 }
  assert.equal(scratch.call(state, w, { id: 1 }), f)
  // A pair tensor of one texture (Gated DeltaNet's beta|decay buffer), 4-bit weights, or a scratch that isn't a buffer of h2 texels.
  assert.equal(scratch.call(state, w, { id: 2 }), false)
  assert.equal(scratch.call(state, { ...w, bits: 4 }, { id: 1 }), false)
  assert.equal(scratch.call({ ...state, f16Pack: () => 0 }, w, { id: 1 }), false)
  assert.equal(scratch.call({ ...state, f16Scratch: () => ({ kMax: 6144 }) }, w, { id: 1 }), false)
  assert.equal(scratch.call({ ...state, usesF16: () => false }, w, { id: 1 }), false)
})
