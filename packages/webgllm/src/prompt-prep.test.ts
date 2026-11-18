import assert from "node:assert/strict"
import { describe, test } from "node:test"
import { promptPrep } from "./gpu/native.js"
import * as K from "./gpu/kernels.js"

describe("a prompt batch's queries, keys and values in one pass (gpu/native.ts's promptPrep)", () => {
  // LFM3's shape: 16 heads of 128, 4 key/value heads.
  const shape = { headDim: 128, heads: 16, kvHeads: 4, ropeLow: K.ROPE_LOW }
  const spec = promptPrep({ ...shape, y: "quad", slices: 4 })

  test("a workgroup a head of a token: the q heads, the k heads, the v heads", () => {
    assert.deepEqual(spec.groups({ width: 512, height: 548 }), [24, 548])
    assert.match(spec.code, /@compute @workgroup_size\(32\)/)
    assert.deepEqual(spec.inputs.map((i) => `${i.name}:${i.kind}`), ["uQKVG:u", "uQKVG_B:u", "uQNorm:f", "uKNorm:f", "uRope:f"])
    // The projection's q rows, then k, then v (texels of 4).
    assert.match(spec.code, /let base = select\(select\(640, 512, isK\), 0, isQ\) \+ head \* 32;/)
  })

  test("reads the projection as pfq does: tokens 4q, 4q + 1 in one texture, the others in the second, K-slices added", () => {
    assert.match(spec.code, /for \(var k = 0; k < 4; k\+\+\) \{\n {4}let at = vec2<i32>\(x, \(m >> 2u\) \+ k \* u\.stride\);/)
    assert.match(spec.code, /if \(\(\(m >> 1u\) & 1\) == 1\) \{ t = textureLoad\(tPB, at, 0\); \} else \{ t = textureLoad\(tP, at, 0\); \}/)
    assert.match(spec.code, /acc \+= select\(lo\(t\), hi\(t\), \(m & 1\) == 1\);/)
    const pair = promptPrep({ ...shape, y: "pair", slices: 1 })
    assert.deepEqual(pair.inputs.map((i) => i.name), ["uQKVG", "uQNorm", "uKNorm", "uRope"])
    assert.match(pair.code, /let at = vec2<i32>\(x, \(m >> 1u\) \+ k \* u\.stride\);\n {4}let t = textureLoad\(tP, at, 0\);/)
  })

  test("the head's norm and rotation as headPrep has them", () => {
    const glsl = K.headPrep({ out: "f32", rope: true, norm: true, headDim: 128 })
    assert.match(glsl, /inv = inversesqrt\(dot\(s, vec4\(1\.0\)\) \/ 128\.0 \+ uEps\);/)
    // The sum of squares as a fused multiply-add a texel, in order: what the GLSL pass's `s += v * v` compiles to (the logprobs agree to the last digit).
    assert.match(spec.code, /for \(var i = 0; i < 32; i\+\+\) \{ s4 = fma\(sq\[i\], sq\[i\], s4\); \}/)
    assert.match(spec.code, /let inv = inverseSqrt\(dot\(s4, vec4<f32>\(1\.0\)\) \/ 128\.0 \+ u\.eps\);/)
    assert.match(glsl, /return first \? v \* cs - p \* sn : v \* cs \+ p \* sn;/)
    assert.match(spec.code, /r = select\(n \* cs \+ p \* sn, n \* cs - p \* sn, first\);/)
    // Each row's own position: the batch's first plus the row.
    assert.match(glsl, /int pos = uPos0 \+ m;/)
    assert.match(spec.code, /let pos = u\.pos0 \+ m;/)
  })

  test("queries to their f32 texel, keys and values to the token's slot of the cache", () => {
    assert.match(spec.code, /textureStore\(outQ, vec2<i32>\(head \* 32 \+ t, m\), r \* u\.scale\);/)
    // The slot wraps at a ring, and lies in blocks of 2^kvShift rows, kvWidth texels apart.
    assert.match(spec.code, /var slot = u\.slot0 \+ m;\n {4}if \(u\.ring > 0\) \{ slot = slot % u\.ring; \}/)
    assert.match(spec.code, /let kx = \(slot >> u32\(u\.kvShift\)\) \* u\.kvWidth \+ head \* 16 \+ t;/)
    // Keys as h2 (two rotated texels a texel), values beside them (after the 4 key heads' 64 texels).
    assert.match(spec.code, /textureStore\(outKV, vec2<i32>\(kx, ky\), h2\(nv\[2 \* t\], nv\[2 \* t \+ 1\]\)\);/)
    assert.match(spec.code, /textureStore\(outKV, vec2<i32>\(kx \+ 64, ky\), h2\(pf\(base \+ 2 \* t, m\), pf\(base \+ 2 \* t \+ 1, m\)\)\);/)
  })
})
