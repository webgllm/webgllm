import assert from "node:assert/strict"
import { describe, test } from "node:test"
import { promptConv } from "./gpu/native.js"
import * as K from "./gpu/kernels.js"

describe("a prompt batch's short convolution in one pass (gpu/native.ts's promptConv)", () => {
  // LFM3's: 7 taps, in_proj's output a 4-token matmul's in 4 K-slices.
  const spec = promptConv({ width: 7, y: "quad", slices: 4 })

  test("a workgroup an h2 texel of the output and 64 rows of the tensor (256 tokens)", () => {
    assert.deepEqual(spec.groups({ width: 256, height: 548 }), [256, 3])
    assert.deepEqual(spec.groups({ width: 256, height: 256 }), [256, 1])
    assert.match(spec.code, /@compute @workgroup_size\(64\)/)
    assert.deepEqual(spec.inputs.map((i) => `${i.name}:${i.kind}`), ["uBCx:u", "uBCx_B:u", "uState:f", "uConv:f"])
    // B·x of the block's 256 tokens and of the 8 before it (2 rows hold the 6 the taps reach), two texels each.
    assert.match(spec.code, /var<workgroup> Y: array<vec4<f32>, 528>;/)
  })

  test("B | C | x as shortConv reads them, each token's B·x once", () => {
    const glsl = K.shortConv(7)
    assert.match(glsl, /y0 = pf\(uBCx, c0, t\) \* pf\(uBCx, 2 \* dt \+ c0, t\);/)
    assert.match(glsl, /vec4 C0 = pf\(uBCx, dt \+ c0, m\);/)
    assert.match(spec.code, /let B0 = row\(c0, q\);/)
    assert.match(spec.code, /let X0 = row\(2 \* u\.dt \+ c0, q\);/)
    assert.match(spec.code, /let C0 = row\(u\.dt \+ c0, q\);/)
    assert.match(spec.code, /let y0_3 = B0\.t3 \* X0\.t3;\n {2}let y1_3 = B1\.t3 \* X1\.t3;\n {2}Y\[2 \* \(8 \+ 4 \* l \+ 3\)\] = y0_3;/)
    // Tokens before the batch from the state, oldest first: token t at row width - 1 + t.
    assert.match(glsl, /y0 = texelFetch\(uState, ivec2\(c0, 6 \+ t\), 0\);/)
    assert.match(spec.code, /let s = 6 \+ 4 \* qh \+ 1;\n {8}Y\[2 \* \(4 \* l \+ 1\)\] = select\(vec4<f32>\(0\.0\), textureLoad\(tState, vec2<i32>\(c0, s\), 0\), s >= 0\);/)
  })

  test("the taps in order, each a fused multiply-add (as the GLSL pass's `a0 += w * y0` compiles), from 6 tokens back", () => {
    assert.match(K.shortConv(7), /a0 \+= texelFetch\(uConv, ivec2\(c0, 0\), 0\) \* y0;/)
    assert.match(spec.code, /let at = 2 \* \(8 \+ 4 \* l \+ 2 - 6\);/)
    for (let k = 0; k < 7; k++) assert.match(spec.code, new RegExp(`a0 = fma\\(w0_${k}, Y\\[at \\+ ${2 * k}\\], a0\\);\\n {4}a1 = fma\\(w1_${k}, Y\\[at \\+ ${2 * k + 1}\\], a1\\);`))
    assert.match(spec.code, /textureStore\(outZ, vec2<i32>\(x, m0 \+ 2\), h2\(C0\.t2 \* a0, C1\.t2 \* a1\)\);/)
  })

  test("the batch's last width - 1 tokens' B·x are the state after it", () => {
    assert.match(K.convState(7), /int t = uM - 6 \+ i;/)
    assert.match(spec.code, /let i = m0 \+ 0 - \(rows - 6\);\n {4}if \(i >= 0\) \{\n {6}textureStore\(outState, vec2<i32>\(c0, i\), y0_0\);\n {6}textureStore\(outState, vec2<i32>\(c0 \+ 1, i\), y1_0\);/)
  })

  test("a pair tensor's rows hold 2 tokens: LFM2.5's 3 taps reach one row back", () => {
    const pair = promptConv({ width: 3, y: "pair", slices: 2 })
    assert.deepEqual(pair.groups({ width: 128, height: 540 }), [128, 5])
    assert.match(pair.code, /var<workgroup> Y: array<vec4<f32>, 260>;/)
    assert.match(pair.code, /if \(l < 1\) \{/)
    assert.doesNotMatch(pair.code, /tPB/)
    assert.throws(() => promptConv({ width: 1, y: "pair", slices: 1 }))
  })
})
