import assert from "node:assert/strict"
import { describe, test } from "node:test"
import { linChunkLoop } from "./gpu/native.js"
import * as K from "./gpu/kernels.js"

describe("the chunked recurrence in one dispatch (gpu/native.ts's linChunkLoop)", () => {
  const spec = linChunkLoop({ keyHeadDim: 128, valueHeadDim: 128, chunk: 32 })

  test("a workgroup a value texel, a thread a key row, the state and a chunk's deltas in workgroup memory", () => {
    assert.deepEqual(spec.groups({ width: 512, height: 524 }), [512, 1])
    assert.match(spec.code, /@compute @workgroup_size\(128\)/)
    assert.match(spec.code, /var<workgroup> S: array<vec4<f32>, 128>;/)
    assert.match(spec.code, /var<workgroup> D: array<vec4<f32>, 32>;/)
    assert.deepEqual(spec.inputs.map((i) => i.name), ["uS", "uU", "uW", "uOi", "uQp", "uQK", "uCum"])
    assert.deepEqual(spec.uniforms.map((u) => u.name), ["uK4", "uC0", "uC1", "uFirst"])
    // The output's rows, then the state after the last chunk.
    assert.match(spec.code, /textureStore\(outO, vec2<i32>\(x, t\), textureLoad\(tOi, vec2<i32>\(x, t\), 0\) \+ qs\);/)
    assert.match(spec.code, /textureStore\(outS, vec2<i32>\(x, k\), S\[k\]\);/)
  })

  test("chunks as chunkStart has them: a first one of uFirst rows, then `chunk` each, to uC1", () => {
    assert.match(spec.code, /let c1 = min\(c0 \+ select\(32, u\.first, c0 == 0\), u\.c1\);/)
    assert.match(K.linCum(32), /return t < uFirst \? 0 : uFirst \+ \(\(t - uFirst\) \/ 32\) \* 32;/)
  })

  test("the passes' own sums, term for term", () => {
    // linChunkOut: delta = U - W S0 and the output's Q' S0, over the head's keys in order.
    assert.match(K.linChunkOut({ keyHeadDim: 128, valueHeadDim: 128, out: "both" }), /ws \+= s0 \* w\.x \+ s1 \* w\.y \+ s2 \* w\.z \+ s3 \* w\.w;\n {4}qs \+= s0 \* q\.x \+ s1 \* q\.y \+ s2 \* q\.z \+ s3 \* q\.w;/)
    assert.match(spec.code, /ws \+= s0 \* wv\.x \+ s1 \* wv\.y \+ s2 \* wv\.z \+ s3 \* wv\.w;\n {8}qs \+= s0 \* q\.x \+ s1 \* q\.y \+ s2 \* q\.z \+ s3 \* q\.w;/)
    assert.match(spec.code, /for \(var i = 0; i < 32; i\+\+\) \{/)
    // linChunkState: the decayed state, then each token's k̂ times its delta, decayed to the chunk's end.
    assert.match(K.linChunkState({ keyHeadDim: 128, valueHeadDim: 128 }), /vec4 acc = exp\(last\) \* texelFetch\(uS, ivec2\(x, k\), 0\);[\s\S]*acc \+= exp\(last - linG\(uCum, h, s\)\) \* kk \* texelFetch\(uDelta, ivec2\(x, s\), 0\);/)
    assert.match(spec.code, /var acc = exp\(last\) \* S\[k\];[\s\S]*acc \+= exp\(last - G\(h, s\)\) \* kk \* D\[s - c0\];/)
  })

  test("shapes it doesn't fit throw: more key rows than a workgroup's threads, or a chunk of more tokens than key rows", () => {
    assert.throws(() => linChunkLoop({ keyHeadDim: 512, valueHeadDim: 128, chunk: 32 }))
    assert.throws(() => linChunkLoop({ keyHeadDim: 16, valueHeadDim: 16, chunk: 32 }))
  })
})
