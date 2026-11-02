import assert from "node:assert/strict"
import { describe, test } from "node:test"
import * as K from "./gpu/kernels.js"

describe("the f16 prompt matmul's h2 reads (TUNING.f16Pack)", () => {
  test("f16Pack 0 builds the RGBA16F kernels as they were", () => {
    assert.equal(K.matmulF16({ half: false, packed: 0 }), K.matmulF16({ half: false }))
    assert.equal(K.matmulF16({ half: true, packed: 0 }), K.matmulF16({ half: true }))
    assert.equal(K.dequantRgba4({ zeros: true, h2: false }), K.dequantRgba4({ zeros: true }))
    const bytes = { zeros: false, words: false, unorm: true, packed: true }
    assert.equal(K.dequantBytes({ ...bytes, h2: false }), K.dequantBytes(bytes))
    assert.match(K.dequantRgba4({ zeros: true }), /out vec4 o;/)
  })

  test("each bit reads its input as h2 texels, two K texels a step", () => {
    for (const packed of [1, 2, 3]) {
      const source = K.matmulF16({ half: false, packed })
      const h2A = (packed & 1) !== 0
      const h2W = (packed & 2) !== 0
      assert.match(source, new RegExp(`uniform ${h2W ? "usampler2D" : "sampler2D"} uW;`))
      assert.match(source, new RegExp(`uniform ${h2A ? "usampler2D" : "sampler2D"} uA;`))
      assert.match(source, /kq \+= 2\)/)
      assert.equal(source.includes("texelFetch(uA, ivec2(kq >> 1"), h2A)
      assert.equal(source.includes("texelFetch(uW, ivec2((wx + kq) >> 1"), h2W)
    }
    assert.match(K.matmulF16({ half: true, packed: 3 }), /mediump vec4 w0 = h2lo\(W0\);/)
  })

  test("h2 dequantization calls the texel's math twice into one h2 texel", () => {
    const sources = [
      K.dequantRgba4({ zeros: false, h2: true }),
      K.dequantRgba4({ zeros: true, h2: true }),
      K.withPackedScales(K.dequantRgba4({ zeros: true, h2: true })),
      ...[false, true].flatMap((words) =>
        [false, true].map((packed) =>
          K.dequantBytes({ zeros: true, words, unorm: !words, packed, h2: true })
        )
      )
    ]
    for (const source of sources) {
      assert.match(source, /vec4 deq\(int x, int y\) \{/)
      assert.match(source, /if \(row >= uN\) return vec4\(0\.0\);/)
      assert.match(source, /\n {2}return \([^;]*\) \* d;\n\}/)
      assert.match(source, /out uvec4 o;/)
      assert.match(source, /o = packH2\(deq\(x, y\), deq\(x \+ 1, y\)\);/)
      assert.doesNotMatch(source, /out vec4 o;/)
    }
    // Packed scales still apply to the function's body.
    assert.match(
      K.withPackedScales(K.dequantRgba4({ zeros: true, h2: true })),
      /float d = comp\(h2lo\(sz\), r\);/
    )
  })
})
