import assert from "node:assert/strict"
import { describe, test } from "node:test"
import { fusedAdd } from "./gpu/native.js"

describe("the native residual add (TUNING.fusedAdd)", () => {
  const names = (spec: { inputs: Array<{ name: string; kind: string }> }) => spec.inputs.map((i) => `${i.name}:${i.kind}`)

  test("a token's add reads a split matmul's partial sums, a workgroup a row", () => {
    const spec = fusedAdd({ texels: 512, normed: true, next: true })
    assert.deepEqual(names(spec), ["uH:f", "uY:f", "uNorm:f", "uNext:f"])
    assert.deepEqual(spec.groups({ width: 512, height: 3 }), [3, 1])
    assert.match(spec.code, /textureLoad\(tY, vec2<i32>\(x, q \* u\.split \+ k\), 0\)/)
    assert.match(spec.code, /var<workgroup> red: array<f32, 256>;/)
    // (Its rows are all the rectangle's: no row past its end to leave alone.)
    assert.doesNotMatch(spec.code, /< end\)/)
    assert.deepEqual(spec.uniforms.map((u) => u.name), ["uSplitIn", "uParts", "uEps", "uBase", "uSliceStride"])
  })

  test("a prompt's add reads a 4-token matmul's two textures, a workgroup their row of 4 tokens", () => {
    const spec = fusedAdd({ texels: 512, normed: true, next: false, y: "quad", slices: 4 })
    assert.deepEqual(names(spec), ["uH:f", "uY:u", "uY_B:u", "uNorm:f"])
    assert.deepEqual(spec.groups({ width: 512, height: 548 }), [137, 1])
    assert.deepEqual(spec.groups({ width: 512, height: 9 }), [3, 1])
    // Its K-slices, uSliceStride rows apart, as pfq adds them; tokens 4q, 4q + 1 in uY, the others in uY_B.
    assert.match(spec.code, /for \(var k = 0; k < 4; k\+\+\) \{\n {4}let at = vec2<i32>\(x, q \+ k \* u\.stride\);/)
    assert.match(spec.code, /let a = textureLoad\(tY, at, 0\);\n {4}r\.t0 \+= lo\(a\);\n {4}r\.t1 \+= hi\(a\);\n {4}let b = textureLoad\(tYB, at, 0\);\n {4}r\.t2 \+= lo\(b\);\n {4}r\.t3 \+= hi\(b\);/)
    assert.match(spec.code, /var<workgroup> red: array<vec4<f32>, 256>;/)
    // A batch that isn't whole rows of 4 leaves the rows past its end alone.
    for (const t of [0, 1, 2, 3]) assert.match(spec.code, new RegExp(`if \\(m${t} < end\\) \\{\\n {2}textureStore\\(out0, vec2<i32>\\(xa0, m${t}\\), va0_${t}\\);`))
    // Each token's sum of squares in column `texels`, zeros in the norm's other columns.
    assert.match(spec.code, /textureStore\(out0, vec2<i32>\(512 \+ c, m3\), select\(vec4<f32>\(0\.0\), vec4<f32>\(sum\[3\], 0\.0, 0\.0, 0\.0\), c == 0\)\);/)
    // One output without `next`.
    assert.doesNotMatch(spec.code, /out1/)
  })

  test("a pair tensor's add takes its row's 2 tokens, with the next matmuls' scaled input", () => {
    const spec = fusedAdd({ texels: 256, normed: false, next: true, y: "pair", slices: 2 })
    assert.deepEqual(names(spec), ["uH:f", "uY:u", "uNext:f"])
    assert.deepEqual(spec.groups({ width: 256, height: 531 }), [266, 1])
    assert.match(spec.code, /var<workgroup> red: array<vec2<f32>, 128>;/)
    assert.doesNotMatch(spec.code, /tYB|tNorm|inverseSqrt/)
    assert.match(spec.code, /textureStore\(out1, vec2<i32>\(u\.base \+ j0 \+ 0, m1\), h2\(va0_1 \* ta0, vb0_1 \* tb0\)\);/)
  })

  test("the threads are a power of two (the sums halve), each as many texel pairs", () => {
    // 384 texels (a hidden size of 1,536): 192 pairs in 64 threads of 3.
    const spec = fusedAdd({ texels: 384, normed: true, next: false })
    assert.match(spec.code, /@compute @workgroup_size\(64\)/)
    assert.match(spec.code, /let j0 = i32\(l\) \* 3;/)
    assert.match(spec.code, /let xa2 = 2 \* \(j0 \+ 2\);/)
    // (Fewer threads than the norm sums columns: each writes several.)
    assert.match(spec.code, /for \(var c = i32\(l\); c < u\.parts; c \+= 64\)/)
    assert.throws(() => fusedAdd({ texels: 3, normed: false, next: false }))
  })
})
