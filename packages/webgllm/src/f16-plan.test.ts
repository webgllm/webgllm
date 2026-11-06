import assert from "node:assert/strict"
import { describe, test } from "node:test"
import { Engine } from "./engine.js"

describe("the f16 matmul's chunks (Engine.f16Plan)", () => {
  const scratch = { hd: 2048, width: 2048 }

  test("whole matrices where they fit, whole K-slices where one does", () => {
    // LFM3's down (2048 × 7680, 4 slices of 1920): fits whole.
    assert.deepEqual(Engine.f16Plan(scratch, 2048, 60, 4), { group: 4, rows: 2048 })
    // Its gate|up (15360 × 2048, 2 slices of 1024): a slice at a time.
    assert.deepEqual(Engine.f16Plan(scratch, 15360, 32, 2), { group: 1, rows: 15360 })
    // 4 slices, 3 fit: two groups of 2, evened out.
    assert.deepEqual(Engine.f16Plan({ hd: 2048, width: 1536 }, 2048, 64, 4), {
      group: 2,
      rows: 2048
    })
  })

  test("rows in columns when one slice doesn't fit", () => {
    assert.deepEqual(Engine.f16Plan({ hd: 64, width: 512 }, 15360, 32, 2), {
      group: 1,
      rows: 128
    })
    // 6144 rows, at most 4096 a chunk: two of 3072.
    assert.deepEqual(Engine.f16Plan({ hd: 1024, width: 1024 }, 6144, 32, 2), {
      group: 1,
      rows: 3072
    })
  })

  test("every chunk fits the scratch and covers the matrix", () => {
    for (const hd of [64, 512, 2048, 16384])
      for (const width of [512, 1920, 4096, 16384])
        for (const N of [256, 2048, 6144, 15360])
          for (const [kbSlice, slices] of [[16, 4], [32, 2], [60, 4], [64, 1], [240, 1]]) {
            if (width < kbSlice * 8) continue
            const { group, rows } = Engine.f16Plan({ hd, width }, N, kbSlice, slices)
            assert.ok(group >= 1 && group <= slices)
            assert.ok(rows === N || rows % 4 === 0, `${rows}`)
            assert.ok(rows >= 1 && rows <= N)
            assert.ok(
              Math.ceil(rows / hd) * group * kbSlice * 8 <= width,
              `hd ${hd} width ${width} N ${N} slices ${slices}×${kbSlice}: ${group}, ${rows}`
            )
          }
  })
})
