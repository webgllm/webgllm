import assert from "node:assert/strict"
import { describe, test } from "node:test"
import {
  chunkPartial,
  combinePartials,
  panelAt,
  panelChunk,
  panelLayout,
  type Partial
} from "./panels.js"

/** A small deterministic generator (mulberry32). */
function random(seed: number) {
  let a = seed >>> 0
  return () => {
    a = (a + 0x6d2b79f5) >>> 0
    let t = a
    t = Math.imul(t ^ (t >>> 15), t | 1)
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61)
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}

const cases = [
  // LFM3 and Qwen3.5 (128 texels a panel) writing from 600, 2k and 8k
  // tokens; 3- and 8-row draft batches; a long cache in 4 segments; the
  // SEGMENT CHECK's 512-key segments; a 4096-texel GPU.
  { kt: 151, segTexels: 2048, slice: 1, maxWidth: 16384 },
  { kt: 513, segTexels: 2048, slice: 1, maxWidth: 16384 },
  { kt: 1976, segTexels: 2048, slice: 3, maxWidth: 16384 },
  { kt: 1977, segTexels: 2048, slice: 8, maxWidth: 16384 },
  { kt: 8000, segTexels: 2048, slice: 1, maxWidth: 16384 },
  { kt: 1980, segTexels: 128, slice: 1, maxWidth: 16384 },
  { kt: 1980, segTexels: 2048, slice: 2, maxWidth: 4096 }
]

describe("panel layout", () => {
  for (const o of cases) {
    test(`kt ${o.kt}, segments of ${o.segTexels}, ${o.slice} rows, ${o.maxWidth} wide`, () => {
      const panelW = 128
      const segments = Math.ceil(o.kt / o.segTexels)
      const lay = panelLayout({
        ...o,
        segments,
        panelW,
        chunkTexels: 32,
        maxChunks: 128
      })
      assert.ok(lay.width <= o.maxWidth)
      assert.equal(lay.chunks, segments * lay.segChunks)
      // Each segment's chunks cover its texels; the combine's chunk count
      // stays near the cap.
      assert.ok(lay.segChunks * lay.chunkTexels >= Math.min(o.kt, o.segTexels))
      assert.ok(lay.chunks <= 128 + segments, `${lay.chunks} chunks`)
      // Every fragment of the pass is one (chunk, query, texel), and every
      // chunk's texels are where the combine looks for them.
      const seen = new Set<string>()
      for (let y = 0; y < lay.height; y++)
        for (let px = 0; px < lay.width; px++) {
          const { x, qi, k } = panelChunk(lay, panelW, o.slice, px, y)
          if (k >= lay.chunks) continue
          const at = panelAt(lay, panelW, o.slice, k, qi)
          assert.deepEqual([at.x + x, at.y], [px, y])
          seen.add(`${k} ${qi} ${x}`)
        }
      assert.equal(seen.size, lay.chunks * o.slice * panelW)
    })
  }
})

describe("chunked softmax", () => {
  // One query and head: attention over `keys` keys split as attendPanel
  // splits them (segments of segTexels texels, chunks of chunkTexels,
  // keys past the last masked), each chunk's online softmax, then the
  // combine, against the softmax in float64.
  const run = (
    seed: number,
    keys: number,
    segTexels: number,
    masked: (j: number) => boolean
  ) => {
    const rand = random(seed)
    const dims = 8
    const scores = Array.from({ length: keys }, (_, j) =>
      masked(j) ? -1e30 : 12 * (rand() - 0.5)
    )
    const values = Array.from({ length: keys }, () =>
      Array.from({ length: dims }, () => rand() - 0.5)
    )
    const kt = Math.ceil(keys / 4)
    const segments = Math.ceil(kt / segTexels)
    const lay = panelLayout({
      kt,
      segTexels,
      segments,
      slice: 1,
      panelW: 128,
      maxWidth: 16384,
      chunkTexels: 4,
      maxChunks: 128
    })
    const parts: Array<Partial> = []
    for (let k = 0; k < lay.chunks; k++) {
      const s = Math.floor(k / lay.segChunks)
      const c = k - s * lay.segChunks
      const segKeys = Math.max(
        0,
        Math.min(keys - s * 4 * segTexels, 4 * segTexels)
      )
      const from = s * 4 * segTexels + c * 4 * lay.chunkTexels
      const to =
        s * 4 * segTexels + Math.min(segKeys, (c + 1) * 4 * lay.chunkTexels)
      parts.push(
        from < to
          ? chunkPartial(scores.slice(from, to), values.slice(from, to))
          : { max: -1e30, sum: 0, acc: new Float64Array(dims) }
      )
    }
    const got = combinePartials(parts)
    let mx = -Infinity
    for (const s of scores) mx = Math.max(mx, s)
    const want = new Float64Array(dims)
    let l = 0
    scores.forEach((s, j) => {
      const e = masked(j) ? 0 : Math.exp(s - mx)
      l += e
      for (let d = 0; d < dims; d++) want[d] += e * values[j][d]
    })
    let err = 0
    let ref = 0
    for (let d = 0; d < dims; d++) {
      err += (got[d] - want[d] / l) ** 2
      ref += (want[d] / l) ** 2
    }
    return { rel: Math.sqrt(err / ref), chunks: lay.chunks }
  }

  test("a cache of 1,000 keys", () => {
    const { rel, chunks } = run(1, 1000, 2048, () => false)
    assert.ok(chunks >= 16, `${chunks} chunks`)
    assert.ok(rel < 1e-5, `relative error ${rel}`)
  })

  test("a sliding window: the first 300 keys masked, whole chunks of them empty", () => {
    const { rel } = run(2, 777, 2048, (j) => j < 300)
    assert.ok(rel < 1e-5, `relative error ${rel}`)
  })

  test("masked keys inside chunks, before and after live ones", () => {
    const { rel } = run(3, 513, 2048, (j) => j % 16 < 9 || j > 500)
    assert.ok(rel < 1e-5, `relative error ${rel}`)
  })

  test("segments, the last one short (empty chunks at its end)", () => {
    const { rel, chunks } = run(4, 2100, 64, () => false)
    assert.ok(chunks > 100, `${chunks} chunks`)
    assert.ok(rel < 1e-5, `relative error ${rel}`)
  })
})
