import test from "node:test"
import assert from "node:assert/strict"
import { expandQ6 } from "./q6-expanded.js"
import { packQ6_K, newQuantMatrix, dequantizeQ6_K } from "./repack.js"
import { halfTable } from "./half.js"

test("byte expansion preserves every Q6_K weight exactly across row groups and chunks", () => {
  const rows = 12,
    K = 512,
    blocks = K / 256,
    raw = new Uint8Array(rows * blocks * 210)
  let seed = 7
  for (let i = 0; i < raw.length; i++) {
    seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0
    raw[i] = seed >>> 24
  }
  for (let i = 0; i < rows * blocks; i++) {
    raw[i * 210 + 208] = 0x00
    raw[i * 210 + 209] = 0x24
  }
  const m = newQuantMatrix(rows, K, 6, false)
  packQ6_K(m, 0, raw, rows)
  const reference = new Float32Array(rows * K)
  dequantizeQ6_K(raw, rows, K, reference)
  const half = halfTable()
  for (const [start, count] of [
    [0, 3],
    [1, 2],
    [2, 1]
  ]) {
    const got = expandQ6(m, start, count)
    for (let g = 0; g < count; g++)
      for (let r = 0; r < 4; r++)
        for (let k = 0; k < K; k++) {
          const sb = k >> 8,
            j = k & 255,
            at = ((g * blocks + sb) * 4 + r) * 256 + j
          const scaleAt = ((g * blocks + sb) * 4 + r) * 256 + (j >> 4) * 16
          let packedScale = 0
          for (let part = 0; part < 4; part++)
            packedScale |= (got.quants[scaleAt + part] >> 6) << (2 * part)
          const scale = packedScale >= 128 ? packedScale - 256 : packedScale
          const d = half[m.scales[((start + g) * blocks + sb) * 4 + r]]
          const actual = Math.fround(d * scale * ((got.quants[at] & 63) - 32))
          assert.equal(actual, reference[((start + g) * 4 + r) * K + k])
        }
  }
})

test("head expansion is rejected unless scratch savings cover both head and activation memory", async () => {
  const { fitsExpandedHead } = await import("./q6-expanded.js")
  const mib = 2 ** 20
  assert.equal(
    fitsExpandedHead(46.875 * mib, 60 * mib, 4 * mib, 8.4375 * mib),
    true
  )
  assert.equal(fitsExpandedHead(46.875 * mib, 32 * mib, 32 * mib, 0), false)
  assert.equal(
    fitsExpandedHead(46.875 * mib, 60 * mib, 4 * mib, 16.875 * mib),
    false
  )
  assert.equal(fitsExpandedHead(1, Infinity, 0, 0), false)
})

test("extra head partial-sum storage cannot consume an unfunded memory budget", async () => {
  const { fitsExpandedHead } = await import("./q6-expanded.js"),
    mib = 2 ** 20
  assert.equal(
    fitsExpandedHead((46.875 + 8) * mib, 60 * mib, 4 * mib, 8.4375 * mib),
    false
  )
})

test("head scratch accounting includes the whole new target when switching from an unsplit head", async () => {
  const { headScratchBytes } = await import("./q6-expanded.js")
  assert.equal(headScratchBytes(1, 8, 32, 1024), 0)
  assert.equal(headScratchBytes(2, 8, 32, 1024), 8 * 2 ** 20)
  assert.equal(headScratchBytes(4, 8, 32, 1024), 16 * 2 ** 20)
})

test("automatic scratch reuses the former tuner's two-buffer budget without increasing its retained ceiling", async () => {
  const { fundedF16Scratch } = await import("./q6-expanded.js")
  const mib = 2 ** 20,
    oldScratch = 60 * mib,
    oneActivation = 8.4375 * mib,
    head = 46.875 * mib
  const next = fundedF16Scratch(
    oldScratch,
    2 * oneActivation,
    oneActivation,
    head,
    7680 * 128
  )
  assert.equal(next, 20 * mib)
  assert.ok(next + oneActivation + head <= oldScratch + 2 * oneActivation)
  assert.equal(
    fundedF16Scratch(
      oldScratch,
      2 * oneActivation,
      oneActivation,
      0,
      7680 * 128
    ),
    oldScratch
  )
  assert.equal(fundedF16Scratch(2 * mib, 0, 0, 0, 1024 * 128), 2 * mib)
})
