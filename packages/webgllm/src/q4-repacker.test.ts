import assert from "node:assert/strict"
import test from "node:test"
import { q4Repacker, quantsFromRaw } from "./repack.js"

test("the WebAssembly repack of raw MatMulNBits rows matches quantsFromRaw", () => {
  const repack = q4Repacker()
  assert.ok(repack, "WebAssembly SIMD is available in Node")
  for (const [N, K] of [[4, 32], [12, 2048], [64, 7680], [8, 64]]) {
    const raw = new Uint8Array((N * K) / 2)
    for (let i = 0; i < raw.length; i++) raw[i] = Math.imul(i + 1, 2654435761) >>> 24
    assert.deepEqual(Array.from(repack(raw, N, K)), Array.from(quantsFromRaw(raw, N, K)), `${N}×${K}`)
  }
})
