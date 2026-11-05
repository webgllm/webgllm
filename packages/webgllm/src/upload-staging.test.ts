import test from "node:test"
import assert from "node:assert/strict"
import { newQuantMatrix } from "./repack.js"
import { uploadedGgufWeights, canReleaseStaging } from "./upload-staging.js"
import type { WeightPiece } from "./weights.js"

test("owned GGUF matrices remain intact until consumed, then release their backing stores", async () => {
  const matrix = newQuantMatrix(4, 32, 4, false)
  matrix.quants[0] = 123
  async function* source(): AsyncGenerator<WeightPiece> {
    yield { kind: "embedding", value: matrix }
  }
  const iterator = uploadedGgufWeights(source())
  const piece = await iterator.next()
  assert.equal(piece.done, false)
  assert.equal(matrix.quants[0], 123)
  await iterator.next()
  if (canReleaseStaging) {
    assert.equal(matrix.quants.byteLength, 0)
    assert.equal(matrix.scales.byteLength, 0)
  }
})

test("raw/aliased sources are retained and closing an interrupted upload releases owned matrices", async () => {
  const raw = newQuantMatrix(4, 32, 4, false),
    owned = newQuantMatrix(4, 32, 4, false)
  raw.raw = new Uint8Array(64)
  async function* source(): AsyncGenerator<WeightPiece> {
    yield { kind: "embedding", value: raw }
    yield { kind: "embedding", value: owned }
  }
  const iterator = uploadedGgufWeights(source())
  await iterator.next()
  await iterator.next()
  assert.ok(raw.quants.byteLength > 0)
  await iterator.return(undefined)
  if (canReleaseStaging) assert.equal(owned.quants.byteLength, 0)
  assert.equal(raw.raw.byteLength, 64)
})
