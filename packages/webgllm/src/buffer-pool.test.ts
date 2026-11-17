import assert from "node:assert/strict"
import test from "node:test"
import { BufferPool } from "./byte-source.js"

test("a buffer pool hands back a given buffer for the same length only, up to its cap", () => {
  const pool = new BufferPool(48)
  const a = pool.take(16)
  pool.give(a.buffer)
  assert.equal(pool.take(32).buffer === a.buffer, false, "another length gets a new buffer")
  assert.equal(pool.take(16).buffer, a.buffer)
  // Past the cap, a buffer isn't kept.
  const big = new ArrayBuffer(40)
  pool.give(new ArrayBuffer(16))
  pool.give(big)
  assert.notEqual(pool.take(40).buffer, big)
  pool.clear()
  assert.equal(pool.take(16).byteLength, 16)
})
