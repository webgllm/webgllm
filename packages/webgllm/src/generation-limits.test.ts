import assert from "node:assert/strict"
import { test } from "node:test"
import { outputTokenLimit } from "./generation-limits.js"

test("omitted output limits use remaining model context, including beyond 4096", () => {
  assert.equal(outputTokenLimit(12_000), 12_000)
  assert.equal(outputTokenLimit(123), 123)
})
test("explicit output limits take precedence while respecting the context boundary", () => {
  assert.equal(outputTokenLimit(12_000, 64), 64)
  assert.equal(outputTokenLimit(12_000, 6000), 6000)
  assert.equal(outputTokenLimit(123, 6000), 123)
  assert.equal(outputTokenLimit(123, 0), 0)
})
test("invalid explicit output limits fail instead of being ignored", () => {
  for (const value of [-1, 1.5, NaN, Infinity, Number.MAX_SAFE_INTEGER + 1])
    assert.throws(() => outputTokenLimit(12_000, value), /nonnegative safe integer/)
})
