import test from "node:test"
import assert from "node:assert/strict"
import { ReadPlan } from "./read-plan.js"
function fixture(retire = true, aliased = false) {
  const data = Uint8Array.from({ length: 128 }, (_, i) => i)
  const source = {
    size: 128,
    read: async (offset: number, length: number) =>
      aliased
        ? data.subarray(offset, offset + length)
        : data.slice(offset, offset + length)
  }
  const plan = new ReadPlan(3, 64, retire)
  plan.add({ key: "a", source, offset: 16, length: 16 })
  plan.add({ key: "b", source, offset: 32, length: 16 })
  plan.add({ key: "c", source, offset: 64, length: 64 })
  return { plan, data }
}
test("shared read survives until every slice was converted, then retires its backing store", async () => {
  const { plan } = fixture(),
    a = await plan.take("a"),
    buffer = a.buffer
  plan.release(a)
  assert.equal(buffer.byteLength, 32)
  const b = await plan.take("b")
  assert.deepEqual(
    [...b],
    Array.from({ length: 16 }, (_, i) => i + 32)
  )
  plan.release(b)
  assert.equal(buffer.byteLength, 0)
})
test("concurrent borrowed slices and duplicate releases cannot detach a live consumer", async () => {
  const { plan } = fixture(),
    [a, b] = await Promise.all([plan.take("a"), plan.take("b")])
  plan.release(a)
  plan.release(a)
  assert.equal(b.buffer.byteLength, 32)
  assert.equal(b[0], 32)
  plan.release(b)
  assert.equal(b.buffer.byteLength, 0)
})
test("default and external aliased sources preserve their input ownership", async () => {
  for (const [retire, aliased] of [
    [false, false],
    [true, true]
  ]) {
    const { plan, data } = fixture(retire, aliased),
      a = await plan.take("a"),
      b = await plan.take("b")
    plan.release(a)
    plan.release(b)
    assert.equal(a[0], 16)
    assert.equal(b[0], 32)
    assert.equal(data.byteLength, 128)
  }
})
test("inline tensor bytes remain caller-owned even with retirement enabled", async () => {
  const data = new Uint8Array(16).fill(7),
    source = { size: 16, read: async () => data },
    plan = new ReadPlan(3, 64, true)
  plan.add({ key: "inline", source, offset: 0, length: 16, bytes: data })
  const got = await plan.take("inline")
  plan.release(got)
  assert.equal(data.byteLength, 16)
  assert.equal(data[0], 7)
})
