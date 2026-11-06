import assert from "node:assert/strict"
import { describe, test } from "node:test"
import { fromHalf, halfTable, halvesToFloats, toHalf } from "./half.js"

describe("half floats", () => {
  test("values a half holds exactly round-trip", () => {
    for (const v of [0, 1, -1, 0.5, 2, 65504, -65504, 6.103515625e-5, 5.960464477539063e-8, 0.333251953125]) {
      assert.equal(fromHalf(toHalf(v)), v, `${v}`)
    }
  })

  test("rounds to the nearest half", () => {
    assert.equal(fromHalf(toHalf(1 + 2 ** -11)), 1) // a tie rounds to even
    assert.equal(fromHalf(toHalf(1 + 3 * 2 ** -12)), 1 + 2 ** -10)
  })

  test("overflow is infinity, NaN stays NaN, signed zero kept", () => {
    assert.equal(fromHalf(toHalf(1e6)), Infinity)
    assert.equal(fromHalf(toHalf(-1e6)), -Infinity)
    assert.ok(Number.isNaN(fromHalf(toHalf(NaN))))
    assert.ok(Object.is(fromHalf(toHalf(-0)), -0))
  })

  test("the table and halvesToFloats agree with fromHalf", () => {
    const table = halfTable()
    for (const h of [0, 1, 0x3c00, 0x7bff, 0x8001, 0xc000, 0x7c00, 0x0400]) {
      assert.equal(table[h], fromHalf(h), `0x${h.toString(16)}`)
    }
    const bytes = new Uint8Array(new Uint16Array([0x3c00, 0xc000, 0x3800]).buffer)
    assert.deepEqual([...halvesToFloats(bytes)], [1, -2, 0.5])
  })
})
