import assert from "node:assert/strict"
import test from "node:test"
import { ReadPlan } from "./read-plan.js"

test("six-way read-ahead keeps the byte cap plus only the current consumed slice", async () => {
  const cap = 512,
    plan = new ReadPlan(6, cap)
  let pendingBytes = 0,
    pendingCount = 0,
    peakBytes = 0,
    peakCount = 0
  const lengths = Array.from({ length: 30 }, (_, i) => (i % 9 === 8 ? 400 : 64))
  const source = {
    size: 8192,
    async read(_offset: number, length: number) {
      pendingBytes += length
      pendingCount++
      peakBytes = Math.max(peakBytes, pendingBytes)
      peakCount = Math.max(peakCount, pendingCount)
      return new Uint8Array(length)
    }
  }
  let offset = 0
  for (let i = 0; i < lengths.length; i++) {
    plan.add({ key: String(i), source, offset, length: lengths[i] })
    offset += lengths[i] + 16
  }
  for (let i = 0; i < lengths.length; i++) {
    const bytes = await plan.take(String(i))
    pendingBytes -= bytes.length
    pendingCount--
    assert.ok(pendingBytes <= cap)
  }
  assert.ok(peakCount > 4 && peakCount <= 7)
  assert.ok(peakBytes <= cap + Math.max(...lengths))
})
