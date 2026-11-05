import assert from "node:assert/strict"
import test from "node:test"
import { ReadPlan } from "./read-plan.js"

test("contiguous reads share one bounded backing store and preserve every slice's byte offset", async () => {
  const data = Uint8Array.from({ length: 4096 }, (_, i) => (i * 31) & 255)
  const calls: [number, number][] = []
  const source = {
    size: data.length,
    async read(offset: number, length: number) {
      calls.push([offset, length])
      return data.slice(offset, offset + length)
    }
  }
  const plan = new ReadPlan(6, 1024)
  const reads = [
    [10, 100],
    [110, 200],
    [310, 128],
    [600, 512]
  ]
  reads.forEach(([offset, length], i) =>
    plan.add({ key: String(i), source, offset, length })
  )
  const results: Uint8Array[] = []
  for (let i = 0; i < reads.length; i++) {
    const got = await plan.take(String(i))
    results.push(got)
    assert.deepEqual(got, data.subarray(reads[i][0], reads[i][0] + reads[i][1]))
  }
  assert.deepEqual(calls, [
    [10, 428],
    [600, 512]
  ])
  assert.equal(results[0].buffer, results[1].buffer)
  assert.equal(results[1].byteOffset, 100)
  assert.ok(calls.every(([, bytes]) => bytes <= 512))
})

test("coalescing cannot retain a backing store beyond the read-ahead byte cap", async () => {
  const reads = Array.from({ length: 24 }, (_, i) => ({
    offset: i * 64,
    length: i === 23 ? 256 : 64,
    done: false
  }))
  const allocations: { length: number; items: typeof reads }[] = []
  let live = 0,
    peak = 0
  const source = {
    size: 4096,
    async read(offset: number, length: number) {
      allocations.push({
        length,
        items: reads.filter(
          (r) => r.offset >= offset && r.offset + r.length <= offset + length
        )
      })
      live += length
      peak = Math.max(peak, live)
      return new Uint8Array(length)
    }
  }
  const plan = new ReadPlan(6, 512)
  reads.forEach((r, i) =>
    plan.add({ key: String(i), source, offset: r.offset, length: r.length })
  )
  for (let i = 0; i < reads.length; i++) {
    assert.equal((await plan.take(String(i))).length, reads[i].length)
    reads[i].done = true
    for (const group of allocations)
      if (group.length && group.items.every((r) => r.done)) {
        live -= group.length
        group.length = 0
      }
    assert.ok(live <= 512)
  }
  assert.ok(peak <= 512 + 256)
  assert.equal(live, 0)
})
