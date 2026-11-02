import assert from "node:assert/strict"
import { describe, test } from "node:test"
import { queriesFit, slotRuns } from "./cache-slots.js"

describe("queries in the cache", () => {
  test("runs cover the slots in order, never crossing a block or a ring's end", () => {
    for (const [slot0, count, blockRows, ring] of [
      [10, 8, 16, 0],
      [14, 8, 16, 0],
      [1020, 32, 1024, 0],
      [250, 12, 1 << 30, 256],
      [255, 1, 1 << 30, 256],
      [0, 4, 4, 0]
    ]) {
      const runs = slotRuns(slot0, count, blockRows, ring)
      let next = 0
      for (const { s0, row0, block, run } of runs) {
        assert.equal(s0, next)
        assert.ok(run > 0 && row0 + run <= blockRows)
        for (let k = 0; k < run; k++) {
          const slot = ring ? (slot0 + s0 + k) % ring : slot0 + s0 + k
          assert.equal(block * blockRows + row0 + k, slot)
          if (ring) assert.ok(slot < ring)
        }
        next += run
      }
      assert.equal(next, count)
    }
  })

  test("a ring never gives up a key the tokens being written still read", () => {
    const window = 32
    for (const capacity of [8, 16, 40]) {
      const ring = window + capacity
      for (const rows of [1, 2, 4, 8]) {
        for (const slots of [1, 2, 4]) {
          for (let pos0 = 0; pos0 < 3 * ring; pos0 += 7) {
            if (!queriesFit({ pos0, rows, slots, ring, window, positions: 0 })) continue
            // The slots the queries take...
            const taken = new Set<number>()
            for (let s = 0; s < rows * slots; s++) taken.add((pos0 + rows + s) % ring)
            // ...hold no key a token at pos0 .. pos0 + rows - 1 reads.
            for (let q = pos0; q < pos0 + rows; q++)
              for (let j = Math.max(0, q - window + 1); j <= q; j++)
                assert.ok(!taken.has(j % ring), `pos0 ${pos0}, rows ${rows}, slots ${slots}, key ${j}`)
          }
        }
      }
    }
  })

  test("a full cache must hold the queries' slots", () => {
    const fit = (pos0: number, positions: number) =>
      queriesFit({ pos0, rows: 1, slots: 4, ring: 0, window: 0, positions })
    assert.ok(fit(1019, 1024))
    assert.ok(!fit(1020, 1024))
  })
})
