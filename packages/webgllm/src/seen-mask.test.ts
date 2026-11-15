import test from "node:test"
import assert from "node:assert/strict"
import { Engine } from "./engine.js"
function fixture() {
  const writes: Array<{
    rect: { x: number; y: number; width: number; height: number }
    pixels: Uint8Array
  }> = []
  const engine = Object.create(Engine.prototype) as Engine
  Object.assign(engine, {
    buffers: { mask: { width: 16, height: 2 } },
    gpu: {
      write: (
        _target: unknown,
        rect: { x: number; y: number; width: number; height: number },
        data: { pixels: Uint8Array }
      ) => writes.push({ rect, pixels: data.pixels.slice() })
    }
  })
  return { engine, writes }
}
test("compact CPU bits preserve prompt, generated and padding flags", () => {
  const { engine, writes } = fixture()
  engine.setSeen([0, 7, 8, 15, 18, 19, 0], 19)
  const expected = new Uint8Array(32).fill(2, 19)
  for (const t of [0, 7, 8, 15, 18]) expected[t] = 1
  assert.deepEqual(writes[0].pixels, expected)
  assert.equal(Reflect.get(engine, "maskStates").byteLength, 19)
  engine.markSeen(7)
  engine.markSeen(7)
  engine.markSeen(5)
  engine.markSeen(5)
  engine.markSeen(20)
  engine.markSeen(-1)
  engine.markSeen(32)
  assert.equal(writes.length, 3)
  assert.deepEqual(writes[1], {
    rect: { x: 7, y: 0, width: 1, height: 1 },
    pixels: new Uint8Array([3])
  })
  assert.deepEqual(writes[2], {
    rect: { x: 5, y: 0, width: 1, height: 1 },
    pixels: new Uint8Array([3])
  })
  engine.markSeen(17)
  assert.deepEqual(writes[3].rect, { x: 1, y: 1, width: 1, height: 1 })
})
test("resetting the seen set clears old bits and retains the full GPU mask semantics", () => {
  const { engine, writes } = fixture()
  engine.setSeen([0, 1], 32)
  engine.markSeen(0)
  const bits = Reflect.get(engine, "maskStates")
  engine.setSeen([2], 32)
  assert.equal(Reflect.get(engine, "maskStates"), bits)
  engine.markSeen(0)
  assert.equal(writes.length, 4)
  assert.equal(writes[3].pixels[0], 3)
  engine.setSeen([], 0)
  engine.markSeen(0)
  assert.equal(writes.length, 5)
  assert.ok(writes[4].pixels.every((x) => x === 2))
})

test("a token leaving the penalty window goes back to unseen; only changes are written", () => {
  const { engine, writes } = fixture()
  engine.setSeen([4], 32)
  engine.setSeenState(4, 0)
  engine.setSeenState(4, 0)
  engine.setSeenState(9, 3)
  engine.setSeenState(9, 1)
  assert.deepEqual(writes.slice(1).map((w) => [w.rect.x + w.rect.y * 16, w.pixels[0]]), [[4, 0], [9, 3], [9, 1]])
})
