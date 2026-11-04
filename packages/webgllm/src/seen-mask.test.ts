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
test("compact CPU seen bits preserve every GPU seen flag and padding ban", () => {
  const { engine, writes } = fixture()
  engine.setSeen([0, 7, 8, 15, 18, 19, 0], 19)
  const expected = new Uint8Array(32).fill(2, 19)
  for (const t of [0, 7, 8, 15, 18]) expected[t] = 1
  assert.deepEqual(writes[0].pixels, expected)
  assert.equal(Reflect.get(engine, "maskData").byteLength, 4)
  engine.markSeen(5)
  engine.markSeen(5)
  engine.markSeen(20)
  engine.markSeen(-1)
  engine.markSeen(32)
  assert.equal(writes.length, 2)
  assert.deepEqual(writes[1], {
    rect: { x: 5, y: 0, width: 1, height: 1 },
    pixels: new Uint8Array([1])
  })
  engine.markSeen(17)
  assert.deepEqual(writes[2].rect, { x: 1, y: 1, width: 1, height: 1 })
})
test("resetting the seen set clears old bits and retains the full GPU mask semantics", () => {
  const { engine, writes } = fixture()
  engine.setSeen([0, 1], 32)
  const bits = Reflect.get(engine, "maskData")
  engine.setSeen([2], 32)
  assert.equal(Reflect.get(engine, "maskData"), bits)
  engine.markSeen(0)
  assert.equal(writes.length, 3)
  assert.equal(writes[2].pixels[0], 1)
  engine.setSeen([], 0)
  engine.markSeen(0)
  assert.equal(writes.length, 4)
  assert.ok(writes[3].pixels.every((x) => x === 2))
})
