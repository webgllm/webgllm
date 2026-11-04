import assert from "node:assert/strict"
import test from "node:test"
import { Engine } from "./engine.js"
import { GL } from "./gpu/gl.js"

for (const [decodeWins, prefillWins] of [[false, true], [true, false], [true, true]])
test(`dependent storage timing: decode win ${decodeWins}, prefill win ${prefillWins}`, async (t) => {
  let clock = 0, prefillCalls = 0
  t.mock.method(performance, "now", () => clock)
  let pending = false,
    previous: unknown,
    last: unknown,
    reads = 0,
    uploads = 0
  const deleted: unknown[] = []
  const state = {
    buffers: { normed: {}, partial: {} },
    layoutPrefill: async () => { prefillCalls++; return { rgba4: prefillWins ? 6 : 10, bytes: 8 } },
    allocateMatrix: (
      _N: number,
      _K: number,
      _bits: number,
      _zeros: boolean,
      rgba4: boolean
    ) => ({ groups: 256, quants: {}, scales: {}, zeros: null, rgba4 }),
    writeMatrix: () => undefined,
    splitPartials(_input: unknown, gm: { rgba4: boolean }) {
      clock += gm.rgba4 ? decodeWins ? 2 : 4 : 3
      assert.equal(pending, false)
      pending = true
      return 4
    },
    gpu: {
      texture: (_format: number, width: number, height: number) => ({
        width,
        height
      }),
      write: (
        _texture: unknown,
        _rect: unknown,
        data: { format: number; pixels: Uint32Array }
      ) => {
        assert.equal(data.format, GL.RGBA_INTEGER)
        assert.ok(data.pixels.every((word) => word === 0x3c003800))
        uploads++
      },
      program: (source: string) => source,
      run: (pass: {
        inputs: { uPrev: unknown }
        output: unknown
        rect: { width: number; height: number }
      }) => {
        assert.notEqual(pass.inputs.uPrev, pass.output)
        if (previous) assert.equal(pass.inputs.uPrev, previous)
        assert.equal(pass.rect.width, 4)
        assert.equal(pass.rect.height, 4)
        previous = last = pass.output
      },
      pace: async (ahead: number) => {
        assert.equal(ahead, 1)
        await Promise.resolve()
        pending = false
      },
      read: async (
        texture: unknown,
        rect: { width: number; height: number }
      ) => {
        assert.equal(pending, false)
        assert.equal(texture, last)
        assert.deepEqual(rect, { x: 0, y: 0, width: 4, height: 4 })
        reads++
        return new Float32Array(64)
      },
      delete: (texture: unknown) => deleted.push(texture)
    }
  }
  await Reflect.get(Engine.prototype, "chooseLayout").call(state, {
    N: 1024,
    K: 128,
    bits: 4,
    scales: new Uint16Array(4096),
    zeros: null
  })
  assert.equal(Reflect.get(state, "rgba4Layers"), decodeWins && prefillWins)
  assert.equal(prefillCalls, decodeWins ? 1 : 0)
  assert.equal(uploads, 1)
  assert.ok(reads >= 3)
  assert.equal(deleted.length, 6)
})
