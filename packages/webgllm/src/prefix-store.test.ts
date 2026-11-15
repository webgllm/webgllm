import assert from "node:assert/strict"
import test from "node:test"
import { Engine } from "./engine.js"
import type { SavedPrefix } from "./engine.js"
import { GL } from "./gpu/gl.js"
import * as K from "./gpu/kernels.js"
import type { Pass, Program, Texture } from "./gpu/gl.js"
import { PrefixStore, savePromptPrefix } from "./prefix-store.js"
import { persistPrefix, PrefixDisk, untilAborted } from "./prefix-disk.js"
import type { PrefixFolder } from "./prefix-disk.js"

// The store's own bookkeeping, with plain values for states.

function store(budget: number) {
  const freed: string[] = []
  return { freed, s: new PrefixStore<string>((state) => freed.push(state), budget) }
}

test("match finds the longest stored prefix of the keys, within the limit", () => {
  const { s } = store(100)
  s.add([1, 2], 10, "a")
  s.add([1, 2, 3, 4], 10, "b")
  s.add([1, 5, 6], 10, "c")
  assert.equal(s.match([1, 2, 3, 4, 9])?.state, "b")
  assert.equal(s.match([1, 2, 3, 9])?.state, "a")
  assert.equal(s.match([1, 5, 6, 7])?.state, "c")
  // (The prompt's last token is read for its logits: never the whole prompt.)
  assert.equal(s.match([1, 2, 3, 4], 3)?.state, "a")
  assert.equal(s.match([7, 1, 2]), undefined)
  assert.equal(s.match([1]), undefined)
})

test("the least recently used entries go first past the budget", () => {
  const { s, freed } = store(30)
  s.add([1], 10, "a")
  s.add([2], 10, "b")
  s.add([3], 10, "c")
  // A match uses "a": "b" is the oldest now.
  assert.equal(s.match([1, 9])?.state, "a")
  s.add([4], 10, "d")
  assert.deepEqual(freed, ["b"])
  // (has counts as a use too: "d" is the oldest now.)
  assert.ok(s.has([3]))
  s.add([5], 20, "e")
  assert.deepEqual(freed, ["b", "a", "d"])
  assert.equal(s.bytes, 30)
  assert.equal(s.size, 2)
})

test("a state past the whole budget is freed, not kept", () => {
  const { s, freed } = store(10)
  assert.equal(s.add([1], 11, "big"), false)
  assert.deepEqual(freed, ["big"])
  assert.equal(s.size, 0)
  assert.equal(s.fits(10), true)
  assert.equal(s.fits(0), false)
})

test("the same keys again replace the entry, freeing the old state", () => {
  const { s, freed } = store(100)
  s.add([1, 2], 10, "old")
  s.add([1, 2], 12, "new")
  assert.deepEqual(freed, ["old"])
  assert.equal(s.size, 1)
  assert.equal(s.match([1, 2, 3])?.state, "new")
})

test("a smaller budget frees entries now; 0 keeps none; clear frees all", () => {
  const { s, freed } = store(100)
  s.add([1], 40, "a")
  s.add([2], 40, "b")
  s.setBudget(50)
  assert.deepEqual(freed, ["a"])
  s.setBudget(0)
  assert.deepEqual(freed, ["a", "b"])
  assert.equal(s.add([3], 1, "c"), false)
  s.setBudget(100)
  s.add([4], 1, "d")
  s.clear()
  assert.deepEqual(freed, ["a", "b", "c", "d"])
  assert.throws(() => s.setBudget(-1), RangeError)
  assert.throws(() => s.setBudget(Number.NaN), RangeError)
})

test("savePromptPrefix keeps the system prompt's end once, and never a prefix with an image in it", () => {
  const saves: number[] = []
  const engine = {
    at: 4 as number | undefined,
    snapshotPosition() { return this.at },
    savedPrefixBytes: (at: number) => at * 10,
    savePrefix: (at: number) => { saves.push(at); return `state@${at}` }
  }
  const { s } = store(1000)
  const keys = [1, 2, 3, 4, 5, 6]
  savePromptPrefix(engine, s, 6, keys, [], 4)
  savePromptPrefix(engine, s, 6, keys, [], 4)
  assert.deepEqual(saves, [4])
  assert.equal(s.match([1, 2, 3, 4, 7])?.state, "state@4")
  // The prefix snapshot elsewhere (a shorter boundary learned from the live history): not the system prompt's.
  engine.at = 2
  savePromptPrefix(engine, s, 6, keys, [], 5)
  // An image before the prefix's end: its rows hang on more than the keys.
  engine.at = 5
  savePromptPrefix(engine, s, 6, keys, [{ start: 2 }], 5)
  // Not read that far, no system prompt, or past the budget: nothing kept.
  savePromptPrefix(engine, s, 4, keys, [], 5)
  savePromptPrefix(engine, s, 6, keys, [], undefined)
  s.setBudget(49)
  savePromptPrefix(engine, s, 6, keys, [], 5)
  assert.deepEqual(saves, [4])
})

// The engine's save and restore, on a GPU that runs its copy kernels on
// plain arrays: one number per texel.

type Layer = {
  conv?: { state: Texture[] }
  attention?: { kv: Texture; ring?: number }
}

function engineFixture() {
  const engine = Object.create(Engine.prototype) as Engine
  const live = new Set<Texture>()
  const data = new Map<Texture, Float64Array>()
  const sources = new Map<Program, string>()
  let id = 0
  let failAt = -1
  let allocations = 0
  const gpu = {
    readsPerSlice: 0,
    activePhase: "decode",
    isLost: false,
    setPhase() {},
    outOfMemory: () => false,
    pace: async () => undefined,
    texture(internal: number, width: number, height: number, upload?: { pixels: ArrayBufferView }): Texture {
      if (++allocations === failAt) throw new Error("injected allocation failure")
      const tex = { id: id++, tex: {}, internal, width, height, bytes: width * height * 16 } as Texture
      live.add(tex)
      const values = new Float64Array(width * height)
      // (A texel's value in its first word, as readBytes writes it.)
      if (upload) {
        const p = upload.pixels as Float32Array | Uint32Array
        const words = new Float32Array(p.buffer, p.byteOffset, p.byteLength / 4)
        for (let i = 0; i < values.length; i++) values[i] = words[i * 4]
      }
      data.set(tex, values)
      return tex
    },
    async readBytes(tex: Texture) {
      assert.ok(live.has(tex), `read of a missing or freed texture ${tex.id}`)
      const words = new Float32Array(tex.width * tex.height * 4)
      data.get(tex)!.forEach((v, i) => { words[i * 4] = v })
      return new Uint8Array(words.buffer)
    },
    program(source: string): Program {
      const program = { program: {}, uniforms: new Map(), samplers: [] } as unknown as Program
      sources.set(program, source)
      return program
    },
    run(pass: Pass) {
      const input = pass.inputs.uX as Texture
      const output = pass.output as Texture
      for (const tex of [input, output]) assert.ok(live.has(tex), `use of a missing or freed texture ${tex?.id}`)
      const from = data.get(input)!
      const to = data.get(output)!
      const source = sources.get(pass.program)
      const rect = pass.rect ?? { x: 0, y: 0, width: output.width, height: output.height }
      const u = (name: string) => (pass.uniforms?.[name] as { int: number }).int
      for (let y = rect.y; y < rect.y + rect.height; y++)
        for (let x = rect.x; x < rect.x + rect.width; x++) {
          const fromY = source === K.COPY_POSITIONS ? (u("uPos0") + (y - u("uRow0"))) % u("uMod") : y
          assert.ok(x < input.width && fromY < input.height, "read outside the input")
          to[y * output.width + x] = from[fromY * input.width + x]
        }
    },
    write(tex: Texture) {
      assert.ok(live.has(tex))
      data.get(tex)!.fill(0)
    },
    delete(tex: Texture) {
      assert.ok(live.delete(tex), `double deletion of texture ${tex.id}`)
    }
  }
  // 1 KV head of 8: 2 texels a cache row; blocks of 4 rows side by side.
  const kvShift = 2
  const kvWidth = 2
  const kvFor = (positions: number) => gpu.texture(GL.RGBA32UI, kvWidth * Math.ceil(positions / 4), Math.min(positions, 4))
  const layers: Layer[] = [
    { conv: { state: [gpu.texture(GL.RGBA32F, 2, 3)] } },
    { attention: { kv: kvFor(8) } },
    // A sliding window of 2, in a ring of 4 rows.
    { attention: { kv: gpu.texture(GL.RGBA32UI, kvWidth, 4), ring: 4 } },
    { attention: { kv: kvFor(8) } }
  ]
  Object.assign(engine, {
    gpu, layers, position: 0, contextLength: 64, kvShift, kvPositions: 8,
    guessAt: null, draft: null, images: [], snapshots: new Map(),
    config: { hiddenSize: 8, convKernel: 4, slidingWindow: 2, numHeads: 1, numKvHeads: 1, headDim: 8 }
  })
  const full = () => layers.filter((l) => l.attention && !l.attention.ring).map((l) => l.attention!.kv)
  /** Writes what a prompt `tag` leaves in the cache after `n` tokens: position p of each full layer holds tag·1000 + p. */
  const read = (tag: number, n: number) => {
    const e = engine as unknown as { position: number; kvPositions: number }
    for (const kv of full()) {
      const d = data.get(kv)!
      for (let p = 0; p < n; p++)
        for (let x = 0; x < kvWidth; x++) d[(p & 3) * kv.width + (p >> 2) * kvWidth + x] = tag * 1000 + p
    }
    data.get(layers[0].conv!.state[0])!.fill(tag)
    const ring = layers[2].attention!.kv
    for (let p = Math.max(0, n - 2); p < n; p++) data.get(ring)!.fill(tag * 1000 + p, (p % 4) * kvWidth, (p % 4 + 1) * kvWidth)
    e.position = n
  }
  /** Position p of the full layers' caches (one per layer). */
  const at = (p: number) => full().map((kv) => data.get(kv)![(p & 3) * kv.width + (p >> 2) * kvWidth])
  const snapshots = () => (engine as unknown as { snapshots: Map<number, { prefix?: boolean }> }).snapshots
  return {
    engine, live, data, layers, read, at, snapshots,
    failOn: (n: number) => { failAt = allocations + n },
    growTo: (positions: number) => {
      for (const l of layers) if (l.attention && !l.attention.ring) {
        const bigger = kvFor(positions)
        gpu.run({ program: gpu.program(K.COPY_POSITIONS), inputs: { uX: l.attention.kv }, output: bigger,
          uniforms: { uPos0: { int: 0 }, uRow0: { int: 0 }, uMod: { int: l.attention.kv.height } },
          rect: { x: 0, y: 0, width: l.attention.kv.width, height: l.attention.kv.height } } as unknown as Pass)
        gpu.delete(l.attention.kv)
        l.attention.kv = bigger
      }
      ;(engine as unknown as { kvPositions: number }).kvPositions = positions
    }
  }
}

test("a saved prefix brings back its cache rows after another prompt overwrote them (A, B, A)", async () => {
  const f = engineFixture()
  const { engine } = f
  // Prompt A: 8 tokens, its system prompt's end (6, inside the second block) saved as the prefix.
  f.read(1, 6)
  engine.snapshot(undefined, "prefix")
  f.read(1, 8)
  const bytes = engine.savedPrefixBytes(6)
  const before = f.live.size
  const saved = engine.savePrefix(6)!
  assert.equal(saved.position, 6)
  // Its own textures: the conv state, the window, and each full layer's 2 blocks of 4 rows.
  assert.equal(f.live.size - before, 4)
  assert.equal(bytes, [...saved.state.conv, ...saved.state.windows, ...saved.kv].reduce((sum, t) => sum + t.bytes, 0))
  assert.deepEqual(saved.kv.map((t) => [t.width, t.height]), [[4, 4], [4, 4]])

  // Prompt B overwrites everything, and has its own snapshots; the cache grows meanwhile.
  engine.rewind(0)
  f.growTo(16)
  f.read(2, 8)
  engine.snapshot(undefined, "prefix")
  f.read(2, 12)
  engine.snapshot()
  assert.deepEqual(f.at(3), [2003, 2003])

  // Back to A: its rows, its conv state, its window, at 6.
  const restored = await engine.restorePrefix(saved)
  assert.equal(restored, 6)
  assert.equal(engine.position, 6)
  for (let p = 0; p < 6; p++) assert.deepEqual(f.at(p), [1000 + p, 1000 + p])
  assert.ok(f.data.get(f.layers[0].conv!.state[0])!.every((v) => v === 1))
  const ring = f.data.get(f.layers[2].attention!.kv)!
  assert.equal(ring[(4 % 4) * 2], 1004)
  assert.equal(ring[(5 % 4) * 2], 1005)
  // B's snapshots went; a copy of A's is the prefix, its own textures.
  assert.deepEqual([...f.snapshots().keys()], [6])
  assert.equal(engine.snapshotPosition("prefix"), 6)

  // The entry and the live snapshot free apart (no double deletion), and a
  // rewind along the new history stops at the prefix.
  engine.freePrefix(saved)
  f.read(1, 10)
  assert.equal(engine.rewindPosition(9), 6)
  assert.equal(engine.rewind(9), 6)
  for (let p = 0; p < 6; p++) assert.deepEqual(f.at(p), [1000 + p, 1000 + p])
})

test("restoring grows a cache too small for the prefix", async () => {
  const f = engineFixture()
  const { engine } = f
  f.growTo(16)
  f.read(1, 14)
  engine.snapshot(undefined, "prefix")
  const saved = engine.savePrefix(14)!
  assert.deepEqual(saved.kv.map((t) => [t.width, t.height]), [[8, 4], [8, 4]])
  // A new sequence starts small again.
  engine.reset()
  assert.equal((engine as unknown as { kvPositions: number }).kvPositions, 4)
  await engine.restorePrefix(saved)
  assert.ok((engine as unknown as { kvPositions: number }).kvPositions >= 14)
  for (let p = 0; p < 14; p++) assert.deepEqual(f.at(p), [1000 + p, 1000 + p])
  engine.freePrefix(saved)
})

test("no saved prefix without a snapshot there, or past what the cache holds", () => {
  const f = engineFixture()
  f.read(1, 9)
  assert.equal(f.engine.savePrefix(6), undefined)
  assert.equal(f.engine.savedPrefixBytes(6), 0)
  f.read(1, 6)
  f.engine.snapshot(undefined, "prefix")
  ;(f.engine as unknown as { position: number }).position = 5
  assert.equal(f.engine.savePrefix(6), undefined)
})

test("a save that fails partway frees what it made", () => {
  for (let n = 1; n <= 4; n++) {
    const f = engineFixture()
    f.read(1, 6)
    f.engine.snapshot(undefined, "prefix")
    const before = f.live.size
    f.failOn(n)
    assert.throws(() => f.engine.savePrefix(6), /injected allocation failure/)
    assert.equal(f.live.size, before, `textures left after failing at allocation ${n}`)
  }
})

test("prefixes saved in a store are freed with it", () => {
  const f = engineFixture()
  const s = new PrefixStore<SavedPrefix>((saved) => f.engine.freePrefix(saved), 1 << 20)
  f.read(1, 6)
  f.engine.snapshot(undefined, "prefix")
  const before = f.live.size
  savePromptPrefix(f.engine, s, 6, [1, 2, 3, 4, 5, 6], [], 6)
  assert.equal(s.size, 1)
  assert.ok(f.live.size > before)
  s.clear()
  assert.equal(f.live.size, before)
})

// On disk: a folder in memory, as OPFS would keep it.

function memoryFolder() {
  const files = new Map<string, { bytes: Uint8Array }>()
  const reads: string[] = []
  const folder: PrefixFolder = {
    names: async () => [...files.keys()],
    async read(name, from = 0, to) {
      const file = files.get(name)
      if (!file) throw new Error(`No file ${name}`)
      reads.push(`${name}@${from}`)
      const bytes = file.bytes.slice(from, to ?? file.bytes.byteLength)
      return { size: file.bytes.byteLength, bytes: bytes.buffer }
    },
    async write(name, parts) {
      const size = parts.reduce((sum, p) => sum + p.byteLength, 0)
      const bytes = new Uint8Array(size)
      let at = 0
      for (const p of parts) { bytes.set(p, at); at += p.byteLength }
      files.set(name, { bytes })
    },
    async writeAt(name, at, bytes) {
      const file = files.get(name)
      if (!file) throw new Error(`No file ${name}`)
      file.bytes.set(bytes, at)
    },
    async remove(name) { files.delete(name) }
  }
  return { files, folder, reads }
}

const texture = (value: number, width = 2, height = 2) => {
  const bytes = new Uint8Array(width * height * 16)
  bytes.fill(value)
  return { internal: GL.RGBA32UI, width, height, bytes }
}

test("a prefix saved to disk is found again by a page loaded later, byte for byte", async () => {
  const { folder } = memoryFolder()
  await new PrefixDisk(folder, "id").save([1, 2, 3], [texture(7), texture(9, 4, 1)])
  const later = new PrefixDisk(folder, "id")
  assert.ok(await later.has([1, 2, 3]))
  const found = await later.match([1, 2, 3, 4], 3)
  assert.deepEqual(found?.keys, [1, 2, 3])
  assert.equal(found?.position, 3)
  assert.deepEqual(found?.textures.map((t) => [t.internal, t.width, t.height]), [[GL.RGBA32UI, 2, 2], [GL.RGBA32UI, 4, 1]])
  assert.ok(found?.textures[0].bytes.every((b) => b === 7))
  assert.ok(found?.textures[1].bytes.every((b) => b === 9))
  // Not the whole prompt, nor one no longer than what memory has.
  assert.equal(await later.match([1, 2, 3], 2), undefined)
  assert.equal(await later.match([1, 2, 3, 4], 3, 3), undefined)
  assert.equal(await later.match([1, 2, 9, 4], 3), undefined)
})

test("another identity's files are left for it, never matched; files cut short are deleted when found", async () => {
  const { files, folder } = memoryFolder()
  await new PrefixDisk(folder, "another bundle").save([1, 2], [texture(1)])
  await new PrefixDisk(folder, "id").save([5, 6], [texture(2)])
  // A write that never finished: this identity's file, a byte short.
  const cut = [...files.keys()].find((name) => name.startsWith("2-") && new TextDecoder().decode(files.get(name)!.bytes).includes('"identity":"id"'))!
  const whole = files.get(cut)!
  files.set(cut, { ...whole, bytes: whole.bytes.slice(0, whole.bytes.byteLength - 1) })
  await folder.write("notes.txt", [new Uint8Array([1])])
  const disk = new PrefixDisk(folder, "id")
  assert.equal(await disk.match([1, 2, 3], 2), undefined, "another identity's")
  assert.equal(await disk.has([1, 2]), false)
  assert.equal(await disk.match([5, 6, 7], 2), undefined, "cut short")
  // The cut one went; the other identity's is still there, and still its own (and notes.txt is left alone).
  assert.equal(files.size, 2)
  assert.ok(files.has("notes.txt"))
  assert.ok(await new PrefixDisk(folder, "another bundle").match([1, 2, 3], 2))
})

test("another identity's files count toward the budget, and go least recently used first like any", async () => {
  const { files, folder } = memoryFolder()
  const size = await (async () => {
    const probe = memoryFolder()
    await new PrefixDisk(probe.folder, "id").save([1], [texture(1)])
    return [...probe.files.values()][0].bytes.byteLength
  })()
  await new PrefixDisk(folder, "another bundle", 2 * size).save([9], [texture(9)])
  const disk = new PrefixDisk(folder, "id", 2 * size)
  await disk.save([1], [texture(1)])
  await disk.save([2], [texture(2)])
  assert.equal(files.size, 2)
  assert.equal(await new PrefixDisk(folder, "another bundle").has([9]), false, "the oldest went")
  // clear() deletes only its own.
  await new PrefixDisk(folder, "another bundle", 4 * size).save([8], [texture(8)])
  await new PrefixDisk(folder, "id", 4 * size).clear()
  assert.equal(files.size, 1)
  assert.ok(await new PrefixDisk(folder, "another bundle").has([8]))
})

test("files past the disk budget go least recently used first; 0 deletes them all", async () => {
  const { files, folder } = memoryFolder()
  const size = (await (async () => {
    const probe = memoryFolder()
    await new PrefixDisk(probe.folder, "id").save([1], [texture(1)])
    return [...probe.files.values()][0].bytes.byteLength
  })())
  const disk = new PrefixDisk(folder, "id", 2 * size)
  await disk.save([1], [texture(1)])
  await disk.save([2], [texture(2)])
  // [1] used: [2] is the oldest now.
  assert.ok(await disk.match([1, 0], 1))
  await disk.save([3], [texture(3)])
  assert.ok(await disk.has([1]))
  assert.equal(await disk.has([2]), false)
  assert.ok(await disk.has([3]))
  assert.equal(files.size, 2)
  // Too big for the budget at all: nothing written, nothing deleted.
  await disk.save([4], [texture(4, 8, 8)])
  assert.equal(files.size, 2)
  // An image's key isn't a token: not saved.
  await disk.save([2 ** 40], [texture(5)])
  assert.equal(files.size, 2)
  await disk.setBudget(0)
  assert.equal(files.size, 0)
  assert.equal(await disk.match([3, 0], 1), undefined)
  await assert.rejects(disk.setBudget(-1), RangeError)
})

test("a prefix goes from the GPU to disk and back to the GPU, and restores as it was", async () => {
  const f = engineFixture()
  const { engine } = f
  const { folder } = memoryFolder()
  const store = new PrefixStore<SavedPrefix>((saved) => engine.freePrefix(saved), 1 << 20)
  const keys = [11, 12, 13, 14, 15, 16, 17, 18]
  f.read(1, 6)
  engine.snapshot(undefined, "prefix")
  f.read(1, 8)
  const fresh = savePromptPrefix(engine, store, 8, keys, [], 6)
  assert.deepEqual(fresh?.keys, keys.slice(0, 6))
  await persistPrefix(new PrefixDisk(folder, "id"), (s) => store.holds(s), engine, fresh, assert.fail)
  // A page loaded again: a new engine state, nothing in memory.
  const g = engineFixture()
  g.read(2, 8)
  const found = await new PrefixDisk(folder, "id").match(keys, 7)
  assert.equal(found?.position, 6)
  const saved = g.engine.importPrefix(found!.position, found!.textures)!
  assert.ok(saved)
  await g.engine.restorePrefix(saved)
  for (let p = 0; p < 6; p++) assert.deepEqual(g.at(p), [1000 + p, 1000 + p])
  assert.ok(g.data.get(g.layers[0].conv!.state[0])!.every((v) => v === 1))
  const ring = g.data.get(g.layers[2].attention!.kv)!
  assert.equal(ring[0], 1004)
  assert.equal(ring[2], 1005)
  g.engine.freePrefix(saved)
  // Shapes another engine wouldn't make: refused, nothing allocated.
  const before = g.live.size
  assert.equal(g.engine.importPrefix(6, found!.textures.slice(1)), undefined)
  assert.equal(g.engine.importPrefix(3, found!.textures), undefined)
  assert.equal(g.live.size, before)
})

test("nothing goes to disk once the store has freed the prefix, or with no disk", async () => {
  const f = engineFixture()
  const { folder, files } = memoryFolder()
  const store = new PrefixStore<SavedPrefix>((saved) => f.engine.freePrefix(saved), 1 << 20)
  f.read(1, 6)
  f.engine.snapshot(undefined, "prefix")
  const fresh = savePromptPrefix(f.engine, store, 6, [1, 2, 3, 4, 5, 6], [], 6)
  store.clear()
  await persistPrefix(new PrefixDisk(folder, "id"), (s) => store.holds(s), f.engine, fresh, assert.fail)
  await persistPrefix(undefined, () => true, f.engine, fresh, assert.fail)
  assert.equal(files.size, 0)
})

// The review's findings.

test("savePromptPrefix makes room before copying: the store never holds more than its budget", () => {
  const { s, freed } = store(25)
  s.add([9], 10, "old")
  s.add([8], 10, "older")
  let peak = 0
  const engine = {
    snapshotPosition: () => 4,
    savedPrefixBytes: () => 10,
    savePrefix: () => { peak = s.bytes + 10; return "new" }
  }
  savePromptPrefix(engine, s, 6, [1, 2, 3, 4, 5, 6], [], 4)
  assert.ok(peak <= 25, `${peak} bytes at the copy`)
  assert.deepEqual(freed, ["old"])
  assert.ok(s.has([1, 2, 3, 4]))
})

test("a failed prefix copy is skipped while the GPU is there, and thrown once it's lost", () => {
  const logs: string[] = []
  const { s } = store(100)
  let lost = false
  const engine = { snapshotPosition: () => 4, savedPrefixBytes: () => 10, savePrefix: (): string => { throw new Error("Out of memory.") } }
  const failure = { lost: () => lost, log: (text: string) => logs.push(text) }
  assert.equal(savePromptPrefix(engine, s, 6, [1, 2, 3, 4, 5, 6], [], 4, failure), undefined)
  assert.equal(s.size, 0)
  assert.match(logs[0] ?? "", /Out of memory/)
  lost = true
  assert.throws(() => savePromptPrefix(engine, s, 6, [1, 2, 3, 4, 5, 6], [], 4, failure), /Out of memory/)
})

test("the disk's least recently used order survives a reload", async () => {
  const { files, folder } = memoryFolder()
  const size = await (async () => {
    const probe = memoryFolder()
    await new PrefixDisk(probe.folder, "id").save([1], [texture(1)])
    return [...probe.files.values()][0].bytes.byteLength
  })()
  const first = new PrefixDisk(folder, "id", 2 * size)
  await first.save([1], [texture(1)])
  await first.save([2], [texture(2)])
  // A page loaded again uses [1]...
  assert.ok(await new PrefixDisk(folder, "id", 2 * size).match([1, 0], 1))
  // ...and the next one saves [3]: [2] goes, the least recently used.
  const third = new PrefixDisk(folder, "id", 2 * size)
  await third.save([3], [texture(3)])
  assert.ok(await third.has([1]))
  assert.equal(await third.has([2]), false)
  assert.equal(files.size, 2)
})

test("a prefix used from memory counts as used on disk too", async () => {
  const { folder } = memoryFolder()
  const size = await (async () => {
    const probe = memoryFolder()
    await new PrefixDisk(probe.folder, "id").save([1], [texture(1)])
    return [...probe.files.values()][0].bytes.byteLength
  })()
  const disk = new PrefixDisk(folder, "id", 2 * size)
  await disk.save([1], [texture(1)])
  await disk.save([2], [texture(2)])
  await disk.touch([1])
  await disk.touch([7])
  const later = new PrefixDisk(folder, "id", 2 * size)
  await later.save([3], [texture(3)])
  assert.ok(await later.has([1]))
  assert.equal(await later.has([2]), false)
})

test("a prefix on disk past the memory budget isn't read at all", async () => {
  const { folder, reads } = memoryFolder()
  await new PrefixDisk(folder, "id").save([1, 2], [texture(1, 4, 4)])
  const disk = new PrefixDisk(folder, "id")
  // (The index reads each file's header once.)
  assert.ok(await disk.has([1, 2]))
  const indexed = reads.length
  assert.equal(await disk.match([1, 2, 3], 2, 0, 255), undefined)
  assert.equal(reads.length, indexed)
  assert.ok(await disk.match([1, 2, 3], 2, 0, 256))
  assert.ok(reads.length > indexed)
})

test("untilAborted stops waiting when the signal aborts", async () => {
  const controller = new AbortController()
  let release!: (value: string) => void
  const pending = untilAborted(new Promise<string>((resolve) => { release = resolve }), controller.signal)
  controller.abort()
  assert.equal(await pending, undefined)
  release("late")
  assert.equal(await untilAborted(Promise.resolve("now"), new AbortController().signal), "now")
  assert.equal(await untilAborted(Promise.resolve("no signal")), "no signal")
})

test("two identities saving the same tokens keep a file each; a stale index never deletes the other's", async () => {
  const { files, folder } = memoryFolder()
  const a = new PrefixDisk(folder, "bundle a")
  const b = new PrefixDisk(folder, "bundle b")
  await a.save([1, 2], [texture(1)])
  assert.ok(await a.has([1, 2]))
  await b.save([1, 2], [texture(2)])
  assert.equal(files.size, 2, "a file each")
  // a's index, read before b saved, still finds its own file, and b's stays.
  const found = await a.match([1, 2, 3], 2)
  assert.ok(found?.textures[0].bytes.every((x) => x === 1))
  assert.equal(files.size, 2)
  assert.ok((await b.match([1, 2, 3], 2))?.textures[0].bytes.every((x) => x === 2))
  // A file rewritten under a's name by another identity: a leaves it.
  const [name] = [...files.keys()].filter((n) => new TextDecoder().decode(files.get(n)!.bytes).includes('"identity":"bundle a"'))
  files.set(name, { bytes: files.get([...files.keys()].find((n) => n !== name)!)!.bytes.slice() })
  assert.equal(await a.match([1, 2, 3], 2), undefined)
  assert.equal(files.size, 2, "not deleted")
})
