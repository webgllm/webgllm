import assert from "node:assert/strict"
import { test } from "node:test"

import { encodePng } from "./png.js"

/** The PNG's chunks, each CRC checked, and its pixels inflated. */
async function decode(png: Uint8Array) {
  assert.deepEqual([...png.subarray(0, 8)], [137, 80, 78, 71, 13, 10, 26, 10])
  const view = new DataView(png.buffer, png.byteOffset, png.byteLength)
  const chunks: Array<{ type: string; data: Uint8Array }> = []
  for (let at = 8; at < png.length; ) {
    const length = view.getUint32(at)
    const type = String.fromCharCode(...png.subarray(at + 4, at + 8))
    chunks.push({ type, data: png.subarray(at + 8, at + 8 + length) })
    at += 12 + length
  }
  const header = new DataView(chunks[0]!.data.buffer, chunks[0]!.data.byteOffset)
  const idat = chunks.find((c) => c.type === "IDAT")!.data
  const raw = new Uint8Array(await new Response(new Blob([idat as BlobPart]).stream().pipeThrough(new DecompressionStream("deflate"))).arrayBuffer())
  return { types: chunks.map((c) => c.type), width: header.getUint32(0), height: header.getUint32(4), depth: chunks[0]!.data[8], color: chunks[0]!.data[9], raw }
}

test("a PNG of RGBA pixels: its header, and its rows unfiltered", async () => {
  const rgba = new Uint8Array([255, 0, 0, 255, 0, 255, 0, 128, 0, 0, 255, 0, 9, 9, 9, 9])
  const png = await decode(await encodePng(rgba, 2, 2))
  assert.deepEqual(png.types, ["IHDR", "IDAT", "IEND"])
  assert.deepEqual([png.width, png.height, png.depth, png.color], [2, 2, 8, 6])
  assert.deepEqual([...png.raw], [0, ...rgba.subarray(0, 8), 0, ...rgba.subarray(8)])
})

test("a scaled PNG repeats each pixel", async () => {
  const rgba = new Uint8Array([1, 2, 3, 4, 5, 6, 7, 8])
  const png = await decode(await encodePng(rgba, 2, 1, 2))
  assert.deepEqual([png.width, png.height], [4, 2])
  const row = [0, 1, 2, 3, 4, 1, 2, 3, 4, 5, 6, 7, 8, 5, 6, 7, 8]
  assert.deepEqual([...png.raw], [...row, ...row])
})

test("a PNG's CRCs are right (zlib's crc32 of each chunk's type and data)", async () => {
  const png = await encodePng(new Uint8Array(4), 1, 1)
  // IEND's CRC is fixed: crc32("IEND") = 0xAE426082.
  assert.deepEqual([...png.subarray(png.length - 4)], [0xae, 0x42, 0x60, 0x82])
})

test("wrong sizes are refused", async () => {
  await assert.rejects(encodePng(new Uint8Array(3), 1, 1), /bytes/)
  await assert.rejects(encodePng(new Uint8Array(4), 1, 1, 0), /scale/)
})
