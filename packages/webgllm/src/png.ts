/**
 * RGBA pixels as a PNG file: 8 bits a channel, no interlacing, each row
 * unfiltered, compressed with the platform's zlib (CompressionStream).
 * `scale` repeats each pixel scale × scale times (crisp pixel art).
 */
const CRC_TABLE = (() => {
  const table = new Uint32Array(256)
  for (let n = 0; n < 256; n++) {
    let c = n
    for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1
    table[n] = c >>> 0
  }
  return table
})()

function crc32(bytes: Uint8Array): number {
  let c = 0xffffffff
  for (const b of bytes) c = CRC_TABLE[(c ^ b) & 0xff]! ^ (c >>> 8)
  return (c ^ 0xffffffff) >>> 0
}

function chunk(type: string, data: Uint8Array): Uint8Array {
  const out = new Uint8Array(12 + data.length)
  const view = new DataView(out.buffer)
  view.setUint32(0, data.length)
  for (let i = 0; i < 4; i++) out[4 + i] = type.charCodeAt(i)
  out.set(data, 8)
  view.setUint32(8 + data.length, crc32(out.subarray(4, 8 + data.length)))
  return out
}

async function zlib(bytes: Uint8Array): Promise<Uint8Array> {
  const stream = new Blob([bytes as BlobPart]).stream().pipeThrough(new CompressionStream("deflate"))
  return new Uint8Array(await new Response(stream).arrayBuffer())
}

export async function encodePng(rgba: Uint8Array, width: number, height: number, scale = 1): Promise<Uint8Array> {
  if (rgba.length !== width * height * 4) throw new Error(`A ${width} × ${height} image has ${rgba.length} bytes, not ${width * height * 4}.`)
  if (!Number.isInteger(scale) || scale < 1) throw new Error(`The scale must be a whole number of at least 1, not ${scale}.`)
  const w = width * scale
  const h = height * scale
  const raw = new Uint8Array(h * (1 + w * 4))
  for (let y = 0; y < h; y++) {
    const row = y * (1 + w * 4)
    // (Filter type 0: none.)
    for (let x = 0; x < w; x++) raw.set(rgba.subarray(((Math.floor(y / scale) * width + Math.floor(x / scale)) * 4), (Math.floor(y / scale) * width + Math.floor(x / scale)) * 4 + 4), row + 1 + x * 4)
  }
  const header = new Uint8Array(13)
  const view = new DataView(header.buffer)
  view.setUint32(0, w)
  view.setUint32(4, h)
  header.set([8, 6, 0, 0, 0], 8)
  const parts = [new Uint8Array([137, 80, 78, 71, 13, 10, 26, 10]), chunk("IHDR", header), chunk("IDAT", await zlib(raw)), chunk("IEND", new Uint8Array(0))]
  const out = new Uint8Array(parts.reduce((s, p) => s + p.length, 0))
  let at = 0
  for (const p of parts) {
    out.set(p, at)
    at += p.length
  }
  return out
}
