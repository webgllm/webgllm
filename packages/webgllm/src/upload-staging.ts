import type { QuantMatrix, WeightPiece } from "./weights.js"

type TransferableBuffer = ArrayBuffer & {
  transfer?: (length: number) => ArrayBuffer
}

export const canReleaseStaging =
  typeof (ArrayBuffer.prototype as TransferableBuffer).transfer === "function"

/** Retire a buffer whose contents are exclusively owned and fully consumed. */
export function releaseOwnedBuffer(buffer: ArrayBufferLike) {
  const owned = buffer as TransferableBuffer
  if (owned.byteLength) owned.transfer?.(0)
}

/** Only for the worker's exclusively owned GGUF matrices, after their upload.
 * Public loaders keep their normal non-destructive ownership semantics.
 */
function releaseMatrices(piece: WeightPiece) {
  if (!canReleaseStaging || piece.kind === "config") return
  const buffers = new Set<TransferableBuffer>()
  const visit = (value: unknown) => {
    if (!value || typeof value !== "object" || ArrayBuffer.isView(value)) return
    if ("quants" in value && "scales" in value) {
      const matrix = value as QuantMatrix
      // Raw-format/aliased storage belongs to the input source, not this stage.
      if (matrix.raw) return
      for (const view of [matrix.quants, matrix.scales, matrix.zeros]) {
        if (
          !view ||
          view.byteOffset ||
          view.byteLength !== view.buffer.byteLength
        )
          continue
        buffers.add(view.buffer as TransferableBuffer)
      }
      return
    }
    for (const item of Object.values(value)) visit(item)
  }
  visit(piece)
  for (const buffer of buffers) releaseOwnedBuffer(buffer)
}

/** Resuming after yield means Engine.load has copied the piece into WebGL. */
export async function* uploadedGgufWeights(pieces: AsyncIterable<WeightPiece>) {
  for await (const piece of pieces) {
    try {
      yield piece
    } finally {
      releaseMatrices(piece)
    }
  }
}
