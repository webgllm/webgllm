import { tokenAllowed } from "./constraint/index.js"
import type { Constraint } from "./constraint/index.js"

/** Keep reasoning separate from the schema-constrained answer. */
export class StructuredGeneration {
  private reasoning: boolean
  private readonly thinkingMask: Uint32Array

  constructor(
    private readonly constraint: Constraint,
    eos: number,
    private readonly thinkEnd: number | null,
    startsInThinking: boolean,
    vocabSize: number
  ) {
    this.reasoning = startsInThinking
    this.thinkingMask = new Uint32Array(Math.ceil(vocabSize / 32)).fill(0xffffffff)
    this.thinkingMask[eos >>> 5] &= ~(1 << (eos & 31))
  }

  get answering() { return !this.reasoning }
  get complete() { return !this.reasoning && this.constraint.complete }

  mask(): Uint32Array {
    return this.reasoning ? this.thinkingMask : this.constraint.computeMask()
  }

  observe(token: number) {
    if (this.reasoning) {
      if (token === this.thinkEnd) this.reasoning = false
    } else this.constraint.accept(token)
  }

  /** Masks for hypothetical answer prefixes; rejected drafts never touch live state. */
  previewAnswerDrafts(proposed: readonly number[]): { drafts: number[]; masks: Uint32Array[] } {
    if (this.reasoning) throw new Error("Answer drafts require the structured-answer phase.")
    const preview = this.constraint.clone()
    const drafts: number[] = []
    const masks: Uint32Array[] = []
    try {
      masks.push(preview.computeMask().slice())
      for (const token of proposed) {
        // EOS is drawn after the last forwarded row rather than drafted into KV.
        if (token === preview.eosTokenId || !tokenAllowed(masks[masks.length - 1], token)) break
        preview.accept(token)
        drafts.push(token)
        masks.push(preview.computeMask().slice())
      }
      return { drafts, masks }
    } finally { preview.dispose() }
  }

  dispose() { this.constraint.dispose() }
}

/** Apply the same mask to full CPU logits as GPU candidate selection uses. */
export function maskLogits(logits: Float32Array, mask: Uint32Array) {
  for (let id = 0; id < logits.length; id++) {
    if (((mask[id >>> 5] ?? 0) & (1 << (id & 31))) === 0) logits[id] = -Infinity
  }
}
