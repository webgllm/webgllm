import assert from "node:assert/strict"
import { test } from "node:test"
import { StructuredGeneration, maskLogits } from "./structured-generation.js"
import { createConstraint, tokenAllowed } from "@tanstack-ai-browser/structured-output"
import type { Constraint } from "@tanstack-ai-browser/structured-output"

function fixture() {
  const accepted: number[] = []
  let disposed = false
  const mask = Uint32Array.of(1 << 3)
  const constraint: Constraint = {
    vocabSize: 64, eosTokenId: 63,
    get complete() { return accepted.includes(3) },
    computeMask: () => mask,
    accept: token => { accepted.push(token) },
    clone: () => constraint,
    dispose: () => { disposed = true }
  }
  return { constraint, mask, accepted, get disposed() { return disposed } }
}

test("thinking masks EOS but does not consume schema tokens until the closing marker", () => {
  const f = fixture()
  const s = new StructuredGeneration(f.constraint, 63, 62, true, 64)
  assert.equal(s.answering, false)
  assert.equal(s.mask()[1] >>> 31, 0)
  s.observe(12)
  assert.deepEqual(f.accepted, [])
  s.observe(62)
  assert.equal(s.answering, true)
  assert.equal(s.mask(), f.mask)
  s.observe(3)
  assert.deepEqual(f.accepted, [3])
  assert.equal(s.complete, true)
  s.dispose()
  assert.equal(f.disposed, true)
})

test("without template thinking the first token is schema constrained", () => {
  const f = fixture()
  const s = new StructuredGeneration(f.constraint, 63, 62, false, 64)
  assert.equal(s.answering, true)
  assert.equal(s.mask(), f.mask)
  assert.equal(s.complete, false)
})

test("CPU masks tokens before selection, including the sign bit of each word", () => {
  const logits = new Float32Array(65).fill(100)
  const mask = Uint32Array.of(1 << 31, 1, 0)
  maskLogits(logits, mask)
  assert.equal(logits[31], 100)
  assert.equal(logits[32], 100)
  for (let i = 0; i < logits.length; i++) if (i !== 31 && i !== 32) assert.equal(logits[i], -Infinity)
})


test("a seeded model-generated opening marker stays in reasoning and never enters the JSON matcher", () => {
  const f = fixture()
  const s = new StructuredGeneration(f.constraint, 63, 62, true, 64)
  s.observe(61) // Model's actual <think> token, absent from its template.
  s.observe(12)
  assert.deepEqual(f.accepted, [])
  s.observe(62)
  s.observe(3)
  assert.deepEqual(f.accepted, [3])
  assert.equal(s.complete, true)
})

test("answer draft masks follow hypothetical prefixes without advancing live grammar", async () => {
  const eos = 128
  const tokenizer = { tokens: [...Array.from({ length: 128 }, (_, id) => Uint8Array.of(id)), new Uint8Array()], eos_token_id: eos, special_token_ids: [eos] }
  const constraint = await createConstraint({ tokenizer, schema: { type: "string", enum: ["ab", "ac"] } })
  const s = new StructuredGeneration(constraint, eos, null, false, 129)
  const code = (text: string) => text.charCodeAt(0)
  s.observe(code('"'))
  s.observe(code("a"))
  const preview = s.previewAnswerDrafts([code("b"), code('"'), eos, code("x")])
  assert.deepEqual(preview.drafts, [code("b"), code('"')])
  assert.equal(preview.masks.length, 3)
  assert.equal(tokenAllowed(preview.masks[0], code("b")), true)
  assert.equal(tokenAllowed(preview.masks[0], code("c")), true)
  assert.equal(tokenAllowed(preview.masks[1], code('"')), true)
  assert.equal(tokenAllowed(preview.masks[1], code("a")), false)
  assert.equal(tokenAllowed(preview.masks[2], eos), true)
  // Rejecting b at the first proposal leaves the live alternative c available.
  assert.equal(tokenAllowed(s.mask(), code("c")), true)
  s.observe(code("c"))
  s.observe(code('"'))
  assert.equal(s.complete, true)
  s.dispose()
})

test("answer speculation discards an invalid suffix before it can enter KV or grammar state", async () => {
  const eos = 128
  const tokenizer = { tokens: [...Array.from({ length: 128 }, (_, id) => Uint8Array.of(id)), new Uint8Array()], eos_token_id: eos, special_token_ids: [eos] }
  const constraint = await createConstraint({ tokenizer, schema: { type: "string", enum: ["ab"] } })
  const s = new StructuredGeneration(constraint, eos, null, false, 129)
  const code = (text: string) => text.charCodeAt(0)
  s.observe(code('"'))
  const preview = s.previewAnswerDrafts([code("a"), code("x"), code('"')])
  assert.deepEqual(preview.drafts, [code("a")])
  assert.equal(preview.masks.length, 2)
  assert.equal(tokenAllowed(preview.masks[1], code("b")), true)
  assert.equal(tokenAllowed(s.mask(), code("a")), true)
  assert.equal(s.complete, false)
  s.dispose()
})
