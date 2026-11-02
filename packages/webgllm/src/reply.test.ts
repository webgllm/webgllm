import assert from "node:assert/strict"
import { test } from "node:test"
import { createThinkingSplitter, splitThinking } from "./reply.js"

function streamed(chunks: string[]) {
  const splitter = createThinkingSplitter()
  return [...chunks.flatMap(chunk => splitter.push(chunk)), ...splitter.flush()]
}

test("leading repeated empty thought blocks are not answer or reasoning text", () => {
  const pieces = streamed(['<think>plan</think>', '<think>\n</think>', ' \n<think> </think>', '\n21'])
  assert.equal(pieces.filter(piece => piece.kind === "thinking").map(piece => piece.text).join(""), "plan")
  assert.equal(pieces.filter(piece => piece.kind === "answer").map(piece => piece.text).join(""), "21")
})

test("empty block markers can arrive one character at a time", () => {
  const raw = '<think>plan</think>\n<think> \n</think><think></think>21'
  const pieces = streamed([...raw])
  assert.equal(pieces.filter(piece => piece.kind === "thinking").map(piece => piece.text).join(""), "plan")
  assert.equal(pieces.filter(piece => piece.kind === "answer").map(piece => piece.text).join(""), "21")
})

test("JSON, code spans, body markers and nonempty later blocks remain literal answer data", () => {
  for (const answer of ['{"value":"<think></think>"}', '`<think></think>`', 'Use <think></think> literally', '<think>not empty</think>answer']) {
    const pieces = streamed(['<think>plan</think>', ...answer])
    assert.equal(pieces.filter(piece => piece.kind === "answer").map(piece => piece.text).join(""), answer)
  }
})

test("flush preserves incomplete marker-shaped answer prefixes and removes complete empty blocks", () => {
  for (const suffix of ['<th', '<think> ', '<think> </thi']) {
    const pieces = streamed(['<think>plan</think>', suffix])
    assert.equal(pieces.filter(piece => piece.kind === "answer").map(piece => piece.text).join(""), suffix)
  }
  assert.deepEqual(splitThinking('<think>plan</think><think> </think>'), { thinking: 'plan', answer: '' })
})
