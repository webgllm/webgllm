import assert from "node:assert/strict"
import { test } from "node:test"
import { createStopFilter } from "./stop-sequences.js"

/** Streams `chunks` through a filter: what's sent, and whether a stop ended it. */
function run(stops: string[], chunks: string[]) {
  const filter = createStopFilter(stops)!
  let sent = ""
  let stopped = false
  const pieces: string[] = []
  for (const chunk of chunks) {
    const out = filter.push(chunk)
    if (out.text) pieces.push(out.text)
    sent += out.text
    if (out.stopped) {
      stopped = true
      break
    }
  }
  if (!stopped) sent += filter.flush()
  return { sent, stopped, pieces }
}

test("no stop sequences: no filter", () => {
  assert.equal(createStopFilter(undefined), null)
  assert.equal(createStopFilter([]), null)
  assert.equal(createStopFilter([""]), null)
})

test("ends at the first stop sequence, which is never sent", () => {
  assert.deepEqual(run(["END"], ["Hello ", "world END more"]), { sent: "Hello world ", stopped: true, pieces: ["Hello ", "world "] })
})

test("the earliest of several stop sequences wins", () => {
  assert.equal(run(["b", "cd"], ["xxcdb"]).sent, "xx")
  assert.equal(run(["cd", "b"], ["xxbcd"]).sent, "xx")
})

test("a stop sequence split across tokens is held back, then dropped", () => {
  const { sent, stopped, pieces } = run(["</answer>"], ["42", "</", "ans", "wer>", "tail"])
  assert.equal(sent, "42")
  assert.equal(stopped, true)
  assert.deepEqual(pieces, ["42"])
})

test("held-back text that turns out not to be a stop is sent", () => {
  const { sent, stopped, pieces } = run(["STOP"], ["a ST", "OOP b"])
  assert.equal(sent, "a STOOP b")
  assert.equal(stopped, false)
  assert.deepEqual(pieces, ["a ", "STOOP b"])
})

test("a held-back start of a stop at the end of the reply is sent", () => {
  assert.deepEqual(run(["STOP"], ["done ST"]), { sent: "done ST", stopped: false, pieces: ["done "] })
})

test("thinking passes through unmatched, markers and all", () => {
  const { sent, stopped } = run(["\n"], ["<think>", "line one\nline two", "\n</thi", "nk>\n\nAnswer\nmore"])
  assert.equal(sent, "<think>line one\nline two\n</think>\n\nAnswer")
  assert.equal(stopped, true)
})

test("thinking text streams without waiting for the close", () => {
  const filter = createStopFilter(["x"])!
  assert.equal(filter.push("<think>xx").text, "<think>xx")
  assert.equal(filter.push(" </th").text, " ")
  assert.equal(filter.push("ink>x").text, "</think>")
})

test("whitespace and empty thinking before the answer aren't answer", () => {
  assert.equal(run(["\n"], ["\n\n<think>\n</think>\n", "\nHi\nthere"]).sent, "\n\n<think>\n</think>\n\nHi")
  assert.equal(run(["</think>"], ["</think>", "ok</think>no"]).sent, "</think>ok")
})

test("a stop inside the answer's first token", () => {
  assert.deepEqual(run(["Hi"], ["Hi there"]), { sent: "", stopped: true, pieces: [] })
})

test("marker-like text after the answer starts is matched as answer", () => {
  assert.equal(run(["<think>"], ["Say <thi", "nk> please"]).sent, "Say ")
})

test("undecided text at the end is matched as answer", () => {
  assert.equal(run(["<t"], ["<th"]).sent, "")
})

test("nothing more is sent after a stop", () => {
  const filter = createStopFilter(["."])!
  assert.deepEqual(filter.push("a.b"), { text: "a", stopped: true })
  assert.deepEqual(filter.push("c"), { text: "", stopped: true })
  assert.equal(filter.flush(), "")
})
