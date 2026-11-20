import assert from "node:assert/strict"
import test from "node:test"
import { Engine } from "./engine.js"
import { PacedGpu } from "./gpu/paced.js"
import { bufferMatvec } from "./gpu/native.js"
import * as R from "./gpu/raon-kernels.js"
import { WGSL_CODE } from "./gpu/wgsl.js"
import { WebGpu } from "./gpu/webgpu.js"

test("the engine fingerprint covers the WebGPU backend, its native kernels, the GLSL translator and pacing", () => {
  const text = Engine.fingerprintText()
  const covered = (what: string, source: string) => assert.ok(text.includes(source), `${what} isn't in the fingerprint`)
  covered("WebGpu", WebGpu.toString())
  covered("bufferMatvec", bufferMatvec.toString())
  covered("PacedGpu", PacedGpu.toString())
  // The translator's classes and helpers, not only its exported entry points.
  for (const part of WGSL_CODE.filter((v) => typeof v === "function")) covered((part as { name: string }).name, String(part))
  assert.match(Engine.fingerprint, /^[0-9a-f]+$/)
})

test("the engine fingerprint covers Raon's kernels: a change to one of them alone is a change to it", () => {
  const text = Engine.fingerprintText()
  for (const [name, value] of Object.entries(R)) {
    if (typeof value === "string") assert.ok(text.includes(value), `${name} isn't in the fingerprint`)
    else if (typeof value === "function") assert.ok(text.includes((value as { source?: string }).source ?? String(value)), `${name}'s source isn't in the fingerprint`)
  }
  // (What they're built from, too: the shared helpers aren't exported one by one.)
  for (const part of R.RAON_CODE) assert.ok(text.includes(String(part)), "a helper of Raon's kernels isn't in the fingerprint")
  assert.ok(text.includes(R.STREAM_MIX) && !text.includes(R.STREAM_MIX.replace("o = acc;", "o = vec4(0.0);")))
})
