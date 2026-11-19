import assert from "node:assert/strict"
import test from "node:test"
import { Engine } from "./engine.js"
import { PacedGpu } from "./gpu/paced.js"
import { bufferMatvec } from "./gpu/native.js"
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
