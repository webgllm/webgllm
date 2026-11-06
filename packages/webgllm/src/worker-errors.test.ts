import assert from "node:assert/strict"
import { test } from "node:test"
import { CodedError, downloadError, errorCode } from "./worker-errors.js"
import { GpuError } from "./gpu/gl.js"
import { StructuredOutputError } from "./constraint/index.js"

test("codes known where thrown, and classified failures", () => {
  assert.equal(errorCode(new CodedError("long", "context-length-exceeded")), "context-length-exceeded")
  assert.equal(errorCode(new StructuredOutputError("unsupported-schema", "bad")), "invalid-schema")
  assert.equal(errorCode(new StructuredOutputError("constraint-dead-end", "dead")), undefined)
  assert.equal(errorCode(new GpuError("The GPU context was lost.")), "gpu-context-lost")
  assert.equal(errorCode(new GpuError("The GPU couldn't grow the cache to 9000 tokens (Out of memory.).")), "out-of-memory")
  assert.equal(errorCode(new GpuError("This browser has no WebGL 2.")), "webgl2-unavailable")
  assert.equal(errorCode(new GpuError("This browser has no OffscreenCanvas in workers.")), "webgl2-unavailable")
  assert.equal(errorCode(new GpuError("Shader didn't compile")), undefined)
  assert.equal(errorCode(new Error("other")), undefined)
  assert.equal(errorCode("text"), undefined)
})

test("download failures: storage-full for the quota, download-failed otherwise, aborts as they are", () => {
  const quota = { name: "QuotaExceededError", message: "quota" }
  assert.equal(errorCode(quota), "storage-full")
  assert.equal(errorCode(downloadError(quota)), "storage-full")
  assert.equal(errorCode(downloadError(new TypeError("Failed to fetch"))), "download-failed")
  assert.match((downloadError(new Error("HTTP 404 for x")) as Error).message, /HTTP 404/)
  const abort = new AbortController()
  abort.abort()
  const reason = abort.signal.reason
  assert.equal(downloadError(reason, abort.signal), reason)
})
