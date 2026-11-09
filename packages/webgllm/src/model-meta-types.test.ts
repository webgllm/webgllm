import assert from "node:assert/strict"
import { test } from "node:test"
import { WEBGL_MODEL_META, WEBGL_TEXT_MODELS } from "./model-meta.js"
import type { WebGlModelReasoning, WebGlTextModel } from "./model-meta.js"

// Compile-time checks: typecheck fails if the catalog's types widen again.
type Equal<A, B> = (<T>() => T extends A ? 1 : 2) extends <T>() => T extends B ? 1 : 2 ? true : false
const exact = <T extends true>(_: T) => undefined

exact<Equal<WebGlModelReasoning<"LiquidAI/LFM3-1.7B-GGUF">, { readonly levels: { readonly off: "none"; readonly high: "high" }; readonly budget: true }>>(true)
exact<Equal<WebGlModelReasoning<"unsloth/Qwen3.5-2B-GGUF">, { readonly levels: { readonly off: "none"; readonly high: "high" }; readonly budget: true }>>(true)
exact<Equal<WebGlModelReasoning<"LiquidAI/LFM2.5-1.2B-Thinking-ONNX">, { readonly levels: { readonly high: "high" }; readonly budget: true }>>(true)
exact<Equal<WebGlModelReasoning<"LiquidAI/LFM2.5-1.2B-Thinking-GGUF">, { readonly levels: { readonly high: "high" }; readonly budget: true }>>(true)
exact<Equal<WebGlModelReasoning<"LiquidAI/LFM2.5-350M-GGUF">, undefined>>(true)
exact<Equal<WebGlModelReasoning<"LiquidAI/LFM2.5-1.2B-Instruct-ONNX">, undefined>>(true)
exact<Equal<WebGlModelReasoning<"LiquidAI/LFM2-1.2B-Tool-GGUF">, undefined>>(true)
// Model names stay a union of literals, not string.
exact<Equal<string extends WebGlTextModel ? true : false, false>>(true)

test("the catalog's runtime reasoning matches its types", () => {
  for (const model of WEBGL_TEXT_MODELS) {
    const reasoning = WEBGL_MODEL_META[model].reasoning
    if (model.includes("Thinking")) assert.deepEqual(reasoning, { levels: { high: "high" }, budget: true })
    else if (/LFM2[.-]/.test(model)) assert.equal(reasoning, undefined)
    else assert.deepEqual(reasoning, { levels: { off: "none", high: "high" }, budget: true })
  }
})
