/**
 * logits.html's worker: webgllm's handler with a logit processor for the
 * page's model (WebLLM's LogitProcessor), which keeps only the likeliest
 * token, so sampling is greedy whatever the model card says, and reports
 * what it's given to the page.
 */
import { WebGLLMWorkerHandler } from "../src/index.js"
import type { LogitProcessor } from "../src/index.js"

const model = new URLSearchParams(self.location.search).get("model") ?? ""
const processor: LogitProcessor = {
  processLogits(logits) {
    let best = 0
    for (let i = 1; i < logits.length; i++) if (logits[i] > logits[best]) best = i
    postMessage({ type: "bench-logits", length: logits.length, argmax: best })
    const only = new Float32Array(logits.length).fill(-Infinity)
    only[best] = logits[best]
    return only
  },
  processSampledToken: (token) => postMessage({ type: "bench-token", token }),
  resetState: () => postMessage({ type: "bench-reset" })
}

const handler = new WebGLLMWorkerHandler()
handler.setLogitProcessorRegistry(new Map([[model, processor]]))
self.onmessage = (event) => handler.onmessage(event)
