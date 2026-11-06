/**
 * The transformers-js package's worker without WebGPU, for a CPU benchmark
 * in a browser that has it: `navigator.gpu` is hidden before the worker
 * loads, so it picks onnxruntime's WebAssembly build and the q4 weights, as
 * in a browser without WebGPU.
 */
Object.defineProperty(navigator, "gpu", { value: undefined, configurable: true })

// Messages that arrive while the worker loads are handed over once it has.
const queued: Array<MessageEvent> = []
self.onmessage = (event) => queued.push(event)
const url = "/bundle-tjs/src/worker.js"
await import(/* @vite-ignore */ url)
for (const event of queued) self.onmessage?.(event)
