/** Bundle untouched engine sources, adding only allocation counters for memory runs. */
import { build } from "esbuild"
import { mkdir } from "node:fs/promises"
import { resolve } from "node:path"
const root = process.argv[2]
const trace = process.argv[4] === "trace"
const outputName = process.argv[3] ?? "baseline-worker.js"
if (!root || (process.argv[4] !== undefined && !trace))
  throw new Error(
    "Usage: node bench/build-workers.mjs /path/to/baseline-checkout [output.js] [trace]"
  )
await mkdir(new URL("./dist/", import.meta.url), { recursive: true })
const gpu = resolve(root, "packages/webgl/src/gpu/gl.ts"),
  worker = resolve(root, "packages/webgl/src/worker.ts")
await build({
  stdin: {
    contents: `import {Gpu} from ${JSON.stringify(gpu)};
let peak=0,last;const capture=gpu=>{last=gpu;peak=Math.max(peak,gpu.bytes)};const texture=Gpu.prototype.texture;
Gpu.prototype.texture=function(...args){const result=Reflect.apply(texture,this,args);capture(this);return result};
const send=postMessage.bind(self);postMessage=message=>{send(message);if(last&&(message.type==="ready"||message.type==="done"))send({type:"bench-memory",when:message.type,gpuBytes:last.bytes,peakGpuBytes:peak,pacing:Object.fromEntries(Object.entries(Reflect.get(last,"budgets")??{}).map(([name,b])=>[name,{reads:b.reads,initial:Reflect.get(b,"initial"),window:b.presentationWindow,maxWindow:b.presentationMax,relaxation:Reflect.get(b,"relaxedReadsAllowed"),frameMs:Reflect.get(b,"frameMs"),fastMatrixAgeMs:Number.isFinite(Reflect.get(b,"fastMatrixAt"))?Math.round(performance.now()-Reflect.get(b,"fastMatrixAt")):null}]))})};
Object.assign(globalThis,{__benchMemory:()=>({gpuBytes:last?.bytes??0,peakGpuBytes:peak})});
${trace ? `let tag="";for(const name of ["run","program","texture","write","readBack","pace"]){const original=Gpu.prototype[name];Gpu.prototype[name]=function(...args){const start=performance.now();if(name==="run")tag=args[0].name??"other";const pass=tag;const report=()=>{const ms=performance.now()-start;if(ms>10)postMessage({type:"bench-gpu",name:name+":"+pass,start:performance.timeOrigin+start,ms,shape:name==="run"?{rect:args[0].rect,uniforms:args[0].uniforms}:name==="texture"?args.slice(0,3):undefined})};const result=Reflect.apply(original,this,args);if(result?.then)return result.finally(report);report();return result}}` : ""}
${trace ? `for(const name of ["configurePacing","setPhase","resetPacing","observeFrames","pace","primePresentation"]){const original=Gpu.prototype[name];if(!original)continue;const snap=g=>Object.fromEntries(Object.entries(Reflect.get(g,"budgets")??{}).map(([k,b])=>[k,{reads:b.reads,initial:Reflect.get(b,"initial"),relaxation:Reflect.get(b,"relaxedReadsAllowed"),lastFrameAge:performance.now()-Reflect.get(b,"lastFrame"),healthy:Reflect.get(b,"healthyFrame"),window:b.presentationWindow}]));Gpu.prototype[name]=function(...args){const before=snap(this),start=performance.now();const report=()=>{const after=snap(this);if(Object.keys(after).some(k=>before[k]?.reads!==after[k].reads||before[k]?.relaxation!==after[k].relaxation))send({type:"bench-gpu",name:"pacing:"+name,start:performance.timeOrigin+start,ms:performance.now()-start,shape:{args:args.slice(0,5),phase:this.activePhase,before,after}})};const result=Reflect.apply(original,this,args);if(result?.then)return result.finally(report);report();return result}}` : ""}
await import(${JSON.stringify(worker)});
const originalHandler=onmessage;onmessage=async event=>{if(event.data.type==="bench-dispose"){if(last?.disposePaced)await last.disposePaced();else last?.dispose();send({type:"bench-disposed"})}else {if(last&&event.data.type==="tune"&&Number.isFinite(event.data.values?.submissionReads)){const configure=last.configurePacing,reads=event.data.values.submissionReads;last.configurePacing=function(_reads,ms){return configure.call(this,reads,ms)}}originalHandler?.call(self,event)}};`,
    resolveDir: root,
    sourcefile: "benchmark-memory.ts",
    loader: "ts"
  },
  outfile: new URL(`./dist/${outputName}`, import.meta.url).pathname,
  bundle: true,
  format: "esm",
  platform: "browser",
  target: "es2022"
})
