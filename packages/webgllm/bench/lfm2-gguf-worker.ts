import { Engine } from "../src/engine.js"
import { httpSource } from "../src/byte-source.js"
import { loadGguf } from "../src/load-gguf.js"

type Reference = { key: string; file: string; sha256: string; reference: string; prompts: { ids: number[]; steps: { id: number; referenceTop?: number; token: string; logProbs: number[] }[] }[] }
const send = (type: string, value: unknown) => postMessage({ type, value })
function normalized(values: Float32Array): number[] {
  const max = values.reduce((a,b)=>Math.max(a,b),-Infinity)
  const logSum = max + Math.log(values.reduce((sum,value)=>sum+Math.exp(value-max),0))
  return Array.from(values,value=>value-logSum)
}
function compare(actual: Float32Array, expected: number[]) {
  if(actual.length!==expected.length)throw Error("Reference vocabulary length differs")
  let top = 0, max = -Infinity, sum = 0, kl = 0, maxLogProbDifference = 0
  for (let i=0;i<actual.length;i++) { if(!Number.isFinite(actual[i]))throw Error("Non-finite GPU logits");if(actual[i]>max){max=actual[i];top=i} }
  for(const value of actual)sum+=Math.exp(value-max)
  const normalizer=max+Math.log(sum)
  for(let i=0;i<actual.length;i++){const logp=actual[i]-normalizer;kl+=Math.exp(expected[i])*(expected[i]-logp);maxLogProbDifference=Math.max(maxLogProbDifference,Math.abs(expected[i]-logp))}
  return {top,kl,maxLogProbDifference}
}
const distance=(a:Float32Array,b:Float32Array)=>a.reduce((max,value,i)=>Math.max(max,Math.abs(value-b[i])),0)
onmessage = async ({data}:MessageEvent<{key:string;reference?:string}>) => {
  let engine:Engine|undefined
  const report:Record<string,unknown>={key:data.key,comparisonReference:data.reference??"q4",results:[]}
  try{
    const reference=await(await fetch(`/models/${data.key}/${data.reference==="f16"?"refs-f16.json":"refs.json"}`)).json() as Reference
    Object.assign(report,{file:reference.file,sha256:reference.sha256,reference:reference.reference})
    send("status",`Loading ${reference.file}`)
    engine=new Engine(new OffscreenCanvas(1,1),{batchSize:512})
    await engine.load(loadGguf(await httpSource(`/models/${reference.file}`)))
    Object.assign(report,{config:engine.config,renderer:engine.gpu.limits.renderer,gpuBytes:engine.gpu.bytes})
    const results=report.results as Record<string,unknown>[]
    for(const [index,prompt] of reference.prompts.entries()){
      engine.reset()
      const row=await engine.forwardHidden(prompt.ids)
      let logits=await engine.logits(row)
      const cold=logits.slice()
      engine.snapshot(row,"prefix")
      let next:Float32Array|undefined
      for(const [step,expected] of prompt.steps.entries()){
        const metric=compare(logits,expected.logProbs)
        results.push({prompt:index,step,expectedTop:expected.referenceTop??expected.id,top1:metric.top===(expected.referenceTop??expected.id),...metric})
        send("status",`${data.key}: prompt ${index+1}, step ${step+1}, KL ${metric.kl.toExponential(2)}`)
        if(step<prompt.steps.length-1){const row=await engine.forwardHidden([expected.id]);logits=await engine.logits(row);if(step===0)next=logits.slice()}
      }
      engine.resume(prompt.ids.length)
      const restored=await engine.logits(0)
      const hiddenMaxAbs=distance(cold,restored)
      // Resume recomputes the head norm instead of reusing a folded half input.
      const cacheHiddenKl=compare(restored,normalized(cold)).kl
      const nextRow=await engine.forwardHidden([prompt.steps[0].id])
      const continued=await engine.logits(nextRow)
      results.push({prompt:index,cacheHiddenMaxAbs:hiddenMaxAbs,cacheHiddenKl,cacheContinueMaxAbs:next?distance(next,continued):null,snapshotsWithinLimit:Reflect.get(engine,"snapshots").size<=engine.snapshotLimit})
    }
    report.pass=results.every(r=>r.kl===undefined? (r.cacheHiddenKl as number)<1e-5&&r.cacheContinueMaxAbs===0&&r.snapshotsWithinLimit : (r.kl as number)<0.01&&r.top1===true)
  }catch(error){report.error=error instanceof Error?error.stack:String(error);report.pass=false}
  finally{engine?.gpu.dispose();send("done",report)}
}
