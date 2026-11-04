/** Diagnostic canvas motion in Android recordings; recording changes GPU load.
 * OUT.json FRAMES.jsonl BEFORE-capture.json AFTER-capture.json
 * Metadata records video path, capture start, device/host clock offset and phase rows.
 */
import fs from 'node:fs'
import { execFileSync, spawn } from 'node:child_process'
const [output, frameOutput, ...inputs] = process.argv.slice(2)
if (!output || !frameOutput || inputs.length !== 2) throw Error('Expected outputs and two capture metadata files')
const captures = inputs.map(file => JSON.parse(fs.readFileSync(file)))
const run = args => execFileSync(args[0], args.slice(1), { maxBuffer: 16 * 1024 * 1024 })
const green = (r,g,b) => r < 160 && g > 175 && b > 145 && b < 230
const first = captures[0]
const stream = capture => JSON.parse(run(['ffprobe','-v','error','-select_streams','v:0','-show_entries','stream=width,height,avg_frame_rate,duration','-of','json',capture.video])).streams[0]
const base = stream(first)
const rgb = run(['ffmpeg','-v','error','-ss','2','-i',first.video,'-frames:v','1','-f','rawvideo','-pix_fmt','rgb24','pipe:1'])
let center = 0, maximum = 0
for (let y = 250; y < Math.min(base.height - 200, 1100); y++) {
  let count = 0
  for (let x = 0; x < base.width; x++) {
    const i = (y * base.width + x) * 3
    if (green(rgb[i], rgb[i+1], rgb[i+2])) count++
  }
  if (count > maximum) { maximum = count; center = y }
}
if (maximum < 25) throw Error('Could not identify the animated canvas ball')
const allFrames = [], results = []
for (const capture of captures) {
  const info = stream(capture)
  const pts = JSON.parse(run(['ffprobe','-v','error','-select_streams','v:0','-show_entries','frame=best_effort_timestamp_time','-of','json',capture.video])).frames.map(f => Number(f.best_effort_timestamp_time))
  const deltaY = (capture.canvas.y - first.canvas.y) * info.height / capture.canvas.screenHeight
  const y0 = Math.round(center + deltaY - 60), height = 120, bytes = info.width * height * 3
  const ff = spawn('ffmpeg',['-v','error','-i',capture.video,'-vf',`crop=${info.width}:${height}:0:${y0}`,'-fps_mode','passthrough','-f','rawvideo','-pix_fmt','rgb24','pipe:1'],{stdio:['ignore','pipe','pipe']})
  let pending = Buffer.alloc(0), stderr = '', index = 0
  const frames = []
  ff.stderr.on('data',b=>stderr+=b)
  ff.stdout.on('data',chunk => {
    pending = Buffer.concat([pending,chunk])
    while (pending.length >= bytes) {
      const data = pending.subarray(0,bytes)
      let sum = 0, pixels = 0
      for (let i = 0; i < bytes; i+=3) if (green(data[i],data[i+1],data[i+2])) { sum += (i / 3) % info.width; pixels++ }
      frames.push({mode:capture.mode,t:pts[index++],x:pixels > 500 ? sum / pixels : null,pixels})
      pending = pending.subarray(bytes)
    }
  })
  await new Promise((resolve,reject)=>{ff.on('error',reject);ff.on('close',code=>code===0?resolve():reject(Error(stderr)))})
  if (frames.length !== pts.length) throw Error('Video timestamps and decoded frames differ')
  const start = capture.rows.findLastIndex(r=>r.phase==='environment'), rows = capture.rows.slice(Math.max(start,0))
  const time = row => (Date.parse(row.at)-capture.deviceMinusHostMs-capture.recordHostStartMs)/1000
  const idle = rows.find(r=>r.phase==='idle'), load=rows.find(r=>r.phase==='load'), prefill=rows.find(r=>r.phase==='prefill'), decode=rows.find(r=>r.phase==='decode')
  const windows = {idle:[time(idle)-idle.durationMs/1000,time(idle)],prefill:[time(load),time(prefill)],decode:[time(prefill),time(decode)]}
  const phases = {}
  for (const [phase,[from,to]] of Object.entries(windows)) {
    const chosen = frames.filter(f=>f.t>=from+0.35&&f.t<=to-0.35), valid=chosen.filter(f=>f.x!==null)
    if (!valid.length || chosen.length!==valid.length) throw Error(`Missing ball in ${capture.mode} ${phase}`)
    let anchor=valid[0].x, began=valid[0].t, longest=0, motion=0
    for (const f of valid.slice(1)) {
      if (Math.abs(f.x-anchor)>0.25) { longest=Math.max(longest,f.t-began); anchor=f.x; began=f.t; motion++ }
    }
    longest=Math.max(longest,valid.at(-1).t-began)
    phases[phase]={frames:chosen.length,spanSeconds:valid.at(-1).t-valid[0].t,motionEventsPerSecond:motion/(valid.at(-1).t-valid[0].t),longestPositionHoldMs:longest*1000}
  }
  allFrames.push(...frames)
  results.push({mode:capture.mode,id:capture.id,videoFile:capture.video.split('/').at(-1),recording:info,roi:{x:0,y:y0,width:info.width,height},phases})
}
fs.writeFileSync(output,JSON.stringify({method:'Green-ball centroid in 720x1600 Android screenrecord frames; 0.25 px motion threshold; discard 350 ms at phase edges. This recording adds compositor/encoder work, so these are diagnostic motion events, not primary model throughput or an exact display FPS counter.',results},null,2)+'\n')
fs.writeFileSync(frameOutput,allFrames.map(JSON.stringify).join('\n')+'\n')
console.log(JSON.stringify(results.map(x=>({mode:x.mode,phases:x.phases}))))
