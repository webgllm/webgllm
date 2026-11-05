"""Independent llama.cpp CPU distributions. No runtime or upstream source patches.
Usage: python3 lfm2-gguf-reference.py MODEL_DIR LLAMA_SERVER [--key instruct --f16-control FILE]
Run prepare-lfm2-gguf first. Model files stay outside the repository.
"""
import argparse, hashlib, json, math, pathlib, subprocess, time, urllib.request
parser=argparse.ArgumentParser()
parser.add_argument('directory',type=pathlib.Path)
parser.add_argument('server')
parser.add_argument('--port',type=int,default=8799)
parser.add_argument('--key',choices=['350m','instruct','thinking'])
parser.add_argument('--f16-control',type=pathlib.Path)
args=parser.parse_args()
models={'350m':'LFM2.5-350M-QAD-Q4_0.gguf','instruct':'LFM2.5-1.2B-Instruct-QAD-Q4_0.gguf','thinking':'LFM2.5-1.2B-Thinking-Q4_0.gguf'}
if args.f16_control and args.key!='instruct':parser.error('F16 control requires --key instruct')
version=subprocess.check_output([args.server,'--version'],stderr=subprocess.STDOUT,text=True).strip()
def post(path,body):
 request=urllib.request.Request(f'http://127.0.0.1:{args.port}'+path,data=json.dumps(body).encode(),headers={'Content-Type':'application/json'})
 return json.load(urllib.request.urlopen(request,timeout=120))
def full(step):
 values=[None]*65536
 for item in step['top_logprobs']:values[item['id']]=item['logprob']
 if any(value is None for value in values):raise RuntimeError('Reference did not return every vocabulary probability')
 return values
for key,file in models.items():
 if args.key and args.key!=key:continue
 model=args.f16_control or args.directory/file
 with (args.directory/f'llama-{key}-reference.log').open('wb') as log:
  process=subprocess.Popen([args.server,'-m',str(model),'-ngl','0','--ctx-size','4096','--parallel','1','--host','127.0.0.1','--port',str(args.port)],stdout=log,stderr=subprocess.STDOUT)
  try:
   deadline=time.monotonic()+60
   while True:
    try:
     if json.load(urllib.request.urlopen(f'http://127.0.0.1:{args.port}/health',timeout=1)).get('status')=='ok':break
    except Exception:
     if process.poll() is not None or time.monotonic()>deadline:raise RuntimeError('Reference server failed to start')
     time.sleep(.1)
   fixtures=json.loads((args.directory/key/'fixtures.json').read_text())
   original=json.loads((args.directory/key/'refs.json').read_text()) if args.f16_control else None
   prompts=[]
   for index,prompt in enumerate(fixtures['prompts']):
    ids=post('/tokenize',{'content':prompt['rendered'],'add_special':False,'parse_special':True})['tokens']
    if ids!=prompt['ids']:raise RuntimeError('Native and HF tokenization differ')
    if original:
     prefix=ids[:];steps=[]
     for expected in original['prompts'][index]['steps']:
      step=post('/completion',{'prompt':prefix,'n_predict':1,'temperature':0,'n_probs':65536,'cache_prompt':False,'post_sampling_probs':False})['completion_probabilities'][0]
      values=full(step)
      steps.append({'id':expected['id'],'referenceTop':step['id'],'token':step['token'],'logProbs':values,'cpuQ4VsF16Kl':sum(math.exp(a)*(a-b) for a,b in zip(expected['logProbs'],values))})
      prefix.append(expected['id'])
    else:
     result=post('/completion',{'prompt':ids,'n_predict':4,'temperature':0,'n_probs':65536,'cache_prompt':False,'post_sampling_probs':False})
     steps=[{'id':step['id'],'token':step['token'],'logProbs':full(step)} for step in result['completion_probabilities']]
    prompts.append({'ids':ids,'steps':steps})
   with (args.directory/file).open('rb') as source:digest=hashlib.file_digest(source,'sha256').hexdigest()
   report={'key':key,'file':file,'sha256':digest,'reference':version+('; dequantized F16 control' if original else '; native quantized CPU')+'; context4096 fixture only','prompts':prompts}
   output=args.directory/key/('refs-f16.json' if original else 'refs.json')
   output.write_text(json.dumps(report,separators=(',',':')))
   print(key,digest,[len(prompt['steps']) for prompt in prompts],flush=True)
  finally:
   process.terminate()
   try:process.wait(timeout=10)
   except subprocess.TimeoutExpired:process.kill();process.wait()
