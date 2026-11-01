"""LFM3 reference set for tools.html: the 20 prompts of tool_rate.py (models/lfm3-1.7b-web-onnx)
as token ids, and onnxruntime CPU's logits at the first reply position for fp16 and onnx-community's
q4f16 (as shipped), plus q4f16's greedy replies.
usage: make_lfm3_ref.py <models/lfm3-1.7b-web-onnx/q4f16/scripts> <onnx-community/LFM3-1.7B-ONNX dir> <out dir>
The model dir needs CPU copies of the graphs (cpu_patch.py): onnx/model_fp16.cpu.onnx, onnx/model_q4f16.cpu.onnx."""
import sys, os, json, re, numpy as np
scripts, dl, out = sys.argv[1:4]
sys.path.insert(0, scripts)
import lm as L, jinja2, corpus
src = open(os.path.join(scripts, 'tool_rate.py')).read()
Q = eval(re.search(r"Q = (\[.*?\])\n", src, re.S).group(1))
tok, _ = L.load_tok(dl)
tmpl = re.sub(r'\{%-?\s*(end)?generation\s*-?%\}', '', open(os.path.join(dl, 'chat_template.jinja')).read())
env = jinja2.Environment(trim_blocks=True, lstrip_blocks=True); env.filters['tojson'] = lambda v, **k: json.dumps(v, ensure_ascii=False)
T = env.from_string(tmpl)
prompts = []
for q in Q:
    p = T.render(messages=[{'role': 'user', 'content': q}], tools=corpus.TOOLS, bos_token='<|startoftext|>', add_generation_prompt=True, enable_thinking=False)
    prompts.append({'question': q, 'ids': tok.encode(p, add_special_tokens=False).ids})
tc = tok.token_to_id('<tool_call>')
meta = {'tool_call_id': tc, 'eos_id': L.EOS, 'vocab': None, 'prompts': prompts, 'refs': ['q4f16', 'fp16']}
for name, path in [('fp16', 'onnx/model_fp16.cpu.onnx'), ('q4f16', 'onnx/model_q4f16.cpu.onnx')]:
    lm = L.LM(os.path.join(dl, path)); rows = []; ps = []; greedy = 0; replies = []
    for pr in prompts:
        ids = pr['ids']
        if name == 'q4f16':
            toks, logits = L.generate(lm, ids, 48); lg = logits[0]
            replies.append(tok.decode(toks, skip_special_tokens=False))
        else:
            lg, _ = lm.step(ids, lm.empty_cache(), len(ids))
        rows.append(lg); p = np.exp(lg - lg.max()); p /= p.sum(); ps.append(float(p[tc])); greedy += int(lg.argmax() == tc)
    arr = np.stack(rows).astype(np.float32); meta['vocab'] = arr.shape[1]
    arr.tofile(os.path.join(out, f'logits_{name}.bin'))
    meta[name] = {'p_tool_call': ps, 'mean_p': float(np.mean(ps)), 'greedy_tool_calls': greedy}
    if replies: meta[name]['replies'] = replies
    print(f'{name}: mean P(<tool_call> first)={np.mean(ps):.3f}  greedy {greedy}/{len(prompts)}', flush=True)
json.dump(meta, open(os.path.join(out, 'tool_prompts.json'), 'w'), ensure_ascii=False, indent=1)
print('prompt lengths', [len(p['ids']) for p in prompts])
