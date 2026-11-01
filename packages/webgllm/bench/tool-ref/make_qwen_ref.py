"""Qwen3.5 reference set for tools.html: the same 20 prompts and tools as the LFM3 set, rendered with
the model's chat template (thinking off) as token ids. No logits: references come from llama_ref.py.
usage: make_qwen_ref.py <models/lfm3-1.7b-web-onnx/q4f16/scripts> <Qwen3.5 model dir> <out dir> <vocab rows>"""
import sys, os, re, json
scripts, model, out, vocab = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
sys.path.insert(0, scripts)
import corpus
from jinja2.sandbox import ImmutableSandboxedEnvironment
from tokenizers import Tokenizer
src = open(os.path.join(scripts, 'tool_rate.py')).read()
Q = eval(re.search(r"Q = (\[.*?\])\n", src, re.S).group(1))
tok = Tokenizer.from_file(os.path.join(model, 'tokenizer.json'))
def raise_exception(msg): raise RuntimeError(msg)
env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True, extensions=['jinja2.ext.loopcontrols'])
env.filters['tojson'] = lambda v, indent=None, ensure_ascii=False, **k: json.dumps(v, indent=indent, ensure_ascii=ensure_ascii)
env.globals['raise_exception'] = raise_exception
T = env.from_string(open(os.path.join(model, 'chat_template.jinja')).read())
prompts = []
for q in Q:
    p = T.render(messages=[{'role': 'user', 'content': q}], tools=corpus.TOOLS, add_generation_prompt=True, enable_thinking=False)
    prompts.append({'question': q, 'ids': tok.encode(p, add_special_tokens=False).ids})
os.makedirs(out, exist_ok=True)
meta = {'tool_call_id': tok.token_to_id('<tool_call>'), 'eos_id': tok.token_to_id('<|im_end|>'), 'vocab': vocab, 'prompts': prompts, 'refs': []}
json.dump(meta, open(os.path.join(out, 'tool_prompts.json'), 'w'), ensure_ascii=False, indent=1)
print('tool_call', meta['tool_call_id'], 'eos', meta['eos_id'], 'lengths', [len(p['ids']) for p in prompts])
print(repr(T.render(messages=[{'role': 'user', 'content': Q[0]}], tools=corpus.TOOLS, add_generation_prompt=True, enable_thinking=False)[-300:]))
