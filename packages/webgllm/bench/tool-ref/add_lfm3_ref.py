"""Add an onnxruntime CPU reference (logits at the first reply position, greedy replies) to an LFM3
set's tool_prompts.json, first in its refs. For exact 4-bit math, set every MatMulNBits node's
accuracy_level to 0 in a copy of the CPU graph (onnx-community's q4f16 ships with 4, which quantizes
activations to 8 bits on the CPU) and add it as q4f16_exact.
usage: add_lfm3_ref.py <scripts dir> <model dir> <set dir> <name> <graph path in model dir>"""
import sys, os, json, numpy as np
scripts, dl, out, name, graph = sys.argv[1:6]
sys.path.insert(0, scripts)
import lm as L
tok, _ = L.load_tok(dl)
m = json.load(open(os.path.join(out, 'tool_prompts.json'))); tc = m['tool_call_id']
lm = L.LM(os.path.join(dl, graph)); rows = []; ps = []; greedy = 0; replies = []
for pr in m['prompts']:
    toks, logits = L.generate(lm, pr['ids'], 48); lg = logits[0]
    replies.append(tok.decode(toks, skip_special_tokens=False))
    rows.append(lg); p = np.exp(lg - lg.max()); p /= p.sum(); ps.append(float(p[tc])); greedy += int(lg.argmax() == tc)
np.stack(rows).astype(np.float32).tofile(os.path.join(out, f'logits_{name}.bin'))
m['refs'] = [name] + [r for r in m.get('refs', []) if r != name]
m[name] = {'p_tool_call': ps, 'mean_p': float(np.mean(ps)), 'greedy_tool_calls': greedy, 'replies': replies}
json.dump(m, open(os.path.join(out, 'tool_prompts.json'), 'w'), ensure_ascii=False, indent=1)
print(f'{name}: mean P(<tool_call> first)={np.mean(ps):.3f}  greedy {greedy}/{len(ps)}')
