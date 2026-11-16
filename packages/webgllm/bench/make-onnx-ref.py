"""onnxruntime reference tensors for webgllm's check.html (format=onnx).

usage: python -I bench/make-onnx-ref.py <onnx-community/LFM3-1.7B-ONNX dir> <out dir> [--accuracy-level N] [--steps 32]

Needs Python packages numpy, onnx, onnxruntime, tokenizers and jinja2 (a venv of their own).

Writes, as check-worker.ts reads them (REF_ONNX_DIR, served at /ref-onnx/):
- manifest.json   {files: [{file, name, shape}]}
- tokens.json     {prompt, messages, token_ids, pieces, greedy_continuation: {token_ids, text}}
- tokens.bin      the prompt ids as int32
- logits.bin      [n][vocab] float32: the whole prompt in one run
- gen_logits.bin  [steps][vocab]: row i = logits after feeding greedy token i (one token a run)
- per layer, all prompt positions, row-major [token][feature] float32:
  l_out-N, operator_norm-N, resid_mid-N, ffn_out-N,
  conv layers: conv_in_proj-N (B | C | x), conv_gated-N (C * conv, the out_proj input), conv_out_proj-N
  attention layers: q_proj-N (before q_norm), k_proj-N, v_proj-N, attn_gate-N (after its sigmoid),
  attn_gated-N (the o_proj input), attn_out_proj-N
  result_norm (the final norm's output)

The graph is copied (graph only; the weights file is the original, hard-linked) with every intermediate
above added as a graph output, the final norm's residual sum exposed, and MatMulNBits'
accuracy_level set to --accuracy-level (default 0: no int8 quantization of activations on the CPU;
the shipped graph says 4).
"""
import argparse, json, os, re, sys
import numpy as np
import onnx
import onnxruntime as ort
from tokenizers import Tokenizer
import jinja2

ap = argparse.ArgumentParser()
ap.add_argument("model_dir")
ap.add_argument("out_dir")
ap.add_argument("--accuracy-level", type=int, default=0)
ap.add_argument("--steps", type=int, default=32)
ap.add_argument("--graph", default="onnx/model_q4f16.onnx")
args = ap.parse_args()
os.makedirs(args.out_dir, exist_ok=True)

# ---- The prompt: a system and a user message, thinking off, generation prompt on. Longer than the
# 256-token sliding window so the sliding layers' window is exercised.
SYSTEM = (
    "You are a careful technical assistant for a small team that maintains a weather station network "
    "in the mountains. Stations report temperature, humidity, wind speed, wind direction, pressure and "
    "battery voltage every ten minutes over a low-power radio link to a gateway in the valley. The "
    "gateway batches the readings and uploads them to a server once an hour. Answer questions about the "
    "network precisely and briefly. When you are not sure, say what you would check first. Prefer plain "
    "language over jargon, give numbers with their units, and never invent readings that were not given."
)
USER = (
    "Station 14 sits at 2,310 metres on a north-facing ridge. Since Tuesday its battery voltage has "
    "dropped from 12.8 V at noon to 11.9 V at noon, while the solar panel current at noon fell from "
    "1.4 A to 0.3 A. Overnight the temperature stayed between -9 and -4 degrees Celsius, the humidity "
    "was above 95 percent for most of the night, and the wind was light, under 3 metres per second, "
    "from the north-east. Two other stations on the same ridge, 12 and 15, show normal panel currents. "
    "Station 14 also missed four of its last thirty reports, always between 05:00 and 07:00. The "
    "radio link budget to the gateway has about 6 dB of margin in good weather.\n\n"
    "What is the most likely cause of the falling panel current, why do the reports go missing in "
    "the early morning, and what should the person who hikes up on Saturday bring and do? Keep it to "
    "a short list."
)
messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": USER}]

tok = Tokenizer.from_file(os.path.join(args.model_dir, "tokenizer.json"))
tcfg = json.load(open(os.path.join(args.model_dir, "tokenizer_config.json")))
tmpl_path = os.path.join(args.model_dir, "chat_template.jinja")
template_src = open(tmpl_path).read() if os.path.exists(tmpl_path) else tcfg["chat_template"]
# {% generation %} is a transformers-only tag (assistant masks); it renders nothing.
template_src = re.sub(r"\{%-?\s*(end)?generation\s*-?%\}", "", template_src)
env = jinja2.Environment(trim_blocks=True, lstrip_blocks=True, extensions=["jinja2.ext.loopcontrols"])
env.filters["tojson"] = lambda v, **k: json.dumps(v, ensure_ascii=False)
env.globals["raise_exception"] = lambda m: (_ for _ in ()).throw(Exception(m))
bos = tcfg.get("bos_token")
if isinstance(bos, dict):
    bos = bos.get("content")
prompt = env.from_string(template_src).render(
    messages=messages, bos_token=bos or "", add_generation_prompt=True, enable_thinking=False
)
ids = tok.encode(prompt, add_special_tokens=False).ids
n = len(ids)
print(f"prompt: {n} tokens", flush=True)

# ---- The graph with intermediates as outputs.
src = os.path.join(args.model_dir, args.graph)
m = onnx.load(src, load_external_data=False)
g = m.graph
cfg = json.load(open(os.path.join(args.model_dir, "config.json")))
L = cfg["num_hidden_layers"]
types = cfg["layer_types"]
for node in g.node:
    if node.op_type == "MatMulNBits":
        for a in node.attribute:
            if a.name == "accuracy_level":
                a.i = args.accuracy_level
    if node.name == f"/model/layers.{L}/final_norm_layernorm/SkipLayerNorm":
        outs = list(node.output) + [""] * (4 - len(node.output))
        outs[3] = "ref/final_residual"
        del node.output[:]
        node.output.extend(outs)

P = "/model/layers.{}/"
want = {}  # file name -> graph tensor name
for i in range(L):
    p = P.format(i)
    want[f"operator_norm-{i}"] = p + "operator_layernorm/output_0"
    want[f"resid_mid-{i}"] = p + "ffn_layernorm/output_3"
    want[f"ffn_out-{i}"] = p + "mlp/down_proj/MatMul/output_0"
    want[f"l_out-{i}"] = (P.format(i + 1) + "operator_layernorm/output_3") if i + 1 < L else "ref/final_residual"
    if types[i] == "conv":
        want[f"conv_in_proj-{i}"] = p + "conv/in_proj/MatMul/output_0"
        want[f"conv_gated-{i}"] = p + "conv/Mul_2/output_0"
        want[f"conv_out_proj-{i}"] = p + "conv/out_proj/MatMul/output_0"
    else:
        want[f"q_proj-{i}"] = p + "attn/q_proj/Split/output_0"
        want[f"k_proj-{i}"] = p + "attn/k_proj/MatMul/output_0"
        want[f"v_proj-{i}"] = p + "attn/v_proj/MatMul/output_0"
        want[f"attn_gate-{i}"] = p + "attn/gate_proj/Sigmoid/output_0"
        want[f"attn_gated-{i}"] = p + "attn/gate_proj/Reshape_2/output_0"
        want[f"attn_out_proj-{i}"] = p + "attn/o_proj/MatMul/output_0"
want["result_norm"] = f"/model/layers.{L}/final_norm_layernorm/output_0"
produced = {o for node in g.node for o in node.output}
missing = [k for k, v in want.items() if v not in produced]
if missing:
    sys.exit(f"tensors not in the graph: {missing[:8]}")
existing = {o.name for o in g.output}
for t in want.values():
    if t not in existing:
        g.output.append(onnx.ValueInfoProto(name=t))

gdir = os.path.join(args.out_dir, "graph")
os.makedirs(gdir, exist_ok=True)
data_name = os.path.basename(src) + "_data"
link = os.path.join(gdir, data_name)
weights = os.path.abspath(os.path.join(os.path.dirname(src), data_name))
# A hard link: onnxruntime refuses data outside the graph's directory. One left
# by an earlier run (another model, a reused out dir) is replaced, never kept.
if os.path.lexists(link) and not (os.path.exists(link) and os.path.samefile(link, weights)):
    os.remove(link)
if not os.path.exists(link):
    os.link(weights, link)
gpath = os.path.join(gdir, os.path.basename(src))
onnx.save(m, gpath)

so = ort.SessionOptions()
so.log_severity_level = 3
sess = ort.InferenceSession(gpath, so, providers=["CPUExecutionProvider"])
print("onnxruntime", ort.__version__, "accuracy_level", args.accuracy_level, flush=True)
in_types = {i.name: i for i in sess.get_inputs()}
out_names = [o.name for o in sess.get_outputs()]
state_names = [o for o in out_names if o.startswith("present")]


def empty_feeds():
    f = {}
    for name, i in in_types.items():
        if name.startswith("past"):
            dt = np.float16 if "float16" in i.type else np.float32
            shape = [1 if d == "batch_size" else (0 if isinstance(d, str) else d) for d in i.shape]
            f[name] = np.zeros(shape, dt)
    return f


def carry(f, o):
    for k, v in o.items():
        if k.startswith("present_conv."):
            f["past_conv." + k[len("present_conv."):]] = v
        elif k.startswith("present."):
            f["past_key_values." + k[len("present."):]] = v


feeds = empty_feeds()
feeds["input_ids"] = np.array([ids], np.int64)
feeds["attention_mask"] = np.ones((1, n), np.int64)
feeds["num_logits_to_keep"] = np.array(n, np.int64)
names = ["logits"] + state_names + list(want.values())
res = dict(zip(names, sess.run(names, feeds)))
logits = res["logits"][0].astype(np.float32)
assert logits.shape[0] == n, logits.shape
V = logits.shape[1]
files = []


def write(file, arr, name):
    arr = np.ascontiguousarray(arr.astype(np.float32))
    arr.tofile(os.path.join(args.out_dir, file))
    files.append({"file": file, "name": name, "shape": list(arr.shape)})


write("logits.bin", logits, "logits")
for key, t in want.items():
    a = res[t][0]  # drop the batch
    write(f"{key}.bin", a.reshape(n, -1), t)

# ---- Greedy continuation, a token a run.
carry(feeds, {k: res[k] for k in state_names})
cont = [int(logits[-1].argmax())]
gen = []
total = n
for step in range(args.steps):
    total += 1
    feeds["input_ids"] = np.array([[cont[-1]]], np.int64)
    feeds["attention_mask"] = np.ones((1, total), np.int64)
    feeds["num_logits_to_keep"] = np.array(1, np.int64)
    o = dict(zip(["logits"] + state_names, sess.run(["logits"] + state_names, feeds)))
    row = o["logits"][0, -1].astype(np.float32)
    gen.append(row)
    carry(feeds, o)
    if step + 1 < args.steps:
        cont.append(int(row.argmax()))
write("gen_logits.bin", np.stack(gen), "logits (decode)")
np.array(ids, np.int32).tofile(os.path.join(args.out_dir, "tokens.bin"))

json.dump(
    {
        "prompt": prompt,
        "messages": messages,
        "token_ids": ids,
        "pieces": [tok.id_to_token(t) for t in ids],
        "greedy_continuation": {"token_ids": cont, "text": tok.decode(cont, skip_special_tokens=False)},
        "generator": {
            "onnxruntime": ort.__version__,
            "provider": "CPUExecutionProvider",
            "graph": args.graph,
            "accuracy_level": args.accuracy_level,
            "prompt_in_one_run": True,
        },
    },
    open(os.path.join(args.out_dir, "tokens.json"), "w"),
    ensure_ascii=False,
    indent=1,
)
json.dump({"files": files}, open(os.path.join(args.out_dir, "manifest.json"), "w"), indent=1)
print("continuation:", repr(tok.decode(cont, skip_special_tokens=False)))
print(f"wrote {len(files)} files to {args.out_dir}")
