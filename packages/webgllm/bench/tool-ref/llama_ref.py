"""A llama.cpp reference for a set: each prompt's first-token probabilities (top 50) and a 48-token
greedy reply from a running llama-server, written to <set dir>/llama_<name>.json.
usage: llama_ref.py <set dir> <name> [port] [reply tokens]   (llama-server -m model.gguf -ngl 0 --port 8791)"""
import json, sys, urllib.request, math
D, NAME = sys.argv[1], sys.argv[2]
PORT = sys.argv[3] if len(sys.argv) > 3 else "8791"
STEPS = int(sys.argv[4]) if len(sys.argv) > 4 else 48
def post(path, body):
    r = urllib.request.Request(f"http://127.0.0.1:{PORT}" + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(r, timeout=600))
m = json.load(open(f"{D}/tool_prompts.json"))
print("tokenize <tool_call>:", post("/tokenize", {"content": "<tool_call>", "add_special": False, "parse_special": True})["tokens"], "expected", m["tool_call_id"])
print("detokenize first 12 ids:", repr(post("/detokenize", {"tokens": m["prompts"][0]["ids"][:12]})["content"]))
out = []; sp = 0; g = 0
for i, p in enumerate(m["prompts"]):
    r = post("/completion", {"prompt": p["ids"], "n_predict": STEPS, "temperature": 0, "top_k": 1, "n_probs": 50, "cache_prompt": False, "return_tokens": True, "samplers": ["top_k"]})
    first = r["completion_probabilities"][0]
    ptc = 0.0
    for c in first["top_logprobs"]:
        if c["id"] == m["tool_call_id"]: ptc = math.exp(c["logprob"])
    top = first["top_logprobs"][0]["id"]
    sp += ptc; g += top == m["tool_call_id"]
    out.append({"i": i, "p_tool_call": ptc, "top": top, "reply": r["content"], "evaluated": r["tokens_evaluated"], "n_ids": len(p["ids"])})
    print(f"#{i:2d} P(<tool_call>) {ptc:.3f} top {'<tool_call>' if top == m['tool_call_id'] else top} eval {r['tokens_evaluated']}/{len(p['ids'])}  {r['content'][:90]!r}", flush=True)
print(f"SUMMARY llama.cpp {NAME}: mean P(<tool_call> first) {sp/len(out):.3f}, greedy {g}/{len(out)}")
json.dump(out, open(f"{D}/llama_{NAME}.json", "w"), indent=1, ensure_ascii=False)
