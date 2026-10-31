"""
Checks that a GGUF file's embedded tokenizer and chat template are the ones
src/tokenizer.ts loads from the original repo (LiquidAI/LFM3-1.7B):

- `tokenizer.chat_template` byte for byte against chat_template.jinja,
- `tokenizer.ggml.tokens` against tokenizer.json's vocabulary and added
  tokens, id by id (ids past tokenizer.json's are listed: padding rows),
- `tokenizer.ggml.merges` against tokenizer.json's merges,
- the special token ids (bos, eos, pad, ...) and token types.

    python check-gguf-tokenizer.py MODEL.gguf DIR

DIR holds tokenizer.json and chat_template.jinja. Standard library only.
"""

import json
import os
import struct
import sys

SCALARS = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}


def read_metadata(path):
    """The GGUF header's key/value pairs (strings as bytes)."""
    with open(path, "rb") as f:

        def rd(fmt):
            n = struct.calcsize("<" + fmt)
            return struct.unpack("<" + fmt, f.read(n))[0]

        def rstr():
            return f.read(rd("Q"))

        def rval(t):
            if t == 8:
                return rstr()
            if t == 9:
                et = rd("I")
                return [rval(et) for _ in range(rd("Q"))]
            return rd(SCALARS[t])

        if f.read(4) != b"GGUF":
            raise SystemExit("Not a GGUF file.")
        version = rd("I")
        rd("Q")  # tensor count
        kv = {}
        for _ in range(rd("Q")):
            key = rstr().decode()
            kv[key] = rval(rd("I"))
        return version, kv


def main():
    gguf_path, directory = sys.argv[1], sys.argv[2]
    version, kv = read_metadata(gguf_path)
    failures = []

    template = open(os.path.join(directory, "chat_template.jinja"), "rb").read()
    embedded = kv.get("tokenizer.chat_template")
    same_template = embedded == template
    print(f"GGUF v{version}, {len(kv)} keys; model {kv.get('tokenizer.ggml.model', b'?').decode()}"
          f", pre {kv.get('tokenizer.ggml.pre', b'?').decode()}")
    print(f"chat template: {len(embedded or b'')} bytes embedded, {len(template)} in chat_template.jinja, "
          f"{'identical' if same_template else 'DIFFERENT'}")
    if not same_template:
        failures.append("chat template")

    tj = json.load(open(os.path.join(directory, "tokenizer.json"), encoding="utf-8"))
    by_id = {i: t for t, i in tj["model"]["vocab"].items()}
    added = {a["id"]: a for a in tj["added_tokens"]}
    for i, a in added.items():
        by_id[i] = a["content"]
    tokens = [t.decode("utf-8") for t in kv["tokenizer.ggml.tokens"]]
    types = kv.get("tokenizer.ggml.token_type", [])
    known = max(by_id) + 1
    mismatched = [i for i in range(min(len(tokens), known)) if tokens[i] != by_id.get(i)]
    print(f"tokens: {len(tokens)} in the GGUF, {known} in tokenizer.json; "
          f"{len(mismatched)} of the first {min(len(tokens), known)} differ")
    for i in mismatched[:10]:
        print(f"  id {i}: GGUF {tokens[i]!r}, tokenizer.json {by_id.get(i)!r}")
    if mismatched or len(tokens) < known:
        failures.append("tokens")
    if len(tokens) > known:
        extra = tokens[known:]
        print(f"  ids {known}..{len(tokens) - 1} only in the GGUF: {extra[:3]} ... {extra[-1:]}"
              f" (types {sorted(set(types[known:]))})")

    # Token types: 3 = control (special), 4 = user-defined (added, not special).
    wrong_types = []
    for i, a in added.items():
        want = 3 if a["special"] else 4
        if i < len(types) and types[i] != want:
            wrong_types.append((i, a["content"], types[i], want))
    print(f"added tokens: {len(added)}; {len(wrong_types)} with an unexpected GGUF token type")
    for w in wrong_types[:10]:
        print("  ", w)

    merges = kv.get("tokenizer.ggml.merges")
    if merges is not None:
        ours = [" ".join(m) if isinstance(m, list) else m for m in tj["model"]["merges"]]
        theirs = [m.decode("utf-8") for m in merges]
        same_merges = ours == theirs
        print(f"merges: {len(theirs)} in the GGUF, {len(ours)} in tokenizer.json, "
              f"{'identical' if same_merges else 'DIFFERENT'}")
        if not same_merges:
            failures.append("merges")

    for key, name in [
        ("bos_token_id", "<|startoftext|>"),
        ("eos_token_id", "<|im_end|>"),
        ("padding_token_id", "<|pad|>"),
    ]:
        value = kv.get(f"tokenizer.ggml.{key}")
        ok = value is not None and tokens[value] == name and by_id.get(value) == name
        print(f"{key}: {value} = {tokens[value] if value is not None else None!r} (want {name}) {'ok' if ok else 'DIFFERENT'}")
        if not ok:
            failures.append(key)
    for key in ["add_bos_token", "add_eos_token", "add_space_prefix"]:
        if f"tokenizer.ggml.{key}" in kv:
            print(f"{key}: {kv[f'tokenizer.ggml.{key}']}")

    print("All checks passed." if not failures else f"Failed: {', '.join(failures)}")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
