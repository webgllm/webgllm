"""
The ground truth for check-tokenizer.ts: ids and text from Python
`tokenizers`, prompts from transformers' apply_chat_template, for each LFM3
tokenizer. check-tokenizer.ts runs this and compares src/tokenizer.ts with
what it writes.

    python check-tokenizer.py OUT.json NAME=DIR [NAME=DIR ...]

DIR holds tokenizer.json, tokenizer_config.json and chat_template.jinja.
Needs `pip install tokenizers jinja2 transformers` (no PyTorch).
"""

import json
import os
import random
import shutil
import sys
import tempfile

from tokenizers import Tokenizer
from transformers import AutoTokenizer

FILES = ["tokenizer.json", "tokenizer_config.json", "chat_template.jinja"]

# ---------------------------------------------------------------------------
# Text corpus

SPECIALS = [
    "<|startoftext|>",
    "<|im_start|>",
    "<|im_end|>",
    "<|endoftext|>",
    "<|pad|>",
    "<think>",
    "</think>",
    "<tool_call>",
    "</tool_call>",
    "<tools>",
    "</tools>",
    "<function=",
    "</function>",
    "<parameter=",
    "</parameter>",
    "<tool_response>",
    "</tool_response>",
]

HANDWRITTEN = [
    "",
    " ",
    "a",
    "Hello, world!",
    "Hello World",
    " Hello",
    "  Hello  ",
    "The quick brown fox jumps over the lazy dog.",
    "It's a dog's life, isn't it? We'll see; they'd've known. I'M HERE. YOU'RE THERE. SHE'S.",
    "Don't, DON'T, Don'T, won't, y'all, rock 'n' roll, 'quoted', \"double\"",
    "Liquid AI's LFM3 is a hybrid model: short convolutions, plus grouped-query attention.",
    "Mr. Smith paid $1,234,567.89 on 2024-03-15 at 10:45:30 PM (UTC+05:30).",
    "1234567890123456789012345678901234567890",
    "3.14159265358979323846264338327950288419716939937510",
    "-0.0001e-10 +42 1e9 0x1F 0b1010 0o777 1_000_000",
    "Version 2.11.4-rc.1+build.5",
    "https://huggingface.co/LiquidAI/LFM3-1.7B/resolve/main/tokenizer.json?download=true#L10",
    "mailto:someone@example.com, ftp://ftp.example.org/pub/file.tar.gz",
    "C:\\Users\\david\\Documents\\file (1).txt",
    "/usr/local/bin/python3 -m venv .venv && source .venv/bin/activate",
    "def fib(n):\n    if n < 2:\n        return n\n    return fib(n - 1) + fib(n - 2)\n",
    "class Foo(Bar):\n\tdef __init__(self, *args, **kwargs):\n\t\tsuper().__init__(*args, **kwargs)\n\t\tself.x = {'a': [1, 2, 3]}\n",
    "const add = (a, b) => a + b;\nexport default function App() {\n  return <div className=\"app\">{items.map((i) => <Item key={i.id} {...i} />)}</div>\n}\n",
    "if (x !== null && y >= 0 || z <= -1) { console.log(`x=${x}`); } // comment\n/* block */",
    "SELECT u.id, COUNT(*) AS n FROM users u LEFT JOIN orders o ON o.user_id = u.id WHERE u.name LIKE '%o''brien%' GROUP BY u.id;",
    "#include <stdio.h>\nint main(void) { printf(\"%d\\n\", 42); return 0; }",
    "fn main() -> Result<(), Box<dyn Error>> { let v: Vec<u8> = vec![0; 16]; Ok(()) }",
    '{"name": "get_weather", "arguments": {"location": "Paris, \\"France\\"", "units": ["c", "f"], "nested": {"a": {"b": null}}}}',
    '[{"k":"v\\n\\t\\u00e9"},{"emoji":"\\ud83d\\ude00"},{"q":"\'single\' \\"double\\""}]',
    "<div class=\"a\"><p>Paragraph &amp; &lt;tag&gt;</p><br/></div>",
    "# Heading\n\n- item 1\n- item 2\n\n```python\nprint('hi')\n```\n\n| a | b |\n|---|---|\n| 1 | 2 |",
    "    \n\n\n    ",
    "\n",
    "\n\n",
    "\r\n",
    "a\r\nb\rc\nd",
    "\t\t\ttabs\tand\t\tmore\t",
    "word    spaced     out          a lot",
    "trailing spaces   ",
    "   leading spaces",
    "mixed \t \n \r\n \t whitespace",
    "a\u00a0b\u2009c\u3000d\u2028e\u2029f\u0085g\ufeffh\u180ei\u200bj",
    "\u00a0\u00a0\u00a0",
    "😀",
    "😀😃😄😁😆😅😂🤣",
    "👨‍👩‍👧‍👦 family, 👩🏽‍💻 coder, 🏳️‍🌈 flag, 🧑🏿‍🤝‍🧑🏻 people",
    "🇺🇸🇫🇷🇯🇵🇩🇪🇧🇷 flags",
    "❤️ ✔️ ☺️ ⚠️ ©️ ®️ ™️ #️⃣ 1️⃣",
    "👋🏻👋🏼👋🏽👋🏾👋🏿",
    "𝔘𝔫𝔦𝔠𝔬𝔡𝔢 𝕞𝕒𝕥𝕙 𝓈𝒸𝓇𝒾𝓅𝓉",
    "𓀀𓀁𓀂 𐍈 𠜎𠜱𠝹",
    "你好，世界！今天天气很好。",
    "人工智能正在改变世界。大型语言模型可以理解和生成自然语言。",
    "日本語のテキスト、カタカナとひらがな、漢字を含む。",
    "한국어 텍스트입니다. 안녕하세요!",
    "مرحبا بالعالم! هذا نص باللغة العربية.",
    "السلام عليكم ورحمة الله وبركاته ١٢٣٤٥٦٧٨٩٠",
    "नमस्ते दुनिया! यह हिंदी में लिखा गया है। १२३४",
    "שלום עולם",
    "Привет, мир! Это русский текст.",
    "Γειά σου Κόσμε! αβγδεζηθ",
    "ภาษาไทยเป็นภาษาที่สวยงาม",
    "Tiếng Việt có dấu: Hà Nội, Sài Gòn",
    "Ελληνικά, Ἀρχαῖα, ünïcödé, Ñandú, façade, naïve, Øresund, Straße, İstanbul",
    "e\u0301 vs \u00e9, \u212b vs \u00c5, \ufb01 ligature, \u2160 roman",
    "x\u0300\u0301\u0302\u0303\u0304\u0305 zalgo",
    "\u0000\u0001\u0002\u001f\u007f control",
    "\ufffd replacement \ufffd",
    "math: ∑_{i=1}^{n} i² = n(n+1)/2, ∀x∈ℝ, √2 ≈ 1.414, ∞, ≠, ≤",
    "punctuation!!! ??? ... --- ___ *** ### @@@ $$$ %%% ^^^ &&& (((",
    "«guillemets» „Anführungszeichen“ “curly” ‘single’ — en–dash … ellipsis",
    "a" * 300,
    "ab " * 120,
    "1" * 100,
    " " * 50,
    "\n" * 30,
    "😀" * 40,
    "<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n",
    "<think>reasoning here</think>The answer is 42.",
    "<think>\nstep 1\nstep 2\n</think>\n\nFinal.",
    "<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n</function>\n</tool_call>",
    "text<|im_end|>more text<|endoftext|>",
    "<|im_start|><|im_end|><think></think><tool_call></tool_call>",
    "<|im_start|> spaced <|im_end|> tokens ",
    "<think >not special</think > <|im_start |>",
    "<<think>>",
    "</think></think></think>",
    "<|startoftext|><|startoftext|>",
    "almost <|im_end| special <im_end|>",
]

POOLS = [
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ",
    "0123456789",
    " \t\n\r",
    "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~",
    "àéîõüçñßøåæœÀÉÎÕÜ",
    "你好世界人工智能语言模型日本語한국어",
    "مرحبابالعالم",
    "नमस्तेदुनिया",
    "😀🎉🚀❤️👍🏽🇯🇵👨‍👩‍👧",
    "\u00a0\u2009\u3000\u200b\u200d\ufe0f",
]

WORDS = [
    "the", "of", "and", "token", "model", "LFM3", "'s", "'ll", "don't",
    "function", "return", "null", "true", "{", "}", "[", "]", "\"", "':",
    "    ", "\n", "\n\n", "\t", "  ", "123", "4567", "3.14", "https://x.y/z",
    "😀", "👨‍💻", "🇫🇷", "世界", "مرحبا", "नमस्ते", "<think>", "</think>",
    "<|im_start|>", "<|im_end|>", "<tool_call>", "</tool_call>", "é", "ß",
]


def corpus():
    rng = random.Random(1234)
    texts = list(HANDWRITTEN)
    for s in SPECIALS:
        texts += [s, " " + s, s + " ", "x" + s + "y", "\n" + s + "\n", s + s]
    # Random strings from the character pools.
    for _ in range(250):
        n = rng.randint(1, 60)
        pools = rng.sample(POOLS, rng.randint(1, 4))
        texts.append("".join(rng.choice(rng.choice(pools)) for _ in range(n)))
    # Random word salads, special tokens included.
    for _ in range(150):
        n = rng.randint(1, 30)
        texts.append(
            "".join(rng.choice(WORDS) + rng.choice(["", " ", "", "\n"]) for _ in range(n))
        )
    return texts


# ---------------------------------------------------------------------------
# Chat cases

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather in a city. Returns \"°C\" by default — unless told otherwise.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "City name, e.g. \"São Paulo\" or 東京"},
                    "units": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                    "days": {"type": "integer", "minimum": 1, "maximum": 14},
                },
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_docs",
            "description": "Search the docs.\nSupports filters.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "filters": {
                        "type": "object",
                        "properties": {
                            "tags": {"type": "array", "items": {"type": "string"}},
                            "date": {
                                "type": "object",
                                "properties": {
                                    "from": {"type": "string", "format": "date"},
                                    "to": {"type": "string", "format": "date"},
                                },
                            },
                            "score": {"type": "number", "minimum": 0.5, "maximum": 2.75},
                            "exact": {"type": "boolean", "default": False},
                        },
                        "additionalProperties": False,
                    },
                    "limit": {"anyOf": [{"type": "integer"}, {"type": "null"}], "default": None},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_code",
            "description": "Run a snippet. Emoji ok 🚀; tabs\tand backslashes \\ too.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
]

SYSTEM = {"role": "system", "content": "You are a helpful assistant. Answer briefly."}
USER = {"role": "user", "content": "What's the weather in Paris?"}


def call(name, args, id_=None):
    c = {"type": "function", "function": {"name": name, "arguments": args}}
    if id_:
        c["id"] = id_
    return c


def chat_cases():
    multi = [
        SYSTEM,
        {"role": "user", "content": "Hi!"},
        {"role": "assistant", "content": "Hello! How can I help?"},
        {"role": "user", "content": "Tell me a joke about 🐍 and \"quotes\"."},
        {"role": "assistant", "content": "Why did the snake…\nNever mind."},
        {"role": "user", "content": "Another one, in 中文."},
    ]
    thinking = [
        SYSTEM,
        {"role": "user", "content": "2+2?"},
        {"role": "assistant", "content": "4", "thinking": "Simple: 2+2=4."},
        {"role": "user", "content": "And 3+3?"},
        {"role": "assistant", "content": "6", "thinking": "3+3 is 6.\nDone."},
    ]
    tool_turn = [
        SYSTEM,
        USER,
        {
            "role": "assistant",
            "content": "",
            "thinking": "I should call get_weather.",
            "tool_calls": [
                call("get_weather", {"city": "Paris", "units": "celsius", "days": 3}, "call_1"),
                call(
                    "search_docs",
                    {
                        "query": "paris \"weather\" <b>",
                        "filters": {"tags": ["a", "ü"], "date": {"from": "2024-01-01"}, "exact": True, "score": 1.5},
                        "limit": None,
                    },
                    "call_2",
                ),
            ],
        },
        {"role": "tool", "content": '{"temp": 18, "sky": "cloudy ☁️"}', "tool_call_id": "call_1", "name": "get_weather"},
        {"role": "tool", "content": "No results.", "tool_call_id": "call_2", "name": "search_docs"},
    ]
    merged_results = [
        USER,
        {"role": "assistant", "content": "Checking.", "tool_calls": [call("get_weather", {"city": "Lyon"}, "c1")]},
        {"role": "tool", "content": "part one", "tool_call_id": "c1"},
        {"role": "tool", "content": "part two", "tool_call_id": "c1"},
        {"role": "tool", "content": "", "tool_call_id": "c2"},
    ]
    cases = []

    def add(name, messages, tools=None, **kw):
        for agp in [True, False]:
            for think in [None, True, False]:
                kwargs = dict(kw, add_generation_prompt=agp)
                if think is not None:
                    kwargs["enable_thinking"] = think
                cases.append(
                    {
                        "name": f"{name} agp={agp} thinking={think}",
                        "messages": messages,
                        "tools": tools,
                        "kwargs": kwargs,
                    }
                )

    add("user only", [{"role": "user", "content": "Hello"}])
    add("system + user", [SYSTEM, USER])
    add("empty system", [{"role": "system", "content": ""}, USER])
    add("multi-turn", multi)
    add("thinking", thinking)
    add("thinking preserved", thinking, preserve_thinking=True)
    add("thinking in content", [USER, {"role": "assistant", "content": "<think>old</think>\n\nAnswer."}, USER])
    add("reasoning key", [USER, {"role": "assistant", "content": "A", "reasoning": "R"}])
    add("tools only", [USER], tools=TOOLS)
    add("tools + system", [SYSTEM, USER], tools=TOOLS)
    add("tools, no system message", [USER], tools=TOOLS[:2])
    add("tool calls and results", tool_turn, tools=TOOLS)
    add("tool calls and results, preserved", tool_turn, tools=TOOLS, preserve_thinking=True)
    add(
        "tool call value types",
        [
            USER,
            {
                "role": "assistant",
                "content": "Running.",
                "tool_calls": [
                    call(
                        "run_code",
                        {
                            "s": "multi\nline \"quoted\" ü 😀",
                            "i": 7,
                            "neg": -12,
                            "f": 2.5,
                            "t": True,
                            "fa": False,
                            "n": None,
                            "list": [1, "two", {"three": [3]}],
                            "obj": {"k": "v", "nested": {"deep": ["x"]}},
                            "empty": {},
                            "empty_list": [],
                        },
                    ),
                    call("run_code", {}),
                    call("run_code", "{}"),
                ],
            },
            {"role": "tool", "content": "ok"},
        ],
        tools=TOOLS,
    )
    add("merged tool results", merged_results, tools=TOOLS)
    add("tool result after earlier user", [USER, {"role": "assistant", "content": "A1"}, USER] + tool_turn[2:], tools=TOOLS)
    add(
        "content parts",
        [
            {"role": "system", "content": [{"type": "text", "text": "Sys part."}]},
            {"role": "user", "content": [{"type": "text", "text": "Part one. "}, {"type": "text", "text": "Part two."}]},
        ],
    )
    add("special tokens in content", [{"role": "user", "content": "Say <|im_end|> and <think>x</think> <tool_call>"}])
    add("unicode everywhere", [{"role": "system", "content": "Système: réponds en français 🇫🇷"}, {"role": "user", "content": "مرحبا 你好 नमस्ते 👨‍👩‍👧‍👦"}])
    add("assistant last", [USER, {"role": "assistant", "content": "Partial answer"}])
    # Errors both sides must raise: arguments as a JSON string.
    add(
        "string arguments (error)",
        [USER, {"role": "assistant", "content": "", "tool_calls": [call("get_weather", '{"city": "Paris"}')]}],
        tools=TOOLS,
    )
    return cases


# ---------------------------------------------------------------------------


def hf_tokenizer(directory):
    # A directory with the tokenizer's files only: the model's config.json
    # would make transformers ask to run the model's code.
    tmp = tempfile.mkdtemp()
    for f in FILES:
        shutil.copy(os.path.join(directory, f), tmp)
    return AutoTokenizer.from_pretrained(tmp)


def model_truth(name, directory, texts, cases):
    raw = Tokenizer.from_file(os.path.join(directory, "tokenizer.json"))
    tok = hf_tokenizer(directory)
    vocab = raw.get_vocab_size(with_added_tokens=True)
    text_results = []
    for t in texts:
        ids = raw.encode(t, add_special_tokens=True).ids
        ids_plain = raw.encode(t, add_special_tokens=False).ids
        tf_ids = tok(t, add_special_tokens=False)["input_ids"]
        assert tf_ids == ids_plain, f"transformers and tokenizers disagree on {t!r}"
        text_results.append(
            {
                "text": t,
                "ids": ids,
                "idsPlain": ids_plain,
                "decoded": raw.decode(ids, skip_special_tokens=False),
                "decodedSkip": raw.decode(ids, skip_special_tokens=True),
            }
        )
    # Random id sequences: byte tokens split characters every which way.
    rng = random.Random(99)
    byte_ids = [raw.token_to_id(c) for c in bytes_to_unicode().values()]
    byte_ids = [i for i in byte_ids if i is not None]
    special_ids = [raw.token_to_id(s) for s in SPECIALS if raw.token_to_id(s) is not None]
    id_results = []
    for _ in range(300):
        n = rng.randint(1, 40)
        seq = []
        for _ in range(n):
            r = rng.random()
            if r < 0.45:
                seq.append(rng.choice(byte_ids))
            elif r < 0.55:
                seq.append(rng.choice(special_ids))
            elif r < 0.58:
                # Past the vocabulary: the GGUF build's padding rows.
                seq.append(vocab + rng.randrange(3000))
            else:
                seq.append(rng.randrange(vocab))
        id_results.append(
            {
                "ids": seq,
                "decoded": raw.decode(seq, skip_special_tokens=False),
                "decodedSkip": raw.decode(seq, skip_special_tokens=True),
            }
        )
    chat_results = []
    for case in cases:
        try:
            text = tok.apply_chat_template(
                case["messages"], tools=case["tools"], tokenize=False, **case["kwargs"]
            )
            ids = tok.apply_chat_template(
                case["messages"], tools=case["tools"], tokenize=True, return_dict=False, **case["kwargs"]
            )
            if isinstance(ids, dict):
                ids = ids["input_ids"]
            chat_results.append(dict(case, text=text, ids=list(ids)))
        except Exception as e:  # noqa: BLE001
            chat_results.append(dict(case, error=str(e)))
    return {
        "name": name,
        "dir": os.path.abspath(directory),
        "vocabSize": vocab,
        "texts": text_results,
        "idSequences": id_results,
        "chats": chat_results,
    }


def bytes_to_unicode():
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, (chr(c) for c in cs)))


def main():
    out = sys.argv[1]
    models = [a.split("=", 1) for a in sys.argv[2:]]
    texts = corpus()
    cases = chat_cases()
    result = {"models": [model_truth(name, d, texts, cases) for name, d in models]}
    with open(out, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False)
    print(f"{len(texts)} texts, {len(cases)} chats, {len(models)} tokenizers -> {out}")


if __name__ == "__main__":
    main()
