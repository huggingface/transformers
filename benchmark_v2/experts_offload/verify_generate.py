"""Public-API verification on MPS: `generate` with a static cache + compile (all devices), greedy, batch 1.
Correctness vs eager, decode tok/s on 4 prompts x 128 tokens, sustained tok/s over tokens 512-1024."""

import sys
import time

import torch
from dense_quant import quantize_dense

from transformers import AutoModelForCausalLM, AutoTokenizer, CompileConfig


repo = sys.argv[1] if len(sys.argv) > 1 else "ArthurZ/Qwen3-30B-A3B-metal-experts-4bit"
spec = sys.argv[2] if len(sys.argv) > 2 else "4g32c"
cache_gib = float(sys.argv[3]) if len(sys.argv) > 3 else 0  # > 0: stream experts (and n-gram rows) through this cache
tok = AutoTokenizer.from_pretrained(repo)
if cache_gib:
    import glob
    import os

    from huggingface_hub import snapshot_download
    from safetensors import safe_open

    path = repo if os.path.isdir(repo) else snapshot_download(repo)
    resident = 0  # bytes of everything not streamed
    for file in glob.glob(f"{path}/*.safetensors"):
        with safe_open(file, "pt") as f:
            for key in f.keys():
                if ".experts." not in key and ".ngram_embedding." not in key:
                    meta = f.get_tensor_meta(key) if hasattr(f, "get_tensor_meta") else None
                    shape = meta.shape if meta else f.get_slice(key).get_shape()
                    resident += torch.Size(shape).numel() * 2  # bf16 once loaded
    model = AutoModelForCausalLM.from_pretrained(
        repo,
        dtype=torch.bfloat16,
        device_map={"": "mps", "experts": "disk", "ngram_embedding": "disk"},
        max_memory={"mps": int(resident + cache_gib * 2**30)},
    )
else:
    model = AutoModelForCausalLM.from_pretrained(repo, dtype=torch.bfloat16, device_map="mps")
quantize_dense(model, spec)
compile_config = CompileConfig()
compile_config._compile_all_devices = True  # generate only auto-compiles on cuda / xpu / ... otherwise
import os as _os2
compiled = {"cache_implementation": "static"}
if _os2.environ.get("NO_COMPILE") != "1":
    compiled["compile_config"] = compile_config
prompts = [
    "Write a short story about a lighthouse keeper who finds a message in a bottle.",
    "Write a Python function that parses an ISO 8601 date string without using datetime, with tests.",
    "Explain the causes of the French Revolution to a high school student.",
    "Translate into French and explain the grammar: 'The children would have gone if it had not rained.'",
]


def ids(prompt):
    return tok.apply_chat_template(
        [{"role": "user", "content": prompt}], add_generation_prompt=True, return_tensors="pt", return_dict=True,
        enable_thinking=False,
    ).input_ids.to("mps")  # fmt: skip


class Stamps:
    def __init__(self):
        self.t = []

    def put(self, _):
        self.t.append(time.perf_counter())

    def end(self):
        pass


def generate(prompt, n, **kwargs):
    stamps = Stamps()
    with torch.inference_mode():
        out = model.generate(
            ids(prompt), max_new_tokens=n, min_new_tokens=n, do_sample=False, streamer=stamps, **kwargs
        )
    steps = [b - a for a, b in zip(stamps.t[2:], stamps.t[3:])]  # decode steps
    return out[0, -n:].tolist(), steps


eager_tokens, _ = generate(prompts[1], 128)
generate(prompts[0], 32, **compiled)  # compile
compiled_tokens, _ = generate(prompts[1], 128, **compiled)
import json, os as _os
if _os.environ.get("DUMP_TOKENS"):
    json.dump({"eager": eager_tokens, "compiled": compiled_tokens}, open(_os.environ["DUMP_TOKENS"], "w"))
same = next((i for i, (a, b) in enumerate(zip(eager_tokens, compiled_tokens)) if a != b), 128)
print(f"correctness: compiled greedy tokens == eager for {same}/128", flush=True)
rates = []
for p in prompts:
    _, steps = generate(p, 128, **compiled)
    rates.append(len(steps) / sum(steps))
    print(f"  decode {rates[-1]:.2f} tok/s (mean of 127 steps): {p[:40]!r}", flush=True)
_, steps = generate(prompts[0], 1024, **compiled)
tail = steps[len(steps) // 2 :]
print(f"decode mean over 4 prompts: {sum(rates) / 4:.2f} tok/s", flush=True)
print(
    f"sustained: {len(tail) / sum(tail):.2f} tok/s (mean over tokens {len(steps) - len(tail)}-{len(steps)} of 1024)",
    flush=True,
)
