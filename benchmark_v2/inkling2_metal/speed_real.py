"""Inkling-2-Mini decode speed on MPS from the real checkpoint (experts in their NVFP4 / MXFP8 banks), `generate` with
a static cache and torch.compile.   PYTHONPATH=src:benchmark_v2/inkling2_metal python .../speed_real.py [new_tokens]"""

import sys
import time

import inductor_mps_fixes
import torch

from transformers import AutoModelForImageTextToText, AutoTokenizer, CompileConfig


REPO = "thinkingmachines/Inkling-2-Mini"
inductor_mps_fixes.apply()
new_tokens = int(sys.argv[1]) if len(sys.argv) > 1 else 128
model = AutoModelForImageTextToText.from_pretrained(REPO, device_map="mps", dtype=torch.bfloat16).eval()
tokenizer = AutoTokenizer.from_pretrained(REPO)
messages = [{"role": "user", "content": "Write a short story about a lighthouse keeper."}]
ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt", return_dict=True)
ids = ids["input_ids"].to("mps")
compile_config = CompileConfig()
compile_config._compile_all_devices = True
kwargs = {"cache_implementation": "static", "compile_config": compile_config, "do_sample": False}


class Stamps:
    def __init__(self):
        self.t = []

    def put(self, _):
        self.t.append(time.perf_counter())

    def end(self):
        pass


with torch.inference_mode():
    for phase in ("compile", "warm"):
        start = time.perf_counter()
        model.generate(ids, max_new_tokens=16, min_new_tokens=16, **kwargs)
        print(f"{phase}: {time.perf_counter() - start:.1f}s", flush=True)
    stamps = Stamps()
    out = model.generate(ids, max_new_tokens=new_tokens, min_new_tokens=new_tokens, streamer=stamps, **kwargs)
steps = sorted(b - a for a, b in zip(stamps.t[2:], stamps.t[3:]))
mean = sum(steps) / len(steps)
print(f"Inkling-2-Mini (real, native fp4 experts), {new_tokens} tokens: median {1 / steps[len(steps) // 2]:.1f} tok/s, "
      f"mean {1 / mean:.1f} tok/s ({mean * 1e3:.2f} ms/token)", flush=True)  # fmt: skip
print(repr(tokenizer.decode(out[0, ids.shape[1] :])[:300]), flush=True)
