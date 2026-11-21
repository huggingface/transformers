"""Inkling-2-Mini decode on MPS loaded through the quantization plan: the checkpoint's NVFP4 / MXFP8 experts as stored,
plus `{"*": MetalConfig(bits=4)}` quantizing the bf16 rest on the fly.   PYTHONPATH=src:benchmark_v2/inkling2_metal ..."""

import sys
import time

import inductor_mps_fixes
import torch

from transformers import AutoModelForImageTextToText, AutoTokenizer, CompileConfig, MetalConfig


inductor_mps_fixes.apply()
REPO = "thinkingmachines/Inkling-2-Mini"
plan = {"model.visual.*": None, "*_message_gate": None, "lm_head": MetalConfig(bits=4, group_size=64), "*": MetalConfig(bits=4, group_size=64)}
start = time.perf_counter()
model = AutoModelForImageTextToText.from_pretrained(REPO, device_map="mps", dtype=torch.bfloat16, quantization_config=plan)
kinds = {}
for name, module in model.named_modules():
    if name.startswith("model.language_model") and not list(module.children()):
        kinds[type(module).__name__] = kinds.get(type(module).__name__, 0) + 1
print(f"loaded in {time.perf_counter() - start:.0f}s, {torch.mps.current_allocated_memory() / 2**30:.1f} GiB, lm_head {type(model.lm_head).__name__}, {kinds}", flush=True)
for module in model.modules():
    if type(module).__name__.endswith("RelativePositionBias"):
        module.proj_t = module.proj.detach().t().to(torch.bfloat16).contiguous()
tokenizer = AutoTokenizer.from_pretrained(REPO)
messages = [{"role": "user", "content": "Write a short story about a lighthouse keeper."}]
ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt", return_dict=True)["input_ids"].to("mps")
compile_config = CompileConfig()
compile_config._compile_all_devices = True
kwargs = {"cache_implementation": "static", "compile_config": compile_config, "do_sample": False}
stamps = []


class Stamps:
    def put(self, _):
        stamps.append(time.perf_counter())

    def end(self):
        pass


new_tokens = int(next((a for a in sys.argv[1:] if a.isdigit()), 128))
with torch.inference_mode():
    model.generate(ids, max_new_tokens=16, min_new_tokens=16, **kwargs)
    model.generate(ids, max_new_tokens=16, min_new_tokens=16, **kwargs)
    out = model.generate(ids, max_new_tokens=new_tokens, min_new_tokens=new_tokens, streamer=Stamps(), **kwargs)
steps = sorted(b - a for a, b in zip(stamps[2:], stamps[3:]))
mean = sum(steps) / len(steps)
print(f"plan (stored fp4 experts + Metal 4-bit rest): median {1 / steps[len(steps) // 2]:.1f} tok/s, mean {1 / mean:.1f} tok/s", flush=True)
print(repr(tokenizer.decode(out[0, ids.shape[1] :])[:200]), flush=True)
