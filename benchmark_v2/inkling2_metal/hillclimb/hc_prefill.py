"""Prefill tok/s of Inkling-2-Mini on MPS (4-bit dense plan, same load as hc_bench): time to the first token at several
prompt lengths, median of `samples` after a warm-up, one ledger line.

    PYTHONPATH=src:benchmark_v2/inkling2_metal python benchmark_v2/inkling2_metal/hillclimb/hc_prefill.py [name]
"""
import json, os, statistics, sys, time

from hc_bench import gpu_busy, on_ac

HERE = os.path.dirname(os.path.abspath(__file__))
if not on_ac() or gpu_busy() > 20:
    sys.exit("refusing to run: battery or busy GPU")
import torch
import inductor_mps_fixes
from transformers import AutoModelForImageTextToText, MetalConfig

inductor_mps_fixes.apply()
metal = MetalConfig(bits=4, group_size=64)
plan = {"model.visual.*": None, "*_message_gate": None, "lm_head": metal, "*": metal}
model = AutoModelForImageTextToText.from_pretrained("thinkingmachines/Inkling-2-Mini", device_map="mps",
                                                    dtype=torch.bfloat16, quantization_config=plan).eval()  # fmt: skip
for module in model.modules():
    if type(module).__name__.endswith("RelativePositionBias"):
        module.proj_t = module.proj.detach().t().to(torch.bfloat16).contiguous()
results = {}
with torch.inference_mode():
    for i, n in enumerate((512, 2048)):
        ids = torch.randint(1000, 50000, (1, n), device="mps")
        model.generate(ids, max_new_tokens=1, do_sample=False)  # warm
        ts = []
        for _ in range(4):
            torch.mps.synchronize(); t = time.perf_counter()
            model.generate(ids, max_new_tokens=1, do_sample=False)
            torch.mps.synchronize(); ts.append(time.perf_counter() - t)
        results[n] = n / statistics.median(ts)
        print(f"[{i + 1}/2] prefill {n:5d} tokens: {results[n]:7.0f} tok/s (ttft {statistics.median(ts) * 1e3:.0f} ms)", flush=True)
record = {"config": sys.argv[1] if len(sys.argv) > 1 else "prefill", "time": time.strftime("%Y-%m-%d %H:%M:%S"),
          "prefill_tok_s": results, "ac": on_ac()}  # fmt: skip
open(os.path.join(HERE, "ledger.jsonl"), "a").write(json.dumps(record) + "\n")
