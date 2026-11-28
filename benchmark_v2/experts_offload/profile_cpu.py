"""CPU-side profile of one decode: the events-on wall is 78% main-thread CPU issue time."""
import sys, time, torch
from dense_quant import quantize_dense
from transformers import AutoModelForCausalLM, AutoTokenizer, CompileConfig

repo = sys.argv[1]; spec = sys.argv[2]; cache_gib = float(sys.argv[3])
N = int(sys.argv[4]) if len(sys.argv) > 4 else 48
tok = AutoTokenizer.from_pretrained(repo)
import glob, os
from safetensors import safe_open
path = repo
resident = 0
for file in glob.glob(f"{path}/*.safetensors"):
    with safe_open(file, "pt") as f:
        for key in f.keys():
            if ".experts." not in key and ".ngram_embedding." not in key:
                shape = f.get_slice(key).get_shape()
                resident += torch.Size(shape).numel() * 2
model = AutoModelForCausalLM.from_pretrained(
    repo, dtype=torch.bfloat16, device_map={"": "mps", "experts": "disk", "ngram_embedding": "disk"},
    max_memory={"mps": int(resident + cache_gib * 2**30)})
quantize_dense(model, spec)
cc = CompileConfig(); cc._compile_all_devices = True
compiled = {"cache_implementation": "static", "compile_config": cc}
prompt = "Write a short story about a lighthouse keeper who finds a message in a bottle."
ids = lambda p: tok.apply_chat_template([{"role": "user", "content": p}], add_generation_prompt=True,
    return_tensors="pt", return_dict=True, enable_thinking=False).input_ids.to("mps")
class S:
    def __init__(self): self.t = []
    def put(self, _): self.t.append(time.perf_counter())
    def end(self): pass
def gen(n):
    s = S()
    with torch.inference_mode():
        model.generate(ids(prompt), max_new_tokens=n, min_new_tokens=n, do_sample=False, streamer=s, **compiled)
    return s
gen(32); torch.mps.synchronize()
from torch.profiler import profile, ProfilerActivity
import os
with_stack = os.environ.get("WITH_STACK", "0") == "1"
with profile(activities=[ProfilerActivity.CPU], with_stack=with_stack) as prof:
    gen(N)
    torch.mps.synchronize()
if not with_stack:
    print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=30))
else:
    seen = {}
    for evt in prof.events():
        if evt.name != "aten::_local_scalar_dense":
            continue
        st = getattr(evt, "stack", None)
        if not st:
            continue
        key = "\n".join(line.strip() for line in st[-6:])
        seen[key] = seen.get(key, 0) + 1
    for key, n in sorted(seen.items(), key=lambda kv: -kv[1]):
        print(f"\n=== x{n} ===")
        print(key)
