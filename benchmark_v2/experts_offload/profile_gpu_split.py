"""Split the GPU 55 ms/token into pass-1 experts, pass-2 (masked_decode), and the rest."""
import sys, time, torch
from dense_quant import quantize_dense
from transformers import AutoModelForCausalLM, AutoTokenizer

repo, spec, cache_gib = sys.argv[1], sys.argv[2], float(sys.argv[3])
N = int(sys.argv[4]) if len(sys.argv) > 4 else 128
tok = AutoTokenizer.from_pretrained(repo)
import glob
from safetensors import safe_open
resident = 0
for f_ in glob.glob(f"{repo}/*.safetensors"):
    with safe_open(f_, "pt") as f:
        for k in f.keys():
            if ".experts." not in k and ".ngram_embedding." not in k:
                resident += torch.Size(f.get_slice(k).get_shape()).numel() * 2
model = AutoModelForCausalLM.from_pretrained(
    repo, dtype=torch.bfloat16, device_map={"": "mps", "experts": "disk", "ngram_embedding": "disk"},
    max_memory={"mps": int(resident + cache_gib * 2**30)})
quantize_dense(model, spec)

import transformers.distributed.offload as off

SPANS = []  # (kind, start_event, end_event), read after one sync
def ev():
    e = torch.mps.Event(enable_timing=True); e.record(); return e

_orig_pool = off._run_pool
def pool(*a, **k):
    s = ev(); out = _orig_pool(*a, **k); SPANS.append(("pass1", s, ev())); return out
off._run_pool = pool

# masked_decode is per-instance; wrap it on each streamed experts module.
for name, m in model.named_modules():
    if hasattr(m, "masked_decode") and getattr(m, "offload_store", None) is not None:
        orig = m.masked_decode
        def make(orig=orig):
            def inner(*a, **k):
                s = ev(); out = orig(*a, **k); SPANS.append(("pass2", s, ev())); return out
            return inner
        m.masked_decode = make()

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
        model.generate(ids(prompt), max_new_tokens=n, min_new_tokens=n, do_sample=False,
                       streamer=s, cache_implementation="static")
    return s

gen(16)
torch.mps.synchronize()
SPANS.clear()
s = gen(N)
torch.mps.synchronize()
steps = [b - a for a, b in zip(s.t[2:], s.t[3:])]
per = len(steps)
agg = {}
for kind, a, b in SPANS:
    agg[kind] = agg.get(kind, 0.0) + a.elapsed_time(b)
wall = sum(steps)
print(f"\ntokens {per}  wall {wall/per*1000:.2f} ms/token  ({per/wall:.2f} tok/s)")
tot = 0.0
for kind, v in sorted(agg.items(), key=lambda kv: -kv[1]):
    print(f"  {kind:8s} {v/per:8.2f} ms/token  {v/wall*100:5.1f}%  ({len([1 for k,_,_ in SPANS if k==kind])//per}x/token)")
    tot += v
print(f"  {'rest':8s} {(wall-tot)/per*1000:8.2f} ms/token  {(wall-tot)/wall*100:5.1f}%  (attention, router, python)")
