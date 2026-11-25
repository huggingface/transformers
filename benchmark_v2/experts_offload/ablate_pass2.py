"""How much of the 51 ms/token is the second expert pass? Zero `masked_decode` and re-time.
Diagnostic only: output is wrong on the ablated arm, so no correctness claim is made from it."""
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

layers = [m for m in model.modules()
          if hasattr(m, "masked_decode") and getattr(m, "offload_store", None) is not None]
print(f"layers with masked_decode: {len(layers)}", flush=True)
real = [m.masked_decode for m in layers]

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
    steps = [b - a for a, b in zip(s.t[2:], s.t[3:])]
    return sum(steps) / len(steps)

gen(16); torch.mps.synchronize()
a = gen(N); torch.mps.synchronize()
print(f"baseline (pass 2 on)   {a*1000:7.2f} ms/token  {1/a:6.2f} tok/s", flush=True)

for m in layers:
    m.masked_decode = lambda h, s, w, **k: torch.zeros_like(h)
b = gen(N); torch.mps.synchronize()
print(f"pass 2 zeroed          {b*1000:7.2f} ms/token  {1/b:6.2f} tok/s", flush=True)
print(f"pass-2 share           {(a-b)*1000:7.2f} ms/token  ({(a-b)/a*100:.1f}% of wall)", flush=True)

for m, f in zip(layers, real):
    m.masked_decode = f
c = gen(N); torch.mps.synchronize()
print(f"restored (pass 2 on)   {c*1000:7.2f} ms/token  {1/c:6.2f} tok/s  <- same-process control", flush=True)
