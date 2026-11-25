"""Who forces a GPU sync during decode? Patches Tensor.item/__int__/__float__/__bool__ to dump callers."""
import sys, time, traceback, collections, torch
from dense_quant import quantize_dense
from transformers import AutoModelForCausalLM, AutoTokenizer, CompileConfig

repo = sys.argv[1]; spec = sys.argv[2]; cache_gib = float(sys.argv[3]); N = int(sys.argv[4]) if len(sys.argv) > 4 else 8
tok = AutoTokenizer.from_pretrained(repo)
import glob
from safetensors import safe_open
resident = 0
for file in glob.glob(f"{repo}/*.safetensors"):
    with safe_open(file, "pt") as f:
        for key in f.keys():
            if ".experts." not in key and ".ngram_embedding." not in key:
                resident += torch.Size(f.get_slice(key).get_shape()).numel() * 2
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

HITS = collections.Counter()
SAMPLES = {}
ARMED = [False]

def wrap(name, orig):
    def inner(self, *a, **k):
        if ARMED[0]:
            st = traceback.extract_stack()[:-1]
            frames = [f"{f.filename.split('/')[-1]}:{f.lineno} {f.name}" for f in st[-7:]]
            key = " <- ".join(reversed(frames))
            HITS[key] += 1
            SAMPLES.setdefault(key, st)
        return orig(self, *a, **k)
    return inner

torch.Tensor.item = wrap("item", torch.Tensor.item)
torch.Tensor.__int__ = wrap("int", torch.Tensor.__int__)
torch.Tensor.__float__ = wrap("float", torch.Tensor.__float__)
torch.Tensor.__bool__ = wrap("bool", torch.Tensor.__bool__)

def gen(n):
    s = S()
    with torch.inference_mode():
        model.generate(ids(prompt), max_new_tokens=n, min_new_tokens=n, do_sample=False, streamer=s, **compiled)
    return s

gen(32); torch.mps.synchronize()
ARMED[0] = True
gen(N); torch.mps.synchronize()
ARMED[0] = False
print(f"\n=== {N} tokens: {sum(HITS.values())} scalar syncs ===", flush=True)
for key, n in HITS.most_common(6):
    print(f"\n--- x{n} ---")
    for fr in SAMPLES[key][-7:]:
        print(f"  {fr.filename.split('/')[-1]}:{fr.lineno} {fr.name}")
