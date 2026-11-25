"""Sweep `_MPSHostRouter.MAX_AHEAD` in one process: how far the host may run ahead of the device.

Reloading the 30B checkpoint per value costs ~8 minutes, so this loads once, warms the compiled step up, then walks
`MA_VALUES` ascending. Results print (and flush) per value, so a deadlock at the top of the range still leaves the
lower values on disk. `MAX_AHEAD` is set on the live router, which is the same field `route()` spins on.

The point of the sweep: `route()` blocks the main thread once the host is `MAX_AHEAD` layers ahead, and `wait()`
commits a command buffer every `COMMIT_EVERY`. Metal stops accepting buffers at ~64 in flight, so too deep a queue
deadlocks the flush; too shallow one under-overlaps the miss reads with the GPU. We want the knee, not a guess.

Run through the bounded runner so the same loader patches apply:
  MA_VALUES=1,2,4,8,12,16,24,32 MA_TOKENS=16 \
    python .bench/mem_bounded_run.py max_ahead_sweep.py ~/qwen38-metal/final-ckpt 4g32c 16
"""

import os
import sys
import time

import torch
from dense_quant import quantize_dense

from transformers import AutoModelForCausalLM, AutoTokenizer, CompileConfig

repo = sys.argv[1] if len(sys.argv) > 1 else "ArthurZ/Qwen3-30B-A3B-metal-experts-4bit"
spec = sys.argv[2] if len(sys.argv) > 2 else "4g32c"
cache_gib = float(sys.argv[3]) if len(sys.argv) > 3 else 0
N = int(os.environ.get("MA_TOKENS", "16"))
values = [int(v) for v in os.environ.get("MA_VALUES", "1,2,4,8,12,16,24,32,48").split(",") if v.strip()]

tok = AutoTokenizer.from_pretrained(repo)
if cache_gib:
    import glob

    from huggingface_hub import snapshot_download
    from safetensors import safe_open

    path = repo if os.path.isdir(repo) else snapshot_download(repo)
    resident = 0
    for file in glob.glob(f"{path}/*.safetensors"):
        with safe_open(file, "pt") as f:
            for key in f.keys():
                if ".experts." not in key and ".ngram_embedding." not in key:
                    meta = f.get_tensor_meta(key) if hasattr(f, "get_tensor_meta") else None
                    shape = meta.shape if meta else f.get_slice(key).get_shape()
                    resident += torch.Size(shape).numel() * 2
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
compile_config._compile_all_devices = True
compiled = {"cache_implementation": "static", "compile_config": compile_config}

prompt = "Write a short story about a lighthouse keeper who finds a message in a bottle."


def ids(p):
    return tok.apply_chat_template(
        [{"role": "user", "content": p}], add_generation_prompt=True, return_tensors="pt", return_dict=True,
        enable_thinking=False,
    ).input_ids.to("mps")  # fmt: skip


stores = []
seen = set()
for name, module in model.named_modules():
    store = getattr(module, "offload_store", None)
    if store is not None and id(store) not in seen:
        seen.add(id(store))
        stores.append((name, store))
if not stores:
    raise SystemExit("no offload store found: is this a streamed checkpoint?")


def routers():
    out = []
    for _, store in stores:
        r = getattr(store, "mps_router", None)
        if r is not None:
            out.append(r)
    return out


class Stamps:
    """Per-token timestamps (and the token ids) from the generate streamer, so prefill is excluded."""

    def __init__(self):
        self.t = []
        self.tokens = []

    def put(self, token):
        self.t.append(time.perf_counter())
        try:
            self.tokens.append(int(token))
        except (TypeError, ValueError):
            pass

    def end(self):
        pass


def generate(n):
    stamps = Stamps()
    with torch.inference_mode():
        model.generate(ids(prompt), max_new_tokens=n, min_new_tokens=n, do_sample=False, streamer=stamps, **compiled)
    steps = [b - a for a, b in zip(stamps.t[2:], stamps.t[3:])]
    return steps, stamps.tokens


def snapshot():
    out = {}
    for _, store in stores:
        for k, v in store.stats().items():
            out[k] = out.get(k, 0) + v
    for r in routers():
        for k in ("spin_seconds", "work_seconds", "ahead_reads", "jobs_done"):
            out[k] = out.get(k, 0) + getattr(r, k)
        # Steady-state lag: how many routings the host was ahead of the device when the window closed.
        out["lag"] = max(out.get("lag", 0), r.counter - r.ext.signaled(r.event))
    return out


def reset():
    for _, store in stores:
        store.reset_stats()
    for r in routers():
        r.spin_seconds = r.work_seconds = 0.0
        r.ahead_reads = r.jobs_done = 0


generate(32)  # compile warmup
torch.mps.synchronize()
print(f"routers: {len(routers())}  commit_every: {[r.COMMIT_EVERY for r in routers()]}", flush=True)

baseline_tokens = None
rows = []
for ma in values:
    for r in routers():
        r.MAX_AHEAD = ma
    reset()
    print(f"\n=== MAX_AHEAD={ma} ===", flush=True)
    try:
        steps, tokens = generate(N)
    except Exception as error:  # noqa: BLE001 - a deadlock surfaces here as a Metal timeout
        print(f"  FAILED: {type(error).__name__}: {str(error)[:300]}", flush=True)
        rows.append((ma, None))
        break
    if not steps:
        print("  no steps measured", flush=True)
        rows.append((ma, None))
        continue
    wall = sum(steps)
    per_tok = wall / len(steps)
    snap = snapshot()
    n = len(steps)
    host = snap["spin_seconds"] + snap["work_seconds"]
    same = "n/a" if baseline_tokens is None else ("same" if tokens == baseline_tokens else "DIFFERENT")
    if baseline_tokens is None:
        baseline_tokens = tokens
    hits, misses = snap.get("hits", 0), snap.get("misses", 0)
    print(
        f"  wall {per_tok * 1000:7.2f} ms/token  {1 / per_tok:6.2f} tok/s   "
        f"spin {snap['spin_seconds'] / n * 1000:6.2f}  work {snap['work_seconds'] / n * 1000:6.2f}  "
        f"lag {snap.get('lag', 0):3d}   "
        f"hit {hits / max(hits + misses, 1) * 100:5.2f}%  ahead_reads {snap.get('ahead_reads', 0):6d}  "
        f"tokens {same}",
        flush=True,
    )
    rows.append((ma, per_tok))

print("\n=== summary ===", flush=True)
best = None
for ma, per_tok in rows:
    if per_tok is None:
        print(f"MAX_AHEAD={ma:3d}  failed", flush=True)
        continue
    print(f"MAX_AHEAD={ma:3d}  {per_tok * 1000:7.2f} ms/token  {1 / per_tok:6.2f} tok/s", flush=True)
    if best is None or per_tok < best[1]:
        best = (ma, per_tok)
if best:
    print(f"\nbest: MAX_AHEAD={best[0]}  {1 / best[1]:.2f} tok/s", flush=True)
