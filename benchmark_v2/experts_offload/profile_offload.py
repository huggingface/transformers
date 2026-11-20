# /// script
# requires-python = ">=3.10"
# dependencies = ["torch", "numpy", "accelerate", "kernels", "triton", "huggingface_hub[hf_xet]"]
# ///
"""Where does a streamed-experts decode token go? Host phases (store tracing), IO (prefetch batches), GPU (profiler).

hf jobs uv run --flavor a100-large ... profile_offload.py [model] [cache fraction] [attn] [use_kernels]
"""

import glob
import os
import statistics
import sys
import time
from collections import defaultdict

import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open

from transformers import AutoModelForCausalLM, AutoTokenizer


model_id = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen3-30B-A3B"
fraction = float(sys.argv[2]) if len(sys.argv) > 2 else 0.25
attn = sys.argv[3] if len(sys.argv) > 3 else "sdpa"
use_kernels = len(sys.argv) > 4 and sys.argv[4] == "kernels"
device = "cuda" if torch.cuda.is_available() else "mps"
sync = torch.cuda.synchronize if device == "cuda" else torch.mps.synchronize
print(f"{model_id} cache={fraction} attn={attn} use_kernels={use_kernels} on {device}", flush=True)

path = snapshot_download(model_id, allow_patterns=["*.json", "*.safetensors", "*.txt", "*.jinja", "tokenizer*"])
sizes = {"BF16": 2, "F16": 2, "F32": 4, "U32": 4, "I32": 4, "F8_E4M3": 1}
expert_bytes = rest_bytes = 0
for file in glob.glob(f"{path}/*.safetensors"):
    with safe_open(file, "pt") as f:
        for key in f.keys():
            s = f.get_slice(key)
            n = torch.Size(s.get_shape()).numel() * sizes.get(s.get_dtype(), 2)
            expert_bytes, rest_bytes = (
                (expert_bytes + n, rest_bytes) if ".experts." in key else (expert_bytes, rest_bytes + n)
            )
for file in glob.glob(f"{path}/*.safetensors"):  # cold reads
    fd = os.open(file, os.O_RDONLY)
    if hasattr(os, "posix_fadvise"):
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    os.close(fd)

key = 0 if device == "cuda" else "mps"
model = AutoModelForCausalLM.from_pretrained(
    path,
    dtype=torch.bfloat16,
    device_map={"": device, "experts": "disk"},
    max_memory={key: int(rest_bytes + fraction * expert_bytes)},
    attn_implementation=attn,
)
if use_kernels:
    model.set_use_kernels(True)
store = model._experts_offload
tok = AutoTokenizer.from_pretrained(path)
prompts = [
    "Write a short story about a lighthouse keeper who finds a message in a bottle.",
    "Explain the causes of the French Revolution to a high school student.",
]


def ids(prompt):
    return tok.apply_chat_template(
        [{"role": "user", "content": prompt}], add_generation_prompt=True, return_tensors="pt", return_dict=True,
        enable_thinking=False,
    ).input_ids.to(device)  # fmt: skip


class Stamps:
    def __init__(self):
        self.t = []

    def put(self, _):
        self.t.append(time.perf_counter())

    def end(self):
        pass


with torch.inference_mode():
    model.generate(ids(prompts[0]), max_new_tokens=64, do_sample=False)  # warm the cache and kernels
    sync()
    store.reset_stats()
    store.trace = []
    stamps = Stamps()
    model.generate(ids(prompts[1]), max_new_tokens=64, min_new_tokens=64, do_sample=False, streamer=stamps)
    sync()
    trace, store.trace = store.trace, None
    stats = store.stats()
    # GPU view of a few decode tokens
    activities = [torch.profiler.ProfilerActivity.CPU]
    if device == "cuda":
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    with torch.profiler.profile(activities=activities) as prof:
        model.generate(ids(prompts[1]), max_new_tokens=4, min_new_tokens=4, do_sample=False)
        sync()

# ---------- analysis ----------
token_times = [b - a for a, b in zip(stamps.t[1:], stamps.t[2:])]  # decode steps (first put is the prompt)
decode_start, decode_end = stamps.t[1], stamps.t[-1]
n = len(token_times)
ms = 1e3
print(
    f"\n=== decode: {n} tokens, median {statistics.median(token_times) * ms:.1f} ms/token -> {1 / statistics.median(token_times):.2f} tok/s"
)
print("stats per token:", {k: round(v / n, 3) for k, v in stats.items() if isinstance(v, (int, float))})

events = [e for e in trace if decode_start <= e[0] <= decode_end]
main = events[0][1] if events else None
phase_time = defaultdict(float)
open_ = {}
for t, thread, phase, layer, info in events:
    if thread != main:
        continue
    for name in ("sync", "wait"):
        if phase == f"{name}_start":
            open_[name] = t
        elif phase == f"{name}_end" and name in open_:
            phase_time[name] += t - open_.pop(name)
misses = [info for t, th, phase, layer, info in events if phase == "misses"]
layers_per_token = len(misses) / n
total = decode_end - decode_start
print(f"\n=== where a decode token goes (main thread), {total / n * ms:.1f} ms/token")
print(f"  routing sync (device drains pass 1 + earlier work) {phase_time['sync'] / n * ms:7.1f} ms")
print(f"  waiting for expert reads                             {phase_time['wait'] / n * ms:7.1f} ms")
print(
    f"  everything else (launches, Python, pass 2, attn...)  {(total - phase_time['sync'] - phase_time['wait']) / n * ms:7.1f} ms"
)
print(
    f"  misses/token {sum(m['misses'] for m in misses) / n:.1f}, routed experts/token {sum(m['experts'] for m in misses) / n:.1f}, "
    f"misses already read ahead {sum(m['covered'] for m in misses) / max(sum(m['misses'] for m in misses), 1):.0%}, "
    f"layers with a miss {sum(m['misses'] > 0 for m in misses) / max(len(misses), 1):.0%} of {layers_per_token:.0f}/token"
)

# IO threads: batches, loader start latency, per-expert take time, effective bandwidth
batches = defaultdict(list)
for t, thread, phase, layer, info in events:
    if phase in ("batch_start", "loaders_started", "take_start", "take_end"):
        batches[thread].append((t, phase, layer, info))
loader_setup, takes, bytes_, io_busy = [], [], 0, 0.0
for thread, evs in batches.items():
    start = None
    for t, phase, layer, info in evs:
        if phase == "batch_start":
            start = t
        elif phase == "loaders_started" and start is not None:
            loader_setup.append(t - start)
        elif phase == "take_start":
            take_start = t
        elif phase == "take_end":
            takes.append(t - take_start)
            bytes_ += info["bytes"]
unused = sum(info.get("unused_guesses", 0) for t, th, phase, layer, info in events if phase == "layer_end")
print(f"\n=== IO ({len(batches)} io threads used)")
if loader_setup:
    print(
        f"  prefetch batches {len(loader_setup)} ({len(loader_setup) / n:.1f}/token), loader start {statistics.median(loader_setup) * ms:.2f} ms median"
    )
if takes:
    print(
        f"  experts read {len(takes)} ({len(takes) / n:.1f}/token), take {statistics.median(takes) * ms:.2f} ms median / {max(takes) * ms:.1f} max"
    )
    print(f"  bytes read {bytes_ / n / 1e6:.0f} MB/token -> {bytes_ / total / 1e9:.2f} GB/s over decode wall time")
print(f"  unused lookahead guesses cancelled {unused / n:.1f}/token")

# One token, layer by layer: ASCII timeline of the main thread
mid = token_times.index(statistics.median(token_times)) + 1
t0, t1 = stamps.t[mid], stamps.t[mid + 1]
print(f"\n=== one median token, main thread ({(t1 - t0) * ms:.1f} ms): layer | sync | misses | wait")
rows = []
current = {}
for t, thread, phase, layer, info in trace:
    if not (t0 <= t <= t1) or thread != main:
        continue
    if phase == "sync_start":
        current = {"layer": layer, "sync": t}
    elif phase == "sync_end":
        current["sync"] = t - current["sync"]
    elif phase == "misses":
        current["misses"] = info["misses"]
        current["wait"] = 0.0
    elif phase == "wait_start":
        current["w0"] = t
    elif phase == "wait_end":
        current["wait"] += t - current.pop("w0")
    elif phase == "layer_end" and "misses" in current:
        rows.append(current)
for row in rows:
    bar = "#" * int(row["wait"] * ms) + "s" * int(row["sync"] * ms)
    print(
        f"  {row['layer'].split('.')[-3]:>4} | {row['sync'] * ms:5.2f} | {row['misses']:3d} | {row['wait'] * ms:6.2f} {bar}"
    )

print("\n=== GPU / ops (4 decode tokens)")
print(
    prof.key_averages().table(
        sort_by="self_cuda_time_total" if device == "cuda" else "self_cpu_time_total", row_limit=15
    )
)
