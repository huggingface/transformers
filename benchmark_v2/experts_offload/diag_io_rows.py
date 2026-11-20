# /// script
# requires-python = ">=3.10"
# dependencies = ["torch", "numpy", "huggingface_hub[hf_xet]"]
# ///
"""Expert rows of stacked banks (transformers format) to the GPU: safetensors multi-slice prefetch vs per-expert
loaders vs our own preadv into a pinned slab. Batches of random (layer, expert) rows, cold and warm page cache."""

import glob
import json
import os
import random
import struct
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open


repo = sys.argv[1] if len(sys.argv) > 1 else "ArthurZ/Qwen3-30B-A3B-native"
path = snapshot_download(repo, allow_patterns=["*.json", "*.safetensors"])
index = json.load(open(f"{path}/model.safetensors.index.json", encoding="utf-8"))["weight_map"]
banks = sorted(k for k in index if k.endswith(("experts.gate_up_proj", "experts.down_proj")))
handles = {f: safe_open(f"{path}/{f}", "pt") for f in {index[k] for k in banks}}
meta = {k: handles[index[k]].get_tensor_meta(k) for k in banks}
num_experts = meta[banks[0]].shape[0]
layers = sorted({k.rsplit(".mlp", 1)[0] for k in banks})
print(torch.cuda.get_device_name(), f"{len(layers)} layers x {num_experts} experts, {len(banks)} banks", flush=True)


def header_size(file):
    with open(f"{path}/{file}", "rb") as f:
        return 8 + struct.unpack("<Q", f.read(8))[0]


data_start = {f: header_size(f) for f in handles}


def drop():
    for file in glob.glob(f"{path}/*.safetensors"):
        fd = os.open(file, os.O_RDONLY)
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        os.close(fd)


def batch(n, seed):
    rnd = random.Random(seed)
    layer = rnd.choice(layers)  # one layer's misses: rows of the same banks
    experts = rnd.sample(range(num_experts), n)
    return [(f"{layer}.mlp.experts.{p}", e) for p in ("gate_up_proj", "down_proj") for e in experts]


def multi_slice(rows):
    plans = {}
    for key, e in rows:
        plans.setdefault(index[key], {}).setdefault(key, []).append(slice(e, e + 1))
    out = []
    for file, plan in plans.items():
        with handles[file].prefetch(plan, device="cuda:0", threads=16) as loader:
            out += [t for _, t in loader]
    return out


def per_expert(rows):
    out = []
    loaders = [handles[index[key]].prefetch({key: slice(e, e + 1)}, device="cuda:0", threads=16) for key, e in rows]
    for loader, (key, _) in zip(loaders, rows):
        out.append(loader.take(key))
        loader.close()
    return out


pool = ThreadPoolExecutor(16)
stream = torch.cuda.Stream()
fds = {f: os.open(f"{path}/{f}", os.O_RDONLY) for f in handles}
slabs = []


def pread_rows(rows):
    def read(item):
        key, e = item
        m, file = meta[key], index[key]
        row_bytes = (m.data_offsets[1] - m.data_offsets[0]) // m.shape[0]
        buf = torch.empty(row_bytes, dtype=torch.uint8, pin_memory=True)
        view, done = memoryview(buf.numpy()), 0
        offset = data_start[file] + m.data_offsets[0] + e * row_bytes
        while done < row_bytes:
            done += os.preadv(fds[file], [view[done:]], offset + done)
        with torch.cuda.stream(stream):
            return buf.to("cuda", non_blocking=True)

    out = list(pool.map(read, rows))
    stream.synchronize()
    return out


for n in (1, 2, 4, 8, 128):  # batch-1 decode misses per layer; 128 = a long prompt reading a whole layer
    for name, fn in (
        ("A multi-slice prefetch", multi_slice),
        ("B per-expert loaders", per_expert),
        ("C preadv pinned", pread_rows),
    ):
        for cold in (True, False):
            rows = batch(n, seed=n)  # same rows for every method
            if cold:
                drop()
            torch.cuda.synchronize()
            start = time.perf_counter()
            out = fn(rows)
            torch.cuda.synchronize()
            seconds = time.perf_counter() - start
            nbytes = sum(t.numel() * t.element_size() for t in out)
            print(
                f"n={n:3d} {name:24s} {'cold' if cold else 'warm'} {seconds * 1e3:8.1f} ms  {nbytes / seconds / 1e9:5.2f} GB/s  "
                f"{seconds / n * 1e3:6.2f} ms/expert",
                flush=True,
            )
