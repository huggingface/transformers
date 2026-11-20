# /// script
# requires-python = ">=3.10"
# dependencies = ["torch", "numpy", "huggingface_hub[hf_xet]", "safetensors==0.9.0rc1"]
# ///
"""Expert read throughput on this machine: 128 random Qwen3-30B-A3B experts (gate, up, down; 9.4 MB) to the GPU."""

import glob
import json
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor

import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open


path = snapshot_download("Qwen/Qwen3-30B-A3B", allow_patterns=["*.json", "*.safetensors"])
index = json.load(open(f"{path}/model.safetensors.index.json", encoding="utf-8"))["weight_map"]
random.seed(0)
experts = [(random.randrange(48), random.randrange(128)) for _ in range(128)]
keys = [
    f"model.layers.{layer}.mlp.experts.{expert}.{proj}.weight"
    for layer, expert in experts
    for proj in ("gate_proj", "up_proj", "down_proj")
]
total = sum(9437184 for _ in experts)


def drop():
    for file in glob.glob(f"{path}/*.safetensors"):
        fd = os.open(file, os.O_RDONLY)
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        os.close(fd)


def handles(backend, device="cpu"):
    return {f: safe_open(f"{path}/{f}", "pt", device=device, backend=backend) for f in set(index.values())}


def run(name, fn, threads=4):
    for cold in (True, False):
        if cold:
            drop()
        torch.cuda.synchronize()
        start = time.perf_counter()
        fn(threads)
        torch.cuda.synchronize()
        seconds = time.perf_counter() - start
        print(f"{name:36s} {'cold' if cold else 'warm'}  {seconds * 1e3:7.0f} ms  {total / seconds / 1e9:5.2f} GB/s"
              f"  {seconds / len(experts) * 1e3:5.2f} ms/expert", flush=True)


stream = torch.cuda.Stream()


def via_pinned(backend):
    h = handles(backend)

    def read(key):
        t = h[index[key]].get_slice(key)[...]
        with torch.cuda.stream(stream):
            return t.pin_memory().to("cuda", non_blocking=True)

    def fn(threads):
        with ThreadPoolExecutor(threads) as pool:
            list(pool.map(read, keys))

    return fn


def direct(threads):
    h = handles("pread", "cuda")
    with ThreadPoolExecutor(threads) as pool:
        list(pool.map(lambda k: h[index[k]].get_slice(k)[...], keys))


def prefetch(threads):
    h = handles("pread")
    by_file = {}
    for key in keys:
        by_file.setdefault(index[key], {})[key] = None
    loaders = [(h[f].prefetch(plan, device="cuda:0", threads=threads), plan) for f, plan in by_file.items()]
    for loader, plan in loaders:
        for key in plan:
            loader.take(key)
    for loader, _ in loaders:
        loader.close()


print(torch.cuda.get_device_name(), os.cpu_count(), "cpus", flush=True)
run("(a) mmap -> pin -> H2D, 4 threads", via_pinned("mmap"))
run("(a) mmap -> pin -> H2D, 16 threads", via_pinned("mmap"), 16)
run("(b) pread -> pin -> H2D, 4 threads", via_pinned("pread"))
run("(b) pread -> pin -> H2D, 16 threads", via_pinned("pread"), 16)
run("(c) pread -> cuda, 4 threads", direct)
run("(c) pread -> cuda, 16 threads", direct, 16)
run("(d) prefetch 8 threads", prefetch, 8)
run("(d) prefetch 16 threads", prefetch, 16)
