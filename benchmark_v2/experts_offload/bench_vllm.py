# /// script
# requires-python = ">=3.10"
# dependencies = ["vllm"]
# ///
"""vLLM batch-1 latency for Qwen3-30B-A3B on this GPU (`vllm bench latency`): all on GPU, then CPU-offloaded weights."""

import subprocess
import sys


model = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen3-30B-A3B"
common = ["vllm", "bench", "latency", "--model", model, "--batch-size", "1", "--input-len", "64"]
common += ["--output-len", "128", "--num-iters-warmup", "2", "--num-iters", "5", "--max-model-len", "4096"]
runs = {
    "vllm full (CUDA graphs)": [],
    "vllm cpu-offload 40GB": ["--cpu-offload-gb", "40"],
}
for name, extra in runs.items():
    print(f"=== {name}: {' '.join(common + extra)}", flush=True)
    subprocess.run(common + extra, check=False)
