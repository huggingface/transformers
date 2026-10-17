# Copyright 2026 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Measure device-to-host (D2H) synchronizations in the Qwen VL vision encoders.

Diagnostic script for PR #47703 / issue #47649 ("D2H sync in Qwen VL vision encoder
causes ~250ms stalls").

For a Qwen-VL model (qwen2_vl / qwen2_5_vl / qwen3_vl) it reports, on a GPU runner:

  * wall-clock forward latency (mean / median / p90, ms) -- the user-visible symptom
  * number of `.item()` / `int()` scalar reads on device tensors (``_local_scalar_dense``)
  * number of CUDA ``cudaMemcpyAsync`` copies (device <-> host traffic)
  * number and total bytes of *device-to-host* copies (from the Chrome trace, when ``--trace``
    is set)

How to use
~~~~~~~~~~
Run the script twice -- once on ``main`` (baseline) and once on the PR branch -- and compare:

.. code-block:: bash

    git checkout main
    python benchmark/d2h_syncs.py --model Qwen/Qwen3-VL-2B --iters 30 --trace /tmp/main.json
    git checkout <pr-branch>
    python benchmark/d2h_syncs.py --model Qwen/Qwen3-VL-2B --iters 30 --trace /tmp/pr.json

The ``--scope vision`` mode isolates the vision encoder (the hot path from the vLLM PRs
vllm-project/vllm#14377 / #14684, where D2H syncs dropped from 222 to 0 and QPS rose 7-15%).
The ``--scope full`` mode exercises the whole multimodal forward, including
``get_rope_index`` which runs a Python loop per batch item.
"""

import argparse
import json
import statistics
import time
from pathlib import Path

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="Qwen/Qwen3-VL-2B", help="Model id to benchmark (a Qwen VL model).")
    parser.add_argument(
        "--scope",
        choices=["full", "vision"],
        default="full",
        help="'full' = multimodal LLM forward; 'vision' = vision encoder only.",
    )
    parser.add_argument("--iters", type=int, default=20, help="Number of timed forward passes.")
    parser.add_argument("--warmup", type=int, default=3, help="Number of warmup forward passes.")
    parser.add_argument("--profile-iters", type=int, default=5, help="Number of iterations profiled for sync counts.")
    parser.add_argument("--batch", type=int, default=1, help="Number of image+text pairs per batch.")
    parser.add_argument("--image-size", type=int, default=1024, help="Side length of the synthetic image (px).")
    parser.add_argument("--text-len", type=int, default=64, help="Approximate number of text tokens per sample.")
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="float16")
    parser.add_argument(
        "--attn",
        choices=["eager", "sdpa", "flash_attention_2"],
        default=None,
        help="Attention implementation; defaults to the model default (sdpa).",
    )
    parser.add_argument(
        "--trace",
        type=Path,
        default=None,
        help="Export a Chrome trace to this path and report device-to-host copy stats from it.",
    )
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def import_torch():
    import torch

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required. Run this script on a GPU runner (e.g. GitHub Actions 'gpu' image).")
    return torch


def count_from_profiler(prof, torch):
    """Return (scalar_syncs, memcpy_total) from a finished profiler run."""
    scalar_syncs = 0
    memcpy_total = 0
    for evt in prof.events():
        name = evt.name or ""
        if "local_scalar" in name:
            scalar_syncs += evt.count
        elif "memcpy" in name.lower():
            memcpy_total += evt.count
    return scalar_syncs, memcpy_total


def parse_trace_d2h(trace_path):
    """Parse a Chrome trace exported by torch.profiler and count device-to-host memcpys.

    Torch versions encode the direction either as ``"d2h"/"h2d"/"d2d"`` or as
    ``"DtoH"/"HtoD"/"DtoD"`` inside the event ``args``; both are handled here.
    """
    d2h_count = 0
    d2h_bytes = 0
    memcpy_count = 0
    with open(trace_path, "r") as f:
        for line in f:
            if "memcpy" not in line.lower():
                continue
            try:
                evt = json.loads(line.rstrip(",\n"))
            except json.JSONDecodeError:
                continue
            name = str(evt.get("name", "")).lower()
            if "memcpy" not in name:
                continue
            memcpy_count += 1
            args = evt.get("args", {})
            direction = str(args.get("direction", "")).lower()
            if direction in ("d2h", "dtoh"):
                d2h_count += 1
                d2h_bytes += int(args.get("bytes", args.get("cudaMemcpyAsync", 0)) or 0)
    return memcpy_count, d2h_count, d2h_bytes


def build_inputs(processor, args, torch, device, dtype):
    try:
        from PIL import Image
    except ImportError:
        raise SystemExit("Pillow is required: `pip install pillow` (or `pip install transformers[image]`).")

    rng = np.random.default_rng(args.seed)
    images = [
        Image.fromarray(rng.integers(0, 255, (args.image_size, args.image_size, 3), dtype=np.uint8))
        for _ in range(args.batch)
    ]
    # Each sample sees one image; the text is padded so the batch mixes text and vision tokens.
    texts = ["Describe this image in detail. " * max(1, args.text_len // 5) for _ in range(args.batch)]

    inputs = processor(text=texts, images=images, return_tensors="pt", padding=True)
    inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
    if dtype != torch.float32:
        for key in ("pixel_values", "pixel_values_videos"):
            if key in inputs and isinstance(inputs[key], torch.Tensor):
                inputs[key] = inputs[key].to(dtype)
    return inputs


def main():
    args = parse_args()
    torch = import_torch()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if args.dtype == "float16":
        dtype = torch.float16
    elif args.dtype == "bfloat16":
        dtype = torch.bfloat16
    else:
        dtype = torch.float32

    from transformers import AutoModelForImageTextToText, AutoProcessor

    device = torch.device("cuda")

    print(f"[setup] loading processor + model {args.model} ({args.dtype})")
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    kwargs = {}
    if args.attn is not None:
        kwargs["attn_implementation"] = args.attn
    model = (
        AutoModelForImageTextToText.from_pretrained(args.model, dtype=dtype, trust_remote_code=True, **kwargs)
        .to(device)
        .eval()
    )

    print(
        f"[setup] building inputs: batch={args.batch}, image={args.image_size}x{args.image_size}, text_len={args.text_len}"
    )
    inputs = build_inputs(processor, args, torch, device, dtype)

    if args.scope == "vision":

        def forward():
            return model.get_image_features(inputs["pixel_values"], inputs["image_grid_thw"])

        input_desc = "vision encoder"
    else:
        forward_kwargs = {
            k: v
            for k, v in inputs.items()
            if k
            in (
                "input_ids",
                "attention_mask",
                "pixel_values",
                "image_grid_thw",
                "pixel_values_videos",
                "video_grid_thw",
                "mm_token_type_ids",
                "position_ids",
            )
        }

        def forward():
            return model(**forward_kwargs)

        input_desc = "full multimodal forward"

    # Warmup
    for _ in range(args.warmup):
        forward()
    torch.cuda.synchronize()

    # Timed iterations (no profiler, so the numbers are not skewed by tracing)
    latencies_ms = []
    for _ in range(args.iters):
        torch.cuda.synchronize()
        start = time.perf_counter()
        forward()
        torch.cuda.synchronize()
        latencies_ms.append((time.perf_counter() - start) * 1e3)

    latencies_ms = sorted(latencies_ms)
    mean_ms = statistics.mean(latencies_ms)
    median_ms = latencies_ms[len(latencies_ms) // 2]
    p90_ms = latencies_ms[int(len(latencies_ms) * 0.90)]

    # Profiled iterations (sync counts). A small number of iterations keeps the trace size sane.
    from torch.profiler import ProfilerActivity, profile

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=False, with_stack=False
    ) as prof:
        for _ in range(args.profile_iters):
            forward()
        torch.cuda.synchronize()

    scalar_syncs, memcpy_total = count_from_profiler(prof, torch)

    print(f"\n=== {args.model} | {args.scope} ({input_desc}) | dtype={args.dtype} | attn={args.attn or 'default'} ===")
    print(f"latency  mean={mean_ms:.2f} ms  median={median_ms:.2f} ms  p90={p90_ms:.2f} ms  (iters={args.iters})")
    print(f"syncs    .item()/int() scalar reads = {scalar_syncs}  (per iter: {scalar_syncs / args.profile_iters:.1f})")
    print(
        f"syncs    cudaMemcpyAsync events      = {memcpy_total}  (per iter: {memcpy_total / args.profile_iters:.1f})"
    )

    if args.trace is not None:
        prof.export_chrome_trace(str(args.trace))
        memcpy_total_trace, d2h_count, d2h_bytes = parse_trace_d2h(args.trace)
        print(f"trace    {args.trace}  (memcpy events={memcpy_total_trace})")
        print(
            f"syncs    D2H copies (from trace)  = {d2h_count}  (per iter: {d2h_count / args.profile_iters:.1f}, "
            f"bytes/iter: {d2h_bytes / args.profile_iters:,.0f})"
        )


if __name__ == "__main__":
    main()
