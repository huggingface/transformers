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
"""
Measure how HELIX and full attention scale with context length.

Two things are reported per model and context length:

* **attention pair count** -- how many query/key products the model actually forms. This is exact and
  hardware-independent, counted by instrumenting the attention call itself, so it is the honest statement
  of the asymptotics. Full attention grows as `N^2`; HELIX grows as `N` times a constant that depends only
  on `local_blocks`, `index_topk` and `block_size`.
* **wall-clock forward time**, fitted as `t ~ N^alpha`. HELIX is a reference implementation in plain
  PyTorch with no fused kernels, so its *absolute* time is not competitive with a fused attention kernel;
  the exponent is the point, not the constant.

Two other modes:

* `--cache` prefills the context and weighs everything the model must then hold to keep generating. This is
  the number that decides what long-context serving costs. Full attention keeps every key and value on
  every layer; HELIX keeps a fixed-size recurrent and convolution state everywhere, a window-capped
  key/value cache on its `"helix_local"` layers, and the full history only on the layers whose index can
  actually reach into it.
* `--memory` runs each measurement in a fresh subprocess and reports the transient memory of the forward
  pass on top of the weights. Note that a memory-efficient attention kernel already keeps *full* attention
  linear in memory -- what is quadratic there is compute, and what grows without bound at decode time is
  the cache, which is what `--cache` measures.

Usage:
    python scaling.py --lengths 512 1024 2048 4096
    python scaling.py --lengths 8192 16384 32768 --cache
"""

import argparse
import math
import resource
import subprocess
import sys
import time

import torch

from transformers import HelixConfig, HelixForCausalLM, LlamaConfig, LlamaForCausalLM
from transformers.models.helix.modeling_helix import HelixBraid


COMMON = {
    "vocab_size": 256,
    "hidden_size": 256,
    "intermediate_size": 512,
    "num_hidden_layers": 4,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 64,
    "max_position_embeddings": 1 << 20,
}


def build_helix():
    return HelixForCausalLM(
        HelixConfig(
            **COMMON,
            block_size=64,
            local_blocks=3,
            num_window_scales=2,
            index_layer_stride=2,
            landmark_dim=64,
            index_branching=8,
            index_beam_width=4,
            index_topk=8,
            num_recurrent_heads=2,
            recurrent_head_dim=64,
            recurrent_value_head_dim=64,
            recurrent_chunk_size=64,
        )
    )


def build_llama():
    return LlamaForCausalLM(LlamaConfig(**COMMON))


def count_helix_pairs(model, length):
    """Instrument the block attention to count the query/key products HELIX actually forms."""
    total = 0

    def wrap(braid):
        original = braid._blocked_attention

        def counted(query_blocks, keys, *args, **kwargs):
            nonlocal total
            batch, heads, num_query_blocks, block_size, _ = query_blocks.shape
            total += batch * heads * num_query_blocks * block_size * keys.shape[3]
            return original(query_blocks, keys, *args, **kwargs)

        braid._blocked_attention = counted

    for module in model.modules():
        if isinstance(module, HelixBraid):
            wrap(module)
    with torch.no_grad():
        model(torch.zeros(1, length, dtype=torch.long), use_cache=False)
    return total


def count_llama_pairs(model, length):
    layers = model.config.num_hidden_layers
    return layers * model.config.num_attention_heads * length * (length + 1) // 2


def timed(model, length, repeats):
    input_ids = torch.zeros(1, length, dtype=torch.long)
    with torch.no_grad():
        model(input_ids, use_cache=False)  # warm up
        started = time.perf_counter()
        for _ in range(repeats):
            model(input_ids, use_cache=False)
    return (time.perf_counter() - started) / repeats


def _proc_memory_mib(field):
    """Read VmRSS (current) or VmHWM (peak) from /proc, in MiB. Falls back to ru_maxrss elsewhere."""
    try:
        with open("/proc/self/status") as handle:
            for line in handle:
                if line.startswith(field):
                    return int(line.split()[1]) / 1024
    except OSError:
        pass
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / 1024 if sys.platform != "darwin" else peak / (1024 * 1024)


def probe(name, length):
    """
    Run one forward in this (fresh) process and print how much memory it added on top of the weights.

    The baseline is resident memory *after* the model is built, so the number is the transient cost of the
    forward pass itself -- the gathers, masks and score matrices -- which is what the tiling is meant to cap.
    """
    model = (build_helix() if name == "HELIX" else build_llama()).eval()
    input_ids = torch.zeros(1, length, dtype=torch.long)
    baseline = _proc_memory_mib("VmHWM")
    with torch.no_grad():
        model(input_ids, use_cache=False)
    print(f"{max(_proc_memory_mib('VmHWM') - baseline, 0.0):.1f}")


def measure_memory(name, length):
    """Peak memory of a forward, measured in a subprocess so the numbers do not accumulate."""
    result = subprocess.run(
        [sys.executable, __file__, "--probe", name, str(length)], capture_output=True, text=True, check=True
    )
    return float(result.stdout.strip().splitlines()[-1])


def cache_bytes(model, length):
    """Total bytes the model must keep after prefilling `length` tokens in order to keep generating."""
    outputs = model(torch.zeros(1, length, dtype=torch.long), use_cache=True)
    total = 0
    for layer in outputs.past_key_values.layers:
        tensors = [getattr(layer, "keys", None), getattr(layer, "values", None)]
        for attribute in ("conv_states", "recurrent_states"):
            tensors += list(getattr(layer, attribute, {}).values())
        tensors += list(getattr(layer, "landmark_levels", []))
        tensors.append(getattr(layer, "route_hidden", None))
        # `leaf_landmarks` is level 0 of `landmark_levels`, so it is already counted.
        total += sum(t.numel() * t.element_size() for t in tensors if t is not None)
    return total / (1024 * 1024)


def slope(lengths, values):
    """Least-squares exponent of `value ~ length**alpha`."""
    xs = [math.log(v) for v in lengths]
    ys = [math.log(v) for v in values]
    mean_x, mean_y = sum(xs) / len(xs), sum(ys) / len(ys)
    numerator = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    denominator = sum((x - mean_x) ** 2 for x in xs)
    return numerator / denominator


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lengths", type=int, nargs="+", default=[512, 1024, 2048, 4096])
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--skip-timing", action="store_true")
    parser.add_argument("--memory", action="store_true", help="measure peak memory instead of time")
    parser.add_argument("--cache", action="store_true", help="measure the decode cache instead of time")
    parser.add_argument("--probe", nargs=2, metavar=("MODEL", "LENGTH"), help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.probe:
        probe(args.probe[0], int(args.probe[1]))
        return

    torch.manual_seed(0)
    models = {"HELIX": (build_helix(), count_helix_pairs), "Llama": (build_llama(), count_llama_pairs)}
    for model, _ in models.values():
        model.eval()

    pairs = {name: [] for name in models}
    costs = {name: [] for name in models}
    unit = "cache MiB" if args.cache else "peak MiB" if args.memory else "seconds"
    print(f"{'length':>8}  {'model':6}  {'attention pairs':>18}  {'per token':>10}  {unit:>9}")
    for length in args.lengths:
        for name, (model, counter) in models.items():
            count = counter(model, length)
            pairs[name].append(count)
            if args.cache:
                with torch.no_grad():
                    cost = cache_bytes(model, length)
            elif args.memory:
                cost = measure_memory(name, length)
            elif args.skip_timing:
                cost = float("nan")
            else:
                cost = timed(model, length, args.repeats)
            costs[name].append(cost)
            print(f"{length:>8}  {name:6}  {count:>18,}  {count / length:>10,.0f}  {cost:>9.1f}", flush=True)

    print("\nfitted exponent alpha in  quantity ~ N^alpha")
    for name in models:
        line = f"  {name:6}  attention pairs: {slope(args.lengths, pairs[name]):.2f}"
        if (args.cache or args.memory or not args.skip_timing) and all(cost > 0 for cost in costs[name]):
            line += f"   {unit}: {slope(args.lengths, costs[name]):.2f}"
        print(line)


if __name__ == "__main__":
    main()
