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

Usage:
    python scaling.py --lengths 512 1024 2048 4096
"""

import argparse
import math
import time

import torch

from transformers import HelixConfig, HelixForCausalLM, LlamaConfig, LlamaForCausalLM
from transformers.models.helix.modeling_helix import HelixBraid


COMMON = {
    "vocab_size": 256,
    "hidden_size": 256,
    "intermediate_size": 512,
    "num_hidden_layers": 2,
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
            index_layer_stride=1,
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
    args = parser.parse_args()

    torch.manual_seed(0)
    models = {"HELIX": (build_helix(), count_helix_pairs), "Llama": (build_llama(), count_llama_pairs)}
    for model, _ in models.values():
        model.eval()

    pairs = {name: [] for name in models}
    times = {name: [] for name in models}
    print(f"{'length':>8}  {'model':6}  {'attention pairs':>18}  {'per token':>10}  {'seconds':>9}")
    for length in args.lengths:
        for name, (model, counter) in models.items():
            count = counter(model, length)
            pairs[name].append(count)
            elapsed = float("nan") if args.skip_timing else timed(model, length, args.repeats)
            times[name].append(elapsed)
            print(f"{length:>8}  {name:6}  {count:>18,}  {count / length:>10,.0f}  {elapsed:>9.3f}", flush=True)

    print("\nfitted exponent alpha in  quantity ~ N^alpha")
    for name in models:
        line = f"  {name:6}  attention pairs: {slope(args.lengths, pairs[name]):.2f}"
        if not args.skip_timing:
            line += f"   wall clock: {slope(args.lengths, times[name]):.2f}"
        print(line)


if __name__ == "__main__":
    main()
