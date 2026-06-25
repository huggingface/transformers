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
Reproduce the two empirical contracts of μP on a tiny Llama:

* `--mode coord`: per-module mean of `|activation|` is approximately width-invariant under μP and fans out under
  standard parametrization (SP, the default training regime that μP replaces) — the coordinate check from
  "Tensor Programs V".
* `--mode lr-transfer`: the loss-vs-learning-rate curves overlap across widths under μP and shift under SP, which
  is what makes hyperparameters tuned at the base width transferable to wider models (μTransfer).

Usage:
    python examples/pytorch/mup_demo.py --mode coord
    python examples/pytorch/mup_demo.py --mode lr-transfer
"""

import argparse

import torch

from transformers import LlamaConfig, LlamaForCausalLM
from transformers.integrations.mup import build_mup_param_groups, coord_check


def make_model(width: int, mup: bool, base_width: int, vocab: int = 128):
    config = LlamaConfig(
        vocab_size=vocab,
        hidden_size=width,
        intermediate_size=4 * width,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=64,
        tie_word_embeddings=True,
        mup=mup,
        mup_base_width=base_width if mup else None,
        attn_implementation="eager",
    )
    torch.manual_seed(0)
    return LlamaForCausalLM(config)


def _coord_spread(records, layer, widths):
    """Return max/min ratio of mean(|activation|) across widths for one layer (1.0 = perfectly flat)."""
    vals = [records[w][layer][-1] for w in widths if layer in records[w] and records[w][layer]]
    vals = [v for v in vals if v > 1e-9]
    if len(vals) < 2:
        return None
    return max(vals) / min(vals)


def summarize_coord(records, label):
    widths = sorted(records.keys())
    layers = sorted(set().union(*[records[w].keys() for w in widths]))
    header = f"{'layer':45s}" + "".join(f"  w={w:<6d}" for w in widths) + "    spread"
    print(f"\n{label}")
    print(header)
    print("-" * len(header))
    spreads = []
    for layer in layers:
        row = f"{layer:45s}"
        for w in widths:
            vals = records[w].get(layer)
            row += f"  {vals[-1]:8.4f}" if vals else f"  {'-':>8s}"
        spread = _coord_spread(records, layer, widths)
        if spread is not None:
            spreads.append(spread)
            row += f"   {spread:6.2f}x"
        print(row)
    mean_spread = sum(spreads) / len(spreads) if spreads else float("nan")
    print(f"\n  → mean spread: {mean_spread:.2f}x")
    return mean_spread


def run_coord(widths, batch, n_steps, lr):
    print("=" * 72)
    print("COORDINATE CHECK — per-module mean(|activation|) after a few steps")
    print("=" * 72)
    print("What to look for:")
    print(f"  • Under μP each row should be roughly constant across w={widths[0]}…{widths[-1]} (spread ≈ 1).")
    print("  • Under SP (standard parametrization, the default) values fan out with width.")

    rec_mup = coord_check(
        lambda w: make_model(w, mup=True, base_width=widths[0]),
        widths=widths,
        batch=batch,
        n_steps=n_steps,
        lr=lr,
    )
    mup_spread = summarize_coord(rec_mup, f"=== μP (base_width={widths[0]}) ===")

    rec_sp = coord_check(
        lambda w: make_model(w, mup=False, base_width=widths[0]),
        widths=widths,
        batch=batch,
        n_steps=n_steps,
        lr=lr,
    )
    sp_spread = summarize_coord(rec_sp, "=== SP ===")

    print("\n" + "=" * 72)
    print(f"SUMMARY  μP mean spread = {mup_spread:.2f}x   |   SP mean spread = {sp_spread:.2f}x")
    width_range = widths[-1] / widths[0]
    if mup_spread < sp_spread:
        print(f"μP keeps activations roughly width-invariant; SP fans out (≈ width range = {width_range:.0f}x).")
    print("=" * 72)


@torch.no_grad()
def loss_after_steps(width, mup, base_width, batch, n_steps, lr):
    model = make_model(width, mup=mup, base_width=base_width)
    model.train()
    if mup:
        groups = build_mup_param_groups(model, lr=lr)
    else:
        groups = [{"params": list(model.parameters()), "lr": lr}]
    opt = torch.optim.AdamW(groups)
    final = float("nan")
    with torch.enable_grad():
        for _ in range(n_steps):
            opt.zero_grad()
            out = model(**batch)
            out.loss.backward()
            opt.step()
            final = out.loss.item()
    return final


def _print_lr_table(label, lrs, widths, losses):
    """`losses` is shape (len(lrs), len(widths)). Mark column-wise minima with an arrow."""
    print(f"\n{label}")
    header = f"{'lr':>10s}" + "".join(f"  w={w:<8d}" for w in widths)
    print(header)
    print("-" * len(header))
    best_per_width = [min(range(len(lrs)), key=lambda i: losses[i][j]) for j in range(len(widths))]
    for i, lr in enumerate(lrs):
        row = f"{lr:10.1e}"
        for j in range(len(widths)):
            mark = " ←" if best_per_width[j] == i else "  "
            row += f"  {losses[i][j]:6.3f}{mark}"
        print(row)
    optimal_lrs = [lrs[best_per_width[j]] for j in range(len(widths))]
    return optimal_lrs


def run_lr_transfer(widths, lrs, batch, n_steps):
    base_width = widths[0]
    print("=" * 72)
    print("LEARNING-RATE TRANSFER — final loss after a few steps")
    print("=" * 72)
    print("What to look for:")
    print("  • Under μP the optimal LR (← marker) should be the SAME row at every width.")
    print("  • Under SP (standard parametrization, the default) the optimal LR shifts as width grows.")

    results = {}
    for label, mup in (("μP", True), ("SP", False)):
        losses = [[loss_after_steps(w, mup, base_width, batch, n_steps, lr) for w in widths] for lr in lrs]
        title = f"=== {label} (base_width={base_width}) ===" if mup else f"=== {label} ==="
        results[label] = _print_lr_table(title, lrs, widths, losses)

    print("\n" + "=" * 72)
    print("SUMMARY  optimal LR per width:")
    fmt = lambda xs: ", ".join(f"{x:.0e}" for x in xs)  # noqa: E731
    print(f"  μP : {fmt(results['μP'])}   {'✓ width-invariant' if len(set(results['μP'])) == 1 else '✗ shifts'}")
    print(f"  SP : {fmt(results['SP'])}   {'✓ width-invariant' if len(set(results['SP'])) == 1 else '✗ shifts'}")
    print("=" * 72)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("coord", "lr-transfer"), default="coord")
    parser.add_argument("--widths", type=int, nargs="+", default=[64, 128, 256])
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lrs", type=float, nargs="+", default=[1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 1e-1])
    args = parser.parse_args()

    torch.manual_seed(0)
    input_ids = torch.randint(0, 128, (4, 32))
    batch = {"input_ids": input_ids, "labels": input_ids.clone()}

    if args.mode == "coord":
        run_coord(args.widths, batch, args.steps, args.lr)
    else:
        run_lr_transfer(args.widths, args.lrs, batch, args.steps)


if __name__ == "__main__":
    main()
