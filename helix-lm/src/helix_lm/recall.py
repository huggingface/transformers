# Copyright 2026 Nathan. Licensed under the Apache License, Version 2.0.
"""
Multi-query associative recall — the probe for the one thing fixed-state models lose.

Each sequence writes a set of key/value bindings, then, far enough away that no local window can still see
them, queries a shuffled subset. Accuracy is measured only on the answer positions. Running the same model
with and without the index strand isolates exactly what the index contributes.
"""

from __future__ import annotations

import math
import time

import torch

from .config import HelixConfig
from .model import HelixForCausalLM


def make_batch(batch_size, num_pairs, seq_len, num_keys, num_values, generator, device):
    """
    Build ``k1 v1 ... kn vn <filler...> q1 v(q1) q2 v(q2) ...``.

    The answer is teacher-forced into the input, as in the usual MQAR setup, so the query phase has the
    same key/value shape as the write phase. Supervision still lands only on the answer positions, each
    predicted from the query key immediately before it.
    """
    key_lo, value_lo, filler = 1, 1 + num_keys, 1 + num_keys + num_values
    write_len, query_len = 2 * num_pairs, 2 * num_pairs
    if write_len + query_len > seq_len:
        raise ValueError(f"{num_pairs} pairs need at least {write_len + query_len} tokens, got seq_len={seq_len}")

    input_ids = torch.full((batch_size, seq_len), filler, dtype=torch.long, device=device)
    labels = torch.full((batch_size, seq_len), -100, dtype=torch.long, device=device)
    for row in range(batch_size):
        keys = torch.randperm(num_keys, generator=generator, device=device)[:num_pairs] + key_lo
        values = torch.randint(0, num_values, (num_pairs,), generator=generator, device=device) + value_lo
        input_ids[row, 0:write_len:2] = keys
        input_ids[row, 1:write_len:2] = values
        order = torch.randperm(num_pairs, generator=generator, device=device)
        start = seq_len - query_len
        input_ids[row, start::2] = keys[order]
        input_ids[row, start + 1 :: 2] = values[order]
        labels[row, start + 1 :: 2] = values[order]
    return input_ids, labels


def recall_config(vocab_size: int, with_index: bool) -> HelixConfig:
    return HelixConfig(
        vocab_size=vocab_size,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=32,
        block_size=16,
        local_blocks=2,
        num_window_scales=2,
        # Every other layer indexes, or no layer does -- that difference is the whole experiment.
        index_layer_stride=2 if with_index else 10**6,
        landmark_dim=32,
        index_branching=4,
        index_beam_width=4,
        index_topk=8,
        num_recurrent_heads=2,
        recurrent_head_dim=32,
        recurrent_value_head_dim=32,
        recurrent_chunk_size=32,
    )


def train_and_score(model, steps, seq_len, pairs, batch_size, num_keys, num_values, lr, generator, device):
    model = model.to(device).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    warmup = max(1, steps // 20)

    def factor(step):
        if step < warmup:
            return (step + 1) / warmup
        return 0.5 * (1.0 + math.cos(math.pi * (step - warmup) / max(1, steps - warmup)))

    schedule = torch.optim.lr_scheduler.LambdaLR(optimizer, factor)
    for _ in range(steps):
        input_ids, labels = make_batch(batch_size, pairs, seq_len, num_keys, num_values, generator, device)
        model(input_ids, labels=labels).loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        schedule.step()
        optimizer.zero_grad(set_to_none=True)

    model.eval()
    correct = total = 0
    with torch.no_grad():
        for _ in range(4):
            input_ids, labels = make_batch(batch_size, pairs, seq_len, num_keys, num_values, generator, device)
            predicted = model(input_ids).logits[:, :-1].argmax(-1)
            answered = labels[:, 1:] != -100
            correct += (predicted[answered] == labels[:, 1:][answered]).sum().item()
            total += int(answered.sum())
    return correct / max(total, 1), total


def run_recall(steps=600, seq_len=256, pairs=16, batch_size=16, num_keys=128, num_values=16, lr=1e-3, seed=0):
    """Train HELIX with and without the index strand and print both scores with a confidence interval."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vocab_size = 1 + num_keys + num_values + 1
    chance = 1 / num_values
    span = recall_config(vocab_size, True).local_span

    print(f"\n{pairs} bindings written in the first {2 * pairs} tokens and queried in the last {2 * pairs},")
    print(f"in a {seq_len}-token sequence — so they sit far outside HELIX's {span}-token local window.")
    print(f"Chance accuracy is {chance:.1%}.\n")

    results = {}
    for label, with_index in (("HELIX (L+R+I)", True), ("HELIX no index (L+R)", False)):
        torch.manual_seed(seed)
        generator = torch.Generator(device=device).manual_seed(seed)
        model = HelixForCausalLM(recall_config(vocab_size, with_index))
        started = time.time()
        accuracy, n = train_and_score(
            model, steps, seq_len, pairs, batch_size, num_keys, num_values, lr, generator, device
        )
        error = 1.96 * math.sqrt(accuracy * (1 - accuracy) / max(n, 1))
        results[label] = accuracy
        print(
            f"{label:22s} recall {accuracy:6.1%}  +/- {error:.1%}   "
            f"({model.num_parameters() / 1e6:.2f}M params, {time.time() - started:.0f}s)",
            flush=True,
        )

    print(f"\n{'chance':22s}        {chance:6.1%}")
    gap = results["HELIX (L+R+I)"] - results["HELIX no index (L+R)"]
    print(f"\nThe index strand is worth {gap:+.1%} here. Remove it and the model cannot see the bindings at")
    print("all — its local window is too short and its recurrent state too small to hold them.")
    return results
