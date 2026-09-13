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
Multi-query associative recall (MQAR) for HELIX.

MQAR is the standard probe for the one property that fixed-state sequence models lose: pulling an exact
value back out of the distant past. Each sequence first *writes* a set of key/value bindings, then, far
enough away that no local window can still see them, *queries* a random subset of those keys. Accuracy is
measured only on the answer positions.

The number of bindings is the knob. A model whose entire memory is a fixed-size state has to compress all
of them into that state, so its accuracy falls away as the bindings outnumber what the state can hold.
Full attention keeps every token, so it stays flat -- and pays `O(N^2)` for the privilege. The question
this script asks is whether HELIX's index strand buys back the attention curve at the recurrent price.

Usage:
    python mqar.py                       # HELIX vs. the L+R ablation vs. full attention
    python mqar.py --pairs 16 64 --steps 400
"""

import argparse
import math
import time

import torch
from torch import nn

from transformers import HelixConfig, HelixForCausalLM, LlamaConfig, LlamaForCausalLM


def make_batch(batch_size, num_pairs, seq_len, num_keys, num_values, generator, device):
    """
    Build `k1 v1 ... kn vn <filler...> q1 v(q1) q2 v(q2) ...`.

    Returns the token ids and a label tensor that is `-100` everywhere except the answer slots. The answer
    is teacher-forced into the input as well, as in the usual MQAR setup, so the query phase has the same
    key/value shape as the write phase; supervision still only lands on the answer positions, each of which
    is predicted from the query key immediately before it.
    """
    key_lo, value_lo, filler = 1, 1 + num_keys, 1 + num_keys + num_values
    num_queries = num_pairs
    write_len, query_len = 2 * num_pairs, 2 * num_queries
    if write_len + query_len > seq_len:
        raise ValueError(f"{num_pairs} pairs need at least {write_len + query_len} tokens, got seq_len={seq_len}")

    input_ids = torch.full((batch_size, seq_len), filler, dtype=torch.long, device=device)
    labels = torch.full((batch_size, seq_len), -100, dtype=torch.long, device=device)

    for row in range(batch_size):
        keys = torch.randperm(num_keys, generator=generator, device=device)[:num_pairs] + key_lo
        values = torch.randint(0, num_values, (num_pairs,), generator=generator, device=device) + value_lo
        input_ids[row, 0:write_len:2] = keys
        input_ids[row, 1:write_len:2] = values
        # Query the bindings back in a shuffled order, right at the end of the sequence.
        order = torch.randperm(num_pairs, generator=generator, device=device)[:num_queries]
        start = seq_len - query_len
        input_ids[row, start::2] = keys[order]
        input_ids[row, start + 1 :: 2] = values[order]
        labels[row, start + 1 :: 2] = values[order]
    # The label at position t is predicted from position t - 1, which is where the query key sits.
    return input_ids, labels


def helix_config(vocab_size, seq_len, with_index):
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
        # 2 of the 4 layers get the index strand, or none of them for the ablation.
        index_layer_stride=2 if with_index else 10**6,
        landmark_dim=32,
        index_branching=4,
        index_beam_width=4,
        index_topk=8,
        num_recurrent_heads=2,
        recurrent_head_dim=32,
        recurrent_value_head_dim=32,
        recurrent_chunk_size=32,
        max_position_embeddings=seq_len,
    )


def llama_config(vocab_size, seq_len):
    return LlamaConfig(
        vocab_size=vocab_size,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=32,
        max_position_embeddings=seq_len,
    )


def evaluate(model, batches):
    correct = total = 0
    with torch.no_grad():
        for input_ids, labels in batches:
            logits = model(input_ids, use_cache=False).logits
            answered = labels[:, 1:] != -100
            predicted = logits[:, :-1].argmax(-1)[answered]
            correct += (predicted == labels[:, 1:][answered]).sum().item()
            total += int(answered.sum())
    return correct / max(total, 1)


def train_one(name, model, args, num_pairs, generator, device):
    model = model.to(device).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    warmup = max(1, args.steps // 20)

    def factor(step):
        if step < warmup:
            return (step + 1) / warmup
        progress = (step - warmup) / max(1, args.steps - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    schedule = torch.optim.lr_scheduler.LambdaLR(optimizer, factor)
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

    started = time.time()
    for step in range(args.steps):
        input_ids, labels = make_batch(
            args.batch_size, num_pairs, args.seq_len, args.num_keys, args.num_values, generator, device
        )
        logits = model(input_ids, use_cache=False).logits
        loss = loss_fn(logits[:, :-1].reshape(-1, logits.shape[-1]), labels[:, 1:].reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        schedule.step()
        optimizer.zero_grad(set_to_none=True)
        if args.verbose and (step + 1) % max(1, args.steps // 5) == 0:
            print(f"    {name}: step {step + 1}/{args.steps}  loss {loss.item():.4f}", flush=True)

    model.eval()
    evaluation = [
        make_batch(args.batch_size, num_pairs, args.seq_len, args.num_keys, args.num_values, generator, device)
        for _ in range(args.eval_batches)
    ]
    return evaluate(model, evaluation), time.time() - started


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", type=int, nargs="+", default=[8, 32, 64])
    parser.add_argument("--seq-len", type=int, default=320)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--eval-batches", type=int, default=4)
    parser.add_argument("--num-keys", type=int, default=128)
    parser.add_argument("--num-values", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--models", nargs="+", default=None, help="substrings selecting which models to run")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    device = torch.device(args.device)
    vocab_size = 1 + args.num_keys + args.num_values + 1

    builders = {
        "HELIX (L+R+I)": lambda: HelixForCausalLM(helix_config(vocab_size, args.seq_len, with_index=True)),
        "HELIX no index (L+R)": lambda: HelixForCausalLM(helix_config(vocab_size, args.seq_len, with_index=False)),
        "Llama (full attention)": lambda: LlamaForCausalLM(llama_config(vocab_size, args.seq_len)),
    }

    if args.models is not None:
        builders = {
            name: build
            for name, build in builders.items()
            if any(wanted.lower() in name.lower() for wanted in args.models)
        }

    helix = helix_config(vocab_size, args.seq_len, with_index=True)
    print(f"sequence length {args.seq_len}, widest HELIX local window {helix.local_span} tokens")
    print(f"bindings are written in the first tokens and queried in the last {2 * max(args.pairs)}")
    print(f"chance accuracy is {1 / args.num_values:.1%}\n")

    results = {}
    for name, build in builders.items():
        torch.manual_seed(args.seed)
        parameters = sum(p.numel() for p in build().parameters())
        for num_pairs in args.pairs:
            torch.manual_seed(args.seed)
            generator = torch.Generator(device=device).manual_seed(args.seed)
            accuracy, elapsed = train_one(name, build(), args, num_pairs, generator, device)
            results[(name, num_pairs)] = accuracy
            print(
                f"{name:24s} pairs={num_pairs:3d}  recall={accuracy:6.1%}  ({elapsed:5.1f}s, {parameters / 1e6:.2f}M params)",
                flush=True,
            )

    print("\nrecall accuracy")
    header = "".join(f"{p:>10d}" for p in args.pairs)
    print(f"{'model':24s}{header}")
    for name in builders:
        row = "".join(f"{results[(name, p)]:>9.1%} " for p in args.pairs)
        print(f"{name:24s}{row}")


if __name__ == "__main__":
    main()
