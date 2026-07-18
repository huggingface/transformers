# Copyright 2026 The SwissAI Initiative and The HuggingFace Inc. team. All rights reserved.
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
Cross-check a locally pruned Apertus 1.5 backbone against a reference pruned checkpoint on the hub:
compares the pruning-related config entries and the `lm_head.weight` tensor bit-exactly, downloading only
the reference config, index, and the single shard containing the head. Exits non-zero on any mismatch.

Example:
    python scripts/check_apertus1p5_pruning_crosscheck.py \
        --local /path/to/Apertus-1.5-8B-SFT-RL-DPO-SDPO-pruned \
        --reference_repo apertus-ai/Apertus-v1.5-8B-integration --revision refs/pr/1
"""

import argparse
import json
import os

import torch
from huggingface_hub import hf_hub_download
from safetensors import safe_open


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local", required=True, help="Locally pruned backbone checkpoint dir")
    parser.add_argument("--reference_repo", default="apertus-ai/Apertus-v1.5-8B-integration")
    parser.add_argument("--revision", default="refs/pr/1")
    args = parser.parse_args()

    failed = False

    hub_config = json.load(open(hf_hub_download(args.reference_repo, "config.json", revision=args.revision)))
    local_config = json.load(open(os.path.join(args.local, "config.json")))
    for key in ("vocab_size", "output_vocab_size", "tie_word_embeddings"):
        match = hub_config.get(key) == local_config.get(key)
        failed |= not match
        print(
            f"[{'PASS' if match else 'FAIL'}] config {key}: local {local_config.get(key)} "
            f"vs reference {hub_config.get(key)}"
        )

    hub_index = json.load(
        open(hf_hub_download(args.reference_repo, "model.safetensors.index.json", revision=args.revision))
    )
    hub_shard_name = hub_index["weight_map"]["lm_head.weight"]
    print(f"downloading reference shard containing lm_head: {hub_shard_name} ...")
    hub_shard = hf_hub_download(args.reference_repo, hub_shard_name, revision=args.revision)

    local_index = json.load(open(os.path.join(args.local, "model.safetensors.index.json")))
    local_shard = os.path.join(args.local, local_index["weight_map"]["lm_head.weight"])

    with safe_open(hub_shard, framework="pt") as file:
        hub_head = file.get_tensor("lm_head.weight")
    with safe_open(local_shard, framework="pt") as file:
        local_head = file.get_tensor("lm_head.weight")

    shape_match = hub_head.shape == local_head.shape and hub_head.dtype == local_head.dtype
    failed |= not shape_match
    print(
        f"[{'PASS' if shape_match else 'FAIL'}] lm_head shape/dtype: local "
        f"{tuple(local_head.shape)}/{local_head.dtype} vs reference {tuple(hub_head.shape)}/{hub_head.dtype}"
    )

    if shape_match:
        equal = torch.equal(hub_head, local_head)
        failed |= not equal
        print(f"[{'PASS' if equal else 'FAIL'}] lm_head bit-exact equality: {equal}")
        if not equal:
            differing_rows = int((hub_head != local_head).any(dim=1).sum())
            max_diff = (hub_head.float() - local_head.float()).abs().max().item()
            print(f"  differing rows: {differing_rows}, max abs diff: {max_diff}")

    if failed:
        raise SystemExit(1)
    print("\nPRUNING CROSS-CHECK PASSED")


if __name__ == "__main__":
    main()
