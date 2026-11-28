# Copyright 2026 Dustin Loring and the HuggingFace Inc. team. All rights reserved.
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

import argparse
import json
import pickle
from pathlib import Path

import torch
from tokenizers.processors import TemplateProcessing

from ...convert_slow_tokenizer import TikTokenConverter
from ...tokenization_utils_tokenizers import TokenizersBackend
from ..nanochat.convert_nanochat_checkpoints import convert_layer
from .configuration_blueberry import BlueberryConfig
from .modeling_blueberry import BlueberryForCausalLM


class BlueberryTikTokenConverter(TikTokenConverter):
    def __init__(self, encoding):
        special_tokens = sorted(encoding._special_tokens, key=encoding._special_tokens.__getitem__)
        super().__init__(pattern=encoding._pat_str, extra_special_tokens=special_tokens)
        self.mergeable_ranks = encoding._mergeable_ranks

    def load_tiktoken_bpe(self, vocab_file):
        return self.mergeable_ranks


def convert_state_dict(state_dict: dict[str, torch.Tensor], config: BlueberryConfig) -> dict[str, torch.Tensor]:
    state_dict = {key.removeprefix("_orig_mod."): tensor for key, tensor in state_dict.items()}
    mapping = {
        "transformer.wte.weight": "model.embed_tokens.weight",
        "lm_head.weight": "lm_head.weight",
        "resid_lambdas": "model.resid_lambdas",
        "x0_lambdas": "model.x0_lambdas",
        "smear_gate.weight": "model.smear_gate.weight",
        "smear_lambda": "model.smear_lambda",
        "backout_lambda": "model.backout_lambda",
    }
    if config.loop_gate:
        mapping.update({"loop_gate.weight": "model.loop_gate.weight", "loop_gate_bias": "model.loop_gate_bias"})

    vocab_weights = {"transformer.wte.weight", "lm_head.weight"}
    for idx in range(config.num_unique_layers):
        old_prefix = f"transformer.h.{idx}"
        new_prefix = f"model.layers.{idx}"
        mapping.update(convert_layer(old_prefix, new_prefix))
        if config.has_value_embedding(idx):
            mapping[f"{old_prefix}.attn.ve_gate.weight"] = f"{new_prefix}.self_attn.value_gate.weight"
            key = f"value_embeds.{idx}.weight"
            mapping[key] = f"model.{key}"
            vocab_weights.add(key)

    missing = mapping.keys() - state_dict.keys()
    unexpected = state_dict.keys() - mapping.keys()
    if missing or unexpected:
        raise ValueError(
            f"Checkpoint tensors do not match Blueberry. Missing: {sorted(missing)}. Unexpected: {sorted(unexpected)}."
        )

    converted = {}
    for old_key, new_key in mapping.items():
        tensor = state_dict[old_key]
        if old_key in vocab_weights:
            # The source checkpoint pads the vocabulary to a multiple of 64 for training.
            tensor = tensor[: config.vocab_size].clone()
        converted[new_key] = tensor.float()
    return converted


def convert_tokenizer(tokenizer_path: Path, config: BlueberryConfig) -> TokenizersBackend:
    with tokenizer_path.open("rb") as handle:
        encoding = pickle.load(handle)
    if encoding.n_vocab != config.vocab_size:
        raise ValueError(
            f"The tokenizer has {encoding.n_vocab} tokens, but the checkpoint expects {config.vocab_size}."
        )

    bos_token = "<|bos|>"
    bos_id = encoding.encode_single_token(bos_token)
    tokenizer = TokenizersBackend(
        tokenizer_object=BlueberryTikTokenConverter(encoding).converted(),
        bos_token=bos_token,
        eos_token=bos_token,
        pad_token=bos_token,
        additional_special_tokens=sorted(encoding.special_tokens_set - {bos_token}),
        padding_side="left",
        model_max_length=config.max_position_embeddings,
        model_input_names=["input_ids", "attention_mask"],
    )
    for token in encoding.special_tokens_set:
        if tokenizer.convert_tokens_to_ids(token) != encoding.encode_single_token(token):
            raise ValueError(f"The token ID changed during conversion for {token!r}.")
    tokenizer.backend_tokenizer.post_processor = TemplateProcessing(
        single=f"{bos_token} $A",
        pair=f"{bos_token} $A {bos_token}:1 $B:1",
        special_tokens=[(bos_token, bos_id)],
    )
    return tokenizer


def convert_checkpoint(checkpoint_dir: Path, step: int, tokenizer_path: Path, output_dir: Path) -> None:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError(f"Choose an empty output directory. Found existing files in {output_dir}.")

    metadata = json.loads((checkpoint_dir / f"meta_{step:06d}.json").read_text(encoding="utf-8"))
    source = metadata["model_config"]
    config = BlueberryConfig(
        vocab_size=source["vocab_size"],
        hidden_size=source["n_embd"],
        num_unique_layers=source["n_layer"],
        num_attention_heads=source["n_head"],
        num_key_value_heads=source["n_kv_head"],
        max_position_embeddings=source["sequence_len"],
        window_pattern=source.get("window_pattern", "L"),
        loop_repeats=source.get("loop_repeats", 1),
        loop_gate=source.get("loop_gate", False),
    )
    tokenizer = convert_tokenizer(tokenizer_path, config)
    config.bos_token_id = tokenizer.bos_token_id
    config.eos_token_id = tokenizer.eos_token_id
    config.pad_token_id = tokenizer.pad_token_id

    state_dict = torch.load(checkpoint_dir / f"model_{step:06d}.pt", map_location="cpu", weights_only=True)
    with torch.device("meta"):
        model = BlueberryForCausalLM(config)
    model.load_state_dict(convert_state_dict(state_dict, config), strict=True, assign=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    (output_dir / "README.md").write_text(
        f"""---
library_name: transformers
pipeline_tag: text-generation
---

# Blueberry

Blueberry is a looped decoder-only language model by Dustin Loring (dustinwloring1988@gmail.com).
The release includes [Blueberry](https://huggingface.co/CrowdMind/Blueberry) for chat and
[Blueberry-base](https://huggingface.co/CrowdMind/Blueberry-base) for text completion.

This checkpoint has {parameter_count:,} parameters and was exported from training step {step:,}.
Its {config.num_unique_layers} distinct blocks run {config.num_hidden_layers} times per forward pass,
with {config.loop_repeats} passes through the middle half.

Load the checkpoint with `AutoModelForCausalLM` and `AutoTokenizer` using
the [Transformers fork with Blueberry support](https://github.com/blueberry-llm/transformers).
The training code is in the [Blueberry project](https://github.com/blueberry-llm/blueberry).
""",
        encoding="utf-8",
    )


def main():
    parser = argparse.ArgumentParser(
        description="Convert a Blueberry checkpoint and tokenizer to Transformers format."
    )
    parser.add_argument("--checkpoint-dir", type=Path, required=True, help="Directory with model and metadata files.")
    parser.add_argument(
        "--step", type=int, required=True, help="Training step shared by the model and metadata files."
    )
    parser.add_argument("--tokenizer", type=Path, required=True, help="Path to the trusted local tokenizer.pkl.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Empty directory for the converted checkpoint.")
    args = parser.parse_args()
    convert_checkpoint(args.checkpoint_dir, args.step, args.tokenizer, args.output_dir)


if __name__ == "__main__":
    main()
