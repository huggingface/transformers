# coding=utf-8
# Copyright 2022 The HuggingFace Inc. team.
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
import re
from pathlib import Path

import torch
from torch import nn

from transformers import LLaMaConfig, LLaMaForCausalLM, LLaMaTokenizer


def generate_config(params_json_path: Path, vocab_size: int) -> LLaMaConfig:
    with open(params_json_path, "r") as fi:
        hyperparameters = json.load(fi)

    assert hyperparameters["vocab_size"] == -1, "We get vocab size information from the tokenizer"
    assert vocab_size > 0

    hidden_size = hyperparameters["dim"]
    multiple_of = hyperparameters["multiple_of"]
    intermediate_size = int(2 * 4 * hidden_size / 3)
    intermediate_size = multiple_of * ((intermediate_size + multiple_of - 1) // multiple_of)

    return LLaMaConfig(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        num_hidden_layers=hyperparameters["n_layers"],
        num_attention_heads=hyperparameters["n_heads"],
        intermediate_size=intermediate_size,
        max_position_embeddings=2048,
        layer_norm_eps=hyperparameters["norm_eps"],
        tie_word_embeddings=False,
        use_cache=True
    )

def get_tokenzier(tokenizer_path: Path) -> LLaMaTokenizer:
    return LLaMaTokenizer(str(tokenizer_path.absolute()))

original_name_to_transformers_name = {
    "tok_embeddings.weight" : "llama.embed.weight",
    "norm.weight": "llama.final_layer_norm.weight",
    "output.weight": "lm_head.weight",
    r"layers.(\d*).attention_norm.weight": r"llama.layers.\1.attention_norm.weight",
    r"layers.(\d*).attention.wq.weight": r"llama.layers.\1.attention.qkv.weight",
    r"layers.(\d*).attention.wk.weight": r"llama.layers.\1.attention.qkv.weight",
    r"layers.(\d*).attention.wv.weight": r"llama.layers.\1.attention.qkv.weight",
    r"layers.(\d*).attention.wo.weight": r"llama.layers.\1.attention.o.weight",
    r"layers.(\d*).ffn_norm.weight": r"llama.layers.\1.ff_norm.weight",
    r"layers.(\d*).feed_forward.w1.weight": r"llama.layers.\1.ff.wi_0.weight",
    r"layers.(\d*).feed_forward.w2.weight": r"llama.layers.\1.ff.wo.weight",
    r"layers.(\d*).feed_forward.w3.weight": r"llama.layers.\1.ff.wi_1.weight",
}
def map_original_names_to_transformers_names(original_name: str):
    for pattern, repl in original_name_to_transformers_name.items():
        if re.match(pattern, original_name) is None:
            continue
        return re.sub(pattern, repl, original_name)
    raise ValueError(f"Did not expect {original_name}")

@torch.no_grad()
def convert_model(model_path: Path, config:LLaMaConfig) -> LLaMaForCausalLM:
    # HACK @thomasw21: Bypasses `reset_parameters` which can be quite costly.
    nn.Linear.reset_parameters = lambda *args: None
    model = LLaMaForCausalLM(config=config)

    paths = sorted(model_path.glob("*.pth"))
    tp_size = len(paths)
    for tp_rank, path in enumerate(paths):
        weights = torch.load(path)

        for original_name, original_param in weights.items():
            if original_name.endswith(".attention.inner_attention.rope.freqs"):
                print(f"We ignore {original_name} as it stores the rotary embeddings which are not in fact parameters")
                continue

            transformers_name = map_original_names_to_transformers_names(original_name)
            transformers_param = model.get_parameter(transformers_name)

            if original_name.endswith("norm.weight"):
                transformers_param.copy_(original_param)
                continue

            # weights are sharded across TP
            if any(original_name.endswith(suffix) for suffix in [".feed_forward.w2.weight", ".attention.wo.weight"]):
                # Row Linear weight
                input_dim = transformers_param.shape[1]
                assert input_dim % tp_size == 0
                step = input_dim // tp_size
                start = tp_rank * step
                end = (tp_rank + 1) * step
                transformers_param[:, start:end].copy_(original_param)
                continue

            # Column linear
            if any(original_name.endswith(suffix) for suffix in [".wq.weight", ".wk.weight", "wv.weight"]):
                # We fuse all the weights into a single qkv matrix.
                index, suffix = [(i, suffix) for i, suffix in enumerate([".wq.weight", ".wk.weight", "wv.weight"]) if original_name.endswith(suffix)][0]
                assert config.num_attention_heads % tp_size == 0
                heads_per_tp_rank = config.num_attention_heads // tp_size
                transformer_shard = transformers_param \
                    .view(config.num_attention_heads, 3, config.hidden_size // config.num_attention_heads, config.hidden_size) \
                    [tp_rank * heads_per_tp_rank: (tp_rank+1) * heads_per_tp_rank, index]
                original_param = original_param.view(*transformer_shard.shape)
            else:
                output_dim = transformers_param.shape[0]
                assert output_dim % tp_size == 0
                step = output_dim // tp_size
                start = tp_rank * step
                end = (tp_rank + 1) * step
                transformer_shard = transformers_param[start: end]

            transformer_shard.copy_(original_param)

    return model

def main(args):
    tokenizer = get_tokenzier(tokenizer_path=args.checkpoint_directory / "tokenizer.model")

    model_path = args.checkpoint_directory / args.model_subpath
    config = generate_config(model_path / "params.json", vocab_size=tokenizer.vocab_size)
    model = convert_model(model_path=model_path, config=config)

    config.save_pretrained(args.pytorch_dump_folder_path)
    model.save_pretrained(args.pytorch_dump_folder_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument(
        "--checkpoint-directory",
        type=Path,
        required=True,
        help="Path to the checkpoint path containing `tokenizer.json` and different model size checkpoints.",
    )
    parser.add_argument(
        "--pytorch-dump-folder-path", type=Path, required=True, help="Path to the output PyTorch model."
    )
    parser.add_argument(
        "--model-subpath", type=Path, required=True, help="Subpath after going into checkpoint directory where the model checkpoint lies. Typically `7B` or `13B`"
    )
    args = parser.parse_args()
    main(args)