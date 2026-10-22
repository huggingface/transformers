# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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
"""Convert the original NeoBERT masked language model to native EuroBERT weights.

python src/transformers/models/eurobert/convert_neobert_to_eurobert.py \
    --checkpoint_path chandar-lab/NeoBERT --output_dir /path/to/neobert-eurobert
"""

import argparse
import json
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from safetensors.torch import load_file

from transformers import BertTokenizer, EuroBertConfig, EuroBertForMaskedLM


def convert_config(original_config: dict, intermediate_size: int, tokenizer: BertTokenizer) -> EuroBertConfig:
    rope_parameters = {"rope_type": "default", "rope_theta": 10000.0}
    return EuroBertConfig(
        vocab_size=original_config["vocab_size"],
        hidden_size=original_config["hidden_size"],
        intermediate_size=intermediate_size,
        num_hidden_layers=original_config["num_hidden_layers"],
        num_attention_heads=original_config["num_attention_heads"],
        num_key_value_heads=original_config["num_attention_heads"],
        max_position_embeddings=original_config["max_length"],
        rms_norm_eps=original_config["norm_eps"],
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.cls_token_id,
        eos_token_id=tokenizer.sep_token_id,
        mask_token_id=tokenizer.mask_token_id,
        rope_parameters=rope_parameters,
        hidden_act="silu",
        attention_bias=False,
        mlp_bias=True,
        tie_word_embeddings=False,
    )


def convert_state_dict(state_dict: dict[str, torch.Tensor], config: EuroBertConfig) -> dict[str, torch.Tensor]:
    state_dict = state_dict.copy()
    hidden_size = config.hidden_size
    num_heads = config.num_attention_heads
    head_dim = config.head_dim

    def permute_for_rope(weight):
        # NeoBERT rotates adjacent complex pairs; EuroBERT rotates pairs from the two halves of each head.
        return (
            weight.reshape(num_heads, head_dim // 2, 2, hidden_size).transpose(1, 2).reshape(hidden_size, hidden_size)
        )

    converted = {
        "model.embed_tokens.weight": state_dict.pop("model.encoder.weight"),
        "model.norm.weight": state_dict.pop("model.layer_norm.weight"),
        "lm_head.weight": state_dict.pop("decoder.weight"),
        "lm_head.bias": state_dict.pop("decoder.bias"),
    }
    for layer_idx in range(config.num_hidden_layers):
        source = f"model.transformer_encoder.{layer_idx}"
        target = f"model.layers.{layer_idx}"
        # QKV blocks are grouped within each head, rather than across all heads.
        qkv = state_dict.pop(f"{source}.qkv.weight").reshape(num_heads, 3, head_dim, hidden_size)
        query, key, value = qkv.unbind(dim=1)
        converted[f"{target}.self_attn.q_proj.weight"] = permute_for_rope(query)
        converted[f"{target}.self_attn.k_proj.weight"] = permute_for_rope(key)
        converted[f"{target}.self_attn.v_proj.weight"] = value.reshape(hidden_size, hidden_size)
        converted[f"{target}.self_attn.o_proj.weight"] = state_dict.pop(f"{source}.wo.weight")
        gate, up = state_dict.pop(f"{source}.ffn.w12.weight").chunk(2, dim=0)
        converted[f"{target}.mlp.gate_proj.weight"] = gate
        converted[f"{target}.mlp.up_proj.weight"] = up
        converted[f"{target}.mlp.down_proj.weight"] = state_dict.pop(f"{source}.ffn.w3.weight")
        converted[f"{target}.input_layernorm.weight"] = state_dict.pop(f"{source}.attention_norm.weight")
        converted[f"{target}.post_attention_layernorm.weight"] = state_dict.pop(f"{source}.ffn_norm.weight")
        # EuroBERT's mlp_bias enables both the LM head bias and all MLP biases. NeoBERT has only the former.
        for projection in ("gate_proj", "up_proj", "down_proj"):
            weight = converted[f"{target}.mlp.{projection}.weight"]
            converted[f"{target}.mlp.{projection}.bias"] = weight.new_zeros(weight.shape[0])

    if state_dict:
        raise ValueError(f"Unexpected NeoBERT checkpoint keys: {sorted(state_dict)}")
    return {key: value.contiguous() for key, value in converted.items()}


@torch.no_grad()
def convert_neobert_checkpoint(checkpoint_path: str, output_dir: str, revision: str | None = None):
    input_dir = Path(checkpoint_path)
    if not input_dir.is_dir():
        input_dir = Path(
            snapshot_download(
                checkpoint_path,
                revision=revision,
                allow_patterns=["*.json", "model.safetensors", "vocab.txt"],
            )
        )
    with open(input_dir / "config.json", encoding="utf-8") as config_file:
        original_config = json.load(config_file)
    if original_config.get("architectures", ["NeoBERTLMHead"]) != ["NeoBERTLMHead"]:
        raise ValueError("This converter supports NeoBERTLMHead checkpoints only.")

    state_dict = load_file(input_dir / "model.safetensors")
    # The original config's intermediate_size precedes SwiGLU's 2/3 reduction and rounding to a multiple of 8.
    intermediate_size = state_dict["model.transformer_encoder.0.ffn.w12.weight"].shape[0] // 2
    tokenizer = BertTokenizer.from_pretrained(input_dir, model_input_names=["input_ids", "attention_mask"])
    config = convert_config(original_config, intermediate_size, tokenizer)
    converted = convert_state_dict(state_dict, config)
    with torch.device("meta"):
        model = EuroBertForMaskedLM(config)
    model.load_state_dict(converted, strict=True, assign=True)
    model.save_pretrained(output_dir)
    tokenizer.model_max_length = config.max_position_embeddings
    tokenizer.save_pretrained(output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint_path", default="chandar-lab/NeoBERT", help="Original Hub model ID or local directory."
    )
    parser.add_argument("--output_dir", required=True, help="Directory for the native EuroBERT model and tokenizer.")
    parser.add_argument("--revision", help="Hub revision of the original checkpoint.")
    args = parser.parse_args()
    convert_neobert_checkpoint(args.checkpoint_path, args.output_dir, args.revision)
