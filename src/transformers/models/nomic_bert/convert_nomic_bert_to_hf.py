# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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

import torch

from transformers import (
    AutoConfig,
    AutoModelForMaskedLM,
    NomicBertConfig,
    NomicBertModel,
)


EPILOG_TXT = """Example:
    python transformers/src/transformers/models/nomic_bert/convert_nomic_bert_to_hf.py --original_model_id nomic-ai/nomic-embed-text-v1.5 --output_hub_path org/nomic_bert
"""


def split_qkv_weight(qkv_weight, num_heads, head_dim):
    """
    Split combined QKV weight into separate Q, K, V weights.
    Original shape: (3 * num_heads * head_dim, hidden_size)
    Output: query, key, value weights each of shape (num_heads * head_dim, hidden_size)
    """
    qkv_weight = qkv_weight.view(3, num_heads * head_dim, -1)
    return qkv_weight[0], qkv_weight[1], qkv_weight[2]


def convert_state_dict_to_hf(state_dict, config):
    """
    Convert state dict keys from original nomic-ai format to HuggingFace format.
    The original model uses a different architecture:
    - Combined QKV attention (Wqkv) instead of separate Q/K/V
    - Different MLP structure (fc11, fc12, fc2) for SwiGLU
    - Different normalization names (norm1, norm2)
    - emb_ln instead of embeddings.LayerNorm
    - No position embeddings (uses RoPE instead)
    """
    new_state_dict = {}
    num_heads = config.num_attention_heads
    head_dim = config.hidden_size // num_heads

    for key, value in state_dict.items():
        # Skip MLM head keys since we're converting to base model (NomicBertModel)
        if key.startswith("cls.predictions"):
            continue

        # Embeddings
        if key == "bert.embeddings.word_embeddings.weight":
            new_state_dict["embeddings.word_embeddings.weight"] = value
        elif key == "bert.embeddings.token_type_embeddings.weight":
            new_state_dict["embeddings.token_type_embeddings.weight"] = value
        # Note: Original model doesn't have position embeddings (uses RoPE)
        elif key == "bert.emb_ln.weight":
            new_state_dict["embeddings.LayerNorm.weight"] = value
        elif key == "bert.emb_ln.bias":
            new_state_dict["embeddings.LayerNorm.bias"] = value

        # Encoder layers: bert.encoder.layers.X -> encoder.layer.X
        elif "bert.encoder.layers." in key:
            # Replace bert.encoder.layers with encoder.layer
            new_key = key.replace("bert.encoder.layers.", "encoder.layer.")

            # Handle combined QKV attention weights
            if ".attn.Wqkv.weight" in new_key:
                # Split QKV into separate Q, K, V
                q_weight, k_weight, v_weight = split_qkv_weight(value, num_heads, head_dim)
                new_state_dict[new_key.replace(".attn.Wqkv.weight", ".attention.self.query.weight")] = q_weight
                new_state_dict[new_key.replace(".attn.Wqkv.weight", ".attention.self.key.weight")] = k_weight
                new_state_dict[new_key.replace(".attn.Wqkv.weight", ".attention.self.value.weight")] = v_weight
                continue
            # Handle attention output projection
            elif ".attn.out_proj.weight" in new_key:
                new_key = new_key.replace(".attn.out_proj.weight", ".attention.output.dense.weight")
            elif ".attn.out_proj.bias" in new_key:
                new_key = new_key.replace(".attn.out_proj.bias", ".attention.output.dense.bias")
            # Handle MLP layers (SwiGLU: fc11=gate, fc12=value, fc2=output)
            elif ".mlp.fc11.weight" in new_key:
                new_key = new_key.replace(".mlp.fc11.weight", ".intermediate.dense_gate.weight")
            elif ".mlp.fc11.bias" in new_key:
                new_key = new_key.replace(".mlp.fc11.bias", ".intermediate.dense_gate.bias")
            elif ".mlp.fc12.weight" in new_key:
                new_key = new_key.replace(".mlp.fc12.weight", ".intermediate.dense.weight")
            elif ".mlp.fc12.bias" in new_key:
                new_key = new_key.replace(".mlp.fc12.bias", ".intermediate.dense.bias")
            elif ".mlp.fc2.weight" in new_key:
                new_key = new_key.replace(".mlp.fc2.weight", ".output.dense.weight")
            elif ".mlp.fc2.bias" in new_key:
                new_key = new_key.replace(".mlp.fc2.bias", ".output.dense.bias")
            # Handle layer norms
            elif ".norm1.weight" in new_key:
                new_key = new_key.replace(".norm1.weight", ".attention.output.LayerNorm.weight")
            elif ".norm1.bias" in new_key:
                new_key = new_key.replace(".norm1.bias", ".attention.output.LayerNorm.bias")
            elif ".norm2.weight" in new_key:
                new_key = new_key.replace(".norm2.weight", ".output.LayerNorm.weight")
            elif ".norm2.bias" in new_key:
                new_key = new_key.replace(".norm2.bias", ".output.LayerNorm.bias")

            new_state_dict[new_key] = value

        # Pooler (if present)
        elif key.startswith("bert.pooler"):
            new_key = key.replace("bert.pooler", "pooler")
            new_state_dict[new_key] = value

    return new_state_dict


def get_config(checkpoint):
    base_config = AutoConfig.from_pretrained(checkpoint)
    if checkpoint == "nomic-ai/nomic-embed-text-v1.5":
        return NomicBertConfig(
            rotary_emb_fraction=base_config.rotary_emb_fraction,
            rotary_emb_base=base_config.rotary_emb_base,
            rotary_emb_scale_base=base_config.rotary_emb_scale_base,
            rotary_emb_interleaved=base_config.rotary_emb_interleaved,
            type_vocab_size=base_config.type_vocab_size,
            pad_vocab_size_multiple=base_config.pad_vocab_size_multiple,
            tie_word_embeddings=base_config.tie_word_embeddings,
            max_position_embeddings=base_config.max_position_embeddings,
        )

    return base_config


def convert_nomic_hub_to_hf(original_model_id, output_hub_path, push_to_hub):
    config = AutoConfig.from_pretrained(original_model_id, trust_remote_code=True)  # the config needs to be passed in
    original_model = AutoModelForMaskedLM.from_pretrained(original_model_id, config=config, trust_remote_code=True)

    config = get_config(original_model_id)

    with torch.device("meta"):
        model = NomicBertModel(config)

    state_dict = original_model.state_dict()
    state_dict = convert_state_dict_to_hf(state_dict, config)

    # Get the expected state dict from the model to fill in missing keys
    # We need to create the model on CPU to get the actual shapes
    with torch.device("cpu"):
        cpu_model = NomicBertModel(config)
        expected_state_dict = cpu_model.state_dict()

    # Initialize missing keys with zeros
    for key in expected_state_dict.keys():
        if key not in state_dict:
            expected_shape = expected_state_dict[key].shape
            if "position_embeddings" in key:
                # Position embeddings are not used (RoPE instead), but we need to initialize them
                print(f"Initializing {key} with zeros (RoPE is used instead)")
                state_dict[key] = torch.zeros(expected_shape, dtype=expected_state_dict[key].dtype)
            elif "bias" in key:
                # Initialize missing biases with zeros
                print(f"Initializing {key} with zeros")
                state_dict[key] = torch.zeros(expected_shape, dtype=expected_state_dict[key].dtype)
            elif "pooler" in key:
                # Initialize pooler if missing
                print(f"Initializing {key} with zeros")
                state_dict[key] = torch.zeros(expected_shape, dtype=expected_state_dict[key].dtype)
            else:
                print(f"Warning: Missing key {key}, initializing with zeros")
                state_dict[key] = torch.zeros(expected_shape, dtype=expected_state_dict[key].dtype)

    # Move model to CPU to load state dict
    model = cpu_model
    model.load_state_dict(state_dict, strict=True)
    model.save_pretrained(output_hub_path)

    if push_to_hub:
        model.push_to_hub(output_hub_path, private=True)


def main():
    parser = argparse.ArgumentParser(
        epilog=EPILOG_TXT,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--original_model_id",
        default="nomic-ai/nomic-embed-text-v1.5",
        help="Hub location of the model",
    )
    parser.add_argument(
        "--output_hub_path",
        default="org/nomic_bert",
        help="Location on the hub of the converted model",
    )
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        help="If set, the model will be pushed to the hub after conversion.",
    )
    args = parser.parse_args()
    convert_nomic_hub_to_hf(args.original_model_id, args.output_hub_path, args.push_to_hub)


if __name__ == "__main__":
    main()
