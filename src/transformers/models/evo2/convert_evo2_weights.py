# Copyright 2025 Arc Institute and The HuggingFace Inc. team. All rights reserved.
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
"""Convert original Evo 2 (StripedHyena 2 / Vortex) checkpoints to Hugging Face format."""

import argparse

import torch
from huggingface_hub import hf_hub_download

from transformers import Evo2Config, Evo2ForCausalLM, Evo2Tokenizer


def infer_config_from_original_weights(
    original_weights: dict[str, torch.Tensor],
    max_position_embeddings: int = 8192,
    rope_scaling_factor: float | None = None,
) -> Evo2Config:
    vocab_size, hidden_size = original_weights["embedding_layer.weight"].shape
    intermediate_size = original_weights["blocks.0.mlp.l1.weight"].shape[0]
    num_hidden_layers = sum(1 for key in original_weights if key.endswith(".pre_norm.scale"))

    mixer_types = []
    head_dim = 128
    short_conv_kernel_size = 3
    hcs_filter_length = 7
    hcm_filter_length = 128
    filter_groups = 128
    state_size = 16

    for i in range(num_hidden_layers):
        prefix = f"blocks.{i}"
        if f"{prefix}.inner_mha_cls.Wqkv.weight" in original_weights:
            mixer_types.append("attention")
            if f"{prefix}.inner_mha_cls.rotary_emb.inv_freq" in original_weights:
                head_dim = 2 * original_weights[f"{prefix}.inner_mha_cls.rotary_emb.inv_freq"].numel()
        elif f"{prefix}.filter.log_poles" in original_weights:
            mixer_types.append("hcl")
            state_size = original_weights[f"{prefix}.filter.log_poles"].shape[1]
        elif f"{prefix}.filter.D" in original_weights:
            mixer_types.append("hcm")
            h = original_weights[f"{prefix}.filter.h"]
            hcm_filter_length = h.shape[-1]
        else:
            mixer_types.append("hcs")
            h = original_weights[f"{prefix}.filter.h"]
            hcs_filter_length = h.shape[-1]
            filter_groups = h.numel() // hcs_filter_length

        if f"{prefix}.filter.short_filter_weight" in original_weights:
            short_conv_kernel_size = original_weights[f"{prefix}.filter.short_filter_weight"].shape[-1]

    rope_parameters = None
    if rope_scaling_factor is not None:
        rope_parameters = {"rope_type": "linear", "factor": float(rope_scaling_factor)}

    return Evo2Config(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=hidden_size // head_dim,
        max_position_embeddings=max_position_embeddings,
        rope_parameters=rope_parameters,
        mixer_types=mixer_types,
        short_conv_kernel_size=short_conv_kernel_size,
        hcs_filter_length=hcs_filter_length,
        hcm_filter_length=hcm_filter_length,
        filter_groups=filter_groups,
        state_size=state_size,
        tie_word_embeddings=True,
    )


def convert_original_weights_to_transformers(
    original_weights: dict[str, torch.Tensor],
    config: Evo2Config | None = None,
) -> tuple[dict[str, torch.Tensor], Evo2Config]:
    """Convert weights from original Evo 2 (Vortex) format to Hugging Face format."""
    if isinstance(original_weights, dict) and "state_dict" in original_weights:
        original_weights = original_weights["state_dict"]
    elif (
        isinstance(original_weights, dict)
        and "model" in original_weights
        and isinstance(original_weights["model"], dict)
    ):
        original_weights = original_weights["model"]

    if config is None:
        config = infer_config_from_original_weights(original_weights)

    new_state_dict: dict[str, torch.Tensor] = {}

    # Token embeddings and LM head
    new_state_dict["model.embed_tokens.weight"] = original_weights["embedding_layer.weight"]
    new_state_dict["lm_head.weight"] = original_weights.get(
        "unembed.weight", original_weights["embedding_layer.weight"]
    )

    # Decoder layers
    for layer_idx in range(config.num_hidden_layers):
        mixer_type = config.mixer_types[layer_idx]
        orig_prefix = f"blocks.{layer_idx}"
        new_prefix = f"model.layers.{layer_idx}"

        # Pre/post norms
        new_state_dict[f"{new_prefix}.operator_norm.weight"] = original_weights[f"{orig_prefix}.pre_norm.scale"]
        new_state_dict[f"{new_prefix}.ffn_norm.weight"] = original_weights[f"{orig_prefix}.post_norm.scale"]

        # MLP layers: l1 (gate), l2 (up), l3 (down)
        new_state_dict[f"{new_prefix}.feed_forward.gate_proj.weight"] = original_weights[
            f"{orig_prefix}.mlp.l1.weight"
        ]
        new_state_dict[f"{new_prefix}.feed_forward.up_proj.weight"] = original_weights[f"{orig_prefix}.mlp.l2.weight"]
        new_state_dict[f"{new_prefix}.feed_forward.down_proj.weight"] = original_weights[
            f"{orig_prefix}.mlp.l3.weight"
        ]

        if mixer_type == "attention":
            wqkv = original_weights[f"{orig_prefix}.inner_mha_cls.Wqkv.weight"]
            q, k, v = torch.split(wqkv, config.hidden_size, dim=0)
            new_state_dict[f"{new_prefix}.self_attn.q_proj.weight"] = q
            new_state_dict[f"{new_prefix}.self_attn.k_proj.weight"] = k
            new_state_dict[f"{new_prefix}.self_attn.v_proj.weight"] = v
            new_state_dict[f"{new_prefix}.self_attn.o_proj.weight"] = original_weights[
                f"{orig_prefix}.inner_mha_cls.out_proj.weight"
            ]
            new_state_dict[f"{new_prefix}.self_attn.o_proj.bias"] = original_weights[
                f"{orig_prefix}.inner_mha_cls.out_proj.bias"
            ]
        else:
            new_state_dict[f"{new_prefix}.conv.in_proj.weight"] = original_weights[f"{orig_prefix}.projections.weight"]
            new_state_dict[f"{new_prefix}.conv.short_conv"] = original_weights[
                f"{orig_prefix}.filter.short_filter_weight"
            ].reshape(3 * config.hidden_size, 1, config.short_conv_kernel_size)
            new_state_dict[f"{new_prefix}.conv.out_proj.weight"] = original_weights[
                f"{orig_prefix}.out_filter_dense.weight"
            ]
            new_state_dict[f"{new_prefix}.conv.out_proj.bias"] = original_weights[
                f"{orig_prefix}.out_filter_dense.bias"
            ]

            if mixer_type == "hcl":
                new_state_dict[f"{new_prefix}.conv.log_poles"] = original_weights[
                    f"{orig_prefix}.filter.log_poles"
                ].reshape(config.hidden_size, config.state_size, 1)
                new_state_dict[f"{new_prefix}.conv.residues"] = original_weights[
                    f"{orig_prefix}.filter.residues"
                ].reshape(config.hidden_size, config.state_size)
            else:
                filter_len = config.hcs_filter_length if mixer_type == "hcs" else config.hcm_filter_length
                new_state_dict[f"{new_prefix}.conv.filter"] = original_weights[f"{orig_prefix}.filter.h"].reshape(
                    config.filter_groups, 1, filter_len
                )

            if mixer_type != "hcs":
                new_state_dict[f"{new_prefix}.conv.D"] = original_weights[f"{orig_prefix}.filter.D"].reshape(
                    config.hidden_size
                )

    # Final RMSNorm
    new_state_dict["model.embedding_norm.weight"] = original_weights["norm.scale"]

    return new_state_dict, config


def main():
    parser = argparse.ArgumentParser(description="Convert original Evo 2 checkpoint to Hugging Face format.")
    parser.add_argument(
        "--model_id",
        default="arcinstitute/evo2_1b_base",
        help="Hub model id or local checkpoint path.",
    )
    parser.add_argument(
        "--filename",
        default=None,
        help="Checkpoint filename in the Hub repository (defaults to '<model_name>.pt').",
    )
    parser.add_argument(
        "--max_position_embeddings",
        type=int,
        default=8192,
        help="Maximum sequence length supported by the checkpoint.",
    )
    parser.add_argument(
        "--rope_scaling_factor",
        type=float,
        default=None,
        help="Linear RoPE scaling factor (e.g. 32.0 for 262k checkpoints, 128.0 for 1M checkpoints).",
    )
    parser.add_argument(
        "--output_dir",
        default="evo2_converted",
        help="The output directory to save the converted model and tokenizer.",
    )
    args = parser.parse_args()

    filename = args.filename or f"{args.model_id.split('/')[-1]}.pt"
    weights_path = hf_hub_download(args.model_id, filename)
    original_weights = torch.load(weights_path, map_location="cpu", weights_only=True)

    config = infer_config_from_original_weights(
        original_weights,
        max_position_embeddings=args.max_position_embeddings,
        rope_scaling_factor=args.rope_scaling_factor,
    )
    new_state_dict, config = convert_original_weights_to_transformers(original_weights, config=config)

    model = Evo2ForCausalLM(config)
    model.load_state_dict(new_state_dict, strict=True)

    tokenizer = Evo2Tokenizer()
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
