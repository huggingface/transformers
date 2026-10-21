import argparse
import json
from pathlib import Path
from typing import Any

from transformers.models.mistral_large4.modular_mistral_large4 import MistralLarge4Config


VLLM_TO_HF_MAPPING = {
    # Top level
    "vision_encoder.spatial_merge_size": "spatial_merge_size",
    "vision_encoder.image_token_id": "image_token_index",
    "tied_embeddings": "tie_word_embeddings",
    "vision_encoder.adapter_bias": "multimodal_projector_bias",
    # Text model: general
    "dim": "text_config.hidden_size",
    "n_layers": "text_config.num_hidden_layers",
    "head_dim": "text_config.head_dim",
    "hidden_dim": "text_config.intermediate_size",
    "n_heads": "text_config.num_attention_heads",
    "n_kv_heads": "text_config.num_key_value_heads",
    "norm_eps": "text_config.rms_norm_eps",
    "rope_theta": "text_config.rope_parameters.rope_theta",
    "max_position_embeddings": "text_config.max_position_embeddings",
    "vocab_size": "text_config.vocab_size",
    # Text model: YaRN (`apply_scale: false` means no magnitude scaling, see the `attention_factor` constant)
    "yarn.factor": "text_config.rope_parameters.factor",
    "yarn.original_max_position_embeddings": "text_config.rope_parameters.original_max_position_embeddings",
    "yarn.beta": "text_config.rope_parameters.beta_fast",
    "yarn.alpha": "text_config.rope_parameters.beta_slow",
    # Text model: MLA
    "q_lora_rank": "text_config.q_lora_rank",
    "kv_lora_rank": "text_config.kv_lora_rank",
    "qk_rope_head_dim": "text_config.qk_rope_head_dim",
    "qk_nope_head_dim": "text_config.qk_nope_head_dim",
    "v_head_dim": "text_config.v_head_dim",
    # Text model: MoE
    "moe.num_experts": "text_config.n_routed_experts",
    "moe.num_experts_per_tok": "text_config.num_experts_per_tok",
    "moe.num_shared_experts": "text_config.n_shared_experts",
    "moe.expert_hidden_dim": "text_config.moe_intermediate_size",
    "moe.routed_scale": "text_config.routed_scaling_factor",
    "moe.first_k_dense_replace": "text_config.first_k_dense_replace",
    "moe.num_expert_groups": "text_config.n_group",
    "moe.num_expert_groups_per_tok": "text_config.topk_group",
    # Text model: DSA indexer
    "dsa.topk": "text_config.index_topk",
    "dsa.n_heads": "text_config.index_n_heads",
    "dsa.head_dim": "text_config.index_head_dim",
    # Vision model
    "vision_encoder.hidden_size": "vision_config.hidden_size",
    "vision_encoder.intermediate_size": "vision_config.intermediate_size",
    "vision_encoder.num_hidden_layers": "vision_config.num_hidden_layers",
    "vision_encoder.num_attention_heads": "vision_config.num_attention_heads",
    "vision_encoder.num_channels": "vision_config.num_channels",
    "vision_encoder.image_size": "vision_config.image_size",
    "vision_encoder.patch_size": "vision_config.patch_size",
    "vision_encoder.rope_theta": "vision_config.rope_parameters.rope_theta",
}


# Need a transformation of the source value (pattern / string expansion), not a plain copy
#   text_config.layer_types
#   text_config.indexer_types          (dsa.sharing)
#   text_config.no_rope_layer_interval (nope)
#   text_config.mlp_layer_types        (moe.first_k_dense_replace, moe.route_every_n)


def recursive_get(d: dict, key: tuple[str, ...]) -> Any:
    # Error case
    if not isinstance(key, tuple) or len(key) == 0:
        raise ValueError(f"Invalid key: {key}")
    # Stop case
    if len(key) == 1:
        return d[key[0]]
    # Recursive case
    else:
        return recursive_get(d[key[0]], key[1:])


def recursive_set(d: dict, key: tuple[str, ...], value: Any) -> None:
    # Error case
    if not isinstance(key, tuple) or len(key) == 0:
        raise ValueError(f"Invalid key: {key}")
    # Stop case
    if len(key) == 1:
        d[key[0]] = value
    # Recursive case
    else:
        if key[0] not in d:
            d[key[0]] = {}
        recursive_set(d[key[0]], key[1:], value)


def convert_config(source: dict) -> dict:
    config_dict = {
        "model_type": "mistral_large4",
        "projector_hidden_act": "gelu",
        "vision_feature_layer": -1,
        "text_config": {
            "model_type": "mistral_large4_text",
            "hidden_act": "silu",
            "norm_topk_prob": True,
            "rope_parameters": {"rope_type": "yarn", "attention_factor": 1.0},
        },
        "vision_config": {
            "model_type": "pixtral",
            "hidden_act": "silu",
            "rope_parameters": {"rope_type": "axial"},
        },
    }
    for src_key, dst_key in VLLM_TO_HF_MAPPING.items():
        src_key, dst_key = (tuple(x.split(".")) for x in (src_key, dst_key))
        value = recursive_get(source, src_key)
        recursive_set(config_dict, dst_key, value)
    return config_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", help="Location of the mistral_large4 checkpoint")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Location to write the HF-formatted checkpoint. If not specified, only the config will be created.",
    )
    parser.add_argument("--config_name", type=str, default="params.json", help="The name of the original config file.")
    # parser.add_argument("--tokenizer_only", action="store_true", help="If passed, will only convert the tokenizer.")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    config_path = input_dir / args.config_name
    with open(config_path, encoding="utf-8") as f:
        params = json.load(f)
    config = convert_config(params)

    # Compare the keys of the config to the reference, recursively
    def flatten_keys(d: str | dict):
        if isinstance(d, str):
            return [d]
        flattened = []
        for key, value in d.items():
            if isinstance(value, dict):
                flattened += [f"{key}.{x}" for x in flatten_keys(value)]
            else:
                flattened.append(key)
        return flattened

    reference = json.loads(MistralLarge4Config().to_json_string())
    with open("reference.json", "w", encoding="utf-8") as f:
        json.dump(reference, f, indent=4)

    reference_keys = set(flatten_keys(reference))
    config_keys = set(flatten_keys(config))

    missing_keys = reference_keys - config_keys
    print("\nMissing keys:")
    for key in missing_keys:
        print(f"  {key}")
