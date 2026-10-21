import argparse
import json
from pathlib import Path
from typing import Any

from transformers.models.mistral_large4.modular_mistral_large4 import MistralLarge4Config


HF_TO_VLLM_MAPPING = {
    # Top level
    "spatial_merge_size": "vision_encoder.spatial_merge_size",
    "image_token_index": "vision_encoder.image_token_id",
    "tie_word_embeddings": "tied_embeddings",
    "text_config.tie_word_embeddings": "tied_embeddings",
    "multimodal_projector_bias": "vision_encoder.adapter_bias",
    # Text model: general
    "text_config.hidden_size": "dim",
    "text_config.num_hidden_layers": "n_layers",
    "text_config.head_dim": "head_dim",
    "text_config.intermediate_size": "hidden_dim",
    "text_config.num_attention_heads": "n_heads",
    "text_config.num_key_value_heads": "n_kv_heads",
    "text_config.rms_norm_eps": "norm_eps",
    "text_config.rope_parameters.rope_theta": "rope_theta",
    "text_config.max_position_embeddings": "max_position_embeddings",
    "text_config.vocab_size": "vocab_size",
    # Text model: YaRN (`apply_scale: false` means no magnitude scaling, see the `attention_factor` constant)
    "text_config.rope_parameters.factor": "yarn.factor",
    "text_config.rope_parameters.original_max_position_embeddings": "yarn.original_max_position_embeddings",
    "text_config.rope_parameters.beta_fast": "yarn.beta",
    "text_config.rope_parameters.beta_slow": "yarn.alpha",
    # Text model: MLA
    "text_config.q_lora_rank": "q_lora_rank",
    "text_config.kv_lora_rank": "kv_lora_rank",
    "text_config.qk_rope_head_dim": "qk_rope_head_dim",
    "text_config.qk_nope_head_dim": "qk_nope_head_dim",
    "text_config.v_head_dim": "v_head_dim",
    # Text model: MoE
    "text_config.n_routed_experts": "moe.num_experts",
    "text_config.num_experts_per_tok": "moe.num_experts_per_tok",
    "text_config.n_shared_experts": "moe.num_shared_experts",
    "text_config.moe_intermediate_size": "moe.expert_hidden_dim",
    "text_config.routed_scaling_factor": "moe.routed_scale",
    "text_config.first_k_dense_replace": "moe.first_k_dense_replace",
    "text_config.n_group": "moe.num_expert_groups",
    "text_config.topk_group": "moe.num_expert_groups_per_tok",
    # Text model: DSA indexer
    "text_config.index_topk": "dsa.topk",
    "text_config.index_n_heads": "dsa.n_heads",
    "text_config.index_head_dim": "dsa.head_dim",
    # Vision model
    "vision_config.hidden_size": "vision_encoder.hidden_size",
    "vision_config.intermediate_size": "vision_encoder.intermediate_size",
    "vision_config.num_hidden_layers": "vision_encoder.num_hidden_layers",
    "vision_config.num_attention_heads": "vision_encoder.num_attention_heads",
    "vision_config.num_channels": "vision_encoder.num_channels",
    "vision_config.image_size": "vision_encoder.image_size",
    "vision_config.patch_size": "vision_encoder.patch_size",
    "vision_config.rope_parameters.rope_theta": "vision_encoder.rope_theta",
}


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
    # Start with a config dict that has the parameters vLLM hardcodes
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

    # Map the parameters from the source config to the new config
    for hf_key, vllm_key in HF_TO_VLLM_MAPPING.items():
        hf_key, vllm_key = (tuple(x.split(".")) for x in (hf_key, vllm_key))
        value = recursive_get(source, vllm_key)
        recursive_set(config_dict, hf_key, value)

    # For some parameters, the mapping is more complex, and done here
    config_dict["text_config"]["indexer_types"] = source["dsa"]["sharing"].split(",")

    layers_pe = source["nope"].split(",")
    num_hidden_layers = config_dict["text_config"]["num_hidden_layers"]
    if num_hidden_layers % len(layers_pe) != 0:
        raise ValueError(
            f"{num_hidden_layers = } should be divisible by the number of no-rope layers ({len(layers_pe)})"
        )
    repeats = num_hidden_layers // len(layers_pe)
    no_rope_layers = [{"rope": 1, "nope": 0}[layer_pe] for layer_pe in layers_pe] * repeats
    config_dict["text_config"]["no_rope_layers"] = no_rope_layers

    kv_source_layers = [int(layer_idx) for layer_idx in source["kv_sharing"]["source_layers_str"].split(",")]
    if len(kv_source_layers) != num_hidden_layers:
        raise ValueError(
            f"kv_sharing.source_layers_str has {len(kv_source_layers)} entries, expected {num_hidden_layers}"
        )
    config_dict["text_config"]["kv_source_layers"] = kv_source_layers

    first_k_dense_replace = source["moe"]["first_k_dense_replace"]
    mlp_layer_types = ["dense"] * first_k_dense_replace + ["sparse"] * (num_hidden_layers - first_k_dense_replace)
    config_dict["text_config"]["mlp_layer_types"] = mlp_layer_types
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
