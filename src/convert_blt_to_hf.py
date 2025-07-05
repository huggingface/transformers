import argparse
import json
import logging
import os
from typing import Any, Dict, Optional

import torch
from huggingface_hub import hf_hub_download, upload_folder
from safetensors.torch import load_file, save_file

from transformers.models.blt_wip.configuration_blt import BLTConfig
from transformers.models.blt_wip.modeling_blt import BLTModel
from transformers.models.blt_wip.modeling_blt_dev import BLTForCausalLM
from transformers.utils import logging as transformers_logging


logger = transformers_logging.get_logger(__name__)
transformers_logging.set_verbosity_info()


def merge_configurations(config_path: str, entropy_params_path: str) -> Dict[str, Any]:
    logger.info("Merging configurations")

    with open(config_path, "r") as f:
        main_config = json.load(f)

    with open(entropy_params_path, "r") as f:
        entropy_data = json.load(f)

    entropy_model_params = entropy_data.get("entropy_model", {})
    patcher_args = entropy_data.get("data", {}).get("patcher_args", {})

    unified_config = main_config.copy()["args"]

    for key in ["vocab_size", "dim", "n_layers", "n_heads", "max_seqlen"]:
        if key in unified_config and not isinstance(unified_config[key], int):
            unified_config[key] = int(unified_config[key])

    patch_size = patcher_args.get("patch_size", 8)
    if isinstance(patch_size, float):
        patch_size = int(patch_size)

    # Create patcher config
    patcher_hidden_size = int(entropy_model_params.get("dim", 512))
    patcher_multiple_of = int(entropy_model_params.get("multiple_of", 256))
    patcher_intermediate_size = patcher_multiple_of * ((int(8 * patcher_hidden_size / 3) + patcher_multiple_of - 1) // patcher_multiple_of)
    
    patcher_config = {
        "vocab_size": int(entropy_model_params.get("vocab_size", 256)),
        "hidden_size": patcher_hidden_size,
        "num_hidden_layers": int(entropy_model_params.get("n_layers", 8)),
        "num_attention_heads": int(entropy_model_params.get("n_heads", 8)),
        "num_key_value_heads": int(entropy_model_params.get("n_kv_heads"))
        if entropy_model_params.get("n_kv_heads") is not None
        else None,
        "max_position_embeddings": int(entropy_model_params.get("max_seqlen", 1024)),
        "norm_eps": entropy_model_params.get("norm_eps", 1e-5),
        "dropout": entropy_model_params.get("dropout", 0.0),
        "rope_theta": entropy_model_params.get("rope_theta", 10000.0),
        "attn_impl": entropy_model_params.get("attn_impl", "sdpa"),
        "attn_bias_type": entropy_model_params.get("attn_bias_type", "causal"),
        "intermediate_size": patcher_intermediate_size,
    }

    # Create encoder config
    encoder_hidden_size = unified_config.get("dim_local_encoder", 1024)
    encoder_multiple_of = unified_config.get("multiple_of", 256)
    encoder_intermediate_size = encoder_multiple_of * ((int(8 * encoder_hidden_size / 3) + encoder_multiple_of - 1) // encoder_multiple_of)
    
    encoder_config = {
        "vocab_size": unified_config.get("vocab_size", 256),
        "cross_attn_all_layers": unified_config.get("cross_attn_all_layers_encoder", False),
        "cross_attn_k": unified_config.get("cross_attn_k", 2),
        "hidden_size_global": unified_config.get("hidden_size_global", 2048),
        "pm_size": unified_config.get("pm_size", 0),
        "hidden_size": encoder_hidden_size,
        "num_attention_heads": unified_config.get("n_heads_local_encoder", 16),
        "num_key_value_heads": unified_config.get("n_kv_heads"),
        "num_hidden_layers": unified_config.get("n_layers_local_encoder", 1),
        "norm_eps": unified_config.get("norm_eps", 1e-5),
        "dropout": unified_config.get("dropout", 0.0),
        "max_position_embeddings": unified_config.get("max_encoder_seq_length") or unified_config.get("max_seqlen", 1024),
        "rope_theta": unified_config.get("rope_theta", 10000.0),
        "rope_scaling": {"rope_type": "default"},
        "hidden_act": unified_config.get("hidden_act", "silu"),
        "_attn_implementation": unified_config.get("_attn_implementation", "sdpa"),
        "intermediate_size": encoder_intermediate_size,
    }

    # Create decoder config
    decoder_hidden_size = unified_config.get("dim_local_decoder", 1024)
    decoder_multiple_of = unified_config.get("multiple_of", 256)
    decoder_intermediate_size = decoder_multiple_of * ((int(8 * decoder_hidden_size / 3) + decoder_multiple_of - 1) // decoder_multiple_of)
    
    decoder_config = {
        "vocab_size": unified_config.get("vocab_size", 256),
        "cross_attn_all_layers": unified_config.get("cross_attn_all_layers_decoder", False),
        "cross_attn_k": unified_config.get("cross_attn_k", 2),
        "hidden_size_global": unified_config.get("hidden_size_global", 2048),
        "hidden_size": decoder_hidden_size,
        "num_attention_heads": unified_config.get("n_heads_local_decoder", 16),
        "num_key_value_heads": unified_config.get("n_kv_heads"),
        "num_hidden_layers": unified_config.get("n_layers_local_decoder", 9),
        "norm_eps": unified_config.get("norm_eps", 1e-5),
        "dropout": unified_config.get("dropout", 0.0),
        "max_position_embeddings": unified_config.get("max_encoder_seq_length") or unified_config.get("max_seqlen", 1024),
        "rope_theta": unified_config.get("rope_theta", 10000.0),
        "rope_scaling": {"rope_type": "default"},
        "hidden_act": unified_config.get("hidden_act", "silu"),
        "_attn_implementation": unified_config.get("_attn_implementation", "sdpa"),
        "intermediate_size": decoder_intermediate_size,
    }

    # Create global transformer config
    global_hidden_size = unified_config.get("dim_global", 2048)
    global_multiple_of = unified_config.get("multiple_of", 256)
    global_intermediate_size = global_multiple_of * ((int(8 * global_hidden_size / 3) + global_multiple_of - 1) // global_multiple_of)
    
    global_config = {
        "hidden_size": global_hidden_size,
        "num_attention_heads": unified_config.get("n_heads_global", 16),
        "num_key_value_heads": unified_config.get("n_kv_heads_global"),
        "num_hidden_layers": unified_config.get("n_layers_global", 25),
        "norm_eps": unified_config.get("norm_eps", 1e-5),
        "dropout": unified_config.get("dropout", 0.0),
        "max_position_embeddings": unified_config.get("max_seqlen", 1024),
        "rope_theta": unified_config.get("rope_theta", 10000.0),
        "rope_scaling": {"rope_type": "default"},
        "hidden_act": unified_config.get("hidden_act", "silu"),
        "_attn_implementation": unified_config.get("_attn_implementation", "sdpa"),
        "intermediate_size": global_intermediate_size,
    }

    # Create main config with sub-configs
    main_config_dict = {
        "model_type": "blt",
        "vocab_size": unified_config.get("vocab_size", 256),
        "max_position_embeddings": unified_config.get("max_seqlen", 1024),
        "patch_in_forward": True,
        "realtime_patching": True,
        "patching_mode": "entropy",
        "patch_size": patch_size,
        "patching_threshold": patcher_args.get("threshold", 0.5),
        "patching_threshold_add": patcher_args.get("threshold_add", 0.0),
        "max_patch_length": patcher_args.get("max_patch_length"),
        "patching_batch_size": patcher_args.get("patching_batch_size", 1),
        "patching_device": patcher_args.get("patching_device", "cuda"),
        "monotonicity": patcher_args.get("monotonicity", False),
        "cross_attn_k": unified_config.get("cross_attn_k", 2),
        "encoder_hash_byte_group_size": unified_config.get("encoder_hash_byte_group_size"),
        "encoder_hash_byte_group_vocab": unified_config.get("encoder_hash_byte_group_vocab", 30000),
        "encoder_hash_byte_group_nb_functions": unified_config.get("encoder_hash_byte_group_nb_functions", 3),
        "pm_size": unified_config.get("pm_size", 0),
        "patcher_config": patcher_config,
        "encoder_config": encoder_config,
        "decoder_config": decoder_config,
        "global_config": global_config,
    }

    main_config_dict["tie_word_embeddings"] = False

    logger.info(f"Merged configuration with {len(main_config_dict)} parameters")
    return main_config_dict


def apply_weight_mapping(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:    
    component_mappings = {
        ".attention.": ".self_attn.",
        ".feed_forward.": ".mlp.",
        ".attention_norm.": ".input_layernorm.",
        ".ffn_norm.": ".post_attention_layernorm.",
        ".tok_embeddings.": ".embed_tokens.",
        ".cross_attn_norm_q.": ".q_norm.",
        ".cross_attn_norm_kv.": ".k_norm.",
        ".w1.": ".gate_proj.",
        ".w2.": ".down_proj.",
        ".w3.": ".up_proj.",
        ".wq.": ".q_proj.",
        ".wk.": ".k_proj.",
        ".wv.": ".v_proj.",
        ".wo.": ".o_proj.",
        ".output.": ".lm_head.",
    }
    
    new_state_dict = {}
    
    for old_key, tensor in state_dict.items():
        new_key = old_key
        
        for old_pattern, new_pattern in component_mappings.items():
            if old_pattern in new_key:
                new_key = new_key.replace(old_pattern, new_pattern)
        
        new_state_dict[new_key] = tensor
    
    return new_state_dict


def merge_weights(weights_path: str, entropy_weights_path: str) -> Dict[str, torch.Tensor]:
    main_weights = load_file(weights_path)

    entropy_weights = torch.load(entropy_weights_path, map_location="cpu", weights_only=True)

    if "model" in entropy_weights:
        entropy_weights = entropy_weights["model"]
    elif "state_dict" in entropy_weights:
        entropy_weights = entropy_weights["state_dict"]

    unified_weights = main_weights.copy()

    for key, tensor in entropy_weights.items():
        patcher_key = f"patcher.{key}"
        unified_weights[patcher_key] = tensor
    
    unified_weights = apply_weight_mapping(unified_weights)
    
    decoder_lm_head_key = "local_decoder.lm_head.weight"
    top_lm_head_key = "lm_head.weight"
    unified_weights[top_lm_head_key] = unified_weights[decoder_lm_head_key]
    del unified_weights[decoder_lm_head_key]

    prefixed_weights = {}
    for key, tensor in unified_weights.items():
        if key == top_lm_head_key:
            prefixed_weights[key] = tensor
        elif not key.startswith("model."):
            prefixed_weights[f"model.{key}"] = tensor
        else:
            prefixed_weights[key] = tensor
    
    unified_weights = prefixed_weights
    
    return unified_weights


def create_tokenizer_config(output_dir: str, config: Dict[str, Any]):
    tokenizer_config = {
        "tokenizer_class": "BltTokenizer",
        "vocab_size": config.get("vocab_size", 256),
        "model_max_length": config.get("max_seqlen", 1024),
        "add_bos_token": True,
        "add_eos_token": True,
        "bos_token": "<s>",
        "eos_token": "</s>",
        "pad_token": "<pad>",
        "unk_token": "<unk>",
    }

    tokenizer_path = os.path.join(output_dir, "tokenizer_config.json")
    with open(tokenizer_path, "w") as f:
        json.dump(tokenizer_config, f, indent=2)


def push_to_hub(
    local_dir: str,
    repo_id: str,
    commit_message: str = "Upload converted BLT model",
    private: bool = False,
    token: Optional[str] = None,
) -> None:
    try:
        upload_folder(
            folder_path=local_dir,
            repo_id=repo_id,
            commit_message=commit_message,
            repo_type="model",
            token=token,
        )
        logger.info(f"Successfully pushed model to {repo_id}")

    except Exception as e:
        logger.error(f"Failed to push model to Hub: {e}")
        raise


def convert_hf_blt_to_unified(
    model_id: str,
    output_dir: str,
    config_name: str = "config.json",
    weights_name: str = "model.bin",
    cache_dir: Optional[str] = None,
    push_to_hub_repo: Optional[str] = None,
    hub_private: bool = False,
    hub_token: Optional[str] = None,
) -> None:
    # Download model files
    config_path = hf_hub_download(repo_id=model_id, filename="config.json", cache_dir=cache_dir)
    weights_path = hf_hub_download(repo_id=model_id, filename="model.safetensors", cache_dir=cache_dir)
    entropy_params_path = hf_hub_download(repo_id=model_id, filename="entropy_model/params.json", cache_dir=cache_dir)
    entropy_weights_path = hf_hub_download(
        repo_id=model_id, filename="entropy_model/consolidated.pth", cache_dir=cache_dir
    )

    unified_config = merge_configurations(config_path, entropy_params_path)
    unified_weights = merge_weights(weights_path, entropy_weights_path)

    os.makedirs(output_dir, exist_ok=True)

    config_path = os.path.join(output_dir, config_name)
    with open(config_path, "w") as f:
        json.dump(unified_config, f, indent=2)

    if weights_name.endswith(".bin"):
        weights_name = weights_name.replace(".bin", ".safetensors")

    weights_path = os.path.join(output_dir, weights_name)
    save_file(unified_weights, weights_path)

    create_tokenizer_config(output_dir, unified_config)

    logger.info(f"Conversion completed, model saved to: {output_dir}")

    if push_to_hub_repo:
        push_to_hub(
            local_dir=output_dir,
            repo_id=push_to_hub_repo,
            commit_message="Upload BLT model converted",
            private=hub_private,
            token=hub_token,
        )


def main():
    parser = argparse.ArgumentParser(
        description="Convert BLT models from HuggingFace Hub format to unified format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--model_id",
        type=str,
        default="facebook/blt-1b",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./blt_converted",
    )
    parser.add_argument(
        "--config_name",
        type=str,
        default="config.json",
    )
    parser.add_argument(
        "--weights_name",
        type=str,
        default="model.bin",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        default=True,
    )
    parser.add_argument(
        "--push_to_hub",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--hub_private",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--hub_token",
        type=str,
        default="hf_token",
    )

    args = parser.parse_args()

    transformers_logging.set_verbosity_debug()
    logging.basicConfig(level=logging.DEBUG)

    try:
        convert_hf_blt_to_unified(
            model_id=args.model_id,
            output_dir=args.output_dir,
            config_name=args.config_name,
            weights_name=args.weights_name,
            cache_dir=args.cache_dir,
            push_to_hub_repo=args.push_to_hub,
            hub_private=args.hub_private,
            hub_token=args.hub_token,
        )
    except Exception as e:
        logger.error(f"Conversion failed: {e}")
        raise


if __name__ == "__main__":
    main()
