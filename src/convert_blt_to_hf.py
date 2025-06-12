import argparse
import json
import logging
import os
from typing import Dict, Any, Optional

import torch
from huggingface_hub import hf_hub_download, snapshot_download, upload_folder
from safetensors.torch import load_file, save_file

from transformers.utils import logging as transformers_logging

logger = transformers_logging.get_logger(__name__)
transformers_logging.set_verbosity_info()

from transformers.models.blt_wip.modeling_blt_wip import BLTModel
from transformers.models.blt_wip.configuration_blt import BLTConfig


def download_model_files(model_id: str, cache_dir: Optional[str] = None) -> Dict[str, str]:
    config_path = hf_hub_download(
        repo_id=model_id,
        filename="config.json",
        cache_dir=cache_dir
    )
    
    weights_path = hf_hub_download(
        repo_id=model_id,
        filename="model.safetensors",
        cache_dir=cache_dir
    )
    
    entropy_params_path = hf_hub_download(
        repo_id=model_id,
        filename="entropy_model/params.json",
        cache_dir=cache_dir
    )
    
    entropy_weights_path = hf_hub_download(
        repo_id=model_id,
        filename="entropy_model/consolidated.pth",
        cache_dir=cache_dir
    )
    
    return {
        "config": config_path,
        "weights": weights_path,
        "entropy_params": entropy_params_path,
        "entropy_weights": entropy_weights_path
    }
    

def merge_configurations(config_path: str, entropy_params_path: str) -> Dict[str, Any]:

    logger.info("Merging confi")
    
    # Load BLT configuration
    with open(config_path, 'r') as f:
        main_config = json.load(f)
    
    # Load Patcher entropy model parameters
    with open(entropy_params_path, 'r') as f:
        entropy_data = json.load(f)
    
    entropy_model_params = entropy_data.get("entropy_model", {})
    patcher_args = entropy_data.get("data", {}).get("patcher_args", {})
    
    # Create unified configuration
    unified_config = main_config.copy()['args']
    
    # Ensure other integer parameters are properly typed
    for key in ["vocab_size", "dim", "n_layers", "n_heads", "max_seqlen"]:
        if key in unified_config and not isinstance(unified_config[key], int):
            unified_config[key] = int(unified_config[key])
    
    patch_size = patcher_args.get("patch_size", 8)
    if isinstance(patch_size, float):
        patch_size = int(patch_size)
    
    unified_config.update({
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
        
        "patcher_vocab_size": int(entropy_model_params.get("vocab_size", 256)),
        "patcher_dim": int(entropy_model_params.get("dim", 512)),
        "patcher_n_layers": int(entropy_model_params.get("n_layers", 8)),
        "patcher_n_heads": int(entropy_model_params.get("n_heads", 8)),
        "patcher_head_dim": int(entropy_model_params.get("head_dim")) if entropy_model_params.get("head_dim") is not None else None,
        "patcher_n_kv_heads": int(entropy_model_params.get("n_kv_heads")) if entropy_model_params.get("n_kv_heads") is not None else None,
        "patcher_max_seqlen": int(entropy_model_params.get("max_seqlen", 1024)),
        "patcher_norm_eps": entropy_model_params.get("norm_eps", 1e-5),
        "patcher_dropout": entropy_model_params.get("dropout", 0.0),
        "patcher_sliding_window": int(entropy_model_params.get("sliding_window", 512)) if entropy_model_params.get("sliding_window") is not None else None,
        "patcher_ffn_dim_multiplier": entropy_model_params.get("ffn_dim_multiplier"),
        "patcher_multiple_of": int(entropy_model_params.get("multiple_of", 256)),
        "patcher_rope_theta": entropy_model_params.get("rope_theta", 10000.0),
        "patcher_rope_use_fp32_in_outer_product": entropy_model_params.get("rope_use_fp32_in_outer_product", False),
        "patcher_attn_impl": entropy_model_params.get("attn_impl", "sdpa"),
        "patcher_attn_bias_type": entropy_model_params.get("attn_bias_type", "causal"),
        "patcher_init_base_std": entropy_model_params.get("init_base_std"),
        "patcher_init_std_factor": entropy_model_params.get("init_std_factor", "disabled"),
        "patcher_dim_token_emb": entropy_model_params.get("dim_token_emb"),
        "patcher_weight_tying": entropy_model_params.get("weight_tying", False),
        "patcher_bos_token_id": entropy_model_params.get("bos_token_id", 1),
        "patcher_eos_token_id": entropy_model_params.get("eos_token_id", 2),
    })
    
    logger.info(f"Merged configuration with {len(unified_config)} parameters")
    return unified_config


def merge_weights(weights_path: str, entropy_weights_path: str) -> Dict[str, torch.Tensor]:
    logger.info("Merging model weights")
    
    main_weights = load_file(weights_path)
    logger.info(f"Loaded main model weights: {len(main_weights)} tensors")
    
    entropy_weights = torch.load(entropy_weights_path, map_location='cpu', weights_only=True)
    
    if 'model' in entropy_weights:
        entropy_weights = entropy_weights['model']
    elif 'state_dict' in entropy_weights:
        entropy_weights = entropy_weights['state_dict']
    
    logger.info(f"Loaded entropy model weights: {len(entropy_weights)} tensors")
    
    # unified state dict
    unified_weights = main_weights.copy()
    
    # Add entropy model weights with "patcher." prefix
    for key, tensor in entropy_weights.items():
        patcher_key = f"patcher.{key}"
        unified_weights[patcher_key] = tensor
    
    logger.info(f"Merged weights: {len(unified_weights)} tensors total")
    return unified_weights


def create_tokenizer_config(output_dir: str, config: Dict[str, Any]):
    logger.info("Creating tokenizer config")
    
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
    with open(tokenizer_path, 'w') as f:
        json.dump(tokenizer_config, f, indent=2)
    
    logger.info(f"Tokenizer config saved to {tokenizer_path}")


def validate_unified_model(config: Dict[str, Any], weights: Dict[str, torch.Tensor]):
    logger.info("Validating unified model")
    
    required_keys = [
        "vocab_size", "dim", "n_layers", "n_heads",
        "patch_in_forward", "patcher_vocab_size", "patcher_dim"
    ]
    
    missing_keys = [key for key in required_keys if key not in config]
    if missing_keys:
        logger.warning(f"Missing configuration keys: {missing_keys}")
    
    # Check for patcher weights
    patcher_weights = [key for key in weights.keys() if key.startswith("patcher.")]
    if not patcher_weights:
        logger.warning("No patcher weights found in unified weights")
    else:
        logger.info(f"Found {len(patcher_weights)} patcher weight tensors")
    
    main_weights = [key for key in weights.keys() if not key.startswith("patcher.")]
    logger.info(f"Found {len(main_weights)} main model weight tensors")
    
    try:
        logger.info("Testing model instantiation...")
        blt_config = BLTConfig(**config)
        model = BLTModel(blt_config)

        logger.info("Testing weight loading...")
        try:
            missing_keys, unexpected_keys = model.load_state_dict(weights, strict=False)
            if missing_keys:
                logger.warning(f"Missing keys during weight loading: {missing_keys}")
            if unexpected_keys:
                logger.warning(f"Unexpected keys during weight loading: {unexpected_keys}")
            logger.info("Weight loading successful")
        except Exception as weight_error:
            logger.warning(f"Weight loading failed: {weight_error}")
            logger.info("Model instantiation successful, but weight loading had issues")
        
    except Exception as e:
        logger.error(f"Model validation failed: {e}")
    
    logger.info("Model validation completed")


def push_to_hub(
    local_dir: str,
    repo_id: str,
    commit_message: str = "Upload converted BLT model",
    private: bool = False,
    token: Optional[str] = None,
) -> None:
    """
    Push the converted model to Hugging Face Hub.
    
    Args:
        local_dir: Local directory containing the converted model files
        repo_id: Repository ID on Hugging Face Hub (e.g., "username/model-name")
        commit_message: Commit message for the upload
        private: Whether to create a private repository
        token: Hugging Face authentication token (if not provided, will use cached token)
    """
    logger.info(f"Pushing converted model to Hub: {repo_id}")
    
    try:
        # Upload the entire directory to the Hub
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
    safe_serialization: bool = True,
    cache_dir: Optional[str] = None,
    push_to_hub_repo: Optional[str] = None,
    hub_private: bool = False,
    hub_token: Optional[str] = None,
) -> None:
    """
    Convert BLT model from HuggingFace Hub format to unified format.
    
    Args:
        model_id: HuggingFace model ID (e.g., "facebook/blt-1b")
        output_dir: Output directory for unified model
        config_name: Name for unified config file
        weights_name: Name for unified weights file  
        safe_serialization: Whether to use safetensors format
        cache_dir: Cache directory for downloads
        push_to_hub_repo: Repository ID to push the converted model to (optional)
        hub_private: Whether to create a private repository on the Hub
        hub_token: Hugging Face authentication token
    """
    logger.info(f"Converting {model_id} to unified transformers format")
    
    file_paths = download_model_files(model_id, cache_dir)
    
    # Merge configurations
    unified_config = merge_configurations(
        file_paths["config"], 
        file_paths["entropy_params"]
    )
    
    # Merge weights
    unified_weights = merge_weights(
        file_paths["weights"],
        file_paths["entropy_weights"]
    )
    
    validate_unified_model(unified_config, unified_weights)
    
    os.makedirs(output_dir, exist_ok=True)
    
    config_path = os.path.join(output_dir, config_name)
    with open(config_path, 'w') as f:
        json.dump(unified_config, f, indent=2)

    if safe_serialization and weights_name.endswith('.bin'):
        weights_name = weights_name.replace('.bin', '.safetensors')
    elif not safe_serialization and weights_name.endswith('.safetensors'):
        weights_name = weights_name.replace('.safetensors', '.bin')
    
    weights_path = os.path.join(output_dir, weights_name)
    if safe_serialization:
        save_file(unified_weights, weights_path)
    else:
        torch.save(unified_weights, weights_path)

    logger.info(f"Unified config and weights saved to {weights_path}")
    
    # Create tokenizer config
    create_tokenizer_config(output_dir, unified_config)

    logger.info(f"Conversion completed, model saved to: {output_dir}")
    
    # Push to Hub if requested
    if push_to_hub_repo:
        push_to_hub(
            local_dir=output_dir,
            repo_id=push_to_hub_repo,
            commit_message=f"Upload unified BLT model converted from {model_id}",
            private=hub_private,
            token=hub_token,
        )


def main():
    parser = argparse.ArgumentParser(
        description="Convert BLT models from HuggingFace Hub format to unified format",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "--model_id",
        type=str,
        default="facebook/blt-1b",
        help="HuggingFace model ID (e.g., facebook/blt-1b)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./new_unified_blt_debug",
        help="Output directory for unified model"
    )
    
    # Optional
    parser.add_argument(
        "--config_name",
        type=str,
        default="config.json",
        help="Name for unified config file (default: config.json)"
    )
    parser.add_argument(
        "--weights_name",
        type=str,
        default="model.bin",
    )
    parser.add_argument(
        "--safe_serialization",
        action="store_true",
        default=True,
    )
    parser.add_argument(
        "--no_safe_serialization",
        dest="safe_serialization",
        action="store_false",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        default=True,  # Enable debug by default for easier debugging
    )
    
    # Hub upload arguments
    parser.add_argument(
        "--push_to_hub",
        type=str,
        default="itazap/blt-1b",
    )
    parser.add_argument(
        "--hub_private",
        action="store_true",
        default=False,
        help="Whether to create a private repository on the Hub"
    )
    parser.add_argument(
        "--hub_token",
        type=str,
        default="hf_your_token_here",
        help="Hugging Face authentication token (if not provided, will use cached token)"
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
            safe_serialization=args.safe_serialization,
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