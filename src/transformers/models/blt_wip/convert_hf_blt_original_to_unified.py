import argparse
import json
import logging
import os
from typing import Dict, Any, Optional

import torch
from huggingface_hub import hf_hub_download, snapshot_download
from safetensors.torch import load_file, save_file

from transformers.utils import logging as transformers_logging

logger = transformers_logging.get_logger(__name__)
transformers_logging.set_verbosity_info()

# For standalone execution, we'll skip the model validation to avoid import issues
# The script will create the unified config and weights files without testing model instantiation
ENABLE_MODEL_VALIDATION = False

import sys
import os

from transformers.models.blt_wip.modeling_blt_wip import BLTModel
from transformers.models.blt_wip.configuration_blt import BLTConfig


ENABLE_MODEL_VALIDATION = True

def download_model_files(model_id: str, cache_dir: Optional[str] = None) -> Dict[str, str]:
    """
    Download all necessary files from HuggingFace Hub.
    
    Args:
        model_id: HuggingFace model ID (e.g., "facebook/blt-1b")
        cache_dir: Optional cache directory
        
    Returns:
        Dictionary with paths to downloaded files
    """
    logger.info(f"Downloading model files from {model_id}...")
    
    try:
        # Download main config
        config_path = hf_hub_download(
            repo_id=model_id,
            filename="config.json",
            cache_dir=cache_dir
        )
        
        # Download main model weights
        weights_path = hf_hub_download(
            repo_id=model_id,
            filename="model.safetensors",
            cache_dir=cache_dir
        )
        
        # Download entropy model params
        entropy_params_path = hf_hub_download(
            repo_id=model_id,
            filename="entropy_model/params.json",
            cache_dir=cache_dir
        )
        
        # Download entropy model weights
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
        
    except Exception as e:
        logger.error(f"Failed to download files from {model_id}: {e}")
        raise


def merge_configurations(config_path: str, entropy_params_path: str) -> Dict[str, Any]:
    """
    Merge main configuration with entropy model parameters.
    
    Args:
        config_path: Path to main config.json
        entropy_params_path: Path to entropy_model/params.json
        
    Returns:
        Merged configuration dictionary
    """
    logger.info("Merging configurations...")
    
    # Load main configuration
    with open(config_path, 'r') as f:
        main_config = json.load(f)
    
    # Load entropy model parameters
    with open(entropy_params_path, 'r') as f:
        entropy_data = json.load(f)
    
    # Extract entropy model and patcher parameters
    entropy_model_params = entropy_data.get("entropy_model", {})
    patcher_args = entropy_data.get("data", {}).get("patcher_args", {})
    
    # Create unified configuration
    unified_config = main_config.copy()
    
    # Ensure required main model parameters are present with correct types
    # Sometimes the original config may have different key names
    if "vocab_size" not in unified_config:
        unified_config["vocab_size"] = int(main_config.get("vocab_size", 256))
    if "dim" not in unified_config:
        unified_config["dim"] = int(main_config.get("dim", main_config.get("hidden_size", main_config.get("d_model", 512))))
    if "n_layers" not in unified_config:
        unified_config["n_layers"] = int(main_config.get("n_layers", main_config.get("num_layers", main_config.get("num_hidden_layers", 8))))
    if "n_heads" not in unified_config:
        unified_config["n_heads"] = int(main_config.get("n_heads", main_config.get("num_attention_heads", main_config.get("num_heads", 8))))
    if "max_seqlen" not in unified_config:
        unified_config["max_seqlen"] = int(main_config.get("max_seqlen", main_config.get("max_position_embeddings", main_config.get("seq_length", 1024))))
    
    # Ensure other integer parameters are properly typed
    for key in ["vocab_size", "dim", "n_layers", "n_heads", "max_seqlen"]:
        if key in unified_config and not isinstance(unified_config[key], int):
            unified_config[key] = int(unified_config[key])
    
    # Convert all patch_size values to integers to avoid float/int type errors
    patch_size = patcher_args.get("patch_size", 8)
    if isinstance(patch_size, float):
        patch_size = int(patch_size)
    
    # Add patching configuration
    unified_config.update({
        "patch_in_forward": True,
        "realtime_patching": True,
        "patching_mode": "entropy",
        
        # Patcher arguments
        "patch_size": patch_size,
        "patching_threshold": patcher_args.get("threshold", 0.5),
        "patching_threshold_add": patcher_args.get("threshold_add", 0.0),
        "max_patch_length": patcher_args.get("max_patch_length"),
        "patching_batch_size": patcher_args.get("patching_batch_size", 1),
        "patching_device": patcher_args.get("patching_device", "cuda"),
        "monotonicity": patcher_args.get("monotonicity", False),
        
        # Entropy model (patcher) architecture parameters
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
    """
    Merge main model weights with entropy model weights.
    
    Args:
        weights_path: Path to main model.safetensors
        entropy_weights_path: Path to entropy_model/consolidated.pth
        
    Returns:
        Merged state dictionary
    """
    logger.info("Merging model weights...")
    
    # Load main model weights
    main_weights = load_file(weights_path)
    logger.info(f"Loaded main model weights: {len(main_weights)} tensors")
    
    # Load entropy model weights
    entropy_weights = torch.load(entropy_weights_path, map_location='cpu', weights_only=True)
    
    # Handle nested entropy model structure
    if 'model' in entropy_weights:
        entropy_weights = entropy_weights['model']
    elif 'state_dict' in entropy_weights:
        entropy_weights = entropy_weights['state_dict']
    
    logger.info(f"Loaded entropy model weights: {len(entropy_weights)} tensors")
    
    # Create unified state dict
    unified_weights = main_weights.copy()
    
    # Add entropy model weights with "patcher." prefix
    for key, tensor in entropy_weights.items():
        patcher_key = f"patcher.{key}"
        unified_weights[patcher_key] = tensor
    
    logger.info(f"Merged weights: {len(unified_weights)} tensors total")
    return unified_weights


def create_tokenizer_config(output_dir: str, config: Dict[str, Any]):
    """
    Create tokenizer configuration file.
    
    Args:
        output_dir: Output directory
        config: Model configuration
    """
    logger.info("Creating tokenizer configuration...")
    
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
    """
    Validate the unified model configuration and weights.
    
    Args:
        config: Unified configuration
        weights: Unified weights
    """
    logger.info("Validating unified model...")
    
    # Check required configuration keys
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
    
    # Check for main model weights
    main_weights = [key for key in weights.keys() if not key.startswith("patcher.")]
    logger.info(f"Found {len(main_weights)} main model weight tensors")
    
    # Try to create the model with the configuration (if imports are available)
    if ENABLE_MODEL_VALIDATION and BLTConfig is not None and BLTModel is not None:
        try:
            logger.info("Testing model instantiation...")
            
            # Debug: Print config keys to help diagnose issues
            logger.debug(f"Config keys: {list(config.keys())}")
            logger.debug(f"Config vocab_size: {config.get('vocab_size')} (type: {type(config.get('vocab_size'))})")
            logger.debug(f"Config dim: {config.get('dim')} (type: {type(config.get('dim'))})")
            
            blt_config = BLTConfig(**config)
            model = BLTModel(blt_config)
            logger.info("✓ Model instantiation successful")
            
            # Try to load the weights
            logger.info("Testing weight loading...")
            try:
                missing_keys, unexpected_keys = model.load_state_dict(weights, strict=False)
                if missing_keys:
                    logger.warning(f"Missing keys during weight loading: {missing_keys[:5]}...")  # Show first 5
                if unexpected_keys:
                    logger.warning(f"Unexpected keys during weight loading: {unexpected_keys[:5]}...")  # Show first 5
                logger.info("✓ Weight loading successful")
            except Exception as weight_error:
                logger.warning(f"Weight loading failed: {weight_error}")
                logger.info("Model instantiation successful, but weight loading had issues")
            
        except Exception as e:
            logger.error(f"Model validation failed: {e}")
            logger.debug(f"Full error details:", exc_info=True)
            logger.warning("Model may not be compatible with modeling_blt_wip.py")
            logger.info("You can still use the converted files and test manually")
    else:
        logger.info("Skipping model instantiation test (BLT classes not available)")
        logger.info("You can test the model manually after conversion")
    
    logger.info("Model validation completed")


def convert_hf_blt_to_unified(
    model_id: str,
    output_dir: str,
    config_name: str = "config.json",
    weights_name: str = "pytorch_model.bin",
    safe_serialization: bool = True,
    cache_dir: Optional[str] = None,
    validate: bool = True,
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
        validate: Whether to validate the unified model
    """
    logger.info(f"Converting {model_id} to unified format...")
    
    # Download model files
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
    
    # Validate if requested
    if validate:
        validate_unified_model(unified_config, unified_weights)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Save unified configuration
    config_path = os.path.join(output_dir, config_name)
    with open(config_path, 'w') as f:
        json.dump(unified_config, f, indent=2)
    logger.info(f"Unified config saved to {config_path}")
    
    # Save unified weights
    if safe_serialization and weights_name.endswith('.bin'):
        weights_name = weights_name.replace('.bin', '.safetensors')
    elif not safe_serialization and weights_name.endswith('.safetensors'):
        weights_name = weights_name.replace('.safetensors', '.bin')
    
    weights_path = os.path.join(output_dir, weights_name)
    if safe_serialization:
        save_file(unified_weights, weights_path)
    else:
        torch.save(unified_weights, weights_path)
    logger.info(f"Unified weights saved to {weights_path}")
    
    # Create tokenizer config
    create_tokenizer_config(output_dir, unified_config)
    
    # Create README
    readme_path = os.path.join(output_dir, "README.md")
    with open(readme_path, 'w') as f:
        f.write(f"""# Unified BLT Model

This model was converted from {model_id} to unified format compatible with modeling_blt_wip.py.

## Files

- `{config_name}`: Unified configuration (main config + entropy model params)
- `{weights_name}`: Unified weights (main model + entropy model weights with "patcher." prefix)
- `tokenizer_config.json`: Tokenizer configuration

## Usage

```python
import torch
import json
from modeling_blt_wip import BLTModel, BLTConfig

# Load configuration
with open('{config_name}', 'r') as f:
    config_dict = json.load(f)

config = BLTConfig(**config_dict)

# Load model
model = BLTModel(config)

# Load weights
if '{weights_name}'.endswith('.safetensors'):
    from safetensors.torch import load_file
    state_dict = load_file('{weights_name}')
else:
    state_dict = torch.load('{weights_name}', map_location='cpu')

model.load_state_dict(state_dict, strict=False)
```

## Original Model

Converted from: {model_id}
""")
    
    logger.info(f"Conversion completed! Unified model saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert BLT models from HuggingFace Hub format to unified format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert facebook/blt-1b to unified format
  python convert_hf_blt_to_unified.py \\
      --model_id facebook/blt-1b \\
      --output_dir ./unified_blt_1b

  # Convert with custom file names
  python convert_hf_blt_to_unified.py \\
      --model_id facebook/blt-7b \\
      --output_dir ./unified_blt_7b \\
      --config_name unified_config.json \\
      --weights_name unified_model.safetensors

  # Convert without validation
  python convert_hf_blt_to_unified.py \\
      --model_id facebook/blt-1b \\
      --output_dir ./my_blt \\
      --no_validate
        """
    )
    
    # Required arguments (with defaults for debugging)
    parser.add_argument(
        "--model_id",
        type=str,
        default="facebook/blt-1b",
        help="HuggingFace model ID (e.g., facebook/blt-1b)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./unified_blt_debug",
        help="Output directory for unified model"
    )
    
    # Optional arguments
    parser.add_argument(
        "--config_name",
        type=str,
        default="config.json",
        help="Name for unified config file (default: config.json)"
    )
    parser.add_argument(
        "--weights_name",
        type=str,
        default="pytorch_model.bin",
        help="Name for unified weights file (default: pytorch_model.bin)"
    )
    parser.add_argument(
        "--safe_serialization",
        action="store_true",
        default=True,
        help="Use safetensors format for weights (default: True)"
    )
    parser.add_argument(
        "--no_safe_serialization",
        dest="safe_serialization",
        action="store_false",
        help="Use .bin format instead of safetensors"
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="Cache directory for downloads"
    )
    parser.add_argument(
        "--no_validate",
        dest="validate",
        action="store_false",
        default=True,
        help="Skip model validation"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        default=True,  # Enable debug by default for easier debugging
        help="Enable debug logging"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    if args.debug:
        transformers_logging.set_verbosity_debug()
        logging.basicConfig(level=logging.DEBUG)
    
    # Run conversion
    try:
        convert_hf_blt_to_unified(
            model_id=args.model_id,
            output_dir=args.output_dir,
            config_name=args.config_name,
            weights_name=args.weights_name,
            safe_serialization=args.safe_serialization,
            cache_dir=args.cache_dir,
            validate=args.validate,
        )
    except Exception as e:
        logger.error(f"Conversion failed: {e}")
        raise


if __name__ == "__main__":
    main() 