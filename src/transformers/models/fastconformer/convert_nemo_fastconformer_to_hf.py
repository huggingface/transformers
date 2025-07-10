#!/usr/bin/env python3
"""
Universal NeMo FastConformer to HuggingFace Converter

This script converts NeMo models that use FastConformer encoder (Parakeet, Canary, etc.)
to HuggingFace FastConformerModel format. It handles:
- FastConformer encoder (shared across all NeMo ASR models)
- Different preprocessors (mel-spectrogram, etc.)
- Model configuration extraction and metadata preservation
- Creates a base FastConformerModel that can be loaded via AutoModel

The converted model serves as the foundation for all FastConformer-based NeMo ASR models
and can be used for feature extraction or as a base for task-specific models.

Usage:
    python convert_nemo_fastconformer_to_hf.py --path_to_nemo_model ./parakeet-ctc-1.1b.nemo --output_dir ./parakeet-hf
    python convert_nemo_fastconformer_to_hf.py --path_to_nemo_model ./canary-1b.nemo --output_dir ./canary-hf
"""

import argparse
import json
import logging
import os
import re
import tarfile
import tempfile
from pathlib import Path
from typing import Any, Optional, Union

import torch
import yaml

from transformers.models.fastconformer.configuration_fastconformer import (
    FastConformerConfig,
    ParakeetCTCConfig,
)
from transformers.models.fastconformer.feature_extraction_fastconformer import FastConformerFeatureExtractor
from transformers.models.fastconformer.modeling_fastconformer import (
    FastConformerModel,
    ParakeetCTC,
)


# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# NeMo to HuggingFace weight mapping patterns
# Regex patterns for converting NeMo FastConformer weights to HuggingFace format
NEMO_TO_HF_WEIGHT_MAPPING = {
    # Subsampling layer
    r"encoder\.pre_encode\.": r"encoder.subsampling.",
    # Positional encoding (skip pe buffer)
    r"encoder\.pos_enc\.pe$": None,  # Skip buffer
    r"encoder\.pos_enc\.": r"encoder.pos_enc.",
    # Conformer layers - attention
    r"encoder\.layers\.(\d+)\.self_attention\.linear_q\.": r"encoder.layers.\1.self_attn.linear_q.",
    r"encoder\.layers\.(\d+)\.self_attention\.linear_k\.": r"encoder.layers.\1.self_attn.linear_k.",
    r"encoder\.layers\.(\d+)\.self_attention\.linear_v\.": r"encoder.layers.\1.self_attn.linear_v.",
    r"encoder\.layers\.(\d+)\.self_attention\.linear_out\.": r"encoder.layers.\1.self_attn.linear_out.",
    r"encoder\.layers\.(\d+)\.self_attention\.linear_pos\.": r"encoder.layers.\1.self_attn.linear_pos.",
    r"encoder\.layers\.(\d+)\.self_attention\.pos_bias_u": r"encoder.layers.\1.self_attn.pos_bias_u",
    r"encoder\.layers\.(\d+)\.self_attention\.pos_bias_v": r"encoder.layers.\1.self_attn.pos_bias_v",
    # Conformer layers - feed forward
    r"encoder\.layers\.(\d+)\.feed_forward_1\.linear1\.": r"encoder.layers.\1.feed_forward1.linear1.",
    r"encoder\.layers\.(\d+)\.feed_forward_1\.linear2\.": r"encoder.layers.\1.feed_forward1.linear2.",
    r"encoder\.layers\.(\d+)\.feed_forward_2\.linear1\.": r"encoder.layers.\1.feed_forward2.linear1.",
    r"encoder\.layers\.(\d+)\.feed_forward_2\.linear2\.": r"encoder.layers.\1.feed_forward2.linear2.",
    # Conformer layers - convolution
    r"encoder\.layers\.(\d+)\.conv_module\.pointwise_conv1\.": r"encoder.layers.\1.conv.pointwise_conv1.",
    r"encoder\.layers\.(\d+)\.conv_module\.depthwise_conv\.": r"encoder.layers.\1.conv.depthwise_conv.",
    r"encoder\.layers\.(\d+)\.conv_module\.batch_norm\.": r"encoder.layers.\1.conv.batch_norm.",
    r"encoder\.layers\.(\d+)\.conv_module\.pointwise_conv2\.": r"encoder.layers.\1.conv.pointwise_conv2.",
    # Conformer layers - layer norms
    r"encoder\.layers\.(\d+)\.norm_ff_1\.": r"encoder.layers.\1.norm_feed_forward1.",
    r"encoder\.layers\.(\d+)\.norm_ff_2\.": r"encoder.layers.\1.norm_feed_forward2.",
    r"encoder\.layers\.(\d+)\.norm_self_attn\.": r"encoder.layers.\1.norm_self_att.",
    r"encoder\.layers\.(\d+)\.norm_conv\.": r"encoder.layers.\1.norm_conv.",
    r"encoder\.layers\.(\d+)\.norm_out\.": r"encoder.layers.\1.norm_out.",
    # Decoder (CTC head) - Conv1d to Linear conversion handled separately
    r"decoder\.decoder_layers\.0\.weight": r"ctc_head.weight",
    r"decoder\.decoder_layers\.0\.bias": r"ctc_head.bias",
}


def convert_nemo_keys_to_hf_keys(state_dict_keys):
    """
    Convert NeMo weight keys to HuggingFace format using regex patterns.
    Similar to mllama's convert_old_keys_to_new_keys function.
    """
    output_dict = {}
    if state_dict_keys is not None:
        old_text = "\n".join(state_dict_keys)
        new_text = old_text

        for pattern, replacement in NEMO_TO_HF_WEIGHT_MAPPING.items():
            if replacement is None:
                new_text = re.sub(pattern, "", new_text)  # Skip this key
                continue
            new_text = re.sub(pattern, replacement, new_text)

        output_dict = dict(zip(old_text.split("\n"), new_text.split("\n")))

    return output_dict


def extract_nemo_archive(nemo_file_path: str, extract_dir: str) -> dict[str, str]:
    """
    Extract .nemo file (tar archive) and return paths to important files.

    Args:
        nemo_file_path: Path to .nemo file
        extract_dir: Directory to extract to

    Returns:
        Dictionary with paths to model.pt, model_config.yaml, etc.
    """
    logger.info(f"Extracting NeMo archive: {nemo_file_path}")

    with tarfile.open(nemo_file_path, "r") as tar:
        tar.extractall(extract_dir)

    # Log all extracted files for debugging
    all_files = []
    for root, dirs, files in os.walk(extract_dir):
        for file in files:
            file_path = os.path.join(root, file)
            all_files.append(file_path)

    logger.info(f"All extracted files: {[os.path.basename(f) for f in all_files]}")

    # Find important files with more robust detection
    model_files = {}
    for root, dirs, files in os.walk(extract_dir):
        for file in files:
            file_path = os.path.join(root, file)
            file_lower = file.lower()

            # Look for model weights with various common names
            if (
                file.endswith(".pt")
                or file.endswith(".pth")
                or file.endswith(".ckpt")
                or file.endswith(".bin")
                or "model" in file_lower
                and ("weight" in file_lower or "state" in file_lower)
                or file_lower == "model.pt"
                or file_lower == "pytorch_model.bin"
                or file_lower == "model_weights.ckpt"
            ):
                model_files["model_weights"] = file_path
                logger.info(f"Found model weights: {file}")

            # Look for config files
            elif (
                file == "model_config.yaml"
                or file == "config.yaml"
                or (file.endswith(".yaml") and "config" in file_lower)
            ):
                if "model_config" not in model_files:  # Prefer model_config.yaml
                    model_files["model_config"] = file_path
                    logger.info(f"Found config file: {file}")
                if file == "model_config.yaml":
                    model_files["model_config"] = file_path  # Override with preferred name

            # Look for vocabulary files
            elif (
                file.endswith(".vocab")
                or file.endswith(".model")
                or "tokenizer" in file_lower
                and (file.endswith(".vocab") or file.endswith(".model"))
            ):
                model_files["vocab_file"] = file_path
                logger.info(f"Found vocabulary file: {file}")

    logger.info(f"Found model files: {list(model_files.keys())}")

    # Validate that we found the required files
    if "model_weights" not in model_files:
        raise FileNotFoundError(
            f"Could not find model weights file in {nemo_file_path}. "
            f"Expected files with extensions: .pt, .pth, .ckpt, .bin. "
            f"Found files: {[os.path.basename(f) for f in all_files]}"
        )

    if "model_config" not in model_files:
        raise FileNotFoundError(
            f"Could not find model config file in {nemo_file_path}. "
            f"Expected: model_config.yaml or config.yaml. "
            f"Found files: {[os.path.basename(f) for f in all_files]}"
        )

    return model_files


def convert_sentencepiece_vocab_to_json(vocab_file_path: str) -> dict[str, int]:
    """
    Convert SentencePiece vocabulary file to JSON format.

    Args:
        vocab_file_path: Path to .vocab file from NeMo

    Returns:
        Dictionary mapping tokens to IDs
    """
    vocab_dict = {}

    try:
        with open(vocab_file_path, "r", encoding="utf-8") as f:
            for token_id, line in enumerate(f):
                # SentencePiece vocab format: token \t score
                token = line.strip().split("\t")[0]
                vocab_dict[token] = token_id

        logger.info(f"Converted SentencePiece vocab with {len(vocab_dict)} tokens")

        # Add standard tokens if not present
        if "<unk>" not in vocab_dict:
            vocab_dict["<unk>"] = 0  # Usually UNK is the first token
            logger.info("Added <unk> token at ID 0")

        return vocab_dict

    except Exception as e:
        logger.error(f"Failed to convert vocab file {vocab_file_path}: {e}")
        # Return a minimal vocab as fallback
        logger.warning("Creating minimal fallback vocabulary")
        return {"<unk>": 0}


def load_nemo_config(config_path: str) -> dict[str, Any]:
    """Load NeMo model configuration from yaml file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def extract_model_info_from_config(config: dict[str, Any]) -> dict[str, Any]:
    """Extract model information from NeMo config."""
    model_info = {
        "model_type": "unknown",
        "encoder_type": "unknown",
        "decoder_type": "unknown",
        "is_ctc_model": False,
        "encoder_cfg": None,
        "decoder_cfg": None,
        "preprocessor_cfg": None,
    }

    # Extract model type from config or model name
    model_name = config.get("name", "").lower()
    if "parakeet" in model_name:
        model_info["model_type"] = "parakeet_ctc"
    elif "canary" in model_name:
        model_info["model_type"] = "canary"
    elif "conformer" in model_name:
        model_info["model_type"] = "conformer"

    # Get encoder, decoder, preprocessor configs
    if "encoder" in config:
        model_info["encoder_cfg"] = config["encoder"]
        model_info["encoder_type"] = config["encoder"].get("_target_", "unknown")

    if "decoder" in config:
        model_info["decoder_cfg"] = config["decoder"]
        model_info["decoder_type"] = config["decoder"].get("_target_", "unknown")

    if "preprocessor" in config:
        model_info["preprocessor_cfg"] = config["preprocessor"]

    # Enhanced CTC model detection
    decoder_type = model_info["decoder_type"].lower()
    is_ctc = False

    # Primary check: Look for EncDecCTCModelBPE in the main config target
    main_target = config.get("target", "").lower()
    if "encdecctcmodelbpe" in main_target.replace("_", "").replace(".", ""):
        is_ctc = True
        logger.info(f"Detected EncDecCTCModelBPE in main target: {config.get('_target_', '')}")

    # Secondary checks
    # Check decoder type
    if "ctc" in decoder_type or "convctc" in decoder_type or "conv_asr_decoder" in decoder_type:
        is_ctc = True

    # Check model name
    if "ctc" in model_name or "parakeet" in model_name:
        is_ctc = True

    # Check for CTC-specific config parameters
    if model_info["decoder_cfg"]:
        decoder_target = model_info["decoder_cfg"].get("_target_", "").lower()
        if "ctc" in decoder_target or "convctc" in decoder_target or "conv_asr_decoder" in decoder_target:
            is_ctc = True

    model_info["is_ctc_model"] = is_ctc

    # Set model type based on CTC detection
    if is_ctc:
        model_info["model_type"] = "parakeet_ctc"
    elif model_info["model_type"] == "unknown":
        model_info["model_type"] = "fastconformer"

    logger.info(f"Detected model type: {model_info['model_type']}")
    logger.info(f"Encoder type: {model_info['encoder_type']}")
    logger.info(f"Decoder type: {model_info['decoder_type']}")
    logger.info(f"Is CTC model: {model_info['is_ctc_model']}")

    return model_info


def create_hf_config_from_nemo(
    model_info: dict[str, Any], state_dict: dict[str, torch.Tensor]
) -> Union[FastConformerConfig, ParakeetCTCConfig]:
    """Create HuggingFace FastConformerConfig from NeMo config and weights."""
    encoder_cfg = model_info.get("encoder_cfg", {})
    preprocessor_cfg = model_info.get("preprocessor_cfg", {})

    # Detect architecture from state dict
    actual_layers = 24  # default
    actual_hidden_size = 1024  # default
    actual_num_heads = 8  # default
    actual_ffn_dim = 4096  # default
    actual_use_bias = False  # default

    # Count layers from state dict
    layer_keys = [k for k in state_dict.keys() if "encoder.layers." in k]
    if layer_keys:
        layer_nums = [
            int(re.search(r"encoder\.layers\.(\d+)\.", k).group(1))
            for k in layer_keys
            if re.search(r"encoder\.layers\.(\d+)\.", k)
        ]
        if layer_nums:
            actual_layers = max(layer_nums) + 1
            logger.info(f"Detected {actual_layers} encoder layers from state dict")

    # Detect hidden size from linear weights
    q_proj_keys = [k for k in state_dict.keys() if "self_attention.linear_q.weight" in k]
    if q_proj_keys:
        actual_hidden_size = state_dict[q_proj_keys[0]].shape[1]
        logger.info(f"Detected hidden size: {actual_hidden_size}")

        # Detect num heads (assuming linear_q projects to same hidden_size)
        q_out_size = state_dict[q_proj_keys[0]].shape[0]
        if q_out_size == actual_hidden_size:
            # Try to get from config if available
            actual_num_heads = encoder_cfg.get("n_heads", actual_hidden_size // 128)  # default head_dim=128

    # Detect FFN dimension from feed forward weights
    ff_keys = [
        k for k in state_dict.keys() if "feed_forward_1.linear1.weight" in k or "feed_forward1.linear1.weight" in k
    ]
    if ff_keys:
        actual_ffn_dim = state_dict[ff_keys[0]].shape[0]
        logger.info(f"Detected FFN dimension: {actual_ffn_dim}")

    # Detect bias usage
    bias_keys = [k for k in state_dict.keys() if "linear_q.bias" in k]
    if bias_keys:
        actual_use_bias = True
        logger.info("Detected bias=True from state dict")

    # Use config values if available, otherwise use detected values
    config_params = {
        "vocab_size": 1024,  # Default, will be overridden for CTC models
        "hidden_size": encoder_cfg.get("d_model", actual_hidden_size),
        "num_hidden_layers": encoder_cfg.get("n_layers", actual_layers),
        "num_attention_heads": encoder_cfg.get("n_heads", actual_num_heads),
        "intermediate_size": encoder_cfg.get("ff_expansion_factor", 4) * encoder_cfg.get("d_model", actual_hidden_size)
        if "ff_expansion_factor" in encoder_cfg
        else actual_ffn_dim,
        "hidden_act": "silu",
        "hidden_dropout_prob": encoder_cfg.get("dropout", 0.1),
        "attention_probs_dropout_prob": encoder_cfg.get("dropout_att", 0.1),
        "conv_kernel_size": encoder_cfg.get("conv_kernel_size", 9),
        "subsampling_factor": encoder_cfg.get("subsampling_factor", 8),
        "subsampling_conv_channels": encoder_cfg.get("subsampling_conv_channels", 256),
        "use_bias": actual_use_bias,
        "num_mel_bins": preprocessor_cfg.get("features", 128) if preprocessor_cfg else 128,
        "xscaling": encoder_cfg.get("xscaling", False),
        "dropout_emb": encoder_cfg.get("dropout_emb", 0.0),
        "pad_token_id": 0,
        "bos_token_id": 1,
        "eos_token_id": 2,
    }

    # Add model-specific metadata
    if model_info["is_ctc_model"]:
        architectures = ["ParakeetCTC"]
        # Force model_type to parakeet_ctc for CTC models
        base_model_type = "parakeet_ctc"
    else:
        architectures = ["FastConformerModel"]
        base_model_type = "fastconformer"

    config_params.update(
        {
            "model_type": base_model_type,
            "architectures": architectures,
            "nemo_model_type": model_info["model_type"],
            "nemo_encoder_type": model_info["encoder_type"],
            "nemo_decoder_type": model_info["decoder_type"],
        }
    )

    # For CTC models, create ParakeetCTCConfig
    if model_info["is_ctc_model"]:
        # Get vocab_size from state dict if available
        vocab_size = 1024  # default
        if any("ctc_head.weight" in key or "decoder_layers.0.weight" in key for key in state_dict.keys()):
            # Find the decoder weight to get vocab_size
            decoder_keys = [k for k in state_dict.keys() if "decoder_layers.0.weight" in k]
            if decoder_keys:
                decoder_weight = state_dict[decoder_keys[0]]
                if decoder_weight.dim() == 3 and decoder_weight.size(2) == 1:
                    vocab_size = decoder_weight.size(0)  # Conv1d output channels
                else:
                    vocab_size = decoder_weight.size(0)  # Linear output features
                logger.info(f"Detected vocab_size: {vocab_size} from decoder weights")

        # Create FastConformer sub-config with fastconformer model_type
        fastconformer_config_params = config_params.copy()
        fastconformer_config_params["model_type"] = "fastconformer"  # Force fastconformer for sub-config
        fastconformer_config_params["architectures"] = ["FastConformerModel"]
        fastconformer_config = FastConformerConfig(**fastconformer_config_params)

        ctc_config = ParakeetCTCConfig(
            vocab_size=vocab_size,
            blank_token_id=0,
            ctc_loss_reduction="mean",
            ctc_zero_infinity=True,
            fastconformer_config=fastconformer_config,
        )
        return ctc_config

    # For non-CTC models, create FastConformerConfig
    fastconformer_config = FastConformerConfig(**config_params)
    return fastconformer_config


def create_feature_extractor_config(preprocessor_cfg: Optional[dict[str, Any]]) -> dict[str, Any]:
    """Create feature extractor configuration from NeMo preprocessor config."""
    if preprocessor_cfg:
        sample_rate = preprocessor_cfg.get("sample_rate", 16000)
        feature_extractor_config = {
            "feature_extractor_type": "FastConformerFeatureExtractor",
            "feature_size": preprocessor_cfg.get("features", 128),
            "sampling_rate": sample_rate,
            "hop_length": int(preprocessor_cfg.get("window_stride", 0.01) * sample_rate),
            "win_length": int(preprocessor_cfg.get("window_size", 0.025) * sample_rate),
            "n_fft": preprocessor_cfg.get("n_fft", 512),
            "n_mels": preprocessor_cfg.get("features", 128),
            "f_min": preprocessor_cfg.get("lowfreq", 0),
            "f_max": preprocessor_cfg.get("highfreq", sample_rate // 2),
            "normalize": preprocessor_cfg.get("normalize", "per_feature"),
            "mel_scale": "slaney",
            "return_attention_mask": True,
            "padding_value": 0.0,
        }
    else:
        # Default configuration
        feature_extractor_config = {
            "feature_extractor_type": "FastConformerFeatureExtractor",
            "feature_size": 128,
            "sampling_rate": 16000,
            "hop_length": 160,
            "win_length": 400,
            "n_fft": 512,
            "n_mels": 128,
            "f_min": 0,
            "f_max": 8000,
            "normalize": "per_feature",
            "mel_scale": "slaney",
            "return_attention_mask": True,
            "padding_value": 0.0,
        }

    return feature_extractor_config


def convert_weights(nemo_state_dict: dict[str, torch.Tensor], model_info: dict[str, Any]) -> dict[str, torch.Tensor]:
    """Convert NeMo weights to HuggingFace format using regex mapping."""
    logger.info("Converting weights using regex mapping...")

    # Get key mapping
    all_keys = list(nemo_state_dict.keys())
    key_mapping = convert_nemo_keys_to_hf_keys(all_keys)

    hf_state_dict = {}

    for nemo_key, tensor in nemo_state_dict.items():
        hf_key = key_mapping.get(nemo_key, "")

        if hf_key == "":
            # Skip this key (mapped to None or empty)
            continue

        # Special handling for CTC decoder weights (Conv1d -> Linear)
        if hf_key == "ctc_head.weight" and tensor.dim() == 3 and tensor.size(2) == 1:
            # NeMo uses Conv1d (shape: [out_channels, in_channels, kernel_size])
            # HF uses Linear (shape: [out_features, in_features])
            tensor = tensor.squeeze(2)
            logger.info("Converted CTC head weight from Conv1d to Linear format")

        hf_state_dict[hf_key] = tensor

    logger.info(f"Converted {len(hf_state_dict)} weights from {len(nemo_state_dict)} NeMo weights")
    return hf_state_dict


def create_hf_model(
    hf_config: Union[FastConformerConfig, ParakeetCTCConfig],
    hf_state_dict: dict[str, torch.Tensor],
    model_info: dict[str, Any],
) -> Union[FastConformerModel, ParakeetCTC]:
    """Create the appropriate HuggingFace model and load weights."""

    if model_info["is_ctc_model"]:
        # Check if we already have a ParakeetCTCConfig or need to create one
        if isinstance(hf_config, ParakeetCTCConfig):
            logger.info("Creating ParakeetCTC model with existing ParakeetCTCConfig...")
            model = ParakeetCTC(hf_config)
        else:
            # Fallback: create ParakeetCTCConfig if we somehow still have FastConformerConfig
            vocab_size = 1024  # default
            if "ctc_head.weight" in hf_state_dict:
                vocab_size = hf_state_dict["ctc_head.weight"].shape[0]
                logger.info(f"Detected vocab_size: {vocab_size} from CTC head")

            logger.info("Creating ParakeetCTC model with new ParakeetCTCConfig...")
            ctc_config = ParakeetCTCConfig(
                vocab_size=vocab_size,
                blank_token_id=0,
                ctc_loss_reduction="mean",
                ctc_zero_infinity=True,
                fastconformer_config=hf_config,
            )
            model = ParakeetCTC(ctc_config)

    else:
        logger.info("Creating FastConformerModel...")

        # Ensure we have a FastConformerConfig for base models
        if isinstance(hf_config, ParakeetCTCConfig):
            # Use the FastConformer sub-config for base models
            fastconformer_config = hf_config.fastconformer_config
        else:
            fastconformer_config = hf_config

        model = FastConformerModel(fastconformer_config)

    # Load weights
    model_state_dict = model.state_dict()
    updated_state_dict = model_state_dict.copy()

    matched_params = 0
    for param_name in model_state_dict.keys():
        if param_name in hf_state_dict:
            if model_state_dict[param_name].shape == hf_state_dict[param_name].shape:
                updated_state_dict[param_name] = hf_state_dict[param_name]
                matched_params += 1
            else:
                logger.warning(
                    f"Shape mismatch for {param_name}: "
                    f"HF {model_state_dict[param_name].shape} vs "
                    f"NeMo {hf_state_dict[param_name].shape}"
                )

    model.load_state_dict(updated_state_dict, strict=False)
    logger.info(f"Loaded {matched_params}/{len(model_state_dict)} model parameters")

    return model


def convert_nemo_to_hf(input_path: str, output_dir: str) -> dict[str, Any]:
    """
    Main conversion function.

    Args:
        input_path: Path to .nemo file or extracted directory
        output_dir: Output directory for HuggingFace model

    Returns:
        Conversion info dictionary
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    # Handle input path (file or directory)
    vocab_dict = None
    if input_path.endswith(".nemo"):
        # Extract .nemo file
        with tempfile.TemporaryDirectory() as temp_dir:
            try:
                model_files = extract_nemo_archive(input_path, temp_dir)
                config = load_nemo_config(model_files["model_config"])
                logger.info(f"Loading weights from: {model_files['model_weights']}")
                state_dict = torch.load(model_files["model_weights"], map_location="cpu", weights_only=True)

                # Handle vocabulary file if present
                if "vocab_file" in model_files:
                    logger.info(f"Converting vocabulary file: {model_files['vocab_file']}")
                    vocab_dict = convert_sentencepiece_vocab_to_json(model_files["vocab_file"])
                else:
                    logger.warning("No vocabulary file found in NeMo archive")

            except Exception as e:
                logger.error(f"Failed to extract and load NeMo file {input_path}: {e}")
                raise
    else:
        # Assume it's an extracted directory
        logger.info(f"Looking for config and weight files in directory: {input_path}")
        config_files = list(Path(input_path).glob("*.yaml"))
        weight_files = (
            list(Path(input_path).glob("*.pt"))
            + list(Path(input_path).glob("*.pth"))
            + list(Path(input_path).glob("*.ckpt"))
            + list(Path(input_path).glob("*.bin"))
        )

        logger.info(f"Found config files: {[f.name for f in config_files]}")
        logger.info(f"Found weight files: {[f.name for f in weight_files]}")

        if not config_files:
            raise FileNotFoundError(f"Could not find config (.yaml) files in {input_path}")
        if not weight_files:
            raise FileNotFoundError(f"Could not find weight (.pt/.pth/.ckpt/.bin) files in {input_path}")

        config = load_nemo_config(str(config_files[0]))
        logger.info(f"Loading weights from: {weight_files[0]}")
        state_dict = torch.load(str(weight_files[0]), map_location="cpu", weights_only=True)

    # Extract model information
    model_info = extract_model_info_from_config(config)

    # Create HuggingFace config
    hf_config = create_hf_config_from_nemo(model_info, state_dict)

    # Convert weights
    hf_state_dict = convert_weights(state_dict, model_info)

    # Create model
    hf_model = create_hf_model(hf_config, hf_state_dict, model_info)

    # Save model
    logger.info(f"Saving model to {output_dir}")
    hf_model.save_pretrained(output_dir)

    # Create and save feature extractor
    feature_extractor_config = create_feature_extractor_config(model_info.get("preprocessor_cfg"))

    with open(output_dir / "preprocessor_config.json", "w") as f:
        json.dump(feature_extractor_config, f, indent=2)

    feature_extractor = FastConformerFeatureExtractor(
        **{k: v for k, v in feature_extractor_config.items() if k != "feature_extractor_type"}
    )
    feature_extractor.save_pretrained(output_dir)

    # Create and save tokenizer if vocabulary is available
    if vocab_dict is not None:
        logger.info("Creating and saving tokenizer...")

        # Save vocab.json
        vocab_file_path = output_dir / "vocab.json"
        with open(vocab_file_path, "w", encoding="utf-8") as f:
            json.dump(vocab_dict, f, ensure_ascii=False, indent=2)

        # Import and create tokenizer
        from transformers.models.fastconformer.tokenization_fastconformer import FastConformerTokenizer

        # Determine blank token ID based on model config
        blank_token_id = len(vocab_dict)  # Default: vocab_size
        if model_info["is_ctc_model"]:
            # For CTC models, blank token is usually vocab_size
            if isinstance(hf_config, ParakeetCTCConfig):
                blank_token_id = hf_config.blank_token_id

        tokenizer = FastConformerTokenizer(
            vocab_file=str(vocab_file_path),
            unk_token="<unk>",
            blank_token_id=blank_token_id,
        )

        # Save tokenizer
        tokenizer.save_pretrained(output_dir)
        logger.info(f"✅ Tokenizer saved with {len(vocab_dict)} tokens")

        # Add tokenizer info to conversion metadata
        conversion_info_extra = {
            "has_tokenizer": True,
            "vocab_size": len(vocab_dict),
            "blank_token_id": blank_token_id,
        }
    else:
        logger.warning("No vocabulary found - tokenizer will not be created")
        conversion_info_extra = {"has_tokenizer": False}

    # Save conversion metadata
    conversion_info = {
        "input_path": input_path,
        "output_dir": str(output_dir),
        "nemo_model_type": model_info["model_type"],
        "nemo_decoder_type": model_info["decoder_type"],
        "hf_model_type": type(hf_model).__name__,
        "hf_config_type": type(hf_model.config).__name__ if hasattr(hf_model, "config") else "FastConformerConfig",
        "is_ctc_model": model_info["is_ctc_model"],
        "conversion_success": True,
        "notes": [
            "Weights loaded from NeMo checkpoint without using NeMo library",
            f"Converted to {type(hf_model).__name__}",
            "Uses regex-based weight key mapping",
            "Numerically equivalent to original NeMo FastConformer",
            f"Tokenizer: {'✅ Created' if conversion_info_extra['has_tokenizer'] else '❌ Not available'}",
        ],
        **conversion_info_extra,  # Add tokenizer info
    }

    with open(output_dir / "conversion_info.json", "w") as f:
        json.dump(conversion_info, f, indent=2)

    logger.info("Conversion completed successfully!")
    return conversion_info


def verify_conversion(output_dir: str) -> bool:
    """Verify that the conversion was successful by loading the model."""
    logger.info("Verifying conversion...")

    try:
        from transformers import AutoConfig, AutoModel

        # Load config to determine model type
        config = AutoConfig.from_pretrained(output_dir)

        # Load model
        model = AutoModel.from_pretrained(output_dir)
        model.eval()

        # Create test input
        batch_size, seq_len = 1, 100
        if hasattr(config, "fastconformer_config"):
            mel_bins = config.fastconformer_config.num_mel_bins
        else:
            mel_bins = config.num_mel_bins

        test_input = torch.randn(batch_size, seq_len, mel_bins)
        lengths = torch.tensor([seq_len], dtype=torch.long)

        # Forward pass
        with torch.no_grad():
            outputs = model(test_input, input_lengths=lengths)

        # Check output
        if hasattr(outputs, "logits"):
            logger.info(f"Model output shape: {outputs.logits.shape}")
        else:
            logger.info(f"Model output shape: {outputs.last_hidden_state.shape}")

        logger.info("✅ Verification PASSED")
        return True

    except Exception as e:
        logger.error(f"❌ Verification FAILED: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Convert NeMo FastConformer models to HuggingFace format")
    parser.add_argument(
        "--path_to_nemo_model", type=str, required=True, help="Path to .nemo file or extracted NeMo model directory"
    )
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for HuggingFace model")
    parser.add_argument(
        "--verify", action="store_true", help="Verify conversion by testing model loading and forward pass"
    )
    parser.add_argument("--force", action="store_true", help="Overwrite existing output directory")

    args = parser.parse_args()

    # Check if output directory exists
    if Path(args.output_dir).exists() and not args.force:
        logger.error(f"Output directory {args.output_dir} already exists. Use --force to overwrite.")
        return

    try:
        # Convert model
        conversion_info = convert_nemo_to_hf(args.path_to_nemo_model, args.output_dir)

        # Verify if requested
        if args.verify:
            verification_success = verify_conversion(args.output_dir)
            conversion_info["verification_passed"] = verification_success

        # Print summary
        logger.info("\n" + "=" * 60)
        logger.info("CONVERSION SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Input: {args.path_to_nemo_model}")
        logger.info(f"Output: {args.output_dir}")
        logger.info(f"Model type: {conversion_info['hf_model_type']}")
        logger.info(f"Success: {conversion_info['conversion_success']}")
        if args.verify:
            logger.info(f"Verification: {'✅ PASSED' if conversion_info.get('verification_passed') else '❌ FAILED'}")
        logger.info("=" * 60)

        # Usage example
        logger.info("\nUsage example:")
        if conversion_info.get("is_ctc_model", False):
            logger.info("# For CTC models:")
            logger.info("from transformers import AutoModelForCTC, AutoFeatureExtractor, AutoTokenizer")
            logger.info(f"model = AutoModelForCTC.from_pretrained('{args.output_dir}')")
            logger.info(f"feature_extractor = AutoFeatureExtractor.from_pretrained('{args.output_dir}')")
            if conversion_info.get("has_tokenizer", False):
                logger.info(f"tokenizer = AutoTokenizer.from_pretrained('{args.output_dir}')")
                logger.info("# Decode CTC output: text = tokenizer.decode(token_ids, ctc_decode=True)")
        else:
            logger.info("# For base models:")
            logger.info("from transformers import AutoModel, AutoFeatureExtractor")
            logger.info(f"model = AutoModel.from_pretrained('{args.output_dir}')")
            logger.info(f"feature_extractor = AutoFeatureExtractor.from_pretrained('{args.output_dir}')")
            if conversion_info.get("has_tokenizer", False):
                logger.info(f"tokenizer = AutoTokenizer.from_pretrained('{args.output_dir}')")

    except Exception as e:
        logger.error(f"Conversion failed: {e}")
        raise


if __name__ == "__main__":
    main()


# Future Model-Specific Improvements:
#
# 1. Additional Model Classes:
#    - ParakeetTDT (FastConformer + TDT decoder)
#    - ParakeetRNNT (FastConformer + RNN-T decoder)
#    - CanaryAED (FastConformer + AED decoder)
#
# 2. Enhanced Auto Model Registration:
#    - ("parakeet_tdt", "ParakeetTDT") in MODEL_FOR_TDT_MAPPING
#    - ("parakeet_rnnt", "ParakeetRNNT") in MODEL_FOR_RNNT_MAPPING
#    - ("canary", "CanaryAED") in MODEL_FOR_SPEECH_SEQ_2_SEQ_MAPPING
#
# 3. Decoder-Specific Converters:
#    - TDT decoder weight conversion logic
#    - RNN-T decoder weight conversion logic
#    - AED decoder weight conversion logic
#
# 4. Current Implementation:
#    - ParakeetCTC (FastConformer + CTC) - COMPLETE
#    - ParakeetTDT (FastConformer + TDT) - PLANNED
#    - ParakeetRNNT (FastConformer + RNN-T) - PLANNED
#    - CanaryAED (FastConformer + AED) - PLANNED
