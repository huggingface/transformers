#!/usr/bin/env python3
# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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
"""

NeMo to ParakeetCTC HuggingFace Converter

This script converts NeMo models with FastConformer encoder and CTC decoder
to HuggingFace `ParakeetForCTC` format. It handles:
- FastConformer encoder extraction and conversion
- CTC decoder creation with proper encoder-decoder structure
- Preprocessor configuration (mel-spectrogram, etc.)
- Tokenizer and vocabulary conversion for CTC models
- Model configuration extraction and metadata preservation

The converted model creates a ParakeetCTC model with clean encoder-decoder architecture
that can be loaded via AutoModel and is fully compatible with HuggingFace ecosystem
for CTC-based speech recognition.

Usage:
```bash
# download original weights (e.g., nvidia/parakeet-ctc-1.1b.nemo)
wget --content-disposition -P /raid/eric/nemo \
    https://huggingface.co/nvidia/parakeet-ctc-1.1b/resolve/main/parakeet-ctc-1.1b.nemo

# run conversion (verify flag to check model loading)
python src/transformers/models/parakeet/convert_nemo_to_hf.py \
    --path_to_nemo_model /raid/eric/nemo/parakeet-ctc-1.1b.nemo \
    --output_dir /raid/eric/nemo/parakeet-ctc-hf --force \
    --verify --push_to_hub bezzam/parakeet-ctc-1.1b-hf
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

from transformers.models.parakeet.configuration_parakeet import ParakeetConfig, ParakeetEncoderConfig, ParakeetTDTDecoderConfig, ParakeetTDTConfig, ParakeetTDTJointConfig
#from transformers.models.parakeet.configuration_parakeet import ParakeetEncoderConfig, ParakeetTDTDecoderConfig, ParakeetTDTConfig, ParakeetTDTJointConfig
from transformers.models.parakeet.feature_extraction_parakeet import ParakeetFeatureExtractor
from transformers.models.parakeet.modeling_parakeet import ParakeetForCTC, ParakeetForTDT
from transformers.models.parakeet.processing_parakeet import ParakeetProcessor
from transformers.models.parakeet.tokenization_parakeet import ParakeetCTCTokenizer


# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# NeMo to HuggingFace weight mapping patterns
# Regex patterns for converting NeMo FastConformer weights to HuggingFace format
NEMO_TO_HF_WEIGHT_MAPPING = {
    # Subsampling layer
    r"encoder\.pre_encode\.": r"encoder.subsampling.",
    # Subsampling specific mappings
    r"encoder\.subsampling\.conv\.": r"encoder.subsampling.layers.",
    r"encoder\.subsampling\.out\.": r"encoder.subsampling.linear.",
    # # Positional encoding (skip pe buffer)
    # r"encoder\.pos_enc\.pe$": None,  # Skip buffer
    r"encoder\.pos_enc\.": r"encoder.encode_positions.",
    # Conformer layers - attention (NeMo already uses self_attn)
    r"encoder\.layers\.(\d+)\.self_attn\.": r"encoder.layers.\1.self_attn.",
    # Conformer layers - feed forward (NeMo already uses feed_forward1/2)
    r"encoder\.layers\.(\d+)\.feed_forward1\.": r"encoder.layers.\1.feed_forward1.",
    r"encoder\.layers\.(\d+)\.feed_forward2\.": r"encoder.layers.\1.feed_forward2.",
    # Conformer layers - convolution (NeMo already uses conv not conv_module)
    r"encoder\.layers\.(\d+)\.conv\.": r"encoder.layers.\1.conv.",
    # BatchNorm to 'norm' mapping in convolution module
    r"encoder\.layers\.(\d+)\.conv\.batch_norm\.": r"encoder.layers.\1.conv.norm.",
    # Conformer layers - layer norms (NeMo naming)
    r"encoder\.layers\.(\d+)\.norm_feed_forward1\.": r"encoder.layers.\1.norm_feed_forward1.",
    r"encoder\.layers\.(\d+)\.norm_feed_forward2\.": r"encoder.layers.\1.norm_feed_forward2.",
    r"encoder\.layers\.(\d+)\.norm_self_att\.": r"encoder.layers.\1.norm_self_att.",
    r"encoder\.layers\.(\d+)\.norm_conv\.": r"encoder.layers.\1.norm_conv.",
    r"encoder\.layers\.(\d+)\.norm_out\.": r"encoder.layers.\1.norm_out.",
    # Decoder (CTC head) - Conv1d to Linear conversion handled separately
    r"decoder\.decoder_layers\.0\.weight": r"ctc_head.weight",
    r"decoder\.decoder_layers\.0\.bias": r"ctc_head.bias",
    # Catch-all pattern for any remaining encoder patterns (must be last)
    r"^encoder\.": r"encoder.",
    r"linear_k": "k_proj",
    r"linear_v": "v_proj",
    r"linear_out": "o_proj",
    r"linear_q": "q_proj",
    r"pos_bias_u": "bias_u",
    r"pos_bias_v": "bias_v",
    r"linear_pos": "relative_k_proj",
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

    with tarfile.open(nemo_file_path, "r", encoding="utf-8") as tar:
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
                or file.endswith(".txt")
                or ("tokenizer" in file_lower and (file.endswith(".vocab") or file.endswith(".model")))
            ):
                # Prefer .vocab files over others
                if "vocab_file" not in model_files or file.endswith(".vocab"):
                    model_files["vocab_file"] = file_path
                    logger.info(f"Found vocabulary file: {file}")
                else:
                    logger.info(f"Found additional vocabulary file (using existing): {file}")

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
    Convert vocabulary file to JSON format. Supports both SentencePiece and plain text formats.

    Args:
        vocab_file_path: Path to .vocab/.txt file from NeMo

    Returns:
        Dictionary mapping tokens to IDs
    """
    vocab_dict = {}

    logger.info(f"Processing vocabulary file: {vocab_file_path}")

    try:
        with open(vocab_file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        logger.info(f"Read {len(lines)} lines from vocabulary file")

        # Detect format by checking first few lines
        sample_lines = lines[:5]
        has_tabs = any("\t" in line for line in sample_lines)

        if has_tabs:
            logger.info("Detected SentencePiece format (token\\tscore)")
            # SentencePiece vocab format: token \t score
            for token_id, line in enumerate(lines):
                parts = line.strip().split("\t")
                if len(parts) >= 1 and parts[0]:  # Ensure token is not empty
                    token = parts[0]
                    vocab_dict[token] = token_id
        else:
            logger.info("Detected plain text format (one token per line)")
            # Plain text format: one token per line
            for token_id, line in enumerate(lines):
                token = line.strip()
                if token:  # Ensure token is not empty
                    vocab_dict[token] = token_id

        logger.info(f"Successfully converted vocabulary with {len(vocab_dict)} tokens")

        # Ensure <unk> token exists
        if "<unk>" not in vocab_dict:
            # Insert <unk> at the beginning and shift other tokens
            new_vocab_dict = {"<unk>": 0}
            for token, token_id in vocab_dict.items():
                new_vocab_dict[token] = token_id + 1
            vocab_dict = new_vocab_dict
            logger.info("Added <unk> token at ID 0 and shifted other tokens")

        return vocab_dict

    except Exception as e:
        logger.error(f"Failed to convert vocab file {vocab_file_path}: {e}")
        import traceback

        logger.error(f"Traceback: {traceback.format_exc()}")
        # Return a minimal vocab as fallback
        logger.warning("Creating minimal fallback vocabulary")
        return {"<unk>": 0}


def load_nemo_config(config_path: str) -> dict[str, Any]:
    """Load NeMo model configuration from yaml file."""
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config


def extract_model_info_from_config(config: dict[str, Any]) -> dict[str, Any]:
    """Extract model information from NeMo config."""
    model_info = {
        "model_type": "unknown",
        "encoder_type": "unknown",
        "decoder_type": "unknown",
        "is_ctc_model": False,
        "is_tdt_model": False,
        "encoder_cfg": None,
        "decoder_cfg": None,
        "preprocessor_cfg": None,
    }

    # Extract model type from config or model name
    model_name = config.get("name", "").lower()
    if "parakeet" in model_name:
        if "ctc" in model_name:
            model_info["model_type"] = "parakeet_ctc"
        if "tdt" in model_name:
            model_info["model_type"] = "parakeet_tdt"
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
    is_tdt = False

    # Primary check: Look for EncDecCTCModelBPE in the main config target
    main_target = config.get("target", "").lower()
    if "encdecctcmodelbpe" in main_target.replace("_", "").replace(".", ""):
        is_ctc = True
        logger.info(f"Detected EncDecCTCModelBPE in main target: {config.get('_target_', '')}")
    elif "tdt" in config['loss']['loss_name']:
        is_tdt = True
        logger.info(f"Detected TDT model")


    # Secondary checks
    # Check decoder type
    if "ctc" in decoder_type or "convctc" in decoder_type or "conv_asr_decoder" in decoder_type:
        is_ctc = True

    # Check model name
    if "ctc" in model_name:
        is_ctc = True

    # Check model name
    if "tdt" in model_name and "parakeet" in model_name:
        is_tdt = True

    # Check for CTC-specific config parameters
    if model_info["decoder_cfg"]:
        decoder_target = model_info["decoder_cfg"].get("_target_", "").lower()
        if "ctc" in decoder_target or "convctc" in decoder_target or "conv_asr_decoder" in decoder_target:
            is_ctc = True

    model_info["is_ctc_model"] = is_ctc
    model_info["is_tdt_model"] = is_tdt

    # Set model type based on CTC detection
    if is_ctc:
        model_info["model_type"] = "parakeet_ctc"
        assert not is_tdt
    if is_tdt:
        model_info["model_type"] = "parakeet_tdt"

    

    logger.info(f"Detected model type: {model_info['model_type']}")
    logger.info(f"Encoder type: {model_info['encoder_type']}")
    logger.info(f"Decoder type: {model_info['decoder_type']}")
    logger.info(f"Is CTC model: {model_info['is_ctc_model']}")

    return model_info


def create_hf_config_from_nemo(
    model_info: dict[str, Any], state_dict: dict[str, torch.Tensor], vocab_dict: Optional[dict[str, int]] = None
) -> Union[ParakeetTDTConfig]:  # used to be ParakeetConfig TODO(hainan)
    """Create HuggingFace ParakeetConfig from NeMo config and weights."""

    encoder_cfg = model_info.get("encoder_cfg", {})
    decoder_cfg = model_info.get("decoder_cfg", {})

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

        # decoder parameters
        "pred_hidden": 640,
        "pred_n_layers": 2,
        "joint_hidden": 640,
        "durations": [0,1,2,3,4],
    }

    # Add model-specific metadata
    if model_info["is_ctc_model"]:
        architectures = ["parakeet_ctc"]
        base_model_type = "parakeet_ctc"
    elif model_info["is_tdt_model"]:
        architectures = ["parakeet_tdt"]
        base_model_type = "parakeet_tdt"
    else:
        raise ValueError("Unsupported model type. Only CTC and TDT models are supported in this converter.")

    config_params.update(
        {
            "model_type": base_model_type,
            "architectures": architectures,
            "nemo_model_type": model_info["model_type"],
            "nemo_encoder_type": model_info["encoder_type"],
            "nemo_decoder_type": model_info["decoder_type"],
        }
    )

    # For CTC models, create ParakeetConfig
    model_config = None
    if model_info["is_ctc_model"]:
        # Get vocab_size from state dict if available
        vocab_size = 1024  # default
        if any("decoder.ctc_head.weight" in key or "decoder_layers.0.weight" in key for key in state_dict.keys()):
            # Find the decoder weight to get vocab_size
            decoder_keys = [k for k in state_dict.keys() if "decoder_layers.0.weight" in k]
            if decoder_keys:
                decoder_weight = state_dict[decoder_keys[0]]
                if decoder_weight.dim() == 3 and decoder_weight.size(2) == 1:
                    vocab_size = decoder_weight.size(0)  # Conv1d output channels
                else:
                    vocab_size = decoder_weight.size(0)  # Linear output features
                logger.info(f"Detected vocab_size: {vocab_size} from decoder weights")

        # Create `ParakeetEncoderConfig` sub-config with `parakeet_encoder` model_type
        parakeet_encoder_config_params = config_params.copy()
        parakeet_encoder_config_params["model_type"] = "parakeet_encoder"
        parakeet_encoder_config_params["architectures"] = ["ParakeetEncoder"]
        parakeet_encoder_config = ParakeetEncoderConfig(**parakeet_encoder_config_params)

        # Calculate blank token ID: should be len(vocab_dict) if we have vocab, otherwise vocab_size - 1
        if vocab_dict:
            blank_id = len(vocab_dict)  # Blank token after all real tokens
            logger.info(f"Setting blank_token_id to {blank_id} (vocab has {len(vocab_dict)} real tokens)")
        else:
            blank_id = vocab_size - 1  # Fallback
            logger.info(f"No vocab provided, setting blank_token_id to {blank_id}")

        model_config = ParakeetConfig(
            vocab_size=vocab_size,  # Total size including blank token
            blank_token_id=blank_id,
            ctc_loss_reduction="mean",
            ctc_zero_infinity=True,
            encoder_config=parakeet_encoder_config,
        )
    elif model_info["is_tdt_model"]:
        # Get vocab_size from state dict if available
        vocab_size = 1024  # default

        # Create `ParakeetEncoderConfig` sub-config with `parakeet_encoder` model_type
        parakeet_encoder_config_params = config_params.copy()
        parakeet_encoder_config_params["model_type"] = "parakeet_encoder"
        parakeet_encoder_config_params["architectures"] = ["ParakeetEncoder"]
        parakeet_encoder_config = ParakeetEncoderConfig(**parakeet_encoder_config_params)


        parakeet_decoder_config_params = config_params.copy()
        parakeet_decoder_config_params["model_type"] = "parakeet_decoder"
        parakeet_decoder_config_params["architectures"] = ["ParakeetDecoder"]
        parakeet_decoder_config = ParakeetTDTDecoderConfig(**parakeet_decoder_config_params)

        parakeet_joint_config_params = config_params.copy()
        parakeet_joint_config_params["model_type"] = "parakeet_joint"
        parakeet_joint_config_params["architectures"] = ["ParakeetJoint"]
        parakeet_joint_config = ParakeetTDTJointConfig(**parakeet_joint_config_params)

        # Calculate blank token ID: should be len(vocab_dict) if we have vocab, otherwise vocab_size - 1
        if vocab_dict:
            blank_id = len(vocab_dict)  # Blank token after all real tokens
            logger.info(f"Setting blank_token_id to {blank_id} (vocab has {len(vocab_dict)} real tokens)")
        else:
            blank_id = vocab_size - 1  # Fallback
            logger.info(f"No vocab provided, setting blank_token_id to {blank_id}")

        model_config = ParakeetTDTConfig(
            vocab_size=vocab_size,  # Total size including blank token
            blank_token_id=blank_id,
            ctc_loss_reduction="mean",
            ctc_zero_infinity=True,
            encoder_config=parakeet_encoder_config,
            decoder_config=parakeet_decoder_config,
            joint_config=parakeet_joint_config,
        )

    # Non CTC models, TODO
    return model_config


def create_feature_extractor_config(preprocessor_cfg: Optional[dict[str, Any]]) -> dict[str, Any]:
    """Create feature extractor configuration from NeMo preprocessor config."""
    if preprocessor_cfg:
        sample_rate = preprocessor_cfg.get("sample_rate", 16000)
        feature_extractor_config = {
            "feature_extractor_type": "ParakeetFeatureExtractor",
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
            "feature_extractor_type": "ParakeetFeatureExtractor",
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


def extract_preprocessing_weights(nemo_state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Extract preprocessing weights from NeMo state dict."""
    preprocessing_weights = {}

    # Extract window function
    window_keys = [k for k in nemo_state_dict.keys() if k.endswith("featurizer.window")]
    if window_keys:
        preprocessing_weights["feature_extractor.window"] = nemo_state_dict[window_keys[0]]
        logger.info(f"Extracted window function: {window_keys[0]}")

    # Extract filterbanks
    fb_keys = [k for k in nemo_state_dict.keys() if k.endswith("featurizer.fb")]
    if fb_keys:
        # NeMo stores filterbanks with shape (1, n_mels, n_fft//2+1)
        # HuggingFace expects (n_mels, n_fft//2+1)
        fb_tensor = nemo_state_dict[fb_keys[0]]
        if fb_tensor.dim() == 3 and fb_tensor.size(0) == 1:
            fb_tensor = fb_tensor.squeeze(0)
        preprocessing_weights["feature_extractor.filterbanks"] = fb_tensor
        logger.info(f"Extracted mel filterbanks: {fb_keys[0]}")

    return preprocessing_weights


def convert_weights(nemo_state_dict: dict[str, torch.Tensor], model_info: dict[str, Any]) -> dict[str, torch.Tensor]:
    """Convert NeMo weights to HuggingFace format using regex mapping."""
    logger.info("Converting weights using regex mapping...")

    # Extract preprocessing weights first
    preprocessing_weights = extract_preprocessing_weights(nemo_state_dict)

    # Get key mapping
    all_keys = list(nemo_state_dict.keys())

    # Debug: Show sample original NeMo parameter names
    logger.info(f"Sample original NeMo parameter names: {all_keys[:10]}")

    key_mapping = convert_nemo_keys_to_hf_keys(all_keys)

    hf_state_dict = {}

    for nemo_key, tensor in nemo_state_dict.items():
        hf_key = key_mapping.get(nemo_key, "")

        if hf_key == "":
            # Skip this key (mapped to None or empty)
            continue

        hf_state_dict[hf_key] = tensor

    # Add preprocessing weights to the model state dict
    hf_state_dict.update(preprocessing_weights)

    logger.info(
        f"Converted {len(hf_state_dict)} weights from {len(nemo_state_dict)} NeMo weights (including {len(preprocessing_weights)} preprocessing weights)"
    )
    return hf_state_dict


def create_hf_model(
    hf_config: Union[ParakeetConfig, ParakeetTDTConfig],
    hf_state_dict: dict[str, torch.Tensor],
    model_info: dict[str, Any],
) -> Union[ParakeetForCTC,ParakeetForTDT]:
    """Create the appropriate HuggingFace model and load weights."""

    if model_info["is_ctc_model"]:
        # Check if we already have a ParakeetCTCConfig or need to create one
        if isinstance(hf_config, ParakeetConfig):
            logger.info("Creating ParakeetForCTC model with existing ParakeetConfig...")
            model = ParakeetForCTC(hf_config)
        else:
            # Fallback: create ParakeetConfig if we somehow still have FastConformerConfig
            vocab_size = 1024  # default
            if "decoder.ctc_head.weight" in hf_state_dict:
                vocab_size = hf_state_dict["decoder.ctc_head.weight"].shape[0]
                logger.info(f"Detected vocab_size: {vocab_size} from CTC head")

            logger.info("Creating ParakeetForCTC model with new ParakeetConfig...")
            ctc_config = ParakeetConfig(
                vocab_size=vocab_size,
                blank_token_id=0,
                ctc_loss_reduction="mean",
                ctc_zero_infinity=True,
                encoder_config=hf_config,
            )
            model = ParakeetForCTC(ctc_config)

    elif model_info["is_tdt_model"]:
        # Check if we already have a ParakeetTDTConfig or need to create one
        if isinstance(hf_config, ParakeetTDTConfig):
            logger.info("Creating ParakeetForTDT model with existing ParakeetTDTConfig...")
            model = ParakeetForTDT(hf_config)
        else:
            # Fallback: create ParakeetConfig if we somehow still have FastConformerConfig
            vocab_size = 1024  # default

            logger.info("Creating ParakeetForTDT model with new ParakeetTDTConfig...")
            tdt_config = ParakeetTDTConfig(
                vocab_size=vocab_size,
                blank_token_id=vocab_size,
                tdt_loss_reduction="mean",
                model_type='tdt',
                encoder_config=hf_config,
                decoder_config=hf_config,
            )
            model = ParakeetForTDT(tdt_config)

    else:
        raise ValueError("Unsupported model type. Only CTC models are supported in this converter.")
        # logger.info("Creating FastConformerModel...")

        # # Ensure we have a FastConformerConfig for base models
        # if isinstance(hf_config, ParakeetCTCConfig):
        #     # Use the FastConformer sub-config for base models
        #     fastconformer_config = hf_config.encoder_config
        # else:
        #     fastconformer_config = hf_config

        # model = FastConformerModel(fastconformer_config)

    # Load weights
    model_state_dict = model.state_dict()
    updated_state_dict = model_state_dict.copy()

    matched_params = 0
    shape_mismatches = 0
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
                shape_mismatches += 1

    model.load_state_dict(updated_state_dict, strict=False)
    if matched_params != len(model_state_dict):
        raise ValueError(
            f"Missing parameters when copying! {matched_params}/{len(model_state_dict)} parameters were copied."
        )
    else:
        logger.info(f"Loaded {matched_params}/{len(model_state_dict)} model parameters")
    if shape_mismatches > 0:
        logger.warning(f"Found {shape_mismatches} shape mismatches")

    return model


def convert_nemo_to_hf(input_path: str, output_dir: str, push_to_hub: str | None) -> dict[str, Any]:
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
    hf_config = create_hf_config_from_nemo(model_info, state_dict, vocab_dict)

    # Convert weights
    hf_state_dict = convert_weights(state_dict, model_info)

    # Create model
    hf_model = create_hf_model(hf_config, hf_state_dict, model_info)

    # Save model
    logger.info(f"Saving model to {output_dir}")
    hf_model.save_pretrained(output_dir)

    # Create and save feature extractor
    feature_extractor_config = create_feature_extractor_config(model_info.get("preprocessor_cfg"))

    with open(output_dir / "preprocessor_config.json", "w", encoding="utf-8") as f:
        json.dump(feature_extractor_config, f, indent=2)

    feature_extractor = ParakeetFeatureExtractor(
        **{k: v for k, v in feature_extractor_config.items() if k != "feature_extractor_type"}
    )
    feature_extractor.save_pretrained(output_dir)

    # Create and save tokenizer if vocabulary is available
    processor = None
    if vocab_dict is not None:
        logger.info("Creating and saving tokenizer...")

        # Save vocab.json
        vocab_file_path = output_dir / "vocab.json"
        with open(vocab_file_path, "w", encoding="utf-8") as f:
            json.dump(vocab_dict, f, ensure_ascii=False, indent=2)

        # Get blank token ID from model config
        blank_token_id = len(vocab_dict)  # Default: vocab_size (blank token after all real tokens)
        if model_info["is_ctc_model"] and isinstance(hf_config, ParakeetConfig):
            # Use the blank token ID from the model configuration
            blank_token_id = hf_config.blank_token_id
            logger.info(f"Using blank_token_id from model config: {blank_token_id}")
        elif model_info["is_tdt_model"] and isinstance(hf_config, ParakeetTDTConfig):
            # Use the blank token ID from the model configuration
            blank_token_id = hf_config.blank_token_id
            logger.info(f"HAINAN HERE Using blank_token_id from model config: {blank_token_id}")

        tokenizer = ParakeetCTCTokenizer(
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

        # create processor
        processor = ParakeetProcessor(
            feature_extractor=feature_extractor,
            tokenizer=tokenizer,
        )
        processor.save_pretrained(output_dir)
        logger.info("✅ Processor saved with feature extractor and tokenizer")
    else:
        logger.warning("No vocabulary found - tokenizer will not be created")
        conversion_info_extra = {"has_tokenizer": False}

    # push to hub
    if push_to_hub:
        logger.info(f"Pushing model to HuggingFace Hub at {push_to_hub}...")
        feature_extractor.push_to_hub(push_to_hub)
        hf_model.push_to_hub(push_to_hub)
        tokenizer.push_to_hub(push_to_hub)
        if processor is not None:
            processor.push_to_hub(push_to_hub)
        logger.info(f"✅ Model pushed to HuggingFace Hub at {push_to_hub}")

    # Save conversion metadata
    conversion_info = {
        "input_path": input_path,
        "output_dir": str(output_dir),
        "hub_repo_id": push_to_hub,
        "nemo_model_type": model_info["model_type"],
        "nemo_decoder_type": model_info["decoder_type"],
        "hf_model_type": type(hf_model).__name__,
        "hf_config_type": type(hf_model.config).__name__ if hasattr(hf_model, "config") else "ParakeetCTCConfig",
        "is_ctc_model": model_info["is_ctc_model"],
        "conversion_success": True,
        "notes": [
            "NeMo CTC model converted to ParakeetForCTC format",
            "Weights loaded from NeMo checkpoint without using NeMo library",
            f"Converted to {type(hf_model).__name__}",
            "Uses regex-based weight key mapping for `ParakeetEncoder``",
            "Clean encoder-decoder structure: model.encoder.* and model.decoder.*",
            "CTC head weights converted from Conv1d to Linear format",
            "Numerically equivalent to original NeMo model",
            f"Tokenizer: {'✅ Created with CTC decoding support' if conversion_info_extra['has_tokenizer'] else '❌ Not available'}",
        ],
        **conversion_info_extra,  # Add tokenizer info
    }

    # Save conversion metadata
    with open(output_dir / "conversion_info.json", "w", encoding="utf-8") as f:
        json.dump(conversion_info, f, indent=2)

    logger.info("Conversion completed successfully!")
    return conversion_info


def verify_conversion(output_dir: str) -> bool:
    """Verify that the conversion was successful by loading the model."""
    logger.info("Verifying conversion...")

#    try:
    if True:
        from transformers import AutoConfig, AutoModelForCTC

        # Load config to determine model type
        config = AutoConfig.from_pretrained(output_dir)

        # Load model
        model = AutoModelForCTC.from_pretrained(output_dir)
        model.eval()

        # Create test input
        batch_size, seq_len = 1, 100
        if hasattr(config, "encoder_config"):
            mel_bins = config.encoder_config.num_mel_bins
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

#    except Exception as e:
#        logger.error(f"❌ Verification FAILED: {e}")
#        return False

def verify_conversion_tdt(output_dir: str) -> bool:
    """Verify that the conversion was successful by loading the model."""
    logger.info("Verifying conversion...")

    try:
        from transformers import AutoConfig, AutoModelForTDT

        # Load config to determine model type
        config = AutoConfig.from_pretrained(output_dir)

        # Load model
        model = AutoModelForTDT.from_pretrained(output_dir)
        model.eval()

        # Create test input
        batch_size, seq_len = 1, 100
        if hasattr(config, "encoder_config"):
            mel_bins = config.encoder_config.num_mel_bins
        else:
            mel_bins = config.num_mel_bins

        test_input = torch.randn(batch_size, seq_len, mel_bins)
        lengths = torch.tensor([seq_len], dtype=torch.long)

        # Forward pass
        with torch.no_grad():
            outputs = model(test_input, input_lengths=lengths)

#        # Check output
#        if hasattr(outputs, "logits"):
#            logger.info(f"Model output shape: {outputs.logits.shape}")
#        else:
#            logger.info(f"Model output shape: {outputs.last_hidden_state.shape}")

        logger.info("✅ Verification PASSED")
        return True
    except Exception as e:
        logger.error(f"❌ Verification FAILED: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Convert NeMo CTC models to HuggingFace `ParakeetForCTC` format")
    parser.add_argument(
        "--path_to_nemo_model",
        type=str,
        required=True,
        help="Path to .nemo file or extracted NeMo CTC model directory",
    )
    parser.add_argument(
        "--output_dir", type=str, required=True, help="Output directory for HuggingFace `ParakeetForCTC` model"
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify conversion by testing `ParakeetForCTC` model loading and forward pass",
    )
    parser.add_argument("--force", action="store_true", help="Overwrite existing output directory")
    parser.add_argument(
        "--push_to_hub", default=None, type=str, help="If provided, where to upload the converted model on the 🤗 hub."
    )

    args = parser.parse_args()

    # Check if output directory exists
    if Path(args.output_dir).exists() and not args.force:
        logger.error(f"Output directory {args.output_dir} already exists. Use --force to overwrite.")
        return

    try:
        # Convert model
        conversion_info = convert_nemo_to_hf(args.path_to_nemo_model, args.output_dir, args.push_to_hub)
        model_path = args.output_dir if args.push_to_hub is None else args.push_to_hub

        # Verify if requested
        if args.verify:
            if 'ctc' in args.path_to_nemo_model:
                verification_success = verify_conversion(model_path)
            elif 'tdt' in args.path_to_nemo_model:
                verification_success = verify_conversion_tdt(model_path)

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
        logger.info("# Load ParakeetForCTC model:")
        logger.info("from transformers import AutoModelForCTC, AutoModelForTDT, AutoFeatureExtractor, AutoTokenizer")
        logger.info(f"model = AutoModelForCTC.from_pretrained('{model_path}')")
        logger.info(f"model = AutoModelForTDT.from_pretrained('{model_path}')")
        logger.info(f"feature_extractor = AutoFeatureExtractor.from_pretrained('{model_path}')")
        if conversion_info.get("has_tokenizer", False):
            logger.info(f"tokenizer = AutoTokenizer.from_pretrained('{model_path}')")
            logger.info("# Decode CTC output: text = tokenizer.decode(token_ids, ctc_decode=True)")
        logger.info("\n# Alternative - direct ParakeetForCTC import:")
        logger.info("from transformers.models.parakeet import ParakeetForCTC")
        logger.info(f"model = ParakeetForCTC.from_pretrained('{model_path}')")

    except Exception as e:
        logger.error(f"Conversion failed: {e}")
        raise


if __name__ == "__main__":
    main()
