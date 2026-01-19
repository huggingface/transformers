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

import argparse
import gc
import os
import re
import tarfile

import torch
import yaml
from tokenizers import AddedToken

from transformers import (
    ParakeetCTCConfig,
    ParakeetEncoder,
    ParakeetEncoderConfig,
    ParakeetFeatureExtractor,
    ParakeetForCTC,
    ParakeetForTDT,
    ParakeetProcessor,
    ParakeetTDTConfig,
    ParakeetTokenizer,
)
from transformers.convert_slow_tokenizer import ParakeetConverter
from transformers.utils.hub import cached_file


NEMO_TO_HF_WEIGHT_MAPPING = {
    r"encoder\.pre_encode\.conv\.": r"encoder.subsampling.layers.",
    r"encoder\.pre_encode\.out\.": r"encoder.subsampling.linear.",
    r"encoder\.pos_enc\.": r"encoder.encode_positions.",
    r"encoder\.layers\.(\d+)\.conv\.batch_norm\.": r"encoder.layers.\1.conv.norm.",
    r"decoder\.decoder_layers\.0\.(weight|bias)": r"ctc_head.\1",
    r"linear_([kv])": r"\1_proj",
    r"linear_out": r"o_proj",
    r"linear_q": r"q_proj",
    r"pos_bias_([uv])": r"bias_\1",
    r"linear_pos": r"relative_k_proj",
}

# Additional mappings for TDT decoder and joint network
NEMO_TDT_WEIGHT_MAPPING = {
    # Decoder embedding
    r"decoder\.prediction\.embed\.": r"decoder.embedding.",
    # Decoder LSTM - remove extra nesting
    r"decoder\.prediction\.dec_rnn\.lstm\.": r"decoder.lstm.",
    # Joint network encoder projection
    r"joint\.enc\.": r"joint.encoder_proj.",
    # Decoder output projection (NeMo puts this in joint, HF puts in decoder)
    r"joint\.pred\.": r"decoder.output_proj.",
    # Note: joint.joint_net.2 (combined head) needs special handling - see write_tdt_model
}


def convert_key(key, mapping):
    for pattern, replacement in mapping.items():
        key = re.sub(pattern, replacement, key)
    return key


def extract_nemo_archive(nemo_file_path: str, extract_dir: str) -> dict[str, str]:
    """
    Extract .nemo file (tar archive) and return paths to important files.

    Args:
        nemo_file_path: Path to .nemo file
        extract_dir: Directory to extract to

    Returns:
        Dictionary with paths to model.pt, model_config.yaml, etc.
    """
    print(f"Extracting NeMo archive: {nemo_file_path}")

    with tarfile.open(nemo_file_path, "r", encoding="utf-8") as tar:
        tar.extractall(extract_dir)

    # Log all extracted files for debugging
    all_files = []
    for root, dirs, files in os.walk(extract_dir):
        for file in files:
            file_path = os.path.join(root, file)
            all_files.append(file_path)

    print(f"All extracted files: {[os.path.basename(f) for f in all_files]}")

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
                print(f"Found model weights: {file}")

            # Look for config files
            elif (
                file == "model_config.yaml"
                or file == "config.yaml"
                or (file.endswith(".yaml") and "config" in file_lower)
            ):
                if "model_config" not in model_files:  # Prefer model_config.yaml
                    model_files["model_config"] = file_path
                    print(f"Found config file: {file}")
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
                if "tokenizer_model_file" not in model_files or file.endswith(".model"):
                    model_files["tokenizer_model_file"] = file_path
                    print(f"Found tokenizer model file: {file}")
                else:
                    print(f"Found additional vocabulary file (using existing): {file}")

    print(f"Found model files: {list(model_files.keys())}")

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


def write_processor(nemo_config: dict, model_files, output_dir, push_to_repo_id=None):
    tokenizer_converted = ParakeetConverter(model_files["tokenizer_model_file"]).converted()
    tokenizer_converted_fast = ParakeetTokenizer(
        tokenizer_object=tokenizer_converted,
        clean_up_tokenization_spaces=False,
    )
    tokenizer_converted_fast.add_tokens(
        [AddedToken("<unk>", normalized=False, special=True), AddedToken("<pad>", normalized=False, special=True)]
    )
    tokenizer_converted_fast.add_special_tokens(
        {
            "pad_token": AddedToken("<pad>", normalized=False, special=True),
            "unk_token": AddedToken("<unk>", normalized=False, special=True),
        }
    )

    feature_extractor_keys_to_ignore = ["_target_", "pad_to", "frame_splicing", "dither", "normalize", "window", "log"]
    feature_extractor_config_keys_mapping = {
        "sample_rate": "sampling_rate",
        "window_size": "win_length",
        "window_stride": "hop_length",
        "window": "window",
        "n_fft": "n_fft",
        "log": "log",
        "features": "feature_size",
        "dither": "dither",
        "pad_to": "pad_to",
        "pad_value": "padding_value",
        "frame_splicing": "frame_splicing",
        "preemphasis": "preemphasis",
        "hop_length": "hop_length",
    }
    converted_feature_extractor_config = {}

    for key, value in nemo_config["preprocessor"].items():
        if key in feature_extractor_keys_to_ignore:
            continue
        if key in feature_extractor_config_keys_mapping:
            if key in ["window_size", "window_stride"]:
                value = int(value * nemo_config["preprocessor"]["sample_rate"])
            converted_feature_extractor_config[feature_extractor_config_keys_mapping[key]] = value
        else:
            raise ValueError(f"Key {key} not found in feature_extractor_keys_mapping")

    feature_extractor = ParakeetFeatureExtractor(**converted_feature_extractor_config)

    processor = ParakeetProcessor(
        feature_extractor=feature_extractor,
        tokenizer=tokenizer_converted_fast,
    )
    processor.save_pretrained(output_dir)

    if push_to_repo_id:
        processor.push_to_hub(push_to_repo_id)


def convert_encoder_config(nemo_config):
    """Convert NeMo encoder config to HF encoder config."""
    encoder_keys_to_ignore = [
        "att_context_size",
        "causal_downsampling",
        "stochastic_depth_start_layer",
        "feat_out",
        "stochastic_depth_drop_prob",
        "_target_",
        "ff_expansion_factor",
        "untie_biases",
        "att_context_style",
        "self_attention_model",
        "conv_norm_type",
        "subsampling",
        "stochastic_depth_mode",
        "conv_context_size",
        "dropout_pre_encoder",
        "reduction",
        "reduction_factor",
        "reduction_position",
    ]
    encoder_config_keys_mapping = {
        "d_model": "hidden_size",
        "n_heads": "num_attention_heads",
        "n_layers": "num_hidden_layers",
        "feat_in": "num_mel_bins",
        "conv_kernel_size": "conv_kernel_size",
        "subsampling_factor": "subsampling_factor",
        "subsampling_conv_channels": "subsampling_conv_channels",
        "pos_emb_max_len": "max_position_embeddings",
        "dropout": "dropout",
        "dropout_emb": "dropout_positions",
        "dropout_att": "attention_dropout",
        "xscaling": "scale_input",
        "use_bias": "attention_bias",
    }
    converted_encoder_config = {}

    for key, value in nemo_config["encoder"].items():
        if key in encoder_keys_to_ignore:
            continue
        if key in encoder_config_keys_mapping:
            converted_encoder_config[encoder_config_keys_mapping[key]] = value
            # NeMo uses 'use_bias' for both attention and convolution bias, but HF separates them
            if key == "use_bias":
                converted_encoder_config["convolution_bias"] = value
        else:
            raise ValueError(f"Key {key} not found in encoder_config_keys_mapping")

    return ParakeetEncoderConfig(**converted_encoder_config)


def load_and_convert_state_dict(model_files):
    """Load NeMo state dict and convert keys to HF format."""
    state_dict = torch.load(model_files["model_weights"], map_location="cpu", weights_only=True)
    converted_state_dict = {}
    for key, value in state_dict.items():
        # Skip preprocessing weights (featurizer components)
        if key.endswith("featurizer.window") or key.endswith("featurizer.fb"):
            print(f"Skipping preprocessing weight: {key}")
            continue
        converted_key = convert_key(key, NEMO_TO_HF_WEIGHT_MAPPING)
        converted_state_dict[converted_key] = value

    return converted_state_dict


def write_ctc_model(encoder_config, converted_state_dict, output_dir, push_to_repo_id=None):
    """Write CTC model using encoder config and converted state dict."""
    model_config = ParakeetCTCConfig.from_encoder_config(encoder_config)

    print("Loading the checkpoint in a Parakeet CTC model.")
    with torch.device("meta"):
        model = ParakeetForCTC(model_config)
    model.load_state_dict(converted_state_dict, strict=True, assign=True)
    print("Checkpoint loaded successfully.")
    del model.config._name_or_path

    print("Saving the model.")
    model.save_pretrained(output_dir)

    if push_to_repo_id:
        model.push_to_hub(push_to_repo_id)

    del model

    # Safety check: reload the converted model
    gc.collect()
    print("Reloading the model to check if it's saved correctly.")
    ParakeetForCTC.from_pretrained(output_dir, dtype=torch.bfloat16, device_map="auto")
    print("Model reloaded successfully.")


def write_encoder_model(encoder_config, converted_state_dict, output_dir, push_to_repo_id=None):
    """Write encoder model using encoder config and converted state dict."""
    # Filter to only encoder weights (exclude CTC head if present)
    encoder_state_dict = {
        k.replace("encoder.", "", 1) if k.startswith("encoder.") else k: v
        for k, v in converted_state_dict.items()
        if k.startswith("encoder.")
    }

    print("Loading the checkpoint in a Parakeet Encoder model (for TDT).")
    with torch.device("meta"):
        model = ParakeetEncoder(encoder_config)

    model.load_state_dict(encoder_state_dict, strict=True, assign=True)
    print("Checkpoint loaded successfully.")
    del model.config._name_or_path

    print("Saving the model.")
    model.save_pretrained(output_dir)

    if push_to_repo_id:
        model.push_to_hub(push_to_repo_id)
    del model

    # Safety check: reload the converted model
    gc.collect()
    print("Reloading the model to check if it's saved correctly.")
    ParakeetEncoder.from_pretrained(output_dir, dtype=torch.bfloat16, device_map="auto")
    print("Model reloaded successfully.")


def convert_tdt_config(nemo_config, encoder_config):
    """Convert NeMo TDT config to HF TDT config."""
    decoder_config = nemo_config.get("decoder", {})
    joint_config = nemo_config.get("joint", {})
    decoding_config = nemo_config.get("decoding", {})

    # Extract vocab size from labels
    labels = nemo_config.get("labels", [])
    vocab_size = len(labels) if labels else decoder_config.get("vocab_size", 1024)

    # Prediction network config
    prednet = decoder_config.get("prednet", {})
    decoder_hidden_size = prednet.get("pred_hidden", 640)
    decoder_num_layers = prednet.get("pred_rnn_layers", 2)

    # Joint network config
    jointnet = joint_config.get("jointnet", {})
    joint_hidden_size = jointnet.get("joint_hidden", 640)

    # Duration config from decoding
    durations = decoding_config.get("durations", [0, 1, 2, 3, 4])
    num_duration_bins = len(durations)

    print(
        f"TDT config: vocab_size={vocab_size}, decoder_hidden={decoder_hidden_size}, "
        f"decoder_layers={decoder_num_layers}, joint_hidden={joint_hidden_size}, "
        f"num_durations={num_duration_bins}"
    )

    return ParakeetTDTConfig(
        vocab_size=vocab_size,
        decoder_hidden_size=decoder_hidden_size,
        decoder_num_layers=decoder_num_layers,
        joint_hidden_size=joint_hidden_size,
        num_duration_bins=num_duration_bins,
        encoder_config=encoder_config.to_dict(),
        blank_token_id=vocab_size,
    )


def load_and_convert_tdt_state_dict(model_files, vocab_size, num_duration_bins):
    """Load NeMo TDT state dict and convert keys to HF format, splitting combined head."""
    state_dict = torch.load(model_files["model_weights"], map_location="cpu", weights_only=True)
    converted_state_dict = {}

    # Combine encoder and TDT mappings
    all_mappings = {**NEMO_TO_HF_WEIGHT_MAPPING, **NEMO_TDT_WEIGHT_MAPPING}

    for key, value in state_dict.items():
        # Skip preprocessing weights
        if key.endswith("featurizer.window") or key.endswith("featurizer.fb"):
            print(f"Skipping preprocessing weight: {key}")
            continue

        # Handle combined output head - needs to be split
        if key == "joint.joint_net.2.weight":
            # NeMo combines token and duration heads: [vocab_size+1+num_durations, joint_hidden]
            # Split into separate heads
            token_weight = value[: vocab_size + 1, :]  # First vocab_size+1 rows for tokens
            duration_weight = value[vocab_size + 1 :, :]  # Last num_duration_bins rows for durations
            converted_state_dict["joint.token_head.weight"] = token_weight
            converted_state_dict["joint.duration_head.weight"] = duration_weight
            print(f"Split combined weight: token_head {token_weight.shape}, duration_head {duration_weight.shape}")
            continue

        if key == "joint.joint_net.2.bias":
            # Same split for bias
            token_bias = value[: vocab_size + 1]
            duration_bias = value[vocab_size + 1 :]
            converted_state_dict["joint.token_head.bias"] = token_bias
            converted_state_dict["joint.duration_head.bias"] = duration_bias
            print(f"Split combined bias: token_head {token_bias.shape}, duration_head {duration_bias.shape}")
            continue

        # Standard key conversion
        converted_key = convert_key(key, all_mappings)
        converted_state_dict[converted_key] = value

    return converted_state_dict


def write_tdt_model(nemo_config, encoder_config, model_files, output_dir, push_to_repo_id=None):
    """Write TDT model using encoder config, TDT config, and converted state dict."""
    # Step 1: Convert TDT config
    model_config = convert_tdt_config(nemo_config, encoder_config)
    print(f"Converted TDT config: {model_config}")

    # Step 2: Load and convert state dict with TDT-specific handling
    converted_state_dict = load_and_convert_tdt_state_dict(
        model_files, model_config.vocab_size, model_config.num_duration_bins
    )

    print("Loading the checkpoint in a Parakeet TDT model.")
    with torch.device("meta"):
        model = ParakeetForTDT(model_config)

    # Load weights
    missing_keys, unexpected_keys = model.load_state_dict(converted_state_dict, strict=False, assign=True)

    if missing_keys:
        print(f"Warning: Missing keys: {missing_keys}")
    if unexpected_keys:
        print(f"Warning: Unexpected keys: {unexpected_keys}")

    if not missing_keys and not unexpected_keys:
        print("All weights loaded successfully!")
    else:
        # Re-try with strict to get detailed error if there are issues
        try:
            model.load_state_dict(converted_state_dict, strict=True, assign=True)
        except Exception as e:
            print(f"Strict loading failed: {e}")
            print("Continuing with partial weights...")

    del model.config._name_or_path

    print("Saving the model.")
    model.save_pretrained(output_dir)

    if push_to_repo_id:
        model.push_to_hub(push_to_repo_id)

    del model

    # Safety check: reload the converted model
    gc.collect()
    print("Reloading the model to check if it's saved correctly.")
    ParakeetForTDT.from_pretrained(output_dir, torch_dtype=torch.bfloat16, device_map="auto")
    print("Model reloaded successfully.")


def write_model(nemo_config, model_files, model_type, output_dir, push_to_repo_id=None):
    """Main model conversion function."""
    # Step 1: Convert encoder config (shared across all model types)
    encoder_config = convert_encoder_config(nemo_config)
    print(f"Converted encoder config: {encoder_config}")

    # Step 2: Write model based on type
    if model_type == "encoder":
        converted_state_dict = load_and_convert_state_dict(model_files)
        write_encoder_model(encoder_config, converted_state_dict, output_dir, push_to_repo_id)
    elif model_type == "ctc":
        converted_state_dict = load_and_convert_state_dict(model_files)
        write_ctc_model(encoder_config, converted_state_dict, output_dir, push_to_repo_id)
    elif model_type == "tdt":
        # TDT has its own state dict loading with combined head splitting
        write_tdt_model(nemo_config, encoder_config, model_files, output_dir, push_to_repo_id)
    else:
        raise ValueError(f"Model type {model_type} not supported.")


def main(
    hf_repo_id,
    output_dir,
    model_type,
    push_to_repo_id=None,
    skip_processor=False,
):
    nemo_filename = f"{hf_repo_id.split('/')[-1]}.nemo"
    filepath = cached_file(hf_repo_id, nemo_filename)

    model_files = extract_nemo_archive(filepath, os.path.dirname(filepath))
    nemo_config = yaml.load(open(model_files["model_config"], "r"), Loader=yaml.FullLoader)

    if not skip_processor:
        write_processor(nemo_config, model_files, output_dir, push_to_repo_id)
    else:
        print("Skipping processor conversion (--skip_processor flag set)")
    write_model(nemo_config, model_files, model_type, output_dir, push_to_repo_id)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf_repo_id", required=True, help="Model repo on huggingface.co")
    parser.add_argument(
        "--model_type", required=True, choices=["encoder", "ctc", "tdt"], help="Model type (`encoder`, `ctc`, `tdt`)"
    )
    parser.add_argument("--output_dir", required=True, help="Output directory for HuggingFace model")
    parser.add_argument("--push_to_repo_id", help="Repository ID to push the model to on the Hub")
    parser.add_argument("--skip_processor", action="store_true", help="Skip processor conversion")
    args = parser.parse_args()
    main(
        args.hf_repo_id,
        args.output_dir,
        args.model_type,
        args.push_to_repo_id,
        args.skip_processor,
    )
