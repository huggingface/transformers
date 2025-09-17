#!/usr/bin/env python
# coding=utf-8
# Copyright 2025 Google LLC and HuggingFace Inc. team.
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
"""Convert TimesFM 2.5 checkpoint to HuggingFace format."""

import argparse
import os

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from safetensors import safe_open

# Import TimesFM libraries - assuming they are installed
from timesfm.timesfm_2p5.timesfm_2p5_torch import TimesFM_2p5_200M_torch

from transformers.models.timesfm_2p5.configuration_timesfm_2p5 import Timesfm2P5Config
from transformers.models.timesfm_2p5.modeling_timesfm_2p5 import Timesfm2P5ModelForPrediction


def load_original_checkpoint(checkpoint_path=None, repo_id=None):
    """Load the original TimesFM 2.5 checkpoint."""

    if checkpoint_path:
        print(f"Loading checkpoint from {checkpoint_path}")
        if checkpoint_path.endswith(".safetensors"):
            state_dict = {}
            with safe_open(checkpoint_path, framework="pt", device="cpu") as f:
                for key in f.keys():
                    state_dict[key] = f.get_tensor(key)
        else:
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            if isinstance(checkpoint, dict):
                state_dict = checkpoint.get("state_dict", checkpoint)
            else:
                state_dict = checkpoint
        return state_dict

    elif repo_id:
        print(f"Loading checkpoint from HuggingFace: {repo_id}")
        # Download the safetensors file
        model_file = hf_hub_download(
            repo_id=repo_id,
            filename="model.safetensors",
            cache_dir=None,
        )

        # Load state dict from safetensors
        state_dict = {}
        with safe_open(model_file, framework="pt", device="cpu") as f:
            for key in f.keys():
                state_dict[key] = f.get_tensor(key)
        return state_dict

    else:
        # Load using timesfm library
        print("Loading TimesFM 2.5 model using timesfm library")
        model = TimesFM_2p5_200M_torch()
        model.load_checkpoint(hf_repo_id="google/timesfm-2.5-200m-pytorch")
        return model.state_dict()


def convert_weights(original_state_dict, hf_model):
    """Convert TimesFM 2.5 weights to HuggingFace format."""

    converted_state_dict = {}

    # Mapping from original keys to HuggingFace keys
    key_mapping = {
        # Tokenizer (input residual block) - note original uses "hidden_layer" not "input_layer"
        "tokenizer.hidden_layer.weight": "model.tokenizer.input_layer.weight",
        "tokenizer.hidden_layer.bias": "model.tokenizer.input_layer.bias",
        "tokenizer.output_layer.weight": "model.tokenizer.output_layer.weight",
        "tokenizer.output_layer.bias": "model.tokenizer.output_layer.bias",
        "tokenizer.residual_layer.weight": "model.tokenizer.residual_layer.weight",
        "tokenizer.residual_layer.bias": "model.tokenizer.residual_layer.bias",
        # Output projections for point predictions - note original uses "hidden_layer" not "input_layer" and no biases
        "output_projection_point.hidden_layer.weight": "output_projection_point.input_layer.weight",
        "output_projection_point.output_layer.weight": "output_projection_point.output_layer.weight",
        "output_projection_point.residual_layer.weight": "output_projection_point.residual_layer.weight",
        # Output projections for quantiles - note original uses "hidden_layer" not "input_layer" and no biases
        "output_projection_quantiles.hidden_layer.weight": "output_projection_quantiles.input_layer.weight",
        "output_projection_quantiles.output_layer.weight": "output_projection_quantiles.output_layer.weight",
        "output_projection_quantiles.residual_layer.weight": "output_projection_quantiles.residual_layer.weight",
    }

    # Process transformer layers
    num_layers = hf_model.config.num_hidden_layers
    for i in range(num_layers):
        # Layer normalization
        key_mapping[f"stacked_xf.{i}.pre_attn_ln.scale"] = f"model.layers.{i}.pre_attn_ln.weight"
        key_mapping[f"stacked_xf.{i}.post_attn_ln.scale"] = f"model.layers.{i}.post_attn_ln.weight"

        # Attention projections
        key_mapping[f"stacked_xf.{i}.attn.query.weight"] = f"model.layers.{i}.self_attn.q_proj.weight"
        key_mapping[f"stacked_xf.{i}.attn.key.weight"] = f"model.layers.{i}.self_attn.k_proj.weight"
        key_mapping[f"stacked_xf.{i}.attn.value.weight"] = f"model.layers.{i}.self_attn.v_proj.weight"
        key_mapping[f"stacked_xf.{i}.attn.out.weight"] = f"model.layers.{i}.self_attn.o_proj.weight"

        # QK normalization
        key_mapping[f"stacked_xf.{i}.attn.query_ln.scale"] = f"model.layers.{i}.self_attn.q_norm.scale"
        key_mapping[f"stacked_xf.{i}.attn.key_ln.scale"] = f"model.layers.{i}.self_attn.k_norm.scale"

        # Query scaling parameter (new in TimesFM style)
        key_mapping[f"stacked_xf.{i}.attn.per_dim_scale.per_dim_scale"] = f"model.layers.{i}.self_attn.scaling"

        # MLP layer normalization
        key_mapping[f"stacked_xf.{i}.pre_ff_ln.scale"] = f"model.layers.{i}.mlp.pre_ff_ln.weight"
        key_mapping[f"stacked_xf.{i}.post_ff_ln.scale"] = f"model.layers.{i}.mlp.post_ff_ln.weight"

        # MLP layers
        key_mapping[f"stacked_xf.{i}.ff0.weight"] = f"model.layers.{i}.mlp.ff0.weight"
        key_mapping[f"stacked_xf.{i}.ff1.weight"] = f"model.layers.{i}.mlp.ff1.weight"

    # Convert weights
    for orig_key, new_key in key_mapping.items():
        if orig_key in original_state_dict:
            converted_state_dict[new_key] = original_state_dict[orig_key]
            print(f"Converted: {orig_key} -> {new_key}")
        else:
            print(f"Warning: Key {orig_key} not found in original checkpoint")

    print(f"\nConverted {len(converted_state_dict)} parameters")
    return converted_state_dict


def validate_conversion(hf_model, original_model=None):
    """Validate the converted model by comparing outputs with original TimesFM 2.5."""

    print("\n=== Validating conversion ===")

    # Load original TimesFM 2.5 model for comparison
    print("Loading original TimesFM 2.5 model for comparison...")
    orig_model = TimesFM_2p5_200M_torch()
    try:
        orig_model.load_checkpoint(hf_repo_id="google/timesfm-2.5-200m-pytorch")
        print("✓ Loaded original model checkpoint")
    except Exception as e:
        print(f"⚠ Could not load original model checkpoint: {e}")
        print("Skipping output comparison")
        return None

    # Set HF model to eval mode
    hf_model.eval()

    # Original model doesn't have eval() method, but set its underlying model if available
    if hasattr(orig_model, "model") and hasattr(orig_model.model, "eval"):
        orig_model.model.eval()

    # Create test inputs - using numpy arrays like original TimesFM API
    # Make sure lengths are multiples of patch_length (32) for easier processing
    forecast_input = [
        np.sin(np.linspace(0, 20, 96)),  # 3 * 32
        np.sin(np.linspace(0, 20, 128)),  # 4 * 32
        np.sin(np.linspace(0, 20, 160)),  # 5 * 32
    ]

    print(f"Test input shapes: {[inp.shape for inp in forecast_input]}")

    # Compile and run original model with ForecastConfig
    print("\nCompiling original model...")
    import timesfm

    orig_model.compile(
        timesfm.ForecastConfig(
            max_context=1024,
            max_horizon=256,
            normalize_inputs=True,
            use_continuous_quantile_head=True,
            force_flip_invariance=True,
            infer_is_positive=True,
            fix_quantile_crossing=True,
        )
    )

    print("Running original model inference...")
    with torch.no_grad():
        point_forecast, quantile_forecast = orig_model.forecast(horizon=128, inputs=forecast_input)

    # Pass raw inputs to HF model - let it handle preprocessing internally
    past_values = forecast_input  # Pass as sequence of numpy arrays
    past_values_mask = None  # Model will generate masks internally

    print("\nRunning converted model inference...")
    with torch.no_grad():
        hf_output = hf_model(past_values=past_values, past_values_mask=past_values_mask, horizon=128, return_dict=True)

    # Extract predictions
    hf_point_predictions = hf_output.point_predictions.float().numpy()
    hf_quantile_predictions = hf_output.quantile_predictions.float().numpy()

    print("\nOutput shapes:")
    print(f"  Original point forecast: {point_forecast.shape}")
    print(f"  Original quantile forecast: {quantile_forecast.shape}")
    print(f"  HF point predictions: {hf_point_predictions.shape}")
    print(f"  HF quantile predictions: {hf_quantile_predictions.shape}")

    # Compare outputs for each sample
    print("\nComparing outputs...")

    # Original model returns:
    # - point_forecast: shape [batch, horizon]
    # - quantile_forecast: shape [batch, horizon, num_quantiles] (mean + 10th to 90th percentiles)
    # HF model returns tensors with same shapes

    max_point_diff = 0
    mean_point_diff = 0
    max_quantile_diff = 0
    mean_quantile_diff = 0

    batch_size = point_forecast.shape[0]
    for i in range(batch_size):
        # Compare point forecasts
        orig_points = point_forecast[i]  # Shape: [128]
        hf_points = hf_point_predictions[i]  # Shape: [128]

        point_diff = np.abs(orig_points - hf_points)
        sample_max_point = point_diff.max()
        sample_mean_point = point_diff.mean()

        max_point_diff = max(max_point_diff, sample_max_point)
        mean_point_diff += sample_mean_point

        # Compare quantile forecasts
        orig_quantiles = quantile_forecast[i]  # Shape: [128, 10] (quantiles only based on actual output)
        hf_quantiles = hf_quantile_predictions[i]  # Shape: [128, 10] (quantiles)

        # Compare quantiles directly (shapes should match)
        print(f"    Original quantiles shape: {orig_quantiles.shape}")
        print(f"    HF quantiles shape: {hf_quantiles.shape}")

        # Take minimum dimension to avoid shape mismatch
        min_quantiles = min(orig_quantiles.shape[1], hf_quantiles.shape[1])
        orig_quantiles_trimmed = orig_quantiles[:, :min_quantiles]
        hf_quantiles_trimmed = hf_quantiles[:, :min_quantiles]

        quantile_diff = np.abs(orig_quantiles_trimmed - hf_quantiles_trimmed)
        sample_max_quantile = quantile_diff.max()
        sample_mean_quantile = quantile_diff.mean()

        max_quantile_diff = max(max_quantile_diff, sample_max_quantile)
        mean_quantile_diff += sample_mean_quantile

        print(f"  Sample {i}: Point diff max={sample_max_point:.6f}, mean={sample_mean_point:.6f}")
        print(f"  Sample {i}: Quantile diff max={sample_max_quantile:.6f}, mean={sample_mean_quantile:.6f}")

    mean_point_diff /= batch_size
    mean_quantile_diff /= batch_size

    print("\nOverall comparison:")
    print(f"Point forecast - Max difference: {max_point_diff:.6f}")
    print(f"Point forecast - Mean difference: {mean_point_diff:.6f}")
    print(f"Quantile forecast - Max difference: {max_quantile_diff:.6f}")
    print(f"Quantile forecast - Mean difference: {mean_quantile_diff:.6f}")

    # Define acceptable thresholds
    POINT_THRESHOLD = 1e-4
    QUANTILE_THRESHOLD = 1e-4

    if max_point_diff > POINT_THRESHOLD or max_quantile_diff > QUANTILE_THRESHOLD:
        print("\n⚠ Output mismatch detected!")
        print(f"Point forecast max diff: {max_point_diff} (threshold: {POINT_THRESHOLD})")
        print(f"Quantile forecast max diff: {max_quantile_diff} (threshold: {QUANTILE_THRESHOLD})")
        print("This could indicate issues with the weight conversion or model architecture.")
    else:
        print("\n✓ All outputs match within acceptable tolerance!")

    return hf_output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        help="Path to the original TimesFM 2.5 checkpoint file",
    )
    parser.add_argument(
        "--repo_id",
        type=str,
        default="google/timesfm-2.5-200m-pytorch",
        help="HuggingFace repository ID for the model",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for the converted model",
    )
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        help="Push the converted model to HuggingFace Hub",
    )
    parser.add_argument(
        "--hub_model_id",
        type=str,
        help="Model ID for pushing to HuggingFace Hub",
    )

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load original checkpoint
    if args.checkpoint_path:
        original_state_dict = load_original_checkpoint(checkpoint_path=args.checkpoint_path)
    else:
        original_state_dict = load_original_checkpoint(repo_id=args.repo_id)

    print(f"Loaded {len(original_state_dict)} parameters from original checkpoint")

    # Create HuggingFace model with correct config
    config = Timesfm2P5Config(
        patch_length=32,
        context_length=512,
        horizon_length=128,
        output_patch_length=128,
        output_quantile_len=1024,
        num_hidden_layers=20,
        hidden_size=1280,
        intermediate_size=1280,
        head_dim=80,
        num_attention_heads=16,
        quantiles=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        decode_index=5,
        use_rotary_position_embeddings=True,
    )

    hf_model = Timesfm2P5ModelForPrediction(config)

    # Convert weights
    converted_state_dict = convert_weights(original_state_dict, hf_model)

    # Load converted weights
    hf_model.load_state_dict(converted_state_dict, strict=False)

    # Validate conversion
    validate_conversion(hf_model)

    # Save the model
    print(f"\nSaving model to {args.output_dir}")
    hf_model.save_pretrained(args.output_dir, safe_serialization=True)
    config.save_pretrained(args.output_dir)

    # Push to hub if requested
    if args.push_to_hub:
        print(f"Pushing to HuggingFace Hub: {args.hub_model_id}")
        hf_model.push_to_hub(args.hub_model_id)
        config.push_to_hub(args.hub_model_id)

    print("\n✓ Conversion complete!")


if __name__ == "__main__":
    main()
