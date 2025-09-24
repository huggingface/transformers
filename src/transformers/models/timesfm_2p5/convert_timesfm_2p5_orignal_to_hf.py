import argparse
import os
import re
import shutil

import numpy as np
import timesfm
import torch

from transformers import Timesfm2P5Config, Timesfm2P5ModelForPrediction


"""
Sample usage:

```
python src/transformers/models/timesfm_2p5/convert_timesfm_2p5_orignal_to_hf.py \
    --output_dir /output/path
```
"""


def get_nested_attr(obj, key):
    """Recursively retrieves an attribute from an object, handling list/tuple indexing if present."""
    parts = key.split(".")
    for part in parts:
        match = re.match(r"(.*)\[(\d+)\]", part)  # Handle list indexing like `layers[0]`
        if match:
            attr_name, index = match.groups()
            obj = getattr(obj, attr_name)[int(index)]  # Access list/tuple element
        else:
            obj = getattr(obj, part)  # Regular attribute access
    return obj


def write_model(model_path, safe_serialization=True, huggingface_repo_id="google/timesfm-2.5-200m-pytorch"):
    os.makedirs(model_path, exist_ok=True)
    tmp_model_path = os.path.join(model_path, "tmp")
    os.makedirs(tmp_model_path, exist_ok=True)

    # Initialize TimesFM 2.5 model with 200M parameters configuration
    tfm = timesfm.TimesFM_2p5_200M_torch()
    tfm.load_checkpoint()

    # Compile with forecasting configuration
    forecast_config = timesfm.ForecastConfig(
        frequency="D",  # Daily frequency as example
        horizon_len=128,
        context_len=16384,
        input_patch_len=32,
        output_patch_len=128,
        quantiles=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    )
    tfm.compile(forecast_config)

    # Create TimesFM 2.5 config for Transformers
    timesfm_config = Timesfm2P5Config(
        patch_length=32,  # input_patch_len
        context_length=16384,
        horizon_length=128,  # output_patch_len
        num_hidden_layers=20,
        hidden_size=1280,
        intermediate_size=1280,
        head_dim=80,  # 1280 / 16
        num_attention_heads=16,
        output_quantile_len=1024,
        decode_index=5,
        use_positional_embedding=False,
        use_rotary_embeddings=True,
        use_qk_norm=True,
        use_per_dim_scale=True,
        use_bias=False,
        activation="swish",
        query_pre_attn_scalar=256.0,
        quantiles=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    )
    timesfm_config.save_pretrained(tmp_model_path)
    timesfm_model = Timesfm2P5ModelForPrediction(timesfm_config)

    # Get the original model for weight copying
    original_model = tfm._model

    # Mapping for TimesFM 2.5 specific components
    MODEL_LAYER_MAPPING = {
        # Input projection (tokenizer) using ResidualBlock
        "tokenizer.hidden_layer.weight": "decoder.input_ff_layer.hidden_layer.weight",
        "tokenizer.output_layer.weight": "decoder.input_ff_layer.output_layer.weight",
        "tokenizer.residual_layer.weight": "decoder.input_ff_layer.residual_layer.weight",

        # No frequency embedding in TimesFM 2.5 - removed for simplified API

        # Output projection for quantiles and mean
        "output_projection_quantiles.hidden_layer.weight": "horizon_ff_layer.hidden_layer.weight",
        "output_projection_quantiles.output_layer.weight": "horizon_ff_layer.output_layer.weight",
        "output_projection_quantiles.residual_layer.weight": "horizon_ff_layer.residual_layer.weight",
    }

    # Transformer layers mapping for TimesFM 2.5 (20 layers)
    TRANSFORMER_LAYER_MAPPING = {
        # Attention layers with rotary embeddings and query scaling
        "stacked_transformers.layers[{i}].self_attn.q_proj.weight": "decoder.layers[{i}].self_attn.q_proj.weight",
        "stacked_transformers.layers[{i}].self_attn.k_proj.weight": "decoder.layers[{i}].self_attn.k_proj.weight",
        "stacked_transformers.layers[{i}].self_attn.v_proj.weight": "decoder.layers[{i}].self_attn.v_proj.weight",
        "stacked_transformers.layers[{i}].self_attn.o_proj.weight": "decoder.layers[{i}].self_attn.o_proj.weight",

        # QK normalization layers
        "stacked_transformers.layers[{i}].self_attn.query_ln.weight": "decoder.layers[{i}].self_attn.query_ln.weight",
        "stacked_transformers.layers[{i}].self_attn.key_ln.weight": "decoder.layers[{i}].self_attn.key_ln.weight",

        # MLP layers with configurable activation
        "stacked_transformers.layers[{i}].mlp.ff0.weight": "decoder.layers[{i}].mlp.ff0.weight",
        "stacked_transformers.layers[{i}].mlp.ff1.weight": "decoder.layers[{i}].mlp.ff1.weight",

        # Normalization layers (pre/post attention and feedforward)
        "stacked_transformers.layers[{i}].pre_attn_ln.weight": "decoder.layers[{i}].pre_attn_ln.weight",
        "stacked_transformers.layers[{i}].post_attn_ln.weight": "decoder.layers[{i}].post_attn_ln.weight",
        "stacked_transformers.layers[{i}].pre_ff_ln.weight": "decoder.layers[{i}].pre_ff_ln.weight",
        "stacked_transformers.layers[{i}].post_ff_ln.weight": "decoder.layers[{i}].post_ff_ln.weight",
    }

    # Copy model-level weights
    for old_key, new_key in MODEL_LAYER_MAPPING.items():
        try:
            old_attr = get_nested_attr(original_model, old_key)
            new_attr = get_nested_attr(timesfm_model, new_key)
            new_attr.data.copy_(old_attr.data)
        except AttributeError:
            print(f"Skipping {old_key} (not found in original model).")

    # Copy transformer layer weights
    num_layers = len(timesfm_model.decoder.layers)
    for i in range(num_layers):
        for old_template, new_template in TRANSFORMER_LAYER_MAPPING.items():
            old_key = old_template.format(i=i)
            new_key = new_template.format(i=i)

            try:
                old_attr = get_nested_attr(original_model, old_key)
                new_attr = get_nested_attr(timesfm_model, new_key)
                new_attr.data.copy_(old_attr.data)
            except AttributeError:
                print(f"Skipping {old_key} (not found in original model).")

    timesfm_model.save_pretrained(model_path, safe_serialization=safe_serialization)
    shutil.rmtree(tmp_model_path)


def check_outputs(model_path, huggingface_repo_id):
    """Compares outputs between original and converted models."""
    print("\nChecking model outputs...")

    # Load original TimesFM 2.5 model
    tfm = timesfm.TimesFM_2p5_200M_torch()
    tfm.load_checkpoint()

    # Compile with forecasting configuration
    forecast_config = timesfm.ForecastConfig(
        frequency="D",
        horizon_len=128,
        context_len=16384,
        input_patch_len=32,
        output_patch_len=128,
        quantiles=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    )
    tfm.compile(forecast_config)

    # Load converted model
    converted_model = Timesfm2P5ModelForPrediction.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
    ).to("cuda" if torch.cuda.is_available() else "cpu")
    converted_model.eval()

    # Create test inputs - multiple time series
    forecast_input = [
        np.sin(np.linspace(0, 20, 100)),
        np.sin(np.linspace(0, 20, 200)),
        np.sin(np.linspace(0, 20, 400)),
    ]
    frequency_input = [0, 1, 2]  # Daily, weekly, monthly frequencies

    # Get predictions from original TimesFM 2.5 model
    point_forecast_orig, quantile_forecast_orig = tfm.forecast(
        horizon=128,
        inputs=forecast_input,
    )

    # Convert inputs to tensors for Transformers model
    forecast_input_tensor = [
        torch.tensor(ts, dtype=torch.bfloat16).to("cuda" if torch.cuda.is_available() else "cpu")
        for ts in forecast_input
    ]
    frequency_input_tensor = torch.tensor(frequency_input, dtype=torch.long).to(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    # Get predictions from converted model
    with torch.no_grad():
        outputs = converted_model(past_values=forecast_input_tensor, freq=frequency_input_tensor, return_dict=True)
        point_forecast_conv = outputs.mean_predictions.float().cpu().numpy()
        quantile_forecast_conv = outputs.full_predictions.float().cpu().numpy()

    # Compare outputs
    point_forecast_diff = np.abs(point_forecast_orig - point_forecast_conv)
    quantile_forecast_diff = np.abs(quantile_forecast_orig - quantile_forecast_conv)

    max_point_diff = point_forecast_diff.max()
    mean_point_diff = point_forecast_diff.mean()
    max_quantile_diff = quantile_forecast_diff.max()
    mean_quantile_diff = quantile_forecast_diff.mean()

    print("\nOutput comparison:")
    print(f"Point forecast - Max difference: {max_point_diff:.6f}")
    print(f"Point forecast - Mean difference: {mean_point_diff:.6f}")
    print(f"Quantile forecast - Max difference: {max_quantile_diff:.6f}")
    print(f"Quantile forecast - Mean difference: {mean_quantile_diff:.6f}")

    # Define acceptable thresholds for TimesFM 2.5
    POINT_THRESHOLD = 1e-5
    QUANTILE_THRESHOLD = 1e-5

    if max_point_diff > POINT_THRESHOLD or max_quantile_diff > QUANTILE_THRESHOLD:
        raise ValueError(
            f"Output mismatch detected!\n"
            f"Point forecast max diff: {max_point_diff} (threshold: {POINT_THRESHOLD})\n"
            f"Quantile forecast max diff: {max_quantile_diff} (threshold: {QUANTILE_THRESHOLD})"
        )

    print("\n All outputs match within acceptable tolerance!")

    # Print shapes for verification
    print("\nOutput shapes:")
    print(f"Original point forecast: {point_forecast_orig.shape}")
    print(f"Converted point forecast: {point_forecast_conv.shape}")
    print(f"Original quantile forecast: {quantile_forecast_orig.shape}")
    print(f"Converted quantile forecast: {quantile_forecast_conv.shape}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Location to write HF model and tokenizer",
    )
    parser.add_argument(
        "--safe_serialization", type=bool, default=True, help="Whether or not to save using `safetensors`."
    )
    parser.add_argument(
        "--huggingface_repo_id",
        type=str,
        default="google/timesfm-2.5-200m-pytorch",
        help="The Hugging Face repository ID to use for the model.",
    )
    args = parser.parse_args()

    # Check if model already exists
    if os.path.exists(
        os.path.join(args.output_dir, "model.safetensors" if args.safe_serialization else "pytorch_model.bin")
    ):
        print(f"Model already exists in {args.output_dir}, skipping conversion.")
    else:
        write_model(
            model_path=args.output_dir,
            safe_serialization=args.safe_serialization,
            huggingface_repo_id=args.huggingface_repo_id,
        )
    check_outputs(args.output_dir, args.huggingface_repo_id)


if __name__ == "__main__":
    main()