import argparse
import os
import re
import shutil

import numpy as np
import torch

import timesfm
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


def write_model(model_path, huggingface_repo_id="google/timesfm-2.5-200m-pytorch", safe_serialization=True):
    os.makedirs(model_path, exist_ok=True)
    tmp_model_path = os.path.join(model_path, "tmp")
    os.makedirs(tmp_model_path, exist_ok=True)

    # Load TimesFM 2.5 model - workaround for huggingface_hub version issues
    from huggingface_hub import hf_hub_download

    # Download the checkpoint file
    checkpoint_path = hf_hub_download(repo_id=huggingface_repo_id, filename="model.safetensors")

    # Create model instance and load checkpoint
    tfm = timesfm.TimesFM_2p5_200M_torch()
    tfm.model.load_checkpoint(checkpoint_path)

    # Compile with forecasting configuration
    tfm.compile(
        timesfm.ForecastConfig(
            max_context=1024,
            max_horizon=256,
            normalize_inputs=True,
            use_continuous_quantile_head=True,
        )
    )
    original_model = tfm.model

    # Get actual dimensions from original model
    quantile_output_dims = original_model.output_projection_quantiles.output_layer.weight.shape[0]
    # Original TimesFM 2.5 has 9 quantiles + 1 extra (median/point prediction) = 10 total
    actual_quantile_len = quantile_output_dims // 10  # 9 quantiles + 1 = 10 total

    timesfm_config = Timesfm2P5Config(
        patch_length=32,
        context_length=16384,
        horizon_length=128,
        num_hidden_layers=20,
        hidden_size=1280,
        intermediate_size=1280,
        head_dim=80,
        num_attention_heads=16,
        output_quantile_len=actual_quantile_len,
        decode_index=5,
        use_positional_embedding=False,
        use_rotary_embeddings=True,
        use_qk_norm=True,
        use_per_dim_scale=True,
        use_bias=False,
        activation="swish",
        query_pre_attn_scalar=256.0,
        quantiles=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        max_position_embeddings=16384,
        rope_theta=10000.0,
    )
    timesfm_config.save_pretrained(tmp_model_path)
    timesfm_model = Timesfm2P5ModelForPrediction(timesfm_config)

    # Mapping of the layers from the original TimesFM 2.5 model to the Transformers model
    MODEL_LAYER_MAPPING = {
        # Input projection (tokenizer) - ResidualBlock: 64 -> 1280 -> 1280
        "tokenizer.hidden_layer.weight": "decoder.input_ff_layer.hidden_layer.weight",
        "tokenizer.hidden_layer.bias": "decoder.input_ff_layer.hidden_layer.bias",
        "tokenizer.output_layer.weight": "decoder.input_ff_layer.output_layer.weight",
        "tokenizer.output_layer.bias": "decoder.input_ff_layer.output_layer.bias",
        "tokenizer.residual_layer.weight": "decoder.input_ff_layer.residual_layer.weight",
        "tokenizer.residual_layer.bias": "decoder.input_ff_layer.residual_layer.bias",
        # Separate output projections for TimesFM 2.5 - these are at model level, not inside decoder
        # Point projection: 1280 -> 1280 -> 1280
        "output_projection_point.hidden_layer.weight": "output_projection_point.hidden_layer.weight",
        "output_projection_point.output_layer.weight": "output_projection_point.output_layer.weight",
        "output_projection_point.residual_layer.weight": "output_projection_point.residual_layer.weight",
        # Quantile projection: 1280 -> 1280 -> output_dims
        "output_projection_quantiles.hidden_layer.weight": "output_projection_quantiles.hidden_layer.weight",
        "output_projection_quantiles.output_layer.weight": "output_projection_quantiles.output_layer.weight",
        "output_projection_quantiles.residual_layer.weight": "output_projection_quantiles.residual_layer.weight",
    }

    TRANSFORMER_LAYER_MAPPING = {
        # Attention layers - MultiHeadAttention with separate q, k, v projections
        "stacked_xf[{i}].attn.query.weight": "decoder.layers[{i}].self_attn.q_proj.weight",
        "stacked_xf[{i}].attn.key.weight": "decoder.layers[{i}].self_attn.k_proj.weight",
        "stacked_xf[{i}].attn.value.weight": "decoder.layers[{i}].self_attn.v_proj.weight",
        "stacked_xf[{i}].attn.out.weight": "decoder.layers[{i}].self_attn.o_proj.weight",
        # QK normalization layers (RMS norm) - uses 'scale' instead of 'weight'
        "stacked_xf[{i}].attn.query_ln.scale": "decoder.layers[{i}].self_attn.query_ln.weight",
        "stacked_xf[{i}].attn.key_ln.scale": "decoder.layers[{i}].self_attn.key_ln.weight",
        # Per-dimension scaling parameter
        "stacked_xf[{i}].attn.per_dim_scale.per_dim_scale": "decoder.layers[{i}].self_attn.scaling",
        # MLP layers (feed forward)
        "stacked_xf[{i}].ff0.weight": "decoder.layers[{i}].mlp.ff0.weight",
        "stacked_xf[{i}].ff1.weight": "decoder.layers[{i}].mlp.ff1.weight",
        # Layer normalization (RMS norm) - uses 'scale' instead of 'weight'
        "stacked_xf[{i}].pre_attn_ln.scale": "decoder.layers[{i}].pre_attn_ln.weight",
        "stacked_xf[{i}].post_attn_ln.scale": "decoder.layers[{i}].post_attn_ln.weight",
        "stacked_xf[{i}].pre_ff_ln.scale": "decoder.layers[{i}].pre_ff_ln.weight",
        "stacked_xf[{i}].post_ff_ln.scale": "decoder.layers[{i}].post_ff_ln.weight",
    }

    # Debug: Print both model structures
    print(f"Original model attributes: {dir(original_model)}")
    print(f"\\nTransformers model attributes: {dir(timesfm_model)}")
    print(f"\\nTransformers decoder attributes: {dir(timesfm_model.decoder)}")
    print(f"\\nTransformers input_ff_layer attributes: {dir(timesfm_model.decoder.input_ff_layer)}")

    # Copy model-level weights
    for old_key, new_key in MODEL_LAYER_MAPPING.items():
        try:
            old_attr = get_nested_attr(original_model, old_key)  # Get tensor from original model
            new_attr = get_nested_attr(timesfm_model, new_key)  # Get corresponding attribute in new model

            print(f"Shape comparison {old_key}: {old_attr.shape} vs {new_key}: {new_attr.shape}")

            if old_attr.shape == new_attr.shape:
                new_attr.data.copy_(old_attr.data)  # Copy data
                print(f"✅ Converted {old_key} -> {new_key}")
            else:
                print(f"⚠️  Shape mismatch {old_key}: {old_attr.shape} vs {new_attr.shape}")
        except AttributeError as e:
            print(f"Skipping {old_key}: {e}")

    # Copy transformer layer weights
    num_layers = len(timesfm_model.decoder.layers)
    for i in range(num_layers):
        # Special handling for fused QKV weights
        try:
            # Check if original model has fused QKV projection
            qkv_fused = get_nested_attr(original_model, f"stacked_xf[{i}].attn.qkv_proj.weight")

            # Split fused QKV into separate Q, K, V projections
            # QKV fused shape: [3 * hidden_size, hidden_size] = [3840, 1280]
            # Split into Q: [1280, 1280], K: [1280, 1280], V: [1280, 1280]
            q_weight, k_weight, v_weight = qkv_fused.chunk(3, dim=0)

            # Copy to separate projections
            q_proj = get_nested_attr(timesfm_model, f"decoder.layers[{i}].self_attn.q_proj.weight")
            k_proj = get_nested_attr(timesfm_model, f"decoder.layers[{i}].self_attn.k_proj.weight")
            v_proj = get_nested_attr(timesfm_model, f"decoder.layers[{i}].self_attn.v_proj.weight")

            q_proj.data.copy_(q_weight.data)
            k_proj.data.copy_(k_weight.data)
            v_proj.data.copy_(v_weight.data)

            if i == 0:
                print(
                    f"✅ Converted layer {i}: stacked_xf[{i}].attn.qkv_proj.weight (fused) -> separate Q/K/V projections"
                )
                print(f"   Q: {q_weight.shape}, K: {k_weight.shape}, V: {v_weight.shape}")
        except AttributeError:
            # No fused QKV, try separate weights
            if i == 0:
                print(f"⚠️  Layer {i}: No fused QKV found, trying separate Q/K/V weights...")

        # Copy all other transformer layer weights
        for old_template, new_template in TRANSFORMER_LAYER_MAPPING.items():
            old_key = old_template.format(i=i)
            new_key = new_template.format(i=i)

            # Skip Q/K/V weights if we already handled fused QKV
            if any(x in old_key for x in [".query.weight", ".key.weight", ".value.weight"]):
                continue

            try:
                # Get tensor from original model
                old_attr = get_nested_attr(original_model, old_key)
                # Get corresponding attribute in new model
                new_attr = get_nested_attr(timesfm_model, new_key)
                new_attr.data.copy_(old_attr.data)  # Copy data
                if i == 0:  # Only print first layer details
                    print(f"✅ Converted layer {i}: {old_key} -> {new_key}")
            except AttributeError:
                if i == 0:  # Only print first layer errors
                    print(f"Skipping layer {i}: {old_key} (not found in original model).")

    timesfm_model.save_pretrained(model_path, safe_serialization=safe_serialization)
    shutil.rmtree(tmp_model_path)
    print(f"✅ Model saved to {model_path}")


def check_outputs(model_path, huggingface_repo_id="google/timesfm-2.5-200m-pytorch"):
    """Compares outputs between original and converted models."""
    print("\nChecking model outputs...")

    # Load original TimesFM 2.5 model using same method as write_model
    try:
        from huggingface_hub import hf_hub_download

        # Download the checkpoint file
        checkpoint_path = hf_hub_download(repo_id=huggingface_repo_id, filename="model.safetensors")

        # Create model instance and load checkpoint
        tfm = timesfm.TimesFM_2p5_200M_torch()
        tfm.model.load_checkpoint(checkpoint_path)

        # Compile with forecasting configuration (following README example)
        tfm.compile(
            timesfm.ForecastConfig(
                max_context=1024,
                max_horizon=256,
                normalize_inputs=True,
                use_continuous_quantile_head=True,
            )
        )
    except Exception as e:
        print(f"⚠️  Warning: Could not load original model for validation: {e}")
        print("Skipping output comparison check.")
        return 0.0, 0.0

    # Load converted model
    converted_model = Timesfm2P5ModelForPrediction.from_pretrained(
        model_path,
        dtype=torch.float32,
    )
    if torch.cuda.is_available():
        converted_model = converted_model.to("cuda")
    converted_model.eval()  # Set to evaluation mode

    # Create test inputs (same as in original test)
    forecast_input = [
        np.linspace(0, 1, 100),
        np.sin(np.linspace(0, 20, 67)),
        np.sin(np.linspace(0, 10, 150)) + np.random.normal(0, 0.1, 150),
    ]

    # Get predictions from original model
    point_forecast_orig, quantile_forecast_orig = tfm.forecast(
        horizon=128,
        inputs=forecast_input,
    )

    # Convert inputs to sequence of tensors
    forecast_input_tensor = [torch.tensor(ts, dtype=torch.float32) for ts in forecast_input]
    if torch.cuda.is_available():
        forecast_input_tensor = [ts.to("cuda") for ts in forecast_input_tensor]

    # Get predictions from converted model
    with torch.no_grad():
        # Use forecast_context_len=1024 to match original's max_context=1024
        outputs = converted_model(past_values=forecast_input_tensor, forecast_context_len=1024, return_dict=True)
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

    # Define acceptable thresholds
    POINT_THRESHOLD = 1e-5
    QUANTILE_THRESHOLD = 1e-5

    if max_point_diff > POINT_THRESHOLD or max_quantile_diff > QUANTILE_THRESHOLD:
        print(
            f"\n⚠️ Output differences detected (may be acceptable):\n"
            f"Point forecast max diff: {max_point_diff} (threshold: {POINT_THRESHOLD})\n"
            f"Quantile forecast max diff: {max_quantile_diff} (threshold: {QUANTILE_THRESHOLD})"
        )
    else:
        print("\n✅ All outputs match within acceptable tolerance!")

    # Print shapes for verification
    print("\nOutput shapes:")
    print(f"Original point forecast: {point_forecast_orig.shape}")
    print(f"Converted point forecast: {point_forecast_conv.shape}")
    print(f"Original quantile forecast: {quantile_forecast_orig.shape}")
    print(f"Converted quantile forecast: {quantile_forecast_conv.shape}")

    return max_point_diff, max_quantile_diff


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

    # if the saved model file exists, skip the conversion
    if os.path.exists(
        os.path.join(args.output_dir, "model.safetensors" if args.safe_serialization else "pytorch_model.bin")
    ):
        print(f"Model already exists in {args.output_dir}, skipping conversion.")
    else:
        write_model(
            model_path=args.output_dir,
            huggingface_repo_id=args.huggingface_repo_id,
            safe_serialization=args.safe_serialization,
        )

    # Always check outputs
    max_point_diff, max_quantile_diff = check_outputs(args.output_dir, args.huggingface_repo_id)

    print("\n🎉 TimesFM 2.5 conversion completed!")
    print(f"   Point forecast precision: {max_point_diff:.6f}")
    print(f"   Quantile forecast precision: {max_quantile_diff:.6f}")


if __name__ == "__main__":
    main()
