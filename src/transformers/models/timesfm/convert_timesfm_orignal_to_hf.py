import argparse
import os
import re
import shutil

import numpy as np
import timesfm
import torch

from transformers import TimesFmConfig, TimesFmModelForPrediction


"""
Sample usage:

```
python src/transformers/models/timesfm/convert_timesfm_orignal_to_hf.py \
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


def write_model(model_path, safe_serialization=True, huggingface_repo_id="google/timesfm-2.0-500m-pytorch"):
    os.makedirs(model_path, exist_ok=True)
    tmp_model_path = os.path.join(model_path, "tmp")
    os.makedirs(tmp_model_path, exist_ok=True)

    tfm = timesfm.TimesFm(
        hparams=timesfm.TimesFmHparams(
            backend="cuda" if torch.cuda.is_available() else "cpu",
            per_core_batch_size=32,
            horizon_len=128,
            input_patch_len=32,
            output_patch_len=128,
            num_layers=50,
            model_dims=1280,
            use_positional_embedding=False,
            context_len=2048,
        ),
        checkpoint=timesfm.TimesFmCheckpoint(huggingface_repo_id=huggingface_repo_id),
    )

    timesfm_config = TimesFmConfig(
        patch_length=tfm.hparams.input_patch_len,
        context_length=tfm.hparams.context_len,
        horizon_length=tfm.hparams.horizon_len,
        num_hidden_layers=tfm.hparams.num_layers,
        hidden_size=tfm.hparams.model_dims,
        intermediate_size=tfm.hparams.model_dims,
        head_dim=tfm.hparams.model_dims // tfm.hparams.num_heads,
        num_attention_heads=tfm.hparams.num_heads,
        use_positional_embedding=tfm.hparams.use_positional_embedding,
    )
    timesfm_config.save_pretrained(tmp_model_path)
    timesfm_model = TimesFmModelForPrediction(timesfm_config)

    # copy the weights from the original model to the new model making
    original_model = tfm._model

    # mapping of the layers from the original model to the transformer model
    MODEL_LAYER_MAPPING = {
        "input_ff_layer.hidden_layer[0].weight": "decoder.input_ff_layer.input_layer.weight",
        "input_ff_layer.hidden_layer[0].bias": "decoder.input_ff_layer.input_layer.bias",
        "input_ff_layer.output_layer.weight": "decoder.input_ff_layer.output_layer.weight",
        "input_ff_layer.output_layer.bias": "decoder.input_ff_layer.output_layer.bias",
        "input_ff_layer.residual_layer.weight": "decoder.input_ff_layer.residual_layer.weight",
        "input_ff_layer.residual_layer.bias": "decoder.input_ff_layer.residual_layer.bias",
        "freq_emb.weight": "decoder.freq_emb.weight",
        "horizon_ff_layer.hidden_layer[0].weight": "horizon_ff_layer.input_layer.weight",
        "horizon_ff_layer.hidden_layer[0].bias": "horizon_ff_layer.input_layer.bias",
        "horizon_ff_layer.output_layer.weight": "horizon_ff_layer.output_layer.weight",
        "horizon_ff_layer.output_layer.bias": "horizon_ff_layer.output_layer.bias",
        "horizon_ff_layer.residual_layer.weight": "horizon_ff_layer.residual_layer.weight",
        "horizon_ff_layer.residual_layer.bias": "horizon_ff_layer.residual_layer.bias",
    }

    TRANSFORMER_LAYER_MAPPING = {
        "stacked_transformer.layers[{i}].self_attn.qkv_proj.weight": "decoder.layers[{i}].self_attn.qkv_proj.weight",
        "stacked_transformer.layers[{i}].self_attn.qkv_proj.bias": "decoder.layers[{i}].self_attn.qkv_proj.bias",
        "stacked_transformer.layers[{i}].self_attn.o_proj.weight": "decoder.layers[{i}].self_attn.o_proj.weight",
        "stacked_transformer.layers[{i}].self_attn.o_proj.bias": "decoder.layers[{i}].self_attn.o_proj.bias",
        "stacked_transformer.layers[{i}].self_attn.scaling": "decoder.layers[{i}].self_attn.scaling",
        "stacked_transformer.layers[{i}].mlp.gate_proj.weight": "decoder.layers[{i}].mlp.gate_proj.weight",
        "stacked_transformer.layers[{i}].mlp.gate_proj.bias": "decoder.layers[{i}].mlp.gate_proj.bias",
        "stacked_transformer.layers[{i}].mlp.down_proj.weight": "decoder.layers[{i}].mlp.down_proj.weight",
        "stacked_transformer.layers[{i}].mlp.down_proj.bias": "decoder.layers[{i}].mlp.down_proj.bias",
        "stacked_transformer.layers[{i}].mlp.layer_norm.weight": "decoder.layers[{i}].mlp.layer_norm.weight",
        "stacked_transformer.layers[{i}].mlp.layer_norm.bias": "decoder.layers[{i}].mlp.layer_norm.bias",
        "stacked_transformer.layers[{i}].input_layernorm.weight": "decoder.layers[{i}].input_layernorm.weight",
    }

    for old_key, new_key in MODEL_LAYER_MAPPING.items():
        try:
            old_attr = get_nested_attr(original_model, old_key)  # Get tensor from original model
            new_attr = get_nested_attr(timesfm_model, new_key)  # Get corresponding attribute in new model
            new_attr.data.copy_(old_attr.data)  # Copy data
        except AttributeError:
            print(f"Skipping {old_key} (not found in original model).")

    num_layers = len(timesfm_model.decoder.layers)
    for i in range(num_layers):
        for old_template, new_template in TRANSFORMER_LAYER_MAPPING.items():
            old_key = old_template.format(i=i)
            new_key = new_template.format(i=i)

            try:
                # Get tensor from original model
                old_attr = get_nested_attr(original_model, old_key)
                if "qkv_proj" in old_key:
                    # Split the tensor into q, k, v projections
                    q_proj, k_proj, v_proj = (
                        old_attr[: tfm.hparams.model_dims, ...],
                        old_attr[tfm.hparams.model_dims : tfm.hparams.model_dims * 2, ...],
                        old_attr[tfm.hparams.model_dims * 2 :, ...],
                    )
                    # Get corresponding attribute in new model
                    q_key = new_key.replace("qkv_proj", "q_proj")
                    q_attr = get_nested_attr(timesfm_model, q_key)
                    q_attr.data.copy_(q_proj.data)  # Copy data
                    k_key = new_key.replace("qkv_proj", "k_proj")
                    k_attr = get_nested_attr(timesfm_model, k_key)
                    k_attr.data.copy_(k_proj.data)  # Copy data
                    v_key = new_key.replace("qkv_proj", "v_proj")
                    v_attr = get_nested_attr(timesfm_model, v_key)
                    v_attr.data.copy_(v_proj.data)  # Copy data
                else:
                    # Get corresponding attribute in new model
                    new_attr = get_nested_attr(timesfm_model, new_key)
                    new_attr.data.copy_(old_attr.data)  # Copy data
            except AttributeError:
                print(f"Skipping {old_key} (not found in original model).")

    timesfm_model.save_pretrained(model_path, safe_serialization=safe_serialization)
    shutil.rmtree(tmp_model_path)


def check_outputs(model_path, huggingface_repo_id):
    """Compares outputs between original and converted models."""
    print("\nChecking model outputs...")

    # Load original model
    tfm = timesfm.TimesFm(
        hparams=timesfm.TimesFmHparams(
            backend="cuda" if torch.cuda.is_available() else "cpu",
            per_core_batch_size=32,
            horizon_len=128,
            input_patch_len=32,
            output_patch_len=128,
            num_layers=50,
            context_len=2048,
            model_dims=1280,
            use_positional_embedding=False,
            point_forecast_mode="mean",
        ),
        checkpoint=timesfm.TimesFmCheckpoint(huggingface_repo_id=huggingface_repo_id),
    )

    # Load converted model
    converted_model = TimesFmModelForPrediction.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
    ).to("cuda" if torch.cuda.is_available() else "cpu")
    converted_model.eval()  # Set to evaluation mode

    # Create test inputs
    forecast_input = [
        np.sin(np.linspace(0, 20, 100)),
        np.sin(np.linspace(0, 20, 200)),
        np.sin(np.linspace(0, 20, 400)),
    ]
    frequency_input = [0, 1, 2]

    # Get predictions from original model
    point_forecast_orig, quantile_forecast_orig = tfm.forecast(
        forecast_input,
        freq=frequency_input,
    )

    # Convert inputs to sequence of tensors
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

    # Define acceptable thresholds
    POINT_THRESHOLD = 1e-5
    QUANTILE_THRESHOLD = 1e-5

    if max_point_diff > POINT_THRESHOLD or max_quantile_diff > QUANTILE_THRESHOLD:
        raise ValueError(
            f"Output mismatch detected!\n"
            f"Point forecast max diff: {max_point_diff} (threshold: {POINT_THRESHOLD})\n"
            f"Quantile forecast max diff: {max_quantile_diff} (threshold: {QUANTILE_THRESHOLD})"
        )

    print("\n✓ All outputs match within acceptable tolerance!")

    # Optional: Print shapes for verification
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
        default="google/timesfm-2.0-500m-pytorch",
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
            safe_serialization=args.safe_serialization,
            huggingface_repo_id=args.huggingface_repo_id,
        )
    check_outputs(args.output_dir, args.huggingface_repo_id)


if __name__ == "__main__":
    main()
