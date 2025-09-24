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
    --output_dir /output/path --test_original_only
```
"""


def test_original_model_only(output_dir):
    """Test original TimesFM 2.5 model functionality."""
    print("=== Testing Original TimesFM 2.5 Model ===")

    # Initialize TimesFM 2.5 model
    tfm = timesfm.TimesFM_2p5_200M_torch()
    tfm.load_checkpoint()

    # Compile with forecasting configuration
    forecast_config = timesfm.ForecastConfig(
        max_context=512,  # Small context for testing
        max_horizon=128,
        normalize_inputs=True,
        use_continuous_quantile_head=True,
    )
    tfm.compile(forecast_config)

    # Create test inputs - multiple time series
    forecast_input = [
        np.sin(np.linspace(0, 20, 100)),
        np.sin(np.linspace(0, 20, 200)),
        np.sin(np.linspace(0, 20, 300)),
    ]

    print(f"Input time series shapes: {[ts.shape for ts in forecast_input]}")

    # Get predictions from original TimesFM 2.5 model
    try:
        # Use correct TimesFM 2.5 API with horizon parameter
        point_forecast, quantile_forecast = tfm.forecast(
            horizon=128,  # Forecast 128 steps ahead
            inputs=forecast_input,
        )
        print("✅ Original TimesFM 2.5 forecast successful!")
        print(f"Point forecast shape: {point_forecast.shape}")
        print(f"Quantile forecast shape: {quantile_forecast.shape}")

        # Check quantile dimensions to understand the model structure
        print(f"Model quantile output layer shape: {tfm.model.output_projection_quantiles.output_layer.weight.shape}")
        print(f"Quantile output dims: {tfm.model.output_projection_quantiles.output_layer.weight.shape[0]}")
        print(f"If 10 quantiles: {tfm.model.output_projection_quantiles.output_layer.weight.shape[0] // 1024}")

        # Save results for comparison
        results = {
            'point_forecast': point_forecast,
            'quantile_forecast': quantile_forecast,
            'input_shapes': [ts.shape for ts in forecast_input]
        }
        np.savez(os.path.join(output_dir, 'original_results.npz'), **results)
        print(f"✅ Results saved to {output_dir}/original_results.npz")

        # Inspect model structure
        print("\n=== Model Structure Inspection ===")
        model = tfm.model
        print(f"Model type: {type(model)}")
        print(f"Tokenizer input shape: {model.tokenizer.hidden_layer.weight.shape}")
        print(f"Tokenizer output shape: {model.tokenizer.output_layer.weight.shape}")
        print(f"Point projection output shape: {model.output_projection_point.output_layer.weight.shape}")
        print(f"Quantile projection output shape: {model.output_projection_quantiles.output_layer.weight.shape}")
        print(f"Number of transformer layers: {len(model.stacked_xf)}")

        if len(model.stacked_xf) > 0:
            layer0 = model.stacked_xf[0]
            print(f"Layer 0 attention query shape: {layer0.attn.query.weight.shape}")
            print(f"Layer 0 ff0 shape: {layer0.ff0.weight.shape}")
            print(f"Layer 0 ff1 shape: {layer0.ff1.weight.shape}")

        return True
    except Exception as e:
        print(f"❌ Original TimesFM 2.5 forecast failed: {e}")
        return False


def create_config_only(output_dir):
    """Create TimesFM 2.5 config based on original model."""
    print("=== Creating TimesFM 2.5 Config ===")

    # Initialize TimesFM 2.5 model to inspect structure
    tfm = timesfm.TimesFM_2p5_200M_torch()
    tfm.load_checkpoint()

    original_model = tfm.model

    # Check actual quantile output dimensions from original model
    quantile_output_dims = original_model.output_projection_quantiles.output_layer.weight.shape[0]
    actual_quantile_len = quantile_output_dims // 10  # Should be 1024

    print(f"Original model quantile output dims: {quantile_output_dims}")
    print(f"Calculated quantile_len: {actual_quantile_len}")

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
        output_quantile_len=actual_quantile_len,  # 1024
        decode_index=5,
        use_positional_embedding=False,
        use_rotary_embeddings=True,
        use_qk_norm=True,
        use_per_dim_scale=True,
        use_bias=False,
        activation="swish",
        query_pre_attn_scalar=256.0,
        quantiles=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],  # 10 quantiles
    )

    timesfm_config.save_pretrained(output_dir)
    print(f"✅ Config saved to {output_dir}")

    return timesfm_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Location to write HF model and tokenizer",
    )
    parser.add_argument(
        "--test_original_only",
        action="store_true",
        help="Only test the original TimesFM 2.5 model"
    )
    parser.add_argument(
        "--config_only",
        action="store_true",
        help="Only create and save the config"
    )
    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    if args.test_original_only:
        # Only test original model for now
        success = test_original_model_only(args.output_dir)
        if success:
            print("\n✅ Original TimesFM 2.5 model test completed successfully!")
        else:
            print("\n❌ Original TimesFM 2.5 model test failed!")
    elif args.config_only:
        # Only create config
        config = create_config_only(args.output_dir)
        print("\n✅ Config creation completed!")
    else:
        print("⚠️ Full conversion not yet implemented")
        print("Use --test_original_only to test original model")
        print("Use --config_only to create config")


if __name__ == "__main__":
    main()