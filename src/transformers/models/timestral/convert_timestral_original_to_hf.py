# Copyright 2026 The HuggingFace Team. All rights reserved.
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
Convert Timestral checkpoints from the original repository (https://github.com/mistralai/timestral).

Example:

```bash
python src/transformers/models/timestral/convert_timestral_original_to_hf.py \
    --repo_id mistralai/Timestral-209M \
    --pytorch_dump_folder_path ./timestral-209m \
    --verify
```

`--verify` compares the converted model with the original implementation and needs the original package:
`pip install git+https://github.com/mistralai/timestral`.
"""

import argparse
import json
import re

import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

from transformers import TimestralConfig, TimestralModelForPrediction


# Renames of the original keys, applied in order. The fused attention and MLP projections are split separately.
ORIGINAL_TO_CONVERTED_KEY_MAPPING = {
    r"^representation\.patch_proj\.hidden_layer\.": r"model.input_ff_layer.input_layer.",
    r"^representation\.patch_proj\.": r"model.input_ff_layer.",
    r"^representation\.forecast_query$": r"model.forecast_query",
    r"^transformer\.layers\.(\d+)\.attn_norms\.(\d+)\.scale$": r"model.layers.\1.attention_layernorms.\2.weight",
    r"^transformer\.layers\.(\d+)\.attns\.(\d+)\.wO\.": r"model.layers.\1.attentions.\2.o_proj.",
    r"^transformer\.layers\.(\d+)\.attns\.(\d+)\._gate\.": r"model.layers.\1.attentions.\2.gate_proj.",
    r"^transformer\.layers\.(\d+)\.mlp_norm\.scale$": r"model.layers.\1.mlp_layernorm.weight",
    r"^transformer\.layers\.(\d+)\.mlp\.w3\.": r"model.layers.\1.mlp.down_proj.",
    r"^transformer\.out_norm\.scale$": r"model.norm.weight",
    r"^head\.projection\.hidden_layer\.": r"output_projection.input_layer.",
    r"^head\.projection\.": r"output_projection.",
}

FUSED_ATTENTION_PATTERN = re.compile(r"^transformer\.layers\.(\d+)\.attns\.(\d+)\.wQKV\.(weight|bias)$")
FUSED_MLP_PATTERN = re.compile(r"^transformer\.layers\.(\d+)\.mlp\.w12\.(weight|bias)$")


def convert_config(original_config: dict) -> TimestralConfig:
    """Builds a [`TimestralConfig`] from the `config.json` of an original checkpoint."""
    original_config = dict(original_config)
    if original_config.pop("stabilize_with_global", False):
        raise ValueError("Checkpoints with `stabilize_with_global=True` are not supported.")
    # Only changes how the original implementation evaluates the attention, not its result.
    original_config.pop("split_context_horizon_attn", None)

    gates = {}
    for axis in ("time", "variate"):
        gate = original_config.pop(f"{axis}_attn_gate", "sdpa_sigmoid")
        if gate not in ("none", "sdpa_sigmoid"):
            raise ValueError(f"Unknown attention gate {gate!r}.")
        gates[f"{axis}_attention_gate"] = gate == "sdpa_sigmoid"

    config = TimestralConfig(
        patch_length=original_config.pop("patch_size"),
        hidden_size=original_config.pop("d_model"),
        num_attention_heads=original_config.pop("num_heads"),
        intermediate_size=original_config.pop("mlp_hidden_dim"),
        block_types=original_config.pop("block_schedule"),
        quantiles=original_config.pop("quantile_knots"),
        block_size=original_config.pop("block_size"),
        context_length=original_config.pop("context_length"),
        rope_parameters={
            "rope_type": "default",
            "rope_theta": original_config.pop("rope_theta", 10000.0),
            "partial_rotary_factor": original_config.pop("rope_partial_factor", 0.5),
        },
        rms_norm_eps=original_config.pop("norm_eps", 1e-6),
        attention_bias=original_config.pop("attn_bias", False),
        mlp_bias=original_config.pop("mlp_bias", False),
        minimum_scale=original_config.pop("minimum_scale", 1e-6),
        use_asinh=original_config.pop("use_asinh", True),
        use_covariates=original_config.pop("covariate_enabled", True),
        attend_missing_patches=original_config.pop("attend_missing_patches", True),
        quantile_width_factor=original_config.pop("quantile_width_factor", 2.0),
        **gates,
    )
    if original_config:
        raise ValueError(f"Unknown keys in the original config: {sorted(original_config)}.")
    return config


def convert_state_dict(original_state_dict: dict[str, torch.Tensor], config: TimestralConfig) -> dict:
    """Renames the original parameters and splits the fused attention and MLP projections."""
    hidden_size = config.hidden_size
    num_heads = config.num_attention_heads
    head_dim = hidden_size // num_heads

    state_dict = {}
    for key, value in original_state_dict.items():
        if match := FUSED_ATTENTION_PATTERN.match(key):
            layer_idx, attention_idx, name = match.groups()
            prefix = f"model.layers.{layer_idx}.attentions.{attention_idx}"
            # The fused projection is laid out as (query/key/value, head_dim, num_heads). Each projection is
            # reordered to the (num_heads, head_dim) layout.
            value = value.reshape(3, head_dim, num_heads, *value.shape[1:]).transpose(1, 2)
            value = value.reshape(3, hidden_size, *value.shape[3:])
            for projection, weights in zip(("q_proj", "k_proj", "v_proj"), value):
                state_dict[f"{prefix}.{projection}.{name}"] = weights.clone()
        elif match := FUSED_MLP_PATTERN.match(key):
            layer_idx, name = match.groups()
            prefix = f"model.layers.{layer_idx}.mlp"
            # The original MLP computes `first_half * silu(second_half)`.
            first_half, second_half = value.chunk(2, dim=0)
            state_dict[f"{prefix}.up_proj.{name}"] = first_half.clone()
            state_dict[f"{prefix}.gate_proj.{name}"] = second_half.clone()
        else:
            new_key = key
            for pattern, replacement in ORIGINAL_TO_CONVERTED_KEY_MAPPING.items():
                new_key, num_replacements = re.subn(pattern, replacement, new_key)
                if num_replacements:
                    break
            else:
                raise ValueError(f"Unexpected key in the original checkpoint: {key}.")
            state_dict[new_key] = value
    return state_dict


def convert_checkpoint(original_config: dict, original_state_dict: dict[str, torch.Tensor]):
    config = convert_config(original_config)
    model = TimestralModelForPrediction(config)
    model.load_state_dict(convert_state_dict(original_state_dict, config), strict=True)
    return model.eval()


@torch.no_grad()
def verify_conversion(model: TimestralModelForPrediction, original_config: dict, original_state_dict: dict):
    """
    Compares the converted model with the original implementation on a few inputs.

    For these inputs the forecasts reach the thousands, after a `sinh`. With float32 weights, two implementations
    of the same computation then differ by rounding errors of about 1e-5 of the largest value. The comparison is
    therefore run twice: with float32 weights at a loose tolerance, and with float64 weights, where the two
    implementations have to agree closely.
    """
    try:
        from timestral import TimestralConfig as OriginalConfig
        from timestral import TimestralForecaster
        from timestral import TimestralModel as OriginalModel
    except ImportError as error:
        raise ImportError(
            "Verifying the conversion needs the original implementation: "
            "`pip install git+https://github.com/mistralai/timestral`."
        ) from error

    generator = torch.Generator().manual_seed(0)
    patch_length = model.config.patch_length
    context_length = 8 * patch_length + 5
    prediction_length = (model.config.block_size + 1) * patch_length + 3
    steps = torch.arange(context_length, dtype=torch.float32)
    targets = 10 * torch.sin(2 * torch.pi * steps / 24) + torch.randn(2, 2, context_length, generator=generator)
    targets[0, 0, 5:9] = float("nan")
    past_covariates = torch.randn(2, 1, context_length, generator=generator)
    known_covariates = torch.randn(2, 1, context_length + prediction_length, generator=generator)
    cases = {
        "targets only": {},
        "with covariates": {"past_covariates": past_covariates, "known_covariates": known_covariates},
    }

    for dtype, rtol, atol in ((torch.float32, 1e-3, 1e-3), (torch.float64, 1e-5, 1e-4)):
        original_model = OriginalModel(OriginalConfig.from_dict(original_config))
        original_model.load_state_dict(original_state_dict, strict=True)
        forecaster = TimestralForecaster(original_model.eval().to(dtype))
        model.to(dtype)
        for name, covariates in cases.items():
            expected = forecaster.forecast(targets, prediction_length=prediction_length, **covariates).quantiles
            outputs = model(past_values=targets, prediction_length=prediction_length, **covariates)
            predictions, expected = outputs.full_predictions.double(), expected.double()
            max_difference = (predictions - expected).abs().max().item()
            print(f"{name}, {str(dtype).split('.')[-1]} weights: max absolute difference {max_difference:.3e}")
            torch.testing.assert_close(predictions, expected, rtol=rtol, atol=atol)
    model.to(torch.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo_id", default="mistralai/Timestral-209M", help="Hub repository of the checkpoint.")
    parser.add_argument("--pytorch_dump_folder_path", required=True, help="Where to save the converted model.")
    parser.add_argument("--verify", action="store_true", help="Compare with the original implementation.")
    parser.add_argument("--push_to_hub", default=None, help="Hub repository to push the converted model to.")
    args = parser.parse_args()

    with open(hf_hub_download(args.repo_id, "config.json"), encoding="utf-8") as config_file:
        original_config = json.load(config_file)
    original_state_dict = load_file(hf_hub_download(args.repo_id, "model.safetensors"))

    model = convert_checkpoint(original_config, original_state_dict)
    if args.verify:
        verify_conversion(model, original_config, original_state_dict)

    model.save_pretrained(args.pytorch_dump_folder_path)
    print(f"Saved the converted model to {args.pytorch_dump_folder_path}")
    if args.push_to_hub:
        model.push_to_hub(args.push_to_hub)


if __name__ == "__main__":
    main()
