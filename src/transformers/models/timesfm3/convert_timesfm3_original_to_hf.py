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
Convert TimesFM 3 checkpoints from the original repository (https://github.com/google-research/timesfm).

Example:

```bash
python src/transformers/models/timesfm3/convert_timesfm3_original_to_hf.py \
    --repo_id google/timesfm-3.0-pytorch \
    --pytorch_dump_folder_path ./timesfm-3.0 \
    --verify
```

`--verify` compares the converted model with the original implementation and needs the original package:
`pip install "timesfm[torch] @ git+https://github.com/google-research/timesfm"`.
"""

import argparse
import json
import os
import re

import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

from transformers import TimesFm3Config, TimesFm3ModelForPrediction


# Renames of the original keys, applied in order.
ORIGINAL_TO_CONVERTED_KEY_MAPPING = {
    r"^pre_transformer_resblock\.hidden_layer\.": r"model.input_ff_layer.input_layer.",
    r"^pre_transformer_resblock\.": r"model.input_ff_layer.",
    r"^transformer_stack\.layers\.(\d+)\.pre_seq_attn_ln\.": r"model.layers.\1.input_layernorm.",
    r"^transformer_stack\.layers\.(\d+)\.post_seq_attn_ln\.": r"model.layers.\1.post_attention_layernorm.",
    r"^transformer_stack\.layers\.(\d+)\.pre_var_attn_ln\.": r"model.layers.\1.pre_variate_layernorm.",
    r"^transformer_stack\.layers\.(\d+)\.post_var_attn_ln\.": r"model.layers.\1.post_variate_layernorm.",
    r"^transformer_stack\.layers\.(\d+)\.pre_ff_ln\.": r"model.layers.\1.pre_feedforward_layernorm.",
    r"^transformer_stack\.layers\.(\d+)\.post_ff_ln\.": r"model.layers.\1.post_feedforward_layernorm.",
    r"^transformer_stack\.layers\.(\d+)\.ff0\.": r"model.layers.\1.mlp.fc1.",
    r"^transformer_stack\.layers\.(\d+)\.ff1\.": r"model.layers.\1.mlp.fc2.",
    r"^transformer_stack\.layers\.(\d+)\.seq_attn\.": r"model.layers.\1.self_attn.",
    r"^transformer_stack\.layers\.(\d+)\.var_attn\.": r"model.layers.\1.variate_attn.",
    r"^output_head\.": r"output_projection.",
}

# Renames inside an attention module, applied after the ones above.
ATTENTION_KEY_MAPPING = {
    r"\.query_proj\.": r".q_proj.",
    r"\.key_proj\.": r".k_proj.",
    r"\.value_proj\.": r".v_proj.",
    r"\.out_proj\.": r".o_proj.",
    r"\.query_ln\.": r".q_norm.",
    r"\.key_ln\.": r".k_norm.",
    r"\.per_dim_scale\.per_dim_scale$": r".scaling",
}

# Options of the original implementation that the port does not expose, with the only value it supports: the
# value of the released checkpoint.
SUPPORTED_MODEL_OPTIONS = {
    "input_transform": "identity",
    "use_frozen_running_stats": False,
    "use_iterative_cpm_revin": True,
    "use_linear_detrending": True,
    "use_stitching": True,
    "use_variate_attention": True,
}
SUPPORTED_RESIDUAL_BLOCK_OPTIONS = {"dropout": 0.0, "identity_skip": False, "prenorm": "none"}
SUPPORTED_TRANSFORMER_OPTIONS = {
    "attention_norm": "rms",
    "feedforward_norm": "rms",
    "qk_norm": "rms",
    "v_norm": "none",
    "causal_attention": True,
    "debug_no_masking": False,
    "paired_token_skip_second": False,
    "use_memory_efficient_attention": True,
    "use_rope_seq": True,
    "use_rope_var": False,
}
# Options without effect on the forecasts.
IGNORED_TRANSFORMER_OPTIONS = ("deterministic", "training", "use_sdpa", "max_variates")


def _check_supported_options(options: dict, supported_options: dict, name: str):
    for key, supported_value in supported_options.items():
        if key in options and (value := options.pop(key)) != supported_value:
            raise ValueError(f"`{key}={value!r}` in `{name}` is not supported, only `{supported_value!r}` is.")


def convert_config(original_config: dict) -> TimesFm3Config:
    """Builds a [`TimesFm3Config`] from the `config.json` of an original checkpoint."""
    original_config = dict(original_config)
    residual_block = dict(original_config.pop("residual_block_config"))
    transformer_stack = dict(original_config.pop("transformer_config"))
    transformer = dict(transformer_stack.pop("transformer"))

    _check_supported_options(original_config, SUPPORTED_MODEL_OPTIONS, "config")
    _check_supported_options(residual_block, SUPPORTED_RESIDUAL_BLOCK_OPTIONS, "residual_block_config")
    _check_supported_options(transformer, SUPPORTED_TRANSFORMER_OPTIONS, "transformer_config.transformer")
    for key in IGNORED_TRANSFORMER_OPTIONS:
        transformer.pop(key, None)
    # Gradient checkpointing of the Flax implementation.
    transformer_stack.pop("use_remat", None)

    hidden_size = transformer.pop("model_dims")
    num_attention_heads = transformer.pop("num_heads")
    use_bias = transformer.pop("use_bias")
    activation = transformer.pop("ff_activation")
    shared_options = {
        "hidden_dims": hidden_size,
        "output_dims": hidden_size,
        "use_bias": use_bias,
        "activation": activation,
    }
    for key, value in shared_options.items():
        if (block_value := residual_block.pop(key)) != value:
            raise ValueError(
                f"`{key}={block_value!r}` in `residual_block_config` is not supported with a transformer that "
                f"uses `{value!r}`."
            )

    quantiles = original_config.pop("quantiles")
    config = TimesFm3Config(
        patch_length=original_config.pop("input_patch_len"),
        horizon_length=original_config.pop("output_patch_len"),
        quantiles=quantiles,
        decode_index=len(quantiles) // 2,
        num_hidden_layers=transformer_stack.pop("num_layers"),
        hidden_size=hidden_size,
        intermediate_size=transformer.pop("hidden_dims"),
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_attention_heads,
        head_dim=hidden_size // num_attention_heads,
        use_bias=use_bias,
        attention_bias=use_bias,
        activation=activation,
        value_clip=original_config.pop("value_clip"),
        linear_detrending_threshold=original_config.pop("linear_detrending_threshold"),
    )
    unknown_keys = sorted({**original_config, **residual_block, **transformer_stack, **transformer})
    if unknown_keys:
        raise ValueError(f"Unknown keys in the original config: {unknown_keys}.")
    return config


def convert_state_dict(original_state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Renames the original parameters."""
    state_dict = {}
    for key, value in original_state_dict.items():
        new_key = key
        for pattern, replacement in ORIGINAL_TO_CONVERTED_KEY_MAPPING.items():
            new_key, num_replacements = re.subn(pattern, replacement, new_key)
            if num_replacements:
                break
        else:
            raise ValueError(f"Unexpected key in the original checkpoint: {key}.")
        for pattern, replacement in ATTENTION_KEY_MAPPING.items():
            new_key = re.sub(pattern, replacement, new_key)
        state_dict[new_key] = value
    return state_dict


def convert_checkpoint(original_config: dict, original_state_dict: dict[str, torch.Tensor]):
    config = convert_config(original_config)
    model = TimesFm3ModelForPrediction(config)
    model.load_state_dict(convert_state_dict(original_state_dict), strict=True)
    return model.eval()


@torch.no_grad()
def verify_conversion(model: TimesFm3ModelForPrediction, checkpoint: str, tolerance: float = 1e-4):
    """
    Compares the forecasts of the converted model with those of the original implementation, with and without
    covariates and averaging over the negated inputs.

    The difference is measured relative to the largest forecast. With random weights at the size of the released
    checkpoint, the two attention implementations of the original code (`use_sdpa=True` and `use_sdpa=False`)
    differ from each other by about 1e-5 on these inputs, and so does the converted model from either.
    """
    try:
        from timesfm3 import TimesFM3Forecaster
    except ImportError as error:
        raise ImportError(
            "Verifying the conversion needs the original implementation: "
            '`pip install "timesfm[torch] @ git+https://github.com/google-research/timesfm"`.'
        ) from error

    forecaster = TimesFM3Forecaster.from_pretrained(checkpoint, device="cpu")

    generator = torch.Generator().manual_seed(0)
    patch_length = model.config.patch_length
    context_length = 8 * patch_length + 5
    prediction_length = 4 * patch_length + 3
    steps = torch.arange(context_length, dtype=torch.float32)
    targets = 10 * torch.sin(2 * torch.pi * steps / 24) + torch.randn(2, 2, context_length, generator=generator)
    targets[1] += 0.05 * steps
    targets[0, :, :3] = float("nan")
    targets[0, 0, 15:19] = float("nan")
    past_covariates = torch.randn(2, 1, context_length, generator=generator)
    known_covariates = torch.randn(2, 1, context_length + prediction_length, generator=generator)

    for with_covariates in (False, True):
        for flip_invariance in (False, True):
            covariates = {"past_covariates": past_covariates, "known_covariates": known_covariates}
            original_covariates = {
                "past_only_covariates": list(past_covariates.numpy()),
                "past_future_covariates": list(known_covariates.numpy()),
            }
            expected = forecaster.predict_batch(
                list(targets.numpy()),
                horizon=prediction_length,
                return_quantiles=True,
                use_symmetric_averaging=flip_invariance,
                make_positive=True,
                **(original_covariates if with_covariates else {}),
            )
            expected = torch.stack([torch.from_numpy(output.quantiles) for output in expected])
            predictions = model(
                past_values=targets,
                prediction_length=prediction_length,
                force_flip_invariance=flip_invariance,
                truncate_negative=True,
                **(covariates if with_covariates else {}),
            ).full_predictions

            difference = (predictions - expected).abs().max().item() / expected.abs().max().item()
            print(
                f"covariates={with_covariates}, flip invariance={flip_invariance}: "
                f"largest difference {difference:.1e} of the largest forecast"
            )
            if not difference < tolerance:
                raise ValueError(f"The converted model differs from the original one by {difference:.1e}.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo_id", default="google/timesfm-3.0-pytorch", help="Hub repository or local folder of the checkpoint."
    )
    parser.add_argument("--pytorch_dump_folder_path", required=True, help="Where to save the converted model.")
    parser.add_argument("--verify", action="store_true", help="Compare with the original implementation.")
    parser.add_argument("--push_to_hub", default=None, help="Hub repository to push the converted model to.")
    args = parser.parse_args()

    if os.path.isdir(args.repo_id):
        config_path = os.path.join(args.repo_id, "config.json")
        weights_path = os.path.join(args.repo_id, "model.safetensors")
    else:
        config_path = hf_hub_download(args.repo_id, "config.json")
        weights_path = hf_hub_download(args.repo_id, "model.safetensors")
    with open(config_path, encoding="utf-8") as config_file:
        original_config = json.load(config_file)

    model = convert_checkpoint(original_config, load_file(weights_path))
    if args.verify:
        verify_conversion(model, args.repo_id)

    model.save_pretrained(args.pytorch_dump_folder_path)
    print(f"Saved the converted model to {args.pytorch_dump_folder_path}")
    if args.push_to_hub:
        model.push_to_hub(args.push_to_hub)


if __name__ == "__main__":
    main()
