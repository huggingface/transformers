# Copyright 2024 The HuggingFace Team. All rights reserved.
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

import math
from typing import Any, Dict, Set

import torch


ROPE_CONFIG_DOCSTRING = r"""
    rope_scaling (`Dict`, *optional*):
        Dictionary containing the scaling configuration for the RoPE embeddings. When using this flag, don't update
        `max_position_embeddings` to the expected new maximum. Expected contents:
            `type` (`str`):
                The scaling strategy to use. Can be one of ['linear', 'dynamic', 'yarn'].
            `factor` (`float`):
                The scaling factor to apply to the RoPE embeddings. Must be a float greater than 1.
            `attention_factor` (`float`, *optional*):
                Optional, only used with 'yarn'. The attention scaling factor. If unspecified, it defaults to
                `0.1 ln(factor) + 1`.
            `beta_fast` (`float`, *optional*):
                Optional, only used with 'yarn'. Parameter to set the boundary for extrapolation (only) in the linear
                ramp function. If unspecified, it defaults to 32.
            `beta_slow` (`float`, *optional*):
                Optional, only used with 'yarn'. Parameter to set the boundary for interpolation (only) in the linear
                ramp function. If unspecified, it defaults to 1.
"""


def rope_config_validation(rope_scaling):
    """
    Validate the `rope_scaling` configuration.
    """
    if rope_scaling is None:
        return

    if not isinstance(rope_scaling, dict) or len(rope_scaling) < 2:
        raise ValueError(
            "`rope_scaling` must be a dictionary with a minimum of two fields, `type` and `factor`, "
            f"got {rope_scaling}"
        )
    rope_scaling_type = rope_scaling.get("type", None)
    rope_scaling_factor = rope_scaling.get("factor", None)
    if rope_scaling_type is None or rope_scaling_type not in ["linear", "dynamic", "yarn"]:
        raise ValueError(
            f"`rope_scaling`'s type field must be one of ['linear', 'dynamic', 'yarn'], got {rope_scaling_type}"
        )
    if rope_scaling_factor is None or not isinstance(rope_scaling_factor, float) or rope_scaling_factor <= 1.0:
        raise ValueError(f"`rope_scaling`'s factor field must be a float > 1, got {rope_scaling_factor}")

    if rope_scaling_type != "yarn":
        return

    if not isinstance(rope_scaling, dict) or len(rope_scaling) > 6:
        raise ValueError(
            "`rope_scaling` with type "
            f"{rope_scaling_type}"
            " must be a dictionary with a maximum of six fields, `type`, `factor`,"
            "`original_max_position_embeddings`, `attention_factor`, `beta_fast`, `beta_slow`, "
            f"got {rope_scaling}"
        )
    original_max_position_embeddings = rope_scaling.get("original_max_position_embeddings", None)
    attention_factor = rope_scaling.get("attention_factor", None)
    beta_fast = rope_scaling.get("beta_fast", None)
    beta_slow = rope_scaling.get("beta_slow", None)

    if original_max_position_embeddings is not None and not isinstance(original_max_position_embeddings, int):
        raise ValueError(
            "`rope_scaling`'s original_max_position_embeddings field must be an int, got "
            f"{original_max_position_embeddings}"
        )
    if attention_factor is not None and not isinstance(attention_factor, float) or attention_factor < 0:
        raise ValueError(
            f"`rope_scaling`'s attention_factor field must be a float greater than 0, got {attention_factor}"
        )
    if beta_fast is not None and not isinstance(beta_fast, float):
        raise ValueError(f"`rope_scaling`'s beta_fast field must be a float, got {beta_fast}")
    if beta_slow is not None and not isinstance(beta_slow, float):
        raise ValueError(f"`rope_scaling`'s beta_slow field must be a float, got {beta_slow}")

    b_fast = beta_fast if beta_fast is not None else 32
    b_slow = beta_slow if beta_slow is not None else 1
    if b_fast < b_slow:
        raise ValueError(
            f"`rope_scaling`'s beta_fast field must be greater than beta_slow, got beta_fast={b_fast} and "
            f"beta_slow={b_slow}"
        )


def _check_rope_config_keys(rope_config: Dict[str, Any], required_keys: Set, permitted_keys: Set):
    """Check if the keys in the RoPE config are valid"""
    keys_in_rope_config = set(rope_config.keys())
    required_keys_not_in_config = required_keys - keys_in_rope_config
    if len(required_keys_not_in_config) > 0:
        raise ValueError(
            f"Missing required keys '{required_keys_not_in_config}' in the (internally prepared) RoPE config."
        )
    all_permitted_keys = permitted_keys + required_keys
    keys_not_permitted = keys_in_rope_config - all_permitted_keys
    if len(keys_not_permitted) > 0:
        raise ValueError(f"Unrecognized keys '{keys_not_permitted}' in the (internally prepared) RoPE config.")


def _compute_default_frequencies(rope_config: Dict[str, Any], device: torch.device) -> torch.Tensor:
    """Computes the inverse frequencies according to the original RoPE implementation"""
    required_keys = {"base", "dim"}
    permitted_keys = {"type", "max_position_embeddings"}
    _check_rope_config_keys(rope_config, required_keys, permitted_keys)

    base = rope_config["base"]
    dim = rope_config["dim"]

    # Compute the inverse frequencies
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.int64).float().to(device) / dim))
    return inv_freq


def _compute_dynamic_ntk_frequencies(rope_config: Dict[str, Any], device: torch.device) -> torch.Tensor:
    """Computes he inverse frequencies with NTK scaling. Credits to the Reddit users /u/bloc97 and /u/emozilla"""
    required_keys = {"base", "dim", "scaling_factor", "max_position_embeddings"}
    permitted_keys = {"type"}
    _check_rope_config_keys(rope_config, required_keys, permitted_keys)

    base = rope_config["base"]
    dim = rope_config["dim"]
    scaling_factor = rope_config["scaling_factor"]
    max_position_embeddings = rope_config["max_position_embeddings"]

    # Optional config options
    # seq_len: default to max_position_embeddings, e.g. at init time
    seq_len = rope_config.get("seq_len") or max_position_embeddings

    # Compute the inverse frequencies
    base = base * ((scaling_factor * seq_len / max_position_embeddings) - (scaling_factor - 1)) ** (dim / (dim - 2))
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.int64).float().to(device) / dim))
    return inv_freq


def _compute_yarn_frequencies(rope_config: Dict[str, Any], device: torch.device) -> torch.Tensor:
    """
    Computes he inverse frequencies with NTK scaling. Please refer to the
    [original paper](https://arxiv.org/abs/2309.00071)
    """
    required_keys = {"base", "dim", "scaling_factor", "max_position_embeddings"}
    permitted_keys = {"type", "beta_fast", "beta_slow", "attention_factor"}
    _check_rope_config_keys(rope_config, required_keys, permitted_keys)

    base = rope_config["base"]
    dim = rope_config["dim"]
    scaling_factor = rope_config["scaling_factor"]
    max_position_embeddings = rope_config["max_position_embeddings"]

    # Optional config options
    # beta_fast/beta_slow: as suggested in the paper, default to 32/1 (correspondingly)
    beta_fast = rope_config.get("beta_fast") or 32
    beta_slow = rope_config.get("beta_slow") or 1

    # Compute the inverse frequencies

    # Inverse dimension formula to find the dimension based on the number of rotations
    def find_correction_dim(num_rotations, dim, base, max_position_embeddings):
        return (dim * math.log(max_position_embeddings / (num_rotations * 2 * math.pi))) / (2 * math.log(base))

    # Find dimension range bounds based on rotations
    def find_correction_range(low_rot, high_rot, dim, base, max_position_embeddings):
        low = math.floor(find_correction_dim(low_rot, dim, base, max_position_embeddings))
        high = math.ceil(find_correction_dim(high_rot, dim, base, max_position_embeddings))
        return max(low, 0), min(high, dim - 1)

    def linear_ramp_mask(min, max, dim):
        if min == max:
            max += 0.001  # Prevent singularity

        linear_func = (torch.arange(dim, dtype=torch.float32) - min) / (max - min)
        ramp_func = torch.clamp(linear_func, 0, 1)
        return ramp_func

    pos_freqs = base ** (torch.arange(0, dim, 2).float().to(device) / dim)
    inv_freq_extrapolation = 1.0 / pos_freqs
    inv_freq_interpolation = 1.0 / (scaling_factor * pos_freqs)

    low, high = find_correction_range(beta_fast, beta_slow, dim, base, max_position_embeddings)

    # Get n-dimensional rotational scaling corrected for extrapolation
    inv_freq_mask = 1 - linear_ramp_mask(low, high, dim // 2).float().to(device)
    inv_freq = inv_freq_interpolation * (1 - inv_freq_mask) + inv_freq_extrapolation * inv_freq_mask

    return inv_freq


# This maps the "type" string in rope config to the corresponding config. Can be expanded externally to support
# new RoPE types
ROPE_TYPE_TO_FUNCTION = {
    "default": _compute_default_frequencies,
    "dynamic": _compute_dynamic_ntk_frequencies,
    "yarn": _compute_yarn_frequencies,
}


def compute_frequencies(rope_config: Dict[str, Any], device: torch.device) -> torch.Tensor:
    rope_type = rope_config.get("type", "default")
    rope_fn = ROPE_TYPE_TO_FUNCTION.get(rope_type)
    if rope_fn is None:
        raise ValueError(
            f"Unrecognized RoPE type: {rope_type}.\n\nIf you want to use custom RoPE frequencies, there are two "
            "options: 1: Compute RoPE (cos, sin) externally, passing it through `position_embeddings` to the model's "
            "forward method. 2: Update the inverse frequencies in RoPE, updating `ROPE_TYPE_TO_FUNCTION` with "
            "{'your_rope_type': Callable[rope_config, device] -> torch.Tensor}."
        )
    return rope_fn(rope_config, device)
