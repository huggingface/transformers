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
from typing import Any, Dict, Optional, Tuple

import torch

from .configuration_utils import PretrainedConfig


ROPE_CONFIG_DOCSTRING = r"""
    rope_scaling (`Dict`, *optional*):
        Dictionary containing the scaling configuration for the RoPE embeddings. When using this flag, don't update
        `max_position_embeddings` to the expected new maximum. Expected contents:
            `type` (`str`):
                The scaling strategy to use. Can be one of ['linear', 'dynamic', 'yarn', 'llama3'].
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


def _compute_default_rope_parameters(
    config: PretrainedConfig, device: torch.device, seq_len: Optional[int] = None
) -> Tuple[torch.Tensor, float]:
    """
    Computes the inverse frequencies according to the original RoPE implementation

    Args:
        config ([`~transformers.PretrainedConfig`]):
            The model configuration.
        device (`torch.device`):
            The device to use for initialization of the inverse frequencies.
        seq_len (`int`, *optional*):
            The current sequence length. Unused for this type of RoPE.

    Returns:
        Tuple of (`torch.Tensor`, `float`), containing the inverse frequencies for the RoPE embeddings and the
        post-processing scaling factor applied to the computed cos/sin (unused in this type of RoPE).
    """
    base = config.rope_theta
    if hasattr(config, "head_dim"):  # TODO (joao): BC -- remove `if` in v4.45, keep `else`
        dim = config.head_dim
    else:
        dim = config.hidden_size // config.num_attention_heads
    attention_factor = 1.0  # Unused in this type of RoPE

    # Compute the inverse frequencies
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.int64).float().to(device) / dim))
    return inv_freq, attention_factor


def _compute_linear_scaling_rope_parameters(
    config: PretrainedConfig, device: torch.device, seq_len: Optional[int] = None
) -> Tuple[torch.Tensor, float]:
    """
    Computes the inverse frequencies with linear scaling. Credits to the Reddit user /u/kaiokendev

    Args:
        config ([`~transformers.PretrainedConfig`]):
            The model configuration.
        device (`torch.device`):
            The device to use for initialization of the inverse frequencies.
        seq_len (`int`, *optional*):
            The current sequence length. Unused for this type of RoPE.

    Returns:
        Tuple of (`torch.Tensor`, `float`), containing the inverse frequencies for the RoPE embeddings and the
        post-processing scaling factor applied to the computed cos/sin (unused in this type of RoPE).
    """
    # Gets the default RoPE parameters
    inv_freq, attention_factor = _compute_default_rope_parameters(config, device, seq_len)

    # Then applies linear scaling to the frequencies.
    # NOTE: originally, scaling was applied to the position_ids. However, we get `embs = inv_freq @ position_ids`, so
    # applying scaling to the inverse frequencies is equivalent.
    scaling_factor = config.rope_scaling["factor"]
    inv_freq /= scaling_factor
    return inv_freq, attention_factor


def _compute_dynamic_ntk_parameters(
    config: PretrainedConfig, device: torch.device, seq_len: Optional[int] = None
) -> Tuple[torch.Tensor, float]:
    """
    Computes the inverse frequencies with NTK scaling. Credits to the Reddit users /u/bloc97 and /u/emozilla

    Args:
        config ([`~transformers.PretrainedConfig`]):
            The model configuration.
        device (`torch.device`):
            The device to use for initialization of the inverse frequencies.
        seq_len (`int`, *optional*):
            The current sequence length, used to update the dynamic RoPE at inference time.

    Returns:
        Tuple of (`torch.Tensor`, `float`), containing the inverse frequencies for the RoPE embeddings and the
        post-processing scaling factor applied to the computed cos/sin (unused in this type of RoPE).
    """
    base = config.rope_theta
    if hasattr(config, "head_dim"):  # TODO (joao): BC -- remove `if` in v4.45, keep `else`
        dim = config.head_dim
    else:
        dim = config.hidden_size // config.num_attention_heads
    scaling_factor = config.rope_scaling["factor"]
    max_position_embeddings = config.max_position_embeddings
    attention_factor = 1.0  # Unused in this type of RoPE

    # seq_len: default to max_position_embeddings, e.g. at init time
    seq_len = seq_len if seq_len is not None else max_position_embeddings

    # Compute the inverse frequencies
    base = base * ((scaling_factor * seq_len / max_position_embeddings) - (scaling_factor - 1)) ** (dim / (dim - 2))
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.int64).float().to(device) / dim))
    return inv_freq, attention_factor


def _compute_yarn_parameters(
    config: PretrainedConfig, device: torch.device, seq_len: Optional[int] = None
) -> Tuple[torch.Tensor, float]:
    """
    Computes the inverse frequencies with NTK scaling. Please refer to the
    [original paper](https://arxiv.org/abs/2309.00071)

    Args:
        config ([`~transformers.PretrainedConfig`]):
            The model configuration.
        device (`torch.device`):
            The device to use for initialization of the inverse frequencies.
        seq_len (`int`, *optional*):
            The current sequence length. Unused for this type of RoPE.

    Returns:
        Tuple of (`torch.Tensor`, `float`), containing the inverse frequencies for the RoPE embeddings and the
        post-processing scaling factor applied to the computed cos/sin.
    """
    base = config.rope_theta
    if hasattr(config, "head_dim"):  # TODO (joao): BC -- remove `if` in v4.45, keep `else`
        dim = config.head_dim
    else:
        dim = config.hidden_size // config.num_attention_heads
    scaling_factor = config.rope_scaling["factor"]
    max_position_embeddings = config.max_position_embeddings

    # Sets the attention factor as suggested in the paper
    attention_factor = config.rope_scaling.get("attention_factor")
    if attention_factor is None:
        attention_factor = 0.1 * math.log(scaling_factor) + 1.0

    # Optional config options
    # beta_fast/beta_slow: as suggested in the paper, default to 32/1 (correspondingly)
    beta_fast = config.rope_scaling.get("beta_fast") or 32
    beta_slow = config.rope_scaling.get("beta_slow") or 1

    # Compute the inverse frequencies
    def find_correction_dim(num_rotations, dim, base, max_position_embeddings):
        """Inverse dimension formula to find the dimension based on the number of rotations"""
        return (dim * math.log(max_position_embeddings / (num_rotations * 2 * math.pi))) / (2 * math.log(base))

    def find_correction_range(low_rot, high_rot, dim, base, max_position_embeddings):
        """Find dimension range bounds based on rotations"""
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

    return inv_freq, attention_factor


# This maps the "type" string field in rope config to the corresponding function to compute the RoPE parameters from
# the model config. You can append new {'type': callable} pairs to this dictionary to enable custom RoPE
# parameterizations, as long as the callable has the same signature.
ROPE_PARAMETER_FUNCTIONS = {
    "default": _compute_default_rope_parameters,
    "linear": _compute_linear_scaling_rope_parameters,
    "dynamic": _compute_dynamic_ntk_parameters,
    "yarn": _compute_yarn_parameters,
}


def rope_config_validation(rope_scaling: Optional[Dict[str, Any]]):
    """
    Validate the `rope_scaling` config argument.
    """
    if rope_scaling is None:
        return

    required_keys = {"type", "factor"}
    received_keys = set(rope_scaling.keys())

    missing_keys = required_keys - received_keys
    if missing_keys:
        raise ValueError(f"Missing required keys in `rope_scaling`: {missing_keys}")

    rope_type = rope_scaling["type"]
    possible_rope_types = set(ROPE_PARAMETER_FUNCTIONS.keys())
    if rope_type is None or rope_type not in possible_rope_types:
        raise ValueError(f"`rope_scaling`'s 'type' field must be one of {possible_rope_types}, got {rope_type}")

    scaling_factor = rope_scaling["factor"]
    if scaling_factor is None or not isinstance(scaling_factor, float) or scaling_factor < 1.0:
        raise ValueError(f"`rope_scaling`'s factor field must be a float >= 1, got {scaling_factor}")

    if rope_type in ("linear", "dynamic"):
        unused_keys = received_keys - received_keys
        if unused_keys:
            raise ValueError(f"Unrecognized keys in `rope_scaling` for 'type'='{rope_type}': {unused_keys}")
    elif rope_type in ("yarn"):
        optional_keys = {"attention_factor", "beta_fast", "beta_slow"}
        unused_keys = received_keys - required_keys - optional_keys
        if unused_keys:
            raise ValueError(f"Unrecognized keys in `rope_scaling` for 'type'='yarn': {unused_keys}")

        attention_factor = rope_scaling.get("attention_factor")
        if attention_factor is not None and not isinstance(attention_factor, float) or attention_factor < 0:
            raise ValueError(
                f"`rope_scaling`'s attention_factor field must be a float greater than 0, got {attention_factor}"
            )
        beta_fast = rope_scaling.get("beta_fast")
        if beta_fast is not None and not isinstance(beta_fast, float):
            raise ValueError(f"`rope_scaling`'s beta_fast field must be a float, got {beta_fast}")
        beta_slow = rope_scaling.get("beta_slow")
        if beta_slow is not None and not isinstance(beta_slow, float):
            raise ValueError(f"`rope_scaling`'s beta_slow field must be a float, got {beta_slow}")

        if (beta_fast or 32) < (beta_slow or 1):
            raise ValueError(
                f"`rope_scaling`'s beta_fast field must be greater than beta_slow, got beta_fast={beta_fast} "
                f"(defaults to 32 if None) and beta_slow={beta_slow} (defaults to 1 if None)"
            )
    # else: no validation, it is a registered custom RoPE type
