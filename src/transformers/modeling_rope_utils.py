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


ROPE_CONFIG_DOCSTRING = r"""
    rope_scaling (`Dict`, *optional*):
        Dictionary containing the scaling configuration for the RoPE embeddings. Currently supports three scaling
        strategies: linear, dynamic and yarn. Their scaling factor must be a float greater than 1. The expected format is
        `{"type": strategy name, "factor": scaling factor}`. When using this flag, don't update
        `max_position_embeddings` to the expected new maximum. See the following thread for more information on how
        these scaling strategies behave:
        https://www.reddit.com/r/LocalLLaMA/comments/14mrgpr/dynamically_scaled_rope_further_increases/. This is an
        experimental feature, subject to breaking API changes in future versions.
        For the `yarn` strategy, the dictionary may also contain the following fields:
            `original_max_position_embeddings` (`int`, *optional*):
                The original maximum sequence length. This is used to scale the RoPE embeddings.
            `attention_factor` (`float`, *optional*):
                The attention scaling factor. If unspecified, it defaults to `0.1 ln(s) + 1`, where `s` is the
                `original_max_position_embeddings/max_position_embeddings` ratio.
            `beta_fast` (`float`, *optional*):
                Parameter to set the boundary for extrapolation (only) in the linear ramp function.
            `beta_slow` (`float`, *optional*):
                Parameter to set the boundary for interpolation (only) in the linear ramp function.
"""


class RopeModelMixin:
    """
    Provides utilities for a model to set and retrieve RoPE embeddings.
    """

    def get_rope_embeddings(
        self, maximum_position_embeddings: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns the RoPE embeddings for the model, i.e. the cos and sin values for all positions up to
        `maximum_position_embeddings`.

        Args:
            maximum_position_embeddings (`int`, *optional*):
                The maximum number of positions to generate RoPE embeddings for. If not provided, defaults to the
                model's `config.max_position_embeddings`.

        Returns:
            Tuple of `torch.Tensor`: The RoPE embeddings for the model, i.e. the cos and sin values for all positions
            up to `maximum_position_embeddings`.
        """
        # Assumption: all layers hold the same RoPE embeddings
        layers = self.layers if hasattr(self, "layers") else getattr(self, self.base_model_prefix).layers
        rope_layer = layers[0].self_attn.rotary_emb
        all_position_ids = torch.arange(
            maximum_position_embeddings or self.config.max_position_embeddings,
            dtype=torch.long,
            device=self.rope_layer.device,
        )
        dummy_hidden_states = torch.zeros((1,), device=self.rope_layer.device, dtype=self.dtype)
        cos, sin = rope_layer(dummy_hidden_states, all_position_ids)
        return cos, sin

    def set_rope_embeddings(
        self, frequencies: torch.Tensor, scaling_factor: float = 1.0, attention_factor: Optional[float] = None
    ):
        """
        Sets the RoPE embeddings, parameterized by the frequencies and scaling factor.

        Args:
            frequencies (`torch.Tensor`):
                The **inverse** frequencies of the RoPE embeddings.
            scaling_factor (`float`, *optional*, defaults to 1.0):
                A scaling factor to be applied to `position_ids` before computing the RoPE embeddings.
            attention_factor (`float`, *optional*):
                A scaling factor to be applied to `cos` and `sin` after they are computed. Used in advaced RoPE types,
                like YaRN.
        """
        layers = self.layers if hasattr(self, "layers") else getattr(self, self.base_model_prefix).layers
        for layer in layers:
            layer.self_attn.rotary_emb.inv_freq = frequencies
            layer.self_attn.rotary_emb.rope_config["scaling_factor"] = scaling_factor
            layer.self_attn.rotary_emb.rope_config["attention_factor"] = attention_factor


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
            f"`rope_scaling`'s original_max_position_embeddings field must be an int, got {original_max_position_embeddings}"
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
            f"`rope_scaling`'s beta_fast field must be greater than beta_slow, got beta_fast={b_fast} and beta_slow={b_slow}"
        )


def compute_frequencies(rope_config: Dict[str, Any], device: torch.device) -> torch.Tensor:
    rope_type = rope_config.get("rope_type", "default")
    if rope_type == "default":
        return _compute_default_frequencies(rope_config, device)
    elif rope_type == "dynamic":
        return _compute_dynamic_ntk_frequencies(rope_config, device)
    elif rope_type == "yarn":
        return _compute_yarn_frequencies(rope_config, device)
    else:
        raise ValueError(
            f"Unrecognized RoPE type: {rope_type}. If you want to use custom RoPE frequencies, use "
            "`model.set_rope_embeddings()`"
        )


def _compute_default_frequencies(rope_config: Dict[str, Any], device: torch.device) -> torch.Tensor:
    # Mandatory config options
    required_keys = ["base", "dim"]
    for key in required_keys:
        if key not in rope_config:
            raise ValueError(f"Missing required key '{key}' in RoPE config.")

    base = rope_config["base"]
    dim = rope_config["dim"]

    # Compute the inverse frequencies
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.int64).float().to(device) / dim))
    return inv_freq


def _compute_dynamic_ntk_frequencies(rope_config: Dict[str, Any], device: torch.device) -> torch.Tensor:
    """Credits to the Reddit users /u/bloc97 and /u/emozilla"""
    # Mandatory config options
    required_keys = ["base", "dim", "scaling_factor", "max_position_embeddings"]
    for key in required_keys:
        if key not in rope_config:
            raise ValueError(f"Missing required key '{key}' in RoPE config for RoPE type = 'dynamic'.")

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
    # Mandatory config options
    required_keys = ["base", "dim", "scaling_factor", "max_position_embeddings"]
    for key in required_keys:
        if key not in rope_config:
            raise ValueError(f"Missing required key '{key}' in RoPE config for RoPE type = 'dynamic'.")

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
