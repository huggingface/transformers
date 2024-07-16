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

import torch
from typing import Any, Optional, Tuple, Dict


class RopeModelMixin:
    """
    Provides utilities for a model to set and retrieve RoPE embeddings.
    """

    def get_rope_embeddings(self, maximum_position_embeddings: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
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
            device=self.rope_layer.device
        )
        dummy_hidden_states = torch.zeros((1,), device=self.rope_layer.device, dtype=self.dtype)
        cos, sin = rope_layer(dummy_hidden_states, all_position_ids)
        return cos, sin

    def set_rope_embeddings(self, frequencies: torch.Tensor, scaling_factor: float):
        """
        Sets the RoPE embeddings, parameterized by the frequencies and scaling factor.

        Args:
            frequencies (`torch.Tensor`):
                The **inverse** frequencies of the RoPE embeddings.
            scaling_factor (`float`):
                A scaling factor to be applied to `position_ids` before computing the RoPE embeddings.
        """
        layers = self.layers if hasattr(self, "layers") else getattr(self, self.base_model_prefix).layers
        for layer in layers:
            layer.self_attn.rotary_emb.inv_freq = frequencies
            layer.self_attn.rotary_emb.scaling_factor = scaling_factor


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
    required_keys = ["base", "dim"]
    for key in required_keys:
        if key not in rope_config:
            raise ValueError(f"Missing required key '{key}' in RoPE config.")

    base = rope_config["base"]
    dim = rope_config["dim"]

    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.int64).float().to(device) / dim))
    return inv_freq


def _compute_dynamic_ntk_frequencies(rope_config: Dict[str, Any], device: torch.device) -> torch.Tensor:
    required_keys = ["base", "dim", "scaling_factor", "max_position_embeddings"]
    for key in required_keys:
        if key not in rope_config:
            raise ValueError(f"Missing required key '{key}' in RoPE config for RoPE type = 'dynamic'.")

    base = rope_config["base"]
    dim = rope_config["dim"]
    scaling_factor = rope_config["scaling_factor"]
    max_position_embeddings = rope_config["max_position_embeddings"]

    # default to max_position_embeddings, e.g. at init time
    seq_len = rope_config.get("seq_len") or max_position_embeddings

    base = base * ((scaling_factor * seq_len / max_position_embeddings) - (scaling_factor - 1)) ** (dim / (dim - 2))
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.int64).float().to(device) / dim))
    return inv_freq


def _compute_yarn_frequencies(rope_config: Dict[str, Any], device: torch.device) -> torch.Tensor:
    required_keys = ["base", "dim", "scaling_factor", "max_position_embeddings"]
    for key in required_keys:
        if key not in rope_config:
            raise ValueError(f"Missing required key '{key}' in RoPE config for RoPE type = 'dynamic'.")
