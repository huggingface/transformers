# Copyright 2026 H Company and the HuggingFace Inc. team. All rights reserved.
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
"""NeoMME model configuration."""

import math
from typing import Literal

from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring, logging
from ...utils.type_validators import positive_int_field


logger = logging.get_logger(__name__)

NEOMME_LAYER_TYPES = ("full_attention", "sliding_attention")


@auto_docstring(checkpoint="Hcompany/NeoMME-260M")
@strict
class NeoMMEConfig(PreTrainedConfig):
    r"""
    embedding_rank (`int`, *optional*, defaults to 256):
        Width of the factorized token embedding table before projection to `hidden_size`.
    layer_types (`list[str]`, *optional*):
        By default, every sixth layer and the final layer use full attention.
    rope_parameters (`dict`, *optional*):
        Rotary-position settings for `"full_attention"` and `"sliding_attention"` layers. The rotated dimensions,
        `head_dim * partial_rotary_factor`, must be a positive multiple of four.
    residual_multiplier (`float`, *optional*):
        Scale applied to attention and MLP residual branches. Defaults to `1 / sqrt(2 * num_hidden_layers)`.
    embedding_dim (`int`, *optional*, defaults to 128):
        Width of the token-level embeddings returned by [`NeoMMEForRetrieval`]. This setting is unrelated to
        `embedding_rank`.
    document_token_id (`int`, *optional*, defaults to 5):
        Token ID for the `<doc>` marker.
    tie_word_embeddings (`bool`, *optional*, defaults to `True`):
        Whether the masked token decoder reuses both factorized token embedding weights.

    ```python
    >>> from transformers import NeoMMEModel, NeoMMEConfig

    >>> configuration = NeoMMEConfig()
    >>> model = NeoMMEModel(configuration)
    ```
    """

    model_type = "neomme"
    base_model_tp_plan = {
        "layers.*.self_attn.q_proj": "colwise",
        "layers.*.self_attn.k_proj": "colwise",
        "layers.*.self_attn.v_proj": "colwise",
        "layers.*.self_attn.output_projection.gate_proj": "colwise",
        "layers.*.self_attn.output_projection.o_proj": "rowwise",
        "layers.*.mlp.up_proj": "colwise",
        "layers.*.mlp.down_proj": "rowwise",
    }
    # Per-layer-type RoPE: sliding layers rotate every head dim at short range; global layers rotate 25%
    # at long range and leave the rest of each head unrotated for content matching.
    default_theta = {"full_attention": 1_000_000.0, "sliding_attention": 10_000.0}
    default_partial_rotary_factor = {"full_attention": 0.25, "sliding_attention": 1.0}
    default_long_sliding_window = 1024

    vocab_size: int = positive_int_field(default=131072)
    embedding_rank: int = positive_int_field(default=256)
    hidden_size: int = positive_int_field(default=1024)
    intermediate_size: int = positive_int_field(default=3584)
    hidden_act: Literal["relu2"] = "relu2"
    mlp_bias: bool = False
    num_hidden_layers: int = positive_int_field(default=17)
    num_attention_heads: int = positive_int_field(default=16)
    num_key_value_heads: int = positive_int_field(default=4)
    head_dim: int = positive_int_field(default=64)
    max_position_embeddings: int = positive_int_field(default=16384)
    norm_eps: float = 1e-6
    initializer_range: float = 0.02
    attention_dropout: float | int = 0.0
    attention_bias: bool = False
    layer_types: list[str] | None = None
    rope_parameters: dict[Literal["full_attention", "sliding_attention"], dict] | None = None
    sliding_window: int | None = 256
    residual_multiplier: float | None = None
    patch_size: int = positive_int_field(default=32)
    embedding_dim: int = positive_int_field(default=128)
    pad_token_id: int | None = 0
    document_token_id: int | None = 5
    image_token_id: int | None = 6
    tie_word_embeddings: bool = True

    def __post_init__(self, **kwargs):
        if self.layer_types is None:
            self.layer_types = [
                "full_attention" if (i + 1) % 6 == 0 or i == self.num_hidden_layers - 1 else "sliding_attention"
                for i in range(self.num_hidden_layers)
            ]
        self.validate_layer_type()

        if "per_layer_config" not in kwargs:
            sliding_idx = 0
            kwargs["per_layer_config"] = {}
            for layer_idx, layer_type in enumerate(self.layer_types):
                if layer_type == "full_attention":
                    kwargs["per_layer_config"][layer_idx] = {"sliding_window": None}
                elif layer_type == "sliding_attention":
                    # Alternate short and long windows by sliding-layer index.
                    if sliding_idx % 2:
                        kwargs["per_layer_config"][layer_idx] = {"sliding_window": self.default_long_sliding_window}
                    sliding_idx += 1
        if self.residual_multiplier is None:
            self.residual_multiplier = (2 * self.num_hidden_layers) ** -0.5

        super().__post_init__(**kwargs)

    def validate_architecture(self):
        """Part of `@strict`-powered validation. Validates the architecture of the config."""
        if self.num_attention_heads % self.num_key_value_heads:
            raise ValueError("num_key_value_heads must divide num_attention_heads")

        for layer_idx, layer_config in enumerate(self.per_layer_config):
            sliding_window = layer_config.sliding_window
            if sliding_window is not None and (not isinstance(sliding_window, int) or sliding_window <= 0):
                raise ValueError(
                    f"sliding_window for layer={layer_idx} must be a positive integer or None, got {sliding_window!r}."
                )

        if not math.isfinite(self.residual_multiplier) or self.residual_multiplier <= 0:
            raise ValueError("residual_multiplier must be finite and positive")

    def convert_rope_params_to_dict(self, **kwargs):
        rope_theta = kwargs.pop("rope_theta", None)
        self.rope_parameters = self.rope_parameters if self.rope_parameters is not None else {}

        for layer_type in set(self.layer_types):
            layer_params = self.rope_parameters.setdefault(layer_type, {})
            layer_params.setdefault("rope_type", "default")
            layer_params.setdefault(
                "rope_theta", rope_theta if rope_theta is not None else self.default_theta.get(layer_type)
            )
            layer_params.setdefault("partial_rotary_factor", self.default_partial_rotary_factor.get(layer_type))

        self.standardize_rope_params()
        self._validate_rotary_dims()
        return kwargs

    def validate_layer_type(self) -> None:
        if len(self.layer_types) != self.num_hidden_layers:
            raise ValueError(
                f"`num_hidden_layers` ({self.num_hidden_layers}) must be equal to the number of "
                f"`layer_types` ({len(self.layer_types)})"
            )
        invalid_layer_types = sorted(set(self.layer_types).difference(NEOMME_LAYER_TYPES))
        if invalid_layer_types:
            raise ValueError(
                f"`layer_types` entries must be one of {NEOMME_LAYER_TYPES} for NeoMME, got {invalid_layer_types}."
            )
        if "full_attention" not in self.layer_types:
            raise ValueError("layer_types must contain at least one full_attention layer.")

    def _validate_rotary_dims(self) -> None:
        """Ensure each layer type rotates a multiple of 4 head dimensions."""
        for layer_type in sorted(set(self.layer_types)):
            rope_theta = self.rope_parameters[layer_type]["rope_theta"]
            if (
                not isinstance(rope_theta, (int, float))
                or isinstance(rope_theta, bool)
                or not math.isfinite(rope_theta)
                or rope_theta <= 0
            ):
                raise ValueError(f"rope_parameters[{layer_type!r}]['rope_theta'] must be finite and positive")

            partial_rotary_factor = self.rope_parameters[layer_type].get("partial_rotary_factor", 1.0)
            rotary_dim = int(self.head_dim * partial_rotary_factor)
            if not 0.0 < partial_rotary_factor <= 1.0:
                raise ValueError(
                    "`partial_rotary_factor` must be in (0.0, 1.0] but got "
                    f"rope_parameters[{layer_type!r}]['partial_rotary_factor']={partial_rotary_factor}"
                )
            if rotary_dim < 4 or rotary_dim % 4:
                raise ValueError(
                    f"rope_parameters[{layer_type!r}]['partial_rotary_factor']={partial_rotary_factor} rotates "
                    f"{rotary_dim} of head_dim={self.head_dim} dims, which is not a multiple of 4: the two "
                    f"M-RoPE axes consume frequencies in alternating pairs. Nearest usable factors: "
                    f"{(rotary_dim - rotary_dim % 4) / self.head_dim} or "
                    f"{(rotary_dim + 4 - rotary_dim % 4) / self.head_dim}."
                )

    @property
    def patch_dim(self) -> int:
        """Width of one flattened image patch, `3 * patch_size ** 2`."""
        return 3 * self.patch_size**2


__all__ = ["NeoMMEConfig"]
