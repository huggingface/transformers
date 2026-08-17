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

from typing import Literal

from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring, logging


logger = logging.get_logger(__name__)


@auto_docstring(checkpoint="Hcompany/NeoMME-260M")
@strict
class NeoMMEConfig(PreTrainedConfig):
    r"""
    embedding_rank (`int`, *optional*, defaults to 256):
        Width of the factorized token embedding table before projection to `hidden_size`.
    layer_types (`list[str]`, *optional*):
        Attention type for each layer. Use `"full_attention"` or `"sliding_attention"`. When set, this list must
        contain one value per layer. By default, every sixth layer and the final layer use full attention.
    rope_parameters (`dict`, *optional*):
        Rotary-position settings for `"full_attention"` and `"sliding_attention"` layers. The rotated dimensions,
        `head_dim * partial_rotary_factor`, must be a positive multiple of four.
    sliding_window_short (`int`, *optional*, defaults to 256):
        Number of tokens on either side that a short sliding-attention layer can attend to.
    sliding_window_long (`int`, *optional*, defaults to 1024):
        Number of tokens on either side that a long sliding-attention layer can attend to. Short and long windows
        alternate between full-attention layers.
    use_value_embeds (`bool`, *optional*, defaults to `True`):
        Whether to add learned token value embeddings in the first and last full-attention layers.
    residual_scale (`float`, *optional*):
        Scale applied to attention and MLP residual branches. Defaults to `1 / sqrt(2 * num_hidden_layers)`.
    embedding_dim (`int`, *optional*, defaults to 128):
        Width of the token-level embeddings returned by [`NeoMMEForRetrieval`]. This setting is unrelated to
        `embedding_rank`.
    document_token_id (`int`, *optional*, defaults to 5):
        Token ID for the `<doc>` marker.

    ```python
    >>> from transformers import NeoMMEModel, NeoMMEConfig

    >>> configuration = NeoMMEConfig()
    >>> model = NeoMMEModel(configuration)
    ```
    """

    model_type = "neomme"
    # Per-layer-type RoPE: sliding layers rotate every head dim at short range; global layers rotate 25%
    # at long range and leave the rest of each head unrotated for content matching.
    default_theta = {"full_attention": 1_000_000.0, "sliding_attention": 10_000.0}
    default_partial_rotary_factor = {"full_attention": 0.25, "sliding_attention": 1.0}

    vocab_size: int = 131072
    embedding_rank: int = 256
    hidden_size: int = 1024
    intermediate_size: int = 3584
    num_hidden_layers: int = 17
    num_attention_heads: int = 16
    num_key_value_heads: int = 4
    head_dim: int = 64
    max_position_embeddings: int = 16384
    norm_eps: float = 1e-6
    initializer_range: float = 0.02
    attention_dropout: float | int = 0.0

    layer_types: list[str] | None = None
    rope_parameters: dict[Literal["full_attention", "sliding_attention"], dict] | None = None
    sliding_window_short: int = 256
    sliding_window_long: int = 1024

    use_value_embeds: bool = True
    residual_scale: float | None = None

    patch_size: int = 32
    embedding_dim: int = 128

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
        if self.residual_scale is None:
            self.residual_scale = (2 * self.num_hidden_layers) ** -0.5
        self.validate_layer_types()

        super().__post_init__(**kwargs)

    def validate_architecture(self):
        """Part of `@strict`-powered validation. Validates the architecture of the config."""
        if self.num_key_value_heads <= 0 or self.num_attention_heads % self.num_key_value_heads:
            raise ValueError("num_key_value_heads must divide num_attention_heads")
        if not 0 < self.sliding_window_short <= self.sliding_window_long:
            raise ValueError(
                f"expected 0 < sliding_window_short <= sliding_window_long, got {self.sliding_window_short} "
                f"and {self.sliding_window_long}. Pass two equal widths for a single band; the research "
                "encoding of `sliding_window_long = 0` for 'uniform' is resolved by the conversion script."
            )
        if self.residual_scale <= 0:
            raise ValueError("residual_scale must be positive")
        self._validate_rotary_dims()

    def convert_rope_params_to_dict(self, **kwargs):
        rope_scaling = kwargs.pop("rope_scaling", None)
        rope_theta = kwargs.pop("rope_theta", None)
        self.rope_parameters = self.rope_parameters if self.rope_parameters is not None else {}
        if rope_scaling is not None:
            rope_scaling = dict(rope_scaling)
            legacy_rope_type = rope_scaling.pop("type", None)
            if legacy_rope_type is not None:
                rope_scaling.setdefault("rope_type", legacy_rope_type)

        for layer_type in set(self.layer_types):
            layer_params = self.rope_parameters.setdefault(layer_type, {})
            if rope_scaling is not None:
                layer_params.update(rope_scaling)
            layer_params.setdefault("rope_type", "default")
            layer_params.setdefault("rope_theta", rope_theta or self.default_theta[layer_type])
            layer_params.setdefault("partial_rotary_factor", self.default_partial_rotary_factor[layer_type])

        self.standardize_rope_params()
        return kwargs

    def validate_layer_types(self) -> None:
        """Validate `layer_types`."""
        if len(self.layer_types) != self.num_hidden_layers:
            raise ValueError(
                f"layer_types has {len(self.layer_types)} entries but num_hidden_layers is "
                f"{self.num_hidden_layers}; there must be exactly one entry per layer."
            )
        unknown = sorted(set(self.layer_types) - {"full_attention", "sliding_attention"})
        if unknown:
            raise ValueError(f"layer_types contains unknown values {unknown}; expected full/sliding_attention.")

    def _validate_rotary_dims(self) -> None:
        """Ensure each layer type rotates a multiple of 4 head dimensions."""
        for layer_type in sorted(set(self.layer_types)):
            partial_rotary_factor = self.rope_parameters[layer_type].get("partial_rotary_factor", 1.0)
            rotary_dim = int(self.head_dim * partial_rotary_factor)
            if not 0.0 < partial_rotary_factor <= 1.0:
                # Above 1.0 the rotary slice is wider than the head and the failure lands inside attention as
                # a shape error; at or below 0.0 there is no rotation at all and positions vanish silently.
                raise ValueError(
                    f"rope_parameters[{layer_type!r}]['partial_rotary_factor']={partial_rotary_factor} is "
                    "outside (0.0, 1.0]: it is the fraction of each head's dims that carries position."
                )
            if rotary_dim < 4:
                raise ValueError(
                    f"rope_parameters[{layer_type!r}]['partial_rotary_factor']={partial_rotary_factor} rotates "
                    f"{rotary_dim} of head_dim={self.head_dim} dims, but two-axis M-RoPE needs at least 4."
                )
            if rotary_dim % 4:
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

    @property
    def layer_window_sizes(self) -> list[int | None]:
        """Per-layer sliding-window half-width; `None` on global layers.

        Sliding layers alternate short/long by their ordinal among sliding layers, so the pattern does
        not shift when the global layers move.
        """
        windows: list[int | None] = []
        sliding_idx = 0
        for layer_type in self.layer_types:
            if layer_type == "full_attention":
                windows.append(None)
                continue
            windows.append(self.sliding_window_long if sliding_idx % 2 else self.sliding_window_short)
            sliding_idx += 1
        return windows


__all__ = ["NeoMMEConfig"]
