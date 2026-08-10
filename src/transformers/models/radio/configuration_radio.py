# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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
"""RADIO vision encoder configuration."""

from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...image_utils import OPENAI_CLIP_MEAN, OPENAI_CLIP_STD
from ...utils import auto_docstring


__all__ = ["RadioConfig"]


@auto_docstring(checkpoint="nvidia/C-RADIOv4-H")
@strict
class RadioConfig(PreTrainedConfig):
    r"""
    mlp_ratio (`float`, *optional*, defaults to 4.0):
        Ratio of the hidden size of the MLP relative to `hidden_size`.
    use_swiglu_ffn (`bool`, *optional*, defaults to `False`):
        Whether to use a SwiGLU feed-forward network in the encoder layers instead of the standard MLP.
    layerscale_value (`float`, *optional*, defaults to 1.0):
        Initial value for the LayerScale parameters. C-RADIO has no LayerScale; the default of `1.0` makes the
        (inherited) LayerScale an identity operation.
    max_img_size (`int`, *optional*, defaults to 2048):
        Maximum supported image size (in pixels) used to size the position embedding table of the CPE patch generator.
    num_cls_tokens (`int`, *optional*, defaults to 3):
        Number of learned class (summary) tokens prepended to the patch sequence.
    num_registers (`int`, *optional*, defaults to 7):
        Number of learned register tokens prepended to the patch sequence.
    summary_idxs (`list[int]`, *optional*, defaults to `[0, 1]`):
        Indices of the class tokens to gather and flatten into the `summary` output embedding.
    norm_mean (`tuple[float, float, float]`, *optional*, defaults to `OPENAI_CLIP_MEAN`):
        Per-channel mean used by the input conditioner to normalize pixel values.
    norm_std (`tuple[float, float, float]`, *optional*, defaults to `OPENAI_CLIP_STD`):
        Per-channel standard deviation used by the input conditioner to normalize pixel values.
    """

    model_type = "radio"

    hidden_size: int = 1280
    num_hidden_layers: int = 32
    num_attention_heads: int = 16
    mlp_ratio: float = 4.0
    hidden_act: str = "gelu"
    layer_norm_eps: float = 1e-6
    attention_probs_dropout_prob: float = 0.0
    hidden_dropout_prob: float = 0.0
    drop_path_rate: float = 0.0
    use_swiglu_ffn: bool = False
    qkv_bias: bool = True
    # C-RADIO has no layerscale; 1.0 makes the (inherited) layerscale an identity op.
    layerscale_value: float = 1.0
    # patch / image
    num_channels: int = 3
    patch_size: int = 16
    image_size: int = 224
    # CPE patch generator
    max_img_size: int = 2048
    num_cls_tokens: int = 3
    num_registers: int = 7
    summary_idxs: list[int] | None = None
    # input conditioner
    norm_mean: list[float] | tuple[float, float, float] = tuple(OPENAI_CLIP_MEAN)
    norm_std: list[float] | tuple[float, float, float] = tuple(OPENAI_CLIP_STD)
    initializer_range: float = 0.02

    def __post_init__(self, **kwargs):
        if self.summary_idxs is None:
            self.summary_idxs = [0, 1]
        self.norm_mean = list(self.norm_mean)
        self.norm_std = list(self.norm_std)
        super().__post_init__(**kwargs)

    @property
    def num_summary_tokens(self) -> int:
        """Number of skipped prefix tokens (cls + registers) before spatial features."""
        return self.num_cls_tokens + self.num_registers
