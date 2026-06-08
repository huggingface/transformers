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

from ...configuration_utils import PretrainedConfig
from ...image_utils import OPENAI_CLIP_MEAN, OPENAI_CLIP_STD


__all__ = ["RADIOConfig"]


class RADIOConfig(PretrainedConfig):
    """Configuration for the RADIO vision encoder (native transformers port).

    The defaults correspond to ``nvidia/C-RADIOv4-H`` (a ViT-H/16 backbone with a
    Cropped Position Embedding (CPE) patch generator).
    """

    model_type = "radio"

    def __init__(
        self,
        hidden_size: int = 1280,
        num_hidden_layers: int = 32,
        num_attention_heads: int = 16,
        mlp_ratio: float = 4.0,
        hidden_act: str = "gelu",
        layer_norm_eps: float = 1e-6,
        attention_probs_dropout_prob: float = 0.0,
        hidden_dropout_prob: float = 0.0,
        drop_path_rate: float = 0.0,
        use_swiglu_ffn: bool = False,
        qkv_bias: bool = True,
        # C-RADIO has no layerscale; 1.0 makes the (inherited) layerscale an identity op.
        layerscale_value: float = 1.0,
        # patch / image
        num_channels: int = 3,
        patch_size: int = 16,
        image_size: int = 224,
        # CPE patch generator
        max_img_size: int = 2048,
        num_cls_tokens: int = 3,
        num_registers: int = 7,
        summary_idxs: list[int] | None = None,
        # input conditioner
        norm_mean: tuple[float, float, float] = OPENAI_CLIP_MEAN,
        norm_std: tuple[float, float, float] = OPENAI_CLIP_STD,
        # resolution metadata (inference convenience)
        max_resolution: int = 2048,
        preferred_resolution: tuple[int, int] = (512, 512),
        initializer_range: float = 0.02,
        **kwargs,
    ):
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.mlp_ratio = mlp_ratio
        self.hidden_act = hidden_act
        self.layer_norm_eps = layer_norm_eps
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.hidden_dropout_prob = hidden_dropout_prob
        self.drop_path_rate = drop_path_rate
        self.use_swiglu_ffn = use_swiglu_ffn
        self.qkv_bias = qkv_bias
        self.layerscale_value = layerscale_value

        self.num_channels = num_channels
        self.patch_size = patch_size
        self.image_size = image_size

        self.max_img_size = max_img_size
        self.num_cls_tokens = num_cls_tokens
        self.num_registers = num_registers
        self.summary_idxs = summary_idxs if summary_idxs is not None else [0, 1]

        self.norm_mean = list(norm_mean)
        self.norm_std = list(norm_std)

        self.max_resolution = max_resolution
        self.preferred_resolution = preferred_resolution
        self.initializer_range = initializer_range

        super().__init__(**kwargs)

    @property
    def num_summary_tokens(self) -> int:
        """Number of skipped prefix tokens (cls + registers) before spatial features."""
        return self.num_cls_tokens + self.num_registers
