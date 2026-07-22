# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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
"""PyTorch LingBot-Vision model.

LingBot-Vision is a DINOv3-style ViT: axial RoPE over the patch grid, register (storage) tokens, LayerScale and
an optional SwiGLU feed-forward. Every block is therefore inherited from `dinov3_vit`; the deltas are the fused
QKV projection of the original checkpoints (split at load time by `conversion_mapping.py`), the `rope_parameters`
config format, and the flat mask token.
"""

import torch
from torch import nn

from ...activations import ACT2FN
from ...utils import auto_docstring, logging
from ..dinov2.modeling_dinov2 import Dinov2MLP
from ..dinov3_vit.modeling_dinov3_vit import (
    DINOv3ViTAttention,
    DINOv3ViTBackbone,
    DINOv3ViTBackboneOutput,
    Dinov3ViTDropPath,
    DINOv3ViTEmbeddings,
    DINOv3ViTEncoder,
    DINOv3ViTGatedMLP,
    DINOv3ViTLayer,
    DINOv3ViTLayerScale,
    DINOv3ViTModel,
    DINOv3ViTPreTrainedModel,
    DINOv3ViTRopePositionEmbedding,
)
from .configuration_lingbot_vision import LingbotVisionConfig


logger = logging.get_logger(__name__)


class LingbotVisionBackboneOutput(DINOv3ViTBackboneOutput):
    pass


class LingbotVisionEmbeddings(DINOv3ViTEmbeddings):
    def __init__(self, config: LingbotVisionConfig):
        super().__init__(config)
        # The original implementation stores the mask token flat; it broadcasts the same way over patch embeddings.
        self.mask_token = nn.Parameter(torch.zeros(1, config.hidden_size))


class LingbotVisionRopePositionEmbedding(DINOv3ViTRopePositionEmbedding):
    def __init__(self, config: LingbotVisionConfig):
        nn.Module.__init__(self)

        self.config = config
        self.base = config.rope_parameters["rope_theta"]
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.num_patches_h = config.image_size // config.patch_size
        self.num_patches_w = config.image_size // config.patch_size

        inv_freq = 1 / self.base ** torch.arange(0, 1, 4 / self.head_dim, dtype=torch.float32)  # (head_dim / 4,)
        self.register_buffer("inv_freq", inv_freq, persistent=False)


class LingbotVisionAttention(DINOv3ViTAttention):
    pass


class LingbotVisionLayerScale(DINOv3ViTLayerScale):
    pass


class LingbotVisionMLP(Dinov2MLP):
    # `fc1`/`fc2` are the names the original checkpoints use, so the plain MLP needs no renaming and the gated
    # variant (`w1`/`w2`/`w3` -> `gate_proj`/`up_proj`/`down_proj`) stays unambiguous in both directions.
    def __init__(self, config: LingbotVisionConfig):
        nn.Module.__init__(self)
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size, bias=config.mlp_bias)
        self.activation = ACT2FN[config.hidden_act]
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size, bias=config.mlp_bias)


class LingbotVisionGatedMLP(DINOv3ViTGatedMLP):
    pass


class LingbotVisionDropPath(Dinov3ViTDropPath):
    pass


class LingbotVisionLayer(DINOv3ViTLayer):
    pass


@auto_docstring
class LingbotVisionPreTrainedModel(DINOv3ViTPreTrainedModel):
    config: LingbotVisionConfig
    _no_split_modules = ["LingbotVisionLayer"]
    # The original checkpoints carry two constants that this implementation derives instead of storing: the RoPE
    # period table (recomputed from `rope_parameters` as `inv_freq`) and the bias mask of the fused QKV projection
    # (its only effect is an all-zero key bias, which the checkpoints already materialize).
    _keys_to_ignore_on_load_unexpected = [r"^rope_embed\.periods$", r"\.attn\.qkv\.bias_mask$"]


class LingbotVisionEncoder(DINOv3ViTEncoder):
    pass


@auto_docstring
class LingbotVisionModel(DINOv3ViTModel):
    pass


@auto_docstring(custom_intro="LingBot-Vision backbone, to be used with dense prediction frameworks.")
class LingbotVisionBackbone(DINOv3ViTBackbone):
    pass


__all__ = [
    "LingbotVisionBackbone",
    "LingbotVisionModel",
    "LingbotVisionPreTrainedModel",
]
