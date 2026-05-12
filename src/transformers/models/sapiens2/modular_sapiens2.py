# Copyright 2026 the HuggingFace Team. All rights reserved.
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

from huggingface_hub.dataclasses import strict

from ...image_processing_backends import TorchvisionBackend
from ...image_utils import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, PILImageResampling
from ...utils import auto_docstring
from ..dinov3_vit.configuration_dinov3_vit import DINOv3ViTConfig
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
    DINOv3ViTMLP,
    DINOv3ViTModel,
    DINOv3ViTPreTrainedModel,
    DINOv3ViTRopePositionEmbedding,
)

# TODO (guarin): Double check if we want this checkpoint as default. Motiviation is that
# it is the smallest checkpoint which supports all tasks.
@auto_docstring(checkpoint="facebook/sapiens2-pretrain-0.4b")
@strict
class Sapiens2Config(DINOv3ViTConfig):
    r"""
    use_qk_norm (`bool`, *optional*, defaults to `True`):
        Whether to apply RMSNorm to queries and keys before RoPE in attention layers.
    num_key_value_heads (`int`, *optional*):
        Number of key/value heads for GQA layers. Defaults to `num_attention_heads // 2`.
        Set to `None` to disable GQA and use full multi-head attention everywhere.
    first_k_full_attention_layers (`int`, *optional*, defaults to 8):
        Number of initial transformer layers that use full multi-head attention.
        Layers at or after this index switch to GQA with `num_key_value_heads`.
    last_k_full_attention_layers (`int`, *optional*, defaults to 8):
        Number of final transformer layers that use full multi-head attention.
        Layers before `num_hidden_layers - last_k_full_attention_layers` use GQA with `num_key_value_heads`.
    """

    model_type = "sapiens2"

    hidden_size: int = 1024
    num_hidden_layers: int = 24
    num_attention_heads: int = 16
    intermediate_size: int = 2816
    use_gated_mlp: bool = True
    hidden_act: str = "silu"
    num_register_tokens: int = 8
    use_qk_norm: bool = True
    num_key_value_heads: int | None = None
    first_k_full_attention_layers: int = 8
    last_k_full_attention_layers: int = 8

    def __post_init__(self, **kwargs):
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads // 2
        super().__post_init__(**kwargs)


class Sapiens2BackboneOutput(DINOv3ViTBackboneOutput):
    pass


class Sapiens2Embeddings(DINOv3ViTEmbeddings):
    pass


class Sapiens2RopePositionEmbedding(DINOv3ViTRopePositionEmbedding):
    pass


class Sapiens2Attention(DINOv3ViTAttention):
    pass


class Sapiens2LayerScale(DINOv3ViTLayerScale):
    pass


class Sapiens2MLP(DINOv3ViTMLP):
    pass


class Sapiens2GatedMLP(DINOv3ViTGatedMLP):
    pass


class Sapiens2DropPath(Dinov3ViTDropPath):
    pass


class Sapiens2Layer(DINOv3ViTLayer):
    pass


class Sapiens2PreTrainedModel(DINOv3ViTPreTrainedModel):
    pass


class Sapiens2Encoder(DINOv3ViTEncoder):
    pass


class Sapiens2Model(DINOv3ViTModel):
    pass


class Sapiens2Backbone(DINOv3ViTBackbone):
    pass


class Sapiens2ImageProcessor(TorchvisionBackend):
    resample = PILImageResampling.BILINEAR
    image_mean = IMAGENET_DEFAULT_MEAN
    image_std = IMAGENET_DEFAULT_STD
    size = {"height": 768, "width": 1024}
    do_resize = True
    do_rescale = False
    do_normalize = True


__all__ = [
    "Sapiens2Config",
    "Sapiens2Model",
    "Sapiens2PreTrainedModel",
    "Sapiens2Backbone",
    "Sapiens2ImageProcessor",
]
