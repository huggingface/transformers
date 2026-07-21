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
"""LingBot-Vision model configuration."""

from huggingface_hub.dataclasses import strict

from ...backbone_utils import BackboneConfigMixin
from ...configuration_utils import PreTrainedConfig
from ...modeling_rope_utils import RopeParameters
from ...utils import auto_docstring


@auto_docstring(checkpoint="IMvision12/lingbot-vision-vit-giant-hf")
@strict
class LingbotVisionConfig(BackboneConfigMixin, PreTrainedConfig):
    r"""
    image_size (`int`, *optional*, defaults to 224):
        The nominal image size used to initialize the patch embedding metadata.
    patch_size (`int`, *optional*, defaults to 16):
        Patch size used by the ViT patch embedding.
    num_channels (`int`, *optional*, defaults to 3):
        Number of input image channels.
    hidden_size (`int`, *optional*, defaults to 768):
        Dimensionality of the encoder layers and the pooler layer.
    num_hidden_layers (`int`, *optional*, defaults to 12):
        Number of hidden layers in the Transformer encoder.
    num_attention_heads (`int`, *optional*, defaults to 12):
        Number of attention heads for each attention layer.
    mlp_ratio (`float`, *optional*, defaults to 4.0):
        Expansion ratio for the feed-forward layers.
    qkv_bias (`bool`, *optional*, defaults to `True`):
        Whether to add a bias to the query, key and value projections.
    proj_bias (`bool`, *optional*, defaults to `True`):
        Whether to add a bias to the attention output projection.
    ffn_bias (`bool`, *optional*, defaults to `True`):
        Whether to add biases to the feed-forward projections.
    layer_scale_init_value (`float`, *optional*):
        Initial value for LayerScale. If `None`, LayerScale is disabled.
    num_storage_tokens (`int`, *optional*, defaults to 0):
        Number of LingBot-Vision storage/register tokens inserted after the class token.
    ffn_layer (`str`, *optional*, defaults to `"mlp"`):
        Feed-forward layer type. Supports `"mlp"`, `"swiglu"`, `"swiglu32"`, `"swiglu64"` and `"swiglu128"`.
    norm_layer (`str`, *optional*, defaults to `"layernorm"`):
        Normalization layer type. Supports `"layernorm"`, `"layernormbf16"` and `"rmsnorm"`.
    mask_k_bias (`bool`, *optional*, defaults to `False`):
        Whether to zero the key-bias slice in the fused QKV projection.
    rope_parameters (`dict`, *optional*):
        Parameters for axial rotary position embeddings. `rope_theta` sets the base period and defaults to 100.0.
    rope_min_period (`float`, *optional*):
        Minimum period for axial rotary position embeddings.
    rope_max_period (`float`, *optional*):
        Maximum period for axial rotary position embeddings.
    rope_normalize_coords (`str`, *optional*, defaults to `"separate"`):
        Coordinate normalization strategy for axial RoPE. Supports `"min"`, `"max"` and `"separate"`.
    rope_shift_coords (`float`, *optional*):
        Random coordinate shift used by LingBot-Vision RoPE during training.
    rope_jitter_coords (`float`, *optional*):
        Random coordinate jitter used by LingBot-Vision RoPE during training.
    rope_rescale_coords (`float`, *optional*):
        Random coordinate rescaling used by LingBot-Vision RoPE during training.
    rope_dtype (`str`, *optional*, defaults to `"bf16"`):
        Dtype used to build RoPE tables. Supports `"fp32"`, `"fp16"` and `"bf16"`.
    untie_cls_and_patch_norms (`bool`, *optional*, defaults to `False`):
        Whether to use a separate final norm for class/storage tokens.
    apply_layernorm (`bool`, *optional*, defaults to `True`):
        Whether `LingbotVisionBackbone` should apply the final layer norm to selected feature maps.
    reshape_hidden_states (`bool`, *optional*, defaults to `True`):
        Whether `LingbotVisionBackbone` should reshape feature maps to `(batch, channels, height, width)`.
    """

    model_type = "lingbot_vision"
    default_theta = 100.0

    image_size: int = 224
    patch_size: int = 16
    num_channels: int = 3
    hidden_size: int = 768
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    mlp_ratio: float = 4.0
    qkv_bias: bool = True
    proj_bias: bool = True
    ffn_bias: bool = True
    hidden_dropout_prob: float = 0.0
    attention_probs_dropout_prob: float = 0.0
    drop_path_rate: float = 0.0
    initializer_range: float = 0.02
    layer_norm_eps: float = 1e-6
    layer_scale_init_value: float | None = None
    num_storage_tokens: int = 0
    ffn_layer: str = "mlp"
    norm_layer: str = "layernorm"
    mask_k_bias: bool = False
    rope_parameters: RopeParameters | dict | None = None
    rope_min_period: float | None = None
    rope_max_period: float | None = None
    rope_normalize_coords: str = "separate"
    rope_shift_coords: float | None = None
    rope_jitter_coords: float | None = None
    rope_rescale_coords: float | None = None
    rope_dtype: str = "bf16"
    untie_cls_and_patch_norms: bool = False
    apply_layernorm: bool = True
    reshape_hidden_states: bool = True
    _out_features: list[str] | None = None
    _out_indices: list[int] | None = None

    def __post_init__(self, **kwargs):
        self.stage_names = ["stem"] + [f"stage{idx}" for idx in range(1, self.num_hidden_layers + 1)]
        self.set_output_features_output_indices(
            out_indices=kwargs.pop("out_indices", None), out_features=kwargs.pop("out_features", None)
        )
        super().__post_init__(**kwargs)


__all__ = ["LingbotVisionConfig"]
