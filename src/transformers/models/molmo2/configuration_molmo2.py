# Copyright 2026 The HuggingFace Team. All rights reserved.
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

"""
Molmo2 configuration
"""

from dataclasses import field
from typing import Any

from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...utils import logging
from ...utils.auto_docstring import auto_docstring


logger = logging.get_logger(__name__)


@auto_docstring(checkpoint="allenai/Molmo2-8B")
@strict
class Molmo2VitConfig(PreTrainedConfig):
    r"""
    image_default_input_size (`list[int]`, *optional*, defaults to `[378, 378]`):
        Default input image size (height, width).
    image_patch_size (`int`, *optional*, defaults to 14):
        Size of each image patch.
    image_num_pos (`int`, *optional*, defaults to 577):
        Number of positional embeddings for the image.
    """

    model_type = "molmo2"
    base_config_key = "vit_config"

    hidden_size: int = 1152
    intermediate_size: int = 4304
    num_hidden_layers: int = 27
    num_attention_heads: int = 16
    num_key_value_heads: int = 16
    head_dim: int = 72
    hidden_act: str = "gelu_pytorch_tanh"
    layer_norm_eps: float = 1e-6
    image_default_input_size: list[int] = field(default_factory=lambda: [378, 378])
    image_patch_size: int = 14
    image_num_pos: int = 577
    attention_dropout: float = 0.0
    residual_dropout: float = 0.0
    initializer_range: float = 0.02

    @property
    def image_num_patch(self):
        h, w = self.image_default_input_size
        return h // self.image_patch_size, w // self.image_patch_size


@auto_docstring(checkpoint="allenai/Molmo2-8B")
@strict
class Molmo2AdapterConfig(PreTrainedConfig):
    r"""
    vit_layers (`list[int]`, *optional*, defaults to `[-3, -9]`):
        Indices of ViT layers to extract features from.
    pooling_attention_mask (`bool`, *optional*, defaults to `False`):
        Whether to use attention mask during pooling.
    text_hidden_size (`int`, *optional*, defaults to 3584):
        Hidden size of the text model (used for projection).
    image_feature_dropout (`float`, *optional*, defaults to 0.0):
        Dropout rate for image features.
    """

    model_type = "molmo2"
    base_config_key = "adapter_config"

    vit_layers: list[int] = field(default_factory=lambda: [-3, -9])
    pooling_attention_mask: bool = False
    hidden_size: int = 1152
    num_attention_heads: int = 16
    num_key_value_heads: int = 16
    head_dim: int = 72
    attention_dropout: float = 0.0
    residual_dropout: float = 0.0
    hidden_act: str = "silu"
    intermediate_size: int = 18944
    text_hidden_size: int = 3584
    image_feature_dropout: float = 0.0
    initializer_range: float = 0.02


@auto_docstring(checkpoint="allenai/Molmo2-8B")
@strict
class Molmo2TextConfig(PreTrainedConfig):
    r"""
    additional_vocab_size (`int`, *optional*, defaults to 128):
        Number of additional vocabulary tokens beyond the base vocabulary.
    qkv_bias (`bool`, *optional*, defaults to `True`):
        Whether to use bias in query, key, and value projections.
    embedding_dropout (`float`, *optional*, defaults to 0.0):
        The dropout ratio for the embedding layer.
    residual_dropout (`float`, *optional*, defaults to 0.0):
        The dropout ratio applied after residual connections.
    rope_theta (`float`, *optional*, defaults to 1000000.0):
        The base period of the RoPE embeddings.
    rope_scaling (`dict[str, Any]`, *optional*):
        Dictionary containing the scaling configuration for the RoPE embeddings.
    rope_scaling_layers (`list[int]`, *optional*):
        List of layer indices where rope scaling is applied.
    norm_after (`bool`, *optional*, defaults to `False`):
        Whether to apply layer normalization after the attention/FFN blocks instead of before.
    """

    model_type = "molmo2_text"
    base_config_key = "text_config"
    keys_to_ignore_at_inference = ["past_key_values"]
    base_model_tp_plan = {
        "blocks.*.self_attn.att_proj": "colwise",
        "blocks.*.self_attn.attn_out": "rowwise",
        "blocks.*.mlp.ff_proj": "colwise",
        "blocks.*.mlp.ff_out": "rowwise",
    }
    base_model_pp_plan = {
        "wte": (["input_ids"], ["inputs_embeds"]),
        "blocks": (["hidden_states", "attention_mask"], ["hidden_states"]),
        "norm": (["hidden_states"], ["hidden_states"]),
    }

    hidden_size: int = 3584
    num_attention_heads: int = 28
    num_key_value_heads: int | None = 4
    head_dim: int = 128
    vocab_size: int = 152064
    additional_vocab_size: int = 128
    qkv_bias: bool = True
    num_hidden_layers: int = 48
    intermediate_size: int = 18944
    hidden_act: str = "silu"
    embedding_dropout: float = 0.0
    attention_dropout: float = 0.0
    residual_dropout: float = 0.0
    max_position_embeddings: int = 4096
    rope_theta: float = 1000000.0
    rope_scaling: dict[str, Any] | None = None
    rope_scaling_layers: list[int] | None = None
    layer_norm_eps: float = 1e-6
    norm_after: bool = False
    initializer_range: float = 0.02
    use_cache: bool = True
    tie_word_embeddings: bool = False

    def __post_init__(self, **kwargs):
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads
        super().__post_init__(**kwargs)


@auto_docstring(checkpoint="allenai/Molmo2-8B")
@strict
class Molmo2Config(PreTrainedConfig):
    r"""
    vit_config (`Molmo2VitConfig`, *optional*):
        Configuration for the vision transformer backbone.
    adapter_config (`Molmo2AdapterConfig`, *optional*):
        Configuration for the vision-to-language adapter.
    image_start_token_id (`int`, *optional*):
        Token ID marking the start of an image region.
    low_res_image_start_token_id (`int`, *optional*):
        Token ID marking the start of a low-resolution image crop.
    image_end_token_id (`int`, *optional*):
        Token ID marking the end of an image region.
    image_low_res_id (`int`, *optional*):
        Token ID for low-resolution image patches.
    image_patch_id (`int`, *optional*):
        Token ID for image patches.
    image_col_id (`int`, *optional*):
        Token ID for column separators in image patch sequences.
    frame_start_token_id (`int`, *optional*):
        Token ID marking the start of a video frame.
    frame_end_token_id (`int`, *optional*):
        Token ID marking the end of a video frame.
    use_frame_special_tokens (`bool`, *optional*, defaults to `True`):
        Whether to use special tokens to delineate video frames.
    tie_word_embeddings (`bool`, *optional*, defaults to `False`):
        Whether the model's input and output word embeddings should be tied.
    """

    model_type = "molmo2"
    sub_configs = {
        "text_config": Molmo2TextConfig,
        "vit_config": Molmo2VitConfig,
        "adapter_config": Molmo2AdapterConfig,
    }

    vit_config: dict | PreTrainedConfig | None = None
    adapter_config: dict | PreTrainedConfig | None = None
    text_config: dict | PreTrainedConfig | None = None
    image_start_token_id: int | None = None
    low_res_image_start_token_id: int | None = None
    image_end_token_id: int | None = None
    image_low_res_id: int | None = None
    image_patch_id: int | None = None
    image_col_id: int | None = None
    frame_start_token_id: int | None = None
    frame_end_token_id: int | None = None
    use_frame_special_tokens: bool = True
    initializer_range: float = 0.02
    tie_word_embeddings: bool = False

    def __post_init__(self, **kwargs):
        if isinstance(self.vit_config, dict):
            self.vit_config = self.sub_configs["vit_config"](**self.vit_config)
        elif self.vit_config is None:
            self.vit_config = self.sub_configs["vit_config"]()

        if isinstance(self.adapter_config, dict):
            self.adapter_config = self.sub_configs["adapter_config"](**self.adapter_config)
        elif self.adapter_config is None:
            self.adapter_config = self.sub_configs["adapter_config"]()

        if isinstance(self.text_config, dict):
            self.text_config = self.sub_configs["text_config"](**self.text_config)
        elif self.text_config is None:
            self.text_config = self.sub_configs["text_config"]()

        # Normalize negative `vit_layers` indices and trim the ViT to the deepest layer the adapter actually reads.
        num_vit_layers = self.vit_config.num_hidden_layers
        self.adapter_config.vit_layers = [
            layer if layer >= 0 else layer + num_vit_layers for layer in self.adapter_config.vit_layers
        ]
        last_layer_needed = max(self.adapter_config.vit_layers) + 1
        if last_layer_needed < num_vit_layers:
            self.vit_config.num_hidden_layers = last_layer_needed

        self.image_high_res_id = self.image_patch_id
        self.use_cache = self.text_config.use_cache
        super().__post_init__(**kwargs)


__all__ = [
    "Molmo2AdapterConfig",
    "Molmo2Config",
    "Molmo2TextConfig",
    "Molmo2VitConfig",
]
