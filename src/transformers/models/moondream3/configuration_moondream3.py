# coding=utf-8
# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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

from typing import Optional, List

from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_rope_utils import RopeParameters, rope_config_validation, standardize_rope_params


class Moondream3TextConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`Moondream3TextModel`]. It is used to instantiate a
    Moondream3 model according to the specified arguments, defining the model architecture.

    Configuration objects inherit from [`PreTrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PreTrainedConfig`] for more information.

    Args:
        vocab_size (`int`, *optional*, defaults to 51200):
            Vocabulary size of the Moondream3 model.
        hidden_size (`int`, *optional*, defaults to 2048):
            Dimension of the hidden representations.
        intermediate_size (`int`, *optional*, defaults to 8192):
            Dimension of the MLP representations.
        num_hidden_layers (`int`, *optional*, defaults to 24):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 32):
            Number of attention heads for each attention layer in the Transformer encoder.
        num_key_value_heads (`int`, *optional*, defaults to 32):
            This is the number of key_value heads that should be used to implement Grouped Query Attention.
        max_position_embeddings (`int`, *optional*, defaults to 4096):
            The maximum sequence length that this model might ever be used with.
        num_experts (`int`, *optional*, defaults to 64):
            Number of experts for MoE layers.
        num_experts_per_tok (`int`, *optional*, defaults to 8):
            Number of selected experts per token.
        moe_intermediate_size (`int`, *optional*, defaults to 1024):
            Intermediate size of the routed expert.
        moe_start_layer (`int`, *optional*, defaults to 4):
            The layer index where MoE layers start.
        bos_token_id (`int`, *optional*, defaults to 0):
            The id of the beginning-of-sequence token.
        eos_token_id (`int`, *optional*, defaults to 0):
            The id of the end-of-sequence token.
        coord_token_id (`int`, *optional*, defaults to 5):
            The id of the coordinate token used for region detection.
        hidden_act (`str` or `function`, *optional*, defaults to `"gelu_pytorch_tanh"`):
            The non-linear activation function.
        moe_hidden_act (`str` or `function`, *optional*, defaults to `"gelu"`):
            The non-linear activation function used inside MoE experts.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer.
        rms_norm_eps (`float`, *optional*, defaults to 1e-5):
            The epsilon used by the rms normalization layers.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions.
        tie_word_embeddings (`bool`, *optional*, defaults to `False`):
            Whether the model's input and output word embeddings should be tied.
        attention_bias (`bool`, *optional*, defaults to `False`):
            Whether to use a bias in the query, key, value and output projection layers.
        rope_parameters (`dict`, *optional*):
            The dictionary containing parameters for RoPE (Rotary Positional Embeddings), such as `rope_theta` and `rope_type`.
        head_dim (`int`, *optional*):
            The dimension of the head. If not specified, will default to `hidden_size // num_attention_heads`.
    """

    model_type = "moondream3_text"
    base_config_key = "text_config"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size: int = 51200,
        hidden_size: int = 2048,
        intermediate_size: int = 8192,
        num_hidden_layers: int = 24,
        num_attention_heads: int = 32,
        num_key_value_heads: int = 32,
        max_position_embeddings: int = 4096,
        num_experts: int = 64,
        num_experts_per_tok: int = 8,
        moe_intermediate_size: int = 1024,
        moe_start_layer: int = 4,
        bos_token_id: int = 0,
        eos_token_id: int = 0,
        coord_token_id: int = 5,
        hidden_act: str = "gelu_pytorch_tanh",
        moe_hidden_act: str = "gelu",
        initializer_range: float = 0.02,
        rms_norm_eps: float = 1e-5,
        use_cache: bool = True,
        tie_word_embeddings: bool = False,
        attention_bias: bool = True,
        rope_parameters: Optional[RopeParameters | dict[str, RopeParameters]] = None,
        head_dim: Optional[int] = None,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.attention_bias = attention_bias
        self.head_dim = head_dim or hidden_size // num_attention_heads
        self.bos_token_id = bos_token_id
        self.coord_token_id = coord_token_id
        self.eos_token_id = eos_token_id

        # MoE parameters (merged from TextMoeConfig)
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.moe_intermediate_size = moe_intermediate_size
        self.moe_start_layer = moe_start_layer
        self.moe_hidden_act = moe_hidden_act

        # Try to set `rope_scaling` if available, otherwise use `rope_parameters`
        rope_scaling = kwargs.pop("rope_scaling", None)
        self.rope_parameters = rope_scaling or rope_parameters

        # Validate the correctness of rotary position embeddings parameters
        rope_theta = kwargs.get("rope_theta", 1500000.0)
        standardize_rope_params(self, rope_theta=rope_theta)
        rope_config_validation(self)

        # HF compatibility attributes
        self.output_router_logits = False
        self.output_attentions = False
        self.output_hidden_states = False
        self.attention_dropout = 0.0

        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)


class Moondream3VisionConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of the Moondream3 vision encoder.

    Args:
        hidden_size (`int`, *optional*, defaults to 1152):
            Dimension of the encoder's hidden states.
        intermediate_size (`int`, *optional*, defaults to 4304):
            Dimension of the encoder's MLP representations.
        num_hidden_layers (`int`, *optional*, defaults to 27):
            Number of hidden layers in the vision encoder.
        num_attention_heads (`int`, *optional*, defaults to 16):
            Number of attention heads in the vision encoder.
        patch_size (`int`, *optional*, defaults to 14):
            The size of each patch in the vision encoder.
        in_channels (`int`, *optional*, defaults to 3):
            Number of input channels.
        proj_out_dim (`int`, *optional*, defaults to 2048):
            Output dimension of the projection layer.
        crop_size (`int`, *optional*, defaults to 378):
            Size of image crops.
        max_crops (`int`, *optional*, defaults to 12):
            Maximum number of crops.
        overlap_margin (`int`, *optional*, defaults to 4):
            Overlap margin for crops.
        proj_inner_dim (`int`, *optional*, defaults to 8192):
            Inner dimension of the projection MLP.
        prefix_len (`int`, *optional*, defaults to 730):
            The number of tokens used to represent the visual input (prefix length).
        hidden_act (`str`, *optional*, defaults to `"gelu_pytorch_tanh"`):
            The non-linear activation function.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer.
        attention_bias (`bool`, *optional*, defaults to `True`):
            Whether to use a bias in the query, key, value and output projection layers.
    """

    model_type = "moondream3_vision"
    base_config_key = "vision_config"

    def __init__(
        self,
        hidden_size: int = 1152,
        intermediate_size: int = 4304,
        num_hidden_layers: int = 27,
        num_attention_heads: int = 16,
        patch_size: int = 14,
        in_channels: int = 3,
        proj_out_dim: int = 2048,
        crop_size: int = 378,
        max_crops: int = 12,
        overlap_margin: int = 4,
        proj_inner_dim: int = 8192,
        prefix_len: int = 730,
        hidden_act: str = "gelu_pytorch_tanh",
        initializer_range: float = 0.02,
        attention_bias: bool = True,
        **kwargs,
    ):
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.proj_out_dim = proj_out_dim
        self.crop_size = crop_size
        self.max_crops = max_crops
        self.prefix_len = prefix_len
        self.overlap_margin = overlap_margin
        self.proj_inner_dim = proj_inner_dim
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.attention_dropout = 0.0
        self.attention_bias = attention_bias

        super().__init__(**kwargs)


class Moondream3RegionConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of the Moondream3 region encoder for object detection and grounding.

    Args:
        hidden_size (`int`, *optional*, defaults to 2048):
            Dimension of the hidden representations for region features.
        coord_feat_dim (`int`, *optional*, defaults to 256):
            Dimension of coordinate feature embeddings.
        coord_out_dim (`int`, *optional*, defaults to 1024):
            Output dimension for coordinate features.
        size_feat_dim (`int`, *optional*, defaults to 512):
            Dimension of size feature embeddings.
        size_out_dim (`int`, *optional*, defaults to 2048):
            Output dimension for size features.
    """

    model_type = "moondream3_region"
    base_config_key = "region_config"

    def __init__(
        self,
        hidden_size: int = 2048,
        coord_feat_dim: int = 256,
        coord_out_dim: int = 1024,
        size_feat_dim: int = 512,
        size_out_dim: int = 2048,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.hidden_size = hidden_size
        self.coord_feat_dim = coord_feat_dim
        self.coord_out_dim = coord_out_dim
        self.size_feat_dim = size_feat_dim
        self.size_out_dim = size_out_dim


class Moondream3Config(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`Moondream3Model`].

    Args:
        text_config (`Union[PreTrainedConfig, dict]`, *optional*, defaults to `Moondream3TextConfig`):
            The config object or dictionary of the text backbone.
        vision_config (`Union[PreTrainedConfig, dict]`, *optional*, defaults to `Moondream3VisionConfig`):
            The config object or dictionary of the vision backbone.
        region_config (`Union[PreTrainedConfig, dict]`, *optional*, defaults to `Moondream3RegionConfig`):
            The config object or dictionary of the region backbone for object detection and grounding.
        bos_token_id (`int`, *optional*, defaults to 0):
            The id of the beginning-of-sequence token.
        tie_word_embeddings (`bool`, *optional*, defaults to `False`):
            Whether to tie the word embeddings.
    """

    model_type = "moondream3"
    sub_configs = {
        "vision_config": Moondream3VisionConfig,
        "text_config": Moondream3TextConfig,
        "region_config": Moondream3RegionConfig,
    }
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        text_config=None,
        vision_config=None,
        region_config=None,
        bos_token_id=0,
        tie_word_embeddings: bool = False,
        **kwargs,
    ):
        if isinstance(vision_config, dict):
            self.vision_config = self.sub_configs["vision_config"](**vision_config)
        elif vision_config is None:
            self.vision_config = self.sub_configs["vision_config"]()

        if isinstance(text_config, dict):
            self.text_config = self.sub_configs["text_config"](**text_config)
        elif text_config is None:
            self.text_config = self.sub_configs["text_config"]()

        if isinstance(region_config, dict):
            self.region_config = self.sub_configs["region_config"](**region_config)
        elif region_config is None:
            self.region_config = self.sub_configs["region_config"]()

        super().__init__(**kwargs, tie_word_embeddings=tie_word_embeddings)


__all__ = [
    "Moondream3Config",
    "Moondream3TextConfig",
    "Moondream3VisionConfig",
    "Moondream3RegionConfig",
]
