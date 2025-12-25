# coding=utf-8
# Copyright 2025 the HuggingFace Team. All rights reserved.
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

from typing import Optional

import torch.nn as nn

from ...modeling_rope_utils import RopeParameters
from ..glm4v.configuration_glm4v import Glm4vConfig, Glm4vTextConfig
from ..glm4v.modeling_glm4v import (
    Glm4vForConditionalGeneration,
    Glm4vModel,
    Glm4vPreTrainedModel,
    Glm4vTextDecoderLayer,
    Glm4vTextModel,
)


class GlmImageTextConfig(Glm4vTextConfig):
    r"""
    This is the configuration class to store the configuration of a [`GlmImageModel`]. It is used to instantiate a
    GLM-Image model according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of
    GLM-Image [zai-org/GLM-Image](https://huggingface.co/zai-org/GLM-Image).

    Configuration objects inherit from [`PreTrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PreTrainedConfig`] for more information.

    Args:
        vocab_size (`int`, *optional*, defaults to 168064):
            Vocabulary size of the GlmImage model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`GlmImageModel`]
        vision_vocab_size (`int`, *optional*, defaults to 16512):
            Vision vocabulary size of the GlmImage model. Defines the number of different tokens that can be represented
            by the `inputs_ids` passed when calling [`GlmImageVisionModel`]
        hidden_size (`int`, *optional*, defaults to 4096):
            Dimension of the hidden representations.
        intermediate_size (`int`, *optional*, defaults to 13696):
            Dimension of the MLP representations.
        num_hidden_layers (`int`, *optional*, defaults to 40):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 32):
            Number of attention heads for each attention layer in the Transformer encoder.
        num_key_value_heads (`int`, *optional*, defaults to 2):
            This is the number of key_value heads that should be used to implement Grouped Query Attention. If
            `num_key_value_heads=num_attention_heads`, the model will use Multi Head Attention (MHA), if
            `num_key_value_heads=1` the model will use Multi Query Attention (MQA) otherwise GQA is used. When
            converting a multi-head checkpoint to a GQA checkpoint, each group key and value head should be constructed
            by meanpooling all the original heads within that group. For more details checkout [this
            paper](https://huggingface.co/papers/2305.13245). If it is not specified, will default to `32`.
        hidden_act (`str` or `function`, *optional*, defaults to `"silu"`):
            The non-linear activation function (function or string) in the decoder.
        max_position_embeddings (`int`, *optional*, defaults to 32768):
            The maximum sequence length that this model might ever be used with.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        rms_norm_eps (`float`, *optional*, defaults to 1e-05):
            The epsilon used by the rms normalization layers.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models). Only
            relevant if `config.is_decoder=True`.
        tie_word_embeddings (`bool`, *optional*, defaults to `False`):
            Whether the model's input and output word embeddings should be tied.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        rope_parameters (`RopeParameters`, *optional*):
            Dictionary containing the configuration parameters for the RoPE embeddings. The dictionary should contain
            a value for `rope_theta` and optionally parameters used for scaling in case you want to use RoPE
            with longer `max_position_embeddings`.

    ```python
    >>> from transformers import GlmImageTextModel, GlmImageConfig

    >>> # Initializing a GLM-4.1V style configuration
    >>> configuration = GlmImageConfig()

    >>> # Initializing a model from the GLM-4.1V style configuration
    >>> model = GlmImageTextModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "glm_image_text"
    base_config_key = "text_config"
    keys_to_ignore_at_inference = ["past_key_values"]
    # Default tensor parallel plan for base model `GlmImage`
    base_model_tp_plan = {
        "layers.*.self_attn.q_proj": "colwise",
        "layers.*.self_attn.k_proj": "colwise",
        "layers.*.self_attn.v_proj": "colwise",
        "layers.*.self_attn.o_proj": "rowwise",
        "layers.*.mlp.gate_up_proj": "colwise_rep",  # we need to replicate here due to the `chunk` operation
        "layers.*.mlp.down_proj": "rowwise_rep",  # we need to replicate here due to the `chunk` operation
    }
    base_model_pp_plan = {
        "embed_tokens": (["input_ids"], ["inputs_embeds"]),
        "layers": (["hidden_states", "attention_mask"], ["hidden_states"]),
        "norm": (["hidden_states"], ["hidden_states"]),
    }

    def __init__(
        self,
        vocab_size: Optional[int] = 168064,
        vision_vocab_size: Optional[int] = 16512,
        hidden_size: Optional[int] = 4096,
        intermediate_size: Optional[int] = 13696,
        num_hidden_layers: Optional[int] = 40,
        num_attention_heads: Optional[int] = 32,
        num_key_value_heads: Optional[int] = 2,
        hidden_act: Optional[str] = "silu",
        max_position_embeddings: Optional[int] = 32768,
        initializer_range: Optional[float] = 0.02,
        rms_norm_eps: Optional[int] = 1e-05,
        use_cache: Optional[bool] = True,
        tie_word_embeddings: Optional[bool] = False,
        attention_dropout: Optional[float] = 0.0,
        rope_parameters: Optional[RopeParameters | dict[str, RopeParameters]] = None,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.vision_vocab_size = vision_vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads

        # for backward compatibility
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads

        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.attention_dropout = attention_dropout
        self.rope_parameters = rope_parameters

        super().__init__(
            tie_word_embeddings=tie_word_embeddings, ignore_keys_at_rope_validation={"mrope_section"}, **kwargs
        )


class GlmImageConfig(Glm4vConfig):
    pass


class GlmImageDecoderLayer(Glm4vTextDecoderLayer):
    pass


class GlmImagePreTrainedModel(Glm4vPreTrainedModel):
    pass


class GlmImageTextModel(Glm4vTextModel):
    pass


class GlmImageModel(Glm4vModel):
    pass


class GlmImageForConditionalGeneration(Glm4vForConditionalGeneration):
    def __init__(self, config):
        super().__init__(config)
        self.model = GlmImageModel(config)
        self.lm_head = nn.Linear(config.text_config.hidden_size, config.text_config.vision_vocab_size, bias=False)


__all__ = ["GlmImageConfig", "GlmImagePreTrainedModel", "GlmImageTextModel", "GlmImageForConditionalGeneration"]
