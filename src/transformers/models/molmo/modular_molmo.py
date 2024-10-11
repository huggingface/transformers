# coding=utf-8
# Copyright 2024 HuggingFace Inc. team. All rights reserved.
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

import torch
from torch import nn
from ...modeling_rope_utils import rope_config_validation
from ..clip.configuration_clip import CLIPVisionConfig
from ..qwen2.configuration_qwen2 import Qwen2Config
from ...configuration_utils import PretrainedConfig
from ...utils import logging
from ..auto import CONFIG_MAPPING
from ..clip.modeling_clip import (
    CLIPAttention,
    CLIPEncoder,
    CLIPEncoderLayer,
    CLIPFlashAttention2,
    CLIPSdpaAttention,
    CLIPVisionEmbeddings,
    CLIPVisionModel,
    CLIPVisionTransformer,
)
from ..llava.modeling_llava import (
    LlavaForConditionalGeneration,
    LlavaMultiModalProjector,
)
from ..qwen2.modeling_qwen2 import (
    Qwen2Attention,
    Qwen2DecoderLayer,
    Qwen2FlashAttention2,
    Qwen2ForCausalLM,
    Qwen2MLP,
    Qwen2Model,
    Qwen2SdpaAttention,
)


logger = logging.get_logger(__name__)


class MolmoVisionConfig(CLIPVisionConfig):
    def __init__(
        self,
        hidden_size=1024,
        num_attention_heads=32,
        intermediate_size = 4096,
        image_num_key_value_heads=16,
        num_hidden_layers = 23,
        num_image_positions = 577,
        projection_dim=512,
        num_channels=3,
        image_size=336,
        patch_size=14,
        hidden_act="quick_gelu",
        layer_norm_eps=1e-5,
        attention_dropout=0.0,
        initializer_range=0.02,
        initializer_factor=1.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.image_num_key_value_heads = image_num_key_value_heads
        self.num_hidden_layers = num_hidden_layers
        self.num_image_positions = num_image_positions
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.projection_dim = projection_dim
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_channels = num_channels
        self.patch_size = patch_size
        self.image_size = image_size
        self.initializer_range = initializer_range
        self.initializer_factor = initializer_factor
        self.attention_dropout = attention_dropout
        self.layer_norm_eps = layer_norm_eps
        self.hidden_act = hidden_act

class MolmoTextConfig(Qwen2Config):
    def __init__(
        self,
        hidden_size = 3584,
        num_key_value_heads = 4,
        num_attention_heads = 28,
        num_hidden_layers = 28,
        head_dim = 128,
        vocab_size = 152064,
        additional_vocab_size = 128,
        intermediate_size = 37888,
        hidden_act="silu",
        max_position_embeddings=32768,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=True,
        tie_word_embeddings=False,
        rope_theta=10000.0,
        rope_scaling=None,
        use_sliding_window=False,
        sliding_window=4096,
        max_window_layers=28,
        attention_dropout=0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.num_key_value_heads = num_key_value_heads
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers
        self.head_dim = head_dim
        self.vocab_size = vocab_size
        self.additional_vocab_size = additional_vocab_size
        self.intermediate_size = intermediate_size
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.use_sliding_window = use_sliding_window
        self.sliding_window = sliding_window if use_sliding_window else None
        self.max_window_layers = max_window_layers

        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.attention_dropout = attention_dropout
        # Validate the correctness of rotary position embeddings parameters
        # BC: if there is a 'type' field, move it to 'rope_type'.
        if self.rope_scaling is not None and "type" in self.rope_scaling:
            self.rope_scaling["rope_type"] = self.rope_scaling["type"]
        rope_config_validation(self)

        super().__init__(
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )
class MolmoConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`LlavaForConditionalGeneration`]. It is used to instantiate an
    Llava model according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of the Llava-9B.

    e.g. [llava-hf/llava-9b](https://huggingface.co/llava-hf/llava-9b)

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        vision_config (`Union[AutoConfig, dict]`,  *optional*, defaults to `CLIPVisionConfig`):
            The config object or dictionary of the vision backbone.
        text_config (`Union[AutoConfig, dict]`, *optional*, defaults to `LlamaConfig`):
            The config object or dictionary of the text backbone.
        ignore_index (`int`, *optional*, defaults to -100):
            The ignore index for the loss function.
        image_token_index (`int`, *optional*, defaults to 32000):
            The image token index to encode the image prompt.
        projector_hidden_act (`str`, *optional*, defaults to `"gelu"`):
            The activation function used by the multimodal projector.
        vision_feature_select_strategy (`str`, *optional*, defaults to `"default"`):
            The feature selection strategy used to select the vision feature from the vision backbone.
            Can be one of `"default"` or `"full"`.
        vision_feature_layer (`int`, *optional*, defaults to -2):
            The index of the layer to select the vision feature.
        image_seq_length (`int`, *optional*, defaults to 576):
            Sequence length of one image embedding.

    Example:

    ```python
    >>> from transformers import LlavaForConditionalGeneration, LlavaConfig, CLIPVisionConfig, LlamaConfig

    >>> # Initializing a CLIP-vision config
    >>> vision_config = CLIPVisionConfig()

    >>> # Initializing a Llama config
    >>> text_config = LlamaConfig()

    >>> # Initializing a Llava llava-1.5-7b style configuration
    >>> configuration = LlavaConfig(vision_config, text_config)

    >>> # Initializing a model from the llava-1.5-7b style configuration
    >>> model = LlavaForConditionalGeneration(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "llava"
    is_composition = True

    def __init__(
        self,
        vision_config=None,
        text_config=None,
        ignore_index=-100,
        image_token_index=32000,
        projector_hidden_act="gelu",
        image_seq_length=576,
        initializer_range=0.02,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.ignore_index = ignore_index
        self.image_token_index = image_token_index
        self.projector_hidden_act = projector_hidden_act
        self.image_seq_length = image_seq_length
        if vision_config is None:
            vision_config = {}
            logger.info("vision_config is None. initializing the MolmoVisionConfig with default values.")
        if text_config is None:
            text_config = {}
            logger.info("text_config is None. initializing the MolmoTextConfig with default values.")
        self.vision_config = MolmoVisionConfig(**vision_config)
        self.text_config = MolmoTextConfig(**text_config)
        self.initializer_range = initializer_range

    @classmethod
    def from_text_vision_configs(cls, text_config: MolmoTextConfig, vision_config: MolmoVisionConfig, **kwargs):
        r"""
        Instantiate a [`MolmoConfig`] (or a derived class) from molmo text model configuration and molmo vision model
        configuration.

        Returns:
            [`MolmoConfig`]: An instance of a configuration object
        """

        return cls(text_config=text_config.to_dict(), vision_config=vision_config.to_dict(), **kwargs)

# text modules inherited from Qwen2


class MolmoMLP(Qwen2MLP):
    def __init__(self, config):
        super().__init__()
        self.down_proj = nn.Linear(config.intermediate_size // 2, config.hidden_size, bias=False)


# We have different attention classes for the txt and the image components, they need to be propagated back correctly
class MolmoTextAttention(Qwen2Attention):
    pass


class MolmoTextSdpaAttention(MolmoTextAttention, Qwen2SdpaAttention):
    pass


class MolmoTextFlashAttention2(MolmoTextAttention, Qwen2FlashAttention2):
    pass


MOLMO_TEXT_ATTENTION_CLASSES = {
    "eager": MolmoTextAttention,
    "sdpa": MolmoTextSdpaAttention,
    "flash_attention_2": MolmoTextFlashAttention2,
}


class MolmoDecoderLayer(Qwen2DecoderLayer):
    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.mlp = MolmoMLP(config)
        self.self_attn = MOLMO_TEXT_ATTENTION_CLASSES[config._attn_implementation](config, layer_idx)


class MolmoTextModel(Qwen2Model):
    def __init__(self, config):
        super().__init__(config)
        self.embed_tokens = nn.Embedding(
            config.vocab_size + config.additional_vocab_size,
            config.hidden_size,
        )

        self.layers = nn.ModuleList(
            [MolmoDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.post_init()


# TODO the name matching here is error-inducing as MolmoForCausalLM isn't a standalone generative model
class MolmoForCausalLM(Qwen2ForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        self.model = MolmoTextModel(config)
        self.post_init()


# New Molmo multimodal projection and image pooling


class MolmoMultiModalProjector(LlavaMultiModalProjector):
    def __init__(self, config: MolmoConfig):
        super().__init__()
        self.linear_1 = nn.Linear(
            config.vision_config.hidden_size,
            config.text_config.intermediate_size // 2,
            bias=False,
        )
        self.linear_2 = nn.Linear(
            config.text_config.intermediate_size // 2,
            config.text_config.hidden_size,
            bias=False,
        )
        self.linear_3 = nn.Linear(
            config.vision_config.hidden_size,
            config.text_config.intermediate_size // 2,
            bias=False,
        )

    def forward(self, image_features):
        hidden_states = self.linear_1(image_features)
        hidden_states = self.act(hidden_states)
        intermediate_states = self.linear_3(image_features)
        hidden_states = self.linear_2(hidden_states, intermediate_states)
        return hidden_states


# Molmo image components inherited from CLIPVision


# We have different attention classes for the txt and the image components, they need to be propagated back correctly


class MolmoVisionAttention(CLIPAttention):
    pass


class MolmoVisionSdpaAttention(MolmoVisionAttention, CLIPSdpaAttention):
    pass


class MolmoVisionFlashAttention2(MolmoVisionAttention, CLIPFlashAttention2):
    pass


MOLMO_VISION_ATTENTION_CLASSES = {
    "eager": MolmoVisionAttention,
    "sdpa": MolmoVisionSdpaAttention,
    "flash_attention_2": MolmoVisionFlashAttention2,
}


class MolmoVisionEmbeddings(CLIPVisionEmbeddings):
    def __init__(self, config: MolmoVisionConfig):
        super().__init__()
        self.position_embedding = nn.Embedding(config.num_image_positions, config.hidden_size)


class MolmoEncoderLayer(CLIPEncoderLayer):
    def __init__(self, config: MolmoVisionConfig):
        super().__init__()
        self.self_attn = MOLMO_VISION_ATTENTION_CLASSES[config._attn_implementation](config)


class MolmoEncoder(CLIPEncoder):
    """
    Transformer encoder consisting of `config.num_hidden_layers` self attention layers. Each layer is a
    [`MolmoEncoderLayer`].

    Args:
        config: MolmoConfig
    """

    def __init__(self, config: MolmoVisionConfig):
        super().__init__()
        self.layers = nn.ModuleList([MolmoEncoderLayer(config) for _ in range(config.num_hidden_layers)])


class MolmoVisionTransformer(CLIPVisionTransformer):
    def __init__(self, config: MolmoVisionConfig):
        super().__init__()
        self.embeddings = MolmoVisionEmbeddings(config)
        self.post_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps, bias=True)
        self.encoder = MolmoEncoder(config)  # necessary because of renaming issue in modular


class MolmoImagePooling2d(nn.Module):  # It's an attention layer, so should be doable to take from CLIP?
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`:"
                f" {self.num_heads})."
            )
        self.scale = self.head_dim**-0.5
        self.dropout = config.attention_dropout

        self.q_proj = nn.Linear(
            2 * self.embed_dim,
            self.num_heads * self.head_dim,
            bias=True,
        )
        self.k_proj = nn.Linear(
            2 * self.embed_dim,
            config.image_num_key_value_heads * self.head_dim,
            bias=True,
        )
        self.v_proj = nn.Linear(
            2 * self.embed_dim,
            config.image_num_key_value_heads * self.head_dim,
            bias=True,
        )
        self.out_proj = nn.Linear(
            self.num_heads * self.head_dim,
            config.hidden_size,
            bias=True,
        )

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        causal_attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Input shape: Batch x Time x Channel"""

        bsz, tgt_len, embed_dim = hidden_states.size()

        # get query proj
        query_states = self.q_proj(hidden_states) * self.scale
        key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
        value_states = self._shape(self.v_proj(hidden_states), -1, bsz)

        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
        key_states = key_states.view(*proj_shape)
        value_states = value_states.view(*proj_shape)

        src_len = key_states.size(1)
        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))

        if attn_weights.size() != (bsz * self.num_heads, tgt_len, src_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, but is"
                f" {attn_weights.size()}"
            )

        # apply the causal_attention_mask first
        if causal_attention_mask is not None:
            if causal_attention_mask.size() != (bsz, 1, tgt_len, src_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is"
                    f" {causal_attention_mask.size()}"
                )
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + causal_attention_mask
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, tgt_len, src_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        if output_attentions:
            # this operation is a bit akward, but it's required to
            # make sure that attn_weights keeps its gradient.
            # In order to do so, attn_weights have to reshaped
            # twice and have to be reused in the following
            attn_weights_reshaped = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights_reshaped.view(bsz * self.num_heads, tgt_len, src_len)
        else:
            attn_weights_reshaped = None

        attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)

        attn_output = torch.bmm(attn_probs, value_states)

        if attn_output.size() != (bsz * self.num_heads, tgt_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, tgt_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(bsz, tgt_len, embed_dim)

        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights_reshaped


class MolmoVisionModel(CLIPVisionModel):
    config_class = MolmoVisionConfig  # needed because renames

    def __init__(self, config: MolmoVisionConfig):
        super().__init__()

        self.vision_model = MolmoVisionTransformer(config)
        self.image_pooling_2d = MolmoImagePooling2d(config)


class MolmoForConditionalGeneration(LlavaForConditionalGeneration):
    def __init__(self, config: MolmoConfig):
        super().__init__(config)
        self.multi_modal_projector = MolmoMultiModalProjector(config)

        self.language_model = MolmoForCausalLM._from_config(
            config.text_config, attn_implementation=config._attn_implementation
        )
        self.vision_tower = MolmoVisionModel._from_config(config.vision_config)
        self.post_init()


__all__ = [
    "MolmoConfig",
    "MolmoVisionConfig",
    "MolmoVisionEmbeddings",
    "MolmoVisionModel",
    "MolmoTextAttention",
    "MolmoVisionAttention",
    "MolmoImagePooling2d",
    "MolmoForConditionalGeneration",
]
