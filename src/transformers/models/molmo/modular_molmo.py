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


from torch import nn

from transformers.models.clip.configuration_clip import CLIPVisionConfig
from transformers.models.qwen2.configuration_qwen2 import Qwen2Config
from transformers.models.llava.configuration_llava import (
    LlavaConfig,
)

from ...utils import logging
from ..clip.modeling_clip import (
    CLIPEncoder,
    CLIPEncoderLayer,
    CLIPVisionEmbeddings,
    CLIPVisionModel,
    CLIPVisionTransformer,
    CLIPAttention,
    CLIPSdpaAttention,
    CLIPFlashAttention2,
)
from ..llava.modeling_llava import (
    LlavaForConditionalGeneration,
    LlavaMultiModalProjector,
)
from ..qwen2.modeling_qwen2 import (
    Qwen2DecoderLayer,
    Qwen2ForCausalLM,
    Qwen2MLP,
    Qwen2Model,
    Qwen2Attention,
    Qwen2FlashAttention2,
    Qwen2SdpaAttention,
)
from ...configuration_utils import PretrainedConfig
from ..auto import CONFIG_MAPPING
from typing import Optional

logger = logging.get_logger(__name__)


class MolmoVisionConfig(CLIPVisionConfig):
    model_type = "clip_vision_model"

    def __init__(
        self,
        **kwargs,
    ):
        super().__init__(**kwargs)



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
        vision_feature_select_strategy="default",
        vision_feature_layer=-2,
        image_seq_length=576,
        **kwargs,
    ):
        self.ignore_index = ignore_index
        self.image_token_index = image_token_index
        self.projector_hidden_act = projector_hidden_act
        self.image_seq_length = image_seq_length

        if vision_feature_select_strategy not in ["default", "full"]:
            raise ValueError(
                "vision_feature_select_strategy should be one of 'default', 'full'."
                f"Got: {vision_feature_select_strategy}"
            )

        self.vision_feature_select_strategy = vision_feature_select_strategy
        self.vision_feature_layer = vision_feature_layer

        if isinstance(vision_config, dict):
            vision_config["model_type"] = (
                vision_config["model_type"] if "model_type" in vision_config else "clip_vision_model"
            )
            vision_config = CONFIG_MAPPING[vision_config["model_type"]](**vision_config)
        elif vision_config is None:
            vision_config = {}
            logger.info("vision_config is None. Initializing MolmoVisionConfig with default values.")

        self.vision_config = MolmoVisionConfig(**vision_config)

        if isinstance(text_config, dict):
            text_config["model_type"] = text_config["model_type"] if "model_type" in text_config else "llama"
            text_config = CONFIG_MAPPING[text_config["model_type"]](**text_config)
        elif text_config is None:
            text_config = CONFIG_MAPPING["llama"]()

        self.text_config = text_config

        super().__init__(**kwargs)

# text modules inherited from Qwen2


class MolmoMLP(Qwen2MLP):
    def __init__(self, config):
        super().__init__()
        self.down_proj = nn.Linear(config.intermediate_size // 2, config.hidden_size, bias=False)



# We have different attention classes for the txt and the image components, they need to be propagated back correctly
class MolmoTextAttention(Qwen2Attention):
    pass

class MolmoTextSdpaAttention(Qwen2SdpaAttention):
    pass

class MolmoTextFlashAttention2(Qwen2FlashAttention2):
    pass

MOLMO_TEXT_ATTENTION_CLASSES = {
    "eager": MolmoTextAttention,
    "sdpa": MolmoTextSdpaAttention,
    "flash_attention_2": MolmoTextFlashAttention2
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

class MolmoVisionSdpaAttention(CLIPSdpaAttention):
    pass 

class MolmoVisionFlashAttention2(CLIPFlashAttention2):
    pass  

MOLMO_VISION_ATTENTION_CLASSES = {
    "eager": MolmoVisionAttention,
    "sdpa": MolmoVisionSdpaAttention,
    "flash_attention_2": MolmoVisionFlashAttention2
    }


# This needs to be in caps for some reason in the modular renaming
class MolmoVisionEmbeddings(CLIPVisionEmbeddings):
    def __init__(self, config: MolmoVisionConfig):
        super().__init__()
        self.position_embedding = nn.Embedding(config.num_image_positions, config.hidden_size)


# this class is not needed, just here while renaming issue persists
class MolmoEncoderLayer(CLIPEncoderLayer):
    def __init__(self, config: MolmoVisionConfig):
        super().__init__()

# this class is not needed, just here while renaming issue persists
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

class MolmoImagePooling2d(CLIPAttention): # It's an attention layer, so should be doable to take from CLIP?
    def __init__(self, config, is_vit_layer: Optional[bool] = True):
        super().__init__()

        self.q_proj = nn.Linear(2 * config.hidden_size,
            config.num_heads * config.head_dim,
            bias=True,
            device=config.init_device,
            )
        self.k_proj = nn.Linear(
            2 * config.hidden_size,
            config.num_key_value_heads * config.head_dim,
            bias=True,
            device=config.init_device,
            )
        self.v_proj = nn.Linear(
            2 * config.hidden_size,
            config.num_key_value_heads * config.head_dim,
            bias=True,
            device=config.init_device,
            )
        self.out_proj = nn.Linear(
            config.num_heads * config.head_dim,
            config.hidden_size,
            bias=True,
            device=config.init_device,
            )



class MolmoVisionModel(CLIPVisionModel):
    config_class = MolmoVisionConfig  # needed because renames

    def __init__(self, config: MolmoVisionConfig):
        super().__init__()

        self.vision_model = MolmoVisionTransformer(config)
        self.image_pooling_2d = MolmoImagePooling2d(config, is_vit_layer=False)


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
    "MolmoTextConfig",
    "MolmoVisionConfig",
    "MolmoVisionEmbeddings",
    "MolmoVisionModel",
    "MolmoTextAttention",
    "MolmoModel",
    "MolmoForConditionalGeneration",
]
