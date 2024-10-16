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


from typing import Optional, Tuple, Union, List, Dict

import torch
from torch import nn
from ...modeling_outputs import BaseModelOutputWithPooling, BaseModelOutput
from ...modeling_rope_utils import rope_config_validation
from ..clip.configuration_clip import CLIPVisionConfig
from ..qwen2.configuration_qwen2 import Qwen2Config
from ...configuration_utils import PretrainedConfig
from ...utils import logging
from ..auto import CONFIG_MAPPING
from ..clip.modeling_clip import (
    CLIPMLP,
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
    LlavaCausalLMOutputWithPast
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
import math

from ...utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_flash_attn_2_available,
    is_flash_attn_greater_or_equal_2_10,
    logging,
    replace_return_docstrings,
)

logger = logging.get_logger(__name__)


class MolmoVisionConfig(CLIPVisionConfig):
    def __init__(
        self,
        hidden_size=1024,
        num_attention_heads=16,
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
        residual_dropout=0.0,
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
        self.residual_dropout = residual_dropout

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
        hidden_act="swiglu",
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
        self.max_position_embeddings = max_position_embeddings
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
        vision_feature_select_strategy="full",
        vision_feature_layers=[-2, -9],
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.ignore_index = ignore_index
        self.image_token_index = image_token_index
        self.projector_hidden_act = projector_hidden_act
        self.image_seq_length = image_seq_length
        self.vision_feature_select_strategy = vision_feature_select_strategy
        self.vision_feature_layers = vision_feature_layers
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



# swiglu activation 

class MolmoSwiGLU(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, gate = x.chunk(2, dim=-1)
        return nn.functional.silu(gate) * x
    
# text modules inherited from Qwen2


class MolmoMLP(CLIPMLP):
    def __init__(self, config):
        super().__init__()
        self.activation_fn = MolmoSwiGLU()
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.fc2 = nn.Linear(config.intermediate_size // 2, config.hidden_size, bias=False)

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
        self.patch_embedding = nn.Linear(
            self.patch_size ** 2 * 3,
            self.embed_dim,
            bias=False,
            )
class MolmoVisionMLP(CLIPMLP):
    pass


class MolmoEncoderLayer(CLIPEncoderLayer):
    def __init__(self, config: MolmoVisionConfig):
        super().__init__()
        self.self_attn = MOLMO_VISION_ATTENTION_CLASSES[config._attn_implementation](config)
        self.mlp = MolmoVisionMLP(config)



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

# TODO add pooling call + embed here
class MolmoVisionTransformer(CLIPVisionTransformer):
    def __init__(self, config: MolmoVisionConfig):
        super().__init__()
        self.embeddings = MolmoVisionEmbeddings(config)
        self.encoder = MolmoEncoder(config)  # necessary because of renaming issue in modular
        del self.post_layernorm


    def forward(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        interpolate_pos_encoding: Optional[bool] = False,
    ) -> Union[Tuple, BaseModelOutput]:
        r"""
        Returns:

        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        hidden_states = self.embeddings(pixel_values, interpolate_pos_encoding=interpolate_pos_encoding)
        hidden_states = self.pre_layrnorm(hidden_states)

        encoder_outputs = self.encoder(
            inputs_embeds=hidden_states,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        last_hidden_state = encoder_outputs[0]
        # TODO add pooling operations here! 

        if not return_dict:
            return (last_hidden_state) + encoder_outputs[1:]

        return BaseModelOutput(
            last_hidden_state=last_hidden_state,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )

class MolmoImagePooling2d(nn.Module):  # It's an attention layer, so should be doable to take from CLIP?
    def __init__(self, config: MolmoVisionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.image_num_key_value_heads = config.image_num_key_value_heads
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
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim,
            config.hidden_size,
            bias=True,
        )
        self.residual_dropout = nn.Dropout(config.residual_dropout)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
   
    def _split_heads(self, hidden_states, num_heads) -> torch.Tensor:
        return hidden_states.reshape(hidden_states.shape[:2] + (num_heads, self.head_dim))

    def _merge_heads(self, hidden_states) -> torch.Tensor:
        return hidden_states.reshape(hidden_states.shape[:2] + (self.embed_dim,))

    def forward(self, inputs_q: torch.Tensor, inputs_kv: Optional[torch.Tensor] = None) -> torch.Tensor:
        if inputs_kv is not None:
            inputs_k = inputs_kv
            inputs_v = inputs_kv
        else:
            inputs_k = inputs_q
            inputs_v = inputs_q

        queries, keys, values = self.q_proj(inputs_q), self.k_proj(inputs_k), self.v_proj(inputs_v)

        queries = self._split_heads(queries, self.num_heads)
        keys = self._split_heads(keys, self.image_num_key_value_heads)
        values = self._split_heads(values, self.image_num_key_value_heads)

        # TODO do we need this to be here?
        if self.num_heads != self.image_num_key_value_heads:
            keys = keys.repeat_interleave(self.num_key_value_groups, dim=2, output_size=self.num_heads)
            values = values.repeat_interleave(self.num_key_value_groups, dim=2, output_size=self.num_heads)

        original_queries_dtype = queries.dtype

        #if self.config.float32_attention:
        # Seems that the default is float32
        queries = queries.to(torch.float)
        keys = keys.to(torch.float)

        if self.config._attn_implementation == "eager":
            attn_weights = torch.einsum("...qhd,...khd->...hqk", queries / math.sqrt(queries.size(-1)), keys)
            attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(queries.dtype)
            if self.attention_dropout is not None:
                attn_weights = self.attention_dropout(attn_weights)
            # TODO remove einsum!
            attn_output = torch.einsum("...hqk,...khd->...qhd", attn_weights.to(values.dtype), values)

        elif self.config._attn_implementation == "sdpa":
            attn_output = nn.functional.scaled_dot_product_attention(
                queries.transpose(1, 2).contiguous(),
                keys.transpose(1, 2).contiguous(),
                values.transpose(1, 2).contiguous(),
                is_causal=False,
                dropout_p=self.config.vision_backbone.attention_dropout
            ).transpose(1, 2)
        else:
            raise NotImplementedError(f"{self.config._attn_implementation} is not supported.")
        attn_output = attn_output.to(original_queries_dtype)
        attn_output = self._merge_heads(attn_output)
        attn_output = self.o_proj(attn_output)
        attn_output = self.residual_dropout(attn_output)

        return attn_output

class MolmoVisionModel(CLIPVisionModel):
    config_class = MolmoVisionConfig  # needed because renames

    def __init__(self, config: MolmoVisionConfig):
        super().__init__()
        self.image_hidden_size = 2 * config.hidden_size

        self.vision_model = MolmoVisionTransformer(config)
        self.image_pooling_2d = MolmoImagePooling2d(config)
        self.pad_embed = nn.Parameter(torch.zeros((2, self.image_hidden_size)))

class MolmoCausalLMOutputWithPast(LlavaCausalLMOutputWithPast):
    pass

class MolmoForConditionalGeneration(LlavaForConditionalGeneration):
    def __init__(self, config: MolmoConfig):
        super().__init__(config)
        self.multi_modal_projector = MolmoMultiModalProjector(config)

        self.language_model = MolmoForCausalLM._from_config(
            config.text_config, attn_implementation=config._attn_implementation
        )
        self.vision_tower = MolmoVisionModel._from_config(config.vision_config)
        self.post_init()

    def get_image_features(
        self, pixel_values: torch.FloatTensor, vision_feature_layers: List, vision_feature_select_strategy: str
    ):
        image_outputs = self.vision_tower(pixel_values, output_hidden_states=True)
        # this is not memory efficient at all (output_hidden_states=True) will save all the hidden stated.
        features = []
        image_features = image_outputs.hidden_states
        for layer in vision_feature_layers:
            features.append(image_features[layer])
        image_features = torch.cat(features, dim=-1)
        # TODO add pad embed, dropout, pooling, reshaping, then multimodal projection
        return image_features
    
    # redefinition of forward to include the vision feature selection
    # TODO (modular): how do we change this kind of attribute within a method
    # without changing the whole method? 
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        pixel_values: torch.FloatTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        vision_feature_layers: Optional[int] = None,
        vision_feature_select_strategy: Optional[str] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        num_logits_to_keep: int = 0,
    ) -> Union[Tuple, MolmoCausalLMOutputWithPast]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

            num_logits_to_keep (`int`, *optional*):
                Calculate logits for the last `num_logits_to_keep` tokens. If `0`, calculate logits for all
                `input_ids` (special case). Only last token logits are needed for generation, and calculating them only for that
                token can save memory, which becomes pretty significant for long sequences or large vocabulary size.


        Returns:

        Example:

        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import AutoProcessor, MolmoForConditionalGeneration

        >>> model = MolmoForConditionalGeneration.from_pretrained("molmo-hf/molmo-1.5-7b-hf")
        >>> processor = AutoProcessor.from_pretrained("molmo-hf/molmo-1.5-7b-hf")

        >>> prompt = "USER: <image>\nWhat's the content of the image? ASSISTANT:"
        >>> url = "https://www.ilankelman.org/stopsigns/australia.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> inputs = processor(images=image, text=prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(**inputs, max_new_tokens=15)
        >>> processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "USER:  \nWhat's the content of the image? ASSISTANT: The image features a busy city street with a stop sign prominently displayed"
        ```"""

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        vision_feature_layers = (
            vision_feature_layers if vision_feature_layers is not None else self.config.vision_feature_layers
        )
        vision_feature_select_strategy = (
            vision_feature_select_strategy
            if vision_feature_select_strategy is not None
            else self.config.vision_feature_select_strategy
        )

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if pixel_values is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both pixel_values and inputs_embeds at the same time, and must specify either one"
            )

        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

        if pixel_values is not None:
            image_features = self.get_image_features(
                pixel_values=pixel_values,
                vision_feature_layers=vision_feature_layers,
                vision_feature_select_strategy=vision_feature_select_strategy,
            )

            special_image_mask = (
                (input_ids == self.config.image_token_index)
                .unsqueeze(-1)
                .expand_as(inputs_embeds)
                .to(inputs_embeds.device)
            )
            image_features = image_features.to(inputs_embeds.device, inputs_embeds.dtype)
            inputs_embeds = inputs_embeds.masked_scatter(special_image_mask, image_features)

        outputs = self.language_model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            num_logits_to_keep=num_logits_to_keep,
        )

        logits = outputs[0]

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            if attention_mask is not None:
                shift_attention_mask = attention_mask[..., 1:]
                shift_logits = logits[..., :-1, :][shift_attention_mask.to(logits.device) != 0].contiguous()
                shift_labels = labels[..., 1:][shift_attention_mask.to(labels.device) != 0].contiguous()
            else:
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1).to(shift_logits.device)
            )

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return MolmoCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            image_hidden_states=image_features if pixel_values is not None else None,
        )
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
