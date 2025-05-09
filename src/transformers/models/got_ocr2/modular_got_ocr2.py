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


from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.utils.checkpoint

from transformers.models.llava.modeling_llava import (
    KwargsForCausalLM,
    LlavaCausalLMOutputWithPast,
    LlavaForConditionalGeneration,
    LlavaModel,
    LlavaModelOutputWithPast,
    LlavaPreTrainedModel,
)
from transformers.models.sam.modeling_sam import SamMLPBlock, SamVisionAttention, SamVisionEncoder, SamVisionLayer

from ...configuration_utils import PretrainedConfig
from ...modeling_flash_attention_utils import FlashAttentionKwargs
from ...processing_utils import Unpack
from ...utils import auto_docstring, can_return_tuple, is_vision_available, logging
from ..auto import CONFIG_MAPPING, AutoConfig


if is_vision_available():
    pass

logger = logging.get_logger(__name__)


class GotOcr2VisionConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`GotOcr2VisionModel`]. It is used to instantiate a GOT_OCR2
    vision encoder according to the specified arguments, defining the model architecture. Instantiating a configuration
    defaults will yield a similar configuration to that of the SAM ViT-h
    [facebook/sam-vit-huge](https://huggingface.co/facebook/sam-vit-huge) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        hidden_size (`int`, *optional*, defaults to 768):
            Dimensionality of the encoder layers and the pooler layer.
        output_channels (`int`, *optional*, defaults to 256):
            Dimensionality of the output channels in the Patch Encoder.
        num_hidden_layers (`int`, *optional*, defaults to 12):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 12):
            Number of attention heads for each attention layer in the Transformer encoder.
        num_channels (`int`, *optional*, defaults to 3):
            Number of channels in the input image.
        image_size (`int`, *optional*, defaults to 1024):
            Expected resolution. Target size of the resized input image.
        patch_size (`int`, *optional*, defaults to 16):
            Size of the patches to be extracted from the input image.
        hidden_act (`str`, *optional*, defaults to `"gelu"`):
            The non-linear activation function (function or string)
        layer_norm_eps (`float`, *optional*, defaults to 1e-06):
            The epsilon used by the layer normalization layers.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        initializer_range (`float`, *optional*, defaults to 1e-10):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        qkv_bias (`bool`, *optional*, defaults to `True`):
            Whether to add a bias to query, key, value projections.
        use_abs_pos (`bool`, *optional*, defaults to `True`):
            Whether to use absolute position embedding.
        use_rel_pos (`bool`, *optional*, defaults to `True`):
            Whether to use relative position embedding.
        window_size (`int`, *optional*, defaults to 14):
            Window size for relative position.
        global_attn_indexes (`List[int]`, *optional*, defaults to `[2, 5, 8, 11]`):
            The indexes of the global attention layers.
        mlp_dim (`int`, *optional*, defaults to 3072):
            The dimensionality of the MLP layer in the Transformer encoder.
    """

    base_config_key = "vision_config"

    def __init__(
        self,
        hidden_size=768,
        output_channels=256,
        num_hidden_layers=12,
        num_attention_heads=12,
        num_channels=3,
        image_size=1024,
        patch_size=16,
        hidden_act="gelu",
        layer_norm_eps=1e-06,
        attention_dropout=0.0,
        initializer_range=1e-10,
        qkv_bias=True,
        use_abs_pos=True,
        use_rel_pos=True,
        window_size=14,
        global_attn_indexes=[2, 5, 8, 11],
        mlp_dim=3072,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.hidden_size = hidden_size
        self.output_channels = output_channels
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_channels = num_channels
        self.image_size = image_size
        self.patch_size = patch_size
        self.hidden_act = hidden_act
        self.layer_norm_eps = layer_norm_eps
        self.attention_dropout = attention_dropout
        self.initializer_range = initializer_range
        self.qkv_bias = qkv_bias
        self.use_abs_pos = use_abs_pos
        self.use_rel_pos = use_rel_pos
        self.window_size = window_size
        self.global_attn_indexes = global_attn_indexes
        self.mlp_dim = mlp_dim


class GotOcr2Config(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`GotOcr2ForConditionalGeneration`]. It is used to instantiate a
    GotOcr2 model according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of GOT-OCR-2.0.

    e.g [stepfun-ai/GOT-OCR-2.0-hf](https://huggingface.co/stepfun-ai/GOT-OCR-2.0-hf)

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Args:
        vision_config (`Union[AutoConfig, dict]`,  *optional*, defaults to `CLIPVisionConfig`):
            The config object or dictionary of the vision backbone.
        text_config (`Union[AutoConfig, dict]`, *optional*, defaults to `LlamaConfig`):
            The config object or dictionary of the text backbone.
        image_token_index (`int`, *optional*, defaults to 151859):
            The image token index to encode the image prompt.
        image_seq_length (`int`, *optional*, defaults to 576):
            Sequence length of one image embedding.
        pad_token_id (`int`, *optional*, defaults to -1):
            Padding token id.

    ```python
    >>> from transformers import GotOcr2ForConditionalGeneration, GotOcr2Config

    >>> # Initializing a GotOcr2 style configuration
    >>> configuration = GotOcr2Config()

    >>> # Initializing a model from the Qwen2-VL-7B style configuration
    >>> model = GotOcr2ForConditionalGeneration(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "got_ocr2"
    attribute_map = {
        "image_token_id": "image_token_index",
    }
    sub_configs = {"text_config": AutoConfig, "vision_config": GotOcr2VisionConfig}

    def __init__(
        self,
        vision_config=None,
        text_config=None,
        image_token_index=151859,
        image_seq_length=576,
        pad_token_id=-1,
        **kwargs,
    ):
        self.image_token_index = image_token_index
        self.image_seq_length = image_seq_length
        self.pad_token_id = pad_token_id

        if vision_config is None:
            self.vision_config = GotOcr2VisionConfig()
        elif isinstance(vision_config, dict):
            self.vision_config = GotOcr2VisionConfig(**vision_config)
        elif isinstance(vision_config, GotOcr2VisionConfig):
            self.vision_config = vision_config

        if isinstance(text_config, dict):
            text_config["model_type"] = text_config["model_type"] if "model_type" in text_config else "qwen2"
            text_config = CONFIG_MAPPING[text_config["model_type"]](**text_config)
        elif text_config is None:
            text_config = CONFIG_MAPPING["qwen2"](
                vocab_size=151860,
                hidden_size=1024,
                intermediate_size=2816,
                num_hidden_layers=24,
                num_attention_heads=16,
                num_key_value_heads=16,
                hidden_act="silu",
                max_position_embeddings=32768,
                initializer_range=0.02,
                rms_norm_eps=1e-6,
                use_cache=True,
                tie_word_embeddings=True,
                rope_theta=1000000.0,
                rope_scaling=None,
                use_sliding_window=False,
                sliding_window=4096,
                max_window_layers=21,
                attention_dropout=0.0,
            )

        self.text_config = text_config

        super().__init__(**kwargs)


class GotOcr2MLPBlock(SamMLPBlock):
    pass


class GotOcr2VisionAttention(SamVisionAttention):
    pass


class GotOcr2VisionLayer(SamVisionLayer):
    def __init__(self, config, window_size):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.attn = GotOcr2VisionAttention(config, window_size)
        self.layer_norm2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.mlp = GotOcr2MLPBlock(config)
        self.window_size = window_size


class GotOcr2VisionEncoder(SamVisionEncoder):
    pass


class GotOcr2MultiModalProjector(nn.Module):
    def __init__(self, config: GotOcr2Config):
        super().__init__()
        vision_output_channels = config.vision_config.output_channels
        language_hidden_size = config.text_config.hidden_size
        self.conv_upsampler1 = nn.Conv2d(
            vision_output_channels, vision_output_channels * 2, kernel_size=3, stride=2, padding=1, bias=False
        )
        self.conv_upsampler2 = nn.Conv2d(
            vision_output_channels * 2, language_hidden_size, kernel_size=3, stride=2, padding=1, bias=False
        )
        self.multimodal_projector = nn.Linear(language_hidden_size, language_hidden_size)

    def forward(self, vision_embeddings: torch.Tensor) -> torch.Tensor:
        hidden_state = self.conv_upsampler1(vision_embeddings)
        hidden_state = self.conv_upsampler2(hidden_state)
        hidden_state = hidden_state.flatten(2).permute(0, 2, 1)
        hidden_state = self.multimodal_projector(hidden_state)
        return hidden_state


class GotOcr2CausalLMOutputWithPast(LlavaCausalLMOutputWithPast):
    pass


class GotOcr2ModelOutputWithPast(LlavaModelOutputWithPast):
    pass


class GotOcr2PreTrainedModel(LlavaPreTrainedModel):
    def _init_weights(self, module):
        std = getattr(self.config, "initializer_range", self.config.get_text_config().initializer_range)

        if isinstance(module, (nn.Linear, nn.Conv2d)):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, (nn.LayerNorm, GotOcr2LayerNorm)):  # noqa: F821
            module.weight.data.fill_(1.0)
            module.bias.data.zero_()
        elif isinstance(module, GotOcr2VisionAttention):
            if module.use_rel_pos:
                module.rel_pos_h.data.zero_()
                module.rel_pos_w.data.zero_()
        elif isinstance(module, GotOcr2VisionEncoder):
            if module.pos_embed is not None:
                module.pos_embed.data.zero_()


class GotOcr2Model(LlavaModel):
    def __init__(self, config: GotOcr2Config):
        super().__init__(config)
        self.vision_tower = GotOcr2VisionEncoder(config.vision_config)

    def get_image_features(
        self,
        pixel_values: torch.FloatTensor,
    ):
        """
        Obtains image last hidden states from the vision tower and apply multimodal projection.

        Args:
            pixel_values (`torch.FloatTensor]` of shape `(batch_size, channels, height, width)`)
        Returns:
            image_features (`torch.Tensor`): Image feature tensor of shape `(num_images, image_length, embed_dim)`).
        """
        image_outputs = self.vision_tower(pixel_values).last_hidden_state
        return self.multi_modal_projector(image_outputs)

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        pixel_values: torch.FloatTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Union[Tuple, GotOcr2ModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if pixel_values is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both pixel_values and inputs_embeds at the same time, and must specify either one"
            )

        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

        if pixel_values is not None:
            image_features = self.get_image_features(pixel_values=pixel_values.to(inputs_embeds.dtype))
            n_image_tokens = (input_ids == self.config.image_token_id).sum()
            n_image_features = image_features.shape[0] * image_features.shape[1]
            if n_image_tokens != n_image_features:
                raise ValueError(
                    f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
                )
            special_image_mask = (input_ids == self.config.image_token_id).unsqueeze(-1)
            special_image_mask = special_image_mask.expand_as(inputs_embeds).to(inputs_embeds.device)
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
            return_dict=True,
            cache_position=cache_position,
            **kwargs,
        )

        return GotOcr2ModelOutputWithPast(
            last_hidden_state=outputs.last_hidden_state,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            image_hidden_states=image_features if pixel_values is not None else None,
        )


class GotOcr2ForConditionalGeneration(LlavaForConditionalGeneration):
    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        pixel_values: torch.FloatTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs: Unpack[KwargsForCausalLM],
    ) -> Union[Tuple, GotOcr2CausalLMOutputWithPast]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Example:

        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import AutoProcessor, GotOcr2ForConditionalGeneration, TextStreamer

        >>> model = GotOcr2ForConditionalGeneration.from_pretrained("stepfun-ai/GOT-OCR-2.0-hf").to("cuda")
        >>> processor = AutoProcessor.from_pretrained("stepfun-ai/GOT-OCR-2.0-hf")

        >>> url = "https://huggingface.co/datasets/hf-internal-testing/fixtures_got_ocr/resolve/main/multi_box.png"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> inputs = processor(image, return_tensors="pt", color="green").to("cuda")

        >>> # Generate
        >>> streamer = TextStreamer(processor.tokenizer, skip_prompt=True, skip_special_tokens=True)
        >>> generate_ids = model.generate(
        ...     **inputs,
        ...     do_sample=False,
        ...     tokenizer = processor.tokenizer,
        ...     stop_strings='<|im_end|>',
        ...     streamer=streamer,
        ...     max_new_tokens=4096,
        ... )
        "You should keep in mind what features from the module should be used, especially
        when you're planning to sell a template."
        ```"""
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
            cache_position=cache_position,
            logits_to_keep=logits_to_keep,
            **kwargs,
        )

        hidden_states = outputs[0]
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(
                logits=logits, labels=labels, vocab_size=self.config.text_config.vocab_size, **kwargs
            )

        return GotOcr2CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            image_hidden_states=outputs.image_hidden_states,
        )


__all__ = [
    "GotOcr2VisionConfig",
    "GotOcr2Config",
    "GotOcr2PreTrainedModel",
    "GotOcr2Model",
    "GotOcr2ForConditionalGeneration",
]
