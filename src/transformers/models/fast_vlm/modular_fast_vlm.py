# Copyright 2025 The HuggingFace Team. All rights reserved.
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

import torch
from torch import nn

from ...activations import ACT2FN
from ...cache_utils import Cache
from ...configuration_utils import PreTrainedConfig
from ...modeling_outputs import BaseModelOutputWithPooling
from ...processing_utils import Unpack
from ...utils import TransformersKwargs, auto_docstring
from ...utils.generic import check_model_inputs
from ..auto import CONFIG_MAPPING
from ..llava.configuration_llava import LlavaConfig
from ..llava.modeling_llava import (
    LlavaCausalLMOutputWithPast,
    LlavaForConditionalGeneration,
    LlavaModel,
    LlavaModelOutputWithPast,
    LlavaMultiModalProjector,
    LlavaPreTrainedModel,
)


class FastVlmConfig(LlavaConfig):
    r"""
    This is the configuration class to store the configuration of a [`FastVlmForConditionalGeneration`]. It is used to instantiate a
    FastVLM model according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield the same configuration as the one of FastVLM-7B.

    e.g. [KamilaMila/FastVLM-7B](https://huggingface.co/KamilaMila/FastVLM-7B)

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        vision_config (`Union[AutoConfig, dict]`,  *optional*, defaults to `TimmWrapperConfig` for `fastvit_mci3`):
            The config object or dictionary of the vision backbone.
        text_config (`Union[AutoConfig, dict]`, *optional*, defaults to `Qwen2Config`):
            The config object or dictionary of the text backbone.
        image_token_id (`int`, *optional*, defaults to 151646):
            The image token index to encode the image prompt.
        projector_hidden_act (`str`, *optional*, defaults to `"gelu"`):
            The activation function used by the multimodal projector.
        vision_feature_select_strategy (`str`, *optional*, defaults to `"full"`):
            The feature selection strategy used to select the vision feature from the vision backbone.
            Only "full" supported.
        vision_feature_layer (`Union[int, list[int]]`, *optional*, defaults to -1):
            The index of the layer to select the vision feature. If multiple indices are provided,
            the vision feature of the corresponding indices will be concatenated to form the
            vision features. Only -1 supported.
        multimodal_projector_bias (`bool`, *optional*, defaults to `True`):
            Whether to use bias in the multimodal projector.

    Example:

    ```python
    >>> from transformers import FastVlmForConditionalGeneration, FastVlmConfig

    >>> # Initializing a FastVLM-7B style configuration
    >>> configuration = FastVlmConfig()

    >>> # Initializing a model from the FastVLM-7B style configuration
    >>> model = FastVlmForConditionalGeneration(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "fast_vlm"

    def __init__(
        self,
        vision_config=None,
        text_config=None,
        image_token_id=151646,
        projector_hidden_act="gelu",
        vision_feature_select_strategy="full",
        vision_feature_layer=-1,
        multimodal_projector_bias=True,
        **kwargs,
    ):
        self.image_token_id = image_token_id
        self.projector_hidden_act = projector_hidden_act

        if vision_feature_select_strategy != "full":
            raise ValueError(
                f"Unexpected select feature strategy: {vision_feature_select_strategy}. Only 'full' is supported in FastVLM."
            )

        if vision_feature_layer != -1:
            raise ValueError(
                f"Unexpected vision feature layer: {vision_feature_layer}. Only -1 is supported in FastVLM."
            )

        self.vision_feature_select_strategy = vision_feature_select_strategy
        self.vision_feature_layer = vision_feature_layer

        if isinstance(vision_config, dict):
            vision_config["model_type"] = vision_config.get("model_type", "timm_wrapper")
            vision_config = CONFIG_MAPPING[vision_config["model_type"]](**vision_config)
        elif vision_config is None:
            vision_config = CONFIG_MAPPING["timm_wrapper"](
                architecture="fastvit_mci3",
                do_pooling=True,
                global_pool="avg",
                hidden_size=3072,
                initializer_range=0.02,
                model_args={"inference_mode": True},
            )

        self.vision_config = vision_config

        if isinstance(text_config, dict):
            text_config["model_type"] = text_config.get("model_type", "qwen2")
            text_config = CONFIG_MAPPING[text_config["model_type"]](**text_config)
        elif text_config is None:
            text_config = CONFIG_MAPPING["qwen2"](
                hidden_size=3584,
                vocab_size=152128,
                intermediate_size=18944,
                num_attention_heads=28,
                num_key_value_heads=4,
                num_hidden_layers=28,
            )

        self.text_config = text_config
        self.multimodal_projector_bias = multimodal_projector_bias

        PreTrainedConfig.__init__(**kwargs)


class FastVlmMultiModalProjector(LlavaMultiModalProjector):
    def __init__(self, config: FastVlmConfig):
        nn.Module.__init__()
        self.linear_1 = nn.Linear(
            config.vision_config.hidden_size,
            config.text_config.hidden_size,
            bias=config.multimodal_projector_bias,
        )
        self.act = ACT2FN[config.projector_hidden_act]
        self.linear_2 = nn.Linear(
            config.text_config.hidden_size, config.text_config.hidden_size, bias=config.multimodal_projector_bias
        )


class FastVlmPreTrainedModel(LlavaPreTrainedModel):
    pass


class FastVlmModelOutputWithPast(LlavaModelOutputWithPast):
    pass


class FastVlmModel(LlavaModel):
    _checkpoint_conversion_mapping = {}

    def __init__(self, config: FastVlmConfig):
        super().__init__(config)

    @check_model_inputs(tie_last_hidden_states=False)
    @auto_docstring(
        custom_intro="Obtains image last hidden states from the vision tower and apply multimodal projection."
    )
    def get_image_features(
        self,
        pixel_values: torch.FloatTensor,
        vision_feature_layer: int | list[int] | None = None,
        vision_feature_select_strategy: str | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPooling:
        r"""
        pixel_values (`torch.FloatTensor]` of shape `(batch_size, channels, height, width)`):
            The tensors corresponding to the input images.
        vision_feature_layer (`Union[int, list[int]]`, *optional*):
            The index/indices of the layer to select the vision feature. Only -1 supported.
        vision_feature_select_strategy (`str`, *optional*):
            The feature selection strategy used to select the vision feature from the vision backbone.
            Only "full" supported.
        """
        kwargs = {k: v for k, v in kwargs.items() if v is not None}
        image_outputs = self.vision_tower(pixel_values, return_dict=True, **kwargs)

        # since the vision tower is hybrid in FastVLM, its output needs to be handled differently from Llava
        selected_image_feature = image_outputs.last_hidden_state
        selected_image_feature = selected_image_feature.flatten(2).permute(0, 2, 1)
        image_features = self.multi_modal_projector(selected_image_feature)
        image_outputs.pooler_output = list(image_features)

        return image_outputs

    @check_model_inputs(tie_last_hidden_states=False)
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        pixel_values: torch.FloatTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        vision_feature_layer: int | list[int] | None = None,
        vision_feature_select_strategy: str | None = None,
        cache_position: torch.LongTensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> FastVlmModelOutputWithPast:
        r"""
        vision_feature_layer (`Union[int, list[int], NoneType]`, *optional*):
            The index of the layer to select the vision feature. If multiple indices are provided, the vision feature of the
            corresponding indices will be concatenated to form the vision features. Only -1 supported.
        vision_feature_select_strategy (`str`, *optional*):
            The feature selection strategy used to select the vision feature from the vision backbone. Only "full" supported.
        """
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

        if pixel_values is not None:
            image_features = self.get_image_features(
                pixel_values=pixel_values,
                vision_feature_layer=vision_feature_layer,
                vision_feature_select_strategy=vision_feature_select_strategy,
                return_dict=True,
            ).pooler_output
            image_features = torch.cat(image_features, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
            special_image_mask = self.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, image_features=image_features
            )
            inputs_embeds = inputs_embeds.masked_scatter(special_image_mask, image_features)

        outputs = self.language_model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            **kwargs,
        )

        return FastVlmModelOutputWithPast(
            last_hidden_state=outputs.last_hidden_state,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            image_hidden_states=image_features if pixel_values is not None else None,
        )


class FastVlmCausalLMOutputWithPast(LlavaCausalLMOutputWithPast):
    pass


@auto_docstring(
    custom_intro="""
    The FastVlm model which consists of a vision backbone and a language model.
    """
)
class FastVlmForConditionalGeneration(LlavaForConditionalGeneration):
    _checkpoint_conversion_mapping = {}

    @check_model_inputs(tie_last_hidden_states=False)
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        pixel_values: torch.FloatTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        vision_feature_layer: int | list[int] | None = None,
        vision_feature_select_strategy: str | None = None,
        labels: torch.LongTensor | None = None,
        cache_position: torch.LongTensor | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        **kwargs: Unpack[TransformersKwargs],
    ) -> FastVlmCausalLMOutputWithPast:
        r"""
        vision_feature_layer (`Union[int, list[int], NoneType]`, *optional*):
            The index of the layer to select the vision feature. If multiple indices are provided, the vision feature of the
            corresponding indices will be concatenated to form the vision features. Only -1 supported.
        vision_feature_select_strategy (`str`, *optional*):
            The feature selection strategy used to select the vision feature from the vision backbone. Only "full" supported.
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Example:

        ```python
        >>> from PIL import Image
        >>> import httpx
        >>> from io import BytesIO
        >>> from transformers import AutoProcessor, AutoModelForImageTextToText
        >>> import torch

        >>> device = "cuda" if torch.cuda.is_available() else "cpu"

        >>> model = AutoModelForImageTextToText.from_pretrained("KamilaMila/FastVLM-0.5B").to(device)
        >>> processor = AutoProcessor.from_pretrained("KamilaMila/FastVLM-0.5B")

        >>> conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What are these?"},
                        {"type": "image"}
                    ]
                }
            ]

        >>> prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
        >>> url = "https://www.ilankelman.org/stopsigns/australia.jpg"
        >>> with httpx.stream("GET", url) as response:
        ...     image = Image.open(BytesIO(response.read()))

        >>> inputs = processor(images=image, text=prompt, return_tensors="pt").to(device)

        >>> # Generate
        >>> generated_ids = model.generate(**inputs, max_new_tokens=15)
        >>> print(processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0])
        system\n You are a helpful assistant.\n user\n What are these?\n assistant\n The image depicts a traditional Chinese street...
        ```"""
        outputs = self.model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            vision_feature_layer=vision_feature_layer,
            vision_feature_select_strategy=vision_feature_select_strategy,
            cache_position=cache_position,
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

        return FastVlmCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            image_hidden_states=outputs.image_hidden_states,
        )


__all__ = ["FastVlmForConditionalGeneration", "FastVlmModel", "FastVlmPreTrainedModel", "FastVlmConfig"]
