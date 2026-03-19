# Copyright 2026 NAVER Corp. and the HuggingFace Inc. team. All rights reserved.
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
"""PyTorch HyperCLOVAX Vision model."""

import torch
from torch import nn

from ...cache_utils import Cache
from ...generation import GenerationMixin
from ...modeling_layers import GenericForSequenceClassification
from ...modeling_outputs import BaseModelOutputWithPooling, CausalLMOutputWithPast
from ...processing_utils import Unpack
from ...utils import TransformersKwargs, auto_docstring, can_return_tuple, logging
from ..auto.modeling_auto import AutoModel
from ..granite.modeling_granite import GraniteAttention, GraniteDecoderLayer, GraniteForCausalLM, GraniteModel
from ..llama.modeling_llama import LlamaPreTrainedModel
from ..video_llama_3.modeling_video_llama_3 import VideoLlama3Model
from .configuration_hyperclovax_vision import HCXVisionConfig, HyperClovaXConfig


logger = logging.get_logger(__name__)


class HyperClovaXAttention(GraniteAttention):
    pass


class HyperClovaXDecoderLayer(GraniteDecoderLayer):
    pass


@auto_docstring
class HCXVisionPreTrainedModel(LlamaPreTrainedModel):
    config: HCXVisionConfig
    _no_split_modules = ["HyperClovaXDecoderLayer"]
    input_modalities = ("image", "video", "text")
    _can_record_outputs = {"hidden_states": HyperClovaXDecoderLayer, "attentions": HyperClovaXAttention}


@auto_docstring
class HyperClovaXModel(GraniteModel):
    config_class = HyperClovaXConfig
    input_modalities = ("text",)


@auto_docstring
class HyperClovaXForCausalLM(GraniteForCausalLM):
    accepts_loss_kwargs = False
    config_class = HyperClovaXConfig
    input_modalities = ("text",)


@auto_docstring
class HCXVisionModel(VideoLlama3Model):
    config: HCXVisionConfig
    _no_split_modules = ["HyperClovaXDecoderLayer"]
    _can_record_outputs = {"hidden_states": HyperClovaXDecoderLayer, "attentions": HyperClovaXAttention}

    def __init__(self, config: HCXVisionConfig):
        super().__init__(config)

        self.vision_model = AutoModel.from_config(config.vision_config)

        self.projector = nn.Linear(config.vision_output_size, config.text_hidden_size)

        self.language_model = AutoModel.from_config(config.text_config)
        self.post_init()

    @can_return_tuple
    @auto_docstring
    def get_image_features(
        self,
        pixel_values: torch.FloatTensor,
        image_grid_thw: torch.LongTensor,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | BaseModelOutputWithPooling:
        r"""
        pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)`):
            The tensors corresponding to the input images.
        image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
            The temporal, height and width of feature shape of each image in LLM.
        """
        vision_outputs = self.vision_model(pixel_values, grid_thw=image_grid_thw, **kwargs)
        vision_outputs.pooler_output = self.projector(vision_outputs.pooler_output)

        return vision_outputs


@auto_docstring
class HCXVisionForConditionalGeneration(HCXVisionPreTrainedModel, GenerationMixin):
    accepts_loss_kwargs = False
    _tied_weights_keys = {"lm_head.weight": "model.language_model.embed_tokens.weight"}

    def __init__(self, config: HCXVisionConfig):
        super().__init__(config)
        self.model = HCXVisionModel(config)
        self.vocab_size = config.text_config.vocab_size
        self.lm_head = nn.Linear(config.text_config.hidden_size, config.text_config.vocab_size, bias=False)
        self.post_init()

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.model.set_input_embeddings(value)

    @auto_docstring
    def get_video_features(
        self,
        pixel_values_videos: torch.FloatTensor,
        video_grid_thw: torch.LongTensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | BaseModelOutputWithPooling:
        r"""
        pixel_values_videos (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)`):
            The tensors corresponding to the input videos.
        video_grid_thw (`torch.LongTensor` of shape `(num_videos, 3)`, *optional*):
            The temporal, height and width of feature shape of each video in LLM.
        """
        return self.model.get_video_features(
            pixel_values_videos=pixel_values_videos, video_grid_thw=video_grid_thw, **kwargs
        )

    @auto_docstring
    def get_image_features(
        self,
        pixel_values: torch.FloatTensor,
        image_grid_thw: torch.LongTensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | BaseModelOutputWithPooling:
        r"""
        pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)`):
            The tensors corresponding to the input images.
        image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
            The temporal, height and width of feature shape of each image in LLM.
        """
        return self.model.get_image_features(pixel_values=pixel_values, image_grid_thw=image_grid_thw, **kwargs)

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        pixel_values: torch.FloatTensor | None = None,
        pixel_values_videos: torch.FloatTensor | None = None,
        image_grid_thw: torch.LongTensor | None = None,
        video_grid_thw: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        cache_position: torch.LongTensor | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | CausalLMOutputWithPast:
        r"""
        pixel_values (`torch.FloatTensor`, *optional*):
            Pixel values of input images after preprocessing.
        pixel_values_videos (`torch.FloatTensor`, *optional*):
            Pixel values of input videos, same format as ``pixel_values``.
        image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
            ``[temporal, height, width]`` grid counts per image.
        video_grid_thw (`torch.LongTensor` of shape `(num_videos, 3)`, *optional*):
            ``[temporal, height, width]`` grid counts per video.
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss.
        logits_to_keep (`int` or `torch.Tensor`, *optional*, defaults to 0):
            If an ``int``, compute logits for the last ``logits_to_keep`` tokens.

        Example:

        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import AutoProcessor, HCXVisionForConditionalGeneration

        >>> model = HCXVisionForConditionalGeneration.from_pretrained(
        ...     "naver-hyperclovax/HyperCLOVAX-SEED-Think-32B",
        ...     torch_dtype="auto",
        ...     device_map="auto",
        ... )
        >>> processor = AutoProcessor.from_pretrained("naver-hyperclovax/HyperCLOVAX-SEED-Think-32B")

        >>> messages = [
        ...     {"role": "user", "content": [
        ...         {"type": "image", "url": "http://images.cocodataset.org/val2017/000000039769.jpg"},
        ...         {"type": "text", "text": "Describe this image in detail."},
        ...     ]}
        ... ]
        >>> inputs = processor.apply_chat_template(
        ...     messages, tokenize=True, return_dict=True, return_tensors="pt"
        ... ).to(model.device)
        >>> output = model.generate(**inputs, max_new_tokens=200)
        >>> processor.decode(output[0], skip_special_tokens=True)
        ```
        """
        outputs = self.model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )
        hidden_states = outputs[0]
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :]) * getattr(
            self.config.text_config, "logits_scaling", 1
        )

        loss = None
        if labels is not None:
            loss = self.loss_function(logits, labels, self.vocab_size)

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        cache_position=None,
        position_ids=None,
        use_cache=True,
        pixel_values=None,
        pixel_values_videos=None,
        image_grid_thw=None,
        video_grid_thw=None,
        is_first_iteration=False,
        **kwargs,
    ):
        model_inputs = super().prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            position_ids=position_ids,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            use_cache=use_cache,
            is_first_iteration=is_first_iteration,
            **kwargs,
        )

        if not is_first_iteration and use_cache:
            model_inputs["pixel_values"] = None
            model_inputs["pixel_values_videos"] = None

        return model_inputs


class HCXVisionForSequenceClassification(HCXVisionPreTrainedModel, GenericForSequenceClassification):
    accepts_loss_kwargs = False


__all__ = [
    "HCXVisionForConditionalGeneration",
    "HCXVisionForSequenceClassification",
    "HCXVisionModel",
    "HCXVisionPreTrainedModel",
    "HyperClovaXForCausalLM",
    "HyperClovaXModel",
]
