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
from ...modeling_outputs import BaseModelOutputWithPast, BaseModelOutputWithPooling, CausalLMOutputWithPast
from ...processing_utils import Unpack
from ...utils import TransformersKwargs, auto_docstring, can_return_tuple, logging, torch_compilable_check
from ..auto.modeling_auto import AutoModel
from ..granite.modeling_granite import GraniteAttention, GraniteDecoderLayer, GraniteForCausalLM, GraniteModel
from ..llama.modeling_llama import LlamaPreTrainedModel
from .configuration_hyperclovax_vision import HCXVisionConfig, HyperClovaXConfig


logger = logging.get_logger(__name__)


class HyperClovaXAttention(GraniteAttention):
    pass


class HyperClovaXDecoderLayer(GraniteDecoderLayer):
    pass


@auto_docstring
class HCXVisionPreTrainedModel(LlamaPreTrainedModel):
    config_class = HCXVisionConfig
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
class HCXVisionModel(HCXVisionPreTrainedModel):
    def __init__(self, config: HCXVisionConfig):
        super().__init__(config)

        self.vision_model = AutoModel.from_config(config.vision_config)

        self.multi_modal_projector = nn.Linear(config.vision_config.out_hidden_size, config.text_config.hidden_size)

        self.language_model = AutoModel.from_config(config.text_config)
        self.post_init()

    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.language_model.set_input_embeddings(value)

    def get_rope_index(self):
        raise AttributeError("Not needed for HCX")

    def get_vision_position_ids(self):
        raise AttributeError("Not needed for HCX")

    def compute_3d_position_ids(self):
        raise AttributeError("Not needed for HCX")

    @can_return_tuple
    @auto_docstring
    def get_video_features(
        self,
        pixel_values_videos: torch.FloatTensor,
        video_grid_thw: torch.LongTensor,
        video_merge_sizes: torch.LongTensor,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | BaseModelOutputWithPooling:
        r"""
        pixel_values_videos (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)`):
            The tensors corresponding to the input videos.
        video_grid_thw (`torch.LongTensor` of shape `(num_videos, 3)`, *optional*):
            The temporal, height and width of feature shape of each video in LLM.
        video_merge_sizes (`torch.Tensor` of shape `(num_videos,)`):
            The spatial downsampling ratio of each video feature.
        """
        return self.get_image_features(
            pixel_values=pixel_values_videos,
            image_grid_thw=video_grid_thw,
            image_merge_sizes=video_merge_sizes,
            **kwargs,
        )

    @can_return_tuple
    @auto_docstring
    def get_image_features(
        self,
        pixel_values: torch.FloatTensor,
        image_grid_thw: torch.LongTensor,
        image_merge_sizes: torch.LongTensor,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | BaseModelOutputWithPooling:
        r"""
        pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)`):
            The tensors corresponding to the input images.
        image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
            The temporal, height and width of feature shape of each image in LLM.
        image_merge_sizes (`torch.Tensor` of shape `(num_images,)`):
            The spatial downsampling ratio of each image feature.
        """
        vision_outputs = self.vision_model(
            pixel_values=pixel_values,
            grid_thw=image_grid_thw,
            merge_sizes=image_merge_sizes,
            return_dict=True,
            **kwargs,
        )
        last_hidden_state = vision_outputs.last_hidden_state
        image_embeds = self.projector(last_hidden_state)

        split_sizes = image_grid_thw.prod(dim=1) // (image_merge_sizes**2)
        image_embeds = torch.split(image_embeds, split_sizes.tolist())
        vision_outputs.pooler_output = image_embeds

        return vision_outputs

    def get_placeholder_mask(
        self,
        input_ids: torch.LongTensor,
        inputs_embeds: torch.FloatTensor,
        image_features: torch.FloatTensor | None = None,
        video_features: torch.FloatTensor | None = None,
    ) -> tuple[torch.BoolTensor, torch.BoolTensor]:
        if input_ids is None:
            special_image_mask = inputs_embeds == self.get_input_embeddings()(
                torch.tensor(self.config.img_start_id, dtype=torch.long, device=inputs_embeds.device)
            )
            special_image_mask = special_image_mask.all(-1)
            special_video_mask = inputs_embeds == self.get_input_embeddings()(
                torch.tensor(self.config.video_start_id, dtype=torch.long, device=inputs_embeds.device)
            )
            special_video_mask = special_video_mask.all(-1)
        else:
            special_image_mask = input_ids == self.config.img_start_id
            special_video_mask = input_ids == self.config.video_start_id

        n_image_tokens = special_image_mask.sum()
        special_image_mask = special_image_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
        if image_features is not None:
            torch_compilable_check(
                inputs_embeds[special_image_mask].numel() == image_features.numel(),
                f"Image features and image tokens do not match, tokens: {n_image_tokens}, features: {image_features.shape[0]}",
            )

        n_video_tokens = special_video_mask.sum()
        special_video_mask = special_video_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
        if video_features is not None:
            torch_compilable_check(
                inputs_embeds[special_video_mask].numel() == video_features.numel(),
                f"Video features and video tokens do not match, tokens: {n_video_tokens}, features: {video_features.shape[0]}",
            )
        return special_image_mask, special_video_mask

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        cache_position: torch.LongTensor | None = None,
        pixel_values: torch.Tensor | None = None,
        pixel_values_videos: torch.FloatTensor | None = None,
        image_grid_thw: torch.LongTensor | None = None,
        video_grid_thw: torch.LongTensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | BaseModelOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if pixel_values is not None:
            image_embeds = self.get_image_features(pixel_values, image_grid_thw).pooler_output
            image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
            image_mask, _ = self.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds
            )
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

        if pixel_values_videos is not None:
            video_embeds = self.get_video_features(pixel_values_videos, video_grid_thw).pooler_output
            video_embeds = video_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
            _, video_mask = self.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, video_features=video_embeds
            )
            inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

        outputs = self.language_model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )

        return BaseModelOutputWithPast(
            last_hidden_state=outputs.last_hidden_state,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


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
    def get_image_features(
        self,
        pixel_values: torch.FloatTensor,
        image_grid_thw: torch.LongTensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ):
        r"""
        pixel_values (`torch.FloatTensor`):
            The tensors corresponding to the input images.
        image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
            The temporal, height and width of feature shape of each image in LLM.
        """
        return self.model.get_image_features(pixel_values, image_grid_thw=image_grid_thw, **kwargs)

    @auto_docstring
    def get_video_features(
        self,
        pixel_values_videos: torch.FloatTensor,
        video_grid_thw: torch.LongTensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ):
        r"""
        pixel_values_videos (`torch.FloatTensor`):
            The tensors corresponding to the input videos.
        video_grid_thw (`torch.LongTensor` of shape `(num_videos, 3)`, *optional*):
            The temporal, height and width of feature shape of each video in LLM.
        """
        return self.model.get_video_features(pixel_values_videos, video_grid_thw=video_grid_thw, **kwargs)

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

    def _prepare_position_ids_for_generation(self):
        raise AttributeError("Not needed for HCX")


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
