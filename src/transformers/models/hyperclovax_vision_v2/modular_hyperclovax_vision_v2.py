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
from ...modeling_outputs import (
    BaseModelOutputWithPast,
    BaseModelOutputWithPooling,
    CausalLMOutputWithPast,
)
from ...processing_utils import Unpack
from ...utils import TransformersKwargs, auto_docstring, can_return_tuple, logging
from ..auto.modeling_auto import AutoModel
from ..gemma3.modeling_gemma3 import Gemma3ForSequenceClassification
from ..granite.modeling_granite import (
    GraniteAttention,
    GraniteDecoderLayer,
    GraniteForCausalLM,
    GraniteModel,
    GraniteRMSNorm,
)
from ..llama.modeling_llama import LlamaPreTrainedModel
from ..video_llama_3.modeling_video_llama_3 import VideoLlama3Model
from .configuration_hyperclovax_vision_v2 import HCXVisionV2Config, HyperCLOVAXConfig


logger = logging.get_logger(__name__)


class HyperCLOVAXAttention(GraniteAttention):
    pass


class HyperCLOVAXRMSNorm(GraniteRMSNorm):
    pass


class HyperCLOVAXDecoderLayer(GraniteDecoderLayer):
    def __init__(self, config: HyperCLOVAXConfig, layer_idx: int):
        super().__init__(config, layer_idx)

        self.use_post_norm = getattr(config, "use_post_norm", False)
        if self.use_post_norm:  # Peri-LN (post-norm)
            self.post_norm1 = HyperCLOVAXRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            self.post_norm2 = HyperCLOVAXRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        use_cache: bool | None = False,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*):
                attention mask of size `(batch_size, sequence_length)` if flash attention is used or `(batch_size, 1,
                query_sequence_length, key_sequence_length)` if default attention is used.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_values (`Cache`, *optional*): cached past key and value projection states
            position_embeddings (`tuple[torch.FloatTensor, torch.FloatTensor]`, *optional*):
                Tuple containing the cosine and sine positional embeddings of shape `(batch_size, seq_len, head_dim)`,
                with `head_dim` being the embedding dimension of each attention head.
            kwargs (`dict`, *optional*):
                Arbitrary kwargs to be ignored, used for FSDP and other methods that injects code
                into the model
        """
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        if self.use_post_norm:  # Peri-LN
            hidden_states = self.post_norm1(hidden_states)
        hidden_states = residual + hidden_states * self.residual_multiplier

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)

        if self.use_post_norm:  # Peri-LN
            hidden_states = self.post_norm2(hidden_states)

        hidden_states = residual + hidden_states * self.residual_multiplier

        return hidden_states


@auto_docstring
class HCXVisionV2PreTrainedModel(LlamaPreTrainedModel):
    config: HCXVisionV2Config
    _no_split_modules = ["HyperCLOVAXDecoderLayer"]
    input_modalities = ("image", "video", "text")
    _can_record_outputs = {
        "hidden_states": HyperCLOVAXDecoderLayer,
        "attentions": HyperCLOVAXAttention,
    }


@auto_docstring
class HyperCLOVAXModel(GraniteModel):
    config_class = HyperCLOVAXConfig
    input_modalities = ("text",)

    def __init__(self, config: HyperCLOVAXConfig):
        super().__init__(config)

        self.layers = nn.ModuleList(
            [HyperCLOVAXDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )

        self.post_init()


@auto_docstring
class HyperCLOVAXForCausalLM(GraniteForCausalLM):
    accepts_loss_kwargs = False
    config_class = HyperCLOVAXConfig
    input_modalities = ("text",)

    def __init__(self, config):
        super().__init__(config)
        self.model = HyperCLOVAXModel(config)


@auto_docstring
class HCXVisionV2Model(VideoLlama3Model):
    config: HCXVisionV2Config
    _no_split_modules = ["HyperCLOVAXDecoderLayer"]
    _can_record_outputs = {
        "hidden_states": HyperCLOVAXDecoderLayer,
        "attentions": HyperCLOVAXAttention,
    }

    def __init__(self, config: HCXVisionV2Config):
        super().__init__(config)

        self.vision_model = AutoModel.from_config(config.vision_config)

        self.projector = nn.Linear(config.vision_config.out_hidden_size, config.text_config.hidden_size)

        self.language_model = AutoModel.from_config(config.text_config)
        self.post_init()

    @can_return_tuple
    @auto_docstring
    def get_video_features(
        self,
        pixel_values_videos: torch.FloatTensor,
        video_grid_thw: torch.LongTensor,
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
        return self.get_image_features(pixel_values=pixel_values_videos, image_grid_thw=video_grid_thw, **kwargs)

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
        vision_outputs = self.vision_model(pixel_values, grid_thw=image_grid_thw, return_dict=True, **kwargs)
        vision_outputs.pooler_output = self.projector(vision_outputs.pooler_output)

        return vision_outputs

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
        pixel_values: torch.Tensor | None = None,
        pixel_values_videos: torch.FloatTensor | None = None,
        image_grid_thw: torch.LongTensor | None = None,
        video_grid_thw: torch.LongTensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | CausalLMOutputWithPast:
        r"""
        pixel_values (`torch.FloatTensor`, *optional*):
            Pixel values of input images after preprocessing by [`Qwen2VLImageProcessor`].
            A 2D tensor of shape `(total_num_patches, channels * patch_size^2 * temporal_patch_size)`.
            In the input token sequence, each image position should contain `config.image_token_id`.
        pixel_values_videos (`torch.FloatTensor`, *optional*):
            Pixel values of input videos, with the same format as `pixel_values`.
        image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
            The temporal, height and width dimensions of the feature grid for each image.
            Each row contains `[temporal, height, width]` grid counts.
        video_grid_thw (`torch.LongTensor` of shape `(num_videos, 3)`, *optional*):
            The temporal, height and width dimensions of the feature grid for each video.
        """
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

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
            **kwargs,
        )
        return BaseModelOutputWithPast(
            last_hidden_state=outputs.last_hidden_state,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


@auto_docstring
class HCXVisionV2ForConditionalGeneration(HCXVisionV2PreTrainedModel, GenerationMixin):
    accepts_loss_kwargs = False
    _tied_weights_keys = {"lm_head.weight": "model.language_model.embed_tokens.weight"}

    def __init__(self, config: HCXVisionV2Config):
        super().__init__(config)
        self.model = HCXVisionV2Model(config)
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
        >>> from transformers import AutoProcessor, HCXVisionV2ForConditionalGeneration

        >>> model = HCXVisionV2ForConditionalGeneration.from_pretrained(
        ...     "naver-hyperclovax/HyperCLOVAX-SEED-Think-32B",
        ...     torch_dtype="auto",
        ...     device_map="auto",
        ... )
        >>> processor = AutoProcessor.from_pretrained("naver-hyperclovax/HyperCLOVAX-SEED-Think-32B")

        >>> messages = [
        ...     {"role": "user", "content": [
        ...         {"type": "image_url", "image_url": {"url": "http://images.cocodataset.org/val2017/000000039769.jpg"}},
        ...         {"type": "text", "text": "Describe this image in detail."},
        ...     ]}
        ... ]
        >>> inputs = processor.tokenizer.apply_chat_template(
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


class HCXVisionV2ForSequenceClassification(Gemma3ForSequenceClassification, HCXVisionV2PreTrainedModel):
    config: HCXVisionV2Config
    input_modalities = ("text",)

    def __init__(self, config):
        super().__init__(config)
        self.model = HCXVisionV2Model(config)


class HyperCLOVAXForSequenceClassification(GenericForSequenceClassification, HCXVisionV2PreTrainedModel):
    config: HyperCLOVAXConfig
    input_modalities = ("text",)


__all__ = [
    "HCXVisionV2ForConditionalGeneration",
    "HCXVisionV2ForSequenceClassification",
    "HCXVisionV2Model",
    "HCXVisionV2PreTrainedModel",
    "HyperCLOVAXForCausalLM",
    "HyperCLOVAXModel",
    "HyperCLOVAXForSequenceClassification",
]
