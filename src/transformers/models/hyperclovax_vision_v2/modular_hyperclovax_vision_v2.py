# Copyright 2026 NAVER Corp. and The HuggingFace Team. All rights reserved.
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
"""PyTorch HyperCLOVAX Vision V2 model."""

import torch
from huggingface_hub.dataclasses import strict
from torch import nn

from ...cache_utils import Cache
from ...configuration_utils import PreTrainedConfig
from ...generation.utils import GenerationMixin
from ...modeling_outputs import (
    BaseModelOutputWithPast,
    BaseModelOutputWithPooling,
    CausalLMOutputWithPast,
)
from ...processing_utils import Unpack
from ...utils import TransformersKwargs, auto_docstring, can_return_tuple, logging
from ..auto import CONFIG_MAPPING, AutoConfig
from ..exaone4_5.modeling_exaone4_5 import Exaone4_5_ForConditionalGeneration
from ..exaone4_5.processing_exaone4_5 import Exaone4_5_Processor
from ..gemma3.modeling_gemma3 import Gemma3ForSequenceClassification
from ..video_llama_3.modeling_video_llama_3 import VideoLlama3Model, VideoLlama3PreTrainedModel


logger = logging.get_logger(__name__)


@auto_docstring(checkpoint="naver-hyperclovax/HyperCLOVAX-SEED-Think-32B")
@strict
class HyperCLOVAXVisionV2Config(PreTrainedConfig):
    r"""
    ```python
    >>> from transformers import HyperCLOVAXVisionV2Config, HyperCLOVAXVisionV2ForConditionalGeneration

    >>> # Initializing a HyperCLOVAX Vision V2 configuration with defaults
    >>> configuration = HyperCLOVAXVisionV2Config()

    >>> # Initializing a model from the configuration
    >>> model = HyperCLOVAXVisionV2ForConditionalGeneration(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```
    """

    model_type = "hyperclovax_vision_v2"
    sub_configs = {"text_config": AutoConfig, "vision_config": AutoConfig}
    keys_to_ignore_at_inference = ["past_key_values"]

    text_config: dict | PreTrainedConfig | None = None
    vision_config: dict | PreTrainedConfig | None = None
    image_token_id: int = 128060
    video_token_id: int = 128061
    tie_word_embeddings: bool = True

    def __post_init__(self, **kwargs):
        if isinstance(self.vision_config, dict):
            model_type = self.vision_config.get("model_type", "qwen2_5_vl_vision")
            # The Hub config uses the full Qwen2.5-VL type for the vision transformer.
            model_type = "qwen2_5_vl_vision" if model_type == "qwen2_5_vl" else model_type
            self.vision_config["model_type"] = model_type
            self.vision_config = CONFIG_MAPPING[model_type](**self.vision_config)
        elif self.vision_config is None:
            self.vision_config = CONFIG_MAPPING["qwen2_5_vl_vision"]()

        if isinstance(self.text_config, dict):
            model_type = self.text_config.get("model_type", "hyperclovax")
            self.text_config = CONFIG_MAPPING[model_type](**self.text_config)
        elif self.text_config is None:
            self.text_config = CONFIG_MAPPING["hyperclovax"]()

        # This is necessary to properly find the weight conversion mapping.
        if kwargs.get("model_type") == "vlm":
            kwargs["model_type"] = "hyperclovax_vision_v2"

        super().__post_init__(**kwargs)


@auto_docstring
class HyperCLOVAXVisionV2Processor(Exaone4_5_Processor):
    @property
    def model_input_names(self):
        # HyperCLOVAX Vision V2 does not use second_per_grid_ts (no temporal RoPE)
        names = super().model_input_names
        return [n for n in names if n not in ("second_per_grid_ts")]


@auto_docstring
class HyperCLOVAXVisionV2PreTrainedModel(VideoLlama3PreTrainedModel):
    config: HyperCLOVAXVisionV2Config
    input_modalities = ("image", "video", "text")
    _no_split_modules = ["HyperCLOVAXDecoderLayer"]
    _can_record_outputs = {
        "hidden_states": "HyperCLOVAXDecoderLayer",
        "attentions": "HyperCLOVAXAttention",
    }


@auto_docstring
class HyperCLOVAXVisionV2Model(HyperCLOVAXVisionV2PreTrainedModel, VideoLlama3Model):
    def __init__(self, config: HyperCLOVAXVisionV2Config):
        super().__init__(config)
        self.projector = nn.Linear(config.vision_config.out_hidden_size, config.text_config.hidden_size)

    def get_rope_index(self):
        raise AttributeError("HyperCLOVAX Vision V2 does not need 3D positions")

    def get_vision_position_ids(self):
        raise AttributeError("HyperCLOVAX Vision V2 does not need 3D positions")

    def compute_3d_position_ids(self):
        raise AttributeError("HyperCLOVAX Vision V2 does not need 3D positions")

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
class HyperCLOVAXVisionV2ForConditionalGeneration(
    HyperCLOVAXVisionV2PreTrainedModel, Exaone4_5_ForConditionalGeneration
):
    # Reference: fix gemma3 grad acc #37208
    accepts_loss_kwargs = False
    _tied_weights_keys = {"lm_head.weight": "model.language_model.embed_tokens.weight"}

    def __init__(self, config: HyperCLOVAXVisionV2Config):
        super().__init__(config)
        self.model = HyperCLOVAXVisionV2Model(config)
        self.vocab_size = config.text_config.vocab_size
        self.lm_head = nn.Linear(config.text_config.hidden_size, config.text_config.vocab_size, bias=False)
        self.post_init()

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
            Pixel values of input videos, same format as `pixel_values`.
        image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
            `[temporal, height, width]` grid counts per image.
        video_grid_thw (`torch.LongTensor` of shape `(num_videos, 3)`, *optional*):
            `[temporal, height, width]` grid counts per video.
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss.
        logits_to_keep (`int` or `torch.Tensor`, *optional*, defaults to 0):
            If an `int`, compute logits for the last `logits_to_keep` tokens.

        Example:

        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import HyperCLOVAXVisionV2Processor, HyperCLOVAXVisionV2ForConditionalGeneration

        >>> model = HyperCLOVAXVisionV2ForConditionalGeneration.from_pretrained(
        ...     "naver-hyperclovax/HyperCLOVAX-SEED-Think-32B",
        ...     torch_dtype="auto",
        ...     device_map="auto",
        ... )
        >>> processor = HyperCLOVAXVisionV2Processor.from_pretrained("naver-hyperclovax/HyperCLOVAX-SEED-Think-32B")

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
        logits = self.lm_head(hidden_states[:, slice_indices, :]) * self.config.text_config.logits_scaling

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

    def _prepare_position_ids_for_generation(self, inputs_tensor, model_kwargs):
        # HyperCLOVAX Vision V2 uses 1D position_ids (not Qwen2.5-VL's 3D/4D rope_deltas-based ids)

        return GenerationMixin._prepare_position_ids_for_generation(self, inputs_tensor, model_kwargs)

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

        # HyperCLOVAX Vision V2 uses 1D position_ids — do NOT force None like Exaone4.5 does for 2D-RoPE
        return model_inputs


class HyperCLOVAXVisionV2ForSequenceClassification(
    Gemma3ForSequenceClassification, HyperCLOVAXVisionV2PreTrainedModel
):
    config: HyperCLOVAXVisionV2Config
    input_modalities = ("text",)


__all__ = [
    "HyperCLOVAXVisionV2Config",
    "HyperCLOVAXVisionV2ForConditionalGeneration",
    "HyperCLOVAXVisionV2ForSequenceClassification",
    "HyperCLOVAXVisionV2Model",
    "HyperCLOVAXVisionV2PreTrainedModel",
    "HyperCLOVAXVisionV2Processor",
]
