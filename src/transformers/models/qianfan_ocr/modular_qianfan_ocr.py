# Copyright 2026 HuggingFace Inc. team. All rights reserved.
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


from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub.dataclasses import strict

from ...cache_utils import Cache
from ...configuration_utils import PreTrainedConfig
from ...image_processing_utils import BatchFeature
from ...image_utils import ImageInput
from ...modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling, ModelOutput
from ...modeling_utils import PreTrainedModel
from ...processing_utils import ProcessorMixin, Unpack
from ...tokenization_utils_base import PreTokenizedInput, TextInput
from ...utils import TransformersKwargs, auto_docstring
from ...utils.generic import can_return_tuple, merge_with_config_defaults
from ...utils.output_capturing import OutputRecorder, capture_outputs
from ..auto import CONFIG_MAPPING, AutoConfig
from ..beit.modeling_beit import BeitDropPath
from ..internvl.configuration_internvl import InternVLConfig, InternVLVisionConfig
from ..internvl.modeling_internvl import (
    InternVLForConditionalGeneration,
    InternVLPreTrainedModel,
    InternVLModel,
    InternVLCausalLMOutputWithPast,
    InternVLModelOutputWithPast,
    InternVLMultiModalProjector,
    InternVLVisionAttention,
    InternVLVisionEmbeddings,
    InternVLVisionLayer,
    InternVLVisionMLP,
    InternVLVisionModel,
    InternVLVisionPreTrainedModel,
)
from ..internvl.processing_internvl import InternVLProcessor


@strict
@auto_docstring(checkpoint="baidu/Qianfan-OCR")
class QianfanOCRVisionConfig(InternVLVisionConfig):
    r"""
    drop_path_rate (`float`, *optional*, defaults to 0.1):
        Dropout rate for stochastic depth.

    Example:

    ```python
    >>> from transformers import QianfanOCRVisionConfig

    >>> configuration = QianfanOCRVisionConfig()
    >>> configuration.hidden_size
    1024
    ```"""

    model_type = "qianfan_ocr_vision"
    base_config_key = "vision_config"

    attention_bias: bool = True
    drop_path_rate: float = 0.1


@strict
@auto_docstring(checkpoint="baidu/Qianfan-OCR")
class QianfanOCRConfig(InternVLConfig):
    r"""
    downsample_ratio (`float`, *optional*, defaults to 0.5):
        Factor by which to downsample the image.

    Example:

    ```python
    >>> from transformers import QianfanOCRConfig

    >>> configuration = QianfanOCRConfig()
    >>> configuration.downsample_ratio
    0.5
    ```"""

    model_type = "qianfan_ocr"
    sub_configs = {"text_config": AutoConfig, "vision_config": QianfanOCRVisionConfig}

    tie_word_embeddings: bool = False

    def __post_init__(self, **kwargs):
        if isinstance(self.vision_config, dict):
            self.vision_config = QianfanOCRVisionConfig(**self.vision_config)
        elif self.vision_config is None:
            self.vision_config = QianfanOCRVisionConfig()

        if isinstance(self.text_config, dict):
            self.text_config["model_type"] = self.text_config.get("model_type", "qwen3")
            self.text_config = CONFIG_MAPPING[self.text_config["model_type"]](**self.text_config)
        elif self.text_config is None:
            self.text_config = CONFIG_MAPPING["qwen3"]()

        super(InternVLConfig, self).__post_init__(**kwargs)


class QianfanOCRDropPath(BeitDropPath):
    pass


class QianfanOCRVisionAttention(InternVLVisionAttention):
    pass


class QianfanOCRVisionMLP(InternVLVisionMLP):
    pass


class QianfanOCRVisionLayer(InternVLVisionLayer):
    """Vision transformer layer with stochastic depth (DropPath) support."""

    def __init__(self, config: QianfanOCRVisionConfig, drop_path_rate: float = 0.0) -> None:
        super().__init__(config)
        self.drop_path1 = nn.Identity() if drop_path_rate <= 0.0 else QianfanOCRDropPath(drop_path_rate)
        self.drop_path2 = nn.Identity() if drop_path_rate <= 0.0 else QianfanOCRDropPath(drop_path_rate)

    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> tuple[torch.Tensor] | tuple[torch.Tensor, torch.Tensor]:
        attention_output, _ = self.attention(
            self.layernorm_before(hidden_states),
        )

        attention_output = self.lambda_1 * attention_output

        # first residual connection with drop path
        hidden_states = self.drop_path1(attention_output) + hidden_states

        # layernorm after self-attention
        layer_output = self.layernorm_after(hidden_states)

        layer_output = self.mlp(layer_output)
        layer_output = self.dropout(layer_output)

        if self.lambda_2 is not None:
            layer_output = self.lambda_2 * layer_output

        # second residual connection with drop path
        layer_output = self.drop_path2(layer_output) + hidden_states

        return layer_output


class QianfanOCRVisionEncoder(nn.Module):
    def __init__(self, config: QianfanOCRVisionConfig) -> None:
        super().__init__()
        self.config = config
        self.gradient_checkpointing = False

        dpr = [x.item() for x in torch.linspace(0, config.drop_path_rate, config.num_hidden_layers, device="cpu")]
        self.layer = nn.ModuleList([QianfanOCRVisionLayer(config, drop_path_rate=dpr[i]) for i in range(config.num_hidden_layers)])

    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> tuple | BaseModelOutput:
        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states)

        return BaseModelOutput(
            last_hidden_state=hidden_states,
        )


class QianfanOCRVisionEmbeddings(InternVLVisionEmbeddings):
    pass


class QianfanOCRVisionModelOutputWithPooling(BaseModelOutputWithPooling):
    r"""
    pooler_output (`torch.FloatTensor` of shape `(batch_size, hidden_size)`):
        Average of the last layer hidden states of the patch tokens (excluding the *[CLS]* token) if
        *config.use_mean_pooling* is set to True. If set to False, then the final hidden state of the *[CLS]* token
        will be returned.
    """

    pass


@auto_docstring
class QianfanOCRVisionPreTrainedModel(InternVLVisionPreTrainedModel):
    config_class = QianfanOCRVisionConfig
    base_model_prefix = "vision_model"
    _no_split_modules = ["QianfanOCRVisionLayer"]
    _can_record_outputs = {
        "hidden_states": OutputRecorder(QianfanOCRVisionLayer, index=0),
        "attentions": OutputRecorder(QianfanOCRVisionAttention, index=1),
    }


@auto_docstring
class QianfanOCRVisionModel(InternVLVisionModel):
    @merge_with_config_defaults
    @capture_outputs(tie_last_hidden_states=False)
    @auto_docstring
    def forward(
        self, pixel_values: torch.Tensor, bool_masked_pos: torch.BoolTensor | None = None, **kwargs
    ) -> tuple | QianfanOCRVisionModelOutputWithPooling:
        r"""
        bool_masked_pos (`torch.BoolTensor` of shape `(batch_size, num_patches)`, *optional*):
            Boolean masked positions. Indicates which patches are masked (1) and which aren't (0).
        """
        embedding_output = self.embeddings(pixel_values, bool_masked_pos=bool_masked_pos)
        encoder_outputs = self.encoder(embedding_output)
        sequence_output = encoder_outputs[0]
        sequence_output = self.layernorm(sequence_output)

        return QianfanOCRVisionModelOutputWithPooling(
            last_hidden_state=sequence_output,
        )


class QianfanOCRMultiModalProjector(InternVLMultiModalProjector):
    pass


class QianfanOCRPreTrainedModel(InternVLPreTrainedModel):
    config_class = QianfanOCRConfig
    input_modalities = ("image", "text")


@dataclass
@auto_docstring(
    custom_intro="""
    Base class for QianfanOCR outputs, with hidden states and attentions.
    """
)
class QianfanOCRModelOutputWithPast(InternVLModelOutputWithPast):
    r"""
    image_hidden_states (`torch.FloatTensor`, *optional*):
        A `torch.FloatTensor` of size `(batch_size, num_images, sequence_length, hidden_size)`.
        image_hidden_states of the model produced by the vision encoder and after projecting the last hidden state.
    """


class QianfanOCRModel(InternVLModel):
    @merge_with_config_defaults
    @can_return_tuple
    @auto_docstring(
        custom_intro="Obtains image last hidden states from the vision tower and apply multimodal projection."
    )
    def get_image_features(
        self,
        pixel_values: torch.FloatTensor,
        vision_feature_layer: int | list[int] | list[int] | None = None,
        vision_feature_select_strategy: str | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | BaseModelOutputWithPooling:
        r"""
        pixel_values (`torch.FloatTensor]` of shape `(batch_size, channels, height, width)`)
            The tensors corresponding to the input images.
        vision_feature_layer (`int` or `list[int]`):
            Layer index or list of layer indices to extract features from.
        """
        # Use vision_tower parameter dtype instead of self.dtype for DataParallel compatibility.
        try:
            target_dtype = next(self.vision_tower.parameters()).dtype
        except StopIteration:
            target_dtype = pixel_values.dtype
        pixel_values = pixel_values.to(dtype=target_dtype)  # fp16 compatibility

        downsample_ratio = self.config.downsample_ratio
        if vision_feature_layer != -1:
            kwargs["output_hidden_states"] = True
        vision_outputs = self.vision_tower(pixel_values=pixel_values, return_dict=True, **kwargs)
        if vision_feature_layer == -1:
            vision_features = vision_outputs.last_hidden_state
        else:
            vision_features = vision_outputs.hidden_states[vision_feature_layer]
        if vision_feature_select_strategy == "default":
            vision_features = vision_features[:, 1:, :]

        channels = vision_features.shape[1]
        feature_size = int(channels**0.5)
        batch_size = vision_features.shape[0]

        vision_features = vision_features.reshape(batch_size, feature_size, feature_size, -1)

        vision_features = self.pixel_shuffle(vision_features, scale_factor=downsample_ratio)

        vision_features = vision_features.reshape(batch_size, -1, vision_features.shape[-1])

        vision_features = self.multi_modal_projector(vision_features)
        vision_outputs.pooler_output = vision_features

        return vision_outputs

class QianfanOCRCausalLMOutputWithPast(InternVLCausalLMOutputWithPast):
    pass


class QianfanOCRForConditionalGeneration(InternVLForConditionalGeneration):
    @can_return_tuple
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
        logits_to_keep: int | torch.Tensor = 0,
        image_sizes: torch.Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | QianfanOCRCausalLMOutputWithPast:
        r"""
        Example:

        ```python
        >>> import torch
        >>> from transformers import AutoProcessor, AutoModelForImageTextToText

        >>> torch_device = "cuda"
        >>> processor = AutoProcessor.from_pretrained("baidu/Qianfan-OCR")
        >>> model = AutoModelForImageTextToText.from_pretrained(
        ...     "baidu/Qianfan-OCR", dtype=torch.bfloat16, device_map=torch_device
        ... )

        >>> messages = [
        ...     {
        ...         "role": "user",
        ...         "content": [
        ...             {"type": "image", "url": "https://example.com/image.jpg"},
        ...             {"type": "text", "text": "Describe this image."},
        ...         ],
        ...     },
        ... ]

        >>> inputs = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt").to(torch_device)
        >>> generate_ids = model.generate(**inputs, max_new_tokens=200)
        >>> print(processor.decode(generate_ids[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True))
        ```"""
        return super().forward(
            input_ids=input_ids,
            pixel_values=pixel_values,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            vision_feature_layer=vision_feature_layer,
            vision_feature_select_strategy=vision_feature_select_strategy,
            labels=labels,
            logits_to_keep=logits_to_keep,
            image_sizes=image_sizes,
            **kwargs,
        )


class QianfanOCRProcessor(InternVLProcessor):
    def __init__(
        self,
        image_processor=None,
        tokenizer=None,
        image_seq_length: int = 256,
        chat_template=None,
        start_image_token: str = "<img>",
        end_image_token: str = "</img>",
        context_image_token: str = "<IMG_CONTEXT>",
        **kwargs,
    ):
        r"""
        start_image_token (`str`, *optional*, defaults to `"<img>"`):
            The token used to mark the start of an image sequence.
        end_image_token (`str`, *optional*, defaults to `"</img>"`):
            The token used to mark the end of an image sequence.
        context_image_token (`str`, *optional*, defaults to `"<IMG_CONTEXT>"`):
            The token used as an image context placeholder in the input sequence.
        """
        # InternVLProcessor.__init__ reads these as tokenizer attributes.
        # Inject them so it works with tokenizers that don't expose them (e.g. Qwen2Tokenizer).
        if tokenizer is not None:
            for attr, value in (
                ("start_image_token", start_image_token),
                ("end_image_token", end_image_token),
                ("start_image_token_id", tokenizer.convert_tokens_to_ids(start_image_token)),
                ("end_image_token_id", tokenizer.convert_tokens_to_ids(end_image_token)),
                ("context_image_token", context_image_token),
                ("context_image_token_id", tokenizer.convert_tokens_to_ids(context_image_token)),
            ):
                if not hasattr(tokenizer, attr):
                    setattr(tokenizer, attr, value)
        ProcessorMixin.__init__(
            self,
            image_processor=image_processor,
            tokenizer=tokenizer,
            chat_template=chat_template,
            **kwargs,
        )
        self.image_seq_length = image_seq_length
        # Override with the config values to ensure they are always used,
        # regardless of what the tokenizer may have returned.
        self.start_image_token = start_image_token
        self.end_image_token = end_image_token
        self.image_token = context_image_token
        self.video_token = None
        self.start_image_token_id = tokenizer.convert_tokens_to_ids(start_image_token)
        self.end_image_token_id = tokenizer.convert_tokens_to_ids(end_image_token)
        self.image_token_id = tokenizer.convert_tokens_to_ids(context_image_token)
        self.image_ids = [self.image_token_id, self.start_image_token_id, self.end_image_token_id]
        self.video_processor = None

    def _insert_media_placeholders(
        self,
        text: list[str],
        image_pixel_values,
        video_pixel_values,
        image_num_patches: list[int],
        video_num_patches: list[int],
        image_num_patches_indices: np.ndarray,
        video_num_patches_indices: np.ndarray,
        video_patch_indices: np.ndarray,
    ):
        """
        Processes interleaved text with <image> placeholders, replacing them with appropriate image tokens.
        """
        image_index = 0
        processed_text = []
        image_patches = []
        replace_strings = []
        for prompt in text:
            new_prompt = prompt
            while self.image_token in new_prompt:
                start_index = image_num_patches_indices[image_index - 1] if image_index > 0 else 0
                end_index = image_num_patches_indices[image_index]
                image_patches.append(image_pixel_values[start_index:end_index])
                new_prompt = new_prompt.replace(self.image_token, "<placeholder>", 1)
                replace_strings.append(
                    f"{self.start_image_token}{self.image_token * self.image_seq_length * image_num_patches[image_index]}{self.end_image_token}"
                )
                image_index += 1
            while "<placeholder>" in new_prompt:
                replace_str = replace_strings.pop(0)
                new_prompt = new_prompt.replace("<placeholder>", replace_str, 1)
            processed_text.append(new_prompt)
        return processed_text, image_patches, image_index, 0

    def __call__(
        self,
        images: ImageInput | None = None,
        text: TextInput | PreTokenizedInput | list[TextInput] | list[PreTokenizedInput] | None = None,
        videos=None,
        **kwargs,
    ) -> BatchFeature:
        # QianfanOCR has no video or audio support. Drop those keys if they arrive
        # from apply_chat_template's internal self(...) call
        kwargs.pop("audio", None)
        videos = None  # QianfanOCR has no video support
        return super().__call__(images=images, text=text, videos=videos, **kwargs)


__all__ = [
    "QianfanOCRVisionConfig",
    "QianfanOCRConfig",
    "QianfanOCRVisionPreTrainedModel",
    "QianfanOCRVisionModel",
    "QianfanOCRPreTrainedModel",
    "QianfanOCRModel",
    "QianfanOCRForConditionalGeneration",
    "QianfanOCRProcessor",
]
