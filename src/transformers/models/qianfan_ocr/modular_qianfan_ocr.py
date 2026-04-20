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
from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...image_processing_utils import BatchFeature
from ...image_utils import ImageInput
from ...modeling_outputs import BaseModelOutputWithPooling
from ...processing_utils import ProcessorMixin, Unpack
from ...tokenization_utils_base import PreTokenizedInput, TextInput
from ...utils import TransformersKwargs, auto_docstring
from ...utils.generic import can_return_tuple, merge_with_config_defaults
from ...utils.output_capturing import capture_outputs
from ..auto import CONFIG_MAPPING, AutoConfig
from ..beit.modeling_beit import BeitDropPath
from ..internvl.configuration_internvl import InternVLConfig, InternVLVisionConfig
from ..internvl.modeling_internvl import (
    InternVLCausalLMOutputWithPast,
    InternVLForConditionalGeneration,
    InternVLModel,
    InternVLModelOutputWithPast,
    InternVLMultiModalProjector,
    InternVLPreTrainedModel,
    InternVLVisionAttention,
    InternVLVisionEmbeddings,
    InternVLVisionLayer,
    InternVLVisionMLP,
    InternVLVisionModel,
    InternVLVisionPreTrainedModel,
)
from ..internvl.processing_internvl import InternVLProcessor


@auto_docstring(checkpoint="baidu/Qianfan-OCR")
@strict
class QianfanOCRVisionConfig(InternVLVisionConfig):
    r"""
    projection_dropout (`float`, *optional*, defaults to 0.0):
        Dropout probability for the projection layer.
    norm_type (`str`, *optional*, defaults to `"layer_norm"`):
        The type of normalization to use in the encoder. Can be `"layer_norm"` or `"rms_norm"`.
    use_mask_token (`bool`, *optional*, defaults to `False`):
        Whether to use a mask token for masked image modeling.
    use_mean_pooling (`bool`, *optional*, defaults to `True`):
        Whether to mean pool the final hidden states of the patches instead of using the final hidden state of the
        CLS token, before applying the classification head.
    drop_path_rate (`float`, *optional*, defaults to 0.1):
        Dropout rate for stochastic depth.

    Example:

    ```python
    >>> # Initializing a QianfanOCR vision style configuration
    >>> configuration = QianfanOCRVisionConfig()

    >>> # Initializing a model from the configuration
    >>> model = QianfanOCRVisionModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "qianfan_ocr_vision"
    base_config_key = "vision_config"

    attention_bias: bool = True
    drop_path_rate: float = 0.1


@auto_docstring(checkpoint="baidu/Qianfan-OCR")
@strict
class QianfanOCRConfig(InternVLConfig):
    r"""
    downsample_ratio (`float`, *optional*, defaults to 0.5):
        Factor by which to downsample the image.

    Example:

    ```python
    >>> # Initializing a QianfanOCR style configuration
    >>> configuration = QianfanOCRConfig()

    >>> # Initializing a model from the configuration
    >>> model = QianfanOCRForConditionalGeneration(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
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

        PreTrainedConfig.__post_init__(self, **kwargs)


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
        del self.seq_len_dim
        del self.chunk_size_feed_forward
        self.drop_path1 = nn.Identity() if drop_path_rate <= 0.0 else QianfanOCRDropPath(drop_path_rate)
        self.drop_path2 = nn.Identity() if drop_path_rate <= 0.0 else QianfanOCRDropPath(drop_path_rate)

    def forward(
        self,
        hidden_states: torch.Tensor,
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.layernorm_before(hidden_states)
        # Self Attention
        hidden_states, _ = self.attention(hidden_states, **kwargs)
        hidden_states = self.lambda_1 * hidden_states
        hidden_states = self.drop_path1(hidden_states)
        hidden_states = hidden_states + residual

        residual = hidden_states
        hidden_states = self.layernorm_after(hidden_states)
        # Fully Connected
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.lambda_2 * hidden_states
        hidden_states = self.drop_path2(hidden_states) + residual

        return hidden_states


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
        "hidden_states": QianfanOCRVisionLayer,
        "attentions": QianfanOCRVisionAttention,
    }


@auto_docstring
class QianfanOCRVisionModel(InternVLVisionModel):
    def __init__(self, config: QianfanOCRVisionConfig) -> None:
        super().__init__(config)
        del self.encoder
        dpr = [x.item() for x in torch.linspace(0, config.drop_path_rate, config.num_hidden_layers, device="cpu")]
        self.layers = nn.ModuleList(
            [QianfanOCRVisionLayer(config, drop_path_rate=dpr[i]) for i in range(config.num_hidden_layers)]
        )

    @merge_with_config_defaults
    @capture_outputs
    @auto_docstring
    def forward(
        self,
        pixel_values: torch.Tensor,
        bool_masked_pos: torch.BoolTensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | QianfanOCRVisionModelOutputWithPooling:
        r"""
        bool_masked_pos (`torch.BoolTensor` of shape `(batch_size, num_patches)`, *optional*):
            Boolean masked positions. Indicates which patches are masked (1) and which aren't (0).
        """
        hidden_states = self.embeddings(pixel_values, bool_masked_pos=bool_masked_pos)
        for layer_module in self.layers:
            hidden_states = layer_module(hidden_states, **kwargs)
        hidden_states = self.layernorm(hidden_states)

        return QianfanOCRVisionModelOutputWithPooling(
            last_hidden_state=hidden_states,
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
    pass


class QianfanOCRCausalLMOutputWithPast(InternVLCausalLMOutputWithPast):
    pass


class QianfanOCRForConditionalGeneration(InternVLForConditionalGeneration):
    @can_return_tuple
    @auto_docstring
    def forward(self, **super_kwargs) -> tuple | QianfanOCRCausalLMOutputWithPast:
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
        return super().forward(**super_kwargs)


class QianfanOCRProcessor(InternVLProcessor):
    def __init__(
        self,
        image_processor=None,
        tokenizer=None,
        image_seq_length: int = 256,
        chat_template=None,
        image_placeholder_token: str = "<image>",
        **kwargs,
    ):
        r"""
        image_placeholder_token (`str`, *optional*, defaults to `"<image>"`):
            The token emitted by the chat template to mark image positions.
            It is replaced by the full ``<img><IMG_CONTEXT>...<IMG_CONTEXT></img>``
            sequence during processing.
        """
        ProcessorMixin.__init__(self, image_processor, tokenizer, chat_template=chat_template, **kwargs)
        self.image_seq_length = image_seq_length
        self.start_image_token = tokenizer.start_image_token
        self.end_image_token = tokenizer.end_image_token
        self.start_image_token_id = tokenizer.start_image_token_id
        self.end_image_token_id = tokenizer.end_image_token_id
        self.image_token = tokenizer.context_image_token
        self.image_token_id = tokenizer.context_image_token_id
        self.image_ids = [self.image_token_id, self.start_image_token_id, self.end_image_token_id]
        self.image_placeholder_token = image_placeholder_token
        self.video_token = None
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
            while self.image_placeholder_token in new_prompt:
                start_index = image_num_patches_indices[image_index - 1] if image_index > 0 else 0
                end_index = image_num_patches_indices[image_index]
                image_patches.append(image_pixel_values[start_index:end_index])
                new_prompt = new_prompt.replace(self.image_placeholder_token, "<placeholder>", 1)
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
        if videos is not None:
            raise ValueError("QianfanOCR does not support video input.")

        return super().__call__(images=images, text=text, videos=None, **kwargs)


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
