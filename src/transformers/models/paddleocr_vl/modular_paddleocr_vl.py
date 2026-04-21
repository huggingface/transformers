# Copyright 2025 The PaddlePaddle Team and The HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
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

import math

import numpy as np
import torch
from huggingface_hub.dataclasses import strict
from torch import nn

from ... import initialization as init
from ...activations import GELUActivation
from ...cache_utils import Cache, DynamicCache
from ...image_processing_utils import BatchFeature
from ...image_transforms import group_images_by_shape, reorder_images
from ...image_utils import (
    ImageInput,
    PILImageResampling,
    SizeDict,
)
from ...masking_utils import create_bidirectional_mask, create_causal_mask
from ...modeling_outputs import BaseModelOutput, BaseModelOutputWithPast, BaseModelOutputWithPooling
from ...modeling_utils import PreTrainedModel
from ...models.qwen2_vl.image_processing_pil_qwen2_vl import Qwen2VLImageProcessorPil
from ...models.qwen2_vl.image_processing_qwen2_vl import Qwen2VLImageProcessor
from ...processing_utils import (
    ImagesKwargs,
    ProcessingKwargs,
    ProcessorMixin,
    Unpack,
)
from ...tokenization_utils_base import PreTokenizedInput, TextInput
from ...utils import (
    TensorType,
    TransformersKwargs,
    auto_docstring,
    can_return_tuple,
    logging,
    torch_compilable_check,
    torch_int,
)
from ...utils.generic import merge_with_config_defaults
from ...utils.output_capturing import capture_outputs
from ..ernie4_5.configuration_ernie4_5 import Ernie4_5Config
from ..ernie4_5.modeling_ernie4_5 import (
    Ernie4_5DecoderLayer,
    Ernie4_5MLP,
    Ernie4_5Model,
    Ernie4_5RMSNorm,
)
from ..qwen2_5_omni.modeling_qwen2_5_omni import (
    Qwen2_5OmniAttention,
)
from ..qwen2_vl.configuration_qwen2_vl import Qwen2VLConfig
from ..qwen2_vl.modeling_qwen2_vl import (
    Qwen2VLCausalLMOutputWithPast,
    Qwen2VLForConditionalGeneration,
    Qwen2VLModel,
    Qwen2VLModelOutputWithPast,
    Qwen2VLRotaryEmbedding,
    VisionRotaryEmbedding,
)
from ..siglip.configuration_siglip import SiglipVisionConfig
from ..siglip.modeling_siglip import (
    SiglipMLP,
    SiglipVisionEmbeddings,
)
from ..video_llama_3.modeling_video_llama_3 import (
    VideoLlama3VisionAttention,
    VideoLlama3VisionEncoder,
    VideoLlama3VisionEncoderLayer,
)


logger = logging.get_logger(__name__)


def smart_resize(
    height: int,
    width: int,
    factor: int = 28,
    min_pixels: int = 384 * 384,
    max_pixels: int = 1536 * 1536,
):
    if height < factor:
        width = round((width * factor) / height)
        height = factor

    if width < factor:
        height = round((height * factor) / width)
        width = factor

    if max(height, width) / min(height, width) > 200:
        raise ValueError(
            f"absolute aspect ratio must be smaller than 200, got {max(height, width) / min(height, width)}"
        )
    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = max(factor, math.floor(height / beta / factor) * factor)
        w_bar = max(factor, math.floor(width / beta / factor) * factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar


class PaddleOCRVLImageProcessorKwargs(ImagesKwargs, total=False):
    r"""
    patch_size (`int`, *optional*, defaults to 14):
        The spatial patch size of the vision encoder.
    temporal_patch_size (`int`, *optional*, defaults to 1):
        The temporal patch size of the vision encoder.
    merge_size (`int`, *optional*, defaults to 2):
        The merge size of the vision encoder to llm encoder.
    """

    min_pixels: int
    max_pixels: int
    patch_size: int
    temporal_patch_size: int
    merge_size: int


class PaddleOCRVLImageProcessorPil(Qwen2VLImageProcessorPil):
    size = {"shortest_edge": 384 * 384, "longest_edge": 1536 * 1536}
    temporal_patch_size = 1

    def _preprocess(
        self,
        images: list[np.ndarray],
        do_resize: bool,
        size: SizeDict,
        resample: "PILImageResampling | None",
        do_rescale: bool,
        rescale_factor: float,
        do_normalize: bool,
        image_mean: float | list[float] | None,
        image_std: float | list[float] | None,
        patch_size: int,
        temporal_patch_size: int,
        merge_size: int,
        return_tensors: str | TensorType | None,
        **kwargs,
    ) -> BatchFeature:
        all_patches = []
        all_grids = []

        for image in images:
            height, width = image.shape[-2:]
            if do_resize:
                resized_height, resized_width = smart_resize(
                    height,
                    width,
                    factor=patch_size * merge_size,
                    min_pixels=size.shortest_edge,
                    max_pixels=size.longest_edge,
                )
                image = self.resize(
                    image,
                    size=SizeDict(height=resized_height, width=resized_width),
                    resample=resample,
                )
            else:
                resized_height, resized_width = height, width

            if do_rescale:
                image = self.rescale(image, rescale_factor)
            if do_normalize:
                image = self.normalize(image, image_mean, image_std)

            patches = np.expand_dims(image, axis=0)
            if patches.ndim == 4:
                patches = np.expand_dims(patches, axis=1)
            if patches.shape[1] % temporal_patch_size != 0:
                repeats = np.repeat(
                    patches[:, -1:], temporal_patch_size - (patches.shape[1] % temporal_patch_size), axis=1
                )
                patches = np.concatenate([patches, repeats], axis=1)

            batch_size = 1
            grid_t = patches.shape[1] // temporal_patch_size
            channel = patches.shape[2]
            grid_h, grid_w = resized_height // patch_size, resized_width // patch_size

            patches = patches.reshape(
                batch_size,
                grid_t,
                temporal_patch_size,
                channel,
                grid_h,
                patch_size,
                grid_w,
                patch_size,
            )
            patches = patches.transpose(0, 1, 4, 6, 3, 2, 5, 7)
            flatten_patches = patches.reshape(batch_size, grid_t * grid_h * grid_w, channel, patch_size, patch_size)

            all_patches.append(flatten_patches.squeeze(0))
            all_grids.append([grid_t, grid_h, grid_w])

        pixel_values = np.concatenate(all_patches, axis=0)
        image_grid_thw = np.array(all_grids, dtype=np.int64)

        return BatchFeature(
            data={"pixel_values": pixel_values, "image_grid_thw": image_grid_thw}, tensor_type=return_tensors
        )

    def get_number_of_image_patches(self, height: int, width: int, images_kwargs=None):
        """
        A utility that returns number of image patches for a given image size.

        Args:
            height (`int`):
                Height of the input image.
            width (`int`):
                Width of the input image.
            images_kwargs (`dict`, *optional*)
                Any kwargs to override defaults of the image processor.
        Returns:
            `int`: Number of image patches per image.
        """
        min_pixels = images_kwargs["min_pixels"] if "min_pixels" in images_kwargs else self.size["shortest_edge"]
        max_pixels = images_kwargs["max_pixels"] if "max_pixels" in images_kwargs else self.size["longest_edge"]
        patch_size = images_kwargs.get("patch_size", self.patch_size)
        merge_size = images_kwargs.get("merge_size", self.merge_size)

        factor = patch_size * merge_size
        resized_height, resized_width = smart_resize(
            height, width, factor, min_pixels=min_pixels, max_pixels=max_pixels
        )
        grid_h, grid_w = resized_height // patch_size, resized_width // patch_size
        return grid_h * grid_w


class PaddleOCRVLImageProcessor(Qwen2VLImageProcessor):
    size = {"shortest_edge": 384 * 384, "longest_edge": 1536 * 1536}
    temporal_patch_size = 1

    def _preprocess(
        self,
        images: list["torch.Tensor"],
        do_resize: bool,
        size: SizeDict,
        resample: "PILImageResampling | int | None",
        do_rescale: bool,
        rescale_factor: float,
        do_normalize: bool,
        image_mean: float | list[float] | None,
        image_std: float | list[float] | None,
        patch_size: int,
        temporal_patch_size: int,
        merge_size: int,
        disable_grouping: bool | None,
        return_tensors: str | TensorType | None,
        **kwargs,
    ) -> BatchFeature:
        grouped_images, grouped_images_index = group_images_by_shape(images, disable_grouping=disable_grouping)
        resized_images_grouped = {}
        for shape, stacked_images in grouped_images.items():
            height, width = stacked_images.shape[-2:]
            if do_resize:
                resized_height, resized_width = smart_resize(
                    height,
                    width,
                    factor=patch_size * merge_size,
                    min_pixels=size.shortest_edge,
                    max_pixels=size.longest_edge,
                )
                stacked_images = self.resize(
                    image=stacked_images,
                    size=SizeDict(height=resized_height, width=resized_width),
                    resample=resample,
                )
            resized_images_grouped[shape] = stacked_images
        resized_images = reorder_images(resized_images_grouped, grouped_images_index)

        grouped_images, grouped_images_index = group_images_by_shape(resized_images, disable_grouping=disable_grouping)
        processed_images_grouped = {}
        processed_grids = {}
        for shape, stacked_images in grouped_images.items():
            resized_height, resized_width = stacked_images.shape[-2:]
            patches = self.rescale_and_normalize(
                stacked_images, do_rescale, rescale_factor, do_normalize, image_mean, image_std
            )
            if patches.ndim == 4:
                patches = patches.unsqueeze(1)
            if patches.shape[1] % temporal_patch_size != 0:
                repeats = patches[:, -1:].repeat(1, temporal_patch_size - 1, 1, 1, 1)
                patches = torch.cat([patches, repeats], dim=1)

            batch_size, grid_t, channel = patches.shape[:3]
            grid_t = grid_t // temporal_patch_size
            grid_h, grid_w = resized_height // patch_size, resized_width // patch_size

            patches = patches.view(
                batch_size,
                grid_t,
                temporal_patch_size,
                channel,
                grid_h,
                patch_size,
                grid_w,
                patch_size,
            )
            patches = patches.permute(0, 1, 4, 6, 3, 2, 5, 7)
            flatten_patches = patches.reshape(batch_size, grid_t * grid_h * grid_w, channel, patch_size, patch_size)

            processed_images_grouped[shape] = flatten_patches
            processed_grids[shape] = [[grid_t, grid_h, grid_w]] * batch_size

        processed_images = reorder_images(processed_images_grouped, grouped_images_index)
        processed_grids = reorder_images(processed_grids, grouped_images_index)
        pixel_values = torch.cat(processed_images, dim=0)
        image_grid_thw = torch.tensor(processed_grids)

        return BatchFeature(
            data={"pixel_values": pixel_values, "image_grid_thw": image_grid_thw}, tensor_type=return_tensors
        )

    def get_number_of_image_patches(self, height: int, width: int, images_kwargs=None):
        """
        A utility that returns number of image patches for a given image size.

        Args:
            height (`int`):
                Height of the input image.
            width (`int`):
                Width of the input image.
            images_kwargs (`dict`, *optional*)
                Any kwargs to override defaults of the image processor.
        Returns:
            `int`: Number of image patches per image.
        """
        min_pixels = images_kwargs["min_pixels"] if "min_pixels" in images_kwargs else self.size["shortest_edge"]
        max_pixels = images_kwargs["max_pixels"] if "max_pixels" in images_kwargs else self.size["longest_edge"]
        patch_size = images_kwargs.get("patch_size", self.patch_size)
        merge_size = images_kwargs.get("merge_size", self.merge_size)

        factor = patch_size * merge_size
        resized_height, resized_width = smart_resize(
            height, width, factor, min_pixels=min_pixels, max_pixels=max_pixels
        )
        grid_h, grid_w = resized_height // patch_size, resized_width // patch_size
        return grid_h * grid_w


class PaddleOCRVLProcessorKwargs(ProcessingKwargs, total=False):
    _defaults = {
        "text_kwargs": {
            "padding": False,
            "return_mm_token_type_ids": True,
        },
    }


class PaddleOCRVLProcessor(ProcessorMixin):
    r"""
    [`PaddleOCRVLProcessor`] offers all the functionalities of [`PaddleOCRVLImageProcessor`] and [`LLamaTokenizerFast`]. See the
    [`~PaddleOCRVLProcessor.__call__`] and [`~PaddleOCRVLProcessor.decode`] for more information.
    Args:
        image_processor ([`PaddleOCRVLImageProcessor`], *optional*):
            The image processor is a required input.
        tokenizer ([`LLamaTokenizerFast`], *optional*):
            The tokenizer is a required input.
        chat_template (`str`, *optional*): A Jinja template which will be used to convert lists of messages
            in a chat into a tokenizable string.
    """

    image_processor_class = "AutoImageProcessor"
    tokenizer_class = "AutoTokenizer"

    def __init__(self, image_processor=None, tokenizer=None, chat_template=None, **kwargs):
        self.image_token = tokenizer.image_token
        self.image_token_id = tokenizer.image_token_id
        super().__init__(image_processor, tokenizer, chat_template=chat_template)

    def __call__(
        self,
        images: ImageInput = None,
        text: TextInput | PreTokenizedInput | list[TextInput] | list[PreTokenizedInput] = None,
        **kwargs: Unpack[PaddleOCRVLProcessorKwargs],
    ) -> BatchFeature:
        """
        Args:
            images (`PIL.Image.Image`, `np.ndarray`, `torch.Tensor`, `List[PIL.Image.Image]`, `List[np.ndarray]`, `List[torch.Tensor]`):
                The image or batch of images to be prepared. Each image can be a PIL image, NumPy array or PyTorch
                tensor. Both channels-first and channels-last formats are supported.
            text (`str`, `List[str]`, `List[List[str]]`):
                The sequence or batch of sequences to be encoded. Each sequence can be a string or a list of strings
                (pretokenized string). If the sequences are provided as list of strings (pretokenized), you must set
                `is_split_into_words=True` (to lift the ambiguity with a batch of sequences).
            return_tensors (`str` or [`~utils.TensorType`], *optional*):
                If set, will return tensors of a particular framework. Acceptable values are:
                - `'tf'`: Return TensorFlow `tf.constant` objects.
                - `'pt'`: Return PyTorch `torch.Tensor` objects.
                - `'np'`: Return NumPy `np.ndarray` objects.
                - `'jax'`: Return JAX `jnp.ndarray` objects.

        Returns:
            [`BatchFeature`]: A [`BatchFeature`] with the following fields:

            - **input_ids** -- List of token ids to be fed to a model. Returned when `text` is not `None`.
            - **attention_mask** -- List of indices specifying which tokens should be attended to by the model (when
              `return_attention_mask=True` or if *"attention_mask"* is in `self.model_input_names` and if `text` is not
              `None`).
            - **pixel_values** -- Pixel values to be fed to a model. Returned when `images` is not `None`.
            - **image_grid_thw** -- List of image 3D grid in LLM. Returned when `images` is not `None`.
        """
        output_kwargs = self._merge_kwargs(
            PaddleOCRVLProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            **kwargs,
        )

        if images is not None:
            image_inputs = self.image_processor(images=images, **output_kwargs["images_kwargs"])
            image_grid_thw = image_inputs["image_grid_thw"]

        else:
            image_inputs = {}
            image_grid_thw = None

        if not isinstance(text, list):
            text = [text]

        text = text.copy()

        if image_grid_thw is not None:
            index = 0
            for i in range(len(text)):
                while self.image_token in text[i]:
                    text[i] = text[i].replace(
                        self.image_token,
                        "<|placeholder|>"
                        * (
                            image_grid_thw[index].prod()
                            // self.image_processor.merge_size
                            // self.image_processor.merge_size
                        ),
                        1,
                    )
                    index += 1
                text[i] = text[i].replace("<|placeholder|>", self.image_token)

        return_tensors = output_kwargs["text_kwargs"].pop("return_tensors", None)
        return_mm_token_type_ids = output_kwargs["text_kwargs"].pop("return_mm_token_type_ids", False)
        text_inputs = self.tokenizer(text, **output_kwargs["text_kwargs"], return_tensors=None)

        if return_mm_token_type_ids:
            text_inputs["mm_token_type_ids"] = self.create_mm_token_type_ids(text_inputs["input_ids"])
        return BatchFeature(data={**text_inputs, **image_inputs}, tensor_type=return_tensors)


@auto_docstring(checkpoint="PaddlePaddle/PaddleOCR-VL")
@strict
class PaddleOCRVisionConfig(SiglipVisionConfig):
    r"""
    Example:

    ```python
    >>> from transformers import PaddleOCRVisionConfig, PaddleOCRVisionModel

    >>> # Initializing a PaddleOCRVisionConfig with PaddlePaddle/PaddleOCR-VL style configuration
    >>> configuration = PaddleOCRVisionConfig()

    >>> # Initializing a PaddleOCRVisionModel (with random weights) from the PaddlePaddle/PaddleOCR-VL style configuration
    >>> model = PaddleOCRVisionModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```
    """

    model_type = "paddleocr_vl_vision"
    base_config_key = "vision_config"

    hidden_size: int = 1152
    intermediate_size: int = 4304
    num_hidden_layers: int = 27
    num_attention_heads: int = 16
    image_size: int = 384
    patch_size: int = 14
    spatial_merge_size: int = 2


@auto_docstring(checkpoint="PaddlePaddle/PaddleOCR-VL")
@strict
class PaddleOCRTextConfig(Ernie4_5Config):
    model_type = "paddleocr_vl_text"


@auto_docstring(checkpoint="PaddlePaddle/PaddleOCR-VL")
@strict
class PaddleOCRVLConfig(Qwen2VLConfig):
    r"""
    Example:

    ```python
    >>> from transformers import PaddleOCRVLForConditionalGeneration, PaddleOCRVLConfig

    >>> # Initializing a PaddleOCRVL style configuration
    >>> configuration = PaddleOCRVLConfig()

    >>> # Initializing a model from the PaddleOCRVL style configuration
    >>> model = PaddleOCRVLForConditionalGeneration(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    sub_configs = {"vision_config": PaddleOCRVisionConfig, "text_config": PaddleOCRTextConfig}

    image_token_id: int = 100295
    video_token_id: int = 100296
    vision_start_token_id: int = 101305
    vision_end_token_id: int = 101306
    tie_word_embeddings: bool = True


class PaddleOCRProjector(nn.Module):
    def __init__(self, config: PaddleOCRVLConfig):
        super().__init__()
        self.merge_kernel_size = (config.vision_config.spatial_merge_size, config.vision_config.spatial_merge_size)

        hidden_size = config.vision_config.hidden_size * self.merge_kernel_size[0] * self.merge_kernel_size[1]

        self.pre_norm = torch.nn.LayerNorm(config.vision_config.hidden_size, eps=1e-05)
        self.linear_1 = nn.Linear(hidden_size, hidden_size, bias=True)
        self.act = GELUActivation()
        self.linear_2 = nn.Linear(hidden_size, config.text_config.hidden_size, bias=True)

    def forward(self, image_features: torch.Tensor, image_grid_thw: torch.Tensor) -> torch.Tensor:
        image_features_chunks = image_features.split(image_grid_thw.prod(dim=1).tolist(), dim=0)
        m1, m2 = self.merge_kernel_size

        processed_features = []
        for image_feature, image_grid in zip(image_features_chunks, image_grid_thw):
            image_feature = self.pre_norm(image_feature)
            t, h, w = image_grid
            d = image_feature.shape[-1]
            h_block = h // m1
            w_block = w // m2

            image_feature = image_feature.reshape(t, h_block, m1, w_block, m2, d)
            image_feature = image_feature.transpose(2, 3)
            image_feature = image_feature.reshape(t * h_block * w_block, m1 * m2 * d)

            hidden_states = self.linear_1(image_feature)
            hidden_states = self.act(hidden_states)
            hidden_states = self.linear_2(hidden_states)
            processed_features.append(hidden_states)

        return torch.cat(processed_features, dim=0)


class PaddleOCRVisionRotaryEmbedding(VisionRotaryEmbedding):
    pass


class PaddleOCRRotaryEmbedding(Qwen2VLRotaryEmbedding):
    pass


class PaddleOCRMLP(Ernie4_5MLP):
    def __init__(self, config: PaddleOCRTextConfig):
        super().__init__()


class PaddleOCRAttention(Qwen2_5OmniAttention):
    def __init__(self, config: PaddleOCRVLConfig, layer_idx: int | None = None):
        super().__init__()

        self.attention_dropout = 0.0
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=config.use_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.use_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.use_bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=config.use_bias)


class PaddleOCRRMSNorm(Ernie4_5RMSNorm):
    pass


class PaddleOCRDecoderLayer(Ernie4_5DecoderLayer):
    def __init__(self, config: PaddleOCRTextConfig, layer_idx: int):
        super().__init__()


@auto_docstring
class PaddleOCRVLPreTrainedModel(PreTrainedModel):
    config: PaddleOCRVLConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["PaddleOCRDecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn = True
    _supports_sdpa = True
    _supports_flex_attn = True

    _can_compile_fullgraph = True
    _supports_attention_backend = True

    _can_record_outputs = {
        "hidden_states": PaddleOCRDecoderLayer,
        "attentions": PaddleOCRAttention,
    }

    def _init_weights(self, module):
        super()._init_weights(module)
        if isinstance(module, PaddleOCRVisionEmbeddings):
            init.copy_(module.position_ids, torch.arange(module.position_ids.shape[-1]).expand((1, -1)))
        elif isinstance(module, PaddleOCRVisionRotaryEmbedding):
            inv_freq = 1.0 / (module.theta ** (torch.arange(0, module.dim, 2, dtype=torch.float) / module.dim))
            init.copy_(module.inv_freq, inv_freq)


class PaddleOCRTextModel(PaddleOCRVLPreTrainedModel, Ernie4_5Model):
    def __init__(self, config: PaddleOCRTextConfig):
        super().__init__(config)

    @merge_with_config_defaults
    @capture_outputs
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds: torch.Tensor = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)

        if position_ids is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device) + past_seen_tokens
            position_ids = position_ids.view(1, 1, -1).expand(3, inputs_embeds.shape[0], -1)
        elif position_ids.ndim == 2:
            position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)

        if position_ids.ndim == 3 and position_ids.shape[0] == 4:
            text_position_ids = position_ids[0]
            position_ids = position_ids[1:]
        else:
            text_position_ids = None

        causal_mask = create_causal_mask(
            config=self.config,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            position_ids=text_position_ids,
        )

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids=position_ids)

        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_embeddings=position_embeddings,
                position_ids=text_position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                **kwargs,
            )

        hidden_states = self.norm(hidden_states)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
        )


class PaddleOCRVisionEmbeddings(SiglipVisionEmbeddings):
    def __init__(self, config: PaddleOCRVisionConfig):
        super().__init__()

    def interpolate_pos_encoding(self, embeddings: torch.Tensor, height: int, width: int) -> torch.Tensor:
        num_positions = self.position_embedding.weight.shape[0]

        patch_pos_embed = self.position_embedding.weight.unsqueeze(0)

        dim = embeddings.shape[-1]

        sqrt_num_positions = torch_int(num_positions**0.5)
        patch_pos_embed = patch_pos_embed.reshape(1, sqrt_num_positions, sqrt_num_positions, dim)
        patch_pos_embed = patch_pos_embed.permute(0, 3, 1, 2)

        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed,
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        )

        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return patch_pos_embed

    def forward(
        self,
        pixel_values: torch.FloatTensor,
        image_grid_thw: list[tuple[int, int, int] | list[tuple[int, int, int]]] | None = None,
    ) -> torch.Tensor:
        """
        Args:
            pixel_values (`torch.FloatTensor` of shape `(batch_size, sequence_length, image_channels, patch_size, patch_size)`):
                The tensors corresponding to the input images.
            image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
                The temporal, height and width of feature shape of each image in LLM.
        """
        batch_size, squence_len, channel, height, width = pixel_values.shape
        target_dtype = self.patch_embedding.weight.dtype
        pixel_values = pixel_values.reshape(batch_size * squence_len, channel, height, width)
        patch_embeds = self.patch_embedding(pixel_values.to(dtype=target_dtype))  # shape = [*, width, grid, grid]
        embeddings = patch_embeds.flatten(-2).squeeze(-1)
        embeddings = embeddings.reshape(batch_size, squence_len, -1)

        start = 0
        embeddings = embeddings.squeeze(0)
        tmp_embeddings = []
        for image_grid in image_grid_thw:
            t, h, w = image_grid
            end = start + t * h * w
            image_embeddings = embeddings[start:end, :]
            position_embedding = self.interpolate_pos_encoding(image_embeddings, h, w).squeeze(0).repeat(t, 1)
            image_embeddings = image_embeddings + position_embedding
            tmp_embeddings.append(image_embeddings)
            start = end
        embeddings = torch.concat(tmp_embeddings, dim=0)

        return embeddings


class PaddleOCRVisionAttention(VideoLlama3VisionAttention):
    def __init__(self, config: PaddleOCRVisionConfig):
        super().__init__()


class PaddleOCRVisionMLP(SiglipMLP):
    def __init__(self, config: PaddleOCRVisionConfig):
        super().__init__()


class PaddleOCRVisionEncoderLayer(VideoLlama3VisionEncoderLayer):
    def __init__(self, config: PaddleOCRVisionConfig):
        super().__init__()


class PaddleOCRVisionEncoder(VideoLlama3VisionEncoder):
    def __init__(self, config: PaddleOCRVisionConfig):
        super().__init__()
        embed_dim = config.hidden_size
        num_heads = config.num_attention_heads
        head_dim = embed_dim // num_heads
        self.rotary_pos_emb = PaddleOCRVisionRotaryEmbedding(head_dim // 2)

    def forward(
        self,
        inputs_embeds: torch.FloatTensor,
        cu_seqlens: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        image_grid_thw: list[tuple[int, int, int] | list[tuple[int, int, int]]] | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutput:
        r"""
        inputs_embeds (`torch.FloatTensor` of shape `(sequence_length, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation.
            This is useful if you want more control over how to convert `input_ids` indices into associated vectors
            than the model's internal embedding lookup matrix.
        cu_seqlens (`torch.Tensor` of shape `(num_images + 1,)`):
            The cumulative sequence lengths of each image or video feature.
        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            The attention_mask used in forward function shape [batch_size X sequence_length] if not None.
        image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
            The temporal, height and width of feature shape of each image in LLM.
        """
        device = inputs_embeds.device
        hidden_states = inputs_embeds
        attention_mask = create_bidirectional_mask(
            config=self.config,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
        )
        split_hids = []
        split_wids = []
        for t, h, w in image_grid_thw:
            image_pids = torch.arange(t * h * w, device=device) % (h * w)
            sample_hids = image_pids // w
            sample_wids = image_pids % w
            split_hids.append(sample_hids)
            split_wids.append(sample_wids)
        width_position_ids = torch.concat(split_wids, dim=0)
        height_position_ids = torch.concat(split_hids, dim=0)

        pids = torch.stack([height_position_ids, width_position_ids], dim=-1)
        max_grid_size = pids.max() + 1
        rotary_embeddings_max_grid = self.rotary_pos_emb(max_grid_size)
        rotary_embeddings = rotary_embeddings_max_grid[pids].flatten(1)
        rotary_embeddings = rotary_embeddings.repeat(1, 2)
        position_embeddings = (rotary_embeddings.cos(), rotary_embeddings.sin())

        for encoder_layer in self.layers:
            hidden_states = encoder_layer(
                hidden_states,
                cu_seqlens=cu_seqlens,
                position_embeddings=position_embeddings,
                **kwargs,
            )

        return BaseModelOutput(
            last_hidden_state=hidden_states,
        )


class PaddleOCRVisionTransformer(PaddleOCRVLPreTrainedModel):
    config: PaddleOCRVisionConfig
    main_input_name = "pixel_values"
    input_modalities = "image"
    _can_record_outputs = {
        "hidden_states": PaddleOCRVisionEncoderLayer,
        "attentions": PaddleOCRVisionAttention,
    }

    def __init__(self, config: PaddleOCRVisionConfig):
        super().__init__(config)
        self.config = config
        embed_dim = config.hidden_size

        self.embeddings = PaddleOCRVisionEmbeddings(config)
        self.encoder = PaddleOCRVisionEncoder(config)
        self.post_layernorm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)

        self.post_init()

    @merge_with_config_defaults
    @capture_outputs(tie_last_hidden_states=False)
    def forward(
        self,
        pixel_values: torch.FloatTensor,
        cu_seqlens: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        image_grid_thw: list[tuple[int, int, int] | list[tuple[int, int, int]]] | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPooling:
        """
        Args:
            pixel_values (`torch.FloatTensor` of shape `(batch_size, sequence_length, patch_size * patch_size * image_channels)`):
                The tensors corresponding to the input images.
            cu_seqlens (`torch.Tensor` of shape `(num_images + 1,)`):
                The cumulative sequence lengths of each image or video feature.
            attention_mask (`torch.Tensor`, *optional*):
                The attention_mask used in forward function shape [batch_size X sequence_length] if not None.
            image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
                The temporal, height and width of feature shape of each image in LLM.
        """
        hidden_states = self.embeddings(pixel_values, image_grid_thw=image_grid_thw)

        encoder_outputs: BaseModelOutput = self.encoder(
            inputs_embeds=hidden_states,
            cu_seqlens=cu_seqlens,
            attention_mask=attention_mask,
            image_grid_thw=image_grid_thw,
            **kwargs,
        )

        last_hidden_state = encoder_outputs.last_hidden_state
        last_hidden_state = self.post_layernorm(last_hidden_state)

        return BaseModelOutputWithPooling(
            last_hidden_state=last_hidden_state,
            pooler_output=None,
        )


class PaddleOCRVisionModel(PaddleOCRVLPreTrainedModel):
    config: PaddleOCRVisionConfig
    main_input_name = "pixel_values"
    input_modalities = "image"

    def __init__(self, config: PaddleOCRVisionConfig):
        super().__init__(config)

        self.vision_model = PaddleOCRVisionTransformer(config)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        pixel_values: torch.FloatTensor,
        cu_seqlens: torch.Tensor,
        image_grid_thw: list[tuple[int, int, int] | list[tuple[int, int, int]]] | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | BaseModelOutputWithPooling:
        """
        Args:
            pixel_values (`torch.FloatTensor` of shape `(batch_size, sequence_length, image_channels, patch_size, patch_size)`):
                The tensors corresponding to the input images.
            cu_seqlens (`torch.Tensor` of shape `(num_images + 1,)`):
                The cumulative sequence lengths of each image or video feature.
            image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
                The temporal, height and width of feature shape of each image in LLM.
        """
        return self.vision_model(
            pixel_values=pixel_values,
            cu_seqlens=cu_seqlens,
            image_grid_thw=image_grid_thw,
            **kwargs,
        )


class PaddleOCRVLModelOutputWithPast(Qwen2VLModelOutputWithPast):
    pass


class PaddleOCRVLCausalLMOutputWithPast(Qwen2VLCausalLMOutputWithPast):
    pass


class PaddleOCRVLModel(Qwen2VLModel):
    _keys_to_ignore_on_load_unexpected = ["packing_position_embedding", "vision_model.head"]

    def __init__(self, config: PaddleOCRVLConfig):
        super().__init__(config)
        self.visual = PaddleOCRVisionModel._from_config(config.vision_config)
        self.projector = PaddleOCRProjector(config)
        self.language_model = PaddleOCRTextModel._from_config(config.text_config)
        self.rope_deltas = None

        self.post_init()

    def get_input_embeddings(self):
        return self.language_model.embed_tokens

    def set_input_embeddings(self, value):
        self.language_model.embed_tokens = value

    def get_video_features(self):
        raise AttributeError("PaddleOCRVLModel does not support video.")

    @can_return_tuple
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
        pixel_values = pixel_values.type(self.visual.dtype).unsqueeze(0)
        cu_seqlens = torch.repeat_interleave(image_grid_thw[:, 1] * image_grid_thw[:, 2], image_grid_thw[:, 0]).cumsum(
            dim=0,
            # Select dtype based on the following factors:
            #  - FA2 requires that cu_seqlens_q must have dtype int32
            #  - torch.onnx.export requires that cu_seqlens_q must have same dtype as grid_thw
            # See https://github.com/huggingface/transformers/pull/34852 for more information
            dtype=image_grid_thw.dtype if torch.jit.is_tracing() else torch.int32,
        )
        cu_seqlens = torch.nn.functional.pad(cu_seqlens, (1, 0), value=0)
        vision_outputs = self.visual(
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            cu_seqlens=cu_seqlens,
            return_dict=True,
            **kwargs,
        )
        image_embeds = vision_outputs.last_hidden_state
        image_embeds = self.projector(image_embeds, image_grid_thw)
        vision_outputs.pooler_output = image_embeds

        return vision_outputs

    def get_placeholder_mask(
        self, input_ids: torch.LongTensor, inputs_embeds: torch.FloatTensor, image_features: torch.FloatTensor
    ):
        """
        Obtains multimodal placeholder mask from `input_ids` or `inputs_embeds`, and checks that the placeholder token count is
        equal to the length of multimodal features. If the lengths are different, an error is raised.
        """
        if input_ids is None:
            special_image_mask = inputs_embeds == self.get_input_embeddings()(
                torch.tensor(self.config.image_token_id, dtype=torch.long, device=inputs_embeds.device)
            )
            special_image_mask = special_image_mask.all(-1)
        else:
            special_image_mask = input_ids == self.config.image_token_id

        n_image_tokens = special_image_mask.sum()
        special_image_mask = special_image_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
        n_image_features = image_features.shape[0] * image_features.shape[1]
        torch_compilable_check(
            inputs_embeds[special_image_mask].numel() == image_features.numel(),
            f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}",
        )
        return special_image_mask

    @can_return_tuple
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: list[torch.FloatTensor] | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        pixel_values: torch.Tensor | None = None,
        image_grid_thw: torch.LongTensor | None = None,
        mm_token_type_ids: torch.IntTensor | None = None,
        rope_deltas: torch.LongTensor | None = None,
        **kwargs,
    ) -> tuple | PaddleOCRVLModelOutputWithPast:
        r"""
        image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
            The temporal, height and width of feature shape of each image in LLM.
        rope_deltas (`torch.LongTensor` of shape `(batch_size, )`, *optional*):
            The rope index difference between sequence length and multimodal rope.
        """
        if inputs_embeds is None:
            inputs_embeds = self.language_model.embed_tokens(input_ids)

        if pixel_values is not None:
            image_embeds = self.get_image_features(pixel_values, image_grid_thw, return_dict=True).pooler_output
            image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
            image_mask = self.get_placeholder_mask(input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds)
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

        if position_ids is None:
            position_ids = self.compute_3d_position_ids(
                input_ids=input_ids,
                image_grid_thw=image_grid_thw,
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                mm_token_type_ids=mm_token_type_ids,
            )

        outputs = self.language_model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            **kwargs,
        )

        output = PaddleOCRVLModelOutputWithPast(
            last_hidden_state=outputs.last_hidden_state,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            rope_deltas=self.rope_deltas,
        )

        return output


class PaddleOCRVLForConditionalGeneration(Qwen2VLForConditionalGeneration):
    _keys_to_ignore_on_load_unexpected = ["packing_position_embedding", "vision_model.head"]

    def get_video_features(self):
        raise AttributeError("PaddleOCRVLForConditionalGeneration does not support video.")

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        pixel_values: torch.Tensor | None = None,
        image_grid_thw: torch.LongTensor | None = None,
        rope_deltas: torch.LongTensor | None = None,
        mm_token_type_ids: torch.IntTensor | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | PaddleOCRVLCausalLMOutputWithPast:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
        image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
            The temporal, height and width of feature shape of each image in LLM.
        rope_deltas (`torch.LongTensor` of shape `(batch_size, )`, *optional*):
            The rope index difference between sequence length and multimodal rope.

        Example:

        ```python
        >>> from transformers import AutoProcessor, PaddleOCRVLForConditionalGeneration

        >>> model = PaddleOCRVLForConditionalGeneration.from_pretrained("PaddlePaddle/PaddleOCR-VL", dtype="bfloat16")
        >>> processor = AutoProcessor.from_pretrained("PaddlePaddle/PaddleOCR-VL")

        >>> messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": "https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/ocr_demo.jpg",
                    },
                    {"type": "text", "text": "OCR:"},
                ],
            }
        ]

        >>> inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt"
        ).to(model.device)

        >>> # Generate
        >>> generated_ids = model.generate(**inputs, max_new_tokens=1024)
        >>> generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
        >>> output_text = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        >>> print(output_text)
        ```
        """
        outputs: PaddleOCRVLModelOutputWithPast = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            image_grid_thw=image_grid_thw,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            pixel_values=pixel_values,
            rope_deltas=rope_deltas,
            mm_token_type_ids=mm_token_type_ids,
            **kwargs,
        )
        hidden_states = outputs.last_hidden_state

        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(
                logits=logits, labels=labels, vocab_size=self.config.text_config.vocab_size, **kwargs
            )

        return PaddleOCRVLCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            rope_deltas=outputs.rope_deltas,
        )


__all__ = [
    "PaddleOCRVLForConditionalGeneration",
    "PaddleOCRVLModel",
    "PaddleOCRVLPreTrainedModel",
    "PaddleOCRVisionTransformer",
    "PaddleOCRVLConfig",
    "PaddleOCRTextModel",
    "PaddleOCRVisionModel",
    "PaddleOCRVisionConfig",
    "PaddleOCRTextConfig",
    "PaddleOCRVLImageProcessor",
    "PaddleOCRVLImageProcessorPil",
    "PaddleOCRVLProcessor",
]
