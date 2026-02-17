# Copyright 2022 The HuggingFace Inc. team.
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
"""
Processing saving/loading class for common processors.
"""

import bisect
import copy
import inspect
import json
import os
import sys
import typing
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, Literal, TypedDict, TypeVar, Union

import numpy as np
import typing_extensions
from huggingface_hub import create_repo, is_offline_mode
from huggingface_hub.dataclasses import validate_typed_dict
from huggingface_hub.errors import EntryNotFoundError

from .audio_utils import AudioInput, load_audio
from .dynamic_module_utils import custom_object_save
from .feature_extraction_utils import BatchFeature
from .image_utils import ChannelDimension, ImageInput, is_vision_available
from .tokenization_utils_base import (
    PaddingStrategy,
    PreTokenizedInput,
    PreTrainedTokenizerBase,
    TextInput,
    TruncationStrategy,
)
from .utils import (
    AUDIO_TOKENIZER_NAME,
    CHAT_TEMPLATE_DIR,
    CHAT_TEMPLATE_FILE,
    LEGACY_PROCESSOR_CHAT_TEMPLATE_FILE,
    PROCESSOR_NAME,
    PushToHubMixin,
    TensorType,
    cached_file,
    copy_func,
    direct_transformers_import,
    is_torch_available,
    list_repo_templates,
    logging,
)
from .utils.chat_template_utils import render_jinja_template
from .utils.type_validators import (
    device_validator,
    image_size_validator,
    padding_validator,
    positive_any_number,
    positive_int,
    resampling_validator,
    tensor_type_validator,
    truncation_validator,
    video_metadata_validator,
)
from .video_utils import VideoInput, VideoMetadataType


if is_torch_available():
    import torch

    from .modeling_utils import PreTrainedAudioTokenizerBase

if is_vision_available():
    from .image_utils import PILImageResampling

logger = logging.get_logger(__name__)

# type hinting: specifying the type of processor class that inherits from ProcessorMixin
SpecificProcessorType = TypeVar("SpecificProcessorType", bound="ProcessorMixin")

# Dynamically import the Transformers module to grab the attribute classes of the processor from their names.
transformers_module = direct_transformers_import(Path(__file__).parent)


class _LazyAutoProcessorMapping(dict):
    """
    Lazy dictionary to avoid circular imports.
    The mapping names are only imported when accessed.
    """

    _MAPPING_NAMES = {
        "image_processor": ("transformers.models.auto.image_processing_auto", "AutoImageProcessor"),
        "video_processor": ("transformers.models.auto.video_processing_auto", "AutoVideoProcessor"),
        "feature_extractor": ("transformers.models.auto.feature_extraction_auto", "AutoFeatureExtractor"),
        "audio_processor": ("transformers.models.auto.feature_extraction_auto", "AutoFeatureExtractor"),
        "tokenizer": ("transformers.models.auto.tokenization_auto", "AutoTokenizer"),
    }

    def __getitem__(self, key):
        if key not in self._MAPPING_NAMES:
            raise KeyError(key)
        module_name, attr_name = self._MAPPING_NAMES[key]
        module = __import__(module_name, fromlist=[attr_name])
        return getattr(module, attr_name)

    def __contains__(self, key):
        return key in self._MAPPING_NAMES

    def keys(self):
        return self._MAPPING_NAMES.keys()


MODALITY_TO_AUTOPROCESSOR_MAPPING = _LazyAutoProcessorMapping()

MODALITY_TO_BASE_CLASS_MAPPING = {
    "audio_tokenizer": (
        "HiggsAudioV2TokenizerModel",
        "DacModel",
    ),  # TODO: @eustlb, to be replaced with PreTrainedAudioTokenizerBase
    "audio_processor": "FeatureExtractionMixin",
    "tokenizer": ("PreTrainedTokenizerBase", "MistralCommonBackend"),
    "feature_extractor": "FeatureExtractionMixin",
    "image_processor": "ImageProcessingMixin",
    "video_processor": "BaseVideoProcessor",
}


def _get_modality_for_attribute(attribute_name: str) -> str:
    """
    Get the canonical modality type for a given attribute name.

    For example:
    - "image_processor" -> "image_processor"
    - "encoder_image_processor" -> "image_processor"
    - "text_tokenizer" -> "tokenizer"
    - "my_feature_extractor" -> "feature_extractor"
    """
    for modality in MODALITY_TO_AUTOPROCESSOR_MAPPING.keys():
        if modality in attribute_name:
            return modality
    raise ValueError(
        f"Cannot determine modality for attribute '{attribute_name}'. "
        f"Attribute name must contain one of: {list(MODALITY_TO_AUTOPROCESSOR_MAPPING.keys())}"
    )


if sys.version_info >= (3, 11):
    Unpack = typing.Unpack
else:
    Unpack = typing_extensions.Unpack


class TextKwargs(TypedDict, total=False):
    """
    Keyword arguments for text processing. For extended documentation, check out tokenization_utils_base methods and
    docstrings associated.

    Attributes:
        add_special_tokens (`bool`, *optional*)
            Whether or not to add special tokens when encoding the sequences.
        padding (`bool`, `str` or [`~utils.PaddingStrategy`], *optional*)
            Activates and controls padding.
        truncation (`bool`, `str` or [`~tokenization_utils_base.TruncationStrategy`], *optional*):
            Activates and controls truncation.
        max_length (`int`, *optional*):
            Controls the maximum length to use by one of the truncation/padding parameters.
        stride (`int`, *optional*):
            If set, the overflowing tokens will contain some tokens from the end of the truncated sequence.
        is_split_into_words (`bool`, *optional*):
            Whether or not the input is already pre-tokenized.
        pad_to_multiple_of (`int`, *optional*):
            If set, will pad the sequence to a multiple of the provided value.
        return_token_type_ids (`bool`, *optional*):
            Whether to return token type IDs.
        return_attention_mask (`bool`, *optional*):
            Whether to return the attention mask.
        return_overflowing_tokens (`bool`, *optional*):
            Whether or not to return overflowing token sequences.
        return_special_tokens_mask (`bool`, *optional*):
            Whether or not to return special tokens mask information.
        return_offsets_mapping (`bool`, *optional*):
            Whether or not to return `(char_start, char_end)` for each token.
        return_length (`bool`, *optional*):
            Whether or not to return the lengths of the encoded inputs.
        verbose (`bool`, *optional*):
            Whether or not to print more information and warnings.
        padding_side (`str`, *optional*):
            The side on which padding will be applied.
        return_mm_token_type_ids (`bool`, *optional*):
            Whether to return multimodal token type ids indicating mm placeholder token positions.
        return_tensors (`str` or [`~utils.TensorType`], *optional*):
            If set, will return tensors of a particular framework. Acceptable values are:
            - `'pt'`: Return PyTorch `torch.Tensor` objects.
            - `'np'`: Return NumPy `np.ndarray` objects.
    """

    text_pair: TextInput | PreTokenizedInput | list[TextInput] | list[PreTokenizedInput] | None
    text_target: TextInput | PreTokenizedInput | list[TextInput] | list[PreTokenizedInput] | None
    text_pair_target: TextInput | PreTokenizedInput | list[TextInput] | list[PreTokenizedInput] | None
    add_special_tokens: bool | None
    padding: Annotated[bool | str | PaddingStrategy | None, padding_validator()]
    truncation: Annotated[bool | str | TruncationStrategy | None, truncation_validator()]
    max_length: Annotated[int | None, positive_int()]
    stride: Annotated[int | None, positive_int()]
    is_split_into_words: bool | None
    pad_to_multiple_of: Annotated[int | None, positive_int()]
    return_token_type_ids: bool | None
    return_attention_mask: bool | None
    return_overflowing_tokens: bool | None
    return_special_tokens_mask: bool | None
    return_offsets_mapping: bool | None
    return_length: bool | None
    verbose: bool | None
    padding_side: Literal["left", "right"] | None
    return_mm_token_type_ids: bool | None
    return_tensors: Annotated[str | TensorType | None, tensor_type_validator()]


class ImagesKwargs(TypedDict, total=False):
    """
    Keyword arguments for image processing. For extended documentation, check the appropriate ImageProcessor
    class methods and docstrings.

    Attributes:
        do_convert_rgb (`bool`):
            Whether to convert the image to RGB format.
        do_resize (`bool`, *optional*):
            Whether to resize the image.
        size (`dict[str, int]`, *optional*):
            Resize the shorter side of the input to `size["shortest_edge"]`.
        crop_size (`dict[str, int]`, *optional*):
            Desired output size when applying center-cropping.
        resample (`PILImageResampling`, *optional*):
            Resampling filter to use if resizing the image.
        do_rescale (`bool`, *optional*):
            Whether to rescale the image by the specified scale `rescale_factor`.
        rescale_factor (`int` or `float`, *optional*):
            Scale factor to use if rescaling the image.
        do_normalize (`bool`, *optional*):
            Whether to normalize the image.
        image_mean (`float` or `list[float] or tuple[float, float, float]`, *optional*):
            Mean to use if normalizing the image.
        image_std (`float` or `list[float] or tuple[float, float, float]`, *optional*):
            Standard deviation to use if normalizing the image.
        do_pad (`bool`, *optional*):
            Whether to pad the images in the batch.
        pad_size (`dict[str, int]`, *optional*):
            The size `{"height": int, "width" int}` to pad the images to.
        do_center_crop (`bool`, *optional*):
            Whether to center crop the image.
        data_format (`ChannelDimension` or `str`, *optional*):
            The channel dimension format for the output image.
        input_data_format (`ChannelDimension` or `str`, *optional*):
            The channel dimension format for the input image.
        device (`Union[str, torch.Tensor]`, *optional*):
            The device to use for processing (e.g. "cpu", "cuda"), only relevant for fast image processing.
        return_tensors (`str` or [`~utils.TensorType`], *optional*):
            If set, will return tensors of a particular framework. Acceptable values are:
            - `'pt'`: Return PyTorch `torch.Tensor` objects.
            - `'np'`: Return NumPy `np.ndarray` objects.
        disable_grouping (`bool`, *optional*):
            Whether to group images by shapes when processing or not, only relevant for fast image processing.
        image_seq_length (`int`, *optional*):
            The number of image tokens to be used for each image in the input.
            Added for backward compatibility but this should be set as a processor attribute in future models.
    """

    do_convert_rgb: bool | None
    do_resize: bool | None
    size: Annotated[int | list[int] | tuple[int, ...] | dict[str, int] | None, image_size_validator()]
    crop_size: Annotated[int | list[int] | tuple[int, ...] | dict[str, int] | None, image_size_validator()]
    resample: Annotated[Union["PILImageResampling", int] | None, resampling_validator()]
    do_rescale: bool | None
    rescale_factor: float | None
    do_normalize: bool | None
    image_mean: float | list[float] | tuple[float, ...] | None
    image_std: float | list[float] | tuple[float, ...] | None
    do_pad: bool | None
    pad_size: Annotated[int | list[int] | tuple[int, ...] | dict[str, int] | None, image_size_validator()]
    do_center_crop: bool | None
    data_format: str | ChannelDimension | None
    input_data_format: str | ChannelDimension | None
    device: Annotated[Union[str, "torch.device"] | None, device_validator()]
    return_tensors: Annotated[str | TensorType | None, tensor_type_validator()]
    disable_grouping: bool | None
    image_seq_length: int | None


class VideosKwargs(TypedDict, total=False):
    """
    Keyword arguments for video processing.

    Attributes:
        do_convert_rgb (`bool`):
            Whether to convert the video to RGB format.
        do_resize (`bool`):
            Whether to resize the video.
        size (`dict[str, int]`, *optional*):
            Resize the shorter side of the input to `size["shortest_edge"]`.
        default_to_square (`bool`, *optional*, defaults to `self.default_to_square`):
            Whether to default to a square when resizing, if size is an int.
        resample (`PILImageResampling`, *optional*):
            Resampling filter to use if resizing the video.
        do_rescale (`bool`, *optional*):
            Whether to rescale the video by the specified scale `rescale_factor`.
        rescale_factor (`int` or `float`, *optional*):
            Scale factor to use if rescaling the video.
        do_normalize (`bool`, *optional*):
            Whether to normalize the video.
        image_mean (`float` or `list[float] or tuple[float, float, float]`, *optional*):
            Mean to use if normalizing the video.
        image_std (`float` or `list[float] or tuple[float, float, float]`, *optional*):
            Standard deviation to use if normalizing the video.
        do_center_crop (`bool`, *optional*):
            Whether to center crop the video.
        do_pad (`bool`, *optional*):
            Whether to pad the images in the batch.
        do_sample_frames (`bool`, *optional*):
            Whether to sample frames from the video before processing or to process the whole video.
        video_metadata (`Union[VideoMetadata, dict]`, *optional*):
            Metadata of the video containing information about total duration, fps and total number of frames.
        num_frames (`int`, *optional*):
            Maximum number of frames to sample when `do_sample_frames=True`.
        fps (`int` or `float`, *optional*):
            Target frames to sample per second when `do_sample_frames=True`.
        crop_size (`dict[str, int]`, *optional*):
            Desired output size when applying center-cropping.
        data_format (`ChannelDimension` or `str`, *optional*):
            The channel dimension format for the output video.
        input_data_format (`ChannelDimension` or `str`, *optional*):
            The channel dimension format for the input video.
        device (`Union[str, torch.Tensor]`, *optional*):
            The device to use for processing (e.g. "cpu", "cuda"), only relevant for fast image processing.
        return_metadata (`bool`, *optional*):
            Whether to return video metadata or not.
        return_tensors (`str` or [`~utils.TensorType`], *optional*):
            If set, will return tensors of a particular framework. Acceptable values are:
            - `'pt'`: Return PyTorch `torch.Tensor` objects.
            - `'np'`: Return NumPy `np.ndarray` objects.
    """

    do_convert_rgb: bool | None
    do_resize: bool | None
    size: Annotated[int | list[int] | tuple[int, ...] | dict[str, int] | None, image_size_validator()]
    default_to_square: bool | None
    resample: Annotated[Union["PILImageResampling", int] | None, resampling_validator()]
    do_rescale: bool | None
    rescale_factor: float | None
    do_normalize: bool | None
    image_mean: float | list[float] | tuple[float, ...] | None
    image_std: float | list[float] | tuple[float, ...] | None
    do_center_crop: bool | None
    do_pad: bool | None
    crop_size: Annotated[int | list[int] | tuple[int, ...] | dict[str, int] | None, image_size_validator()]
    data_format: str | ChannelDimension | None
    input_data_format: str | ChannelDimension | None
    device: Annotated[Union[str, "torch.device"] | None, device_validator()]
    do_sample_frames: bool | None
    video_metadata: Annotated[VideoMetadataType | None, video_metadata_validator()]
    fps: Annotated[int | float | None, positive_any_number()]
    num_frames: Annotated[int | None, positive_int()]
    return_metadata: bool | None
    return_tensors: Annotated[str | TensorType | None, tensor_type_validator()]


class AudioKwargs(TypedDict, total=False):
    """
    Keyword arguments for audio processing.

    Attributes:
        sampling_rate (`int`, *optional*):
            The sampling rate at which the `raw_speech` input was sampled.
        raw_speech (`np.ndarray`, `list[float]`, `list[np.ndarray]`, `list[list[float]]`):
            The sequence or batch of sequences to be padded. Each sequence can be a numpy array, a list of float
            values, a list of numpy arrays or a list of list of float values. Must be mono channel audio, not
            stereo, i.e. single float per timestep.
        padding (`bool`, `str` or [`~utils.PaddingStrategy`], *optional*):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding
            index) among:

            - `True` or `'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
                sequence if provided).
            - `'max_length'`: Pad to a maximum length specified with the argument `max_length` or to the maximum
                acceptable input length for the model if that argument is not provided.
            - `False` or `'do_not_pad'`
        max_length (`int`, *optional*):
            Maximum length of the returned list and optionally padding length (see above).
        truncation (`bool`, *optional*):
            Activates truncation to cut input sequences longer than *max_length* to *max_length*.
        pad_to_multiple_of (`int`, *optional*):
            If set, will pad the sequence to a multiple of the provided value.
        return_attention_mask (`bool`, *optional*):
            Whether or not [`~ASTFeatureExtractor.__call__`] should return `attention_mask`.
        return_tensors (`str` or [`~utils.TensorType`], *optional*):
            If set, will return tensors of a particular framework. Acceptable values are:
            - `'pt'`: Return PyTorch `torch.Tensor` objects.
            - `'np'`: Return NumPy `np.ndarray` objects.
    """

    sampling_rate: Annotated[int | None, positive_int()]
    raw_speech: Union["np.ndarray", list[float], list["np.ndarray"], list[list[float]]] | None
    padding: Annotated[bool | str | PaddingStrategy | None, padding_validator()]
    max_length: Annotated[int | None, positive_int()]
    truncation: Annotated[bool | str | TruncationStrategy | None, truncation_validator()]
    pad_to_multiple_of: Annotated[int | None, positive_int()]
    return_attention_mask: bool | None
    return_tensors: Annotated[str | TensorType | None, tensor_type_validator()]


class ProcessingKwargs(TypedDict, total=False):
    """
    Base class for kwargs passing to processors.
    In case a model has specific kwargs that are not present in the base class or default values for existing keys,
    it should have its own `ModelProcessorKwargs` class that inherits from `ProcessingKwargs` to provide:
        1) Additional typed keys and that this model requires to process inputs.
        2) Default values for existing keys under a `_defaults` attribute.
    New keys have to be defined as follows to ensure type hinting is done correctly.

    ```python
    # adding a new image kwarg for this model
    class ModelImagesKwargs(ImagesKwargs, total=False):
        new_image_kwarg: Optional[bool]

    class ModelProcessorKwargs(ProcessingKwargs, total=False):
        images_kwargs: ModelImagesKwargs
        _defaults = {
            "images_kwargs: {
                "new_image_kwarg": False,
            }
            "text_kwargs": {
                "padding": "max_length",
            },
        }

    ```

    For Python 3.8 compatibility, when inheriting from this class and overriding one of the kwargs,
    you need to manually update the __annotations__ dictionary. This can be done as follows:

    ```python
    class CustomProcessorKwargs(ProcessingKwargs, total=False):
        images_kwargs: CustomImagesKwargs

    CustomProcessorKwargs.__annotations__["images_kwargs"] = CustomImagesKwargs  # python 3.8 compatibility
    ```python

    """

    _defaults = {}

    text_kwargs: TextKwargs = {
        **TextKwargs.__annotations__,
    }
    images_kwargs: ImagesKwargs = {
        **ImagesKwargs.__annotations__,
    }
    videos_kwargs: VideosKwargs = {
        **VideosKwargs.__annotations__,
    }
    audio_kwargs: AudioKwargs = {
        **AudioKwargs.__annotations__,
    }


class TokenizerChatTemplateKwargs(TypedDict, total=False):
    """
    Keyword arguments for tokenizer's `apply_chat_template`, when it is called from within a processor.

    tools (`list[Dict]`, *optional*):
        A list of tools (callable functions) that will be accessible to the model. If the template does not
        support function calling, this argument will have no effect. Each tool should be passed as a JSON Schema,
        giving the name, description and argument types for the tool. See our
        [chat templating guide](https://huggingface.co/docs/transformers/main/en/chat_templating#automated-function-conversion-for-tool-use)
        for more information.
    documents (`list[dict[str, str]]`, *optional*):
        A list of dicts representing documents that will be accessible to the model if it is performing RAG
        (retrieval-augmented generation). If the template does not support RAG, this argument will have no
        effect. We recommend that each document should be a dict containing "title" and "text" keys. Please
        see the RAG section of the [chat templating guide](https://huggingface.co/docs/transformers/main/en/chat_templating#arguments-for-RAG)
        for examples of passing documents with chat templates.
    add_generation_prompt (bool, *optional*):
        If this is set, a prompt with the token(s) that indicate
        the start of an assistant message will be appended to the formatted output. This is useful when you want to generate a response from the model.
        Note that this argument will be passed to the chat template, and so it must be supported in the
        template for this argument to have any effect.
    continue_final_message (bool, *optional*):
        If this is set, the chat will be formatted so that the final
        message in the chat is open-ended, without any EOS tokens. The model will continue this message
        rather than starting a new one. This allows you to "prefill" part of
        the model's response for it. Cannot be used at the same time as `add_generation_prompt`.
    return_assistant_tokens_mask (`bool`, defaults to `False`):
        Whether to return a mask of the assistant generated tokens. For tokens generated by the assistant,
        the mask will contain 1. For user and system tokens, the mask will contain 0.
        This functionality is only available for chat templates that support it via the `{% generation %}` keyword.
    """

    tools: list[dict] | None = None
    documents: list[dict[str, str]] | None = None
    add_generation_prompt: bool | None = False
    continue_final_message: bool | None = False
    return_assistant_tokens_mask: bool | None = False


class ProcessorChatTemplateKwargs(TokenizerChatTemplateKwargs, total=False):
    """
    Keyword arguments for processor's `apply_chat_template`.

    tokenize (`bool`, *optional*, defaults to `False`):
        Whether to tokenize the output or not.
    return_dict (`bool`, defaults to `False`):
        Whether to return a dictionary with named outputs. Has no effect if tokenize is `False`.
    load_audio_from_video (`bool`, *optional*, defaults to `False`):
        Whether to use the audio track of input video. If `True` the audio track will be loaded and passed to the
        processor. This flag has no effect if the model doesn't support audio modality.
    """

    tokenize: bool | None = False
    return_dict: bool | None = False
    load_audio_from_video: bool | None = False


class AllKwargsForChatTemplate(TypedDict, total=False):
    processor_kwargs: ProcessingKwargs
    template_kwargs: ProcessorChatTemplateKwargs


@dataclass
class MultiModalData:
    """
    Dataclass that holds extra useful data for processing
    multimodal data. Processors currently cannot return keys,
    unless it is used in model's forward. Thus we have helper
    methods that calculate and return useful data from processing
    input multimodals (images/videos).
    Note that this dataclass is aimed to be used only in vLLM
    and we might change its API in the future.
    """

    num_image_tokens: list[int] | None = None
    num_video_tokens: list[int] | None = None
    num_audio_tokens: list[int] | None = None
    num_image_patches: list[int] | None = None

    def __contains__(self, key):
        return hasattr(self, key) and getattr(self, key) is not None

    def __getitem__(self, key):
        if hasattr(self, key):
            return getattr(self, key)
        raise AttributeError(f"{self.__class__.__name__} has no attribute {key}")


class ProcessorMixin(PushToHubMixin):
    """
    This is a mixin used to provide saving/loading functionality for all processor classes.
    """

    # Names need to be attr_class for attr in attributes
    _auto_class = None
    valid_processor_kwargs = ProcessingKwargs

    # args have to match the attributes class attribute
    def __init__(self, *args, **kwargs):
        # First, extract chat template from kwargs. It can never be a positional arg
        setattr(self, "chat_template", kwargs.pop("chat_template", None))

        # Check audio tokenizer for its class but do not treat it as attr to avoid saving weights
        if (audio_tokenizer := kwargs.pop("audio_tokenizer", None)) is not None:
            proper_class = self.check_argument_for_proper_class("audio_tokenizer", audio_tokenizer)
            if not (is_torch_available() and isinstance(audio_tokenizer, PreTrainedAudioTokenizerBase)):
                raise ValueError(
                    f"Tried to use `{proper_class}` for audio tokenization. However, this class is not"
                    " registered for audio tokenization."
                )
            setattr(self, "audio_tokenizer", audio_tokenizer)

        # Sanitize args and kwargs
        for key in kwargs:
            if key not in self.get_attributes():
                raise TypeError(f"Unexpected keyword argument {key}.")
        for arg, attribute_name in zip(args, self.get_attributes()):
            if attribute_name in kwargs:
                raise TypeError(f"Got multiple values for argument {attribute_name}.")
            else:
                kwargs[attribute_name] = arg

        if len(kwargs) != len(self.get_attributes()):
            raise ValueError(
                f"This processor requires {len(self.get_attributes())} arguments: {', '.join(self.get_attributes())}. Got "
                f"{len(args)} arguments instead."
            )

        # Check each arg is of the proper class (this will also catch a user initializing in the wrong order)
        for attribute_name, arg in kwargs.items():
            self.check_argument_for_proper_class(attribute_name, arg)
            setattr(self, attribute_name, arg)

    def __call__(
        self,
        images: ImageInput | None = None,
        text: TextInput | PreTokenizedInput | list[TextInput] | list[PreTokenizedInput] | None = None,
        videos: VideoInput | None = None,
        audio: AudioInput | None = None,
        **kwargs: Unpack[ProcessingKwargs],
    ):
        """
        Main method to prepare for model inputs. This method forwards the each modality argument to its own processor
        along with `kwargs`. Please refer to the docstring of the each processor attributes for more information.

        Args:
            images (`PIL.Image.Image`, `np.ndarray`, `torch.Tensor`, `list[PIL.Image.Image]`, `list[np.ndarray]`, `list[torch.Tensor]`):
                The image or batch of images to be prepared. Each image can be a PIL image, NumPy array or PyTorch
                tensor. Both channels-first and channels-last formats are supported.
            text (`TextInput`, `PreTokenizedInput`, `list[TextInput]`, `list[PreTokenizedInput]`, *optional*):
                The sequence or batch of sequences to be encoded. Each sequence can be a string or a list of strings
                (pretokenized string). If the sequences are provided as list of strings (pretokenized), you must set
                `is_split_into_words=True` (to lift the ambiguity with a batch of sequences).
            videos (`np.ndarray`, `torch.Tensor`, `List[np.ndarray]`, `List[torch.Tensor]`):
                The video or batch of videos to be prepared. Each video can be a 4D NumPy array or PyTorch
                tensor, or a nested list of 3D frames. Both channels-first and channels-last formats are supported.
            audio (`np.ndarray`, `torch.Tensor`, `list[np.ndarray]`, `list[torch.Tensor]`):
                The audio or batch of audio to be prepared. Each audio can be a NumPy array or PyTorch
                tensor.
            return_tensors (`str` or [`~utils.TensorType`], *optional*):
                If set, will return tensors of a particular framework. Acceptable values are:

                - `'pt'`: Return PyTorch `torch.Tensor` objects.
                - `'np'`: Return NumPy `np.ndarray` objects.

        Returns:
            [`BatchFeature`]: A [`BatchFeature`] object with processed inputs in a dict format.
        """
        if "audios" in kwargs and audio is None:
            raise ValueError("You passed keyword argument `audios` which is deprecated. Please use `audio` instead.")

        if images is None and text is None and videos is None and audio is None:
            raise ValueError(f"You need to provide at least one input to call {self.__class__.__name__}")

        kwargs = self._merge_kwargs(
            self.valid_processor_kwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs if hasattr(self, "tokenizer") else {},
            **kwargs,
        )

        attribute_to_kwargs = {
            "tokenizer": (text, "text_kwargs"),
            "image_processor": (images, "images_kwargs"),
            "video_processor": (videos, "videos_kwargs"),
            "feature_extractor": (audio, "audio_kwargs"),
        }
        outputs = {}
        for attribute_name in self.get_attributes():
            attribute = getattr(self, attribute_name, None)
            input_data, input_kwargs = attribute_to_kwargs[attribute_name]
            if input_data is not None and attribute is not None:
                attribute_output = attribute(input_data, **kwargs[input_kwargs])
                outputs.update(attribute_output)

        return BatchFeature(outputs)

    def check_argument_for_proper_class(self, argument_name, argument):
        """
        Checks the passed argument's class against the expected transformers class. In case of an unexpected
        mismatch between expected and actual class, an error is raise. Otherwise, the proper retrieved class
        is returned.
        """
        # If the exact attribute name is not in the mapping, use its canonical modality
        # (e.g., "encoder_tokenizer" -> "tokenizer")
        if argument_name not in MODALITY_TO_BASE_CLASS_MAPPING:
            argument_name = _get_modality_for_attribute(argument_name)
        class_name = MODALITY_TO_BASE_CLASS_MAPPING.get(argument_name)
        if isinstance(class_name, tuple):
            proper_class = tuple(self.get_possibly_dynamic_module(n) for n in class_name if n is not None)
        else:
            proper_class = self.get_possibly_dynamic_module(class_name)

        if not isinstance(argument, proper_class):
            raise TypeError(
                f"Received a {type(argument).__name__} for argument {argument_name}, but a {class_name} was expected."
            )

        return proper_class

    def to_dict(self) -> dict[str, Any]:
        """
        Serializes this instance to a Python dictionary.

        Returns:
            `dict[str, Any]`: Dictionary of all the attributes that make up this processor instance.
        """
        output = copy.deepcopy(self.__dict__)

        # Get the kwargs in `__init__`.
        sig = inspect.signature(self.__init__)
        # Only save the attributes that are presented in the kwargs of `__init__`.
        # or in the attributes
        attrs_to_save = list(sig.parameters) + self.__class__.get_attributes()
        # extra attributes to be kept
        attrs_to_save += ["auto_map"]

        # Remove tokenizers from output - they have their own vocab files and are saved separately.
        # All other sub-processors (image_processor, feature_extractor, etc.) are kept in processor_config.json.
        for attribute in self.__class__.get_attributes():
            if attribute in output:
                modality = _get_modality_for_attribute(attribute)
                if modality == "tokenizer":
                    del output[attribute]

        if "chat_template" in output:
            del output["chat_template"]

        def cast_array_to_list(dictionary):
            """
            Numpy arrays are not serialiazable but can be in pre-processing dicts.
            This function casts arrays to list, recusring through the nested configs as well.
            """
            for key, value in dictionary.items():
                if isinstance(value, np.ndarray):
                    dictionary[key] = value.tolist()
                elif isinstance(value, dict):
                    dictionary[key] = cast_array_to_list(value)
            return dictionary

        # Special case, add `audio_tokenizer` dict which points to model weights and path
        if "audio_tokenizer" in output:
            audio_tokenizer_dict = {
                "audio_tokenizer_class": self.audio_tokenizer.__class__.__name__,
                "audio_tokenizer_name_or_path": self.audio_tokenizer.name_or_path,
            }
            output["audio_tokenizer"] = audio_tokenizer_dict

        # Serialize attributes as a dict
        output = {
            k: v.to_dict() if isinstance(v, PushToHubMixin) else v
            for k, v in output.items()
            if (
                k in attrs_to_save  # keep all attributes that have to be serialized
                and v.__class__.__name__ != "BeamSearchDecoderCTC"  # remove attributes with that are objects
            )
        }
        output = cast_array_to_list(output)
        output["processor_class"] = self.__class__.__name__

        return output

    def to_json_string(self) -> str:
        """
        Serializes this instance to a JSON string.

        Returns:
            `str`: String containing all the attributes that make up this feature_extractor instance in JSON format.
        """
        dictionary = self.to_dict()

        return json.dumps(dictionary, indent=2, sort_keys=True) + "\n"

    def to_json_file(self, json_file_path: str | os.PathLike):
        """
        Save this instance to a JSON file.

        Args:
            json_file_path (`str` or `os.PathLike`):
                Path to the JSON file in which this processor instance's parameters will be saved.
        """
        with open(json_file_path, "w", encoding="utf-8") as writer:
            writer.write(self.to_json_string())

    def __repr__(self):
        attributes_repr = [f"- {name}: {repr(getattr(self, name))}" for name in self.get_attributes()]
        attributes_repr = "\n".join(attributes_repr)
        return f"{self.__class__.__name__}:\n{attributes_repr}\n\n{self.to_json_string()}"

    def save_pretrained(self, save_directory, push_to_hub: bool = False, **kwargs):
        """
        Saves the attributes of this processor (feature extractor, tokenizer...) in the specified directory so that it
        can be reloaded using the [`~ProcessorMixin.from_pretrained`] method.

        <Tip>

        This class method is simply calling [`~feature_extraction_utils.FeatureExtractionMixin.save_pretrained`] and
        [`~tokenization_utils_base.PreTrainedTokenizerBase.save_pretrained`]. Please refer to the docstrings of the
        methods above for more information.

        </Tip>

        Args:
            save_directory (`str` or `os.PathLike`):
                Directory where the feature extractor JSON file and the tokenizer files will be saved (directory will
                be created if it does not exist).
            push_to_hub (`bool`, *optional*, defaults to `False`):
                Whether or not to push your model to the Hugging Face model hub after saving it. You can specify the
                repository you want to push to with `repo_id` (will default to the name of `save_directory` in your
                namespace).
            kwargs (`dict[str, Any]`, *optional*):
                Additional key word arguments passed along to the [`~utils.PushToHubMixin.push_to_hub`] method.
        """
        os.makedirs(save_directory, exist_ok=True)

        if push_to_hub:
            commit_message = kwargs.pop("commit_message", None)
            repo_id = kwargs.pop("repo_id", save_directory.split(os.path.sep)[-1])
            repo_id = create_repo(repo_id, exist_ok=True, **kwargs).repo_id
            files_timestamps = self._get_files_timestamps(save_directory)
        # If we have a custom config, we copy the file defining it in the folder and set the attributes so it can be
        # loaded from the Hub.
        if self._auto_class is not None:
            attrs = [getattr(self, attribute_name) for attribute_name in self.get_attributes()]
            configs = [(a.init_kwargs if isinstance(a, PreTrainedTokenizerBase) else a) for a in attrs]
            configs.append(self)
            custom_object_save(self, save_directory, config=configs)

        for attribute_name in self.get_attributes():
            attribute = getattr(self, attribute_name)

            modality = _get_modality_for_attribute(attribute_name)
            is_primary = attribute_name == modality
            if modality == "tokenizer":
                attribute._set_processor_class(self.__class__.__name__)
                # Save the tokenizer in its own vocab file. The other attributes are saved as part of `processor_config.json`
                if is_primary:
                    attribute.save_pretrained(save_directory)
                else:
                    # if a model has multiple tokenizers, save the additional tokenizers in their own folders.
                    attribute.save_pretrained(os.path.join(save_directory, attribute_name))
            elif attribute._auto_class is not None:
                custom_object_save(attribute, save_directory, config=attribute)

        if self._auto_class is not None:
            # We added an attribute to the init_kwargs of the tokenizers, which needs to be cleaned up.
            for attribute_name in self.get_attributes():
                attribute = getattr(self, attribute_name)
                if isinstance(attribute, PreTrainedTokenizerBase):
                    del attribute.init_kwargs["auto_map"]

        # If we save using the predefined names, we can load using `from_pretrained`
        # plus we save chat_template in its own file
        output_processor_file = os.path.join(save_directory, PROCESSOR_NAME)
        output_chat_template_file_jinja = os.path.join(save_directory, CHAT_TEMPLATE_FILE)
        chat_template_dir = os.path.join(save_directory, CHAT_TEMPLATE_DIR)

        # Save `chat_template` in its own file. We can't get it from `processor_dict` as we popped it in `to_dict`
        # to avoid serializing chat template in json config file. So let's get it from `self` directly
        if isinstance(self.chat_template, str):
            # New format for single templates is to save them as chat_template.jinja
            with open(output_chat_template_file_jinja, "w", encoding="utf-8") as f:
                f.write(self.chat_template)
            logger.info(f"chat template saved in {output_chat_template_file_jinja}")
        elif isinstance(self.chat_template, dict):
            # New format for multiple templates is to save the default as chat_template.jinja
            # and the other templates in the chat_templates/ directory
            for template_name, template in self.chat_template.items():
                if template_name == "default":
                    with open(output_chat_template_file_jinja, "w", encoding="utf-8") as f:
                        f.write(self.chat_template["default"])
                    logger.info(f"chat template saved in {output_chat_template_file_jinja}")
                else:
                    os.makedirs(chat_template_dir, exist_ok=True)
                    template_filepath = os.path.join(chat_template_dir, f"{template_name}.jinja")
                    with open(template_filepath, "w", encoding="utf-8") as f:
                        f.write(template)
                    logger.info(f"chat template saved in {template_filepath}")

        # Create a unified `preprocessor_config.json` and save all attributes as a composite config, except for tokenizers
        self.to_json_file(output_processor_file)
        logger.info(f"processor saved in {output_processor_file}")
        return_files = [output_processor_file]

        if push_to_hub:
            self._upload_modified_files(
                save_directory,
                repo_id,
                files_timestamps,
                commit_message=commit_message,
                token=kwargs.get("token"),
            )

        return return_files

    @classmethod
    def get_processor_dict(
        cls, pretrained_model_name_or_path: str | os.PathLike, **kwargs
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """
        From a `pretrained_model_name_or_path`, resolve to a dictionary of parameters, to be used for instantiating a
        processor of type [`~processing_utils.ProcessingMixin`] using `from_args_and_dict`.

        Parameters:
            pretrained_model_name_or_path (`str` or `os.PathLike`):
                The identifier of the pre-trained checkpoint from which we want the dictionary of parameters.
            subfolder (`str`, *optional*, defaults to `""`):
                In case the relevant files are located inside a subfolder of the model repo on huggingface.co, you can
                specify the folder name here.

        Returns:
            `tuple[Dict, Dict]`: The dictionary(ies) that will be used to instantiate the processor object.
        """
        # holding a copy for optionally loading the audio tokenizer (if available)
        audio_tokenizer_kwargs = copy.deepcopy(kwargs)

        cache_dir = kwargs.pop("cache_dir", None)
        force_download = kwargs.pop("force_download", False)
        proxies = kwargs.pop("proxies", None)
        token = kwargs.pop("token", None)
        local_files_only = kwargs.pop("local_files_only", False)
        revision = kwargs.pop("revision", None)
        subfolder = kwargs.pop("subfolder", "")

        from_pipeline = kwargs.pop("_from_pipeline", None)
        from_auto_class = kwargs.pop("_from_auto", False)

        user_agent = {"file_type": "processor", "from_auto_class": from_auto_class}
        if from_pipeline is not None:
            user_agent["using_pipeline"] = from_pipeline

        if is_offline_mode() and not local_files_only:
            logger.info("Offline mode: forcing local_files_only=True")
            local_files_only = True

        pretrained_model_name_or_path = str(pretrained_model_name_or_path)
        is_local = os.path.isdir(pretrained_model_name_or_path)
        if os.path.isdir(pretrained_model_name_or_path):
            processor_file = os.path.join(pretrained_model_name_or_path, PROCESSOR_NAME)

        additional_chat_template_files = {}
        resolved_additional_chat_template_files = {}
        if os.path.isfile(pretrained_model_name_or_path):
            resolved_processor_file = pretrained_model_name_or_path
            # can't load chat-template and audio tokenizer when given a file as pretrained_model_name_or_path
            resolved_chat_template_file = None
            resolved_raw_chat_template_file = None
            resolved_audio_tokenizer_file = None
            is_local = True
        else:
            if is_local:
                template_dir = Path(pretrained_model_name_or_path, CHAT_TEMPLATE_DIR)
                if template_dir.is_dir():
                    for template_file in template_dir.glob("*.jinja"):
                        template_name = template_file.stem
                        additional_chat_template_files[template_name] = f"{CHAT_TEMPLATE_DIR}/{template_file.name}"
            else:
                try:
                    for template in list_repo_templates(
                        pretrained_model_name_or_path,
                        local_files_only=local_files_only,
                        revision=revision,
                        cache_dir=cache_dir,
                        token=token,
                    ):
                        template = template.removesuffix(".jinja")
                        additional_chat_template_files[template] = f"{CHAT_TEMPLATE_DIR}/{template}.jinja"
                except EntryNotFoundError:
                    pass  # No template dir means no template files
            processor_file = PROCESSOR_NAME

            try:
                # Load from local folder or from cache or download from model Hub and cache
                resolved_processor_file = cached_file(
                    pretrained_model_name_or_path,
                    processor_file,
                    cache_dir=cache_dir,
                    force_download=force_download,
                    proxies=proxies,
                    local_files_only=local_files_only,
                    token=token,
                    user_agent=user_agent,
                    revision=revision,
                    subfolder=subfolder,
                    _raise_exceptions_for_missing_entries=False,
                )

                # chat_template.json is a legacy file used by the processor class
                # a raw chat_template.jinja is preferred in future
                resolved_chat_template_file = cached_file(
                    pretrained_model_name_or_path,
                    LEGACY_PROCESSOR_CHAT_TEMPLATE_FILE,
                    cache_dir=cache_dir,
                    force_download=force_download,
                    proxies=proxies,
                    local_files_only=local_files_only,
                    token=token,
                    user_agent=user_agent,
                    revision=revision,
                    subfolder=subfolder,
                    _raise_exceptions_for_missing_entries=False,
                )

                resolved_raw_chat_template_file = cached_file(
                    pretrained_model_name_or_path,
                    CHAT_TEMPLATE_FILE,
                    cache_dir=cache_dir,
                    force_download=force_download,
                    proxies=proxies,
                    local_files_only=local_files_only,
                    token=token,
                    user_agent=user_agent,
                    revision=revision,
                    subfolder=subfolder,
                    _raise_exceptions_for_missing_entries=False,
                )

                resolved_additional_chat_template_files = {
                    template_name: cached_file(
                        pretrained_model_name_or_path,
                        template_file,
                        cache_dir=cache_dir,
                        force_download=force_download,
                        proxies=proxies,
                        local_files_only=local_files_only,
                        token=token,
                        user_agent=user_agent,
                        revision=revision,
                        subfolder=subfolder,
                        _raise_exceptions_for_missing_entries=False,
                    )
                    for template_name, template_file in additional_chat_template_files.items()
                }

                resolved_audio_tokenizer_file = cached_file(
                    pretrained_model_name_or_path,
                    AUDIO_TOKENIZER_NAME,
                    cache_dir=cache_dir,
                    force_download=force_download,
                    proxies=proxies,
                    local_files_only=local_files_only,
                    token=token,
                    user_agent=user_agent,
                    revision=revision,
                    subfolder=subfolder,
                    _raise_exceptions_for_missing_entries=False,
                )
            except OSError:
                # Raise any environment error raise by `cached_file`. It will have a helpful error message adapted to
                # the original exception.
                raise
            except Exception:
                # For any other exception, we throw a generic error.
                raise OSError(
                    f"Can't load processor for '{pretrained_model_name_or_path}'. If you were trying to load"
                    " it from 'https://huggingface.co/models', make sure you don't have a local directory with the"
                    f" same name. Otherwise, make sure '{pretrained_model_name_or_path}' is the correct path to a"
                    f" directory containing a {PROCESSOR_NAME} file"
                )

        # Add chat template as kwarg before returning because most models don't have processor config
        if resolved_chat_template_file is not None:
            # This is the legacy path
            with open(resolved_chat_template_file, encoding="utf-8") as reader:
                chat_template_json = json.loads(reader.read())
                chat_templates = {"default": chat_template_json["chat_template"]}
                if resolved_additional_chat_template_files:
                    raise ValueError(
                        "Cannot load chat template due to conflicting files - this checkpoint combines "
                        "a legacy chat_template.json file with separate template files, which is not "
                        "supported. To resolve this error, replace the legacy chat_template.json file "
                        "with a modern chat_template.jinja file."
                    )
        else:
            chat_templates = {
                template_name: open(template_file, "r", encoding="utf-8").read()
                for template_name, template_file in resolved_additional_chat_template_files.items()
            }
            if resolved_raw_chat_template_file is not None:
                with open(resolved_raw_chat_template_file, "r", encoding="utf-8") as reader:
                    chat_templates["default"] = reader.read()
        if isinstance(chat_templates, dict) and "default" in chat_templates and len(chat_templates) == 1:
            chat_templates = chat_templates["default"]  # Flatten when we just have a single template/file

        # Existing processors on the Hub created before #27761 being merged don't have `processor_config.json` (if not
        # updated afterward), and we need to keep `from_pretrained` work. So here it fallbacks to the empty dict.
        # (`cached_file` called using `_raise_exceptions_for_missing_entries=False` to avoid exception)
        # However, for models added in the future, we won't get the expected error if this file is missing.
        if resolved_processor_file is None:
            # In any case we need to pass `chat_template` if it is available
            processor_dict = {}
        else:
            try:
                # Load processor dict
                with open(resolved_processor_file, encoding="utf-8") as reader:
                    text = reader.read()
                processor_dict = json.loads(text)

            except json.JSONDecodeError:
                raise OSError(
                    f"It looks like the config file at '{resolved_processor_file}' is not a valid JSON file."
                )

        if is_local:
            logger.info(f"loading configuration file {resolved_processor_file}")
        else:
            logger.info(f"loading configuration file {processor_file} from cache at {resolved_processor_file}")

        if processor_dict.get("chat_template") is not None:
            logger.warning_once(
                "Chat templates should be in a 'chat_template.jinja' file but found key='chat_template' "
                "in the processor's config. Make sure to move your template to its own file."
            )
        elif chat_templates:
            processor_dict["chat_template"] = chat_templates

        # Audio tokenizer needs to load the model checkpoint first, because the saved
        # json file contains only references to the model path and repo id
        if resolved_audio_tokenizer_file is not None or "audio_tokenizer" in processor_dict:
            if resolved_audio_tokenizer_file is not None:
                reader = open(resolved_audio_tokenizer_file, "r", encoding="utf-8")
                audio_tokenizer_dict = reader.read()
                audio_tokenizer_dict = json.loads(audio_tokenizer_dict)
            else:
                audio_tokenizer_dict = processor_dict["audio_tokenizer"]

            audio_tokenizer_class = cls.get_possibly_dynamic_module(audio_tokenizer_dict["audio_tokenizer_class"])
            audio_tokenizer_path = audio_tokenizer_dict["audio_tokenizer_name_or_path"]
            processor_dict["audio_tokenizer"] = audio_tokenizer_class.from_pretrained(
                audio_tokenizer_path, **audio_tokenizer_kwargs
            )

        return processor_dict, kwargs

    @classmethod
    def from_args_and_dict(cls, args, processor_dict: dict[str, Any], **kwargs):
        """
        Instantiates a type of [`~processing_utils.ProcessingMixin`] from a Python dictionary of parameters.

        Args:
            processor_dict (`dict[str, Any]`):
                Dictionary that will be used to instantiate the processor object. Such a dictionary can be
                retrieved from a pretrained checkpoint by leveraging the
                [`~processing_utils.ProcessingMixin.to_dict`] method.
            kwargs (`dict[str, Any]`):
                Additional parameters from which to initialize the processor object.

        Returns:
            [`~processing_utils.ProcessingMixin`]: The processor object instantiated from those
            parameters.
        """
        processor_dict = processor_dict.copy()
        return_unused_kwargs = kwargs.pop("return_unused_kwargs", False)

        # We have to pop up some unused (but specific) kwargs and then validate that it doesn't contain unused kwargs
        # If we don't pop, some specific kwargs will raise a warning or error
        for unused_kwarg in cls.get_attributes() + ["auto_map", "processor_class"]:
            processor_dict.pop(unused_kwarg, None)

        # override processor_dict with given kwargs
        processor_dict.update(kwargs)

        # check if there is an overlap between args and processor_dict
        accepted_args_and_kwargs = cls.__init__.__code__.co_varnames[: cls.__init__.__code__.co_argcount][1:]

        # validate both processor_dict and given kwargs
        unused_kwargs, valid_kwargs = cls.validate_init_kwargs(
            processor_config=processor_dict, valid_kwargs=accepted_args_and_kwargs
        )

        # update args that are already in processor_dict to avoid duplicate arguments
        args_to_update = {
            i: valid_kwargs.pop(arg)
            for i, arg in enumerate(accepted_args_and_kwargs)
            if (arg in valid_kwargs and i < len(args))
        }
        args = [args_to_update.get(i, arg) for i, arg in enumerate(args)]

        # instantiate processor with used (and valid) kwargs only
        processor = cls(*args, **valid_kwargs)

        logger.info(f"Processor {processor}")
        if return_unused_kwargs:
            return processor, unused_kwargs
        else:
            return processor

    def _merge_kwargs(
        self,
        ModelProcessorKwargs: ProcessingKwargs,
        tokenizer_init_kwargs: dict | None = None,
        **kwargs,
    ) -> dict[str, dict]:
        """
        Method to merge dictionaries of kwargs cleanly separated by modality within a Processor instance.
        The order of operations is as follows:
            1) kwargs passed as before have highest priority to preserve BC.
                ```python
                high_priority_kwargs = {"crop_size" = {"height": 222, "width": 222}, "padding" = "max_length"}
                processor(..., **high_priority_kwargs)
                ```
            2) kwargs passed as modality-specific kwargs have second priority. This is the recommended API.
                ```python
                processor(..., text_kwargs={"padding": "max_length"}, images_kwargs={"crop_size": {"height": 222, "width": 222}}})
                ```
            3) kwargs passed during instantiation of a modality processor have fourth priority.
                ```python
                tokenizer = tokenizer_class(..., {"padding": "max_length"})
                image_processor = image_processor_class(...)
                processor(tokenizer, image_processor) # will pass max_length unless overridden by kwargs at call
                ```
            4) defaults kwargs specified at processor level have lowest priority.
                ```python
                class MyProcessingKwargs(ProcessingKwargs, CommonKwargs, TextKwargs, ImagesKwargs, total=False):
                    _defaults = {
                        "text_kwargs": {
                            "padding": "max_length",
                            "max_length": 64,
                        },
                    }
                ```
        Args:
            ModelProcessorKwargs (`ProcessingKwargs`):
                Typed dictionary of kwargs specifically required by the model passed.
            tokenizer_init_kwargs (`Dict`, *optional*):
                Dictionary of kwargs the tokenizer was instantiated with and need to take precedence over defaults.

        Returns:
            output_kwargs (`Dict`):
                Dictionary of per-modality kwargs to be passed to each modality-specific processor.

        """
        # holding a copy to avoid mutating user-provided arguments
        # Use deepcopy to also copy nested dicts (like videos_kwargs) that will be modified via pop()
        kwargs = copy.deepcopy(kwargs)

        # Initialize dictionaries
        output_kwargs = {
            "text_kwargs": {},
            "images_kwargs": {},
            "audio_kwargs": {},
            "videos_kwargs": {},
        }

        default_kwargs = {
            "text_kwargs": {},
            "images_kwargs": {},
            "audio_kwargs": {},
            "videos_kwargs": {},
        }

        map_preprocessor_kwargs = {
            "text_kwargs": "tokenizer",
            "images_kwargs": "image_processor",
            "audio_kwargs": "feature_extractor",
            "videos_kwargs": "video_processor",
        }

        possible_modality_keywords = {"text", "audio", "videos", "images"}
        used_keys = set()

        # get defaults from set model processor kwargs if they exist
        for modality in default_kwargs:
            default_kwargs[modality] = ModelProcessorKwargs._defaults.get(modality, {}).copy()
            # Some preprocessors define a set of accepted "valid_kwargs" (currently only vision).
            # In those cases, we don’t declare a `ModalityKwargs` attribute in the TypedDict.
            # Instead, we dynamically obtain the kwargs from the preprocessor and merge them
            # with the general kwargs set. This ensures consistency between preprocessor and
            # processor classes, and helps prevent accidental mismatches.
            modality_valid_kwargs = set(ModelProcessorKwargs.__annotations__[modality].__annotations__)
            if modality in map_preprocessor_kwargs:
                preprocessor = getattr(self, map_preprocessor_kwargs[modality], None)
                preprocessor_valid_kwargs = (
                    getattr(preprocessor, "valid_kwargs", None) if preprocessor is not None else None
                )
                modality_valid_kwargs.update(
                    set(preprocessor_valid_kwargs.__annotations__ if preprocessor_valid_kwargs is not None else [])
                )
            # update defaults with arguments from tokenizer init
            for modality_key in modality_valid_kwargs:
                # init with tokenizer init kwargs if necessary
                if tokenizer_init_kwargs is not None and modality_key in tokenizer_init_kwargs:
                    value = (
                        getattr(self.tokenizer, modality_key)
                        if hasattr(self.tokenizer, modality_key)
                        else tokenizer_init_kwargs[modality_key]
                    )
                    default_kwargs[modality][modality_key] = value
        # now defaults kwargs are updated with the tokenizers defaults.
        # pass defaults to output dictionary
        output_kwargs.update(default_kwargs)

        # For `common_kwargs` just update all modality-specific kwargs with same key/values
        common_kwargs = ModelProcessorKwargs._defaults.get("common_kwargs", {})
        common_kwargs.update(kwargs.get("common_kwargs", {}))
        if common_kwargs:
            for kwarg in output_kwargs.values():
                kwarg.update(common_kwargs)

        # update modality kwargs with passed kwargs
        non_modality_kwargs = set(kwargs) - set(output_kwargs)
        for modality, output_kwarg in output_kwargs.items():
            modality_valid_kwargs = set(ModelProcessorKwargs.__annotations__[modality].__annotations__)
            if modality in map_preprocessor_kwargs:
                preprocessor = getattr(self, map_preprocessor_kwargs[modality], None)
                preprocessor_valid_kwargs = (
                    getattr(preprocessor, "valid_kwargs", None) if preprocessor is not None else None
                )
                modality_valid_kwargs.update(
                    set(preprocessor_valid_kwargs.__annotations__ if preprocessor_valid_kwargs is not None else [])
                )
            for modality_key in modality_valid_kwargs:
                # check if we received a structured kwarg dict or not to handle it correctly
                if modality in kwargs:
                    kwarg_value = kwargs[modality].pop(modality_key, "__empty__")
                    # check if this key was passed as a flat kwarg.
                    if kwarg_value != "__empty__" and modality_key in non_modality_kwargs:
                        raise ValueError(
                            f"Keyword argument {modality_key} was passed two times:\n"
                            f"in a dictionary for {modality} and as a **kwarg."
                        )
                elif modality_key in kwargs:
                    # we get a modality_key instead of popping it because modality-specific processors
                    # can have overlapping kwargs
                    kwarg_value = kwargs.get(modality_key, "__empty__")
                else:
                    kwarg_value = "__empty__"
                if not isinstance(kwarg_value, str) or kwarg_value != "__empty__":
                    output_kwarg[modality_key] = kwarg_value
                    used_keys.add(modality_key)

        # Determine if kwargs is a flat dictionary or contains nested dictionaries
        if any(key in default_kwargs for key in kwargs):
            # kwargs is dictionary-based, and some keys match modality names
            for modality, subdict in kwargs.items():
                if modality in default_kwargs:
                    for subkey, subvalue in subdict.items():
                        if subkey not in used_keys:
                            output_kwargs[modality][subkey] = subvalue
                            used_keys.add(subkey)
        else:
            # kwargs is a flat dictionary
            for key, kwarg in kwargs.items():
                if key not in used_keys and key not in possible_modality_keywords:
                    logger.warning_once(
                        f"Keyword argument `{key}` is not a valid argument for this processor and will be ignored."
                    )

        for key, typed_dict_obj in ModelProcessorKwargs.__annotations__.items():
            if key in map_preprocessor_kwargs:
                preprocessor = getattr(self, map_preprocessor_kwargs[key], None)
                if preprocessor is None or getattr(preprocessor, "valid_kwargs", None) is None:
                    continue
                preprocessor_typed_dict_obj = getattr(preprocessor, "valid_kwargs")
                typed_dict_obj = TypedDict(
                    "merged_typed_dict",
                    {**preprocessor_typed_dict_obj.__annotations__, **typed_dict_obj.__annotations__},
                    total=False,
                )
            validate_typed_dict(typed_dict_obj, output_kwargs[key])
        return output_kwargs

    @classmethod
    def from_pretrained(
        cls: type[SpecificProcessorType],
        pretrained_model_name_or_path: str | os.PathLike,
        cache_dir: str | os.PathLike | None = None,
        force_download: bool = False,
        local_files_only: bool = False,
        token: str | bool | None = None,
        revision: str = "main",
        **kwargs,
    ) -> SpecificProcessorType:
        r"""
        Instantiate a processor associated with a pretrained model.

        <Tip>

        This class method is simply calling the feature extractor
        [`~feature_extraction_utils.FeatureExtractionMixin.from_pretrained`], image processor
        [`~image_processing_utils.ImageProcessingMixin`] and the tokenizer
        [`~tokenization_utils_base.PreTrainedTokenizer.from_pretrained`] methods. Please refer to the docstrings of the
        methods above for more information.

        </Tip>

        Args:
            pretrained_model_name_or_path (`str` or `os.PathLike`):
                This can be either:

                - a string, the *model id* of a pretrained feature_extractor hosted inside a model repo on
                  huggingface.co.
                - a path to a *directory* containing a feature extractor file saved using the
                  [`~SequenceFeatureExtractor.save_pretrained`] method, e.g., `./my_model_directory/`.
                - a path or url to a saved feature extractor JSON *file*, e.g.,
                  `./my_model_directory/preprocessor_config.json`.
            **kwargs
                Additional keyword arguments passed along to both
                [`~feature_extraction_utils.FeatureExtractionMixin.from_pretrained`] and
                [`~tokenization_utils_base.PreTrainedTokenizer.from_pretrained`].
        """
        kwargs["cache_dir"] = cache_dir
        kwargs["force_download"] = force_download
        kwargs["local_files_only"] = local_files_only
        kwargs["revision"] = revision

        if token is not None:
            kwargs["token"] = token

        # Get processor_dict first so we can use it to instantiate non-tokenizer sub-processors
        processor_dict, instantiation_kwargs = cls.get_processor_dict(pretrained_model_name_or_path, **kwargs)
        args = cls._get_arguments_from_pretrained(pretrained_model_name_or_path, processor_dict, **kwargs)
        return cls.from_args_and_dict(args, processor_dict, **instantiation_kwargs)

    @classmethod
    def get_attributes(cls):
        args_in_init = inspect.signature(cls.__init__).parameters.keys()
        attributes = []
        for sub_processor_type in args_in_init:
            # don't treat audio_tokenizer as an attribute
            if sub_processor_type == "audio_tokenizer":
                continue
            if any(modality in sub_processor_type for modality in MODALITY_TO_AUTOPROCESSOR_MAPPING.keys()):
                attributes.append(sub_processor_type)

        # Legacy processors may not override `__init__` and instead expose modality
        # attributes via `<attribute>_class`. In that case, `args_in_init` only exposes
        # `*args`/`**kwargs`, so we need to infer the attributes from those class-level
        # hints to keep backward compatibility (e.g. dynamic processors stored on the Hub).
        if not attributes:
            for attribute_name, value in cls.__dict__.items():
                if value is None or attribute_name == "audio_tokenizer_class" or not attribute_name.endswith("_class"):
                    continue
                inferred_attribute = attribute_name[: -len("_class")]
                if inferred_attribute == "audio_tokenizer":
                    continue
                if any(modality in inferred_attribute for modality in MODALITY_TO_AUTOPROCESSOR_MAPPING.keys()):
                    attributes.append(inferred_attribute)

        return attributes

    @classmethod
    def register_for_auto_class(cls, auto_class="AutoProcessor"):
        """
        Register this class with a given auto class. This should only be used for custom feature extractors as the ones
        in the library are already mapped with `AutoProcessor`.



        Args:
            auto_class (`str` or `type`, *optional*, defaults to `"AutoProcessor"`):
                The auto class to register this new feature extractor with.
        """
        if not isinstance(auto_class, str):
            auto_class = auto_class.__name__

        import transformers.models.auto as auto_module

        if not hasattr(auto_module, auto_class):
            raise ValueError(f"{auto_class} is not a valid auto class.")

        cls._auto_class = auto_class

    @classmethod
    def _load_tokenizer_from_pretrained(
        cls, sub_processor_type, pretrained_model_name_or_path, subfolder="", **kwargs
    ):
        auto_processor_class = MODALITY_TO_AUTOPROCESSOR_MAPPING["tokenizer"]
        is_primary = sub_processor_type == "tokenizer"

        if is_primary:
            # Primary tokenizer: load from root
            tokenizer = auto_processor_class.from_pretrained(
                pretrained_model_name_or_path, subfolder=subfolder, **kwargs
            )
        else:
            # Additional tokenizer: load from subfolder (e.g., "decoder_tokenizer")
            tokenizer_subfolder = os.path.join(subfolder, sub_processor_type) if subfolder else sub_processor_type
            tokenizer = auto_processor_class.from_pretrained(
                pretrained_model_name_or_path, subfolder=tokenizer_subfolder, **kwargs
            )
        return tokenizer

    @classmethod
    def _get_arguments_from_pretrained(cls, pretrained_model_name_or_path, processor_dict=None, **kwargs):
        """
        Identify and instantiate the subcomponents of Processor classes, such as image processors, tokenizers,
        and feature extractors. This method inspects the processor's `__init__` signature to identify parameters
        that correspond to known modality types (image_processor, tokenizer, feature_extractor, etc.) or contain
        modality names in their attribute name.

        For tokenizers: Uses the appropriate Auto class (AutoTokenizer) to load via `.from_pretrained()`.
        Additional tokenizers (e.g., "decoder_tokenizer") are loaded from subfolders.

        For other sub-processors (image_processor, feature_extractor, etc.): Primary ones are loaded via
        Auto class. Additional ones are instantiated from the config stored in processor_config.json
        (passed as processor_dict).

        Args:
            pretrained_model_name_or_path: Path or model id to load from.
            processor_dict: Optional dict containing processor config (from processor_config.json).
                Required when loading additional non-tokenizer sub-processors.
        """
        args = []
        processor_dict = processor_dict if processor_dict is not None else {}
        # Remove subfolder from kwargs to avoid duplicate keyword arguments
        subfolder = kwargs.pop("subfolder", "")

        # get args from processor init signature
        sub_processors = cls.get_attributes()
        for sub_processor_type in sub_processors:
            modality = _get_modality_for_attribute(sub_processor_type)
            is_primary = sub_processor_type == modality

            if (
                "tokenizer" in sub_processor_type
            ):  # This is only necessary for the checkpoint in test_processing_mistral3.py which has no config.json and
                # the tokenizer_config.json references LlamaTokenizerFast. TODO: update the config on the hub.
                if "PixtralProcessor" in cls.__name__:
                    from .tokenization_utils_tokenizers import TokenizersBackend

                    tokenizer = TokenizersBackend.from_pretrained(
                        pretrained_model_name_or_path, subfolder=subfolder, **kwargs
                    )
                else:
                    tokenizer = cls._load_tokenizer_from_pretrained(
                        sub_processor_type, pretrained_model_name_or_path, subfolder=subfolder, **kwargs
                    )
                args.append(tokenizer)
            elif is_primary:
                # Primary non-tokenizer sub-processor: load via Auto class
                auto_processor_class = MODALITY_TO_AUTOPROCESSOR_MAPPING[sub_processor_type]
                sub_processor = auto_processor_class.from_pretrained(
                    pretrained_model_name_or_path, subfolder=subfolder, **kwargs
                )
                args.append(sub_processor)

            elif sub_processor_type in processor_dict:
                # Additional non-tokenizer sub-processor: instantiate from config in processor_dict
                sub_processor_config = processor_dict[sub_processor_type]
                if isinstance(sub_processor_config, dict):
                    # Determine the class to instantiate
                    # Image processors have 'image_processor_type', feature extractors have 'feature_extractor_type'
                    type_key = f"{modality}_type"
                    class_name = sub_processor_config.get(type_key)
                    if class_name is None:
                        raise ValueError(
                            f"Cannot instantiate {sub_processor_type}: missing '{type_key}' in config. "
                            f"Config keys: {list(sub_processor_config.keys())}"
                        )
                    processor_class = cls.get_possibly_dynamic_module(class_name)
                    sub_processor = processor_class(**sub_processor_config)
                    args.append(sub_processor)
                else:
                    raise ValueError(
                        f"Expected dict for {sub_processor_type} in processor_config.json, "
                        f"got {type(sub_processor_config)}"
                    )
            else:
                raise ValueError(
                    f"Cannot find config for {sub_processor_type} in processor_config.json. "
                    f"Available keys: {list(processor_dict.keys())}"
                )

        return args

    @staticmethod
    def get_possibly_dynamic_module(module_name):
        if hasattr(transformers_module, module_name):
            return getattr(transformers_module, module_name)
        lookup_locations = [
            transformers_module.IMAGE_PROCESSOR_MAPPING,
            transformers_module.VIDEO_PROCESSOR_MAPPING,
            transformers_module.TOKENIZER_MAPPING,
            transformers_module.FEATURE_EXTRACTOR_MAPPING,
            transformers_module.MODEL_FOR_AUDIO_TOKENIZATION_MAPPING,
        ]
        for lookup_location in lookup_locations:
            for custom_class in lookup_location._extra_content.values():
                if isinstance(custom_class, tuple):
                    for custom_subclass in custom_class:
                        if custom_subclass is not None and custom_subclass.__name__ == module_name:
                            return custom_subclass
                elif custom_class is not None and custom_class.__name__ == module_name:
                    return custom_class
        raise ValueError(
            f"Could not find module {module_name} in `transformers`. If this is a custom class, "
            f"it should be registered using the relevant `AutoClass.register()` function so that "
            f"other functions can find it!"
        )

    def batch_decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to PreTrainedTokenizer's [`~PreTrainedTokenizer.batch_decode`]. Please
        refer to the docstring of this method for more information.
        """
        if not hasattr(self, "tokenizer"):
            raise ValueError(f"Cannot batch decode text: {self.__class__.__name__} has no tokenizer.")
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to PreTrainedTokenizer's [`~PreTrainedTokenizer.decode`]. Please refer to
        the docstring of this method for more information.
        """
        if not hasattr(self, "tokenizer"):
            raise ValueError(f"Cannot decode text: {self.__class__.__name__} has no tokenizer.")
        return self.tokenizer.decode(*args, **kwargs)

    @property
    def model_input_names(self):
        model_input_names = []
        for attribute_name in self.get_attributes():
            attribute = getattr(self, attribute_name, None)
            attr_input_names = getattr(attribute, "model_input_names")
            model_input_names.extend(attr_input_names)
        return model_input_names

    @staticmethod
    def validate_init_kwargs(processor_config, valid_kwargs):
        kwargs_from_config = set(processor_config.keys())
        valid_kwargs_set = set(valid_kwargs)

        unused_keys = kwargs_from_config - valid_kwargs_set
        valid_keys = kwargs_from_config & valid_kwargs_set

        unused_kwargs = {k: processor_config[k] for k in unused_keys} if unused_keys else {}
        valid_kwargs = {k: processor_config[k] for k in valid_keys} if valid_keys else {}

        return unused_kwargs, valid_kwargs

    def apply_chat_template(
        self,
        conversation: list[dict[str, str]] | list[list[dict[str, str]]],
        chat_template: str | None = None,
        **kwargs: Unpack[AllKwargsForChatTemplate],
    ) -> str:
        """
        Similar to the `apply_chat_template` method on tokenizers, this method applies a Jinja template to input
        conversations to turn them into a single tokenizable string.

        The input is expected to be in the following format, where each message content is a list consisting of text and
        optionally image or video inputs. One can also provide an image, video, URL or local path which will be used to form
        `pixel_values` when `return_dict=True`. If not provided, one will get only the formatted text, optionally tokenized text.

        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "url": "https://www.ilankelman.org/stopsigns/australia.jpg"},
                    {"type": "text", "text": "Please describe this image in detail."},
                ],
            },
        ]

        Args:
            conversation (`Union[list[Dict, [str, str]], list[list[dict[str, str]]]]`):
                The conversation to format.
            chat_template (`Optional[str]`, *optional*):
                The Jinja template to use for formatting the conversation. If not provided, the tokenizer's
                chat template is used.
        """
        if chat_template is None:
            if isinstance(self.chat_template, dict) and "default" in self.chat_template:
                chat_template = self.chat_template["default"]
            elif isinstance(self.chat_template, dict):
                raise ValueError(
                    'The processor has multiple chat templates but none of them are named "default". You need to specify'
                    " which one to use by passing the `chat_template` argument. Available templates are: "
                    f"{', '.join(self.chat_template.keys())}"
                )
            elif self.chat_template is not None:
                chat_template = self.chat_template
            else:
                raise ValueError(
                    "Cannot use apply_chat_template because this processor does not have a chat template."
                )
        else:
            if isinstance(self.chat_template, dict) and chat_template in self.chat_template:
                # It's the name of a template, not a full template string
                chat_template = self.chat_template[chat_template]
            else:
                # It's a template string, render it directly
                pass

        # Check if tokenizer is fast - use backend attribute if available, otherwise fall back to class name
        is_tokenizers_fast = False
        if hasattr(self, "tokenizer"):
            if hasattr(self.tokenizer, "backend"):
                is_tokenizers_fast = self.tokenizer.backend == "tokenizers"
            else:
                # Fallback to class name check
                is_tokenizers_fast = self.tokenizer.__class__.__name__.endswith("Fast")

        if kwargs.get("continue_final_message", False):
            if kwargs.get("add_generation_prompt", False):
                raise ValueError(
                    "continue_final_message and add_generation_prompt are not compatible. Use continue_final_message when you want the model to continue the final message, and add_generation_prompt when you want to add a header that will prompt it to start a new assistant message instead."
                )
            if kwargs.get("return_assistant_tokens_mask", False):
                raise ValueError("continue_final_message is not compatible with return_assistant_tokens_mask.")

        if kwargs.get("return_assistant_tokens_mask", False):
            if not is_tokenizers_fast:
                raise ValueError(
                    "`return_assistant_tokens_mask` is not possible with slow tokenizers. Make sure you have `tokenizers` installed. "
                    "If the error persists, open an issue to support a Fast tokenizer for your model."
                )
            else:
                kwargs["return_offsets_mapping"] = True  # force offset mapping so we can infer token boundaries

        # Fill sets of kwargs that should be used by jinja template, filtering out kwargs used in `processor.__call__`
        # NOTE: we don't only filter but also set the default values here. Without default values, we can remove it
        template_kwargs = {}
        for key in AllKwargsForChatTemplate.__annotations__["template_kwargs"].__annotations__:
            kwarg_type_defaults = AllKwargsForChatTemplate.__annotations__["template_kwargs"]
            default_value = getattr(kwarg_type_defaults, key, None)
            value = kwargs.pop(key, default_value)
            if value is not None and not isinstance(value, dict):
                template_kwargs[key] = value

        # Pass unprocessed custom kwargs
        template_kwargs.update(kwargs)

        # Set the sampling rate to load the audio files if user hasn't already passed with `kwargs`
        if "sampling_rate" not in template_kwargs:
            if hasattr(self, "feature_extractor") and hasattr(self.feature_extractor, "sampling_rate"):
                template_kwargs["sampling_rate"] = self.feature_extractor.sampling_rate
            else:
                template_kwargs["sampling_rate"] = 16_000

        if isinstance(conversation, (list, tuple)) and (
            isinstance(conversation[0], (list, tuple)) or hasattr(conversation[0], "content")
        ):
            is_batched = True
            conversations = conversation
        else:
            is_batched = False
            conversations = [conversation]

        # Normalize OpenAI-style "image_url" content blocks to HuggingFace-style "image" blocks
        # OpenAI format: {"type": "image_url", "image_url": {"url": "..."}}
        # HuggingFace format: {"type": "image", "url": "..."}
        for conversation_idx, conversation in enumerate(conversations):
            for message in conversation:
                if not isinstance(message.get("content"), list):
                    continue
                new_content = []
                for content in message["content"]:
                    if isinstance(content, dict) and content.get("type") == "image_url" and "image_url" in content:
                        image_url_info = content["image_url"]
                        url = image_url_info.get("url", "") if isinstance(image_url_info, dict) else image_url_info
                        new_content.append({"type": "image", "url": url})
                    else:
                        new_content.append(content)
                message["content"] = new_content

        tokenize = template_kwargs.pop("tokenize", False)
        return_dict = template_kwargs.pop("return_dict", True)

        if tokenize:
            batch_images, batch_videos = [], []
            batch_audios = []
            for conversation in conversations:
                images, videos = [], []
                for message in conversation:
                    visuals = [content for content in message["content"] if content["type"] in ["image", "video"]]
                    audio_fnames = [
                        content[key]
                        for content in message["content"]
                        for key in ["audio", "url", "path"]
                        if key in content and content["type"] == "audio"
                    ]
                    image_fnames = [
                        vision_info[key]
                        for vision_info in visuals
                        for key in ["image", "url", "path", "base64"]
                        if key in vision_info and vision_info["type"] == "image"
                    ]
                    images.extend(image_fnames)
                    video_fnames = [
                        vision_info[key]
                        for vision_info in visuals
                        for key in ["video", "url", "path"]
                        if key in vision_info and vision_info["type"] == "video"
                    ]
                    videos.extend(video_fnames)

                    # Audio models do not accept nested list of audios (yet!) so we construct a flat input audio list
                    if not template_kwargs["load_audio_from_video"]:
                        for fname in audio_fnames:
                            batch_audios.append(load_audio(fname, sampling_rate=template_kwargs["sampling_rate"]))
                    else:
                        for fname in video_fnames:
                            batch_audios.append(load_audio(fname, sampling_rate=template_kwargs["sampling_rate"]))

                # Currently all processors can accept nested list of batches, but not flat list of visuals
                # So we'll make a batched list of images and let the processor handle it
                batch_images.append(images)
                batch_videos.append(videos)

        special_tokens_map = {}
        if hasattr(self, "tokenizer") and hasattr(self.tokenizer, "special_tokens_map"):
            special_tokens = self.tokenizer.special_tokens_map
            # Filter out tokens that conflict with template kwargs
            special_tokens_map = {k: v for k, v in special_tokens.items() if k not in template_kwargs}

        prompt, generation_indices = render_jinja_template(
            conversations=conversations,
            chat_template=chat_template,
            **template_kwargs,  # different flags such as `return_assistant_mask`
            **special_tokens_map,  # tokenizer special tokens are used by some templates
        )

        if not is_batched:
            prompt = prompt[0]

        if tokenize:
            # Tokenizer's `apply_chat_template` never adds special tokens when tokenizing
            # But processor's `apply_chat_template` didn't have an option to tokenize, so users had to format the prompt
            # and pass it to the processor. Users thus never worried about special tokens relying on processor handling
            # everything internally. The below line is to keep BC for that and be able to work with model that have
            # special tokens in the template (consistent with tokenizers). We dont want to raise warning, it will flood command line
            # without actionable solution for users
            single_prompt = prompt[0] if is_batched else prompt
            if self.tokenizer.bos_token is not None and single_prompt.startswith(self.tokenizer.bos_token):
                kwargs["add_special_tokens"] = False

            # Always sample frames by default unless explicitly set to `False` by users. If users do not pass `num_frames`/`fps`
            # sampling should not done for BC.
            if "do_sample_frames" not in kwargs and (
                kwargs.get("fps") is not None or kwargs.get("num_frames") is not None
            ):
                kwargs["do_sample_frames"] = True

            images_exist = any((im is not None) for im_list in batch_images for im in im_list)
            videos_exist = any((vid is not None) for vid_list in batch_videos for vid in vid_list)
            out = self(
                text=prompt,
                images=batch_images if images_exist else None,
                videos=batch_videos if videos_exist else None,
                audio=batch_audios if batch_audios else None,
                **kwargs,
            )

            if return_dict:
                if template_kwargs.get("return_assistant_tokens_mask", False):
                    assistant_masks = []
                    offset_mapping = out.pop("offset_mapping")
                    input_ids = out["input_ids"]
                    for i in range(len(input_ids)):
                        current_mask = [0] * len(input_ids[i])
                        offsets = offset_mapping[i]
                        offset_starts = [start for start, end in offsets]
                        for assistant_start_char, assistant_end_char in generation_indices[i]:
                            start_pos = bisect.bisect_left(offset_starts, assistant_start_char)
                            end_pos = bisect.bisect_left(offset_starts, assistant_end_char)

                            if not (
                                start_pos >= 0
                                and start_pos < len(offsets)
                                and offsets[start_pos][0] <= assistant_start_char < offsets[start_pos][1]
                            ):
                                # start_token is out of bounds maybe due to truncation.
                                continue
                            # Ensure end_pos is also within bounds
                            if end_pos > len(input_ids[i]):
                                end_pos = len(input_ids[i])
                            for token_id in range(start_pos, end_pos if end_pos else len(input_ids[i])):
                                current_mask[token_id] = 1
                        assistant_masks.append(current_mask)
                    out["assistant_masks"] = assistant_masks
                    out.convert_to_tensors(tensor_type=kwargs.get("return_tensors"))
                return out
            else:
                return out["input_ids"]
        return prompt

    def post_process_multimodal_output(
        self, generated_outputs, skip_special_tokens=True, generation_mode=None, **kwargs
    ):
        """
        Post-process the output of a multimodal model to return the requested modality output.
        If the model cannot generated the requested modality, an error will be raised.

        Args:
            generated_outputs (`torch.Tensor` or `np.ndarray`):
                The output of the model `generate` function. The output is expected to be a tensor of shape `(batch_size, sequence_length)`
                or `(sequence_length,)`.
            skip_special_tokens (`bool`, *optional*, defaults to `True`):
                Whether or not to remove special tokens in the output. Argument passed to the tokenizer's `batch_decode` method.
            generation_mode (`str`, *optional*):
                Generation mode indicated which modality to output and can be one of `["text", "image", "audio"]`.
            **kwargs:
                Additional arguments to be passed to the tokenizer's `batch_decode method`.

        Returns:
            `list[str]`: The decoded text.
        """
        if generation_mode is not None and generation_mode != "text":
            raise ValueError(
                f"{self.__class__.__name__} got an unexpected generation_mode={generation_mode}. Supported options are only [`text`]"
            )
        return self.post_process_image_text_to_text(
            generated_outputs, skip_special_tokens=skip_special_tokens, **kwargs
        )

    def post_process_image_text_to_text(self, generated_outputs, skip_special_tokens=True, **kwargs):
        """
        Post-process the output of a vlm to decode the text.

        Args:
            generated_outputs (`torch.Tensor` or `np.ndarray`):
                The output of the model `generate` function. The output is expected to be a tensor of shape `(batch_size, sequence_length)`
                or `(sequence_length,)`.
            skip_special_tokens (`bool`, *optional*, defaults to `True`):
                Whether or not to remove special tokens in the output. Argument passed to the tokenizer's `decode` method.
            **kwargs:
                Additional arguments to be passed to the tokenizer's `decode` method.

        Returns:
            `list[str]`: The decoded text.
        """
        return self.tokenizer.decode(generated_outputs, skip_special_tokens=skip_special_tokens, **kwargs)

    def _check_special_mm_tokens(self, text: list[str], text_inputs: "BatchFeature", modalities: list[str]):
        """
        Checks that number of special tokens in text and processed text is same. The count can be different
        if tokenized text was truncated, leading to issues in model code.
        """
        for modality in modalities:
            token_str = getattr(self, f"{modality}_token")
            token_id = getattr(self, f"{modality}_token_id")
            ids_count = [list(ids).count(token_id) for ids in text_inputs["input_ids"]]
            text_count = [sample.count(token_str) for sample in text]

            if ids_count != text_count:
                raise ValueError(
                    f"Mismatch in `{modality}` token count between text and `input_ids`. Got ids={ids_count} and text={text_count}. "
                    "Likely due to `truncation='max_length'`. Please disable truncation or increase `max_length`."
                )


ProcessorMixin.push_to_hub = copy_func(ProcessorMixin.push_to_hub)
if ProcessorMixin.push_to_hub.__doc__ is not None:
    ProcessorMixin.push_to_hub.__doc__ = ProcessorMixin.push_to_hub.__doc__.format(
        object="processor", object_class="AutoProcessor", object_files="processor files"
    )
