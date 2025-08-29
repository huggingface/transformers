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
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, TypedDict, TypeVar, Union

import numpy as np
import typing_extensions
from huggingface_hub.errors import EntryNotFoundError

from .audio_utils import load_audio
from .dynamic_module_utils import custom_object_save
from .feature_extraction_utils import BatchFeature
from .image_utils import ChannelDimension, is_vision_available
from .utils.chat_template_utils import render_jinja_template
from .video_utils import VideoMetadata


if is_vision_available():
    from .image_utils import PILImageResampling


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
    download_url,
    is_offline_mode,
    is_remote_url,
    is_torch_available,
    list_repo_templates,
    logging,
)
from .utils.deprecation import deprecate_kwarg


if is_torch_available():
    from .modeling_utils import PreTrainedAudioTokenizerBase


logger = logging.get_logger(__name__)

# type hinting: specifying the type of processor class that inherits from ProcessorMixin
SpecificProcessorType = TypeVar("SpecificProcessorType", bound="ProcessorMixin")

# Dynamically import the Transformers module to grab the attribute classes of the processor from their names.
transformers_module = direct_transformers_import(Path(__file__).parent)


AUTO_TO_BASE_CLASS_MAPPING = {
    "AutoTokenizer": "PreTrainedTokenizerBase",
    "AutoFeatureExtractor": "FeatureExtractionMixin",
    "AutoImageProcessor": "ImageProcessingMixin",
    "AutoVideoProcessor": "BaseVideoProcessor",
}

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
    """

    text_pair: Optional[Union[TextInput, PreTokenizedInput, list[TextInput], list[PreTokenizedInput]]]
    text_target: Union[TextInput, PreTokenizedInput, list[TextInput], list[PreTokenizedInput]]
    text_pair_target: Optional[Union[TextInput, PreTokenizedInput, list[TextInput], list[PreTokenizedInput]]]
    add_special_tokens: Optional[bool]
    padding: Union[bool, str, PaddingStrategy]
    truncation: Union[bool, str, TruncationStrategy]
    max_length: Optional[int]
    stride: Optional[int]
    is_split_into_words: Optional[bool]
    pad_to_multiple_of: Optional[int]
    return_token_type_ids: Optional[bool]
    return_attention_mask: Optional[bool]
    return_overflowing_tokens: Optional[bool]
    return_special_tokens_mask: Optional[bool]
    return_offsets_mapping: Optional[bool]
    return_length: Optional[bool]
    verbose: Optional[bool]
    padding_side: Optional[str]
    return_mm_token_type_ids: Optional[bool]


class ImagesKwargs(TypedDict, total=False):
    """
    Keyword arguments for image processing. For extended documentation, check the appropriate ImageProcessor
    class methods and docstrings.

    Attributes:
        do_resize (`bool`, *optional*):
            Whether to resize the image.
        size (`dict[str, int]`, *optional*):
            Resize the shorter side of the input to `size["shortest_edge"]`.
        size_divisor (`int`, *optional*):
            The size by which to make sure both the height and width can be divided.
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
        image_mean (`float` or `list[float]`, *optional*):
            Mean to use if normalizing the image.
        image_std (`float` or `list[float]`, *optional*):
            Standard deviation to use if normalizing the image.
        do_pad (`bool`, *optional*):
            Whether to pad the image to the `(max_height, max_width)` of the images in the batch.
        pad_size (`dict[str, int]`, *optional*):
            The size `{"height": int, "width" int}` to pad the images to.
        do_center_crop (`bool`, *optional*):
            Whether to center crop the image.
        data_format (`ChannelDimension` or `str`, *optional*):
            The channel dimension format for the output image.
        input_data_format (`ChannelDimension` or `str`, *optional*):
            The channel dimension format for the input image.
        device (`str`, *optional*):
            The device to use for processing (e.g. "cpu", "cuda"), only relevant for fast image processing.
    """

    do_resize: Optional[bool]
    size: Optional[dict[str, int]]
    size_divisor: Optional[int]
    crop_size: Optional[dict[str, int]]
    resample: Optional[Union["PILImageResampling", int]]
    do_rescale: Optional[bool]
    rescale_factor: Optional[float]
    do_normalize: Optional[bool]
    image_mean: Optional[Union[float, list[float]]]
    image_std: Optional[Union[float, list[float]]]
    do_pad: Optional[bool]
    pad_size: Optional[dict[str, int]]
    do_center_crop: Optional[bool]
    data_format: Optional[ChannelDimension]
    input_data_format: Optional[Union[str, ChannelDimension]]
    device: Optional[str]


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
        size_divisor (`int`, *optional*):
            The size by which to make sure both the height and width can be divided.
        resample (`PILImageResampling`, *optional*):
            Resampling filter to use if resizing the video.
        do_rescale (`bool`, *optional*):
            Whether to rescale the video by the specified scale `rescale_factor`.
        rescale_factor (`int` or `float`, *optional*):
            Scale factor to use if rescaling the video.
        do_normalize (`bool`, *optional*):
            Whether to normalize the video.
        image_mean (`float` or `list[float]`, *optional*):
            Mean to use if normalizing the video.
        image_std (`float` or `list[float]`, *optional*):
            Standard deviation to use if normalizing the video.
        do_pad (`bool`, *optional*):
            Whether to pad the video to the `(max_height, max_width)` of the videos in the batch.
        do_center_crop (`bool`, *optional*):
            Whether to center crop the video.
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
        return_metadata (`ChannelDimension` or `str`, *optional*):
            Whether to return video metadata or not.
    """

    do_convert_rgb: Optional[bool]
    do_resize: Optional[bool]
    size: Optional[dict[str, int]]
    size_divisor: Optional[int]
    default_to_square: Optional[bool]
    resample: Optional["PILImageResampling"]
    do_rescale: Optional[bool]
    rescale_factor: Optional[float]
    do_normalize: Optional[bool]
    image_mean: Optional[Union[float, list[float]]]
    image_std: Optional[Union[float, list[float]]]
    do_pad: Optional[bool]
    do_center_crop: Optional[bool]
    crop_size: Optional[dict[str, int]]
    data_format: Optional[ChannelDimension]
    input_data_format: Optional[Union[str, ChannelDimension]]
    device: Optional[str]
    do_sample_frames: Optional[bool]
    video_metadata: Optional[Union[VideoMetadata, dict]]
    fps: Optional[Union[int, float]]
    num_frames: Optional[int]
    return_metadata: Optional[bool]


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
    """

    sampling_rate: Optional[int]
    raw_speech: Optional[Union["np.ndarray", list[float], list["np.ndarray"], list[list[float]]]]
    padding: Optional[Union[bool, str, PaddingStrategy]]
    max_length: Optional[int]
    truncation: Optional[bool]
    pad_to_multiple_of: Optional[int]
    return_attention_mask: Optional[bool]


class CommonKwargs(TypedDict, total=False):
    return_tensors: Optional[Union[str, TensorType]]


class ProcessingKwargs(TextKwargs, ImagesKwargs, VideosKwargs, AudioKwargs, CommonKwargs, total=False):
    """
    Base class for kwargs passing to processors.
    A model should have its own `ModelProcessorKwargs` class that inherits from `ProcessingKwargs` to provide:
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

    common_kwargs: CommonKwargs = {
        **CommonKwargs.__annotations__,
    }
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

    tools: Optional[list[dict]] = None
    documents: Optional[list[dict[str, str]]] = None
    add_generation_prompt: Optional[bool] = False
    continue_final_message: Optional[bool] = False
    return_assistant_tokens_mask: Optional[bool] = False


class ChatTemplateLoadKwargs(TypedDict, total=False):
    """
    Keyword arguments used to load multimodal data in processor chat templates.

    num_frames (`int`, *optional*):
        Number of frames to sample uniformly. If not passed, the whole video is loaded.
    load_audio_from_video (`bool`, *optional*):
            Whether to use the audio track of input video. If `True` the audio track will be loaded and passed to the
            processor. This flag has no effect if the model doesn't support audio modality.
    """

    sampling_rate: Optional[int] = 16_000
    load_audio_from_video: Optional[bool] = False


class ProcessorChatTemplateKwargs(ChatTemplateLoadKwargs, TokenizerChatTemplateKwargs, total=False):
    """
    Keyword arguments for processor's `apply_chat_template`.

    tokenize (`bool`, *optional*, defaults to `False`):
        Whether to tokenize the output or not.
    return_dict (`bool`, defaults to `False`):
        Whether to return a dictionary with named outputs. Has no effect if tokenize is `False`.
    """

    tokenize: Optional[bool] = False
    return_dict: Optional[bool] = False


class AllKwargsForChatTemplate(
    TextKwargs, ImagesKwargs, VideosKwargs, AudioKwargs, CommonKwargs, ProcessorChatTemplateKwargs
):
    processor_kwargs: ProcessingKwargs = {
        **ProcessingKwargs.__annotations__,
    }
    mm_load_kwargs: ChatTemplateLoadKwargs = {
        **TextKwargs.__annotations__,
    }
    template_kwargs: ProcessorChatTemplateKwargs = {
        **ProcessorChatTemplateKwargs.__annotations__,
    }


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

    num_image_tokens: list[int] = None
    num_video_tokens: list[int] = None
    num_audio_tokens: list[int] = None
    num_image_patches: list[int] = None

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

    attributes = ["feature_extractor", "tokenizer"]
    optional_attributes = ["chat_template", "audio_tokenizer"]
    optional_call_args: list[str] = []
    # Names need to be attr_class for attr in attributes
    feature_extractor_class = None
    tokenizer_class = None
    _auto_class = None

    # args have to match the attributes class attribute
    def __init__(self, *args, **kwargs):
        # First, extract optional attributes from kwargs if present
        # Optional attributes can never be positional arguments
        for optional_attribute in self.optional_attributes:
            optional_attribute_value = kwargs.pop(optional_attribute, None)
            setattr(self, optional_attribute, optional_attribute_value)

            # Check audio tokenizer for its class but do not treat it as attr to avoid saving weights
            if optional_attribute == "audio_tokenizer" and optional_attribute_value is not None:
                proper_class = self.check_argument_for_proper_class(optional_attribute, optional_attribute_value)

                if not (is_torch_available() and isinstance(optional_attribute_value, PreTrainedAudioTokenizerBase)):
                    raise ValueError(
                        f"Tried to use `{proper_class}` for audio tokenization. However, this class is not"
                        " registered for audio tokenization."
                    )

        # Sanitize args and kwargs
        for key in kwargs:
            if key not in self.attributes:
                raise TypeError(f"Unexpected keyword argument {key}.")
        for arg, attribute_name in zip(args, self.attributes):
            if attribute_name in kwargs:
                raise TypeError(f"Got multiple values for argument {attribute_name}.")
            else:
                kwargs[attribute_name] = arg

        if len(kwargs) != len(self.attributes):
            raise ValueError(
                f"This processor requires {len(self.attributes)} arguments: {', '.join(self.attributes)}. Got "
                f"{len(args)} arguments instead."
            )

        # Check each arg is of the proper class (this will also catch a user initializing in the wrong order)
        for attribute_name, arg in kwargs.items():
            self.check_argument_for_proper_class(attribute_name, arg)
            setattr(self, attribute_name, arg)

    def check_argument_for_proper_class(self, argument_name, argument):
        """
        Checks the passed argument's class against the expected transformers class. In case of an unexpected
        mismatch between expected and actual class, an error is raise. Otherwise, the proper retrieved class
        is returned.
        """
        class_name = getattr(self, f"{argument_name}_class")
        # Nothing is ever going to be an instance of "AutoXxx", in that case we check the base class.
        class_name = AUTO_TO_BASE_CLASS_MAPPING.get(class_name, class_name)
        if isinstance(class_name, tuple):
            proper_class = tuple(self.get_possibly_dynamic_module(n) for n in class_name if n is not None)
        else:
            proper_class = self.get_possibly_dynamic_module(class_name)

        if not isinstance(argument, proper_class):
            raise TypeError(
                f"Received a {type(argument).__name__} for argument {argument_name}, but a {class_name} was expected."
            )

        return proper_class

    def to_dict(self, legacy_serialization=True) -> dict[str, Any]:
        """
        Serializes this instance to a Python dictionary.

        Returns:
            `dict[str, Any]`: Dictionary of all the attributes that make up this processor instance.
        """
        output = copy.deepcopy(self.__dict__)

        # Get the kwargs in `__init__`.
        sig = inspect.signature(self.__init__)
        # Only save the attributes that are presented in the kwargs of `__init__`.
        attrs_to_save = list(sig.parameters)
        # extra attributes to be kept
        attrs_to_save += ["auto_map"]

        if legacy_serialization:
            # Don't save attributes like `tokenizer`, `image processor` etc. in processor config if `legacy=True`
            attrs_to_save = [x for x in attrs_to_save if x not in self.__class__.attributes]

        if "tokenizer" in output:
            del output["tokenizer"]
        if "qformer_tokenizer" in output:
            del output["qformer_tokenizer"]
        if "protein_tokenizer" in output:
            del output["protein_tokenizer"]
        if "chat_template" in output:
            del output["chat_template"]

        # Serialize attributes as a dict
        output = {
            k: v.to_dict() if isinstance(v, PushToHubMixin) else v
            for k, v in output.items()
            if (
                k in attrs_to_save  # keep all attributes that have to be serialized
                and v.__class__.__name__ != "BeamSearchDecoderCTC"  # remove attributes with that are objects
                and (
                    (legacy_serialization and not isinstance(v, PushToHubMixin)) or not legacy_serialization
                )  # remove `PushToHubMixin` objects
            )
        }

        # Special case, add `audio_tokenizer` dict which points to model weights and path
        if not legacy_serialization and "audio_tokenizer" in output:
            audio_tokenizer_dict = {
                "audio_tokenizer_class": self.audio_tokenizer.__class__.__name__,
                "audio_tokenizer_name_or_path": self.audio_tokenizer.name_or_path,
            }
            # Update or overwrite, what do audio tokenizers expect when loading?
            output["audio_tokenizer"] = audio_tokenizer_dict

        output["processor_class"] = self.__class__.__name__

        return output

    def to_json_string(self, legacy_serialization=True) -> str:
        """
        Serializes this instance to a JSON string.

        Returns:
            `str`: String containing all the attributes that make up this feature_extractor instance in JSON format.
        """
        dictionary = self.to_dict(legacy_serialization=legacy_serialization)

        return json.dumps(dictionary, indent=2, sort_keys=True) + "\n"

    def to_json_file(self, json_file_path: Union[str, os.PathLike], legacy_serialization=True):
        """
        Save this instance to a JSON file.

        Args:
            json_file_path (`str` or `os.PathLike`):
                Path to the JSON file in which this processor instance's parameters will be saved.
        """
        with open(json_file_path, "w", encoding="utf-8") as writer:
            writer.write(self.to_json_string(legacy_serialization=legacy_serialization))

    def __repr__(self):
        attributes_repr = [f"- {name}: {repr(getattr(self, name))}" for name in self.attributes]
        attributes_repr = "\n".join(attributes_repr)
        return f"{self.__class__.__name__}:\n{attributes_repr}\n\n{self.to_json_string()}"

    def save_pretrained(self, save_directory, push_to_hub: bool = False, legacy_serialization: bool = True, **kwargs):
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
            legacy_serialization (`bool`, *optional*, defaults to `True`):
                Whether or not to save processor attributes in separate config files (legacy) or in processor's config
                file as a nested dict. Saving all attributes in a single dict will become the default in future versions.
                Set to `legacy_serialization=True` until then.
            kwargs (`dict[str, Any]`, *optional*):
                Additional key word arguments passed along to the [`~utils.PushToHubMixin.push_to_hub`] method.
        """
        use_auth_token = kwargs.pop("use_auth_token", None)

        if use_auth_token is not None:
            warnings.warn(
                "The `use_auth_token` argument is deprecated and will be removed in v5 of Transformers. Please use `token` instead.",
                FutureWarning,
            )
            if kwargs.get("token") is not None:
                raise ValueError(
                    "`token` and `use_auth_token` are both specified. Please set only the argument `token`."
                )
            kwargs["token"] = use_auth_token

        os.makedirs(save_directory, exist_ok=True)

        if push_to_hub:
            commit_message = kwargs.pop("commit_message", None)
            repo_id = kwargs.pop("repo_id", save_directory.split(os.path.sep)[-1])
            repo_id = self._create_repo(repo_id, **kwargs)
            files_timestamps = self._get_files_timestamps(save_directory)
        # If we have a custom config, we copy the file defining it in the folder and set the attributes so it can be
        # loaded from the Hub.
        if self._auto_class is not None:
            attrs = [getattr(self, attribute_name) for attribute_name in self.attributes]
            configs = [(a.init_kwargs if isinstance(a, PreTrainedTokenizerBase) else a) for a in attrs]
            configs.append(self)
            custom_object_save(self, save_directory, config=configs)

        save_jinja_files = kwargs.get("save_jinja_files", True)

        for attribute_name in self.attributes:
            # Save the tokenizer in its own vocab file. The other attributes are saved as part of `processor_config.json`
            if attribute_name == "tokenizer":
                attribute = getattr(self, attribute_name)
                if hasattr(attribute, "_set_processor_class"):
                    attribute._set_processor_class(self.__class__.__name__)

                # Propagate save_jinja_files to tokenizer to ensure we don't get conflicts
                attribute.save_pretrained(save_directory, save_jinja_files=save_jinja_files)
            elif legacy_serialization:
                attribute = getattr(self, attribute_name)
                # Include the processor class in attribute config so this processor can then be reloaded with `AutoProcessor` API.
                if hasattr(attribute, "_set_processor_class"):
                    attribute._set_processor_class(self.__class__.__name__)
                attribute.save_pretrained(save_directory)

        if self._auto_class is not None:
            # We added an attribute to the init_kwargs of the tokenizers, which needs to be cleaned up.
            for attribute_name in self.attributes:
                attribute = getattr(self, attribute_name)
                if isinstance(attribute, PreTrainedTokenizerBase):
                    del attribute.init_kwargs["auto_map"]

        # If we save using the predefined names, we can load using `from_pretrained`
        # plus we save chat_template in its own file
        output_processor_file = os.path.join(save_directory, PROCESSOR_NAME)
        output_chat_template_file_jinja = os.path.join(save_directory, CHAT_TEMPLATE_FILE)
        output_chat_template_file_legacy = os.path.join(
            save_directory, LEGACY_PROCESSOR_CHAT_TEMPLATE_FILE
        )  # Legacy filename
        chat_template_dir = os.path.join(save_directory, CHAT_TEMPLATE_DIR)

        # Save `chat_template` in its own file. We can't get it from `processor_dict` as we popped it in `to_dict`
        # to avoid serializing chat template in json config file. So let's get it from `self` directly
        if self.chat_template is not None:
            save_jinja_files = kwargs.get("save_jinja_files", True)
            is_single_template = isinstance(self.chat_template, str)
            if save_jinja_files and is_single_template:
                # New format for single templates is to save them as chat_template.jinja
                with open(output_chat_template_file_jinja, "w", encoding="utf-8") as f:
                    f.write(self.chat_template)
                logger.info(f"chat template saved in {output_chat_template_file_jinja}")
            elif save_jinja_files and not is_single_template:
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
            elif is_single_template:
                # Legacy format for single templates: Put them in chat_template.json
                chat_template_json_string = (
                    json.dumps({"chat_template": self.chat_template}, indent=2, sort_keys=True) + "\n"
                )
                with open(output_chat_template_file_legacy, "w", encoding="utf-8") as writer:
                    writer.write(chat_template_json_string)
                logger.info(f"chat template saved in {output_chat_template_file_legacy}")
            elif self.chat_template is not None:
                # At this point we have multiple templates in the legacy format, which is not supported
                # chat template dicts are saved to chat_template.json as lists of dicts with fixed key names.
                raise ValueError(
                    "Multiple chat templates are not supported in the legacy format. Please save them as "
                    "separate files using the `save_jinja_files` argument."
                )

        if legacy_serialization:
            output_audio_tokenizer_file = os.path.join(save_directory, AUDIO_TOKENIZER_NAME)
            processor_dict = self.to_dict()

            # For now, let's not save to `processor_config.json` if the processor doesn't have extra attributes and
            # `auto_map` is not specified.
            if set(processor_dict.keys()) != {"processor_class"}:
                self.to_json_file(output_processor_file)
                logger.info(f"processor saved in {output_processor_file}")

            if set(processor_dict.keys()) == {"processor_class"}:
                return_files = []
            else:
                return_files = [output_processor_file]

            if self.audio_tokenizer is not None:
                audio_tokenizer_class = self.audio_tokenizer.__class__.__name__
                audio_tokenizer_name_or_path = self.audio_tokenizer.name_or_path
                audio_tokenizer_dict = {
                    "audio_tokenizer_class": audio_tokenizer_class,
                    "audio_tokenizer_name_or_path": audio_tokenizer_name_or_path,
                }
                audio_tokenizer_json = json.dumps(audio_tokenizer_dict, indent=2, sort_keys=True) + "\n"
                with open(output_audio_tokenizer_file, "w", encoding="utf-8") as writer:
                    writer.write(audio_tokenizer_json)

        # Create a unified `preprocessor_config.json` and save all attributes as a composite config, except for tokenizers
        # NOTE: this will become the default way to save all processor attrbiutes in future versions. Toggled off for now to give
        # us time for smoother transition
        else:
            self.to_json_file(output_processor_file, legacy_serialization=False)
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
        cls, pretrained_model_name_or_path: Union[str, os.PathLike], **kwargs
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
        resume_download = kwargs.pop("resume_download", None)
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
        elif is_remote_url(pretrained_model_name_or_path):
            processor_file = pretrained_model_name_or_path
            resolved_processor_file = download_url(pretrained_model_name_or_path)
            # can't load chat-template and audio tokenizer when given a file url as pretrained_model_name_or_path
            resolved_chat_template_file = None
            resolved_raw_chat_template_file = None
            resolved_audio_tokenizer_file = None
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
                    ):
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
                    resume_download=resume_download,
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
                    resume_download=resume_download,
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
                    resume_download=resume_download,
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
                        resume_download=resume_download,
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
                    resume_download=resume_download,
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

        if chat_templates:
            kwargs["chat_template"] = chat_templates

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

        if "chat_template" in processor_dict and processor_dict["chat_template"] is not None:
            logger.warning_once(
                "Chat templates should be in a 'chat_template.jinja' file but found key='chat_template' "
                "in the processor's config. Make sure to move your template to its own file."
            )

        if "chat_template" in kwargs:
            processor_dict["chat_template"] = kwargs.pop("chat_template")

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

        # Pop attributes if saved in a single processor dict, they are loaded in `_get_arguments_from_pretrained`
        for attribute in cls.attributes:
            processor_dict.pop(attribute, None)

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
        # If we don't pop, some specific kwargs will raise a warning
        if "processor_class" in processor_dict:
            del processor_dict["processor_class"]

        if "auto_map" in processor_dict:
            del processor_dict["auto_map"]

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
        tokenizer_init_kwargs: Optional[dict] = None,
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
        # Initialize dictionaries
        output_kwargs = {
            "text_kwargs": {},
            "images_kwargs": {},
            "audio_kwargs": {},
            "videos_kwargs": {},
            "common_kwargs": {},
        }

        default_kwargs = {
            "text_kwargs": {},
            "images_kwargs": {},
            "audio_kwargs": {},
            "videos_kwargs": {},
            "common_kwargs": {},
        }

        possible_modality_keywords = {"text", "audio", "videos", "images"}
        used_keys = set()

        # get defaults from set model processor kwargs if they exist
        for modality in default_kwargs:  # noqa: PLC0206
            default_kwargs[modality] = ModelProcessorKwargs._defaults.get(modality, {}).copy()
            # update defaults with arguments from tokenizer init
            for modality_key in ModelProcessorKwargs.__annotations__[modality].__annotations__:
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

        # update modality kwargs with passed kwargs
        non_modality_kwargs = set(kwargs) - set(output_kwargs)
        for modality, output_kwarg in output_kwargs.items():
            for modality_key in ModelProcessorKwargs.__annotations__[modality].__annotations__:
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
                if key not in used_keys:
                    if key in ModelProcessorKwargs.__annotations__["common_kwargs"].__annotations__:
                        output_kwargs["common_kwargs"][key] = kwarg
                    elif key not in possible_modality_keywords:
                        logger.warning_once(
                            f"Keyword argument `{key}` is not a valid argument for this processor and will be ignored."
                        )

        # all modality-specific kwargs are updated with common kwargs
        for kwarg in output_kwargs.values():
            kwarg.update(output_kwargs["common_kwargs"])
        return output_kwargs

    @classmethod
    def from_pretrained(
        cls: type[SpecificProcessorType],
        pretrained_model_name_or_path: Union[str, os.PathLike],
        cache_dir: Optional[Union[str, os.PathLike]] = None,
        force_download: bool = False,
        local_files_only: bool = False,
        token: Optional[Union[str, bool]] = None,
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

        use_auth_token = kwargs.pop("use_auth_token", None)
        if use_auth_token is not None:
            warnings.warn(
                "The `use_auth_token` argument is deprecated and will be removed in v5 of Transformers. Please use `token` instead.",
                FutureWarning,
            )
            if token is not None:
                raise ValueError(
                    "`token` and `use_auth_token` are both specified. Please set only the argument `token`."
                )
            token = use_auth_token

        if token is not None:
            kwargs["token"] = token

        args = cls._get_arguments_from_pretrained(pretrained_model_name_or_path, **kwargs)
        processor_dict, kwargs = cls.get_processor_dict(pretrained_model_name_or_path, **kwargs)
        return cls.from_args_and_dict(args, processor_dict, **kwargs)

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
    def _get_arguments_from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        """
        Identify and instantiate the subcomponents of Processor classes, like image processors and
        tokenizers. This method uses the Processor attributes like `tokenizer_class` to figure out what class those
        subcomponents should be. Note that any subcomponents must either be library classes that are accessible in
        the `transformers` root, or they must be custom code that has been registered with the relevant autoclass,
        via methods like `AutoTokenizer.register()`. If neither of these conditions are fulfilled, this method
        will be unable to find the relevant subcomponent class and will raise an error.
        """
        args = []
        for attribute_name in cls.attributes:
            class_name = getattr(cls, f"{attribute_name}_class")
            if isinstance(class_name, tuple):
                classes = tuple(cls.get_possibly_dynamic_module(n) if n is not None else None for n in class_name)
                if attribute_name == "image_processor":
                    # TODO: @yoni, change logic in v4.52 (when use_fast set to True by default)
                    use_fast = kwargs.get("use_fast")
                    if use_fast is None:
                        logger.warning_once(
                            "Using a slow image processor as `use_fast` is unset and a slow processor was saved with this model. "
                            "`use_fast=True` will be the default behavior in v4.52, even if the model was saved with a slow processor. "
                            "This will result in minor differences in outputs. You'll still be able to use a slow processor with `use_fast=False`."
                        )
                else:
                    use_fast = kwargs.get("use_fast", True)
                if use_fast and classes[1] is not None:
                    attribute_class = classes[1]
                else:
                    attribute_class = classes[0]
            else:
                attribute_class = cls.get_possibly_dynamic_module(class_name)

            args.append(attribute_class.from_pretrained(pretrained_model_name_or_path, **kwargs))

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
        for attribute_name in self.attributes:
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

    @deprecate_kwarg("video_fps", version="4.58", new_name="fps")
    @deprecate_kwarg(
        "video_load_backend",
        version="4.59",
        additional_message=". This function will use `torchcodec` by default, or `torchvision` if `torchcodec` is not installed.",
    )
    def apply_chat_template(
        self,
        conversation: Union[list[dict[str, str]], list[list[dict[str, str]]]],
        chat_template: Optional[str] = None,
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

        is_tokenizers_fast = hasattr(self, "tokenizer") and self.tokenizer.__class__.__name__.endswith("Fast")

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

        # Fill sets of kwargs that should be used by different parts of template
        processed_kwargs = {
            "mm_load_kwargs": {},
            "template_kwargs": {},
        }

        for kwarg_type in processed_kwargs:
            for key in AllKwargsForChatTemplate.__annotations__[kwarg_type].__annotations__:
                kwarg_type_defaults = AllKwargsForChatTemplate.__annotations__[kwarg_type]
                default_value = getattr(kwarg_type_defaults, key, None)
                value = kwargs.pop(key, default_value)
                if value is not None and not isinstance(value, dict):
                    processed_kwargs[kwarg_type][key] = value

        # pop unused and deprecated kwarg
        kwargs.pop("video_load_backend", None)

        # Pass unprocessed custom kwargs
        processed_kwargs["template_kwargs"].update(kwargs)

        if isinstance(conversation, (list, tuple)) and (
            isinstance(conversation[0], (list, tuple)) or hasattr(conversation[0], "content")
        ):
            is_batched = True
            conversations = conversation
        else:
            is_batched = False
            conversations = [conversation]

        tokenize = processed_kwargs["template_kwargs"].pop("tokenize", False)
        return_dict = processed_kwargs["template_kwargs"].pop("return_dict", False)
        mm_load_kwargs = processed_kwargs["mm_load_kwargs"]

        if tokenize:
            batch_images, batch_videos = [], []
            batch_audios = []
            for conversation in conversations:
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
                    video_fnames = [
                        vision_info[key]
                        for vision_info in visuals
                        for key in ["video", "url", "path"]
                        if key in vision_info and vision_info["type"] == "video"
                    ]

                    # Audio models do not accept nested list of audios (yet!) so we construct a flat input audio list
                    if not mm_load_kwargs["load_audio_from_video"]:
                        for fname in audio_fnames:
                            batch_audios.append(load_audio(fname, sampling_rate=mm_load_kwargs["sampling_rate"]))
                    else:
                        for fname in video_fnames:
                            batch_audios.append(load_audio(fname, sampling_rate=mm_load_kwargs["sampling_rate"]))

                    # Currently all processors can accept nested list of batches, but not flat list of visuals
                    # So we'll make a batched list of images and let the processor handle it
                    if image_fnames:
                        batch_images.append(image_fnames)
                    if video_fnames:
                        batch_videos.append(video_fnames)

        prompt, generation_indices = render_jinja_template(
            conversations=conversations,
            chat_template=chat_template,
            **processed_kwargs["template_kwargs"],  # different flags such as `return_assistant_mask`
            **self.tokenizer.special_tokens_map,  # tokenizer special tokens are used by some templates
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

            # Always sample frames by default unless explicitly set to `False` by users. If users do not pass `num_frames`/`video_fps`
            # sampling should not done for BC.
            if "do_sample_frames" not in kwargs and ("fps" in kwargs or "num_frames" in kwargs):
                kwargs["do_sample_frames"] = True

            out = self(
                text=prompt,
                images=batch_images if batch_images else None,
                videos=batch_videos if batch_videos else None,
                audio=batch_audios if batch_audios else None,
                **kwargs,
            )

            if return_dict:
                if processed_kwargs["template_kwargs"].get("return_assistant_tokens_mask", False):
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
                                and offsets[start_pos][0] <= assistant_start_char < offsets[start_pos][1]
                            ):
                                # start_token is out of bounds maybe due to truncation.
                                continue
                            for token_id in range(start_pos, end_pos if end_pos else len(input_ids[i])):
                                current_mask[token_id] = 1
                        assistant_masks.append(current_mask)
                    out["assistant_masks"] = assistant_masks
                    out.convert_to_tensors(tensor_type=kwargs.get("return_tensors"))
                return out
            else:
                return out["input_ids"]
        return prompt

    def post_process_image_text_to_text(self, generated_outputs, skip_special_tokens=True, **kwargs):
        """
        Post-process the output of a vlm to decode the text.

        Args:
            generated_outputs (`torch.Tensor` or `np.ndarray`):
                The output of the model `generate` function. The output is expected to be a tensor of shape `(batch_size, sequence_length)`
                or `(sequence_length,)`.
            skip_special_tokens (`bool`, *optional*, defaults to `True`):
                Whether or not to remove special tokens in the output. Argument passed to the tokenizer's `batch_decode` method.
            **kwargs:
                Additional arguments to be passed to the tokenizer's `batch_decode method`.

        Returns:
            `list[str]`: The decoded text.
        """
        return self.tokenizer.batch_decode(generated_outputs, skip_special_tokens=skip_special_tokens, **kwargs)

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
