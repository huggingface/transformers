# Copyright 2025 The HuggingFace Inc. team.
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

import os
from typing import Any, TypeVar

from .audio_utils import is_valid_audio, load_audio
from .feature_extraction_utils import BatchFeature as BaseBatchFeature
from .preprocessing_base import PreprocessingMixin
from .utils import (
    FEATURE_EXTRACTOR_NAME,
    copy_func,
    logging,
)


AudioProcessorType = TypeVar("AudioProcessorType", bound="AudioProcessingMixin")


logger = logging.get_logger(__name__)


class BatchFeature(BaseBatchFeature):
    r"""
    Holds the output of the audio processor specific `__call__` methods.

    This class is derived from a python dictionary and can be used as a dictionary.

    Args:
        data (`dict`):
            Dictionary of lists/arrays/tensors returned by the __call__ method ('input_values', 'input_features', etc.).
        tensor_type (`Union[None, str, TensorType]`, *optional*):
            You can give a tensor_type here to convert the lists of integers in PyTorch/Numpy Tensors at
            initialization.
    """


class AudioProcessingMixin(PreprocessingMixin):
    """
    This is an audio processor mixin used to provide saving/loading functionality for audio processors.
    """

    _config_name = FEATURE_EXTRACTOR_NAME
    _type_key = "audio_processor_type"
    _nested_config_keys = ["audio_processor", "feature_extractor"]
    _auto_class_default = "AutoFeatureExtractor"
    _file_type_label = "audio processor"
    _excluded_dict_keys = {"mel_filters", "window"}
    _extra_init_pops = ["feature_extractor_type"]
    _config_filename_kwarg = "audio_processor_filename"
    _subfolder_default = ""

    @classmethod
    def get_audio_processor_dict(
        cls, pretrained_model_name_or_path: str | os.PathLike, **kwargs
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """
        From a `pretrained_model_name_or_path`, resolve to a dictionary of parameters, to be used for instantiating an
        audio processor of type [`~audio_processing_base.AudioProcessingMixin`] using `from_dict`.

        Parameters:
            pretrained_model_name_or_path (`str` or `os.PathLike`):
                The identifier of the pre-trained checkpoint from which we want the dictionary of parameters.
            subfolder (`str`, *optional*, defaults to `""`):
                In case the relevant files are located inside a subfolder of the model repo on huggingface.co, you can
                specify the folder name here.
            audio_processor_filename (`str`, *optional*, defaults to `"preprocessor_config.json"`):
                The name of the file in the model directory to use for the audio processor config.

        Returns:
            `tuple[Dict, Dict]`: The dictionary(ies) that will be used to instantiate the audio processor object.
        """
        return cls._get_config_dict(pretrained_model_name_or_path, **kwargs)

    def fetch_audio(self, audio_url_or_urls: str | list[str] | list[list[str]]):
        """
        Convert a single or a list of urls into the corresponding `np.ndarray` objects.

        If a single url is passed, the return value will be a single object. If a list is passed a list of objects is
        returned.
        """
        if isinstance(audio_url_or_urls, list):
            return [self.fetch_audio(x) for x in audio_url_or_urls]
        elif isinstance(audio_url_or_urls, str):
            return load_audio(audio_url_or_urls)
        elif is_valid_audio(audio_url_or_urls):
            return audio_url_or_urls
        else:
            raise TypeError(f"only a single or a list of entries is supported but got type={type(audio_url_or_urls)}")


AudioProcessingMixin.push_to_hub = copy_func(AudioProcessingMixin.push_to_hub)
if AudioProcessingMixin.push_to_hub.__doc__ is not None:
    AudioProcessingMixin.push_to_hub.__doc__ = AudioProcessingMixin.push_to_hub.__doc__.format(
        object="audio processor", object_class="AutoFeatureExtractor", object_files="audio processor file"
    )
