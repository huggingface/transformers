# Copyright 2021 The HuggingFace Inc. team.
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
Feature extraction saving/loading class for common feature extractors.
"""

import os
from typing import TYPE_CHECKING, Any, TypeVar, Union

from .preprocessing_base import BatchFeature as BatchFeature
from .preprocessing_base import PreprocessingMixin
from .utils import (
    FEATURE_EXTRACTOR_NAME,
    copy_func,
    logging,
)


if TYPE_CHECKING:
    from .feature_extraction_sequence_utils import SequenceFeatureExtractor


logger = logging.get_logger(__name__)

PreTrainedFeatureExtractor = Union["SequenceFeatureExtractor"]

# type hinting: specifying the type of feature extractor class that inherits from FeatureExtractionMixin
SpecificFeatureExtractorType = TypeVar("SpecificFeatureExtractorType", bound="FeatureExtractionMixin")


class FeatureExtractionMixin(PreprocessingMixin):
    """
    Saving/loading mixin for feature extractors.

    All the shared config resolution / serialization / hub logic now lives in [`PreprocessingMixin`];
    this class only carries the feature-extractor identity attributes and the legacy method aliases.
    """

    _config_name = FEATURE_EXTRACTOR_NAME
    _type_key = "feature_extractor_type"
    # `audio_processor` is checked too: processors whose audio attribute is named that way
    # (e.g. GraniteSpeechProcessor) nest their feature-extractor config under that key.
    _nested_config_keys = ["feature_extractor", "audio_processor"]
    _auto_class_default = "AutoFeatureExtractor"
    _file_type_label = "feature extractor"
    _config_filename_kwarg = None

    @classmethod
    def get_feature_extractor_dict(
        cls, pretrained_model_name_or_path: str | os.PathLike, **kwargs
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """
        From a `pretrained_model_name_or_path`, resolve to a dictionary of parameters, to be used for instantiating a
        feature extractor of type [`~feature_extraction_utils.FeatureExtractionMixin`] using `from_dict`.

        Parameters:
            pretrained_model_name_or_path (`str` or `os.PathLike`):
                The identifier of the pre-trained checkpoint from which we want the dictionary of parameters.

        Returns:
            `tuple[Dict, Dict]`: The dictionary(ies) that will be used to instantiate the feature extractor object.
        """
        return cls._get_config_dict(pretrained_model_name_or_path, **kwargs)

    @classmethod
    def get_config_dict(
        cls, pretrained_model_name_or_path: str | os.PathLike, **kwargs
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        # Subclasses may override `get_feature_extractor_dict`.
        return cls.get_feature_extractor_dict(pretrained_model_name_or_path, **kwargs)

    @classmethod
    def from_dict(
        cls, feature_extractor_dict: dict[str, Any], **kwargs
    ) -> Union["FeatureExtractionMixin", tuple["FeatureExtractionMixin", dict[str, Any]]]:
        """
        Instantiates a type of [`~feature_extraction_utils.FeatureExtractionMixin`] from a Python dictionary of
        parameters.

        Args:
            feature_extractor_dict (`dict[str, Any]`):
                Dictionary that will be used to instantiate the feature extractor object. Such a dictionary can be
                retrieved from a pretrained checkpoint by leveraging the
                [`~feature_extraction_utils.FeatureExtractionMixin.to_dict`] method.
            kwargs (`dict[str, Any]`):
                Additional parameters from which to initialize the feature extractor object.

        Returns:
            [`~feature_extraction_utils.FeatureExtractionMixin`]: The feature extractor object instantiated from those
            parameters.
        """
        return_unused_kwargs = kwargs.pop("return_unused_kwargs", False)

        # Update feature_extractor with kwargs if needed
        to_remove = []
        for key, value in kwargs.items():
            if key in feature_extractor_dict:
                feature_extractor_dict[key] = value
                to_remove.append(key)
        for key in to_remove:
            kwargs.pop(key, None)

        feature_extractor = cls(**feature_extractor_dict)

        logger.info(f"Feature extractor {feature_extractor}")
        if return_unused_kwargs:
            return feature_extractor, kwargs
        else:
            return feature_extractor


FeatureExtractionMixin.push_to_hub = copy_func(FeatureExtractionMixin.push_to_hub)
if FeatureExtractionMixin.push_to_hub.__doc__ is not None:
    FeatureExtractionMixin.push_to_hub.__doc__ = FeatureExtractionMixin.push_to_hub.__doc__.format(
        object="feature extractor", object_class="AutoFeatureExtractor", object_files="feature extractor file"
    )
