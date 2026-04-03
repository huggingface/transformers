# Copyright 2020 The HuggingFace Inc. team.
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

from .feature_extraction_utils import BatchFeature as BaseBatchFeature
from .image_utils import is_valid_image, load_image
from .preprocessing_base import PreprocessingMixin
from .utils import (
    IMAGE_PROCESSOR_NAME,
    copy_func,
    logging,
)


ImageProcessorType = TypeVar("ImageProcessorType", bound="ImageProcessingMixin")


logger = logging.get_logger(__name__)


# TODO: Move BatchFeature to be imported by both image_processing_utils and image_processing_utils_fast
# We override the class string here, but logic is the same.
class BatchFeature(BaseBatchFeature):
    r"""
    Holds the output of the image processor specific `__call__` methods.

    This class is derived from a python dictionary and can be used as a dictionary.

    Args:
        data (`dict`):
            Dictionary of lists/arrays/tensors returned by the __call__ method ('pixel_values', etc.).
        tensor_type (`Union[None, str, TensorType]`, *optional*):
            You can give a tensor_type here to convert the lists of integers in PyTorch/Numpy Tensors at
            initialization.
    """


# TODO: (Amy) - factor out the common parts of this and the feature extractor
class ImageProcessingMixin(PreprocessingMixin):
    """
    This is an image processor mixin used to provide saving/loading functionality for sequential and image feature
    extractors.
    """

    _config_name = IMAGE_PROCESSOR_NAME
    _type_key = "image_processor_type"
    _nested_config_keys = ["image_processor"]
    _auto_class_default = "AutoImageProcessor"
    _file_type_label = "image processor"
    _excluded_dict_keys = set()
    _extra_init_pops = ["feature_extractor_type"]
    _config_filename_kwarg = "image_processor_filename"
    _subfolder_default = ""

    @classmethod
    def get_image_processor_dict(
        cls, pretrained_model_name_or_path: str | os.PathLike, **kwargs
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """
        From a `pretrained_model_name_or_path`, resolve to a dictionary of parameters, to be used for instantiating a
        image processor of type [`~image_processor_utils.ImageProcessingMixin`] using `from_dict`.

        Parameters:
            pretrained_model_name_or_path (`str` or `os.PathLike`):
                The identifier of the pre-trained checkpoint from which we want the dictionary of parameters.
            subfolder (`str`, *optional*, defaults to `""`):
                In case the relevant files are located inside a subfolder of the model repo on huggingface.co, you can
                specify the folder name here.
            image_processor_filename (`str`, *optional*, defaults to `"config.json"`):
                The name of the file in the model directory to use for the image processor config.

        Returns:
            `tuple[Dict, Dict]`: The dictionary(ies) that will be used to instantiate the image processor object.
        """
        return cls._get_config_dict(pretrained_model_name_or_path, **kwargs)

    def fetch_images(self, image_url_or_urls: str | list[str] | list[list[str]]):
        """
        Convert a single or a list of urls into the corresponding `PIL.Image` objects.

        If a single url is passed, the return value will be a single object. If a list is passed a list of objects is
        returned.
        """
        if isinstance(image_url_or_urls, list):
            return [self.fetch_images(x) for x in image_url_or_urls]
        elif isinstance(image_url_or_urls, str):
            return load_image(image_url_or_urls)
        elif is_valid_image(image_url_or_urls):
            return image_url_or_urls
        else:
            raise TypeError(f"only a single or a list of entries is supported but got type={type(image_url_or_urls)}")


ImageProcessingMixin.push_to_hub = copy_func(ImageProcessingMixin.push_to_hub)
if ImageProcessingMixin.push_to_hub.__doc__ is not None:
    ImageProcessingMixin.push_to_hub.__doc__ = ImageProcessingMixin.push_to_hub.__doc__.format(
        object="image processor", object_class="AutoImageProcessor", object_files="image processor file"
    )
