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
from .processing_base import ProcessingMixin
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
class ImageProcessingMixin(ProcessingMixin):
    """
    This is an image processor mixin used to provide saving/loading functionality for sequential and image feature
    extractors.
    """

<<<<<<< HEAD
    _auto_class = None

    def __init__(self, **kwargs):
        """Set elements of `kwargs` as attributes."""
        # This key was saved while we still used `XXXFeatureExtractor` for image processing. Now we use
        # `XXXImageProcessor`, this attribute and its value are misleading.
        kwargs.pop("feature_extractor_type", None)
        # Pop "processor_class", should not be saved with image processing config anymore
        kwargs.pop("processor_class", None)
        # Additional attributes without default values
        for key, value in kwargs.items():
            try:
                setattr(self, key, value)
            except AttributeError as err:
                logger.error(f"Can't set {key} with value {value} for {self}")
                raise err

    @classmethod
    def from_pretrained(
        cls: type[ImageProcessorType],
        pretrained_model_name_or_path: str | os.PathLike,
        cache_dir: str | os.PathLike | None = None,
        force_download: bool = False,
        local_files_only: bool = False,
        token: str | bool | None = None,
        revision: str = "main",
        **kwargs,
    ) -> ImageProcessorType:
        r"""
        Instantiate a type of [`~image_processing_utils.ImageProcessingMixin`] from an image processor.

        Args:
            pretrained_model_name_or_path (`str` or `os.PathLike`):
                This can be either:

                - a string, the *model id* of a pretrained image_processor hosted inside a model repo on
                  huggingface.co.
                - a path to a *directory* containing a image processor file saved using the
                  [`~image_processing_utils.ImageProcessingMixin.save_pretrained`] method, e.g.,
                  `./my_model_directory/`.
                - a path to a saved image processor JSON *file*, e.g.,
                  `./my_model_directory/preprocessor_config.json`.
            cache_dir (`str` or `os.PathLike`, *optional*):
                Path to a directory in which a downloaded pretrained model image processor should be cached if the
                standard cache should not be used.
            force_download (`bool`, *optional*, defaults to `False`):
                Whether or not to force to (re-)download the image processor files and override the cached versions if
                they exist.
            proxies (`dict[str, str]`, *optional*):
                A dictionary of proxy servers to use by protocol or endpoint, e.g., `{'http': 'foo.bar:3128',
                'http://hostname': 'foo.bar:4012'}.` The proxies are used on each request.
            token (`str` or `bool`, *optional*):
                The token to use as HTTP bearer authorization for remote files. If `True`, or not specified, will use
                the token generated when running `hf auth login` (stored in `~/.huggingface`).
            revision (`str`, *optional*, defaults to `"main"`):
                The specific model version to use. It can be a branch name, a tag name, or a commit id, since we use a
                git-based system for storing models and other artifacts on huggingface.co, so `revision` can be any
                identifier allowed by git.


                <Tip>

                To test a pull request you made on the Hub, you can pass `revision="refs/pr/<pr_number>"`.

                </Tip>

            return_unused_kwargs (`bool`, *optional*, defaults to `False`):
                If `False`, then this function returns just the final image processor object. If `True`, then this
                functions returns a `Tuple(image_processor, unused_kwargs)` where *unused_kwargs* is a dictionary
                consisting of the key/value pairs whose keys are not image processor attributes: i.e., the part of
                `kwargs` which has not been used to update `image_processor` and is otherwise ignored.
            subfolder (`str`, *optional*, defaults to `""`):
                In case the relevant files are located inside a subfolder of the model repo on huggingface.co, you can
                specify the folder name here.
            kwargs (`dict[str, Any]`, *optional*):
                The values in kwargs of any keys which are image processor attributes will be used to override the
                loaded values. Behavior concerning key/value pairs whose keys are *not* image processor attributes is
                controlled by the `return_unused_kwargs` keyword parameter.

        Returns:
            A image processor of type [`~image_processing_utils.ImageProcessingMixin`].

        Examples:

        ```python
        # We can't instantiate directly the base class *ImageProcessingMixin* so let's show the examples on a
        # derived class: *CLIPImageProcessor*
        image_processor = CLIPImageProcessor.from_pretrained(
            "openai/clip-vit-base-patch32"
        )  # Download image_processing_config from huggingface.co and cache.
        image_processor = CLIPImageProcessor.from_pretrained(
            "./test/saved_model/"
        )  # E.g. image processor (or model) was saved using *save_pretrained('./test/saved_model/')*
        image_processor = CLIPImageProcessor.from_pretrained("./test/saved_model/preprocessor_config.json")
        image_processor = CLIPImageProcessor.from_pretrained(
            "openai/clip-vit-base-patch32", do_normalize=False, foo=False
        )
        assert image_processor.do_normalize is False
        image_processor, unused_kwargs = CLIPImageProcessor.from_pretrained(
            "openai/clip-vit-base-patch32", do_normalize=False, foo=False, return_unused_kwargs=True
        )
        assert image_processor.do_normalize is False
        assert unused_kwargs == {"foo": False}
        ```"""
        kwargs["cache_dir"] = cache_dir
        kwargs["force_download"] = force_download
        kwargs["local_files_only"] = local_files_only
        kwargs["revision"] = revision

        if token is not None:
            kwargs["token"] = token

        image_processor_dict, kwargs = cls.get_image_processor_dict(pretrained_model_name_or_path, **kwargs)

        return cls.from_dict(image_processor_dict, **kwargs)

    def save_pretrained(self, save_directory: str | os.PathLike, push_to_hub: bool = False, **kwargs):
        """
        Save an image processor object to the directory `save_directory`, so that it can be re-loaded using the
        [`~image_processing_utils.ImageProcessingMixin.from_pretrained`] class method.

        Args:
            save_directory (`str` or `os.PathLike`):
                Directory where the image processor JSON file will be saved (will be created if it does not exist).
            push_to_hub (`bool`, *optional*, defaults to `False`):
                Whether or not to push your model to the Hugging Face model hub after saving it. You can specify the
                repository you want to push to with `repo_id` (will default to the name of `save_directory` in your
                namespace).
            kwargs (`dict[str, Any]`, *optional*):
                Additional key word arguments passed along to the [`~utils.PushToHubMixin.push_to_hub`] method.
        """
        if os.path.isfile(save_directory):
            raise AssertionError(f"Provided path ({save_directory}) should be a directory, not a file")

        os.makedirs(save_directory, exist_ok=True)

        if push_to_hub:
            commit_message = kwargs.pop("commit_message", None)
            repo_id = kwargs.pop("repo_id", save_directory.split(os.path.sep)[-1])
            repo_id = create_repo(repo_id, exist_ok=True, **kwargs).repo_id
            files_timestamps = self._get_files_timestamps(save_directory)

        # If we have a custom config, we copy the file defining it in the folder and set the attributes so it can be
        # loaded from the Hub.
        if self._auto_class is not None:
            custom_object_save(self, save_directory, config=self)

        # If we save using the predefined names, we can load using `from_pretrained`
        output_image_processor_file = os.path.join(save_directory, IMAGE_PROCESSOR_NAME)

        self.to_json_file(output_image_processor_file)
        logger.info(f"Image processor saved in {output_image_processor_file}")

        if push_to_hub:
            self._upload_modified_files(
                save_directory,
                repo_id,
                files_timestamps,
                commit_message=commit_message,
                token=kwargs.get("token"),
            )

        return [output_image_processor_file]
=======
    _config_name = IMAGE_PROCESSOR_NAME
    _type_key = "image_processor_type"
    _nested_config_keys = ["image_processor"]
    _auto_class_default = "AutoImageProcessor"
    _file_type_label = "image processor"
    _excluded_dict_keys = set()
    _extra_init_pops = ["feature_extractor_type"]
    _config_filename_kwarg = "image_processor_filename"
    _subfolder_default = ""
>>>>>>> refacto to introduce ProcessingMixin

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

    @classmethod
    def from_dict(cls, image_processor_dict: dict[str, Any], **kwargs):
        """
        Instantiates a type of [`~image_processing_utils.ImageProcessingMixin`] from a Python dictionary of parameters.

        Args:
            image_processor_dict (`dict[str, Any]`):
                Dictionary that will be used to instantiate the image processor object. Such a dictionary can be
                retrieved from a pretrained checkpoint by leveraging the
                [`~image_processing_utils.ImageProcessingMixin.to_dict`] method.
            kwargs (`dict[str, Any]`):
                Additional parameters from which to initialize the image processor object.

        Returns:
            [`~image_processing_utils.ImageProcessingMixin`]: The image processor object instantiated from those
            parameters.
        """
        image_processor_dict = image_processor_dict.copy()
        return_unused_kwargs = kwargs.pop("return_unused_kwargs", False)
        image_processor_dict.update({k: v for k, v in kwargs.items() if k in cls.valid_kwargs.__annotations__})
        image_processor = cls(**image_processor_dict)

        # Apply extra kwargs to instance (BC for remote code, e.g. phi4_multimodal)
        extra_keys = []
        for key in reversed(list(kwargs.keys())):
            if hasattr(image_processor, key) and key not in cls.valid_kwargs.__annotations__:
                setattr(image_processor, key, kwargs.pop(key, None))
                extra_keys.append(key)
        if extra_keys:
            logger.warning_once(
                f"Image processor {cls.__name__}: kwargs {extra_keys} were applied for backward compatibility. "
                f"To avoid this warning, add them to valid_kwargs: create a custom TypedDict extending "
                f"ImagesKwargs with these keys and set it as the `valid_kwargs` class attribute."
            )

        logger.info(f"Image processor {image_processor}")
        if return_unused_kwargs:
            return image_processor, kwargs
        else:
            return image_processor

    def fetch_images(self, image_url_or_urls: str | list[str] | list[list[str]]):
        """
        Convert a single or a list of urls into the corresponding `PIL.Image` objects.

        If a single url is passed, the return value will be a single object. If a list is passed a list of objects is
        returned.
        """
        if isinstance(image_url_or_urls, (list, tuple)):
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
