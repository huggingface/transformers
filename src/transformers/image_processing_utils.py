# coding=utf-8
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

import copy
import json
import os
from typing import Any, Dict, Tuple, Union

import numpy as np

from requests import HTTPError

from .dynamic_module_utils import custom_object_save
from .feature_extraction_utils import BatchFeature as BaseBatchFeature
from .utils import (
    HUGGINGFACE_CO_RESOLVE_ENDPOINT,
    IMAGE_PROCESSOR_NAME,
    EntryNotFoundError,
    PushToHubMixin,
    RepositoryNotFoundError,
    RevisionNotFoundError,
    cached_path,
    copy_func,
    hf_bucket_url,
    is_offline_mode,
    is_remote_url,
    logging,
)


logger = logging.get_logger(__name__)


# TODO: Move BatchFeature to be imported by both feature_extraction_utils and image_processing_utils
# We override the class string here, but logic is the same.
class BatchFeature(BaseBatchFeature):
    r"""
    Holds the output of the image processor specific `__call__` methods.

    This class is derived from a python dictionary and can be used as a dictionary.

    Args:
        data (`dict`):
            Dictionary of lists/arrays/tensors returned by the __call__/pad methods ('pixel_values', 'attention_mask',
            etc.).
        tensor_type (`Union[None, str, TensorType]`, *optional*):
            You can give a tensor_type here to convert the lists of integers in PyTorch/TensorFlow/Numpy Tensors at
            initialization.
    """


# TODO: Abstract out common logic between ImageProcessorMixin and FeatureExtractorMixin
class ImageProcessorMixin(PushToHubMixin):
    """
    Image processor mixin used to provide saving/loading functionality
    """

    _auto_class = None

    def __init__(self, **kwargs):
        """Set elements of `kwargs` as attributes."""
        # Pop "processor_class" as it should be saved as private attribute
        self._processor_class = kwargs.pop("processor_class", None)
        # Additional attributes without default values
        for key, value in kwargs.items():
            try:
                setattr(self, key, value)
            except AttributeError as err:
                logger.error(f"Can't set {key} with value {value} for {self}")
                raise err

    def _set_processor_class(self, processor_class: str):
        """Sets processor class as an attribute."""
        self._processor_class = processor_class

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Union[str, os.PathLike], **kwargs):
        r"""
        Instantiate a type of [`~image_processing_utils.ImageProcessorMixin`] from a image processor, *e.g.* a derived
        class of [`BaseImageProcessor`].

        Args:
            pretrained_model_name_or_path (`str` or `os.PathLike`):
                This can be either:

                - a string, the *model id* of a pretrained image_processor hosted inside a model repo on
                  huggingface.co. Valid model ids can be located at the root-level, like `bert-base-uncased`, or
                  namespaced under a user or organization name, like `dbmdz/bert-base-german-cased`.
                - a path to a *directory* containing a image processor file saved using the
                  [`~image_processing_utils.ImageProcessorMixin.save_pretrained`] method, e.g.,
                  `./my_model_directory/`.
                - a path or url to a saved image processor JSON *file*, e.g.,
                  `./my_model_directory/preprocessor_config.json`.
            cache_dir (`str` or `os.PathLike`, *optional*):
                Path to a directory in which a downloaded pretrained model image processor should be cached if the
                standard cache should not be used.
            force_download (`bool`, *optional*, defaults to `False`):
                Whether or not to force to (re-)download the image processor files and override the cached versions if
                they exist.
            resume_download (`bool`, *optional*, defaults to `False`):
                Whether or not to delete incompletely received file. Attempts to resume the download if such a file
                exists.
            proxies (`Dict[str, str]`, *optional*):
                A dictionary of proxy servers to use by protocol or endpoint, e.g., `{'http': 'foo.bar:3128',
                'http://hostname': 'foo.bar:4012'}.` The proxies are used on each request.
            use_auth_token (`str` or *bool*, *optional*):
                The token to use as HTTP bearer authorization for remote files. If `True`, will use the token generated
                when running `transformers-cli login` (stored in `~/.huggingface`).
            revision (`str`, *optional*, defaults to `"main"`):
                The specific model version to use. It can be a branch name, a tag name, or a commit id, since we use a
                git-based system for storing models and other artifacts on huggingface.co, so `revision` can be any
                identifier allowed by git.
            return_unused_kwargs (`bool`, *optional*, defaults to `False`):
                If `False`, then this function returns just the final image processor object. If `True`, then this
                functions returns a `Tuple(image_processor, unused_kwargs)` where *unused_kwargs* is a dictionary
                consisting of the key/value pairs whose keys are not image processor attributes: i.e., the part of
                `kwargs` which has not been used to update `image_processor` and is otherwise ignored.
            kwargs (`Dict[str, Any]`, *optional*):
                The values in kwargs of any keys which are image processor attributes will be used to override the
                loaded values. Behavior concerning key/value pairs whose keys are *not* image processor attributes is
                controlled by the `return_unused_kwargs` keyword parameter.

        <Tip>

        Passing `use_auth_token=True` is required when you want to use a private model.

        </Tip>

        Returns:
            An image processor of type [`~image_processing_utils.ImageProcessorMixin`].

        Examples: FIXME

        """
        image_processor_dict, kwargs = cls.get_image_processor_dict(pretrained_model_name_or_path, **kwargs)

        return cls.from_dict(image_processor_dict, **kwargs)

    def save_pretrained(self, save_directory: Union[str, os.PathLike], push_to_hub: bool = False, **kwargs):
        """
        Save an image_processor object to the directory `save_directory`, so that it can be re-loaded using the
        [`~image_processing_utils.ImageProcessorMixin.from_pretrained`] instance method.

        Args:
            save_directory (`str` or `os.PathLike`):
                Directory where the image processor JSON file will be saved (will be created if it does not exist).
            push_to_hub (`bool`, *optional*, defaults to `False`):
                Whether or not to push your image processor to the Hugging Face model hub after saving it.

                <Tip warning={true}>

                Using `push_to_hub=True` will synchronize the repository you are pushing to with `save_directory`,
                which requires `save_directory` to be a local clone of the repo you are pushing to if it's an existing
                folder. Pass along `temp_dir=True` to use a temporary directory instead.

                </Tip>

            kwargs:
                Additional key word arguments passed along to the [`~utils.PushToHubMixin.push_to_hub`] method.
        """
        if os.path.isfile(save_directory):
            raise AssertionError(f"Provided path ({save_directory}) should be a directory, not a file")

        if push_to_hub:
            commit_message = kwargs.pop("commit_message", None)
            repo = self._create_or_get_repo(save_directory, **kwargs)

        # If we have a custom config, we copy the file defining it in the folder and set the attributes so it can be
        # loaded from the Hub.
        if self._auto_class is not None:
            custom_object_save(self, save_directory, config=self)

        os.makedirs(save_directory, exist_ok=True)
        # If we save using the predefined names, we can load using `from_pretrained`
        output_image_processor_file = os.path.join(save_directory, IMAGE_PROCESSOR_NAME)

        self.to_json_file(output_image_processor_file)
        logger.info(f"Image processor saved in {output_image_processor_file}")

        if push_to_hub:
            url = self._push_to_hub(repo, commit_message=commit_message)
            logger.info(f"Image processor pushed to the hub in this commit: {url}")

        return [output_image_processor_file]

    @classmethod
    def get_image_processor_dict(
        cls, pretrained_model_name_or_path: Union[str, os.PathLike], **kwargs
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        From a `pretrained_model_name_or_path`, resolve to a dictionary of parameters, to be used for instantiating a
        image processor of type [`~image_processing_utils.ImageProcessorMixin`] using `from_dict`.

        Parameters:
            pretrained_model_name_or_path (`str` or `os.PathLike`):
                The identifier of the pre-trained checkpoint from which we want the dictionary of parameters.

        Returns:
            `Tuple[Dict, Dict]`: The dictionary(ies) that will be used to instantiate the image processor object.
        """
        cache_dir = kwargs.pop("cache_dir", None)
        force_download = kwargs.pop("force_download", False)
        resume_download = kwargs.pop("resume_download", False)
        proxies = kwargs.pop("proxies", None)
        use_auth_token = kwargs.pop("use_auth_token", None)
        local_files_only = kwargs.pop("local_files_only", False)
        revision = kwargs.pop("revision", None)

        from_pipeline = kwargs.pop("_from_pipeline", None)
        from_auto_class = kwargs.pop("_from_auto", False)

        user_agent = {"file_type": "image processor", "from_auto_class": from_auto_class}
        if from_pipeline is not None:
            user_agent["using_pipeline"] = from_pipeline

        if is_offline_mode() and not local_files_only:
            logger.info("Offline mode: forcing local_files_only=True")
            local_files_only = True

        pretrained_model_name_or_path = str(pretrained_model_name_or_path)
        if os.path.isdir(pretrained_model_name_or_path):
            image_processor_file = os.path.join(pretrained_model_name_or_path, IMAGE_PROCESSOR_NAME)
        elif os.path.isfile(pretrained_model_name_or_path) or is_remote_url(pretrained_model_name_or_path):
            image_processor_file = pretrained_model_name_or_path
        else:
            image_processor_file = hf_bucket_url(
                pretrained_model_name_or_path, filename=IMAGE_PROCESSOR_NAME, revision=revision, mirror=None
            )

        try:
            # Load from URL or cache if already cached
            resolved_image_processor_file = cached_path(
                image_processor_file,
                cache_dir=cache_dir,
                force_download=force_download,
                proxies=proxies,
                resume_download=resume_download,
                local_files_only=local_files_only,
                use_auth_token=use_auth_token,
                user_agent=user_agent,
            )

        except RepositoryNotFoundError:
            raise EnvironmentError(
                f"{pretrained_model_name_or_path} is not a local folder and is not a valid model identifier listed on "
                "'https://huggingface.co/models'\nIf this is a private repository, make sure to pass a token having "
                "permission to this repo with `use_auth_token` or log in with `huggingface-cli login` and pass "
                "`use_auth_token=True`."
            )
        except RevisionNotFoundError:
            raise EnvironmentError(
                f"{revision} is not a valid git identifier (branch name, tag name or commit id) that exists for this "
                f"model name. Check the model page at 'https://huggingface.co/{pretrained_model_name_or_path}' for "
                "available revisions."
            )
        except EntryNotFoundError:
            raise EnvironmentError(
                f"{pretrained_model_name_or_path} does not appear to have a file named {IMAGE_PROCESSOR_NAME}."
            )
        except HTTPError as err:
            raise EnvironmentError(
                f"There was a specific connection error when trying to load {pretrained_model_name_or_path}:\n{err}"
            )
        except ValueError:
            raise EnvironmentError(
                f"We couldn't connect to '{HUGGINGFACE_CO_RESOLVE_ENDPOINT}' to load this model, couldn't find it in"
                f" the cached files and it looks like {pretrained_model_name_or_path} is not the path to a directory"
                f" containing a {IMAGE_PROCESSOR_NAME} file.\nCheckout your internet connection or see how to run"
                " the library in offline mode at"
                " 'https://huggingface.co/docs/transformers/installation#offline-mode'."
            )
        except EnvironmentError:
            raise EnvironmentError(
                f"Can't load image processor for '{pretrained_model_name_or_path}'. If you were trying to load it "
                "from 'https://huggingface.co/models', make sure you don't have a local directory with the same name. "
                f"Otherwise, make sure '{pretrained_model_name_or_path}' is the correct path to a directory "
                f"containing a {IMAGE_PROCESSOR_NAME} file"
            )

        try:
            # Load image_processor dict
            with open(resolved_image_processor_file, "r", encoding="utf-8") as reader:
                text = reader.read()
            image_processor_dict = json.loads(text)

        except json.JSONDecodeError:
            raise EnvironmentError(
                f"It looks like the config file at '{resolved_image_processor_file}' is not a valid JSON file."
            )

        if resolved_image_processor_file == image_processor_file:
            logger.info(f"loading image processor configuration file {image_processor_file}")
        else:
            logger.info(
                f"loading image processor configuration file {image_processor_file} from cache at"
                f" {resolved_image_processor_file}"
            )

        return image_processor_dict, kwargs

    @classmethod
    def from_dict(cls, image_processor_dict: Dict[str, Any], **kwargs):
        """
        Instantiates a type of [`~image_processing_utils.ImageProcessorMixin`] from a Python dictionary of parameters.

        Args:
            image_processor_dict (`Dict[str, Any]`):
                Dictionary that will be used to instantiate the image processor object. Such a dictionary can be
                retrieved from a pretrained checkpoint by leveraging the
                [`~image_processing_utils.ImageProcessorMixin.to_dict`] method.
            kwargs (`Dict[str, Any]`):
                Additional parameters from which to initialize the image processor object.

        Returns:
            [`~image_processing_utils.ImageProcessorMixin`]: The image processor object instantiated from those
            parameters.
        """
        return_unused_kwargs = kwargs.pop("return_unused_kwargs", False)

        image_processor = cls(**image_processor_dict)

        # Update image_processor with kwargs if needed
        to_remove = []
        for key, value in kwargs.items():
            if hasattr(image_processor, key):
                setattr(image_processor, key, value)
                to_remove.append(key)
        for key in to_remove:
            kwargs.pop(key, None)

        logger.info(f"image processor {image_processor}")
        if return_unused_kwargs:
            return image_processor, kwargs
        else:
            return image_processor

    def to_dict(self) -> Dict[str, Any]:
        """
        Serializes this instance to a Python dictionary.

        Returns:
            `Dict[str, Any]`: Dictionary of all the attributes that make up this image processor instance.
        """
        output = copy.deepcopy(self.__dict__)
        output["image_processor_type"] = self.__class__.__name__

        return output

    @classmethod
    def from_json_file(cls, json_file: Union[str, os.PathLike]):
        """
        Instantiates an image processor of type [`~image_processing_utils.ImageProcessorMixin`] from the path to a JSON
        file of parameters.

        Args:
            json_file (`str` or `os.PathLike`):
                Path to the JSON file containing the parameters.

        Returns:
            A image processor of type [`~image_processing_utils.ImageProcessorMixin`]: The image_processor object
            instantiated from that JSON file.
        """
        with open(json_file, "r", encoding="utf-8") as reader:
            text = reader.read()
        image_processor_dict = json.loads(text)
        return cls(**image_processor_dict)

    def to_json_string(self) -> str:
        """
        Serializes this instance to a JSON string.

        Returns:
            `str`: String containing all the attributes that make up this image_processor instance in JSON format.
        """
        dictionary = self.to_dict()

        for key, value in dictionary.items():
            if isinstance(value, np.ndarray):
                dictionary[key] = value.tolist()

        # make sure private name "_processor_class" is correctly
        # saved as "processor_class"
        _processor_class = dictionary.pop("_processor_class", None)
        if _processor_class is not None:
            dictionary["processor_class"] = _processor_class

        return json.dumps(dictionary, indent=2, sort_keys=True) + "\n"

    def to_json_file(self, json_file_path: Union[str, os.PathLike]):
        """
        Save this instance to a JSON file.

        Args:
            json_file_path (`str` or `os.PathLike`):
                Path to the JSON file in which this image_processor instance's parameters will be saved.
        """
        with open(json_file_path, "w", encoding="utf-8") as writer:
            writer.write(self.to_json_string())

    def __repr__(self):
        return f"{self.__class__.__name__} {self.to_json_string()}"

    @classmethod
    def register_for_auto_class(cls, auto_class="AutoImageProcessor"):
        """
        Register this class with a given auto class. This should only be used for custom image processors as the ones
        in the library are already mapped with `AutoImageProcessor`.

        <Tip warning={true}>

        This API is experimental and may have some slight breaking changes in the next releases.

        </Tip>

        Args:
            auto_class (`str` or `type`, *optional*, defaults to `"AutoFeatureExtractor"`):
                The auto class to register this new image processor with.
        """
        if not isinstance(auto_class, str):
            auto_class = auto_class.__name__

        import transformers.models.auto as auto_module

        if not hasattr(auto_module, auto_class):
            raise ValueError(f"{auto_class} is not a valid auto class.")

        cls._auto_class = auto_class


ImageProcessorMixin.push_to_hub = copy_func(ImageProcessorMixin.push_to_hub)
ImageProcessorMixin.push_to_hub.__doc__ = ImageProcessorMixin.push_to_hub.__doc__.format(
    object="image processor", object_class="AutoImageProcessor", object_files="image processor file"
)


class BaseImageProcessor(ImageProcessorMixin):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __call__(self, images, **kwargs) -> BatchFeature:
        return self.preprocess(images, **kwargs)

    def preprocess(self, images, **kwargs) -> BatchFeature:
        raise NotImplementedError("Each image processor must implement its own preprocess method")
