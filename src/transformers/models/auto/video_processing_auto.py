# coding=utf-8
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
"""AutoVideoProcessor class."""

import importlib
import json
import os
import warnings
from collections import OrderedDict
from typing import TYPE_CHECKING, Dict, Optional, Tuple, Union

# Build the list of all video processors
from ...configuration_utils import PretrainedConfig
from ...dynamic_module_utils import get_class_from_dynamic_module, resolve_trust_remote_code
from ...utils import (
    CONFIG_NAME,
    VIDEO_PROCESSOR_NAME,
    get_file_from_repo,
    is_torchvision_available,
    is_vision_available,
    logging,
)
from ...video_processing_utils import BaseVideoProcessor
from .auto_factory import _LazyAutoMapping
from .configuration_auto import (
    CONFIG_MAPPING_NAMES,
    AutoConfig,
    model_type_to_module_name,
    replace_list_option_in_docstrings,
)


logger = logging.get_logger(__name__)


if TYPE_CHECKING:
    # This significantly improves completion suggestion performance when
    # the transformers package is used with Microsoft's Pylance language server.
    VIDEO_PROCESSOR_MAPPING_NAMES: OrderedDict[str, Tuple[Optional[str], Optional[str]]] = OrderedDict()
else:
    VIDEO_PROCESSOR_MAPPING_NAMES = OrderedDict(
        [
            ("instructblipvideo", ("InstructBlipVideoVideoProcessor",)),
            ("llava_next_video", ("LlavaNextVideoVideoProcessor",)),
            ("llava_onevision", ("LlavaOnevisionVideoProcessor",)),
            ("qwen2_vl", ("Qwen2VLVideoProcessor",)),
            ("video_llava", ("VideoLlavaVideoProcessor",)),
        ]
    )

for model_type, video_processors in VIDEO_PROCESSOR_MAPPING_NAMES.items():
    slow_video_processor_class, *fast_video_processor_class = video_processors
    if not is_vision_available():
        slow_video_processor_class = None

    # If the fast video processor is not defined, or torchvision is not available, we set it to None
    if not fast_video_processor_class or fast_video_processor_class[0] is None or not is_torchvision_available():
        fast_video_processor_class = None
    else:
        fast_video_processor_class = fast_video_processor_class[0]

    VIDEO_PROCESSOR_MAPPING_NAMES[model_type] = (slow_video_processor_class, fast_video_processor_class)

VIDEO_PROCESSOR_MAPPING = _LazyAutoMapping(CONFIG_MAPPING_NAMES, VIDEO_PROCESSOR_MAPPING_NAMES)


def video_processor_class_from_name(class_name: str):
    for module_name, extractors in VIDEO_PROCESSOR_MAPPING_NAMES.items():
        if class_name in extractors:
            module_name = model_type_to_module_name(module_name)

            module = importlib.import_module(f".{module_name}", "transformers.models")
            try:
                return getattr(module, class_name)
            except AttributeError:
                continue

    for _, extractors in VIDEO_PROCESSOR_MAPPING._extra_content.items():
        for extractor in extractors:
            if getattr(extractor, "__name__", None) == class_name:
                return extractor

    # We did not find the class, but maybe it's because a dep is missing. In that case, the class will be in the main
    # init and we return the proper dummy to get an appropriate error message.
    main_module = importlib.import_module("transformers")
    if hasattr(main_module, class_name):
        return getattr(main_module, class_name)

    return None


def get_video_processor_config(
    pretrained_model_name_or_path: Union[str, os.PathLike],
    cache_dir: Optional[Union[str, os.PathLike]] = None,
    force_download: bool = False,
    resume_download: Optional[bool] = None,
    proxies: Optional[Dict[str, str]] = None,
    token: Optional[Union[bool, str]] = None,
    revision: Optional[str] = None,
    local_files_only: bool = False,
    **kwargs,
):
    """
    Loads the video processor configuration from a pretrained model video processor configuration.

    Args:
        pretrained_model_name_or_path (`str` or `os.PathLike`):
            This can be either:

            - a string, the *model id* of a pretrained model configuration hosted inside a model repo on
              huggingface.co.
            - a path to a *directory* containing a configuration file saved using the
              [`~PreTrainedTokenizer.save_pretrained`] method, e.g., `./my_model_directory/`.

        cache_dir (`str` or `os.PathLike`, *optional*):
            Path to a directory in which a downloaded pretrained model configuration should be cached if the standard
            cache should not be used.
        force_download (`bool`, *optional*, defaults to `False`):
            Whether or not to force to (re-)download the configuration files and override the cached versions if they
            exist.
        resume_download:
            Deprecated and ignored. All downloads are now resumed by default when possible.
            Will be removed in v5 of Transformers.
        proxies (`Dict[str, str]`, *optional*):
            A dictionary of proxy servers to use by protocol or endpoint, e.g., `{'http': 'foo.bar:3128',
            'http://hostname': 'foo.bar:4012'}.` The proxies are used on each request.
        token (`str` or *bool*, *optional*):
            The token to use as HTTP bearer authorization for remote files. If `True`, will use the token generated
            when running `huggingface-cli login` (stored in `~/.huggingface`).
        revision (`str`, *optional*, defaults to `"main"`):
            The specific model version to use. It can be a branch name, a tag name, or a commit id, since we use a
            git-based system for storing models and other artifacts on huggingface.co, so `revision` can be any
            identifier allowed by git.
        local_files_only (`bool`, *optional*, defaults to `False`):
            If `True`, will only try to load the video processor configuration from local files.

    <Tip>

    Passing `token=True` is required when you want to use a private model.

    </Tip>

    Returns:
        `Dict`: The configuration of the video processor.

    Examples:

    ```python
    # Download configuration from huggingface.co and cache.
    video_processor_config = get_video_processor_config("google-bert/bert-base-uncased")
    # This model does not have a video processor config so the result will be an empty dict.
    video_processor_config = get_video_processor_config("FacebookAI/xlm-roberta-base")

    # Save a pretrained video processor locally and you can reload its config
    from transformers import AutoVideoProcessor

    video_processor = AutoVideoProcessor.from_pretrained("llava-hf/llava-onevision-qwen2-0.5b-ov-hf")
    video_processor.save_pretrained("video-processor-test")
    video_processor = get_video_processor_config("video-processor-test")
    ```"""
    use_auth_token = kwargs.pop("use_auth_token", None)
    if use_auth_token is not None:
        warnings.warn(
            "The `use_auth_token` argument is deprecated and will be removed in v5 of Transformers. Please use `token` instead.",
            FutureWarning,
        )
        if token is not None:
            raise ValueError("`token` and `use_auth_token` are both specified. Please set only the argument `token`.")
        token = use_auth_token

    resolved_config_file = get_file_from_repo(
        pretrained_model_name_or_path,
        VIDEO_PROCESSOR_NAME,
        cache_dir=cache_dir,
        force_download=force_download,
        resume_download=resume_download,
        proxies=proxies,
        token=token,
        revision=revision,
        local_files_only=local_files_only,
    )
    if resolved_config_file is None:
        logger.info(
            "Could not locate the video processor configuration file, will try to use the model config instead."
        )
        return {}

    with open(resolved_config_file, encoding="utf-8") as reader:
        return json.load(reader)


def _warning_fast_video_processor_available(fast_class):
    logger.warning(
        f"Fast video processor class {fast_class} is available for this model. "
        "Using slow video processor class. To use the fast video processor class set `use_fast=True`."
    )


class AutoVideoProcessor:
    r"""
    This is a generic video processor class that will be instantiated as one of the video processor classes of the
    library when created with the [`AutoVideoProcessor.from_pretrained`] class method.

    This class cannot be instantiated directly using `__init__()` (throws an error).
    """

    def __init__(self):
        raise EnvironmentError(
            "AutoVideoProcessor is designed to be instantiated "
            "using the `AutoVideoProcessor.from_pretrained(pretrained_model_name_or_path)` method."
        )

    @classmethod
    @replace_list_option_in_docstrings(VIDEO_PROCESSOR_MAPPING_NAMES)
    def from_pretrained(cls, pretrained_model_name_or_path, *inputs, **kwargs):
        r"""
        Instantiate one of the video processor classes of the library from a pretrained model vocabulary.

        The video processor class to instantiate is selected based on the `model_type` property of the config object
        (either passed as an argument or loaded from `pretrained_model_name_or_path` if possible), or when it's
        missing, by falling back to using pattern matching on `pretrained_model_name_or_path`:

        List options

        Params:
            pretrained_model_name_or_path (`str` or `os.PathLike`):
                This can be either:

                - a string, the *model id* of a pretrained video_processor hosted inside a model repo on
                  huggingface.co.
                - a path to a *directory* containing a video processor file saved using the
                  [`~video_processing_utils.BaseVideoProcessor.save_pretrained`] method, e.g.,
                  `./my_model_directory/`.
                - a path or url to a saved video processor JSON *file*, e.g.,
                  `./my_model_directory/preprocessor_config.json`.
            cache_dir (`str` or `os.PathLike`, *optional*):
                Path to a directory in which a downloaded pretrained model video processor should be cached if the
                standard cache should not be used.
            force_download (`bool`, *optional*, defaults to `False`):
                Whether or not to force to (re-)download the video processor files and override the cached versions if
                they exist.
            resume_download:
                Deprecated and ignored. All downloads are now resumed by default when possible.
                Will be removed in v5 of Transformers.
            proxies (`Dict[str, str]`, *optional*):
                A dictionary of proxy servers to use by protocol or endpoint, e.g., `{'http': 'foo.bar:3128',
                'http://hostname': 'foo.bar:4012'}.` The proxies are used on each request.
            token (`str` or *bool*, *optional*):
                The token to use as HTTP bearer authorization for remote files. If `True`, will use the token generated
                when running `huggingface-cli login` (stored in `~/.huggingface`).
            revision (`str`, *optional*, defaults to `"main"`):
                The specific model version to use. It can be a branch name, a tag name, or a commit id, since we use a
                git-based system for storing models and other artifacts on huggingface.co, so `revision` can be any
                identifier allowed by git.
            use_fast (`bool`, *optional*, defaults to `False`):
                Use a fast torchvision-base video processor if it is supported for a given model.
                If a fast tokenizer is not available for a given model, a normal numpy-based video processor
                is returned instead.
            return_unused_kwargs (`bool`, *optional*, defaults to `False`):
                If `False`, then this function returns just the final video processor object. If `True`, then this
                functions returns a `Tuple(video_processor, unused_kwargs)` where *unused_kwargs* is a dictionary
                consisting of the key/value pairs whose keys are not video processor attributes: i.e., the part of
                `kwargs` which has not been used to update `video_processor` and is otherwise ignored.
            trust_remote_code (`bool`, *optional*, defaults to `False`):
                Whether or not to allow for custom models defined on the Hub in their own modeling files. This option
                should only be set to `True` for repositories you trust and in which you have read the code, as it will
                execute code present on the Hub on your local machine.
            kwargs (`Dict[str, Any]`, *optional*):
                The values in kwargs of any keys which are video processor attributes will be used to override the
                loaded values. Behavior concerning key/value pairs whose keys are *not* video processor attributes is
                controlled by the `return_unused_kwargs` keyword parameter.

        <Tip>

        Passing `token=True` is required when you want to use a private model.

        </Tip>

        Examples:

        ```python
        >>> from transformers import AutoVideoProcessor

        >>> # Download video processor from huggingface.co and cache.
        >>> video_processor = AutoVideoProcessor.from_pretrained("llava-hf/llava-onevision-qwen2-0.5b-ov-hf")

        >>> # If video processor files are in a directory (e.g. video processor was saved using *save_pretrained('./test/saved_model/')*)
        >>> # video_processor = AutoVideoProcessor.from_pretrained("./test/saved_model/")
        ```"""
        use_auth_token = kwargs.pop("use_auth_token", None)
        if use_auth_token is not None:
            warnings.warn(
                "The `use_auth_token` argument is deprecated and will be removed in v5 of Transformers. Please use `token` instead.",
                FutureWarning,
            )
            if kwargs.get("token", None) is not None:
                raise ValueError(
                    "`token` and `use_auth_token` are both specified. Please set only the argument `token`."
                )
            kwargs["token"] = use_auth_token

        config = kwargs.pop("config", None)
        use_fast = kwargs.pop("use_fast", None)
        trust_remote_code = kwargs.pop("trust_remote_code", None)
        kwargs["_from_auto"] = True

        config_dict, _ = BaseVideoProcessor.get_video_processor_dict(pretrained_model_name_or_path, **kwargs)
        video_processor_class = config_dict.get("video_processor_type", None)
        video_processor_auto_map = None
        if "AutoVideoProcessor" in config_dict.get("auto_map", {}):
            video_processor_auto_map = config_dict["auto_map"]["AutoVideoProcessor"]

        # If we still don't have the video processor class, check if we're loading from a previous feature extractor config
        # and if so, infer the video processor class from there.
        if video_processor_class is None and video_processor_auto_map is None:
            feature_extractor_class = config_dict.pop("feature_extractor_type", None)
            if feature_extractor_class is not None:
                video_processor_class = feature_extractor_class.replace("FeatureExtractor", "VideoProcessor")
            if "AutoFeatureExtractor" in config_dict.get("auto_map", {}):
                feature_extractor_auto_map = config_dict["auto_map"]["AutoFeatureExtractor"]
                video_processor_auto_map = feature_extractor_auto_map.replace("FeatureExtractor", "VideoProcessor")

        # If we don't find the video processor class in the video processor config, let's try the model config.
        if video_processor_class is None and video_processor_auto_map is None:
            if not isinstance(config, PretrainedConfig):
                config = AutoConfig.from_pretrained(
                    pretrained_model_name_or_path, trust_remote_code=trust_remote_code, **kwargs
                )
            # It could be in `config.video_processor_type``
            video_processor_class = getattr(config, "video_processor_type", None)
            if hasattr(config, "auto_map") and "AutoVideoProcessor" in config.auto_map:
                video_processor_auto_map = config.auto_map["AutoVideoProcessor"]

        if video_processor_class is not None:
            # Update class name to reflect the use_fast option. If class is not found, None is returned.
            if use_fast is not None:
                if use_fast and not video_processor_class.endswith("Fast"):
                    video_processor_class += "Fast"
                elif not use_fast and video_processor_class.endswith("Fast"):
                    video_processor_class = video_processor_class[:-4]
            video_processor_class = video_processor_class_from_name(video_processor_class)

        has_remote_code = video_processor_auto_map is not None
        has_local_code = video_processor_class is not None or type(config) in VIDEO_PROCESSOR_MAPPING
        trust_remote_code = resolve_trust_remote_code(
            trust_remote_code, pretrained_model_name_or_path, has_local_code, has_remote_code
        )

        if video_processor_auto_map is not None and not isinstance(video_processor_auto_map, tuple):
            # In some configs, only the slow video processor class is stored
            video_processor_auto_map = (video_processor_auto_map, None)

        if has_remote_code and trust_remote_code:
            if not use_fast and video_processor_auto_map[1] is not None:
                _warning_fast_video_processor_available(video_processor_auto_map[1])

            if use_fast and video_processor_auto_map[1] is not None:
                class_ref = video_processor_auto_map[1]
            else:
                class_ref = video_processor_auto_map[0]
            video_processor_class = get_class_from_dynamic_module(class_ref, pretrained_model_name_or_path, **kwargs)
            _ = kwargs.pop("code_revision", None)
            if os.path.isdir(pretrained_model_name_or_path):
                video_processor_class.register_for_auto_class()
            return video_processor_class.from_dict(config_dict, **kwargs)
        elif video_processor_class is not None:
            return video_processor_class.from_dict(config_dict, **kwargs)
        # Last try: we use the VIDEO_PROCESSOR_MAPPING.
        elif type(config) in VIDEO_PROCESSOR_MAPPING:
            video_processor_tuple = VIDEO_PROCESSOR_MAPPING[type(config)]

            video_processor_class_py, video_processor_class_fast = video_processor_tuple

            if not use_fast and video_processor_class_fast is not None:
                _warning_fast_video_processor_available(video_processor_class_fast)

            if video_processor_class_fast and (use_fast or video_processor_class_py is None):
                return video_processor_class_fast.from_pretrained(pretrained_model_name_or_path, *inputs, **kwargs)
            else:
                if video_processor_class_py is not None:
                    return video_processor_class_py.from_pretrained(pretrained_model_name_or_path, *inputs, **kwargs)
                else:
                    raise ValueError(
                        "This video processor cannot be instantiated. Please make sure you have `Pillow` installed."
                    )

        raise ValueError(
            f"Unrecognized video processor in {pretrained_model_name_or_path}. Should have a "
            f"`video_processor_type` key in its {VIDEO_PROCESSOR_NAME} of {CONFIG_NAME}, or one of the following "
            f"`model_type` keys in its {CONFIG_NAME}: {', '.join(c for c in VIDEO_PROCESSOR_MAPPING_NAMES.keys())}"
        )

    @staticmethod
    def register(
        config_class,
        slow_video_processor_class=None,
        fast_video_processor_class=None,
        exist_ok=False,
    ):
        """
        Register a new video processor for this class.

        Args:
            config_class ([`PretrainedConfig`]):
                The configuration corresponding to the model to register.
            slow_video_processor_class ([`BaseVideoProcessor`]):
                The video processor to register.
        """
        if slow_video_processor_class is None and fast_video_processor_class is None:
            raise ValueError("You need to specify either slow_video_processor_class or fast_video_processor_class")
        if fast_video_processor_class is not None:
            raise ValueError("We do not support `fast_video_processor_class` yet.")

        # Avoid resetting a set slow/fast video processor if we are passing just the other ones.
        if config_class in VIDEO_PROCESSOR_MAPPING._extra_content:
            existing_slow, existing_fast = VIDEO_PROCESSOR_MAPPING[config_class]
            if slow_video_processor_class is None:
                slow_video_processor_class = existing_slow
            if fast_video_processor_class is None:
                fast_video_processor_class = existing_fast

        VIDEO_PROCESSOR_MAPPING.register(
            config_class, (slow_video_processor_class, fast_video_processor_class), exist_ok=exist_ok
        )
