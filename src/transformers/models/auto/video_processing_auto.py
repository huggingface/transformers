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
import os
from collections import OrderedDict
from typing import TYPE_CHECKING

# Build the list of all video processors
from ...configuration_utils import PreTrainedConfig
from ...dynamic_module_utils import get_class_from_dynamic_module, resolve_trust_remote_code
from ...utils import (
    CONFIG_NAME,
    IMAGE_PROCESSOR_NAME,
    PROCESSOR_NAME,
    VIDEO_PROCESSOR_NAME,
    cached_file,
    is_torchvision_available,
    logging,
    safe_load_json_file,
)
from ...utils.import_utils import requires
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
    VIDEO_PROCESSOR_MAPPING_NAMES: OrderedDict[str, tuple[str | None, str | None]] = OrderedDict()
else:
    VIDEO_PROCESSOR_MAPPING_NAMES = OrderedDict(
        [
            ("ernie4_5_vl_moe", "Ernie4_5_VL_MoeVideoProcessor"),
            ("glm46v", "Glm46VVideoProcessor"),
            ("glm4v", "Glm4vVideoProcessor"),
            ("instructblip", "InstructBlipVideoVideoProcessor"),
            ("instructblipvideo", "InstructBlipVideoVideoProcessor"),
            ("internvl", "InternVLVideoProcessor"),
            ("llava_next_video", "LlavaNextVideoVideoProcessor"),
            ("llava_onevision", "LlavaOnevisionVideoProcessor"),
            ("pe_audio_video", "PeVideoVideoProcessor"),
            ("pe_video", "PeVideoVideoProcessor"),
            ("perception_lm", "PerceptionLMVideoProcessor"),
            ("qwen2_5_omni", "Qwen2VLVideoProcessor"),
            ("qwen2_5_vl", "Qwen2VLVideoProcessor"),
            ("qwen2_vl", "Qwen2VLVideoProcessor"),
            ("qwen3_5", "Qwen3VLVideoProcessor"),
            ("qwen3_5_moe", "Qwen3VLVideoProcessor"),
            ("qwen3_omni_moe", "Qwen2VLVideoProcessor"),
            ("qwen3_vl", "Qwen3VLVideoProcessor"),
            ("qwen3_vl_moe", "Qwen3VLVideoProcessor"),
            ("sam2_video", "Sam2VideoVideoProcessor"),
            ("sam3_video", "Sam3VideoVideoProcessor"),
            ("smolvlm", "SmolVLMVideoProcessor"),
            ("video_llama_3", "VideoLlama3VideoProcessor"),
            ("video_llava", "VideoLlavaVideoProcessor"),
            ("videomae", "VideoMAEVideoProcessor"),
            ("vjepa2", "VJEPA2VideoProcessor"),
        ]
    )

for model_type, video_processors in VIDEO_PROCESSOR_MAPPING_NAMES.items():
    fast_video_processor_class = video_processors

    # If the torchvision is not available, we set it to None
    if not is_torchvision_available():
        fast_video_processor_class = None

    VIDEO_PROCESSOR_MAPPING_NAMES[model_type] = fast_video_processor_class

VIDEO_PROCESSOR_MAPPING = _LazyAutoMapping(CONFIG_MAPPING_NAMES, VIDEO_PROCESSOR_MAPPING_NAMES)


def video_processor_class_from_name(class_name: str):
    for module_name, extractor in VIDEO_PROCESSOR_MAPPING_NAMES.items():
        if class_name == extractor:
            module_name = model_type_to_module_name(module_name)

            module = importlib.import_module(f".{module_name}", "transformers.models")
            try:
                return getattr(module, class_name)
            except AttributeError:
                continue

    for extractor in VIDEO_PROCESSOR_MAPPING._extra_content.values():
        if getattr(extractor, "__name__", None) == class_name:
            return extractor

    # We did not find the class, but maybe it's because a dep is missing. In that case, the class will be in the main
    # init and we return the proper dummy to get an appropriate error message.
    main_module = importlib.import_module("transformers")
    if hasattr(main_module, class_name):
        return getattr(main_module, class_name)

    return None


def get_video_processor_config(
    pretrained_model_name_or_path: str | os.PathLike,
    cache_dir: str | os.PathLike | None = None,
    force_download: bool = False,
    proxies: dict[str, str] | None = None,
    token: bool | str | None = None,
    revision: str | None = None,
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
              [`~BaseVideoProcessor.save_pretrained`] method, e.g., `./my_model_directory/`.

        cache_dir (`str` or `os.PathLike`, *optional*):
            Path to a directory in which a downloaded pretrained model configuration should be cached if the standard
            cache should not be used.
        force_download (`bool`, *optional*, defaults to `False`):
            Whether or not to force to (re-)download the configuration files and override the cached versions if they
            exist.
        proxies (`dict[str, str]`, *optional*):
            A dictionary of proxy servers to use by protocol or endpoint, e.g., `{'http': 'foo.bar:3128',
            'http://hostname': 'foo.bar:4012'}.` The proxies are used on each request.
        token (`str` or *bool*, *optional*):
            The token to use as HTTP bearer authorization for remote files. If `True`, will use the token generated
            when running `hf auth login` (stored in `~/.huggingface`).
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
    video_processor_config = get_video_processor_config("llava-hf/llava-onevision-qwen2-0.5b-ov-hf")
    # This model does not have a video processor config so the result will be an empty dict.
    video_processor_config = get_video_processor_config("FacebookAI/xlm-roberta-base")

    # Save a pretrained video processor locally and you can reload its config
    from transformers import AutoVideoProcessor

    video_processor = AutoVideoProcessor.from_pretrained("llava-hf/llava-onevision-qwen2-0.5b-ov-hf")
    video_processor.save_pretrained("video-processor-test")
    video_processor = get_video_processor_config("video-processor-test")
    ```"""
    # Load with a priority given to the nested processor config, if available in repo
    resolved_processor_file = cached_file(
        pretrained_model_name_or_path,
        filename=PROCESSOR_NAME,
        cache_dir=cache_dir,
        force_download=force_download,
        proxies=proxies,
        token=token,
        revision=revision,
        local_files_only=local_files_only,
        _raise_exceptions_for_gated_repo=False,
        _raise_exceptions_for_missing_entries=False,
    )
    resolved_video_processor_files = [
        resolved_file
        for filename in [VIDEO_PROCESSOR_NAME, IMAGE_PROCESSOR_NAME]
        if (
            resolved_file := cached_file(
                pretrained_model_name_or_path,
                filename=filename,
                cache_dir=cache_dir,
                force_download=force_download,
                proxies=proxies,
                token=token,
                revision=revision,
                local_files_only=local_files_only,
                _raise_exceptions_for_gated_repo=False,
                _raise_exceptions_for_missing_entries=False,
                _raise_exceptions_for_connection_errors=False,
            )
        )
        is not None
    ]
    resolved_video_processor_file = resolved_video_processor_files[0] if resolved_video_processor_files else None

    # An empty list if none of the possible files is found in the repo
    if not resolved_video_processor_file and not resolved_processor_file:
        logger.info("Could not locate the video processor configuration file.")
        return {}

    # Load video_processor dict. Priority goes as (nested config if found -> video processor config -> image processor config)
    # We are downloading both configs because almost all models have a `processor_config.json` but
    # not all of these are nested. We need to check if it was saved recebtly as nested or if it is legacy style
    video_processor_dict = {}
    if resolved_processor_file is not None:
        processor_dict = safe_load_json_file(resolved_processor_file)
        if "video_processor" in processor_dict:
            video_processor_dict = processor_dict["video_processor"]

    if resolved_video_processor_file is not None and video_processor_dict is None:
        video_processor_dict = safe_load_json_file(resolved_video_processor_file)

    return video_processor_dict


@requires(backends=("vision", "torchvision"))
class AutoVideoProcessor:
    r"""
    This is a generic video processor class that will be instantiated as one of the video processor classes of the
    library when created with the [`AutoVideoProcessor.from_pretrained`] class method.

    This class cannot be instantiated directly using `__init__()` (throws an error).
    """

    def __init__(self):
        raise OSError(
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
            proxies (`dict[str, str]`, *optional*):
                A dictionary of proxy servers to use by protocol or endpoint, e.g., `{'http': 'foo.bar:3128',
                'http://hostname': 'foo.bar:4012'}.` The proxies are used on each request.
            token (`str` or *bool*, *optional*):
                The token to use as HTTP bearer authorization for remote files. If `True`, will use the token generated
                when running `hf auth login` (stored in `~/.huggingface`).
            revision (`str`, *optional*, defaults to `"main"`):
                The specific model version to use. It can be a branch name, a tag name, or a commit id, since we use a
                git-based system for storing models and other artifacts on huggingface.co, so `revision` can be any
                identifier allowed by git.
            return_unused_kwargs (`bool`, *optional*, defaults to `False`):
                If `False`, then this function returns just the final video processor object. If `True`, then this
                functions returns a `Tuple(video_processor, unused_kwargs)` where *unused_kwargs* is a dictionary
                consisting of the key/value pairs whose keys are not video processor attributes: i.e., the part of
                `kwargs` which has not been used to update `video_processor` and is otherwise ignored.
            trust_remote_code (`bool`, *optional*, defaults to `False`):
                Whether or not to allow for custom models defined on the Hub in their own modeling files. This option
                should only be set to `True` for repositories you trust and in which you have read the code, as it will
                execute code present on the Hub on your local machine.
            kwargs (`dict[str, Any]`, *optional*):
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
        config = kwargs.pop("config", None)
        trust_remote_code = kwargs.pop("trust_remote_code", None)
        kwargs["_from_auto"] = True

        config_dict, _ = BaseVideoProcessor.get_video_processor_dict(pretrained_model_name_or_path, **kwargs)
        video_processor_class = config_dict.get("video_processor_type", None)
        video_processor_auto_map = None
        if "AutoVideoProcessor" in config_dict.get("auto_map", {}):
            video_processor_auto_map = config_dict["auto_map"]["AutoVideoProcessor"]

        # If we still don't have the video processor class, check if we're loading from a previous image processor config
        # and if so, infer the video processor class from there.
        if video_processor_class is None and video_processor_auto_map is None:
            image_processor_class = config_dict.pop("image_processor_type", None)
            if image_processor_class is not None:
                video_processor_class_inferred = image_processor_class.replace("ImageProcessor", "VideoProcessor")

                # Some models have different image processors, e.g. InternVL uses GotOCRImageProcessor
                # We cannot use GotOCRVideoProcessor when falling back for BC and should try to infer from config later on
                if video_processor_class_from_name(video_processor_class_inferred) is not None:
                    video_processor_class = video_processor_class_inferred
            if "AutoImageProcessor" in config_dict.get("auto_map", {}):
                image_processor_auto_map = config_dict["auto_map"]["AutoImageProcessor"]
                video_processor_auto_map = image_processor_auto_map.replace("ImageProcessor", "VideoProcessor")

        # If we don't find the video processor class in the video processor config, let's try the model config.
        if video_processor_class is None and video_processor_auto_map is None:
            if not isinstance(config, PreTrainedConfig):
                config = AutoConfig.from_pretrained(
                    pretrained_model_name_or_path, trust_remote_code=trust_remote_code, **kwargs
                )
            # It could be in `config.video_processor_type``
            video_processor_class = getattr(config, "video_processor_type", None)
            if hasattr(config, "auto_map") and "AutoVideoProcessor" in config.auto_map:
                video_processor_auto_map = config.auto_map["AutoVideoProcessor"]

        if video_processor_class is not None:
            video_processor_class = video_processor_class_from_name(video_processor_class)

        has_remote_code = video_processor_auto_map is not None
        has_local_code = video_processor_class is not None or type(config) in VIDEO_PROCESSOR_MAPPING
        if has_remote_code:
            if "--" in video_processor_auto_map:
                upstream_repo = video_processor_auto_map.split("--")[0]
            else:
                upstream_repo = None
            trust_remote_code = resolve_trust_remote_code(
                trust_remote_code, pretrained_model_name_or_path, has_local_code, has_remote_code, upstream_repo
            )

        if has_remote_code and trust_remote_code:
            class_ref = video_processor_auto_map
            video_processor_class = get_class_from_dynamic_module(class_ref, pretrained_model_name_or_path, **kwargs)
            _ = kwargs.pop("code_revision", None)
            video_processor_class.register_for_auto_class()
            return video_processor_class.from_pretrained(pretrained_model_name_or_path, *inputs, **kwargs)
        elif video_processor_class is not None:
            return video_processor_class.from_pretrained(pretrained_model_name_or_path, *inputs, **kwargs)
        # Last try: we use the VIDEO_PROCESSOR_MAPPING.
        elif type(config) in VIDEO_PROCESSOR_MAPPING:
            video_processor_class = VIDEO_PROCESSOR_MAPPING[type(config)]
            if video_processor_class is not None:
                return video_processor_class.from_pretrained(pretrained_model_name_or_path, *inputs, **kwargs)

        # Raise a more informative error message if torchvision isn't found, otherwise just fallback to default
        if not is_torchvision_available():
            raise ValueError(
                f"{pretrained_model_name_or_path} requires `torchvision` to be installed. Please install `torchvision` and try again."
            )

        raise ValueError(
            f"Unrecognized video processor in {pretrained_model_name_or_path}. Should have a "
            f"`video_processor_type` key in its {VIDEO_PROCESSOR_NAME} of {CONFIG_NAME}, or one of the following "
            f"`model_type` keys in its {CONFIG_NAME}: {', '.join(c for c in VIDEO_PROCESSOR_MAPPING_NAMES)}"
        )

    @staticmethod
    def register(
        config_class,
        video_processor_class,
        exist_ok=False,
    ):
        """
        Register a new video processor for this class.

        Args:
            config_class ([`PreTrainedConfig`]):
                The configuration corresponding to the model to register.
            video_processor_class ([`BaseVideoProcessor`]):
                The video processor to register.
        """
        VIDEO_PROCESSOR_MAPPING.register(config_class, video_processor_class, exist_ok=exist_ok)


__all__ = ["VIDEO_PROCESSOR_MAPPING", "AutoVideoProcessor"]
