# coding=utf-8
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
""" AutoProcessor class."""
import importlib
import inspect
import json
from collections import OrderedDict

# Build the list of all feature extractors
from ...configuration_utils import PretrainedConfig
from ...feature_extraction_utils import FeatureExtractionMixin
from ...file_utils import CONFIG_NAME, FEATURE_EXTRACTOR_NAME, get_file_from_repo
from ...tokenization_utils import TOKENIZER_CONFIG_FILE
from .auto_factory import _LazyAutoMapping
from .configuration_auto import (
    CONFIG_MAPPING_NAMES,
    AutoConfig,
    config_class_to_model_type,
    model_type_to_module_name,
    replace_list_option_in_docstrings,
)


PROCESSOR_MAPPING_NAMES = OrderedDict(
    [
        ("clip", "CLIPProcessor"),
        ("layoutlmv2", "LayoutLMv2Processor"),
        ("layoutxlm", "LayoutXLMProcessor"),
        ("speech_to_text", "Speech2TextProcessor"),
        ("speech_to_text_2", "Speech2Text2Processor"),
        ("trocr", "TrOCRProcessor"),
        ("wav2vec2", "Wav2Vec2Processor"),
        ("wav2vec2_with_lm", "Wav2Vec2ProcessorWithLM"),
        ("vision-text-dual-encoder", "VisionTextDualEncoderProcessor"),
    ]
)

PROCESSOR_MAPPING = _LazyAutoMapping(CONFIG_MAPPING_NAMES, PROCESSOR_MAPPING_NAMES)


def processor_class_from_name(class_name: str):
    for module_name, processors in PROCESSOR_MAPPING_NAMES.items():
        if class_name in processors:
            module_name = model_type_to_module_name(module_name)

            module = importlib.import_module(f".{module_name}", "transformers.models")
            return getattr(module, class_name)
            break

    return None


class AutoProcessor:
    r"""
    This is a generic processor class that will be instantiated as one of the processor classes of the library when
    created with the [`AutoProcessor.from_pretrained`] class method.

    This class cannot be instantiated directly using `__init__()` (throws an error).
    """

    def __init__(self):
        raise EnvironmentError(
            "AutoProcessor is designed to be instantiated "
            "using the `AutoProcessor.from_pretrained(pretrained_model_name_or_path)` method."
        )

    @classmethod
    @replace_list_option_in_docstrings(PROCESSOR_MAPPING_NAMES)
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        r"""
        Instantiate one of the processor classes of the library from a pretrained model vocabulary.

        The processor class to instantiate is selected based on the `model_type` property of the config object (either
        passed as an argument or loaded from `pretrained_model_name_or_path` if possible):

        List options

        Params:
            pretrained_model_name_or_path (`str` or `os.PathLike`):
                This can be either:

                - a string, the *model id* of a pretrained feature_extractor hosted inside a model repo on
                  huggingface.co. Valid model ids can be located at the root-level, like `bert-base-uncased`, or
                  namespaced under a user or organization name, like `dbmdz/bert-base-german-cased`.
                - a path to a *directory* containing a processor files saved using the `save_pretrained()` method,
                  e.g., `./my_model_directory/`.
            cache_dir (`str` or `os.PathLike`, *optional*):
                Path to a directory in which a downloaded pretrained model feature extractor should be cached if the
                standard cache should not be used.
            force_download (`bool`, *optional*, defaults to `False`):
                Whether or not to force to (re-)download the feature extractor files and override the cached versions
                if they exist.
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
                If `False`, then this function returns just the final feature extractor object. If `True`, then this
                functions returns a `Tuple(feature_extractor, unused_kwargs)` where *unused_kwargs* is a dictionary
                consisting of the key/value pairs whose keys are not feature extractor attributes: i.e., the part of
                `kwargs` which has not been used to update `feature_extractor` and is otherwise ignored.
            kwargs (`Dict[str, Any]`, *optional*):
                The values in kwargs of any keys which are feature extractor attributes will be used to override the
                loaded values. Behavior concerning key/value pairs whose keys are *not* feature extractor attributes is
                controlled by the `return_unused_kwargs` keyword parameter.

        <Tip>

        Passing `use_auth_token=True` is required when you want to use a private model.

        </Tip>

        Examples:

        ```python
        >>> from transformers import AutoProcessor

        >>> # Download processor from huggingface.co and cache.
        >>> processor = AutoProcessor.from_pretrained("facebook/wav2vec2-base-960h")

        >>> # If processor files are in a directory (e.g. processor was saved using *save_pretrained('./test/saved_model/')*)
        >>> processor = AutoProcessor.from_pretrained("./test/saved_model/")
        ```"""
        config = kwargs.pop("config", None)
        kwargs["_from_auto"] = True

        # First, let's see if we have a preprocessor config.
        # Filter the kwargs for `get_file_from_repo``.
        get_file_from_repo_kwargs = {
            key: kwargs[key] for key in inspect.signature(get_file_from_repo).parameters.keys() if key in kwargs
        }
        # Let's start by checking whether the processor class is saved in a feature extractor
        preprocessor_config_file = get_file_from_repo(
            pretrained_model_name_or_path, FEATURE_EXTRACTOR_NAME, **get_file_from_repo_kwargs
        )
        if preprocessor_config_file is not None:
            config_dict, _ = FeatureExtractionMixin.get_feature_extractor_dict(pretrained_model_name_or_path, **kwargs)
            if "processor_class" in config_dict:
                processor_class = processor_class_from_name(config_dict["processor_class"])
                return processor_class.from_pretrained(pretrained_model_name_or_path, **kwargs)

        # Next, let's check whether the processor class is saved in a tokenizer
        # Let's start by checking whether the processor class is saved in a feature extractor
        tokenizer_config_file = get_file_from_repo(
            pretrained_model_name_or_path, TOKENIZER_CONFIG_FILE, **get_file_from_repo_kwargs
        )
        if tokenizer_config_file is not None:
            with open(tokenizer_config_file, encoding="utf-8") as reader:
                config_dict = json.load(reader)

            if "processor_class" in config_dict:
                processor_class = processor_class_from_name(config_dict["processor_class"])
                return processor_class.from_pretrained(pretrained_model_name_or_path, **kwargs)

        # Otherwise, load config, if it can be loaded.
        if not isinstance(config, PretrainedConfig):
            config = AutoConfig.from_pretrained(pretrained_model_name_or_path, **kwargs)

        model_type = config_class_to_model_type(type(config).__name__)

        if getattr(config, "processor_class", None) is not None:
            processor_class = processor_class_from_name(config.processor_class)
            return processor_class.from_pretrained(pretrained_model_name_or_path, **kwargs)

        model_type = config_class_to_model_type(type(config).__name__)
        if model_type is not None:
            return PROCESSOR_MAPPING[type(config)].from_pretrained(pretrained_model_name_or_path, **kwargs)

        raise ValueError(
            f"Unrecognized processor in {pretrained_model_name_or_path}. Should have a `processor_type` key in "
            f"its {FEATURE_EXTRACTOR_NAME}, or one of the following `model_type` keys in its {CONFIG_NAME}: "
            f"{', '.join(c for c in PROCESSOR_MAPPING_NAMES.keys())}"
        )
