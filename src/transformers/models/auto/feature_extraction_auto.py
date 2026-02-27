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
"""AutoFeatureExtractor class."""

import importlib
import os
from collections import OrderedDict

# Build the list of all feature extractors
from ...configuration_utils import PreTrainedConfig
from ...dynamic_module_utils import get_class_from_dynamic_module, resolve_trust_remote_code
from ...feature_extraction_utils import FeatureExtractionMixin
from ...utils import CONFIG_NAME, FEATURE_EXTRACTOR_NAME, PROCESSOR_NAME, cached_file, logging, safe_load_json_file
from .auto_factory import _LazyAutoMapping
from .configuration_auto import (
    CONFIG_MAPPING_NAMES,
    AutoConfig,
    model_type_to_module_name,
    replace_list_option_in_docstrings,
)


logger = logging.get_logger(__name__)

FEATURE_EXTRACTOR_MAPPING_NAMES = OrderedDict(
    [
        ("audio-spectrogram-transformer", "ASTFeatureExtractor"),
        ("audioflamingo3", "WhisperFeatureExtractor"),
        ("clap", "ClapFeatureExtractor"),
        ("clvp", "ClvpFeatureExtractor"),
        ("csm", "EncodecFeatureExtractor"),
        ("dac", "DacFeatureExtractor"),
        ("data2vec-audio", "Wav2Vec2FeatureExtractor"),
        ("dia", "DiaFeatureExtractor"),
        ("encodec", "EncodecFeatureExtractor"),
        ("gemma3n", "Gemma3nAudioFeatureExtractor"),
        ("glmasr", "WhisperFeatureExtractor"),
        ("granite_speech", "GraniteSpeechFeatureExtractor"),
        ("higgs_audio_v2_tokenizer", "DacFeatureExtractor"),
        ("hubert", "Wav2Vec2FeatureExtractor"),
        ("kyutai_speech_to_text", "KyutaiSpeechToTextFeatureExtractor"),
        ("lasr_ctc", "LasrFeatureExtractor"),
        ("lasr_encoder", "LasrFeatureExtractor"),
        ("markuplm", "MarkupLMFeatureExtractor"),
        ("mimi", "EncodecFeatureExtractor"),
        ("moonshine", "Wav2Vec2FeatureExtractor"),
        ("moshi", "EncodecFeatureExtractor"),
        ("musicgen", "EncodecFeatureExtractor"),
        ("musicgen_melody", "MusicgenMelodyFeatureExtractor"),
        ("parakeet_ctc", "ParakeetFeatureExtractor"),
        ("parakeet_encoder", "ParakeetFeatureExtractor"),
        ("pe_audio", "PeAudioFeatureExtractor"),
        ("pe_audio_video", "PeAudioFeatureExtractor"),
        ("phi4_multimodal", "Phi4MultimodalFeatureExtractor"),
        ("pop2piano", "Pop2PianoFeatureExtractor"),
        ("qwen2_5_omni", "WhisperFeatureExtractor"),
        ("qwen2_audio", "WhisperFeatureExtractor"),
        ("qwen3_omni_moe", "WhisperFeatureExtractor"),
        ("seamless_m4t", "SeamlessM4TFeatureExtractor"),
        ("seamless_m4t_v2", "SeamlessM4TFeatureExtractor"),
        ("sew", "Wav2Vec2FeatureExtractor"),
        ("sew-d", "Wav2Vec2FeatureExtractor"),
        ("speech_to_text", "Speech2TextFeatureExtractor"),
        ("speecht5", "SpeechT5FeatureExtractor"),
        ("unispeech", "Wav2Vec2FeatureExtractor"),
        ("unispeech-sat", "Wav2Vec2FeatureExtractor"),
        ("univnet", "UnivNetFeatureExtractor"),
        ("vibevoice_acoustic_tokenizer", "VibeVoiceAcousticTokenizerFeatureExtractor"),
        ("voxtral", "WhisperFeatureExtractor"),
        ("voxtral_realtime", "VoxtralRealtimeFeatureExtractor"),
        ("wav2vec2", "Wav2Vec2FeatureExtractor"),
        ("wav2vec2-bert", "Wav2Vec2FeatureExtractor"),
        ("wav2vec2-conformer", "Wav2Vec2FeatureExtractor"),
        ("wavlm", "Wav2Vec2FeatureExtractor"),
        ("whisper", "WhisperFeatureExtractor"),
        ("xcodec", "DacFeatureExtractor"),
    ]
)

FEATURE_EXTRACTOR_MAPPING = _LazyAutoMapping(CONFIG_MAPPING_NAMES, FEATURE_EXTRACTOR_MAPPING_NAMES)


def feature_extractor_class_from_name(class_name: str):
    for module_name, extractors in FEATURE_EXTRACTOR_MAPPING_NAMES.items():
        if class_name in extractors:
            module_name = model_type_to_module_name(module_name)

            module = importlib.import_module(f".{module_name}", "transformers.models")
            try:
                return getattr(module, class_name)
            except AttributeError:
                continue

    for extractor in FEATURE_EXTRACTOR_MAPPING._extra_content.values():
        if getattr(extractor, "__name__", None) == class_name:
            return extractor

    # We did not find the class, but maybe it's because a dep is missing. In that case, the class will be in the main
    # init and we return the proper dummy to get an appropriate error message.
    main_module = importlib.import_module("transformers")
    if hasattr(main_module, class_name):
        return getattr(main_module, class_name)

    return None


def get_feature_extractor_config(
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
    Loads the feature extractor configuration from a pretrained model feature extractor configuration.

    Args:
        pretrained_model_name_or_path (`str` or `os.PathLike`):
            This can be either:

            - a string, the *model id* of a pretrained model configuration hosted inside a model repo on
              huggingface.co.
            - a path to a *directory* containing a configuration file saved using the
              [`~FeatureExtractionMixin.save_pretrained`] method, e.g., `./my_model_directory/`.

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
            If `True`, will only try to load the feature extractor configuration from local files.

    <Tip>

    Passing `token=True` is required when you want to use a private model.

    </Tip>

    Returns:
        `Dict`: The configuration of the feature extractor.

    Examples:

    ```python
    # Download configuration from huggingface.co and cache.
    feature_extractor_config = get_feature_extractor_config("facebook/wav2vec2-base-960h")
    # This model does not have a feature extractor config so the result will be an empty dict.
    feature_extractor_config = get_feature_extractor_config("FacebookAI/xlm-roberta-base")

    # Save a pretrained feature extractor locally and you can reload its config
    from transformers import AutoFeatureExtractor

    feature_extractor = AutoFeatureExtractor.from_pretrained("facebook/wav2vec2-base-960h")
    feature_extractor.save_pretrained("feature-extractor-test")
    feature_extractor_config = get_feature_extractor_config("feature-extractor-test")
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
    resolved_feature_extractor_file = cached_file(
        pretrained_model_name_or_path,
        filename=FEATURE_EXTRACTOR_NAME,
        cache_dir=cache_dir,
        force_download=force_download,
        proxies=proxies,
        token=token,
        revision=revision,
        local_files_only=local_files_only,
        _raise_exceptions_for_gated_repo=False,
        _raise_exceptions_for_missing_entries=False,
    )

    # An empty list if none of the possible files is found in the repo
    if not resolved_feature_extractor_file and not resolved_processor_file:
        logger.info("Could not locate the feature extractor configuration file.")
        return {}

    # Load feature_extractor dict. Priority goes as (nested config if found -> feature extractor config)
    # We are downloading both configs because almost all models have a `processor_config.json` but
    # not all of these are nested. We need to check if it was saved recently as nested or if it is legacy style
    feature_extractor_dict = {}
    if resolved_processor_file is not None:
        processor_dict = safe_load_json_file(resolved_processor_file)
        if "feature_extractor" in processor_dict:
            feature_extractor_dict = processor_dict["feature_extractor"]

    if resolved_feature_extractor_file is not None and feature_extractor_dict is None:
        feature_extractor_dict = safe_load_json_file(resolved_feature_extractor_file)
    return feature_extractor_dict


class AutoFeatureExtractor:
    r"""
    This is a generic feature extractor class that will be instantiated as one of the feature extractor classes of the
    library when created with the [`AutoFeatureExtractor.from_pretrained`] class method.

    This class cannot be instantiated directly using `__init__()` (throws an error).
    """

    def __init__(self):
        raise OSError(
            "AutoFeatureExtractor is designed to be instantiated "
            "using the `AutoFeatureExtractor.from_pretrained(pretrained_model_name_or_path)` method."
        )

    @classmethod
    @replace_list_option_in_docstrings(FEATURE_EXTRACTOR_MAPPING_NAMES)
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        r"""
        Instantiate one of the feature extractor classes of the library from a pretrained model vocabulary.

        The feature extractor class to instantiate is selected based on the `model_type` property of the config object
        (either passed as an argument or loaded from `pretrained_model_name_or_path` if possible), or when it's
        missing, by falling back to using pattern matching on `pretrained_model_name_or_path`:

        List options

        Params:
            pretrained_model_name_or_path (`str` or `os.PathLike`):
                This can be either:

                - a string, the *model id* of a pretrained feature_extractor hosted inside a model repo on
                  huggingface.co.
                - a path to a *directory* containing a feature extractor file saved using the
                  [`~feature_extraction_utils.FeatureExtractionMixin.save_pretrained`] method, e.g.,
                  `./my_model_directory/`.
                - a path or url to a saved feature extractor JSON *file*, e.g.,
                  `./my_model_directory/preprocessor_config.json`.
            cache_dir (`str` or `os.PathLike`, *optional*):
                Path to a directory in which a downloaded pretrained model feature extractor should be cached if the
                standard cache should not be used.
            force_download (`bool`, *optional*, defaults to `False`):
                Whether or not to force to (re-)download the feature extractor files and override the cached versions
                if they exist.
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
                If `False`, then this function returns just the final feature extractor object. If `True`, then this
                functions returns a `Tuple(feature_extractor, unused_kwargs)` where *unused_kwargs* is a dictionary
                consisting of the key/value pairs whose keys are not feature extractor attributes: i.e., the part of
                `kwargs` which has not been used to update `feature_extractor` and is otherwise ignored.
            trust_remote_code (`bool`, *optional*, defaults to `False`):
                Whether or not to allow for custom models defined on the Hub in their own modeling files. This option
                should only be set to `True` for repositories you trust and in which you have read the code, as it will
                execute code present on the Hub on your local machine.
            kwargs (`dict[str, Any]`, *optional*):
                The values in kwargs of any keys which are feature extractor attributes will be used to override the
                loaded values. Behavior concerning key/value pairs whose keys are *not* feature extractor attributes is
                controlled by the `return_unused_kwargs` keyword parameter.

        <Tip>

        Passing `token=True` is required when you want to use a private model.

        </Tip>

        Examples:

        ```python
        >>> from transformers import AutoFeatureExtractor

        >>> # Download feature extractor from huggingface.co and cache.
        >>> feature_extractor = AutoFeatureExtractor.from_pretrained("facebook/wav2vec2-base-960h")

        >>> # If feature extractor files are in a directory (e.g. feature extractor was saved using *save_pretrained('./test/saved_model/')*)
        >>> # feature_extractor = AutoFeatureExtractor.from_pretrained("./test/saved_model/")
        ```"""
        config = kwargs.pop("config", None)
        trust_remote_code = kwargs.pop("trust_remote_code", None)
        kwargs["_from_auto"] = True

        config_dict, _ = FeatureExtractionMixin.get_feature_extractor_dict(pretrained_model_name_or_path, **kwargs)
        feature_extractor_class = config_dict.get("feature_extractor_type", None)
        feature_extractor_auto_map = None
        if "AutoFeatureExtractor" in config_dict.get("auto_map", {}):
            feature_extractor_auto_map = config_dict["auto_map"]["AutoFeatureExtractor"]

        # If we don't find the feature extractor class in the feature extractor config, let's try the model config.
        if feature_extractor_class is None and feature_extractor_auto_map is None:
            if not isinstance(config, PreTrainedConfig):
                config = AutoConfig.from_pretrained(
                    pretrained_model_name_or_path, trust_remote_code=trust_remote_code, **kwargs
                )
            # It could be in `config.feature_extractor_type``
            feature_extractor_class = getattr(config, "feature_extractor_type", None)
            if hasattr(config, "auto_map") and "AutoFeatureExtractor" in config.auto_map:
                feature_extractor_auto_map = config.auto_map["AutoFeatureExtractor"]

        if feature_extractor_class is not None:
            feature_extractor_class = feature_extractor_class_from_name(feature_extractor_class)

        has_remote_code = feature_extractor_auto_map is not None
        has_local_code = feature_extractor_class is not None or type(config) in FEATURE_EXTRACTOR_MAPPING
        if has_remote_code:
            if "--" in feature_extractor_auto_map:
                upstream_repo = feature_extractor_auto_map.split("--")[0]
            else:
                upstream_repo = None
            trust_remote_code = resolve_trust_remote_code(
                trust_remote_code, pretrained_model_name_or_path, has_local_code, has_remote_code, upstream_repo
            )

        if has_remote_code and trust_remote_code:
            feature_extractor_class = get_class_from_dynamic_module(
                feature_extractor_auto_map, pretrained_model_name_or_path, **kwargs
            )
            _ = kwargs.pop("code_revision", None)
            feature_extractor_class.register_for_auto_class()
            return feature_extractor_class.from_pretrained(pretrained_model_name_or_path, **kwargs)
        elif feature_extractor_class is not None:
            return feature_extractor_class.from_pretrained(pretrained_model_name_or_path, **kwargs)
        # Last try: we use the FEATURE_EXTRACTOR_MAPPING.
        elif type(config) in FEATURE_EXTRACTOR_MAPPING:
            feature_extractor_class = FEATURE_EXTRACTOR_MAPPING[type(config)]
            return feature_extractor_class.from_pretrained(pretrained_model_name_or_path, **kwargs)

        raise ValueError(
            f"Unrecognized feature extractor in {pretrained_model_name_or_path}. Should have a "
            f"`feature_extractor_type` key in its {FEATURE_EXTRACTOR_NAME} of {CONFIG_NAME}, or one of the following "
            f"`model_type` keys in its {CONFIG_NAME}: {', '.join(c for c in FEATURE_EXTRACTOR_MAPPING_NAMES)}"
        )

    @staticmethod
    def register(config_class, feature_extractor_class, exist_ok=False):
        """
        Register a new feature extractor for this class.

        Args:
            config_class ([`PreTrainedConfig`]):
                The configuration corresponding to the model to register.
            feature_extractor_class ([`FeatureExtractorMixin`]): The feature extractor to register.
        """
        FEATURE_EXTRACTOR_MAPPING.register(config_class, feature_extractor_class, exist_ok=exist_ok)


__all__ = ["FEATURE_EXTRACTOR_MAPPING", "AutoFeatureExtractor"]
