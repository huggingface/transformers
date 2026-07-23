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
"""AutoAudioProcessor and (deprecated) AutoFeatureExtractor classes."""

import importlib
import os
import warnings
from collections import OrderedDict

from ...audio_processing_base import AudioProcessingMixin
from ...configuration_utils import PreTrainedConfig
from ...dynamic_module_utils import get_class_from_dynamic_module, resolve_trust_remote_code
from ...feature_extraction_utils import FeatureExtractionMixin
from ...utils import CONFIG_NAME, FEATURE_EXTRACTOR_NAME, PROCESSOR_NAME, cached_file, logging, safe_load_json_file
from .auto_factory import _LazyAutoMapping
from .auto_mappings import FEATURE_EXTRACTOR_MAPPING_NAMES
from .configuration_auto import (
    CONFIG_MAPPING_NAMES,
    AutoConfig,
    model_type_to_module_name,
    replace_list_option_in_docstrings,
)


logger = logging.get_logger(__name__)

# Each entry maps a `model_type` to a dict of backend → audio-processor class name. The torch
# entry is the default returned by `AutoAudioProcessor.from_pretrained`; the numpy entry, when
# present, is the bit-exact CPU-only sibling (see docs/adr/0001-bit-exact-backend-parity.md).
# Non-audio feature extractors (e.g. MarkupLM) keep a single-key dict for back-compat.
FEATURE_EXTRACTOR_MAPPING_NAMES = OrderedDict(
    [
        # The backend label reflects the actual base class of the registered processor. When a
        # model has only one sibling today the other backend's lookup falls back via
        # `_load_class_with_fallback` with a warning. Whisper is the first model with both.
        ("audio-spectrogram-transformer", {"torch": "AudioSpectrogramTransformerAudioProcessor", "numpy": "AudioSpectrogramTransformerAudioProcessorNumpy"}),
        ("audioflamingo3", {"torch": "WhisperAudioProcessor", "numpy": "WhisperAudioProcessorNumpy"}),
        ("clap", {"torch": "ClapAudioProcessor", "numpy": "ClapAudioProcessorNumpy"}),
        ("clvp", {"torch": "ClvpAudioProcessor", "numpy": "ClvpAudioProcessorNumpy"}),
        ("cohere_asr", {"torch": "CohereAsrAudioProcessor", "numpy": "CohereAsrAudioProcessorNumpy"}),
        ("csm", {"torch": "EncodecAudioProcessor", "numpy": "EncodecAudioProcessorNumpy"}),
        ("dac", {"torch": "DacAudioProcessor", "numpy": "DacAudioProcessorNumpy"}),
        ("data2vec-audio", {"torch": "Wav2Vec2AudioProcessor", "numpy": "Wav2Vec2AudioProcessorNumpy"}),
        ("dia", {"torch": "DiaAudioProcessor", "numpy": "DiaAudioProcessorNumpy"}),
        ("encodec", {"torch": "EncodecAudioProcessor", "numpy": "EncodecAudioProcessorNumpy"}),
        ("gemma3n", {"torch": "Gemma3nAudioProcessor", "numpy": "Gemma3nAudioProcessorNumpy"}),
        ("gemma4", {"torch": "Gemma4AudioProcessor", "numpy": "Gemma4AudioProcessorNumpy"}),
        ("glmasr", {"torch": "WhisperAudioProcessor", "numpy": "WhisperAudioProcessorNumpy"}),
        ("granite_speech", {"torch": "GraniteSpeechAudioProcessor"}),
        ("granite_speech_plus", {"torch": "GraniteSpeechAudioProcessor"}),
        ("higgs_audio_v2_tokenizer", {"torch": "DacAudioProcessor", "numpy": "DacAudioProcessorNumpy"}),
        ("hubert", {"torch": "Wav2Vec2AudioProcessor", "numpy": "Wav2Vec2AudioProcessorNumpy"}),
        ("kyutai_speech_to_text", {"torch": "KyutaiSpeechToTextAudioProcessor", "numpy": "KyutaiSpeechToTextAudioProcessorNumpy"}),
        ("lasr_ctc", {"torch": "LasrAudioProcessor"}),
        ("lasr_encoder", {"torch": "LasrAudioProcessor"}),
        ("markuplm", {"torch": "MarkupLMFeatureExtractor"}),
        ("mimi", {"torch": "EncodecAudioProcessor", "numpy": "EncodecAudioProcessorNumpy"}),
        ("moonshine", {"torch": "Wav2Vec2AudioProcessor", "numpy": "Wav2Vec2AudioProcessorNumpy"}),
        ("moshi", {"torch": "EncodecAudioProcessor", "numpy": "EncodecAudioProcessorNumpy"}),
        ("musicgen", {"torch": "EncodecAudioProcessor", "numpy": "EncodecAudioProcessorNumpy"}),
        ("musicgen_melody", {"torch": "MusicgenMelodyAudioProcessor"}),
        ("parakeet_ctc", {"torch": "ParakeetAudioProcessor", "numpy": "ParakeetAudioProcessorNumpy"}),
        ("parakeet_encoder", {"torch": "ParakeetAudioProcessor", "numpy": "ParakeetAudioProcessorNumpy"}),
        ("parakeet_rnnt", {"torch": "ParakeetAudioProcessor", "numpy": "ParakeetAudioProcessorNumpy"}),
        ("parakeet_tdt", {"torch": "ParakeetAudioProcessor", "numpy": "ParakeetAudioProcessorNumpy"}),
        ("pe_audio", {"torch": "PeAudioAudioProcessor", "numpy": "PeAudioAudioProcessorNumpy"}),
        ("pe_audio_video", {"torch": "PeAudioAudioProcessor", "numpy": "PeAudioAudioProcessorNumpy"}),
        ("phi4_multimodal", {"torch": "Phi4MultimodalAudioProcessor"}),
        ("pop2piano", {"torch": "Pop2PianoAudioProcessor", "numpy": "Pop2PianoAudioProcessorNumpy"}),
        ("qwen2_5_omni", {"torch": "WhisperAudioProcessor", "numpy": "WhisperAudioProcessorNumpy"}),
        ("qwen2_audio", {"torch": "WhisperAudioProcessor", "numpy": "WhisperAudioProcessorNumpy"}),
        ("qwen3_omni_moe", {"torch": "WhisperAudioProcessor", "numpy": "WhisperAudioProcessorNumpy"}),
        ("seamless_m4t", {"torch": "SeamlessM4tAudioProcessor", "numpy": "SeamlessM4tAudioProcessorNumpy"}),
        ("seamless_m4t_v2", {"torch": "SeamlessM4tAudioProcessor", "numpy": "SeamlessM4tAudioProcessorNumpy"}),
        ("sew", {"torch": "Wav2Vec2AudioProcessor", "numpy": "Wav2Vec2AudioProcessorNumpy"}),
        ("sew-d", {"torch": "Wav2Vec2AudioProcessor", "numpy": "Wav2Vec2AudioProcessorNumpy"}),
        ("speech_to_text", {"torch": "SpeechToTextAudioProcessor", "numpy": "SpeechToTextAudioProcessorNumpy"}),
        ("speecht5", {"torch": "SpeechT5AudioProcessor", "numpy": "SpeechT5AudioProcessorNumpy"}),
        ("unispeech", {"torch": "Wav2Vec2AudioProcessor", "numpy": "Wav2Vec2AudioProcessorNumpy"}),
        ("unispeech-sat", {"torch": "Wav2Vec2AudioProcessor", "numpy": "Wav2Vec2AudioProcessorNumpy"}),
        ("univnet", {"torch": "UnivNetAudioProcessor", "numpy": "UnivNetAudioProcessorNumpy"}),
        ("vibevoice_acoustic_tokenizer", {"torch": "VibevoiceAcousticTokenizerAudioProcessor"}),
        ("vibevoice_asr", {"torch": "VibevoiceAcousticTokenizerAudioProcessor"}),
        ("voxtral", {"torch": "WhisperAudioProcessor", "numpy": "WhisperAudioProcessorNumpy"}),
        ("voxtral_realtime", {"torch": "VoxtralRealtimeAudioProcessor"}),
        ("wav2vec2", {"torch": "Wav2Vec2AudioProcessor", "numpy": "Wav2Vec2AudioProcessorNumpy"}),
        ("wav2vec2-bert", {"torch": "Wav2Vec2AudioProcessor", "numpy": "Wav2Vec2AudioProcessorNumpy"}),
        ("wav2vec2-conformer", {"torch": "Wav2Vec2AudioProcessor", "numpy": "Wav2Vec2AudioProcessorNumpy"}),
        ("wavlm", {"torch": "Wav2Vec2AudioProcessor", "numpy": "Wav2Vec2AudioProcessorNumpy"}),
        ("whisper", {"torch": "WhisperAudioProcessor", "numpy": "WhisperAudioProcessorNumpy"}),
        ("xcodec", {"torch": "DacAudioProcessor", "numpy": "DacAudioProcessorNumpy"}),
    ]
)

# Irregular legacy `XxxFeatureExtractor` names that do not derive from the new
# `XxxAudioProcessor` by simple suffix substitution. Used by
# `feature_extractor_class_from_name` to resolve hub JSON `feature_extractor_type` strings.
LEGACY_FEATURE_EXTRACTOR_NAME_MAP = {
    "ASTFeatureExtractor": "AudioSpectrogramTransformerAudioProcessor",
    "Gemma3nAudioFeatureExtractor": "Gemma3nAudioProcessor",
    "Gemma4AudioFeatureExtractor": "Gemma4AudioProcessor",
    "Speech2TextFeatureExtractor": "SpeechToTextAudioProcessor",
    "SeamlessM4TFeatureExtractor": "SeamlessM4tAudioProcessor",
    "VibeVoiceAcousticTokenizerFeatureExtractor": "VibevoiceAcousticTokenizerAudioProcessor",
    "PeAudioFeatureExtractor": "PeAudioAudioProcessor",
}


FEATURE_EXTRACTOR_MAPPING = _LazyAutoMapping(CONFIG_MAPPING_NAMES, FEATURE_EXTRACTOR_MAPPING_NAMES)


def _legacy_name_candidates(class_name: str) -> list[str]:
    """Translate a legacy `XxxFeatureExtractor` name into modern candidates.

    Hub `preprocessor_config.json` files have `feature_extractor_type: "WhisperFeatureExtractor"`
    (legacy) or `audio_processor_type: "WhisperAudioProcessor"` (new). When loading legacy
    configs we need to find the new class by name. Tries, in order:
      1. The name as-is (modern names already match)
      2. An explicit override in `LEGACY_FEATURE_EXTRACTOR_NAME_MAP`
      3. The simple `FeatureExtractor → AudioProcessor` suffix substitution
    """
    candidates = [class_name]
    if class_name in LEGACY_FEATURE_EXTRACTOR_NAME_MAP:
        # Explicit override wins over the generic suffix substitution
        candidates.append(LEGACY_FEATURE_EXTRACTOR_NAME_MAP[class_name])
    elif class_name.endswith("FeatureExtractor"):
        candidates.append(class_name.replace("FeatureExtractor", "AudioProcessor"))
    # De-duplicate while preserving order
    seen = set()
    return [c for c in candidates if not (c in seen or seen.add(c))]


def feature_extractor_class_from_name(class_name: str):
    """Resolve an audio-processor or legacy feature-extractor name to its class object.

    Handles both modern names (`WhisperAudioProcessor`, `WhisperAudioProcessorNumpy`) and the
    legacy `XxxFeatureExtractor` names still found in hub `preprocessor_config.json` files.
    """
    for candidate in _legacy_name_candidates(class_name):
        for model_type, extractors_dict in FEATURE_EXTRACTOR_MAPPING_NAMES.items():
            if candidate in extractors_dict.values():
                module_name = model_type_to_module_name(model_type)
                module = importlib.import_module(f".{module_name}", "transformers.models")
                try:
                    return getattr(module, candidate)
                except AttributeError:
                    continue

    for mapping in FEATURE_EXTRACTOR_MAPPING._extra_content.values():
        if isinstance(mapping, dict):
            for cls in mapping.values():
                if getattr(cls, "__name__", None) == class_name:
                    return cls
        elif getattr(mapping, "__name__", None) == class_name:
            return mapping

    # We did not find the class, but maybe it's because a dep is missing. In that case, the class will be in the main
    # init and we return the proper dummy to get an appropriate error message.
    main_module = importlib.import_module("transformers")
    if hasattr(main_module, class_name):
        return getattr(main_module, class_name)

    return None


def _resolve_audio_backend(backend: str | None) -> str:
    """Resolve raw backend input to a concrete backend name (`'torch'` or `'numpy'`).

    Default is `'torch'`; `'numpy'` is the bit-exact CPU-only sibling. If a model has no
    sibling for the requested backend, `_load_class_with_fallback` warns and falls back to
    the available one.
    """
    if backend is None:
        return "torch"
    if backend not in {"torch", "numpy"}:
        raise ValueError(f"Unknown audio-processor backend: {backend!r}. Expected 'torch' or 'numpy'.")
    return backend


def _load_class_with_fallback(mapping, backend):
    """Load an audio-processor class from a backend→class mapping, with fallback.

    Tries the requested backend first; if the model has no sibling for it, falls back to any
    other available backend and warns. Returns `None` if the mapping is empty.
    """
    backends_to_try = [backend] + [b for b in mapping if b != backend]

    for b in backends_to_try:
        value = mapping.get(b)
        if value is None:
            continue

        if isinstance(value, type):
            processor_class = value
        else:
            processor_class = feature_extractor_class_from_name(value)

        if processor_class is None or getattr(processor_class, "is_dummy", False):
            continue

        if b != backend:
            logger.warning_once(
                f"Requested audio-processor backend {backend!r} is not available for this model. "
                f"Falling back to {b!r} backend."
            )
        return processor_class

    return None


def _find_mapping_for_audio_processor(base_class_name: str) -> dict | None:
    """Find the backend→class mapping that contains `base_class_name` in its values."""

    def _value_matches(val, name: str) -> bool:
        if val is None:
            return False
        if isinstance(val, str):
            return val == name
        if isinstance(val, type):
            return getattr(val, "__name__", None) == name
        return False

    for mapping_dict in FEATURE_EXTRACTOR_MAPPING_NAMES.values():
        if any(_value_matches(v, base_class_name) for v in mapping_dict.values()):
            return mapping_dict

    for content in FEATURE_EXTRACTOR_MAPPING._extra_content.values():
        if isinstance(content, dict) and any(_value_matches(v, base_class_name) for v in content.values()):
            return content

    return None


def _load_backend_class(base_class_name: str, backend: str):
    """Load an audio-processor class for the requested backend, with fallback."""
    mapping = _find_mapping_for_audio_processor(base_class_name)
    if mapping is None:
        # Unknown class name (e.g. remote code, custom registration): default to the literal name
        mapping = {"torch": base_class_name}
    return _load_class_with_fallback(mapping, backend)


def _resolve_auto_map_class_ref(auto_map, backend: str) -> str:
    """Extract the class reference string from an `auto_map` entry based on backend preference."""
    if isinstance(auto_map, dict):
        return auto_map.get(backend) or next(iter(auto_map.values()))
    if isinstance(auto_map, (list, tuple)):
        return auto_map[0]
    return auto_map


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
    Loads the audio-processor / feature-extractor configuration from a pretrained model.

    Args:
        pretrained_model_name_or_path (`str` or `os.PathLike`):
            This can be either:

            - a string, the *model id* of a pretrained model configuration hosted inside a model repo on
              huggingface.co.
            - a path to a *directory* containing a configuration file saved using the
              [`~AudioProcessingMixin.save_pretrained`] method, e.g., `./my_model_directory/`.

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
            If `True`, will only try to load the audio-processor configuration from local files.

    Returns:
        `Dict`: The configuration of the audio processor.
    """
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

    if not resolved_feature_extractor_file and not resolved_processor_file:
        logger.info("Could not locate the audio-processor configuration file.")
        return {}

    feature_extractor_dict = {}
    if resolved_processor_file is not None:
        processor_dict = safe_load_json_file(resolved_processor_file)
        # New nested key takes priority; legacy `feature_extractor` key is the fallback.
        if "audio_processor" in processor_dict:
            feature_extractor_dict = processor_dict["audio_processor"]
        elif "feature_extractor" in processor_dict:
            feature_extractor_dict = processor_dict["feature_extractor"]

    if resolved_feature_extractor_file is not None and not feature_extractor_dict:
        feature_extractor_dict = safe_load_json_file(resolved_feature_extractor_file)
    return feature_extractor_dict


def _resolve_audio_processor_from_pretrained(pretrained_model_name_or_path, *, backend: str, **kwargs):
    """Shared resolution logic used by `AutoAudioProcessor` and the deprecated `AutoFeatureExtractor`.

    Reads the hub config, identifies the class via the new `audio_processor_type` key, falling
    back to the legacy `feature_extractor_type` key, then picks the right backend sibling.
    """
    config = kwargs.pop("config", None)
    trust_remote_code = kwargs.pop("trust_remote_code", None)
    kwargs["_from_auto"] = True

    config_dict, _ = AudioProcessingMixin.get_audio_processor_dict(pretrained_model_name_or_path, **kwargs)

    class_name_in_config = (
        config_dict.get("audio_processor_type")
        or config_dict.get("feature_extractor_type")
    )
    auto_map = config_dict.get("auto_map") or {}
    audio_processor_auto_map = auto_map.get("AutoAudioProcessor") or auto_map.get("AutoFeatureExtractor")

    if class_name_in_config is None and audio_processor_auto_map is None:
        if not isinstance(config, PreTrainedConfig):
            config = AutoConfig.from_pretrained(
                pretrained_model_name_or_path, trust_remote_code=trust_remote_code, **kwargs
            )
        class_name_in_config = getattr(config, "audio_processor_type", None) or getattr(
            config, "feature_extractor_type", None
        )
        if hasattr(config, "auto_map"):
            audio_processor_auto_map = (
                config.auto_map.get("AutoAudioProcessor")
                or config.auto_map.get("AutoFeatureExtractor")
            )

    audio_processor_class = None
    if class_name_in_config is not None:
        # Translate legacy → modern, then dispatch to the requested backend
        modern_name = _legacy_name_candidates(class_name_in_config)[-1]
        audio_processor_class = _load_backend_class(modern_name, backend)

    has_remote_code = audio_processor_auto_map is not None
    has_local_code = audio_processor_class is not None or type(config) in FEATURE_EXTRACTOR_MAPPING
    if has_local_code and audio_processor_class is None:
        audio_processor_class = _load_class_with_fallback(FEATURE_EXTRACTOR_MAPPING[type(config)], backend)
    explicit_local_code = has_local_code and audio_processor_class is not None and not audio_processor_class.__module__.startswith("transformers.")

    if has_remote_code:
        class_ref = _resolve_auto_map_class_ref(audio_processor_auto_map, backend)
        upstream_repo = class_ref.split("--")[0] if "--" in class_ref else None
        trust_remote_code = resolve_trust_remote_code(
            trust_remote_code, pretrained_model_name_or_path, has_local_code, has_remote_code, upstream_repo
        )

    if has_remote_code and trust_remote_code and not explicit_local_code:
        audio_processor_class = get_class_from_dynamic_module(
            class_ref, pretrained_model_name_or_path, **kwargs
        )
        _ = kwargs.pop("code_revision", None)
        audio_processor_class.register_for_auto_class()
        return audio_processor_class.from_pretrained(pretrained_model_name_or_path, **kwargs)

    if audio_processor_class is not None:
        return audio_processor_class.from_pretrained(pretrained_model_name_or_path, **kwargs)

    raise ValueError(
        f"Unrecognized audio processor in {pretrained_model_name_or_path}. Should have an "
        f"`audio_processor_type` or `feature_extractor_type` key in its {FEATURE_EXTRACTOR_NAME} or {CONFIG_NAME}, "
        f"or one of the following `model_type` keys in its {CONFIG_NAME}: "
        f"{', '.join(c for c in FEATURE_EXTRACTOR_MAPPING_NAMES)}"
    )


class AutoAudioProcessor:
    r"""
    This is a generic audio processor class that will be instantiated as one of the
    backend-specific [`~audio_processing_base.AudioProcessingMixin`] subclasses when created with the
    [`AutoAudioProcessor.from_pretrained`] class method.

    This class cannot be instantiated directly using `__init__()` (throws an error).
    """

    def __init__(self):
        raise OSError(
            "AutoAudioProcessor is designed to be instantiated "
            "using the `AutoAudioProcessor.from_pretrained(pretrained_model_name_or_path)` method."
        )

    @classmethod
    @replace_list_option_in_docstrings(FEATURE_EXTRACTOR_MAPPING_NAMES)
    def from_pretrained(cls, pretrained_model_name_or_path, *inputs, **kwargs):
        r"""
        Instantiate one of the audio processor classes of the library from a pretrained model.

        The class to instantiate is selected by reading the `audio_processor_type`
        (or legacy `feature_extractor_type`) entry in the hub `preprocessor_config.json`, then
        picking the right backend sibling (`backend="torch"` by default; `"numpy"` for the
        bit-exact CPU-only variant — see [`~docs/adr/0001-bit-exact-backend-parity`]).

        Params:
            pretrained_model_name_or_path (`str` or `os.PathLike`):
                A model identifier on huggingface.co, a path to a saved model directory, or a path to
                a saved audio-processor JSON file.
            backend (`str`, *optional*, defaults to `"torch"`):
                Which backend sibling to load. `"torch"` returns the `XxxAudioProcessor` class;
                `"numpy"` returns `XxxAudioProcessorNumpy` when it exists. If the requested
                backend has no sibling for the resolved model, falls back to the available one
                with a warning.

        List options
        """
        backend = _resolve_audio_backend(kwargs.pop("backend", None))
        return _resolve_audio_processor_from_pretrained(pretrained_model_name_or_path, backend=backend, **kwargs)

    @staticmethod
    def register(config_class, audio_processor_class, exist_ok=False):
        """
        Register a new audio processor for this class.

        Args:
            config_class ([`PreTrainedConfig`]):
                The configuration corresponding to the model to register.
            audio_processor_class ([`AudioProcessingMixin`] or `dict[str, type]`):
                Either a single class (treated as the torch backend) or a backend→class dict.
        """
        if not isinstance(audio_processor_class, dict):
            audio_processor_class = {"torch": audio_processor_class}
        FEATURE_EXTRACTOR_MAPPING.register(config_class, audio_processor_class, exist_ok=exist_ok)


class AutoFeatureExtractor:
    r"""
    Deprecated alias for [`AutoAudioProcessor`].

    Returns the same class as `AutoAudioProcessor.from_pretrained` (torch backend by default).
    Removal target: transformers v5.15. See [ADR 0002](docs/adr/0002-legacy-field-mapping.md).
    """

    def __init__(self):
        raise OSError(
            "AutoFeatureExtractor is designed to be instantiated "
            "using the `AutoFeatureExtractor.from_pretrained(pretrained_model_name_or_path)` method."
        )

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        r"""Deprecated alias for [`AutoAudioProcessor.from_pretrained`]."""
        warnings.warn(
            "`AutoFeatureExtractor` is deprecated and will be removed in transformers v5.15. "
            "Use `AutoAudioProcessor` instead.",
            FutureWarning,
            stacklevel=2,
        )
        backend = _resolve_audio_backend(kwargs.pop("backend", None))
        return _resolve_audio_processor_from_pretrained(pretrained_model_name_or_path, backend=backend, **kwargs)

    @staticmethod
    def register(config_class, feature_extractor_class, exist_ok=False):
        """Deprecated alias for [`AutoAudioProcessor.register`]."""
        warnings.warn(
            "`AutoFeatureExtractor.register` is deprecated and will be removed in transformers v5.15. "
            "Use `AutoAudioProcessor.register` instead.",
            FutureWarning,
            stacklevel=2,
        )
        AutoAudioProcessor.register(config_class, feature_extractor_class, exist_ok=exist_ok)


__all__ = ["FEATURE_EXTRACTOR_MAPPING", "AutoAudioProcessor", "AutoFeatureExtractor"]
