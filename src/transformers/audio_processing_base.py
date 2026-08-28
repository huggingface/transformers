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
import warnings
from dataclasses import fields, replace
from typing import Any, ClassVar, TypeVar

from .audio_utils import MelScaleConfig, is_valid_audio, load_audio
from .preprocessing_base import BatchFeature as BaseBatchFeature
from .preprocessing_base import PreprocessingMixin
from .utils import (
    FEATURE_EXTRACTOR_NAME,
    copy_func,
    logging,
)


_LEGACY_KEY_MAP = {
    "input_features": "audio_features",
    "input_values": "audio_values",
    "audio_input_features": "audio_features",
}


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

    _warned_keys: ClassVar[set] = set()

    def __getitem__(self, item):
        if isinstance(item, str) and item not in self.data:
            new_key = self._resolve_legacy_key(item)
            if new_key is not None and new_key in self.data:
                if item not in BatchFeature._warned_keys:
                    warnings.warn(
                        f"Accessing '{item}' is deprecated, use '{new_key}' instead.",
                        FutureWarning,
                        stacklevel=2,
                    )
                    BatchFeature._warned_keys.add(item)
                return self.data[new_key]
        return super().__getitem__(item)

    def __contains__(self, item):
        if item in self.data:
            return True
        new_key = self._resolve_legacy_key(item)
        return new_key is not None and new_key in self.data

    def __getattr__(self, item):
        # `outputs.input_features` is as common in the wild as `outputs["input_features"]`.
        if isinstance(item, str) and item not in ("data", "skip_tensor_conversion"):
            new_key = self._resolve_legacy_key(item)
            if new_key is not None and new_key in self.data:
                return self[item]
        return super().__getattr__(item)

    def __delitem__(self, item):
        # Makes `pop`/`del` on a legacy key work, since both route through here.
        if isinstance(item, str) and item not in self.data:
            new_key = self._resolve_legacy_key(item)
            if new_key is not None and new_key in self.data:
                item = new_key
        del self.data[item]

    def _resolve_legacy_key(self, old_key):
        if old_key in ("attention_mask", "padding_mask"):
            if "audio_features_mask" in self.data:
                return "audio_features_mask"
            if "audio_values_mask" in self.data:
                return "audio_values_mask"
            return None
        return _LEGACY_KEY_MAP.get(old_key)


class AudioProcessingMixin(PreprocessingMixin):
    """
    This is an audio processor mixin used to provide saving/loading functionality for audio processors.
    """

    _config_name = FEATURE_EXTRACTOR_NAME
    _type_key = "audio_processor_type"
    _nested_config_keys = ["audio_processor", "feature_extractor"]
    _auto_class_default = "AutoAudioProcessor"
    _file_type_label = "audio processor"
    _excluded_dict_keys = {"mel_filters", "window"}
    _extra_init_pops = ["feature_extractor_type"]
    _config_filename_kwarg = "audio_processor_filename"
    _subfolder_default = ""

    # Legacy hub-config translation. Hub `preprocessor_config.json` files written by the
    # old `XxxFeatureExtractor` classes use a flat key schema that doesn't match the new
    # nested `SpectrogramConfig` API. `from_dict` applies `_legacy_field_mapping_base`
    # first, then any per-model `legacy_field_mapping` last (highest priority). Values:
    #
    #   - str:       dot-path to the nested target (e.g. ``"spectrogram_config.stft_config.hop_length"``)
    #                — `from_dict` walks/creates intermediate dicts and writes the value.
    #   - callable:  invoked as ``f(value, config_dict)`` and expected to mutate
    #                ``config_dict`` in place. For non-1:1 mappings, where one legacy key has to be
    #                spread across several modern ones. Note the callable *consumes* the legacy key:
    #                if the modern API keeps a field of the same name, leave it out of the mapping
    #                entirely so it passes through untranslated (see `sampling_rate` below).
    #   - None:      drop the legacy key with no translation.
    #
    # The base mapping covers both universal keys (`return_attention_mask`, `feature_extractor_type`,
    # …) and spectrogram-domain keys (`n_fft`, `hop_length`, …). For non-spectrogram models
    # the spectrogram keys are simply absent from the hub config — translation is a no-op.
    # See docs/adr/0002-legacy-field-mapping.md.
    _legacy_field_mapping_base: dict = {
        # Universal keys (apply to every audio processor).
        # NOTE: `sampling_rate` is intentionally not listed — the hub key matches the modern
        # instance attribute `sampling_rate` verbatim, so it passes through `from_dict` untranslated.
        "feature_extractor_type": None,
        "audio_processor_type": None,
        "processor_class": None,
        "return_attention_mask": "return_padding_mask",
        # Length keys, in samples.
        "n_samples": "max_length",
        "nb_max_samples": "max_length",
        # Spectrogram-domain keys, skipped for models that declare no `spectrogram_config`
        # (see `_legacy_target_exists`). Several appear under more than one legacy spelling.
        "hop_length": "spectrogram_config.stft_config.hop_length",
        "n_fft": "spectrogram_config.stft_config.n_fft",
        "fft_length": "spectrogram_config.stft_config.n_fft",
        "fft_window_size": "spectrogram_config.stft_config.n_fft",
        "win_length": "spectrogram_config.stft_config.win_length",
        "frame_length": "spectrogram_config.stft_config.win_length",
        "window_fn": "spectrogram_config.stft_config.window_fn",
        "power": "spectrogram_config.stft_config.power",
        "center": "spectrogram_config.stft_config.center",
        "pad_mode": "spectrogram_config.stft_config.pad_mode",
        # `feature_size` is the mel count for the 17 spectrogram FEs that persisted it. The
        # guard skips the raw-audio models whose configs also carry it (always as `1`); the
        # three that produce mels *and* saved `feature_size=1` keep the count in
        # `num_mel_bins` and opt out with `"feature_size": None`.
        "feature_size": "spectrogram_config.mel_scale_config.n_mels",
        "num_mel_bins": "spectrogram_config.mel_scale_config.n_mels",
        "f_min": "spectrogram_config.mel_scale_config.f_min",
        "f_max": "spectrogram_config.mel_scale_config.f_max",
        "min_frequency": "spectrogram_config.mel_scale_config.f_min",
        "max_frequency": "spectrogram_config.mel_scale_config.f_max",
        "frequency_min": "spectrogram_config.mel_scale_config.f_min",
        "frequency_max": "spectrogram_config.mel_scale_config.f_max",
        "preemphasis": "spectrogram_config.preemphasis",
        "mel_floor": "spectrogram_config.mel_floor",
        # Derived from the keys above; the modern pipeline recomputes them. `max_length_s` is
        # deliberately absent: UnivNet still reads it off the instance.
        "nb_max_frames": None,
        "nb_frequency_bins": None,
    }
    legacy_field_mapping: dict | None = None

    @classmethod
    def _legacy_target_exists(cls, target: str) -> bool:
        """Whether the config a dot-path ``target`` writes into is declared on the class.

        Guards the spectrogram-domain mappings: a raw-audio model whose hub config happens to
        carry `hop_length` or `num_mel_bins` has no `spectrogram_config` to receive them, and
        conjuring one out of those keys alone would silently turn it into a spectrogram
        extractor (`_standardize_kwargs` derives `do_extract_spectrogram` from its presence).
        """
        config = cls
        for part in target.split(".")[:-1]:
            config = getattr(config, part, None)
            if config is None:
                return False
        return True

    @classmethod
    def _apply_legacy_field_mapping(cls, config_dict: dict) -> dict:
        """Translate legacy hub-config keys to the new nested schema. Mutates and returns ``config_dict``."""
        merged = {**cls._legacy_field_mapping_base, **(cls.legacy_field_mapping or {})}
        for legacy_key, target in merged.items():
            if legacy_key not in config_dict:
                continue
            if isinstance(target, str) and not cls._legacy_target_exists(target):
                # Leave the key flat rather than dropping it: models that compute their own
                # features read it straight off the instance (e.g. Musicgen-Melody's `n_fft`).
                continue
            value = config_dict.pop(legacy_key)
            if target is None:
                continue
            if callable(target):
                target(value, config_dict)
                continue
            # Dot-path target: walk/create intermediate dicts. Don't overwrite an
            # existing modern value if both legacy and modern keys are present.
            parts = target.split(".")
            d = config_dict
            for part in parts[:-1]:
                next_d = d.get(part)
                if next_d is None:
                    next_d = {}
                    d[part] = next_d
                elif not isinstance(next_d, dict):
                    raise TypeError(
                        f"Cannot apply legacy mapping {legacy_key!r}→{target!r}: "
                        f"intermediate key {part!r} is not a dict ({type(next_d).__name__})."
                    )
                d = next_d
            if parts[-1] not in d:
                d[parts[-1]] = value
        return config_dict

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any], **kwargs):
        config_dict = dict(config_dict)
        cls._apply_legacy_field_mapping(config_dict)
        cls._merge_spectrogram_config(config_dict)
        return super().from_dict(config_dict, **kwargs)

    @classmethod
    def _merge_spectrogram_config(cls, config_dict: dict[str, Any]) -> None:
        """Merge an incoming (usually partial) ``spectrogram_config`` onto the class-level one.

        Legacy hub configs only carry a handful of flat keys, which `_apply_legacy_field_mapping`
        turns into a *partial* nested dict such as ``{"stft_config": {"hop_length": 160}}``.
        Instantiating from that alone would fall back to bare `SpectrogramConfig` defaults and
        silently drop everything the model class defines (mel scale, log mode, dtypes, ...), so
        the incoming values are layered on top of the class config instead.
        """
        incoming = config_dict.get("spectrogram_config")
        default = getattr(cls, "spectrogram_config", None)
        if not isinstance(incoming, dict) or default is None:
            return

        incoming = dict(incoming)
        merged = default
        for key, nested_cls in (("stft_config", None), ("mel_scale_config", MelScaleConfig)):
            nested = incoming.pop(key, None)
            if not isinstance(nested, dict):
                continue
            current = getattr(merged, key)
            if current is None:
                # No class-level nested config (e.g. a raw-audio model gaining a mel scale).
                if nested_cls is None:
                    continue
                current = nested_cls()
            valid = {f.name for f in fields(current)}
            merged = replace(merged, **{key: replace(current, **{k: v for k, v in nested.items() if k in valid})})

        valid = {f.name for f in fields(merged)}
        config_dict["spectrogram_config"] = replace(merged, **{k: v for k, v in incoming.items() if k in valid})

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

    def fetch_audio(self, audio_url_or_urls: str | list[str] | list[list[str]], sampling_rate: int | None = None):
        """
        Convert a single or a list of urls into the corresponding `np.ndarray` objects.

        If a single url is passed, the return value will be a single object. If a list is passed a list of objects is
        returned.
        """
        if sampling_rate is None:
            sampling_rate = getattr(self, "sampling_rate", 16000)
        if isinstance(audio_url_or_urls, list):
            return [self.fetch_audio(x, sampling_rate=sampling_rate) for x in audio_url_or_urls]
        elif isinstance(audio_url_or_urls, str):
            return load_audio(audio_url_or_urls, sampling_rate=sampling_rate)
        elif is_valid_audio(audio_url_or_urls):
            return audio_url_or_urls
        else:
            raise TypeError(f"only a single or a list of entries is supported but got type={type(audio_url_or_urls)}")


def make_legacy_audio_processor_alias(new_class: type, legacy_name: str) -> type:
    """Create a deprecated subclass alias for the legacy ``XxxFeatureExtractor`` name.

    Instantiating the alias emits a ``FutureWarning`` directing users to the new class. The
    alias overrides ``to_dict`` so that saved configs identify themselves under the new
    class name (``audio_processor_type: "WhisperAudioProcessor"``, not the legacy name) —
    a `from_pretrained` followed by `save_pretrained` is enough to migrate a checkpoint.

    Removal target: transformers v5.15. See [ADR 0002](docs/adr/0002-legacy-field-mapping.md).
    """

    def __init__(self, *args, **kwargs):
        warnings.warn(
            f"`{legacy_name}` is deprecated and will be removed in transformers v5.15. "
            f"Use `{new_class.__name__}` instead.",
            FutureWarning,
            stacklevel=2,
        )
        new_class.__init__(self, *args, **kwargs)

    def to_dict(self):
        output = new_class.to_dict(self)
        # Migrate only direct alias instances; user subclasses keep their own class name.
        if type(self).__name__ == legacy_name:
            output[self._type_key] = new_class.__name__
        return output

    return type(
        legacy_name,
        (new_class,),
        {
            "__init__": __init__,
            "to_dict": to_dict,
            "__doc__": f"Deprecated alias for [`{new_class.__name__}`]. Removal: transformers v5.15.",
        },
    )


AudioProcessingMixin.push_to_hub = copy_func(AudioProcessingMixin.push_to_hub)
if AudioProcessingMixin.push_to_hub.__doc__ is not None:
    AudioProcessingMixin.push_to_hub.__doc__ = AudioProcessingMixin.push_to_hub.__doc__.format(
        object="audio processor", object_class="AutoFeatureExtractor", object_files="audio processor file"
    )
