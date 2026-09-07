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
Deprecated sequence feature extraction class, kept as a thin shim over [`BaseAudioProcessor`].
"""

import warnings
from dataclasses import replace

from .audio_processing_utils import BaseAudioProcessor
from .utils import logging


logger = logging.get_logger(__name__)


class SequenceFeatureExtractor(BaseAudioProcessor):
    """
    Deprecated base class for speech feature extractors. Subclass [`BaseAudioProcessor`] (through
    `NumpyAudioBackend` or `TorchAudioBackend`) instead; every in-library `XxxFeatureExtractor` is
    now a deprecated alias of the corresponding `XxxAudioProcessor`.

    Padding, truncation, masking and audio fetching all come from [`BaseAudioProcessor`]. Note that
    the inherited `pad` operates on raw audio (as used by the audio-processor pipeline) rather than
    on already-extracted feature dicts like the legacy `SequenceFeatureExtractor.pad` did.

    Args:
        feature_size (`int`, *optional*):
            The feature dimension of the extracted features. Translated to
            `spectrogram_config.mel_scale_config.n_mels` when the subclass defines a mel configuration.
        sampling_rate (`int`, *optional*):
            The sampling rate at which the audio files should be digitalized, expressed in hertz (Hz).
        padding_value (`float`, *optional*, defaults to 0.0):
            The value that is used to fill the padding values / vectors.
    """

    def __init__(
        self,
        feature_size: int | None = None,
        sampling_rate: int | None = None,
        padding_value: float = 0.0,
        **kwargs,
    ):
        warnings.warn(
            "`SequenceFeatureExtractor` is deprecated and will be removed in transformers v5.15. Use the "
            "model's `XxxAudioProcessor` (a `BaseAudioProcessor` subclass) instead.",
            FutureWarning,
        )

        # Legacy `return_attention_mask` drives the modern `return_padding_mask` (same meaning, same
        # default), mirroring the `_legacy_field_mapping_base` translation used for hub configs.
        if "return_attention_mask" in kwargs:
            kwargs.setdefault("return_padding_mask", kwargs["return_attention_mask"])

        super().__init__(sampling_rate=sampling_rate, padding_value=padding_value, **kwargs)

        # Legacy `feature_size` is the number of mel bins; it only has a target when the subclass
        # provides a mel configuration (raw-audio processors have no `n_mels` to set).
        mel_scale_config = getattr(self.spectrogram_config, "mel_scale_config", None)
        if feature_size is not None and mel_scale_config is not None:
            self.spectrogram_config = replace(
                self.spectrogram_config,
                mel_scale_config=replace(mel_scale_config, n_mels=feature_size),
            )


__all__ = ["SequenceFeatureExtractor"]
