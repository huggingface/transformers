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
"""
Feature extractor class for Speech Granite
"""
from typing import Optional, List
import math
from transformers.feature_extraction_utils import BatchFeature, FeatureExtractionMixin
from transformers.utils import logging, is_torch_available, is_torchaudio_available

logger = logging.get_logger(__name__)

if is_torch_available():
    import torch

if is_torchaudio_available():
    import torchaudio

class GraniteSpeechFeatureExtractor(FeatureExtractionMixin):
    model_input_names = ["input_features"]

    def __init__(
        self,
        sampling_rate=16000,
        n_fft=512,
        win_length=400,
        hop_length=160,
        n_mels=80,
        projector_window_size=15,
        projector_downsample_rate=5,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.melspec_kwargs = {
            "sample_rate": sampling_rate,
            "n_fft": n_fft,
            "win_length": win_length,
            "hop_length": hop_length,
            "n_mels": n_mels
        }
        # HACK - for now, lazily initialize the mel spectrogram transform;
        # the feature extractor mixin explodes otherwise because
        # it tries to log the feature extractor, and the melspectrogram
        # transform isn't json serializable...
        self.melspec = None
        self.projector_window_size = projector_window_size
        self.projector_downsample_rate = projector_downsample_rate

    def _ensure_melspec_transform_is_initialized(self):
        if self.melspec is None:
            self.melspec = torchaudio.transforms.MelSpectrogram(
                **self.melspec_kwargs
            )

    def __call__(
        self,
        x: torch.Tensor,
        device: Optional[str]="cpu",
    ) -> BatchFeature:
        # TODO there is probably a better way to do both of these things...
        self._ensure_melspec_transform_is_initialized()
        if device is not None:
            melspec = self.melspec.to(device)
            x = x.to(device)
        else:
            melspec = self.melspec

        B, _ = x.shape
        with torch.no_grad():
            mel = melspec(x.float())
            logmel = mel.transpose(-1,-2).clip_(min=1e-10).log10_()
            mx = logmel.amax(dim=(-2,-1), keepdim=True)
            logmel = torch.maximum(logmel, mx - 8.0).div_(4).add_(1)
            if logmel.shape[1] % 2 == 1:
                logmel = logmel[:,:-1]                       # remove last frame if odd
            x = logmel.reshape(B, -1, 2 * logmel.shape[-1])  # stacking and skipping by 2

        if x.device != "cpu":
            return x.detach().cpu()
        return x

    
    def _get_num_audio_features(self, audio_lengths: List[int]) -> List[int]:
        """
        Gets the (variable length) variable length number of features
        (i.e., projector output) for the sequences being considered.
        """
        hop_length = self.melspec_kwargs["hop_length"]
        effective_window_size = self.projector_window_size // self.projector_downsample_rate

        projector_lengths = []
        for raw_length in audio_lengths:
            # mel sequence length computation
            mel_length = raw_length // hop_length + 1
            # encoder frame takes two mel features
            encoder_length = mel_length // 2
            nblocks = math.ceil(encoder_length / self.projector_window_size)
            # projector output length
            projector_length = nblocks * effective_window_size
            projector_lengths.append(projector_length)
        
        return projector_lengths

        
__all__ = ["GraniteSpeechFeatureExtractor"]