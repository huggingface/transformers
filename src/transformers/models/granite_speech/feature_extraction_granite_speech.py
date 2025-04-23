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
"""Feature extractor class for Granite Speech."""

import math
from collections.abc import Sequence
from typing import Optional

import numpy as np

from ...feature_extraction_utils import BatchFeature, FeatureExtractionMixin
from ...tokenization_utils_base import AudioInput
from ...utils import is_torch_available, is_torchaudio_available, logging
from ...utils.import_utils import requires_backends


logger = logging.get_logger(__name__)

if is_torch_available():
    import torch

if is_torchaudio_available():
    import torchaudio


class GraniteSpeechFeatureExtractor(FeatureExtractionMixin):
    model_input_names = ["input_features"]

    def __init__(
        self,
        sampling_rate: int = 16000,
        n_fft: int = 512,
        win_length: int = 400,
        hop_length: int = 160,
        n_mels: int = 80,
        projector_window_size: int = 15,
        projector_downsample_rate: int = 5,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.melspec_kwargs = {
            "sample_rate": sampling_rate,
            "n_fft": n_fft,
            "win_length": win_length,
            "hop_length": hop_length,
            "n_mels": n_mels,
        }
        # Currently lazily initialized
        self.melspec = None
        self.projector_window_size = projector_window_size
        self.projector_downsample_rate = projector_downsample_rate

    def __call__(
        self,
        audios: AudioInput,
        device: Optional[str] = "cpu",
    ) -> BatchFeature:
        requires_backends(self, ["torchaudio"])

        speech_inputs = {}
        batched_audio, audio_lengths = self._get_audios_and_audio_lengths(audios)
        speech_inputs["input_features"] = self._extract_mel_spectrograms(
            batched_audio,
            device=device,
        )
        audio_embed_sizes = self._get_num_audio_features(audio_lengths)
        speech_inputs["audio_embed_sizes"] = audio_embed_sizes
        # TODO (@alex-jw-brooks): Currently input_features_mask is not
        # a great name, because input_features and input_features_mask
        # have different shapes (before/after the projector).
        #
        # We should align this with other multimodal models, e.g,. llava
        # and qwen2audio and refactor this to ensure input_feature_mask
        # has the same dimensionality as input_features, or compute it in
        # the model based on the audio embedding sizes (since we do not
        # have an attention mask for the audio features to infer padding from).
        speech_inputs["input_features_mask"] = torch.arange(max(audio_embed_sizes)).view(1, -1) < torch.tensor(
            audio_embed_sizes
        ).view(-1, 1)
        return BatchFeature(data=speech_inputs)

    def _ensure_melspec_transform_is_initialized(self):
        """
        Ensures the mel spectrogram transform on this instance is initialized.

        We do this for now since some logging explodes since the mel spectrogram
        transform is not JSON serializable.
        """
        requires_backends(self, ["torchaudio"])

        if self.melspec is None:
            # TODO (@alex-jw-brooks / @eustlb) move this to common batch
            # feature extraction in audio utils once they are written!
            self.melspec = torchaudio.transforms.MelSpectrogram(**self.melspec_kwargs)

    def _extract_mel_spectrograms(self, audio: "torch.Tensor", device="cpu"):
        """
        Compute the Mel features to be passed to the conformer encoder.
        """
        requires_backends(self, ["torchaudio"])

        # Initialize the mel spectrogram if isn't not already and
        # move the melspec / audio to the computation device.
        self._ensure_melspec_transform_is_initialized()
        if device is not None:
            melspec = self.melspec.to(device)
            audio = audio.to(device)
        else:
            melspec = self.melspec

        bsz = audio.shape[0]
        with torch.no_grad():
            # Compute mel features
            mel = melspec(audio.float())
            logmel = mel.transpose(-1, -2).clip_(min=1e-10).log10_()
            mx = logmel.amax(dim=(-2, -1), keepdim=True)
            logmel = torch.maximum(logmel, mx - 8.0).div_(4).add_(1)
            # remove last frame if odd
            if logmel.shape[1] % 2 == 1:
                logmel = logmel[:, :-1]

            # stacking and skipping by 2
            audio = logmel.reshape(bsz, -1, 2 * logmel.shape[-1])

        if audio.device != "cpu":
            return audio.detach().cpu()
        return audio

    def _get_num_audio_features(self, audio_lengths: Sequence[int]) -> Sequence[int]:
        """
        Gets the (variable length) number of features (i.e., projector output) for the sequences
        being considered.

        Args:
            audio_lengths (`Sequence[int]`):
                Sequence of one or more raw audio lengths.
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

    def _get_audios_and_audio_lengths(self, audios: AudioInput) -> Sequence["torch.Tensor", Sequence[int]]:
        """
        Coerces audio inputs to torch tensors and extracts audio lengths prior to stacking.

        Args:
            audios (`AudioInput`):
                Audio sequence, numpy array, or torch tensor.
        """
        requires_backends(self, ["torch"])

        # Coerce to PyTorch tensors if we have numpy arrays, since
        # currently we have a dependency on torch/torchaudio anyway
        if isinstance(audios, np.ndarray):
            audios = torch.from_numpy(audios)
        elif isinstance(audios, Sequence) and isinstance(audios[0], np.ndarray):
            audios = [torch.from_numpy(arr) for arr in audios]

        if isinstance(audios, torch.Tensor):
            if audios.ndim == 1:
                audios = audios.unsqueeze(0)
            if not torch.is_floating_point(audios):
                raise ValueError("Invalid audio provided. Audio should be a floating point between 0 and 1")

            if audios.shape[0] > 1:
                logger.warning("Audio samples are already collated; assuming they all have the same length")
            lengths = [audios.shape[-1]] * audios.shape[0]
            return audios, lengths

        elif isinstance(audios, Sequence) and isinstance(audios[0], torch.Tensor):
            if not torch.is_floating_point(audios[0]):
                raise ValueError("Invalid audio provided. Audio should be a floating point between 0 and 1")
            lengths = [audio.shape[-1] for audio in audios]
            padding = [max(lengths) - length for length in lengths]
            # ensure all audios have a batch dimension:
            audios = [audio.view(1, -1) for audio in audios]
            padded = [torch.nn.functional.pad(audio, (0, pad)) for audio, pad in zip(audios, padding)]
            audios = torch.cat(padded, dim=0)
            return audios, lengths

        raise TypeError("Invalid audio provided. Audio should be a one or more torch tensors or numpy arrays")


__all__ = ["GraniteSpeechFeatureExtractor"]
