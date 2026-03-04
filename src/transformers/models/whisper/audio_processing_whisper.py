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
Audio processor class for Whisper
"""


import torch

from ...audio_processing_backends import TorchBackend
from ...audio_utils import mel_filter_bank
from ...feature_extraction_utils import BatchFeature
from ...utils import logging


logger = logging.get_logger(__name__)


class WhisperAudioProcessor(TorchBackend):
    r"""
    Constructs a Whisper audio processor.

    This audio processor inherits from [`~audio_processing_utils.BaseAudioProcessor`] which contains
    most of the main methods. Users should refer to this superclass for more information regarding those methods.

    This class extracts mel-filter bank features from raw speech using PyTorch's `torch.stft`.

    Args:
        feature_size (`int`, *optional*, defaults to 80):
            The feature dimension of the extracted features (number of mel bins).
        sample_rate (`int`, *optional*, defaults to 16000):
            The sampling rate at which the audio files should be digitalized expressed in hertz (Hz).
        hop_length (`int`, *optional*, defaults to 160):
            Length of the overlapping windows for the STFT used to obtain the Mel Frequency coefficients.
        chunk_length (`int`, *optional*, defaults to 30):
            The maximum number of seconds of audio used to trim and pad sequences.
        n_fft (`int`, *optional*, defaults to 400):
            Size of the Fourier transform.
        dither (`float`, *optional*, defaults to 0.0):
            Adds dithering (small Gaussian noise) to each frame. Use 0.0 for no dithering.
    """

    model_input_names = ["input_features"]

    def __init__(
        self,
        feature_size: int = 80,
        sample_rate: int = 16000,
        hop_length: int = 160,
        chunk_length: int = 30,
        n_fft: int = 400,
        dither: float = 0.0,
        force_mono: bool = True,
        **kwargs,
    ):
        super().__init__(
            sample_rate=sample_rate,
            force_mono=force_mono,
            **kwargs,
        )
        self.feature_size = feature_size
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.chunk_length = chunk_length
        self.n_samples = chunk_length * sample_rate
        self.nb_max_frames = self.n_samples // hop_length
        self.dither = dither
        self.mel_filters = mel_filter_bank(
            num_frequency_bins=1 + n_fft // 2,
            num_mel_filters=feature_size,
            min_frequency=0.0,
            max_frequency=8000.0,
            sampling_rate=sample_rate,
            norm="slaney",
            mel_scale="slaney",
        )

    def _preprocess(
        self,
        audio: list[torch.Tensor],
        padding,
        max_length,
        truncation,
        pad_to_multiple_of,
        return_tensors,
        do_normalize: bool | None = None,
        device: str | None = "cpu",
        **kwargs,
    ) -> BatchFeature:
        # Default max_length to n_samples (chunk_length * sample_rate)
        if max_length is None:
            max_length = self.n_samples

        # Use base class for truncation + padding
        result = super()._preprocess(
            audio,
            padding=padding if padding is not None else True,
            max_length=max_length,
            truncation=truncation if truncation is not None else True,
            pad_to_multiple_of=pad_to_multiple_of,
            return_tensors=None,  # we handle conversion after feature extraction
        )

        audio_tensors = result["audio"]

        # Zero-mean unit-variance normalization (before spectrogram)
        if do_normalize:
            audio_tensors = [(t - t.mean()) / torch.sqrt(t.var() + 1e-7) for t in audio_tensors]

        # Stack into batch for spectrogram extraction
        waveform_batch = torch.stack(audio_tensors, dim=0).to(device, torch.float32)

        # Extract log-mel spectrogram
        input_features = self._extract_fbank_features(waveform_batch, device)

        return BatchFeature(data={"input_features": input_features}, tensor_type=return_tensors)

    def _extract_fbank_features(self, waveform: torch.Tensor, device: str = "cpu") -> torch.Tensor:
        """
        Compute the log-mel spectrogram of the audio using PyTorch's GPU-accelerated STFT implementation.
        """
        window = torch.hann_window(self.n_fft, device=device)

        if self.dither != 0.0:
            waveform = waveform + self.dither * torch.randn(waveform.shape, dtype=waveform.dtype, device=waveform.device)

        stft = torch.stft(waveform, self.n_fft, self.hop_length, window=window, return_complex=True)
        magnitudes = stft[..., :-1].abs() ** 2

        mel_filters = torch.from_numpy(self.mel_filters).to(device, torch.float32)
        mel_spec = mel_filters.T @ magnitudes

        log_spec = torch.clamp(mel_spec, min=1e-10).log10()
        if waveform.dim() == 2:
            max_val = log_spec.max(dim=2, keepdim=True)[0].max(dim=1, keepdim=True)[0]
            log_spec = torch.maximum(log_spec, max_val - 8.0)
        else:
            log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
        log_spec = (log_spec + 4.0) / 4.0

        if device != "cpu":
            log_spec = log_spec.detach().cpu()

        return log_spec


__all__ = ["WhisperAudioProcessor"]
