# Copyright 2026 the HuggingFace Team. All rights reserved.
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
"""Feature extractor class for CLAP2."""

import math

import numpy as np

from ...audio_utils import AudioInput, make_list_of_audio, mel_filter_bank
from ...feature_extraction_sequence_utils import SequenceFeatureExtractor
from ...feature_extraction_utils import BatchFeature
from ...processing_utils import AudioKwargs
from ...utils import TensorType, is_torch_available, logging
from ...utils.import_utils import requires


if is_torch_available():
    import torch


logger = logging.get_logger(__name__)


class Clap2AudioProcessorKwargs(AudioKwargs, total=False):
    """
    max_num_patches (`int`, *optional*):
        Longer clips keep only their first time columns of patches that fit in `max_num_patches`. Defaults to the
        `max_num_patches` of the feature extractor.
    """

    max_num_patches: int | None


@requires(backends=("torch",))
class Clap2FeatureExtractor(SequenceFeatureExtractor):
    r"""
    Constructs a CLAP2 feature extractor.

    Computes the log-mel spectrogram (in dB) of each mono clip, cuts it into `patch_size` x `patch_size` patches,
    frequency-major, and pads the patch sequences to the longest clip of the batch. The spectrogram follows
    `torchaudio.transforms.MelSpectrogram` (HTK mel scale, unnormalized filters, power 2) and
    `torchaudio.transforms.AmplitudeToDB`, which the model was trained with.

    This feature extractor inherits from [`~feature_extraction_sequence_utils.SequenceFeatureExtractor`] which contains
    most of the main methods. Users should refer to this superclass for more information regarding those methods.

    Args:
        feature_size (`int`, *optional*, defaults to 128):
            The number of mel bins.
        sampling_rate (`int`, *optional*, defaults to 32000):
            The sampling rate at which the audio files should be digitalized expressed in hertz (Hz).
        hop_length (`int`, *optional*, defaults to 320):
            The number of samples between successive STFT frames.
        n_fft (`int`, *optional*, defaults to 1024):
            The size of the Fourier transform and of the Hann window.
        f_min (`float`, *optional*, defaults to 50.0):
            The lowest frequency of the mel filters, in Hz.
        f_max (`float`, *optional*, defaults to 16000.0):
            The highest frequency of the mel filters, in Hz.
        patch_size (`int`, *optional*, defaults to 16):
            The number of mel bins and of frames per patch.
        max_num_patches (`int`, *optional*):
            If set, longer clips keep only their first time columns of patches that fit in `max_num_patches`, which
            must hold at least one column (`feature_size // patch_size` patches).
        padding_value (`float`, *optional*, defaults to 0.0):
            The value of the padding patches.
    """

    model_input_names = ["input_features", "input_features_mask"]
    valid_kwargs = Clap2AudioProcessorKwargs

    def __init__(
        self,
        feature_size: int = 128,
        sampling_rate: int = 32000,
        hop_length: int = 320,
        n_fft: int = 1024,
        f_min: float = 50.0,
        f_max: float = 16000.0,
        patch_size: int = 16,
        max_num_patches: int | None = None,
        padding_value: float = 0.0,
        **kwargs,
    ):
        super().__init__(feature_size=feature_size, sampling_rate=sampling_rate, padding_value=padding_value, **kwargs)
        self.hop_length = hop_length
        self.n_fft = n_fft
        self.f_min = f_min
        self.f_max = f_max
        self.patch_size = patch_size
        self.max_num_patches = max_num_patches
        self.window = torch.hann_window(n_fft)
        mel_filters = mel_filter_bank(
            num_frequency_bins=n_fft // 2 + 1,
            num_mel_filters=feature_size,
            min_frequency=f_min,
            max_frequency=f_max,
            sampling_rate=sampling_rate,
            norm=None,
            mel_scale="htk",
        )
        self.mel_filters = torch.from_numpy(mel_filters).float()

    def _extract_patches(
        self, waveform: np.ndarray, max_num_patches: int | None, device: str | None
    ) -> "torch.Tensor":
        waveform = torch.from_numpy(waveform).to(device)
        waveform = torch.nn.functional.pad(waveform, (0, max(0, self.n_fft - waveform.shape[0])))
        window = self.window.to(device)
        stft = torch.stft(
            waveform, self.n_fft, self.hop_length, window=window, pad_mode="reflect", return_complex=True
        )
        log_mel = 10.0 * torch.clamp(stft.abs().pow(2).T @ self.mel_filters.to(device), min=1e-10).log10()

        num_freq_patches = self.feature_size // self.patch_size
        if max_num_patches is not None:
            log_mel = log_mel[: max_num_patches // num_freq_patches * self.patch_size]
        num_time_patches = max(1, math.ceil(log_mel.shape[0] / self.patch_size))
        # the last time patch is completed with the quietest value of the clip
        num_fill_frames = num_time_patches * self.patch_size - log_mel.shape[0]
        log_mel = torch.cat([log_mel, log_mel.new_full((num_fill_frames, self.feature_size), log_mel.min().item())])
        patches = log_mel.reshape(num_time_patches, self.patch_size, num_freq_patches, self.patch_size)
        return patches.permute(2, 0, 3, 1).reshape(num_freq_patches * num_time_patches, self.patch_size**2).cpu()

    def __call__(
        self,
        raw_audio: AudioInput,
        sampling_rate: int | None = None,
        max_num_patches: int | None = None,
        return_tensors: str | TensorType | None = None,
        device: str | None = "cpu",
        **kwargs,
    ) -> BatchFeature:
        """
        Main method to featurize one or several mono clips.

        Args:
            raw_audio (`np.ndarray`, `torch.Tensor`, `list[float]`, `list[np.ndarray]`, `list[torch.Tensor]`):
                The clip or batch of clips. Each clip is a 1D array of mono samples at `sampling_rate`.
            sampling_rate (`int`, *optional*):
                The sampling rate of `raw_audio`. It is strongly recommended to pass it, to catch clips at the wrong
                rate: resample to the rate of the feature extractor first.
            max_num_patches (`int`, *optional*):
                Overrides `self.max_num_patches` for this call.
            return_tensors (`str` or [`~utils.TensorType`], *optional*):
                If set, will return tensors of this type instead of `torch.Tensor` objects.
            device (`str`, *optional*, defaults to `"cpu"`):
                The device to compute the log-mel spectrogram on (e.g. `"cpu"`, `"cuda"`). The features are returned
                on CPU.
        """
        if sampling_rate is None:
            logger.warning_once(
                f"It is strongly recommended to pass the `sampling_rate` argument to `{self.__class__.__name__}()`. "
                "Failing to do so can result in silent errors that might be hard to debug."
            )
        elif sampling_rate != self.sampling_rate:
            raise ValueError(
                f"The model corresponding to this feature extractor: {self.__class__.__name__} was trained using a"
                f" sampling rate of {self.sampling_rate}. Please make sure that the provided `raw_audio` input"
                f" was sampled with {self.sampling_rate} and not {sampling_rate}."
            )
        raw_audio = [np.asarray(audio) for audio in make_list_of_audio(raw_audio)]
        if any(not np.issubdtype(audio.dtype, np.floating) for audio in raw_audio):
            raise ValueError(
                f"{self.__class__.__name__} expects floating point audio in [-1, 1], but got integer samples."
                " Convert them first, e.g. divide int16 samples by 32768."
            )
        raw_audio = [audio.astype(np.float32) for audio in raw_audio]
        if any(audio.ndim != 1 for audio in raw_audio):
            raise ValueError(f"Only mono-channel audio is supported for input to {self.__class__.__name__}")

        max_num_patches = max_num_patches if max_num_patches is not None else self.max_num_patches
        num_freq_patches = self.feature_size // self.patch_size
        if max_num_patches is not None and max_num_patches < num_freq_patches:
            raise ValueError(f"`max_num_patches` must be at least {num_freq_patches}, one time column of patches.")
        patches = [self._extract_patches(audio, max_num_patches, device) for audio in raw_audio]
        num_patches = max(len(clip_patches) for clip_patches in patches)
        input_features = torch.full((len(patches), num_patches, self.patch_size**2), float(self.padding_value))
        input_features_mask = torch.zeros(len(patches), num_patches, dtype=torch.long)
        for i, clip_patches in enumerate(patches):
            input_features[i, : len(clip_patches)] = clip_patches
            input_features_mask[i, : len(clip_patches)] = 1
        return BatchFeature(
            {"input_features": input_features, "input_features_mask": input_features_mask}, tensor_type=return_tensors
        )


__all__ = ["Clap2FeatureExtractor"]
