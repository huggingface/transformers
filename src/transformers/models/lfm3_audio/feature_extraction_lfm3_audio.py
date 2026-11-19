# Copyright 2026 The HuggingFace Team. All rights reserved.
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

import numpy as np
import torch

from ...audio_utils import mel_filter_bank
from ...feature_extraction_sequence_utils import SequenceFeatureExtractor
from ...feature_extraction_utils import BatchFeature
from ...utils import TensorType, logging


logger = logging.get_logger(__name__)


class Lfm3AudioFeatureExtractor(SequenceFeatureExtractor):
    r"""
    Constructs an LFM3-Audio feature extractor.

    This feature extractor inherits from [`~feature_extraction_sequence_utils.SequenceFeatureExtractor`] which contains
    most of the main methods. Users should refer to this superclass for more information regarding those methods.

    Log-mel spectrogram with `torch.stft`: Whisper's recipe without the dynamic range clamp. No centering (clips are
    right-padded to whole frames), `log10` clamped at `log_floor`, then
    `(log_mel + normalization_offset) / normalization_divisor`. A chunk is featurized exactly as the same samples of the
    whole clip, which live transcription relies on.

    Args:
        feature_size (`int`, *optional*, defaults to 128):
            The feature dimension of the extracted features: the number of mel filters.
        sampling_rate (`int`, *optional*, defaults to 16000):
            The sampling rate at which the audio files should be digitalized expressed in hertz (Hz).
        hop_length (`int`, *optional*, defaults to 160):
            Length of the overlapping windows for the STFT used to obtain the Mel Frequency coefficients.
        n_fft (`int`, *optional*, defaults to 400):
            Size of the Fourier transform.
        win_length (`int`, *optional*, defaults to 400):
            The window length for the STFT computation.
        log_floor (`float`, *optional*, defaults to -12.0):
            The lower clamp of the base-10 log-mel values.
        normalization_offset (`float`, *optional*, defaults to 4.0):
            The value added to the log-mel values before dividing by `normalization_divisor`.
        normalization_divisor (`float`, *optional*, defaults to 4.0):
            The divisor of the normalization.
        encoder_stride (`int`, *optional*, defaults to 8):
            The number of spectrogram frames per speech encoder token.
        padding_value (`float`, *optional*, defaults to 0.0):
            Padding value used to pad the spectrograms of a batch.
    """

    model_input_names = ["input_features", "input_features_mask"]

    def __init__(
        self,
        feature_size=128,
        sampling_rate=16000,
        hop_length=160,
        n_fft=400,
        win_length=400,
        log_floor=-12.0,
        normalization_offset=4.0,
        normalization_divisor=4.0,
        encoder_stride=8,
        padding_value=0.0,
        **kwargs,
    ):
        super().__init__(feature_size=feature_size, sampling_rate=sampling_rate, padding_value=padding_value, **kwargs)
        self.hop_length = hop_length
        self.n_fft = n_fft
        self.win_length = win_length
        self.log_floor = log_floor
        self.normalization_offset = normalization_offset
        self.normalization_divisor = normalization_divisor
        self.encoder_stride = encoder_stride

        mel_filters = mel_filter_bank(
            num_frequency_bins=n_fft // 2 + 1,
            num_mel_filters=feature_size,
            min_frequency=0.0,
            max_frequency=sampling_rate / 2,
            sampling_rate=sampling_rate,
            norm="slaney",
            mel_scale="slaney",
        )
        self.mel_filters = torch.from_numpy(mel_filters).to(torch.float32)

    def _torch_extract_fbank_features(self, waveform, device="cpu"):
        window = torch.hann_window(self.win_length, device=device)
        stft = torch.stft(
            waveform.to(device),
            self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=window,
            center=False,
            return_complex=True,
        )

        magnitudes = stft.abs() ** 2
        mel_spec = self.mel_filters.to(device).T @ magnitudes
        log_spec = mel_spec.log10().clamp(min=self.log_floor)
        return (log_spec + self.normalization_offset) / self.normalization_divisor

    def __call__(
        self,
        raw_speech: np.ndarray | list[float] | list[np.ndarray] | list[list[float]] | torch.Tensor,
        sampling_rate: int | None = None,
        return_tensors: str | TensorType | None = None,
        device: str | None = "cpu",
        **kwargs,
    ) -> BatchFeature:
        """
        Main method to featurize and prepare for the model one or several sequence(s).

        Args:
            raw_speech (`np.ndarray`, `list[float]`, `list[np.ndarray]`, `list[list[float]]`, `torch.Tensor`):
                The sequence or batch of sequences to featurize. Must be mono channel audio, not stereo, i.e. single
                float per timestep.
            sampling_rate (`int`, *optional*):
                The sampling rate at which the `raw_speech` input was sampled. It is strongly recommended to pass
                `sampling_rate` at the forward call to prevent silent errors.
            return_tensors (`str` or [`~utils.TensorType`], *optional*):
                If set, will return tensors instead of list of python integers. Acceptable values are:
                - `'pt'`: Return PyTorch `torch.Tensor` objects.
                - `'np'`: Return Numpy `np.ndarray` objects.
            device (`str`, *optional*, defaults to `'cpu'`):
                Specifies the device for computation of the log-mel spectrogram of audio signals in the
                `_torch_extract_fbank_features` method. (e.g., "cpu", "cuda")

        Returns:
            [`BatchFeature`]: `input_features` of shape `(batch_size, num_frames, feature_size)`, right-padded to the
            longest clip, and `input_features_mask` of shape `(batch_size, num_frames)`.
        """
        if sampling_rate is not None:
            if sampling_rate != self.sampling_rate:
                raise ValueError(
                    f"The model corresponding to this feature extractor: {self.__class__.__name__} was trained using a"
                    f" sampling rate of {self.sampling_rate}. Please make sure that the provided `raw_speech` input"
                    f" was sampled with {self.sampling_rate} and not {sampling_rate}."
                )
        else:
            logger.warning(
                f"It is strongly recommended to pass the `sampling_rate` argument to `{self.__class__.__name__}()`. "
                "Failing to do so can result in silent errors that might be hard to debug."
            )

        # Convert to torch tensor
        if isinstance(raw_speech, np.ndarray):
            raw_speech = torch.tensor(raw_speech)
        elif isinstance(raw_speech, (list, tuple)) and isinstance(raw_speech[0], np.ndarray):
            raw_speech = [torch.tensor(speech) for speech in raw_speech]

        is_batched_torch = isinstance(raw_speech, torch.Tensor) and len(raw_speech.shape) > 1
        if is_batched_torch and len(raw_speech.shape) > 2:
            logger.warning(
                f"Only mono-channel audio is supported for input to {self.__class__.__name__}. "
                "We will take the mean of the channels to convert to mono."
            )
            raw_speech = raw_speech.mean(-1)

        is_batched_sequence = isinstance(raw_speech, (list, tuple)) and not isinstance(raw_speech[0], (int, float))
        if is_batched_torch or is_batched_sequence:
            raw_speech = [torch.as_tensor(speech, dtype=torch.float32) for speech in raw_speech]
        else:
            raw_speech = [torch.as_tensor(raw_speech, dtype=torch.float32)]

        for speech in raw_speech:
            if speech.ndim != 1:
                raise ValueError(
                    f"{self.__class__.__name__} expects mono clips, got one of shape {tuple(speech.shape)}"
                )

        # right-pad each clip to whole frames, the batch to its longest clip
        audio_lengths = torch.tensor([len(speech) for speech in raw_speech])
        num_frames = 1 + (audio_lengths - self.n_fft).clamp(min=0).add(self.hop_length - 1) // self.hop_length
        num_samples = self.n_fft + (int(num_frames.max()) - 1) * self.hop_length
        input_values = torch.zeros(len(raw_speech), num_samples)
        for index, speech in enumerate(raw_speech):
            input_values[index, : len(speech)] = speech

        input_features = self._torch_extract_fbank_features(input_values, device).transpose(1, 2)
        input_features_mask = (
            torch.arange(input_features.shape[1], device=device)[None, :] < num_frames.to(device)[:, None]
        )
        input_features = input_features.masked_fill(~input_features_mask[:, :, None], self.padding_value)

        return BatchFeature(
            data={"input_features": input_features, "input_features_mask": input_features_mask.long()},
            tensor_type=return_tensors,
        )


__all__ = ["Lfm3AudioFeatureExtractor"]
