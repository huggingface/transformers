# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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

from __future__ import annotations

import numpy as np

from ...feature_extraction_sequence_utils import SequenceFeatureExtractor
from ...feature_extraction_utils import BatchFeature
from ...utils import TensorType, is_librosa_available, is_torch_available, logging
from ...utils.import_utils import requires


if is_torch_available():
    import torch

if is_librosa_available():
    import librosa


EPSILON = 1e-5
LOG_ZERO_GUARD_VALUE = 2**-24


logger = logging.get_logger(__name__)


@requires(backends=("torch", "librosa"))
class Lfm2AudioFeatureExtractor(SequenceFeatureExtractor):
    r"""
    Constructs a Lfm2Audio feature extractor.

    This feature extractor inherits from [`~feature_extraction_sequence_utils.SequenceFeatureExtractor`] which contains
    most of the main methods. Users should refer to this superclass for more information regarding those methods.

    This class extracts normalized log-mel features with PyTorch. The attention mask includes the terminal
    centered-STFT frame used by LFM2-Audio, whose feature values are zeroed after normalization.

    Args:
        feature_size (`int`, *optional*, defaults to 128):
            The feature dimension of the extracted features.
        sampling_rate (`int`, *optional*, defaults to 16000):
            The sampling rate at which the audio files should be digitalized expressed in hertz (Hz).
        hop_length (`int`, *optional*, defaults to 160):
            Length of the overlapping windows for the STFT used to obtain the Mel Frequency coefficients.
        n_fft (`int`, *optional*, defaults to 512):
            Size of the Fourier transform.
        win_length (`int`, *optional*, defaults to 400):
            The window length for the STFT computation.
        preemphasis (`float`, *optional*, defaults to 0.97):
            A preemphasis filter coefficient. 0.0 means no preemphasis filter.
        padding_value (`float`, *optional*, defaults to 0.0):
            Padding value used to pad the audio. Should correspond to silences.
    """

    model_input_names = ["input_features", "attention_mask"]

    def __init__(
        self,
        feature_size=128,
        sampling_rate=16000,
        hop_length=160,
        n_fft=512,
        win_length=400,
        preemphasis=0.97,
        padding_value=0.0,
        **kwargs,
    ):
        super().__init__(feature_size=feature_size, sampling_rate=sampling_rate, padding_value=padding_value, **kwargs)

        self.hop_length = hop_length
        self.n_fft = n_fft
        self.win_length = win_length
        self.preemphasis = preemphasis
        self.window = torch.hann_window(self.win_length, periodic=False)

        # Use librosa's float32 filters to match the original frontend numerically.
        mel_filters = librosa.filters.mel(
            sr=sampling_rate, n_fft=n_fft, n_mels=feature_size, fmin=0.0, fmax=sampling_rate / 2, norm="slaney"
        )
        self.mel_filters = torch.from_numpy(mel_filters).to(torch.float32)

    def _torch_extract_fbank_features(self, waveform, device="cpu"):
        # Keep the static frontend tensors on the last-used device instead of copying them for every audio sample.
        device = waveform.device
        window = self.window
        if window.device != device:
            window = window.to(device)
            self.window = window
        mel_filters = self.mel_filters
        if mel_filters.device != device:
            mel_filters = mel_filters.to(device)
            self.mel_filters = mel_filters

        # spectrogram
        stft = torch.stft(
            waveform,
            self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=window,
            return_complex=True,
            pad_mode="constant",
        )
        # Preserve the original frontend's magnitude computation and rounding.
        magnitudes = torch.view_as_real(stft)
        magnitudes = torch.sqrt(magnitudes.pow(2).sum(-1))
        magnitudes = magnitudes.pow(2)

        # log mel spectrogram
        mel_spec = mel_filters @ magnitudes
        mel_spec = torch.log(mel_spec + LOG_ZERO_GUARD_VALUE)

        # (batch_size, num_mel_filters, num_frames) -> (batch_size, num_frames, num_mel_filters)
        mel_spec = mel_spec.permute(0, 2, 1)

        return mel_spec

    def __call__(
        self,
        raw_speech: np.ndarray | list[float] | list[np.ndarray] | list[list[float]],
        truncation: bool = False,
        pad_to_multiple_of: int | None = None,
        return_tensors: str | TensorType | None = None,
        return_attention_mask: bool | None = None,
        padding: str | bool | None = "longest",
        max_length: int | None = None,
        sampling_rate: int | None = None,
        device: str | torch.device | None = "cpu",
        **kwargs,
    ) -> BatchFeature:
        """
        Prepare one or more mono waveforms using PyTorch STFT and per-feature normalization.

        Args:
            raw_speech (`np.ndarray`, `list[float]`, `list[np.ndarray]`, `list[list[float]]`):
                The sequence or batch of sequences to be padded. Each sequence can be a numpy array, a list of float
                values, a list of numpy arrays or a list of list of float values. Must be mono channel audio, not
                stereo, i.e. single float per timestep.
            truncation (`bool`, *optional*, defaults to `False`):
                Activates truncation to cut input sequences longer than *max_length* to *max_length*.
            pad_to_multiple_of (`int`, *optional*, defaults to None):
                If set will pad the sequence to a multiple of the provided value.

                This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability
                `>= 7.5` (Volta), or on TPUs which benefit from having sequence lengths be a multiple of 128.
            return_attention_mask (`bool`, *optional*):
                Whether to return the attention mask. If left to the default, will return the attention mask according
                to the specific feature_extractor's default.

                [What are attention masks?](../glossary#attention-mask)

                <Tip>

                For Lfm2Audio models, `attention_mask` should always be passed for batched inference, to avoid subtle
                bugs.

                </Tip>

            return_tensors (`str` or [`~utils.TensorType`], *optional*):
                If set, will return tensors instead of list of python integers. Acceptable values are:

                - `'pt'`: Return PyTorch `torch.Tensor` objects.
                - `'np'`: Return Numpy `np.ndarray` objects.
            sampling_rate (`int`, *optional*):
                The sampling rate at which the `raw_speech` input was sampled. It is strongly recommended to pass
                `sampling_rate` at the forward call to prevent silent errors and allow automatic speech recognition
                pipeline.
            padding_value (`float`, *optional*, defaults to 0.0):
                The value that is used to fill the padding values / vectors.
            device (`str`, *optional*, defaults to `'cpu'`):
                Specifies the device for computation of the log-mel spectrogram of audio signals in the
                `_torch_extract_fbank_features` method. (e.g., "cpu", "cuda")
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

        if isinstance(raw_speech, (list, tuple)):
            if not raw_speech:
                raise ValueError("Audio must contain at least one sample.")
            if np.isscalar(raw_speech[0]):
                raw_speech = [raw_speech]
        elif isinstance(raw_speech, (np.ndarray, torch.Tensor)):
            raw_speech = [raw_speech] if raw_speech.ndim == 1 else list(raw_speech)
        else:
            raise ValueError("Expected a waveform or a batch of waveforms.")
        waveforms = []
        for speech in raw_speech:
            if isinstance(speech, np.ndarray):
                speech = np.ascontiguousarray(speech)
            speech = torch.as_tensor(speech, dtype=torch.float32)
            if speech.ndim != 1:
                raise ValueError("Only mono audio waveforms are supported.")
            if speech.numel() < 2 * self.hop_length:
                raise ValueError("Audio must contain at least two analysis frames for normalization.")
            waveforms.append(speech[:, None])
        raw_speech = waveforms

        audio_lengths = [len(speech) for speech in raw_speech]
        device = torch.device("cpu" if device is None else device)
        use_fast_padding = (
            (padding is True or padding == "longest")
            and not truncation
            and max_length is None
            and pad_to_multiple_of is None
            and self.padding_side == "right"
        )
        if use_fast_padding:
            waveforms = [speech.squeeze(-1).to(device) for speech in raw_speech]
            input_features = (
                waveforms[0].unsqueeze(0)
                if len(waveforms) == 1
                else torch.nn.utils.rnn.pad_sequence(waveforms, batch_first=True, padding_value=self.padding_value)
            )
            audio_lengths = torch.tensor(audio_lengths, dtype=torch.long, device=device)
        else:
            batched_speech = BatchFeature({"input_features": raw_speech, "audio_lengths": audio_lengths})
            padded_inputs = self.pad(
                batched_speech,
                padding=padding,
                max_length=max_length,
                truncation=truncation,
                pad_to_multiple_of=pad_to_multiple_of,
                return_tensors="pt",
            )
            input_features = padded_inputs.input_features.squeeze(-1).to(device)
            audio_lengths = padded_inputs.audio_lengths.to(device)

        audio_lengths = audio_lengths.clamp_max(input_features.shape[1])
        if torch.any(audio_lengths < 2 * self.hop_length):
            raise ValueError("Audio must contain at least two analysis frames after truncation.")

        # preemphasis
        if self.preemphasis is not None:
            timemask = torch.arange(input_features.shape[1], device=input_features.device).unsqueeze(
                0
            ) < audio_lengths.unsqueeze(1)
            input_features = torch.cat(
                [input_features[:, :1], input_features[:, 1:] - self.preemphasis * input_features[:, :-1]], dim=1
            )
            input_features = input_features.masked_fill(~timemask, 0.0)

        input_features = self._torch_extract_fbank_features(input_features, device)
        features_lengths = torch.floor_divide(audio_lengths + self.n_fft // 2 * 2 - self.n_fft, self.hop_length)
        attention_mask = torch.arange(input_features.shape[1], device=device)[None, :] < features_lengths[:, None]

        # normalize mel features, ignoring padding
        mask = attention_mask.unsqueeze(-1)
        input_features_masked = input_features * mask
        mean = input_features_masked.sum(dim=1) / features_lengths.unsqueeze(-1)
        mean = mean.unsqueeze(1)
        variance = ((input_features_masked - mean) ** 2 * mask).sum(dim=1) / (features_lengths - 1).unsqueeze(-1)
        std = torch.sqrt(variance).unsqueeze(1)
        input_features = (input_features - mean) / (std + EPSILON)
        input_features *= mask

        # Normalization excludes the terminal centered-STFT frame, but LFM's encoder consumes it.
        feature_lengths = (features_lengths + 1).clamp_max(input_features.shape[1])
        attention_mask = torch.arange(input_features.shape[1], device=device)[None, :] < feature_lengths[:, None]

        return BatchFeature(
            data={
                "input_features": input_features,
                "attention_mask": attention_mask,
            },
            tensor_type=return_tensors,
        )


__all__ = ["Lfm2AudioFeatureExtractor"]
