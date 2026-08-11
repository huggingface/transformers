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
import math

import numpy as np
import torch
import torch.nn.functional as F

from ...audio_utils import mel_filter_bank
from ...feature_extraction_sequence_utils import SequenceFeatureExtractor
from ...feature_extraction_utils import BatchFeature
from ...utils import PaddingStrategy, TensorType, logging
from ...utils.import_utils import requires


logger = logging.get_logger(__name__)


def _to_exact_int(value: float, name: str, tolerance: float = 1e-6) -> int:
    rounded = round(value)
    if abs(value - rounded) > tolerance:
        raise ValueError(f"{name} must resolve to an integer sample count, got {value}")
    return int(rounded)


@requires(backends=("torch",))
class InklingFeatureExtractor(SequenceFeatureExtractor):
    r"""
    Constructs a TML audio feature extractor, which converts raw audio waveforms into log-mel spectrogram
    features (mel filterbank energies in log10 space). The quantization of these features into discrete
    dMel bins is performed downstream by [`InklingProcessor`].

    This feature extractor inherits from [`~feature_extraction_sequence_utils.SequenceFeatureExtractor`]
    which contains most of the main methods. Users should refer to this superclass for more information
    regarding those methods.

    Args:
        feature_size (`int`, *optional*, defaults to 80):
            The feature dimension of the extracted features, i.e. the number of mel filterbanks.
        sampling_rate (`int`, *optional*, defaults to 16000):
            The sampling rate at which the audio files should be digitized, expressed in hertz (Hz).
        padding_value (`float`, *optional*, defaults to 0.0):
            The value used to pad the log-mel spectrograms to the same length in a batch.
        audio_token_duration_s (`float`, *optional*, defaults to 0.05):
            Duration, in seconds, represented by a single audio token, i.e. the STFT hop length.
        window_size_multiplier (`float`, *optional*, defaults to 2.0):
            Multiplier applied to `audio_token_duration_s` to obtain the STFT window length.
        n_fft (`int`, *optional*):
            FFT size. Defaults to the window length (`audio_token_duration_s * window_size_multiplier *
            sampling_rate`) when not provided.
    """

    model_input_names = ["input_features", "input_features_mask"]

    def __init__(
        self,
        feature_size: int = 80,
        sampling_rate: int = 16_000,
        padding_value: float = 0.0,
        audio_token_duration_s: float = 0.05,
        window_size_multiplier: float = 2.0,
        n_fft: int | None = None,
        **kwargs,
    ):
        super().__init__(
            feature_size=feature_size,
            sampling_rate=sampling_rate,
            padding_value=padding_value,
            **kwargs,
        )
        self.audio_token_duration_s = audio_token_duration_s
        self.window_size_multiplier = window_size_multiplier

        self.hop_length = _to_exact_int(
            audio_token_duration_s * sampling_rate, "audio_token_duration_s * sampling_rate"
        )
        self.window_size = _to_exact_int(
            audio_token_duration_s * window_size_multiplier * sampling_rate,
            "audio_token_duration_s * window_size_multiplier * sampling_rate",
        )
        self.n_fft = n_fft or self.window_size
        if self.hop_length <= 0 or self.window_size <= 0 or self.n_fft <= 0:
            raise ValueError("hop_length, window_size, and n_fft must all be positive")

        # Precomputed once at init, mirrors e.g. WhisperFeatureExtractor.mel_filters.
        self.window = torch.hann_window(self.window_size, periodic=True, dtype=torch.float32)
        # `mel_filter_bank` returns `(num_frequency_bins, feature_size)`; transpose to
        # `(feature_size, num_frequency_bins)` so it left-multiplies the magnitude spectrogram.
        mel_filters = mel_filter_bank(
            num_frequency_bins=self.n_fft // 2 + 1,
            num_mel_filters=feature_size,
            min_frequency=0.0,
            max_frequency=sampling_rate / 2.0,
            sampling_rate=sampling_rate,
            norm="slaney",
            mel_scale="slaney",
        )
        self.mel_filters = torch.from_numpy(np.ascontiguousarray(mel_filters.T, dtype=np.float32))

    def _torch_extract_fbank_features(self, waveform: torch.Tensor, device: str = "cpu") -> torch.Tensor:
        right_pad = math.ceil(waveform.shape[-1] / self.hop_length) * self.hop_length - waveform.shape[-1]
        left_pad = max(self.n_fft - self.hop_length, 0)
        waveform = F.pad(waveform, (left_pad, right_pad))

        stft = torch.stft(
            waveform,
            self.n_fft,
            hop_length=self.hop_length,
            win_length=self.window_size,
            window=self.window.to(device),
            center=False,
            return_complex=True,
        )
        magnitudes = torch.view_as_real(stft)
        magnitudes = magnitudes.pow(2).sum(-1).clamp_min(1e-10).sqrt()

        mel_filters = self.mel_filters.to(device)
        mel_spec = mel_filters @ magnitudes
        mel_spec = mel_spec.clamp_min(1e-10).log10()

        # (batch_size, feature_size, num_frames) -> (batch_size, num_frames, feature_size)
        return mel_spec.transpose(1, 2)

    def __call__(
        self,
        raw_speech: np.ndarray | list[float] | list[np.ndarray] | list[list[float]],
        sampling_rate: int | None = None,
        padding: bool | str | PaddingStrategy = True,
        max_length: int | None = None,
        truncation: bool = False,
        pad_to_multiple_of: int | None = None,
        return_attention_mask: bool | None = True,
        return_tensors: str | TensorType | None = None,
        device: str | None = "cpu",
        **kwargs,
    ) -> BatchFeature:
        """
        Extract log-mel spectrogram features from one or several audio clip(s).

        Args:
            raw_speech (`np.ndarray`, `list[float]`, `list[np.ndarray]`, `list[list[float]]`):
                The sequence or batch of sequences to be padded. Each sequence can be a numpy array, a list
                of float values, a list of numpy arrays or a list of list of float values. Must be mono
                channel audio at `self.sampling_rate`, not stereo, i.e. single float per timestep. Decoding
                and resampling of raw audio (bytes / paths / URLs) is handled upstream by the processor's
                `apply_chat_template`, not here.
            sampling_rate (`int`, *optional*):
                The sampling rate of `raw_speech`, used only to validate against `self.sampling_rate`.
            device (`str`, *optional*, defaults to `"cpu"`):
                The device on which the log-mel spectrogram is computed in `_torch_extract_fbank_features`.
        """
        if sampling_rate is not None:
            if sampling_rate != self.sampling_rate:
                raise ValueError(
                    f"The model corresponding to this feature extractor was trained using a sampling "
                    f"rate of {self.sampling_rate}. Please make sure that the provided audio input "
                    f"was sampled with {self.sampling_rate} and not {sampling_rate}."
                )
        else:
            logger.warning_once(
                "It is strongly recommended to pass the `sampling_rate` argument to this function. "
                "Failing to do so can result in silent errors that might be hard to debug."
            )

        cls_name = self.__class__.__name__

        def _to_mono(clip: "np.ndarray | torch.Tensor | list") -> torch.Tensor:
            tensor = clip if isinstance(clip, torch.Tensor) else torch.as_tensor(np.asarray(clip))
            tensor = tensor.to(torch.float32)
            if tensor.ndim == 2:
                logger.warning_once(
                    f"Only mono-channel audio is supported for input to {cls_name}. "
                    "Taking the mean over the channel (last) axis to convert to mono."
                )
                tensor = tensor.mean(dim=-1)
            elif tensor.ndim != 1:
                raise ValueError(
                    f"Each audio clip must be 1-D (mono) or 2-D (multichannel), got shape {tuple(tensor.shape)}."
                )
            return tensor

        if isinstance(raw_speech, np.ndarray):
            raw_speech = torch.from_numpy(raw_speech)
        if isinstance(raw_speech, torch.Tensor):
            # A single array is one clip: 1-D mono or 2-D multichannel (never a batch).
            if raw_speech.ndim > 2:
                raise ValueError(
                    f"A single array input must be 1-D (mono) or 2-D (multichannel); got {raw_speech.ndim} dims. "
                    "Pass a list of arrays for a batch of clips."
                )
            clips = [raw_speech]
        elif isinstance(raw_speech, (list, tuple)):
            if len(raw_speech) == 0:
                raise ValueError("Received an empty audio input.")
            # A flat list of scalars is a single mono clip; a list of arrays/lists is a batch of clips.
            if isinstance(raw_speech[0], (int, float, np.integer, np.floating)):
                clips = [raw_speech]
            else:
                clips = list(raw_speech)
        else:
            raise TypeError(f"Unsupported audio input type for {cls_name}: {type(raw_speech)}")

        raw_speech = [_to_mono(clip)[:, None] for clip in clips]

        # Stack and pad the raw waveforms to the longest clip in the batch, then extract the log-mel
        # spectrogram on the batched audio in a single `torch.stft` pass (mirrors Parakeet).
        audio_lengths = [len(speech) for speech in raw_speech]
        batched_speech = BatchFeature({"input_features": raw_speech, "audio_lengths": audio_lengths})
        padded_inputs = self.pad(
            batched_speech,
            padding=padding,
            max_length=max_length,
            truncation=truncation,
            pad_to_multiple_of=pad_to_multiple_of,
            return_tensors="pt",
        )
        input_waveforms = padded_inputs.input_features.squeeze(-1)  # (batch_size, num_samples)

        input_features = self._torch_extract_fbank_features(input_waveforms, device)  # (batch_size, T, feature_size)

        # Number of valid frames per clip == ceil(audio_length / hop_length); everything beyond it is
        # padding, which we zero out so it carries `padding_value`.
        num_frames = torch.div(
            padded_inputs.audio_lengths + self.hop_length - 1, self.hop_length, rounding_mode="floor"
        )
        input_features_mask = torch.arange(input_features.shape[1], device=device)[None, :] < num_frames[:, None]
        input_features = input_features * input_features_mask.unsqueeze(-1)

        data = {"input_features": input_features}
        if return_attention_mask:
            # Named `input_features_mask` (not `attention_mask`) so it does not collide with the text
            # `attention_mask` when the processor merges audio and text inputs.
            data["input_features_mask"] = input_features_mask
        return BatchFeature(data=data, tensor_type=return_tensors)


__all__ = ["InklingFeatureExtractor"]
