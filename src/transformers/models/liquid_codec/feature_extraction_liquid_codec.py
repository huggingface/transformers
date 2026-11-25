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
import torch.nn.functional as F

from ...audio_utils import AudioInput
from ...feature_extraction_sequence_utils import SequenceFeatureExtractor
from ...feature_extraction_utils import BatchFeature
from ...utils import PaddingStrategy, TensorType, logging
from ...utils.import_utils import requires


logger = logging.get_logger(__name__)


@requires(backends=("torch",))
class LiquidCodecFeatureExtractor(SequenceFeatureExtractor):
    r"""
    Constructs a Liquid Codec feature extractor.

    This feature extractor inherits from [`~feature_extraction_sequence_utils.SequenceFeatureExtractor`] which contains
    most of the main methods. Users should refer to this superclass for more information regarding those methods.

    Extracts the encoder's spectrogram with `torch.stft`: per frequency bin, log magnitude and phase (real and
    imaginary parts over the magnitude).

    Args:
        sampling_rate (`int`, *optional*, defaults to 24000):
            The sampling rate at which the audio files should be digitalized expressed in hertz (Hz).
        n_fft (`int`, *optional*, defaults to 1280):
            Size of the Fourier transform.
        hop_length (`int`, *optional*, defaults to 320):
            Length of the overlapping windows for the STFT.
        win_length (`int`, *optional*, defaults to 1280):
            The window length for the STFT computation.
        subsampling_factor (`int`, *optional*, defaults to 6):
            Spectrogram frames per codec frame; clips are zero-padded to whole codec frames.
        padding_value (`float`, *optional*, defaults to 0.0):
            Padding value used to pad the spectrograms of a batch.
    """

    model_input_names = ["input_features", "input_features_mask"]

    def __init__(
        self,
        sampling_rate=24000,
        n_fft=1280,
        hop_length=320,
        win_length=1280,
        subsampling_factor=6,
        padding_value=0.0,
        **kwargs,
    ):
        # derived from `n_fft`: a saved config carries it back
        kwargs.pop("feature_size", None)
        # log-magnitude, cos(phase) and sin(phase) per frequency bin
        feature_size = 3 * (n_fft // 2 + 1)
        super().__init__(feature_size=feature_size, sampling_rate=sampling_rate, padding_value=padding_value, **kwargs)
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.subsampling_factor = subsampling_factor

    def _torch_extract_fbank_features(self, waveform: torch.Tensor, device: str = "cpu") -> torch.Tensor:
        window = torch.hann_window(self.win_length, device=device)
        stft = torch.stft(
            waveform.to(device),
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=window,
            center=False,
            return_complex=True,
        )
        magnitude = stft.abs()
        phase_denominator = magnitude.clamp_min(1e-5)
        return torch.cat((torch.log1p(magnitude), stft.real / phase_denominator, stft.imag / phase_denominator), dim=1)

    def __call__(
        self,
        audio: AudioInput | list[float] | list[list[float]],
        padding: bool | str | PaddingStrategy = True,
        max_length: int | None = None,
        truncation: bool = False,
        return_tensors: str | TensorType | None = None,
        sampling_rate: int | None = None,
        device: str | None = "cpu",
        **kwargs,
    ) -> BatchFeature:
        """
        Main method to featurize and prepare for the model one or several sequence(s).

        Args:
            audio (`np.ndarray`, `list[float]`, `list[np.ndarray]`, `list[list[float]]`, `torch.Tensor`):
                The sequence or batch of sequences to featurize. Must be mono channel audio, not stereo, i.e. single
                float per timestep.
            padding (`bool`, `str` or [`~utils.PaddingStrategy`], *optional*, defaults to `True`):
                Select a strategy to pad the returned sequences among:

                - `True` or `'longest'`: Pad to the longest clip of the batch.
                - `'max_length'`: Pad to `max_length` samples.
                - `False` or `'do_not_pad'`: No padding: the clips must have the same number of codec frames.

                Whatever the strategy, each clip is first zero-padded to whole codec frames.
            max_length (`int`, *optional*):
                Length in samples to truncate / pad to, rounded up to whole codec frames.
            truncation (`bool`, *optional*, defaults to `False`):
                Whether to cut clips longer than `max_length`.
            return_tensors (`str` or [`~utils.TensorType`], *optional*):
                If set, will return tensors instead of list of python integers. Acceptable values are:
                - `'pt'`: Return PyTorch `torch.Tensor` objects.
                - `'np'`: Return Numpy `np.ndarray` objects.
            sampling_rate (`int`, *optional*):
                The sampling rate at which the `audio` input was sampled. It is strongly recommended to pass
                `sampling_rate` at the forward call to prevent silent errors.
            device (`str`, *optional*, defaults to `'cpu'`):
                Specifies the device for computation of the spectrogram of audio signals in the
                `_torch_extract_fbank_features` method. (e.g., "cpu", "cuda")

        Returns:
            [`BatchFeature`]: `input_features` of shape `(batch_size, feature_size, num_frames)`, right-padded to
            the longest clip, and `input_features_mask` of shape `(batch_size, num_frames)`.
        """
        if sampling_rate is not None:
            if sampling_rate != self.sampling_rate:
                raise ValueError(
                    f"The model corresponding to this feature extractor: {self.__class__.__name__} was trained using a"
                    f" sampling rate of {self.sampling_rate}. Please make sure that the provided `audio` input"
                    f" was sampled with {self.sampling_rate} and not {sampling_rate}."
                )
        else:
            logger.warning(
                f"It is strongly recommended to pass the `sampling_rate` argument to `{self.__class__.__name__}()`. "
                "Failing to do so can result in silent errors that might be hard to debug."
            )

        if isinstance(audio, np.ndarray):
            audio = torch.tensor(audio)
        elif isinstance(audio, (list, tuple)) and isinstance(audio[0], np.ndarray):
            audio = [torch.tensor(clip) for clip in audio]

        is_batched_torch = isinstance(audio, torch.Tensor) and len(audio.shape) > 1
        if is_batched_torch and len(audio.shape) > 2:
            logger.warning(
                f"Only mono-channel audio is supported for input to {self.__class__.__name__}. "
                "We will take the mean of the channels to convert to mono."
            )
            audio = audio.mean(-1)

        is_batched_sequence = isinstance(audio, (list, tuple)) and not isinstance(audio[0], (int, float))
        if is_batched_torch or is_batched_sequence:
            audio = [torch.as_tensor(clip, dtype=torch.float32) for clip in audio]
        else:
            audio = [torch.as_tensor(audio, dtype=torch.float32)]

        for clip in audio:
            if clip.ndim != 1 or len(clip) == 0:
                raise ValueError(
                    f"{self.__class__.__name__} expects non-empty mono clips, got one of shape {tuple(clip.shape)}"
                )

        padding_strategy = self._get_padding_strategies(padding=padding, max_length=max_length)
        if truncation:
            if max_length is None:
                raise ValueError("When setting `truncation=True`, make sure that `max_length` is defined.")
            audio = [clip[:max_length] for clip in audio]

        # per clip: zero-pad to whole codec frames, then reflect-pad its ends
        total_pad = self.win_length - self.hop_length
        clips = []
        frame_size = self.hop_length * self.subsampling_factor
        for clip in audio:
            clip = F.pad(clip, (0, -len(clip) % frame_size))
            clips.append(F.pad(clip[None], (total_pad // 2, total_pad - total_pad // 2), mode="reflect")[0])

        num_frames = torch.tensor([(len(clip) - self.win_length) // self.hop_length + 1 for clip in clips])
        if padding_strategy == PaddingStrategy.DO_NOT_PAD and len(set(num_frames.tolist())) > 1:
            raise ValueError(
                "Clips of different numbers of codec frames cannot be batched without padding: use `padding=True`."
            )

        waveform = torch.nn.utils.rnn.pad_sequence(clips, batch_first=True)
        if padding_strategy == PaddingStrategy.MAX_LENGTH:
            padded_length = max_length + (-max_length % frame_size) + total_pad
            waveform = F.pad(waveform, (0, max(0, padded_length - waveform.shape[-1])))

        input_features = self._torch_extract_fbank_features(waveform, device)
        input_features_mask = (
            torch.arange(input_features.shape[-1], device=device)[None, :] < num_frames.to(device)[:, None]
        )
        input_features = input_features.masked_fill(~input_features_mask[:, None, :], self.padding_value)

        return BatchFeature(
            data={"input_features": input_features, "input_features_mask": input_features_mask.long()},
            tensor_type=return_tensors,
        )


__all__ = ["LiquidCodecFeatureExtractor"]
