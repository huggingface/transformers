# Copyright 2026 Alibaba DAMO Academy and the HuggingFace Inc. team. All rights reserved.
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

from ...audio_utils import AudioInput, make_list_of_audio
from ...feature_extraction_sequence_utils import SequenceFeatureExtractor
from ...feature_extraction_utils import BatchFeature
from ...utils import TensorType, is_torchaudio_available, logging
from ...utils.import_utils import is_torch_available, requires


if is_torch_available():
    import torch

if is_torchaudio_available():
    import torchaudio


logger = logging.get_logger(__name__)


@requires(backends=("torch", "torchaudio"))
class FunAsrNanoFeatureExtractor(SequenceFeatureExtractor):
    r"""
    Constructs a Fun-ASR-Nano feature extractor.

    This feature extractor inherits from [`~feature_extraction_sequence_utils.SequenceFeatureExtractor`] which contains
    most of the main methods. Users should refer to this superclass for more information regarding those methods.

    It extracts Kaldi-compatible mel-filterbank features (via `torchaudio.compliance.kaldi.fbank`, matching the
    original FunASR front-end) and then applies Low Frame Rate (LFR) processing by stacking `num_frames_lfr`
    consecutive frames and subsampling with stride `stride_lfr`.

    Args:
        feature_size (`int`, *optional*, defaults to 80):
            Number of mel frequency bins.
        sampling_rate (`int`, *optional*, defaults to 16000):
            Sampling rate of the audio.
        frame_length (`int`, *optional*, defaults to 25):
            Frame length in milliseconds for the STFT.
        frame_shift (`int`, *optional*, defaults to 10):
            Frame shift (hop length) in milliseconds for the STFT.
        num_frames_lfr (`int`, *optional*, defaults to 7):
            Number of consecutive frames to stack (LFR stacking factor).
        stride_lfr (`int`, *optional*, defaults to 6):
            Subsampling stride for LFR (take every `stride_lfr`-th stacked frame).
        window (`str`, *optional*, defaults to `"hamming"`):
            Window function for the STFT.
        padding_value (`float`, *optional*, defaults to 0.0):
            Value used for padding shorter sequences.
        return_attention_mask (`bool`, *optional*, defaults to `True`):
            Whether to return an attention mask.

    Example:

    ```python
    >>> import numpy as np
    >>> from transformers import FunAsrNanoFeatureExtractor

    >>> feature_extractor = FunAsrNanoFeatureExtractor()
    >>> audio = np.random.randn(16000)  # 1 second of audio at 16kHz
    >>> features = feature_extractor(audio, sampling_rate=16000, return_tensors="pt")
    >>> features.input_features.shape  # (1, num_frames_after_lfr, 560)
    ```
    """

    model_input_names = ["input_features", "input_features_mask"]

    def __init__(
        self,
        feature_size: int = 80,
        sampling_rate: int = 16000,
        frame_length: int = 25,
        frame_shift: int = 10,
        num_frames_lfr: int = 7,
        stride_lfr: int = 6,
        window: str = "hamming",
        padding_value: float = 0.0,
        return_attention_mask: bool = True,
        **kwargs,
    ):
        super().__init__(
            feature_size=feature_size,
            sampling_rate=sampling_rate,
            padding_value=padding_value,
            return_attention_mask=return_attention_mask,
            **kwargs,
        )
        self.frame_length = frame_length
        self.frame_shift = frame_shift
        self.num_frames_lfr = num_frames_lfr
        self.stride_lfr = stride_lfr
        self.window = window

    def _extract_fbank_features(self, waveform: "torch.Tensor") -> "torch.Tensor":
        """Extract Kaldi mel-filterbank features `(num_frames, feature_size)` from a single 1D waveform tensor."""
        fbank = torchaudio.compliance.kaldi.fbank(
            waveform.unsqueeze(0),
            num_mel_bins=self.feature_size,
            frame_length=self.frame_length,
            frame_shift=self.frame_shift,
            sample_frequency=self.sampling_rate,
            window_type=self.window,
        )
        return fbank

    def _apply_lfr(self, features: "torch.Tensor") -> "torch.Tensor":
        """Apply Low Frame Rate (LFR) by stacking `num_frames_lfr` frames and subsampling with stride `stride_lfr`.

        `features` has shape `(num_frames, feature_size)`; the output has shape
        `(num_lfr_frames, feature_size * num_frames_lfr)`.
        """
        num_input_frames = features.shape[0]
        num_frames_lfr = self.num_frames_lfr
        stride_lfr = self.stride_lfr

        # Repeat the first and last frames so each stacked window is centered.
        left_pad = (num_frames_lfr - 1) // 2
        right_pad = num_frames_lfr - 1 - left_pad
        padded = torch.cat(
            [features[0:1].expand(left_pad, -1), features, features[-1:].expand(right_pad, -1)],
            dim=0,
        )

        # Repeat the last frame so that the final window is complete, then slide over all windows at once.
        num_output_frames = (num_input_frames + stride_lfr - 1) // stride_lfr
        required_length = (num_output_frames - 1) * stride_lfr + num_frames_lfr
        if required_length > padded.shape[0]:
            padded = torch.cat([padded, padded[-1:].expand(required_length - padded.shape[0], -1)], dim=0)

        # `unfold` yields `(num_output_frames, feature_size, num_frames_lfr)`, so the stacked frames are moved
        # back in front of the feature dimension before flattening each window.
        windows = padded.unfold(0, num_frames_lfr, stride_lfr).transpose(1, 2)
        return windows.reshape(num_output_frames, num_frames_lfr * features.shape[1])

    def __call__(
        self,
        raw_speech: AudioInput,
        sampling_rate: int | None = None,
        return_tensors: str | TensorType | None = "pt",
        padding: bool | str = True,
        max_length: int | None = None,
        truncation: bool = False,
        pad_to_multiple_of: int | None = None,
        return_attention_mask: bool | None = None,
        **kwargs,
    ) -> BatchFeature:
        """
        Extract Kaldi mel-filterbank + LFR features from one or several raw audio waveforms.

        Args:
            raw_speech (`np.ndarray`, `list[float]`, `list[np.ndarray]`, `list[list[float]]`):
                The sequence or batch of mono waveforms to featurize.
            sampling_rate (`int`, *optional*):
                Sampling rate of the input audio. Must match `self.sampling_rate` (16000 Hz).
            return_tensors (`str` or [`~utils.TensorType`], *optional*):
                If set, returns tensors of the given framework (`"pt"`, `"np"`, ...).
            padding (`bool` or `str`, *optional*, defaults to `True`):
                Padding strategy forwarded to [`~feature_extraction_sequence_utils.SequenceFeatureExtractor.pad`].
            max_length (`int`, *optional*):
                Maximum LFR sequence length for padding/truncation.
            truncation (`bool`, *optional*, defaults to `False`):
                Whether to truncate sequences longer than `max_length`.
            pad_to_multiple_of (`int`, *optional*):
                If set, pads the LFR sequence length to a multiple of this value.
            return_attention_mask (`bool`, *optional*):
                Whether to return `input_features_mask`. Defaults to `self.return_attention_mask`.

        Returns:
            [`BatchFeature`] with `input_features` of shape
            `(batch, max_lfr_frames, feature_size * num_frames_lfr)` and the frame-level `input_features_mask`.
        """
        if sampling_rate is not None and sampling_rate != self.sampling_rate:
            raise ValueError(
                f"Expected sampling rate {self.sampling_rate}, got {sampling_rate}. "
                f"Please resample your audio to {self.sampling_rate} Hz."
            )

        raw_speech = make_list_of_audio(raw_speech)

        # Kaldi fbank + LFR are computed per sample (variable length), then padded jointly via `self.pad`.
        lfr_features = []
        for waveform in raw_speech:
            waveform = torch.as_tensor(waveform, dtype=torch.float32)
            fbank = self._extract_fbank_features(waveform)
            lfr_features.append(self._apply_lfr(fbank))

        batched = BatchFeature({"input_features": lfr_features})
        padded_inputs = self.pad(
            batched,
            padding=padding,
            max_length=max_length,
            truncation=truncation,
            pad_to_multiple_of=pad_to_multiple_of,
            return_attention_mask=return_attention_mask,
            return_tensors=return_tensors,
        )

        if "attention_mask" in padded_inputs:
            padded_inputs["input_features_mask"] = padded_inputs.pop("attention_mask")

        return padded_inputs


__all__ = ["FunAsrNanoFeatureExtractor"]
