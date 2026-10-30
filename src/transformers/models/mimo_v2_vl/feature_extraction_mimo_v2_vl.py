# Copyright 2026 Xiaomi Corporation and the HuggingFace Inc. team. All rights reserved.
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
"""Audio feature extractor for MiMo V2 multimodal models."""

import math
from functools import lru_cache

import numpy as np

from ...audio_utils import AudioInput, make_list_of_audio, mel_filter_bank, spectrogram, window_function
from ...feature_extraction_sequence_utils import SequenceFeatureExtractor
from ...feature_extraction_utils import BatchFeature
from ...utils import TensorType, is_torch_available, is_torch_tensor


if is_torch_available():
    import torch


@lru_cache(maxsize=8)
def _get_mel_filters(n_fft: int, feature_size: int, f_min: float, f_max: float, sampling_rate: int) -> np.ndarray:
    return mel_filter_bank(
        num_frequency_bins=1 + n_fft // 2,
        num_mel_filters=feature_size,
        min_frequency=f_min,
        max_frequency=f_max,
        sampling_rate=sampling_rate,
        norm=None,
        mel_scale="htk",
    )


@lru_cache(maxsize=8)
def _get_spectral_constants(
    n_fft: int,
    win_length: int,
    feature_size: int,
    f_min: float,
    f_max: float,
    sampling_rate: int,
    device: str | None,
):
    """Cache the STFT window and mel filters. A `None` device selects the NumPy backend."""
    filters = _get_mel_filters(n_fft, feature_size, f_min, f_max, sampling_rate)
    if device is None:
        return window_function(win_length, "hann", frame_length=n_fft), filters
    return (
        torch.hann_window(win_length, dtype=torch.float32, device=device),
        torch.as_tensor(filters, dtype=torch.float32, device=device),
    )


class MiMoV2VLFeatureExtractor(SequenceFeatureExtractor):
    r"""
    Constructs an audio feature extractor for the MiMo V2 VL series.

    Extracts the natural logarithm of magnitude mel spectrograms from mono waveforms, or groups pre-tokenized
    audio codec IDs. Use `MiMoV2VLProcessor.from_pretrained("XiaomiMiMo/MiMo-V2.6-Flash-RL")` to load the
    checkpoint's audio settings together with its tokenizer and visual processors.

    This feature extractor inherits from [`SequenceFeatureExtractor`], which provides feature padding and serialization.

    Args:
        feature_size (`int`, *optional*, defaults to 128):
            Number of mel frequency bins.
        sampling_rate (`int`, *optional*, defaults to 24000):
            Waveform sampling rate in hertz.
        n_fft (`int`, *optional*, defaults to 960):
            Size of the Fourier transform.
        hop_length (`int`, *optional*, defaults to 240):
            Number of samples between consecutive STFT frames.
        win_length (`int`, *optional*, defaults to 960):
            Length of the periodic Hann window.
        f_min (`float`, *optional*, defaults to 0.0):
            Lowest mel frequency in hertz.
        f_max (`float`, *optional*):
            Highest mel frequency in hertz. Defaults to half the sampling rate.
        padding_value (`float`, *optional*, defaults to 0.0):
            Value used to pad mel features.
        audio_kernel_size (`int`, *optional*, defaults to 3):
            Kernel size of the audio encoder's convolutional layers.
        audio_stride_size (`int`, *optional*, defaults to 2):
            Stride of the audio encoder's downsampling convolution.
        audio_avg_pooler (`int`, *optional*, defaults to 2):
            Audio encoder pooling factor.
        audio_group_size (`int`, *optional*, defaults to 4):
            Number of consecutive codec frames represented by one audio token.
        audio_channels (`int`, *optional*, defaults to 8):
            Number of codec channels retained per frame. Loaded from the checkpoint when using the processor.
        audio_input_id_per_second (`float`, *optional*, defaults to 25):
            Number of codec frames per second before grouping.
        audio_segment_size (`int`, *optional*, defaults to 6000):
            Maximum mel frames per learned audio-tokenizer segment. Convolution and pooling lengths are computed
            per segment before the resulting codec frames are grouped.
    """

    model_input_names = ["input_features", "attention_mask", "audio_feature_lens"]

    def __init__(
        self,
        feature_size=128,
        sampling_rate=24000,
        n_fft=960,
        hop_length=240,
        win_length=960,
        f_min=0.0,
        f_max=None,
        padding_value=0.0,
        audio_kernel_size=3,
        audio_stride_size=2,
        audio_avg_pooler=2,
        audio_group_size=4,
        audio_channels=8,
        audio_input_id_per_second=25,
        audio_segment_size=6000,
        **kwargs,
    ):
        kwargs.pop("return_attention_mask", None)
        super().__init__(
            feature_size=feature_size,
            sampling_rate=sampling_rate,
            padding_value=padding_value,
            return_attention_mask=True,
            **kwargs,
        )
        if (
            min(
                feature_size,
                sampling_rate,
                n_fft,
                hop_length,
                win_length,
                audio_kernel_size,
                audio_stride_size,
                audio_avg_pooler,
                audio_group_size,
                audio_channels,
                audio_input_id_per_second,
                audio_segment_size,
            )
            <= 0
        ):
            raise ValueError("FFT, window, stride, pooling and grouping parameters must be positive.")
        if win_length > n_fft:
            raise ValueError("win_length must not exceed n_fft.")
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.f_min = f_min
        self.f_max = f_max
        self.audio_kernel_size = audio_kernel_size
        self.audio_stride_size = audio_stride_size
        self.audio_avg_pooler = audio_avg_pooler
        self.audio_group_size = audio_group_size
        self.audio_channels = audio_channels
        self.audio_input_id_per_second = audio_input_id_per_second
        self.audio_segment_size = audio_segment_size

    @property
    def audio_tokens_per_second(self) -> float:
        return self.audio_input_id_per_second / self.audio_group_size

    def _get_audio_token_length(self, mel_length: int) -> int:
        """Compute the token count after audio convolutions, pooling, and codec grouping."""

        def codec_length(length):
            output_length = length + 3 - self.audio_kernel_size
            output_length = (output_length + 2 - self.audio_kernel_size) // self.audio_stride_size + 1
            return math.ceil(output_length / self.audio_avg_pooler)

        full_segments, remainder = divmod(mel_length, self.audio_segment_size)
        output_length = full_segments * codec_length(self.audio_segment_size)
        if remainder:
            output_length += codec_length(remainder)
        return math.ceil(output_length / self.audio_group_size)

    def _torch_extract_fbank_features(self, waveform: np.ndarray, device: str) -> np.ndarray:
        """Compute a magnitude log-mel spectrogram using PyTorch's STFT."""
        if waveform.strides[0] < 0:
            waveform = np.ascontiguousarray(waveform)
        waveform = torch.as_tensor(waveform, dtype=torch.float32, device=device)
        f_max = self.f_max if self.f_max is not None else self.sampling_rate / 2
        # Include the actual device index in the cache key when callers use device="cuda".
        window, mel_filters = _get_spectral_constants(
            self.n_fft, self.win_length, self.feature_size, self.f_min, f_max, self.sampling_rate, str(waveform.device)
        )
        stft = torch.stft(
            waveform,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=window,
            return_complex=True,
        )
        mel_spectrogram = stft.abs().transpose(0, 1) @ mel_filters
        return mel_spectrogram.clamp(min=1e-7).log().cpu().numpy()

    def _np_extract_fbank_features(self, waveform: np.ndarray, device: str) -> np.ndarray:
        """Compute the same magnitude log-mel features using the NumPy STFT backend."""
        if device != "cpu":
            raise ValueError("Feature extraction on an accelerator requires PyTorch.")
        f_max = self.f_max if self.f_max is not None else self.sampling_rate / 2
        window, mel_filters = _get_spectral_constants(
            self.n_fft, self.win_length, self.feature_size, self.f_min, f_max, self.sampling_rate, None
        )
        return spectrogram(
            waveform,
            window,
            frame_length=self.n_fft,
            hop_length=self.hop_length,
            power=1.0,
            mel_filters=mel_filters,
            mel_floor=1e-7,
            log_mel="log",
        ).T

    def __call__(
        self,
        raw_speech: AudioInput,
        sampling_rate: int | None = None,
        padding: bool | str = True,
        max_length: int | None = None,
        truncation: bool = False,
        pad_to_multiple_of: int | None = None,
        return_attention_mask: bool = True,
        return_tensors: str | TensorType | None = None,
        device: str = "cpu",
        is_tokenized: bool = False,
        **kwargs,
    ) -> BatchFeature:
        """
        Prepare waveforms or pre-tokenized codec IDs for the MiMo V2 VL series.

        Args:
            raw_speech (`AudioInput`):
                A mono waveform or batch of waveforms. With `is_tokenized=True`, a sequence of integer codec IDs
                of shape `(time, codec_channels)`, a list or tuple of such sequences, or a three-dimensional batch.
            sampling_rate (`int`, *optional*):
                Sampling rate of the input waveforms. Must match the extractor's sampling rate.
            padding (`bool` or `str`, *optional*, defaults to `True`):
                Pad to the longest sequence when `True` or `"longest"`, or disable padding with `False` or
                `"do_not_pad"`. Waveforms also support `"max_length"` padding.
            max_length (`int`, *optional*):
                Maximum number of mel frames for waveform padding or truncation.
            truncation (`bool`, *optional*, defaults to `False`):
                Truncate waveform features to `max_length` mel frames.
            pad_to_multiple_of (`int`, *optional*):
                Pad waveform features to a multiple of this number of mel frames.
            return_attention_mask (`bool`, *optional*, defaults to `True`):
                Whether to return a mask distinguishing valid features from padding.
            return_tensors (`str` or [`~utils.TensorType`], *optional*):
                Return PyTorch tensors with `"pt"`, NumPy arrays with `"np"`, or lists when unset.
            device (`str`, *optional*, defaults to `"cpu"`):
                Device used for the PyTorch STFT. The resulting features are returned on the CPU.
            is_tokenized (`bool`, *optional*, defaults to `False`):
                Process codec IDs instead of waveforms. Incomplete groups repeat the final codec frame.

        Returns:
            [`BatchFeature`]: Waveform `input_features` of shape `(batch, mel_time, mel_bins)`, or `input_audio_codes`
            of shape `(batch, grouped_time, group_size, codec_channels)`. Also includes `audio_feature_lens` in
            grouped-token units and, when requested, `attention_mask` in feature-frame units.
        """
        if is_tokenized:
            if truncation or max_length is not None or pad_to_multiple_of is not None:
                raise ValueError("Length overrides are only supported for waveform features.")
            output = self._process_audio_codes(raw_speech, padding=padding, return_tensors=return_tensors)
            if not return_attention_mask:
                output.pop("attention_mask")
            return output
        if sampling_rate is not None and sampling_rate != self.sampling_rate:
            raise ValueError(f"Expected sampling_rate={self.sampling_rate}; resample the waveform before calling.")
        waveforms = make_list_of_audio(raw_speech)
        if not waveforms:
            raise ValueError("Audio batch must contain at least one waveform.")
        extract_fbank_features = (
            self._torch_extract_fbank_features if is_torch_available() else self._np_extract_fbank_features
        )
        features = []
        for waveform in waveforms:
            if is_torch_tensor(waveform):
                waveform = waveform.detach().cpu().numpy()
            waveform = np.asarray(waveform, dtype=np.float32)
            if waveform.ndim != 1:
                raise ValueError("Each waveform must be mono and one-dimensional.")
            if not np.isfinite(waveform).all():
                raise ValueError("Waveform samples must be finite.")
            if len(waveform) <= self.n_fft // 2:
                raise ValueError("Waveform is too short for centered STFT reflection padding.")
            features.append(extract_fbank_features(waveform, device))
        output = self.pad(
            BatchFeature({"input_features": features}),
            padding=padding,
            max_length=max_length,
            truncation=truncation,
            pad_to_multiple_of=pad_to_multiple_of,
            return_attention_mask=True,
            return_tensors=return_tensors,
        )
        mel_lengths = [int(sum(mask) if isinstance(mask, list) else mask.sum()) for mask in output["attention_mask"]]
        output["audio_feature_lens"] = [self._get_audio_token_length(length) for length in mel_lengths]
        if not return_attention_mask:
            output.pop("attention_mask")
        return BatchFeature(dict(output), tensor_type=return_tensors)

    def _process_audio_codes(
        self, audio: AudioInput, padding: bool | str, return_tensors: str | TensorType | None
    ) -> BatchFeature:
        """Group codec frames and pad the batch in grouped-token units."""
        if padding not in (True, False, "longest", "do_not_pad"):
            raise ValueError("Codec inputs support padding=True, False, 'longest', or 'do_not_pad'.")
        padding = padding in (True, "longest")
        if isinstance(audio, np.ndarray) or is_torch_tensor(audio):
            audio = list(audio) if audio.ndim == 3 else [audio]
        elif isinstance(audio, (list, tuple)):
            if audio:
                # A sequence of codec rows is one sample; a sequence of matrices is a batch.
                first = audio[0]
                ndim = first.ndim if isinstance(first, np.ndarray) or is_torch_tensor(first) else np.ndim(first)
                if ndim == 1:
                    audio = [audio]
        else:
            audio = [audio]
        if not audio:
            raise ValueError("Audio batch must contain at least one codec sequence.")
        features, lengths = [], []
        for codes in audio:
            if is_torch_tensor(codes):
                codes = codes.detach().cpu().numpy()
            codes = np.asarray(codes)
            if (
                codes.ndim != 2
                or not len(codes)
                or codes.shape[1] < self.audio_channels
                or not np.issubdtype(codes.dtype, np.integer)
            ):
                raise ValueError("Audio codes must be nonempty integer (time, codec_channels) arrays.")
            codes = codes[:, : self.audio_channels].astype(np.int64, copy=False)
            if pad_length := -len(codes) % self.audio_group_size:
                codes = np.pad(codes, ((0, pad_length), (0, 0)), mode="edge")
            features.append(codes.reshape(-1, self.audio_group_size, self.audio_channels))
            lengths.append(len(features[-1]))
        max_length = max(lengths)
        if padding:
            features = [
                np.pad(value, ((0, max_length - len(value)), (0, 0), (0, 0))) if len(value) < max_length else value
                for value in features
            ]
        masks = [[1] * length + ([0] * (max_length - length) if padding else []) for length in lengths]
        return BatchFeature(
            {"input_audio_codes": features, "attention_mask": masks, "audio_feature_lens": lengths},
            tensor_type=return_tensors,
        )


__all__ = ["MiMoV2VLFeatureExtractor"]
