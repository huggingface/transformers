# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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

from ... import is_torch_available
from ...audio_utils import mel_filter_bank, spectrogram, window_function
from ...feature_extraction_sequence_utils import SequenceFeatureExtractor
from ...feature_extraction_utils import BatchFeature
from ...utils import TensorType, logging


if is_torch_available():
    import torch


logger = logging.get_logger(__name__)


class Qwen3ASRFeatureExtractor(SequenceFeatureExtractor):
    r"""
    Constructs a Qwen3 ASR feature extractor.

    Extracts 128-bin log-mel features from raw speech, then right-pads the mel time axis to a multiple of ``2 * n_window``.

    Args:
        feature_size (`int`, *optional*, defaults to 128):
            Number of mel filter banks.
        sampling_rate (`int`, *optional*, defaults to 16000):
            Audio sampling rate in Hz.
        hop_length (`int`, *optional*, defaults to 160):
            Length of the overlapping windows for the STFT used to obtain the Mel Frequency coefficients.
        chunk_length (`int`, *optional*, defaults to 30):
            Maximum audio length (in seconds) used to trim/pad when ``padding="max_length"``.
        n_fft (`int`, *optional*, defaults to 400):
            Size of the Fourier transform.
        padding_value (`float`, *optional*, defaults to 0.0):
            Padding value used to pad the raw audio.
        dither (`float`, *optional*, defaults to 0.0):
            If non-zero, adds Gaussian noise (`std = dither`) to each STFT frame.
        return_attention_mask (`bool`, *optional*, defaults to `False`):
            Whether to return the attention mask corresponding to the padded mel frames. Recommended for batched inference.
        n_window (`int`, *optional*, defaults to 50):
            Half the mel-frame chunk size used for padding. The log-mel time axis is right-padded to a
            multiple of ``2 * n_window``.
    """

    model_input_names = ["input_features"]

    def __init__(
        self,
        feature_size=128,
        sampling_rate=16000,
        hop_length=160,
        chunk_length=30,
        n_fft=400,
        padding_value=0.0,
        dither=0.0,
        return_attention_mask=False,
        n_window=50,
        **kwargs,
    ):
        super().__init__(
            feature_size=feature_size,
            sampling_rate=sampling_rate,
            padding_value=padding_value,
            return_attention_mask=return_attention_mask,
            **kwargs,
        )
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.chunk_length = chunk_length
        self.n_samples = chunk_length * sampling_rate
        self.nb_max_frames = self.n_samples // hop_length
        self.sampling_rate = sampling_rate
        self.dither = dither
        self.n_window = n_window
        self.mel_filters = mel_filter_bank(
            num_frequency_bins=1 + n_fft // 2,
            num_mel_filters=feature_size,
            min_frequency=0.0,
            max_frequency=8000.0,
            sampling_rate=sampling_rate,
            norm="slaney",
            mel_scale="slaney",
        )

    def _np_extract_fbank_features(self, waveform_batch: np.ndarray, device: str) -> np.ndarray:
        """Compute log-mel spectrograms using a NumPy STFT."""
        if device != "cpu":
            raise ValueError(
                f"Got device `{device}` for feature extraction, but feature extraction on CUDA accelerator "
                "devices requires torch, which is not installed. Either set `device='cpu'`, or "
                "install torch according to the official instructions: https://pytorch.org/get-started/locally/"
            )
        log_spec_batch = []
        for waveform in waveform_batch:
            log_spec = spectrogram(
                waveform,
                window_function(self.n_fft, "hann"),
                frame_length=self.n_fft,
                hop_length=self.hop_length,
                power=2.0,
                dither=self.dither,
                mel_filters=self.mel_filters,
                log_mel="log10",
            )
            log_spec = log_spec[:, :-1]
            log_spec = np.maximum(log_spec, log_spec.max() - 8.0)
            log_spec = (log_spec + 4.0) / 4.0
            log_spec_batch.append(log_spec)
        return np.array(log_spec_batch)

    def _torch_extract_fbank_features(self, waveform: np.ndarray, device: str = "cpu") -> np.ndarray:
        """Compute log-mel spectrograms using PyTorch's (optionally GPU-accelerated) STFT."""
        waveform = torch.from_numpy(waveform).to(device, torch.float32)
        window = torch.hann_window(self.n_fft, device=device)

        if self.dither != 0.0:
            waveform += self.dither * torch.randn(waveform.shape, dtype=waveform.dtype, device=waveform.device)

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
        return log_spec.numpy()

    def __call__(
        self,
        raw_speech: np.ndarray | list[float] | list[np.ndarray] | list[list[float]],
        truncation: bool = True,
        pad_to_multiple_of: int | None = None,
        return_tensors: str | TensorType | None = None,
        return_attention_mask: bool | None = None,
        padding: str | None = "max_length",
        max_length: int | None = None,
        sampling_rate: int | None = None,
        n_window: int | None = None,
        device: str | None = "cpu",
        **kwargs,
    ) -> BatchFeature:
        r"""
        Prepare log-mel features from one or several audio sequences.

        Args:
            raw_speech (`np.ndarray`, `list[float]`, `list[np.ndarray]`, `list[list[float]]`):
                The sequence or batch of sequences to be padded. Mono-channel audio only.
            truncation (`bool`, *optional*, defaults to `True`):
                Truncate audio longer than ``max_length`` samples.
            pad_to_multiple_of (`int`, *optional*):
                If set, pads the raw audio to a multiple of this value (in samples). Separate from
                ``n_window``, which applies to the mel-frame axis.
            return_tensors (`str` or [`~utils.TensorType`], *optional*):
                Return format: ``'pt'`` for PyTorch tensors, ``'np'`` for NumPy arrays.
            return_attention_mask (`bool`, *optional*):
                Whether to return the mel-frame attention mask (recommended for batched inference).
            padding (`str` or [`~utils.PaddingStrategy`], *optional*, defaults to `"max_length"`):
                Padding strategy: ``"longest"``, ``"max_length"`` or ``"do_not_pad"``.
            max_length (`int`, *optional*):
                Maximum audio length (in samples) when ``padding="max_length"``.
            sampling_rate (`int`, *optional*):
                Sampling rate of ``raw_speech``. Must match the feature extractor's sampling rate.
            n_window (`int`, *optional*):
                Override the instance's ``n_window`` for this call. The mel axis is padded to a multiple
                of ``2 * n_window``. Set to ``0`` to skip mel-axis padding entirely.
            device (`str`, *optional*, defaults to `"cpu"`):
                Device used to compute the log-mel spectrogram.
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

        is_batched_numpy = isinstance(raw_speech, np.ndarray) and len(raw_speech.shape) > 1
        if is_batched_numpy and len(raw_speech.shape) > 2:
            raise ValueError(f"Only mono-channel audio is supported for input to {self}")
        is_batched = is_batched_numpy or (
            isinstance(raw_speech, (list, tuple)) and (isinstance(raw_speech[0], (np.ndarray, tuple, list)))
        )

        if is_batched:
            raw_speech = [np.asarray([speech], dtype=np.float32).T for speech in raw_speech]
        elif not is_batched and not isinstance(raw_speech, np.ndarray):
            raw_speech = np.asarray(raw_speech, dtype=np.float32)
        elif isinstance(raw_speech, np.ndarray) and raw_speech.dtype is np.dtype(np.float64):
            raw_speech = raw_speech.astype(np.float32)

        # always return batch
        if not is_batched:
            raw_speech = [np.asarray([raw_speech]).T]

        batched_speech = BatchFeature({"input_features": raw_speech})

        padded_inputs = self.pad(
            batched_speech,
            padding=padding,
            max_length=max_length if max_length else self.n_samples,
            truncation=truncation,
            pad_to_multiple_of=pad_to_multiple_of,
            return_attention_mask=True,
        )

        input_features = padded_inputs.get("input_features").transpose(2, 0, 1)
        extract_fbank_features = (
            self._torch_extract_fbank_features if is_torch_available() else self._np_extract_fbank_features
        )
        input_features = extract_fbank_features(input_features[0], device)
        padded_inputs["input_features"] = input_features

        # Rescale raw-sample attention mask to mel-frame resolution.
        rescaled_attention_mask = padded_inputs["attention_mask"][:, :: self.hop_length]
        if padded_inputs["attention_mask"].shape[1] % self.hop_length != 0:
            rescaled_attention_mask = rescaled_attention_mask[:, :-1]
        padded_inputs["attention_mask"] = rescaled_attention_mask

        # Right-pad the mel time axis to a multiple of `2 * n_window` (needed by `Qwen3ASREncoder`).
        if n_window is None:
            n_window = self.n_window
        multiple = n_window * 2
        if multiple and multiple > 1:
            remainder = padded_inputs["input_features"].shape[-1] % multiple
            if remainder:
                pad = multiple - remainder
                padded_inputs["input_features"] = np.pad(padded_inputs["input_features"], [(0, 0), (0, 0), (0, pad)])
                padded_inputs["attention_mask"] = np.pad(padded_inputs["attention_mask"], [(0, 0), (0, pad)])

        if not return_attention_mask:
            padded_inputs.pop("attention_mask", None)

        if return_tensors is not None:
            padded_inputs = padded_inputs.convert_to_tensors(return_tensors)

        return padded_inputs


__all__ = ["Qwen3ASRFeatureExtractor"]
