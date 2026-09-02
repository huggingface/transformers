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
"""Feature extractor for the Qwen3-TTS single-codebook tokenizer."""

import copy
import math
from typing import Any

import numpy as np

from ...audio_utils import mel_filter_bank, spectrogram, window_function
from ...feature_extraction_sequence_utils import SequenceFeatureExtractor
from ...feature_extraction_utils import BatchFeature
from ...utils import PaddingStrategy, TensorType, logging
from ...utils.import_utils import is_torch_available


if is_torch_available():
    import torch
    import torch.nn.functional as F


logger = logging.get_logger(__name__)


class Qwen3TTSTokenizerSingleCodebookFeatureExtractor(SequenceFeatureExtractor):
    r"""
    Constructs a Qwen3-TTS single-codebook feature extractor.

    This class inherits from [`SequenceFeatureExtractor`]. It pads waveforms with an inner
    [`SequenceFeatureExtractor`] (the Xcodec2 pattern) and computes two mel spectrograms by hand:

    - 128-bin Whisper-style log-mel features for the VQ encoder
    - 80-bin reference mels for the DiT decoder

    Args:
        feature_size (`int`, *optional*, defaults to 128):
            Number of encoder mel bins.
        sampling_rate (`int`, *optional*, defaults to 16000):
            The sampling rate of the input audio in Hz.
        padding_value (`float`, *optional*, defaults to 0.0):
            Padding value for spectrograms and waveforms.
        hop_length (`int`, *optional*, defaults to 160):
            Encoder STFT hop length.
        n_fft (`int`, *optional*, defaults to 400):
            Encoder FFT size.
        dither (`float`, *optional*, defaults to 0.0):
            Optional dither added before the encoder STFT.
        audio_vq_ds_rate (`int`, *optional*, defaults to 2):
            VQ downsample rate. Waveforms are padded to `hop_length * 2 * audio_vq_ds_rate`.
        return_attention_mask (`bool`, *optional*, defaults to `True`):
            Whether to return spectrogram and waveform masks.
        ref_num_mel_bins (`int`, *optional*, defaults to 80):
            Number of reference-mel bins.
        ref_n_fft (`int`, *optional*, defaults to 1024):
            Reference-mel FFT size.
        ref_hop_length (`int`, *optional*, defaults to 160):
            Reference-mel hop length.
        ref_win_length (`int`, *optional*, defaults to 640):
            Reference-mel window length.
        ref_mel_fmin (`float`, *optional*, defaults to 0.0):
            Minimum reference-mel frequency.
        ref_mel_fmax (`float`, *optional*, defaults to 8000.0):
            Maximum reference-mel frequency.
    """

    model_input_names = [
        "input_features",
        "input_features_mask",
        "input_values",
        "padding_mask",
        "ref_mel_features",
        "ref_mel_attention_mask",
    ]

    def __init__(
        self,
        feature_size=128,
        sampling_rate=16000,
        padding_value=0.0,
        hop_length=160,
        n_fft=400,
        dither=0.0,
        audio_vq_ds_rate=2,
        return_attention_mask=True,
        ref_num_mel_bins=80,
        ref_n_fft=1024,
        ref_hop_length=160,
        ref_win_length=640,
        ref_mel_fmin=0.0,
        ref_mel_fmax=8000.0,
        **kwargs,
    ):
        super().__init__(
            feature_size=feature_size,
            sampling_rate=sampling_rate,
            padding_value=padding_value,
            return_attention_mask=return_attention_mask,
            **kwargs,
        )
        self.hop_length = hop_length
        self.n_fft = n_fft
        self.dither = dither
        self.audio_vq_ds_rate = audio_vq_ds_rate
        self.waveform_pad_multiple = hop_length * 2 * audio_vq_ds_rate
        self.ref_num_mel_bins = ref_num_mel_bins
        self.ref_n_fft = ref_n_fft
        self.ref_hop_length = ref_hop_length
        self.ref_win_length = ref_win_length
        self.ref_mel_fmin = ref_mel_fmin
        self.ref_mel_fmax = ref_mel_fmax

        self.waveform_padder = SequenceFeatureExtractor(
            feature_size=1,
            sampling_rate=sampling_rate,
            padding_value=0.0,
        )
        self.waveform_padder.model_input_names = ["audio"]

        self.mel_filters = mel_filter_bank(
            num_frequency_bins=1 + n_fft // 2,
            num_mel_filters=feature_size,
            min_frequency=0.0,
            max_frequency=8000.0,
            sampling_rate=sampling_rate,
            norm="slaney",
            mel_scale="slaney",
        )
        self.ref_mel_filters = mel_filter_bank(
            num_frequency_bins=1 + ref_n_fft // 2,
            num_mel_filters=ref_num_mel_bins,
            min_frequency=ref_mel_fmin,
            max_frequency=ref_mel_fmax,
            sampling_rate=sampling_rate,
            norm="slaney",
            mel_scale="slaney",
        )

    def _batchify_audio(self, raw_speech) -> list[np.ndarray]:
        is_batched = (
            isinstance(raw_speech, (list, tuple))
            and len(raw_speech) > 0
            and isinstance(raw_speech[0], (np.ndarray, list, tuple))
        )
        if is_batched:
            audio_list = [np.asarray(speech, dtype=np.float32) for speech in raw_speech]
        else:
            audio_list = [np.asarray(raw_speech, dtype=np.float32)]

        for audio in audio_list:
            if audio.ndim > 1:
                raise ValueError(f"Expected mono audio of shape (length,) but got shape {audio.shape}")
        return audio_list

    def _extract_encoder_log_mel(self, waveform: np.ndarray) -> np.ndarray:
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
        return log_spec

    def _extract_ref_mel(self, waveform: np.ndarray) -> np.ndarray:
        pad = (self.ref_n_fft - self.ref_hop_length) // 2
        pad_mode = "reflect" if waveform.shape[-1] > pad else "constant"
        if is_torch_available():
            audio = torch.from_numpy(np.asarray(waveform, dtype=np.float32)).unsqueeze(0)
            audio = F.pad(audio.unsqueeze(1), (pad, pad), mode=pad_mode).squeeze(1)
            window = torch.hann_window(self.ref_win_length)
            spec = torch.stft(
                audio,
                self.ref_n_fft,
                hop_length=self.ref_hop_length,
                win_length=self.ref_win_length,
                window=window,
                center=False,
                pad_mode="reflect",
                normalized=False,
                onesided=True,
                return_complex=True,
            )
            spec = torch.sqrt(torch.view_as_real(spec).pow(2).sum(-1) + 1e-9)
            spec = spec.squeeze(0)
            mel_basis = torch.from_numpy(self.ref_mel_filters).float()
            if mel_basis.shape[0] == spec.shape[0]:
                spec = torch.matmul(mel_basis.T, spec)
            else:
                spec = torch.matmul(mel_basis, spec)
            spec = torch.log(torch.clamp(spec, min=1e-5))
            return spec.transpose(0, 1).numpy()

        padded = np.pad(waveform, (pad, pad), mode=pad_mode)
        spec = spectrogram(
            padded,
            window_function(self.ref_win_length, "hann"),
            frame_length=self.ref_n_fft,
            hop_length=self.ref_hop_length,
            power=1.0,
            center=False,
            mel_filters=self.ref_mel_filters,
        )
        spec = np.log(np.clip(spec, a_min=1e-5, a_max=None))
        return spec.T

    def __call__(
        self,
        raw_speech: np.ndarray | list[float] | list[np.ndarray] | list[list[float]],
        padding: bool | str | PaddingStrategy = True,
        max_length: int | None = None,
        truncation: bool = False,
        return_tensors: str | TensorType | None = None,
        return_attention_mask: bool | None = None,
        sampling_rate: int | None = None,
        **kwargs,
    ) -> BatchFeature:
        if sampling_rate is not None:
            if sampling_rate != self.sampling_rate:
                raise ValueError(
                    f"The model corresponding to this feature extractor: {self} was trained using a sampling rate of"
                    f" {self.sampling_rate}. Please make sure that the provided `audio` input was sampled with"
                    f" {self.sampling_rate} and not {sampling_rate}."
                )
        else:
            logger.warning(
                f"It is strongly recommended to pass the `sampling_rate` argument to `{self.__class__.__name__}()`. "
                "Failing to do so can result in silent errors that might be hard to debug."
            )

        if return_attention_mask is None:
            return_attention_mask = self.return_attention_mask

        audio_list = self._batchify_audio(raw_speech)
        padded_waveforms = self.waveform_padder.pad(
            BatchFeature({"audio": audio_list}),
            max_length=max_length,
            truncation=truncation,
            padding=padding,
            return_attention_mask=True,
            pad_to_multiple_of=self.waveform_pad_multiple,
            return_tensors="np",
        )
        padding_mask = np.asarray(padded_waveforms.pop("attention_mask"))
        input_values = np.asarray(padded_waveforms["audio"])
        if input_values.ndim == 1:
            input_values = input_values[None, :]

        encoder_mels = []
        ref_mels = []
        for i, original_audio in enumerate(audio_list):
            original_length = int(original_audio.shape[0])
            vq_length = math.ceil(original_length / self.waveform_pad_multiple) * self.waveform_pad_multiple
            encoder_waveform = input_values[i, :vq_length]
            encoder_mels.append(self._extract_encoder_log_mel(encoder_waveform).T)
            ref_mels.append(self._extract_ref_mel(input_values[i, :original_length]))

        padded_mel = self.pad(
            BatchFeature({"input_features": encoder_mels}),
            padding=padding,
            max_length=max_length,
            truncation=truncation,
            return_attention_mask=return_attention_mask,
            return_tensors=None,
        )
        input_features = np.stack([np.asarray(mel).T for mel in padded_mel["input_features"]], axis=0)
        input_features_mask = padded_mel.get("attention_mask")
        if input_features_mask is not None:
            input_features_mask = np.asarray(input_features_mask)

        max_ref_length = max(mel.shape[0] for mel in ref_mels)
        ref_mel_features = np.stack(
            [
                np.pad(mel, ((0, max_ref_length - mel.shape[0]), (0, 0)), constant_values=self.padding_value)
                for mel in ref_mels
            ]
        )
        ref_mel_attention_mask = np.stack(
            [
                np.pad(np.ones(mel.shape[0], dtype=np.int64), (0, max_ref_length - mel.shape[0]))
                for mel in ref_mels
            ]
        )

        encoded_inputs = BatchFeature(
            {
                "input_features": input_features,
                "input_features_mask": input_features_mask,
                "input_values": input_values,
                "padding_mask": padding_mask,
                "ref_mel_features": ref_mel_features,
                "ref_mel_attention_mask": ref_mel_attention_mask,
            }
        )
        if return_tensors is not None:
            encoded_inputs = encoded_inputs.convert_to_tensors(return_tensors)
        return encoded_inputs

    def to_dict(self) -> dict[str, Any]:
        output = copy.deepcopy(self.__dict__)
        output["feature_extractor_type"] = self.__class__.__name__
        output.pop("waveform_padder", None)
        output.pop("mel_filters", None)
        output.pop("ref_mel_filters", None)
        return output


__all__ = ["Qwen3TTSTokenizerSingleCodebookFeatureExtractor"]
