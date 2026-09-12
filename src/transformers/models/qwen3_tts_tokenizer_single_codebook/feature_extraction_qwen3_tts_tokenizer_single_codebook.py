# Copyright 2026 The Qwen team, Alibaba Group and the HuggingFace Inc. team. All rights reserved.
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
from typing import Any

import numpy as np

from ...audio_utils import AudioInput, make_list_of_audio, mel_filter_bank, spectrogram, window_function
from ...feature_extraction_sequence_utils import SequenceFeatureExtractor
from ...feature_extraction_utils import BatchFeature
from ...utils import PaddingStrategy, TensorType, logging


logger = logging.get_logger(__name__)


class Qwen3TTSTokenizerSingleCodebookFeatureExtractor(SequenceFeatureExtractor):
    r"""
    Constructs a Qwen3-TTS single-codebook feature extractor.

    This class inherits from [`SequenceFeatureExtractor`]. It pads waveforms with an inner
    [`SequenceFeatureExtractor`] and computes two spectrograms from the same 16 kHz audio:

    - 128-bin Whisper-style log-mel features (`input_features`) for the encoder, computed on the waveform
      zero-padded to a multiple of `hop_length * 2 * audio_vq_ds_rate` samples so that every speech code covers
      the same number of samples;
    - 80-bin reference mels (`ref_mels`) for the decoder, computed on the peak-normalised waveform.

    The tokenizer encodes 16 kHz audio and decodes 24 kHz audio, hence `sampling_rate` differs from
    `Qwen3TTSTokenizerSingleCodebookConfig.output_sample_rate`.

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
            Downsample rate of the quantizer. Waveforms are padded to `hop_length * 2 * audio_vq_ds_rate` samples.
        return_attention_mask (`bool`, *optional*, defaults to `True`):
            Whether to return `input_features_mask`.
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
        ref_peak_db (`float`, *optional*, defaults to -6.0):
            Peak level in dBFS the waveform is normalised to before the reference mel is computed.
    """

    model_input_names = ["input_features", "input_features_mask", "ref_mels"]

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
        ref_peak_db=-6.0,
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
        self.ref_num_mel_bins = ref_num_mel_bins
        self.ref_n_fft = ref_n_fft
        self.ref_hop_length = ref_hop_length
        self.ref_win_length = ref_win_length
        self.ref_mel_fmin = ref_mel_fmin
        self.ref_mel_fmax = ref_mel_fmax
        self.ref_peak_db = ref_peak_db

        # `self.pad` is reserved for the spectrograms, so the waveforms get their own padder.
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

    @property
    def waveform_pad_multiple(self) -> int:
        return self.hop_length * 2 * self.audio_vq_ds_rate

    def _extract_encoder_log_mel(self, waveform: np.ndarray) -> np.ndarray:
        """Whisper log-mel features of shape `(num_frames, feature_size)`."""
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
        return log_spec.T

    def _extract_ref_mel(self, waveform: np.ndarray) -> np.ndarray:
        """BigVGAN-style log-mel of the peak-normalised waveform, of shape `(num_frames, ref_num_mel_bins)`."""
        peak = np.abs(waveform).max()
        if peak > 0:
            waveform = waveform * (10 ** (self.ref_peak_db / 20) / peak)

        pad = (self.ref_n_fft - self.ref_hop_length) // 2
        pad_mode = "reflect" if waveform.shape[-1] > pad else "constant"
        waveform = np.pad(waveform, (pad, pad), mode=pad_mode)
        window_pad = (self.ref_n_fft - self.ref_win_length) // 2
        window = np.pad(window_function(self.ref_win_length, "hann"), (window_pad, window_pad))
        complex_spec = spectrogram(
            waveform,
            window,
            frame_length=self.ref_n_fft,
            hop_length=self.ref_hop_length,
            power=None,
            center=False,
        )
        magnitude = np.sqrt(np.abs(complex_spec) ** 2 + 1e-9)
        mel = self.ref_mel_filters.T @ magnitude
        return np.log(np.maximum(mel, 1e-5)).T

    def __call__(
        self,
        raw_speech: AudioInput,
        padding: bool | str | PaddingStrategy = True,
        max_length: int | None = None,
        truncation: bool = False,
        return_tensors: str | TensorType | None = None,
        return_attention_mask: bool | None = None,
        sampling_rate: int | None = None,
        **kwargs,
    ) -> BatchFeature:
        r"""
        Args:
            raw_speech (`np.ndarray`, `torch.Tensor`, `list[float]`, `list[np.ndarray]`, `list[torch.Tensor]`, `list[list[float]]`):
                Mono audio at `sampling_rate`, a single sequence or a batch of sequences.
            padding (`bool`, `str` or [`~utils.PaddingStrategy`], *optional*, defaults to `True`):
                Padding strategy applied to the waveforms and to both spectrograms.
            max_length (`int`, *optional*):
                Maximum waveform length in samples. Spectrograms are padded or truncated to the matching frame count.
            truncation (`bool`, *optional*, defaults to `False`):
                Whether to truncate waveforms longer than `max_length`.
            return_tensors (`str` or [`~utils.TensorType`], *optional*):
                Framework of the returned tensors, `"pt"` or `"np"`.
            return_attention_mask (`bool`, *optional*):
                Whether to return `input_features_mask`. Defaults to `self.return_attention_mask`.
            sampling_rate (`int`, *optional*):
                Sampling rate of `raw_speech`. Passing it guards against silent resampling errors.
        """
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

        audio_list = [np.asarray(audio, dtype=np.float32) for audio in make_list_of_audio(raw_speech)]
        for audio in audio_list:
            if audio.ndim != 1:
                raise ValueError(f"Expected mono audio of shape (length,) but got shape {audio.shape}")

        padded_waveforms = self.waveform_padder.pad(
            BatchFeature({"audio": audio_list}),
            padding=padding,
            max_length=max_length,
            truncation=truncation,
            pad_to_multiple_of=self.waveform_pad_multiple,
            return_attention_mask=True,
            return_tensors="np",
        )
        input_values = padded_waveforms["audio"]
        padding_mask = padded_waveforms["attention_mask"]

        encoder_mels = []
        ref_mels = []
        for waveform, mask in zip(input_values, padding_mask):
            valid_length = int(mask.sum())
            encoder_length = -(-valid_length // self.waveform_pad_multiple) * self.waveform_pad_multiple
            encoder_mels.append(self._extract_encoder_log_mel(waveform[:encoder_length]))
            ref_mels.append(self._extract_ref_mel(waveform[:valid_length]))

        max_frames = max_length // self.hop_length if max_length is not None else None
        padded_encoder_mels = self.pad(
            BatchFeature({"input_features": encoder_mels}),
            padding=padding,
            max_length=max_frames,
            truncation=truncation,
            return_attention_mask=return_attention_mask,
        )
        padded_ref_mels = self.pad(
            BatchFeature({"input_features": ref_mels}),
            padding=padding,
            max_length=max_frames,
            truncation=truncation,
            return_attention_mask=False,
        )

        encoded_inputs = BatchFeature(
            {
                "input_features": np.stack(padded_encoder_mels["input_features"]).transpose(0, 2, 1),
                "ref_mels": np.stack(padded_ref_mels["input_features"]),
            }
        )
        if return_attention_mask:
            encoded_inputs["input_features_mask"] = np.stack(padded_encoder_mels["attention_mask"])
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
