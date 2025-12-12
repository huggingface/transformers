# coding=utf-8
# Copyright 2022 The HuggingFace Inc. team.
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
"""
Feature extractor class for Audio Spectrogram Transformer.
"""

from typing import Optional, Union

import numpy as np

from ...audio_utils import mel_filter_bank, spectrogram, window_function
from ...feature_extraction_sequence_utils import SequenceFeatureExtractor
from ...feature_extraction_utils import BatchFeature
from ...utils import TensorType, is_speech_available, is_torch_available, logging


if is_speech_available():
    import torchaudio.compliance.kaldi as ta_kaldi

if is_torch_available():
    import torch


logger = logging.get_logger(__name__)


class ASTFeatureExtractor(SequenceFeatureExtractor):
    r"""
    Constructs a Audio Spectrogram Transformer (AST) feature extractor.
    """

    model_input_names = ["input_values", "attention_mask"]

    def __init__(
        self,
        feature_size=1,
        sampling_rate=16000,
        num_mel_bins=128,
        max_length=1024,
        padding_value=0.0,
        do_normalize=True,
        mean=-4.2677393,
        std=4.5689974,
        return_attention_mask=False,
        **kwargs,
    ):
        super().__init__(feature_size=feature_size, sampling_rate=sampling_rate, padding_value=padding_value, **kwargs)
        self.num_mel_bins = num_mel_bins
        self.max_length = max_length
        self.do_normalize = do_normalize
        self.mean = mean
        self.std = std
        self.return_attention_mask = return_attention_mask

        if not is_speech_available():
            mel_filters = mel_filter_bank(
                num_frequency_bins=257,
                num_mel_filters=self.num_mel_bins,
                min_frequency=20,
                max_frequency=sampling_rate // 2,
                sampling_rate=sampling_rate,
                norm=None,
                mel_scale="kaldi",
                triangularize_in_mel_space=True,
            )

            self.mel_filters = mel_filters
            self.window = window_function(400, "hann", periodic=False)

    def _extract_fbank_features(
        self,
        waveform: Union[np.ndarray, "torch.Tensor"],
        max_length: int,
        convert_to_tensor: bool = False,
    ) -> Union[np.ndarray, "torch.Tensor"]:
        """
        Get mel-filter bank features using TorchAudio. Note that TorchAudio requires 16-bit signed integers as inputs
        and hence the waveform should not be normalized before feature extraction.
        """
        if is_speech_available():
            if is_torch_available() and isinstance(waveform, torch.Tensor):
                if waveform.ndim != 1:
                    raise ValueError(f"Input waveform must be 1D, but got {waveform.ndim}D tensor.")
                waveform = waveform.unsqueeze(0)
                fbank = ta_kaldi.fbank(
                    waveform,
                    sample_frequency=self.sampling_rate,
                    window_type="hanning",
                    num_mel_bins=self.num_mel_bins,
                )
            else:
                waveform = torch.from_numpy(waveform).unsqueeze(0)
                fbank = ta_kaldi.fbank(
                    waveform,
                    sample_frequency=self.sampling_rate,
                    window_type="hanning",
                    num_mel_bins=self.num_mel_bins,
                )
        else:
            waveform = np.squeeze(waveform)
            fbank = spectrogram(
                waveform,
                self.window,
                frame_length=400,
                hop_length=160,
                fft_length=512,
                power=2.0,
                center=False,
                preemphasis=0.97,
                mel_filters=self.mel_filters,
                log_mel="log",
                mel_floor=1.192092955078125e-07,
                remove_dc_offset=True,
            ).T
            fbank = torch.from_numpy(fbank)

        n_frames = fbank.shape[0]
        difference = max_length - n_frames

        # pad or truncate
        if difference > 0:
            pad_module = torch.nn.ZeroPad2d((0, 0, 0, difference))
            fbank = pad_module(fbank)
        elif difference < 0:
            fbank = fbank[0:max_length, :]

        if not convert_to_tensor and not isinstance(waveform, torch.Tensor):
            fbank = fbank.numpy()

        return fbank

    def normalize(self, input_values: np.ndarray) -> np.ndarray:
        return (input_values - self.mean) / (self.std * 2)

    def __call__(
        self,
        raw_speech: Union[np.ndarray, list[float], list[np.ndarray], list[list[float]]],
        sampling_rate: Optional[int] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        **kwargs,
    ) -> BatchFeature:

        if sampling_rate is not None:
            if sampling_rate != self.sampling_rate:
                raise ValueError(
                    f"The model corresponding to this feature extractor: {self} was trained using a sampling rate of "
                    f"{self.sampling_rate}. Please make sure that the provided `raw_speech` input was sampled with "
                    f"{self.sampling_rate} and not {sampling_rate}."
                )
        else:
            logger.warning(
                f"It is strongly recommended to pass the `sampling_rate` argument to `{self.__class__.__name__}()`."
            )

        is_batched = False

        # Torch tensor path
        if is_torch_available() and isinstance(raw_speech, torch.Tensor):
            if raw_speech.ndim == 2:
                is_batched = True
                raw_speech = list(raw_speech.unbind(0))
            elif raw_speech.ndim == 1:
                raw_speech = [raw_speech]
            else:
                raise ValueError(f"Only mono-channel audio is supported for input to {self}")

        else:
            is_batched_numpy = isinstance(raw_speech, np.ndarray) and len(raw_speech.shape) > 1
            if is_batched_numpy and len(raw_speech.shape) > 2:
                raise ValueError(f"Only mono-channel audio is supported for input to {self}")

            is_batched = is_batched_numpy or (
                isinstance(raw_speech, (list, tuple)) and isinstance(raw_speech[0], (np.ndarray, tuple, list))
            )

            if is_batched:
                raw_speech = [np.asarray(speech, dtype=np.float32) for speech in raw_speech]
            elif not is_batched and not isinstance(raw_speech, np.ndarray):
                raw_speech = np.asarray(raw_speech, dtype=np.float32)
            elif isinstance(raw_speech, np.ndarray) and raw_speech.dtype is np.float64:
                raw_speech = raw_speech.astype(np.float32)

            if not is_batched:
                raw_speech = [raw_speech]

        # extract fbank
        features = [
            self._extract_fbank_features(
                waveform,
                max_length=self.max_length,
                convert_to_tensor=(return_tensors == "pt"),
            )
            for waveform in raw_speech
        ]

        # Tensor output
        if is_torch_available() and isinstance(features[0], torch.Tensor) and return_tensors == "pt":
            features = torch.stack(features)
        elif isinstance(features[0], torch.Tensor) and return_tensors != "pt":
            features = [f.numpy() for f in features]

        # BatchFeature
        padded_inputs = BatchFeature({"input_values": features})

        input_values = padded_inputs.get("input_values")
        if isinstance(input_values[0], list):
            padded_inputs["input_values"] = [np.asarray(feature, dtype=np.float32) for feature in input_values]

        # normalization
        if self.do_normalize:
            if is_torch_available() and isinstance(input_values, torch.Tensor):
                padded_inputs["input_values"] = self.normalize(input_values)
            else:
                padded_inputs["input_values"] = [self.normalize(feature) for feature in input_values]

        if return_tensors is not None:
            padded_inputs = padded_inputs.convert_to_tensors(return_tensors)

        return padded_inputs


__all__ = ["ASTFeatureExtractor"]