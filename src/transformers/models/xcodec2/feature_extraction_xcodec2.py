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

import copy
from typing import Any

import numpy as np

from ... import is_torch_available
from ...audio_utils import (
    AudioInput,
    make_list_of_audio,
    mel_filter_bank,
    spectrogram_torch,
    window_function,
)
from ...feature_extraction_sequence_utils import SequenceFeatureExtractor
from ...feature_extraction_utils import BatchFeature
from ...utils import PaddingStrategy, TensorType, logging


if is_torch_available():
    import torch


logger = logging.get_logger(__name__)


class Xcodec2FeatureExtractor(SequenceFeatureExtractor):
    r"""
    Constructs a Xcodec2 feature extractor, which computes mel-filter bank features for the semantic encoder and padded
    audio for the acoustic encoder.

    This feature extractor inherits from [`SequenceFeatureExtractor`] which contains most of the main methods. Users
    should refer to this superclass for more information regarding those methods.

    Args:
        feature_size (`int`, *optional*, defaults to 80):
            The feature dimension of the extracted features.
        sampling_rate (`int`, *optional*, defaults to 16000):
            The sample rate at which the audio files should be digitalized expressed in hertz (Hz).
        padding_value (`float`, *optional*, defaults to 1.0):
            The value that is used to fill the padding vectors for the mel spectrogram.
        hop_length (`int`, *optional*, defaults to 320):
            Number of audio samples encoded per frame. Equivalent to product of downsampling ratios.
            Needed for acoustic encoder input padding.
    """

    model_input_names = ["audio_spectrogram", "audio", "padding_mask"]

    def __init__(
        self,
        feature_size=80,
        sampling_rate=16000,
        padding_value=1.0,
        hop_length=320,
        **kwargs,
    ):
        super().__init__(feature_size=feature_size, sampling_rate=sampling_rate, padding_value=padding_value, **kwargs)

        # Acoustic encoder feature extraction (similar to DAC)
        self.hop_length = hop_length
        self.acoustic_encoder_padder = SequenceFeatureExtractor(
            feature_size=1,
            sampling_rate=sampling_rate,
            padding_value=0.0,
        )
        self.acoustic_encoder_padder.model_input_names = ["audio", "padding_mask"]

        # Semantic encoder feature extraction (similar to SeamlessM4T)
        self.n_fft = 512
        self.stride = 2
        self.num_mel_bins = 80
        self.window_length = 400
        self.spectrogram_hop_length = 160
        self.mel_filters = mel_filter_bank(
            num_frequency_bins=self.n_fft // 2 + 1,
            num_mel_filters=self.num_mel_bins,
            min_frequency=20,
            max_frequency=self.sampling_rate // 2,
            sampling_rate=self.sampling_rate,
            norm=None,
            mel_scale="kaldi",
            triangularize_in_mel_space=True,
        )
        self.window = window_function(self.window_length, "povey", periodic=False)

    def _extract_fbank_features(
        self,
        audio: np.ndarray,
        device: str = "cpu",
    ) -> list[np.ndarray]:
        """Extract mel-filter bank features using torch backend for batch processing."""
        audio = torch.from_numpy(audio).to(device)
        window = torch.from_numpy(self.window).to(device)
        mel_filters = torch.from_numpy(self.mel_filters).to(device)

        processed = [waveform.squeeze() * (2**15) for waveform in audio]
        features_list = spectrogram_torch(
            processed,
            window,
            mel_filters=mel_filters,
            frame_length=self.window_length,
            hop_length=self.spectrogram_hop_length,
            fft_length=self.n_fft,
            power=2,
            center=False,
            preemphasis=0.97,
            log_mel="log",
            mel_floor=1.192092955078125e-07,
            remove_dc_offset=True,
        )
        return [features.T.cpu().numpy() for features in features_list]

    def __call__(
        self,
        audio: AudioInput,
        padding: bool | str | PaddingStrategy = True,
        max_length: int | None = None,
        truncation: bool = False,
        return_tensors: str | TensorType | None = None,
        sampling_rate: int | None = None,
        device: str = "cpu",
        **kwargs,
    ) -> BatchFeature:
        """
        Args:
            audio (`np.ndarray`, `torch.Tensor`, `list[np.ndarray]`, `list[torch.Tensor]`):
                Numpy array or torch tensor with shape (num_channels, sequence_length). A list of such arrays or
                tensors can also be provided for a batch of inputs.
            padding (`bool`, `str` or [`~utils.PaddingStrategy`], *optional*, defaults to `True`):
                Select a strategy to pad the returned sequences (according to the model's padding side and padding
                index) among:

                - `True` or `'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
                  sequence if provided).
                - `'max_length'`: Pad to a maximum length specified with the argument `max_length` or to the maximum
                  acceptable input length for the model if that argument is not provided.
                - `False` or `'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of different
                  lengths).
            pad_to_multiple_of (`int`, *optional*, defaults to 2):
                If set will pad the sequence to a multiple of the provided value.

                This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability
                `>= 7.5` (Volta), or on TPUs which benefit from having sequence lengths be a multiple of 128.
            max_length (`int`, *optional*):
                Maximum length of the returned list and optionally padding length (see above).
            truncation (`bool`):
                Activates truncation to cut input sequences longer than *max_length* to *max_length*.
            return_tensors (`str` or [`~utils.TensorType`], *optional*):
                If set, will return tensors instead of list of python integers. Acceptable values are:

                - `'tf'`: Return TensorFlow `tf.constant` objects.
                - `'pt'`: Return PyTorch `torch.Tensor` objects.
                - `'np'`: Return Numpy `np.ndarray` objects.
            sampling_rate (`int`, *optional*):
                The sample rate at which the `audio` input was sampled. It is strongly recommended to pass
                `sampling_rate` at the forward call to prevent silent errors.
            device (`str`, *optional*, defaults to `"cpu"`):
                Device for PyTorch tensors during mel-filter bank feature extraction.
            kwargs (*optional*):
                Remaining dictionary of keyword arguments that will be passed to the tokenizer or the feature
                extractor.
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

        if not is_torch_available():
            raise ImportError("PyTorch is required for mel-filter bank feature extraction.")

        audio = make_list_of_audio(audio)
        for example in audio:
            if example.ndim > 2:
                raise ValueError(f"Expected input shape (channels, length) but got shape {example.shape}")
        batch_size = len(audio)

        # 1) Acoustic encoder padding (similar to DAC)
        padded_inputs = self.acoustic_encoder_padder.pad(
            BatchFeature({"audio": audio}),
            max_length=max_length,
            truncation=truncation,
            padding=padding,
            return_attention_mask=padding,
            pad_to_multiple_of=self.hop_length,
            return_tensors="np",
        )
        padding_mask = padded_inputs.pop("attention_mask")
        padded_audio = padded_inputs["audio"][:, np.newaxis, :]

        # Redundant padding to match PyPI xcodec2==0.1.3 behavior
        # See: https://github.com/huggingface/transformers/pull/37868#discussion_r2382396644
        padded_audio = np.pad(padded_audio, ((0, 0), (0, 0), (0, self.hop_length)), constant_values=0.0)
        if padding_mask is not None:
            padding_mask = np.pad(padding_mask, ((0, 0), (0, self.hop_length)), constant_values=0)

        # 2) Compute Mel spectrogram for pretrained semantic encoder (following SeamlessM4T)
        semantic_input = np.pad(
            padded_audio,
            ((0, 0), (0, 0), (self.hop_length // 2, self.hop_length // 2)),
            constant_values=0.0,
        )
        mel_features = self._extract_fbank_features(semantic_input, device=device)
        mel_features = [(x - x.mean(0)) / np.sqrt(x.var(0, ddof=1) + 1e-7) for x in mel_features]
        encoded_inputs = BatchFeature({"audio_spectrogram": mel_features})
        padded_mel = self.pad(
            encoded_inputs,
            padding=padding,
            max_length=max_length,
            truncation=truncation,
            pad_to_multiple_of=self.stride,
            return_attention_mask=False,
            return_tensors="np",
        )
        audio_spectrogram = padded_mel["audio_spectrogram"]
        trimmed_frames = audio_spectrogram.shape[1] - (audio_spectrogram.shape[1] % self.stride)
        audio_spectrogram = audio_spectrogram[:, :trimmed_frames, :].reshape(
            batch_size, trimmed_frames // self.stride, self.num_mel_bins * self.stride
        )

        # 3) Combine outputs from DAC-like padding and SeamlessM4T feature extractor
        padded_inputs = BatchFeature(
            {"audio": padded_audio, "padding_mask": padding_mask, "audio_spectrogram": audio_spectrogram}
        )
        if return_tensors is not None:
            padded_inputs = padded_inputs.convert_to_tensors(return_tensors)
        return padded_inputs

    def to_dict(self) -> dict[str, Any]:
        output = copy.deepcopy(self.__dict__)
        output["feature_extractor_type"] = self.__class__.__name__
        if "mel_filters" in output:
            del output["mel_filters"]
        if "acoustic_encoder_padder" in output:
            del output["acoustic_encoder_padder"]
        if "window" in output:
            del output["window"]
        return output


__all__ = ["Xcodec2FeatureExtractor"]
