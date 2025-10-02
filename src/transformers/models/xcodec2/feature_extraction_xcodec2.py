# coding=utf-8
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
"""Feature extractor class for Xcodec2 model."""

import copy
from typing import Any, Optional, Union

import numpy as np
import torch
import torch.nn.functional as F

from ...audio_utils import mel_filter_bank, spectrogram, window_function
from ...feature_extraction_sequence_utils import SequenceFeatureExtractor
from ...feature_extraction_utils import BatchFeature
from ...utils import PaddingStrategy, TensorType, is_torch_available, logging


logger = logging.get_logger(__name__)


class Xcodec2FeatureExtractor(SequenceFeatureExtractor):
    r"""
    Constructs a Xcodec2 feature extractor.

    This feature extractor inherits from [`SequenceFeatureExtractor`] which contains most of the main methods. Users
    should refer to this superclass for more information regarding those methods.

    This class extracts mel-filter bank features from raw speech.

    Args:
        feature_size (`int`, *optional*, defaults to 80):
            The feature dimension of the extracted features.
        sampling_rate (`int`, *optional*, defaults to 16000):
            The sample rate at which the audio files should be digitalized expressed in hertz (Hz).
        num_mel_bins (`int`, *optional*, defaults to 80):
            Number of Mel-frequency bins.
        padding_value (`float`, *optional*, defaults to 1.0):
            The value that is used to fill the padding vectors for the mel spectrogram.
        stride (`int`, *optional*, defaults to 2):
            Stride used to reshape audios from shape (batch_size,num_frames,num_mel_bins) to
            (batch_size,num_frames//stride,num_mel_bins*stride).
        n_channels (`int`, *optional*, defaults to 1):
            Number of channels in the input audio.
        hop_length (`int`, *optional*, defaults to 320):
            Number of audio samples encoded per frame. Equivalent to product of downsampling ratios.
        pre_padding_value  (`float`, *optional*, defaults to 0.0):
            The value that is used to fill the padding vectors for the input audio (before computing the spectrogram).
    """

    model_input_names = ["audio_spectrogram", "audio", "padding_mask"]

    def __init__(
        self,
        feature_size=80,
        sampling_rate=16000,
        num_mel_bins=80,
        padding_value=1.0,
        stride=2,
        n_channels=1,
        hop_length=320,
        pre_padding_value=0.0,
        return_attention_mask=True,
        **kwargs,
    ):
        super().__init__(
            feature_size=feature_size, 
            sampling_rate=sampling_rate, 
            padding_value=padding_value,
            **kwargs
        )
        self.return_attention_mask = return_attention_mask
        self.num_mel_bins = num_mel_bins
        self.stride = stride 

        # For DAC-like padding before Mel feature extraction
        self.n_channels = n_channels
        self.hop_length = hop_length
        self.pre_padding_value = pre_padding_value
        self.initial_padder = SequenceFeatureExtractor(
            feature_size=n_channels,
            sampling_rate=sampling_rate,
            padding_value=pre_padding_value,
            **kwargs,
        )
        self.initial_padder.model_input_names = ["audio", "padding_mask"]

        # filter bank like SeamlessM4T
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
        self.window = window_function(400, "povey", periodic=False)

    def _extract_fbank_features(
        self,
        waveform: np.ndarray,
    ) -> np.ndarray:
        """
        Get mel-filter bank features using TorchAudio. Note that TorchAudio requires 16-bit signed integers as inputs
        and hence the waveform should not be normalized before feature extraction.
        """
        # by default, it extracts the left channel if stereo
        if len(waveform.shape) == 2:
            waveform = waveform[0]

        waveform = np.squeeze(waveform) * (2**15)  # Kaldi compliance: 16-bit signed integers
        features = spectrogram(
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
        return features

    def __call__(
        self,
        audio: Union[np.ndarray, list[float], list[np.ndarray], list[list[float]]],
        padding: Union[bool, str, PaddingStrategy] = True,
        pad_to_multiple_of: Optional[int] = 2,
        max_length: Optional[int] = None,
        truncation: bool = False,
        return_tensors: Optional[Union[str, TensorType]] = None,
        sampling_rate: Optional[int] = None,
        do_normalize_per_mel_bins: Optional[bool] = True,
        **kwargs,
    ) -> BatchFeature:
        """
        Main method to featurize and prepare for the model one or several sequence(s).

        Args:
            audio (`np.ndarray`, `torch.Tensor`, `list[float]`, `list[np.ndarray]`, `list[torch.Tensor]`,
            `list[list[float]]`, `list[list[list[float]]]`):
                The sequence or batch of sequences to be padded. Each sequence can be a numpy array,
                a torch tensor, a list of float values, a list of numpy arrays, a list of torch tensors,
                a list of list of float values or a list of a list of list of float values.
                If `audio` is a one-dimensional `np.ndarray`, `torch.Tensor` or a `list[float]`, `audio` is
                considered a single-channel, single-sample sound. In all other cases, the first dimension of
                `audio`, whether from an `np.ndarray`, a `torch.Tensor` or a `list[...]`,
                corresponds to the number of samples in the batch, and the number of channels
                (i.e. mono or stereo character) is derived from the other dimensions
                (1D -> single-channel waveform batches; 2D-> stereo-channel waveform batches).
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
            return_attention_mask (`bool`, *optional*):
                Whether to return the attention mask. If left to the default, will return the attention mask according
                to the specific feature_extractor's default.

                [What are attention masks?](../glossary#attention-mask)

                <Tip>

                For Xcodec2 models, `attention_mask` should always be passed for batched inference, to avoid subtle
                bugs.

                </Tip>

            return_tensors (`str` or [`~utils.TensorType`], *optional*):
                If set, will return tensors instead of list of python integers. Acceptable values are:

                - `'tf'`: Return TensorFlow `tf.constant` objects.
                - `'pt'`: Return PyTorch `torch.Tensor` objects.
                - `'np'`: Return Numpy `np.ndarray` objects.
            sampling_rate (`int`, *optional*):
                The sample rate at which the `audio` input was sampled. It is strongly recommended to pass
                `sampling_rate` at the forward call to prevent silent errors.
            do_normalize_per_mel_bins (`bool`, *optional*, defaults to `True`):
                Whether or not to zero-mean unit-variance normalize the input per mel-channel.
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

        is_batched_numpy = isinstance(audio, np.ndarray) and len(audio.shape) > 1
        if is_batched_numpy and len(audio.shape) > 3:
            raise ValueError(f"Only mono-channel or stereo-channel audio is supported for input to {self}")

        acceptable_types = (
            (torch.Tensor, np.ndarray, tuple, list) if is_torch_available() else (np.ndarray, tuple, list)
        )
        is_batched = is_batched_numpy or (
            isinstance(audio, (list, tuple)) and (isinstance(audio[0], acceptable_types))
        )

        if is_batched:
            audio = [np.asarray(speech, dtype=np.float32) for speech in audio]
        elif not is_batched and not isinstance(audio, np.ndarray):
            audio = np.asarray(audio, dtype=np.float32)
        elif isinstance(audio, np.ndarray) and audio.dtype is np.dtype(np.float64):
            audio = audio.astype(np.float32)

        # always return batch
        if not is_batched:
            audio = [audio]

        # DAC-like padding
        for idx, example in enumerate(audio):
            if example.ndim > 2:
                raise ValueError(f"Expected input shape (channels, length) but got shape {example.shape}")
            if self.feature_size == 1 and example.ndim != 1:
                raise ValueError(f"Expected mono audio but example has {example.shape[-1]} channels")
            if self.feature_size == 2:
                raise ValueError("Stereo audio isn't supported for now")

        input_values = BatchFeature({"audio": audio})
        padded_inputs = self.initial_padder.pad(
            input_values,
            max_length=max_length,
            truncation=truncation,
            padding=padding,
            return_attention_mask=padding,
            pad_to_multiple_of=self.hop_length,
        )
        if padding:
            padded_inputs["padding_mask"] = padded_inputs.pop("attention_mask")
            
        # Process audio dimensions consistently
        input_values = []
        for example in padded_inputs.pop("audio"):
            if padding:
                # Add channel dimension if needed for padded inputs
                if example.ndim == 1:
                    example = example[np.newaxis, :]
            if self.feature_size == 1 and example.ndim == 1:
                example = example[..., None]
            input_values.append(example) 
        padded_inputs["audio"] = input_values
        
        # Convert to tensors early if requested for consistent processing
        if return_tensors is not None:
            padded_inputs = padded_inputs.convert_to_tensors(return_tensors)

        # Xcodec2 processing between the two feature extractors
        # See: https://github.com/huggingface/transformers/pull/37868#discussion_r2382396644
        # 1) redundant padding inside modeling of PyPI version (xcodec2==0.1.3)
        # probably accidental on their part, but it is needed to get same results
        # since their logic pads even if input is multiple of hop length
        audio_seq_len = padded_inputs["audio"].shape[-1]
        hop_padding = self.hop_length - (audio_seq_len % self.hop_length)
        padded_inputs["audio"] = F.pad(padded_inputs["audio"], (0, hop_padding))
        padded_inputs["padding_mask"] = F.pad(padded_inputs["padding_mask"], (0, hop_padding))

        # 2) padding before semantic model feature extractor (i.e. that of SeamlessM4TFeatureExtractor)
        semantic_padding = self.hop_length // 2
        semantic_input = F.pad(padded_inputs["audio"], (semantic_padding, semantic_padding)).cpu().tolist()
        semantic_input = [np.asarray(speech, dtype=np.float32) for speech in semantic_input]

        # SeamlessM4TFeatureExtractor logic (except that we use attention/padding mask from above)
        # extract fbank features
        mel_features = [self._extract_fbank_features(waveform) for waveform in semantic_input]

        if do_normalize_per_mel_bins:
            # torch defaults to ddof=1, and numpy defaults to ddof=0
            mel_features = [
                (x - np.expand_dims(x.mean(0), 0)) / np.sqrt(np.expand_dims(x.var(0, ddof=1), 0) + 1e-7)
                for x in mel_features
            ]
        encoded_inputs = BatchFeature({"audio_spectrogram": mel_features})
        padded_mel = self.pad(
            encoded_inputs,
            padding=padding,
            max_length=max_length,
            truncation=truncation,
            pad_to_multiple_of=pad_to_multiple_of,
            return_attention_mask=False,
            return_tensors="np",
        )
        # Process mel features with stride reshaping
        audio_spectrogram = padded_mel.get("audio_spectrogram")
        batch_size, num_frames, num_mel_channels = audio_spectrogram.shape
        
        # Ensure frames are divisible by stride
        frame_remainder = num_frames % self.stride
        if frame_remainder != 0:
            audio_spectrogram = audio_spectrogram[:, : num_frames - frame_remainder, :]
            num_frames = num_frames - frame_remainder
            
        # Reshape to combine stride frames: (batch, frames//stride, channels*stride)
        audio_spectrogram = np.reshape(
            audio_spectrogram, (batch_size, num_frames // self.stride, num_mel_channels * self.stride)
        )

        # Combine output from DAC-like padding and SeamlessM4T feature extractor
        padded_inputs["audio_spectrogram"] = audio_spectrogram

        if return_tensors is not None:
            padded_inputs = padded_inputs.convert_to_tensors(return_tensors)

        return padded_inputs

    def to_dict(self) -> dict[str, Any]:
        """
        Serializes this instance to a Python dictionary. Returns:
            `dict[str, Any]`: Dictionary of all the attributes that make up this configuration instance.
        """
        output = copy.deepcopy(self.__dict__)
        output["feature_extractor_type"] = self.__class__.__name__
        if "mel_filters" in output:
            del output["mel_filters"]
        if "initial_padder" in output:
            del output["initial_padder"]
        if "window" in output:
            del output["window"]
        return output


__all__ = ["Xcodec2FeatureExtractor"]
