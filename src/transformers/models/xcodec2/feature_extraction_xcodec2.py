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

from ...audio_utils import mel_filter_bank, spectrogram_batch, window_function, make_list_of_audio, AudioInput
from ...feature_extraction_sequence_utils import SequenceFeatureExtractor
from ...feature_extraction_utils import BatchFeature
from ...utils import PaddingStrategy, TensorType, logging


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
        n_fft=512,
        window_length=400,
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
        self.n_fft = n_fft
        self.num_mel_bins = num_mel_bins
        self.stride = stride 
        self.window_length = window_length
        self.mel_floor = 1.192092955078125e-07
        self.preemphasis = 0.97
        self.spec_hop_length = 160
        self.spec_power = 2.0
        self.spec_center = False
        self.spec_remove_dc_offset = True
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
    
    def _extract_fbank_features_numpy(
        self,
        waveform_list: list[np.ndarray],
    ) -> list[np.ndarray]:
        """
        Batch version of mel-filter bank feature extraction for improved efficiency for a batch.
        """
        # Process waveforms: extract left channel if stereo and apply Kaldi scaling
        processed_waveforms = []
        for waveform in waveform_list:
            waveform = np.squeeze(waveform) * (2**15)  # Kaldi compliance: 16-bit signed integers
            processed_waveforms.append(waveform)

        # Use batch spectrogram processing
        features_list = spectrogram_batch(
            processed_waveforms,
            self.window,
            frame_length=self.window_length,
            hop_length=self.spec_hop_length,
            fft_length=self.n_fft,
            power=self.spec_power,
            center=self.spec_center,
            preemphasis=self.preemphasis,
            mel_filters=self.mel_filters,
            log_mel="log",
            mel_floor=self.mel_floor,
            remove_dc_offset=self.spec_remove_dc_offset,
        )
        
        # Transpose each feature matrix to match expected format
        return [features.T for features in features_list]

    def _extract_fbank_features_torch(
        self,
        audio: torch.Tensor,
        dtype: torch.dtype = torch.float32,
        device: str = "cpu",
    ) -> list[np.ndarray]:
        """
        torchaudio-based mel-filter bank feature extraction.
        """

        # Ensure each audio is 1D and apply Kaldi scaling (16-bit signed integers)
        audio = audio.to(device, dtype).to(dtype) * (2**15)
        if audio.dim() == 3:
            # (batch, channel, time) -> (batch, time)
            audio = audio.squeeze(1)

        # Arrays to torch
        window = torch.from_numpy(self.window).to(device, dtype)
        mel_filters = torch.from_numpy(self.mel_filters).to(device, dtype)

        # Compute mel spectrograms
        with torch.no_grad():

            # NOTE: Numpy version removes offset and applies preemphasis on each frame, which can cause difference
            if self.spec_remove_dc_offset:
                audio = audio - audio.mean(dim=-1, keepdim=True)
            if self.preemphasis is not None:
                # first sample scaling matches Kaldi
                audio[..., 0] *= (1 - self.preemphasis)
                audio[..., 1:] = audio[..., 1:] - self.preemphasis * audio[..., :-1]

            # Compute mel spectrograms for the batch
            stft = torch.stft(
                audio, 
                self.n_fft, 
                self.spec_hop_length, 
                win_length=self.window_length,
                window=window, 
                center=self.spec_center,
                return_complex=True
            )
            if self.spec_power is not None:
                stft = stft.abs() ** self.spec_power
            audio_spectrogram = mel_filters.T @ stft

            # Apply log mel with floor
            audio_spectrogram = torch.maximum(torch.tensor(self.mel_floor, device=device), audio_spectrogram)
            audio_spectrogram = torch.log(audio_spectrogram)

            # Transpose each feature matrix to match expected format
            audio_spectrogram = [spec.T.cpu().numpy().astype(np.float32) for spec in audio_spectrogram]

        return audio_spectrogram

    def __call__(
        self,
        audio: AudioInput,
        padding: Union[bool, str, PaddingStrategy] = True,
        pad_to_multiple_of: Optional[int] = 2,
        max_length: Optional[int] = None,
        truncation: bool = False,
        return_tensors: Optional[Union[str, TensorType]] = None,
        sampling_rate: Optional[int] = None,
        do_normalize_per_mel_bins: Optional[bool] = True,
        use_torch: bool = False,
        device: str = "cpu",
        **kwargs,
    ) -> BatchFeature:
        """
        Main method to featurize and prepare for the model one or several sequence(s).

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
            use_torch (`bool`, *optional*, defaults to `True`):
                Whether or not to use torch for mel-filter bank feature extraction. If `False`, a numpy
            device (`str`, *optional*, defaults to `"cpu"`):
                Device for PyTorch tensors when using torch.
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

        # ensure batch
        audio = make_list_of_audio(audio)

        # DAC-like padding
        for _, example in enumerate(audio):
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
            
        # Add channel dimension: (batch_size, sequence_length) -> (batch_size, 1, sequence_length)
        padded_inputs["audio"] = padded_inputs["audio"][:, np.newaxis, :]
        
        # Convert to tensors to use torch operations below
        if return_tensors is not None:
            padded_inputs = padded_inputs.convert_to_tensors(return_tensors)

        # Xcodec2 processing between DAC and SeamlessM4T feature extractors
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
        semantic_input = F.pad(padded_inputs["audio"], (semantic_padding, semantic_padding))

        # Compute Mel Spectrogram like in SeamlessM4TFeatureExtractor
        if use_torch:
            mel_features = self._extract_fbank_features_torch(semantic_input, device=device)
        else:
            semantic_input = [np.asarray(speech, dtype=np.float32) for speech in semantic_input]
            mel_features = self._extract_fbank_features_numpy(semantic_input)

        if do_normalize_per_mel_bins:
            # torch defaults to ddof=1, and numpy defaults to ddof=0
            mel_features = [
                (x - x.mean(0)) / np.sqrt(x.var(0, ddof=1) + 1e-7)
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
        audio_spectrogram = padded_mel["audio_spectrogram"]
        batch_size, num_frames, num_mel_channels = audio_spectrogram.shape
        
        # Trim frames to be divisible by stride and reshape
        trimmed_frames = num_frames - (num_frames % self.stride)
        audio_spectrogram = audio_spectrogram[:, :trimmed_frames, :].reshape(
            batch_size, trimmed_frames // self.stride, num_mel_channels * self.stride
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
