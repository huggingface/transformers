# Copyright 2024 Microsoft and the HuggingFace Inc. team. All rights reserved.
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
Processor class for Phi4Multimodal
"""

from typing import Optional, Union

import numpy as np

from ...audio_utils import AudioInput, mel_filter_bank
from ...feature_extraction_sequence_utils import SequenceFeatureExtractor
from ...image_processing_utils import BatchFeature
from ...utils import TensorType, is_torch_available, logging


if is_torch_available():
    import torch


logger = logging.get_logger(__name__)


class Phi4MultimodalFeatureExtractor(SequenceFeatureExtractor):
    model_input_names = ["audio_input_features", "audio_embed_sizes", "audio_attention_mask"]

    def __init__(
        self,
        feature_size: int = 80,
        sampling_rate: int = 16000,
        hop_length: int = 160,
        n_fft: int = 512,
        win_length: int = 400,
        preemphasis: float = 0.97,
        padding_value: float = 0.0,
        audio_compression_rate: int = 8,
        audio_downsample_rate: int = 1,
        audio_feat_stride: int = 1,
        mel_min_frequency: float = 0,
        mel_max_frequency: float = 7690,
        **kwargs,
    ):
        super().__init__(feature_size=feature_size, sampling_rate=sampling_rate, padding_value=padding_value, **kwargs)

        self.hop_length = hop_length
        self.n_fft = n_fft
        self.win_length = win_length
        self.preemphasis = preemphasis
        self.padding_value = padding_value
        self.audio_compression_rate = audio_compression_rate
        self.audio_downsample_rate = audio_downsample_rate
        self.audio_feat_stride = audio_feat_stride

        self.mel_filters = mel_filter_bank(
            num_frequency_bins=self.n_fft // 2 + 1,
            num_mel_filters=self.feature_size,
            min_frequency=mel_min_frequency,
            max_frequency=mel_max_frequency,
            sampling_rate=self.sampling_rate,
            triangularize_in_mel_space=True,
            mel_scale="kaldi",
        )

    def __call__(
        self,
        raw_speech: AudioInput,
        sampling_rate: Optional[int] = None,
        pad_to_multiple_of: Optional[int] = None,
        padding: Optional[str] = "longest",
        max_length: Optional[int] = None,
        truncation: bool = False,
        return_tensors: Optional[Union[str, TensorType]] = None,
        return_attention_mask: Optional[bool] = True,
        device: Optional[str] = "cpu",
        **kwargs,
    ) -> BatchFeature:
        """
        Main method to featurize and prepare for the model one or several audio sequence(s). Implementation uses PyTorch for
        the STFT computation if available, otherwise a slower NumPy based one.

        Args:
            raw_speech (`np.ndarray`, `torch.Tensor`, `list[np.ndarray]`, `list[torch.Tensor]`):
                The sequence or batch of sequences to be processed. Each sequence can be a numpy array or PyTorch tensor.
                For batched inputs, sequences can be a list of numpy arrays or PyTorch tensors, or a single numpy array or
                PyTorch tensor with first dimension being the batch size.
            sampling_rate (`int`, *optional*):
                The sampling rate at which the `raw_speech` input was sampled. It is strongly recommended to pass
                `sampling_rate` at the forward call to prevent silent errors.
            pad_to_multiple_of (`int`, *optional*, defaults to None):
                If set will pad the sequence to a multiple of the provided value.
            padding (`str`, *optional*, defaults to "longest"):
                Padding strategy. Can be "longest" to pad to the longest sequence in the batch, or a specific length.
            max_length (`int`, *optional*):
                Maximum length of the returned list and optionally padding length.
            truncation (`bool`, *optional*, defaults to False):
                Activates truncation to cut input sequences longer than *max_length* to *max_length*.
            return_tensors (`str` or [`~utils.TensorType`], *optional*):
                If set, will return tensors instead of numpy arrays. Acceptable values are:
                - `'pt'`: Return PyTorch `torch.Tensor` objects.
                - `'np'`: Return Numpy `np.ndarray` objects.
                - `'tf'`: Return TensorFlow `tf.constant` objects.
            return_attention_mask (`bool`, *optional*, defaults to `True`):
                Whether to return the extracted audio input features' attention mask.
            device (`str`, *optional*, defaults to "cpu"):
                Specifies the device for computation of the audio features. (e.g., "cpu", "cuda")

        Returns:
            [`BatchFeature`]: A [`BatchFeature`] with the following fields:
                - **audio_input_features** -- Audio features extracted from the raw audio input, shape (batch_size, max_feature_length, feature_size).
                - **audio_lengths** -- Length of each audio sample in the batch, shape (batch_size,).
                - **audio_attention_mask** -- Attention mask for the audio input, shape (batch_size, max_feature_length).
                If `return_tensors` is not specified, the fields will be PyTorch tensors if PyTorch is available, otherwise NumPy arrays.
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

        # Convert to torch tensor
        if isinstance(raw_speech, np.ndarray):
            raw_speech = torch.tensor(raw_speech)
        elif isinstance(raw_speech, (list, tuple)) and isinstance(raw_speech[0], np.ndarray):
            raw_speech = [torch.tensor(speech) for speech in raw_speech]

        is_batched_torch = isinstance(raw_speech, torch.Tensor) and len(raw_speech.shape) > 1
        if is_batched_torch and len(raw_speech.shape) > 2:
            logger.warning(
                f"Only mono-channel audio is supported for input to {self.__class__.__name__}. "
                "We will take the mean of the channels to convert to mono."
            )
            raw_speech = raw_speech.mean(-1)

        is_batched_sequence = isinstance(raw_speech, (list, tuple))
        if is_batched_sequence:
            for speech in raw_speech:
                if len(speech.shape) > 1:
                    logger.warning(
                        f"Only mono-channel audio is supported for input to {self.__class__.__name__}. "
                        "We will take the mean of the channels to convert to mono."
                    )
                    speech = speech.mean(-1)

        if is_batched_torch or is_batched_sequence:
            raw_speech = [speech[:, None].to(torch.float32) for speech in raw_speech]
        else:
            raw_speech = [raw_speech[:, None].to(torch.float32)]

        audio_lengths = [len(speech) for speech in raw_speech]

        # convert into correct format for padding
        batched_speech = BatchFeature(data={"audio_input_features": raw_speech, "audio_lengths": audio_lengths})
        padded_inputs = self.pad(
            batched_speech,
            padding=padding,
            max_length=max_length,
            truncation=truncation,
            pad_to_multiple_of=pad_to_multiple_of,
            return_tensors="pt",
        )
        input_features = padded_inputs.audio_input_features.squeeze(-1)
        audio_lengths = padded_inputs.audio_lengths

        input_features = self._torch_extract_fbank_features(input_features, audio_lengths, device)

        feature_lengths = (audio_lengths - self.win_length) // self.hop_length + 1
        feature_lengths = feature_lengths * self.audio_feat_stride
        audio_embed_sizes = self._compute_audio_embed_size(feature_lengths)

        feature_attention_mask = (
            torch.arange(0, feature_lengths.max()) if is_torch_available() else np.arange(0, feature_lengths.max())
        )
        feature_attention_mask = (
            feature_attention_mask[None, :] < feature_lengths[:, None] if len(feature_lengths) > 1 else None
        )

        data = {
            "audio_input_features": input_features,
            "audio_embed_sizes": audio_embed_sizes,
        }
        if feature_attention_mask is not None and return_attention_mask:
            data["audio_attention_mask"] = feature_attention_mask

        return BatchFeature(data=data, tensor_type=return_tensors)

    # TODO; @eustlb, move this to audio_utils in a general spectogram_batch function that handles torch and numpy
    def _torch_extract_fbank_features(
        self, waveform: "torch.FloatTensor", audio_lengths: "torch.Tensor", device: str = "cpu"
    ) -> "torch.FloatTensor":
        """
        Compute the log mel-scaled spectrogram of batched waveforms using PyTorch's FFT implementation.

        Args:
            waveform (torch.FloatTensor` of shape `(batch_size, max_audio_length)`):
                The batched waveforms.
            audio_lengths (`torch.Tensor` of shape `(batch_size,)`):
                The lengths of the waveforms along the max_audio_length dimension.
            device (`str`, *optional*, defaults to "cpu"):
                The device to run the computation on. (e.g., "cpu", "cuda")

        Returns:
            `torch.FloatTensor` of shape `(batch_size, max_feature_length, feature_size)`:
                The log mel-scaled spectrogram of the batched waveforms.
        """
        fft_window = torch.hamming_window(self.win_length, periodic=False, device=device, dtype=torch.float64)

        # batched implementation
        batch_size = waveform.shape[0]
        frames = waveform.unfold(-1, self.win_length, self.hop_length)

        # ---
        # the unbatched (and unpaded) original implementation skips last few audio values that can't be included in a frame
        # we need to ensure that the corresponding frames for the padded input also mask these values
        if batch_size > 1:
            frames = frames.clone()
            # concerned batch indices
            to_mask_batch_idxs = torch.arange(batch_size)[audio_lengths != audio_lengths.max()]
            if to_mask_batch_idxs.numel() > 0:
                batch_idxs_down = (audio_lengths[to_mask_batch_idxs] - self.win_length) // self.hop_length + 1
                batch_idxs_up = (audio_lengths[to_mask_batch_idxs] // self.hop_length) - 1
                offset_idx = batch_idxs_down.min()
                max_idx = batch_idxs_up.max()

                mask = torch.arange(max_idx - offset_idx, device=device).expand(to_mask_batch_idxs.shape[0], -1)
                mask = ((batch_idxs_down - offset_idx).unsqueeze(1) <= mask) & (
                    mask < (batch_idxs_up - offset_idx).unsqueeze(1)
                )
                mask = mask.unsqueeze(-1).expand(-1, -1, self.win_length)
                masked_frames = frames[to_mask_batch_idxs, offset_idx:max_idx].masked_fill_(mask, 0)
                frames[to_mask_batch_idxs, offset_idx:max_idx] = masked_frames
        # ---

        # apply pre-emphasis first order filter on fft windows
        frames_prev = torch.roll(frames, 1, dims=-1)
        frames_prev[:, :, 0] = frames_prev[:, :, 1]
        frames = (frames - self.preemphasis * frames_prev) * 32768

        # apply fft
        S = torch.fft.rfft(fft_window * frames.view(-1, self.win_length), n=self.n_fft, dim=1)
        S = S.view(frames.shape[0], -1, S.shape[-1])
        S = S.to(torch.complex64)

        spec = torch.abs(S)
        spec_power = spec**2

        # apply triangular mel filter bank
        mel_filters = torch.from_numpy(self.mel_filters).to(device, torch.float32)
        log_spec = torch.clamp(spec_power @ mel_filters, min=1.0)
        log_spec = torch.log(log_spec)

        return log_spec

    def _compute_audio_embed_size(self, audio_frames):
        integer = audio_frames // self.audio_compression_rate
        remainder = audio_frames % self.audio_compression_rate
        result = integer + (remainder > 0).to(integer.dtype)

        integer = result // self.audio_downsample_rate
        remainder = result % self.audio_downsample_rate
        result = integer + (remainder > 0).to(integer.dtype)  # qformer compression

        return result


__all__ = ["Phi4MultimodalFeatureExtractor"]
