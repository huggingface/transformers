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
"""Feature extractor class for Vocos"""

import numpy as np

from ...audio_utils import AudioInput, make_list_of_audio
from ...feature_extraction_sequence_utils import BatchFeature, SequenceFeatureExtractor
from ...utils import PaddingStrategy, TensorType, is_torch_available, is_torchaudio_available, logging
from ...utils.import_utils import requires


if is_torch_available():
    import torch
    import torch.nn.functional as F


if is_torchaudio_available():
    import torchaudio


logger = logging.get_logger(__name__)


@requires(backends=("torchaudio",))
class VocosFeatureExtractor(SequenceFeatureExtractor):
    r"""
    Constructs a Vocos feature extractor.

    This feature extractor inherits from [`~feature_extraction_sequence_utils.SequenceFeatureExtractor`] which contains
    most of the main methods. Users should refer to this superclass for more information regarding those methods.

    This class extracts mel-filter bank features from raw speech using `torchaudio.transforms.MelSpectrogram` which
    performs a Short-Time Fourier Transform (STFT) and then applies a Mel filter bank, computed as in original
    [Vocos](https://github.com/gemelo-ai/vocos/blob/c859e3b7b534f3776a357983029d34170ddd6fc3/vocos/feature_extractors.py#L28)

    Args:
        feature_size (`int`, *optional*, defaults to 100):
            The feature dimension of the extracted features.
        sampling_rate (`int`, *optional*, defaults to 24000):
            The sampling rate at which the audio files should be digitalized expressed in hertz (Hz).
        num_mel_bins (`int`, *optional*, defaults to 100):
            Number of Mel-frequency bins.
        n_fft (`int`, *optional*, defaults to 1024):
            Size of the Fourier transform.
        hop_length (`int`, *optional*, defaults to 256):
            Length of the overlapping windows for the STFT used to obtain the Mel Frequency coefficients.
        padding (`str`, *optional*, defaults to `"center"`):
            Symmetric padding if 'same' and center padding if 'center'.
        padding_value (`float`, *optional*, defaults to 0.0):
            Padding value used to pad the audio. Should correspond to silences.
        return_attention_mask (`bool`, *optional*, defaults to `True`):
            Whether to return the attention mask. If left to the default, it will return the attention mask.

            [What are attention masks?](../glossary#attention-mask)

    """

    model_input_names = ["input_features", "attention_mask"]

    def __init__(
        self,
        feature_size=100,
        sampling_rate=24000,
        num_mel_bins=100,
        n_fft=1024,
        hop_length=256,
        padding="center",
        padding_value=0.0,
        return_attention_mask=True,
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
        self.win_length = n_fft
        self.num_mel_bins = num_mel_bins
        if padding not in ["center", "same"]:
            raise ValueError("Padding must be either `center` or `same`.")
        self.padding = padding

        self.mel_filters = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sampling_rate,
            n_mels=self.num_mel_bins,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            center=(self.padding == "center"),
            power=1,
        )

    def _torch_extract_fbank_features(self, batched_audio, device=None):
        """Extract mel-spectrogram features from audio using torchaudio spectrogram implementation."""

        audio = batched_audio["input_features"]

        if device is None:
            if isinstance(audio[0], torch.Tensor):
                device = audio[0].device
            else:
                device = "cpu"

        if device != next(self.mel_filters.buffers()).device:
            self.mel_filters = self.mel_filters.to(device)

        if self.padding == "same":
            pad = self.win_length - self.hop_length
            audio = F.pad(audio, (pad // 2, pad // 2), mode="reflect")
            batched_audio["attention_mask"] = F.pad(batched_audio["attention_mask"], (pad // 2, pad // 2), value=0)

        audio_spectrogram = self.mel_filters(audio.to(device))

        # compute log mel spectrograms as in original: https://github.com/gemelo-ai/vocos/blob/c859e3b7b534f3776a357983029d34170ddd6fc3/vocos/modules.py#L194
        audio_spectrogram = torch.log(torch.clip(audio_spectrogram, min=1e-7))

        batched_audio["input_features"] = audio_spectrogram

        return batched_audio

    def __call__(
        self,
        audio: AudioInput,
        padding: bool | str | PaddingStrategy | None = True,
        max_length: int | None = None,
        pad_to_multiple_of: int | None = None,
        truncation: bool | None = False,
        sampling_rate: int | None = None,
        return_tensors: str | TensorType | None = None,
        device: str | None = None,
        **kwargs,
    ) -> BatchFeature:
        """
        Main method to featurize and prepare for the model one or several sequence(s).

        Args:
            audio (`np.ndarray`, `torch.Tensor`, `list[np.ndarray]`, `list[torch.Tensor]`):
                The sequence or batch of sequences to be padded. Each sequence can be a numpy array, a list of float
                values, a list of numpy arrays or a list of list of float values. Must be mono channel audio, not
                stereo, i.e. single float per timestep.
            padding (`bool`, `str` or [`~utils.PaddingStrategy`], *optional*, defaults to `True`):
                Select a strategy to pad the returned sequences (according to the model's padding side and padding
                index) among:

                - `True` or `'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
                  sequence if provided).
                - `'max_length'`: Pad to a maximum length specified with the argument `max_length` or to the maximum
                  acceptable input length for the model if that argument is not provided.
                - `False` or `'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of different
                  lengths).
            max_length (`int`, *optional*):
                Maximum length of the returned list and optionally padding length (see above).
            truncation (`bool`):
                Activates truncation to cut input sequences longer than *max_length* to *max_length*.
            sampling_rate (`int`, *optional*):
                The sampling rate at which the `audio` input was sampled. It is strongly recommended to pass
                `sampling_rate` at the forward call to prevent silent errors.
            return_tensors (`str` or [`~utils.TensorType`], *optional*):
                If set, will return tensors instead of list of python integers. Acceptable values are:
                - `'pt'`: Return PyTorch `torch.Tensor` objects.
                - `'np'`: Return Numpy `np.ndarray` objects.
            device (`str`, *optional*):
                Device on which the tensors will be allocated (if left to `None`, uses the device of the first input
                `audio` element, or CPU if the input is a numpy array).
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

        if padding and truncation:
            raise ValueError("Both padding and truncation were set. Make sure you only set one.")

        if pad_to_multiple_of is None and len(audio) > 1 and padding:
            pad_to_multiple_of = self.hop_length

        # always return batch
        audio = make_list_of_audio(audio)

        for _audio in audio:
            if len(_audio.shape) > 1 and _audio.shape[0] > 1:
                raise ValueError(f"Only mono-channel audio is supported for input to {self}, but got {_audio.shape}")

        if isinstance(audio[0], np.ndarray):
            audio = [torch.from_numpy(_audio) for _audio in audio]

        audio = [_audio.view(-1, 1) for _audio in audio]

        batch = BatchFeature({"input_features": audio})

        padded_inputs = self.pad(
            batch,
            padding=padding,
            max_length=max_length,
            pad_to_multiple_of=pad_to_multiple_of,
            truncation=truncation,
            return_attention_mask=True,
            return_tensors="pt",
        )

        padded_inputs["input_features"] = padded_inputs["input_features"].squeeze(-1)

        padded_inputs = self._torch_extract_fbank_features(padded_inputs, device=device)

        if return_tensors is not None:
            padded_inputs = padded_inputs.convert_to_tensors(return_tensors)

        return padded_inputs


__all__ = ["VocosFeatureExtractor"]
