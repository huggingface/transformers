# Copyright 2025 The HuggingFace Inc. team and Google LLC. All rights reserved.
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
import torch

from ...audio_utils import hertz_to_mel
from ...feature_extraction_sequence_utils import SequenceFeatureExtractor
from ...feature_extraction_utils import BatchFeature
from ...utils import TensorType, logging
from ...utils.import_utils import requires


logger = logging.get_logger(__name__)


# TODO: @eustlb, we should be able to remove this and use mel_filter_bank from audio_utils
def linear_to_mel_weight_matrix(
    num_mel_bins: int,
    num_spectrogram_bins: int,
    sample_rate: float,
    lower_edge_hertz: float,
    upper_edge_hertz: float,
    dtype,
) -> np.ndarray:
    """NumPy-port of the JAX mel weight matrix logic."""
    # We use float64 for precision, matching the JAX implementation.
    internal_dtype = np.float64

    # HTK excludes the spectrogram DC bin.
    bands_to_zero = 1
    nyquist_hertz = sample_rate / 2.0
    linear_frequencies = np.linspace(0.0, nyquist_hertz, num_spectrogram_bins, dtype=internal_dtype)[bands_to_zero:]
    spectrogram_bins_mel = hertz_to_mel(linear_frequencies, mel_scale="kaldi")[:, np.newaxis]

    edges = np.linspace(
        hertz_to_mel(lower_edge_hertz, mel_scale="kaldi"),
        hertz_to_mel(upper_edge_hertz, mel_scale="kaldi"),
        num_mel_bins + 2,
        dtype=internal_dtype,
    )

    lower_edge_mel, center_mel, upper_edge_mel = (
        edges[:-2][np.newaxis, :],
        edges[1:-1][np.newaxis, :],
        edges[2:][np.newaxis, :],
    )

    lower_slopes = (spectrogram_bins_mel - lower_edge_mel) / (center_mel - lower_edge_mel)
    upper_slopes = (upper_edge_mel - spectrogram_bins_mel) / (upper_edge_mel - center_mel)
    mel_weights_matrix = np.maximum(0.0, np.minimum(lower_slopes, upper_slopes))
    return np.pad(mel_weights_matrix, [[bands_to_zero, 0], [0, 0]]).astype(dtype)


@requires(backends=("torch",))
class LasrFeatureExtractor(SequenceFeatureExtractor):
    r"""
    Constructs a LASR feature extractor.

    This feature extractor inherits from [`~feature_extraction_sequence_utils.SequenceFeatureExtractor`] which contains
    most of the main methods. Users should refer to this superclass for more information regarding those methods.

    This class extracts mel-filter bank features from raw speech using a custom numpy implementation of the `Short Time
    Fourier Transform` which should match pytorch's `torch.stft` equivalent.

    Args:
            feature_size (`int`, *optional*, defaults to 128):
                The feature dimension of the extracted features.
            sampling_rate (`int`, *optional*, defaults to 16000):
                The sampling rate at which the audio files should be digitalized expressed in hertz (Hz).
            hop_length (`int`, *optional*, defaults to 160):
                Length of the overlapping windows for the STFT used to obtain the Mel Frequency coefficients.
            n_fft (`int`, *optional*, defaults to 512):
                Size of the Fourier transform.
            win_length (`int`, *optional*, defaults to 400):
                The window length for the STFT computation.
            padding_value (`float`, *optional*, defaults to 0.0):
                Padding value used to pad the audio. Should correspond to silences.
    """

    model_input_names = ["input_features", "attention_mask"]

    def __init__(
        self,
        feature_size=128,
        sampling_rate=16000,
        hop_length=160,
        n_fft=512,
        win_length=400,
        padding_value=0.0,
        **kwargs,
    ):
        super().__init__(feature_size=feature_size, sampling_rate=sampling_rate, padding_value=padding_value, **kwargs)

        self.hop_length = hop_length
        self.n_fft = n_fft
        self.win_length = win_length
        self.mel_filters = torch.from_numpy(
            linear_to_mel_weight_matrix(
                num_mel_bins=feature_size,
                num_spectrogram_bins=n_fft // 2 + 1,
                sample_rate=sampling_rate,
                lower_edge_hertz=125.0,
                upper_edge_hertz=7500.0,
                dtype=np.float64,
            )
        )

    def _torch_extract_fbank_features(self, waveform, device="cpu"):
        # spectrogram
        window = torch.hann_window(self.win_length, periodic=False, device=device, dtype=torch.float64)
        waveform = waveform.to(torch.float64)

        # TODO: @eustlb, to be standardized
        # here we cannot use directly torch.stft because every fft frame is padded with zeros
        # due to unfold then rfft, while torch.stft unfolds with the number of fft points
        frames = waveform.unfold(-1, self.win_length, self.hop_length)
        stft = torch.fft.rfft(window * frames, n=self.n_fft)
        power_spec = torch.abs(stft) ** 2

        # log mel spectrogram
        mel_filters = self.mel_filters.to(device)
        mel_spec = torch.clamp(power_spec @ mel_filters, min=1e-5)
        mel_spec = torch.log(mel_spec)

        return mel_spec

    def __call__(
        self,
        raw_speech: np.ndarray | list[float] | list[np.ndarray] | list[list[float]],
        truncation: bool = False,
        pad_to_multiple_of: int | None = None,
        return_tensors: str | TensorType | None = None,
        return_attention_mask: bool | None = None,
        padding: str | None = "longest",
        max_length: int | None = None,
        sampling_rate: int | None = None,
        do_normalize: bool | None = None,
        device: str | None = "cpu",
        return_token_timestamps: bool | None = None,
        **kwargs,
    ) -> BatchFeature:
        """
        Main method to featurize and prepare for the model one or several sequence(s). Implementation uses PyTorch for
        the STFT computation if available, otherwise a slower NumPy based one.

        Args:
            raw_speech (`np.ndarray`, `list[float]`, `list[np.ndarray]`, `list[list[float]]`):
                The sequence or batch of sequences to be padded. Each sequence can be a numpy array, a list of float
                values, a list of numpy arrays or a list of list of float values. Must be mono channel audio, not
                stereo, i.e. single float per timestep.
            truncation (`bool`, *optional*, default to `True`):
                Activates truncation to cut input sequences longer than *max_length* to *max_length*.
            pad_to_multiple_of (`int`, *optional*, defaults to None):
                If set will pad the sequence to a multiple of the provided value.

                This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability
                `>= 7.5` (Volta), or on TPUs which benefit from having sequence lengths be a multiple of 128.
            return_attention_mask (`bool`, *optional*):
                Whether to return the attention mask. If left to the default, will return the attention mask according
                to the specific feature_extractor's default.

                [What are attention masks?](../glossary#attention-mask)

                <Tip>

                For Parakeet models, `attention_mask` should always be passed for batched inference, to avoid subtle
                bugs.

                </Tip>

            return_tensors (`str` or [`~utils.TensorType`], *optional*):
                If set, will return tensors instead of list of python integers. Acceptable values are:

                - `'tf'`: Return TensorFlow `tf.constant` objects.
                - `'pt'`: Return PyTorch `torch.Tensor` objects.
                - `'np'`: Return Numpy `np.ndarray` objects.
            sampling_rate (`int`, *optional*):
                The sampling rate at which the `raw_speech` input was sampled. It is strongly recommended to pass
                `sampling_rate` at the forward call to prevent silent errors and allow automatic speech recognition
                pipeline.
            padding_value (`float`, *optional*, defaults to 0.0):
                The value that is used to fill the padding values / vectors.
            do_normalize (`bool`, *optional*, defaults to `False`):
                Whether or not to zero-mean unit-variance normalize the input. Normalizing can help to significantly
                improve the performance of the model.
            device (`str`, *optional*, defaults to `'cpu'`):
                Specifies the device for computation of the log-mel spectrogram of audio signals in the
                `_torch_extract_fbank_features` method. (e.g., "cpu", "cuda")
            return_token_timestamps (`bool`, *optional*, defaults to `None`):
                Deprecated. Use `return_attention_mask` instead from which the number of frames can be inferred.

                Whether or not to return the number of frames of the input raw_speech.
                These num_frames can be used by the model to compute word level timestamps.
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
        elif isinstance(raw_speech, (list, tuple)):
            if isinstance(raw_speech[0], (list, np.ndarray)):
                raw_speech = [torch.tensor(speech) for speech in raw_speech]
            else:  # list[float]
                raw_speech = torch.tensor(raw_speech)

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

        batched_speech = BatchFeature({"input_features": raw_speech})
        padded_inputs = self.pad(
            batched_speech,
            padding=padding,
            max_length=max_length,
            truncation=truncation,
            pad_to_multiple_of=pad_to_multiple_of,
            return_attention_mask=return_attention_mask,
            return_tensors="pt",
        )
        input_features = padded_inputs.input_features.squeeze(-1)
        input_features = self._torch_extract_fbank_features(input_features, device)
        data = {
            "input_features": input_features.to(torch.float32),
        }

        if return_attention_mask:
            attention_mask = padded_inputs.attention_mask[:, self.win_length - 1 :: self.hop_length]
            data["attention_mask"] = attention_mask.to(torch.bool)

        return BatchFeature(data=data, tensor_type=return_tensors)


__all__ = ["LasrFeatureExtractor"]
