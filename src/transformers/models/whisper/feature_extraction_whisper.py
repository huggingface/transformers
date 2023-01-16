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
Feature extractor class for Whisper
"""

from typing import List, Optional, Tuple, Union

import numpy as np
from numpy.fft import fft

from ...feature_extraction_sequence_utils import SequenceFeatureExtractor
from ...feature_extraction_utils import BatchFeature
from ...utils import TensorType, logging


logger = logging.get_logger(__name__)


# Copied from transformers.models.wav2vec2.modeling_wav2vec2._compute_mask_indices with attention_mask from torch.LongTensor to np.array
def _compute_mask_indices(
    shape: Tuple[int, int],
    mask_prob: float,
    mask_length: int,
    attention_mask: Optional[np.array] = None,
    min_masks: int = 0,
) -> np.ndarray:
    """
    Computes random mask spans for a given shape. Used to implement [SpecAugment: A Simple Data Augmentation Method for
    ASR](https://arxiv.org/abs/1904.08779). Note that this method is not optimized to run on TPU and should be run on
    CPU as part of the preprocessing during training.

    Args:
        shape: The shape for which to compute masks. This should be of a tuple of size 2 where
               the first element is the batch size and the second element is the length of the axis to span.
        mask_prob:  The percentage of the whole axis (between 0 and 1) which will be masked. The number of
                    independently generated mask spans of length `mask_length` is computed by
                    `mask_prob*shape[1]/mask_length`. Note that due to overlaps, `mask_prob` is an upper bound and the
                    actual percentage will be smaller.
        mask_length: size of the mask
        min_masks: minimum number of masked spans
        attention_mask: A (right-padded) attention mask which independently shortens the feature axis of
                        each batch dimension.
    """
    batch_size, sequence_length = shape

    if mask_length < 1:
        raise ValueError("`mask_length` has to be bigger than 0.")

    if mask_length > sequence_length:
        raise ValueError(
            f"`mask_length` has to be smaller than `sequence_length`, but got `mask_length`: {mask_length}"
            f" and `sequence_length`: {sequence_length}`"
        )

    # epsilon is used for probabilistic rounding
    epsilon = np.random.rand(1).item()

    def compute_num_masked_span(input_length):
        """Given input length, compute how many spans should be masked"""
        num_masked_span = int(mask_prob * input_length / mask_length + epsilon)
        num_masked_span = max(num_masked_span, min_masks)

        # make sure num masked span <= sequence_length
        if num_masked_span * mask_length > sequence_length:
            num_masked_span = sequence_length // mask_length

        # make sure num_masked span is also <= input_length - (mask_length - 1)
        if input_length - (mask_length - 1) < num_masked_span:
            num_masked_span = max(input_length - (mask_length - 1), 0)

        return num_masked_span

    # compute number of masked spans in batch
    input_lengths = (
        attention_mask.sum(-1).tolist()
        if attention_mask is not None
        else [sequence_length for _ in range(batch_size)]
    )

    # SpecAugment mask to fill
    spec_aug_mask = np.zeros((batch_size, sequence_length), dtype=bool)
    spec_aug_mask_idxs = []

    max_num_masked_span = compute_num_masked_span(sequence_length)

    if max_num_masked_span == 0:
        return spec_aug_mask

    for input_length in input_lengths:
        # compute num of masked spans for this input
        num_masked_span = compute_num_masked_span(input_length)

        # get random indices to mask
        spec_aug_mask_idx = np.random.choice(
            np.arange(input_length - (mask_length - 1)), num_masked_span, replace=False
        )

        # pick first sampled index that will serve as a dummy index to pad vector
        # to ensure same dimension for all batches due to probabilistic rounding
        # Picking first sample just pads those vectors twice.
        if len(spec_aug_mask_idx) == 0:
            # this case can only happen if `input_length` is strictly smaller then
            # `sequence_length` in which case the last token has to be a padding
            # token which we can use as a dummy mask id
            dummy_mask_idx = sequence_length - 1
        else:
            dummy_mask_idx = spec_aug_mask_idx[0]

        spec_aug_mask_idx = np.concatenate(
            [spec_aug_mask_idx, np.ones(max_num_masked_span - num_masked_span, dtype=np.int32) * dummy_mask_idx]
        )
        spec_aug_mask_idxs.append(spec_aug_mask_idx)

    spec_aug_mask_idxs = np.array(spec_aug_mask_idxs)

    # expand masked indices to masked spans
    spec_aug_mask_idxs = np.broadcast_to(
        spec_aug_mask_idxs[:, :, None], (batch_size, max_num_masked_span, mask_length)
    )
    spec_aug_mask_idxs = spec_aug_mask_idxs.reshape(batch_size, max_num_masked_span * mask_length)

    # add offset to the starting indexes so that indexes now create a span
    offsets = np.arange(mask_length)[None, None, :]
    offsets = np.broadcast_to(offsets, (batch_size, max_num_masked_span, mask_length)).reshape(
        batch_size, max_num_masked_span * mask_length
    )
    spec_aug_mask_idxs = spec_aug_mask_idxs + offsets

    # ensure that we cannot have indices larger than sequence_length
    if spec_aug_mask_idxs.max() > sequence_length - 1:
        spec_aug_mask_idxs[spec_aug_mask_idxs > sequence_length - 1] = sequence_length - 1

    # scatter indices to mask
    np.put_along_axis(spec_aug_mask, spec_aug_mask_idxs, 1, -1)

    return spec_aug_mask


class WhisperFeatureExtractor(SequenceFeatureExtractor):
    r"""
    Constructs a Whisper feature extractor.

    This feature extractor inherits from [`WhisperFeatureExtractor`] which contains most of the main methods. Users
    should refer to this superclass for more information regarding those methods.

    This class extracts mel-filter bank features from raw speech using a custom numpy implementation of the `Short Time
    Fourier Transform` which should match pytorch's `torch.stft` equivalent.

    Args:
        feature_size (`int`, defaults to 80):
            The feature dimension of the extracted features.
        sampling_rate (`int`, defaults to 16000):
            The sampling rate at which the audio files should be digitalized expressed in Hertz per second (Hz).
        hop_length (`int`, defaults to 160):
            Length of the overlaping windows for the STFT used to obtain the Mel Frequency coefficients.
        chunk_length (`int`, defaults to 30):
            The maximum number of chuncks of `sampling_rate` samples used to trim and pad longer or shorter audio
            sequences.
        n_fft (`int`, defaults to 400):
            Size of the Fourier transform.
        padding_value (`float`, *optional*, defaults to 0.0):
            Padding value used to pad the audio. Should correspond to silences.
    """

    model_input_names = ["input_features"]

    def __init__(
        self,
        feature_size=80,
        sampling_rate=16000,
        hop_length=160,
        chunk_length=30,
        n_fft=400,
        padding_value=0.0,
        return_attention_mask=False,  # pad inputs to max length with silence token (zero) and no attention mask
        **kwargs
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
        self.mel_filters = self.get_mel_filters(sampling_rate, n_fft, n_mels=feature_size)

    def get_mel_filters(self, sr, n_fft, n_mels=128, dtype=np.float32):
        # Initialize the weights
        n_mels = int(n_mels)
        weights = np.zeros((n_mels, int(1 + n_fft // 2)), dtype=dtype)

        # Center freqs of each FFT bin
        fftfreqs = np.fft.rfftfreq(n=n_fft, d=1.0 / sr)

        # 'Center freqs' of mel bands - uniformly spaced between limits
        min_mel = 0.0
        max_mel = 45.245640471924965

        mels = np.linspace(min_mel, max_mel, n_mels + 2)

        mels = np.asanyarray(mels)

        # Fill in the linear scale
        f_min = 0.0
        f_sp = 200.0 / 3
        freqs = f_min + f_sp * mels

        # And now the nonlinear scale
        min_log_hz = 1000.0  # beginning of log region (Hz)
        min_log_mel = (min_log_hz - f_min) / f_sp  # same (Mels)
        logstep = np.log(6.4) / 27.0  # step size for log region

        # If we have vector data, vectorize
        log_t = mels >= min_log_mel
        freqs[log_t] = min_log_hz * np.exp(logstep * (mels[log_t] - min_log_mel))

        mel_f = freqs

        fdiff = np.diff(mel_f)
        ramps = np.subtract.outer(mel_f, fftfreqs)

        for i in range(n_mels):
            # lower and upper slopes for all bins
            lower = -ramps[i] / fdiff[i]
            upper = ramps[i + 2] / fdiff[i + 1]

            # .. then intersect them with each other and zero
            weights[i] = np.maximum(0, np.minimum(lower, upper))

        # Slaney-style mel is scaled to be approx constant energy per channel
        enorm = 2.0 / (mel_f[2 : n_mels + 2] - mel_f[:n_mels])
        weights *= enorm[:, np.newaxis]

        return weights

    def fram_wave(self, waveform, center=True):
        """
        Transform a raw waveform into a list of smaller waveforms. The window length defines how much of the signal is
        contain in each frame (smalle waveform), while the hope length defines the step between the beginning of each
        new frame.

        Centering is done by reflecting the waveform which is first centered around `frame_idx * hop_length`.
        """
        frames = []
        for i in range(0, waveform.shape[0] + 1, self.hop_length):
            half_window = (self.n_fft - 1) // 2 + 1
            if center:
                start = i - half_window if i > half_window else 0
                end = i + half_window if i < waveform.shape[0] - half_window else waveform.shape[0]

                frame = waveform[start:end]

                if start == 0:
                    padd_width = (-i + half_window, 0)
                    frame = np.pad(frame, pad_width=padd_width, mode="reflect")

                elif end == waveform.shape[0]:
                    padd_width = (0, (i - waveform.shape[0] + half_window))
                    frame = np.pad(frame, pad_width=padd_width, mode="reflect")

            else:
                frame = waveform[i : i + self.n_fft]
                frame_width = frame.shape[0]
                if frame_width < waveform.shape[0]:
                    frame = np.lib.pad(
                        frame, pad_width=(0, self.n_fft - frame_width), mode="constant", constant_values=0
                    )

            frames.append(frame)
        return np.stack(frames, 0)

    def stft(self, frames, window):
        """
        Calculates the complex Short-Time Fourier Transform (STFT) of the given framed signal. Should give the same
        results as `torch.stft`.
        """
        frame_size = frames.shape[1]
        fft_size = self.n_fft

        if fft_size is None:
            fft_size = frame_size

        if fft_size < frame_size:
            raise ValueError("FFT size must greater or equal the frame size")
        # number of FFT bins to store
        num_fft_bins = (fft_size >> 1) + 1

        data = np.empty((len(frames), num_fft_bins), dtype=np.complex64)
        fft_signal = np.zeros(fft_size)

        for f, frame in enumerate(frames):
            if window is not None:
                np.multiply(frame, window, out=fft_signal[:frame_size])
            else:
                fft_signal[:frame_size] = frame
            data[f] = fft(fft_signal, axis=0)[:num_fft_bins]
        return data.T

    def _np_extract_fbank_features(self, waveform: np.array) -> np.ndarray:
        """
        Compute the log-Mel spectrogram of the provided audio, gives similar results whisper's original torch
        implementation with 1e-5 tolerance.
        """
        window = np.hanning(self.n_fft + 1)[:-1]

        frames = self.fram_wave(waveform)
        stft = self.stft(frames, window=window)
        magnitudes = np.abs(stft[:, :-1]) ** 2

        filters = self.mel_filters
        mel_spec = filters @ magnitudes

        log_spec = np.log10(np.clip(mel_spec, a_min=1e-10, a_max=None))
        log_spec = np.maximum(log_spec, log_spec.max() - 8.0)
        log_spec = (log_spec + 4.0) / 4.0

        return log_spec

    # Modified from transformers.models.wav2vec2.modeling_wav2vec2.Wav2Vec2Model._mask_hidden_states
    def _mask_input_features(
        self,
        input_features: List[np.array],
        mask_time_indices: Optional[np.array] = None,
        attention_mask: Optional[np.array] = None,
    ):
        """
        Masks extracted features along time axis and/or along feature axis according to
        [SpecAugment](https://arxiv.org/abs/1904.08779).
        """

        # generate indices & apply SpecAugment along time axis
        batch_size, hidden_size, sequence_length = input_features.shape

        # todo: move to config
        self.mask_time_prob = 0.05
        self.mask_time_length = 2
        self.mask_time_min_masks = 2

        self.mask_feature_prob = 0.05
        self.mask_feature_length = 10
        self.mask_feature_min_masks = 0

        if mask_time_indices is not None:
            # apply SpecAugment along time axis with given mask_time_indices
            input_features[mask_time_indices] = 0
        elif self.mask_time_prob > 0:
            # generate indices & apply SpecAugment along time axis
            mask_time_indices = _compute_mask_indices(
                (batch_size, sequence_length),
                mask_prob=self.mask_time_prob,
                mask_length=self.mask_time_length,
                attention_mask=attention_mask,
                min_masks=self.mask_time_min_masks,
            )
            mask_time_indices = np.broadcast_to(mask_time_indices[:, None], (batch_size, hidden_size, sequence_length))
            # mask_time_indices = np.tile(mask_time_indices, (1, hidden_size, 1))
            # mask_time_indices = np.repeat(mask_time_indices[:, None, :], hidden_size, axis=1)
            input_features[mask_time_indices] = 0

        if self.mask_feature_prob > 0:
            # generate indices & apply SpecAugment along feature axis
            mask_feature_indices = _compute_mask_indices(
                (batch_size, hidden_size),
                mask_prob=self.mask_feature_prob,
                mask_length=self.mask_feature_length,
                min_masks=self.mask_feature_min_masks,
            )
            input_features[mask_feature_indices] = 0

        return input_features

    def __call__(
        self,
        raw_speech: Union[np.ndarray, List[float], List[np.ndarray], List[List[float]]],
        truncation: bool = True,
        pad_to_multiple_of: Optional[int] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        return_attention_mask: Optional[bool] = None,
        padding: Optional[str] = "max_length",
        max_length: Optional[int] = None,
        sampling_rate: Optional[int] = None,
        **kwargs
    ) -> BatchFeature:
        """
        Main method to featurize and prepare for the model one or several sequence(s).

        Args:
            raw_speech (`np.ndarray`, `List[float]`, `List[np.ndarray]`, `List[List[float]]`):
                The sequence or batch of sequences to be padded. Each sequence can be a numpy array, a list of float
                values, a list of numpy arrays or a list of list of float values.
            truncation (`bool`, *optional*, default to `True`):
                Activates truncation to cut input sequences longer than *max_length* to *max_length*.
            pad_to_multiple_of (`int`, *optional*, defaults to None):
                If set will pad the sequence to a multiple of the provided value.

                This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability
                >= 7.5 (Volta), or on TPUs which benefit from having sequence lengths be a multiple of 128.
            return_attention_mask (`bool`, *optional*):
                Whether to return the attention mask. If left to the default, will return the attention mask according
                to the specific feature_extractor's default.

                [What are attention masks?](../glossary#attention-mask)

                <Tip>

                For WhisperTransoformer models, `attention_mask` should alwys be passed for batched inference, to avoid
                subtle bugs.

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
            padding_value (`float`, defaults to 0.0):
                The value that is used to fill the padding values / vectors.
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
                "It is strongly recommended to pass the `sampling_rate` argument to this function. "
                "Failing to do so can result in silent errors that might be hard to debug."
            )

        is_batched = bool(
            isinstance(raw_speech, (list, tuple))
            and (isinstance(raw_speech[0], np.ndarray) or isinstance(raw_speech[0], (tuple, list)))
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

        # convert into correct format for padding

        # todo: auto return_attention_mask
        padded_inputs = self.pad(
            batched_speech,
            padding=padding,
            max_length=max_length if max_length else self.n_samples,
            truncation=truncation,
            pad_to_multiple_of=pad_to_multiple_of,
            return_attention_mask=True,
        )
        # make sure list is in array format
        input_features = padded_inputs.get("input_features").transpose(2, 0, 1)

        # mono
        input_features = [self._np_extract_fbank_features(waveform) for waveform in input_features[0]]

        if isinstance(input_features[0], List):
            padded_inputs["input_features"] = [np.asarray(feature, dtype=np.float32) for feature in input_features]
        else:
            padded_inputs["input_features"] = input_features

        # todo: move to config
        apply_spec_augment = True
        if apply_spec_augment:
            # todo: input_features to np array
            padded_inputs["input_features"] = np.stack(padded_inputs["input_features"], 0)

            padded_inputs["input_features"] = self._mask_input_features(
                padded_inputs["input_features"],
                attention_mask=padded_inputs["attention_mask"][:, ::self.hop_length],
            )

        if return_tensors is not None:
            padded_inputs = padded_inputs.convert_to_tensors(return_tensors)

        return padded_inputs
