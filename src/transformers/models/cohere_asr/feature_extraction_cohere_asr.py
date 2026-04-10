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

import numpy as np
import torch

from ...feature_extraction_sequence_utils import SequenceFeatureExtractor
from ...feature_extraction_utils import BatchFeature
from ...utils import TensorType, is_librosa_available, logging
from ...utils.import_utils import requires


if is_librosa_available():
    import librosa


EPSILON = 1e-5
LOG_ZERO_GUARD_VALUE = 2**-24


logger = logging.get_logger(__name__)


@requires(backends=("torch", "librosa"))
class CohereAsrFeatureExtractor(SequenceFeatureExtractor):
    r"""
    Constructs a CohereAsr feature extractor.

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
        preemphasis (`float`, *optional*, defaults to 0.97):
            A preemphasis filter coefficient. 0.0 means no preemphasis filter.
        padding_value (`float`, *optional*, defaults to 0.0):
            Padding value used to pad the audio. Should correspond to silences.
        dither (`float`, *optional*, defaults to 1e-05):
            Amount of deterministic dither noise to add before feature extraction. Each sample is seeded by its
            valid waveform length so that dither is batch-composition invariant. Set to 0.0 to disable.
        max_audio_clip_s (`float`, *optional*, defaults to 35.0):
            Maximum duration in seconds for a single audio chunk. Audio longer than
            `max_audio_clip_s - overlap_chunk_second` is split at energy-based boundaries.
        overlap_chunk_second (`float`, *optional*, defaults to 5.0):
            Size in seconds of the boundary search window used when splitting long audio. This is not actual
            overlap between chunks — it defines how far back from the chunk boundary to search for a quiet
            split point.
        min_energy_window_samples (`int`, *optional*, defaults to 1600):
            Size in samples of the sliding window used to find the quietest point when splitting audio chunks.
    """

    model_input_names = ["input_features", "attention_mask"]

    def __init__(
        self,
        feature_size=128,
        sampling_rate=16000,
        hop_length=160,
        n_fft=512,
        win_length=400,
        preemphasis=0.97,
        padding_value=0.0,
        dither=1e-5,
        max_audio_clip_s=35.0,
        overlap_chunk_second=5.0,
        min_energy_window_samples=1600,
        **kwargs,
    ):
        super().__init__(feature_size=feature_size, sampling_rate=sampling_rate, padding_value=padding_value, **kwargs)

        self.hop_length = hop_length
        self.n_fft = n_fft
        self.win_length = win_length
        self.preemphasis = preemphasis
        self.dither = dither
        self.max_audio_clip_s = max_audio_clip_s
        self.overlap_chunk_second = overlap_chunk_second
        self.min_energy_window_samples = min_energy_window_samples

        # TODO: @eustlb, for now we use librosa to compute the mel filters
        # indeed mel_filter_bank uses np.float64 (while librosa uses np.float32), giving numerical differences
        mel_filters = librosa.filters.mel(
            sr=sampling_rate, n_fft=n_fft, n_mels=feature_size, fmin=0.0, fmax=sampling_rate / 2, norm="slaney"
        )
        self.mel_filters = torch.from_numpy(mel_filters).to(torch.float32)

    def _find_split_point_energy(self, waveform: torch.Tensor, start_idx: int, end_idx: int) -> int:
        segment = waveform[start_idx:end_idx]
        if segment.shape[0] <= self.min_energy_window_samples:
            return (start_idx + end_idx) // 2

        min_energy = float("inf")
        quietest_idx = start_idx
        upper = segment.shape[0] - self.min_energy_window_samples
        for i in range(0, upper, self.min_energy_window_samples):
            window = segment[i : i + self.min_energy_window_samples]
            energy = torch.sqrt(torch.mean(window * window)).item()
            if energy < min_energy:
                min_energy = energy
                quietest_idx = start_idx + i
        return quietest_idx

    def _split_audio_chunks_energy(self, waveform: torch.Tensor) -> list[torch.Tensor]:
        chunk_size = max(1, int(round(self.max_audio_clip_s * self.sampling_rate)))
        boundary_context_size = max(1, int(round(self.overlap_chunk_second * self.sampling_rate)))
        total_samples = waveform.shape[0]

        if total_samples <= chunk_size:
            return [waveform]

        chunks_meta: list[tuple[int, int]] = []
        idx = 0
        while idx < total_samples:
            if idx + chunk_size >= total_samples:
                chunks_meta.append((idx, total_samples))
                break

            search_start = max(idx, idx + chunk_size - boundary_context_size)
            search_end = min(idx + chunk_size, total_samples)
            if search_end <= search_start:
                split_point = idx + chunk_size
            else:
                split_point = self._find_split_point_energy(waveform, search_start, search_end)

            split_point = max(idx + 1, min(split_point, total_samples))
            chunks_meta.append((idx, split_point))
            idx = split_point

        return [waveform[start:end] for start, end in chunks_meta if end > start]

    def _apply_dither(self, waveform: torch.Tensor, audio_lengths: torch.Tensor) -> torch.Tensor:
        if self.dither <= 0:
            return waveform
        generator = torch.Generator(device=waveform.device)
        for i in range(waveform.shape[0]):
            valid_samples = min(int(audio_lengths[i].item()), waveform.shape[1])
            if valid_samples <= 0:
                continue
            generator.manual_seed(valid_samples)
            noise = torch.randn(valid_samples, dtype=waveform.dtype, device=waveform.device, generator=generator)
            waveform[i, :valid_samples] += self.dither * noise
        return waveform

    def _torch_extract_fbank_features(self, waveform, device="cpu"):
        # spectrogram
        window = torch.hann_window(self.win_length, periodic=False, device=device)
        stft = torch.stft(
            waveform,
            self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=window,
            return_complex=True,
            pad_mode="constant",
        )
        # Let's match original implementation
        magnitudes = torch.view_as_real(stft)
        magnitudes = torch.sqrt(magnitudes.pow(2).sum(-1))
        magnitudes = magnitudes.pow(2)

        # log mel spectrogram
        mel_filters = self.mel_filters.to(device)
        mel_spec = mel_filters @ magnitudes
        mel_spec = torch.log(mel_spec + LOG_ZERO_GUARD_VALUE)

        # (batch_size, num_mel_filters, num_frames) -> (batch_size, num_frames, num_mel_filters)
        mel_spec = mel_spec.permute(0, 2, 1)

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

                For CohereAsr models, `attention_mask` should always be passed for batched inference, to avoid subtle
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
            raw_speech = [speech.to(torch.float32) for speech in raw_speech]
        else:
            raw_speech = [raw_speech.to(torch.float32)]

        # Chunk long audio at energy-based boundaries
        fast_path_threshold_s = max(0.0, self.max_audio_clip_s - self.overlap_chunk_second)
        audio_chunk_index: list[tuple[int, int | None]] = []
        chunked_speech: list[torch.Tensor] = []
        for sample_idx, speech in enumerate(raw_speech):
            duration_s = speech.shape[0] / self.sampling_rate
            if duration_s <= fast_path_threshold_s:
                chunked_speech.append(speech)
                audio_chunk_index.append((sample_idx, None))
            else:
                chunks = self._split_audio_chunks_energy(speech)
                for chunk_idx, chunk in enumerate(chunks):
                    chunked_speech.append(chunk)
                    audio_chunk_index.append((sample_idx, chunk_idx))

        raw_speech = [speech[:, None] for speech in chunked_speech]

        audio_lengths = [len(speech) for speech in raw_speech]
        batched_speech = BatchFeature({"input_features": raw_speech, "audio_lengths": audio_lengths})

        padded_inputs = self.pad(
            batched_speech,
            padding=padding,
            max_length=max_length,
            truncation=truncation,
            pad_to_multiple_of=pad_to_multiple_of,
            return_tensors="pt",
        )
        input_features = padded_inputs.input_features.squeeze(-1)

        # dithering
        input_features = self._apply_dither(input_features, padded_inputs.audio_lengths)

        # preemphasis
        if self.preemphasis is not None:
            timemask = torch.arange(input_features.shape[1], device=input_features.device).unsqueeze(
                0
            ) < padded_inputs.audio_lengths.unsqueeze(1)
            input_features = torch.cat(
                [input_features[:, :1], input_features[:, 1:] - self.preemphasis * input_features[:, :-1]], dim=1
            )
            input_features = input_features.masked_fill(~timemask, 0.0)

        input_features = self._torch_extract_fbank_features(input_features, device)
        features_lengths = torch.floor_divide(
            padded_inputs.audio_lengths + self.n_fft // 2 * 2 - self.n_fft, self.hop_length
        )
        attention_mask = torch.arange(input_features.shape[1], device=device)[None, :] < features_lengths[:, None]

        # normalize mel features, ignoring padding
        mask = attention_mask.unsqueeze(-1)
        input_features_masked = input_features * mask
        mean = input_features_masked.sum(dim=1) / features_lengths.unsqueeze(-1)
        mean = mean.unsqueeze(1)
        variance = ((input_features_masked - mean) ** 2 * mask).sum(dim=1) / (features_lengths - 1).unsqueeze(-1)
        std = torch.sqrt(variance).unsqueeze(1)
        input_features = (input_features - mean) / (std + EPSILON)
        input_features *= mask

        result = BatchFeature(
            data={
                "input_features": input_features,
                "attention_mask": attention_mask,
            },
            tensor_type=return_tensors,
        )
        result["audio_chunk_index"] = audio_chunk_index
        return result


__all__ = ["CohereAsrFeatureExtractor"]
