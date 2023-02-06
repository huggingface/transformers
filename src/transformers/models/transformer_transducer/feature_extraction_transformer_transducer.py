import math
import warnings
from typing import List, Optional, Union

import numpy as np
from numpy.fft import fft

from ...feature_extraction_sequence_utils import SequenceFeatureExtractor
from ...feature_extraction_utils import BatchFeature
from ...utils import TensorType, logging


logger = logging.get_logger(__name__)


# [NOTE]: copied from https://github.com/pytorch/audio/blob/main/torchaudio/functional/functional.py#L415-L446
def hz_to_mel(freq: float, mel_scale: str = "htk") -> float:
    r"""Convert Hz to Mels.

    Args:
        freqs (float): Frequencies in Hz
        mel_scale (str, optional): Scale to use: ``htk`` or ``slaney``. (Default: ``htk``)
    Returns:
        mels (float): Frequency in Mels
    """

    if mel_scale not in ["slaney", "htk"]:
        raise ValueError('mel_scale should be one of "htk" or "slaney".')

    if mel_scale == "htk":
        return 2595.0 * math.log10(1.0 + (freq / 700.0))

    # Fill in the linear part
    f_min = 0.0
    f_sp = 200.0 / 3

    mels = (freq - f_min) / f_sp

    # Fill in the log-scale part
    min_log_hz = 1000.0
    min_log_mel = (min_log_hz - f_min) / f_sp
    logstep = math.log(6.4) / 27.0

    if freq >= min_log_hz:
        mels = min_log_mel + math.log(freq / min_log_hz) / logstep

    return mels


# [NOTE]: copied from https://github.com/pytorch/audio/blob/main/torchaudio/functional/functional.py#L449-L479
def mel_to_hz(mels: np.ndarray, mel_scale: str = "htk") -> np.ndarray:
    """Convert mel bin numbers to frequencies.

    Args:
        mels (Tensor): Mel frequencies
        mel_scale (str, optional): Scale to use: ``htk`` or ``slaney``. (Default: ``htk``)
    Returns:
        freqs (Tensor): Mels converted in Hz
    """

    if mel_scale not in ["slaney", "htk"]:
        raise ValueError('mel_scale should be one of "htk" or "slaney".')

    if mel_scale == "htk":
        return 700.0 * (10.0 ** (mels / 2595.0) - 1.0)

    # Fill in the linear scale
    f_min = 0.0
    f_sp = 200.0 / 3
    freqs = f_min + f_sp * mels

    # And now the nonlinear scale
    min_log_hz = 1000.0
    min_log_mel = (min_log_hz - f_min) / f_sp
    logstep = math.log(6.4) / 27.0

    log_t = mels >= min_log_mel
    freqs[log_t] = min_log_hz * np.exp(logstep * (mels[log_t] - min_log_mel))

    return freqs


# [NOTE]: At first, it was modeled after Whisper's extractor,
#         but later, it included the emformer's extractor, which made both torchaudio and numpy possible

# [NOTE]: copied from whisper extractor
#         It can be driven by pytorch or numpy.
class TransformerTransducerFeatureExtractor(SequenceFeatureExtractor):
    model_input_names = ["input_features", "attention_mask"]

    def __init__(
        self,
        n_fft: int = 400,
        feature_size: int = 80,
        sampling_rate: int = 16000,
        hop_length: int = 160,
        stack: int = 4,
        stride: int = 3,
        power: int = 2.0,
        center: bool = True,
        mel_scale: str = "slaney",
        filter_norm: Optional[str] = None,
        min_frequency: float = 0.0,
        max_frequency: Optional[float] = None,
        padding_value: float = 0.0,
        return_attention_mask: bool = False,  # pad inputs to max length with silence token (zero) and no attention mask
        **kwargs,
    ):
        """"""
        super().__init__(
            feature_size=feature_size,
            sampling_rate=sampling_rate,
            padding_value=padding_value,
            return_attention_mask=return_attention_mask,
            **kwargs,
        )
        if not (stack and stride):
            raise ValueError("you must be set stride, stack, if you need to compress log_mel")

        # [NOTE]: for Log-MelSpectrogram
        self.n_fft = n_fft
        self.feature_size = feature_size
        self.sampling_rate = sampling_rate
        self.hop_length = hop_length
        self.mel_scale = mel_scale
        self.center = center
        self.power = power
        self.filter_norm = filter_norm
        self.max_frequency = max_frequency if max_frequency else float(sampling_rate // 2)
        self.min_frequency = min_frequency

        # [NOTE]: for window at spectrogram
        self.window_fn = np.hanning(self.n_fft)
        self.mel_filter = self.get_mel_filter(n_mels=feature_size, scale=self.mel_scale, norm=self.filter_norm)

        # [NOTE]: for compressor
        self.stack = stack
        self.stride = stride

    def mel_compressor(self, mel_spectrogram: Union[np.ndarray, List[List[float]]]) -> np.ndarray:
        """"""
        # [NOTE]: 여기서 각각의 멜을 windowing + padding한 뒤 나머지 compress_mel을 padding하는 방식으로 진행해야 할 듯 하다.
        #         compress_features에서 padding을 처리하지 않고 하기에는 self._pad를 overriding해서 새로 만들어야 함.

        if isinstance(mel_spectrogram, list):
            mel_spectrogram = np.array(mel_spectrogram)

        time_steps, _ = mel_spectrogram.shape
        expected_len = math.ceil(time_steps / self.stride)

        mel_store = list()
        for stack_num in range(self.stack):
            idx_iter = range(0, (time_steps - stack_num), self.stride)
            indices = [stack_num + idx for idx in idx_iter]

            features = mel_spectrogram[indices]
            pad_width = ((0, expected_len - features.shape[0]), (0, 0))
            features = np.pad(features, pad_width)

            mel_store.append(features)

        padded_feature = np.concatenate(mel_store, axis=1)
        return padded_feature

    def get_mel_filter(
        self,
        n_mels: int = 128,
        scale: str = "slaney",
        norm: Optional[str] = None,
    ) -> np.ndarray:
        """"""
        n_freqs = (self.n_fft // 2) + 1
        fb = np.zeros((n_freqs, n_mels), dtype=np.float32)
        all_freqs = np.linspace(0, self.sampling_rate // 2, n_freqs)

        m_min = hz_to_mel(self.min_frequency, scale)
        m_max = hz_to_mel(self.max_frequency, scale)

        # n_mels is chennel_size
        m_pts = np.linspace(m_min, m_max, n_mels + 2)
        freqs = mel_to_hz(m_pts, scale)

        f_diff = freqs[1:] - freqs[:-1]
        slopes = freqs[np.newaxis, :] - all_freqs[:, np.newaxis]

        for idx in range(n_mels):
            # lower and upper slopes for all bins
            lower_slope = (-1.0 * slopes[:, :-2][:, idx]) / f_diff[idx]  # 맨 마지막의 1은 n_mels덕분에 자동으로 없어짐
            upper_slope = slopes[:, idx + 2] / f_diff[idx + 1]

            # .. then intersect them with each other and zero
            fb[:, idx] = np.maximum(0, np.minimum(lower_slope, upper_slope))

        # [NOTE]: copied from https://github.com/pytorch/audio/blob/main/torchaudio/functional/functional.py#L565-L575
        if norm is not None and norm == "slaney":
            enorm = 2.0 / (n_mels[2 : n_mels + 2] - n_mels[:n_mels])
            fb *= enorm[:, np.newaxis]

        if (fb.max(axis=0) == 0.0).any():
            warnings.warn(
                "At least one mel filterbank has all zero values. "
                f"The value for `n_mels` ({n_mels}) may be set too high. "
                f"Or, the value for `n_freqs` ({n_freqs}) may be set too low."
            )

        # Slaney-style mel is scaled to be approx constant energy per channel
        fb = np.transpose(fb, (1, 0))  # it's for matmul
        return fb

    # [NOTE]: copied from https://github.com/huggingface/transformers/blob/main/src/transformers/models/whisper/feature_extraction_whisper.py#L135-L169
    def frame_wave(self, waveform, center=True):
        """
        Transform a raw waveform into a list of smaller waveforms. The window length defines how much of the signal is
        contain in each frame (smalle waveform), while the hope length defines the step between the beginning of each
        new frame.

        Centering is done by reflecting the waveform which is first centered around `frame_idx * hop_length`.
        """
        frames = []
        for i in range(0, waveform.shape[0] + 1, self.hop_length):
            half_window = ((self.n_fft - 1) // 2) + 1

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

    # [NOTE]: copied from https://github.com/huggingface/transformers/blob/main/src/transformers/models/whisper/feature_extraction_whisper.py#L171-L196
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

    # [NOTE]: copied from https://github.com/huggingface/transformers/blob/main/src/transformers/models/whisper/feature_extraction_whisper.py#L198-L216
    def get_mel_spectrogram(self, waveform: np.ndarray) -> np.ndarray:
        """
        Compute the log-Mel spectrogram of the provided audio, gives similar results whisper's original torch
        implementation with 1e-5 tolerance.
        """
        # [NOTE]: Mel-Spectrogram
        frames = self.frame_wave(waveform, center=self.center)
        stft = self.stft(frames, self.window_fn)

        magnitudes = np.abs(stft) ** self.power
        mel_spec = np.matmul(self.mel_filter, magnitudes)

        return mel_spec

    def apply_log(self, mel_spectrogram: np.ndarray) -> np.ndarray:
        """"""
        log_mel = np.log10(np.clip(mel_spectrogram, a_min=1e-10, a_max=None))
        log_mel = np.maximum(log_mel, log_mel.max() - 8.0)

        # [NOTE]: apply log scale
        log_mel = (log_mel + 4.0) / 4.0
        return log_mel

    def log_mel_transform(
        self, raw_speechs: Union[List[np.ndarray], np.ndarray], do_numpy: bool = False
    ) -> np.ndarray:
        # [NOTE]: do_numpy is temporarily left because data preprocessing is executed again when the code of the preprocessor changes.
        """"""
        if not isinstance(raw_speechs, list):
            raw_speechs = [raw_speechs]

        mel_spectrogram = [self.get_mel_spectrogram(waveform) for waveform in raw_speechs]
        log_mel_spectrograms = [self.apply_log(mel[:, :-1]) for mel in mel_spectrogram]

        # transpose for model
        log_mel_spectrograms = [np.transpose(log_mel, (1, 0)) for log_mel in log_mel_spectrograms]

        return log_mel_spectrograms

    def __call__(
        self,
        raw_speech: Union[np.ndarray, List[float], List[np.ndarray], List[List[float]]],
        truncation: bool = False,
        pad_to_multiple_of: Optional[int] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        return_attention_mask: Optional[bool] = None,
        padding: Optional[Union[str, bool]] = True,
        max_length: Optional[int] = 512,
        sampling_rate: Optional[int] = None,
        **kwargs,
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
                    f"The model corresponding to this feature extractor: {self} was trained using a sampling rate of"
                    f" {self.sampling_rate}. Please make sure that the provided `raw_speech` input was sampled with"
                    f" {self.sampling_rate} and not {sampling_rate}."
                )
        else:
            logger.warning(
                "It is strongly recommended to pass the `sampling_rate` argument to this function. "
                "Failing to do so can result in silent errors that might be hard to debug."
            )

        if padding == "max_length":
            logger.warning(
                "If PaddingStrategy is max_length, padding may not proceed normally."
                "If time_seq is longer than max_length, padding may not proceed normally."
            )

        is_batched = bool(
            isinstance(raw_speech, (list, tuple))
            and (isinstance(raw_speech[0], np.ndarray) or isinstance(raw_speech[0], (tuple, list)))
        )

        if is_batched:
            raw_speech = [np.asarray(speech, dtype=np.float32) for speech in raw_speech]
        elif not is_batched and not isinstance(raw_speech, np.ndarray):
            raw_speech = np.asarray(raw_speech, dtype=np.float32)
        elif isinstance(raw_speech, np.ndarray) and raw_speech.dtype is np.dtype(np.float64):
            raw_speech = raw_speech.astype(np.float32)

        # always return batch
        if not is_batched:
            raw_speech = [raw_speech]

        log_mel_features = self.log_mel_transform(raw_speech)
        compressed_features = [self.mel_compressor(log_mel) for log_mel in log_mel_features]
        batched_mel = BatchFeature({"input_features": compressed_features})

        padded_inputs = self.pad(
            batched_mel,
            padding=padding,
            max_length=max_length,
            truncation=truncation,
            pad_to_multiple_of=pad_to_multiple_of,
            return_attention_mask=return_attention_mask,
            **kwargs,
        )

        input_features = padded_inputs.get("input_features")
        if isinstance(input_features[0], List):
            padded_inputs["input_features"] = [np.asarray(feature, dtype=np.float32) for feature in input_features]
        else:
            padded_inputs["input_features"] = input_features

        if return_tensors is not None:
            padded_inputs = padded_inputs.convert_to_tensors(return_tensors)

        return padded_inputs
