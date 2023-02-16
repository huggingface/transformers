# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team.
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
 Audio processing functions to extract feature from a raw audio. Should all be in numpy to support all frameworks, and
 remmove unecessary dependencies.
"""
import math
import warnings
from typing import Optional

import numpy as np
from numpy.fft import fft


def hertz_to_mel(freq: float, mel_scale: str = "htk") -> float:
    """Convert Hertz to Mels.

    Args:
        freqs (`float`):
            Frequencies in Hertz
        mel_scale (`str`, *optional*, defaults to `"htk"`):
            Scale to use, `htk` or `slaney`.

    Returns:
        mels (`float`):
            Frequency in Mels
    """

    if mel_scale not in ["slaney", "htk"]:
        raise ValueError('mel_scale should be one of "htk" or "slaney".')

    if mel_scale == "htk":
        return 2595.0 * math.log10(1.0 + (freq / 700.0))

    # Fill in the linear part
    frequency_min = 0.0
    f_sp = 200.0 / 3

    mels = (freq - frequency_min) / f_sp

    # Fill in the log-scale part
    min_log_hertz = 1000.0
    min_log_mel = (min_log_hertz - frequency_min) / f_sp
    logstep = math.log(6.4) / 27.0

    if freq >= min_log_hertz:
        mels = min_log_mel + math.log(freq / min_log_hertz) / logstep

    return mels


def mel_to_hertz(mels: np.array, mel_scale: str = "htk") -> np.array:
    """Convert mel bin numbers to frequencies.

    Args:
        mels (`np.array`):
            Mel frequencies
        mel_scale (`str`, *optional*, `"htk"`):
            Scale to use: `htk` or `slaney`.

    Returns:
        freqs (`np.array`):
            Mels converted to Hertz
    """

    if mel_scale not in ["slaney", "htk"]:
        raise ValueError('mel_scale should be one of "htk" or "slaney".')

    if mel_scale == "htk":
        return 700.0 * (10.0 ** (mels / 2595.0) - 1.0)

    # Fill in the linear scale
    frequency_min = 0.0
    f_sp = 200.0 / 3
    freqs = frequency_min + f_sp * mels

    # And now the nonlinear scale
    min_log_hertz = 1000.0
    min_log_mel = (min_log_hertz - frequency_min) / f_sp
    logstep = math.log(6.4) / 27.0

    log_t = mels >= min_log_mel
    freqs[log_t] = min_log_hertz * np.exp(logstep * (mels[log_t] - min_log_mel))

    return freqs


def _create_triangular_filterbank(
    all_freqs: np.array,
    f_pts: np.array,
) -> np.array:
    """Create a triangular filter bank.


    Args:
        all_freqs (`np.array` of shape (`nb_frequency_bins`, )):
            Discrete frequencies used when the STFT was computed.
        f_pts (`np.array`, of shape (`nb_mel_filters`, )):
            Coordinates of the middle points of the triangular filters to create.

    Returns:
        fb (np.array):
            The filter bank of size (`nb_frequency_bins`, `nb_mel_filters`).
    """
    # Adapted from Librosa
    # calculate the difference between each filter mid point and each stft freq point in hertz
    f_diff = f_pts[1:] - f_pts[:-1]  # (n_filter + 1)
    slopes = np.expand_dims(f_pts, 0) - np.expand_dims(all_freqs, 1)  # (nb_frequency_bins, n_filter + 2)
    # create overlapping triangles
    zero = np.zeros(1)
    down_slopes = (-1.0 * slopes[:, :-2]) / f_diff[:-1]  # (nb_frequency_bins, n_filter)
    up_slopes = slopes[:, 2:] / f_diff[1:]  # (nb_frequency_bins, n_filter)
    fb = np.maximum(zero, np.minimum(down_slopes, up_slopes))

    return fb


def get_mel_filter_banks(
    nb_frequency_bins: int,
    nb_mel_filters: int,
    frequency_min: float,
    frequency_max: float,
    sample_rate: int,
    norm: Optional[str] = None,
    mel_scale: str = "htk",
) -> np.array:
    """
    Create a frequency bin conversion matrix used to obtain the Mel Spectrogram. This is called a *mel filter bank*,
    and various implementation exist, which differ in the number of filters, the shape of the filters, the way the
    filters are spaced, the bandwidth of the filters, and the manner in which the spectrum is warped. The goal of these
    features is to approximate the non-linear human perception of the variation in pitch with respect to the frequency.
    This code is heavily inspired from the *torchaudio* implementation, see
    [here](https://pytorch.org/audio/stable/transforms.html) for more details.


    Tips:
        - Different banks of Mel filters were introduced in the litterature. The following variation are supported:
            - MFCC FB-20: introduced in 1980 by Davis and Mermelstein, it assumes a sampling frequency of 10 kHertz
            and a speech bandwidth of `[0, 4600]` Hertz
            - MFCC FB-24 HTK: from the Cambridge HMM Toolkit (HTK) (1995) uses a filter bank of 24 filters for a
            speech bandwidth `[0, 8000]` Hertz (sampling rate ≥ 16 kHertz).
            - MFCC FB-40: from the Auditory Toolbox for MATLAB written by Slaney in 1998, assumes a sampling rate
            of 16 kHertz, and speech bandwidth [133, 6854] Hertz. This version also includes an area normalization.
            - HFCC-E FB-29 (Human Factor Cepstral Coefficients) of Skowronski and Harris (2004), assumes sampling
            rate of 12.5 kHertz and speech bandwidth [0, 6250] Hertz
        - The default parameters of `torchaudio`'s mel filterbanks implement the `"htk"` filers while `torchlibrosa`
        uses the `"slaney"` implementation.

    Args:
        nb_frequency_bins (`int`):
            Number of frequencies used to compute the spectrogram (should be the same as in `stft`).
        nb_mel_filters (`int`):
            Number of Mel filers to generate.
        frequency_min (`float`):
            Minimum frequency of interest(Hertz).
        frequency_max (`float`):
            Maximum frequency of interest(Hertz).
        sample_rate (`int`):
            Sample rate of the audio waveform.
        norm (`str`, *optional*):
            If "slaney", divide the triangular Mel weights by the width of the mel band (area normalization).
        mel_scale (`str`, *optional*, defaults to `"htk"`):
            Scale to use: `"htk"` or `"slaney"`.

    Returns:
        `np.ndarray`: Triangular filter banks (fb matrix) of shape (`nb_frequency_bins`, `nb_mel_filters`). This matrix
        is a projection matrix to go from a spectrogram to a Mel Spectrogram.

    """

    if norm is not None and norm != "slaney":
        raise ValueError('norm must be one of None or "slaney"')

    # freqency bins
    all_freqs = np.linspace(0, sample_rate // 2, nb_frequency_bins)

    # Compute mim and max frequencies in mel scale
    m_min = hertz_to_mel(frequency_min, mel_scale=mel_scale)
    m_max = hertz_to_mel(frequency_max, mel_scale=mel_scale)

    # create the centers of the triangular mel filters.
    m_pts = np.linspace(m_min, m_max, nb_mel_filters + 2)
    f_pts = mel_to_hertz(m_pts, mel_scale=mel_scale)

    # create the filterbank
    filterbank = _create_triangular_filterbank(all_freqs, f_pts)

    if norm is not None and norm == "slaney":
        # Slaney-style mel is scaled to be approx constant energy per channel
        enorm = 2.0 / (f_pts[2 : nb_mel_filters + 2] - f_pts[:nb_mel_filters])
        filterbank *= np.expand_dims(enorm, 0)

    if (filterbank.max(axis=0) == 0.0).any():
        warnings.warn(
            "At least one mel filterbank has all zero values. "
            f"The value for `nb_mel_filters` ({nb_mel_filters}) may be set too high. "
            f"Or, the value for `nb_frequency_bins` ({nb_frequency_bins}) may be set too low."
        )

    return filterbank


def power_to_db(mel_spectrogram, top_db=None, a_min=1e-10, ref=1.0):
    """
    Convert a mel spectrogram from power to db scale, this function is the numpy implementation of librosa.power_to_lb.
    It computes `10 * log10(mel_spectrogram / ref)`, using basic log properties for stability.

    Tips:
        - The motivation behind applying the log function on the mel spectrogram is that humans do not hear loudness on
          a
        linear scale. Generally to double the percieved volume of a sound we need to put 8 times as much energy into
        it.
        - This means that large variations in energy may not sound all that different if the sound is loud to begin
        with. This compression operation makes the mel features match more closely what humans actually hear.

    Args:
        mel_spectrogram (`np.array`):
            Input mel spectrogram.
        top_db (`int`, *optional*):
            The maximum decibel value.
        a_min (`int`, *optional*, default to 1e-10):
            Minimum value to use when cliping the mel spectrogram.
        ref (`float`, *optional*, default to 1.0):
            Maximum reference value used to scale the mel_spectrogram.

    """
    log_spec = 10 * np.log10(np.clip(mel_spectrogram, a_min=a_min, a_max=None))
    log_spec -= 10.0 * np.log10(np.maximum(a_min, ref))
    if top_db is not None:
        if top_db < 0:
            raise ValueError("top_db must be non-negative")
        log_spec = np.clip(log_spec, min=np.maximum(log_spec) - top_db, max=np.inf)
    return log_spec


# TODO @ArthurZucker: This method does not support batching yet as we are mainly focus on inference.
def fram_wave(waveform: np.array, hop_length: int = 160, fft_window_size: int = 400, center: bool = True):
    """
    In order to compute the short time fourier transform, the waveform needs to be split in overlapping windowed
    segments called `frames`.

    The window length (window_length) defines how much of the signal is contained in each frame, while the hop length
    defines the step between the beginning of each new frame.


    Args:
        waveform (`np.array` of shape `(sample_length,)`):
            The raw waveform which will be split into smaller chunks.
        hop_length (`int`, *optional*, defaults to 160):
            Step between each window of the waveform.
        fft_window_size (`int`, *optional*, defaults to 400):
            Defines the size of the window.
        center (`bool`, defaults to `True`):
            Whether or not to center each frame around the middle of the frame. Centering is done by reflecting the
            waveform on the left and on the right.

    Return:
        framed_waveform (`np.array` of shape `(waveform.shape // hop_length , fft_window_size)`):
            The framed waveforms that can be fed to `np.fft`.
    """
    frames = []
    for i in range(0, waveform.shape[0] + 1, hop_length):
        if center:
            half_window = (fft_window_size - 1) // 2 + 1
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
            frame = waveform[i : i + fft_window_size]
            frame_width = frame.shape[0]
            if frame_width < waveform.shape[0]:
                frame = np.lib.pad(
                    frame, pad_width=(0, fft_window_size - frame_width), mode="constant", constant_values=0
                )
        frames.append(frame)

    frames = np.stack(frames, 0)
    return frames


# TODO @ArthurZucker: This method does not support batching yet as we are mainly focus on inference.


def stft(frames: np.array, windowing_function: np.array, fft_window_size: int = None):
    """
    Calculates the complex Short-Time Fourier Transform (STFT) of the given framed signal. Should give the same results
    as `torch.stft`.

    Args:
        frames (`np.array` of dimension `(num_frames, fft_window_size)`):
            A framed audio signal obtained using `audio_utils.fram_wav`.
        windowing_function (`np.array` of dimension `(nb_frequency_bins, nb_mel_filters)`:
            A array reprensenting the function that will be used to reduces the amplitude of the discontinuities at the
            boundaries of each frame when computing the STFT. Each frame will be multiplied by the windowing_function.
            For more information on the discontinuities, called *Spectral leakage*, refer to [this
            tutorial]https://download.ni.com/evaluation/pxi/Understanding%20FFTs%20and%20Windowing.pdf
        fft_window_size (`int`, *optional*):
            Size of the window om which the Fourier transform is applied. This controls the frequency resolution of the
            spectrogram. 400 means that the fourrier transform is computed on windows of 400 samples. The number of
            frequency bins (`nb_frequency_bins`) used to divide the window into equal strips is equal to
            `(1+fft_window_size)//2`. An increase of the fft_window_size slows the calculus time proportionnally.

    Example:

    ```python
    >>> from transformers.audio_utils import stft, fram_wave
    >>> import numpy as np

    >>> audio = np.random.rand(50)
    >>> fft_window_size = 10
    >>> hop_length = 2
    >>> framed_audio = fram_wave(audio, hop_length, fft_window_size)
    >>> spectrogram = stft(framed_audio, np.hanning(fft_window_size + 1))
    ```

    Returns:
        spectrogram (`np.ndarray`):
            A spectrogram of shape `(num_frames, nb_frequency_bins)` obtained using the STFT algorithm
    """
    frame_size = frames.shape[1]

    if fft_window_size is None:
        fft_window_size = frame_size

    if fft_window_size < frame_size:
        raise ValueError("FFT size must greater or equal the frame size")
    # number of FFT bins to store
    nb_frequency_bins = (fft_window_size >> 1) + 1

    spectrogram = np.empty((len(frames), nb_frequency_bins), dtype=np.complex64)
    fft_signal = np.zeros(fft_window_size)

    for f, frame in enumerate(frames):
        if windowing_function is not None:
            np.multiply(frame, windowing_function, out=fft_signal[:frame_size])
        else:
            fft_signal[:frame_size] = frame
        spectrogram[f] = fft(fft_signal, axis=0)[:nb_frequency_bins]
    return spectrogram.T
