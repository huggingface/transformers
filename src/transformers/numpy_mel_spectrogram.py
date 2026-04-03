"""NumPy implementation of mel spectrogram computation."""

import numpy as np
import librosa


# --- Frequency conversion utilities ---

def hertz_to_mel(freq, mel_scale="htk"):
    if mel_scale == "htk":
        return 2595.0 * np.log10(1.0 + (freq / 700.0))
    elif mel_scale == "kaldi":
        return 1127.0 * np.log(1.0 + (freq / 700.0))
    # slaney
    min_log_hertz = 1000.0
    min_log_mel = 15.0
    logstep = 27.0 / np.log(6.4)
    mels = 3.0 * freq / 200.0
    if isinstance(freq, np.ndarray):
        log_region = freq >= min_log_hertz
        mels[log_region] = min_log_mel + np.log(freq[log_region] / min_log_hertz) * logstep
    elif freq >= min_log_hertz:
        mels = min_log_mel + np.log(freq / min_log_hertz) * logstep
    return mels


def mel_to_hertz(mels, mel_scale="htk"):
    if mel_scale == "htk":
        return 700.0 * (np.power(10, mels / 2595.0) - 1.0)
    elif mel_scale == "kaldi":
        return 700.0 * (np.exp(mels / 1127.0) - 1.0)
    # slaney
    min_log_hertz = 1000.0
    min_log_mel = 15.0
    logstep = np.log(6.4) / 27.0
    freq = 200.0 * mels / 3.0
    if isinstance(mels, np.ndarray):
        log_region = mels >= min_log_mel
        freq[log_region] = min_log_hertz * np.exp(logstep * (mels[log_region] - min_log_mel))
    elif mels >= min_log_mel:
        freq = min_log_hertz * np.exp(logstep * (mels - min_log_mel))
    return freq


# --- Filter bank ---

def _create_triangular_filter_bank(fft_freqs, filter_freqs):
    filter_diff = np.diff(filter_freqs)
    slopes = np.expand_dims(filter_freqs, 0) - np.expand_dims(fft_freqs, 1)
    down_slopes = -slopes[:, :-2] / filter_diff[:-1]
    up_slopes = slopes[:, 2:] / filter_diff[1:]
    return np.maximum(0, np.minimum(down_slopes, up_slopes))


def mel_filter_bank(
    num_frequency_bins,
    num_mel_filters,
    min_frequency,
    max_frequency,
    sampling_rate,
    norm=None,
    mel_scale="htk",
    triangularize_in_mel_space=False,
    frequency_bin_mode="rfft",
):
    mel_min = hertz_to_mel(min_frequency, mel_scale=mel_scale)
    mel_max = hertz_to_mel(max_frequency, mel_scale=mel_scale)
    mel_freqs = np.linspace(mel_min, mel_max, num_mel_filters + 2)
    filter_freqs = mel_to_hertz(mel_freqs, mel_scale=mel_scale)

    n_fft = (num_frequency_bins - 1) * 2

    if triangularize_in_mel_space:
        fft_bin_width = sampling_rate / n_fft
        fft_freqs = hertz_to_mel(
            fft_bin_width * np.arange(num_frequency_bins), mel_scale=mel_scale
        )
        filter_freqs = mel_freqs
    elif frequency_bin_mode == "rfft":
        fft_freqs = np.fft.rfftfreq(n=n_fft, d=1.0 / sampling_rate)
    else:
        fft_freqs = np.linspace(0, sampling_rate // 2, num_frequency_bins)

    mel_filters = _create_triangular_filter_bank(fft_freqs, filter_freqs)

    if norm == "slaney":
        enorm = 2.0 / (
            filter_freqs[2 : num_mel_filters + 2] - filter_freqs[:num_mel_filters]
        )
        mel_filters *= np.expand_dims(enorm, 0)

    return mel_filters


# --- Window ---

def window_function(window_length, name="hann_window", periodic=True):
    N = window_length + 1 if periodic else window_length
    fac = np.linspace(-np.pi, np.pi, N)
    if name in ("hann", "hann_window"):
        w = 0.5 + 0.5 * np.cos(fac)
    elif name in ("hamming", "hamming_window"):
        w = 0.54 + 0.46 * np.cos(fac)
    elif name == "boxcar":
        w = np.ones(N)
    elif name == "povey":
        w = (0.5 + 0.5 * np.cos(fac)) ** 0.85
    else:
        raise ValueError(f"Unknown window function '{name}'")
    return w[:window_length] if periodic else w


# --- Sub-methods ---

def _prepare_window_and_framing(window, win_length, n_fft, needs_manual_framing):
    if needs_manual_framing and win_length < n_fft:
        frame_length = win_length
    else:
        if win_length < n_fft:
            left_pad = (n_fft - win_length) // 2
            right_pad = n_fft - win_length - left_pad
            window = np.pad(window, (left_pad, right_pad))
        frame_length = n_fft
    return window, frame_length


def _frame_waveform(waveform, frame_length, hop_length, n_fft, center, pad_mode):
    squeezed = waveform.ndim == 1
    if squeezed:
        waveform = waveform[np.newaxis, :]
    if center:
        # Use librosa-compatible split-padding to match their STFT exactly
        # This replicates librosa's optimization to avoid copying the entire signal
        start_k = int(np.ceil(n_fft // 2 / hop_length))
        tail_k = (waveform.shape[-1] + n_fft // 2 - n_fft) // hop_length + 1

        if tail_k <= start_k:
            # Head and tail overlap, use simple full padding
            waveform = np.pad(waveform, ((0, 0), (frame_length // 2, frame_length // 2)), mode=pad_mode)
            num_frames = 1 + (waveform.shape[-1] - frame_length) // hop_length
            frame_starts = np.arange(num_frames) * hop_length
            frame_indices = frame_starts[:, np.newaxis] + np.arange(frame_length)
            frames = waveform[:, frame_indices]  # (batch, num_frames, frame_length)
        else:
            # Split padding: handle head and tail separately like librosa
            # Pre-padding: left pad only
            padding = [(0, 0) for _ in range(waveform.ndim)]
            padding[-1] = (frame_length // 2, 0)
            y_pre = np.pad(
                waveform[..., : (start_k - 1) * hop_length - n_fft // 2 + n_fft + 1],
                padding,
                mode=pad_mode,
            )
            y_frames_pre = librosa.util.frame(y_pre, frame_length=frame_length, hop_length=hop_length)
            y_frames_pre = y_frames_pre[..., :start_k]
            y_frames_pre = np.moveaxis(y_frames_pre, -2, -1)  # (batch, frame_length, num_frames) -> (batch, num_frames, frame_length)
            extra = y_frames_pre.shape[-2]

            # Post-padding: right pad only
            padding[-1] = (0, frame_length // 2)
            y_post = np.pad(
                waveform[..., (tail_k) * hop_length - n_fft // 2 :],
                padding,
                mode=pad_mode,
            )
            y_frames_post = librosa.util.frame(y_post, frame_length=frame_length, hop_length=hop_length)
            y_frames_post = np.moveaxis(y_frames_post, -2, -1)  # (batch, frame_length, num_frames) -> (batch, num_frames, frame_length)
            extra += y_frames_post.shape[-2]

            # Middle: no padding
            start = start_k * hop_length - n_fft // 2
            y_frames_middle = librosa.util.frame(
                waveform[..., start:], frame_length=frame_length, hop_length=hop_length
            )
            y_frames_middle = np.moveaxis(y_frames_middle, -2, -1)  # (batch, frame_length, num_frames) -> (batch, num_frames, frame_length)

            # Total frames
            num_frames = y_frames_pre.shape[-2] + y_frames_middle.shape[-2] + y_frames_post.shape[-2]

            # Concatenate frames
            frames = np.concatenate([y_frames_pre, y_frames_middle, y_frames_post], axis=-2)
    else:
        # No centering: no padding
        num_frames = 1 + (waveform.shape[-1] - frame_length) // hop_length
        frame_starts = np.arange(num_frames) * hop_length
        frame_indices = frame_starts[:, np.newaxis] + np.arange(frame_length)
        frames = waveform[:, frame_indices]  # (batch, num_frames, frame_length)

    if squeezed:
        frames = frames.squeeze(0)
    return frames, num_frames


def _apply_frame_processing(frames, *, dither=0.0, preemphasis=None, remove_dc_offset=False):
    compute_dtype = frames.dtype
    if dither != 0.0:
        frames = frames + dither * np.random.randn(*frames.shape).astype(compute_dtype)
    if remove_dc_offset:
        frames = frames - frames.mean(axis=-1, keepdims=True)
    if preemphasis is not None:
        preemph_src = preemphasis * frames[..., :-1]
        frames[..., 1:] = frames[..., 1:] - preemph_src
        frames[..., 0] = frames[..., 0] * (1 - preemphasis)
    return frames


def _windowed_fft(frames, window, fft_length, power, normalized):
    """Apply window, compute FFT, and return power spectrogram of shape (..., freq, time)."""
    frames = frames * window
    spec = np.fft.rfft(frames, n=fft_length, axis=-1).astype(np.complex64)
    if normalized:
        spec = spec / np.sqrt(np.sum(window**2)).astype(spec.real.dtype)
    spec = np.abs(spec, dtype=np.float64) ** power
    return np.moveaxis(spec, -1, -2)


def _apply_mel_scale(
    spectrogram: np.ndarray,
    mel_filters: np.ndarray,
    mel_floor: float = 1e-10,
) -> np.ndarray:
    """Apply mel filterbank to a spectrogram.

    Args:
        spectrogram: Power spectrogram of shape (..., freq, time).
        mel_filters: Mel filterbank of shape (freq, n_mels).
        mel_floor: Minimum value for clamping.

    Returns:
        Mel spectrogram of shape (..., n_mels, time).
    """
    # (n_mels, freq) @ (..., freq, time) -> (..., n_mels, time)
    mel_spec = np.matmul(mel_filters.T, spectrogram)
    return np.maximum(mel_floor, mel_spec)


# --- Main function ---

def mel_spectrogram(
    waveform: np.ndarray,
    sampling_rate: int,
    *,
    n_fft: int = 400,
    win_length: int | None = None,
    hop_length: int | None = None,
    window_fn: str = "hann_window",
    wkwargs: dict | None = None,
    power: float = 2.0,
    center: bool = True,
    pad_mode: str = "reflect",
    normalized: bool = False,
    periodic: bool = True,
    # mel scale kwargs
    n_mels: int = 128,
    f_min: float = 0.0,
    f_max: float | None = None,
    mel_scale: str = "htk",
    norm: str | None = None,
    triangularize_in_mel_space: bool = False,
    # kaldi-specific kwargs
    dither: float = 0.0,
    preemphasis: float | None = None,
    remove_dc_offset: bool = False,
    mel_floor: float = 1e-10,
) -> np.ndarray:
    """Compute mel spectrogram using NumPy.

    Args:
        waveform: Input waveform of shape (..., time).
        sampling_rate: Sample rate in Hz.

    Returns:
        Mel spectrogram of shape (..., n_mels, time).
    """
    if f_max is None:
        f_max = sampling_rate / 2.0

    # --- STFT ---
    if win_length is None:
        win_length = n_fft
    if hop_length is None:
        hop_length = win_length // 2
    window = window_function(win_length, name=window_fn, periodic=periodic)

    needs_manual_framing = (dither != 0.0) or (preemphasis is not None) or remove_dc_offset
    window, frame_length = _prepare_window_and_framing(window, win_length, n_fft, needs_manual_framing)

    is_1d = waveform.ndim == 1
    if is_1d:
        waveform = waveform[np.newaxis, :]
    leading_shape = waveform.shape[:-1]
    waveform = waveform.reshape(-1, waveform.shape[-1])
    frames, num_frames = _frame_waveform(waveform, frame_length, hop_length, n_fft, center, pad_mode)
    compute_dtype = np.result_type(waveform.dtype, window.dtype)
    frames = frames.astype(compute_dtype, copy=False)
    frames = _apply_frame_processing(frames, dither=dither, preemphasis=preemphasis, remove_dc_offset=remove_dc_offset)
    spectrogram = _windowed_fft(frames, window, n_fft, power, normalized)

    num_frequency_bins = n_fft // 2 + 1
    spectrogram = spectrogram.reshape(*leading_shape, num_frequency_bins, num_frames)
    if is_1d:
        spectrogram = spectrogram.squeeze(0)

    num_frequency_bins = spectrogram.shape[-2]
    mel_fb = mel_filter_bank(
        num_frequency_bins, n_mels, f_min, f_max, sampling_rate,
        norm=norm, mel_scale=mel_scale,
        triangularize_in_mel_space=triangularize_in_mel_space,
    )

    return _apply_mel_scale(spectrogram, mel_fb, mel_floor=mel_floor)


class MelSpectrogram:
    """Cached mel spectrogram — precomputes window and mel filterbank.

    Same API and exact same results as the functional ``mel_spectrogram``, but
    avoids recomputing the window and mel filterbank on every call.

    Usage::

        transform = MelSpectrogram(sampling_rate=16000, n_fft=1024, n_mels=80)
        mel = transform(waveform)             # fast repeated calls
    """

    def __init__(
        self,
        sampling_rate: int,
        *,
        n_fft: int = 400,
        win_length: int | None = None,
        hop_length: int | None = None,
        window_fn: str = "hann_window",
        wkwargs: dict | None = None,
        power: float = 2.0,
        center: bool = True,
        pad_mode: str = "reflect",
        normalized: bool = False,
        periodic: bool = True,
        n_mels: int = 128,
        f_min: float = 0.0,
        f_max: float | None = None,
        mel_scale: str = "htk",
        norm: str | None = None,
        triangularize_in_mel_space: bool = False,
        dither: float = 0.0,
        preemphasis: float | None = None,
        remove_dc_offset: bool = False,
        mel_floor: float = 1e-10,
    ):
        self.sampling_rate = sampling_rate
        self.n_fft = n_fft
        self.win_length = win_length if win_length is not None else n_fft
        self.hop_length = hop_length if hop_length is not None else self.win_length // 2
        self.power = power
        self.center = center
        self.pad_mode = pad_mode
        self.normalized = normalized
        self.periodic = periodic
        self.n_mels = n_mels
        self.f_min = f_min
        self.f_max = f_max if f_max is not None else sampling_rate / 2.0
        self.mel_floor = mel_floor
        self.dither = dither
        self.preemphasis = preemphasis
        self.remove_dc_offset = remove_dc_offset
        self.window_fn = window_fn

        # Precompute window
        needs_manual_framing = (dither != 0.0) or (preemphasis is not None) or remove_dc_offset
        window = window_function(self.win_length, name=window_fn, periodic=periodic)
        self._window, self._frame_length = _prepare_window_and_framing(
            window, self.win_length, n_fft, needs_manual_framing,
        )

        # Precompute mel filterbank
        num_frequency_bins = n_fft // 2 + 1
        self._mel_fb = mel_filter_bank(
            num_frequency_bins, n_mels, self.f_min, self.f_max, sampling_rate,
            norm=norm, mel_scale=mel_scale,
            triangularize_in_mel_space=triangularize_in_mel_space,
        )

    def __call__(self, waveform: np.ndarray) -> np.ndarray:
        """Compute mel spectrogram.

        Args:
            waveform: Input of shape (..., time).

        Returns:
            Mel spectrogram of shape (..., n_mels, time).
        """
        is_1d = waveform.ndim == 1
        if is_1d:
            waveform = waveform[np.newaxis, :]
        leading_shape = waveform.shape[:-1]
        waveform = waveform.reshape(-1, waveform.shape[-1])
        frames, num_frames = _frame_waveform(
            waveform, self._frame_length, self.hop_length, self.n_fft, self.center, self.pad_mode,
        )
        compute_dtype = np.result_type(waveform.dtype, self._window.dtype)
        frames = frames.astype(compute_dtype, copy=False)
        frames = _apply_frame_processing(
            frames, dither=self.dither, preemphasis=self.preemphasis, remove_dc_offset=self.remove_dc_offset,
        )
        spectrogram = _windowed_fft(frames, self._window, self.n_fft, self.power, self.normalized)

        num_frequency_bins = self.n_fft // 2 + 1
        spectrogram = spectrogram.reshape(*leading_shape, num_frequency_bins, num_frames)
        if is_1d:
            spectrogram = spectrogram.squeeze(0)

        return _apply_mel_scale(spectrogram, self._mel_fb, mel_floor=self.mel_floor)
