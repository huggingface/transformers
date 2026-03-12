"""PyTorch implementation of mel spectrogram computation."""

import math

import torch


# --- Frequency conversion utilities ---

def _hertz_to_mel_scalar(freq: float, mel_scale: str = "htk") -> float:
    """Convert a single Hz value to mel using Python math (float64)."""
    if mel_scale == "htk":
        return 2595.0 * math.log10(1.0 + freq / 700.0)
    elif mel_scale == "kaldi":
        return 1127.0 * math.log(1.0 + freq / 700.0)
    # slaney
    f_sp = 200.0 / 3
    min_log_hz = 1000.0
    min_log_mel = (min_log_hz - 0.0) / f_sp
    logstep = math.log(6.4) / 27.0
    if freq >= min_log_hz:
        return min_log_mel + math.log(freq / min_log_hz) / logstep
    return (freq - 0.0) / f_sp


def hertz_to_mel(freq: torch.Tensor, mel_scale: str = "htk") -> torch.Tensor:
    if mel_scale == "htk":
        return 2595.0 * torch.log10(1.0 + freq / 700.0)
    elif mel_scale == "kaldi":
        return 1127.0 * torch.log(1.0 + freq / 700.0)
    # slaney
    min_log_hertz = 1000.0
    min_log_mel = 15.0
    logstep = 27.0 / torch.log(torch.tensor(6.4))
    mels = 3.0 * freq / 200.0
    log_region = freq >= min_log_hertz
    mels[log_region] = min_log_mel + torch.log(freq[log_region] / min_log_hertz) * logstep
    return mels


def mel_to_hertz(mels: torch.Tensor, mel_scale: str = "htk") -> torch.Tensor:
    if mel_scale == "htk":
        return 700.0 * (10.0 ** (mels / 2595.0) - 1.0)
    elif mel_scale == "kaldi":
        return 700.0 * (torch.exp(mels / 1127.0) - 1.0)
    # slaney
    f_sp = 200.0 / 3
    min_log_hz = 1000.0
    min_log_mel = (min_log_hz - 0.0) / f_sp
    logstep = math.log(6.4) / 27.0
    freq = 0.0 + f_sp * mels
    log_region = mels >= min_log_mel
    freq[log_region] = min_log_hz * torch.exp(logstep * (mels[log_region] - min_log_mel))
    return freq


def _create_triangular_filter_bank(
    fft_freqs: torch.Tensor, filter_freqs: torch.Tensor
) -> torch.Tensor:
    filter_diff = filter_freqs[1:] - filter_freqs[:-1]
    slopes = filter_freqs.unsqueeze(0) - fft_freqs.unsqueeze(1)
    down_slopes = -slopes[:, :-2] / filter_diff[:-1]
    up_slopes = slopes[:, 2:] / filter_diff[1:]
    return torch.clamp(torch.minimum(down_slopes, up_slopes), min=0)


def _kaldi_mel_filter_bank(
    num_frequency_bins: int,
    num_mel_filters: int,
    min_frequency: float,
    max_frequency: float,
    sampling_rate: int,
) -> torch.Tensor:
    """Compute mel filter bank matching kaldi's exact construction.

    Replicates torchaudio.compliance.kaldi.get_mel_banks exactly:
    - Uses 1127*ln mel scale (not 2595*log10)
    - Computes mel points via mel_low + i * delta (not torch.linspace)
    - Uses n_fft/2 FFT bins (excludes Nyquist), then pads with zero column

    Returns:
        Tensor of shape (num_frequency_bins, num_mel_filters).
    """
    n_fft = (num_frequency_bins - 1) * 2
    num_fft_bins = n_fft // 2  # kaldi excludes Nyquist bin
    fft_bin_width = sampling_rate / n_fft

    mel_low = 1127.0 * math.log(1.0 + min_frequency / 700.0)
    mel_high = 1127.0 * math.log(1.0 + max_frequency / 700.0)
    mel_delta = (mel_high - mel_low) / (num_mel_filters + 1)

    bin_idx = torch.arange(num_mel_filters).unsqueeze(1)
    left_mel = mel_low + bin_idx * mel_delta
    center_mel = mel_low + (bin_idx + 1.0) * mel_delta
    right_mel = mel_low + (bin_idx + 2.0) * mel_delta

    mel = 1127.0 * (1.0 + fft_bin_width * torch.arange(num_fft_bins) / 700.0).log()
    mel = mel.unsqueeze(0)

    up_slope = (mel - left_mel) / (center_mel - left_mel)
    down_slope = (right_mel - mel) / (right_mel - center_mel)
    banks = torch.max(torch.zeros(1), torch.min(up_slope, down_slope))

    # kaldi pads a zero column for the Nyquist bin
    banks = torch.nn.functional.pad(banks, (0, 1), mode="constant", value=0)

    return banks.T  # (num_frequency_bins, num_mel_filters)


def mel_filter_bank_torch(
    num_frequency_bins: int,
    num_mel_filters: int,
    min_frequency: float,
    max_frequency: float,
    sampling_rate: int,
    norm: str | None = None,
    mel_scale: str = "htk",
    triangularize_in_mel_space: bool = False,
    frequency_bin_mode: str = "rfft",
    computation_dtype: "torch.dtype | None" = None,
    bands_to_zero: int = 0,
) -> torch.Tensor:
    """Compute mel filter bank as a pure PyTorch tensor.

    Matches torchaudio's melscale_fbanks: mel range endpoints are computed in
    float64 (Python math), then all tensor work is done in the default dtype
    (float32).

    Args:
        computation_dtype: If provided, all intermediate tensor operations are
            performed in this dtype (e.g. ``torch.float64``), and the result is
            cast back to the default dtype. This is useful to obtain results
            that are numerically identical to a NumPy (float64) reference
            implementation.
        bands_to_zero: Number of lowest frequency bins to zero out before
            building the filter bank. The zeroed rows are restored (as zeros)
            in the output. Set to 1 to exclude the DC bin (HTK / LASR style).

    Returns:
        Tensor of shape (num_frequency_bins, num_mel_filters).
    """
    if triangularize_in_mel_space and bands_to_zero == 0:
        # Kaldi-exact path: matches torchaudio.compliance.kaldi.get_mel_banks.
        # Kept for backward compatibility with models that rely on this behaviour
        # (AST, SeamlessM4T, Speech2Text, etc.).
        return _kaldi_mel_filter_bank(
            num_frequency_bins, num_mel_filters, min_frequency, max_frequency, sampling_rate,
        )

    mel_min = _hertz_to_mel_scalar(min_frequency, mel_scale=mel_scale)
    mel_max = _hertz_to_mel_scalar(max_frequency, mel_scale=mel_scale)

    n_fft = (num_frequency_bins - 1) * 2

    if triangularize_in_mel_space:
        # Kaldi-style direct slope computation in mel space.
        # Uses mel_low + i * delta (not linspace) and direct per-band slopes
        # to match the exact numerical behaviour of kaldi/HTK filter banks.
        mel_delta = (mel_max - mel_min) / (num_mel_filters + 1)
        bin_idx = torch.arange(num_mel_filters, dtype=computation_dtype).unsqueeze(1)
        left_mel = mel_min + bin_idx * mel_delta
        center_mel = mel_min + (bin_idx + 1.0) * mel_delta
        right_mel = mel_min + (bin_idx + 2.0) * mel_delta

        fft_bin_width = sampling_rate / n_fft
        num_fft_bins = num_frequency_bins - bands_to_zero
        hz_freqs = fft_bin_width * torch.arange(bands_to_zero, num_frequency_bins, dtype=computation_dtype)
        mel = hertz_to_mel(hz_freqs, mel_scale=mel_scale).unsqueeze(0)

        up_slope = (mel - left_mel) / (center_mel - left_mel)
        down_slope = (right_mel - mel) / (right_mel - center_mel)
        mel_filters = torch.max(torch.zeros(1, dtype=computation_dtype), torch.min(up_slope, down_slope))

        # Transpose to (num_fft_bins, num_mel_filters) and restore zeroed bands
        mel_filters = mel_filters.T
        if bands_to_zero > 0:
            mel_filters = torch.nn.functional.pad(mel_filters, (0, 0, bands_to_zero, 0))

        return mel_filters

    mel_freqs = torch.linspace(mel_min, mel_max, num_mel_filters + 2, dtype=computation_dtype)

    filter_freqs = mel_to_hertz(mel_freqs, mel_scale=mel_scale)
    if frequency_bin_mode == "rfft":
        fft_freqs = torch.fft.rfftfreq(n=n_fft, d=1.0 / sampling_rate)
    else:
        fft_freqs = torch.linspace(0, sampling_rate // 2, num_frequency_bins)
    if computation_dtype is not None:
        fft_freqs = fft_freqs.to(computation_dtype)

    mel_filters = _create_triangular_filter_bank(fft_freqs, filter_freqs)

    if norm == "slaney":
        enorm = 2.0 / (filter_freqs[2 : num_mel_filters + 2] - filter_freqs[:num_mel_filters])
        mel_filters = mel_filters * enorm.unsqueeze(0)

    if bands_to_zero > 0:
        mel_filters = torch.nn.functional.pad(mel_filters, (0, 0, bands_to_zero, 0))

    return mel_filters


def window_function(window_length, name="hann_window", periodic=True, wkwargs=None):
    """Create a window tensor using torch window functions."""
    if wkwargs is None:
        wkwargs = {}
    if name in ["hann", "hann_window"]:
        return torch.hann_window(window_length, periodic=periodic, **wkwargs)
    elif name in ["hamming", "hamming_window"]:
        return torch.hamming_window(window_length, periodic=periodic, **wkwargs)
    elif name == "boxcar":
        return torch.ones(window_length)
    elif name == "povey":
        return torch.hann_window(window_length, periodic=periodic, **wkwargs).pow(0.85)
    else:
        raise ValueError(f"Unknown window function '{name}'")


# --- Sub-methods ---

def _extract_spectrogram(
    waveform: torch.Tensor,
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
    pad: int = 0,
    periodic: bool = True,
    dither: float = 0.0,
    preemphasis: float | None = None,
    remove_dc_offset: bool = False,
    computation_dtype: "torch.dtype | None" = None,
    left_align_fft: bool = False,
) -> torch.Tensor:
    """Compute the (power) spectrogram via STFT.

    Args:
        waveform: Input waveform of shape (..., time).
        sampling_rate: Sample rate in Hz.
        left_align_fft: If True, use manual framing with unfold(win_length) + zero-pad
            right + rfft(n_fft). This left-aligns the window in the FFT buffer (kaldi
            style), instead of center-padding it (torch.stft default).

    Returns:
        Power spectrogram of shape (..., freq, time).
    """
    if win_length is None:
        win_length = n_fft
    if hop_length is None:
        hop_length = win_length // 2
    if computation_dtype is not None:
        waveform = waveform.to(computation_dtype)
    device = waveform.device
    dtype = waveform.dtype

    needs_manual_framing = (dither != 0.0) or (preemphasis is not None) or remove_dc_offset or left_align_fft

    window_wkwargs = {**(wkwargs or {}), "dtype": dtype}
    window = window_function(win_length, name=window_fn, periodic=periodic, wkwargs=window_wkwargs)
    window = window.to(device=device)

    if needs_manual_framing and win_length < n_fft:
        frame_length = win_length
    else:
        if win_length < n_fft:
            left_pad = (n_fft - win_length) // 2
            right_pad = n_fft - win_length - left_pad
            window = torch.nn.functional.pad(window, (left_pad, right_pad))
        frame_length = n_fft

    fft_length = n_fft
    num_frequency_bins = (fft_length // 2) + 1

    is_1d = waveform.ndim == 1
    if is_1d:
        waveform = waveform.unsqueeze(0)

    leading_shape = waveform.shape[:-1]
    waveform = waveform.reshape(-1, waveform.shape[-1])

    if pad > 0:
        waveform = torch.nn.functional.pad(waveform, (pad, pad))

    if needs_manual_framing:
        result = _manual_stft(
            waveform, window, frame_length, hop_length, fft_length,
            num_frequency_bins, power, normalized, center, pad_mode,
            dither, preemphasis, remove_dc_offset,
        )
    else:
        result = _torch_stft(
            waveform, window, frame_length, hop_length, fft_length,
            power, normalized, center, pad_mode,
        )

    result = result.reshape(*leading_shape, result.shape[-2], result.shape[-1])

    if is_1d:
        result = result.squeeze(0)

    if computation_dtype is not None:
        return result
    return result.float()


def _apply_mel_scale(
    spectrogram: torch.Tensor,
    mel_filters: torch.Tensor,
    mel_floor: float = 1e-10,
) -> torch.Tensor:
    """Apply mel filterbank to a spectrogram.

    Args:
        spectrogram: Power spectrogram of shape (..., freq, time).
        mel_filters: Mel filterbank of shape (freq, n_mels).
        mel_floor: Minimum value for clamping.

    Returns:
        Mel spectrogram of shape (..., n_mels, time).
    """
    # (..., time, freq) @ (freq, n_mels) -> (..., time, n_mels) -> (..., n_mels, time)
    mel_spec = torch.matmul(spectrogram.transpose(-2, -1), mel_filters).transpose(-2, -1)
    return torch.clamp(mel_spec, min=mel_floor)


def _torch_stft(
    waveform, window, frame_length, hop_length, fft_length,
    power, normalized, center, pad_mode,
):
    """Fast path using torch.stft. Returns power spectrogram of shape (batch, freq, time)."""
    stft_out = torch.stft(
        waveform,
        n_fft=fft_length,
        hop_length=hop_length,
        win_length=frame_length,
        window=window,
        center=center,
        pad_mode=pad_mode,
        normalized=False,
        return_complex=True,
    )
    if normalized:
        stft_out = stft_out / window.pow(2.0).sum().sqrt()
    return stft_out.abs() ** power


def _manual_stft(
    waveform, window, frame_length, hop_length, fft_length,
    num_frequency_bins, power, normalized, center, pad_mode,
    dither, preemphasis, remove_dc_offset,
):
    """Manual framing STFT for kaldi-specific features. Returns power spectrogram of shape (batch, freq, time)."""
    if center:
        waveform = torch.nn.functional.pad(
            waveform, (frame_length // 2, frame_length // 2), mode=pad_mode
        )

    # Extract all frames at once: (batch, num_frames, frame_length)
    frames = waveform.unfold(-1, frame_length, hop_length)

    if dither != 0.0:
        frames = frames + dither * torch.randn_like(frames)

    if remove_dc_offset:
        frames = frames - frames.mean(dim=-1, keepdim=True)

    if preemphasis is not None:
        frames = torch.cat([
            frames[..., :1] * (1 - preemphasis),
            frames[..., 1:] - preemphasis * frames[..., :-1],
        ], dim=-1)

    frames = frames * window

    # Zero-pad frames to fft_length if frame_length < fft_length (kaldi left-aligns in FFT buffer)
    if frame_length < fft_length:
        frames = torch.nn.functional.pad(frames, (0, fft_length - frame_length))

    # Batched FFT: (batch, num_frames, fft_length) -> (batch, num_frames, num_frequency_bins)
    spec = torch.fft.rfft(frames, n=fft_length)

    if normalized:
        spec = spec / window.pow(2.0).sum().sqrt()

    spec = spec.abs() ** power

    # (batch, num_frames, freq) -> (batch, freq, num_frames)
    return spec.transpose(-2, -1)


# --- Main function ---

def mel_spectrogram(
    waveform: torch.Tensor,
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
    pad: int = 0,
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
) -> torch.Tensor:
    """Compute mel spectrogram using PyTorch.

    Args:
        waveform: Input waveform of shape (..., time).
        sampling_rate: Sample rate in Hz.

    Returns:
        Mel spectrogram of shape (..., n_mels, time).
    """
    if f_max is None:
        f_max = sampling_rate / 2.0

    spectrogram = _extract_spectrogram(
        waveform, sampling_rate,
        n_fft=n_fft, win_length=win_length, hop_length=hop_length,
        window_fn=window_fn, wkwargs=wkwargs, power=power,
        center=center, pad_mode=pad_mode,         normalized=normalized, pad=pad, periodic=periodic,
        dither=dither, preemphasis=preemphasis, remove_dc_offset=remove_dc_offset,
    )

    num_frequency_bins = spectrogram.shape[-2]
    mel_filters = mel_filter_bank_torch(
        num_frequency_bins, n_mels, f_min, f_max, sampling_rate,
        norm=norm, mel_scale=mel_scale,
        triangularize_in_mel_space=triangularize_in_mel_space,
    ).to(spectrogram.device)

    return _apply_mel_scale(spectrogram, mel_filters, mel_floor=mel_floor)


class MelSpectrogram(torch.nn.Module):
    """Cached mel spectrogram transform — precomputes window and mel filterbank.

    Same API and exact same results as the functional ``mel_spectrogram``, but
    avoids recomputing the window and mel filterbank on every call.

    Usage::

        transform = MelSpectrogram(sampling_rate=16000, n_fft=1024, n_mels=80)
        transform = transform.cuda()          # move buffers to GPU once
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
        pad: int = 0,
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
        super().__init__()
        self.sampling_rate = sampling_rate
        self.n_fft = n_fft
        self.win_length = win_length if win_length is not None else n_fft
        self.hop_length = hop_length if hop_length is not None else self.win_length // 2
        self.power = power
        self.center = center
        self.pad_mode = pad_mode
        self.normalized = normalized
        self.pad = pad
        self.n_mels = n_mels
        self.f_min = f_min
        self.f_max = f_max if f_max is not None else sampling_rate / 2.0
        self.mel_floor = mel_floor
        self.dither = dither
        self.preemphasis = preemphasis
        self.remove_dc_offset = remove_dc_offset

        self._needs_manual_framing = (dither != 0.0) or (preemphasis is not None) or remove_dc_offset

        # Build window
        window = window_function(self.win_length, name=window_fn, periodic=periodic, wkwargs=wkwargs)
        if self._needs_manual_framing and self.win_length < n_fft:
            self._frame_length = self.win_length
        else:
            if self.win_length < n_fft:
                left_pad = (n_fft - self.win_length) // 2
                right_pad = n_fft - self.win_length - left_pad
                window = torch.nn.functional.pad(window, (left_pad, right_pad))
            self._frame_length = n_fft
        self.register_buffer("window", window)

        # Build mel filterbank
        num_frequency_bins = n_fft // 2 + 1
        mel_fb = mel_filter_bank_torch(
            num_frequency_bins, n_mels, self.f_min, self.f_max, sampling_rate,
            norm=norm, mel_scale=mel_scale,
            triangularize_in_mel_space=triangularize_in_mel_space,
        )
        self.register_buffer("mel_filters", mel_fb)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """Compute mel spectrogram.

        Args:
            waveform: Input of shape (..., time).

        Returns:
            Mel spectrogram of shape (..., n_mels, time).
        """
        is_1d = waveform.ndim == 1
        if is_1d:
            waveform = waveform.unsqueeze(0)

        leading_shape = waveform.shape[:-1]
        waveform = waveform.reshape(-1, waveform.shape[-1])

        if self.pad > 0:
            waveform = torch.nn.functional.pad(waveform, (self.pad, self.pad))

        if self._needs_manual_framing:
            spec = _manual_stft(
                waveform, self.window, self._frame_length, self.hop_length,
                self.n_fft, self.n_fft // 2 + 1, self.power, self.normalized,
                self.center, self.pad_mode, self.dither, self.preemphasis,
                self.remove_dc_offset,
            )
        else:
            spec = _torch_stft(
                waveform, self.window, self._frame_length, self.hop_length,
                self.n_fft, self.power, self.normalized, self.center, self.pad_mode,
            )

        spec = spec.reshape(*leading_shape, spec.shape[-2], spec.shape[-1])
        if is_1d:
            spec = spec.squeeze(0)
        spec = spec.float()

        return _apply_mel_scale(spec, self.mel_filters, mel_floor=self.mel_floor)
