# Copyright 2025 The HuggingFace Inc. team.
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


import math

import numpy as np
import librosa

from .audio_processing_utils import BaseAudioProcessor
from .audio_utils import SpectrogramConfig, amplitude_to_db, power_to_db
from .feature_extraction_utils import BatchFeature
from .utils import is_torch_available, logging


logger = logging.get_logger(__name__)


if is_torch_available():
    import torch


# ── NumPy frequency conversion utilities ──────────────────────────────

def _np_hertz_to_mel(freq, mel_scale="htk"):
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


def _np_mel_to_hertz(mels, mel_scale="htk"):
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


# ── Torch frequency conversion utilities ──────────────────────────────

def _torch_hertz_to_mel_scalar(freq: float, mel_scale: str = "htk") -> float:
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


def _torch_hertz_to_mel(freq: "torch.Tensor", mel_scale: str = "htk") -> "torch.Tensor":
    if mel_scale == "htk":
        return 2595.0 * torch.log10(1.0 + freq / 700.0)
    elif mel_scale == "kaldi":
        return 1127.0 * torch.log(1.0 + freq / 700.0)
    # slaney
    f_sp = 200.0 / 3
    min_log_hertz = 1000.0
    min_log_mel = min_log_hertz / f_sp
    logstep = 27.0 / torch.log(torch.tensor(6.4))
    mels = freq / f_sp
    log_region = freq >= min_log_hertz
    mels[log_region] = min_log_mel + torch.log(freq[log_region] / min_log_hertz) * logstep
    return mels


def _torch_mel_to_hertz(mels: "torch.Tensor", mel_scale: str = "htk") -> "torch.Tensor":
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


class NumpyAudioBackend(BaseAudioProcessor):
    """NumPy backend for portable CPU-only audio processing."""

    @property
    def backend(self) -> str:
        return "numpy"

    def _process_audio(self, audio_el):
        """
        Process a single raw audio input into a np.ndarray.

        Handles mono conversion (averaging channels) and numpy conversion.
        Closely mirrors the torch backend logic: expects channel-first.
        """
        if not isinstance(audio_el, np.ndarray):
            audio_el = np.asarray(audio_el)

        if audio_el.ndim > 1:
            # Expecting channel-first: (channels, samples)
            if self.force_mono and audio_el.shape[0] > 1:
                audio_el = audio_el.mean(axis=0)
            elif audio_el.shape[0] == 1:
                audio_el = np.squeeze(audio_el, axis=0)
            else:
                raise ValueError("Audio has more than one channel but force_mono is False")
        return audio_el

    def _pad_single(self, audio: np.ndarray, max_length: int) -> np.ndarray:
        """Pad a single audio array to a target length using np.pad."""
        current_length = audio.shape[-1]
        if current_length >= max_length:
            return audio

        if self.padding_value is None:
            raise ValueError(
                "Asking to pad but the audio processor does not have a padding value. Please select a value to use"
                " as `padding_value`. For example: `audio_processor.padding_value = 0.0`."
            )

        pad_length = max_length - current_length
        if self.padding_side == "right":
            pad_width = [(0, 0)] * (audio.ndim - 1) + [(0, pad_length)]
        elif self.padding_side == "left":
            pad_width = [(0, 0)] * (audio.ndim - 1) + [(pad_length, 0)]
        else:
            raise ValueError(f"Invalid padding side: {self.padding_side}")

        return np.pad(audio, pad_width, mode="constant", constant_values=self.padding_value)

    def _create_stft_window(self, win_length, stft_cfg, audio):
        N = win_length + 1 if stft_cfg.periodic else win_length
        fac = np.linspace(-np.pi, np.pi, N)
        name = stft_cfg.window_fn
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
        return w[:win_length] if stft_cfg.periodic else w

    def _prepare_window_and_framing(self, window, win_length, n_fft, needs_manual_framing):
        if needs_manual_framing and win_length < n_fft:
            frame_length = win_length
        else:
            if win_length < n_fft:
                left_pad = (n_fft - win_length) // 2
                right_pad = n_fft - win_length - left_pad
                window = np.pad(window, (left_pad, right_pad))
            frame_length = n_fft
        return window, frame_length

    def _frame_waveform(self, waveform, frame_length, hop_length, n_fft, center, pad_mode):
        squeezed = waveform.ndim == 1
        if squeezed:
            waveform = waveform[np.newaxis, :]
        if center:
            start_k = int(np.ceil(n_fft // 2 / hop_length))
            tail_k = (waveform.shape[-1] + n_fft // 2 - n_fft) // hop_length + 1

            if tail_k <= start_k:
                waveform = np.pad(waveform, ((0, 0), (frame_length // 2, frame_length // 2)), mode=pad_mode)
                num_frames = 1 + (waveform.shape[-1] - frame_length) // hop_length
                frame_starts = np.arange(num_frames) * hop_length
                frame_indices = frame_starts[:, np.newaxis] + np.arange(frame_length)
                frames = waveform[:, frame_indices]
            else:
                padding = [(0, 0) for _ in range(waveform.ndim)]
                padding[-1] = (frame_length // 2, 0)
                y_pre = np.pad(
                    waveform[..., : (start_k - 1) * hop_length - n_fft // 2 + n_fft + 1],
                    padding,
                    mode=pad_mode,
                )
                y_frames_pre = librosa.util.frame(y_pre, frame_length=frame_length, hop_length=hop_length)
                y_frames_pre = y_frames_pre[..., :start_k]
                y_frames_pre = np.moveaxis(y_frames_pre, -2, -1)
                extra = y_frames_pre.shape[-2]

                padding[-1] = (0, frame_length // 2)
                y_post = np.pad(
                    waveform[..., (tail_k) * hop_length - n_fft // 2 :],
                    padding,
                    mode=pad_mode,
                )
                y_frames_post = librosa.util.frame(y_post, frame_length=frame_length, hop_length=hop_length)
                y_frames_post = np.moveaxis(y_frames_post, -2, -1)
                extra += y_frames_post.shape[-2]

                start = start_k * hop_length - n_fft // 2
                y_frames_middle = librosa.util.frame(
                    waveform[..., start:], frame_length=frame_length, hop_length=hop_length
                )
                y_frames_middle = np.moveaxis(y_frames_middle, -2, -1)

                num_frames = y_frames_pre.shape[-2] + y_frames_middle.shape[-2] + y_frames_post.shape[-2]
                frames = np.concatenate([y_frames_pre, y_frames_middle, y_frames_post], axis=-2)
        else:
            num_frames = 1 + (waveform.shape[-1] - frame_length) // hop_length
            frame_starts = np.arange(num_frames) * hop_length
            frame_indices = frame_starts[:, np.newaxis] + np.arange(frame_length)
            frames = waveform[:, frame_indices]

        if squeezed:
            frames = frames.squeeze(0)
        return frames, num_frames

    def _frame_audio(self, audio, window, frame_length, hop_length, n_fft, stft_cfg):
        frames, _ = self._frame_waveform(audio, frame_length, hop_length, n_fft, stft_cfg.center, stft_cfg.pad_mode)
        compute_dtype = np.result_type(audio.dtype, window.dtype)
        return frames.astype(compute_dtype, copy=False)

    def _window_and_fft(self, frames, window, frame_length, n_fft, stft_cfg):
        frames = frames * window
        spec = np.fft.rfft(frames, n=n_fft, axis=-1).astype(np.complex64)
        if stft_cfg.normalized:
            spec = spec / np.sqrt(np.sum(window**2)).astype(spec.real.dtype)
        return np.moveaxis(spec, -1, -2)

    def _native_stft(self, audio, window, frame_length, hop_length, n_fft, stft_cfg):
        frames, _ = self._frame_waveform(audio, frame_length, hop_length, n_fft, stft_cfg.center, stft_cfg.pad_mode)
        compute_dtype = np.result_type(audio.dtype, window.dtype)
        frames = frames.astype(compute_dtype, copy=False)
        frames = frames * window
        spec = np.fft.rfft(frames, n=n_fft, axis=-1).astype(np.complex64)
        if stft_cfg.normalized:
            spec = spec / np.sqrt(np.sum(window**2)).astype(spec.real.dtype)
        return np.moveaxis(spec, -1, -2)

    def _compute_magnitudes(self, stft_out, power):
        return np.abs(stft_out, dtype=np.float64) ** power

    def _apply_frame_processing(self, frames, *, spectrogram_config, **kwargs):
        """Apply per-frame signal conditioning using the numpy backend."""
        compute_dtype = frames.dtype
        if spectrogram_config.remove_dc_offset:
            frames = frames - frames.mean(axis=-1, keepdims=True)
        preemphasis = spectrogram_config.preemphasis
        if preemphasis is not None:
            preemph_src = preemphasis * frames[..., :-1]
            frames[..., 1:] = frames[..., 1:] - preemph_src
            frames[..., 0] = frames[..., 0] * (1 - preemphasis)
        return frames

    def _apply_mel_scale(
        self,
        features: list[np.ndarray],
        *,
        spectrogram_config: SpectrogramConfig,
        **kwargs,
    ) -> list[np.ndarray]:
        """Apply mel filterbank to spectrogram features using the numpy backend."""
        if spectrogram_config.mel_scale_config.matmul_order == "features_first":
            mel_spec = np.matmul(features, self.mel_filters)
        else:
            mel_spec = np.matmul(self.mel_filters.T, features)
        return np.maximum(spectrogram_config.mel_floor, mel_spec)

    def _normalize_magnitude(
        self,
        features: np.ndarray,
        *,
        spectrogram_config: SpectrogramConfig,
        reference: float = 1.0,
        min_value: float = 1e-10,
        db_range: float | None = None,
        dtype: np.dtype = np.float32,
        **kwargs,
    ) -> np.ndarray:
        """Apply magnitude normalization (log, log10, or dB scaling) to spectrogram features.

        Accepts a single or batched spectrogram (not a list).
        Mirrors the normalization logic in `audio_utils.spectrogram()`.
        """
        log_mel = spectrogram_config.log_mode
        mel_floor = spectrogram_config.mel_floor
        power = spectrogram_config.stft_config.power

        if log_mel is None:
            return features

        # Clamp to mel_floor before taking log
        result = np.maximum(mel_floor, features)

        if log_mel == "log":
            result = np.log(result).astype(dtype)
        elif log_mel == "log10":
            result = np.log10(result).astype(dtype)
        elif log_mel == "dB":
            if power == 1.0:
                result = amplitude_to_db(result, reference, min_value, db_range).astype(dtype)
            elif power == 2.0:
                result = power_to_db(result, reference, min_value, db_range).astype(dtype)
            else:
                raise ValueError(f"Cannot use log_mel option 'dB' with power {power}")
        else:
            raise ValueError(f"Unknown log_mel option: {log_mel}")

        return result

    def _mel_filter_bank(self, spectrogram_config: SpectrogramConfig):
        stft_cfg = spectrogram_config.stft_config
        mel_cfg = spectrogram_config.mel_scale_config
        num_frequency_bins = 1 + stft_cfg.n_fft // 2
        num_mel_filters = mel_cfg.n_mels
        min_frequency = mel_cfg.f_min
        max_frequency = mel_cfg.f_max if mel_cfg.f_max is not None else self.sample_rate / 2
        sampling_rate = self.sample_rate

        mel_min = _np_hertz_to_mel(min_frequency, mel_scale=mel_cfg.mel_scale)
        mel_max = _np_hertz_to_mel(max_frequency, mel_scale=mel_cfg.mel_scale)
        mel_freqs = np.linspace(mel_min, mel_max, num_mel_filters + 2)
        filter_freqs = _np_mel_to_hertz(mel_freqs, mel_scale=mel_cfg.mel_scale)

        n_fft = (num_frequency_bins - 1) * 2

        if mel_cfg.triangularize_in_mel_space:
            fft_bin_width = sampling_rate / n_fft
            fft_freqs = _np_hertz_to_mel(
                fft_bin_width * np.arange(num_frequency_bins), mel_scale=mel_cfg.mel_scale
            )
            filter_freqs = mel_freqs
        elif mel_cfg.frequency_bin_mode == "rfft":
            fft_freqs = np.fft.rfftfreq(n=n_fft, d=1.0 / sampling_rate)
        else:
            fft_freqs = np.linspace(0, sampling_rate // 2, num_frequency_bins)

        # Triangular filter bank
        filter_diff = np.diff(filter_freqs)
        slopes = np.expand_dims(filter_freqs, 0) - np.expand_dims(fft_freqs, 1)
        down_slopes = -slopes[:, :-2] / filter_diff[:-1]
        up_slopes = slopes[:, 2:] / filter_diff[1:]
        mel_filters = np.maximum(0, np.minimum(down_slopes, up_slopes))

        if mel_cfg.norm == "slaney":
            enorm = 2.0 / (filter_freqs[2 : num_mel_filters + 2] - filter_freqs[:num_mel_filters])
            mel_filters *= np.expand_dims(enorm, 0)

        return mel_filters

    def _to_batch(self, audio):
        return np.stack(audio)

    def _get_mask(self, audio_ranges, padded_length, do_extract_spectrogram, spectrogram_config):
        if do_extract_spectrogram:
            spec_cfg = spectrogram_config or self.spectrogram_config
            audio_lengths = np.array([end - start for start, end in audio_ranges])
            features_lengths = self._get_features_lengths(audio_lengths, spec_cfg)
            n_features = self._get_features_lengths(padded_length, spec_cfg, include_center_frame=True)
            mask = (np.arange(n_features)[None, :] < features_lengths[:, None]).astype(np.int32)
            return {"audio_features_mask": mask}
        else:
            mask = np.zeros((len(audio_ranges), padded_length), dtype=np.int32)
            for i, (start, end) in enumerate(audio_ranges):
                mask[i, start:end] = 1
            return {"audio_values_mask": mask}


class TorchAudioBackend(BaseAudioProcessor):
    """Torch backend for audio processing."""

    @property
    def backend(self) -> str:
        return "torch"

    def _process_audio(self, audio_el):
        """
        Process a single raw audio input into a torch.Tensor.

        Handles mono conversion (averaging channels) and numpy-to-torch conversion.
        """
        import torch

        if isinstance(audio_el, np.ndarray):
            audio_el = torch.from_numpy(audio_el)

        if audio_el.ndim > 1:
            # TODO: we would need to ensure somewhere audio is channel first
            if self.force_mono and audio_el.shape[0] > 1:
                audio_el = audio_el.mean(dim=0)
            elif audio_el.shape[0] == 1:
                audio_el = audio_el.squeeze(0)
            else:
                raise ValueError("Audio has more than one channel but force_mono is False")

        return audio_el

    def _pad_single(self, audio: "torch.Tensor", max_length: int) -> "torch.Tensor":
        """Pad a single audio tensor to a target length using torch.nn.functional.pad."""
        import torch.nn.functional as F

        current_length = audio.shape[-1]
        if current_length >= max_length:
            return audio

        if self.padding_value is None:
            raise ValueError(
                "Asking to pad but the audio processor does not have a padding value. Please select a value to use"
                " as `padding_value`. For example: `audio_processor.padding_value = 0.0`."
            )

        if self.padding_side == "right":
            pad_args = (0, max_length - current_length)
        elif self.padding_side == "left":
            pad_args = (max_length - current_length, 0)
        else:
            raise ValueError(f"Invalid padding side: {self.padding_side}")

        return F.pad(audio, pad_args, "constant", self.padding_value)

    def _needs_manual_framing(self, spectrogram_config):
        """Extends the base check with ``left_align_fft`` which also requires manual framing."""
        return super()._needs_manual_framing(spectrogram_config) or spectrogram_config.stft_config.left_align_fft

    def _create_stft_window(self, win_length, stft_cfg, audio):
        dtype = getattr(torch, stft_cfg.window_dtype) if stft_cfg.window_dtype else audio.dtype
        wkwargs = {**(stft_cfg.wkwargs or {}), "dtype": dtype}
        name = stft_cfg.window_fn
        if name in ("hann", "hann_window"):
            window = torch.hann_window(win_length, periodic=stft_cfg.periodic, **wkwargs)
        elif name in ("hamming", "hamming_window"):
            window = torch.hamming_window(win_length, periodic=stft_cfg.periodic, **wkwargs)
        elif name == "boxcar":
            window = torch.ones(win_length)
        elif name == "povey":
            window = torch.hann_window(win_length, periodic=stft_cfg.periodic, **wkwargs).pow(0.85)
        else:
            raise ValueError(f"Unknown window function '{name}'")
        return window.to(device=audio.device)

    def _prepare_window_and_framing(self, window, win_length, n_fft, needs_manual_framing):
        if needs_manual_framing and win_length < n_fft:
            frame_length = win_length
        else:
            if win_length < n_fft:
                left_pad = (n_fft - win_length) // 2
                right_pad = n_fft - win_length - left_pad
                window = torch.nn.functional.pad(window, (left_pad, right_pad))
            frame_length = n_fft
        return window, frame_length

    def _frame_audio(self, audio, window, frame_length, hop_length, n_fft, stft_cfg):
        if stft_cfg.center:
            audio = torch.nn.functional.pad(
                audio, (frame_length // 2, frame_length // 2), mode=stft_cfg.pad_mode
            )
        return audio.unfold(-1, frame_length, hop_length)

    def _window_and_fft(self, frames, window, frame_length, n_fft, stft_cfg):
        frames = frames * window
        if frame_length < n_fft:
            frames = torch.nn.functional.pad(frames, (0, n_fft - frame_length))
        spec = torch.fft.rfft(frames, n=n_fft)
        if stft_cfg.normalized:
            spec = spec / window.pow(2.0).sum().sqrt()
        return spec.transpose(-2, -1)

    def _native_stft(self, audio, window, frame_length, hop_length, n_fft, stft_cfg):
        stft_out = torch.stft(
            audio,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=frame_length,
            window=window,
            center=stft_cfg.center,
            pad_mode=stft_cfg.pad_mode,
            normalized=False,
            return_complex=True,
        )
        if stft_cfg.normalized:
            stft_out = stft_out / window.pow(2.0).sum().sqrt()
        return stft_out

    def _cast_stft_output(self, magnitudes, spectrogram_config):
        if spectrogram_config.computation_dtype:
            return magnitudes
        return magnitudes.float()

    def _compute_magnitudes(self, stft_out, power):
        """Convert complex STFT output to a real-valued magnitude spectrogram."""
        return stft_out.abs() ** power

    def _apply_frame_processing(self, frames, *, spectrogram_config, **kwargs):
        """Apply per-frame signal conditioning using the torch backend."""
        if spectrogram_config.remove_dc_offset:
            frames = frames - frames.mean(dim=-1, keepdim=True)
        preemphasis = spectrogram_config.preemphasis
        if preemphasis is not None:
            frames = torch.cat([
                frames[..., :1] * (1 - preemphasis),
                frames[..., 1:] - preemphasis * frames[..., :-1],
            ], dim=-1)
        return frames

    def _apply_mel_scale(
        self,
        features: list["torch.Tensor"],
        *,
        spectrogram_config: SpectrogramConfig,
        **kwargs,
    ) -> list["torch.Tensor"]:
        """Apply mel filterbank to spectrogram features using the torch backend."""
        mel_filters = self.mel_filters.to(device=features.device)
        if spectrogram_config.mel_scale_config.matmul_order == "features_first":
            mel_spec = torch.matmul(features.transpose(-2, -1), mel_filters)
        else:
            mel_spec = torch.matmul(mel_filters.T, features)
        return torch.clamp(mel_spec, min=spectrogram_config.mel_floor)

    def _normalize_magnitude(
        self,
        features: "torch.Tensor",
        *,
        spectrogram_config: SpectrogramConfig,
        reference: float = 1.0,
        min_value: float = 1e-10,
        db_range: float | None = None,
        dtype: "torch.dtype | None" = None,
        **kwargs,
    ) -> "torch.Tensor":
        """Apply magnitude normalization (log, log10, or dB scaling) to batched spectrogram features (torch.Tensor only)."""
        import torch

        log_mel = spectrogram_config.log_mode
        mel_floor = spectrogram_config.mel_floor
        power = spectrogram_config.stft_config.power

        if dtype is None:
            dtype = torch.float32

        if log_mel is None:
            return features

        # Clamp to mel_floor before taking log
        result = torch.clamp(features, min=mel_floor)

        if log_mel == "log":
            result = torch.log(result).to(dtype)
        elif log_mel == "log10":
            result = torch.log10(result).to(dtype)
        elif log_mel == "dB":
            if reference <= 0.0:
                raise ValueError("reference must be greater than zero")
            if min_value <= 0.0:
                raise ValueError("min_value must be greater than zero")
            reference = max(min_value, reference)
            multiplier = 10.0 if power == 2.0 else 20.0 if power == 1.0 else None
            if multiplier is None:
                raise ValueError(f"Cannot use log_mel option 'dB' with power {power}")
            log_ref = torch.log10(torch.tensor(reference, dtype=result.dtype, device=result.device))
            result = torch.clamp(result, min=min_value)
            result = multiplier * (torch.log10(result) - log_ref)
            if db_range is not None:
                if db_range <= 0.0:
                    raise ValueError("db_range must be greater than zero")
                # Clamp each sample so the minimum value is (max - db_range)
                max_vals = result.amax(dim=-2, keepdim=True) if result.ndim > 2 else result.max()
                result = torch.clamp(result, min=max_vals - db_range)
            result = result.to(dtype)
        else:
            raise ValueError(f"Unknown log_mel option: {log_mel}")

        return result

    def _mel_filter_bank(self, spectrogram_config: SpectrogramConfig):
        stft_cfg = spectrogram_config.stft_config
        mel_cfg = spectrogram_config.mel_scale_config
        computation_dtype = getattr(torch, mel_cfg.computation_dtype) if mel_cfg.computation_dtype else None
        num_frequency_bins = 1 + stft_cfg.n_fft // 2
        num_mel_filters = mel_cfg.n_mels
        min_frequency = mel_cfg.f_min
        max_frequency = mel_cfg.f_max if mel_cfg.f_max is not None else self.sample_rate / 2
        sampling_rate = self.sample_rate

        if mel_cfg.triangularize_in_mel_space and mel_cfg.bands_to_zero == 0:
            # Kaldi-exact path: matches torchaudio.compliance.kaldi.get_mel_banks.
            n_fft = (num_frequency_bins - 1) * 2
            num_fft_bins = n_fft // 2
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
            banks = torch.nn.functional.pad(banks, (0, 1), mode="constant", value=0)

            mel_filters = banks.T
        elif mel_cfg.triangularize_in_mel_space:
            # Kaldi-style with bands_to_zero > 0
            n_fft = (num_frequency_bins - 1) * 2
            mel_min = _torch_hertz_to_mel_scalar(min_frequency, mel_scale=mel_cfg.mel_scale)
            mel_max = _torch_hertz_to_mel_scalar(max_frequency, mel_scale=mel_cfg.mel_scale)
            mel_delta = (mel_max - mel_min) / (num_mel_filters + 1)
            bin_idx = torch.arange(num_mel_filters, dtype=computation_dtype).unsqueeze(1)
            left_mel = mel_min + bin_idx * mel_delta
            center_mel = mel_min + (bin_idx + 1.0) * mel_delta
            right_mel = mel_min + (bin_idx + 2.0) * mel_delta

            fft_bin_width = sampling_rate / n_fft
            hz_freqs = fft_bin_width * torch.arange(mel_cfg.bands_to_zero, num_frequency_bins, dtype=computation_dtype)
            mel = _torch_hertz_to_mel(hz_freqs, mel_scale=mel_cfg.mel_scale).unsqueeze(0)

            up_slope = (mel - left_mel) / (center_mel - left_mel)
            down_slope = (right_mel - mel) / (right_mel - center_mel)
            mel_filters = torch.max(torch.zeros(1, dtype=computation_dtype), torch.min(up_slope, down_slope))

            mel_filters = mel_filters.T
            if mel_cfg.bands_to_zero > 0:
                mel_filters = torch.nn.functional.pad(mel_filters, (0, 0, mel_cfg.bands_to_zero, 0))
        else:
            n_fft = (num_frequency_bins - 1) * 2
            mel_min = _torch_hertz_to_mel_scalar(min_frequency, mel_scale=mel_cfg.mel_scale)
            mel_max = _torch_hertz_to_mel_scalar(max_frequency, mel_scale=mel_cfg.mel_scale)
            mel_freqs = torch.linspace(mel_min, mel_max, num_mel_filters + 2, dtype=computation_dtype)
            filter_freqs = _torch_mel_to_hertz(mel_freqs, mel_scale=mel_cfg.mel_scale)

            if mel_cfg.frequency_bin_mode == "rfft":
                fft_freqs = torch.fft.rfftfreq(n=n_fft, d=1.0 / sampling_rate)
            else:
                fft_freqs = torch.linspace(0, sampling_rate // 2, num_frequency_bins)
            if computation_dtype is not None:
                fft_freqs = fft_freqs.to(computation_dtype)

            # Triangular filter bank
            filter_diff = filter_freqs[1:] - filter_freqs[:-1]
            slopes = filter_freqs.unsqueeze(0) - fft_freqs.unsqueeze(1)
            down_slopes = -slopes[:, :-2] / filter_diff[:-1]
            up_slopes = slopes[:, 2:] / filter_diff[1:]
            mel_filters = torch.clamp(torch.minimum(down_slopes, up_slopes), min=0)

            if mel_cfg.norm == "slaney":
                enorm = 2.0 / (filter_freqs[2 : num_mel_filters + 2] - filter_freqs[:num_mel_filters])
                mel_filters = mel_filters * enorm.unsqueeze(0)

            if mel_cfg.bands_to_zero > 0:
                mel_filters = torch.nn.functional.pad(mel_filters, (0, 0, mel_cfg.bands_to_zero, 0))

        # When computation_dtype is set only on the mel config (not on the
        # spectrogram config), the filters were computed in high precision for
        # accuracy but the spectrogram will be in the default dtype — cast back.
        if computation_dtype is not None and not spectrogram_config.computation_dtype:
            mel_filters = mel_filters.to(torch.get_default_dtype())
        return mel_filters

    def _to_batch(self, audio):
        return torch.stack(audio)

    def _get_mask(self, audio_ranges, padded_length, do_extract_spectrogram, spectrogram_config):
        if do_extract_spectrogram:
            spec_cfg = spectrogram_config or self.spectrogram_config
            audio_lengths = torch.tensor([end - start for start, end in audio_ranges])
            features_lengths = self._get_features_lengths(audio_lengths, spec_cfg)
            n_features = self._get_features_lengths(padded_length, spec_cfg, include_center_frame=True)
            mask = (torch.arange(n_features)[None, :] < features_lengths[:, None]).to(torch.int32)
            return {"audio_features_mask": mask}
        else:
            mask = torch.zeros((len(audio_ranges), padded_length), dtype=torch.int32)
            for i, (start, end) in enumerate(audio_ranges):
                mask[i, start:end] = 1
            return {"audio_values_mask": mask}
