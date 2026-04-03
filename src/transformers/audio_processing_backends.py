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

from .audio_processing_utils import BaseAudioProcessor
from .audio_utils import SpectrogramConfig, amplitude_to_db, mel_filter_bank, power_to_db
from .utils import PaddingStrategy, is_torch_available, logging


logger = logging.get_logger(__name__)


if is_torch_available():
    import torch


# ── Torch frequency conversion utilities (used by TorchAudioBackend._mel_filter_bank) ──


def _torch_hertz_to_mel_scalar(freq: float, mel_scale: str = "htk") -> float:
    if mel_scale == "htk":
        return 2595.0 * math.log10(1.0 + freq / 700.0)
    elif mel_scale == "kaldi":
        return 1127.0 * math.log(1.0 + freq / 700.0)
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
    f_sp = 200.0 / 3
    min_log_hz = 1000.0
    min_log_mel = (min_log_hz - 0.0) / f_sp
    logstep = math.log(6.4) / 27.0
    freq = 0.0 + f_sp * mels
    log_region = mels >= min_log_mel
    freq[log_region] = min_log_hz * torch.exp(logstep * (mels[log_region] - min_log_mel))
    return freq


def _torch_triangular_filter_bank(fft_freqs, filter_freqs, computation_dtype=None):
    """Compute triangular mel filter bank (shared by non-kaldi TorchAudioBackend paths)."""
    num_mel_filters = len(filter_freqs) - 2
    filter_diff = filter_freqs[1:] - filter_freqs[:-1]
    slopes = filter_freqs.unsqueeze(0) - fft_freqs.unsqueeze(1)
    down_slopes = -slopes[:, :-2] / filter_diff[:-1]
    up_slopes = slopes[:, 2:] / filter_diff[1:]
    zero = torch.zeros(1, dtype=computation_dtype) if computation_dtype else torch.zeros(1)
    return torch.clamp(torch.minimum(down_slopes, up_slopes), min=0)


# ═══════════════════════════════════════════════════════════════════════════════
# NumpyAudioBackend
# ═══════════════════════════════════════════════════════════════════════════════


class NumpyAudioBackend(BaseAudioProcessor):
    """NumPy backend for portable CPU-only audio processing."""

    @property
    def backend(self) -> str:
        return "numpy"

    # ── Audio input processing ────────────────────────────────────────────

    def _process_audio(self, audio_el):
        if not isinstance(audio_el, np.ndarray):
            audio_el = np.asarray(audio_el)
        if audio_el.ndim > 1:
            if self.force_mono and audio_el.shape[0] > 1:
                audio_el = audio_el.mean(axis=0)
            elif audio_el.shape[0] == 1:
                audio_el = np.squeeze(audio_el, axis=0)
            else:
                raise ValueError("Audio has more than one channel but force_mono is False")
        return audio_el

    # ── Padding & batching ────────────────────────────────────────────────

    def _pad_single(self, audio: np.ndarray, max_length: int) -> np.ndarray:
        current_length = audio.shape[-1]
        if current_length >= max_length:
            return audio
        pad_length = max_length - current_length
        if self.padding_side == "right":
            pad_width = [(0, 0)] * (audio.ndim - 1) + [(0, pad_length)]
        elif self.padding_side == "left":
            pad_width = [(0, 0)] * (audio.ndim - 1) + [(pad_length, 0)]
        else:
            raise ValueError(f"Invalid padding side: {self.padding_side}")
        return np.pad(audio, pad_width, mode="constant", constant_values=self.padding_value)

    def _to_batch(self, audio):
        batch = np.stack(audio)
        if self.add_channel_dim:
            batch = batch[:, np.newaxis, :]
        return batch

    def _pad_features(self, features, padding, max_length, truncation, pad_to_multiple_of):
        padding_strategy = self._get_padding_strategies(padding=padding, max_length=max_length)
        if truncation and max_length is not None:
            features = [f[:max_length] for f in features]
        actual_lengths = [f.shape[0] for f in features]
        if padding_strategy == PaddingStrategy.LONGEST:
            max_length = max(actual_lengths)
            padding_strategy = PaddingStrategy.MAX_LENGTH
        if max_length is not None and pad_to_multiple_of is not None and (max_length % pad_to_multiple_of != 0):
            max_length = ((max_length // pad_to_multiple_of) + 1) * pad_to_multiple_of
        if padding_strategy == PaddingStrategy.MAX_LENGTH and max_length is not None:
            features = [
                np.pad(f, [(0, max_length - f.shape[0])] + [(0, 0)] * (f.ndim - 1),
                       mode="constant", constant_values=self.padding_value)
                if f.shape[0] < max_length else f
                for f in features
            ]
        return features, [(0, length) for length in actual_lengths]

    def _stack_features(self, features):
        return np.stack(features)

    # ── Masking ───────────────────────────────────────────────────────────

    def _get_mask(self, audio_ranges, padded_length, do_extract_spectrogram, spectrogram_config):
        use_audio_mask = self.mask_level == "audio"
        if do_extract_spectrogram and not use_audio_mask:
            spec_cfg = spectrogram_config or self.spectrogram_config
            audio_lengths = np.array([end - start for start, end in audio_ranges])
            features_lengths = self._get_features_lengths(audio_lengths, spec_cfg)
            n_features = self._get_features_lengths(padded_length, spec_cfg, include_center_frame=True)
            mask = (np.arange(n_features)[None, :] < features_lengths[:, None]).astype(np.int32)
            return {"audio_features_mask": mask}
        mask = np.zeros((len(audio_ranges), padded_length), dtype=np.int32)
        for i, (start, end) in enumerate(audio_ranges):
            mask[i, start:end] = 1
        return {("audio_features_mask" if do_extract_spectrogram else "audio_values_mask"): mask}

    def _get_feature_mask(self, feature_ranges, padded_length):
        mask = np.zeros((len(feature_ranges), padded_length), dtype=np.int32)
        for i, (start, end) in enumerate(feature_ranges):
            mask[i, start:end] = 1
        return {"audio_features_mask": mask}

    # ── STFT pipeline ─────────────────────────────────────────────────────

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
            return window, win_length
        if win_length < n_fft:
            left_pad = (n_fft - win_length) // 2
            right_pad = n_fft - win_length - left_pad
            window = np.pad(window, (left_pad, right_pad))
        return window, n_fft

    @staticmethod
    def _np_frame(x, frame_length, hop_length):
        """Create overlapping frames using stride tricks (replaces librosa.util.frame)."""
        n_frames = 1 + (x.shape[-1] - frame_length) // hop_length
        strides = x.strides[:-1] + (x.strides[-1] * hop_length, x.strides[-1])
        shape = x.shape[:-1] + (n_frames, frame_length)
        return np.lib.stride_tricks.as_strided(x, shape=shape, strides=strides)

    def _frame_waveform(self, waveform, frame_length, hop_length, n_fft, center, pad_mode):
        squeezed = waveform.ndim == 1
        if squeezed:
            waveform = waveform[np.newaxis, :]

        if center:
            start_k = int(np.ceil(n_fft // 2 / hop_length))
            tail_k = (waveform.shape[-1] + n_fft // 2 - n_fft) // hop_length + 1

            if tail_k <= start_k:
                # Short audio: simple center-pad and index-based framing
                waveform = np.pad(waveform, ((0, 0), (frame_length // 2, frame_length // 2)), mode=pad_mode)
                num_frames = 1 + (waveform.shape[-1] - frame_length) // hop_length
                frame_starts = np.arange(num_frames) * hop_length
                frames = waveform[:, frame_starts[:, np.newaxis] + np.arange(frame_length)]
            else:
                # Long audio: split into pre (left-padded), middle (no pad), post (right-padded)
                # to handle edge effects from center padding correctly
                padding = [(0, 0) for _ in range(waveform.ndim)]

                padding[-1] = (frame_length // 2, 0)
                y_pre = np.pad(waveform[..., : (start_k - 1) * hop_length - n_fft // 2 + n_fft + 1], padding, mode=pad_mode)
                y_frames_pre = self._np_frame(y_pre, frame_length, hop_length)[..., :start_k, :]

                padding[-1] = (0, frame_length // 2)
                y_post = np.pad(waveform[..., tail_k * hop_length - n_fft // 2 :], padding, mode=pad_mode)
                y_frames_post = self._np_frame(y_post, frame_length, hop_length)

                start = start_k * hop_length - n_fft // 2
                y_frames_middle = self._np_frame(np.ascontiguousarray(waveform[..., start:]), frame_length, hop_length)

                num_frames = y_frames_pre.shape[-2] + y_frames_middle.shape[-2] + y_frames_post.shape[-2]
                frames = np.concatenate([y_frames_pre, y_frames_middle, y_frames_post], axis=-2)
        else:
            # Non-centered: simple index-based framing
            num_frames = 1 + (waveform.shape[-1] - frame_length) // hop_length
            frame_starts = np.arange(num_frames) * hop_length
            frames = waveform[:, frame_starts[:, np.newaxis] + np.arange(frame_length)]

        if squeezed:
            frames = frames.squeeze(0)
        return frames, num_frames

    def _frame_audio(self, audio, window, frame_length, hop_length, n_fft, stft_cfg):
        frames, _ = self._frame_waveform(audio, frame_length, hop_length, n_fft, stft_cfg.center, stft_cfg.pad_mode)
        compute_dtype = np.result_type(audio.dtype, window.dtype)
        return frames.astype(compute_dtype, copy=False)

    def _apply_frame_processing(self, frames, *, spectrogram_config, **kwargs):
        if spectrogram_config.remove_dc_offset:
            frames = frames - frames.mean(axis=-1, keepdims=True)
        preemphasis = spectrogram_config.preemphasis
        if preemphasis is not None:
            preemph_src = preemphasis * frames[..., :-1]
            frames[..., 1:] = frames[..., 1:] - preemph_src
            frames[..., 0] = frames[..., 0] * (1 - preemphasis)
        return frames

    def _window_and_fft(self, frames, window, frame_length, n_fft, stft_cfg, audio_dtype=None):
        frames = frames * window
        spec = np.fft.rfft(frames, n=n_fft, axis=-1).astype(np.complex64)
        if stft_cfg.normalized:
            spec = spec / np.sqrt(np.sum(window**2)).astype(spec.real.dtype)
        return np.moveaxis(spec, -1, -2)

    def _native_stft(self, audio, window, frame_length, hop_length, n_fft, stft_cfg):
        frames, _ = self._frame_waveform(audio, frame_length, hop_length, n_fft, stft_cfg.center, stft_cfg.pad_mode)
        compute_dtype = np.result_type(audio.dtype, window.dtype)
        frames = frames.astype(compute_dtype, copy=False) * window
        spec = np.fft.rfft(frames, n=n_fft, axis=-1).astype(np.complex64)
        if stft_cfg.normalized:
            spec = spec / np.sqrt(np.sum(window**2)).astype(spec.real.dtype)
        return np.moveaxis(spec, -1, -2)

    def _compute_magnitudes(self, stft_out, power, spectrogram_config=None):
        # computation_dtype signals that upstream FE used float64 magnitudes
        if spectrogram_config and spectrogram_config.computation_dtype:
            return np.abs(stft_out, dtype=np.float64) ** power
        return np.abs(stft_out) ** power

    # ── Mel scale & normalization ─────────────────────────────────────────

    def _mel_filter_bank(self, spectrogram_config: SpectrogramConfig):
        stft_cfg = spectrogram_config.stft_config
        mel_cfg = spectrogram_config.mel_scale_config
        # float32 dtype matches librosa's per-band rounding; computation_dtype keeps float64
        filter_dtype = None if spectrogram_config.computation_dtype else np.float32
        return mel_filter_bank(
            num_frequency_bins=1 + stft_cfg.n_fft // 2,
            num_mel_filters=mel_cfg.n_mels,
            min_frequency=mel_cfg.f_min,
            max_frequency=mel_cfg.f_max if mel_cfg.f_max is not None else self.sample_rate / 2,
            sampling_rate=self.sample_rate,
            norm=mel_cfg.norm,
            mel_scale=mel_cfg.mel_scale,
            triangularize_in_mel_space=mel_cfg.triangularize_in_mel_space,
            dtype=filter_dtype,
        )

    def _apply_mel_scale(self, features, *, spectrogram_config, **kwargs):
        mel_filters = self.mel_filters.astype(features.dtype, copy=False)
        if spectrogram_config.mel_scale_config.matmul_order == "features_first":
            mel_spec = np.matmul(features, mel_filters)
        else:
            mel_spec = np.matmul(mel_filters.T, features)
        return np.maximum(spectrogram_config.mel_floor, mel_spec)

    def _normalize_magnitude(self, features, *, spectrogram_config,
                             reference=1.0, min_value=1e-10, db_range=None,
                             dtype=np.float32, **kwargs):
        log_mel = spectrogram_config.log_mode
        if log_mel is None:
            return features.astype(dtype)

        mel_floor = spectrogram_config.mel_floor
        result = np.maximum(mel_floor, features)

        if log_mel == "log":
            result = np.log(result).astype(dtype)
        elif log_mel == "log10":
            result = np.log10(result).astype(dtype)
        elif log_mel == "dB":
            power = spectrogram_config.stft_config.power
            if power == 1.0:
                result = amplitude_to_db(result, reference, min_value, db_range).astype(dtype)
            elif power == 2.0:
                result = power_to_db(result, reference, min_value, db_range).astype(dtype)
            else:
                raise ValueError(f"Cannot use log_mel option 'dB' with power {power}")
        else:
            raise ValueError(f"Unknown log_mel option: {log_mel}")
        return result

    # ── Kaldi fbank helper ────────────────────────────────────────────────

    def _kaldi_fbank(self, waveform, num_mel_bins, sample_frequency=None, **kwargs):
        """Extract kaldi-compatible fbank features using torchaudio (or fallback to base pipeline).

        Returns numpy array of shape (time, num_mel_bins).
        """
        from .utils import is_speech_available

        if sample_frequency is None:
            sample_frequency = self.sample_rate

        if is_speech_available():
            import torch
            import torchaudio.compliance.kaldi as ta_kaldi

            waveform_tensor = torch.from_numpy(np.asarray(waveform)).unsqueeze(0)
            fbank = ta_kaldi.fbank(waveform_tensor, num_mel_bins=num_mel_bins,
                                   sample_frequency=sample_frequency, **kwargs)
            return fbank.numpy()

        waveform = np.squeeze(waveform)
        features = self.extract_spectrogram([waveform], spectrogram_config=self.spectrogram_config)
        return features[0].T


# ═══════════════════════════════════════════════════════════════════════════════
# TorchAudioBackend
# ═══════════════════════════════════════════════════════════════════════════════


class TorchAudioBackend(BaseAudioProcessor):
    """Torch backend for audio processing."""

    @property
    def backend(self) -> str:
        return "torch"

    # ── Audio input processing ────────────────────────────────────────────

    def _process_audio(self, audio_el):
        if isinstance(audio_el, np.ndarray):
            audio_el = torch.from_numpy(audio_el)
        if audio_el.ndim > 1:
            if self.force_mono and audio_el.shape[0] > 1:
                audio_el = audio_el.mean(dim=0)
            elif audio_el.shape[0] == 1:
                audio_el = audio_el.squeeze(0)
            else:
                raise ValueError("Audio has more than one channel but force_mono is False")
        return audio_el

    # ── Padding & batching ────────────────────────────────────────────────

    def _pad_single(self, audio, max_length):
        current_length = audio.shape[-1]
        if current_length >= max_length:
            return audio
        if self.padding_side == "right":
            pad_args = (0, max_length - current_length)
        elif self.padding_side == "left":
            pad_args = (max_length - current_length, 0)
        else:
            raise ValueError(f"Invalid padding side: {self.padding_side}")
        return torch.nn.functional.pad(audio, pad_args, "constant", self.padding_value)

    def _to_batch(self, audio):
        batch = torch.stack(audio)
        if self.add_channel_dim:
            batch = batch.unsqueeze(1)
        return batch

    def _pad_features(self, features, padding, max_length, truncation, pad_to_multiple_of):
        padding_strategy = self._get_padding_strategies(padding=padding, max_length=max_length)
        if truncation and max_length is not None:
            features = [f[:max_length] for f in features]
        actual_lengths = [f.shape[0] for f in features]
        if padding_strategy == PaddingStrategy.LONGEST:
            max_length = max(actual_lengths)
            padding_strategy = PaddingStrategy.MAX_LENGTH
        if max_length is not None and pad_to_multiple_of is not None and (max_length % pad_to_multiple_of != 0):
            max_length = ((max_length // pad_to_multiple_of) + 1) * pad_to_multiple_of
        if padding_strategy == PaddingStrategy.MAX_LENGTH and max_length is not None:
            padded = []
            for f in features:
                if f.shape[0] < max_length:
                    pad_args = [0, 0] * (f.ndim - 1) + [0, max_length - f.shape[0]]
                    f = torch.nn.functional.pad(f, pad_args, "constant", self.padding_value)
                padded.append(f)
            features = padded
        return features, [(0, length) for length in actual_lengths]

    def _stack_features(self, features):
        return torch.stack(features)

    # ── Masking ───────────────────────────────────────────────────────────

    def _get_mask(self, audio_ranges, padded_length, do_extract_spectrogram, spectrogram_config):
        use_audio_mask = self.mask_level == "audio"
        if do_extract_spectrogram and not use_audio_mask:
            spec_cfg = spectrogram_config or self.spectrogram_config
            audio_lengths = torch.tensor([end - start for start, end in audio_ranges])
            features_lengths = self._get_features_lengths(audio_lengths, spec_cfg)
            n_features = self._get_features_lengths(padded_length, spec_cfg, include_center_frame=True)
            mask = (torch.arange(n_features)[None, :] < features_lengths[:, None]).to(torch.int32)
            return {"audio_features_mask": mask}
        mask = torch.zeros((len(audio_ranges), padded_length), dtype=torch.int32)
        for i, (start, end) in enumerate(audio_ranges):
            mask[i, start:end] = 1
        return {("audio_features_mask" if do_extract_spectrogram else "audio_values_mask"): mask}

    def _get_feature_mask(self, feature_ranges, padded_length):
        mask = torch.zeros((len(feature_ranges), padded_length), dtype=torch.int32)
        for i, (start, end) in enumerate(feature_ranges):
            mask[i, start:end] = 1
        return {"audio_features_mask": mask}

    # ── STFT pipeline ─────────────────────────────────────────────────────

    def _needs_manual_framing(self, spectrogram_config):
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
            return window, win_length
        if win_length < n_fft:
            left_pad = (n_fft - win_length) // 2
            right_pad = n_fft - win_length - left_pad
            window = torch.nn.functional.pad(window, (left_pad, right_pad))
        return window, n_fft

    def _frame_audio(self, audio, window, frame_length, hop_length, n_fft, stft_cfg):
        if stft_cfg.center:
            audio = torch.nn.functional.pad(audio, (frame_length // 2, frame_length // 2), mode=stft_cfg.pad_mode)
        return audio.unfold(-1, frame_length, hop_length)

    def _apply_frame_processing(self, frames, *, spectrogram_config, **kwargs):
        if spectrogram_config.remove_dc_offset:
            frames = frames - frames.mean(dim=-1, keepdim=True)
        preemphasis = spectrogram_config.preemphasis
        if preemphasis is not None:
            frames = torch.cat([
                frames[..., :1] * (1 - preemphasis),
                frames[..., 1:] - preemphasis * frames[..., :-1],
            ], dim=-1)
        return frames

    def _window_and_fft(self, frames, window, frame_length, n_fft, stft_cfg, audio_dtype=None):
        frames = frames * window
        if frame_length < n_fft:
            frames = torch.nn.functional.pad(frames, (0, n_fft - frame_length))
        spec = torch.fft.rfft(frames, n=n_fft)
        if stft_cfg.normalized:
            spec = spec / window.pow(2.0).sum().sqrt()
        return spec.transpose(-2, -1)

    def _native_stft(self, audio, window, frame_length, hop_length, n_fft, stft_cfg):
        stft_out = torch.stft(
            audio, n_fft=n_fft, hop_length=hop_length, win_length=frame_length,
            window=window, center=stft_cfg.center, pad_mode=stft_cfg.pad_mode,
            normalized=False, return_complex=True,
        )
        if stft_cfg.normalized:
            stft_out = stft_out / window.pow(2.0).sum().sqrt()
        return stft_out

    def _cast_stft_output(self, magnitudes, spectrogram_config):
        if spectrogram_config.computation_dtype:
            return magnitudes
        return magnitudes.float()

    def _compute_magnitudes(self, stft_out, power, spectrogram_config=None):
        return stft_out.abs() ** power

    # ── Mel scale & normalization ─────────────────────────────────────────

    def _mel_filter_bank(self, spectrogram_config: SpectrogramConfig):
        stft_cfg = spectrogram_config.stft_config
        mel_cfg = spectrogram_config.mel_scale_config
        computation_dtype = getattr(torch, mel_cfg.computation_dtype) if mel_cfg.computation_dtype else None
        num_frequency_bins = 1 + stft_cfg.n_fft // 2
        num_mel_filters = mel_cfg.n_mels
        min_frequency = mel_cfg.f_min
        max_frequency = mel_cfg.f_max if mel_cfg.f_max is not None else self.sample_rate / 2
        n_fft = (num_frequency_bins - 1) * 2

        if mel_cfg.triangularize_in_mel_space and mel_cfg.bands_to_zero == 0:
            # Kaldi-exact path: matches torchaudio.compliance.kaldi.get_mel_banks
            mel_filters = self._kaldi_exact_mel_banks(
                num_mel_filters, num_frequency_bins, min_frequency, max_frequency,
                self.sample_rate, n_fft,
            )
        elif mel_cfg.triangularize_in_mel_space:
            mel_filters = self._kaldi_mel_banks_with_zero_bands(
                num_mel_filters, num_frequency_bins, min_frequency, max_frequency,
                self.sample_rate, n_fft, mel_cfg, computation_dtype,
            )
        else:
            mel_filters = self._standard_mel_banks(
                num_mel_filters, num_frequency_bins, min_frequency, max_frequency,
                self.sample_rate, n_fft, mel_cfg, computation_dtype,
            )

        # Cast back when mel computation_dtype doesn't match spectrogram computation_dtype
        if computation_dtype is not None and not spectrogram_config.computation_dtype:
            mel_filters = mel_filters.to(torch.get_default_dtype())
        return mel_filters

    @staticmethod
    def _kaldi_exact_mel_banks(num_mel_filters, num_frequency_bins, min_frequency,
                               max_frequency, sampling_rate, n_fft):
        """Matches torchaudio.compliance.kaldi.get_mel_banks exactly."""
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
        return banks.T

    @staticmethod
    def _kaldi_mel_banks_with_zero_bands(num_mel_filters, num_frequency_bins, min_frequency,
                                         max_frequency, sampling_rate, n_fft, mel_cfg, computation_dtype):
        """Kaldi-style with bands_to_zero > 0."""
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
        zero = torch.zeros(1, dtype=computation_dtype)
        mel_filters = torch.max(zero, torch.min(up_slope, down_slope)).T
        if mel_cfg.bands_to_zero > 0:
            mel_filters = torch.nn.functional.pad(mel_filters, (0, 0, mel_cfg.bands_to_zero, 0))
        return mel_filters

    @staticmethod
    def _standard_mel_banks(num_mel_filters, num_frequency_bins, min_frequency,
                            max_frequency, sampling_rate, n_fft, mel_cfg, computation_dtype):
        """Standard (non-kaldi) triangular mel filter bank."""
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
        return mel_filters

    def _apply_mel_scale(self, features, *, spectrogram_config, **kwargs):
        mel_filters = self.mel_filters.to(device=features.device)
        if spectrogram_config.mel_scale_config.matmul_order == "features_first":
            mel_spec = torch.matmul(features.transpose(-2, -1), mel_filters)
        else:
            # F.linear matches torchaudio's MelScale implementation exactly
            mel_spec = torch.nn.functional.linear(features.transpose(-2, -1), mel_filters.T).transpose(-2, -1)
        return torch.clamp(mel_spec, min=spectrogram_config.mel_floor)

    def _normalize_magnitude(self, features, *, spectrogram_config,
                             reference=1.0, min_value=1e-10, db_range=None,
                             dtype=None, **kwargs):
        log_mel = spectrogram_config.log_mode
        mel_floor = spectrogram_config.mel_floor
        power = spectrogram_config.stft_config.power
        if dtype is None:
            dtype = torch.float32

        if log_mel is None:
            return features

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
                max_vals = result.amax(dim=-2, keepdim=True) if result.ndim > 2 else result.max()
                result = torch.clamp(result, min=max_vals - db_range)
            result = result.to(dtype)
        else:
            raise ValueError(f"Unknown log_mel option: {log_mel}")

        if spectrogram_config.skip_last_frame:
            result = result[..., :-1]

        return result
