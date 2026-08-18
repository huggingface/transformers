# Copyright 2026 HuggingFace Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Unit tests for base-layer audio pipeline knobs added by the LOC-minimization refactor."""

import unittest

import numpy as np

from transformers.audio_utils import SpectrogramConfig, StftConfig
from transformers.models.whisper.audio_processing_numpy_whisper import WhisperAudioProcessorNumpy
from transformers.testing_utils import require_torch
from transformers.utils import is_torch_available


if is_torch_available():
    import torch

    from transformers.models.whisper.audio_processing_whisper import WhisperAudioProcessor


def legacy_usm_window(win_length):
    arange = np.arange(win_length, dtype=np.float32)
    return (0.5 * (1 - np.cos(2 * np.pi * arange / win_length))).astype(np.float32)


@require_torch
class WindowF32Test(unittest.TestCase):
    def test_hann_window_f32_bit_exact_both_backends(self):
        np_proc = WhisperAudioProcessorNumpy()
        pt_proc = WhisperAudioProcessor()
        for win_length in (512, 513):
            cfg = StftConfig(window_fn="hann_window_f32", win_length=win_length)
            expected = legacy_usm_window(win_length)
            np_window = np_proc._create_stft_window(win_length, cfg, np.zeros(win_length, dtype=np.float32))
            pt_window = pt_proc._create_stft_window(win_length, cfg, torch.zeros(win_length))
            self.assertEqual(np_window.dtype, np.float32)
            self.assertEqual(pt_window.dtype, torch.float32)
            self.assertTrue(np.array_equal(np_window, expected))
            self.assertTrue(np.array_equal(pt_window.numpy(), expected))


class StftConfigFieldsTest(unittest.TestCase):
    def test_new_stft_config_fields_roundtrip(self):
        cfg = StftConfig(frame_extension=1, fft_dtype="float64", center="left")
        restored = StftConfig.from_dict(cfg.to_dict())
        self.assertEqual(restored.frame_extension, 1)
        self.assertEqual(restored.fft_dtype, "float64")
        self.assertEqual(restored.center, "left")
        # SpectrogramConfig round-trip through to_dict/from_dict
        spec = SpectrogramConfig(stft_config=cfg, preemphasis_mode="htk_per_frame")
        spec2 = SpectrogramConfig.from_dict(spec.to_dict())
        self.assertEqual(spec2.stft_config.frame_extension, 1)
        self.assertEqual(spec2.preemphasis_mode, "htk_per_frame")


class ExtendedFramingNumpyTest(unittest.TestCase):
    """Numpy-only (must not skip on torch-less CI): the new config knobs must reproduce
    Gemma3n's hand-written USM pipeline bit-exactly."""

    def _legacy_gemma3n_stft(self, audio):
        # Verbatim (numpy) legacy implementation: unfold at win_length+1, HTK preemphasis,
        # float32 window, float64 FFT via numpy promotion.
        win_length, hop_length, n_fft, preemphasis = 512, 160, 1024, 0.97
        frames = np.lib.stride_tricks.sliding_window_view(audio, win_length + 1)[::hop_length]
        first = frames[..., :1] * (1.0 - preemphasis)
        rest = frames[..., 1:-1] - preemphasis * frames[..., :-2]
        frames = np.concatenate([first, rest], axis=-1)
        window = legacy_usm_window(win_length)
        frames = frames * window
        return np.abs(np.fft.rfft(frames, n=n_fft, axis=-1))  # (time, freq)

    def test_frame_extension_htk_matches_legacy(self):
        cfg = SpectrogramConfig(
            stft_config=StftConfig(
                n_fft=1024,
                win_length=512,
                hop_length=160,
                power=1.0,
                center=False,
                window_fn="hann_window_f32",
                frame_extension=1,
                fft_dtype="float64",
            ),
            preemphasis=0.97,
            preemphasis_mode="htk_per_frame",
        )
        proc = WhisperAudioProcessorNumpy()
        rng = np.random.RandomState(0)
        audio = rng.uniform(-1, 1, size=16000).astype(np.float32)
        got = proc._stft(audio, spectrogram_config=cfg)  # canonical (freq, time)
        expected = self._legacy_gemma3n_stft(audio).T
        self.assertTrue(np.array_equal(got, expected))


@require_torch
class ExtendedFramingTest(unittest.TestCase):
    """Cross-backend checks for the extended-framing knobs (needs torch)."""

    def _legacy_gemma4_stft(self, audio_batch, win_length=320, hop_length=160, n_fft=512):
        # verbatim legacy implementation: semicausal left-pad, unfold at win_length + 1
        pad_left = win_length // 2
        padded = np.pad(audio_batch, ((0, 0), (pad_left, 0)), mode="constant")
        frame_size = win_length + 1
        num_frames = (padded.shape[-1] - frame_size) // hop_length + 1
        frames = np.stack([padded[:, i * hop_length : i * hop_length + frame_size] for i in range(num_frames)], axis=1)
        frames = frames[..., :-1]
        window = legacy_usm_window(win_length)
        frames = frames * window
        spec = np.fft.rfft(frames, n=n_fft, axis=-1)
        return np.abs(spec).transpose(0, 2, 1)  # (batch, freq, time)

    def test_center_left_preserves_ndim_and_matches_legacy(self):
        """`center == "left"` must preserve input ndim (1-D in -> 2-D out; batched -> 3-D).
        `fft_dtype="float64"` is required for bit-exactness against the raw-rfft reference."""
        cfg = SpectrogramConfig(
            stft_config=StftConfig(
                n_fft=512,
                win_length=320,
                hop_length=160,
                power=1.0,
                center="left",
                window_fn="hann_window_f32",
                frame_extension=1,
                fft_dtype="float64",
            ),
            preemphasis=0.0,
            preemphasis_mode="htk_per_frame",
        )
        rng = np.random.RandomState(0)
        audio_1d = rng.uniform(-1, 1, size=16000).astype(np.float32)
        audio_batch = np.stack([audio_1d, rng.uniform(-1, 1, size=16000).astype(np.float32)])

        np_proc = WhisperAudioProcessorNumpy()
        got_1d = np_proc._stft(audio_1d, spectrogram_config=cfg)
        self.assertEqual(got_1d.ndim, 2)

        got_batched_np = np_proc._stft(audio_batch, spectrogram_config=cfg)
        self.assertEqual(got_batched_np.ndim, 3)  # (batch, freq, time)
        # numpy keeps float64 magnitudes; torch casts back to float32 (computation_dtype unset)
        self.assertEqual(got_batched_np.dtype, np.float64)

        pt_proc = WhisperAudioProcessor()
        got_batched_pt = pt_proc._stft(torch.from_numpy(audio_batch), spectrogram_config=cfg)
        self.assertEqual(tuple(got_batched_pt.shape), got_batched_np.shape)
        self.assertEqual(got_batched_pt.dtype, torch.float32)
        # rtol, not atol: magnitudes run up to ~22 and the divergence is ~1 float32 ulp
        self.assertTrue(np.allclose(got_batched_pt.numpy(), got_batched_np, rtol=1e-6, atol=0))

        expected = self._legacy_gemma4_stft(audio_batch)
        self.assertTrue(np.array_equal(got_batched_np, expected))


class FrameExtensionValidationTest(unittest.TestCase):
    """Numpy-only: must not skip on torch-less CI."""

    def test_frame_extension_validation_errors(self):
        """Unsupported knob combinations must fail loudly."""
        np_proc = WhisperAudioProcessorNumpy()
        audio = np.zeros(1600, dtype=np.float32)

        # htk_per_frame without frame_extension=1
        with self.assertRaises(ValueError):
            np_proc._stft(
                audio,
                spectrogram_config=SpectrogramConfig(
                    stft_config=StftConfig(n_fft=512, win_length=320, hop_length=160),
                    preemphasis_mode="htk_per_frame",
                ),
            )

        # frame_extension != 1
        with self.assertRaises(ValueError):
            np_proc._stft(
                audio,
                spectrogram_config=SpectrogramConfig(
                    stft_config=StftConfig(n_fft=512, win_length=320, hop_length=160, frame_extension=2),
                ),
            )

        # frame_extension with symmetric centering
        with self.assertRaises(ValueError):
            np_proc._stft(
                audio,
                spectrogram_config=SpectrogramConfig(
                    stft_config=StftConfig(n_fft=512, win_length=320, hop_length=160, frame_extension=1, center=True),
                ),
            )

        # remove_dc_offset with frame_extension
        with self.assertRaises(ValueError):
            np_proc._stft(
                audio,
                spectrogram_config=SpectrogramConfig(
                    stft_config=StftConfig(n_fft=512, win_length=320, hop_length=160, frame_extension=1, center=False),
                    remove_dc_offset=True,
                ),
            )

        # invalid fft_dtype value
        with self.assertRaises(ValueError):
            np_proc._stft(
                audio,
                spectrogram_config=SpectrogramConfig(
                    stft_config=StftConfig(
                        n_fft=512, win_length=320, hop_length=160, frame_extension=1, center=False, fft_dtype="f64"
                    ),
                ),
            )


@require_torch
class MaskedMeanVarNormalizeTest(unittest.TestCase):
    def _reference(self, features, lengths, eps=1e-5):
        # Verbatim legacy Cohere-ASR numpy implementation
        features_lengths = np.asarray(lengths)
        attention_mask = np.arange(features.shape[1])[None, :] < features_lengths[:, None]
        mask = np.expand_dims(attention_mask, axis=-1)
        lengths_f = features_lengths.astype(features.dtype)
        mel_masked = features * mask
        mean = np.expand_dims(mel_masked.sum(axis=1) / np.expand_dims(lengths_f, axis=-1), axis=1)
        variance = ((mel_masked - mean) ** 2 * mask).sum(axis=1) / np.expand_dims(lengths_f - 1, axis=-1)
        std = np.expand_dims(np.sqrt(variance), axis=1)
        return (features - mean) / (std + eps) * mask

    def _reference_torch(self, features, lengths, eps=1e-5):
        # Verbatim legacy Cohere-ASR torch implementation (audio_processing_cohere_asr.py::_postprocess_output)
        features_lengths = torch.tensor(lengths)
        attention_mask = torch.arange(features.shape[1])[None, :] < features_lengths[:, None]
        mask = attention_mask.unsqueeze(-1)
        mel_masked = features * mask
        mean = (mel_masked.sum(dim=1) / features_lengths.unsqueeze(-1)).unsqueeze(1)
        variance = ((mel_masked - mean) ** 2 * mask).sum(dim=1) / (features_lengths - 1).unsqueeze(-1)
        std = torch.sqrt(variance).unsqueeze(1)
        return (features - mean) / (std + eps) * mask

    def test_matches_legacy_cohere_numpy_and_torch(self):
        rng = np.random.RandomState(0)
        features = rng.uniform(-1, 1, size=(3, 50, 8)).astype(np.float32)
        lengths = [50, 30, 17]
        expected_np = self._reference(features, lengths)
        expected_pt = self._reference_torch(torch.from_numpy(features), lengths)

        np_out = WhisperAudioProcessorNumpy()._masked_mean_var_normalize(features, lengths, epsilon=1e-5)
        pt_out = WhisperAudioProcessor()._masked_mean_var_normalize(torch.from_numpy(features), lengths, epsilon=1e-5)

        # each backend must bit-match its OWN legacy oracle (they were never identical to each other)
        self.assertTrue(np.array_equal(np_out, expected_np))
        self.assertTrue(np.array_equal(pt_out.numpy(), expected_pt.numpy()))

        # cross-backend sanity: rtol=1e-5 covers the measured ~5.9e-6 kernel-rounding divergence
        self.assertTrue(np.allclose(pt_out.numpy(), np_out, rtol=1e-5, atol=0))
