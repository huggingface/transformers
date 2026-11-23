# Copyright 2026 HuggingFace Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Tests for `ClapAudioProcessor` on the numpy and torch backends."""

from __future__ import annotations

import unittest

import numpy as np
from huggingface_hub.errors import StrictDataclassFieldValidationError

from transformers.testing_utils import require_torch

from ...test_audio_processing_common import AudioProcessingTestMixin


class ClapAudioProcessingTester:
    """Provides init kwargs and fixture parameters for the CLAP audio processor tests."""

    sample_rate = 48000

    def prepare_audio_processor_dict(self) -> dict:
        return {}


@require_torch
class ClapAudioProcessingTest(AudioProcessingTestMixin, unittest.TestCase):
    # CLAP's full-batch padded mel + 48 kHz STFT accumulates a slightly larger cross-backend
    # divergence than the strict float32 noise floor — empirically up to ~8e-5 on batched inputs.
    parity_atol = 1e-4
    parity_rtol = 1e-4

    def setUp(self):
        self.audio_processor_tester = ClapAudioProcessingTester()
        super().setUp()

    def test_class_docstring_is_kept(self):
        # `required_ops` sits after the docstring; above it, `__doc__` would be None.
        self.assertIsNotNone(self.audio_processing_class.__doc__)
        self.assertIn("HTSAT", self.audio_processing_class.__doc__)
        self.assertEqual(self.audio_processing_class.required_ops, ("bilinear_resize",))

    def test_legacy_fusion_config_names_its_mel_bank(self):
        # A legacy fusion config implies torchaudio-default mels; the mapping writes that bank into
        # the config once, at load. The mode alone never changes the bank.
        for backend in self.audio_backends:
            with self.subTest(backend=backend):
                processor = self.audio_processing_class.from_dict({"truncation": "fusion"}, backend=backend)
                mel_config = processor.spectrogram_config.mel_scale_config
                self.assertEqual(processor.truncation_mode, "fusion")
                self.assertEqual(mel_config.mel_scale, "htk")
                self.assertIsNone(mel_config.norm)
                self.assertEqual(processor.mel_filters.shape[-1], mel_config.n_mels)
                restored = self.audio_processing_class.from_dict(processor.to_dict(), backend=backend)
                self.assertEqual(restored.spectrogram_config.mel_scale_config.mel_scale, "htk")
                self.assertEqual(
                    self.audio_processing_class(
                        backend=backend, truncation_mode="fusion"
                    ).spectrogram_config.mel_scale_config.norm,
                    "slaney",
                )

    def test_mode_switching_keeps_config_and_bank_local(self):
        waveform = np.random.RandomState(0).randn(12000).astype(np.float32)
        for backend in self.audio_backends:
            processor = self.audio_processing_class(backend=backend, max_length=4800)
            initial_config = processor.to_dict()
            for mode in ("fusion", "rand_trunc", "fusion"):
                with self.subTest(backend=backend, mode=mode):
                    # Same bank as the instance, so a per-call switch equals a processor configured
                    # with that mode on the same bank.
                    reference = self.audio_processing_class(backend=backend, max_length=4800, truncation_mode=mode)
                    np.random.seed(7)
                    expected = reference(waveform, return_tensors="np")
                    np.random.seed(7)
                    actual = processor(waveform, truncation_mode=mode, return_tensors="np")
                    np.testing.assert_array_equal(actual["audio_features"], expected["audio_features"])
                    np.testing.assert_array_equal(actual["is_longer"], expected["is_longer"])
                    self.assertEqual(processor.to_dict(), initial_config)

    def test_fusion_frame_threshold_and_all_short_batch(self):
        rng = np.random.RandomState(0)
        for backend in self.audio_backends:
            processor = self.audio_processing_class(backend=backend, max_length=4800, truncation_mode="fusion")
            clips = [rng.randn(n).astype(np.float32) for n in (4800, 4801, 5280)]
            output = processor(clips, return_tensors="np")
            self.assertEqual(output["audio_features"].shape, (3, 4, 11, 64))
            np.testing.assert_array_equal(output["is_longer"], [[False], [False], [True]])
            # The batch-level fallback selects exactly one clip even when none needs fusion.
            output = processor(clips[:2], return_tensors="np")
            self.assertEqual(output["is_longer"].sum(), 1)

    def test_padding_boolean(self):
        # `padding=True` fills with the instance's `padding_mode`, like no padding argument or the
        # legacy `padding="repeatpad"` spelling (ported from upstream #49960).
        waveform = np.random.RandomState(0).randn(800).astype(np.float32)
        for backend in self.audio_backends:
            with self.subTest(backend=backend):
                processor = self.audio_processing_class(backend=backend)
                outputs = []
                for kwargs in ({}, {"padding": True}, {"padding": "repeatpad"}):
                    np.random.seed(0)
                    outputs.append(processor(waveform, return_tensors="np", **kwargs)["audio_features"])
                np.testing.assert_array_equal(outputs[1], outputs[0])
                np.testing.assert_array_equal(outputs[1], outputs[2])

    def test_rejects_invalid_clap_workflow_options(self):
        waveform = np.ones(4800, dtype=np.float32)
        incompatible = (
            {"padding": False},
            {"truncation_mode": "unknown"},
            {"padding_mode": "unknown"},
        )
        for backend in self.audio_backends:
            processor = self.audio_processing_class(backend=backend, max_length=4800)
            for kwargs in incompatible:
                with self.subTest(backend=backend, kwargs=kwargs):
                    # Schema violations (`Literal`, `Annotated`) surface as the hub's strict-dataclass error.
                    with self.assertRaises((ValueError, StrictDataclassFieldValidationError)):
                        processor(waveform, **kwargs)

    def test_custom_workflow_inherits_common_schema(self):
        from transformers.processing_utils import AudioKwargs

        for backend in self.audio_backends:
            processor = self.audio_processing_class(backend=backend)
            restored = self.audio_processing_class.from_dict(processor.to_dict(), backend=backend)
            self.assertEqual(restored.to_dict(), processor.to_dict())
            self.assertTrue(AudioKwargs.__annotations__.keys() <= processor.valid_kwargs.__annotations__.keys())
