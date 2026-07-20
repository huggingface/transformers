# Copyright 2026 HuggingFace Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Common tests for `XxxAudioProcessor` classes.

Mirrors `test_image_processing_common.ImageProcessingTestMixin`. Auto-discovers a model's
sibling backend classes (`torch` and optionally `numpy`) from
`FEATURE_EXTRACTOR_MAPPING_NAMES` keyed by model directory name. Per-model test files set
``self.audio_processor_tester`` in their `setUp` and inherit from this mixin.
"""

from __future__ import annotations

import json
import os
import pathlib
import sys
import tempfile
import unittest

import numpy as np

from transformers.models.auto.feature_extraction_auto import (
    FEATURE_EXTRACTOR_MAPPING_NAMES,
    feature_extractor_class_from_name,
)
from transformers.testing_utils import check_json_file_has_correct_format, require_torch
from transformers.utils import is_torch_available


if is_torch_available():
    import torch


def prepare_audio_inputs(
    batch_size: int = 3,
    sample_rate: int = 16000,
    min_length: float = 1.0,
    max_length: float = 3.0,
    equal_length: bool = False,
    seed: int = 0,
):
    """Generate a batch of fake waveforms with varying lengths."""
    rng = np.random.RandomState(seed)
    if equal_length:
        lengths = [int(max_length * sample_rate)] * batch_size
    else:
        lengths = [int(rng.uniform(min_length, max_length) * sample_rate) for _ in range(batch_size)]
    return [rng.uniform(-1.0, 1.0, size=length).astype(np.float32) for length in lengths]


class AudioProcessingTestMixin:
    """Shared tests for every `XxxAudioProcessor` (and its sibling `XxxAudioProcessorNumpy`).

    Subclasses must set ``self.audio_processor_tester`` in their `setUp`. The tester is
    expected to expose:

      - ``prepare_audio_processor_dict()`` → dict of init kwargs
      - ``batch_size``, ``sample_rate`` (optional, used to generate fake inputs)
    """

    audio_processor_tester = None
    test_classes_to_skip: set[str] = set()
    # Per-model override of the cross-backend parity bar (ADR 0001). Default is the float32
    # noise floor; models with longer numerical chains (custom STFT, unfold + preemphasis,
    # float32/64 mixed ops) can relax to 1e-3 / 1e-4 if needed.
    parity_atol: float = 1e-5
    parity_rtol: float = 1e-5

    def setUp(self):
        # Infer the model_name from the test directory (e.g. "whisper" from tests/models/whisper/...).
        test_file_path = pathlib.Path(sys.modules[self.__class__.__module__].__file__).resolve()
        model_name = test_file_path.parent.name
        try:
            class_names_by_backend = FEATURE_EXTRACTOR_MAPPING_NAMES[model_name]
        except KeyError as e:
            raise ValueError(
                f"No entry for model_name={model_name!r} in FEATURE_EXTRACTOR_MAPPING_NAMES. "
                f"Override `setUp` in your test class to provide the backend mapping."
            ) from e

        self.audio_processing_classes = {
            backend: feature_extractor_class_from_name(class_name)
            for backend, class_name in class_names_by_backend.items()
            if class_name not in self.test_classes_to_skip
        }
        self.audio_processing_classes = {b: c for b, c in self.audio_processing_classes.items() if c is not None}

    # ── Cross-backend parity ──────────────────────────────────────────────

    def _to_torch(self, x):
        if isinstance(x, np.ndarray):
            return torch.from_numpy(x)
        if hasattr(x, "numpy"):
            return x
        return torch.as_tensor(x)

    def _assert_outputs_bit_exact(self, output_a, output_b, *, atol=1e-5, rtol=1e-5):
        """Per ADR 0001, sibling backends must agree within the float32 noise floor —
        `torch.allclose(atol=1e-5, rtol=1e-5)`. The bar is intentionally not stricter
        than `np.fft.rfft` vs `torch.fft.rfft` library divergence allows."""
        keys_a = set(output_a.keys())
        keys_b = set(output_b.keys())
        self.assertEqual(keys_a, keys_b, f"Output keys differ: {keys_a} vs {keys_b}")
        for key in keys_a:
            a = self._to_torch(output_a[key])
            b = self._to_torch(output_b[key])
            self.assertEqual(a.shape, b.shape, f"Shape mismatch for {key!r}: {a.shape} vs {b.shape}")
            # Integer masks must match exactly; only float outputs get tolerance.
            if a.dtype in (torch.bool, torch.int32, torch.int64):
                self.assertTrue(
                    torch.equal(a, b),
                    f"Mask/integer output mismatch for {key!r} (max abs diff: {(a.long() - b.long()).abs().max().item()})",
                )
            else:
                self.assertTrue(
                    torch.allclose(a, b, atol=atol, rtol=rtol),
                    f"Numerical parity violated for output key {key!r}: "
                    f"max abs diff {(a - b).abs().max().item():.3e} exceeds atol={atol:.0e}, rtol={rtol:.0e}",
                )

    @require_torch
    def test_backends_equivalence(self):
        if len(self.audio_processing_classes) < 2:
            self.skipTest("Only one backend registered; cross-backend parity test skipped.")
        if self.audio_processor_tester is None:
            self.skipTest("audio_processor_tester not set; cannot generate fixtures.")

        init_dict = self.audio_processor_tester.prepare_audio_processor_dict()
        waveform = prepare_audio_inputs(batch_size=1, seed=0)[0]

        outputs = {}
        for backend, cls in self.audio_processing_classes.items():
            ap = cls(**init_dict)
            outputs[backend] = ap(waveform, sampling_rate=ap.sample_rate, return_tensors="pt")

        reference_backend, reference_output = next(iter(outputs.items()))
        for backend, output in outputs.items():
            if backend == reference_backend:
                continue
            self._assert_outputs_bit_exact(reference_output, output, atol=self.parity_atol, rtol=self.parity_rtol)

    @require_torch
    def test_backends_equivalence_batched(self):
        if len(self.audio_processing_classes) < 2:
            self.skipTest("Only one backend registered; cross-backend parity test skipped.")
        if self.audio_processor_tester is None:
            self.skipTest("audio_processor_tester not set; cannot generate fixtures.")

        init_dict = self.audio_processor_tester.prepare_audio_processor_dict()
        waveforms = prepare_audio_inputs(batch_size=3, equal_length=False, seed=0)

        outputs = {}
        for backend, cls in self.audio_processing_classes.items():
            ap = cls(**init_dict)
            outputs[backend] = ap(waveforms, sampling_rate=ap.sample_rate, return_tensors="pt")

        reference_backend, reference_output = next(iter(outputs.items()))
        for backend, output in outputs.items():
            if backend == reference_backend:
                continue
            self._assert_outputs_bit_exact(reference_output, output, atol=self.parity_atol, rtol=self.parity_rtol)

    # ── JSON round-trip ───────────────────────────────────────────────────

    def test_audio_processor_to_json_string(self):
        if self.audio_processor_tester is None:
            self.skipTest("audio_processor_tester not set.")
        init_dict = self.audio_processor_tester.prepare_audio_processor_dict()
        for cls in self.audio_processing_classes.values():
            ap = cls(**init_dict)
            obj = json.loads(ap.to_json_string())
            self.assertEqual(obj["audio_processor_type"], cls.__name__)

    def test_audio_processor_to_json_file(self):
        if self.audio_processor_tester is None:
            self.skipTest("audio_processor_tester not set.")
        init_dict = self.audio_processor_tester.prepare_audio_processor_dict()
        for cls in self.audio_processing_classes.values():
            ap_first = cls(**init_dict)
            with tempfile.TemporaryDirectory() as tmpdirname:
                json_file_path = os.path.join(tmpdirname, "audio_processor.json")
                ap_first.to_json_file(json_file_path)
                ap_second = cls.from_json_file(json_file_path)
            self.assertEqual(ap_second.to_dict(), ap_first.to_dict())

    def test_audio_processor_from_and_save_pretrained(self):
        if self.audio_processor_tester is None:
            self.skipTest("audio_processor_tester not set.")
        init_dict = self.audio_processor_tester.prepare_audio_processor_dict()
        for cls in self.audio_processing_classes.values():
            ap_first = cls(**init_dict)
            with tempfile.TemporaryDirectory() as tmpdirname:
                saved_file = ap_first.save_pretrained(tmpdirname)[0]
                check_json_file_has_correct_format(saved_file)
                ap_second = cls.from_pretrained(tmpdirname)
            self.assertEqual(ap_second.to_dict(), ap_first.to_dict())

    def test_audio_processor_save_load_with_autoaudioprocessor(self):
        if self.audio_processor_tester is None:
            self.skipTest("audio_processor_tester not set.")
        from transformers.models.auto.feature_extraction_auto import AutoAudioProcessor

        init_dict = self.audio_processor_tester.prepare_audio_processor_dict()
        for backend, cls in self.audio_processing_classes.items():
            ap_first = cls(**init_dict)
            with tempfile.TemporaryDirectory() as tmpdirname:
                ap_first.save_pretrained(tmpdirname)
                ap_second = AutoAudioProcessor.from_pretrained(tmpdirname, backend=backend)
            self.assertEqual(type(ap_second), cls)
            self.assertEqual(ap_second.to_dict(), ap_first.to_dict())

    # ── Basic instantiation ───────────────────────────────────────────────

    def test_init_without_params(self):
        for cls in self.audio_processing_classes.values():
            ap = cls()
            self.assertIsNotNone(ap)
