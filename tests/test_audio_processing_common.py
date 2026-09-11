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

import pathlib
import sys

import numpy as np

from transformers.models.auto.feature_extraction_auto import (
    FEATURE_EXTRACTOR_MAPPING_NAMES,
    feature_extractor_class_from_name,
)
from transformers.testing_utils import require_torch
from transformers.utils import is_torch_available

from .test_preprocessing_common import PreprocessingTesterMixin


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


class AudioProcessingTestMixin(PreprocessingTesterMixin):
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

    # ── `PreprocessingTesterMixin` surface ────────────────────────────────

    @property
    def processing_classes(self) -> dict:
        return self.audio_processing_classes

    @property
    def processor_dict(self) -> dict:
        return self.audio_processor_tester.prepare_audio_processor_dict()

    @property
    def auto_class(self):
        from transformers.models.auto.feature_extraction_auto import AutoAudioProcessor

        return AutoAudioProcessor

    def _prepare_inputs(self):
        return prepare_audio_inputs(batch_size=1, seed=0)[0]

    def _to_torch(self, x):
        if isinstance(x, np.ndarray):
            return torch.from_numpy(x)
        if hasattr(x, "numpy"):
            return x
        return torch.as_tensor(x)

    def _metadata_keys(self):
        """Output keys the processors declare as non-array metadata via `skip_tensor_conversion`
        (e.g. Cohere-ASR's `audio_chunk_index`), compared verbatim instead of as tensors."""
        return {key for cls in self.audio_processing_classes.values() for key in cls.skip_tensor_conversion}

    def _assert_outputs_bit_exact(self, output_a, output_b, *, atol=1e-5, rtol=1e-5):
        """Per ADR 0001, sibling backends must agree within the float32 noise floor —
        `torch.allclose(atol=1e-5, rtol=1e-5)`. The bar is intentionally not stricter
        than `np.fft.rfft` vs `torch.fft.rfft` library divergence allows."""
        keys_a = set(output_a.keys())
        keys_b = set(output_b.keys())
        self.assertEqual(keys_a, keys_b, f"Output keys differ: {keys_a} vs {keys_b}")
        metadata_keys = self._metadata_keys()
        for key in keys_a:
            if key in metadata_keys:
                self.assertEqual(output_a[key], output_b[key], f"Metadata mismatch for {key!r}")
                continue
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
            outputs[backend] = ap(waveform, sampling_rate=ap.sampling_rate, return_tensors="pt")

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
            outputs[backend] = ap(waveforms, sampling_rate=ap.sampling_rate, return_tensors="pt")

        reference_backend, reference_output = next(iter(outputs.items()))
        for backend, output in outputs.items():
            if backend == reference_backend:
                continue
            self._assert_outputs_bit_exact(reference_output, output, atol=self.parity_atol, rtol=self.parity_rtol)

    # ── Padding / truncation ──────────────────────────────────────────────
    # Fixture geometry for the padding/truncation matrix. Lengths are deliberately not round
    # so that `pad_to_multiple_of` rounding is actually exercised -- the legacy fixture's round
    # lengths made several of its own assertions vacuous. Mirrors the legacy tester's
    # `min_seq_length` + i * `seq_length_diff`.
    pad_test_min_length = 8003
    pad_test_length_diff = 4001
    pad_test_batch_size = 3
    # Whether the region outside a padded input's `ranges` holds `padding_value`. False for
    # processors that fill short audio some other way -- CLAP tiles it (`_pad_waveform` with
    # `padding_mode="repeatpad"`) and only zero-pads whatever the tiling leaves over.
    pad_fills_with_padding_value: bool = True

    def _padding_fixture(self, backend: str):
        """Ascending-length waveforms in the array type `backend`'s `pad` expects."""
        lengths = [self.pad_test_min_length + i * self.pad_test_length_diff for i in range(self.pad_test_batch_size)]
        rng = np.random.RandomState(0)
        audio = [rng.uniform(-1.0, 1.0, size=length).astype(np.float32) for length in lengths]
        if backend == "torch":
            audio = [torch.from_numpy(audio_el) for audio_el in audio]
        return audio, lengths

    @staticmethod
    def _lengths(audio) -> list[int]:
        return [audio_el.shape[-1] for audio_el in audio]

    @staticmethod
    def _round_up(length: int, multiple: int) -> int:
        return length if length % multiple == 0 else (length // multiple + 1) * multiple

    def _assert_padding_region(self, processor, audio, ranges):
        """Everything outside `ranges` must hold `padding_value` (side-agnostic, so this
        covers `padding_side="left"` without a separate case)."""
        if processor.padding_value is None or not self.pad_fills_with_padding_value:
            return
        for audio_el, (start, end) in zip(audio, ranges):
            values = np.asarray(audio_el)
            padded_region = np.concatenate([values[:start], values[end:]])
            if padded_region.size:
                self.assertTrue(
                    np.allclose(padded_region, processor.padding_value, atol=1e-6),
                    f"padded region is not filled with padding_value={processor.padding_value}",
                )

    @require_torch
    def test_padding(self):
        """Padding-strategy matrix.

        Ported from `tests/test_sequence_feature_extraction_common.py` (main @ ccba41e1c3):
        `SequenceFeatureExtractionTestMixin._check_padding`, L79-L200.

        Adapted, because `pad` is a different method now -- it takes a raw list and returns
        `(audio, ranges)` rather than consuming and returning a `BatchFeature`, so the legacy
        `pad(BatchFeature({input_name: ...}))[input_name]` mechanics are gone, and there is no
        `return_tensors=`. The legacy `feature_size` shape assertions are dropped (`feature_size`
        exists on 4 of 60 audio processors), and its padding-value sum arithmetic is replaced by
        a direct check that the region outside `ranges` holds `padding_value` -- which also pins
        `ranges`, the new API's replacement for the returned attention mask. The legacy
        from_list/from_array axis becomes the backend loop, since `pad` requires `.shape`.
        """
        init_dict = self.audio_processor_tester.prepare_audio_processor_dict()
        for backend, cls in self.audio_processing_classes.items():
            with self.subTest(backend=backend):
                processor = cls(**init_dict)
                audio, lengths = self._padding_fixture(backend)
                longest = lengths[-1]

                # `padding=False` leaves every length untouched.
                out, _ = processor.pad(audio, padding=False)
                self.assertEqual(self._lengths(out), lengths)

                # `padding="longest"` equalizes to the longest input.
                out_longest, ranges_longest = processor.pad(audio, padding="longest")
                self.assertEqual(self._lengths(out_longest), [longest] * len(lengths))
                self.assertEqual([end - start for start, end in ranges_longest], lengths)
                self._assert_padding_region(processor, out_longest, ranges_longest)

                # `padding="max_length"` at the same target is equivalent to `"longest"`.
                out_max, _ = processor.pad(audio, padding="max_length", max_length=longest)
                self.assertEqual(self._lengths(out_max), [longest] * len(lengths))
                for from_longest, from_max in zip(out_longest, out_max):
                    self.assertTrue(np.allclose(np.asarray(from_longest), np.asarray(from_max), atol=1e-3))

                # `max_length` is required by `padding="max_length"`.
                with self.assertRaises(ValueError):
                    processor.pad(audio, padding="max_length")

                # `pad_to_multiple_of` alone rounds the (implicit longest) target up.
                out_multiple, _ = processor.pad(audio, pad_to_multiple_of=10)
                out_multiple_longest, _ = processor.pad(audio, padding="longest", pad_to_multiple_of=10)
                self.assertEqual(self._lengths(out_multiple), [self._round_up(longest, 10)] * len(lengths))
                self.assertTrue(all(length % 10 == 0 for length in self._lengths(out_multiple)))
                self.assertEqual(self._lengths(out_multiple), self._lengths(out_multiple_longest))

                # `pad_to_multiple_of` rounds an explicit `max_length` up too.
                pad_max_length = longest + self.pad_test_length_diff
                out_rounded, ranges_rounded = processor.pad(
                    audio, padding="max_length", max_length=pad_max_length, pad_to_multiple_of=12
                )
                expected = self._round_up(pad_max_length, 12)
                self.assertEqual(self._lengths(out_rounded), [expected] * len(lengths))
                self._assert_padding_region(processor, out_rounded, ranges_rounded)

    @require_torch
    def test_truncation(self):
        """Truncation matrix, including truncation combined with padding and `pad_to_multiple_of`.

        Ported from `tests/test_sequence_feature_extraction_common.py` (main @ ccba41e1c3):
        `SequenceFeatureExtractionTestMixin._check_truncation`, L201-L330. Same adaptations as
        `test_padding`.

        Deliberate contract change: legacy raised `ValueError` when `truncation=True` was combined
        with anything but `padding="max_length"` (L287-L297 there). The new `pad` truncates first
        and pads afterwards, which makes `truncation=True, padding="longest"` meaningful, so that
        combination is asserted to *work*. What remains required is `max_length` itself -- without
        it there is nothing to truncate to.
        """
        init_dict = self.audio_processor_tester.prepare_audio_processor_dict()
        for backend, cls in self.audio_processing_classes.items():
            with self.subTest(backend=backend):
                processor = cls(**init_dict)
                audio, lengths = self._padding_fixture(backend)
                shortest, middle = lengths[0], lengths[1]

                # Truncating to the shortest input equalizes the batch...
                out, _ = processor.pad(audio, padding="max_length", max_length=shortest, truncation=True)
                self.assertEqual(self._lengths(out), [shortest] * len(lengths))

                # ...whereas without truncation the longer inputs keep their own length.
                out, _ = processor.pad(audio, padding="max_length", max_length=shortest)
                self.assertEqual(self._lengths(out), lengths)

                # Truncating to the middle input truncates the longest and pads the shortest.
                out, ranges = processor.pad(audio, padding="max_length", max_length=middle, truncation=True)
                self.assertEqual(self._lengths(out), [middle] * len(lengths))
                self.assertEqual([end - start for start, end in ranges], [shortest, middle, middle])
                self._assert_padding_region(processor, out, ranges)

                # Truncation composes with `padding="longest"` (see docstring: legacy forbade this).
                out, _ = processor.pad(audio, padding="longest", max_length=middle, truncation=True)
                self.assertEqual(self._lengths(out), [middle] * len(lengths))

                # `pad_to_multiple_of` rounds the truncation target up as well.
                out, _ = processor.pad(
                    audio, padding="max_length", max_length=shortest, pad_to_multiple_of=12, truncation=True
                )
                self.assertEqual(self._lengths(out), [self._round_up(shortest, 12)] * len(lengths))

                # `truncation=True` requires `max_length`, whatever the padding strategy.
                for padding in (False, "longest", "max_length"):
                    with self.subTest(padding=padding), self.assertRaises(ValueError):
                        processor.pad(audio, padding=padding, truncation=True)

    @require_torch
    def test_call_padding_equalizes_batch(self):
        """`pad` is reached from `preprocess`, and the emitted mask tracks the real spans.

        The `pad`-level matrix above cannot catch `preprocess` being wired to something else, and
        `_pad_features` (`audio_processing_utils.py:349`) is a second, separate padding path used
        for already-extracted features -- so this asserts the end-to-end shape and mask contract
        without hardcoding any model's framing arithmetic.
        """
        init_dict = self.audio_processor_tester.prepare_audio_processor_dict()
        for backend, cls in self.audio_processing_classes.items():
            with self.subTest(backend=backend):
                processor = cls(**init_dict)
                _, lengths = self._padding_fixture(backend)
                rng = np.random.RandomState(0)
                waveforms = [rng.uniform(-1.0, 1.0, size=length).astype(np.float32) for length in lengths]

                encoding = processor(waveforms, return_tensors="pt")
                feature_key = processor.model_input_names[0]
                features = encoding[feature_key]

                # A padded batch is rectangular, one row per input.
                self.assertEqual(features.shape[0], len(lengths))

                if not processor.return_padding_mask or len(processor.model_input_names) < 2:
                    continue
                mask = encoding[processor.model_input_names[1]]
                self.assertEqual(mask.shape[0], len(lengths))
                # Longer input => at least as many valid frames. Framing makes the exact count
                # model-specific, but the ordering is a contract.
                valid_per_row = mask.reshape(mask.shape[0], -1).sum(-1).tolist()
                self.assertEqual(
                    valid_per_row,
                    sorted(valid_per_row),
                    f"mask valid-frame counts {valid_per_row} are not monotonic in input length {lengths}",
                )

    # ── JSON round-trip ───────────────────────────────────────────────────

    # ── Basic instantiation ───────────────────────────────────────────────
