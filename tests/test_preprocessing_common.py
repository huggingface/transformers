# Copyright 2026 HuggingFace Inc.
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
"""Shared contract tests for every preprocessor, whatever its modality.

`PreprocessingTesterMixin` owns the tests whose code-under-test lives in
`src/transformers/preprocessing_base.py` -- serialization, `from_dict`/`from_pretrained`,
attribute resolution from `valid_kwargs`, `BatchFeature.to()`, and cross-backend loading. The
modality mixins (`ImageProcessingTestMixin`, `AudioProcessingTestMixin`,
`VideoProcessingTestMixin`) inherit it and keep only what genuinely differs: input fixtures,
output shapes, and the numeric tolerances their backends need.

The test hierarchy mirrors the source hierarchy: `PreprocessingMixin` is subclassed by
`ImageProcessingMixin` / `AudioProcessingMixin`, so a test belongs here exactly when the
behaviour it pins is declared in `preprocessing_base.py`.

Modality mixins supply a three-item surface:

  - ``processing_classes``  -- ``{backend_name: processor_class}``
  - ``processor_dict``      -- init kwargs for those classes
  - ``auto_class``          -- the matching ``AutoXxx`` class

plus, optionally, ``_load_with_auto`` when the Auto class does not take ``backend=``, and
``backend_specific_keys`` for config keys allowed to differ between sibling backends.
"""

from __future__ import annotations

import json
import os
import tempfile
from copy import deepcopy

from transformers.testing_utils import check_json_file_has_correct_format, require_torch
from transformers.utils import is_torch_available


if is_torch_available():
    import torch


class PreprocessingTesterMixin:
    # ── Abstract surface, supplied by the modality mixin ──────────────────

    #: The modality's `AutoXxx` class. Not read off `_auto_class_default`: `BaseVideoProcessor`
    #: inherits `"AutoImageProcessor"` from the image chain, which is not what video loads with.
    auto_class = None
    #: Config keys allowed to differ between sibling backends (e.g. image's `default_to_square`,
    #: `data_format`). Every other extra key must be `None` on both sides.
    backend_specific_keys: set[str] = set()
    #: Set by modality mixins that support `BatchFeature.to()` dtype/device casting.
    test_cast_dtype = None
    #: Boolean switches that cannot be flipped to `None` and re-run in isolation because
    #: enabling them requires companion arguments (video's `do_sample_frames` needs
    #: `video_metadata`). Modality mixins extend this; the default is to test every switch.
    none_resolution_skip_kwargs: set[str] = set()

    @property
    def processing_classes(self) -> dict:
        """`{backend_name: processor_class}` for the model under test."""
        raise NotImplementedError(f"{type(self).__name__} must provide `processing_classes`")

    @property
    def processor_dict(self) -> dict:
        """Init kwargs shared by every backend of the model under test."""
        raise NotImplementedError(f"{type(self).__name__} must provide `processor_dict`")

    def _load_with_auto(self, tmpdirname, backend_name):
        """Reload a saved processor through the modality's Auto class."""
        return self.auto_class.from_pretrained(tmpdirname, backend=backend_name)

    def _main_output_key(self, processor):
        """The key `__call__` puts its primary array under."""
        return processor.model_input_names[0]

    def _prepare_inputs(self):
        """A small valid input batch. Modality mixins override; tests needing it skip without."""
        raise NotImplementedError

    # ── Serialization round-trips ─────────────────────────────────────────

    def test_to_json_string(self):
        for processing_class in self.processing_classes.values():
            processor = processing_class(**self.processor_dict)
            obj = json.loads(processor.to_json_string())
            for key, value in self.processor_dict.items():
                self.assertEqual(obj[key], value, f"`{key}` did not survive `to_json_string`")

    def test_to_json_file(self):
        for processing_class in self.processing_classes.values():
            first = processing_class(**self.processor_dict)
            with tempfile.TemporaryDirectory() as tmpdirname:
                json_file_path = os.path.join(tmpdirname, "processor.json")
                first.to_json_file(json_file_path)
                second = processing_class.from_json_file(json_file_path)
            self.assertEqual(second.to_dict(), first.to_dict())

    def test_from_and_save_pretrained(self):
        for processing_class in self.processing_classes.values():
            first = processing_class(**self.processor_dict)
            with tempfile.TemporaryDirectory() as tmpdirname:
                saved_file = first.save_pretrained(tmpdirname)[0]
                check_json_file_has_correct_format(saved_file)
                second = processing_class.from_pretrained(tmpdirname)
            self.assertEqual(second.to_dict(), first.to_dict())

    def test_save_load_with_auto_class(self):
        if self.auto_class is None:
            self.skipTest("no `auto_class` declared")
        for backend_name, processing_class in self.processing_classes.items():
            first = processing_class(**self.processor_dict)
            with tempfile.TemporaryDirectory() as tmpdirname:
                saved_file = first.save_pretrained(tmpdirname)[0]
                check_json_file_has_correct_format(saved_file)
                second = self._load_with_auto(tmpdirname, backend_name)
            self.assertEqual(second.to_dict(), first.to_dict())

    # ── Init and attribute resolution ─────────────────────────────────────

    def test_init_without_params(self):
        for processing_class in self.processing_classes.values():
            self.assertIsNotNone(processing_class())

    def _find_non_none_default_attr(self, processing_class):
        """An attribute with a non-None class default -- the only kind that can tell
        "explicitly None" apart from "unset"."""
        return next(
            (
                attr
                for attr in ["do_resize", "do_rescale", "do_normalize"]
                if getattr(processing_class, attr, None) is not None
            ),
            None,
        )

    def test_explicit_none_preserved(self):
        """An explicit `None` at init is a stored config value and survives save/load.

        This is a *serialization* contract, deliberately separate from what `__call__` does with a
        `None` (see `test_explicit_none_kwarg_falls_back_to_default`). `_UNSET` in
        `_init_kwargs_from_valid_kwargs` is what distinguishes an omitted kwarg from `None`.
        """
        for processing_class in self.processing_classes.values():
            test_attr = self._find_non_none_default_attr(processing_class)
            if test_attr is None:
                continue
            kwargs = {**self.processor_dict, test_attr: None}
            processor = processing_class(**kwargs)

            self.assertIn(test_attr, processor.to_dict())
            self.assertIsNone(processor.to_dict()[test_attr])

            with tempfile.TemporaryDirectory() as tmpdirname:
                processor.save_pretrained(tmpdirname)
                reloaded = processing_class.from_pretrained(tmpdirname)
            self.assertIsNone(getattr(reloaded, test_attr), f"explicit None for `{test_attr}` was lost after reload")

    def test_override_instance_attributes_does_not_affect_other_instances(self):
        """Mutable class defaults are deep-copied per instance (`_init_kwargs_from_valid_kwargs`)."""
        for backend_name, processing_class in self.processing_classes.items():
            with self.subTest(backend=backend_name):
                first, second = processing_class(), processing_class()
                if not isinstance(getattr(first, "size", None), dict) or not isinstance(
                    getattr(first, "image_mean", None), list
                ):
                    continue

                original_size = deepcopy(second.size)
                for key in first.size:
                    first.size[key] = -1
                modified_size = deepcopy(first.size)

                original_mean = deepcopy(second.image_mean)
                first.image_mean[0] = -1
                modified_mean = deepcopy(first.image_mean)

                self.assertEqual(second.size, original_size)
                self.assertEqual(second.image_mean, original_mean)

                for key in second.size:
                    second.size[key] = -2
                second.image_mean[0] = -2

                self.assertEqual(first.size, modified_size)
                self.assertEqual(first.image_mean, modified_mean)

    # ── Cross-backend config compatibility ────────────────────────────────

    def _assert_cross_backend_configs_match(self, first, second, name_a, name_b):
        dict_a, dict_b = first.to_dict(), second.to_dict()
        # `_type_key` names the concrete class (`WhisperAudioProcessor` vs
        # `WhisperAudioProcessorNumpy`), so it differs between sibling backends by construction.
        for d in (dict_a, dict_b):
            d.pop(getattr(first, "_type_key", None), None)
        difference = {key: dict_a.get(key, dict_b.get(key)) for key in set(dict_a) ^ set(dict_b)}
        self.assertTrue(
            all(value is None for key, value in difference.items() if key not in self.backend_specific_keys),
            f"backends {name_a} and {name_b} differ in unexpected keys: {difference}",
        )
        common = set(dict_a) & set(dict_b)
        self.assertEqual(
            {k: dict_a[k] for k in common},
            {k: dict_b[k] for k in common},
            f"backends {name_a} and {name_b} differ in common keys",
        )

    def test_save_load_backends(self):
        """A config saved by one backend loads in every sibling backend."""
        if len(self.processing_classes) < 2:
            self.skipTest("fewer than 2 backends")
        for name_a, class_a in self.processing_classes.items():
            first = class_a(**self.processor_dict)
            for name_b, class_b in self.processing_classes.items():
                if name_a == name_b:
                    continue
                with tempfile.TemporaryDirectory() as tmpdirname:
                    first.save_pretrained(tmpdirname)
                    second = class_b.from_pretrained(tmpdirname)
                self._assert_cross_backend_configs_match(first, second, name_a, name_b)

    def test_save_load_backends_auto(self):
        """Same, routed through the modality's Auto class."""
        if len(self.processing_classes) < 2:
            self.skipTest("fewer than 2 backends")
        if self.auto_class is None:
            self.skipTest("no `auto_class` declared")
        for name_a, class_a in self.processing_classes.items():
            first = class_a(**self.processor_dict)
            for name_b in self.processing_classes:
                if name_a == name_b:
                    continue
                with tempfile.TemporaryDirectory() as tmpdirname:
                    first.save_pretrained(tmpdirname)
                    second = self._load_with_auto(tmpdirname, name_b)
                self._assert_cross_backend_configs_match(first, second, name_a, name_b)

    # ── Call-time kwarg resolution ────────────────────────────────────────

    @require_torch
    def test_explicit_none_kwarg_falls_back_to_default(self):
        """At call time `None` means "unset" and resolves to the processor's configured value.

        `PreprocessingMixin.preprocess` used to fill defaults with `kwargs.setdefault(...)`, which
        is a no-op when the key is present -- so `processor(x, do_rescale=None)` passed `None`
        through and every `if do_x:` read site downstream evaluated it as `False`, silently
        skipping a step the caller never asked to disable.
        """
        try:
            inputs = self._prepare_inputs()
        except NotImplementedError:
            self.skipTest(f"{type(self).__name__} does not implement `_prepare_inputs`")

        for backend_name, processing_class in self.processing_classes.items():
            with self.subTest(backend=backend_name):
                processor = processing_class(**self.processor_dict)
                # A reference we cannot even produce means the generic fixture does not suit this
                # processor (VitPose needs `boxes`, VitMatte `trimaps`, others a specific channel
                # layout) -- not that the contract is broken. Failures in the `None` call below
                # still surface, since by then the same call has already worked once.
                try:
                    reference = processor(inputs, return_tensors="pt")
                except Exception as exc:
                    self.skipTest(f"generic fixture unsuitable for {processing_class.__name__}: {exc}")

                # Only the universal switches: enabling something like `do_sample_frames`
                # additionally requires companion arguments (`video_metadata`), so a `None` there
                # cannot be resolved and re-run in isolation.
                names = getattr(processor, "_call_kwargs_names", processor._valid_kwargs_names)
                candidates = [
                    name
                    for name in names
                    if name not in self.none_resolution_skip_kwargs
                    and isinstance(getattr(processor, name, None), bool)
                    and getattr(processor, name)
                ]
                if not candidates:
                    self.skipTest("no truthy boolean per-call kwarg to exercise")

                for name in candidates:
                    with self.subTest(kwarg=name):
                        encoding = processor(inputs, return_tensors="pt", **{name: None})
                        self.assertEqual(set(encoding.keys()), set(reference.keys()))
                        for out_key, expected in reference.items():
                            self._assert_output_unchanged(encoding[out_key], expected, name, out_key)

    def _assert_output_unchanged(self, actual, expected, kwarg_name, out_key):
        """Compare one output value exactly. Values may be tensors, ragged lists of tensors
        (uvdoc), or plain metadata, so recurse rather than assume a single shape."""
        if isinstance(expected, torch.Tensor):
            torch.testing.assert_close(
                actual, expected, atol=0, rtol=0, msg=f"`{kwarg_name}=None` changed `{out_key}`"
            )
        elif isinstance(expected, (list, tuple)):
            self.assertEqual(len(actual), len(expected), f"`{kwarg_name}=None` changed `{out_key}` length")
            for got, want in zip(actual, expected):
                self._assert_output_unchanged(got, want, kwarg_name, out_key)
        else:
            self.assertEqual(actual, expected, f"`{kwarg_name}=None` changed `{out_key}`")

    # ── BatchFeature casting ──────────────────────────────────────────────

    @require_torch
    def test_cast_dtype_device(self):
        """`BatchFeature.to()` casts arrays and leaves integer keys alone (`preprocessing_base.py`)."""
        if self.test_cast_dtype is None:
            self.skipTest("`test_cast_dtype` not set")
        try:
            inputs = self._prepare_inputs()
        except NotImplementedError:
            self.skipTest(f"{type(self).__name__} does not implement `_prepare_inputs`")

        for processing_class in self.processing_classes.values():
            processor = processing_class(**self.processor_dict)
            key = self._main_output_key(processor)

            encoding = processor(inputs, return_tensors="pt")
            self.assertEqual(encoding[key].device, torch.device("cpu"))
            self.assertEqual(encoding[key].dtype, torch.float32)

            encoding = processor(inputs, return_tensors="pt").to(torch.float16)
            self.assertEqual(encoding[key].device, torch.device("cpu"))
            self.assertEqual(encoding[key].dtype, torch.float16)

            encoding = processor(inputs, return_tensors="pt").to("cpu", torch.bfloat16)
            self.assertEqual(encoding[key].device, torch.device("cpu"))
            self.assertEqual(encoding[key].dtype, torch.bfloat16)

            with self.assertRaises(TypeError):
                _ = processor(inputs, return_tensors="pt").to(torch.bfloat16, "cpu")

            # A text key alongside the arrays must keep its integer dtype.
            encoding = processor(inputs, return_tensors="pt")
            encoding.update({"input_ids": torch.LongTensor([[1, 2, 3], [4, 5, 6]])})
            encoding = encoding.to(torch.float16)
            self.assertEqual(encoding[key].device, torch.device("cpu"))
            self.assertEqual(encoding[key].dtype, torch.float16)
            self.assertEqual(encoding.input_ids.dtype, torch.long)
