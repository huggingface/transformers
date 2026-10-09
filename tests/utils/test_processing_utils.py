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

import unittest

from transformers import TokenizersBackend
from transformers.processing_utils import ProcessingKwargs, ProcessorMixin
from transformers.testing_utils import CaptureLogger, require_tokenizers
from transformers.utils import logging


class DummyProcessingKwargs(ProcessingKwargs, total=False):
    pass


class DummyProcessor(ProcessorMixin):
    pass


class DefaultsProcessingKwargs(ProcessingKwargs, total=False):
    _defaults = {
        "text_kwargs": {"max_length": 32},
        "common_kwargs": {"return_tensors": "pt", "padding": False},
    }


class DefaultsProcessor(ProcessorMixin):
    valid_processor_kwargs = DummyProcessingKwargs
    text_kwargs = {"max_length": 64}
    images_kwargs = {"do_resize": False}
    return_mm_token_type_ids = True


class ProcessorMixinTest(unittest.TestCase):
    def test_merge_kwargs_propagate_flat_kwargs(self):
        """Flat top level args are propagated"""
        processor = DummyProcessor()
        merged_kwargs = processor._merge_kwargs(max_length=32, do_resize=False, return_mm_token_type_ids=True)
        self.assertEqual(merged_kwargs["text_kwargs"]["max_length"], 32)
        self.assertEqual(merged_kwargs["images_kwargs"]["do_resize"], False)
        self.assertEqual(merged_kwargs["return_mm_token_type_ids"], True)
        # return_mm_token_type_ids must not be propagated as it is in ProcessingKwargs
        self.assertNotIn("return_mm_token_type_ids", merged_kwargs["text_kwargs"])

    def test_merge_kwargs_flat_and_nested(self):
        """Check that flat and nested kwargs do not raise an error if they are defined on
        ProcessingKwargs and the subprocessor kwargs
        """
        processor = DummyProcessor()
        merged_kwargs = processor._merge_kwargs(
            return_mm_token_type_ids=True, text_kwargs={"return_mm_token_type_ids": False}
        )
        self.assertEqual(merged_kwargs["return_mm_token_type_ids"], True)
        self.assertEqual(merged_kwargs["text_kwargs"]["return_mm_token_type_ids"], False)

    def test_merge_kwargs_flat_and_nested_error(self):
        """Check that flat and nested kwargs only defined on the subprocessor raise if passed twice"""
        processor = DummyProcessor()
        with self.assertRaises(ValueError):
            processor._merge_kwargs(max_length=32, text_kwargs={"max_length": 32})

    def test_merge_kwargs_common_kwargs(self):
        """Check that common kwargs are propagated to all modalities"""
        processor = DummyProcessor()
        merged_kwargs = processor._merge_kwargs(common_kwargs={"return_tensors": "pt"})
        self.assertEqual(merged_kwargs["text_kwargs"]["return_tensors"], "pt")
        self.assertEqual(merged_kwargs["images_kwargs"]["return_tensors"], "pt")

        # common_kwargs are not propagated to ProcessorKwargs
        merged_kwargs = processor._merge_kwargs(common_kwargs={"return_mm_token_type_ids": True})
        self.assertNotEqual("return_mm_token_type_ids", True)

        # common_kwargs have lower priority than modality kwargs
        merged_kwargs = processor._merge_kwargs(
            common_kwargs={"return_tensors": "pt"}, text_kwargs={"return_tensors": "np"}
        )
        self.assertEqual(merged_kwargs["text_kwargs"]["return_tensors"], "np")
        self.assertEqual(merged_kwargs["images_kwargs"]["return_tensors"], "pt")

    def test_merge_kwargs_with_typeddict_defaults(self):
        """Check that ProcessorKwargs._defaults are inserted"""
        processor = DummyProcessor()

        merged_kwargs = processor._merge_kwargs(DefaultsProcessingKwargs)
        self.assertEqual(merged_kwargs["text_kwargs"]["max_length"], 32)
        self.assertEqual(merged_kwargs["text_kwargs"]["return_tensors"], "pt")
        self.assertEqual(merged_kwargs["images_kwargs"]["return_tensors"], "pt")
        # common_kwargs are not propagated to ProcessorKwargs
        self.assertNotIn("return_tensors", merged_kwargs)

        # ProcessorKwargs._defaults have lower prio than explicit kwargs
        merged_kwargs = processor._merge_kwargs(
            DefaultsProcessingKwargs, text_kwargs={"max_length": 64}, return_tensors="np"
        )
        self.assertEqual(merged_kwargs["text_kwargs"]["max_length"], 64)
        self.assertEqual(merged_kwargs["text_kwargs"]["return_tensors"], "np")

    @require_tokenizers
    def test_merge_kwargs_with_tokenizer_defaults(self):
        """Check that tokenizer defaults are inserted"""
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel

        processor = DefaultsProcessor()
        processor.tokenizer = TokenizersBackend(
            tokenizer_object=Tokenizer(WordLevel(vocab={"[UNK]": 0}, unk_token="[UNK]")),
            padding_side="left",
            max_length=128,
        )
        processor.tokenizer.padding_side = "right"
        processor.tokenizer.padding = True

        # tokenizer init kwargs have higher prio than processor defaults
        merged_kwargs = processor._merge_kwargs(
            DefaultsProcessingKwargs, tokenizer_init_kwargs=processor.tokenizer.init_kwargs
        )
        self.assertEqual(merged_kwargs["text_kwargs"]["max_length"], 128)

        # tokenizer attributes have higher prio than tokenizer init kwargs
        self.assertEqual(merged_kwargs["text_kwargs"]["padding_side"], "right")

        # ProcessingKwargs._defaults.common_kwargs have higher prio than tokenizer attributes
        self.assertEqual(merged_kwargs["text_kwargs"]["padding"], False)

        # user kwargs have higher prio than tokenizer attributes
        merged_kwargs = processor._merge_kwargs(
            DefaultsProcessingKwargs, tokenizer_init_kwargs=processor.tokenizer.init_kwargs, padding_side="left"
        )
        self.assertEqual(merged_kwargs["text_kwargs"]["padding_side"], "left")

    def test_merge_kwargs_with_default_processor_kwargs(self):
        """Check that processor class attribute defaults are inserted"""
        processor = DefaultsProcessor()

        merged_kwargs = processor._merge_kwargs()
        self.assertEqual(merged_kwargs["text_kwargs"]["max_length"], 64)
        self.assertEqual(merged_kwargs["images_kwargs"]["do_resize"], False)
        self.assertEqual(merged_kwargs["return_mm_token_type_ids"], True)

        # processor class attributes have lower prio than ProcessorKwargs._defaults
        merged_kwargs = processor._merge_kwargs(DefaultsProcessingKwargs)
        self.assertEqual(merged_kwargs["text_kwargs"]["max_length"], 32)
        self.assertEqual(merged_kwargs["text_kwargs"]["return_tensors"], "pt")
        self.assertEqual(merged_kwargs["images_kwargs"]["do_resize"], False)
        self.assertEqual(merged_kwargs["images_kwargs"]["return_tensors"], "pt")
        self.assertEqual(merged_kwargs["return_mm_token_type_ids"], True)

    def test_merge_kwargs_warn_on_unknown_flag_kwargs(self):
        """Checks that passing unknown flat kwargs to the processor logs a warning and doesn't raise an error"""
        processor = DefaultsProcessor()
        logger = logging.get_logger("transformers.processing_utils")
        with CaptureLogger(logger) as caplog:
            processor._merge_kwargs(unknown_kwarg=True)
        self.assertIn("Keyword argument `unknown_kwarg` is not a valid argument", caplog.out)

    def test_merge_kwargs_warn_on_unknown_nested_kwargs(self):
        """Checks that passing unknown nested kwargs to the processor doesn't warn or raise.

        We expect the subprocessor to handle this.
        """
        processor = DefaultsProcessor()
        logger = logging.get_logger("transformers.processing_utils")
        with CaptureLogger(logger) as caplog:
            processor._merge_kwargs(text_kwargs={"unknown_kwarg": True})
        self.assertEqual(caplog.out, "")
