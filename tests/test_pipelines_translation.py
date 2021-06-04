# Copyright 2020 The HuggingFace Team. All rights reserved.
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

import pytest

from transformers import pipeline
from transformers.testing_utils import is_pipeline_test, is_torch_available, require_torch, slow

from .test_pipelines_common import MonoInputPipelineCommonMixin


if is_torch_available():
    from transformers.models.mbart import MBart50TokenizerFast, MBartForConditionalGeneration


class TranslationEnToDePipelineTests(MonoInputPipelineCommonMixin, unittest.TestCase):
    pipeline_task = "translation_en_to_de"
    small_models = ["patrickvonplaten/t5-tiny-random"]  # Default model - Models tested without the @slow decorator
    large_models = [None]  # Models tested with the @slow decorator
    invalid_inputs = [4, "<mask>"]
    mandatory_keys = ["translation_text"]


class TranslationEnToRoPipelineTests(MonoInputPipelineCommonMixin, unittest.TestCase):
    pipeline_task = "translation_en_to_ro"
    small_models = ["patrickvonplaten/t5-tiny-random"]  # Default model - Models tested without the @slow decorator
    large_models = [None]  # Models tested with the @slow decorator
    invalid_inputs = [4, "<mask>"]
    mandatory_keys = ["translation_text"]


@is_pipeline_test
class TranslationNewFormatPipelineTests(unittest.TestCase):
    @require_torch
    @slow
    def test_default_translations(self):
        # We don't provide a default for this pair
        with self.assertRaises(ValueError):
            pipeline(task="translation_cn_to_ar")

        # but we do for this one
        translator = pipeline(task="translation_en_to_de")
        self.assertEquals(translator.src_lang, "en")
        self.assertEquals(translator.tgt_lang, "de")

    @require_torch
    @slow
    def test_multilingual_translation(self):
        model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
        tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")

        translator = pipeline(task="translation", model=model, tokenizer=tokenizer)
        # Missing src_lang, tgt_lang
        with self.assertRaises(ValueError):
            translator("This is a test")

        outputs = translator("This is a test", src_lang="en_XX", tgt_lang="ar_AR")
        self.assertEqual(outputs, [{"translation_text": "هذا إختبار"}])

        outputs = translator("This is a test", src_lang="en_XX", tgt_lang="hi_IN")
        self.assertEqual(outputs, [{"translation_text": "यह एक परीक्षण है"}])

        # src_lang, tgt_lang can be defined at pipeline call time
        translator = pipeline(task="translation", model=model, tokenizer=tokenizer, src_lang="en_XX", tgt_lang="ar_AR")
        outputs = translator("This is a test")
        self.assertEqual(outputs, [{"translation_text": "هذا إختبار"}])

    @require_torch
    def test_translation_on_odd_language(self):
        model = "patrickvonplaten/t5-tiny-random"
        translator = pipeline(task="translation_cn_to_ar", model=model)
        self.assertEquals(translator.src_lang, "cn")
        self.assertEquals(translator.tgt_lang, "ar")

    @require_torch
    def test_translation_default_language_selection(self):
        model = "patrickvonplaten/t5-tiny-random"
        with pytest.warns(UserWarning, match=r".*translation_en_to_de.*"):
            translator = pipeline(task="translation", model=model)
        self.assertEqual(translator.task, "translation_en_to_de")
        self.assertEquals(translator.src_lang, "en")
        self.assertEquals(translator.tgt_lang, "de")

    @require_torch
    def test_translation_with_no_language_no_model_fails(self):
        with self.assertRaises(ValueError):
            pipeline(task="translation")
