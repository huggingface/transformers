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

from transformers import (
    MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING,
    TF_MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING,
    MBart50TokenizerFast,
    MBartConfig,
    MBartForConditionalGeneration,
    TranslationPipeline,
    pipeline,
)
from transformers.testing_utils import is_pipeline_test, require_tf, require_torch, slow

from .test_pipelines_common import ANY


@is_pipeline_test
class TranslationPipelineTests(unittest.TestCase):
    model_mapping = MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING
    tf_model_mapping = TF_MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING

    def get_test_pipeline(self, model, tokenizer, processor):
        if isinstance(model.config, MBartConfig):
            src_lang, tgt_lang = list(tokenizer.lang_code_to_id.keys())[:2]
            translator = TranslationPipeline(model=model, tokenizer=tokenizer, src_lang=src_lang, tgt_lang=tgt_lang)
        else:
            translator = TranslationPipeline(model=model, tokenizer=tokenizer)
        return translator, ["Some string", "Some other text"]

    def run_pipeline_test(self, translator, _):
        outputs = translator("Some string")
        self.assertEqual(outputs, [{"translation_text": ANY(str)}])

        outputs = translator(["Some string"])
        self.assertEqual(outputs, [{"translation_text": ANY(str)}])

        outputs = translator(["Some string", "other string"])
        self.assertEqual(outputs, [{"translation_text": ANY(str)}, {"translation_text": ANY(str)}])

    @require_torch
    def test_small_model_pt(self):
        translator = pipeline("translation_en_to_ro", model="patrickvonplaten/t5-tiny-random", framework="pt")
        outputs = translator("This is a test string", max_length=20)
        self.assertEqual(
            outputs,
            [
                {
                    "translation_text": (
                        "Beide Beide Beide Beide Beide Beide Beide Beide Beide Beide Beide Beide Beide Beide Beide"
                        " Beide Beide"
                    )
                }
            ],
        )

    @require_tf
    def test_small_model_tf(self):
        translator = pipeline("translation_en_to_ro", model="patrickvonplaten/t5-tiny-random", framework="tf")
        outputs = translator("This is a test string", max_length=20)
        self.assertEqual(
            outputs,
            [
                {
                    "translation_text": (
                        "Beide Beide Beide Beide Beide Beide Beide Beide Beide Beide Beide Beide Beide Beide Beide"
                        " Beide Beide"
                    )
                }
            ],
        )

    @require_torch
    def test_en_to_de_pt(self):
        translator = pipeline("translation_en_to_de", model="patrickvonplaten/t5-tiny-random", framework="pt")
        outputs = translator("This is a test string", max_length=20)
        self.assertEqual(
            outputs,
            [
                {
                    "translation_text": (
                        "monoton monoton monoton monoton monoton monoton monoton monoton monoton monoton urine urine"
                        " urine urine urine urine urine urine urine"
                    )
                }
            ],
        )

    @require_tf
    def test_en_to_de_tf(self):
        translator = pipeline("translation_en_to_de", model="patrickvonplaten/t5-tiny-random", framework="tf")
        outputs = translator("This is a test string", max_length=20)
        self.assertEqual(
            outputs,
            [
                {
                    "translation_text": (
                        "monoton monoton monoton monoton monoton monoton monoton monoton monoton monoton urine urine"
                        " urine urine urine urine urine urine urine"
                    )
                }
            ],
        )


class TranslationNewFormatPipelineTests(unittest.TestCase):
    @require_torch
    @slow
    def test_default_translations(self):
        # We don't provide a default for this pair
        with self.assertRaises(ValueError):
            pipeline(task="translation_cn_to_ar")

        # but we do for this one
        translator = pipeline(task="translation_en_to_de")
        self.assertEqual(translator._preprocess_params["src_lang"], "en")
        self.assertEqual(translator._preprocess_params["tgt_lang"], "de")

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
        self.assertEqual(translator._preprocess_params["src_lang"], "cn")
        self.assertEqual(translator._preprocess_params["tgt_lang"], "ar")

    @require_torch
    def test_translation_default_language_selection(self):
        model = "patrickvonplaten/t5-tiny-random"
        with pytest.warns(UserWarning, match=r".*translation_en_to_de.*"):
            translator = pipeline(task="translation", model=model)
        self.assertEqual(translator.task, "translation_en_to_de")
        self.assertEqual(translator._preprocess_params["src_lang"], "en")
        self.assertEqual(translator._preprocess_params["tgt_lang"], "de")

    @require_torch
    def test_translation_with_no_language_no_model_fails(self):
        with self.assertRaises(ValueError):
            pipeline(task="translation")
