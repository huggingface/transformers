import unittest

import pytest

from transformers import pipeline
from transformers.testing_utils import is_pipeline_test, require_torch, slow

from .test_pipelines_common import MonoInputPipelineCommonMixin


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
        pipeline(task="translation_en_to_de")

    @require_torch
    def test_translation_on_odd_language(self):
        model = "patrickvonplaten/t5-tiny-random"
        pipeline(task="translation_cn_to_ar", model=model)

    @require_torch
    def test_translation_default_language_selection(self):
        model = "patrickvonplaten/t5-tiny-random"
        with pytest.warns(UserWarning, match=r".*translation_en_to_de.*"):
            nlp = pipeline(task="translation", model=model)
        self.assertEqual(nlp.task, "translation_en_to_de")

    @require_torch
    def test_translation_with_no_language_no_model_fails(self):
        with self.assertRaises(ValueError):
            pipeline(task="translation")
