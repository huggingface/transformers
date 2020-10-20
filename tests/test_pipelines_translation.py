import unittest

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
