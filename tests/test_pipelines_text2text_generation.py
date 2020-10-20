import unittest

from .test_pipelines_common import MonoInputPipelineCommonMixin


class Text2TextGenerationPipelineTests(MonoInputPipelineCommonMixin, unittest.TestCase):
    pipeline_task = "text2text-generation"
    small_models = ["patrickvonplaten/t5-tiny-random"]  # Default model - Models tested without the @slow decorator
    large_models = []  # Models tested with the @slow decorator
    invalid_inputs = [4, "<mask>"]
    mandatory_keys = ["generated_text"]
