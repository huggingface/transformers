import unittest

from .test_pipelines_common import MonoInputPipelineCommonMixin


class TextGenerationPipelineTests(MonoInputPipelineCommonMixin, unittest.TestCase):
    pipeline_task = "text-generation"
    pipeline_running_kwargs = {"prefix": "This is "}
    small_models = ["sshleifer/tiny-ctrl"]  # Models tested without the @slow decorator
    large_models = []  # Models tested with the @slow decorator
