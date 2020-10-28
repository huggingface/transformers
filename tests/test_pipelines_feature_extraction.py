import unittest

from .test_pipelines_common import MonoInputPipelineCommonMixin


class FeatureExtractionPipelineTests(MonoInputPipelineCommonMixin, unittest.TestCase):
    pipeline_task = "feature-extraction"
    small_models = [
        "sshleifer/tiny-distilbert-base-cased"
    ]  # Default model - Models tested without the @slow decorator
    large_models = [None]  # Models tested with the @slow decorator
    mandatory_keys = {}  # Keys which should be in the output
