import unittest

from .test_pipelines_common import MonoInputPipelineCommonMixin


class SentimentAnalysisPipelineTests(MonoInputPipelineCommonMixin, unittest.TestCase):
    pipeline_task = "sentiment-analysis"
    small_models = [
        "sshleifer/tiny-distilbert-base-uncased-finetuned-sst-2-english"
    ]  # Default model - Models tested without the @slow decorator
    large_models = [None]  # Models tested with the @slow decorator
    mandatory_keys = {"label", "score"}  # Keys which should be in the output
