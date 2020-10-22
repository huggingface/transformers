import unittest

from transformers.pipelines import Pipeline

from .test_pipelines_common import CustomInputPipelineCommonMixin


class QAPipelineTests(CustomInputPipelineCommonMixin, unittest.TestCase):
    pipeline_task = "question-answering"
    small_models = [
        "sshleifer/tiny-distilbert-base-cased-distilled-squad"
    ]  # Models tested without the @slow decorator
    large_models = []  # Models tested with the @slow decorator

    def _test_pipeline(self, nlp: Pipeline):
        output_keys = {"score", "answer", "start", "end"}
        valid_inputs = [
            {"question": "Where was HuggingFace founded ?", "context": "HuggingFace was founded in Paris."},
            {
                "question": "In what field is HuggingFace working ?",
                "context": "HuggingFace is a startup based in New-York founded in Paris which is trying to solve NLP.",
            },
        ]
        invalid_inputs = [
            {"question": "", "context": "This is a test to try empty question edge case"},
            {"question": None, "context": "This is a test to try empty question edge case"},
            {"question": "What is does with empty context ?", "context": ""},
            {"question": "What is does with empty context ?", "context": None},
        ]
        self.assertIsNotNone(nlp)

        mono_result = nlp(valid_inputs[0])
        self.assertIsInstance(mono_result, dict)

        for key in output_keys:
            self.assertIn(key, mono_result)

        multi_result = nlp(valid_inputs)
        self.assertIsInstance(multi_result, list)
        self.assertIsInstance(multi_result[0], dict)

        for result in multi_result:
            for key in output_keys:
                self.assertIn(key, result)
        for bad_input in invalid_inputs:
            self.assertRaises(Exception, nlp, bad_input)
        self.assertRaises(Exception, nlp, invalid_inputs)
