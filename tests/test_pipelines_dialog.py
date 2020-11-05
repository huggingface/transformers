import unittest

from transformers.pipelines import Conversation, Pipeline

from .test_pipelines_common import CustomInputPipelineCommonMixin


class DialoguePipelineTests(CustomInputPipelineCommonMixin, unittest.TestCase):
    pipeline_task = "conversational"
    small_models = []  # Default model - Models tested without the @slow decorator
    large_models = ["microsoft/DialoGPT-medium"]  # Models tested with the @slow decorator

    def _test_pipeline(self, nlp: Pipeline):
        valid_inputs = [Conversation("Hi there!"), [Conversation("Hi there!"), Conversation("How are you?")]]
        invalid_inputs = ["Hi there!", Conversation()]
        self.assertIsNotNone(nlp)

        mono_result = nlp(valid_inputs[0])
        self.assertIsInstance(mono_result, Conversation)

        multi_result = nlp(valid_inputs[1])
        self.assertIsInstance(multi_result, list)
        self.assertIsInstance(multi_result[0], Conversation)
        # Inactive conversations passed to the pipeline raise a ValueError
        self.assertRaises(ValueError, nlp, valid_inputs[1])

        for bad_input in invalid_inputs:
            self.assertRaises(Exception, nlp, bad_input)
        self.assertRaises(Exception, nlp, invalid_inputs)
