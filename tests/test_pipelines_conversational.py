import unittest

from transformers import Conversation, pipeline
from transformers.testing_utils import require_torch, slow, torch_device

from .test_pipelines_common import MonoInputPipelineCommonMixin


DEFAULT_DEVICE_NUM = -1 if torch_device == "cpu" else 0


class ConversationalPipelineTests(MonoInputPipelineCommonMixin, unittest.TestCase):
    pipeline_task = "conversational"
    small_models = []  # Models tested without the @slow decorator
    large_models = ["microsoft/DialoGPT-medium"]  # Models tested with the @slow decorator
    invalid_inputs = ["Hi there!", Conversation()]

    def _test_pipeline(self, nlp):
        # e overide the default test method to check that the output is a `Conversation` object
        self.assertIsNotNone(nlp)

        # We need to recreate conversation for successive tests to pass as
        # Conversation objects get *consumed* by the pipeline
        conversation = Conversation("Hi there!")
        mono_result = nlp(conversation)
        self.assertIsInstance(mono_result, Conversation)

        conversations = [Conversation("Hi there!"), Conversation("How are you?")]
        multi_result = nlp(conversations)
        self.assertIsInstance(multi_result, list)
        self.assertIsInstance(multi_result[0], Conversation)
        # Conversation have been consumed and are not valid anymore
        # Inactive conversations passed to the pipeline raise a ValueError
        self.assertRaises(ValueError, nlp, conversation)
        self.assertRaises(ValueError, nlp, conversations)

        for bad_input in self.invalid_inputs:
            self.assertRaises(Exception, nlp, bad_input)
        self.assertRaises(Exception, nlp, self.invalid_inputs)

    @require_torch
    @slow
    def test_integration_torch_conversation(self):
        # When
        nlp = pipeline(task="conversational", device=DEFAULT_DEVICE_NUM)
        conversation_1 = Conversation("Going to the movies tonight - any suggestions?")
        conversation_2 = Conversation("What's the last book you have read?")
        # Then
        self.assertEqual(len(conversation_1.past_user_inputs), 0)
        self.assertEqual(len(conversation_2.past_user_inputs), 0)
        # When
        result = nlp([conversation_1, conversation_2], do_sample=False, max_length=1000)
        # Then
        self.assertEqual(result, [conversation_1, conversation_2])
        self.assertEqual(len(result[0].past_user_inputs), 1)
        self.assertEqual(len(result[1].past_user_inputs), 1)
        self.assertEqual(len(result[0].generated_responses), 1)
        self.assertEqual(len(result[1].generated_responses), 1)
        self.assertEqual(result[0].past_user_inputs[0], "Going to the movies tonight - any suggestions?")
        self.assertEqual(result[0].generated_responses[0], "The Big Lebowski")
        self.assertEqual(result[1].past_user_inputs[0], "What's the last book you have read?")
        self.assertEqual(result[1].generated_responses[0], "The Last Question")
        # When
        conversation_2.add_user_input("Why do you recommend it?")
        result = nlp(conversation_2, do_sample=False, max_length=1000)
        # Then
        self.assertEqual(result, conversation_2)
        self.assertEqual(len(result.past_user_inputs), 2)
        self.assertEqual(len(result.generated_responses), 2)
        self.assertEqual(result.past_user_inputs[1], "Why do you recommend it?")
        self.assertEqual(result.generated_responses[1], "It's a good book.")

    @require_torch
    @slow
    def test_integration_torch_conversation_truncated_history(self):
        # When
        nlp = pipeline(task="conversational", min_length_for_response=24, device=DEFAULT_DEVICE_NUM)
        conversation_1 = Conversation("Going to the movies tonight - any suggestions?")
        # Then
        self.assertEqual(len(conversation_1.past_user_inputs), 0)
        # When
        result = nlp(conversation_1, do_sample=False, max_length=36)
        # Then
        self.assertEqual(result, conversation_1)
        self.assertEqual(len(result.past_user_inputs), 1)
        self.assertEqual(len(result.generated_responses), 1)
        self.assertEqual(result.past_user_inputs[0], "Going to the movies tonight - any suggestions?")
        self.assertEqual(result.generated_responses[0], "The Big Lebowski")
        # When
        conversation_1.add_user_input("Is it an action movie?")
        result = nlp(conversation_1, do_sample=False, max_length=36)
        # Then
        self.assertEqual(result, conversation_1)
        self.assertEqual(len(result.past_user_inputs), 2)
        self.assertEqual(len(result.generated_responses), 2)
        self.assertEqual(result.past_user_inputs[1], "Is it an action movie?")
        self.assertEqual(result.generated_responses[1], "It's a comedy.")
