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

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, Conversation, ConversationalPipeline, pipeline
from transformers.testing_utils import require_torch, slow, torch_device

from .test_pipelines_common import MonoInputPipelineCommonMixin


DEFAULT_DEVICE_NUM = -1 if torch_device == "cpu" else 0


class ConversationalPipelineTests(MonoInputPipelineCommonMixin, unittest.TestCase):
    pipeline_task = "conversational"
    small_models = []  # Models tested without the @slow decorator
    large_models = ["microsoft/DialoGPT-medium"]  # Models tested with the @slow decorator
    invalid_inputs = ["Hi there!", Conversation()]

    def _test_pipeline(
        self, nlp
    ):  # override the default test method to check that the output is a `Conversation` object
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

    @require_torch
    @slow
    def test_integration_torch_conversation_encoder_decoder(self):
        # When
        tokenizer = AutoTokenizer.from_pretrained("facebook/blenderbot-90M")
        model = AutoModelForSeq2SeqLM.from_pretrained("facebook/blenderbot-90M")
        nlp = ConversationalPipeline(model=model, tokenizer=tokenizer, device=DEFAULT_DEVICE_NUM)

        conversation_1 = Conversation("My name is Sarah and I live in London")
        conversation_2 = Conversation("Going to the movies tonight, What movie would you recommend? ")
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
        self.assertEqual(result[0].past_user_inputs[0], "My name is Sarah and I live in London")
        self.assertEqual(
            result[0].generated_responses[0],
            "hi sarah, i live in london as well. do you have any plans for the weekend?",
        )
        self.assertEqual(
            result[1].past_user_inputs[0], "Going to the movies tonight, What movie would you recommend? "
        )
        self.assertEqual(
            result[1].generated_responses[0], "i don't know... i'm not really sure. what movie are you going to see?"
        )
        # When
        conversation_1.add_user_input("Not yet, what about you?")
        conversation_2.add_user_input("What's your name?")
        result = nlp([conversation_1, conversation_2], do_sample=False, max_length=1000)
        # Then
        self.assertEqual(result, [conversation_1, conversation_2])
        self.assertEqual(len(result[0].past_user_inputs), 2)
        self.assertEqual(len(result[1].past_user_inputs), 2)
        self.assertEqual(len(result[0].generated_responses), 2)
        self.assertEqual(len(result[1].generated_responses), 2)
        self.assertEqual(result[0].past_user_inputs[1], "Not yet, what about you?")
        self.assertEqual(result[0].generated_responses[1], "i don't have any plans yet. i'm not sure what to do yet.")
        self.assertEqual(result[1].past_user_inputs[1], "What's your name?")
        self.assertEqual(result[1].generated_responses[1], "i don't have a name, but i'm going to see a horror movie.")
