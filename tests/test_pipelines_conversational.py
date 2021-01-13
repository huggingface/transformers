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

from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    Conversation,
    ConversationalPipeline,
    is_torch_available,
    pipeline,
)
from transformers.testing_utils import require_torch, slow, torch_device

from .test_pipelines_common import MonoInputPipelineCommonMixin


if is_torch_available():
    import torch

    from transformers.models.gpt2 import GPT2Config, GPT2LMHeadModel

DEFAULT_DEVICE_NUM = -1 if torch_device == "cpu" else 0


class SimpleConversationPipelineTests(unittest.TestCase):
    def get_pipeline(self):
        # When
        config = GPT2Config(
            vocab_size=263,
            n_ctx=128,
            max_length=128,
            n_embd=64,
            n_layer=1,
            n_head=8,
            bos_token_id=256,
            eos_token_id=257,
        )
        model = GPT2LMHeadModel(config)
        # Force model output to be L
        V, D = model.lm_head.weight.shape
        bias = torch.zeros(V, requires_grad=True)
        bias[76] = 1

        model.lm_head.bias = torch.nn.Parameter(bias)

        # # Created with:
        # import tempfile

        # from tokenizers import Tokenizer, models
        # from transformers.tokenization_utils_fast import PreTrainedTokenizerFast

        # vocab = [(chr(i), i) for i in range(256)]
        # tokenizer = Tokenizer(models.Unigram(vocab))
        # with tempfile.NamedTemporaryFile() as f:
        #     tokenizer.save(f.name)
        #     real_tokenizer = PreTrainedTokenizerFast(tokenizer_file=f.name, eos_token="<eos>", bos_token="<bos>")

        # real_tokenizer._tokenizer.save("dummy.json")
        # Special tokens are automatically added at load time.
        tokenizer = AutoTokenizer.from_pretrained("Narsil/small_conversational_test")
        conversation_agent = pipeline(
            task="conversational", device=DEFAULT_DEVICE_NUM, model=model, tokenizer=tokenizer
        )
        return conversation_agent

    @require_torch
    def test_integration_torch_conversation(self):
        conversation_agent = self.get_pipeline()
        conversation_1 = Conversation("Going to the movies tonight - any suggestions?")
        conversation_2 = Conversation("What's the last book you have read?")
        self.assertEqual(len(conversation_1.past_user_inputs), 0)
        self.assertEqual(len(conversation_2.past_user_inputs), 0)

        with self.assertLogs("transformers", level="WARNING") as log:
            result = conversation_agent([conversation_1, conversation_2], max_length=48)
            self.assertEqual(len(log.output), 2)
            self.assertIn("You might consider trimming the early phase of the conversation", log.output[0])
            self.assertIn("Setting `pad_token_id`", log.output[1])

        # Two conversations in one pass
        self.assertEqual(result, [conversation_1, conversation_2])
        self.assertEqual(
            result,
            [
                Conversation(
                    None,
                    past_user_inputs=["Going to the movies tonight - any suggestions?"],
                    generated_responses=["L"],
                ),
                Conversation(
                    None, past_user_inputs=["What's the last book you have read?"], generated_responses=["L"]
                ),
            ],
        )

        # One conversation with history
        conversation_2.add_user_input("Why do you recommend it?")
        with self.assertLogs("transformers", level="WARNING") as log:
            result = conversation_agent(conversation_2, max_length=64)
            self.assertEqual(len(log.output), 3)
            self.assertIn("Cutting history off because it's too long", log.output[0])
            self.assertIn("You might consider trimming the early phase of the conversation", log.output[1])
            self.assertIn("Setting `pad_token_id`", log.output[2])

        self.assertEqual(result, conversation_2)
        self.assertEqual(
            result,
            Conversation(
                None,
                past_user_inputs=["What's the last book you have read?", "Why do you recommend it?"],
                generated_responses=["L", "L"],
            ),
        )

    @require_torch
    def test_history_cache(self):
        conversation_agent = self.get_pipeline()
        conversation = Conversation(
            "Why do you recommend it?",
            past_user_inputs=["What's the last book you have read?"],
            generated_responses=["b"],
        )
        with self.assertLogs("transformers", level="WARNING") as log:
            _ = conversation_agent(conversation, max_length=64)
            self.assertEqual(len(log.output), 3)
            self.assertIn("Cutting history off because it's too long (63 > 32) for underlying model", log.output[0])
            self.assertIn("63 is bigger than 0.9 * max_length: 64", log.output[1])
            self.assertIn("Setting `pad_token_id`", log.output[2])
        self.assertEqual(conversation._index, 1)
        self.assertEqual(
            conversation._history,
            [
                87,
                104,
                97,
                116,
                39,
                115,
                32,
                116,
                104,
                101,
                32,
                108,
                97,
                115,
                116,
                32,
                98,
                111,
                111,
                107,
                32,
                121,
                111,
                117,
                32,
                104,
                97,
                118,
                101,
                32,
                114,
                101,
                97,
                100,
                63,
                259,  # EOS
                98,  # b
                259,  # EOS
            ],
        )


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
        tokenizer = AutoTokenizer.from_pretrained("facebook/blenderbot_small-90M")
        model = AutoModelForSeq2SeqLM.from_pretrained("facebook/blenderbot_small-90M")
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
