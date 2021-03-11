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
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    Conversation,
    ConversationalPipeline,
    is_torch_available,
    pipeline,
)
from transformers.testing_utils import is_pipeline_test, require_torch, slow, torch_device

from .test_pipelines_common import MonoInputPipelineCommonMixin


if is_torch_available():
    import torch

    from transformers.models.gpt2 import GPT2Config, GPT2LMHeadModel

DEFAULT_DEVICE_NUM = -1 if torch_device == "cpu" else 0


@is_pipeline_test
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
        bias = torch.zeros(V)
        bias[76] = 1
        weight = torch.zeros((V, D), requires_grad=True)

        model.lm_head.bias = torch.nn.Parameter(bias)
        model.lm_head.weight = torch.nn.Parameter(weight)

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

        result = conversation_agent([conversation_1, conversation_2], max_length=48)

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
        result = conversation_agent(conversation_2, max_length=64)

        self.assertEqual(result, conversation_2)
        self.assertEqual(
            result,
            Conversation(
                None,
                past_user_inputs=["What's the last book you have read?", "Why do you recommend it?"],
                generated_responses=["L", "L"],
            ),
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
    def test_integration_torch_conversation_dialogpt_input_ids(self):
        tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-small")
        model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-small")
        nlp = ConversationalPipeline(model=model, tokenizer=tokenizer)

        conversation_1 = Conversation("hello")
        inputs = nlp._parse_and_tokenize([conversation_1])
        self.assertEqual(inputs["input_ids"].tolist(), [[31373, 50256]])

        conversation_2 = Conversation("how are you ?", past_user_inputs=["hello"], generated_responses=["Hi there!"])
        inputs = nlp._parse_and_tokenize([conversation_2])
        self.assertEqual(
            inputs["input_ids"].tolist(), [[31373, 50256, 17250, 612, 0, 50256, 4919, 389, 345, 5633, 50256]]
        )

        inputs = nlp._parse_and_tokenize([conversation_1, conversation_2])
        self.assertEqual(
            inputs["input_ids"].tolist(),
            [
                [31373, 50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256],
                [31373, 50256, 17250, 612, 0, 50256, 4919, 389, 345, 5633, 50256],
            ],
        )

    @require_torch
    @slow
    def test_integration_torch_conversation_blenderbot_400M_input_ids(self):
        tokenizer = AutoTokenizer.from_pretrained("facebook/blenderbot-400M-distill")
        model = AutoModelForSeq2SeqLM.from_pretrained("facebook/blenderbot-400M-distill")
        nlp = ConversationalPipeline(model=model, tokenizer=tokenizer)

        # test1
        conversation_1 = Conversation("hello")
        inputs = nlp._parse_and_tokenize([conversation_1])
        self.assertEqual(inputs["input_ids"].tolist(), [[1710, 86, 2]])

        # test2
        conversation_1 = Conversation(
            "I like lasagne.",
            past_user_inputs=["hello"],
            generated_responses=[
                " Do you like lasagne? It is a traditional Italian dish consisting of a shepherd's pie."
            ],
        )
        inputs = nlp._parse_and_tokenize([conversation_1])
        self.assertEqual(
            inputs["input_ids"].tolist(),
            [
                # This should be compared with the same conversation on ParlAI `safe_interactive` demo.
                [
                    1710,  # hello
                    86,
                    228,  # Double space
                    228,
                    946,
                    304,
                    398,
                    6881,
                    558,
                    964,
                    38,
                    452,
                    315,
                    265,
                    6252,
                    452,
                    322,
                    968,
                    6884,
                    3146,
                    278,
                    306,
                    265,
                    617,
                    87,
                    388,
                    75,
                    341,
                    286,
                    521,
                    21,
                    228,  # Double space
                    228,
                    281,  # I like lasagne.
                    398,
                    6881,
                    558,
                    964,
                    21,
                    2,  # EOS
                ]
            ],
        )

    @require_torch
    @slow
    def test_integration_torch_conversation_blenderbot_400M(self):
        tokenizer = AutoTokenizer.from_pretrained("facebook/blenderbot-400M-distill")
        model = AutoModelForSeq2SeqLM.from_pretrained("facebook/blenderbot-400M-distill")
        nlp = ConversationalPipeline(model=model, tokenizer=tokenizer)

        conversation_1 = Conversation("hello")
        result = nlp(
            conversation_1,
        )
        self.assertEqual(
            result.generated_responses[0],
            # ParlAI implementation output, we have a different one, but it's our
            # second best, you can check by using num_return_sequences=10
            # " Hello! How are you? I'm just getting ready to go to work, how about you?",
            " Hello! How are you doing today? I just got back from a walk with my dog.",
        )

        conversation_1 = Conversation("Lasagne   hello")
        result = nlp(conversation_1, encoder_no_repeat_ngram_size=3)
        self.assertEqual(
            result.generated_responses[0],
            " Do you like lasagne? It is a traditional Italian dish consisting of a shepherd's pie.",
        )

        conversation_1 = Conversation(
            "Lasagne   hello   Lasagne is my favorite Italian dish. Do you like lasagne?   I like lasagne."
        )
        result = nlp(
            conversation_1,
            encoder_no_repeat_ngram_size=3,
        )
        self.assertEqual(
            result.generated_responses[0],
            " Me too. I like how it can be topped with vegetables, meats, and condiments.",
        )

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
