# coding=utf-8
# Copyright 2022 The HuggingFace Team. All rights reserved.
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

from transformers import CohereTokenizerFast
from transformers.testing_utils import require_jinja, require_tokenizers, require_torch_multi_gpu

from ...test_tokenization_common import TokenizerTesterMixin


@require_tokenizers
class CohereTokenizationTest(TokenizerTesterMixin, unittest.TestCase):
    slow_tokenizer_class = None
    rust_tokenizer_class = CohereTokenizerFast
    tokenizer_class = CohereTokenizerFast
    test_rust_tokenizer = True
    test_slow_tokenizer = False
    from_pretrained_vocab_key = "tokenizer_file"
    from_pretrained_id = "hf-internal-testing/tiny-random-CohereForCausalLM"
    special_tokens_map = {
        "bos_token": "<BOS_TOKEN>",
        "eos_token": "<|END_OF_TURN_TOKEN|>",
        "unk_token": "<UNK>",
        "pad_token": "<PAD>",
    }

    def setUp(self):
        super().setUp()
        tokenizer = CohereTokenizerFast.from_pretrained("hf-internal-testing/tiny-random-CohereForCausalLM")
        tokenizer.save_pretrained(self.tmpdirname)

    def get_rust_tokenizer(self, **kwargs):
        kwargs.update(self.special_tokens_map)
        return CohereTokenizerFast.from_pretrained(self.tmpdirname, **kwargs)

    # This gives CPU OOM on a single-gpu runner (~60G RAM). On multi-gpu runner, it has ~180G RAM which is enough.
    @require_torch_multi_gpu
    def test_torch_encode_plus_sent_to_model(self):
        super().test_torch_encode_plus_sent_to_model()

    @unittest.skip(reason="This needs a slow tokenizer. Cohere does not have one!")
    def test_encode_decode_with_spaces(self):
        return

    def test_encodings_from_sample_data(self):
        """
        Assert that the created tokens are the same than the hard-coded ones
        """
        tokenizer = self.get_rust_tokenizer()

        INPUT_SENTENCES = ["The quick brown fox<|END_OF_TURN_TOKEN|>", "jumps over the lazy dog<|END_OF_TURN_TOKEN|>"]
        TARGET_TOKENS = [
            [5, 60, 203, 746, 666, 980, 571, 222, 87, 96, 8],
            [5, 82, 332, 88, 91, 544, 206, 257, 930, 97, 239, 435, 8],
        ]

        computed_tokens = tokenizer.batch_encode_plus(INPUT_SENTENCES)["input_ids"]
        self.assertListEqual(TARGET_TOKENS, computed_tokens)

        INPUT_SENTENCES_W_BOS = [
            "<BOS_TOKEN>The quick brown fox<|END_OF_TURN_TOKEN|>",
            "<BOS_TOKEN>jumps over the lazy dog<|END_OF_TURN_TOKEN|>",
        ]
        decoded_tokens = tokenizer.batch_decode(computed_tokens)
        self.assertListEqual(decoded_tokens, INPUT_SENTENCES_W_BOS)

    def test_padding(self, max_length=10):
        for tokenizer, pretrained_name, kwargs in self.tokenizers_list:
            with self.subTest(f"{tokenizer.__class__.__name__} ({pretrained_name})"):
                tokenizer_r = self.rust_tokenizer_class.from_pretrained(pretrained_name, **kwargs)
                # tokenizer_r.pad_token = None # Hotfixing padding = None
                # Simple input
                s = "This is a simple input"
                s2 = ["This is a simple input 1", "This is a simple input 2"]
                p = ("This is a simple input", "This is a pair")
                p2 = [
                    ("This is a simple input 1", "This is a simple input 2"),
                    ("This is a simple pair 1", "This is a simple pair 2"),
                ]

                # Simple input tests
                try:
                    tokenizer_r.encode(s, max_length=max_length)
                    tokenizer_r.encode_plus(s, max_length=max_length)

                    tokenizer_r.batch_encode_plus(s2, max_length=max_length)
                    tokenizer_r.encode(p, max_length=max_length)
                    tokenizer_r.batch_encode_plus(p2, max_length=max_length)
                except ValueError:
                    self.fail("Cohere Tokenizer should be able to deal with padding")

                tokenizer_r.pad_token = None  # Hotfixing padding = None
                self.assertRaises(ValueError, tokenizer_r.encode, s, max_length=max_length, padding="max_length")

                # Simple input
                self.assertRaises(ValueError, tokenizer_r.encode_plus, s, max_length=max_length, padding="max_length")

                # Simple input
                self.assertRaises(
                    ValueError,
                    tokenizer_r.batch_encode_plus,
                    s2,
                    max_length=max_length,
                    padding="max_length",
                )

                # Pair input
                self.assertRaises(ValueError, tokenizer_r.encode, p, max_length=max_length, padding="max_length")

                # Pair input
                self.assertRaises(ValueError, tokenizer_r.encode_plus, p, max_length=max_length, padding="max_length")

                # Pair input
                self.assertRaises(
                    ValueError,
                    tokenizer_r.batch_encode_plus,
                    p2,
                    max_length=max_length,
                    padding="max_length",
                )

    def test_pretrained_model_lists(self):
        # No `max_model_input_sizes` for Cohere model
        self.assertGreaterEqual(len(self.tokenizer_class.pretrained_vocab_files_map), 1)
        self.assertGreaterEqual(len(list(self.tokenizer_class.pretrained_vocab_files_map.values())[0]), 1)

    @require_jinja
    def test_tokenization_for_chat(self):
        tokenizer = self.get_rust_tokenizer()
        test_chats = [
            [{"role": "system", "content": "You are a helpful chatbot."}, {"role": "user", "content": "Hello!"}],
            [
                {"role": "system", "content": "You are a helpful chatbot."},
                {"role": "user", "content": "Hello!"},
                {"role": "assistant", "content": "Nice to meet you."},
            ],
        ]
        tokenized_chats = [tokenizer.apply_chat_template(test_chat) for test_chat in test_chats]
        # fmt: off
        expected_tokens = [
            [5, 36, 99, 59, 60, 41, 58, 60, 71, 55, 46, 71, 60, 61, 58, 54, 71, 60, 55, 51, 45, 54, 99, 38, 36, 99, 59, 65, 59, 60, 45, 53, 71, 60, 55, 51, 45, 54, 99, 38, 65, 243, 394, 204, 336, 84, 88, 887, 374, 216, 74, 286, 22, 8, 36, 99, 59, 60, 41, 58, 60, 71, 55, 46, 71, 60, 61, 58, 54, 71, 60, 55, 51, 45, 54, 99, 38, 36, 99, 61, 59, 45, 58, 71, 60, 55, 51, 45, 54, 99, 38, 48, 420, 87, 9, 8],
            [5, 36, 99, 59, 60, 41, 58, 60, 71, 55, 46, 71, 60, 61, 58, 54, 71, 60, 55, 51, 45, 54, 99, 38, 36, 99, 59, 65,
            59, 60, 45, 53, 71, 60, 55, 51, 45, 54, 99, 38, 65, 243, 394, 204, 336, 84, 88, 887, 374, 216, 74, 286, 22, 8,
            36, 99, 59, 60, 41, 58, 60, 71, 55, 46, 71, 60, 61, 58, 54, 71, 60, 55, 51, 45, 54, 99, 38, 36, 99, 61, 59,
            45, 58, 71, 60, 55, 51, 45, 54, 99, 38, 48, 420, 87, 9, 8, 36, 99, 59, 60, 41, 58, 60, 71, 55, 46, 71, 60, 61,
            58, 54, 71, 60, 55, 51, 45, 54, 99, 38, 36, 99, 43, 48, 41, 60, 42, 55, 60, 71, 60, 55, 51, 45, 54, 99, 38,
            54, 567, 235, 693, 276, 411, 243, 22, 8]
        ]
        # fmt: on
        for tokenized_chat, expected_tokens in zip(tokenized_chats, expected_tokens):
            self.assertListEqual(tokenized_chat, expected_tokens)

    @require_jinja
    def test_tokenization_for_tool_use(self):
        tokenizer = self.get_rust_tokenizer()

        conversation = [{"role": "user", "content": "Whats the biggest penguin in the world?"}]

        tools = [
            {
                "name": "internet_search",
                "description": "Returns a list of relevant document snippets for a textual query retrieved from the internet",
                "parameter_definitions": {
                    "query": {"description": "Query to search the internet with", "type": "str", "required": True}
                },
            },
            {
                "name": "directly_answer",
                "description": "Calls a standard (un-augmented) AI chatbot to generate a response given the conversation history",
                "parameter_definitions": {},
            },
        ]

        tool_use_prompt = tokenizer.apply_tool_use_template(
            conversation,
            tools=tools,
            tokenize=False,
            add_generation_prompt=True,
        )

        expected_prompt = '''<BOS_TOKEN><|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|># Safety Preamble
The instructions in this section override those in the task description and style guide sections. Don't answer questions that are harmful or immoral.

# System Preamble
## Basic Rules
You are a powerful conversational AI trained by Cohere to help people. You are augmented by a number of tools, and your job is to use and consume the output of these tools to best help the user. You will see a conversation history between yourself and a user, ending with an utterance from the user. You will then see a specific instruction instructing you what kind of response to generate. When you answer the user's requests, you cite your sources in your answers, according to those instructions.

# User Preamble
## Task and Context
You help people answer their questions and other requests interactively. You will be asked a very wide array of requests on all kinds of topics. You will be equipped with a wide range of search engines or similar tools to help you, which you use to research your answer. You should focus on serving the user's needs as best you can, which will be wide-ranging.

## Style Guide
Unless the user asks for a different style of answer, you should answer in full sentences, using proper grammar and spelling.

## Available Tools
Here is a list of tools that you have available to you:

```python
def internet_search(query: str) -> List[Dict]:
    """Returns a list of relevant document snippets for a textual query retrieved from the internet

    Args:
        query (str): Query to search the internet with
    """
    pass
```

```python
def directly_answer() -> List[Dict]:
    """Calls a standard (un-augmented) AI chatbot to generate a response given the conversation history
    """
    pass
```<|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|USER_TOKEN|>Whats the biggest penguin in the world?<|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|>Write 'Action:' followed by a json-formatted list of actions that you want to perform in order to produce a good response to the user's last input. You can use any of the supplied tools any number of times, but you should aim to execute the minimum number of necessary actions for the input. You should use the `directly-answer` tool if calling the other tools is unnecessary. The list of actions you want to call should be formatted as a list of json objects, for example:
```json
[
    {
        "tool_name": title of the tool in the specification,
        "parameters": a dict of parameters to input into the tool as they are defined in the specs, or {} if it takes no parameters
    }
]```<|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>'''

        self.assertEqual(tool_use_prompt, expected_prompt)

    @require_jinja
    def test_tokenization_for_grounded_generation(self):
        tokenizer = self.get_rust_tokenizer()
        conversation = [{"role": "user", "content": "Whats the biggest penguin in the world?"}]

        documents = [
            {"title": "Tall penguins", "text": "Emperor penguins are the tallest growing up to 122 cm in height."},
            {"title": "Penguin habitats", "text": "Emperor penguins only live in Antarctica."},
        ]

        grounded_generation_prompt = tokenizer.apply_grounded_generation_template(
            conversation,
            documents=documents,
            citation_mode="accurate",  # or "fast"
            tokenize=False,
            add_generation_prompt=True,
        )

        expected_prompt = """<BOS_TOKEN><|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|># Safety Preamble
The instructions in this section override those in the task description and style guide sections. Don't answer questions that are harmful or immoral.

# System Preamble
## Basic Rules
You are a powerful conversational AI trained by Cohere to help people. You are augmented by a number of tools, and your job is to use and consume the output of these tools to best help the user. You will see a conversation history between yourself and a user, ending with an utterance from the user. You will then see a specific instruction instructing you what kind of response to generate. When you answer the user's requests, you cite your sources in your answers, according to those instructions.

# User Preamble
## Task and Context
You help people answer their questions and other requests interactively. You will be asked a very wide array of requests on all kinds of topics. You will be equipped with a wide range of search engines or similar tools to help you, which you use to research your answer. You should focus on serving the user's needs as best you can, which will be wide-ranging.

## Style Guide
Unless the user asks for a different style of answer, you should answer in full sentences, using proper grammar and spelling.<|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|USER_TOKEN|>Whats the biggest penguin in the world?<|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|><results>
Document: 0
title: Tall penguins
text: Emperor penguins are the tallest growing up to 122 cm in height.

Document: 1
title: Penguin habitats
text: Emperor penguins only live in Antarctica.
</results><|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|>Carefully perform the following instructions, in order, starting each with a new line.
Firstly, Decide which of the retrieved documents are relevant to the user's last input by writing 'Relevant Documents:' followed by comma-separated list of document numbers. If none are relevant, you should instead write 'None'.
Secondly, Decide which of the retrieved documents contain facts that should be cited in a good answer to the user's last input by writing 'Cited Documents:' followed a comma-separated list of document numbers. If you dont want to cite any of them, you should instead write 'None'.
Thirdly, Write 'Answer:' followed by a response to the user's last input in high quality natural english. Use the retrieved documents to help you. Do not insert any citations or grounding markup.
Finally, Write 'Grounded answer:' followed by a response to the user's last input in high quality natural english. Use the symbols <co: doc> and </co: doc> to indicate when a fact comes from a document in the search result, e.g <co: 0>my fact</co: 0> for a fact from document 0.<|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>"""

        self.assertEqual(grounded_generation_prompt, expected_prompt)

    def test_add_prefix_space_fast(self):
        tokenizer_w_prefix = self.get_rust_tokenizer(add_prefix_space=True)
        tokenizer_wo_prefix = self.get_rust_tokenizer(add_prefix_space=False)
        tokens_w_prefix = tokenizer_w_prefix.tokenize("Hey")
        tokens_wo_prefix = tokenizer_wo_prefix.tokenize("Hey")
        self.assertNotEqual(tokens_w_prefix, tokens_wo_prefix)
