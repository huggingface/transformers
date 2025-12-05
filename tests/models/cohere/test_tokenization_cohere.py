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

from transformers import CohereTokenizer
from transformers.testing_utils import (
    require_jinja,
    require_tokenizers,
    require_torch_multi_accelerator,
)

from ...test_tokenization_common import TokenizerTesterMixin


@require_tokenizers
class CohereTokenizationTest(TokenizerTesterMixin, unittest.TestCase):
    tokenizer_class = CohereTokenizer
    from_pretrained_vocab_key = "tokenizer_file"
    from_pretrained_id = "hf-internal-testing/tiny-random-CohereForCausalLM"
    special_tokens_map = {
        "bos_token": "<BOS_TOKEN>",
        "eos_token": "<|END_OF_TURN_TOKEN|>",
        "unk_token": "<UNK>",
        "pad_token": "<PAD>",
    }

    integration_expected_tokens = ['T', 'h', 'is', 'Ġis', 'Ġa', 'Ġt', 'est', 'Ġ', 'Ł', 'ĺ', 'Ĭ', 'Ċ', 'I', 'Ġwas', 'Ġb', 'orn', 'Ġin', 'Ġ', '9', '2', '0', '0', '0', ',', 'Ġand', 'Ġthis', 'Ġis', 'Ġf', 'als', 'Ã©', '.', 'Ċ', 'ç', 'Ķ', 'Ł', 'æ', '´', '»', 'ç', 'ļ', 'Ħ', 'ç', 'ľ', 'Ł', 'è', '°', 'Ľ', 'æ', 'ĺ', '¯', 'Ċ', 'H', 'i', 'Ġ', 'ĠH', 'ell', 'o', 'Ċ', 'H', 'i', 'Ġ', 'Ġ', 'ĠH', 'ell', 'o', 'Ċ', 'Ċ', 'ĠĊ', 'Ġ', 'ĠĊ', 'ĠH', 'ell', 'o', 'Ċ', '<', 's', '>', 'Ċ', 'h', 'i', '<', 's', '>', 't', 'he', 're', 'Ċ', 'T', 'he', 'Ġfollow', 'ing', 'Ġst', 'r', 'ing', 'Ġsh', 'ould', 'Ġbe', 'Ġpro', 'per', 'ly', 'Ġen', 'c', 'od', 'ed', ':', 'ĠH', 'ell', 'o', '.', 'Ċ', 'B', 'ut', 'Ġ', 'ird', 'Ġand', 'Ġ', 'à', '¸', 'Ľ', 'à', '¸', 'µ', 'Ġ', 'Ġ', 'Ġ', 'ird', 'Ġ', 'Ġ', 'Ġ', 'à', '¸', 'Ķ', 'Ċ', 'H', 'ey', 'Ġh', 'ow', 'Ġare', 'Ġy', 'ou', 'Ġdo', 'ing']  # fmt: skip
    integration_expected_token_ids = [60, 80, 223, 307, 204, 202, 333, 167, 199, 192, 178, 166, 49, 265, 227, 712, 229, 167, 33, 26, 24, 24, 24, 20, 233, 524, 307, 222, 632, 1018, 22, 166, 160, 188, 199, 159, 120, 127, 160, 194, 172, 160, 196, 199, 161, 116, 195, 159, 192, 115, 166, 48, 81, 167, 289, 420, 87, 166, 48, 81, 167, 167, 289, 420, 87, 166, 166, 259, 167, 259, 289, 420, 87, 166, 36, 91, 38, 166, 80, 81, 36, 91, 38, 92, 203, 210, 166, 60, 203, 765, 231, 292, 90, 231, 396, 458, 299, 348, 474, 271, 551, 75, 339, 212, 34, 289, 420, 87, 22, 166, 42, 293, 167, 813, 233, 167, 153, 124, 195, 153, 124, 121, 167, 167, 167, 813, 167, 167, 167, 153, 124, 188, 166, 48, 634, 240, 291, 394, 411, 243, 793, 231]  # fmt: skip
    expected_tokens_from_ids = ['T', 'h', 'is', 'Ġis', 'Ġa', 'Ġt', 'est', 'Ġ', 'Ł', 'ĺ', 'Ĭ', 'Ċ', 'I', 'Ġwas', 'Ġb', 'orn', 'Ġin', 'Ġ', '9', '2', '0', '0', '0', ',', 'Ġand', 'Ġthis', 'Ġis', 'Ġf', 'als', 'Ã©', '.', 'Ċ', 'ç', 'Ķ', 'Ł', 'æ', '´', '»', 'ç', 'ļ', 'Ħ', 'ç', 'ľ', 'Ł', 'è', '°', 'Ľ', 'æ', 'ĺ', '¯', 'Ċ', 'H', 'i', 'Ġ', 'ĠH', 'ell', 'o', 'Ċ', 'H', 'i', 'Ġ', 'Ġ', 'ĠH', 'ell', 'o', 'Ċ', 'Ċ', 'ĠĊ', 'Ġ', 'ĠĊ', 'ĠH', 'ell', 'o', 'Ċ', '<', 's', '>', 'Ċ', 'h', 'i', '<', 's', '>', 't', 'he', 're', 'Ċ', 'T', 'he', 'Ġfollow', 'ing', 'Ġst', 'r', 'ing', 'Ġsh', 'ould', 'Ġbe', 'Ġpro', 'per', 'ly', 'Ġen', 'c', 'od', 'ed', ':', 'ĠH', 'ell', 'o', '.', 'Ċ', 'B', 'ut', 'Ġ', 'ird', 'Ġand', 'Ġ', 'à', '¸', 'Ľ', 'à', '¸', 'µ', 'Ġ', 'Ġ', 'Ġ', 'ird', 'Ġ', 'Ġ', 'Ġ', 'à', '¸', 'Ķ', 'Ċ', 'H', 'ey', 'Ġh', 'ow', 'Ġare', 'Ġy', 'ou', 'Ġdo', 'ing']  # fmt: skip
    integration_expected_decoded_text = "This is a test ���\nI was born in 92000, and this is falsé.\n生活的真谛是\nHi  Hello\nHi   Hello\n\n \n  \n Hello\n<s>\nhi<s>there\nThe following string should be properly encoded: Hello.\nBut ird and ปี   ird   ด\nHey how are you doing"

    # This gives CPU OOM on a single-gpu runner (~60G RAM). On multi-gpu runner, it has ~180G RAM which is enough.
    @require_torch_multi_accelerator
    def test_torch_encode_plus_sent_to_model(self):
        super().test_torch_encode_plus_sent_to_model()

    def test_encodings_from_sample_data(self):
        """
        Assert that the created tokens are the same than the hard-coded ones
        """
        tokenizer = self.get_tokenizer()

        INPUT_SENTENCES = ["The quick brown fox<|END_OF_TURN_TOKEN|>", "jumps over the lazy dog<|END_OF_TURN_TOKEN|>"]
        TARGET_TOKENS = [
            [5, 60, 203, 746, 666, 980, 571, 222, 87, 96, 8],
            [5, 82, 332, 88, 91, 544, 206, 257, 930, 97, 239, 435, 8],
        ]

        computed_tokens = tokenizer(INPUT_SENTENCES)["input_ids"]
        self.assertListEqual(TARGET_TOKENS, computed_tokens)

        INPUT_SENTENCES_W_BOS = [
            "<BOS_TOKEN>The quick brown fox<|END_OF_TURN_TOKEN|>",
            "<BOS_TOKEN>jumps over the lazy dog<|END_OF_TURN_TOKEN|>",
        ]
        decoded_tokens = tokenizer.decode(computed_tokens)
        self.assertListEqual(decoded_tokens, INPUT_SENTENCES_W_BOS)

    def test_pretrained_model_lists(self):
        # No `max_model_input_sizes` for Cohere model
        self.assertGreaterEqual(len(self.tokenizer_class.pretrained_vocab_files_map), 1)
        self.assertGreaterEqual(len(list(self.tokenizer_class.pretrained_vocab_files_map.values())[0]), 1)

    @require_jinja
    def test_tokenization_for_tool_use(self):
        tokenizer = self.get_tokenizer()

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
        tokenizer = self.get_tokenizer()
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
        tokenizer_w_prefix = self.get_tokenizer(add_prefix_space=True)
        tokenizer_wo_prefix = self.get_tokenizer(add_prefix_space=False)
        tokens_w_prefix = tokenizer_w_prefix.tokenize("Hey")
        tokens_wo_prefix = tokenizer_wo_prefix.tokenize("Hey")
        self.assertNotEqual(tokens_w_prefix, tokens_wo_prefix)
