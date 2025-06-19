import tempfile
import unittest

from mistral_common.exceptions import InvalidMessageStructureException, TokenizerException

from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.tokenization_mistral_common import MistralCommonTokenizer


class TestMistralCommonTokenizer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tekken_tokenizer: MistralCommonTokenizer = AutoTokenizer.from_pretrained(
            "mistralai/Mistral-Small-3.1-24B-Instruct-2503", tokenizer_type="mistral-common"
        )
        cls.spm_tokenizer: MistralCommonTokenizer = AutoTokenizer.from_pretrained(
            "mistralai/Mistral-Small-Instruct-2409", tokenizer_type="mistral-common"
        )

    def test_vocab_size(self):
        self.assertEqual(self.tekken_tokenizer.vocab_size, 131072)
        self.assertEqual(self.spm_tokenizer.vocab_size, 32768)

    def test_save_pretrained(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_file = self.tekken_tokenizer.save_pretrained(tmp_dir)[0]
            loaded_tokenizer = MistralCommonTokenizer.from_pretrained(tmp_file)

        self.assertIsNotNone(loaded_tokenizer)
        self.assertEqual(self.tekken_tokenizer.get_vocab(), loaded_tokenizer.get_vocab())
        self.assertEqual(
            self.tekken_tokenizer._tokenizer.instruct_tokenizer.tokenizer.version,
            loaded_tokenizer._tokenizer.instruct_tokenizer.tokenizer.version,
        )

        with self.assertRaises(
            ValueError, msg="Kwargs [unk_args] are not supported by `MistralCommonTokenizer.save_pretrained`."
        ):
            with tempfile.TemporaryDirectory() as tmp_dir:
                self.tekken_tokenizer.save_pretrained(tmp_dir, unk_args="")

    def test_encode(self):
        # Encode the same text with both tokenizers
        # "Hello, world!"
        tekken_tokens_with_special = [1, 22177, 1044, 4304, 1033, 2]
        spm_tokens_with_special = [1, 23325, 29493, 2294, 29576, 2]

        self.assertEqual(
            self.tekken_tokenizer.encode("Hello, world!", add_special_tokens=True), tekken_tokens_with_special
        )
        self.assertEqual(self.spm_tokenizer.encode("Hello, world!", add_special_tokens=True), spm_tokens_with_special)
        self.assertEqual(
            self.tekken_tokenizer.encode("Hello, world!", add_special_tokens=False), tekken_tokens_with_special[1:-1]
        )
        self.assertEqual(
            self.spm_tokenizer.encode("Hello, world!", add_special_tokens=False), spm_tokens_with_special[1:-1]
        )

        with self.assertRaises(
            ValueError, msg="Kwargs [unk_args] are not supported by `MistralCommonTokenizer.encode`."
        ):
            self.tekken_tokenizer.encode("Hello, world!", add_special_tokens=True, unk_args="")

    def test_decode(self):
        # Decode the same text with both tokenizers
        # "Hello, world!"
        tekken_tokens = [1, 22177, 1044, 4304, 1033, 2]
        spm_tokens = [1, 23325, 29493, 2294, 29576, 2]
        tekken_tokens_with_space = [1, 22177, 1044, 4304, 2662, 2]
        # "Hello, world !"
        spm_tokens_with_space = [1, 23325, 29493, 2294, 1686, 2]

        # Test decode with and without skip_special_tokens
        self.assertEqual(self.tekken_tokenizer.decode(tekken_tokens, skip_special_tokens=True), "Hello, world!")
        self.assertEqual(self.spm_tokenizer.decode(spm_tokens, skip_special_tokens=True), "Hello, world!")
        self.assertEqual(
            self.tekken_tokenizer.decode(tekken_tokens, skip_special_tokens=False), "<s>Hello, world!</s>"
        )
        self.assertEqual(self.spm_tokenizer.decode(spm_tokens, skip_special_tokens=False), "<s>▁Hello,▁world!</s>")

        # Test decode with and without clean_up_tokenization_spaces
        self.assertEqual(
            self.tekken_tokenizer.decode(
                tekken_tokens_with_space, skip_special_tokens=True, clean_up_tokenization_spaces=True
            ),
            "Hello, world!",
        )
        self.assertEqual(
            self.spm_tokenizer.decode(
                spm_tokens_with_space, skip_special_tokens=True, clean_up_tokenization_spaces=True
            ),
            "Hello, world!",
        )
        self.assertEqual(
            self.tekken_tokenizer.decode(
                tekken_tokens_with_space, skip_special_tokens=True, clean_up_tokenization_spaces=False
            ),
            "Hello, world !",
        )
        self.assertEqual(
            self.spm_tokenizer.decode(
                spm_tokens_with_space, skip_special_tokens=True, clean_up_tokenization_spaces=False
            ),
            "Hello, world !",
        )

        with self.assertRaises(
            ValueError, msg="Kwargs [unk_args] are not supported by `MistralCommonTokenizer.decode`."
        ):
            self.tekken_tokenizer.decode(tekken_tokens, skip_special_tokens=False, unk_args="")

    def test_batch_decode(self):
        # Batch decode the same text with both tokenizers
        # "Hello, world!" and "Hello, world !"
        tekken_tokens = [[1, 22177, 1044, 4304, 1033, 2], [1, 22177, 1044, 4304, 2662, 2]]
        spm_tokens = [[1, 23325, 29493, 2294, 29576, 2], [1, 23325, 29493, 2294, 1686, 2]]

        # Test batch_decode with and without skip_special_tokens
        self.assertEqual(
            self.tekken_tokenizer.batch_decode(tekken_tokens, skip_special_tokens=True),
            ["Hello, world!", "Hello, world !"],
        )
        self.assertEqual(
            self.spm_tokenizer.batch_decode(spm_tokens, skip_special_tokens=True), ["Hello, world!", "Hello, world !"]
        )
        self.assertEqual(
            self.tekken_tokenizer.batch_decode(tekken_tokens, skip_special_tokens=False),
            ["<s>Hello, world!</s>", "<s>Hello, world !</s>"],
        )
        self.assertEqual(
            self.spm_tokenizer.batch_decode(spm_tokens, skip_special_tokens=False),
            ["<s>▁Hello,▁world!</s>", "<s>▁Hello,▁world▁!</s>"],
        )

        # Test batch_decode with and without clean_up_tokenization_spaces
        self.assertEqual(
            self.tekken_tokenizer.batch_decode(
                tekken_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=True
            ),
            ["Hello, world!", "Hello, world!"],
        )
        self.assertEqual(
            self.spm_tokenizer.batch_decode(spm_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=True),
            ["Hello, world!", "Hello, world!"],
        )
        self.assertEqual(
            self.tekken_tokenizer.batch_decode(
                tekken_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False
            ),
            ["Hello, world!", "Hello, world !"],
        )
        self.assertEqual(
            self.spm_tokenizer.batch_decode(spm_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False),
            ["Hello, world!", "Hello, world !"],
        )

        with self.assertRaises(
            ValueError, msg="Kwargs [unk_args] are not supported by `MistralCommonTokenizer.decode`."
        ):
            self.tekken_tokenizer.batch_decode(tekken_tokens, skip_special_tokens=False, unk_args="")

    def test_apply_chat_template(self):
        # Apply chat template with both tokenizers
        conversation = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hi!"},
            {"role": "assistant", "content": "Hello! How can I help you?"},
            {"role": "user", "content": "What is the capital of France?"},
        ]

        # Test apply_chat_template with and without tokenize
        self.assertEqual(
            self.tekken_tokenizer.apply_chat_template(conversation, tokenize=False),
            "<s>[SYSTEM_PROMPT]You are a helpful assistant.[/SYSTEM_PROMPT][INST]Hi![/INST]Hello! How can I help you?</s>[INST]What is the capital of France?[/INST]",
        )
        self.assertEqual(
            self.tekken_tokenizer.apply_chat_template(conversation, tokenize=True),
            [
                1,
                17,
                4568,
                1584,
                1261,
                20351,
                27089,
                1046,
                18,
                3,
                37133,
                1033,
                4,
                22177,
                1033,
                3075,
                1710,
                1362,
                3508,
                1636,
                1063,
                2,
                3,
                7493,
                1395,
                1278,
                8961,
                1307,
                5498,
                1063,
                4,
            ],
        )
        self.assertEqual(
            self.spm_tokenizer.apply_chat_template(conversation, tokenize=False),
            "<s>[INST]▁Hi![/INST]▁Hello!▁How▁can▁I▁help▁you?</s>[INST]▁You▁are▁a▁helpful▁assistant.<0x0A><0x0A>What▁is▁the▁capital▁of▁France?[/INST]",
        )
        self.assertEqual(
            self.spm_tokenizer.apply_chat_template(conversation, tokenize=True),
            [
                1,
                3,
                16127,
                29576,
                4,
                23325,
                29576,
                2370,
                1309,
                1083,
                2084,
                1136,
                29572,
                2,
                3,
                1763,
                1228,
                1032,
                11633,
                14660,
                29491,
                781,
                781,
                3963,
                1117,
                1040,
                6333,
                1070,
                5611,
                29572,
                4,
            ],
        )

        with self.assertRaises(
            ValueError, msg="Kwargs [unk_args] are not supported by `MistralCommonTokenizer.apply_chat_template`."
        ):
            self.tekken_tokenizer.apply_chat_template(conversation, tokenize=True, unk_args="")

    def test_apply_chat_template_continue_final_message(self):
        # Apply chat template with both tokenizers
        conversation = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hi!"},
            {"role": "assistant", "content": "Hello! How can I help you?"},
            {"role": "user", "content": "What is the capital of France?"},
            {"role": "assistant", "content": "Paris"},
        ]

        self.assertEqual(
            self.tekken_tokenizer.apply_chat_template(conversation, tokenize=False, continue_final_message=True),
            "<s>[SYSTEM_PROMPT]You are a helpful assistant.[/SYSTEM_PROMPT][INST]Hi![/INST]Hello! How can I help you?</s>[INST]What is the capital of France?[/INST]Paris",
        )
        self.assertEqual(
            self.tekken_tokenizer.apply_chat_template(conversation, tokenize=True, continue_final_message=True),
            [
                1,
                17,
                4568,
                1584,
                1261,
                20351,
                27089,
                1046,
                18,
                3,
                37133,
                1033,
                4,
                22177,
                1033,
                3075,
                1710,
                1362,
                3508,
                1636,
                1063,
                2,
                3,
                7493,
                1395,
                1278,
                8961,
                1307,
                5498,
                1063,
                4,
                42572,
            ],
        )
        self.assertEqual(
            self.spm_tokenizer.apply_chat_template(conversation, tokenize=False, continue_final_message=True),
            "<s>[INST]▁Hi![/INST]▁Hello!▁How▁can▁I▁help▁you?</s>[INST]▁You▁are▁a▁helpful▁assistant.<0x0A><0x0A>What▁is▁the▁capital▁of▁France?[/INST]▁Paris",
        )
        self.assertEqual(
            self.spm_tokenizer.apply_chat_template(conversation, tokenize=True, continue_final_message=True),
            [
                1,
                3,
                16127,
                29576,
                4,
                23325,
                29576,
                2370,
                1309,
                1083,
                2084,
                1136,
                29572,
                2,
                3,
                1763,
                1228,
                1032,
                11633,
                14660,
                29491,
                781,
                781,
                3963,
                1117,
                1040,
                6333,
                1070,
                5611,
                29572,
                4,
                6233,
            ],
        )

        with self.assertRaises(InvalidMessageStructureException):
            self.tekken_tokenizer.apply_chat_template(conversation, tokenize=False, continue_final_message=False)
        with self.assertRaises(InvalidMessageStructureException):
            self.spm_tokenizer.apply_chat_template(conversation, tokenize=False, continue_final_message=False)

    def test_apply_chat_template_with_tools(self):
        # Apply chat template with tools with both tokenizers
        conversation = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hi!"},
            {"role": "assistant", "content": "Hello! How can I help you?"},
            {"role": "user", "content": "What is the temperature in Paris?"},
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "azerty123",
                        "function": {
                            "name": "get_current_weather",
                            "arguments": {"location": "Paris", "format": "text", "unit": "celsius"},
                        },
                    }
                ],
            },
            {"role": "tool", "name": "get_current_weather", "content": "22", "tool_call_id": "azerty123"},
        ]
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_current_weather",
                    "description": "Get the current weather in a given location",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {
                                "type": "string",
                                "description": "The city and state, e.g. San Francisco, CA",
                                "required": ["location"],
                            },
                            "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                            "format": {
                                "type": "string",
                                "enum": ["text", "json"],
                                "description": "The format of the response",
                                "required": ["format"],
                            },
                        },
                    },
                },
            }
        ]
        self.assertEqual(
            self.tekken_tokenizer.apply_chat_template(conversation, tools=tools, tokenize=False),
            '<s>[SYSTEM_PROMPT]You are a helpful assistant.[/SYSTEM_PROMPT][INST]Hi![/INST]Hello! How can I help you?</s>[AVAILABLE_TOOLS][{"type": "function", "function": {"name": "get_current_weather", "description": "Get the current weather in a given location", "parameters": {"type": "object", "properties": {"location": {"type": "string", "description": "The city and state, e.g. San Francisco, CA", "required": ["location"]}, "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}, "format": {"type": "string", "enum": ["text", "json"], "description": "The format of the response", "required": ["format"]}}}}}][/AVAILABLE_TOOLS][INST]What is the temperature in Paris?[/INST][TOOL_CALLS][{"name": "get_current_weather", "arguments": {"location": "Paris", "format": "text", "unit": "celsius"}, "id": "azerty123"}]</s>[TOOL_RESULTS]azerty123[TOOL_CONTENT]22[/TOOL_RESULTS]',
        )
        self.assertEqual(
            self.spm_tokenizer.apply_chat_template(conversation, tools=tools, tokenize=False),
            '<s>[INST]▁Hi![/INST]▁Hello!▁How▁can▁I▁help▁you?</s>[AVAILABLE_TOOLS]▁[{"type":▁"function",▁"function":▁{"name":▁"get_current_weather",▁"description":▁"Get▁the▁current▁weather▁in▁a▁given▁location",▁"parameters":▁{"type":▁"object",▁"properties":▁{"location":▁{"type":▁"string",▁"description":▁"The▁city▁and▁state,▁e.g.▁San▁Francisco,▁CA",▁"required":▁["location"]},▁"unit":▁{"type":▁"string",▁"enum":▁["celsius",▁"fahrenheit"]},▁"format":▁{"type":▁"string",▁"enum":▁["text",▁"json"],▁"description":▁"The▁format▁of▁the▁response",▁"required":▁["format"]}}}}}][/AVAILABLE_TOOLS][INST]▁You▁are▁a▁helpful▁assistant.<0x0A><0x0A>What▁is▁the▁temperature▁in▁Paris?[/INST][TOOL_CALLS]▁[{"name":▁"get_current_weather",▁"arguments":▁{"location":▁"Paris",▁"format":▁"text",▁"unit":▁"celsius"},▁"id":▁"azerty123"}]</s>[TOOL_RESULTS]▁{"content":▁22,▁"call_id":▁"azerty123"}[/TOOL_RESULTS]',
        )

    def test_apply_chat_template_with_image(self):
        # Apply chat template with image with both tokenizers
        conversation = [
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What is this?"},
                    {
                        "type": "image_url",
                        "image_url": {"url": "https://picsum.photos/id/237/200/300"},
                    },
                ],
            },
        ]

        self.assertEqual(
            self.tekken_tokenizer.apply_chat_template(conversation, tokenize=True),
            [
                1,
                17,
                4568,
                1584,
                1261,
                20351,
                27089,
                1046,
                18,
                3,
                10,
                10,
                10,
                10,
                10,
                10,
                10,
                10,
                12,
                10,
                10,
                10,
                10,
                10,
                10,
                10,
                10,
                12,
                10,
                10,
                10,
                10,
                10,
                10,
                10,
                10,
                12,
                10,
                10,
                10,
                10,
                10,
                10,
                10,
                10,
                12,
                10,
                10,
                10,
                10,
                10,
                10,
                10,
                10,
                12,
                10,
                10,
                10,
                10,
                10,
                10,
                10,
                10,
                12,
                10,
                10,
                10,
                10,
                10,
                10,
                10,
                10,
                12,
                10,
                10,
                10,
                10,
                10,
                10,
                10,
                10,
                12,
                10,
                10,
                10,
                10,
                10,
                10,
                10,
                10,
                12,
                10,
                10,
                10,
                10,
                10,
                10,
                10,
                10,
                12,
                10,
                10,
                10,
                10,
                10,
                10,
                10,
                10,
                13,
                7493,
                1395,
                1593,
                1063,
                4,
            ],
        )

        with self.assertRaises(AssertionError, msg="Make sure to define a multi-modal encoder at init"):
            self.spm_tokenizer.apply_chat_template(conversation, tokenize=True)

    def test_apply_chat_template_with_truncation(self):
        # Apply chat template with image with both tokenizers
        conversation = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hi!"},
            {"role": "assistant", "content": "Hello! How can I help you?"},
            {"role": "user", "content": "What is the capital of France?"},
        ]

        print(self.tekken_tokenizer.apply_chat_template(conversation, tokenize=True, truncation=True, max_length=20))

        self.assertEqual(
            self.tekken_tokenizer.apply_chat_template(conversation, tokenize=True, truncation=True, max_length=20),
            [1, 17, 4568, 1584, 1261, 20351, 27089, 1046, 18, 3, 7493, 1395, 1278, 8961, 1307, 5498, 1063, 4],
        )
        self.assertEqual(
            self.spm_tokenizer.apply_chat_template(conversation, tokenize=True, truncation=True, max_length=20),
            [
                1,
                3,
                16127,
                29576,
                4,
                23325,
                29576,
                2370,
                1309,
                1083,
                2084,
                1136,
                29572,
                2,
                3,
                1763,
                1228,
                1032,
                11633,
                14660,
                29491,
                781,
                781,
                3963,
                1117,
                1040,
                6333,
                1070,
                5611,
                29572,
                4,
            ],
        )

        with self.assertRaises(
            TokenizerException,
            msg="encoding a chat completion request with truncation, but no max model len was provided",
        ):
            self.tekken_tokenizer.apply_chat_template(conversation, truncation=True)
