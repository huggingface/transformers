# Copyright 2026 OpenMOSS and The HuggingFace Inc. team. All rights reserved.
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

from transformers import is_torch_available
from transformers.testing_utils import require_torch, torch_device


if is_torch_available():
    import torch

    from transformers import MossTTSDelayConfig, MossTTSDelayProcessor, PreTrainedModel, PreTrainedTokenizerBase


if is_torch_available():

    class FakeMossTTSDelayTokenizer(PreTrainedTokenizerBase):
        model_input_names = ["input_ids", "attention_mask"]

        def __init__(self, **kwargs):
            super().__init__(
                bos_token="<|endoftext|>",
                eos_token="<|im_end|>",
                pad_token="<|endoftext|>",
                **kwargs,
            )
            self.vocab = {
                "<|endoftext|>": 0,
                "<|im_start|>": 1,
                "<|im_end|>": 2,
                "<|audio_start|>": 3,
                "<|audio_end|>": 4,
                "<|audio_user_slot|>": 5,
                "<|audio_assistant_gen_slot|>": 6,
                "<|audio_assistant_delay_slot|>": 7,
            }
            self.ids_to_tokens = {value: key for key, value in self.vocab.items()}

        def convert_tokens_to_ids(self, tokens):
            if isinstance(tokens, list):
                return [self.convert_tokens_to_ids(token) for token in tokens]
            return self.vocab.get(tokens, 50)

        def convert_ids_to_tokens(self, ids, skip_special_tokens=False):
            if isinstance(ids, list):
                return [self.convert_ids_to_tokens(idx, skip_special_tokens=skip_special_tokens) for idx in ids]
            return self.ids_to_tokens.get(int(ids), f"token_{ids}")

        def encode(self, text, **kwargs):
            token_ids = []
            index = 0
            special_tokens = sorted(self.vocab, key=len, reverse=True)
            while index < len(text):
                for token in special_tokens:
                    if text.startswith(token, index):
                        token_ids.append(self.vocab[token])
                        index += len(token)
                        break
                else:
                    token_ids.append(20 + (ord(text[index]) % 30))
                    index += 1
            return token_ids

        def apply_chat_template(self, messages, add_generation_prompt=False, tokenize=False, **kwargs):
            rendered = "".join(
                f"<|im_start|>{message['role']}\n{message['content']}<|im_end|>\n" for message in messages
            )
            if add_generation_prompt:
                rendered += "<|im_start|>assistant\n"
            if tokenize:
                return self.encode(rendered)
            return rendered

    class FakeMossAudioTokenizer(PreTrainedModel):
        config_class = MossTTSDelayConfig

        def __init__(self):
            super().__init__(MossTTSDelayConfig())
            self.name_or_path = "fake-moss-audio-tokenizer"
            self.proj = torch.nn.Linear(1, 1)


@require_torch
class MossTTSDelayProcessorTest(unittest.TestCase):
    def get_config(self):
        return MossTTSDelayConfig(
            language_config={
                "vocab_size": 99,
                "hidden_size": 32,
                "intermediate_size": 64,
                "num_hidden_layers": 2,
                "num_attention_heads": 4,
                "num_key_value_heads": 4,
            },
            n_vq=2,
            audio_vocab_size=16,
            audio_pad_code=16,
            pad_token_id=0,
            im_start_token_id=1,
            im_end_token_id=2,
            audio_start_token_id=3,
            audio_end_token_id=4,
            audio_user_slot_token_id=5,
            audio_assistant_gen_slot_token_id=6,
            audio_assistant_delay_slot_token_id=7,
        )

    def get_processor(self, audio_tokenizer=None):
        return MossTTSDelayProcessor(
            tokenizer=FakeMossTTSDelayTokenizer(),
            audio_tokenizer=audio_tokenizer,
            model_config=self.get_config(),
        )

    def test_build_user_message(self):
        message = MossTTSDelayProcessor.build_user_message(text="Hello.", language="English")

        self.assertEqual(message["role"], "user")
        self.assertEqual(message["audio_codes_list"], [])
        self.assertIn("- Language:\nEnglish", message["content"])
        self.assertIn("- Text:\nHello.", message["content"])

    def test_build_user_message_wraps_single_reference(self):
        reference = torch.ones((2, 2), dtype=torch.long)
        message = MossTTSDelayProcessor.build_user_message(text="Clone this.", reference=reference)

        self.assertEqual(len(message["audio_codes_list"]), 1)
        self.assertIn("<|audio|>", message["content"])

    def test_build_assistant_message(self):
        audio_codes = torch.ones((2, 2), dtype=torch.long)
        message = MossTTSDelayProcessor.build_assistant_message([audio_codes])

        self.assertEqual(message["role"], "assistant")
        self.assertEqual(message["content"], "<|audio|>")
        self.assertEqual(len(message["audio_codes_list"]), 1)

    def test_delay_pattern_roundtrip(self):
        processor = self.get_processor()
        codes = torch.tensor([[1, 2], [3, 4], [5, 6]], device=torch_device)

        delayed = processor.apply_delay_pattern(codes, pad_code=16)
        restored = processor.apply_de_delay_pattern(delayed)

        self.assertEqual(delayed.shape, (4, 2))
        torch.testing.assert_close(restored, codes)

    def test_replace_audio_placeholders(self):
        processor = self.get_processor()
        content = processor._replace_audio_placeholders(
            content="before <|audio|> after",
            lengths=[2],
            n_vq=2,
            gen_slot_token="<|audio_user_slot|>",
            delay_slot_token="<|audio_user_slot|>",
            audio_start_token="<|audio_start|>",
            audio_end_token="<|audio_end|>",
        )

        self.assertIn("<|audio_start|>", content)
        self.assertIn("<|audio_end|>", content)
        self.assertEqual(content.count("<|audio_user_slot|>"), 3)

    def test_call_text_only_generation(self):
        processor = self.get_processor()
        message = processor.build_user_message(text="Hello.", language="English")
        output = processor([message], mode="generation")

        self.assertIn("input_ids", output)
        self.assertIn("attention_mask", output)
        self.assertEqual(output["input_ids"].shape[0], 1)
        self.assertEqual(output["input_ids"].shape[-1], processor.model_config.n_vq + 1)
        self.assertEqual(output["attention_mask"].dtype, torch.bool)

    def test_audio_tokenizer_accepts_pretrained_model(self):
        processor = self.get_processor(audio_tokenizer=FakeMossAudioTokenizer())

        self.assertIsNotNone(processor.audio_tokenizer)
