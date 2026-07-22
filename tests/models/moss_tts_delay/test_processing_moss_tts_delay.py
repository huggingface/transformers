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
from types import SimpleNamespace

from transformers import is_torch_available
from transformers.testing_utils import require_torch, torch_device


if is_torch_available():
    import torch

    from transformers import (
        MossAudioTokenizerConfig,
        MossAudioTokenizerModel,
        MossAudioTokenizerQuantizerConfig,
        MossTTSDelayConfig,
        MossTTSDelayProcessor,
        PreTrainedTokenizerBase,
    )


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

    class FakeMossAudioTokenizer(MossAudioTokenizerModel):
        def __init__(self):
            super().__init__(
                MossAudioTokenizerConfig(
                    sampling_rate=16000,
                    downsampling_ratios=[4],
                    input_hidden_sizes=[],
                    output_hidden_sizes=[],
                    hidden_sizes=[],
                    num_attention_heads=[],
                    num_hidden_layers=[],
                    intermediate_sizes=[],
                    quantizer_config=MossAudioTokenizerQuantizerConfig(
                        input_hidden_size=4,
                        hidden_size=4,
                        output_hidden_size=4,
                        n_codebooks=2,
                        codebook_size=16,
                        codebook_dim=2,
                    ),
                )
            )
            self.name_or_path = "fake-moss-audio-tokenizer"


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
            n_codebooks=2,
            codebook_size=16,
            codebook_pad_token_id=16,
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
            n_codebooks=2,
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
        self.assertEqual(output["input_ids"].shape[-1], processor.model_config.n_codebooks + 1)
        self.assertEqual(output["attention_mask"].dtype, torch.bool)

    def test_audio_tokenizer_accepts_moss_audio_tokenizer_model(self):
        processor = self.get_processor(audio_tokenizer=FakeMossAudioTokenizer())

        self.assertIsNotNone(processor.audio_tokenizer)

    def test_encode_audios_from_wav_uses_feature_extractor_padding(self):
        class RecordingAudioTokenizer(FakeMossAudioTokenizer):
            def encode(self, input_values, padding_mask=None, num_quantizers=None, return_dict=None):
                self.seen_input_values_shape = tuple(input_values.shape)
                self.seen_padding_mask = padding_mask.detach().cpu()

                hop_length = int(self.config.hop_length)
                num_quantizers = (
                    num_quantizers if num_quantizers is not None else int(self.config.quantizer_config.n_codebooks)
                )
                max_length = input_values.shape[-1] // hop_length
                audio_codes_lengths = (padding_mask.sum(dim=-1).long() + hop_length - 1) // hop_length
                audio_codes = torch.zeros(
                    input_values.shape[0],
                    num_quantizers,
                    max_length,
                    device=input_values.device,
                    dtype=torch.long,
                )
                return SimpleNamespace(audio_codes=audio_codes, audio_codes_lengths=audio_codes_lengths)

        audio_tokenizer = RecordingAudioTokenizer()
        processor = self.get_processor(audio_tokenizer=audio_tokenizer)

        codes = processor.encode_audios_from_wav(
            [torch.ones(1, 5, device=torch_device), torch.ones(1, 8, device=torch_device)],
            sampling_rate=int(processor.model_config.sampling_rate),
            n_codebooks=2,
        )

        self.assertEqual(audio_tokenizer.seen_input_values_shape, (2, 1, 8))
        self.assertEqual(
            audio_tokenizer.seen_padding_mask.tolist(),
            [[True, True, True, True, True, False, False, False], [True, True, True, True, True, True, True, True]],
        )
        self.assertEqual([tuple(code.shape) for code in codes], [(2, 2), (2, 2)])
