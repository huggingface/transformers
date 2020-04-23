# coding=utf-8
# Copyright 2020 HuggingFace Inc. team.
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
from transformers.file_utils import cached_property

from .utils import require_torch, slow, torch_device


if is_torch_available():
    import torch
    from transformers import MarianForConditionalGeneration, MarianSentencePieceTokenizer


@require_torch
class IntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.model_name = "opus/marian-en-de"
        cls.tokenizer = MarianSentencePieceTokenizer.from_pretrained(cls.model_name)
        cls.eos_token_id = cls.tokenizer.eos_token_id
        return cls

    @cached_property
    def model(self):
        return MarianForConditionalGeneration.from_pretrained(self.model_name).to(torch_device)

    @slow
    def test_forward(self):
        src, tgt = ["I am a small frog"], ["▁Ich ▁bin ▁ein ▁kleiner ▁Fro sch"]
        expected = [38, 121, 14, 697, 38848, 0]

        model_inputs: dict = self.tokenizer.prepare_translation_batch(src, tgt_texts=tgt)
        self.assertListEqual(expected, model_inputs["input_ids"][0].tolist())

        desired_keys = {
            "input_ids",
            "attention_mask",
            "decoder_input_ids",
            "decoder_attention_mask",
        }
        self.assertSetEqual(desired_keys, set(model_inputs.keys()))
        with torch.no_grad():
            logits, *enc_features = self.model(**model_inputs)
        max_indices = logits.argmax(-1)
        predicted_words = self.tokenizer.decode_batch(max_indices)
        print(predicted_words)

    @slow
    def test_repl_generate(self):
        src = ["I am a small frog", "Hello"]
        model_inputs: dict = self.tokenizer.prepare_translation_batch(src).to(torch_device)
        generated_ids = self.model.generate(model_inputs["input_ids"], num_beams=2,)
        generated_words = self.tokenizer.decode_batch(generated_ids)[0]
        expected_words = "Ich bin ein kleiner Frosch"
        self.assertEqual(expected_words, generated_words)

    def test_marian_equivalence(self):
        input_ids = self.tokenizer.prepare_translation_batch(["I am a small frog"])["input_ids"][0].to(torch_device)
        expected = [38, 121, 14, 697, 38848, 0]
        self.assertListEqual(expected, input_ids.tolist())

    def test_pad_not_split(self):
        input_ids_w_pad = self.tokenizer.prepare_translation_batch(["I am a small frog <pad>"])["input_ids"][0]
        expected_w_pad = [38, 121, 14, 697, 38848, self.tokenizer.pad_token_id, 0]  # pad
        self.assertListEqual(expected_w_pad, input_ids_w_pad.tolist())
