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
    from transformers import AutoTokenizer, MarianConfig, AutoConfig, AutoModelWithLMHead


@require_torch
class MarianIntegrationTest(unittest.TestCase):
    src = "en"
    tgt = "de"
    src_text = [
        "I am a small frog.",
        "Now I can forget the 100 words of german that I know.",
        "Tom asked his teacher for advice.",
        "That's how I would do it.",
        "Tom really admired Mary's courage.",
        "Turn around and close your eyes.",
    ]
    expected_text = [
        "Ich bin ein kleiner Frosch.",
        "Jetzt kann ich die 100 Wörter des Deutschen vergessen, die ich kenne.",
        "Tom bat seinen Lehrer um Rat.",
        "So würde ich das machen.",
        "Tom bewunderte Marias Mut wirklich.",
        "Drehen Sie sich um und schließen Sie die Augen.",
    ]
    # ^^ actual C++ output differs slightly: (1) des Deutschen removed, (2) ""-> "O", (3) tun -> machen

    @classmethod
    def setUpClass(cls) -> None:
        cls.model_name = f"Helsinki-NLP/opus-mt-{cls.src}-{cls.tgt}"
        cls.tokenizer = AutoTokenizer.from_pretrained(cls.model_name)
        cls.eos_token_id = cls.tokenizer.eos_token_id
        return cls

    @cached_property
    def model(self):
        model = AutoModelWithLMHead.from_pretrained(self.model_name).to(torch_device)
        if torch_device == "cuda":
            return model.half()
        else:
            return model

    def _test_repl_generate_batch(self):
        model_inputs: dict = self.tokenizer.prepare_translation_batch(src_texts=self.src_text).to(torch_device)
        self.assertEqual(self.model.device, model_inputs["input_ids"].device)
        generated_ids = self.model.generate(
            model_inputs["input_ids"], attention_mask=model_inputs["attention_mask"], num_beams=2,
        )
        generated_words = self.tokenizer.decode_batch(generated_ids, skip_special_tokens=True)
        self.assertListEqual(self.expected_text, generated_words)


class TestMarian_EN_DE_More(MarianIntegrationTest):
    @slow
    def test_forward(self):
        src, tgt = ["I am a small frog"], ["▁Ich ▁bin ▁ein ▁kleiner ▁Fro sch"]
        expected = [38, 121, 14, 697, 38848, 0]

        model_inputs: dict = self.tokenizer.prepare_translation_batch(src, tgt_texts=tgt).to(torch_device)
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
        self.tokenizer.decode_batch(max_indices)

    def test_tokenizer_equivalence(self):
        batch = self.tokenizer.prepare_translation_batch(["I am a small frog"]).to(torch_device)
        input_ids = batch["input_ids"][0]
        expected = [38, 121, 14, 697, 38848, 0]
        self.assertListEqual(expected, input_ids.tolist())

    def test_pad_not_split(self):
        input_ids_w_pad = self.tokenizer.prepare_translation_batch(["I am a small frog <pad>"])["input_ids"][0]
        expected_w_pad = [38, 121, 14, 697, 38848, self.tokenizer.pad_token_id, 0]  # pad
        self.assertListEqual(expected_w_pad, input_ids_w_pad.tolist())

    @slow
    def test_batch_generation_en_de(self):
        self._test_repl_generate_batch()

    def test_auto_config(self):
        config = AutoConfig.from_pretrained(self.model_name)
        self.assertIsInstance(config, MarianConfig)


class TestMarian_EN_FR(MarianIntegrationTest):
    src = "en"
    tgt = "fr"
    src_text = [
        "I am a small frog.",
        "Now I can forget the 100 words of german that I know.",
    ]
    expected_text = [
        "Je suis une petite grenouille.",
        "Maintenant, je peux oublier les 100 mots d'allemand que je connais.",
    ]

    @slow
    def test_batch_generation_en_fr(self):
        self._test_repl_generate_batch()


class TestMarian_FR_EN(MarianIntegrationTest):
    src = "fr"
    tgt = "en"
    src_text = [
        "Donnez moi le micro.",
        "Tom et Mary étaient assis à une table.",  # Accents
    ]
    expected_text = [
        "Give me the microphone.",
        "Tom and Mary were sitting at a table.",
    ]

    @slow
    def test_batch_generation_fr_en(self):
        self._test_repl_generate_batch()
