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
from transformers.hf_api import HfApi

from .utils import require_torch, slow, torch_device


if is_torch_available():
    import torch
    from transformers import (
        AutoTokenizer,
        MarianConfig,
        AutoConfig,
        AutoModelWithLMHead,
        MarianTokenizer,
        MarianMTModel,
    )


class ModelManagementTests(unittest.TestCase):
    @slow
    def test_model_count(self):
        model_list = HfApi().model_list()
        expected_num_models = 1012
        actual_num_models = len([x for x in model_list if x.modelId.startswith("Helsinki-NLP")])
        self.assertEqual(expected_num_models, actual_num_models)


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
        cls.tokenizer: MarianTokenizer = AutoTokenizer.from_pretrained(cls.model_name)
        cls.eos_token_id = cls.tokenizer.eos_token_id
        return cls

    @cached_property
    def model(self) -> MarianMTModel:
        model = AutoModelWithLMHead.from_pretrained(self.model_name).to(torch_device)
        c = model.config
        self.assertListEqual(c.bad_words_ids, [[c.pad_token_id]])
        self.assertEqual(c.max_length, 512)
        self.assertEqual(c.decoder_start_token_id, c.pad_token_id)

        if torch_device == "cuda":
            return model.half()
        else:
            return model

    def _assert_generated_batch_equal_expected(self, **tokenizer_kwargs):
        generated_words = self.translate_src_text(**tokenizer_kwargs)
        self.assertListEqual(self.expected_text, generated_words)

    def translate_src_text(self, **tokenizer_kwargs):
        model_inputs: dict = self.tokenizer.prepare_translation_batch(src_texts=self.src_text, **tokenizer_kwargs).to(
            torch_device
        )
        self.assertEqual(self.model.device, model_inputs["input_ids"].device)
        generated_ids = self.model.generate(
            model_inputs["input_ids"], attention_mask=model_inputs["attention_mask"], num_beams=2
        )
        generated_words = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        return generated_words


class TestMarian_EN_DE_More(MarianIntegrationTest):
    @slow
    def test_forward(self):
        src, tgt = ["I am a small frog"], ["Ich bin ein kleiner Frosch."]
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
        self.tokenizer.batch_decode(max_indices)

    def test_tokenizer_equivalence(self):
        batch = self.tokenizer.prepare_translation_batch(["I am a small frog"]).to(torch_device)
        input_ids = batch["input_ids"][0]
        expected = [38, 121, 14, 697, 38848, 0]
        self.assertListEqual(expected, input_ids.tolist())

    def test_unk_support(self):
        t = self.tokenizer
        ids = t.prepare_translation_batch(["||"]).to(torch_device)["input_ids"][0].tolist()
        expected = [t.unk_token_id, t.unk_token_id, t.eos_token_id]
        self.assertEqual(expected, ids)

    def test_pad_not_split(self):
        input_ids_w_pad = self.tokenizer.prepare_translation_batch(["I am a small frog <pad>"])["input_ids"][0]
        expected_w_pad = [38, 121, 14, 697, 38848, self.tokenizer.pad_token_id, 0]  # pad
        self.assertListEqual(expected_w_pad, input_ids_w_pad.tolist())

    @slow
    def test_batch_generation_en_de(self):
        self._assert_generated_batch_equal_expected()

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
        self._assert_generated_batch_equal_expected()


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
        self._assert_generated_batch_equal_expected()


class TestMarian_RU_FR(MarianIntegrationTest):
    src = "ru"
    tgt = "fr"
    src_text = ["Он показал мне рукопись своей новой пьесы."]
    expected_text = ["Il me montre un manuscrit de sa nouvelle pièce."]

    @slow
    def test_batch_generation_ru_fr(self):
        self._assert_generated_batch_equal_expected()


class TestMarian_MT_EN(MarianIntegrationTest):
    src = "mt"
    tgt = "en"
    src_text = ["Il - Babiloniżi b'mod żbaljat ikkonkludew li l - Alla l - veru kien dgħajjef."]
    expected_text = ["The Babylonians wrongly concluded that the true God was weak."]

    @unittest.skip("")  # Known Issue: This model generates a string of .... at the end of the translation.
    def test_batch_generation_mt_en(self):
        self._assert_generated_batch_equal_expected()


class TestMarian_DE_Multi(MarianIntegrationTest):
    src = "de"
    tgt = "ch_group"
    src_text = ["Er aber sprach: Das ist die Gottlosigkeit."]

    @slow
    def test_translation_de_multi_does_not_error(self):
        self.translate_src_text()

    @unittest.skip("")  # "Language codes are not yet supported."
    def test_batch_generation_de_multi_tgt(self):
        self._assert_generated_batch_equal_expected()

    @unittest.skip("")  # "Language codes are not yet supported."
    def test_lang_code(self):
        t = "Er aber sprach"
        zh_code = self.code
        tok_fn = self.tokenizer.prepare_translation_batch
        pass_code = tok_fn(src_texts=[t], tgt_lang_code=zh_code)["input_ids"][0]
        preprocess_with_code = tok_fn(src_texts=[zh_code + "  " + t])["input_ids"][0]
        self.assertListEqual(pass_code.tolist(), preprocess_with_code.tolist())
        for code in self.tokenizer.supported_language_codes:
            self.assertIn(code, self.tokenizer.encoder)
        pass_only_code = tok_fn(src_texts=[""], tgt_lang_code=zh_code)["input_ids"][0].tolist()
        self.assertListEqual(pass_only_code, [self.tokenizer.encoder[zh_code], self.tokenizer.eos_token_id])
