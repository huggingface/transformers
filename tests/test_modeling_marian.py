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

from transformers import AutoConfig, AutoTokenizer, MarianConfig, MarianTokenizer, is_torch_available
from transformers.file_utils import cached_property
from transformers.hf_api import HfApi
from transformers.testing_utils import require_sentencepiece, require_tokenizers, require_torch, slow, torch_device


if is_torch_available():
    import torch

    from transformers import AutoModelWithLMHead, MarianMTModel
    from transformers.convert_marian_to_pytorch import (
        ORG_NAME,
        convert_hf_name_to_opus_name,
        convert_opus_name_to_hf_name,
    )
    from transformers.modeling_bart import shift_tokens_right
    from transformers.pipelines import TranslationPipeline


class ModelManagementTests(unittest.TestCase):
    @slow
    def test_model_names(self):
        model_list = HfApi().model_list()
        model_ids = [x.modelId for x in model_list if x.modelId.startswith(ORG_NAME)]
        bad_model_ids = [mid for mid in model_ids if "+" in model_ids]
        self.assertListEqual([], bad_model_ids)
        self.assertGreater(len(model_ids), 500)


@require_torch
@require_sentencepiece
@require_tokenizers
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
        return cls

    @cached_property
    def tokenizer(self) -> MarianTokenizer:
        return AutoTokenizer.from_pretrained(self.model_name)

    @property
    def eos_token_id(self) -> int:
        return self.tokenizer.eos_token_id

    @cached_property
    def model(self):
        model: MarianMTModel = AutoModelWithLMHead.from_pretrained(self.model_name).to(torch_device)
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
        model_inputs = self.tokenizer.prepare_seq2seq_batch(src_texts=self.src_text, **tokenizer_kwargs).to(
            torch_device
        )
        self.assertEqual(self.model.device, model_inputs.input_ids.device)
        generated_ids = self.model.generate(
            model_inputs.input_ids, attention_mask=model_inputs.attention_mask, num_beams=2
        )
        generated_words = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        return generated_words


@require_sentencepiece
@require_tokenizers
class TestMarian_EN_DE_More(MarianIntegrationTest):
    @slow
    def test_forward(self):
        src, tgt = ["I am a small frog"], ["Ich bin ein kleiner Frosch."]
        expected_ids = [38, 121, 14, 697, 38848, 0]

        model_inputs: dict = self.tokenizer.prepare_seq2seq_batch(src, tgt_texts=tgt).to(torch_device)

        self.assertListEqual(expected_ids, model_inputs.input_ids[0].tolist())

        desired_keys = {
            "input_ids",
            "attention_mask",
            "labels",
        }
        self.assertSetEqual(desired_keys, set(model_inputs.keys()))
        model_inputs["decoder_input_ids"] = shift_tokens_right(model_inputs.labels, self.tokenizer.pad_token_id)
        model_inputs["return_dict"] = True
        model_inputs["use_cache"] = False
        with torch.no_grad():
            outputs = self.model(**model_inputs)
        max_indices = outputs.logits.argmax(-1)
        self.tokenizer.batch_decode(max_indices)

    def test_unk_support(self):
        t = self.tokenizer
        ids = t.prepare_seq2seq_batch(["||"]).to(torch_device).input_ids[0].tolist()
        expected = [t.unk_token_id, t.unk_token_id, t.eos_token_id]
        self.assertEqual(expected, ids)

    def test_pad_not_split(self):
        input_ids_w_pad = self.tokenizer.prepare_seq2seq_batch(["I am a small frog <pad>"]).input_ids[0].tolist()
        expected_w_pad = [38, 121, 14, 697, 38848, self.tokenizer.pad_token_id, 0]  # pad
        self.assertListEqual(expected_w_pad, input_ids_w_pad)

    @slow
    def test_batch_generation_en_de(self):
        self._assert_generated_batch_equal_expected()

    def test_auto_config(self):
        config = AutoConfig.from_pretrained(self.model_name)
        self.assertIsInstance(config, MarianConfig)


@require_sentencepiece
@require_tokenizers
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


@require_sentencepiece
@require_tokenizers
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


@require_sentencepiece
@require_tokenizers
class TestMarian_RU_FR(MarianIntegrationTest):
    src = "ru"
    tgt = "fr"
    src_text = ["Он показал мне рукопись своей новой пьесы."]
    expected_text = ["Il m'a montré le manuscrit de sa nouvelle pièce."]

    @slow
    def test_batch_generation_ru_fr(self):
        self._assert_generated_batch_equal_expected()


@require_sentencepiece
@require_tokenizers
class TestMarian_MT_EN(MarianIntegrationTest):
    src = "mt"
    tgt = "en"
    src_text = ["Billi messu b'mod ġentili, Ġesù fejjaq raġel li kien milqut bil - marda kerha tal - ġdiem."]
    expected_text = ["Touching gently, Jesus healed a man who was affected by the sad disease of leprosy."]

    @slow
    def test_batch_generation_mt_en(self):
        self._assert_generated_batch_equal_expected()


@require_sentencepiece
@require_tokenizers
class TestMarian_en_zh(MarianIntegrationTest):
    src = "en"
    tgt = "zh"
    src_text = ["My name is Wolfgang and I live in Berlin"]
    expected_text = ["我叫沃尔夫冈 我住在柏林"]

    @slow
    def test_batch_generation_eng_zho(self):
        self._assert_generated_batch_equal_expected()


@require_sentencepiece
@require_tokenizers
class TestMarian_en_ROMANCE(MarianIntegrationTest):
    """Multilingual on target side."""

    src = "en"
    tgt = "ROMANCE"
    src_text = [
        ">>fr<< Don't spend so much time watching TV.",
        ">>pt<< Your message has been sent.",
        ">>es<< He's two years older than me.",
    ]
    expected_text = [
        "Ne passez pas autant de temps à regarder la télé.",
        "A sua mensagem foi enviada.",
        "Es dos años más viejo que yo.",
    ]

    @slow
    def test_batch_generation_en_ROMANCE_multi(self):
        self._assert_generated_batch_equal_expected()

    def test_tokenizer_handles_empty(self):
        normalized = self.tokenizer.normalize("")
        self.assertIsInstance(normalized, str)
        with self.assertRaises(ValueError):
            self.tokenizer.prepare_seq2seq_batch([""])

    def test_pipeline(self):
        device = 0 if torch_device == "cuda" else -1
        pipeline = TranslationPipeline(self.model, self.tokenizer, framework="pt", device=device)
        output = pipeline(self.src_text)
        self.assertEqual(self.expected_text, [x["translation_text"] for x in output])


@require_torch
class TestConversionUtils(unittest.TestCase):
    def test_renaming_multilingual(self):
        old_names = [
            "opus-mt-cmn+cn+yue+ze_zh+zh_cn+zh_CN+zh_HK+zh_tw+zh_TW+zh_yue+zhs+zht+zh-fi",
            "opus-mt-cmn+cn-fi",  # no group
            "opus-mt-en-de",  # standard name
            "opus-mt-en-de",  # standard name
        ]
        expected = ["opus-mt-ZH-fi", "opus-mt-cmn_cn-fi", "opus-mt-en-de", "opus-mt-en-de"]
        self.assertListEqual(expected, [convert_opus_name_to_hf_name(x) for x in old_names])

    def test_undoing_renaming(self):
        hf_names = ["opus-mt-ZH-fi", "opus-mt-cmn_cn-fi", "opus-mt-en-de", "opus-mt-en-de"]
        converted_opus_names = [convert_hf_name_to_opus_name(x) for x in hf_names]
        expected_opus_names = [
            "cmn+cn+yue+ze_zh+zh_cn+zh_CN+zh_HK+zh_tw+zh_TW+zh_yue+zhs+zht+zh-fi",
            "cmn+cn-fi",
            "en-de",  # standard name
            "en-de",
        ]
        self.assertListEqual(expected_opus_names, converted_opus_names)
