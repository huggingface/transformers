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

from transformers import AutoTokenizer, MarianTokenizer, TranslationPipeline, is_tf_available
from transformers.file_utils import cached_property
from transformers.testing_utils import require_sentencepiece, require_tf, require_tokenizers, slow


if is_tf_available():

    from transformers import TFAutoModelForSeq2SeqLM, TFMarianMTModel


class AbstractMarianIntegrationTest(unittest.TestCase):
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
        model: TFMarianMTModel = TFAutoModelForSeq2SeqLM.from_pretrained(self.model_name, from_pt=True)
        assert isinstance(model, TFMarianMTModel)
        c = model.config
        self.assertListEqual(c.bad_words_ids, [[c.pad_token_id]])
        self.assertEqual(c.max_length, 512)
        self.assertEqual(c.decoder_start_token_id, c.pad_token_id)
        return model

    def _assert_generated_batch_equal_expected(self, **tokenizer_kwargs):
        generated_words = self.translate_src_text(**tokenizer_kwargs)
        self.assertListEqual(self.expected_text, generated_words)

    def translate_src_text(self, **tokenizer_kwargs):
        model_inputs = self.tokenizer.prepare_seq2seq_batch(
            src_texts=self.src_text, **tokenizer_kwargs, return_tensors="tf"
        )
        generated_ids = self.model.generate(
            model_inputs.input_ids, attention_mask=model_inputs.attention_mask, num_beams=2
        )
        generated_words = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        return generated_words


@require_tf
@require_sentencepiece
@require_tokenizers
class TestMarian_en_zh(AbstractMarianIntegrationTest):
    src = "en"
    tgt = "zh"
    src_text = ["My name is Wolfgang and I live in Berlin"]
    expected_text = ["我叫沃尔夫冈 我住在柏林"]

    @slow
    def test_batch_generation_en_zh(self):
        self._assert_generated_batch_equal_expected()


@require_tf
@require_sentencepiece
@require_tokenizers
class TestMarian_en_ROMANCE(AbstractMarianIntegrationTest):
    """Multilingual on target side."""

    src = "en"
    tgt = "ROMANCE"
    src_text = [
        ">>fr<< Don't spend so much time watching TV.",
        ">>pt<< Your message has been sent.",
        ">>es<< He's two years older than me.",
    ]
    expected_text = [
        "Ne regardez pas tant de temps à la télé.",
        "A sua mensagem foi enviada.",
        "Tiene dos años más que yo.",
    ]

    @slow
    def test_batch_generation_en_ROMANCE_multi(self):
        self._assert_generated_batch_equal_expected()

    def test_tokenizer_handles_empty(self):
        normalized = self.tokenizer.normalize("")
        self.assertIsInstance(normalized, str)
        with self.assertRaises(ValueError):
            self.tokenizer.prepare_seq2seq_batch([""])

    @slow
    def test_pipeline(self):
        pipeline = TranslationPipeline(self.model, self.tokenizer, framework="tf")
        output = pipeline(self.src_text)
        self.assertEqual(self.expected_text, [x["translation_text"] for x in output])
