# coding=utf-8
# Copyright 2020 Huggingface
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

import tempfile
import unittest
from pathlib import Path
from shutil import copyfile

from transformers import BatchEncoding, MarianTokenizer
from transformers.testing_utils import get_tests_dir, require_sentencepiece, slow
from transformers.utils import is_sentencepiece_available, is_tf_available, is_torch_available


if is_sentencepiece_available():
    from transformers.models.marian.tokenization_marian import VOCAB_FILES_NAMES, save_json

from ...test_tokenization_common import TokenizerTesterMixin


SAMPLE_SP = get_tests_dir("fixtures/test_sentencepiece.model")

mock_tokenizer_config = {"target_lang": "fi", "source_lang": "en"}
zh_code = ">>zh<<"
ORG_NAME = "Helsinki-NLP/"

if is_torch_available():
    FRAMEWORK = "pt"
elif is_tf_available():
    FRAMEWORK = "tf"
else:
    FRAMEWORK = "jax"


@require_sentencepiece
class MarianTokenizationTest(TokenizerTesterMixin, unittest.TestCase):
    tokenizer_class = MarianTokenizer
    test_rust_tokenizer = False
    test_sentencepiece = True

    def setUp(self):
        super().setUp()
        vocab = ["</s>", "<unk>", "▁This", "▁is", "▁a", "▁t", "est", "\u0120", "<pad>"]
        vocab_tokens = dict(zip(vocab, range(len(vocab))))
        save_dir = Path(self.tmpdirname)
        save_json(vocab_tokens, save_dir / VOCAB_FILES_NAMES["vocab"])
        save_json(mock_tokenizer_config, save_dir / VOCAB_FILES_NAMES["tokenizer_config_file"])
        if not (save_dir / VOCAB_FILES_NAMES["source_spm"]).exists():
            copyfile(SAMPLE_SP, save_dir / VOCAB_FILES_NAMES["source_spm"])
            copyfile(SAMPLE_SP, save_dir / VOCAB_FILES_NAMES["target_spm"])

        tokenizer = MarianTokenizer.from_pretrained(self.tmpdirname)
        tokenizer.save_pretrained(self.tmpdirname)

    def get_tokenizer(self, **kwargs) -> MarianTokenizer:
        return MarianTokenizer.from_pretrained(self.tmpdirname, **kwargs)

    def get_input_output_texts(self, tokenizer):
        return (
            "This is a test",
            "This is a test",
        )

    def test_convert_token_and_id(self):
        """Test ``_convert_token_to_id`` and ``_convert_id_to_token``."""
        token = "</s>"
        token_id = 0

        self.assertEqual(self.get_tokenizer()._convert_token_to_id(token), token_id)
        self.assertEqual(self.get_tokenizer()._convert_id_to_token(token_id), token)

    def test_get_vocab(self):
        vocab_keys = list(self.get_tokenizer().get_vocab().keys())

        self.assertEqual(vocab_keys[0], "</s>")
        self.assertEqual(vocab_keys[1], "<unk>")
        self.assertEqual(vocab_keys[-1], "<pad>")
        self.assertEqual(len(vocab_keys), 9)

    def test_vocab_size(self):
        self.assertEqual(self.get_tokenizer().vocab_size, 9)

    def test_tokenizer_equivalence_en_de(self):
        en_de_tokenizer = MarianTokenizer.from_pretrained(f"{ORG_NAME}opus-mt-en-de")
        batch = en_de_tokenizer(["I am a small frog"], return_tensors=None)
        self.assertIsInstance(batch, BatchEncoding)
        expected = [38, 121, 14, 697, 38848, 0]
        self.assertListEqual(expected, batch.input_ids[0])

        save_dir = tempfile.mkdtemp()
        en_de_tokenizer.save_pretrained(save_dir)
        contents = [x.name for x in Path(save_dir).glob("*")]
        self.assertIn("source.spm", contents)
        MarianTokenizer.from_pretrained(save_dir)

    def test_outputs_not_longer_than_maxlen(self):
        tok = self.get_tokenizer()

        batch = tok(
            ["I am a small frog" * 1000, "I am a small frog"], padding=True, truncation=True, return_tensors=FRAMEWORK
        )
        self.assertIsInstance(batch, BatchEncoding)
        self.assertEqual(batch.input_ids.shape, (2, 512))

    def test_outputs_can_be_shorter(self):
        tok = self.get_tokenizer()
        batch_smaller = tok(["I am a tiny frog", "I am a small frog"], padding=True, return_tensors=FRAMEWORK)
        self.assertIsInstance(batch_smaller, BatchEncoding)
        self.assertEqual(batch_smaller.input_ids.shape, (2, 10))

    @slow
    def test_tokenizer_integration(self):
        # fmt: off
        expected_encoding = {'input_ids': [[43495, 462, 20, 42164, 1369, 52, 464, 132, 1703, 492, 13, 7491, 38999, 6, 8, 464, 132, 1703, 492, 13, 4669, 37867, 13, 7525, 27, 1593, 988, 13, 33972, 7029, 6, 20, 8251, 383, 2, 270, 5866, 3788, 2, 2353, 8251, 12338, 2, 13958, 387, 2, 3629, 6953, 188, 2900, 2, 13958, 8011, 11501, 23, 8460, 4073, 34009, 20, 435, 11439, 27, 8, 8460, 4073, 6004, 20, 9988, 375, 27, 33, 266, 1945, 1076, 1350, 37867, 3288, 5, 577, 1076, 4374, 8, 5082, 5, 26453, 257, 556, 403, 2, 242, 132, 383, 316, 492, 8, 10767, 6, 316, 304, 4239, 3, 0], [148, 15722, 19, 1839, 12, 1350, 13, 22327, 5082, 5418, 47567, 35938, 59, 318, 19552, 108, 2183, 54, 14976, 4835, 32, 547, 1114, 8, 315, 2417, 5, 92, 19088, 3, 0, 58100, 58100, 58100, 58100, 58100, 58100, 58100, 58100, 58100, 58100, 58100, 58100, 58100, 58100, 58100, 58100, 58100, 58100, 58100, 58100, 58100, 58100, 58100, 58100, 58100, 58100, 58100, 58100, 58100, 58100, 58100, 58100, 58100, 58100, 58100, 58100, 58100, 58100, 58100, 58100, 58100, 58100, 58100, 58100, 58100, 58100, 58100, 58100, 58100, 58100, 58100, 58100, 58100, 58100, 58100, 58100, 58100, 58100, 58100, 58100, 58100, 58100, 58100, 58100, 58100, 58100, 58100, 58100, 58100, 58100, 58100], [36, 6395, 12570, 39147, 11597, 6, 266, 4, 45405, 7296, 3, 0, 58100, 58100, 58100, 58100, 58100, 58100, 58100, 58100, 58100, 58100, 58100, 58100, 58100, 58100, 58100, 58100, 58100, 58100, 58100, 58100, 58100, 58100, 58100, 58100, 58100, 58100, 58100, 58100, 58100, 58100, 58100, 58100, 58100, 58100, 58100, 58100, 58100, 58100, 58100, 58100, 58100, 58100, 58100, 58100, 58100, 58100, 58100, 58100, 58100, 58100, 58100, 58100, 58100, 58100, 58100, 58100, 58100, 58100, 58100, 58100, 58100, 58100, 58100, 58100, 58100, 58100, 58100, 58100, 58100, 58100, 58100, 58100, 58100, 58100, 58100, 58100, 58100, 58100, 58100, 58100, 58100, 58100, 58100, 58100, 58100, 58100, 58100, 58100, 58100, 58100]], 'attention_mask': [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]}  # noqa: E501
        # fmt: on

        self.tokenizer_integration_test_util(
            expected_encoding=expected_encoding,
            model_name="Helsinki-NLP/opus-mt-en-de",
            revision="1a8c2263da11e68e50938f97e10cd57820bd504c",
            decode_kwargs={"use_source_tokenizer": True},
        )

    def test_tokenizer_integration_seperate_vocabs(self):
        tokenizer = MarianTokenizer.from_pretrained("hf-internal-testing/test-marian-two-vocabs")

        source_text = "Tämä on testi"
        target_text = "This is a test"

        expected_src_ids = [76, 7, 2047, 2]
        expected_target_ids = [69, 12, 11, 940, 2]

        src_ids = tokenizer(source_text).input_ids
        self.assertListEqual(src_ids, expected_src_ids)

        target_ids = tokenizer(text_target=target_text).input_ids
        self.assertListEqual(target_ids, expected_target_ids)

        decoded = tokenizer.decode(target_ids, skip_special_tokens=True)
        self.assertEqual(decoded, target_text)
