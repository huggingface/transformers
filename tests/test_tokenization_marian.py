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


import os
import tempfile
import unittest
from pathlib import Path
from shutil import copyfile

from transformers import BatchEncoding, MarianTokenizer
from transformers.file_utils import is_sentencepiece_available, is_tf_available, is_torch_available
from transformers.testing_utils import require_sentencepiece


if is_sentencepiece_available():
    from transformers.models.marian.tokenization_marian import VOCAB_FILES_NAMES, save_json

from .test_tokenization_common import TokenizerTesterMixin


SAMPLE_SP = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures/test_sentencepiece.model")

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
