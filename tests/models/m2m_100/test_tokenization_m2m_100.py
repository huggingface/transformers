# Copyright 2021 The HuggingFace Team. All rights reserved.
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

from transformers import M2M100Tokenizer, is_torch_available
from transformers.testing_utils import (
    get_tests_dir,
    nested_simplify,
    require_sentencepiece,
    require_tokenizers,
    require_torch,
    slow,
)
from transformers.utils import is_sentencepiece_available


if is_sentencepiece_available():
    from transformers.models.m2m_100.tokenization_m2m_100 import VOCAB_FILES_NAMES, save_json

from ...test_tokenization_common import TokenizerTesterMixin


if is_sentencepiece_available():
    SAMPLE_SP = get_tests_dir("fixtures/test_sentencepiece.model")


if is_torch_available():
    from transformers.models.m2m_100.modeling_m2m_100 import shift_tokens_right

EN_CODE = 128022
FR_CODE = 128028


@require_sentencepiece
class M2M100TokenizationTest(TokenizerTesterMixin, unittest.TestCase):
    tokenizer_class = M2M100Tokenizer
    test_rust_tokenizer = False
    test_seq2seq = False
    test_sentencepiece = True

    def setUp(self):
        super().setUp()

        vocab = ["</s>", "<unk>", "▁This", "▁is", "▁a", "▁t", "est", "\u0120", "<pad>"]
        vocab_tokens = dict(zip(vocab, range(len(vocab))))
        save_dir = Path(self.tmpdirname)
        save_json(vocab_tokens, save_dir / VOCAB_FILES_NAMES["vocab_file"])
        if not (save_dir / VOCAB_FILES_NAMES["spm_file"]).exists():
            copyfile(SAMPLE_SP, save_dir / VOCAB_FILES_NAMES["spm_file"])

        tokenizer = M2M100Tokenizer.from_pretrained(self.tmpdirname)
        tokenizer.save_pretrained(self.tmpdirname)

    def get_tokenizer(self, **kwargs):
        return M2M100Tokenizer.from_pretrained(self.tmpdirname, **kwargs)

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
        tokenizer = self.get_tokenizer()
        vocab_keys = list(tokenizer.get_vocab().keys())

        self.assertEqual(vocab_keys[0], "</s>")
        self.assertEqual(vocab_keys[1], "<unk>")
        self.assertEqual(vocab_keys[-1], "<s>")
        self.assertEqual(len(vocab_keys), tokenizer.vocab_size + len(tokenizer.get_added_vocab()))

    @unittest.skip("Skip this test while all models are still to be uploaded.")
    def test_pretrained_model_lists(self):
        pass

    def test_full_tokenizer(self):
        tokenizer = self.get_tokenizer()

        tokens = tokenizer.tokenize("This is a test")
        self.assertListEqual(tokens, ["▁This", "▁is", "▁a", "▁t", "est"])

        self.assertListEqual(
            tokenizer.convert_tokens_to_ids(tokens),
            [2, 3, 4, 5, 6],
        )

        back_tokens = tokenizer.convert_ids_to_tokens([2, 3, 4, 5, 6])
        self.assertListEqual(back_tokens, ["▁This", "▁is", "▁a", "▁t", "est"])

        text = tokenizer.convert_tokens_to_string(tokens)
        self.assertEqual(text, "This is a test")

    @slow
    def test_tokenizer_integration(self):
        # fmt: off
        expected_encoding = {'input_ids': [[128022, 110108, 397, 11, 38272, 2247, 124811, 285, 18105, 1586, 207, 7, 39534, 4428, 397, 1019, 18105, 1586, 207, 7, 41337, 16786, 241, 7, 20214, 17, 125690, 10398, 7, 44378, 58069, 68342, 7798, 7343, 11, 299, 33310, 4, 158, 37350, 94077, 4569, 299, 33310, 90, 4, 52840, 290, 4, 31270, 112, 299, 682, 4, 52840, 39953, 14079, 193, 52519, 90894, 17894, 120697, 11, 40445, 551, 17, 1019, 52519, 90894, 17756, 963, 11, 40445, 480, 17, 9792, 1120, 5173, 1393, 6240, 16786, 241, 120996, 28, 1245, 1393, 118240, 11123, 1019, 93612, 2691, 10618, 98058, 120409, 1928, 279, 4, 40683, 367, 178, 207, 1019, 103, 103121, 506, 65296, 5, 2], [128022, 21217, 367, 117, 125450, 128, 719, 7, 7308, 40, 93612, 12669, 1116, 16704, 71, 17785, 3699, 15592, 35, 144, 9584, 241, 11943, 713, 950, 799, 2247, 88427, 150, 149, 118813, 120706, 1019, 106906, 81518, 28, 1224, 22799, 397, 5, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [128022, 1658, 123311, 5155, 5578, 4722, 279, 14947, 2366, 1120, 1197, 14, 1348, 9232, 5, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]], 'attention_mask': [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]}  # noqa: E501
        # fmt: on

        self.tokenizer_integration_test_util(
            expected_encoding=expected_encoding,
            model_name="facebook/m2m100_418M",
            revision="c168bae485c864188cf9aa0e4108b0b6934dc91e",
        )


@require_torch
@require_sentencepiece
@require_tokenizers
class M2M100TokenizerIntegrationTest(unittest.TestCase):
    checkpoint_name = "facebook/m2m100_418M"
    src_text = [
        "In my opinion, there are two levels of response from the French government.",
        "NSA Affair Emphasizes Complete Lack of Debate on Intelligence",
    ]
    tgt_text = [
        "Selon moi, il y a deux niveaux de réponse de la part du gouvernement français.",
        "L'affaire NSA souligne l'absence totale de débat sur le renseignement",
    ]

    # fmt: off
    expected_src_tokens = [EN_CODE, 593, 1949, 115781, 4, 71586, 4234, 60633, 126233, 432, 123808, 15592, 1197, 117132, 120618, 5, 2]
    # fmt: on

    @classmethod
    def setUpClass(cls):
        cls.tokenizer: M2M100Tokenizer = M2M100Tokenizer.from_pretrained(
            cls.checkpoint_name, src_lang="en", tgt_lang="fr"
        )
        cls.pad_token_id = 1
        return cls

    def check_language_codes(self):
        self.assertEqual(self.tokenizer.get_lang_id("ar"), 128006)
        self.assertEqual(self.tokenizer.get_lang_id("en"), 128022)
        self.assertEqual(self.tokenizer.get_lang_id("ro"), 128076)
        self.assertEqual(self.tokenizer.get_lang_id("mr"), 128063)

    def test_get_vocab(self):
        vocab = self.tokenizer.get_vocab()
        self.assertEqual(len(vocab), self.tokenizer.vocab_size)
        self.assertEqual(vocab["<unk>"], 3)
        self.assertIn(self.tokenizer.get_lang_token("en"), vocab)

    def test_tokenizer_batch_encode_plus(self):
        self.tokenizer.src_lang = "en"
        ids = self.tokenizer.batch_encode_plus(self.src_text).input_ids[0]
        self.assertListEqual(self.expected_src_tokens, ids)

    def test_tokenizer_decode_ignores_language_codes(self):
        self.assertIn(FR_CODE, self.tokenizer.all_special_ids)
        # fmt: off
        generated_ids = [FR_CODE, 5364, 82, 8642, 4, 294, 47, 8, 14028, 136, 3286, 9706, 6, 90797, 6, 144012, 162, 88128, 30061, 5, 2]
        # fmt: on
        result = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        expected_french = self.tokenizer.decode(generated_ids[1:], skip_special_tokens=True)
        self.assertEqual(result, expected_french)
        self.assertNotIn(self.tokenizer.eos_token, result)

    def test_special_tokens_unaffacted_by_save_load(self):
        tmpdirname = tempfile.mkdtemp()
        original_special_tokens = self.tokenizer.lang_token_to_id
        self.tokenizer.save_pretrained(tmpdirname)
        new_tok = M2M100Tokenizer.from_pretrained(tmpdirname)
        self.assertDictEqual(new_tok.lang_token_to_id, original_special_tokens)

    @require_torch
    def test_batch_fairseq_parity(self):
        self.tokenizer.src_lang = "en"
        self.tokenizer.tgt_lang = "fr"

        batch = self.tokenizer(self.src_text, text_target=self.tgt_text, padding=True, return_tensors="pt")

        batch["decoder_input_ids"] = shift_tokens_right(
            batch["labels"], self.tokenizer.pad_token_id, self.tokenizer.eos_token_id
        )

        for k in batch:
            batch[k] = batch[k].tolist()
        # batch = {k: v.tolist() for k,v in batch.items()}
        # fairseq batch: https://gist.github.com/sshleifer/cba08bc2109361a74ac3760a7e30e4f4
        # batch.decoder_inputs_ids[0][0] ==
        assert batch.input_ids[1][0] == EN_CODE
        assert batch.input_ids[1][-1] == 2
        assert batch.labels[1][0] == FR_CODE
        assert batch.labels[1][-1] == 2
        assert batch.decoder_input_ids[1][:2] == [2, FR_CODE]

    @require_torch
    def test_src_lang_setter(self):
        self.tokenizer.src_lang = "mr"
        self.assertListEqual(self.tokenizer.prefix_tokens, [self.tokenizer.get_lang_id("mr")])
        self.assertListEqual(self.tokenizer.suffix_tokens, [self.tokenizer.eos_token_id])

        self.tokenizer.src_lang = "zh"
        self.assertListEqual(self.tokenizer.prefix_tokens, [self.tokenizer.get_lang_id("zh")])
        self.assertListEqual(self.tokenizer.suffix_tokens, [self.tokenizer.eos_token_id])

    @require_torch
    def test_tokenizer_target_mode(self):
        self.tokenizer.tgt_lang = "mr"
        self.tokenizer._switch_to_target_mode()
        self.assertListEqual(self.tokenizer.prefix_tokens, [self.tokenizer.get_lang_id("mr")])
        self.assertListEqual(self.tokenizer.suffix_tokens, [self.tokenizer.eos_token_id])
        self.tokenizer._switch_to_input_mode()
        self.assertListEqual(self.tokenizer.prefix_tokens, [self.tokenizer.get_lang_id(self.tokenizer.src_lang)])

        self.tokenizer.tgt_lang = "zh"
        self.tokenizer._switch_to_target_mode()
        self.assertListEqual(self.tokenizer.prefix_tokens, [self.tokenizer.get_lang_id("zh")])
        self.assertListEqual(self.tokenizer.suffix_tokens, [self.tokenizer.eos_token_id])
        self.tokenizer._switch_to_input_mode()
        self.assertListEqual(self.tokenizer.prefix_tokens, [self.tokenizer.get_lang_id(self.tokenizer.src_lang)])

    @require_torch
    def test_tokenizer_translation(self):
        inputs = self.tokenizer._build_translation_inputs("A test", return_tensors="pt", src_lang="en", tgt_lang="ar")

        self.assertEqual(
            nested_simplify(inputs),
            {
                # en_XX, A, test, EOS
                "input_ids": [[128022, 58, 4183, 2]],
                "attention_mask": [[1, 1, 1, 1]],
                # ar_AR
                "forced_bos_token_id": 128006,
            },
        )
