# coding=utf-8
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
"""Tests for the Wav2Vec2Phoneme tokenizer."""
import json
import os
import unittest
from typing import Tuple

from transformers import Wav2Vec2PhonemeCTCTokenizer
from transformers.models.wav2vec2.tokenization_wav2vec2 import VOCAB_FILES_NAMES
from transformers.models.wav2vec2_phoneme.tokenization_wav2vec2_phoneme import Wav2Vec2PhonemeCTCTokenizerOutput
from transformers.testing_utils import require_phonemizer

from ...test_tokenization_common import TokenizerTesterMixin


@require_phonemizer
class Wav2Vec2PhonemeCTCTokenizerTest(TokenizerTesterMixin, unittest.TestCase):
    tokenizer_class = Wav2Vec2PhonemeCTCTokenizer
    test_rust_tokenizer = False

    def setUp(self):
        super().setUp()

        vocab = (
            "<s> <pad> </s> <unk> n s t ə l a i k d m ɛ ɾ e ɪ p o ɐ z ð f j v b ɹ ʁ ʊ iː r w ʌ u ɡ æ aɪ ʃ h ɔ ɑː "
            "ŋ ɚ eɪ β uː y ɑ̃ oʊ ᵻ eː θ aʊ ts oː ɔ̃ ɣ ɜ ɑ dʒ əl x ɜː ç ʒ tʃ ɔː ɑːɹ ɛ̃ ʎ ɔːɹ ʋ aː ɕ œ ø oːɹ ɲ yː "
            "ʔ iə i5 s. tɕ ?? nʲ ɛː œ̃ ɭ ɔø ʑ tʲ ɨ ɛɹ ts. rʲ ɪɹ ɭʲ i.5 ɔɪ q sʲ u5 ʊɹ iɜ a5 iɛ5 øː ʕ ja əɜ th ɑ5 "
            "oɪ dʲ ə5 tɕh ts.h mʲ ɯ dʑ vʲ e̞ tʃʲ ei5 o5 onɡ5 ɑu5 iɑ5 ai5 aɪɚ kh ə1 ʐ i2 ʉ ħ t[ aɪə ʲ ju ə2 u2 oɜ "
            "pː iɛɜ ou5 y5 uɜ tː uo5 d[ uoɜ tsh ɑɜ ɵ i̪5 uei5 ɟ aɜ ɑɨ i.ɜ eʊ o2 ɐ̃ ä pʲ kʲ n̩ ɒ ph ɑu2 uɨ əɪ ɫ ɬ "
            "yɜ bʲ ɑ2 s̪ aiɜ χ ɐ̃ʊ̃ 1 ə4 yæɜ a2 ɨː t̪ iouɜ ũ onɡɜ aɨ iɛ2 ɔɨ ɑuɜ o̞ ei2 iou2 c kː y2 ɖ oe dˤ yɛɜ "
            'əʊ S ɡʲ onɡ2 u" eiɜ ʈ ɯᵝ iou5 dZ r̝̊ i.2 tS s^ ʝ yə5 iɑɜ uə5 pf ɨu iɑ2 ou2 ər2 fʲ ai2 r̝ uəɜ ɳ əɨ '
            "ua5 uɪ ɽ bː yu5 uo2 yɛ5 l̩ ɻ ərɜ ʂ i̪2 ouɜ uaɜ a. a.ː yæ5 dː r̩ ee ɪu ər5 i̪ ɜ æi u: i.ː t^ o1 ɪ^ "
            "ai ueiɜ æː ɛɪ eə i. ɴ ie ua2 ɑ1 o4 tʃː o: ɑ: u1 N i̪1 au yæ2 u. qː yəɜ y: kʰ tʃʰ iʊ sx õ uo tʰ "
            "uai5 bʰ u.ː uə2 ʊə d^ s̪ː yiɜ dʰ r. oe: i1 ɟː yu2 nʲʲ i̪4 uei2 tsʲ ɸ ĩ ɑ4 t̪ː eɑ u4 e: tsː ʈʰ ɡʰ "
            "ɯɯ dʒʲ ʂʲ X ɵː uaiɜ tɕʲ ã t^ː ẽː yɛ2 cː i.1 ɛʊ dˤdˤ dʒː i4 ɡː yi ɕʲ ɟʰ pʰ dʑʲ yuɜ ua1 ua4 æiː ɐɐ "
            "ui iou1 ʊː a1 iou4 cʰ iɛ1 yə2 ɖʰ ẽ ʒʲ ää ər4 iːː ɪː iɑ1 ər1 œː øi ɪuː cʰcʰ əː1 iː1 ũ kʰː o̞o̞ xʲ "
            "ou1 iɛ4 e̞e̞ y1 dzː dʲʲ dʰː ɯᵝɯᵝ lː uo1 i.4 i: yɛ5ʲ a4"
        ).split(" ")
        vocab_tokens = dict(zip(vocab, range(len(vocab))))

        self.special_tokens_map = {"pad_token": "<pad>", "unk_token": "<unk>", "bos_token": "<s>", "eos_token": "</s>"}

        self.vocab_file = os.path.join(self.tmpdirname, VOCAB_FILES_NAMES["vocab_file"])
        with open(self.vocab_file, "w", encoding="utf-8") as fp:
            fp.write(json.dumps(vocab_tokens) + "\n")

    # overwrite since phonemes require specific creation
    def get_clean_sequence(self, tokenizer, with_prefix_space=False, max_length=20, min_length=5) -> Tuple[str, list]:
        toks = [(i, tokenizer.decode([i], clean_up_tokenization_spaces=False)) for i in range(len(tokenizer))]
        toks = list(filter(lambda t: [t[0]] == tokenizer.encode(t[1], do_phonemize=False), toks))
        if max_length is not None and len(toks) > max_length:
            toks = toks[:max_length]
        if min_length is not None and len(toks) < min_length and len(toks) > 0:
            while len(toks) < min_length:
                toks = toks + toks
        # toks_str = [t[1] for t in toks]
        toks_ids = [t[0] for t in toks]

        # Ensure consistency
        output_txt = tokenizer.decode(toks_ids, clean_up_tokenization_spaces=False)
        if " " not in output_txt and len(toks_ids) > 1:
            output_txt = (
                tokenizer.decode([toks_ids[0]], clean_up_tokenization_spaces=False)
                + " "
                + tokenizer.decode(toks_ids[1:], clean_up_tokenization_spaces=False)
            )
        if with_prefix_space:
            output_txt = " " + output_txt
        output_ids = tokenizer.encode(output_txt, add_special_tokens=False)
        return output_txt, output_ids

    def get_tokenizer(self, **kwargs):
        kwargs.update(self.special_tokens_map)
        return Wav2Vec2PhonemeCTCTokenizer.from_pretrained(self.tmpdirname, **kwargs)

    def test_tokenizer_add_new_tokens(self):
        tokenizer = self.tokenizer_class.from_pretrained("facebook/wav2vec2-lv-60-espeak-cv-ft")

        # check adding a single token
        tokenizer.add_tokens("xxx")
        token_ids = tokenizer("m xxx ɪ", do_phonemize=False).input_ids
        self.assertEqual(token_ids, [13, 392, 17])  # xxx should be last token

        tokenizer.add_tokens(["aaa", "bbb", "ccc"])
        token_ids = tokenizer("m aaa ɪ ccc", do_phonemize=False).input_ids
        self.assertEqual(token_ids, [13, 393, 17, 395])  # aaa and ccc should be after xxx and 2 after aaa

        token_ids = tokenizer("maɪ c", do_phonemize=False).input_ids
        self.assertEqual(token_ids, [3, 200])  # mai should be <unk> (=3)

    def test_phonemize(self):
        tokenizer = self.tokenizer_class.from_pretrained("facebook/wav2vec2-lv-60-espeak-cv-ft")

        input_text = "Hello how are you"
        phonemes = tokenizer.phonemize(input_text, phonemizer_lang="en-us")
        self.assertEqual(phonemes, "h ə l oʊ h aʊ ɑːɹ j uː")

    def test_encode(self):
        tokenizer = self.tokenizer_class.from_pretrained("facebook/wav2vec2-lv-60-espeak-cv-ft")

        input_text = "Hello how are you"
        phonemes = tokenizer.phonemize(input_text, phonemizer_lang="en-us")
        self.assertEqual(tokenizer(input_text).input_ids, tokenizer(phonemes, do_phonemize=False).input_ids)

    def test_encode_decode(self):
        tokenizer = self.tokenizer_class.from_pretrained("facebook/wav2vec2-lv-60-espeak-cv-ft")
        input_text = "Hello how are you"
        phonemes = tokenizer.phonemize(input_text, phonemizer_lang="en-us")

        phonemes_enc_dec = tokenizer.decode(tokenizer(input_text).input_ids)

        self.assertEqual(phonemes, phonemes_enc_dec)

    def test_decode(self):
        tokenizer = self.tokenizer_class.from_pretrained("facebook/wav2vec2-lv-60-espeak-cv-ft")

        sample_ids = [
            [11, 5, 15, tokenizer.pad_token_id, 15, 8, 98],
            [24, 22, 5, 24, 22, 5, 77],
        ]
        tokens = tokenizer.decode(sample_ids[0])
        batch_tokens = tokenizer.batch_decode(sample_ids)
        self.assertEqual(tokens, batch_tokens[0])
        self.assertEqual(batch_tokens, ["k s ɾ ɾ l ɭʲ", "j ð s j ð s oːɹ"])

    def test_phonemize_with_word_del(self):
        tokenizer = self.tokenizer_class.from_pretrained(
            "facebook/wav2vec2-lv-60-espeak-cv-ft", word_delimiter_token="|"
        )
        tokenizer.add_tokens("|")

        input_text = "Hello how are you"
        phonemes = tokenizer.phonemize(input_text, phonemizer_lang="en-us")
        self.assertEqual(phonemes, "h ə l oʊ | h aʊ | ɑːɹ | j uː |")

    def test_encode_with_del(self):
        tokenizer = self.tokenizer_class.from_pretrained(
            "facebook/wav2vec2-lv-60-espeak-cv-ft", word_delimiter_token="|"
        )
        tokenizer.add_tokens("|")

        input_text = "Hello how are you"
        phonemes = tokenizer.phonemize(input_text, phonemizer_lang="en-us")
        self.assertEqual(tokenizer(input_text).input_ids, tokenizer(phonemes, do_phonemize=False).input_ids)

    def test_decode_with_del(self):
        tokenizer = self.tokenizer_class.from_pretrained(
            "facebook/wav2vec2-lv-60-espeak-cv-ft", word_delimiter_token="|"
        )
        tokenizer.add_tokens("|")

        # fmt: off
        sample_ids = [
            [11, 5, 15, tokenizer.pad_token_id, tokenizer.word_delimiter_token_id, 15, 8, tokenizer.word_delimiter_token_id, 98],
            [tokenizer.word_delimiter_token_id, 24, 22, tokenizer.word_delimiter_token_id, 5, 24, 22, 5, 77],
        ]
        # fmt: on

        # decode with word_del_token filter
        tokens = tokenizer.decode(sample_ids[0])
        batch_tokens = tokenizer.batch_decode(sample_ids)
        self.assertEqual(tokens, batch_tokens[0])
        self.assertEqual(batch_tokens, ["k s ɾ ɾ l ɭʲ", "j ð s j ð s oːɹ"])

        # decode with no word_del_token filter
        tokens = tokenizer.decode(sample_ids[0], filter_word_delimiter_token=False)
        batch_tokens = tokenizer.batch_decode(sample_ids, filter_word_delimiter_token=False)
        self.assertEqual(tokens, batch_tokens[0])
        self.assertEqual(batch_tokens, ["k s ɾ | ɾ l | ɭʲ", "| j ð | s j ð s oːɹ"])

    def test_encode_decode_with_del(self):
        tokenizer = self.tokenizer_class.from_pretrained(
            "facebook/wav2vec2-lv-60-espeak-cv-ft", word_delimiter_token="|"
        )
        tokenizer.add_tokens("|")

        input_text = "Hello how are you"
        phonemes = tokenizer.phonemize(input_text, phonemizer_lang="en-us")

        phonemes_enc_dec = tokenizer.decode(tokenizer(input_text).input_ids, filter_word_delimiter_token=False)

        self.assertEqual(phonemes, phonemes_enc_dec)

    def test_encode_decode_with_del_filter(self):
        tokenizer = self.tokenizer_class.from_pretrained(
            "facebook/wav2vec2-lv-60-espeak-cv-ft", word_delimiter_token="|"
        )
        tokenizer.add_tokens("|")

        input_text = "Hello how are you"
        phonemes = tokenizer.phonemize(input_text, phonemizer_lang="en-us")

        phonemes_enc_dec = tokenizer.decode(tokenizer(input_text).input_ids, filter_word_delimiter_token=True)

        self.assertEqual(" ".join([p.strip() for p in phonemes.split(" |")]).strip(), phonemes_enc_dec)

    def test_change_phonemizer_lang(self):
        tokenizer = self.tokenizer_class.from_pretrained(
            "facebook/wav2vec2-lv-60-espeak-cv-ft", word_delimiter_token=None
        )
        input_text = "Hello how are you"

        input_ids_en = tokenizer(input_text, phonemizer_lang="en-us").input_ids
        input_ids_fr = tokenizer(input_text, phonemizer_lang="fr-fr").input_ids

        self.assertNotEqual(input_ids_en, input_ids_fr)

        text_en = tokenizer.decode(input_ids_en)
        text_fr = tokenizer.decode(input_ids_fr)

        self.assertEqual(text_en, "h ə l oʊ h aʊ ɑːɹ j uː")
        self.assertEqual(text_fr, "ɛ l o h aʊ a ʁ j u")

    def test_case_insensitive(self):
        tokenizer = self.tokenizer_class.from_pretrained("facebook/wav2vec2-lv-60-espeak-cv-ft")
        input_text_up = "Hello how Are you"
        input_text_low = "hello how are you"

        input_ids_up = tokenizer(input_text_up).input_ids
        input_ids_low = tokenizer(input_text_low).input_ids

        self.assertEqual(input_ids_up, input_ids_low)

    def test_tokenizer_decode_added_tokens(self):
        tokenizer = self.tokenizer_class.from_pretrained("facebook/wav2vec2-lv-60-espeak-cv-ft")
        tokenizer.add_tokens(["!", "?"])
        tokenizer.add_special_tokens({"cls_token": "$$$"})

        # fmt: off
        sample_ids = [
            [11, 5, 15, tokenizer.pad_token_id, 15, 8, 98, 392, 392, 393, 392, 392, 393, 394, 394],
            [24, 22, 5, 24, 22, 5, 77, tokenizer.pad_token_id, 394, 394],
        ]
        # fmt: on

        batch_tokens = tokenizer.batch_decode(sample_ids)
        self.assertEqual(batch_tokens, ["k s ɾ ɾ l ɭʲ!?!? $$$", "j ð s j ð s oːɹ $$$"])

    @staticmethod
    def get_from_offsets(offsets, key):
        retrieved_list = [d[key] for d in offsets]
        return retrieved_list

    def test_offsets(self):
        tokenizer = self.get_tokenizer(word_delimiter_token="|")
        tokenizer.add_tokens("|")

        # fmt: off
        # ksssɾɾ|ɾɾ<pad>ɾɾ|<pad>ɾlll|ɭʲ -> k s ɾ ɾ | ɾ l | ɭʲ"
        sample_ids = [11, 5, 5, 5, 15, 15, tokenizer.pad_token_id, 15, 15, tokenizer.word_delimiter_token_id, tokenizer.pad_token_id, 15, 8, 8, 8, tokenizer.word_delimiter_token_id, 98]
        # fmt: on

        outputs = tokenizer.decode(sample_ids, output_char_offsets=True, filter_word_delimiter_token=False)
        # check Wav2Vec2CTCTokenizerOutput keys for char
        self.assertEqual(len(outputs.keys()), 2)
        self.assertTrue("text" in outputs)
        self.assertTrue("char_offsets" in outputs)
        self.assertTrue(isinstance(outputs, Wav2Vec2PhonemeCTCTokenizerOutput))

        # check that order of chars is correct and identical for both outputs
        self.assertEqual(" ".join(self.get_from_offsets(outputs["char_offsets"], "char")), outputs.text)
        self.assertListEqual(
            self.get_from_offsets(outputs["char_offsets"], "char"), ["k", "s", "ɾ", "ɾ", "|", "ɾ", "l", "|", "ɭʲ"]
        )

        # check that offsets are actually correct for char
        # 0-1 is 11, 1-4 is 5, 4-6 is first 15, 6-7 is <pad> (thus not shown), 7-9 is second 15, 9-10 is word_delimiter_token,
        # 10-11 is <pad> (thus not shown), 11-12 is third 15, 12-15 is 8, 15-16 is word_delimiter_token, 16-17 is 98
        self.assertListEqual(
            self.get_from_offsets(outputs["char_offsets"], "start_offset"), [0, 1, 4, 7, 9, 11, 12, 15, 16]
        )
        self.assertListEqual(
            self.get_from_offsets(outputs["char_offsets"], "end_offset"), [1, 4, 6, 9, 10, 12, 15, 16, 17]
        )

    def test_offsets_batch(self):
        tokenizer = self.get_tokenizer(word_delimiter_token="|")

        def check_list_tuples_equal(outputs_batch, outputs_list):
            self.assertTrue(isinstance(outputs_batch, Wav2Vec2PhonemeCTCTokenizerOutput))
            self.assertTrue(isinstance(outputs_list[0], Wav2Vec2PhonemeCTCTokenizerOutput))

            # transform list to ModelOutput
            outputs_batch_2 = Wav2Vec2PhonemeCTCTokenizerOutput(
                {k: [d[k] for d in outputs_list] for k in outputs_list[0]}
            )

            self.assertListEqual(outputs_batch["text"], outputs_batch_2["text"])

            def recursive_check(list_or_dict_1, list_or_dict_2):
                if isinstance(list_or_dict_1, list):
                    [recursive_check(l1, l2) for l1, l2 in zip(list_or_dict_1, list_or_dict_2)]
                self.assertEqual(list_or_dict_1, list_or_dict_2)

            if "char_offsets" in outputs_batch:
                recursive_check(outputs_batch["char_offsets"], outputs_batch_2["char_offsets"])

        # fmt: off
        sample_ids = [
            [11, 5, 15, tokenizer.pad_token_id, 15, 4, 8, 98, 32, 32, 32, 32, 4, 33, tokenizer.word_delimiter_token_id, 32, 32, 33, 34, 34],
            [24, 22, 5, tokenizer.word_delimiter_token_id, tokenizer.word_delimiter_token_id, 24, 22, 22, 22, 4, 5, 77, tokenizer.pad_token_id, 22, 22, 4, 34, 34, 34, 34],
        ]
        # fmt: on

        # We assume that `decode` works as expected. All we will check now is
        # the output type is correct and the output is identical to `decode`

        # char
        outputs_char_batch = tokenizer.batch_decode(sample_ids, output_char_offsets=True)
        outputs_char = [tokenizer.decode(ids, output_char_offsets=True) for ids in sample_ids]
        check_list_tuples_equal(outputs_char_batch, outputs_char)

    @unittest.skip("Wav2Vec2PhonemeTokenizer always lower cases letters to correctly map to phonemes")
    def test_added_tokens_do_lower_case(self):
        pass

    @unittest.skip("Wav2Vec2PhonemeTokenizer always puts spaces between phonemes")
    def test_encode_decode_with_spaces(self):
        pass

    @unittest.skip("encodes to text to ids, but decodes ids to phonemes -> not possible to have internal consistency")
    def test_internal_consistency(self):
        pass

    @unittest.skip("Wav2Vec2PhonemeModel has no max model length => no testing")
    def test_pretrained_model_lists(self):
        pass

    # overwrite common
    def test_add_tokens_tokenizer(self):
        tokenizers = self.get_tokenizers(do_lower_case=False)
        for tokenizer in tokenizers:
            with self.subTest(f"{tokenizer.__class__.__name__}"):
                vocab_size = tokenizer.vocab_size
                all_size = len(tokenizer)

                self.assertNotEqual(vocab_size, 0)

                # We usually have added tokens from the start in tests because our vocab fixtures are
                # smaller than the original vocabs - let's not assert this
                # self.assertEqual(vocab_size, all_size)

                new_toks = ["aaaaa bbbbbb", "cccccccccdddddddd"]
                added_toks = tokenizer.add_tokens(new_toks)
                vocab_size_2 = tokenizer.vocab_size
                all_size_2 = len(tokenizer)

                self.assertNotEqual(vocab_size_2, 0)
                self.assertEqual(vocab_size, vocab_size_2)
                self.assertEqual(added_toks, len(new_toks))
                self.assertEqual(all_size_2, all_size + len(new_toks))

                tokens = tokenizer.encode("aaaaa bbbbbb low cccccccccdddddddd l", add_special_tokens=False)

                self.assertGreaterEqual(len(tokens), 4)
                self.assertGreater(tokens[0], tokenizer.vocab_size - 1)
                self.assertGreater(tokens[-3], tokenizer.vocab_size - 1)

                new_toks_2 = {"eos_token": ">>>>|||<||<<|<<", "pad_token": "<<<<<|||>|>>>>|>"}
                added_toks_2 = tokenizer.add_special_tokens(new_toks_2)
                vocab_size_3 = tokenizer.vocab_size
                all_size_3 = len(tokenizer)

                self.assertNotEqual(vocab_size_3, 0)
                self.assertEqual(vocab_size, vocab_size_3)
                self.assertEqual(added_toks_2, len(new_toks_2))
                self.assertEqual(all_size_3, all_size_2 + len(new_toks_2))

                tokens = tokenizer.encode(
                    ">>>>|||<||<<|<< aaaaabbbbbb low cccccccccdddddddd <<<<<|||>|>>>>|> l", add_special_tokens=False
                )

                self.assertGreaterEqual(len(tokens), 6)
                self.assertGreater(tokens[0], tokenizer.vocab_size - 1)
                self.assertGreater(tokens[0], tokens[1])
                self.assertGreater(tokens[-3], tokenizer.vocab_size - 1)
                self.assertGreater(tokens[-3], tokens[-4])
                self.assertEqual(tokens[0], tokenizer.eos_token_id)
                self.assertEqual(tokens[-3], tokenizer.pad_token_id)

    @unittest.skip("The tokenizer shouldn't be used to encode input IDs (except for labels), only to decode.")
    def test_tf_encode_plus_sent_to_model(self):
        pass

    @unittest.skip("The tokenizer shouldn't be used to encode input IDs (except for labels), only to decode.")
    def test_torch_encode_plus_sent_to_model(self):
        pass

    def test_convert_tokens_to_string_format(self):
        # The default common tokenizer tests assumes that the output of `convert_tokens_to_string` is a string which
        # is not the case for Wav2Vec2PhonemeCTCTokenizer.
        tokenizers = self.get_tokenizers(fast=True, do_lower_case=True)
        for tokenizer in tokenizers:
            with self.subTest(f"{tokenizer.__class__.__name__}"):
                tokens = ["ð", "ɪ", "s", "ɪ", "z", "ɐ", "t", "ɛ", "k", "s", "t"]
                output = tokenizer.convert_tokens_to_string(tokens)

                self.assertIsInstance(output["text"], str)
