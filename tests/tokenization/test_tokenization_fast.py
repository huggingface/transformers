# Copyright 2019 HuggingFace Inc.
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

import concurrent.futures
import json
import os
import shutil
import tempfile
import unittest

from tokenizers import Tokenizer, decoders, pre_tokenizers, trainers
from tokenizers.models import BPE, WordLevel

from transformers import AutoTokenizer, PreTrainedTokenizerFast
from transformers.testing_utils import require_tokenizers


@require_tokenizers
class PreTrainedTokenizationFastTest(unittest.TestCase):
    rust_tokenizer_class = PreTrainedTokenizerFast
    from_pretrained_vocab_key = "tokenizer_file"

    @classmethod
    def setUpClass(cls):
        cls.tmpdirname = tempfile.mkdtemp()
        cls.model_paths = cls._create_test_tokenizers()
        cls.bytelevel_bpe_model_name = cls.model_paths[1]
        cls.tokenizers_list = [(cls.rust_tokenizer_class, path, {}) for path in cls.model_paths]

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmpdirname, ignore_errors=True)

    @classmethod
    def _create_test_tokenizers(cls):
        paths = []

        wordlevel_dir = os.path.join(cls.tmpdirname, "wordlevel_tokenizer")
        os.makedirs(wordlevel_dir, exist_ok=True)
        wl_vocab = {"[UNK]": 0, "[PAD]": 1, "hello": 2, "world": 3, "test": 4}
        wordlevel_tokenizer = Tokenizer(WordLevel(wl_vocab, unk_token="[UNK]"))
        wordlevel_tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
        fast_wl = PreTrainedTokenizerFast(
            tokenizer_object=wordlevel_tokenizer,
            unk_token="[UNK]",
            pad_token="[PAD]",
            cls_token="[CLS]",
            sep_token="[SEP]",
            mask_token="[MASK]",
        )
        fast_wl.save_pretrained(wordlevel_dir)
        paths.append(wordlevel_dir)

        bpe_dir = os.path.join(cls.tmpdirname, "bytelevel_bpe_tokenizer")
        os.makedirs(bpe_dir, exist_ok=True)
        bpe_tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
        trainer = trainers.BpeTrainer(
            special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"],
            vocab_size=100,
        )
        corpus = ["Hello world!", "Test the byte level BPE tokenizer.", "Tokenizer fast test."]
        bpe_tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel()
        bpe_tokenizer.train_from_iterator(corpus, trainer=trainer)
        bpe_tokenizer.decoder = decoders.ByteLevel()
        fast_bpe = PreTrainedTokenizerFast(
            tokenizer_object=bpe_tokenizer,
            unk_token="[UNK]",
            pad_token="[PAD]",
            cls_token="[CLS]",
            sep_token="[SEP]",
            mask_token="[MASK]",
        )
        fast_bpe.save_pretrained(bpe_dir)
        paths.append(bpe_dir)

        return paths

    @unittest.skip(
        "We disable this test for PreTrainedTokenizerFast because it is the only tokenizer that is not linked to any model"
    )
    def test_tokenizer_mismatch_warning(self):
        pass

    @unittest.skip(
        "We disable this test for PreTrainedTokenizerFast because it is the only tokenizer that is not linked to any model"
    )
    def test_encode_decode_with_spaces(self):
        pass

    @unittest.skip(
        "We disable this test for PreTrainedTokenizerFast because it is the only tokenizer that is not linked to any model"
    )
    def test_added_tokens_serialization(self):
        pass

    @unittest.skip(
        "We disable this test for PreTrainedTokenizerFast because it is the only tokenizer that is not linked to any model"
    )
    def test_additional_special_tokens_serialization(self):
        pass

    def test_training_new_tokenizer(self):
        tmpdirname_orig = self.tmpdirname
        # Here we want to test the 2 available tokenizers that use 2 different types of models: Unigram and WordLevel.
        for tokenizer, pretrained_name, kwargs in self.tokenizers_list:
            with self.subTest(f"{tokenizer.__class__.__name__} ({pretrained_name})"):
                try:
                    self.tmpdirname = tempfile.mkdtemp()
                    tokenizer = self.rust_tokenizer_class.from_pretrained(pretrained_name, **kwargs)

                    tokenizer.save_pretrained(self.tmpdirname)
                    reloaded = PreTrainedTokenizerFast.from_pretrained(self.tmpdirname)
                    self.assertEqual(reloaded.get_vocab(), tokenizer.get_vocab())
                finally:
                    # Even if the test fails, we must be sure that the folder is deleted and that the default tokenizer
                    # is restored
                    shutil.rmtree(self.tmpdirname)
                    self.tmpdirname = tmpdirname_orig

    def test_training_new_tokenizer_with_special_tokens_change(self):
        tmpdirname_orig = self.tmpdirname
        # Here we want to test the 2 available tokenizers that use 2 different types of models: Unigram and WordLevel.
        for tokenizer, pretrained_name, kwargs in self.tokenizers_list:
            with self.subTest(f"{tokenizer.__class__.__name__} ({pretrained_name})"):
                try:
                    self.tmpdirname = tempfile.mkdtemp()
                    tokenizer = self.rust_tokenizer_class.from_pretrained(pretrained_name, **kwargs)

                    tokenizer.add_special_tokens({"pad_token": "<pad>"})
                    tokenizer.save_pretrained(self.tmpdirname)
                    reloaded = PreTrainedTokenizerFast.from_pretrained(self.tmpdirname)
                    self.assertEqual(reloaded.pad_token, "<pad>")
                finally:
                    # Even if the test fails, we must be sure that the folder is deleted and that the default tokenizer
                    # is restored
                    shutil.rmtree(self.tmpdirname)
                    self.tmpdirname = tmpdirname_orig

    def test_training_new_tokenizer_with_bytelevel(self):
        tokenizer = self.rust_tokenizer_class.from_pretrained(self.bytelevel_bpe_model_name)

        toy_text_iterator = ("a" for _ in range(1000))
        new_tokenizer = tokenizer.train_new_from_iterator(text_iterator=toy_text_iterator, length=1000, vocab_size=50)

        encoding_ids = new_tokenizer.encode("a🤗")
        self.assertGreater(len(encoding_ids), 0)
        self.assertEqual(new_tokenizer.decode(encoding_ids), "a🤗")

    def test_init_from_tokenizers_model(self):
        from tokenizers import Tokenizer

        sentences = ["Hello, y'all!", "How are you 😁 ? There should not be any issue right?"]

        tokenizer = Tokenizer.from_pretrained("google-t5/t5-base")
        # Enable padding
        tokenizer.enable_padding(pad_id=0, pad_token="<pad>", length=512, pad_to_multiple_of=8)
        self.assertEqual(
            tokenizer.padding,
            {
                "length": 512,
                "pad_to_multiple_of": 8,
                "pad_id": 0,
                "pad_token": "<pad>",
                "pad_type_id": 0,
                "direction": "right",
            },
        )
        fast_tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer)
        tmpdirname = tempfile.mkdtemp()
        fast_tokenizer.save_pretrained(tmpdirname)
        fast_from_saved = PreTrainedTokenizerFast.from_pretrained(tmpdirname)
        for tok in [fast_tokenizer, fast_from_saved]:
            self.assertEqual(tok.pad_token_id, 0)
            self.assertEqual(tok.padding_side, "right")
            self.assertEqual(tok.pad_token, "<pad>")
            self.assertEqual(tok.init_kwargs["max_length"], 512)
            self.assertEqual(tok.init_kwargs["pad_to_multiple_of"], 8)
            self.assertEqual(tok(sentences, padding = True), {'input_ids': [[8774, 6, 3, 63, 31, 1748, 55, 1, 0, 0, 0, 0,0, 0, 0, 0],[ 571, 33, 25, 3, 2, 3, 58, 290, 225, 59, 36, 136, 962, 269, 58, 1]], 'token_type_ids': [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], 'attention_mask': [[1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]})  # fmt: skip

        tokenizer.enable_truncation(8, stride=0, strategy="longest_first", direction="right")
        self.assertEqual(
            tokenizer.truncation, {"max_length": 8, "stride": 0, "strategy": "longest_first", "direction": "right"}
        )
        fast_tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer)
        tmpdirname = tempfile.mkdtemp()
        fast_tokenizer.save_pretrained(tmpdirname)
        fast_from_saved = PreTrainedTokenizerFast.from_pretrained(tmpdirname)
        for tok in [fast_tokenizer, fast_from_saved]:
            self.assertEqual(tok.truncation_side, "right")
            self.assertEqual(tok.init_kwargs["truncation_strategy"], "longest_first")
            self.assertEqual(tok.init_kwargs["max_length"], 8)
            self.assertEqual(tok.init_kwargs["stride"], 0)
            # NOTE even if the model has a default max_length, it is not used...
            # thus tok(sentences, truncation = True) does nothing and does not warn either
            self.assertEqual(tok(sentences, truncation = True, max_length = 8), {'input_ids': [[8774, 6, 3, 63, 31, 1748, 55, 1],[ 571, 33, 25, 3, 2, 3, 58, 1]], 'token_type_ids': [[0, 0, 0, 0, 0, 0, 0, 0],[0, 0, 0, 0, 0, 0, 0, 0]], 'attention_mask': [[1, 1, 1, 1, 1, 1, 1, 1],[1, 1, 1, 1, 1, 1, 1, 1]]})  # fmt: skip

    def test_class_after_save_and_reload(self):
        model_id = self.model_paths[0]

        with tempfile.TemporaryDirectory() as temp_dir:
            tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
            self.assertIsInstance(tokenizer, PreTrainedTokenizerFast)

            tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False)
            self.assertIsInstance(tokenizer, PreTrainedTokenizerFast)

            tokenizer.save_pretrained(temp_dir)

            tokenizer = AutoTokenizer.from_pretrained(temp_dir, use_fast=False)
            self.assertIsInstance(tokenizer, PreTrainedTokenizerFast)

            tokenizer = AutoTokenizer.from_pretrained(temp_dir, use_fast=True)
            self.assertIsInstance(tokenizer, PreTrainedTokenizerFast)


@require_tokenizers
class TokenizerVersioningTest(unittest.TestCase):
    def test_local_versioning(self):
        tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-cased")
        json_tokenizer = json.loads(tokenizer._tokenizer.to_str())
        json_tokenizer["model"]["vocab"]["huggingface"] = len(tokenizer)

        with tempfile.TemporaryDirectory() as tmp_dir:
            # Hack to save this in the tokenizer_config.json
            tokenizer.init_kwargs["fast_tokenizer_files"] = ["tokenizer.4.0.0.json"]
            tokenizer.save_pretrained(tmp_dir)
            json.dump(json_tokenizer, open(os.path.join(tmp_dir, "tokenizer.4.0.0.json"), "w"))

            # This should pick the new tokenizer file as the version of Transformers is > 4.0.0
            new_tokenizer = AutoTokenizer.from_pretrained(tmp_dir)
            self.assertEqual(len(new_tokenizer), len(tokenizer) + 1)
            json_tokenizer = json.loads(new_tokenizer._tokenizer.to_str())
            self.assertIn("huggingface", json_tokenizer["model"]["vocab"])

            # Will need to be adjusted if we reach v42 and this test is still here.
            # Should pick the old tokenizer file as the version of Transformers is < 4.0.0
            shutil.move(os.path.join(tmp_dir, "tokenizer.4.0.0.json"), os.path.join(tmp_dir, "tokenizer.42.0.0.json"))
            tokenizer.init_kwargs["fast_tokenizer_files"] = ["tokenizer.42.0.0.json"]
            tokenizer.save_pretrained(tmp_dir)
            new_tokenizer = AutoTokenizer.from_pretrained(tmp_dir)
            self.assertEqual(len(new_tokenizer), len(tokenizer))
            json_tokenizer = json.loads(new_tokenizer._tokenizer.to_str())
            self.assertNotIn("huggingface", json_tokenizer["model"]["vocab"])

    def test_repo_versioning(self):
        # This repo has two tokenizer files, one for v4.0.0 and above with an added token, one for versions lower.
        repo = "hf-internal-testing/test-two-tokenizers"

        # This should pick the new tokenizer file as the version of Transformers is > 4.0.0
        tokenizer = AutoTokenizer.from_pretrained(repo)
        self.assertEqual(len(tokenizer), 28997)
        json_tokenizer = json.loads(tokenizer._tokenizer.to_str())
        self.assertIn("huggingface", json_tokenizer["model"]["vocab"])

        # Testing an older version by monkey-patching the version in the module it's used.
        import transformers as old_transformers

        old_transformers.tokenization_utils_base.__version__ = "3.0.0"
        old_tokenizer = old_transformers.models.auto.AutoTokenizer.from_pretrained(repo)
        self.assertEqual(len(old_tokenizer), 28996)
        json_tokenizer = json.loads(old_tokenizer._tokenizer.to_str())
        self.assertNotIn("huggingface", json_tokenizer["model"]["vocab"])


@require_tokenizers
class ReduceMutableBorrowTests(unittest.TestCase):
    def test_async_share_tokenizer(self):
        # See https://github.com/huggingface/transformers/pull/12550
        # and https://github.com/huggingface/tokenizers/issues/537
        tokenizer = PreTrainedTokenizerFast.from_pretrained("robot-test/dummy-tokenizer-wordlevel")
        text = "The Matrix is a 1999 science fiction action film."

        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [executor.submit(self.fetch, tokenizer, text) for i in range(10)]
            return_value = [future.result() for future in futures]
            self.assertEqual(return_value, [[1, 10, 0, 8, 0, 18, 0, 0, 0, 2] for i in range(10)])

    def fetch(self, tokenizer, text):
        return tokenizer.encode(text, truncation="longest_first", padding="longest")
