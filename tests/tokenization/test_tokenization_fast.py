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
import copy
import json
import multiprocessing
import os
import pickle
import shutil
import tempfile
import unittest

from tokenizers import Tokenizer, decoders, pre_tokenizers, trainers
from tokenizers.models import BPE, WordLevel

from transformers import AutoTokenizer, PreTrainedTokenizerFast
from transformers.testing_utils import require_tokenizers


def _multiprocess_encode(tokenizer_bytes, batch):
    """Worker body for the multiprocessing pickling test.

    The tokenizer is received as already-pickled bytes (exactly how
    ``datasets.map(num_proc=...)`` / ``DataLoader(num_workers=...)`` ship it to
    the child process) and must round-trip through ``pickle.loads`` before being
    used.
    """
    tokenizer = pickle.loads(tokenizer_bytes)
    return tokenizer(batch, padding=True, truncation=True, max_length=8)["input_ids"]


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
        self.assertEqual(new_tokenizer.decode(encoding_ids), " a🤗")

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
            self.assertEqual(tok(sentences, padding = True, return_token_type_ids=True), {'input_ids': [[8774, 6, 3, 63, 31, 1748, 55, 1, 0, 0, 0, 0,0, 0, 0, 0],[ 571, 33, 25, 3, 2, 3, 58, 290, 225, 59, 36, 136, 962, 269, 58, 1]], 'token_type_ids': [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], 'attention_mask': [[1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]})  # fmt: skip

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
            self.assertEqual(tok(sentences, truncation = True, max_length = 8, return_token_type_ids=True), {'input_ids': [[8774, 6, 3, 63, 31, 1748, 55, 1],[ 571, 33, 25, 3, 2, 3, 58, 1]], 'token_type_ids': [[0, 0, 0, 0, 0, 0, 0, 0],[0, 0, 0, 0, 0, 0, 0, 0]], 'attention_mask': [[1, 1, 1, 1, 1, 1, 1, 1],[1, 1, 1, 1, 1, 1, 1, 1]]})  # fmt: skip

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

    def test_bpe_tokenizer_skips_clean_up_tokenization_spaces(self):
        """BPE tokenizers should not apply clean_up_tokenization even when the flag is True.

        clean_up_tokenization strips spaces before punctuation (e.g. " ." -> "."),
        which was designed for WordPiece tokenizers. For BPE tokenizers, spaces are
        encoded as part of tokens and the cleanup is destructive.
        """
        tokenizer = PreTrainedTokenizerFast.from_pretrained(self.bytelevel_bpe_model_name)
        tokenizer.clean_up_tokenization_spaces = True

        # Text with space before punctuation — cleanup would strip it if applied.
        # Leading space accounts for ByteLevel BPE's add_prefix_space behavior.
        text = " Hello world ."
        ids = tokenizer.encode(text, add_special_tokens=False)
        decoded = tokenizer.decode(ids, clean_up_tokenization_spaces=True)

        # The space before "." must be preserved — BPE guard skips the cleanup
        self.assertEqual(decoded, text)

    def test_bpe_override_forces_cleanup(self):
        """The escape hatch flag forces cleanup even for BPE tokenizers."""
        tokenizer = PreTrainedTokenizerFast.from_pretrained(self.bytelevel_bpe_model_name)
        tokenizer.clean_up_tokenization_spaces = True
        tokenizer.clean_up_tokenization_spaces_for_bpe_even_though_it_will_corrupt_output = True

        text = " Hello world ."
        ids = tokenizer.encode(text, add_special_tokens=False)
        decoded = tokenizer.decode(ids, clean_up_tokenization_spaces=True)

        # With the override, cleanup IS applied — spaces before punctuation are stripped
        self.assertEqual(decoded, " Hello world.")

    def test_bpe_override_irrelevant_when_cleanup_false(self):
        """Override flag has no effect when clean_up_tokenization_spaces is False."""
        tokenizer = PreTrainedTokenizerFast.from_pretrained(self.bytelevel_bpe_model_name)
        tokenizer.clean_up_tokenization_spaces = False
        tokenizer.clean_up_tokenization_spaces_for_bpe_even_though_it_will_corrupt_output = True

        # Leading space accounts for ByteLevel BPE's add_prefix_space behavior
        text = " Hello world ."
        ids = tokenizer.encode(text, add_special_tokens=False)
        decoded = tokenizer.decode(ids)

        # cleanup=False takes precedence — text is preserved, override is irrelevant
        self.assertEqual(decoded, text)

    def test_non_bpe_tokenizer_still_cleans_up(self):
        """Non-BPE tokenizers should still apply cleanup normally."""
        # model_paths[0] is a WordLevel tokenizer (non-BPE)
        tokenizer = PreTrainedTokenizerFast.from_pretrained(self.model_paths[0])
        tokenizer.clean_up_tokenization_spaces = True

        text = "hello world ."
        ids = tokenizer.encode(text, add_special_tokens=False)
        decoded = tokenizer.decode(ids, clean_up_tokenization_spaces=True)

        # Non-BPE: cleanup IS applied — space before "." is stripped
        self.assertNotIn(" .", decoded)


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
        from unittest.mock import patch

        import transformers as old_transformers

        # Matt: The old test modified the module level version numbers
        # which was (I think) the cause of strange flaky tests depending on test ordering.
        # Using a context manager ensures the version mutation doesn't leak out of this test
        with patch.object(old_transformers.tokenization_utils_base, "__version__", "3.0.0"):
            old_tokenizer = old_transformers.models.auto.AutoTokenizer.from_pretrained(repo)
            self.assertEqual(len(old_tokenizer), 28996)
            json_tokenizer = json.loads(old_tokenizer._tokenizer.to_str())
            self.assertNotIn("huggingface", json_tokenizer["model"]["vocab"])


@require_tokenizers
class ReduceMutableBorrowTests(unittest.TestCase):
    """Thread-safety stress tests for fast (Rust-backed) tokenizers.

    These tests reproduce the ``RuntimeError: Already borrowed`` race described in
    https://github.com/huggingface/transformers/issues/47085 and verify that the
    per-instance ``threading.RLock`` serialises the critical section.
    """

    @classmethod
    def setUpClass(cls):
        cls.tmpdirname = tempfile.mkdtemp()
        # Build a small WordLevel tokenizer for offline use
        wl_vocab = {"[UNK]": 0, "[PAD]": 1, "hello": 2, "world": 3, "test": 4, "tokenizer": 5, "thread": 6}
        wl_dir = os.path.join(cls.tmpdirname, "wordlevel")
        os.makedirs(wl_dir, exist_ok=True)
        wl_tokenizer = Tokenizer(WordLevel(wl_vocab, unk_token="[UNK]"))
        wl_tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
        fast_wl = PreTrainedTokenizerFast(
            tokenizer_object=wl_tokenizer,
            unk_token="[UNK]",
            pad_token="[PAD]",
            cls_token="[CLS]",
            sep_token="[SEP]",
            mask_token="[MASK]",
        )
        fast_wl.save_pretrained(wl_dir)
        cls.wordlevel_path = wl_dir

        # Build a small BPE tokenizer for offline use
        bpe_dir = os.path.join(cls.tmpdirname, "bpe")
        os.makedirs(bpe_dir, exist_ok=True)
        bpe = Tokenizer(BPE(unk_token="[UNK]"))
        trainer = trainers.BpeTrainer(
            special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"],
            vocab_size=50,
        )
        corpus = [
            "hello world",
            "tokenizer thread safety",
            "test the tokenizer",
            "fast tokenization",
            "concurrent encoding",
        ]
        bpe.pre_tokenizer = pre_tokenizers.ByteLevel()
        bpe.train_from_iterator(corpus, trainer=trainer)
        bpe.decoder = decoders.ByteLevel()
        fast_bpe = PreTrainedTokenizerFast(
            tokenizer_object=bpe,
            unk_token="[UNK]",
            pad_token="[PAD]",
            cls_token="[CLS]",
            sep_token="[SEP]",
            mask_token="[MASK]",
        )
        fast_bpe.save_pretrained(bpe_dir)
        cls.bpe_path = bpe_dir

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmpdirname, ignore_errors=True)

    def _encode_with_varying_settings(self, tokenizer, text, seed):
        """Encode with changing truncation/padding to maximise interleaving."""
        settings = [
            {"truncation": True, "padding": False, "max_length": 8},
            {"truncation": True, "padding": True, "max_length": 16},
            {"truncation": "longest_first", "padding": "longest", "max_length": 12},
            {"truncation": False, "padding": "max_length", "max_length": 8},
            {"truncation": "only_first", "padding": True, "max_length": 10},
            {"truncation": True, "padding": False, "max_length": 6},
        ]
        setting = settings[seed % len(settings)]
        return tokenizer.encode(text, **setting)

    # ------------------------------------------------------------------
    #  Core race-condition reproduction
    # ------------------------------------------------------------------

    def test_concurrent_encode_mixed_settings(self):
        """16 threads, 50 iterations each, varying truncation/padding."""
        tokenizer = PreTrainedTokenizerFast.from_pretrained(self.wordlevel_path)
        text = "hello world test tokenizer thread safety"

        def worker(tid):
            for i in range(50):
                self._encode_with_varying_settings(tokenizer, text, tid * 50 + i)

        with concurrent.futures.ThreadPoolExecutor(max_workers=16) as pool:
            futures = [pool.submit(worker, i) for i in range(16)]
            for f in concurrent.futures.as_completed(futures, timeout=60):
                f.result()  # re-raise any exception

    def test_concurrent_encode_identical_settings(self):
        """32 threads with identical settings — should still not panic."""
        tokenizer = PreTrainedTokenizerFast.from_pretrained(self.wordlevel_path)
        text = "hello world test"

        def worker():
            for _ in range(30):
                tokenizer.encode(text, truncation=True, padding=True, max_length=8)

        with concurrent.futures.ThreadPoolExecutor(max_workers=32) as pool:
            futures = [pool.submit(worker) for _ in range(32)]
            for f in concurrent.futures.as_completed(futures, timeout=60):
                f.result()

    def test_concurrent_encode_with_decode(self):
        """Interleave encoding and decoding on the same tokenizer."""
        tokenizer = PreTrainedTokenizerFast.from_pretrained(self.wordlevel_path)
        texts = ["hello world", "test tokenizer", "thread safety"]
        encoded = [tokenizer.encode(t) for t in texts]

        def encode_worker():
            for _ in range(30):
                for t in texts:
                    tokenizer.encode(t, truncation=True, padding=True, max_length=8)

        def decode_worker():
            for _ in range(30):
                for e in encoded:
                    tokenizer.decode(e)

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            enc_futures = [pool.submit(encode_worker) for _ in range(4)]
            dec_futures = [pool.submit(decode_worker) for _ in range(4)]
            for f in concurrent.futures.as_completed(enc_futures + dec_futures, timeout=60):
                f.result()

    def test_batch_encode_concurrent(self):
        """Concurrent batch-encoding with varying padding/truncation."""
        tokenizer = PreTrainedTokenizerFast.from_pretrained(self.wordlevel_path)
        batch = [
            "hello world",
            "test tokenizer",
            "thread safety is important",
            "fast tokenization",
        ]

        def worker(tid):
            for i in range(30):
                pad = i % 2 == 0
                trunc = i % 3 == 0
                kwargs = {}
                if pad:
                    kwargs["padding"] = True
                if trunc:
                    kwargs["truncation"] = True
                    kwargs["max_length"] = 8 + (i % 4)
                tokenizer(batch, **kwargs)

        with concurrent.futures.ThreadPoolExecutor(max_workers=12) as pool:
            futures = [pool.submit(worker, i) for i in range(12)]
            for f in concurrent.futures.as_completed(futures, timeout=60):
                f.result()

    # ------------------------------------------------------------------
    #  BPE tokenizer tests
    # ------------------------------------------------------------------

    def test_concurrent_bpe_mixed_settings(self):
        """BPE tokenizer with 16 threads and varying settings."""
        tokenizer = PreTrainedTokenizerFast.from_pretrained(self.bpe_path)
        text = "hello world tokenizer thread safety"

        def worker(tid):
            for i in range(40):
                self._encode_with_varying_settings(tokenizer, text, tid * 40 + i)

        with concurrent.futures.ThreadPoolExecutor(max_workers=16) as pool:
            futures = [pool.submit(worker, i) for i in range(16)]
            for f in concurrent.futures.as_completed(futures, timeout=60):
                f.result()

    def test_concurrent_bpe_batch(self):
        """BPE tokenizer with concurrent batch encoding."""
        tokenizer = PreTrainedTokenizerFast.from_pretrained(self.bpe_path)
        batch = [
            "hello world",
            "test the tokenizer thread",
            "safety concurrent encoding",
        ]

        def worker(i):
            for _ in range(30):
                tokenizer(
                    batch,
                    padding=i % 2 == 0,
                    truncation=i % 3 == 0,
                    max_length=8 + (i % 4),
                )

        with concurrent.futures.ThreadPoolExecutor(max_workers=12) as pool:
            futures = [pool.submit(worker, i) for i in range(12)]
            for f in concurrent.futures.as_completed(futures, timeout=60):
                f.result()

    # ------------------------------------------------------------------
    #  Multiple independent tokenizer instances
    # ------------------------------------------------------------------

    def test_multiple_tokenizer_instances(self):
        """Multiple tokenizer instances should not interfere."""
        tok1 = PreTrainedTokenizerFast.from_pretrained(self.wordlevel_path)
        tok2 = PreTrainedTokenizerFast.from_pretrained(self.bpe_path)
        text = "hello world"

        def worker1():
            for _ in range(30):
                tok1.encode(text, truncation=True, padding=True, max_length=8)

        def worker2():
            for _ in range(30):
                tok2.encode(text, truncation=True, padding=True, max_length=8)

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            futures = [pool.submit(worker1) for _ in range(4)] + [pool.submit(worker2) for _ in range(4)]
            for f in concurrent.futures.as_completed(futures, timeout=60):
                f.result()

    # ------------------------------------------------------------------
    #  Single-threaded correctness — lock must not break normal use
    # ------------------------------------------------------------------

    def test_single_threaded_consistency(self):
        """Lock must not alter single-threaded behaviour."""
        tokenizer = PreTrainedTokenizerFast.from_pretrained(self.wordlevel_path)
        text = "hello world test"

        # Encode with various settings
        r1 = tokenizer.encode(text, truncation=True, padding=True, max_length=8)
        r2 = tokenizer.encode(text, truncation=True, padding=True, max_length=8)
        self.assertEqual(r1, r2, "Same settings should produce identical results")

        r3 = tokenizer.encode(text, truncation=True, padding=False, max_length=4)
        self.assertLessEqual(len(r3), 4, "Truncation should limit length")

        r4 = tokenizer.encode(text, truncation=False, padding=False)
        self.assertGreater(len(r4), 0, "No truncation should preserve full sequence")

        # Decode round-trip
        decoded = tokenizer.decode(r1, skip_special_tokens=True)
        self.assertIsInstance(decoded, str)

    def test_single_threaded_batch_consistency(self):
        """Batch encoding in single thread must be consistent."""
        tokenizer = PreTrainedTokenizerFast.from_pretrained(self.wordlevel_path)
        batch = ["hello world", "test tokenizer", "thread safety"]

        out1 = tokenizer(batch, truncation=True, padding=True, max_length=8)
        out2 = tokenizer(batch, truncation=True, padding=True, max_length=8)
        self.assertEqual(out1["input_ids"], out2["input_ids"])

        self.assertEqual(len(out1["input_ids"]), len(batch))
        for ids in out1["input_ids"]:
            self.assertLessEqual(len(ids), 8)

    def test_return_overflowing_tokens(self):
        """Overflowing tokens must work correctly."""
        tokenizer = PreTrainedTokenizerFast.from_pretrained(self.wordlevel_path)
        text = "hello world test tokenizer thread safety"
        encoded = tokenizer(
            text,
            truncation=True,
            max_length=3,
            return_overflowing_tokens=True,
        )
        self.assertIn("overflow_to_sample_mapping", encoded)
        self.assertGreater(len(encoded["input_ids"]), 1)

    # ------------------------------------------------------------------
    #  Thread safety property inspection
    # ------------------------------------------------------------------

    def test_lock_is_per_instance(self):
        """Each tokenizer instance must have its own lock."""
        t1 = PreTrainedTokenizerFast.from_pretrained(self.wordlevel_path)
        t2 = PreTrainedTokenizerFast.from_pretrained(self.wordlevel_path)
        self.assertIsNot(t1._lock, t2._lock, "Lock must be per-instance")

    def test_lock_is_reentrant(self):
        """The lock must allow re-entrant acquisition (RLock behaviour)."""
        tokenizer = PreTrainedTokenizerFast.from_pretrained(self.wordlevel_path)
        lock = tokenizer._lock
        acquired = lock.acquire(blocking=False)
        self.assertTrue(acquired)
        re_acquired = lock.acquire(blocking=False)
        self.assertTrue(re_acquired, "Lock must be reentrant (RLock)")
        lock.release()
        lock.release()

    # ------------------------------------------------------------------
    #  Long-running stress test (short iteration for CI)
    # ------------------------------------------------------------------

    def test_stress_many_threads_many_iterations(self):
        """64 threads, 100 iterations, aggressive truncation toggling."""
        tokenizer = PreTrainedTokenizerFast.from_pretrained(self.wordlevel_path)
        text = "hello world test tokenizer thread safety concurrent fast tokenization"

        def worker():
            for i in range(100):
                trunc = "longest_first" if i % 2 == 0 else "only_first"
                pad = i % 3 == 0
                tokenizer.encode(text, truncation=trunc, padding=pad, max_length=8 + (i % 4))

        with concurrent.futures.ThreadPoolExecutor(max_workers=64) as pool:
            futures = [pool.submit(worker) for _ in range(64)]
            for f in concurrent.futures.as_completed(futures, timeout=120):
                f.result()


@require_tokenizers
class FastTokenizerPickleTests(unittest.TestCase):
    """Regression tests for the picklability of ``TokenizersBackend``.

    The per-instance ``threading.RLock`` added for thread-safety (issue #47085)
    is not picklable, which broke ``pickle.dumps`` / ``copy.deepcopy`` and any
    multiprocessing usage (``datasets.map(num_proc=...)``,
    ``DataLoader(num_workers=...)``). ``__getstate__`` / ``__setstate__`` exclude
    the lock from serialization and rebuild it on unpickle.
    """

    @classmethod
    def setUpClass(cls):
        cls.tmpdirname = tempfile.mkdtemp()
        wl_vocab = {"[UNK]": 0, "[PAD]": 1, "hello": 2, "world": 3, "test": 4, "tokenizer": 5, "thread": 6}
        wl_dir = os.path.join(cls.tmpdirname, "wordlevel")
        os.makedirs(wl_dir, exist_ok=True)
        wl_tokenizer = Tokenizer(WordLevel(wl_vocab, unk_token="[UNK]"))
        wl_tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
        fast_wl = PreTrainedTokenizerFast(
            tokenizer_object=wl_tokenizer,
            unk_token="[UNK]",
            pad_token="[PAD]",
            cls_token="[CLS]",
            sep_token="[SEP]",
            mask_token="[MASK]",
        )
        fast_wl.save_pretrained(wl_dir)
        cls.wordlevel_path = wl_dir

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmpdirname, ignore_errors=True)

    def test_pickle_round_trip(self):
        """pickle.dumps/loads must round-trip and still encode/decode."""
        tokenizer = PreTrainedTokenizerFast.from_pretrained(self.wordlevel_path)
        text = "hello world test tokenizer thread safety"
        expected_ids = tokenizer.encode(text)

        dumped = pickle.dumps(tokenizer)
        restored = pickle.loads(dumped)

        self.assertEqual(restored.encode(text), expected_ids)
        self.assertEqual(
            restored.decode(expected_ids, skip_special_tokens=True),
            tokenizer.decode(expected_ids, skip_special_tokens=True),
        )

    def test_deepcopy_round_trip(self):
        """copy.deepcopy must round-trip and still encode/decode."""
        tokenizer = PreTrainedTokenizerFast.from_pretrained(self.wordlevel_path)
        text = "hello world test tokenizer"
        expected_ids = tokenizer.encode(text)

        copied = copy.deepcopy(tokenizer)
        self.assertEqual(copied.encode(text), expected_ids)
        self.assertEqual(
            copied.decode(expected_ids, skip_special_tokens=True),
            tokenizer.decode(expected_ids, skip_special_tokens=True),
        )

    def test_multiprocessing_pool(self):
        """A pickled tokenizer must work across a process pool (datasets.map)."""
        tokenizer = PreTrainedTokenizerFast.from_pretrained(self.wordlevel_path)
        batch = ["hello world", "test tokenizer", "thread safety"]
        expected = tokenizer(batch, padding=True, truncation=True, max_length=8)["input_ids"]
        # This is the real failure mode: the tokenizer is serialized and shipped to workers.
        tokenizer_bytes = pickle.dumps(tokenizer)

        with multiprocessing.Pool(processes=3) as pool:
            results = pool.starmap(_multiprocess_encode, [(tokenizer_bytes, batch)] * 3)

        for result in results:
            self.assertEqual(result, expected)

    def test_pickle_does_not_break_thread_safety(self):
        """After unpickling, concurrent use must still be race-free (0 errors)."""
        tokenizer = PreTrainedTokenizerFast.from_pretrained(self.wordlevel_path)
        text = "hello world test tokenizer thread safety concurrent fast tokenization"
        restored = pickle.loads(pickle.dumps(tokenizer))

        def worker():
            for i in range(50):
                trunc = "longest_first" if i % 2 == 0 else "only_first"
                pad = i % 3 == 0
                restored.encode(text, truncation=trunc, padding=pad, max_length=8 + (i % 4))

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            futures = [pool.submit(worker) for _ in range(8)]
            for f in concurrent.futures.as_completed(futures, timeout=120):
                f.result()


@require_tokenizers
class PatchMistralRegexHubCallTests(unittest.TestCase):
    """
    Regression tests for https://github.com/huggingface/transformers/issues/44749 /
    https://github.com/huggingface/transformers/issues/43502

    `_patch_mistral_regex` used to unconditionally call `huggingface_hub.model_info`
    for any tokenizer with vocab > 100k (e.g. Qwen3), causing a large slowdown
    and breaking offline / local-path loading.
    """

    def _dummy_backend_tokenizer(self):
        tok = Tokenizer(WordLevel({"[UNK]": 0, "a": 1}, unk_token="[UNK]"))
        tok.pre_tokenizer = pre_tokenizers.Whitespace()
        return tok

    def test_local_files_only_skips_model_info(self):
        from unittest.mock import patch

        tok = self._dummy_backend_tokenizer()
        with patch(
            "huggingface_hub.model_info",
            side_effect=AssertionError("model_info must not be called when local_files_only=True"),
        ):
            result = PreTrainedTokenizerFast._patch_mistral_regex(
                tok,
                "some/non-mistral-repo-id",
                local_files_only=True,
            )
        self.assertIsNotNone(result)

    def test_hub_error_does_not_break_init(self):
        from unittest.mock import patch

        tok = self._dummy_backend_tokenizer()
        with patch("huggingface_hub.model_info", side_effect=RuntimeError("hub down")):
            # Must not raise — a Hub failure should be swallowed for non-Mistral models.
            result = PreTrainedTokenizerFast._patch_mistral_regex(
                tok,
                "not-a-real/hub-id",
            )
        self.assertIsNotNone(result)
