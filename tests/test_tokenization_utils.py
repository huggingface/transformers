# coding=utf-8
# Copyright 2018 HuggingFace Inc..
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

import pickle
import unittest
from typing import Callable, Optional

import numpy as np

from transformers import BatchEncoding, BertTokenizer, BertTokenizerFast, PreTrainedTokenizer, TensorType, TokenSpan
from transformers.models.gpt2.tokenization_gpt2 import GPT2Tokenizer
from transformers.testing_utils import CaptureStderr, require_flax, require_tf, require_tokenizers, require_torch, slow


class TokenizerUtilsTest(unittest.TestCase):
    def check_tokenizer_from_pretrained(self, tokenizer_class):
        s3_models = list(tokenizer_class.max_model_input_sizes.keys())
        for model_name in s3_models[:1]:
            tokenizer = tokenizer_class.from_pretrained(model_name)
            self.assertIsNotNone(tokenizer)
            self.assertIsInstance(tokenizer, tokenizer_class)
            self.assertIsInstance(tokenizer, PreTrainedTokenizer)

            for special_tok in tokenizer.all_special_tokens:
                self.assertIsInstance(special_tok, str)
                special_tok_id = tokenizer.convert_tokens_to_ids(special_tok)
                self.assertIsInstance(special_tok_id, int)

    def assert_dump_and_restore(self, be_original: BatchEncoding, equal_op: Optional[Callable] = None):
        batch_encoding_str = pickle.dumps(be_original)
        self.assertIsNotNone(batch_encoding_str)

        be_restored = pickle.loads(batch_encoding_str)

        # Ensure is_fast is correctly restored
        self.assertEqual(be_restored.is_fast, be_original.is_fast)

        # Ensure encodings are potentially correctly restored
        if be_original.is_fast:
            self.assertIsNotNone(be_restored.encodings)
        else:
            self.assertIsNone(be_restored.encodings)

        # Ensure the keys are the same
        for original_v, restored_v in zip(be_original.values(), be_restored.values()):
            if equal_op:
                self.assertTrue(equal_op(restored_v, original_v))
            else:
                self.assertEqual(restored_v, original_v)

    @slow
    def test_pretrained_tokenizers(self):
        self.check_tokenizer_from_pretrained(GPT2Tokenizer)

    def test_tensor_type_from_str(self):
        self.assertEqual(TensorType("tf"), TensorType.TENSORFLOW)
        self.assertEqual(TensorType("pt"), TensorType.PYTORCH)
        self.assertEqual(TensorType("np"), TensorType.NUMPY)

    @require_tokenizers
    def test_batch_encoding_pickle(self):
        import numpy as np

        tokenizer_p = BertTokenizer.from_pretrained("bert-base-cased")
        tokenizer_r = BertTokenizerFast.from_pretrained("bert-base-cased")

        # Python no tensor
        with self.subTest("BatchEncoding (Python, return_tensors=None)"):
            self.assert_dump_and_restore(tokenizer_p("Small example to encode"))

        with self.subTest("BatchEncoding (Python, return_tensors=NUMPY)"):
            self.assert_dump_and_restore(
                tokenizer_p("Small example to encode", return_tensors=TensorType.NUMPY), np.array_equal
            )

        with self.subTest("BatchEncoding (Rust, return_tensors=None)"):
            self.assert_dump_and_restore(tokenizer_r("Small example to encode"))

        with self.subTest("BatchEncoding (Rust, return_tensors=NUMPY)"):
            self.assert_dump_and_restore(
                tokenizer_r("Small example to encode", return_tensors=TensorType.NUMPY), np.array_equal
            )

    @require_tf
    @require_tokenizers
    def test_batch_encoding_pickle_tf(self):
        import tensorflow as tf

        def tf_array_equals(t1, t2):
            return tf.reduce_all(tf.equal(t1, t2))

        tokenizer_p = BertTokenizer.from_pretrained("bert-base-cased")
        tokenizer_r = BertTokenizerFast.from_pretrained("bert-base-cased")

        with self.subTest("BatchEncoding (Python, return_tensors=TENSORFLOW)"):
            self.assert_dump_and_restore(
                tokenizer_p("Small example to encode", return_tensors=TensorType.TENSORFLOW), tf_array_equals
            )

        with self.subTest("BatchEncoding (Rust, return_tensors=TENSORFLOW)"):
            self.assert_dump_and_restore(
                tokenizer_r("Small example to encode", return_tensors=TensorType.TENSORFLOW), tf_array_equals
            )

    @require_torch
    @require_tokenizers
    def test_batch_encoding_pickle_pt(self):
        import torch

        tokenizer_p = BertTokenizer.from_pretrained("bert-base-cased")
        tokenizer_r = BertTokenizerFast.from_pretrained("bert-base-cased")

        with self.subTest("BatchEncoding (Python, return_tensors=PYTORCH)"):
            self.assert_dump_and_restore(
                tokenizer_p("Small example to encode", return_tensors=TensorType.PYTORCH), torch.equal
            )

        with self.subTest("BatchEncoding (Rust, return_tensors=PYTORCH)"):
            self.assert_dump_and_restore(
                tokenizer_r("Small example to encode", return_tensors=TensorType.PYTORCH), torch.equal
            )

    @require_tokenizers
    def test_batch_encoding_is_fast(self):
        tokenizer_p = BertTokenizer.from_pretrained("bert-base-cased")
        tokenizer_r = BertTokenizerFast.from_pretrained("bert-base-cased")

        with self.subTest("Python Tokenizer"):
            self.assertFalse(tokenizer_p("Small example to_encode").is_fast)

        with self.subTest("Rust Tokenizer"):
            self.assertTrue(tokenizer_r("Small example to_encode").is_fast)

    @require_tokenizers
    def test_batch_encoding_word_to_tokens(self):
        tokenizer_r = BertTokenizerFast.from_pretrained("bert-base-cased")
        encoded = tokenizer_r(["Test", "\xad", "test"], is_split_into_words=True)

        self.assertEqual(encoded.word_to_tokens(0), TokenSpan(start=1, end=2))
        self.assertEqual(encoded.word_to_tokens(1), None)
        self.assertEqual(encoded.word_to_tokens(2), TokenSpan(start=2, end=3))

    def test_batch_encoding_with_labels(self):
        batch = BatchEncoding({"inputs": [[1, 2, 3], [4, 5, 6]], "labels": [0, 1]})
        tensor_batch = batch.convert_to_tensors(tensor_type="np")
        self.assertEqual(tensor_batch["inputs"].shape, (2, 3))
        self.assertEqual(tensor_batch["labels"].shape, (2,))
        # test converting the converted
        with CaptureStderr() as cs:
            tensor_batch = batch.convert_to_tensors(tensor_type="np")
        self.assertFalse(len(cs.err), msg=f"should have no warning, but got {cs.err}")

        batch = BatchEncoding({"inputs": [1, 2, 3], "labels": 0})
        tensor_batch = batch.convert_to_tensors(tensor_type="np", prepend_batch_axis=True)
        self.assertEqual(tensor_batch["inputs"].shape, (1, 3))
        self.assertEqual(tensor_batch["labels"].shape, (1,))

    @require_torch
    def test_batch_encoding_with_labels_pt(self):
        batch = BatchEncoding({"inputs": [[1, 2, 3], [4, 5, 6]], "labels": [0, 1]})
        tensor_batch = batch.convert_to_tensors(tensor_type="pt")
        self.assertEqual(tensor_batch["inputs"].shape, (2, 3))
        self.assertEqual(tensor_batch["labels"].shape, (2,))
        # test converting the converted
        with CaptureStderr() as cs:
            tensor_batch = batch.convert_to_tensors(tensor_type="pt")
        self.assertFalse(len(cs.err), msg=f"should have no warning, but got {cs.err}")

        batch = BatchEncoding({"inputs": [1, 2, 3], "labels": 0})
        tensor_batch = batch.convert_to_tensors(tensor_type="pt", prepend_batch_axis=True)
        self.assertEqual(tensor_batch["inputs"].shape, (1, 3))
        self.assertEqual(tensor_batch["labels"].shape, (1,))

    @require_tf
    def test_batch_encoding_with_labels_tf(self):
        batch = BatchEncoding({"inputs": [[1, 2, 3], [4, 5, 6]], "labels": [0, 1]})
        tensor_batch = batch.convert_to_tensors(tensor_type="tf")
        self.assertEqual(tensor_batch["inputs"].shape, (2, 3))
        self.assertEqual(tensor_batch["labels"].shape, (2,))
        # test converting the converted
        with CaptureStderr() as cs:
            tensor_batch = batch.convert_to_tensors(tensor_type="tf")
        self.assertFalse(len(cs.err), msg=f"should have no warning, but got {cs.err}")

        batch = BatchEncoding({"inputs": [1, 2, 3], "labels": 0})
        tensor_batch = batch.convert_to_tensors(tensor_type="tf", prepend_batch_axis=True)
        self.assertEqual(tensor_batch["inputs"].shape, (1, 3))
        self.assertEqual(tensor_batch["labels"].shape, (1,))

    @require_flax
    def test_batch_encoding_with_labels_jax(self):
        batch = BatchEncoding({"inputs": [[1, 2, 3], [4, 5, 6]], "labels": [0, 1]})
        tensor_batch = batch.convert_to_tensors(tensor_type="jax")
        self.assertEqual(tensor_batch["inputs"].shape, (2, 3))
        self.assertEqual(tensor_batch["labels"].shape, (2,))
        # test converting the converted
        with CaptureStderr() as cs:
            tensor_batch = batch.convert_to_tensors(tensor_type="jax")
        self.assertFalse(len(cs.err), msg=f"should have no warning, but got {cs.err}")

        batch = BatchEncoding({"inputs": [1, 2, 3], "labels": 0})
        tensor_batch = batch.convert_to_tensors(tensor_type="jax", prepend_batch_axis=True)
        self.assertEqual(tensor_batch["inputs"].shape, (1, 3))
        self.assertEqual(tensor_batch["labels"].shape, (1,))

    def test_padding_accepts_tensors(self):
        features = [{"input_ids": np.array([0, 1, 2])}, {"input_ids": np.array([0, 1, 2, 3])}]
        tokenizer = BertTokenizer.from_pretrained("bert-base-cased")

        batch = tokenizer.pad(features, padding=True)
        self.assertTrue(isinstance(batch["input_ids"], np.ndarray))
        self.assertEqual(batch["input_ids"].tolist(), [[0, 1, 2, tokenizer.pad_token_id], [0, 1, 2, 3]])
        batch = tokenizer.pad(features, padding=True, return_tensors="np")
        self.assertTrue(isinstance(batch["input_ids"], np.ndarray))
        self.assertEqual(batch["input_ids"].tolist(), [[0, 1, 2, tokenizer.pad_token_id], [0, 1, 2, 3]])

    @require_torch
    def test_padding_accepts_tensors_pt(self):
        import torch

        features = [{"input_ids": torch.tensor([0, 1, 2])}, {"input_ids": torch.tensor([0, 1, 2, 3])}]
        tokenizer = BertTokenizer.from_pretrained("bert-base-cased")

        batch = tokenizer.pad(features, padding=True)
        self.assertTrue(isinstance(batch["input_ids"], torch.Tensor))
        self.assertEqual(batch["input_ids"].tolist(), [[0, 1, 2, tokenizer.pad_token_id], [0, 1, 2, 3]])
        batch = tokenizer.pad(features, padding=True, return_tensors="pt")
        self.assertTrue(isinstance(batch["input_ids"], torch.Tensor))
        self.assertEqual(batch["input_ids"].tolist(), [[0, 1, 2, tokenizer.pad_token_id], [0, 1, 2, 3]])

    @require_tf
    def test_padding_accepts_tensors_tf(self):
        import tensorflow as tf

        features = [{"input_ids": tf.constant([0, 1, 2])}, {"input_ids": tf.constant([0, 1, 2, 3])}]
        tokenizer = BertTokenizer.from_pretrained("bert-base-cased")

        batch = tokenizer.pad(features, padding=True)
        self.assertTrue(isinstance(batch["input_ids"], tf.Tensor))
        self.assertEqual(batch["input_ids"].numpy().tolist(), [[0, 1, 2, tokenizer.pad_token_id], [0, 1, 2, 3]])
        batch = tokenizer.pad(features, padding=True, return_tensors="tf")
        self.assertTrue(isinstance(batch["input_ids"], tf.Tensor))
        self.assertEqual(batch["input_ids"].numpy().tolist(), [[0, 1, 2, tokenizer.pad_token_id], [0, 1, 2, 3]])
