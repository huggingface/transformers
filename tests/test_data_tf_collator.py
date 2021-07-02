# Copyright 2020 The HuggingFace Team. All rights reserved.
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
import shutil
import tempfile
import unittest

from transformers import BertTokenizer, is_tf_available, set_seed
from transformers.testing_utils import require_tf


if is_tf_available():
    import tensorflow as tf

    from transformers import TFDataCollatorForLanguageModeling


@require_tf
class TFDataCollatorIntegrationTest(unittest.TestCase):
    def setUp(self):
        self.tmpdirname = tempfile.mkdtemp()

        vocab_tokens = ["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"]
        self.vocab_file = os.path.join(self.tmpdirname, "vocab.txt")
        with open(self.vocab_file, "w", encoding="utf-8") as vocab_writer:
            vocab_writer.write("".join([x + "\n" for x in vocab_tokens]))

    def tearDown(self):
        shutil.rmtree(self.tmpdirname)

    def assert_all_equal(self, batches: list, key: str, shape: tuple):
        for batch in batches:
            self.assertEqual(batch[key].shape, tf.TensorShape(shape))

    def _test_normal_with_special_tokens_mask(self, no_pad_features, pad_features):
        tokenizer = BertTokenizer(self.vocab_file)

        tf_data_collator = TFDataCollatorForLanguageModeling(
            tokenizer,
            padding_length=None,
            batch_size=2,
            special_tokens_mask=[0]*5+[1]*5)

        tf_data_collator_with_padding = TFDataCollatorForLanguageModeling(
            tokenizer,
            padding_length=16,
            batch_size=2,
            special_tokens_mask=[0]*11+[1]*5)

        tf_dataset_no_pad = tf.data.Dataset.from_tensor_slices(
            no_pad_features
        )
        tf_dataset_pad = tf.data.Dataset.from_tensor_slices(
            tf.ragged.stack(pad_features)
        )

        # --- Non-Ragged Data ----
        # --- Pre Batching w/ No Padding ----
        batch = tf_data_collator(tf_dataset_no_pad)

        self.assert_all_equal(list(batch), 'input_ids', (2, 10))
        self.assert_all_equal(list(batch), 'labels', (2, 10))

        # # --- Pre Batching w/ Padding ---
        batch = tf_data_collator_with_padding(tf_dataset_no_pad)

        self.assert_all_equal(list(batch), 'input_ids', (2, 16))
        self.assert_all_equal(list(batch), 'labels', (2, 16))

        # --- Ragged Data ----
        # --- Pre Batching w/ No Padding ----
        batch = tf_data_collator(tf_dataset_pad, True)

        self.assert_all_equal(list(batch), 'input_ids', (2, 10))
        self.assert_all_equal(list(batch), 'labels', (2, 10))

        # # --- Pre Batching w/ Padding ---
        batch = tf_data_collator_with_padding(tf_dataset_pad, True)

        self.assert_all_equal(list(batch), 'input_ids', (2, 16))
        self.assert_all_equal(list(batch), 'labels', (2, 16))

        tokenizer = BertTokenizer(self.vocab_file)
        tokenizer._pad_token = None

        no_token_collator = TFDataCollatorForLanguageModeling(
            tokenizer,
            padding_length=16,
            batch_size=2,
            special_tokens_mask=[0]*11+[1]*5)

        with self.assertRaises(ValueError):
            # Expect error due to padding token missing
            no_token_collator(tf_dataset_pad)

        set_seed(42)  # For reproducibility
        tokenizer = BertTokenizer(self.vocab_file)
        mask_data_collator = TFDataCollatorForLanguageModeling(
            tokenizer,
            batch_size=2,
            special_tokens_mask=[0]*5+[1]*5)
        batch = mask_data_collator(tf_dataset_no_pad)

        self.assert_all_equal(list(batch), 'input_ids', (2, 10))
        self.assert_all_equal(list(batch), 'labels', (2, 10))

        output_batch = list(batch)[0]
        masked_tokens = output_batch["input_ids"] == tokenizer.mask_token_id
        self.assertTrue(masked_tokens.numpy().any())
        self.assertTrue(all(x == -100 for x in list(output_batch["labels"][~masked_tokens].numpy())))


    def _test_normal_without_special_tokens_mask(self, no_pad_features, pad_features):
        tokenizer = BertTokenizer(self.vocab_file)

        tf_data_collator = TFDataCollatorForLanguageModeling(
            tokenizer,
            batch_size=2)

        tf_data_collator_with_padding = TFDataCollatorForLanguageModeling(
            tokenizer,
            padding_length=16,
            batch_size=2)

        tf_dataset_no_pad = tf.data.Dataset.from_tensor_slices(
            no_pad_features
        )
        tf_dataset_pad = tf.data.Dataset.from_tensor_slices(
            tf.ragged.stack(pad_features)
        )

        # --- Non-Ragged Data ----
        # --- Pre Batching w/ No Padding ----
        batch = tf_data_collator(tf_dataset_no_pad)

        self.assert_all_equal(list(batch), 'input_ids', (2, 10))
        self.assert_all_equal(list(batch), 'labels', (2, 10))

        # # --- Pre Batching w/ Padding ---
        batch = tf_data_collator_with_padding(tf_dataset_no_pad)

        self.assert_all_equal(list(batch), 'input_ids', (2, 16))
        self.assert_all_equal(list(batch), 'labels', (2, 16))

        # --- Ragged Data ----
        # --- Pre Batching w/ No Padding ----
        batch = tf_data_collator(tf_dataset_pad, True)

        self.assert_all_equal(list(batch), 'input_ids', (2, 10))
        self.assert_all_equal(list(batch), 'labels', (2, 10))

        # # --- Pre Batching w/ Padding ---
        batch = tf_data_collator_with_padding(tf_dataset_pad, True)

        self.assert_all_equal(list(batch), 'input_ids', (2, 16))
        self.assert_all_equal(list(batch), 'labels', (2, 16))

        tokenizer = BertTokenizer(self.vocab_file)
        tokenizer._pad_token = None

        no_token_collator = TFDataCollatorForLanguageModeling(
            tokenizer,
            padding_length=5,
            batch_size=2)

        with self.assertRaises(ValueError):
            # Expect error due to padding token missing
            no_token_collator(tf_dataset_pad)

        set_seed(42)  # For reproducibility
        tokenizer = BertTokenizer(self.vocab_file)
        mask_data_collator = TFDataCollatorForLanguageModeling(
            tokenizer,
            batch_size=2)
        batch = mask_data_collator(tf_dataset_no_pad)

        self.assert_all_equal(list(batch), 'input_ids', (2, 10))
        self.assert_all_equal(list(batch), 'labels', (2, 10))

        output_batch = list(batch)[0]
        masked_tokens = output_batch["input_ids"] == tokenizer.mask_token_id
        self.assertTrue(masked_tokens.numpy().any())
        self.assertTrue(all(x == -100 for x in list(output_batch["labels"][~masked_tokens].numpy())))

    def test_data_collator_for_language_modeling(self):
        # List[int] case
        no_pad_features = [list(range(10)), list(range(10)), list(range(10)), list(range(10))]
        pad_features = [list(range(5)), list(range(10)), list(range(5)), list(range(10))]
        self._test_normal_without_special_tokens_mask(no_pad_features, pad_features)

        self._test_normal_with_special_tokens_mask(no_pad_features, pad_features)

        # tf.Tensor case
        no_pad_features = [tf.constant(range(10)), tf.constant(range(10)),
                           tf.constant(range(10)), tf.constant(range(10))]
        pad_features = [tf.constant(range(5)), tf.constant(range(10)),
                        tf.constant(range(5)), tf.constant(range(10))]
        self._test_normal_without_special_tokens_mask(no_pad_features, pad_features)
        self._test_normal_with_special_tokens_mask(no_pad_features, pad_features)

    def test_nsp(self):
        tokenizer = BertTokenizer(self.vocab_file)
        features = {"input_ids": [[0, 1, 2, 3, 4] for _ in range(2)],
                    "token_type_ids": [[0, 1, 2, 3, 4] for _ in range(2)],
                    "next_sentence_label": [i for i in range(2)]}
        data_collator = TFDataCollatorForLanguageModeling(
            tokenizer,
            batch_size=2)
        tf_dataset_no_pad = tf.data.Dataset.from_tensor_slices(features)
        batch = data_collator(tf_dataset_no_pad)

        self.assertEqual(list(batch)[0]["input_ids"].shape, tf.TensorShape((2, 5)))
        self.assertEqual(list(batch)[0]["token_type_ids"].shape, tf.TensorShape((2, 5)))
        self.assertEqual(list(batch)[0]["labels"].shape, tf.TensorShape((2, 5)))
        self.assertEqual(list(batch)[0]["next_sentence_label"].shape, tf.TensorShape((2,)))

        data_collator = TFDataCollatorForLanguageModeling(tokenizer, padding_length=8, batch_size=2)
        tf_dataset_pad = tf.data.Dataset.from_tensor_slices(features)
        batch = data_collator(tf_dataset_pad)

        self.assertEqual(list(batch)[0]["input_ids"].shape, tf.TensorShape((2, 8)))
        self.assertEqual(list(batch)[0]["token_type_ids"].shape, tf.TensorShape((2, 8)))
        self.assertEqual(list(batch)[0]["labels"].shape, tf.TensorShape((2, 8)))
        self.assertEqual(list(batch)[0]["next_sentence_label"].shape, tf.TensorShape((2,)))

    def test_sop(self):
        tokenizer = BertTokenizer(self.vocab_file)
        features = {"input_ids": [[0, 1, 2, 3, 4] for _ in range(2)],
                    "token_type_ids": [[0, 1, 2, 3, 4] for _ in range(2)],
                    "sentence_order_label": [i for i in range(2)]}
        data_collator = TFDataCollatorForLanguageModeling(
            tokenizer,
            batch_size=2)
        tf_dataset_no_pad = tf.data.Dataset.from_tensor_slices(features)
        batch = data_collator(tf_dataset_no_pad)

        self.assertEqual(list(batch)[0]["input_ids"].shape, tf.TensorShape((2, 5)))
        self.assertEqual(list(batch)[0]["token_type_ids"].shape, tf.TensorShape((2, 5)))
        self.assertEqual(list(batch)[0]["labels"].shape, tf.TensorShape((2, 5)))
        self.assertEqual(list(batch)[0]["sentence_order_label"].shape, tf.TensorShape((2,)))

        data_collator = TFDataCollatorForLanguageModeling(
            tokenizer,
            batch_size=2,
            padding_length=8)
        tf_dataset_pad = tf.data.Dataset.from_tensor_slices(features)
        batch = data_collator(tf_dataset_pad)

        self.assertEqual(list(batch)[0]["input_ids"].shape, tf.TensorShape((2, 8)))
        self.assertEqual(list(batch)[0]["token_type_ids"].shape, tf.TensorShape((2, 8)))
        self.assertEqual(list(batch)[0]["labels"].shape, tf.TensorShape((2, 8)))
        self.assertEqual(list(batch)[0]["sentence_order_label"].shape, tf.TensorShape((2,)))
