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

from copy import deepcopy 
from transformers import BertTokenizer, is_tf_available, set_seed
from transformers.testing_utils import require_tf
from dataclasses import dataclass

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

    def assert_all_equal(self, batches: list, key: str, shape: tuple) -> bool:
        for batch in batches:
            self.assertEqual(batch[key], tf.TensorShape(shape))

    def _test_normal_with_special_tokens_mask(self, features):
        tokenizer = BertTokenizer(self.vocab_file)
        special_tokens_mask = tokenizer.get_special_tokens_mask
        tf_data_collator = TFDataCollatorForLanguageModeling(tokenizer, special_tokens_mask)

        tf_dataset = tf.data.Dataset(features)

        # --- Pre Batching w/ No Padding ----
        batch = (
            tf_dataset
            .batch(2)
            .map(tf_data_collator)
        )

        self.assert_all_equal(batch, 'input_ids', (2, 10))
        self.assert_all_equal(batch, 'labels', (2, 10))

        # --- Post Batching w/ No Padding
        batch = (
            tf_dataset
            .map(tf_data_collator)
            .batch(2)
        )

        self.assertEqual(batch["input_ids"].shape, tf.TensorShape((2, 10)))
        self.assertEqual(batch["labels"].shape, tf.TensorShape((2, 10)))

        # --- Pre Batching w/ Padding ---
        batch = (
            tf_dataset
            .padded_batch(2, 16)
            .map(tf_data_collator)
        )

        self.assertEqual(batch["input_ids"].shape, tf.TensorShape((2, 10)))
        self.assertEqual(batch["labels"].shape, tf.TensorShape((2, 10)))

        # --- Post Batching w/ Padding ---
        batch = (
            tf_dataset
            .map(tf_data_collator)
            .padded_batch(2, 16)
        )

        batch = tf_data_collator(features)
        self.assertEqual(batch["input_ids"].shape, tf.TensorShape((2, 10)))

    def _test_normal_without_special_tokens_mask(self, no_pad_features, pad_features):
        tokenizer = BertTokenizer(self.vocab_file)
        tf_data_collator = TFDataCollatorForLanguageModeling(tokenizer)

        tf_dataset_no_pad = tf.data.Dataset.from_generator(
            lambda: no_pad_features, output_types=tf.int32
        )
        tf_dataset_pad = tf.data.Dataset.from_generator(
            lambda: pad_features, output_types=tf.int32
        )

        # --- Pre Batching w/ No Padding ----
        batch = (
            tf_dataset_no_pad
            .batch(1)
            .map(tf_data_collator)
            .as_numpy_iterator()
        )

        # print(list(batch.as_numpy_iterator()))
        print(type(batch))
        self.assertEqual(batch["input_ids"].shape, tf.TensorShape((2, 10)))
        self.assertEqual(batch["labels"].shape, tf.TensorShape((2, 10)))

        # --- Post Batching w/ No Padding
        #batch = (
        #     deepcopy(tf_dataset_no_pad)
        #     .map(tf_data_collator)
        #     .batch(2)
        # )
        #
        # self.assertEqual(batch["input_ids"].shape, tf.TensorShape((2, 10)))
        # self.assertEqual(batch["labels"].shape, tf.TensorShape((2, 10)))
        #
        # # --- Pre Batching w/ Padding ---
        # batch = (
        #     deepcopy(tf_dataset_pad)
        #     .padded_batch(2, 16)
        #     .map(tf_data_collator)
        # )
        #
        # self.assertEqual(batch["input_ids"].shape, tf.TensorShape((2, 10)))
        # self.assertEqual(batch["labels"].shape, tf.TensorShape((2, 10)))
        #
        # # --- Post Batching w/ Padding ---
        # batch = (
        #     deepcopy(tf_dataset_pad)
        #     .map(tf_data_collator)
        #     .padded_batch(2, 16)
        # )
        #
        # self.assertEqual(batch["input_ids"].shape, tf.TensorShape((2, 10)))
        # self.assertEqual(batch["labels"].shape, tf.TensorShape((2, 10)))

        # batch = tf_data_collator(features)
        # self.assertEqual(batch["input_ids"].shape, tf.TensorShape((2, 10)))

        # tokenizer._pad_token = None
        # data_collator = TFDataCollatorForLanguageModeling(tokenizer, mlm=False)
        # with self.assertRaises(ValueError):
        #     # Expect error due to padding token missing
        #     data_collator(pad_features)

        # set_seed(42)  # For reproducibility
        # tokenizer = BertTokenizer(self.vocab_file)
        # data_collator = TFDataCollatorForLanguageModeling(tokenizer)
        # batch = data_collator(no_pad_features)
        # self.assertEqual(batch["input_ids"].shape, tf.TensorShape((2, 10)))
        # self.assertEqual(batch["labels"].shape, tf.TensorShape((2, 10)))

        # masked_tokens = batch["input_ids"] == tokenizer.mask_token_id
        # self.assertTrue(masked_tokens.numpy().any())
        # self.assertTrue(all(x == -100 for x in list(batch["labels"][~masked_tokens].numpy())))

    def test_data_collator_for_language_modeling(self):
        # List[int] case
        no_pad_features = [list(range(10)), list(range(10))]
        pad_features = [list(range(5)), list(range(10))]
        self._test_normal_without_special_tokens_mask(no_pad_features, pad_features)

        # tf.Tensor case
        # no_pad_features = [(tf.constant(list(range(10))), tf.constant(list(range(10)))),
        #                    (tf.constant(list(range(10))), tf.constant(list(range(10))))]
        # pad_features = [(tf.constant(list(range(5))), tf.constant(list(range(10)))),
        #                 (tf.constant(list(range(5))), tf.constant(list(range(10))))]
        # self._test_normal_without_special_tokens_mask(no_pad_features, pad_features)

    def test_nsp(self):
        tokenizer = BertTokenizer(self.vocab_file)
        features = {"input_data": [[0, 1, 2, 3, 4] for i in range(2)],
                     "token_type_ids": [[0, 1, 2, 3, 4] for i in range(2)],
                     "sentence_prediction_label": [i for i in range(2)]}
        data_collator = TFDataCollatorForLanguageModeling(tokenizer)
        batch = data_collator(features)

        self.assertEqual(batch["input_ids"].shape, tf.TensorShape((2, 5)))
        self.assertEqual(batch["token_type_ids"].shape, tf.TensorShape((2, 5)))
        self.assertEqual(batch["labels"].shape, tf.TensorShape((2, 5)))
        self.assertEqual(batch["next_sentence_label"].shape, tf.TensorShape((2,)))

        data_collator = TFDataCollatorForLanguageModeling(tokenizer, pad_to_multiple_of=8)
        batch = data_collator(features)

        self.assertEqual(batch["input_ids"].shape, tf.TensorShape((2, 8)))
        self.assertEqual(batch["token_type_ids"].shape, tf.TensorShape((2, 8)))
        self.assertEqual(batch["labels"].shape, tf.TensorShape((2, 8)))
        self.assertEqual(batch["next_sentence_label"].shape, tf.TensorShape((2,)))

    def test_sop(self):
        tokenizer = BertTokenizer(self.vocab_file)
        features = {"input_data": [[0, 1, 2, 3, 4] for i in range(2)],
                     "token_type_ids": [[0, 1, 2, 3, 4] for i in range(2)],
                     "sentence_prediction_label": [i for i in range(2)]}
        data_collator = TFDataCollatorForLanguageModeling(tokenizer)
        batch = data_collator(features)

        self.assertEqual(batch["input_ids"].shape, tf.TensorShape((2, 5)))
        self.assertEqual(batch["token_type_ids"].shape, tf.TensorShape((2, 5)))
        self.assertEqual(batch["labels"].shape, tf.TensorShape((2, 5)))
        self.assertEqual(batch["sentence_order_label"].shape, tf.TensorShape((2,)))

        data_collator = TFDataCollatorForLanguageModeling(tokenizer, pad_to_multiple_of=8)
        batch = data_collator(features)

        self.assertEqual(batch["input_ids"].shape, tf.TensorShape((2, 8)))
        self.assertEqual(batch["token_type_ids"].shape, tf.TensorShape((2, 8)))
        self.assertEqual(batch["labels"].shape, tf.TensorShape((2, 8)))
        self.assertEqual(batch["sentence_order_label"].shape, tf.TensorShape((2,)))
