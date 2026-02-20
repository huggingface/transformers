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
"""
Tests for data collators.

Tests are organized by collator type, with each test class containing:
- Functionality tests (PyTorch and NumPy variants)
- Immutability tests (verifying inputs are not mutated)
"""

import copy
import os
import shutil
import tempfile
import unittest

import numpy as np

from transformers import (
    BertTokenizer,
    BertTokenizerFast,
    DataCollatorForLanguageModeling,
    DataCollatorForPermutationLanguageModeling,
    DataCollatorForSeq2Seq,
    DataCollatorForTokenClassification,
    DataCollatorForWholeWordMask,
    DataCollatorWithFlattening,
    DataCollatorWithPadding,
    default_data_collator,
    is_torch_available,
    set_seed,
)
from transformers.testing_utils import require_torch
from transformers.utils import PaddingStrategy


if is_torch_available():
    import torch


class DataCollatorTestMixin:
    """Mixin providing common setup and utility methods for data collator tests."""

    def setUp(self):
        self.tmpdirname = tempfile.mkdtemp()
        vocab_tokens = ["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"]
        self.vocab_file = os.path.join(self.tmpdirname, "vocab.txt")
        with open(self.vocab_file, "w", encoding="utf-8") as vocab_writer:
            vocab_writer.write("".join([x + "\n" for x in vocab_tokens]))

    def tearDown(self):
        shutil.rmtree(self.tmpdirname)

    def _check_immutability(self, collator, features):
        """Verify that collator does not mutate input data."""
        original = copy.deepcopy(features)
        collator(features)

        for orig, feat in zip(original, features):
            for key in orig:
                orig_val, feat_val = orig[key], feat[key]
                if isinstance(orig_val, np.ndarray):
                    self.assertEqual(orig_val.tolist(), feat_val.tolist())
                elif is_torch_available() and isinstance(orig_val, torch.Tensor):
                    self.assertEqual(orig_val.tolist(), feat_val.tolist())
                else:
                    self.assertEqual(orig_val, feat_val)


# =============================================================================
# default_data_collator tests
# =============================================================================


@require_torch
class TestDefaultDataCollator(DataCollatorTestMixin, unittest.TestCase):
    """
    Tests for default_data_collator.

    The default collator handles basic batching of dict features, converting
    lists and arrays to tensors and properly handling labels.
    """

    def test_basic_collation(self):
        """Test basic dict collation with lists."""
        features = [{"label": i, "inputs": [0, 1, 2, 3, 4, 5]} for i in range(4)]
        batch = default_data_collator(features)

        self.assertEqual(batch["labels"].tolist(), list(range(4)))
        self.assertEqual(batch["labels"].dtype, torch.long)
        self.assertEqual(batch["inputs"].shape, torch.Size([4, 6]))

    def test_multi_label(self):
        """Test collation with multiple labels per sample."""
        features = [{"label_ids": [0, 1, 2], "inputs": [0, 1, 2, 3]} for _ in range(4)]
        batch = default_data_collator(features)

        self.assertEqual(batch["labels"].tolist(), [[0, 1, 2]] * 4)
        self.assertEqual(batch["labels"].dtype, torch.long)

    def test_numpy_array_inputs(self):
        """Test collation when features are numpy arrays."""
        features = [{"label": i, "inputs": np.random.randint(0, 10, [10])} for i in range(4)]
        batch = default_data_collator(features)

        self.assertEqual(batch["labels"].tolist(), list(range(4)))
        self.assertEqual(batch["inputs"].shape, torch.Size([4, 10]))

    def test_tensor_labels(self):
        """Test collation when labels are already tensors."""
        features = [{"label": torch.tensor(i), "inputs": [0, 1, 2]} for i in range(4)]
        batch = default_data_collator(features)

        self.assertEqual(batch["labels"].dtype, torch.long)
        self.assertEqual(batch["labels"].tolist(), list(range(4)))

    def test_dtype_inference(self):
        """Test that int labels become long, float labels become float."""
        # Classification: int -> long
        features = [{"input_ids": [0, 1, 2], "label": i} for i in range(4)]
        batch = default_data_collator(features)
        self.assertEqual(batch["labels"].dtype, torch.long)

        # Regression: float -> float
        features = [{"input_ids": [0, 1, 2], "label": float(i)} for i in range(4)]
        batch = default_data_collator(features)
        self.assertEqual(batch["labels"].dtype, torch.float)

    def test_none_labels_excluded(self):
        """Test that None labels are excluded from batch."""
        features = [{"label": None, "inputs": [0, 1, 2, 3]} for _ in range(4)]
        batch = default_data_collator(features)
        self.assertNotIn("labels", batch)

        # With label_ids
        features = [{"label_ids": None, "inputs": [0, 1, 2, 3]} for _ in range(4)]
        batch = default_data_collator(features)
        self.assertNotIn("labels", batch)

    def test_numpy_output(self):
        """Test collation with NumPy output."""
        features = [{"label": i, "inputs": [0, 1, 2, 3]} for i in range(4)]
        batch = default_data_collator(features, return_tensors="np")

        self.assertEqual(batch["labels"].tolist(), list(range(4)))
        self.assertEqual(batch["labels"].dtype, np.int64)
        self.assertEqual(batch["inputs"].shape, (4, 4))

    def test_numpy_dtype_inference(self):
        """Test dtype inference with NumPy output."""
        # Int labels
        features = [{"input_ids": [0, 1, 2], "label": i} for i in range(4)]
        batch = default_data_collator(features, return_tensors="np")
        self.assertEqual(batch["labels"].dtype, np.int64)

        # Float labels
        features = [{"input_ids": [0, 1, 2], "label": float(i)} for i in range(4)]
        batch = default_data_collator(features, return_tensors="np")
        self.assertEqual(batch["labels"].dtype, np.float32)

    def test_immutability(self):
        """Test that collation does not mutate input data."""

        def collator_pt(x):
            return default_data_collator(x, return_tensors="pt")

        def collator_np(x):
            return default_data_collator(x, return_tensors="np")

        for collator in [collator_pt, collator_np]:
            # Test with various input types
            for features in [
                [{"label": i, "inputs": [0, 1, 2, 3]} for i in range(4)],
                [{"label": float(i), "inputs": [0, 1, 2, 3]} for i in range(4)],
                [{"label": None, "inputs": [0, 1, 2, 3]} for _ in range(4)],
                [{"label_ids": [0, 1, 2], "inputs": [0, 1, 2, 3]} for _ in range(4)],
            ]:
                self._check_immutability(collator, features)


# =============================================================================
# DataCollatorWithPadding tests
# =============================================================================


@require_torch
class TestDataCollatorWithPadding(DataCollatorTestMixin, unittest.TestCase):
    """
    Tests for DataCollatorWithPadding.

    This collator pads sequences to the same length within a batch, supporting
    dynamic padding, max_length, and pad_to_multiple_of options.
    """

    def test_dynamic_padding(self):
        """Test padding to longest sequence in batch."""
        tokenizer = BertTokenizer(self.vocab_file)
        features = [{"input_ids": [0, 1, 2]}, {"input_ids": [0, 1, 2, 3, 4, 5]}]

        collator = DataCollatorWithPadding(tokenizer)
        batch = collator(features)

        self.assertEqual(batch["input_ids"].shape, torch.Size([2, 6]))
        self.assertEqual(batch["input_ids"][0].tolist(), [0, 1, 2] + [tokenizer.pad_token_id] * 3)

    def test_max_length_padding(self):
        """Test padding to specified max_length."""
        tokenizer = BertTokenizer(self.vocab_file)
        features = [{"input_ids": [0, 1, 2]}, {"input_ids": [0, 1, 2, 3, 4, 5]}]

        collator = DataCollatorWithPadding(tokenizer, padding="max_length", max_length=10)
        batch = collator(features)

        self.assertEqual(batch["input_ids"].shape, torch.Size([2, 10]))

    def test_pad_to_multiple_of(self):
        """Test padding to multiple of specified value."""
        tokenizer = BertTokenizer(self.vocab_file)
        features = [{"input_ids": [0, 1, 2]}, {"input_ids": [0, 1, 2, 3, 4, 5]}]

        collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)
        batch = collator(features)

        self.assertEqual(batch["input_ids"].shape, torch.Size([2, 8]))

    def test_numpy_output(self):
        """Test padding with NumPy output."""
        tokenizer = BertTokenizer(self.vocab_file)
        features = [{"input_ids": [0, 1, 2]}, {"input_ids": [0, 1, 2, 3, 4, 5]}]

        collator = DataCollatorWithPadding(tokenizer, return_tensors="np")
        batch = collator(features)

        self.assertEqual(batch["input_ids"].shape, (2, 6))
        self.assertEqual(batch["input_ids"][0].tolist(), [0, 1, 2] + [tokenizer.pad_token_id] * 3)

    def test_attention_mask_generated(self):
        """Test that attention_mask is properly generated."""
        tokenizer = BertTokenizer(self.vocab_file)
        features = [{"input_ids": [0, 1, 2]}, {"input_ids": [0, 1, 2, 3, 4, 5]}]

        collator = DataCollatorWithPadding(tokenizer)
        batch = collator(features)

        self.assertIn("attention_mask", batch)
        self.assertEqual(batch["attention_mask"][0].tolist(), [1, 1, 1, 0, 0, 0])
        self.assertEqual(batch["attention_mask"][1].tolist(), [1, 1, 1, 1, 1, 1])

    def test_immutability(self):
        """Test that padding does not mutate input data."""
        tokenizer = BertTokenizer(self.vocab_file)

        for return_tensors in ["pt", "np"]:
            collator = DataCollatorWithPadding(tokenizer, return_tensors=return_tensors)
            features = [{"input_ids": [0, 1, 2]}, {"input_ids": [0, 1, 2, 3, 4, 5]}]
            self._check_immutability(collator, features)


# =============================================================================
# DataCollatorWithFlattening tests
# =============================================================================


@require_torch
class TestDataCollatorWithFlattening(DataCollatorTestMixin, unittest.TestCase):
    """
    Tests for DataCollatorWithFlattening.

    This collator concatenates multiple sequences into a single sequence,
    useful for packing multiple examples efficiently. It generates position_ids
    that reset for each original sequence.
    """

    def _get_features(self):
        return [
            {"input_ids": [10, 11, 12]},
            {"input_ids": [20, 21, 22, 23, 24, 25]},
            {"input_ids": [30, 31, 32, 33, 34, 35, 36]},
        ]

    def test_basic_flattening(self):
        """Test that sequences are concatenated with per-sequence position_ids."""
        collator = DataCollatorWithFlattening(return_tensors="pt")
        batch = collator(self._get_features())

        self.assertEqual(batch["input_ids"].shape, torch.Size([1, 16]))
        self.assertEqual(
            batch["input_ids"][0].tolist(),
            [10, 11, 12, 20, 21, 22, 23, 24, 25, 30, 31, 32, 33, 34, 35, 36],
        )
        self.assertEqual(batch["position_ids"][0].tolist(), [0, 1, 2, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 6])

        # Should not include attention_mask or flash attn kwargs by default
        for key in ["attention_mask", "cu_seq_lens_k", "cu_seq_lens_q", "seq_idx"]:
            self.assertNotIn(key, batch)

    def test_flash_attn_kwargs(self):
        """Test flattening with Flash Attention kwargs."""
        collator = DataCollatorWithFlattening(return_tensors="pt", return_flash_attn_kwargs=True)
        batch = collator(self._get_features())

        self.assertEqual(batch["cu_seq_lens_k"].tolist(), [0, 3, 9, 16])
        self.assertEqual(batch["cu_seq_lens_q"].tolist(), [0, 3, 9, 16])
        self.assertEqual(batch["max_length_k"], 7)
        self.assertEqual(batch["max_length_q"], 7)

    def test_seq_idx(self):
        """Test flattening with seq_idx for sequence identification."""
        collator = DataCollatorWithFlattening(return_tensors="pt", return_seq_idx=True)
        batch = collator(self._get_features())

        self.assertEqual(batch["seq_idx"][0].tolist(), [0, 0, 0, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2])

    def test_with_labels(self):
        """Test flattening with tensor and list labels."""
        # Tensor labels
        features = [
            {"input_ids": torch.tensor([1, 2, 3]), "labels": torch.tensor([10, 11, 12])},
            {"input_ids": torch.tensor([4, 5]), "labels": torch.tensor([13, 14])},
        ]
        collator = DataCollatorWithFlattening(return_tensors="pt")
        batch = collator(features)

        self.assertEqual(batch["input_ids"].shape, (1, 5))
        self.assertEqual(batch["labels"].shape, (1, 5))

        # List labels
        features = [
            {"input_ids": [1, 2, 3], "labels": [10, 11, 12]},
            {"input_ids": [4, 5], "labels": [13, 14]},
        ]
        batch = collator(features)
        self.assertEqual(batch["labels"].shape, (1, 5))

    def test_numpy_output(self):
        """Test flattening with NumPy output."""
        collator = DataCollatorWithFlattening(return_tensors="np")
        batch = collator(self._get_features())

        self.assertEqual(batch["input_ids"].shape, (1, 16))
        self.assertEqual(batch["position_ids"].shape, (1, 16))

    def test_numpy_flash_attn_kwargs(self):
        """Test flattening with Flash Attention kwargs and NumPy output."""
        collator = DataCollatorWithFlattening(return_tensors="np", return_flash_attn_kwargs=True)
        batch = collator(self._get_features())

        self.assertEqual(batch["cu_seq_lens_k"].tolist(), [0, 3, 9, 16])
        self.assertEqual(batch["max_length_k"], 7)

    def test_immutability(self):
        """Test that flattening does not mutate input data."""
        for return_tensors in ["pt", "np"]:
            collator = DataCollatorWithFlattening(return_tensors=return_tensors)
            self._check_immutability(collator, self._get_features())


# =============================================================================
# DataCollatorForTokenClassification tests
# =============================================================================


@require_torch
class TestDataCollatorForTokenClassification(DataCollatorTestMixin, unittest.TestCase):
    """
    Tests for DataCollatorForTokenClassification.

    This collator pads both input_ids and labels for token classification tasks,
    using -100 as the default label padding value (ignored by loss functions).
    """

    def _get_features(self):
        return [
            {"input_ids": [0, 1, 2], "labels": [0, 1, 2]},
            {"input_ids": [0, 1, 2, 3, 4, 5], "labels": [0, 1, 2, 3, 4, 5]},
        ]

    def test_padding(self):
        """Test that both input_ids and labels are padded correctly."""
        tokenizer = BertTokenizer(self.vocab_file)
        collator = DataCollatorForTokenClassification(tokenizer)
        batch = collator(self._get_features())

        self.assertEqual(batch["input_ids"].shape, torch.Size([2, 6]))
        self.assertEqual(batch["input_ids"][0].tolist(), [0, 1, 2] + [tokenizer.pad_token_id] * 3)
        self.assertEqual(batch["labels"].shape, torch.Size([2, 6]))
        self.assertEqual(batch["labels"][0].tolist(), [0, 1, 2, -100, -100, -100])

    def test_max_length_padding(self):
        """Test padding to max_length."""
        tokenizer = BertTokenizer(self.vocab_file)
        collator = DataCollatorForTokenClassification(tokenizer, padding="max_length", max_length=10)
        batch = collator(self._get_features())

        self.assertEqual(batch["input_ids"].shape, torch.Size([2, 10]))
        self.assertEqual(batch["labels"].shape, torch.Size([2, 10]))

    def test_pad_to_multiple_of(self):
        """Test padding to multiple of 8."""
        tokenizer = BertTokenizer(self.vocab_file)
        collator = DataCollatorForTokenClassification(tokenizer, pad_to_multiple_of=8)
        batch = collator(self._get_features())

        self.assertEqual(batch["input_ids"].shape, torch.Size([2, 8]))
        self.assertEqual(batch["labels"].shape, torch.Size([2, 8]))

    def test_custom_label_pad_token(self):
        """Test custom label padding token."""
        tokenizer = BertTokenizer(self.vocab_file)
        collator = DataCollatorForTokenClassification(tokenizer, label_pad_token_id=-1)
        batch = collator(self._get_features())

        self.assertEqual(batch["labels"][0].tolist(), [0, 1, 2, -1, -1, -1])

    def test_without_labels(self):
        """Test collator works without labels (inference mode)."""
        tokenizer = BertTokenizer(self.vocab_file)
        features = [{"input_ids": [0, 1, 2]}, {"input_ids": [0, 1, 2, 3, 4, 5]}]

        collator = DataCollatorForTokenClassification(tokenizer)
        batch = collator(features)

        self.assertEqual(batch["input_ids"].shape, torch.Size([2, 6]))
        self.assertNotIn("labels", batch)

    def test_with_tensor_inputs(self):
        """Test with PyTorch tensor inputs."""
        tokenizer = BertTokenizer(self.vocab_file)
        features = [
            {"input_ids": torch.tensor([0, 1, 2]), "labels": torch.tensor([0, 1, 2])},
            {"input_ids": torch.tensor([0, 1, 2, 3, 4, 5]), "labels": torch.tensor([0, 1, 2, 3, 4, 5])},
        ]

        collator = DataCollatorForTokenClassification(tokenizer)
        batch = collator(features)

        self.assertEqual(batch["labels"][0].tolist(), [0, 1, 2, -100, -100, -100])

    def test_numpy_output(self):
        """Test with NumPy output."""
        tokenizer = BertTokenizer(self.vocab_file)
        collator = DataCollatorForTokenClassification(tokenizer, return_tensors="np")
        batch = collator(self._get_features())

        self.assertEqual(batch["input_ids"].shape, (2, 6))
        self.assertEqual(batch["labels"][0].tolist(), [0, 1, 2, -100, -100, -100])

    def test_immutability(self):
        """Test that collation does not mutate input data."""
        tokenizer = BertTokenizer(self.vocab_file)

        for return_tensors in ["pt", "np"]:
            collator = DataCollatorForTokenClassification(tokenizer, return_tensors=return_tensors)
            self._check_immutability(collator, self._get_features())


# =============================================================================
# DataCollatorForSeq2Seq tests
# =============================================================================


@require_torch
class TestDataCollatorForSeq2Seq(DataCollatorTestMixin, unittest.TestCase):
    """
    Tests for DataCollatorForSeq2Seq.

    This collator handles encoder-decoder models, padding both input sequences
    and labels (decoder input) appropriately.
    """

    def _get_features(self):
        return [
            {"input_ids": list(range(3)), "labels": list(range(3))},
            {"input_ids": list(range(6)), "labels": list(range(6))},
        ]

    def test_padding(self):
        """Test basic padding behavior."""
        tokenizer = BertTokenizer(self.vocab_file)
        collator = DataCollatorForSeq2Seq(tokenizer, padding=PaddingStrategy.LONGEST)
        batch = collator(self._get_features())

        self.assertEqual(batch["input_ids"].shape, torch.Size([2, 6]))
        self.assertEqual(batch["labels"].shape, torch.Size([2, 6]))
        self.assertEqual(batch["input_ids"][0].tolist(), [0, 1, 2] + [tokenizer.pad_token_id] * 3)
        self.assertEqual(batch["labels"][0].tolist(), [0, 1, 2, -100, -100, -100])

    def test_with_tensor_inputs(self):
        """Test with PyTorch tensor inputs."""
        tokenizer = BertTokenizer(self.vocab_file)
        features = [
            {"input_ids": torch.tensor(list(range(3))), "labels": torch.tensor(list(range(3)))},
            {"input_ids": torch.tensor(list(range(6))), "labels": torch.tensor(list(range(6)))},
        ]

        collator = DataCollatorForSeq2Seq(tokenizer, padding=PaddingStrategy.LONGEST)
        batch = collator(features)

        self.assertEqual(batch["input_ids"].shape, torch.Size([2, 6]))
        self.assertEqual(batch["labels"].shape, torch.Size([2, 6]))

    def test_max_length_padding(self):
        """Test padding to max_length."""
        tokenizer = BertTokenizer(self.vocab_file)
        collator = DataCollatorForSeq2Seq(tokenizer, padding=PaddingStrategy.MAX_LENGTH, max_length=10)
        batch = collator(self._get_features())

        self.assertEqual(batch["input_ids"].shape, torch.Size([2, 10]))
        self.assertEqual(batch["labels"].shape, torch.Size([2, 10]))

    def test_pad_to_multiple_of(self):
        """Test padding to multiple of 8."""
        tokenizer = BertTokenizer(self.vocab_file)
        collator = DataCollatorForSeq2Seq(tokenizer, padding=PaddingStrategy.LONGEST, pad_to_multiple_of=8)
        batch = collator(self._get_features())

        self.assertEqual(batch["input_ids"].shape, torch.Size([2, 8]))
        self.assertEqual(batch["labels"].shape, torch.Size([2, 8]))

    def test_custom_label_pad_token(self):
        """Test custom label padding token."""
        tokenizer = BertTokenizer(self.vocab_file)
        collator = DataCollatorForSeq2Seq(tokenizer, padding=PaddingStrategy.LONGEST, label_pad_token_id=-1)
        batch = collator(self._get_features())

        self.assertEqual(batch["labels"][0].tolist(), [0, 1, 2, -1, -1, -1])

    def test_do_not_pad(self):
        """Test DO_NOT_PAD raises on unequal lengths, works on equal."""
        tokenizer = BertTokenizer(self.vocab_file)
        collator = DataCollatorForSeq2Seq(tokenizer, padding=PaddingStrategy.DO_NOT_PAD)

        # Unequal lengths should raise
        with self.assertRaises(ValueError):
            collator(self._get_features())

        # Equal lengths should work
        features = [{"input_ids": list(range(3)), "labels": list(range(3))}] * 2
        batch = collator(features)
        self.assertEqual(batch["input_ids"][0].tolist(), list(range(3)))
        self.assertEqual(batch["labels"][0].tolist(), list(range(3)))

    def test_without_labels(self):
        """Test collator works without labels."""
        tokenizer = BertTokenizer(self.vocab_file)
        features = [{"input_ids": list(range(3))}, {"input_ids": list(range(6))}]

        collator = DataCollatorForSeq2Seq(tokenizer, padding=PaddingStrategy.LONGEST)
        batch = collator(features)

        self.assertEqual(batch["input_ids"].shape, torch.Size([2, 6]))
        # Labels should either not be present or be None
        self.assertTrue("labels" not in batch or batch["labels"] is None)

    def test_numpy_output(self):
        """Test with NumPy output."""
        tokenizer = BertTokenizer(self.vocab_file)
        collator = DataCollatorForSeq2Seq(tokenizer, padding=PaddingStrategy.LONGEST, return_tensors="np")
        batch = collator(self._get_features())

        self.assertEqual(batch["input_ids"].shape, (2, 6))
        self.assertEqual(batch["labels"].shape, (2, 6))

    def test_immutability(self):
        """Test that collation does not mutate input data."""
        tokenizer = BertTokenizer(self.vocab_file)

        for return_tensors in ["pt", "np"]:
            collator = DataCollatorForSeq2Seq(
                tokenizer, padding=PaddingStrategy.LONGEST, return_tensors=return_tensors
            )
            self._check_immutability(collator, self._get_features())


# =============================================================================
# DataCollatorForLanguageModeling tests
# =============================================================================


@require_torch
class TestDataCollatorForLanguageModeling(DataCollatorTestMixin, unittest.TestCase):
    """
    Tests for DataCollatorForLanguageModeling.

    This collator supports both Masked Language Modeling (MLM) and Causal Language
    Modeling (CLM). For MLM, it randomly masks tokens; for CLM, it shifts labels.
    """

    def test_clm_mode(self):
        """Test CLM mode (no masking)."""
        tokenizer = BertTokenizer(self.vocab_file)
        features = [{"input_ids": list(range(10))}, {"input_ids": list(range(10))}]

        collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
        batch = collator(features)

        self.assertEqual(batch["input_ids"].shape, torch.Size([2, 10]))
        self.assertEqual(batch["labels"].shape, torch.Size([2, 10]))

    def test_clm_with_padding(self):
        """Test CLM mode with different length sequences."""
        tokenizer = BertTokenizer(self.vocab_file)
        features = [{"input_ids": list(range(5))}, {"input_ids": list(range(10))}]

        collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
        batch = collator(features)

        self.assertEqual(batch["input_ids"].shape, torch.Size([2, 10]))

    def test_clm_pad_to_multiple_of(self):
        """Test CLM with pad_to_multiple_of."""
        tokenizer = BertTokenizer(self.vocab_file)
        features = [{"input_ids": list(range(10))}, {"input_ids": list(range(10))}]

        collator = DataCollatorForLanguageModeling(tokenizer, mlm=False, pad_to_multiple_of=8)
        batch = collator(features)

        self.assertEqual(batch["input_ids"].shape, torch.Size([2, 16]))

    def test_mlm_mode(self):
        """Test MLM mode with masking."""
        set_seed(42)
        tokenizer = BertTokenizer(self.vocab_file)
        features = [{"input_ids": list(range(10))}, {"input_ids": list(range(10))}]

        collator = DataCollatorForLanguageModeling(tokenizer, mlm=True)
        batch = collator(features)

        self.assertEqual(batch["input_ids"].shape, torch.Size([2, 10]))
        self.assertEqual(batch["labels"].shape, torch.Size([2, 10]))

        # Check that masking occurred and non-masked tokens have -100 labels
        masked_tokens = batch["input_ids"] == tokenizer.mask_token_id
        self.assertTrue(torch.any(masked_tokens))
        self.assertTrue(all(x == -100 for x in batch["labels"][~masked_tokens].tolist()))

    def test_mlm_with_padding(self):
        """Test MLM mode with different-length sequences requiring padding."""
        set_seed(42)
        tokenizer = BertTokenizer(self.vocab_file)
        features = [{"input_ids": list(range(5))}, {"input_ids": list(range(10))}]

        collator = DataCollatorForLanguageModeling(tokenizer, mlm=True)
        batch = collator(features)

        self.assertEqual(batch["input_ids"].shape, torch.Size([2, 10]))
        masked_tokens = batch["input_ids"] == tokenizer.mask_token_id
        self.assertTrue(torch.any(masked_tokens))
        self.assertTrue(all(x == -100 for x in batch["labels"][~masked_tokens].tolist()))

    def test_mlm_pad_to_multiple_of(self):
        """Test MLM mode with pad_to_multiple_of."""
        set_seed(42)
        tokenizer = BertTokenizer(self.vocab_file)
        features = [{"input_ids": list(range(10))}, {"input_ids": list(range(10))}]

        collator = DataCollatorForLanguageModeling(tokenizer, mlm=True, pad_to_multiple_of=8)
        batch = collator(features)

        self.assertEqual(batch["input_ids"].shape, torch.Size([2, 16]))
        masked_tokens = batch["input_ids"] == tokenizer.mask_token_id
        self.assertTrue(torch.any(masked_tokens))
        self.assertTrue(all(x == -100 for x in batch["labels"][~masked_tokens].tolist()))

    def test_with_raw_list_features(self):
        """Test LM collator with raw list features (not dicts)."""
        tokenizer = BertTokenizer(self.vocab_file)

        # CLM with raw lists
        collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
        features = [list(range(10)), list(range(10))]
        batch = collator(features)
        self.assertEqual(batch["input_ids"].shape, torch.Size([2, 10]))

        # CLM with raw lists requiring padding
        features = [list(range(5)), list(range(10))]
        batch = collator(features)
        self.assertEqual(batch["input_ids"].shape, torch.Size([2, 10]))

    def test_mlm_seed_reproducibility(self):
        """Test that masking is reproducible with seed."""
        tokenizer = BertTokenizer(self.vocab_file)
        features = [{"input_ids": list(range(1000))}, {"input_ids": list(range(1000))}]

        collator1 = DataCollatorForLanguageModeling(tokenizer, seed=42)
        batch1 = collator1(features)

        collator2 = DataCollatorForLanguageModeling(tokenizer, seed=42)
        batch2 = collator2(features)

        self.assertTrue(torch.all(batch1["input_ids"] == batch2["input_ids"]))
        self.assertTrue(torch.all(batch1["labels"] == batch2["labels"]))

        # Different seed -> different results
        collator3 = DataCollatorForLanguageModeling(tokenizer, seed=43)
        batch3 = collator3(features)
        self.assertFalse(torch.all(batch1["input_ids"] == batch3["input_ids"]))

    def test_mlm_multiworker_dataloader(self):
        """Test seed works with multi-worker DataLoader."""
        tokenizer = BertTokenizer(self.vocab_file)
        features = [{"input_ids": list(range(1000))} for _ in range(10)]

        dataloader = torch.utils.data.DataLoader(
            features,
            batch_size=2,
            num_workers=2,
            generator=torch.Generator().manual_seed(42),
            collate_fn=DataCollatorForLanguageModeling(tokenizer, seed=42),
        )

        batches = [batch["input_ids"] for batch in dataloader]
        result = torch.stack(batches)
        self.assertEqual(result.shape, torch.Size([5, 2, 1000]))

    def test_missing_pad_token_error(self):
        """Test error when pad token is missing and padding is needed."""
        tokenizer = BertTokenizer(self.vocab_file)
        tokenizer.pad_token = None
        features = [{"input_ids": list(range(5))}, {"input_ids": list(range(10))}]

        collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
        with self.assertRaises(ValueError):
            collator(features)

    def test_numpy_output(self):
        """Test with NumPy output."""
        tokenizer = BertTokenizer(self.vocab_file)
        features = [{"input_ids": list(range(10))}, {"input_ids": list(range(10))}]

        collator = DataCollatorForLanguageModeling(tokenizer, mlm=False, return_tensors="np")
        batch = collator(features)

        self.assertEqual(batch["input_ids"].shape, (2, 10))
        self.assertEqual(batch["labels"].shape, (2, 10))

    def test_numpy_mlm(self):
        """Test MLM mode with NumPy output."""
        set_seed(42)
        tokenizer = BertTokenizer(self.vocab_file)
        features = [{"input_ids": list(range(10))}, {"input_ids": list(range(10))}]

        collator = DataCollatorForLanguageModeling(tokenizer, mlm=True, return_tensors="np")
        batch = collator(features)

        masked_tokens = batch["input_ids"] == tokenizer.mask_token_id
        self.assertTrue(np.any(masked_tokens))

    def test_immutability(self):
        """Test that collation does not mutate input data."""
        tokenizer = BertTokenizer(self.vocab_file)
        features = [{"input_ids": list(range(10))}, {"input_ids": list(range(10))}]

        for return_tensors in ["pt", "np"]:
            collator = DataCollatorForLanguageModeling(tokenizer, mlm=False, return_tensors=return_tensors)
            self._check_immutability(collator, copy.deepcopy(features))

    # -------------------- Unit tests for internal methods --------------------

    def test_calc_word_ids_and_prob_mask(self):
        """Test word ID assignment and probability mask generation."""
        offsets = np.array(
            [
                [(0, 0), (0, 3), (3, 4), (5, 6), (6, 7), (8, 9)],
                [(0, 0), (0, 3), (3, 4), (5, 6), (6, 7), (0, 0)],
                [(0, 0), (0, 3), (3, 4), (0, 0), (6, 7), (0, 0)],
                [(1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7)],
                [(1, 1), (2, 2), (3, 4), (5, 6), (7, 8), (9, 10)],
                [(0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0)],
            ]
        )

        special_tokens_mask = np.array(
            [
                [1, 0, 0, 0, 0, 0],
                [1, 0, 0, 0, 0, 1],
                [1, 0, 0, 1, 0, 1],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [1, 1, 1, 1, 1, 1],
            ]
        )

        word_ids, prob_mask = DataCollatorForLanguageModeling._calc_word_ids_and_prob_mask(
            offsets, special_tokens_mask
        )

        expected_word_ids = np.array(
            [
                [-1, 1, 1, 2, 2, 3],
                [-1, 1, 1, 2, 2, -1],
                [-1, 1, 1, -1, 2, -1],
                [1, 1, 1, 1, 1, 1],
                [1, 2, 3, 4, 5, 6],
                [-1, -1, -1, -1, -1, -1],
            ]
        )

        expected_prob_mask = np.array(
            [
                [1, 0, 1, 0, 1, 0],
                [1, 0, 1, 0, 1, 1],
                [1, 0, 1, 1, 0, 1],
                [0, 1, 1, 1, 1, 1],
                [0, 0, 0, 0, 0, 0],
                [1, 1, 1, 1, 1, 1],
            ]
        )

        np.testing.assert_array_equal(word_ids, expected_word_ids)
        np.testing.assert_array_equal(prob_mask, expected_prob_mask)

    def test_whole_word_mask_internal(self):
        """Test mask expansion to cover all subword tokens of masked words."""
        word_ids = np.array(
            [
                [-1, 1, 1, 2, 2, 3],
                [-1, 1, 1, 2, 2, -1],
                [-1, 1, 1, -1, 2, -1],
                [1, 1, 1, 1, 1, 1],
                [1, 2, 3, 4, 5, 6],
                [1, 2, 3, 4, 5, 6],
                [-1, -1, -1, -1, -1, -1],
            ]
        )

        mask = np.array(
            [
                [0, 1, 0, 0, 0, 0],
                [0, 1, 0, 1, 0, 0],
                [0, 0, 0, 0, 1, 0],
                [1, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 1, 0, 1, 0, 1],
                [0, 0, 0, 0, 0, 0],
            ]
        ).astype(bool)

        output = DataCollatorForLanguageModeling._whole_word_mask(word_ids, mask)

        expected = np.array(
            [
                [0, 1, 1, 0, 0, 0],
                [0, 1, 1, 1, 1, 0],
                [0, 0, 0, 0, 1, 0],
                [1, 1, 1, 1, 1, 1],
                [0, 0, 0, 0, 0, 0],
                [0, 1, 0, 1, 0, 1],
                [0, 0, 0, 0, 0, 0],
            ]
        ).astype(bool)

        np.testing.assert_array_equal(output, expected)


# =============================================================================
# DataCollatorForWholeWordMask tests
# =============================================================================


@require_torch
class TestDataCollatorForWholeWordMask(DataCollatorTestMixin, unittest.TestCase):
    """
    Tests for DataCollatorForWholeWordMask.

    This collator extends MLM to ensure that when a token is masked, all other
    tokens from the same word are also masked (whole word masking).
    """

    def _get_tokenizer_and_features(self):
        tokenizer = BertTokenizerFast(self.vocab_file)
        input_tokens = [f"token_{i}" for i in range(8)]
        tokenizer.add_tokens(input_tokens)
        features = [tokenizer(" ".join(input_tokens), return_offsets_mapping=True) for _ in range(2)]
        return tokenizer, features

    def test_basic(self):
        """Test whole word masking masks complete words."""
        tokenizer, features = self._get_tokenizer_and_features()
        collator = DataCollatorForWholeWordMask(tokenizer, return_tensors="pt")
        batch = collator(features)

        self.assertEqual(batch["input_ids"].shape, (2, 10))
        self.assertEqual(batch["labels"].shape, (2, 10))

    def test_with_numpy_inputs(self):
        """Test with numpy array inputs."""
        tokenizer, _ = self._get_tokenizer_and_features()
        input_tokens = [f"token_{i}" for i in range(8)]
        features = [
            tokenizer(" ".join(input_tokens), return_offsets_mapping=True).convert_to_tensors("np") for _ in range(2)
        ]

        collator = DataCollatorForWholeWordMask(tokenizer, return_tensors="pt")
        batch = collator(features)

        self.assertEqual(batch["input_ids"].shape, torch.Size((2, 10)))

    def test_with_tensor_inputs(self):
        """Test with PyTorch tensor inputs."""
        tokenizer, _ = self._get_tokenizer_and_features()
        input_tokens = [f"token_{i}" for i in range(8)]
        features = [
            tokenizer(" ".join(input_tokens), return_offsets_mapping=True).convert_to_tensors("pt") for _ in range(2)
        ]

        collator = DataCollatorForWholeWordMask(tokenizer, return_tensors="pt")
        batch = collator(features)

        self.assertEqual(batch["input_ids"].shape, torch.Size((2, 10)))

    def test_seed_reproducibility(self):
        """Test reproducibility with seed."""
        tokenizer = BertTokenizerFast(self.vocab_file)
        input_tokens = [f"token_{i}" for i in range(998)]
        tokenizer.add_tokens(input_tokens)
        features = [tokenizer(" ".join(input_tokens), return_offsets_mapping=True) for _ in range(2)]

        collator1 = DataCollatorForWholeWordMask(tokenizer, seed=42, return_tensors="np")
        batch1 = collator1(features)

        collator2 = DataCollatorForWholeWordMask(tokenizer, seed=42, return_tensors="np")
        batch2 = collator2(features)

        np.testing.assert_array_equal(batch1["input_ids"], batch2["input_ids"])

        # Different seed -> different results
        collator3 = DataCollatorForWholeWordMask(tokenizer, seed=43, return_tensors="np")
        batch3 = collator3(features)
        self.assertFalse(np.all(batch1["input_ids"] == batch3["input_ids"]))

    def test_seed_multiworker_dataloader(self):
        """Test seed reproducibility with multi-worker DataLoader."""
        tokenizer = BertTokenizerFast(self.vocab_file)
        input_tokens = [f"token_{i}" for i in range(998)]
        tokenizer.add_tokens(input_tokens)
        features = [tokenizer(" ".join(input_tokens), return_offsets_mapping=True) for _ in range(10)]

        dataloader1 = torch.utils.data.DataLoader(
            features,
            batch_size=2,
            num_workers=2,
            generator=torch.Generator().manual_seed(42),
            collate_fn=DataCollatorForWholeWordMask(tokenizer, seed=42),
        )
        batches1 = torch.stack([batch["input_ids"] for batch in dataloader1])

        dataloader2 = torch.utils.data.DataLoader(
            features,
            batch_size=2,
            num_workers=2,
            collate_fn=DataCollatorForWholeWordMask(tokenizer, seed=42),
        )
        batches2 = torch.stack([batch["input_ids"] for batch in dataloader2])

        self.assertTrue(torch.all(batches1 == batches2))

        # Different seed -> different results
        dataloader3 = torch.utils.data.DataLoader(
            features,
            batch_size=2,
            num_workers=2,
            collate_fn=DataCollatorForWholeWordMask(tokenizer, seed=43),
        )
        batches3 = torch.stack([batch["input_ids"] for batch in dataloader3])
        self.assertFalse(torch.all(batches1 == batches3))

    def test_numpy_output(self):
        """Test with NumPy output."""
        tokenizer, features = self._get_tokenizer_and_features()
        collator = DataCollatorForWholeWordMask(tokenizer, return_tensors="np")
        batch = collator(features)

        self.assertEqual(batch["input_ids"].shape, (2, 10))
        self.assertEqual(batch["labels"].shape, (2, 10))

    def test_immutability(self):
        """Test that collation does not mutate input data."""
        tokenizer, features = self._get_tokenizer_and_features()
        features = [dict(f) for f in features]

        for return_tensors in ["pt", "np"]:
            collator = DataCollatorForWholeWordMask(tokenizer, return_tensors=return_tensors)
            self._check_immutability(collator, copy.deepcopy(features))


# =============================================================================
# DataCollatorForPermutationLanguageModeling tests
# =============================================================================


@require_torch
class TestDataCollatorForPermutationLanguageModeling(DataCollatorTestMixin, unittest.TestCase):
    """
    Tests for DataCollatorForPermutationLanguageModeling.

    This collator implements XLNet-style permutation language modeling,
    generating perm_mask and target_mapping for each batch.
    """

    def test_basic(self):
        """Test basic PLM collation."""
        tokenizer = BertTokenizer(self.vocab_file)
        features = [{"input_ids": list(range(10))}, {"input_ids": list(range(10))}]

        collator = DataCollatorForPermutationLanguageModeling(tokenizer)
        batch = collator(features)

        self.assertEqual(batch["input_ids"].shape, torch.Size([2, 10]))
        self.assertEqual(batch["perm_mask"].shape, torch.Size([2, 10, 10]))
        self.assertEqual(batch["target_mapping"].shape, torch.Size([2, 10, 10]))
        self.assertEqual(batch["labels"].shape, torch.Size([2, 10]))

    def test_with_padding(self):
        """Test PLM with different length sequences."""
        tokenizer = BertTokenizer(self.vocab_file)
        features = [{"input_ids": list(range(4))}, {"input_ids": list(range(10))}]

        collator = DataCollatorForPermutationLanguageModeling(tokenizer)
        batch = collator(features)

        self.assertEqual(batch["input_ids"].shape, torch.Size([2, 10]))

    def test_odd_sequence_error(self):
        """Test that odd sequence lengths raise an error."""
        tokenizer = BertTokenizer(self.vocab_file)
        collator = DataCollatorForPermutationLanguageModeling(tokenizer)

        features = [{"input_ids": list(range(5))}]
        with self.assertRaises(ValueError):
            collator(features)

    def test_numpy_output(self):
        """Test with NumPy output."""
        tokenizer = BertTokenizer(self.vocab_file)
        features = [{"input_ids": list(range(10))}, {"input_ids": list(range(10))}]

        collator = DataCollatorForPermutationLanguageModeling(tokenizer, return_tensors="np")
        batch = collator(features)

        self.assertEqual(batch["input_ids"].shape, (2, 10))
        self.assertEqual(batch["perm_mask"].shape, (2, 10, 10))
        self.assertEqual(batch["target_mapping"].shape, (2, 10, 10))

    def test_immutability(self):
        """Test that collation does not mutate input data."""
        tokenizer = BertTokenizer(self.vocab_file)
        features = [{"input_ids": list(range(10))}, {"input_ids": list(range(10))}]

        for return_tensors in ["pt", "np"]:
            collator = DataCollatorForPermutationLanguageModeling(tokenizer, return_tensors=return_tensors)
            self._check_immutability(collator, copy.deepcopy(features))


# =============================================================================
# Next Sentence Prediction tests
# =============================================================================


@require_torch
class TestNextSentencePrediction(DataCollatorTestMixin, unittest.TestCase):
    """
    Tests for Next Sentence Prediction (NSP) with DataCollatorForLanguageModeling.

    NSP is used in BERT pretraining where the model predicts if two sentences
    follow each other in the original text.
    """

    def _get_features(self):
        return [
            {"input_ids": [0, 1, 2, 3, 4], "token_type_ids": [0, 1, 2, 3, 4], "next_sentence_label": i}
            for i in range(2)
        ]

    def test_nsp(self):
        """Test NSP labels are preserved during collation."""
        tokenizer = BertTokenizer(self.vocab_file)
        collator = DataCollatorForLanguageModeling(tokenizer)
        batch = collator(self._get_features())

        self.assertEqual(batch["input_ids"].shape, torch.Size([2, 5]))
        self.assertEqual(batch["token_type_ids"].shape, torch.Size([2, 5]))
        self.assertEqual(batch["labels"].shape, torch.Size([2, 5]))
        self.assertEqual(batch["next_sentence_label"].shape, torch.Size([2]))

    def test_nsp_with_padding(self):
        """Test NSP with pad_to_multiple_of."""
        tokenizer = BertTokenizer(self.vocab_file)
        collator = DataCollatorForLanguageModeling(tokenizer, pad_to_multiple_of=8)
        batch = collator(self._get_features())

        self.assertEqual(batch["input_ids"].shape, torch.Size([2, 8]))
        self.assertEqual(batch["next_sentence_label"].shape, torch.Size([2]))

    def test_numpy_output(self):
        """Test NSP with NumPy output."""
        tokenizer = BertTokenizer(self.vocab_file)
        collator = DataCollatorForLanguageModeling(tokenizer, return_tensors="np")
        batch = collator(self._get_features())

        self.assertEqual(batch["input_ids"].shape, (2, 5))
        self.assertEqual(batch["next_sentence_label"].shape, (2,))

    def test_immutability(self):
        """Test that NSP collation does not mutate input data."""
        tokenizer = BertTokenizer(self.vocab_file)

        for return_tensors in ["pt", "np"]:
            collator = DataCollatorForLanguageModeling(tokenizer, return_tensors=return_tensors)
            self._check_immutability(collator, self._get_features())


# =============================================================================
# Sentence Order Prediction tests
# =============================================================================


@require_torch
class TestSentenceOrderPrediction(DataCollatorTestMixin, unittest.TestCase):
    """
    Tests for Sentence Order Prediction (SOP) with DataCollatorForLanguageModeling.

    SOP is used in ALBERT pretraining where the model predicts if two sentences
    are in the correct order.
    """

    def _get_features(self):
        return [
            {"input_ids": [0, 1, 2, 3, 4], "token_type_ids": [0, 1, 2, 3, 4], "sentence_order_label": i}
            for i in range(2)
        ]

    def test_sop(self):
        """Test SOP labels are preserved during collation."""
        tokenizer = BertTokenizer(self.vocab_file)
        collator = DataCollatorForLanguageModeling(tokenizer)
        batch = collator(self._get_features())

        self.assertEqual(batch["input_ids"].shape, torch.Size([2, 5]))
        self.assertEqual(batch["token_type_ids"].shape, torch.Size([2, 5]))
        self.assertEqual(batch["labels"].shape, torch.Size([2, 5]))
        self.assertEqual(batch["sentence_order_label"].shape, torch.Size([2]))

    def test_sop_with_tensor_inputs(self):
        """Test SOP with PyTorch tensor inputs."""
        tokenizer = BertTokenizer(self.vocab_file)
        features = [
            {
                "input_ids": torch.tensor([0, 1, 2, 3, 4]),
                "token_type_ids": torch.tensor([0, 1, 2, 3, 4]),
                "sentence_order_label": i,
            }
            for i in range(2)
        ]

        collator = DataCollatorForLanguageModeling(tokenizer)
        batch = collator(features)

        self.assertEqual(batch["sentence_order_label"].shape, torch.Size([2]))

    def test_sop_with_padding(self):
        """Test SOP with pad_to_multiple_of."""
        tokenizer = BertTokenizer(self.vocab_file)
        features = [
            {
                "input_ids": torch.tensor([0, 1, 2, 3, 4]),
                "token_type_ids": torch.tensor([0, 1, 2, 3, 4]),
                "sentence_order_label": i,
            }
            for i in range(2)
        ]

        collator = DataCollatorForLanguageModeling(tokenizer, pad_to_multiple_of=8)
        batch = collator(features)

        self.assertEqual(batch["input_ids"].shape, torch.Size([2, 8]))
        self.assertEqual(batch["sentence_order_label"].shape, torch.Size([2]))

    def test_numpy_output(self):
        """Test SOP with NumPy output."""
        tokenizer = BertTokenizer(self.vocab_file)
        collator = DataCollatorForLanguageModeling(tokenizer, return_tensors="np")
        batch = collator(self._get_features())

        self.assertEqual(batch["input_ids"].shape, (2, 5))
        self.assertEqual(batch["sentence_order_label"].shape, (2,))

    def test_immutability(self):
        """Test that SOP collation does not mutate input data."""
        tokenizer = BertTokenizer(self.vocab_file)

        for return_tensors in ["pt", "np"]:
            collator = DataCollatorForLanguageModeling(tokenizer, return_tensors=return_tensors)
            self._check_immutability(collator, self._get_features())
