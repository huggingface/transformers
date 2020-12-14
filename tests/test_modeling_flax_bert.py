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

import unittest

from numpy import ndarray

from transformers import BertTokenizerFast, TensorType, is_flax_available, is_torch_available
from transformers.testing_utils import require_flax, require_torch


if is_flax_available():
    import os

    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.12"  # assumed parallelism: 8

    import jax
    from transformers.models.bert.modeling_flax_bert import FlaxBertModel

if is_torch_available():
    import torch

    from transformers.models.bert.modeling_bert import BertModel


@require_flax
@require_torch
class FlaxBertModelTest(unittest.TestCase):
    def assert_almost_equals(self, a: ndarray, b: ndarray, tol: float):
        diff = (a - b).sum()
        self.assertLessEqual(diff, tol, f"Difference between torch and flax is {diff} (>= {tol})")

    def test_from_pytorch(self):
        with torch.no_grad():
            with self.subTest("bert-base-cased"):
                tokenizer = BertTokenizerFast.from_pretrained("bert-base-cased")
                fx_model = FlaxBertModel.from_pretrained("bert-base-cased")
                pt_model = BertModel.from_pretrained("bert-base-cased")

                # Check for simple input
                pt_inputs = tokenizer.encode_plus("This is a simple input", return_tensors=TensorType.PYTORCH)
                fx_inputs = tokenizer.encode_plus("This is a simple input", return_tensors=TensorType.JAX)
                pt_outputs = pt_model(**pt_inputs).to_tuple()
                fx_outputs = fx_model(**fx_inputs)

                self.assertEqual(len(fx_outputs), len(pt_outputs), "Output lengths differ between Flax and PyTorch")

                for fx_output, pt_output in zip(fx_outputs, pt_outputs):
                    self.assert_almost_equals(fx_output, pt_output.numpy(), 5e-3)

    def test_multiple_sequences(self):
        tokenizer = BertTokenizerFast.from_pretrained("bert-base-cased")
        model = FlaxBertModel.from_pretrained("bert-base-cased")

        sequences = ["this is an example sentence", "this is another", "and a third one"]
        encodings = tokenizer(sequences, return_tensors=TensorType.JAX, padding=True, truncation=True)

        @jax.jit
        def model_jitted(input_ids, attention_mask=None, token_type_ids=None):
            return model(input_ids, attention_mask, token_type_ids)

        with self.subTest("JIT Disabled"):
            with jax.disable_jit():
                tokens, pooled = model_jitted(**encodings)
                self.assertEqual(tokens.shape, (3, 7, 768))
                self.assertEqual(pooled.shape, (3, 768))

        with self.subTest("JIT Enabled"):
            jitted_tokens, jitted_pooled = model_jitted(**encodings)

            self.assertEqual(jitted_tokens.shape, (3, 7, 768))
            self.assertEqual(jitted_pooled.shape, (3, 768))
