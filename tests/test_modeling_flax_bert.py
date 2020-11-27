import unittest

import pytest
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
                    self.assert_almost_equals(fx_output, pt_output.numpy(), 5e-4)

    def assert_almost_equals(self, a: ndarray, b: ndarray, tol: float):
        diff = (a - b).sum()
        self.assertLessEqual(diff, tol, "Difference between torch and flax is {} (>= {})".format(diff, tol))


@require_flax
@require_torch
@pytest.mark.parametrize("jit", ["disable_jit", "enable_jit"])
def test_multiple_sentences(jit):
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-cased")
    model = FlaxBertModel.from_pretrained("bert-base-cased")

    sentences = ["this is an example sentence", "this is another", "and a third one"]
    encodings = tokenizer(sentences, return_tensors=TensorType.JAX, padding=True, truncation=True)

    @jax.jit
    def model_jitted(input_ids, attention_mask, token_type_ids):
        return model(input_ids, attention_mask, token_type_ids)

    if jit == "disable_jit":
        with jax.disable_jit():
            tokens, pooled = model_jitted(**encodings)
    else:
        tokens, pooled = model_jitted(**encodings)

    assert tokens.shape == (3, 7, 768)
    assert pooled.shape == (3, 768)
