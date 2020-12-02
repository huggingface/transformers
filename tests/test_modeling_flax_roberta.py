import unittest

from numpy import ndarray

from transformers import RobertaTokenizerFast, TensorType, is_flax_available, is_torch_available
from transformers.testing_utils import require_flax, require_torch


if is_flax_available():
    import os

    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.12"  # assumed parallelism: 8

    import jax
    from transformers.models.roberta.modeling_flax_roberta import FlaxRobertaModel

if is_torch_available():
    import torch

    from transformers.models.roberta.modeling_roberta import RobertaModel


@require_flax
@require_torch
class FlaxRobertaModelTest(unittest.TestCase):
    def assert_almost_equals(self, a: ndarray, b: ndarray, tol: float):
        diff = (a - b).sum()
        self.assertLessEqual(diff, tol, f"Difference between torch and flax is {diff} (>= {tol})")

    def test_from_pytorch(self):
        with torch.no_grad():
            with self.subTest("roberta-base"):
                tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")
                fx_model = FlaxRobertaModel.from_pretrained("roberta-base")
                pt_model = RobertaModel.from_pretrained("roberta-base")

                # Check for simple input
                pt_inputs = tokenizer.encode_plus("This is a simple input", return_tensors=TensorType.PYTORCH)
                fx_inputs = tokenizer.encode_plus("This is a simple input", return_tensors=TensorType.JAX)
                pt_outputs = pt_model(**pt_inputs)
                fx_outputs = fx_model(**fx_inputs)

                self.assertEqual(len(fx_outputs), len(pt_outputs), "Output lengths differ between Flax and PyTorch")

                for fx_output, pt_output in zip(fx_outputs, pt_outputs.to_tuple()):
                    self.assert_almost_equals(fx_output, pt_output.numpy(), 6e-4)

    def test_multiple_sequences(self):
        tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")
        model = FlaxRobertaModel.from_pretrained("roberta-base")

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
