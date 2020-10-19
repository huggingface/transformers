import unittest

from numpy import ndarray

from transformers import TensorType, is_flax_available, is_torch_available
from transformers.testing_utils import require_flax, require_torch
from transformers.tokenization_roberta_fast import RobertaTokenizerFast


if is_flax_available():
    from transformers.modeling_flax_roberta import FlaxRobertaModel

if is_torch_available():
    import torch

    from transformers.modeling_roberta import RobertaModel


@require_flax
@require_torch
class FlaxRobertaModelTest(unittest.TestCase):
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

                for fx_output, pt_output in zip(fx_outputs, pt_outputs):
                    self.assert_almost_equals(fx_output, pt_output.numpy(), 5e-4)

    def assert_almost_equals(self, a: ndarray, b: ndarray, tol: float):
        diff = (a - b).sum()
        self.assertLessEqual(diff, tol, "Difference between torch and flax is {} (>= {})".format(diff, tol))
