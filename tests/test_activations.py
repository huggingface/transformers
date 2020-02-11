import unittest

from transformers import is_torch_available

from .utils import require_torch


if is_torch_available():
    from transformers.pytorch_activations import _gelu_python, get_activation, gelu_new
    import torch


@require_torch
class TestActivations(unittest.TestCase):
    def test_gelu_versions(self):
        x = torch.Tensor([-100, -1, -0.1, 0, 0.1, 1.0, 100])
        torch_builtin = get_activation("gelu")
        self.assertTrue(torch.eq(_gelu_python(x), torch_builtin(x)).item())
        self.assertFalse(torch.eq(_gelu_python(x), gelu_new(x)).item())

    def test_get_activation(self):
        get_activation("swish")
        get_activation("relu")
        get_activation("tanh")
        with self.assertRaises(KeyError):
            get_activation("bogus")
        with self.assertRaises(KeyError):
            get_activation(None)


if __name__ == "__main__":
    unittest.main()
