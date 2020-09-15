import unittest

from transformers import is_tf_available
from transformers.testing_utils import require_tf


if is_tf_available():
    from transformers.activations_tf import get_tf_activation


@require_tf
class TestTFActivations(unittest.TestCase):
    def test_get_activation(self):
        get_tf_activation("swish")
        get_tf_activation("gelu")
        get_tf_activation("relu")
        get_tf_activation("tanh")
        get_tf_activation("gelu_new")
        get_tf_activation("gelu_fast")
        get_tf_activation("mish")
        with self.assertRaises(KeyError):
            get_tf_activation("bogus")
        with self.assertRaises(KeyError):
            get_tf_activation(None)
