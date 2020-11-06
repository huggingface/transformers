import unittest

from transformers import TensorType, is_flax_available
from transformers.testing_utils import require_flax
from transformers.tokenization_bert_fast import BertTokenizerFast


if is_flax_available():
    from transformers.modeling_flax_performer import FlaxPerformerModel


@require_flax
class FlaxBertModelTest(unittest.TestCase):
    def test_from_pytorch(self):
        with self.subTest("performer-base-cased"):
            tokenizer = BertTokenizerFast.from_pretrained("bert-base-cased")
            fx_model = FlaxPerformerModel.from_pretrained("bert-base-cased")

            # Check for simple input
            fx_inputs = tokenizer.encode_plus("This is a simple input", return_tensors=TensorType.JAX)
            fx_outputs = fx_model(**fx_inputs)

            self.assertIsNotNone(fx_outputs)
