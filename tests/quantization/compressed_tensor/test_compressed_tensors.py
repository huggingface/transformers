import gc
import unittest

from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.testing_utils import require_compressed_tensors, require_torch
from transformers.utils import is_torch_available


if is_torch_available():
    import torch


@require_compressed_tensors
@require_torch
class CompressedTensorsTest(unittest.TestCase):
    quantized_model_name = "nm-testing/tinyllama-oneshot-w8a8-test-static-shape-change-v3"

    prompt = "Paris is the capital of which country?"

    def tearDown(self):
        gc.collect()
        torch.cuda.empty_cache()
        gc.collect()

    @classmethod
    def setUpClass(self):
        """
        Setup quantized model
        """
        self.tokenizer = AutoTokenizer.from_pretrained(self.quantized_model_name)
        self.quantized_model = AutoModelForCausalLM.from_pretrained(self.quantized_model_name)
        self.device = self.quantized_model.device

    def test_quantized_model(self):
        """Carry out generation"""
        self.assertIsNotNone(
            self.quantized_model.config.quantization_config,
            "quantization_config should not be None",
        )
        self.assertTrue(
            any(
                key
                for key, tensor in self.quantized_model.state_dict().items()
                if "scale" in key and not torch.all(tensor == 1.0)
            ),
            "quantized model should load a non-trivail scale into the state dict",
        )
        inputs = self.tokenizer(self.prompt, return_tensors="pt").to(self.device)
        generated_ids = self.quantized_model.generate(**inputs, max_length=50)
        outputs = self.tokenizer.batch_decode(generated_ids)

        self.assertIsNotNone(outputs)
