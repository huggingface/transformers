# from transformers.quantizers.quantizer_compressed_tensors import CompressedTensorsHfQuantizer
# from transformers.quantizers.quantizer_compressed_tensors import CompressedTensorsHfQuantizer

import gc
import unittest

import torch

from transformers import AutoModelForCausalLM, AutoTokenizer, CompressedTensorsConfig
from transformers.testing_utils import slow


class CompressedTensorsTest(unittest.TestCase):
    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    source_quantized_model_name = "nm-testing/tinyllama-oneshot-w8a8-test-static-shape-change-v3"

    prompt = "Paris is the capital of which country?"

    def tear_down(self):
        gc.collect()
        torch.cuda.empty_cache()
        gc.collect()

    @classmethod
    def setUpClass(self):
        """
        Setup quantized model
        """
        self.tokenizer = AutoTokenizer.from_pretrained(self.source_quantized_model_name)
        self.source_quantized_model = AutoModelForCausalLM.from_pretrained(self.source_quantized_model_name)

        self.device = self.source_quantized_model.device
        compression_config = self.source_quantized_model.config.quantization_config.quantization_config.config_groups

        self.config = CompressedTensorsConfig(
            config_groups=compression_config,
            sparsity_config=self.source_quantized_model.config.quantization_config.sparsity_config.dict(),
        )

        self.assertIsNotNone(self.config.sparsity_config, "sparsity_config should not be None")
        self.assertIsNotNone(self.config.quantization_config, "quantization_config should not be None")

        # apply quantization config to the base model
        self.quantized_model = AutoModelForCausalLM.from_pretrained(self.model_name, quantization_config=self.config)

    def test_quantized_model(self):
        """Carry out generation"""
        inputs = self.tokenizer(self.prompt, return_tensors="pt").to(self.device)
        generated_ids = self.quantized_model.generate(**inputs, max_length=50)
        outputs = self.tokenizer.batch_decode(generated_ids)

        self.assertIsNotNone(outputs)
        self.tear_down()

    @slow
    def test_forward(self):
        batch_size = context_size = 1024
        tensor1 = torch.rand(1024) * 1000
        tensor1 = tensor1.long()
        tensor2 = torch.rand(1024) * 1000
        tensor2 = tensor2.long()

        input_tensor = torch.cat((tensor1, tensor2), dim=0)
        input_tensor = input_tensor.unsqueeze(0)
        with torch.no_grad():
            out = self.quantized_model(input_tensor)
        self.assertEqual(out.shape[0], batch_size)
        self.assertEqual(out.shape[1], context_size)

        self.tear_down()
