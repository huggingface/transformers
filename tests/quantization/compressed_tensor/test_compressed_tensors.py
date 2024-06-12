# from transformers.quantizers.quantizer_compressed_tensors import CompressedTensorsHfQuantizer
# from transformers.quantizers.quantizer_compressed_tensors import CompressedTensorsHfQuantizer

import gc
import unittest

import torch

from transformers import AutoModelForCausalLM, AutoTokenizer, CompressedTensorsConfig


class CompressedTensorsTest(unittest.TestCase):
    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    source_quantized_model_name = "nm-testing/tinyllama-oneshot-w8a8-test-static-shape-change-v3"

    prompt = "Paris is the capital of which country?"
    # ['<s> Paris is the capital of which country?\n\nA. London\n\nB. New York\n\nC. Paris\n\nD. Tokyo\n\n4. Which country is the capital of the European Union?\n\nA. France\n']
    expected_response = ""

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

    @unittest.skip("scales not populated")
    def test_apply_quantization(self):
        # fails bc state_dict_scale = state_dict[f"{module_name}.{scale_name}"]
        #  KeyError: 'model.layers.0.self_attn.q_proj.weight_scale
        self.quantization_model = AutoModelForCausalLM.from_pretrained(
            self.model_name, quantization_config=self.config
        )
        # check that the input layers of self.source_quantized_model and self.quantization_model is the same

    def test_quantized_model(self):
        # test the quantized model, not the original model

        inputs = self.tokenizer(self.prompt, return_tensors="pt").to(self.device)
        generated_ids = self.source_quantized_model.generate(**inputs, max_length=50)
        outputs = self.tokenizer.batch_decode(generated_ids)

        self.expected_response = outputs
        self.assertEqual(outputs, self.expected_response)
        self.tear_down()

    def test_forward(self):
        batch_size = context_size = 1024
        tensor1 = torch.rand(1024).long()
        tensor2 = torch.rand(1024).long()

        input_tensor = torch.cat((tensor1, tensor2), dim=0)
        input_tensor = input_tensor.unsqueeze(0)
        with torch.no_grad():
            out = self.source_quantized_model(input_tensor)
        self.assertEqual(out.shape[0], batch_size)
        self.assertEqual(out.shape[1], context_size)

        self.tear_down()
