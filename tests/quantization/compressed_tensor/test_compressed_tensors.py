import gc
import unittest

from transformers import AutoModelForCausalLM, AutoTokenizer, CompressedTensorsConfig
from transformers.testing_utils import require_compressed_tensors, require_torch
from transformers.utils import is_torch_available


if is_torch_available():
    import torch


@require_compressed_tensors
@require_torch
class CompressedTensorsTest(unittest.TestCase):
    tinyllama_w8a16 = "nm-testing/tinyllama-w8a16-dense-hf-quantizer"
    tinyllama_w4a16 = "nm-testing/tinyllama-w4a16-compressed-hf-quantizer"
    tinyllama_w8a8 = "nm-testing/tinyllama-w8a8-compressed-hf-quantizer"
    llama3_8b_fp8 = "nm-testing/Meta-Llama-3-8B-Instruct-fp8-hf_compat"

    prompt = "Paris is the capital of which country?"

    def tearDown(self):
        gc.collect()
        torch.cuda.empty_cache()
        gc.collect()

    def test_config_args(self):
        with self.assertRaises(ValueError):
            # passing quant scheme directly is not allowed
            CompressedTensorsConfig(config_groups={"weights": {"num_bits": 8}})
        CompressedTensorsConfig(
            config_groups={"FP8": ["Linear"]},
            ignore=["lm_head"],
            quantization_status="frozen",
            sparsity_config={"format": "dense"},
        )

    def test_config_to_from_dict(self):
        config = CompressedTensorsConfig(config_groups={"FP8": ["Linear"]}, sparsity_config={"format": "dense"})
        config_dict = config.to_dict()
        config_from_dict = CompressedTensorsConfig.from_dict(config_dict)

        from compressed_tensors import QuantizationConfig, SparsityCompressionConfig

        self.assertIsInstance(config_from_dict.quantization_config, QuantizationConfig)
        self.assertIsInstance(config_from_dict.sparsity_config, SparsityCompressionConfig)

    def test_tinyllama_w8a8(self):
        expected_out = "<s> Paris is the capital of which country?\n\n**A) Paris**\n\n**Q** ** Paris is the capital of which country?\n\n**A) Paris**\n\n**Q** ** Paris is the capital of which country"
        self._test_quantized_model(self.tinyllama_w8a8, expected_out)

    def test_tinyllama_w4a16(self):
        expected_out = "<s> Paris is the capital of which country?\nAnswer: Paris is the capital of France.\nQuestion: Which country is the capital of which city?\nAnswer: The capital of the city of New York is New York.\nQuestion: Which"
        self._test_quantized_model(self.tinyllama_w4a16, expected_out)

    def test_tinyllama_w8a16(self):
        expected_out = "<s> Paris is the capital of which country?\nA. France\nB. Germany\nC. Spain\nD. Italy\nE. Switzerland\nQ10. Which of the following is not a country in the European Union?\nA."
        self._test_quantized_model(self.tinyllama_w8a16, expected_out)

    def test_llama_8b_fp8(self):
        expected_out = "<|begin_of_text|>Paris is the capital of which country? France\nWhat is the name of the famous art museum in Paris? The Louvre\nWhat is the name of the famous opera house in Paris? Palais Garnier\nWhat is the name of the"
        self._test_quantized_model(self.llama3_8b_fp8, expected_out)

    def _test_quantized_model(self, model_name: str, expected_output: str):
        """Carry out generation"""
        quantized_model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        device = quantized_model.device
        self.assertIsNotNone(
            quantized_model.config.quantization_config,
            "quantization_config should not be None",
        )
        self.assertTrue(
            any(
                key
                for key, tensor in quantized_model.state_dict().items()
                if "scale" in key and not torch.all(tensor == 1.0)
            ),
            "quantized model should load a non-trivial scale into the state dict",
        )
        inputs = tokenizer(self.prompt, return_tensors="pt").to(device)
        generated_ids = quantized_model.generate(**inputs, max_length=50, do_sample=False)
        outputs = tokenizer.batch_decode(generated_ids)

        self.assertIsNotNone(outputs)
        self.assertEqual(outputs[0], expected_output)
