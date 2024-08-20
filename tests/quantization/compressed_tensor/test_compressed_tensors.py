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
    tinyllama_w8a8 = "nm-testing/tinyllama-oneshot-w8a8-test-static-shape-change-v3"
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
        self._test_quantized_model(self.tinyllama_w8a8)

    def test_llama_8b_fp8(self):
        self._test_quantized_model(self.llama3_8b_fp8)

    def _test_quantized_model(self, model_name: str):
        """Carry out generation"""
        quantized_model = AutoModelForCausalLM.from_pretrained(model_name)
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
        generated_ids = quantized_model.generate(**inputs, max_length=50)
        outputs = tokenizer.batch_decode(generated_ids)

        self.assertIsNotNone(outputs)
