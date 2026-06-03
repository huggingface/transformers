import gc
import unittest

from transformers import AutoModelForCausalLM, AutoTokenizer, CompressedTensorsConfig
from transformers.testing_utils import backend_empty_cache, require_compressed_tensors, require_torch, torch_device
from transformers.utils import is_torch_available


if is_torch_available():
    import torch


@require_compressed_tensors
@require_torch
class CompressedTensorsTest(unittest.TestCase):
    tinyllama_w4a16 = "nm-testing/TinyLlama-1.1B-Chat-v1.0-W4A16-G128-compressed"
    tinyllama_int8 = "nm-testing/TinyLlama-1.1B-Chat-v1.0-W8A8-Dynamic-Per-Token-compressed"
    tinyllama_fp8 = "nm-testing/TinyLlama-1.1B-Chat-v1.0-FP8-Dynamic-compressed"
    tinyllama_w8a16 = "nm-testing/TinyLlama-1.1B-Chat-v1.0-W8A16-G128-compressed"

    prompt = "The capital of France is Paris, the capital of Germany is Berlin"

    def tearDown(self):
        gc.collect()
        backend_empty_cache(torch_device)
        gc.collect()

    def test_config_args(self):
        with self.assertRaises(ValueError):
            # passing quant scheme directly is not allowed
            CompressedTensorsConfig(config_groups={"weights": {"num_bits": 8}})
        CompressedTensorsConfig(
            config_groups={"FP8": ["Linear"]},
            ignore=["lm_head"],
            quantization_status="frozen",
        )

    def test_config_to_from_dict(self):
        config = CompressedTensorsConfig(config_groups={"FP8": ["Linear"]})
        config_dict = config.to_dict()
        config_from_dict = CompressedTensorsConfig.from_dict(config_dict)

        from compressed_tensors import QuantizationConfig

        self.assertIsInstance(config_from_dict.quantization_config, QuantizationConfig)

    def test_tinyllama_w4a16(self):
        self._test_quantized_model(self.tinyllama_w4a16, 20.0)

    def test_tinyllama_int8(self):
        self._test_quantized_model(self.tinyllama_int8, 30.0)

    def test_tinyllama_fp8(self):
        self._test_quantized_model(self.tinyllama_fp8, 20.0)

    def test_tinyllama_w8a16(self):
        self._test_quantized_model(self.tinyllama_w8a16, 20.0)

    def _test_quantized_model(self, model_name: str, expected_perplexity: float):
        # load model
        quantized_model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        device = quantized_model.device

        # check config
        self.assertIsNotNone(
            quantized_model.config.quantization_config,
            "quantization_config should not be None",
        )
        # check scales
        self.assertTrue(
            any(
                key
                for key, tensor in quantized_model.state_dict().items()
                if "scale" in key and not torch.all(tensor == 1.0)
            ),
            "quantized model should load a non-trivial scale into the state dict",
        )

        # compute outputs with loss
        inputs = tokenizer(self.prompt, return_tensors="pt").to(device)
        labels = inputs["input_ids"]
        with torch.no_grad():
            outputs = quantized_model(**inputs, labels=labels)

        # check perplexity
        perplexity = torch.exp(outputs.loss)
        self.assertLessEqual(perplexity, expected_perplexity)

    def test_tinyllama_fp8_uses_fp8_kernel(self):
        """Verify FP8 model uses CompressedTensorsFP8Linear on GPU/XPU."""
        if not (torch.cuda.is_available() or (hasattr(torch, "xpu") and torch.xpu.is_available())):
            self.skipTest("FP8 kernel path requires GPU or XPU")

        from transformers.integrations.compressed_tensors_fp8 import CompressedTensorsFP8Linear

        model = AutoModelForCausalLM.from_pretrained(self.tinyllama_fp8, device_map="auto")

        fp8_count = sum(1 for m in model.modules() if isinstance(m, CompressedTensorsFP8Linear))
        self.assertGreater(fp8_count, 0, "FP8 model should use CompressedTensorsFP8Linear on GPU/XPU")

        # Verify weights are in FP8 dtype
        for module in model.modules():
            if isinstance(module, CompressedTensorsFP8Linear):
                self.assertEqual(module.weight.dtype, torch.float8_e4m3fn)
                break

    def test_non_fp8_model_unaffected(self):
        """Verify non-FP8 models (e.g. INT8) do not use the FP8 kernel path."""
        from transformers.integrations.compressed_tensors_fp8 import CompressedTensorsFP8Linear

        int8_model = "nm-testing/tinyllama-w8a8-compressed"
        model = AutoModelForCausalLM.from_pretrained(int8_model, device_map="auto")
        fp8_count = sum(1 for m in model.modules() if isinstance(m, CompressedTensorsFP8Linear))
        self.assertEqual(fp8_count, 0, "INT8 model should NOT use CompressedTensorsFP8Linear")
