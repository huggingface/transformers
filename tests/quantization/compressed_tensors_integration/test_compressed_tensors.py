import gc
import os
import tempfile
import unittest

from transformers import AutoModelForCausalLM, AutoTokenizer, CompressedTensorsConfig
from transformers.testing_utils import (
    backend_empty_cache,
    require_compressed_tensors,
    require_torch,
    require_torch_accelerator,
    torch_device,
)
from transformers.utils import is_torch_available


if is_torch_available():
    import torch


@require_compressed_tensors
@require_torch
class CompressedTensorsTest(unittest.TestCase):
    tinyllama_w4a16 = "nm-testing/TinyLlama-1.1B-Chat-v1.0-W4A16-G128-compressed"
    tinyllama_int8 = "nm-testing/TinyLlama-1.1B-Chat-v1.0-W8A8-Dynamic-Per-Token-compressed"
    tinyllama_fp8 = "nm-testing/TinyLlama-1.1B-Chat-v1.0-FP8-Dynamic-compressed"
    tinyllama_fp8_static = "nm-testing/TinyLlama-1.1B-Chat-v1.0-FP8-e2e"
    tinyllama_w8a16 = "nm-testing/TinyLlama-1.1B-Chat-v1.0-W8A16-G128-compressed"
    llama3_fp8_frozen = "RedHatAI/Llama-3.2-1B-Instruct-FP8"

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
        # Non-FP8 schemes have no kernels and are dequantized at load time.
        self._test_quantized_model(self.tinyllama_w4a16, 20.0, expect_quantized=False)

    def test_tinyllama_int8(self):
        self._test_quantized_model(self.tinyllama_int8, 30.0, expect_quantized=False)

    def test_tinyllama_fp8(self):
        self._test_quantized_model(self.tinyllama_fp8, 20.0)

    def test_tinyllama_w8a16(self):
        self._test_quantized_model(self.tinyllama_w8a16, 20.0, expect_quantized=False)

    def test_frozen_fp8_dequantized_on_load(self):
        quantization_config = CompressedTensorsConfig(run_compressed=False)
        model = AutoModelForCausalLM.from_pretrained(
            self.llama3_fp8_frozen,
            device_map=torch_device,
            torch_dtype=torch.float32,
            quantization_config=quantization_config,
        )
        weight = model.model.layers[0].self_attn.q_proj.weight
        # Dequantized max is small (~0.68); raw fp8 max would be 448.0
        self.assertLess(weight.abs().max().item(), 5.0)

        tokenizer = AutoTokenizer.from_pretrained(self.llama3_fp8_frozen)
        inputs = tokenizer(self.prompt, return_tensors="pt").to(torch_device)
        output_ids = model.generate(**inputs, max_new_tokens=8, do_sample=False)
        self.assertGreater(output_ids.shape[1], inputs["input_ids"].shape[1])

    def _test_quantized_model(self, model_name: str, expected_perplexity: float, expect_quantized: bool = True):
        # load model
        quantized_model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        device = quantized_model.device

        if expect_quantized:
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

    @require_torch_accelerator
    def test_tinyllama_fp8_uses_fp8_kernel(self):
        """Verify FP8 model uses CompressedTensorsFP8Linear on GPU/XPU."""
        from transformers.integrations.compressed_tensors import CompressedTensorsFP8Linear

        model = AutoModelForCausalLM.from_pretrained(self.tinyllama_fp8, device_map="auto")

        fp8_count = sum(1 for m in model.modules() if isinstance(m, CompressedTensorsFP8Linear))
        self.assertGreater(fp8_count, 0, "FP8 model should use CompressedTensorsFP8Linear on GPU/XPU")

        # Verify weights are in FP8 dtype and the scale was reshaped into the row-wise
        # kernel layout at load time (see `ConvertFP8LinearScale`).
        for module in model.modules():
            if isinstance(module, CompressedTensorsFP8Linear):
                self.assertEqual(module.weight.dtype, torch.float8_e4m3fn)
                self.assertEqual(module.weight_scale.shape, (1, module.out_features))
                self.assertEqual(module.weight_scale.dtype, torch.float32)
                self.assertTrue(module.weight_scale.is_contiguous())
                break

    def test_tinyllama_fp8_dequantize(self):
        """With `dequantize=True` the FP8 kernel path is disabled and weights are dequantized."""
        from transformers.integrations.compressed_tensors import CompressedTensorsFP8Linear

        quantization_config = CompressedTensorsConfig(dequantize=True)
        model = AutoModelForCausalLM.from_pretrained(
            self.tinyllama_fp8, device_map="auto", quantization_config=quantization_config
        )

        fp8_count = sum(1 for m in model.modules() if isinstance(m, CompressedTensorsFP8Linear))
        self.assertEqual(fp8_count, 0, "dequantize=True should NOT use CompressedTensorsFP8Linear")

        # Model should still generate sensible outputs after dequantization.
        tokenizer = AutoTokenizer.from_pretrained(self.tinyllama_fp8)
        inputs = tokenizer(self.prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model(**inputs, labels=inputs["input_ids"])
        self.assertLessEqual(torch.exp(outputs.loss), 20.0)

    @require_torch_accelerator
    def test_tinyllama_fp8_save_reload(self):
        """An FP8 model should still work after saving and reloading."""
        model = AutoModelForCausalLM.from_pretrained(self.tinyllama_fp8, device_map="auto")
        tokenizer = AutoTokenizer.from_pretrained(self.tinyllama_fp8)

        with tempfile.TemporaryDirectory() as tmp_dir:
            model.save_pretrained(tmp_dir)
            del model
            gc.collect()
            backend_empty_cache(torch_device)

            # Saving must restore the compressed-tensors checkpoint layout (`RevertFP8LinearScale`
            # reverses the load-time reshape), so the checkpoint stays loadable by other
            # compressed-tensors consumers (llm-compressor, vLLM).
            from safetensors import safe_open

            with safe_open(os.path.join(tmp_dir, "model.safetensors"), framework="pt") as f:
                scale = f.get_tensor("model.layers.0.self_attn.q_proj.weight_scale")
                weight = f.get_tensor("model.layers.0.self_attn.q_proj.weight")
            self.assertEqual(scale.shape, (weight.shape[0], 1))

            reloaded = AutoModelForCausalLM.from_pretrained(tmp_dir, device_map="auto")
            inputs = tokenizer(self.prompt, return_tensors="pt").to(reloaded.device)
            with torch.no_grad():
                outputs = reloaded(**inputs, labels=inputs["input_ids"])
            self.assertLessEqual(torch.exp(outputs.loss), 20.0)

    @require_torch_accelerator
    def test_tinyllama_fp8_per_tensor_save_reload(self):
        """Per-tensor (static) FP8: the single weight scale is expanded to (1, out_features)
        at load time and collapsed back to a single element on save."""
        from transformers.integrations.compressed_tensors import CompressedTensorsFP8Linear

        model = AutoModelForCausalLM.from_pretrained(self.tinyllama_fp8_static, device_map="auto")
        module = next(m for m in model.modules() if isinstance(m, CompressedTensorsFP8Linear))
        self.assertEqual(module.weight_scale.shape, (1, module.out_features))

        with tempfile.TemporaryDirectory() as tmp_dir:
            model.save_pretrained(tmp_dir)
            del model
            gc.collect()
            backend_empty_cache(torch_device)

            from safetensors import safe_open

            with safe_open(os.path.join(tmp_dir, "model.safetensors"), framework="pt") as f:
                scale = f.get_tensor("model.layers.0.self_attn.q_proj.weight_scale")
            self.assertEqual(scale.numel(), 1)

            reloaded = AutoModelForCausalLM.from_pretrained(tmp_dir, device_map="auto")
            tokenizer = AutoTokenizer.from_pretrained(self.tinyllama_fp8_static)
            inputs = tokenizer(self.prompt, return_tensors="pt").to(reloaded.device)
            with torch.no_grad():
                outputs = reloaded(**inputs, labels=inputs["input_ids"])
            self.assertLessEqual(torch.exp(outputs.loss), 20.0)

    def test_non_fp8_model_unaffected(self):
        """Verify non-FP8 models (e.g. INT8) do not use the FP8 kernel path."""
        from transformers.integrations.compressed_tensors import CompressedTensorsFP8Linear

        int8_model = "nm-testing/tinyllama-w8a8-compressed"
        model = AutoModelForCausalLM.from_pretrained(int8_model, device_map="auto")
        fp8_count = sum(1 for m in model.modules() if isinstance(m, CompressedTensorsFP8Linear))
        self.assertEqual(fp8_count, 0, "INT8 model should NOT use CompressedTensorsFP8Linear")
