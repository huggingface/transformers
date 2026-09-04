import gc
import os
import tempfile
import unittest
from types import SimpleNamespace

from transformers import AutoModelForCausalLM, AutoTokenizer, CompressedTensorsConfig
from transformers.testing_utils import (
    backend_empty_cache,
    require_compressed_tensors,
    require_cuda_capability_at_least,
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

    @property
    def optimized(self):
        """The optimized inference path is opt-in; without it the regular compressed-tensors route is used."""
        return CompressedTensorsConfig(use_optimized_inference=True)

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

    def test_laguna_ignore_names_follow_checkpoint_conversion(self):
        from transformers.quantizers.quantizer_compressed_tensors import _normalize_ignore_names_for_model

        ignore = [
            "model.layers.1.mlp.shared_expert.gate_proj",
            r"re:.*\.mlp\.shared_expert\.down_proj$",
            "model.layers.1.mlp.shared_experts.up_proj",
            "lm_head",
        ]

        self.assertEqual(
            _normalize_ignore_names_for_model(ignore, "laguna"),
            [
                "model.layers.1.mlp.shared_experts.gate_proj",
                r"re:.*\.mlp\.shared_experts\.down_proj$",
                "model.layers.1.mlp.shared_experts.up_proj",
                "lm_head",
            ],
        )
        self.assertIs(_normalize_ignore_names_for_model(ignore, "other"), ignore)

    def test_mixed_expert_scheme_selected_per_layer(self):
        """Laguna INT4 stores layers 1-30 in 4-bit and layers 31-39 in 8-bit."""
        from transformers.integrations.compressed_tensors import get_scheme

        def group(target, num_bits):
            return {
                "format": "pack-quantized",
                "targets": [target],
                "weights": {
                    "num_bits": num_bits,
                    "type": "int",
                    "strategy": "group",
                    "group_size": 128,
                    "symmetric": True,
                    "dynamic": False,
                },
            }

        config = CompressedTensorsConfig(
            config_groups={
                "int4": group(r"re:.*layers\.([1-9]|[12]\d|30)\..*(gate_proj|up_proj|down_proj)$", 4),
                "int8": group(r"re:.*layers\.3[1-9]\..*(gate_proj|up_proj|down_proj)$", 8),
            },
            format="pack-quantized",
            quantization_status="compressed",
        ).quantization_config

        self.assertEqual(get_scheme(config, "model.layers.1.mlp.experts.0.gate_proj").weights.num_bits, 4)
        self.assertEqual(get_scheme(config, "model.layers.31.mlp.experts.0.gate_proj").weights.num_bits, 8)
        self.assertIsNone(get_scheme(config, "model.embed_tokens"))

    def test_block_fp8_experts_use_compressor_and_model_dtype(self):
        """Laguna XS 2.1 FP8 uses blockwise scales, which require compressor dequantization."""
        from compressed_tensors.quantization.lifecycle.forward import dequantize

        from transformers.integrations.compressed_tensors import DecompressExperts

        config = CompressedTensorsConfig(
            config_groups={
                "fp8": {
                    "format": "float-quantized",
                    "targets": ["Linear"],
                    "input_activations": {
                        "num_bits": 8,
                        "type": "float",
                        "strategy": "tensor",
                        "symmetric": True,
                        "dynamic": True,
                    },
                    "weights": {
                        "num_bits": 8,
                        "type": "float",
                        "strategy": "block",
                        "block_structure": [2, 2],
                        "symmetric": True,
                        "dynamic": False,
                    },
                }
            },
            format="float-quantized",
            quantization_status="compressed",
        ).quantization_config
        scheme = config.config_groups["fp8"]
        weight = torch.arange(1, 17, dtype=torch.float32).reshape(4, 4).to(torch.float8_e4m3fn)
        scale = torch.tensor([[0.5, 1.0], [1.5, 2.0]])
        weight_key = "mlp.experts.*.gate_proj.weight$"
        scale_key = "mlp.experts.*.gate_proj.weight_scale$"
        quantizer = SimpleNamespace(compressor=SimpleNamespace(quantization_config=config))
        model = SimpleNamespace(get_parameter=lambda _: SimpleNamespace(dtype=torch.bfloat16))

        converted = DecompressExperts(quantizer).convert(
            {weight_key: [weight], scale_key: [scale]},
            source_patterns=[weight_key, scale_key],
            target_patterns=["mlp.experts.gate_up_proj"],
            full_layer_name="model.layers.1.mlp.experts.gate_up_proj",
            model=model,
        )

        expected = dequantize(weight, scale, args=scheme.weights).to(torch.bfloat16)
        self.assertEqual(converted[weight_key].dtype, torch.bfloat16)
        torch.testing.assert_close(converted[weight_key], expected.unsqueeze(0))

    def test_expert_conversion_collects_metadata_by_format(self):
        from transformers.core_model_loading import Concatenate, MergeModulelist, WeightConverter
        from transformers.quantizers.quantizer_compressed_tensors import CompressedTensorsHfQuantizer

        config = SimpleNamespace(
            config_groups={
                name: SimpleNamespace(format=format)
                for name, format in {
                    "fp8": "float-quantized",
                    "int4": "pack-quantized",
                    "nvfp4": "nvfp4-pack-quantized",
                }.items()
            }
        )
        converter = WeightConverter(
            ["mlp.experts.*.gate_proj.weight", "mlp.experts.*.up_proj.weight"],
            "mlp.experts.gate_up_proj",
            [MergeModulelist(dim=0), Concatenate(dim=1)],
        )
        quantizer = SimpleNamespace(
            compressor=SimpleNamespace(quantization_config=config), get_weight_conversions=lambda: []
        )
        converted = CompressedTensorsHfQuantizer.update_weight_conversions(quantizer, [converter])[0]

        self.assertIn("mlp.experts.*.gate_proj.weight$", converted.source_patterns)
        self.assertIn("mlp.experts.*.gate_proj.weight_packed$", converted.source_patterns)
        self.assertIn("mlp.experts.*.gate_proj.weight_shape$", converted.source_patterns)
        self.assertIn("mlp.experts.*.gate_proj.weight_global_scale$", converted.source_patterns)
        self.assertIn("mlp.experts.*.gate_proj.input_global_scale$", converted.source_patterns)
        self.assertNotIn("mlp.experts.*.gate_proj.weight_zero_point$", converted.source_patterns)
        self.assertNotIn("mlp.experts.*.gate_proj.weight_g_idx$", converted.source_patterns)
        self.assertEqual(len(converted.source_patterns), len(set(converted.source_patterns)))

    def test_tinyllama_w4a16(self):
        # Non-FP8 schemes have no kernels and are dequantized at load time.
        self._test_quantized_model(self.tinyllama_w4a16, 20.0, expect_quantized=False)

    def test_tinyllama_int8(self):
        self._test_quantized_model(self.tinyllama_int8, 30.0, expect_quantized=False)

    @require_torch_accelerator
    @require_cuda_capability_at_least(8, 9)
    def test_tinyllama_fp8(self):
        self._test_quantized_model(self.tinyllama_fp8, 20.0, quantization_config=self.optimized)

    def test_tinyllama_w8a16(self):
        self._test_quantized_model(self.tinyllama_w8a16, 20.0, expect_quantized=False)

    def test_frozen_fp8_dequantized_on_load(self):
        quantization_config = CompressedTensorsConfig(dequantize=True)
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

    def _test_quantized_model(
        self,
        model_name: str,
        expected_perplexity: float,
        expect_quantized: bool = True,
        quantization_config=None,
    ):
        # load model. NB: `quantization_config=None` cannot be passed through, it is not the same as
        # not passing it at all -- `from_pretrained` then tries to merge it with the checkpoint's own
        kwargs = {} if quantization_config is None else {"quantization_config": quantization_config}
        quantized_model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", **kwargs)
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
    @require_cuda_capability_at_least(8, 9)
    def test_tinyllama_fp8_uses_fp8_kernel(self):
        """With `use_optimized_inference=True`, an FP8 model uses CompressedTensorsFP8Linear on GPU/XPU."""
        from transformers.integrations.compressed_tensors import CompressedTensorsFP8Linear

        model = AutoModelForCausalLM.from_pretrained(
            self.tinyllama_fp8, device_map="auto", quantization_config=self.optimized
        )

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
        """With `dequantize=True` the weights are dequantized at load time."""
        from transformers.integrations.compressed_tensors import CompressedTensorsFP8Linear

        quantization_config = CompressedTensorsConfig(dequantize=True)
        model = AutoModelForCausalLM.from_pretrained(
            self.tinyllama_fp8, device_map="auto", quantization_config=quantization_config
        )

        fp8_count = sum(1 for m in model.modules() if isinstance(m, CompressedTensorsFP8Linear))
        self.assertEqual(fp8_count, 0, "dequantize=True should NOT use CompressedTensorsFP8Linear")
        self.assertEqual(model.model.layers[0].self_attn.q_proj.weight.dtype, model.dtype)

        # Model should still generate sensible outputs after dequantization.
        tokenizer = AutoTokenizer.from_pretrained(self.tinyllama_fp8)
        inputs = tokenizer(self.prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model(**inputs, labels=inputs["input_ids"])
        self.assertLessEqual(torch.exp(outputs.loss), 20.0)

    @require_torch_accelerator
    @require_cuda_capability_at_least(8, 9)
    def test_tinyllama_fp8_save_reload(self):
        """An FP8 model should still work after saving and reloading."""
        model = AutoModelForCausalLM.from_pretrained(
            self.tinyllama_fp8, device_map="auto", quantization_config=self.optimized
        )
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

            reloaded = AutoModelForCausalLM.from_pretrained(
                tmp_dir, device_map="auto", quantization_config=self.optimized
            )
            inputs = tokenizer(self.prompt, return_tensors="pt").to(reloaded.device)
            with torch.no_grad():
                outputs = reloaded(**inputs, labels=inputs["input_ids"])
            self.assertLessEqual(torch.exp(outputs.loss), 20.0)

    @require_torch_accelerator
    @require_cuda_capability_at_least(8, 9)
    def test_tinyllama_fp8_per_tensor_save_reload(self):
        """Per-tensor (static) FP8: the single weight scale is expanded to (1, out_features)
        at load time and collapsed back to a single element on save."""
        from transformers.integrations.compressed_tensors import CompressedTensorsFP8Linear

        model = AutoModelForCausalLM.from_pretrained(
            self.tinyllama_fp8_static, device_map="auto", quantization_config=self.optimized
        )
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

            reloaded = AutoModelForCausalLM.from_pretrained(
                tmp_dir, device_map="auto", quantization_config=self.optimized
            )
            tokenizer = AutoTokenizer.from_pretrained(self.tinyllama_fp8_static)
            inputs = tokenizer(self.prompt, return_tensors="pt").to(reloaded.device)
            with torch.no_grad():
                outputs = reloaded(**inputs, labels=inputs["input_ids"])
            self.assertLessEqual(torch.exp(outputs.loss), 20.0)

    def test_use_optimized_inference_is_opt_in(self):
        """Without `use_optimized_inference=True`, an FP8 checkpoint takes the regular compressed-tensors
        route: no kernels, and with `dequantize=False` the weights are left compressed until the
        first forward pass decompresses them."""
        from transformers.integrations.compressed_tensors import CompressedTensorsFP8Linear

        model = AutoModelForCausalLM.from_pretrained(self.tinyllama_fp8, device_map=torch_device)
        module = model.model.layers[0].self_attn.q_proj

        self.assertNotIsInstance(module, CompressedTensorsFP8Linear)
        self.assertEqual(module.weight.dtype, torch.float8_e4m3fn, "weights must not be decompressed at load time")

        tokenizer = AutoTokenizer.from_pretrained(self.tinyllama_fp8)
        inputs = tokenizer(self.prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model(**inputs, labels=inputs["input_ids"])
        # compressed-tensors decompressed the model on that first forward pass
        self.assertEqual(module.weight.dtype, model.dtype)
        self.assertLessEqual(torch.exp(outputs.loss), 20.0)

    def test_non_fp8_model_unaffected(self):
        """Verify non-FP8 models (e.g. INT8) do not use the FP8 kernel path."""
        from transformers.integrations.compressed_tensors import CompressedTensorsFP8Linear

        int8_model = "nm-testing/tinyllama-w8a8-compressed"
        model = AutoModelForCausalLM.from_pretrained(int8_model, device_map="auto")
        fp8_count = sum(1 for m in model.modules() if isinstance(m, CompressedTensorsFP8Linear))
        self.assertEqual(fp8_count, 0, "INT8 model should NOT use CompressedTensorsFP8Linear")
