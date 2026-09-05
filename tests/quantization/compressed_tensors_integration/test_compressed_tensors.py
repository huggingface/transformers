import gc
import os
import tempfile
import unittest

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


@require_compressed_tensors
@require_torch
class CompressedTensorsMoEExpertGlobalScaleTest(unittest.TestCase):
    """`tensor_group` (NVFP4) MoE experts must be dequantized with their per-tensor global scale.

    NVFP4 stores `weight_global_scale` beside the group scales, and the compressor reads it with
    `state_dict.get(..., None)`. A missing global scale is therefore not an error: the experts are
    dequantized by a wrong constant factor, nothing is raised, and nothing lands in `missing_keys`.
    These tests are hermetic — they build the packed tensors directly, so no checkpoint is needed.
    """

    group_size = 16

    def _scheme(self, tensor_group: bool):
        from compressed_tensors.quantization import (
            QuantizationArgs,
            QuantizationScheme,
            QuantizationStrategy,
            QuantizationType,
        )

        if tensor_group:
            args = QuantizationArgs(
                num_bits=4,
                type=QuantizationType.FLOAT,
                group_size=self.group_size,
                strategy=QuantizationStrategy.TENSOR_GROUP,
                symmetric=True,
            )
        else:
            args = QuantizationArgs(
                num_bits=4,
                type=QuantizationType.INT,
                group_size=128,
                strategy=QuantizationStrategy.GROUP,
                symmetric=True,
            )
        return QuantizationScheme(targets=["Linear"], weights=args)

    def _derived_sources(self, tensor_group: bool) -> set:
        """Per-module tensor names `update_weight_conversions` derives for an expert converter."""
        import types

        from transformers.core_model_loading import Concatenate, MergeModulelist, WeightConverter
        from transformers.quantizers.quantizer_compressed_tensors import CompressedTensorsHfQuantizer

        quantizer = object.__new__(CompressedTensorsHfQuantizer)
        inner = types.SimpleNamespace(config_groups={"group_0": self._scheme(tensor_group)})
        quantizer.quantization_config = types.SimpleNamespace(quantization_config=inner)
        quantizer.get_weight_conversions = lambda: []
        converter = WeightConverter(
            source_patterns=["experts.*.gate_proj.weight", "experts.*.up_proj.weight"],
            target_patterns="experts.gate_up_proj",
            operations=[MergeModulelist(dim=0), Concatenate(dim=1)],
        )
        updated = CompressedTensorsHfQuantizer.update_weight_conversions(quantizer, [converter])
        return {p.rsplit(".", 1)[-1].removesuffix("$") for conv in updated for p in conv.source_patterns}

    def test_tensor_group_scheme_derives_the_global_scale_source(self):
        """Without this source the global scale never reaches the decompressor."""
        self.assertIn("weight_global_scale", self._derived_sources(tensor_group=True))

    def test_non_tensor_group_scheme_is_unchanged(self):
        """INT `group` schemes have no global scale; their source list must not grow."""
        derived = self._derived_sources(tensor_group=False)
        self.assertNotIn("weight_global_scale", derived)
        self.assertEqual(derived, {"weight_packed", "weight_scale", "weight_shape"})

    def test_experts_are_dequantized_with_their_global_scale(self):
        """`DecompressExperts` must reproduce a faithful per-expert decompression.

        The regression this pins is silent: dropping the global scale returns experts scaled by
        it (a factor of ~100s), with no exception and an empty `missing_keys`.
        """
        import types

        from compressed_tensors.compressors import BaseCompressor
        from compressed_tensors.compressors.format import infer_module_format
        from compressed_tensors.quantization.utils.helpers import calculate_qparams, generate_gparam

        from transformers.integrations.compressed_tensors import DecompressExperts

        scheme = self._scheme(tensor_group=True)
        compressor = BaseCompressor.get_value_from_registry(infer_module_format(torch.nn.Linear, scheme))

        torch.manual_seed(0)
        num_experts = 3
        packed, scales, shapes, global_scales, faithful = [], [], [], [], []
        for expert in range(num_experts):
            # Different magnitudes per expert, so a shared or dropped global scale cannot pass.
            weight = (torch.randn(32, 64) * (expert + 1) * 3.0).to(torch.bfloat16)
            global_scale = generate_gparam(weight.min(), weight.max())
            grouped = weight.reshape(weight.shape[0], -1, self.group_size)
            scale, _ = calculate_qparams(
                grouped.min(dim=-1).values, grouped.max(dim=-1).values, scheme.weights, global_scale=global_scale
            )
            compressed = compressor.compress(
                {"weight": weight, "weight_scale": scale, "weight_global_scale": global_scale}, scheme
            )
            packed.append(compressed["weight_packed"])
            scales.append(compressed["weight_scale"])
            global_scales.append(compressed["weight_global_scale"])
            shapes.append(torch.tensor(list(weight.shape)))
            faithful.append(compressor.decompress(dict(compressed), scheme)["weight"])

        # `convert` reads the config off the quantizer even when handed an explicit scheme.
        quantizer = types.SimpleNamespace(
            compressor=types.SimpleNamespace(
                quantization_config=types.SimpleNamespace(config_groups={"group_0": scheme})
            )
        )
        op = DecompressExperts(quantizer, scheme=scheme)
        out = op.convert(
            {
                "experts.gate_proj.weight_packed": packed,
                "experts.gate_proj.weight_scale": scales,
                "experts.gate_proj.weight_shape": shapes,
                "experts.gate_proj.weight_global_scale": global_scales,
            },
            source_patterns=[],
            target_patterns=[],
        )
        stacked = out["experts.gate_proj.weight_packed"]
        expected = torch.stack([w.to(stacked.dtype) for w in faithful])
        torch.testing.assert_close(stacked, expected, rtol=0, atol=0)
