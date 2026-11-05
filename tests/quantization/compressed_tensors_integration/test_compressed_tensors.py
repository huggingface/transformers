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
class CompressedTensorsUpdateWeightConversionsTest(unittest.TestCase):
    """The rewrite of the model's conversion mapping by the quantizer, exercised without loading any checkpoint.

    Mixed-precision checkpoints (e.g. RedHatAI/GLM-5.3-Flash-NVFP4, whose experts are NVFP4-packed while its
    causal convs stay dense) only serialize the companion tensors (e.g. `weight_scale`) of the modules they
    quantize: `update_weight_conversions` must leave the converters of the other modules as registered, so that
    they keep consuming the plain checkpoint keys instead of silently dropping them.
    """

    # experts-only NVFP4 scheme, like RedHatAI/GLM-5.3-Flash-NVFP4
    nvfp4_experts_config = {
        "config_groups": {
            "group_0": {
                "format": "nvfp4-pack-quantized",
                "input_activations": None,
                "targets": ["re:.*\\.mlp\\.experts\\..*(gate|up|down)_proj$"],
                "weights": {
                    "dynamic": False,
                    "group_size": 16,
                    "num_bits": 4,
                    "strategy": "tensor_group",
                    "symmetric": True,
                    "type": "float",
                },
            }
        },
        "format": "mixed-precision",
        "quantization_status": "compressed",
        "ignore": ["lm_head"],
    }
    # experts-only FP8 scheme: the serialized names then include the plain `weight`
    fp8_experts_config = {
        "config_groups": {
            "group_0": {
                "format": "float-quantized",
                "input_activations": None,
                "targets": ["re:.*\\.mlp\\.experts\\..*(gate|up|down)_proj$"],
                "weights": {
                    "dynamic": False,
                    "num_bits": 8,
                    "strategy": "tensor",
                    "symmetric": True,
                    "type": "float",
                },
            }
        },
        "format": "mixed-precision",
        "quantization_status": "compressed",
        "ignore": ["lm_head"],
    }

    # Keys of a checkpoint which quantizes its experts to NVFP4 but keeps the (fused) causal conv modules
    # dense, like RedHatAI/GLM-5.3-Flash-NVFP4
    checkpoint_keys = {
        "model.layers.0.self_attn.q_conv1d.weight",
        "model.layers.0.self_attn.k_conv1d.weight",
        "model.layers.0.self_attn.v_conv1d.weight",
        "model.layers.1.self_attn.q_conv1d.weight",
        "model.layers.1.self_attn.k_conv1d.weight",
        "model.layers.1.self_attn.v_conv1d.weight",
        "model.layers.1.mlp.experts.0.gate_proj.weight_packed",
        "model.layers.1.mlp.experts.0.gate_proj.weight_scale",
        "model.layers.1.mlp.experts.0.gate_proj.weight_shape",
        "model.layers.1.mlp.experts.0.up_proj.weight_packed",
        "model.layers.1.mlp.experts.0.up_proj.weight_scale",
        "model.layers.1.mlp.experts.0.up_proj.weight_shape",
        "model.layers.1.mlp.experts.0.down_proj.weight_packed",
        "model.layers.1.mlp.experts.0.down_proj.weight_scale",
        "model.layers.1.mlp.experts.0.down_proj.weight_shape",
    }
    packed_names = ("weight_packed", "weight_scale", "weight_shape")
    conv1d_sources = ["self_attn.q_conv1d.weight", "self_attn.k_conv1d.weight", "self_attn.v_conv1d.weight"]

    def _quantizer(self, config):
        from transformers.quantizers.quantizer_compressed_tensors import CompressedTensorsHfQuantizer

        quantizer = CompressedTensorsHfQuantizer(CompressedTensorsConfig(**config), pre_quantized=True)
        quantizer.use_fp8_kernel = False  # normally set by validate_environment()
        return quantizer

    @staticmethod
    def _glm5_next_like_model():
        """Parameter layout of glm5_next layers: a fused causal conv in attention (dense) and a fused MoE block."""

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.model = torch.nn.Module()
                self.model.layers = torch.nn.ModuleList()
                for _ in range(2):
                    layer = torch.nn.Module()
                    layer.self_attn = torch.nn.Module()
                    layer.self_attn.conv1d = torch.nn.Conv1d(8, 8, 4)
                    layer.mlp = torch.nn.Module()
                    layer.mlp.experts = torch.nn.Module()
                    layer.mlp.experts.gate_up_proj = torch.nn.Linear(8, 16)
                    layer.mlp.experts.down_proj = torch.nn.Linear(8, 16)
                    self.model.layers.append(layer)

        return Model()

    @staticmethod
    def _conv1d_converter():
        # the glm5_next q/k/v conv1d fusion registered by transformers
        from transformers.core_model_loading import Concatenate, WeightConverter

        return WeightConverter(
            source_patterns=[
                "self_attn.q_conv1d.weight",
                "self_attn.k_conv1d.weight",
                "self_attn.v_conv1d.weight",
            ],
            target_patterns="self_attn.conv1d.weight",
            operations=[Concatenate(dim=0)],
        )

    @staticmethod
    def _experts_converter():
        # the glm5_next expert fusion registered by transformers
        from transformers.core_model_loading import Concatenate, MergeModulelist, WeightConverter

        return WeightConverter(
            source_patterns=[
                "mlp.experts.*.gate_proj.weight",
                "mlp.experts.*.up_proj.weight",
            ],
            target_patterns="mlp.experts.gate_up_proj",
            operations=[MergeModulelist(dim=0), Concatenate(dim=1)],
        )

    def test_dense_module_converter_left_as_registered(self):
        """A converter whose modules the checkpoint does not quantize keeps consuming the plain keys."""
        quantizer = self._quantizer(self.nvfp4_experts_config)

        updated = quantizer.update_weight_conversions(
            [self._conv1d_converter()], self._glm5_next_like_model(), checkpoint_keys=self.checkpoint_keys
        )

        self.assertEqual(updated[0].source_patterns, self.conv1d_sources)

    def test_dense_module_converter_left_as_registered_fp8(self):
        """Same with FP8 serialized names: the plain `weight` is part of them, but the `weight_scale`
        companions the rewrite would collect are missing for unquantized modules."""
        quantizer = self._quantizer(self.fp8_experts_config)
        self.assertEqual(quantizer.serialized_names, {"weight": ["weight", "weight_scale"]})
        plain_keys = {key for key in self.checkpoint_keys if not key.endswith(self.packed_names)}

        updated = quantizer.update_weight_conversions(
            [self._conv1d_converter()], self._glm5_next_like_model(), checkpoint_keys=plain_keys
        )

        self.assertEqual(updated[0].source_patterns, self.conv1d_sources)

    def test_quantized_module_converter_still_rewritten(self):
        """A converter whose sources have serialized companions in the checkpoint still collects them and
        dequantizes first."""
        from transformers.integrations.compressed_tensors import CompressedTensorsDequantize

        quantizer = self._quantizer(self.nvfp4_experts_config)

        updated = quantizer.update_weight_conversions(
            [self._experts_converter()], self._glm5_next_like_model(), checkpoint_keys=self.checkpoint_keys
        )

        self.assertTrue(all(source.endswith(self.packed_names) for source in updated[0].source_patterns))
        self.assertIsInstance(updated[0].operations[0], CompressedTensorsDequantize)

    def test_fp8_converter_with_companions_still_rewritten(self):
        """FP8 sources that do ship `weight_scale` companions are still rewritten to collect them."""
        quantizer = self._quantizer(self.fp8_experts_config)
        fp8_keys = {
            "model.layers.0.self_attn.q_conv1d.weight",
            "model.layers.0.self_attn.q_conv1d.weight_scale",
            "model.layers.0.self_attn.k_conv1d.weight",
            "model.layers.0.self_attn.k_conv1d.weight_scale",
            "model.layers.0.self_attn.v_conv1d.weight",
            "model.layers.0.self_attn.v_conv1d.weight_scale",
        }

        updated = quantizer.update_weight_conversions(
            [self._conv1d_converter()], self._glm5_next_like_model(), checkpoint_keys=fp8_keys
        )

        self.assertEqual(
            sorted(updated[0].source_patterns),
            sorted(source + suffix for source in self.conv1d_sources for suffix in ("", "_scale")),
        )

    def test_fallback_when_checkpoint_keys_are_unavailable(self):
        """Checkpoints that cannot be inspected (e.g. a `state_dict` was passed directly) keep the previous
        behavior: the converter is rewritten unconditionally."""
        quantizer = self._quantizer(self.nvfp4_experts_config)

        updated = quantizer.update_weight_conversions(
            [self._conv1d_converter()], self._glm5_next_like_model(), checkpoint_keys=None
        )

        self.assertTrue(all(source.endswith(self.packed_names) for source in updated[0].source_patterns))

    def test_plain_checkpoint_keys_still_collected(self):
        """Simulate the renaming performed while loading: under the guard, the plain q/k/v conv1d checkpoint
        keys are collected by the untouched fusion converter instead of being reported as UNEXPECTED."""
        from transformers.core_model_loading import WeightConverter, rename_source_key

        quantizer = self._quantizer(self.nvfp4_experts_config)
        model = self._glm5_next_like_model()
        updated = quantizer.update_weight_conversions(
            [self._conv1d_converter()], model, checkpoint_keys=self.checkpoint_keys
        )
        converters = [entry for entry in updated if isinstance(entry, WeightConverter)]
        state_dict = model.state_dict()

        for proj in ("q", "k", "v"):
            key = f"model.layers.0.self_attn.{proj}_conv1d.weight"
            renamed, source_pattern = rename_source_key(key, [], converters, meta_state_dict=state_dict)
            self.assertEqual(renamed, "model.layers.0.self_attn.conv1d.weight")
            self.assertIn(renamed, state_dict)
            self.assertIsNotNone(source_pattern)

    def test_get_model_conversion_mapping_threads_checkpoint_keys(self):
        """`get_model_conversion_mapping` forwards the checkpoint keys to the quantizer."""
        from transformers import PreTrainedConfig, PreTrainedModel
        from transformers.conversion_mapping import (
            get_model_conversion_mapping,
            register_checkpoint_conversion_mapping,
        )

        model_type = "_ct_update_weight_conversions"
        register_checkpoint_conversion_mapping(model_type, [self._conv1d_converter()], overwrite=True)

        class _Model(PreTrainedModel):
            def __init__(self, config, torch_model):
                super().__init__(config)
                self.model = torch_model.model
                self.post_init()

        model = _Model(PreTrainedConfig(model_type=model_type), self._glm5_next_like_model())
        quantizer = self._quantizer(self.nvfp4_experts_config)

        transforms = get_model_conversion_mapping(
            model, hf_quantizer=quantizer, add_legacy=False, checkpoint_keys=self.checkpoint_keys
        )
        self.assertEqual(transforms[0].source_patterns, self.conv1d_sources)

        # Without the checkpoint keys, the converter falls back to being rewritten
        transforms = get_model_conversion_mapping(model, hf_quantizer=quantizer, add_legacy=False)
        self.assertTrue(all(source.endswith(self.packed_names) for source in transforms[0].source_patterns))
