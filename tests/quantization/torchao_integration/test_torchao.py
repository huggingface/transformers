# Copyright 2024 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import gc
import tempfile
import unittest
from unittest.mock import patch

from parameterized import parameterized

from transformers import AutoModelForCausalLM, AutoTokenizer, Qwen3MoeConfig, TorchAoConfig
from transformers.testing_utils import (
    Expectations,
    backend_empty_cache,
    require_cuda_capability_at_least,
    require_torch_accelerator,
    require_torch_multi_accelerator,
    require_torch_xpu,
    require_torchao,
    slow,
    torch_device,
)
from transformers.utils import is_torch_available, is_torchao_available


if is_torch_available():
    import torch

    from transformers.integrations.torchao import TorchAoQuantize
    from transformers.quantizers.quantizer_torchao import TorchAoHfQuantizer

if is_torchao_available():
    import torchao.quantization as torchao_quantization
    from torchao.prototype.mx_formats import NVFP4DynamicActivationNVFP4WeightConfig
    from torchao.quantization import (
        Float8DynamicActivationFloat8WeightConfig,
        Float8Tensor,
        Float8WeightOnlyConfig,
        FqnToConfig,
        Int4WeightOnlyConfig,
        Int8DynamicActivationInt8WeightConfig,
        Int8DynamicActivationIntxWeightConfig,
        Int8WeightOnlyConfig,
        IntxWeightOnlyConfig,
        MappingType,
        PerAxis,
    )
    from torchao.utils import (
        TorchAOBaseTensor,
    )


@require_torchao
class TorchAoConfigTest(unittest.TestCase):
    def test_to_dict(self):
        """
        Makes sure the config format is properly set
        """
        quantization_config = TorchAoConfig(Int4WeightOnlyConfig(group_size=32))
        torchao_orig_config = quantization_config.to_dict()

        self.assertIn("quant_type", torchao_orig_config)
        self.assertIn("quant_method", torchao_orig_config)
        self.assertEqual(torchao_orig_config["quant_method"], "torchao")

    def test_repr(self):
        """
        Check that there is no error in the repr
        """
        config = Int4WeightOnlyConfig(group_size=8)
        quantization_config = TorchAoConfig(config, modules_to_not_convert=["conv"])
        repr(quantization_config)

    def test_json_serializable(self):
        """
        Check that the config dict can be JSON serialized.
        """
        config = Int4WeightOnlyConfig(group_size=32)
        quantization_config = TorchAoConfig(config)
        d = quantization_config.to_dict()
        self.assertTrue("group_size" in d["quant_type"]["default"]["_data"])
        quantization_config.to_json_string(use_diff=False)


@require_torchao
class TorchAoParameterTargetingTest(unittest.TestCase):
    def setUp(self):
        self.model = torch.nn.Module()
        self.model.linear = torch.nn.Linear(32, 64, device="meta")
        self.model.embed_tokens = torch.nn.Embedding(64, 32, device="meta")
        self.model.norm = torch.nn.LayerNorm(32, device="meta")
        self.model.experts = torch.nn.Module()
        self.model.experts.gate_up_proj = torch.nn.Parameter(torch.empty(2, 64, 32, device="meta"))
        self.model.experts.down_proj = torch.nn.Parameter(torch.empty(2, 32, 64, device="meta"))
        self.model.experts.bias = torch.nn.Parameter(torch.empty(64, device="meta"))
        self.model.get_input_embeddings = lambda: self.model.embed_tokens
        self.config = Int4WeightOnlyConfig(int4_packing_format="plain_int32", group_size=32)

    def make_quantizer(self, config=None, **kwargs):
        quantizer = TorchAoHfQuantizer(
            TorchAoConfig(self.config if config is None else config, **kwargs), pre_quantized=False
        )
        quantizer.modules_to_not_convert = []
        quantizer.offload_to_cpu = False
        return quantizer

    @parameterized.expand([(True,), (False,)])
    def test_backend_parameter_selection(self, eligible):
        quantizer = self.make_quantizer()
        with patch.object(
            torchao_quantization, "config_targets_parameter", return_value=eligible, create=True
        ) as select:
            for name in ("gate_up_proj", "down_proj", "bias"):
                self.assertEqual(quantizer.param_needs_quantization(self.model, f"experts.{name}"), eligible)
                select.assert_called_with(self.model.experts, "experts", name, self.config)

    @parameterized.expand([("plain",), ("non_dynamic",), ("quantized",)])
    def test_backend_selection_preserves_linear_targets(self, kind):
        if kind == "non_dynamic":
            self.model.linear = torch.nn.modules.linear.NonDynamicallyQuantizableLinear(32, 64, device="meta")
        elif kind == "quantized":
            self.model.linear = torch.nn.Linear(32, 64)
            torchao_quantization.quantize_(self.model.linear, Int8WeightOnlyConfig(set_inductor_config=False))
            self.assertIsInstance(self.model.linear.weight, TorchAOBaseTensor)

        quantizer = self.make_quantizer()
        with patch.object(torchao_quantization, "config_targets_parameter", return_value=False, create=True) as select:
            self.assertTrue(quantizer.param_needs_quantization(self.model, "linear.weight"))
            select.assert_not_called()
            quantizer.modules_to_not_convert = ["linear"]
            self.assertFalse(quantizer.param_needs_quantization(self.model, "linear.weight"))
            select.assert_not_called()

    @parameterized.expand([(False,), (True,)])
    def test_legacy_selection_without_backend_helper(self, include_embeddings):
        quantizer = self.make_quantizer(include_input_output_embeddings=include_embeddings)
        with patch.dict(vars(torchao_quantization)):
            torchao_quantization.__dict__.pop("config_targets_parameter", None)
            for name, expected in (
                ("linear.weight", True),
                ("linear.bias", False),
                ("embed_tokens.weight", include_embeddings),
                ("norm.weight", False),
                ("experts.gate_up_proj", False),
                ("experts.down_proj", False),
                ("experts.bias", False),
            ):
                with self.subTest(name=name):
                    self.assertEqual(quantizer.param_needs_quantization(self.model, name), expected)

    @parameterized.expand([(False,), (True,)])
    def test_embedding_opt_in_and_module_exclusion(self, include_embeddings):
        quantizer = self.make_quantizer(include_input_output_embeddings=include_embeddings)
        quantizer.modules_to_not_convert = ["experts"]
        with patch.object(torchao_quantization, "config_targets_parameter", create=True) as select:
            self.assertFalse(quantizer.param_needs_quantization(self.model, "experts.gate_up_proj"))
            self.assertEqual(quantizer.param_needs_quantization(self.model, "embed_tokens.weight"), include_embeddings)
            select.assert_not_called()

    def test_explicit_fqn_does_not_use_backend_selection(self):
        config = FqnToConfig({r"re:experts\.(gate_up_proj|down_proj)": self.config})
        quantizer = self.make_quantizer(config)
        with patch.object(
            torchao_quantization,
            "config_targets_parameter",
            side_effect=AssertionError("unexpected selection"),
            create=True,
        ):
            self.assertTrue(quantizer.param_needs_quantization(self.model, "experts.gate_up_proj"))
            self.assertTrue(quantizer.param_needs_quantization(self.model, "experts.down_proj"))
            self.assertFalse(quantizer.param_needs_quantization(self.model, "experts.bias"))

    def test_backend_selection_errors_are_not_suppressed(self):
        quantizer = self.make_quantizer()
        with patch.object(
            torchao_quantization, "config_targets_parameter", side_effect=ValueError("invalid config"), create=True
        ):
            with self.assertRaisesRegex(ValueError, "invalid config"):
                quantizer.param_needs_quantization(self.model, "experts.gate_up_proj")

    def test_int4_custom_parameter_dispatch(self):
        quantizer = self.make_quantizer()
        value = torch.randn(2, 64, 32)
        missing_keys = {"experts.gate_up_proj", "experts.down_proj"}
        with patch.object(torchao_quantization, "quantize_") as quantize:
            result = TorchAoQuantize(quantizer).convert(
                {"gate_up_proj": [value]},
                model=self.model,
                full_layer_name="experts.gate_up_proj",
                missing_keys=missing_keys,
            )
        quantize.assert_called_once()
        self.assertIs(quantize.call_args.args[0], self.model.experts)
        self.assertEqual(quantize.call_args.args[1].fqn_to_config, {"gate_up_proj": self.config})
        self.assertEqual(quantize.call_args.kwargs, {"filter_fn": None})
        self.assertEqual(result, {})
        self.assertEqual(missing_keys, {"experts.down_proj"})
        self.assertEqual(self.model.experts.down_proj.device.type, "meta")
        torch.testing.assert_close(self.model.experts.gate_up_proj, value)

    @parameterized.expand([(False,), (True,)])
    def test_custom_parameter_quantization(self, explicit_fqn):
        config = Int8WeightOnlyConfig(set_inductor_config=False)
        if explicit_fqn:
            config = FqnToConfig({r"re:experts\.(gate_up_proj|down_proj)": config})
        quantizer = self.make_quantizer(config)
        missing_keys = {"experts.gate_up_proj", "experts.down_proj"}
        for name, shape in (("gate_up_proj", (2, 64, 32)), ("down_proj", (2, 32, 64))):
            result = TorchAoQuantize(quantizer).convert(
                {name: torch.randn(shape)},
                model=self.model,
                full_layer_name=f"experts.{name}",
                missing_keys=missing_keys,
            )
            self.assertEqual(result, {})
            self.assertIsInstance(getattr(self.model.experts, name), TorchAOBaseTensor)
            self.assertTrue(getattr(self.model.experts, name)._is_hf_initialized)
            if name == "gate_up_proj":
                self.assertEqual(self.model.experts.down_proj.device.type, "meta")
        self.assertEqual(missing_keys, set())
        self.assertTrue(self.model.experts._hf_quantized_needs_local_tp)

    @parameterized.expand([(False, False), (False, True), (True, False), (True, True)])
    @require_torch_accelerator
    def test_custom_parameter_cpu_offload(self, explicit_fqn, reverse_order):
        config = Int8WeightOnlyConfig(set_inductor_config=False)
        if explicit_fqn:
            config = FqnToConfig({r"re:experts\.(gate_up_proj|down_proj)": config})
        quantizer = self.make_quantizer(config)
        quantizer.offload_to_cpu = True
        targets = ["gate_up_proj", "down_proj"]
        if reverse_order:
            targets.reverse()
        missing_keys = {f"experts.{name}" for name in targets}
        quantize = torchao_quantization.quantize_

        for name in targets:
            sibling_name = next(other for other in targets if other != name)
            sibling = getattr(self.model.experts, sibling_name)
            value = torch.randn(getattr(self.model.experts, name).shape)

            def quantize_on_accelerator(module, config, **kwargs):
                self.assertEqual(getattr(module, name).device.type, torch_device)
                self.assertIs(getattr(module, sibling_name), sibling)
                return quantize(module, config, **kwargs)

            with (
                patch(
                    "transformers.integrations.torchao.get_current_accelerator",
                    return_value=torch.device(torch_device),
                ),
                patch.object(self.model.experts, "to", side_effect=AssertionError("must not move the entire module")),
                patch.object(torchao_quantization, "quantize_", side_effect=quantize_on_accelerator) as quantize_mock,
            ):
                result = TorchAoQuantize(quantizer).convert(
                    {name: value},
                    model=self.model,
                    full_layer_name=f"experts.{name}",
                    missing_keys=missing_keys,
                )

            quantize_mock.assert_called_once()
            self.assertEqual(result, {})
            self.assertIsInstance(getattr(self.model.experts, name), TorchAOBaseTensor)
            self.assertEqual(getattr(self.model.experts, name).device.type, "cpu")
            self.assertIs(getattr(self.model.experts, sibling_name), sibling)
            self.assertEqual(self.model.experts.bias.device.type, "meta")
            self.assertNotIn(f"experts.{name}", missing_keys)
        self.assertEqual(missing_keys, set())

    def test_linear_quantization_without_backend_helper(self):
        quantizer = self.make_quantizer(Int8WeightOnlyConfig(set_inductor_config=False))
        with patch.dict(vars(torchao_quantization)):
            torchao_quantization.__dict__.pop("config_targets_parameter", None)
            self.assertTrue(quantizer.param_needs_quantization(self.model, "linear.weight"))
            result = TorchAoQuantize(quantizer).convert(
                {"weight": torch.randn(64, 32)},
                model=self.model,
                full_layer_name="linear.weight",
                missing_keys={"linear.weight"},
            )
        self.assertEqual(result, {})
        self.assertIsInstance(self.model.linear.weight, TorchAOBaseTensor)

    @require_torch_xpu
    def test_moe_from_pretrained_xpu(self):
        config = Qwen3MoeConfig(
            vocab_size=128,
            hidden_size=128,
            intermediate_size=256,
            moe_intermediate_size=128,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=4,
            num_experts=2,
            num_experts_per_tok=1,
        )
        model = AutoModelForCausalLM.from_config(config)
        with tempfile.TemporaryDirectory() as checkpoint:
            model.save_pretrained(checkpoint)
            model = AutoModelForCausalLM.from_pretrained(
                checkpoint,
                quantization_config=TorchAoConfig(
                    Int4WeightOnlyConfig(int4_packing_format="plain_int32", group_size=32, set_inductor_config=False)
                ),
                dtype=torch.bfloat16,
                device_map="xpu",
                attn_implementation="eager",
                experts_implementation="eager",
            )
        self.assertIsInstance(model.model.layers[0].self_attn.q_proj.weight, TorchAOBaseTensor)
        has_backend_selection = getattr(torchao_quantization, "config_targets_parameter", None) is not None
        for name in ("gate_up_proj", "down_proj"):
            self.assertEqual(
                isinstance(getattr(model.model.layers[0].mlp.experts, name), TorchAOBaseTensor), has_backend_selection
            )
        with torch.no_grad():
            logits = model(torch.tensor([[1, 2, 3]], device="xpu")).logits
        self.assertEqual(logits.shape, (1, 3, 128))
        self.assertTrue(torch.isfinite(logits).all())


@require_torchao
@slow
class TorchAoTestBase:
    """Base mixin with all torchao test methods. Not a TestCase — subclass with unittest.TestCase to run."""

    input_text = "What are we having for dinner?"
    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    device = None  # must be set by subclass

    def tearDown(self):
        gc.collect()
        backend_empty_cache(torch_device)
        gc.collect()

    def test_int4wo_quant(self):
        """
        Simple LLM model testing int4 weight only quantization
        """
        int4_packing_format = "plain_int32" if self.device == "xpu" else "tile_packed_to_4d"
        config = Int4WeightOnlyConfig(int4_packing_format=int4_packing_format)
        quant_config = TorchAoConfig(config)

        quantized_model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            dtype=torch.bfloat16,
            device_map=self.device,
            quantization_config=quant_config,
        )
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        self.assertIn("Int4", type(quantized_model.model.layers[0].self_attn.v_proj.weight).__name__)

        input_ids = tokenizer(self.input_text, return_tensors="pt").to(self.device)

        output = quantized_model.generate(**input_ids, max_new_tokens=10)
        # fmt: off
        EXPECTED_OUTPUT = Expectations(
            {
                ("cuda", None): "What are we having for dinner?\nRed, white, and green beans,",
                ("xpu", None): "What are we having for dinner?\n\nJessica: (smiling)",
                ("xpu", 5): "What are we having for dinner?\n\n[Scene 2]\n\n[",
            }
        )
        # fmt: on
        self.assertEqual(tokenizer.decode(output[0], skip_special_tokens=True), EXPECTED_OUTPUT.get_expectation())

    def test_int8_dynamic_activation_int8_weight_quant(self):
        """
        Simple LLM model testing int8_dynamic_activation_int8_weight
        """
        config = Int8DynamicActivationInt8WeightConfig()
        quant_config = TorchAoConfig(config)

        quantized_model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            device_map=self.device,
            quantization_config=quant_config,
            torch_dtype=torch.bfloat16,
        )
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        input_ids = tokenizer(self.input_text, return_tensors="pt").to(self.device)

        output = quantized_model.generate(**input_ids, max_new_tokens=10)
        EXPECTED_OUTPUT = "What are we having for dinner?\n\nJessica: (smiling)"
        self.assertEqual(tokenizer.decode(output[0], skip_special_tokens=True), EXPECTED_OUTPUT)

    def test_include_input_output_embeddings(self):
        weight_dtype = torch.int8
        granularity = PerAxis(0)
        mapping_type = MappingType.ASYMMETRIC
        embedding_config = IntxWeightOnlyConfig(
            weight_dtype=weight_dtype,
            granularity=granularity,
            mapping_type=mapping_type,
        )
        config = FqnToConfig({"_default": None, "model.embed_tokens": embedding_config, "lm_head": embedding_config})
        # need set `include_input_output_embeddings` to True
        quant_config = TorchAoConfig(quant_type=config, include_input_output_embeddings=True)
        quantized_model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            device_map=self.device,
            quantization_config=quant_config,
            torch_dtype=torch.bfloat16,
        )
        # making sure embedding is quantized
        self.assertNotEqual(type(quantized_model.model.embed_tokens.weight).__name__, "Parameter")
        self.assertNotEqual(type(quantized_model.lm_head.weight).__name__, "Parameter")
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        input_ids = tokenizer(self.input_text, return_tensors="pt").to(self.device)

        output = quantized_model.generate(**input_ids, max_new_tokens=10)
        EXPECTED_OUTPUT = "What are we having for dinner?\n\nJessica: (smiling)"
        self.assertEqual(tokenizer.decode(output[0], skip_special_tokens=True), EXPECTED_OUTPUT)

    def test_per_module_config_skip(self):
        linear_config = Int8WeightOnlyConfig()
        config = FqnToConfig({"_default": linear_config, "model.layers.0.self_attn.q_proj": None})
        quant_config = TorchAoConfig(quant_type=config)
        quantized_model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            device_map=self.device,
            quantization_config=quant_config,
            torch_dtype=torch.bfloat16,
        )
        # making sure `model.layers.0.self_attn.q_proj` is skipped
        self.assertTrue(not isinstance(quantized_model.model.layers[0].self_attn.q_proj.weight, TorchAOBaseTensor))
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        input_ids = tokenizer(self.input_text, return_tensors="pt").to(self.device)

        output = quantized_model.generate(**input_ids, max_new_tokens=10)
        EXPECTED_OUTPUT = "What are we having for dinner?\n\nJessica: (smiling)"
        self.assertEqual(tokenizer.decode(output[0], skip_special_tokens=True), EXPECTED_OUTPUT)

    def test_fqn_to_config_regex_basic(self):
        linear_config = Int8WeightOnlyConfig()
        config = FqnToConfig({"_default": linear_config, r"re:model\.layers\..+\.self_attn\.q_proj": None})
        quant_config = TorchAoConfig(quant_type=config)
        quantized_model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            device_map=self.device,
            quantization_config=quant_config,
            torch_dtype=torch.bfloat16,
        )
        # making sure `model.layers.0.self_attn.q_proj` is skipped
        self.assertTrue(not isinstance(quantized_model.model.layers[0].self_attn.q_proj.weight, TorchAOBaseTensor))
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        input_ids = tokenizer(self.input_text, return_tensors="pt").to(self.device)

        output = quantized_model.generate(**input_ids, max_new_tokens=10)
        EXPECTED_OUTPUT = "What are we having for dinner?\n\nJessica: (smiling)"
        self.assertEqual(tokenizer.decode(output[0], skip_special_tokens=True), EXPECTED_OUTPUT)

    def test_fqn_to_config_regex_fullmatch(self):
        """Testing that we will only match the fqns that fully
        matches the regex
        """
        linear1_config = Int8WeightOnlyConfig()
        linear2_config = Float8WeightOnlyConfig()
        # intentially removing `j` after `q_proj` so it's not a full match
        config = FqnToConfig(
            {
                r"re:model\.layers\.+\.self_attn\.q_pro": linear1_config,
                "model.layers.3.self_attn.q_proj": linear2_config,
            }
        )
        quant_config = TorchAoConfig(quant_type=config)
        quantized_model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            device_map=self.device,
            quantization_config=quant_config,
            torch_dtype=torch.bfloat16,
        )
        # highest precedence is fully specified module fqn
        self.assertTrue(isinstance(quantized_model.model.layers[3].self_attn.q_proj.weight, Float8Tensor))
        # because regex `model\.layers\.+*\.self_attn\.q_pro` didin't fully match `model.layers.1.self_attn.q_proj` (missing last `j`)
        # this layer is not expected to be quantized to int8
        self.assertTrue(not isinstance(quantized_model.model.layers[1].self_attn.q_proj.weight, TorchAOBaseTensor))
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        input_ids = tokenizer(self.input_text, return_tensors="pt").to(self.device)

        output = quantized_model.generate(**input_ids, max_new_tokens=10)
        EXPECTED_OUTPUT = "What are we having for dinner?\n\nJessica: (smiling)"
        self.assertEqual(tokenizer.decode(output[0], skip_special_tokens=True), EXPECTED_OUTPUT)

    def test_fqn_to_config_module_regex_precedence(self):
        linear1_config = Int8WeightOnlyConfig()
        linear2_config = Float8WeightOnlyConfig()
        config = FqnToConfig(
            {
                r"re:model\.layers\..+\.self_attn\.q_proj": None,
                "model.layers.3.self_attn.q_proj": linear2_config,
                "_default": linear1_config,
            }
        )
        quant_config = TorchAoConfig(quant_type=config)
        quantized_model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            device_map=self.device,
            quantization_config=quant_config,
            torch_dtype=torch.bfloat16,
        )
        # highest precedence is fully specified module fqn
        self.assertTrue(isinstance(quantized_model.model.layers[3].self_attn.q_proj.weight, Float8Tensor))
        # second precedence: regex
        self.assertTrue(not isinstance(quantized_model.model.layers[1].self_attn.q_proj.weight, TorchAOBaseTensor))
        # last precedence: _default
        self.assertTrue(isinstance(quantized_model.model.layers[1].self_attn.k_proj.weight, TorchAOBaseTensor))
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        input_ids = tokenizer(self.input_text, return_tensors="pt").to(self.device)

        output = quantized_model.generate(**input_ids, max_new_tokens=10)
        EXPECTED_OUTPUT = "What are we having for dinner?\n\nJessica: (smiling)"
        self.assertEqual(tokenizer.decode(output[0], skip_special_tokens=True), EXPECTED_OUTPUT)

    def test_fqn_to_config_regex_precedence(self):
        linear1_config = Int8WeightOnlyConfig()
        linear2_config = Float8WeightOnlyConfig()
        config = FqnToConfig(
            {
                r"re:model\.layers\..+\.self_attn\.q_proj.weight": None,
                "model.layers.3.self_attn.q_proj.weight": linear2_config,
                "_default": linear1_config,
            }
        )
        quant_config = TorchAoConfig(quant_type=config)
        quantized_model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            device_map=self.device,
            quantization_config=quant_config,
            torch_dtype=torch.bfloat16,
        )
        self.assertTrue(isinstance(quantized_model.model.layers[3].self_attn.q_proj.weight, Float8Tensor))
        self.assertTrue(not isinstance(quantized_model.model.layers[1].self_attn.q_proj.weight, TorchAOBaseTensor))
        self.assertTrue(isinstance(quantized_model.model.layers[1].self_attn.k_proj.weight, TorchAOBaseTensor))
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        input_ids = tokenizer(self.input_text, return_tensors="pt").to(self.device)

        output = quantized_model.generate(**input_ids, max_new_tokens=10)
        EXPECTED_OUTPUT = "What are we having for dinner?\n\nJessica: (smiling)"
        self.assertEqual(tokenizer.decode(output[0], skip_special_tokens=True), EXPECTED_OUTPUT)

    def test_fqn_to_config_param_over_module_regex_precedence(self):
        linear1_config = Int8WeightOnlyConfig()
        linear2_config = Float8WeightOnlyConfig()
        config = FqnToConfig(
            {
                r"re:model\.layers\..+\.self_attn\.q_proj.weight": None,
                r"re:model\.layers\..+\.self_attn\.q_proj": linear2_config,
                "_default": linear1_config,
            }
        )
        quant_config = TorchAoConfig(quant_type=config)
        quantized_model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            device_map=self.device,
            quantization_config=quant_config,
            torch_dtype=torch.bfloat16,
        )
        self.assertTrue(not isinstance(quantized_model.model.layers[1].self_attn.q_proj.weight, TorchAOBaseTensor))
        self.assertTrue(isinstance(quantized_model.model.layers[1].self_attn.k_proj.weight, TorchAOBaseTensor))
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        input_ids = tokenizer(self.input_text, return_tensors="pt").to(self.device)

        output = quantized_model.generate(**input_ids, max_new_tokens=10)
        EXPECTED_OUTPUT = "What are we having for dinner?\n\nJessica: (smiling)"
        self.assertEqual(tokenizer.decode(output[0], skip_special_tokens=True), EXPECTED_OUTPUT)

    def test_fqn_to_config_param_over_module_precedence(self):
        linear1_config = Int8WeightOnlyConfig()
        linear2_config = Float8WeightOnlyConfig()
        config = FqnToConfig(
            {
                "model.layers.3.self_attn.q_proj.weight": None,
                "model.layers.3.self_attn.q_proj": linear2_config,
                "_default": linear1_config,
            }
        )
        quant_config = TorchAoConfig(quant_type=config)
        quantized_model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            device_map=self.device,
            quantization_config=quant_config,
            torch_dtype=torch.bfloat16,
        )
        self.assertTrue(not isinstance(quantized_model.model.layers[3].self_attn.q_proj.weight, TorchAOBaseTensor))
        self.assertTrue(isinstance(quantized_model.model.layers[3].self_attn.k_proj.weight, TorchAOBaseTensor))
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        input_ids = tokenizer(self.input_text, return_tensors="pt").to(self.device)

        output = quantized_model.generate(**input_ids, max_new_tokens=10)
        EXPECTED_OUTPUT = "What are we having for dinner?\n\nJessica: (smiling)"
        self.assertEqual(tokenizer.decode(output[0], skip_special_tokens=True), EXPECTED_OUTPUT)

    def test_fqn_to_config_exact_over_regex_precedence(self):
        linear1_config = Int8WeightOnlyConfig()
        linear2_config = Float8WeightOnlyConfig()
        config = FqnToConfig(
            {
                "model.layers.3.self_attn.q_proj.weight": None,
                "model.layers.1.self_attn.q_proj": linear1_config,
                r"re:model\.layers\..+\.self_attn\.q_proj.weight": linear2_config,
            }
        )
        quant_config = TorchAoConfig(quant_type=config)
        quantized_model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            device_map=self.device,
            quantization_config=quant_config,
            torch_dtype=torch.bfloat16,
        )
        self.assertTrue(not isinstance(quantized_model.model.layers[3].self_attn.q_proj.weight, TorchAOBaseTensor))
        self.assertTrue(isinstance(quantized_model.model.layers[1].self_attn.q_proj.weight, TorchAOBaseTensor))
        self.assertTrue(isinstance(quantized_model.model.layers[2].self_attn.q_proj.weight, Float8Tensor))

        tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        input_ids = tokenizer(self.input_text, return_tensors="pt").to(self.device)

        output = quantized_model.generate(**input_ids, max_new_tokens=10)
        EXPECTED_OUTPUT = "What are we having for dinner?\n\nJessica: (smiling)"
        self.assertEqual(tokenizer.decode(output[0], skip_special_tokens=True), EXPECTED_OUTPUT)

    @require_cuda_capability_at_least(8, 9)
    def test_fqn_to_config_non_weight_param(self):
        linear1_config = Int8WeightOnlyConfig()
        linear2_config = Float8WeightOnlyConfig()
        config = FqnToConfig(
            {
                r"re:.*gate_up_proj": linear2_config,
                "model.layers.0.feed_forward.experts.gate_up_proj": None,
                "_default": linear1_config,
            }
        )
        quant_config = TorchAoConfig(quant_type=config)
        quantized_model = AutoModelForCausalLM.from_pretrained(
            "jcaip/Llama-4-Scout-17B-two-layers-only-testing",
            device_map=self.device,
            dtype=torch.bfloat16,
            quantization_config=quant_config,
        )

        self.assertTrue(isinstance(quantized_model.model.layers[1].feed_forward.experts.gate_up_proj, Float8Tensor))
        self.assertTrue(
            not isinstance(quantized_model.model.layers[0].feed_forward.experts.gate_up_proj, Float8Tensor)
        )
        self.assertTrue(isinstance(quantized_model.model.layers[1].self_attn.q_proj.weight, TorchAOBaseTensor))

    def test_compute_module_sizes(self):
        r"""
        Test if we compute the right module sizes needed to generate the device map.
        Also test if we get the right values for `total_byte_count` in `caching_allocator_warmup`.
        """
        from transformers import AutoConfig
        from transformers.integrations.accelerate import compute_module_sizes
        from transformers.modeling_utils import expand_device_map, get_total_byte_count
        from transformers.quantizers import AutoHfQuantizer

        # we need to preprocess the model like that because device_map calculation happens before we load the weights inside the model.
        # For normal wieghts, it's fine but for quantized weights, the tensors dtype might change during loading.
        with torch.device("meta"):
            config = AutoConfig.from_pretrained(self.model_name)
            model = AutoModelForCausalLM.from_config(config, dtype=torch.bfloat16)
            model_size, _ = compute_module_sizes(model, only_modules=False)

            expected_keys = [name for name, _ in model.named_parameters()] + [
                name for name, _ in model.named_buffers()
            ]
            expanded_device_map = expand_device_map({"": torch_device}, expected_keys)
            total_byte_count = list(get_total_byte_count(model, expanded_device_map).values())[0]

            # testing prequantized = False should be enough, the shape should be the same whether it is pre-quantized or not
            hf_quantizer = AutoHfQuantizer.from_config(
                TorchAoConfig(quant_type=Int4WeightOnlyConfig()), pre_quantized=False
            )
            hf_quantizer.preprocess_model(model=model, config=model.config)
            quantized_model_size, _ = compute_module_sizes(model, hf_quantizer, only_modules=False)

            expected_keys = [name for name, _ in model.named_parameters()] + [
                name for name, _ in model.named_buffers()
            ]
            expanded_device_map = expand_device_map({"": torch_device}, expected_keys)
            quantized_total_byte_count = list(get_total_byte_count(model, expanded_device_map, hf_quantizer).values())[
                0
            ]

        for name, module in model.named_modules():
            # modules are not replaced when using torchao
            if isinstance(module, torch.nn.Linear) and "lm_head" not in name:
                # from 16 bits to 4 bits
                assert int(model_size[f"{name}.weight"] // 4) == int(quantized_model_size[f"{name}.weight"])

        # check that we get the same value, as we use `compute_module_sizes` in `get_total_byte_count`
        assert total_byte_count == model_size[""]
        assert quantized_total_byte_count == quantized_model_size[""]

        # we should at least have 1.5 times memory reduction in total
        assert model_size[""] > quantized_model_size[""] * 2


class TorchAoCPUTest(TorchAoTestBase, unittest.TestCase):
    device = "cpu"

    @unittest.skip("Int4 does not support CPU")
    def test_int4wo_quant(self):
        pass


@require_torch_accelerator
class TorchAoAcceleratorTest(TorchAoTestBase, unittest.TestCase):
    device = torch_device

    def test_int4wo_offload(self):
        """
        Test Int4 weight-only quantization with CPU offload.
        """
        device_map_offload = {
            "model.embed_tokens": 0,
            "model.layers.0": 0,
            "model.layers.1": 0,
            "model.layers.2": 0,
            "model.layers.3": 0,
            "model.layers.4": 0,
            "model.layers.5": 0,
            "model.layers.6": 0,
            "model.layers.7": 0,
            "model.layers.8": 0,
            "model.layers.9": 0,
            "model.layers.10": 0,
            "model.layers.11": 0,
            "model.layers.12": 0,
            "model.layers.13": 0,
            "model.layers.14": 0,
            "model.layers.15": 0,
            "model.layers.16": 0,
            "model.layers.17": 0,
            "model.layers.18": 0,
            "model.layers.19": "cpu",
            "model.layers.20": "cpu",
            "model.layers.21": "cpu",
            "model.norm": 0,
            "model.rotary_emb": 0,
            "lm_head": 0,
        }

        int4_packing_format = "plain_int32" if self.device == "xpu" else "tile_packed_to_4d"
        config = Int4WeightOnlyConfig(int4_packing_format=int4_packing_format)
        quant_config = TorchAoConfig(config)

        quantized_model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.bfloat16,
            device_map=device_map_offload,
            quantization_config=quant_config,
        )
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        input_ids = tokenizer(self.input_text, return_tensors="pt").to(self.device)

        output = quantized_model.generate(**input_ids, max_new_tokens=10)
        # fmt: off
        EXPECTED_OUTPUT = Expectations(
            {
                ("cuda", None): "What are we having for dinner?\nRed, white, and green beans,",
                ("xpu", None): "What are we having for dinner?\n\nJessica: (smiling)",
                ("xpu", 5): "What are we having for dinner?\n\n[Scene 2]\n\n[",
            }
        )
        # fmt: on
        self.assertEqual(tokenizer.decode(output[0], skip_special_tokens=True), EXPECTED_OUTPUT.get_expectation())

    @require_torch_multi_accelerator
    def test_int4wo_quant_multi_accelerator(self):
        """
        Simple test that checks if the quantized model int4 weight only is working properly with multiple accelerators
        set CUDA_VISIBLE_DEVICES=0,1 if you have more than 2 CUDA GPUs
        set ZE_AFFINITY_MASK=0,1 if you have more than 2 Intel XPUs
        """

        int4_packing_format = "plain_int32" if self.device == "xpu" else "tile_packed_to_4d"
        config = Int4WeightOnlyConfig(int4_packing_format=int4_packing_format)
        quant_config = TorchAoConfig(config)
        quantized_model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            quantization_config=quant_config,
        )
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        self.assertTrue(set(quantized_model.hf_device_map.values()) == {0, 1})

        input_ids = tokenizer(self.input_text, return_tensors="pt").to(self.device)

        output = quantized_model.generate(**input_ids, max_new_tokens=10)
        EXPECTED_OUTPUT = Expectations(
            {
                ("cuda", None): "What are we having for dinner?\nRed, white, and green beans,",
                ("xpu", None): "What are we having for dinner?\n\nJessica: (smiling)",
                ("xpu", 5): "What are we having for dinner?\n\n[Scene 2]\n\n[",
            }
        )
        self.assertEqual(tokenizer.decode(output[0], skip_special_tokens=True), EXPECTED_OUTPUT.get_expectation())


@slow
@require_torchao
class TorchAoSerializationTest(unittest.TestCase):
    """Parameterized serialization tests: quantize, save, reload, check output."""

    input_text = "What are we having for dinner?"
    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

    # fmt: off
    COMMON_OUTPUT = "What are we having for dinner?\n\nJessica: (smiling)"
    ALL_DEVICES_COMMON = Expectations({("cpu", None): COMMON_OUTPUT, ("cuda", None): COMMON_OUTPUT, ("xpu", None): COMMON_OUTPUT})

    test_params = (
        [
            ("Int8WeightOnlyConfig", Int8WeightOnlyConfig(version=2), ALL_DEVICES_COMMON),
            ("Int8DynamicActivationInt8WeightConfig", Int8DynamicActivationInt8WeightConfig(version=2), ALL_DEVICES_COMMON),
            ("Float8DynamicActivationFloat8WeightConfig", Float8DynamicActivationFloat8WeightConfig(), Expectations({("cuda", None): COMMON_OUTPUT, ("xpu", None): "What are we having for dinner?\n\nJess: (smiling) I", ("xpu", 5): COMMON_OUTPUT})),
            ("Float8WeightOnlyConfig", Float8WeightOnlyConfig(), Expectations({("cuda", None): COMMON_OUTPUT, ("xpu", None): COMMON_OUTPUT})),
            ("Int4WeightOnlyConfig", Int4WeightOnlyConfig(int4_packing_format="plain_int32" if torch_device == "xpu" else "tile_packed_to_4d"), Expectations({("cuda", None): "What are we having for dinner?\nRed, white, and green beans,", ("xpu", None): COMMON_OUTPUT, ("xpu", 5): "What are we having for dinner?\n\n[Scene 2]\n\n["})),
            ("Int8DynamicActivationIntxWeightConfig", Int8DynamicActivationIntxWeightConfig(), Expectations({("cpu", None): COMMON_OUTPUT, ("cuda", 9): COMMON_OUTPUT, ("cuda", 8): "What are we having for dinner?\n\nJEN: (smiling) I", ("xpu", None): COMMON_OUTPUT})),
            ("IntxWeightOnlyConfig", IntxWeightOnlyConfig(), ALL_DEVICES_COMMON),
            ("NVFP4DynamicActivationNVFP4WeightConfig", NVFP4DynamicActivationNVFP4WeightConfig(), Expectations({("cuda", None): "What are we having for dinner?\n\n10. Avoid using \"I"})),
        ]
        if is_torchao_available()
        else []
    )
    # fmt: on

    def tearDown(self):
        gc.collect()
        backend_empty_cache(torch_device)
        gc.collect()

    def _check_serialization(self, device, config, expected_output):
        if isinstance(config, (Float8DynamicActivationFloat8WeightConfig, Float8WeightOnlyConfig)):
            if torch.cuda.is_available() and torch.cuda.get_device_capability() < (8, 9):
                self.skipTest(f"{type(config).__name__} requires CUDA capability >= (8, 9)")
        if isinstance(config, NVFP4DynamicActivationNVFP4WeightConfig):
            if torch.cuda.is_available() and torch.cuda.get_device_capability() < (10, 0):
                self.skipTest(f"{type(config).__name__} requires CUDA capability >= (10, 0) (SM100)")
        quant_config = TorchAoConfig(config)
        needs_bfloat16 = isinstance(config, Int4WeightOnlyConfig | NVFP4DynamicActivationNVFP4WeightConfig)
        dtype = torch.bfloat16 if needs_bfloat16 else "auto"
        quantized_model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            dtype=dtype,
            device_map=device,
            quantization_config=quant_config,
        )
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        input_ids = tokenizer(self.input_text, return_tensors="pt").to(device)
        output = quantized_model.generate(**input_ids, max_new_tokens=10)
        self.assertEqual(tokenizer.decode(output[0], skip_special_tokens=True), expected_output)
        with tempfile.TemporaryDirectory() as tmpdirname:
            quantized_model.save_pretrained(tmpdirname)
            loaded_model = AutoModelForCausalLM.from_pretrained(tmpdirname, dtype=dtype, device_map=device)
            input_ids = tokenizer(self.input_text, return_tensors="pt").to(device)
            output = loaded_model.generate(**input_ids, max_new_tokens=10)
            self.assertEqual(tokenizer.decode(output[0], skip_special_tokens=True), expected_output)

    @parameterized.expand(test_params, skip_on_empty=True)
    def test_serialization_cpu(self, _name, config, expected_outputs):
        try:
            expected = expected_outputs.find_expectation(("cpu", None, None))
        except ValueError:
            self.skipTest(f"{type(config).__name__} does not support CPU")
        self._check_serialization("cpu", config, expected)

    @parameterized.expand(test_params, skip_on_empty=True)
    @require_torch_accelerator
    def test_serialization_accelerator(self, _name, config, expected_outputs):
        try:
            expected = expected_outputs.get_expectation()
        except ValueError:
            self.skipTest(f"{type(config).__name__} does not support {torch_device}")
        self._check_serialization(torch_device, config, expected)


if __name__ == "__main__":
    unittest.main()
