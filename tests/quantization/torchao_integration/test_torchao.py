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
import importlib.metadata
import tempfile
import unittest

from packaging import version
from parameterized import parameterized

from transformers import AutoModelForCausalLM, AutoTokenizer, TorchAoConfig
from transformers.testing_utils import (
    Expectations,
    backend_empty_cache,
    get_device_properties,
    require_torch_accelerator,
    require_torch_multi_accelerator,
    require_torchao,
    require_torchao_version_greater_or_equal,
    slow,
    torch_device,
)
from transformers.utils import is_torch_available, is_torchao_available


if is_torch_available():
    import torch

if is_torchao_available():
    import torchao

    # renamed in torchao 0.7.0, please install the latest torchao
    from torchao.dtypes import (
        AffineQuantizedTensor,
        TensorCoreTiledLayout,
    )
    from torchao.quantization import (
        Float8Tensor,
        Float8WeightOnlyConfig,
        Int4WeightOnlyConfig,
        Int8DynamicActivationInt8WeightConfig,
        Int8WeightOnlyConfig,
        IntxWeightOnlyConfig,
        MappingType,
        ModuleFqnToConfig,
        PerAxis,
    )
    from torchao.quantization.autoquant import AQMixin

    if version.parse(importlib.metadata.version("torchao")) >= version.parse("0.8.0"):
        from torchao.dtypes import Int4CPULayout
    if version.parse(importlib.metadata.version("torchao")) >= version.parse("0.11.0"):
        from torchao.dtypes import Int4XPULayout
    if version.parse(importlib.metadata.version("torchao")) >= version.parse("0.15.0"):
        from torchao.quantization import FqnToConfig


def check_torchao_int4_wo_quantized(test_module, qlayer):
    weight = qlayer.weight
    test_module.assertEqual(weight.quant_min, 0)
    test_module.assertEqual(weight.quant_max, 15)
    test_module.assertTrue(isinstance(weight, AffineQuantizedTensor))
    layout = None
    if weight.device.type == "cpu":
        layout = Int4CPULayout
    elif weight.device.type == "xpu":
        layout = Int4XPULayout
    elif weight.device.type == "cuda":
        layout = TensorCoreTiledLayout
    test_module.assertTrue(isinstance(weight.tensor_impl._layout, layout))


def check_autoquantized(test_module, qlayer):
    weight = qlayer.weight
    test_module.assertTrue(isinstance(weight, AQMixin))


def check_forward(test_module, model, batch_size=1, context_size=1024):
    # Test forward pass
    with torch.no_grad():
        out = model(torch.zeros([batch_size, context_size], device=model.device, dtype=torch.int32)).logits
    test_module.assertEqual(out.shape[0], batch_size)
    test_module.assertEqual(out.shape[1], context_size)


@require_torchao
@require_torchao_version_greater_or_equal("0.8.0")
class TorchAoConfigTest(unittest.TestCase):
    def test_to_dict(self):
        """
        Makes sure the config format is properly set
        """
        quantization_config = TorchAoConfig("int4_weight_only")
        torchao_orig_config = quantization_config.to_dict()

        for key in torchao_orig_config:
            self.assertEqual(getattr(quantization_config, key), torchao_orig_config[key])

    def test_post_init_check(self):
        """
        Test kwargs validations in TorchAoConfig
        """
        _ = TorchAoConfig("int4_weight_only")
        with self.assertRaisesRegex(ValueError, "Unsupported string quantization type"):
            _ = TorchAoConfig("fp6")

        with self.assertRaisesRegex(ValueError, "Unexpected keyword arg"):
            _ = TorchAoConfig("int4_weight_only", group_size1=32)

    def test_repr(self):
        """
        Check that there is no error in the repr
        """
        config = Int4WeightOnlyConfig(group_size=8, layout=TensorCoreTiledLayout())
        quantization_config = TorchAoConfig(config, modules_to_not_convert=["conv"])
        repr(quantization_config)

    def test_json_serializable(self):
        """
        Check that the config dict can be JSON serialized.
        """
        config = Int4WeightOnlyConfig(group_size=32, layout=TensorCoreTiledLayout())
        quantization_config = TorchAoConfig(config)
        d = quantization_config.to_dict()
        self.assertTrue("inner_k_tiles" in d["quant_type"]["default"]["_data"]["layout"]["_data"])
        quantization_config.to_json_string(use_diff=False)


@require_torchao
@require_torchao_version_greater_or_equal("0.8.0")
@slow
class TorchAoTest(unittest.TestCase):
    input_text = "What are we having for dinner?"
    max_new_tokens = 10
    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    device = "cpu"
    quant_scheme_kwargs = (
        {"group_size": 32, "layout": Int4CPULayout(), "version": 1}
        if is_torchao_available() and version.parse(importlib.metadata.version("torchao")) >= version.parse("0.8.0")
        else {"group_size": 32}
    )

    # called only once for all test in this class
    @classmethod
    def setUpClass(cls):
        cls.EXPECTED_OUTPUT = "What are we having for dinner?\n- 1. What is the temperature outside"

    def tearDown(self):
        gc.collect()
        backend_empty_cache(torch_device)
        gc.collect()

    def test_int4wo_quant(self):
        """
        Simple LLM model testing int4 weight only quantization
        """
        config = Int4WeightOnlyConfig(**self.quant_scheme_kwargs)
        quant_config = TorchAoConfig(config)

        # Note: we quantize the bfloat16 model on the fly to int4
        quantized_model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            dtype=torch.bfloat16,
            device_map=self.device,
            quantization_config=quant_config,
        )
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        check_torchao_int4_wo_quantized(self, quantized_model.model.layers[0].self_attn.v_proj)

        input_ids = tokenizer(self.input_text, return_tensors="pt").to(self.device)

        output = quantized_model.generate(**input_ids, max_new_tokens=self.max_new_tokens)
        self.assertEqual(tokenizer.decode(output[0], skip_special_tokens=True), self.EXPECTED_OUTPUT)

    def test_int4wo_quant_bfloat16_conversion(self):
        """
        Testing the dtype of model will be modified to be bfloat16 for int4 weight only quantization
        """
        config = Int4WeightOnlyConfig(**self.quant_scheme_kwargs)
        quant_config = TorchAoConfig(config)

        # Note: we quantize the bfloat16 model on the fly to int4
        quantized_model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            dtype=torch.bfloat16,
            device_map=self.device,
            quantization_config=quant_config,
        )
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        check_torchao_int4_wo_quantized(self, quantized_model.model.layers[0].self_attn.v_proj)

        input_ids = tokenizer(self.input_text, return_tensors="pt").to(self.device)

        output = quantized_model.generate(**input_ids, max_new_tokens=self.max_new_tokens)
        self.assertEqual(tokenizer.decode(output[0], skip_special_tokens=True), self.EXPECTED_OUTPUT)

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

        output = quantized_model.generate(**input_ids, max_new_tokens=self.max_new_tokens)
        EXPECTED_OUTPUT = [
            "What are we having for dinner?\n\nJessica: (smiling)",
            "What are we having for dinner?\n\nJess: (smiling) I",
        ]
        self.assertTrue(tokenizer.decode(output[0], skip_special_tokens=True) in EXPECTED_OUTPUT)

    @require_torchao_version_greater_or_equal("0.11.0")
    def test_include_input_output_embeddings(self):
        weight_dtype = torch.int8
        granularity = PerAxis(0)
        mapping_type = MappingType.ASYMMETRIC
        embedding_config = IntxWeightOnlyConfig(
            weight_dtype=weight_dtype,
            granularity=granularity,
            mapping_type=mapping_type,
            version=1,
        )
        config = ModuleFqnToConfig(
            {"_default": None, "model.embed_tokens": embedding_config, "lm_head": embedding_config}
        )
        # need set `include_input_output_embeddings` to True
        quant_config = TorchAoConfig(quant_type=config, include_input_output_embeddings=True)
        quantized_model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            device_map=self.device,
            quantization_config=quant_config,
            torch_dtype=torch.bfloat16,
        )
        # making sure embedding is quantized
        self.assertTrue(isinstance(quantized_model.model.embed_tokens.weight, AffineQuantizedTensor))
        self.assertTrue(isinstance(quantized_model.lm_head.weight, AffineQuantizedTensor))
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        input_ids = tokenizer(self.input_text, return_tensors="pt").to(self.device)

        output = quantized_model.generate(**input_ids, max_new_tokens=self.max_new_tokens)
        EXPECTED_OUTPUT = [
            "What are we having for dinner?\n\nJessica: (smiling)",
            "What are we having for dinner?\n\nJess: (smiling) I",
        ]
        self.assertTrue(tokenizer.decode(output[0], skip_special_tokens=True) in EXPECTED_OUTPUT)

    @require_torchao_version_greater_or_equal("0.11.0")
    def test_per_module_config_skip(self):
        linear_config = Int8WeightOnlyConfig()
        config = ModuleFqnToConfig({"_default": linear_config, "model.layers.0.self_attn.q_proj": None})
        quant_config = TorchAoConfig(quant_type=config)
        quantized_model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            device_map=self.device,
            quantization_config=quant_config,
            torch_dtype=torch.bfloat16,
        )
        # making sure `model.layers.0.self_attn.q_proj` is skipped
        self.assertTrue(not isinstance(quantized_model.model.layers[0].self_attn.q_proj.weight, AffineQuantizedTensor))
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        input_ids = tokenizer(self.input_text, return_tensors="pt").to(self.device)

        output = quantized_model.generate(**input_ids, max_new_tokens=self.max_new_tokens)
        EXPECTED_OUTPUT = [
            "What are we having for dinner?\n\nJessica: (smiling)",
            "What are we having for dinner?\n\nJess: (smiling) I",
        ]
        self.assertTrue(tokenizer.decode(output[0], skip_special_tokens=True) in EXPECTED_OUTPUT)

    @require_torchao_version_greater_or_equal("0.13.0")
    def test_module_fqn_to_config_regex_basic(self):
        linear_config = Int8WeightOnlyConfig()
        config = ModuleFqnToConfig({"_default": linear_config, r"re:model\.layers\..+\.self_attn\.q_proj": None})
        quant_config = TorchAoConfig(quant_type=config)
        quantized_model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            device_map=self.device,
            quantization_config=quant_config,
            torch_dtype=torch.bfloat16,
        )
        # making sure `model.layers.0.self_attn.q_proj` is skipped
        self.assertTrue(not isinstance(quantized_model.model.layers[0].self_attn.q_proj.weight, AffineQuantizedTensor))
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        input_ids = tokenizer(self.input_text, return_tensors="pt").to(self.device)

        output = quantized_model.generate(**input_ids, max_new_tokens=self.max_new_tokens)
        EXPECTED_OUTPUT = [
            "What are we having for dinner?\n\nJessica: (smiling)",
            "What are we having for dinner?\n\nJess: (smiling) I",
        ]
        self.assertTrue(tokenizer.decode(output[0], skip_special_tokens=True) in EXPECTED_OUTPUT)

    @require_torchao_version_greater_or_equal("0.13.0")
    def test_module_fqn_to_config_regex_fullmatch(self):
        """Testing that we will only match the fqns that fully
        matches the regex
        """
        linear1_config = Int8WeightOnlyConfig()
        linear2_config = Float8WeightOnlyConfig()
        # intentially removing `j` after `q_proj` so it's not a full match
        config = ModuleFqnToConfig(
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
        self.assertTrue(not isinstance(quantized_model.model.layers[1].self_attn.q_proj.weight, AffineQuantizedTensor))
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        input_ids = tokenizer(self.input_text, return_tensors="pt").to(self.device)

        output = quantized_model.generate(**input_ids, max_new_tokens=self.max_new_tokens)
        EXPECTED_OUTPUT = [
            "What are we having for dinner?\n\nJessica: (smiling)",
            "What are we having for dinner?\n\nJess: (smiling) I",
        ]
        self.assertTrue(tokenizer.decode(output[0], skip_special_tokens=True) in EXPECTED_OUTPUT)

    @require_torchao_version_greater_or_equal("0.13.0")
    def test_module_fqn_to_config_regex_precedence(self):
        linear1_config = Int8WeightOnlyConfig()
        linear2_config = Float8WeightOnlyConfig()
        config = ModuleFqnToConfig(
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
        self.assertTrue(not isinstance(quantized_model.model.layers[1].self_attn.q_proj.weight, AffineQuantizedTensor))
        # last precedence: _default
        self.assertTrue(isinstance(quantized_model.model.layers[1].self_attn.k_proj.weight, AffineQuantizedTensor))
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        input_ids = tokenizer(self.input_text, return_tensors="pt").to(self.device)

        output = quantized_model.generate(**input_ids, max_new_tokens=self.max_new_tokens)
        EXPECTED_OUTPUT = [
            "What are we having for dinner?\n\nJessica: (smiling)",
            "What are we having for dinner?\n\nJess: (smiling) I",
        ]
        self.assertTrue(tokenizer.decode(output[0], skip_special_tokens=True) in EXPECTED_OUTPUT)

    @require_torchao_version_greater_or_equal("0.15.0")
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
        self.assertTrue(not isinstance(quantized_model.model.layers[1].self_attn.q_proj.weight, AffineQuantizedTensor))
        self.assertTrue(isinstance(quantized_model.model.layers[1].self_attn.k_proj.weight, AffineQuantizedTensor))
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        input_ids = tokenizer(self.input_text, return_tensors="pt").to(self.device)

        output = quantized_model.generate(**input_ids, max_new_tokens=self.max_new_tokens)
        EXPECTED_OUTPUT = [
            "What are we having for dinner?\n\nJessica: (smiling)",
            "What are we having for dinner?\n\nJess: (smiling) I",
        ]
        self.assertTrue(tokenizer.decode(output[0], skip_special_tokens=True) in EXPECTED_OUTPUT)

    @require_torchao_version_greater_or_equal("0.15.0")
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
        self.assertTrue(not isinstance(quantized_model.model.layers[1].self_attn.q_proj.weight, AffineQuantizedTensor))
        self.assertTrue(isinstance(quantized_model.model.layers[1].self_attn.k_proj.weight, AffineQuantizedTensor))
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        input_ids = tokenizer(self.input_text, return_tensors="pt").to(self.device)

        output = quantized_model.generate(**input_ids, max_new_tokens=self.max_new_tokens)
        EXPECTED_OUTPUT = [
            "What are we having for dinner?\n\nJessica: (smiling)",
            "What are we having for dinner?\n\nJess: (smiling) I",
        ]
        self.assertTrue(tokenizer.decode(output[0], skip_special_tokens=True) in EXPECTED_OUTPUT)

    @require_torchao_version_greater_or_equal("0.15.0")
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
        self.assertTrue(not isinstance(quantized_model.model.layers[3].self_attn.q_proj.weight, AffineQuantizedTensor))
        self.assertTrue(isinstance(quantized_model.model.layers[3].self_attn.k_proj.weight, AffineQuantizedTensor))
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        input_ids = tokenizer(self.input_text, return_tensors="pt").to(self.device)

        output = quantized_model.generate(**input_ids, max_new_tokens=self.max_new_tokens)
        EXPECTED_OUTPUT = [
            "What are we having for dinner?\n\nJessica: (smiling)",
            "What are we having for dinner?\n\nJess: (smiling) I",
        ]
        self.assertTrue(tokenizer.decode(output[0], skip_special_tokens=True) in EXPECTED_OUTPUT)

    @require_torchao_version_greater_or_equal("0.15.0")
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
        self.assertTrue(not isinstance(quantized_model.model.layers[3].self_attn.q_proj.weight, AffineQuantizedTensor))
        self.assertTrue(isinstance(quantized_model.model.layers[1].self_attn.q_proj.weight, AffineQuantizedTensor))
        self.assertTrue(isinstance(quantized_model.model.layers[2].self_attn.q_proj.weight, Float8Tensor))

        tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        input_ids = tokenizer(self.input_text, return_tensors="pt").to(self.device)

        output = quantized_model.generate(**input_ids, max_new_tokens=self.max_new_tokens)
        EXPECTED_OUTPUT = [
            "What are we having for dinner?\n\nJessica: (smiling)",
            "What are we having for dinner?\n\nJess: (smiling) I",
        ]
        self.assertTrue(tokenizer.decode(output[0], skip_special_tokens=True) in EXPECTED_OUTPUT)

    @require_torchao_version_greater_or_equal("0.15.0")
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
            device_map="cuda",
            dtype=torch.bfloat16,
            quantization_config=quant_config,
        )

        self.assertTrue(isinstance(quantized_model.model.layers[1].feed_forward.experts.gate_up_proj, Float8Tensor))
        self.assertTrue(
            not isinstance(quantized_model.model.layers[0].feed_forward.experts.gate_up_proj, Float8Tensor)
        )
        self.assertTrue(isinstance(quantized_model.model.layers[1].self_attn.q_proj.weight, AffineQuantizedTensor))

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
                TorchAoConfig(quant_type=Int4WeightOnlyConfig(**self.quant_scheme_kwargs)), pre_quantized=False
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


@require_torch_accelerator
class TorchAoAcceleratorTest(TorchAoTest):
    device = torch_device
    quant_scheme_kwargs = {"group_size": 32, "version": 1}

    # called only once for all test in this class
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        # fmt: off
        EXPECTED_OUTPUTS = Expectations(
            {
                ("xpu", 3): "What are we having for dinner?\n\nJessica: (smiling)",
                ("cuda", 7): "What are we having for dinner?\n- 1. What is the temperature outside",
            }
        )
        # fmt: on
        cls.EXPECTED_OUTPUT = EXPECTED_OUTPUTS.get_expectation()

    def test_int4wo_offload(self):
        """
        Simple test that checks if the quantized model int4 weight only is working properly with cpu/disk offload
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

        config = Int4WeightOnlyConfig(**self.quant_scheme_kwargs)
        quant_config = TorchAoConfig(config)

        quantized_model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.bfloat16,
            device_map=device_map_offload,
            quantization_config=quant_config,
        )
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        input_ids = tokenizer(self.input_text, return_tensors="pt").to(self.device)

        # fmt: off
        EXPECTED_OUTPUTS_DEVICES = Expectations(
            {
                ("xpu", 3): ["What are we having for dinner?\n\nJessica: (smiling)"],
                ("cuda", 7): ["What are we having for dinner?\n- 1. What is the temperature outside",
                              "What are we having for dinner?"],
            }
        )
        # fmt: on
        EXPECTED_OUTPUTS = EXPECTED_OUTPUTS_DEVICES.get_expectation()

        output = quantized_model.generate(**input_ids, max_new_tokens=self.max_new_tokens)
        generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

        self.assertIn(generated_text, EXPECTED_OUTPUTS)

    @require_torch_multi_accelerator
    def test_int4wo_quant_multi_accelerator(self):
        """
        Simple test that checks if the quantized model int4 weight only is working properly with multiple accelerators
        set CUDA_VISIBLE_DEVICES=0,1 if you have more than 2 CUDA GPUs
        set ZE_AFFINITY_MASK=0,1 if you have more than 2 Intel XPUs
        """

        config = Int4WeightOnlyConfig(**self.quant_scheme_kwargs)
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

        output = quantized_model.generate(**input_ids, max_new_tokens=self.max_new_tokens)
        self.assertEqual(tokenizer.decode(output[0], skip_special_tokens=True), self.EXPECTED_OUTPUT)

    def test_autoquant(self):
        """
        Simple LLM model testing autoquant
        """
        quant_config = TorchAoConfig("autoquant")

        quantized_model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype="auto",
            device_map=self.device,
            quantization_config=quant_config,
        )
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        input_ids = tokenizer(self.input_text, return_tensors="pt").to(self.device)
        output = quantized_model.generate(
            **input_ids, max_new_tokens=self.max_new_tokens, cache_implementation="static"
        )
        quantized_model.finalize_autoquant()

        check_autoquantized(self, quantized_model.model.layers[0].self_attn.v_proj)

        EXPECTED_OUTPUTS = ["What are we having for dinner?\n\nJessica: (smiling)", "What are we having for dinner?"]

        output = quantized_model.generate(
            **input_ids, max_new_tokens=self.max_new_tokens, cache_implementation="static"
        )
        self.assertIn(tokenizer.decode(output[0], skip_special_tokens=True), EXPECTED_OUTPUTS)


@require_torchao_version_greater_or_equal("0.15.0")
@slow
class TorchAoSerializationTest(unittest.TestCase):
    input_text = "What are we having for dinner?"
    max_new_tokens = 10
    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

    device = "cpu"

    # called only once for all test in this class
    @classmethod
    def setUpClass(cls):
        cls.quant_scheme_kwargs = (
            {"group_size": 32, "layout": Int4CPULayout(), "version": 1}
            if is_torchao_available()
            and version.parse(importlib.metadata.version("torchao")) >= version.parse("0.8.0")
            else {"group_size": 32}
        )
        cls.quant_scheme = Int4WeightOnlyConfig(**cls.quant_scheme_kwargs)
        cls.tokenizer = AutoTokenizer.from_pretrained(cls.model_name)
        cls.EXPECTED_OUTPUT = "What are we having for dinner?\n- 1. What is the temperature outside"

    def setUp(self):
        self.quant_config = TorchAoConfig(self.quant_scheme)
        dtype = torch.bfloat16 if isinstance(self.quant_scheme, Int4WeightOnlyConfig) else "auto"
        self.quantized_model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            dtype=dtype,
            device_map=self.device,
            quantization_config=self.quant_config,
        )

    def tearDown(self):
        gc.collect()
        backend_empty_cache(torch_device)
        gc.collect()

    def test_original_model_expected_output(self):
        input_ids = self.tokenizer(self.input_text, return_tensors="pt").to(self.device)
        output = self.quantized_model.generate(**input_ids, max_new_tokens=self.max_new_tokens)

        self.assertEqual(self.tokenizer.decode(output[0], skip_special_tokens=True), self.EXPECTED_OUTPUT)

    def check_serialization_expected_output(self, device, expected_output):
        """
        Test if we can serialize and load/infer the model again on the same device
        """
        dtype = torch.bfloat16 if isinstance(self.quant_scheme, Int4WeightOnlyConfig) else "auto"
        with tempfile.TemporaryDirectory() as tmpdirname:
            self.quantized_model.save_pretrained(tmpdirname)

            loaded_quantized_model = AutoModelForCausalLM.from_pretrained(
                tmpdirname, dtype=dtype, device_map=device, torch_dtype=dtype
            )
            input_ids = self.tokenizer(self.input_text, return_tensors="pt").to(device)

            output = loaded_quantized_model.generate(**input_ids, max_new_tokens=self.max_new_tokens)
            self.assertEqual(self.tokenizer.decode(output[0], skip_special_tokens=True), expected_output)

    def test_serialization_expected_output(self):
        self.check_serialization_expected_output(self.device, self.EXPECTED_OUTPUT)


@require_torchao
@require_torchao_version_greater_or_equal("0.15.0")
class TorchAoSafeSerializationTest(TorchAoSerializationTest):
    # called only once for all test in this class
    @classmethod
    def setUpClass(cls):
        cls.tokenizer = AutoTokenizer.from_pretrained(cls.model_name)
        cls.EXPECTED_OUTPUT = "What are we having for dinner?\n\nJessica: (smiling)"
        # placeholder
        cls.quant_scheme = torchao.quantization.Float8WeightOnlyConfig()

    def tearDown(self):
        gc.collect()
        backend_empty_cache(torch_device)
        gc.collect()
        if hasattr(self, "quantized_model"):
            del self.quantized_model
        gc.collect()

    test_params = (
        [
            (
                torchao.quantization.Float8DynamicActivationFloat8WeightConfig(),
                "What are we having for dinner?\n\nJess: (smiling) I",
            ),
            (torchao.quantization.Float8WeightOnlyConfig(), "What are we having for dinner?\n\nJessica: (smiling)"),
            (Int4WeightOnlyConfig(), "What are we having for dinner?"),
            (
                Int4WeightOnlyConfig(int4_packing_format="tile_packed_to_4d"),
                "What are we having for dinner?\nRed, white, and green beans,",
            ),
            (
                torchao.quantization.Int8DynamicActivationIntxWeightConfig(),
                "What are we having for dinner?\n\nJessica: (smiling)",
            ),
            (torchao.quantization.IntxWeightOnlyConfig(), "What are we having for dinner?\n\nJessica: (smiling)"),
        ]
        if is_torchao_available()
        else []
    )

    @parameterized.expand(test_params, skip_on_empty=True)
    def test_serialization_expected_output(self, config, expected_output):
        device = "cuda"
        self.quant_config = TorchAoConfig(config)
        self.quantized_model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            dtype=torch.bfloat16,
            device_map=device,
            quantization_config=self.quant_config,
        )
        self.check_serialization_expected_output(device, expected_output)


class TorchAoSerializationW8A8CPUTest(TorchAoSerializationTest):
    # called only once for all test in this class
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.quant_scheme = Int8DynamicActivationInt8WeightConfig()
        cls.quant_scheme_kwargs = {}
        cls.EXPECTED_OUTPUT = "What are we having for dinner?\n\nJessica: (smiling)"

    @require_torch_accelerator
    def test_serialization_expected_output_on_accelerator(self):
        """
        Test if we can serialize on device (cpu) and load/infer the model on accelerator
        """
        self.check_serialization_expected_output(torch_device, self.EXPECTED_OUTPUT)


class TorchAoSerializationW8CPUTest(TorchAoSerializationTest):
    # called only once for all test in this class
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.quant_scheme = Int8WeightOnlyConfig()
        cls.quant_scheme_kwargs = {}
        cls.EXPECTED_OUTPUT = "What are we having for dinner?\n\nJessica: (smiling)"

    @require_torch_accelerator
    def test_serialization_expected_output_on_accelerator(self):
        """
        Test if we can serialize on device (cpu) and load/infer the model on accelerator
        """
        self.check_serialization_expected_output(torch_device, self.EXPECTED_OUTPUT)


@require_torch_accelerator
@require_torchao
class TorchAoSerializationAcceleratorTest(TorchAoSerializationTest):
    device = f"{torch_device}:0"

    # called only once for all test in this class
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        # fmt: off
        cls.quant_scheme = Int4WeightOnlyConfig(**{"group_size": 32, "version": 1})
        cls.quant_scheme_kwargs = {}
        EXPECTED_OUTPUTS = Expectations(
            {
                ("xpu", 3): "What are we having for dinner?\n\nJessica: (smiling)",
                ("cuda", 7): "What are we having for dinner?\n- 1. What is the temperature outside",
            }
        )
        # fmt: on
        cls.EXPECTED_OUTPUT = EXPECTED_OUTPUTS.get_expectation()


@require_torch_accelerator
class TorchAoSerializationW8A8AcceleratorTest(TorchAoSerializationTest):
    device = f"{torch_device}:0"

    # called only once for all test in this class
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.quant_scheme = Int8DynamicActivationInt8WeightConfig()
        cls.quant_scheme_kwargs = {}
        cls.EXPECTED_OUTPUT = "What are we having for dinner?\n\nJessica: (smiling)"


@require_torch_accelerator
class TorchAoSerializationW8AcceleratorTest(TorchAoSerializationTest):
    device = f"{torch_device}:0"

    # called only once for all test in this class
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.quant_scheme = Int8WeightOnlyConfig()
        cls.quant_scheme_kwargs = {}
        cls.EXPECTED_OUTPUT = "What are we having for dinner?\n\nJessica: (smiling)"


@require_torch_accelerator
@require_torchao_version_greater_or_equal("0.10.0")
class TorchAoSerializationFP8AcceleratorTest(TorchAoSerializationTest):
    device = f"{torch_device}:0"

    # called only once for all test in this class
    @classmethod
    def setUpClass(cls):
        device_type, major, minor = get_device_properties()
        if device_type == "cuda" and major < 9:
            raise unittest.SkipTest("CUDA compute capability 9.0 or higher required for FP8 tests")

        from torchao.quantization import Float8WeightOnlyConfig

        super().setUpClass()
        cls.quant_scheme = Float8WeightOnlyConfig()
        cls.quant_scheme_kwargs = {}

        cls.EXPECTED_OUTPUT = "What are we having for dinner?\n\nJessica: (smiling)"


@require_torch_accelerator
@require_torchao_version_greater_or_equal("0.10.0")
class TorchAoSerializationA8W4Test(TorchAoSerializationTest):
    device = f"{torch_device}:0"

    # called only once for all test in this class
    @classmethod
    def setUpClass(cls):
        device_type, major, minor = get_device_properties()
        if device_type == "cuda" and major < 9:
            raise unittest.SkipTest("CUDA compute capability 9.0 or higher required for FP8 tests")

        from torchao.quantization import Int8DynamicActivationInt4WeightConfig

        super().setUpClass()
        cls.quant_scheme = Int8DynamicActivationInt4WeightConfig()
        cls.quant_scheme_kwargs = {}

        cls.EXPECTED_OUTPUT = "What are we having for dinner?\n\nJessica: (smiling)"


if __name__ == "__main__":
    unittest.main()
