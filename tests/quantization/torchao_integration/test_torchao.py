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

from transformers import AutoModelForCausalLM, AutoTokenizer, TorchAoConfig
from transformers.testing_utils import (
    Expectations,
    backend_empty_cache,
    get_device_properties,
    require_torch_accelerator,
    require_torch_multi_accelerator,
    require_torchao,
    require_torchao_version_greater_or_equal,
    torch_device,
)
from transformers.utils import is_torch_available, is_torchao_available


if is_torch_available():
    import torch

if is_torchao_available():
    # renamed in torchao 0.7.0, please install the latest torchao
    from torchao.dtypes import (
        AffineQuantizedTensor,
        TensorCoreTiledLayout,
    )
    from torchao.quantization import (
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
        quantization_config = TorchAoConfig("int4_weight_only", modules_to_not_convert=["conv"], group_size=8)
        repr(quantization_config)

    def test_json_serializable(self):
        """
        Check that the config dict can be JSON serialized.
        """
        quantization_config = TorchAoConfig("int4_weight_only", group_size=32, layout=TensorCoreTiledLayout())
        d = quantization_config.to_dict()
        self.assertIsInstance(d["quant_type_kwargs"]["layout"], list)
        self.assertTrue("inner_k_tiles" in d["quant_type_kwargs"]["layout"][1])
        quantization_config.to_json_string(use_diff=False)


@require_torchao
@require_torchao_version_greater_or_equal("0.8.0")
class TorchAoTest(unittest.TestCase):
    input_text = "What are we having for dinner?"
    max_new_tokens = 10
    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    device = "cpu"
    quant_scheme_kwargs = (
        {"group_size": 32, "layout": Int4CPULayout()}
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
        quant_config = TorchAoConfig("int4_weight_only", **self.quant_scheme_kwargs)

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
        quant_config = TorchAoConfig("int4_weight_only", **self.quant_scheme_kwargs)

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
        quant_config = TorchAoConfig("int8_dynamic_activation_int8_weight")

        quantized_model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            device_map=self.device,
            quantization_config=quant_config,
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


@require_torch_accelerator
class TorchAoAcceleratorTest(TorchAoTest):
    device = torch_device
    quant_scheme_kwargs = {"group_size": 32}

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
            "model.layers.21": "disk",
            "model.norm": 0,
            "model.rotary_emb": 0,
            "lm_head": 0,
        }

        quant_config = TorchAoConfig("int4_weight_only", group_size=32)

        quantized_model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            dtype=torch.bfloat16,
            device_map=device_map_offload,
            quantization_config=quant_config,
        )
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        input_ids = tokenizer(self.input_text, return_tensors="pt").to(self.device)

        # fmt: off
        EXPECTED_OUTPUTS = Expectations(
            {
                ("xpu", 3): "What are we having for dinner?\n\nJessica: (smiling)",
                ("cuda", 7): "What are we having for dinner?\n- 2. What is the temperature outside",
            }
        )
        # fmt: on
        EXPECTED_OUTPUT = EXPECTED_OUTPUTS.get_expectation()

        output = quantized_model.generate(**input_ids, max_new_tokens=self.max_new_tokens)
        generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

        self.assertEqual(generated_text, EXPECTED_OUTPUT)

    @require_torch_multi_accelerator
    def test_int4wo_quant_multi_accelerator(self):
        """
        Simple test that checks if the quantized model int4 weight only is working properly with multiple accelerators
        set CUDA_VISIBLE_DEVICES=0,1 if you have more than 2 CUDA GPUs
        set ZE_AFFINITY_MASK=0,1 if you have more than 2 Intel XPUs
        """

        quant_config = TorchAoConfig("int4_weight_only", **self.quant_scheme_kwargs)
        quantized_model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            dtype=torch.bfloat16,
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
            dtype="auto",
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

        EXPECTED_OUTPUT = "What are we having for dinner?\n\nJane: (sighs)"
        output = quantized_model.generate(
            **input_ids, max_new_tokens=self.max_new_tokens, cache_implementation="static"
        )
        self.assertEqual(tokenizer.decode(output[0], skip_special_tokens=True), EXPECTED_OUTPUT)


@require_torchao
@require_torchao_version_greater_or_equal("0.8.0")
class TorchAoSerializationTest(unittest.TestCase):
    input_text = "What are we having for dinner?"
    max_new_tokens = 10
    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    quant_scheme = "int4_weight_only"
    quant_scheme_kwargs = (
        {"group_size": 32, "layout": Int4CPULayout()}
        if is_torchao_available() and version.parse(importlib.metadata.version("torchao")) >= version.parse("0.8.0")
        else {"group_size": 32}
    )
    device = "cpu"

    # called only once for all test in this class
    @classmethod
    def setUpClass(cls):
        cls.tokenizer = AutoTokenizer.from_pretrained(cls.model_name)
        cls.EXPECTED_OUTPUT = "What are we having for dinner?\n- 1. What is the temperature outside"

    def setUp(self):
        self.quant_config = TorchAoConfig(self.quant_scheme, **self.quant_scheme_kwargs)
        dtype = torch.bfloat16 if self.quant_scheme == "int4_weight_only" else "auto"
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
        dtype = torch.bfloat16 if self.quant_scheme == "int4_weight_only" else "auto"
        with tempfile.TemporaryDirectory() as tmpdirname:
            self.quantized_model.save_pretrained(tmpdirname, safe_serialization=False)
            loaded_quantized_model = AutoModelForCausalLM.from_pretrained(tmpdirname, dtype=dtype, device_map=device)
            input_ids = self.tokenizer(self.input_text, return_tensors="pt").to(device)

            output = loaded_quantized_model.generate(**input_ids, max_new_tokens=self.max_new_tokens)
            self.assertEqual(self.tokenizer.decode(output[0], skip_special_tokens=True), expected_output)

    def test_serialization_expected_output(self):
        self.check_serialization_expected_output(self.device, self.EXPECTED_OUTPUT)


class TorchAoSerializationW8A8CPUTest(TorchAoSerializationTest):
    quant_scheme, quant_scheme_kwargs = "int8_dynamic_activation_int8_weight", {}

    # called only once for all test in this class
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.EXPECTED_OUTPUT = "What are we having for dinner?\n\nJessica: (smiling)"

    @require_torch_accelerator
    def test_serialization_expected_output_on_accelerator(self):
        """
        Test if we can serialize on device (cpu) and load/infer the model on accelerator
        """
        self.check_serialization_expected_output(torch_device, self.EXPECTED_OUTPUT)


class TorchAoSerializationW8CPUTest(TorchAoSerializationTest):
    quant_scheme, quant_scheme_kwargs = "int8_weight_only", {}

    # called only once for all test in this class
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.EXPECTED_OUTPUT = "What are we having for dinner?\n\nJessica: (smiling)"

    @require_torch_accelerator
    def test_serialization_expected_output_on_accelerator(self):
        """
        Test if we can serialize on device (cpu) and load/infer the model on accelerator
        """
        self.check_serialization_expected_output(torch_device, self.EXPECTED_OUTPUT)


@require_torch_accelerator
class TorchAoSerializationAcceleratorTest(TorchAoSerializationTest):
    quant_scheme, quant_scheme_kwargs = "int4_weight_only", {"group_size": 32}
    device = f"{torch_device}:0"

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


@require_torch_accelerator
class TorchAoSerializationW8A8AcceleratorTest(TorchAoSerializationTest):
    quant_scheme, quant_scheme_kwargs = "int8_dynamic_activation_int8_weight", {}
    device = f"{torch_device}:0"

    # called only once for all test in this class
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.EXPECTED_OUTPUT = "What are we having for dinner?\n\nJessica: (smiling)"


@require_torch_accelerator
class TorchAoSerializationW8AcceleratorTest(TorchAoSerializationTest):
    quant_scheme, quant_scheme_kwargs = "int8_weight_only", {}
    device = f"{torch_device}:0"

    # called only once for all test in this class
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
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

        cls.quant_scheme = Float8WeightOnlyConfig()
        cls.quant_scheme_kwargs = {}

        super().setUpClass()

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

        cls.quant_scheme = Int8DynamicActivationInt4WeightConfig()
        cls.quant_scheme_kwargs = {}

        super().setUpClass()

        cls.EXPECTED_OUTPUT = "What are we having for dinner?\n\nJessica: (smiling)"


if __name__ == "__main__":
    unittest.main()
