# coding=utf-8
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
    require_torch_gpu,
    require_torch_multi_gpu,
    require_torchao,
    require_torchao_version_greater_or_equal,
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
    from torchao.quantization.autoquant import AQMixin

    if version.parse(importlib.metadata.version("torchao")) >= version.parse("0.8.0"):
        from torchao.dtypes import Int4CPULayout


def check_torchao_int4_wo_quantized(test_module, qlayer):
    weight = qlayer.weight
    test_module.assertEqual(weight.quant_min, 0)
    test_module.assertEqual(weight.quant_max, 15)
    test_module.assertTrue(isinstance(weight, AffineQuantizedTensor))
    layout = Int4CPULayout if weight.device.type == "cpu" else TensorCoreTiledLayout
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
        self.assertIsInstance(d["quant_type_kwargs"]["layout"], dict)
        self.assertTrue("inner_k_tiles" in d["quant_type_kwargs"]["layout"])
        quantization_config.to_json_string(use_diff=False)


@require_torchao
@require_torchao_version_greater_or_equal("0.8.0")
class TorchAoTest(unittest.TestCase):
    input_text = "What are we having for dinner?"
    max_new_tokens = 10
    EXPECTED_OUTPUT = "What are we having for dinner?\n- 1. What is the temperature outside"
    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    device = "cpu"
    quant_scheme_kwargs = (
        {"group_size": 32, "layout": Int4CPULayout()}
        if is_torchao_available() and version.parse(importlib.metadata.version("torchao")) >= version.parse("0.8.0")
        else {"group_size": 32}
    )

    def tearDown(self):
        gc.collect()
        torch.cuda.empty_cache()
        gc.collect()

    def test_int4wo_quant(self):
        """
        Simple LLM model testing int4 weight only quantization
        """
        quant_config = TorchAoConfig("int4_weight_only", **self.quant_scheme_kwargs)

        # Note: we quantize the bfloat16 model on the fly to int4
        quantized_model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.bfloat16,
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
            torch_dtype=None,
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


@require_torch_gpu
class TorchAoGPUTest(TorchAoTest):
    device = "cuda"
    quant_scheme_kwargs = {"group_size": 32}

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
            torch_dtype=torch.bfloat16,
            device_map=device_map_offload,
            quantization_config=quant_config,
        )
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        input_ids = tokenizer(self.input_text, return_tensors="pt").to(self.device)

        output = quantized_model.generate(**input_ids, max_new_tokens=self.max_new_tokens)
        EXPECTED_OUTPUT = "What are we having for dinner?\n- 2. What is the temperature outside"

        self.assertEqual(tokenizer.decode(output[0], skip_special_tokens=True), EXPECTED_OUTPUT)

    @require_torch_multi_gpu
    def test_int4wo_quant_multi_gpu(self):
        """
        Simple test that checks if the quantized model int4 weight only is working properly with multiple GPUs
        set CUDA_VISIBLE_DEVICES=0,1 if you have more than 2 GPUs
        """

        quant_config = TorchAoConfig("int4_weight_only", **self.quant_scheme_kwargs)
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
            torch_dtype=torch.bfloat16,
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

        EXPECTED_OUTPUT = 'What are we having for dinner?\n\n10. "Dinner is ready'
        output = quantized_model.generate(
            **input_ids, max_new_tokens=self.max_new_tokens, cache_implementation="static"
        )
        self.assertEqual(tokenizer.decode(output[0], skip_special_tokens=True), EXPECTED_OUTPUT)


@require_torchao
@require_torchao_version_greater_or_equal("0.8.0")
class TorchAoSerializationTest(unittest.TestCase):
    input_text = "What are we having for dinner?"
    max_new_tokens = 10
    ORIGINAL_EXPECTED_OUTPUT = "What are we having for dinner?\n- 1. What is the temperature outside"
    # TODO: investigate why we don't have the same output as the original model for this test
    SERIALIZED_EXPECTED_OUTPUT = "What are we having for dinner?\n\nJessica: (smiling)"
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
        cls.quant_config = TorchAoConfig(cls.quant_scheme, **cls.quant_scheme_kwargs)
        cls.quantized_model = AutoModelForCausalLM.from_pretrained(
            cls.model_name,
            torch_dtype=torch.bfloat16,
            device_map=cls.device,
            quantization_config=cls.quant_config,
        )
        cls.tokenizer = AutoTokenizer.from_pretrained(cls.model_name)

    def tearDown(self):
        gc.collect()
        torch.cuda.empty_cache()
        gc.collect()

    def test_original_model_expected_output(self):
        input_ids = self.tokenizer(self.input_text, return_tensors="pt").to(self.device)
        output = self.quantized_model.generate(**input_ids, max_new_tokens=self.max_new_tokens)

        self.assertEqual(self.tokenizer.decode(output[0], skip_special_tokens=True), self.ORIGINAL_EXPECTED_OUTPUT)

    def check_serialization_expected_output(self, device, expected_output):
        """
        Test if we can serialize and load/infer the model again on the same device
        """
        with tempfile.TemporaryDirectory() as tmpdirname:
            self.quantized_model.save_pretrained(tmpdirname, safe_serialization=False)
            loaded_quantized_model = AutoModelForCausalLM.from_pretrained(
                self.model_name, torch_dtype=torch.bfloat16, device_map=device
            )
            input_ids = self.tokenizer(self.input_text, return_tensors="pt").to(device)

            output = loaded_quantized_model.generate(**input_ids, max_new_tokens=self.max_new_tokens)
            self.assertEqual(self.tokenizer.decode(output[0], skip_special_tokens=True), expected_output)

    def test_serialization_expected_output(self):
        self.check_serialization_expected_output(self.device, self.SERIALIZED_EXPECTED_OUTPUT)


class TorchAoSerializationW8A8CPUTest(TorchAoSerializationTest):
    quant_scheme, quant_scheme_kwargs = "int8_dynamic_activation_int8_weight", {}
    ORIGINAL_EXPECTED_OUTPUT = "What are we having for dinner?\n\nJessica: (smiling)"
    SERIALIZED_EXPECTED_OUTPUT = ORIGINAL_EXPECTED_OUTPUT

    @require_torch_gpu
    def test_serialization_expected_output_on_cuda(self):
        """
        Test if we can serialize on device (cpu) and load/infer the model on cuda
        """
        self.check_serialization_expected_output("cuda", self.SERIALIZED_EXPECTED_OUTPUT)


class TorchAoSerializationW8CPUTest(TorchAoSerializationTest):
    quant_scheme, quant_scheme_kwargs = "int8_weight_only", {}
    ORIGINAL_EXPECTED_OUTPUT = "What are we having for dinner?\n\nJessica: (smiling)"
    SERIALIZED_EXPECTED_OUTPUT = ORIGINAL_EXPECTED_OUTPUT

    @require_torch_gpu
    def test_serialization_expected_output_on_cuda(self):
        """
        Test if we can serialize on device (cpu) and load/infer the model on cuda
        """
        self.check_serialization_expected_output("cuda", self.SERIALIZED_EXPECTED_OUTPUT)


@require_torch_gpu
class TorchAoSerializationGPTTest(TorchAoSerializationTest):
    quant_scheme, quant_scheme_kwargs = "int4_weight_only", {"group_size": 32}
    device = "cuda:0"


@require_torch_gpu
class TorchAoSerializationW8A8GPUTest(TorchAoSerializationTest):
    quant_scheme, quant_scheme_kwargs = "int8_dynamic_activation_int8_weight", {}
    ORIGINAL_EXPECTED_OUTPUT = "What are we having for dinner?\n\nJessica: (smiling)"
    SERIALIZED_EXPECTED_OUTPUT = ORIGINAL_EXPECTED_OUTPUT
    device = "cuda:0"


@require_torch_gpu
class TorchAoSerializationW8GPUTest(TorchAoSerializationTest):
    quant_scheme, quant_scheme_kwargs = "int8_weight_only", {}
    ORIGINAL_EXPECTED_OUTPUT = "What are we having for dinner?\n\nJessica: (smiling)"
    SERIALIZED_EXPECTED_OUTPUT = ORIGINAL_EXPECTED_OUTPUT
    device = "cuda:0"


@require_torch_gpu
@require_torchao_version_greater_or_equal("0.10.0")
class TorchAoSerializationFP8GPUTest(TorchAoSerializationTest):
    ORIGINAL_EXPECTED_OUTPUT = "What are we having for dinner?\n\nJessica: (smiling)"
    SERIALIZED_EXPECTED_OUTPUT = ORIGINAL_EXPECTED_OUTPUT
    device = "cuda:0"

    def setUp(self):
        if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 9:
            raise unittest.SkipTest("CUDA compute capability 9.0 or higher required for FP8 tests")

        from torchao.quantization import Float8WeightOnlyConfig

        self.quant_scheme = Float8WeightOnlyConfig()
        self.quant_scheme_kwargs = {}
        super().setUp()


@require_torch_gpu
@require_torchao_version_greater_or_equal("0.10.0")
class TorchAoSerializationA8W4Test(TorchAoSerializationTest):
    ORIGINAL_EXPECTED_OUTPUT = "What are we having for dinner?\n\nJessica: (smiling)"
    SERIALIZED_EXPECTED_OUTPUT = ORIGINAL_EXPECTED_OUTPUT
    device = "cuda:0"

    def setUp(self):
        if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 9:
            raise unittest.SkipTest("CUDA compute capability 9.0 or higher required for FP8 tests")

        from torchao.quantization import Int8DynamicActivationInt4WeightConfig

        self.quant_scheme = Int8DynamicActivationInt4WeightConfig()
        self.quant_scheme_kwargs = {}
        super().setUp()


if __name__ == "__main__":
    unittest.main()
