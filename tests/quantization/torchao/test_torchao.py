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
import unittest

from transformers import AutoModelForCausalLM, AutoTokenizer, TorchAoConfig
from transformers.testing_utils import (
    require_torch_gpu,
    torch_device,
)
from transformers.utils import is_torch_available, is_torchao_available


if is_torch_available():
    import torch

if is_torchao_available():
    from torchao.dtypes import AffineQuantizedTensor
    from torchao.dtypes.affine_quantized_tensor import TensorCoreTiledLayoutType


class TorchAoLLMRunner:
    def __init__(self, model_id, quant_config, compute_dtype, device):
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=compute_dtype,
            device_map=device,
            quantization_config=quant_config,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.device = self.model.device


def cleanup():
    torch.cuda.empty_cache()
    gc.collect()


def check_torchao_quantized(test_module, qlayer, batch_size=1, context_size=1024):
    weight = qlayer.weight
    test_module.assertTrue(isinstance(weight, AffineQuantizedTensor))
    test_module.assertEqual(weight.quant_min, 0)
    test_module.assertEqual(weight.quant_max, 15)
    test_module.assertTrue(isinstance(weight.layout_type, TensorCoreTiledLayoutType))


def check_forward(test_module, model, batch_size=1, context_size=1024):
    # Test forward pass
    with torch.no_grad():
        out = model(torch.zeros([batch_size, context_size], device=model.device, dtype=torch.int32)).logits
    test_module.assertEqual(out.shape[0], batch_size)
    test_module.assertEqual(out.shape[1], context_size)
    cleanup()


MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"


@require_torch_gpu
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
        with self.assertRaisesRegex(ValueError, "is not supported yet"):
            _ = TorchAoConfig("fp6")

        with self.assertRaisesRegex(ValueError, "Unexpected keyword arg"):
            _ = TorchAoConfig("int4_weight_only", group_size1=32)


@require_torch_gpu
@unittest.skipIf(not is_torchao_available(), "TorchAoTest requires torchao to be available")
class TorchAoTest(unittest.TestCase):
    def tearDown(self):
        cleanup()

    def test_int4wo_quan(self):
        """
        Simple LLM model testing int4 weight only quantization
        """
        quant_config = TorchAoConfig("int4_weight_only", group_size=32)

        # Note: we quantize the bfloat16 model on the fly to int4
        torchao_runner = TorchAoLLMRunner(
            model_id=MODEL_ID, quant_config=quant_config, compute_dtype=torch.bfloat16, device=torch_device
        )

        check_torchao_quantized(self, torchao_runner.model.model.layers[0].self_attn.v_proj)

    def test_int4wo_quant_bfloat16_conversion(self):
        """
        Testing the dtype of model will be modified to be bfloat16 for int4 weight only quantization
        """
        quant_config = TorchAoConfig("int4_weight_only", group_size=32)

        # Note: we quantize the bfloat16 model on the fly to int4
        torchao_runner = TorchAoLLMRunner(
            model_id=MODEL_ID, quant_config=quant_config, compute_dtype=torch.float16, device=torch_device
        )

        check_torchao_quantized(self, torchao_runner.model.model.layers[0].self_attn.v_proj)


if __name__ == "__main__":
    unittest.main()
