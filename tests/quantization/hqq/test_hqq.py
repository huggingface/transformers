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

from transformers import AutoModelForCausalLM, AutoTokenizer, HQQConfig
from transformers.testing_utils import (
    require_accelerate,
    require_torch_gpu,
    slow,
)
from transformers.utils import is_accelerate_available, is_hqq_available, is_torch_available


if is_torch_available():
    import torch

if is_accelerate_available():
    pass

if is_hqq_available():
    from hqq.core.quantize import BaseQuantizeConfig, HQQBackend, HQQLinear


@require_torch_gpu
class HQQConfigTest(unittest.TestCase):
    def test_to_dict(self):
        """
        Makes sure the config format is properly set
        """
        quantization_config = HQQConfig()
        hqq_orig_config = quantization_config.to_dict()

        for key in hqq_orig_config:
            self.assertEqual(quantization_config.quant_config[key], hqq_orig_config[key])


class HQQLLMRunner:
    def __init__(self, model_id, quant_config=None, compute_dtype=torch.float16, device="cuda", cache_dir=None):
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=compute_dtype,
            device_map=device,
            quantization_config=quant_config,
            cache_dir=cache_dir,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir)
        self.device = self.model.device
        HQQLinear.set_backend(HQQBackend.PYTORCH_BACKPROP)


def cleanup():
    torch.cuda.empty_cache()
    gc.collect()


@slow
@require_torch_gpu
@require_accelerate
class HQQTest(unittest.TestCase):
    def tearDown(self):
        gc.collect()
        torch.cuda.empty_cache()
        gc.collect()

    def test_small_mistral_fp16_quantized_model(self):
        """
        Simple LLM model testing fp16
        """
        compute_dtype = torch.float16
        device = "cuda:0"
        cache_dir = None

        quant_config = BaseQuantizeConfig(nbits=4, group_size=64, offload_meta=False)
        hqq_runner = HQQLLMRunner(
            model_id="mistralai/Mistral-7B-Instruct-v0.2",
            quant_config=HQQConfig(quant_config),
            compute_dtype=compute_dtype,
            cache_dir=cache_dir,
            device=device,
        )

        batch_size, context_size = 1, 1024

        # Test HQQ layer
        hqq_layer = hqq_runner.model.model.layers[10].self_attn.v_proj
        W_r = hqq_layer.dequantize()
        x = torch.randn((batch_size, context_size, 4096), device=device, dtype=compute_dtype) / 10.0
        with torch.no_grad():
            y = hqq_layer(x)
        self.assertEqual(y.shape[-1], W_r.shape[0])
        self.assertEqual(y.dtype, compute_dtype)

        del W_r, x, y
        cleanup()

        # Test forward pass
        with torch.no_grad():
            out = hqq_runner.model(
                torch.zeros([batch_size, context_size], device=hqq_runner.model.device, dtype=torch.int32)
            ).logits
        self.assertEqual(out.shape[0], batch_size)
        self.assertEqual(out.shape[1], context_size)

    def test_mistral_bfp16_offloading_quantized_model(self):
        """
        Simple LLM model testing bfp16 with offfloading
        """
        compute_dtype = torch.bfloat16
        device = "cuda:0"
        cache_dir = None

        quant_config = {
            "self_attn.q_proj": BaseQuantizeConfig(nbits=4, group_size=64, offload_meta=False),
            "self_attn.k_proj": BaseQuantizeConfig(nbits=3, group_size=64, offload_meta=True),
            "self_attn.v_proj": BaseQuantizeConfig(nbits=2, group_size=64, offload_meta=True),
            "self_attn.o_proj": BaseQuantizeConfig(nbits=2, group_size=64, offload_meta=True),
            "mlp.gate_proj": BaseQuantizeConfig(nbits=4, group_size=64, offload_meta=False),
            "mlp.up_proj": BaseQuantizeConfig(nbits=4, group_size=64, offload_meta=False),
            "mlp.down_proj": BaseQuantizeConfig(nbits=4, group_size=64, offload_meta=False),
        }

        hqq_runner = HQQLLMRunner(
            model_id="mistralai/Mistral-7B-Instruct-v0.2",
            quant_config=HQQConfig(quant_config),
            compute_dtype=compute_dtype,
            cache_dir=cache_dir,
            device=device,
        )

        batch_size, context_size = 1, 1024

        # Test HQQ layer
        hqq_layer = hqq_runner.model.model.layers[10].self_attn.v_proj
        W_r = hqq_layer.dequantize()
        x = torch.randn((batch_size, context_size, 4096), device=device, dtype=compute_dtype) / 10.0
        with torch.no_grad():
            y = hqq_layer(x)
        self.assertEqual(y.shape[-1], W_r.shape[0])
        self.assertEqual(y.dtype, compute_dtype)

        # Check device
        self.assertEqual(hqq_layer.W_q.device.type, "cuda")
        self.assertEqual(hqq_layer.meta["zero_scale"].device.type, "cpu")

        del W_r, x, y
        cleanup()

        # Test forward pass
        with torch.no_grad():
            out = hqq_runner.model(
                torch.zeros([batch_size, context_size], device=hqq_runner.model.device, dtype=torch.int32)
            ).logits
        self.assertEqual(out.shape[0], batch_size)
        self.assertEqual(out.shape[1], context_size)
