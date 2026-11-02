# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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
import unittest

from huggingface_hub import hf_hub_download

from transformers import AutoConfig, FineGrainedFP8Config, Mistral4ForCausalLM, NVFP4Config
from transformers.testing_utils import require_kernels, require_torch, require_torch_gpu, slow
from transformers.utils import is_torch_available


if is_torch_available():
    import torch

# A tiny Mistral4 saved pre-quantized like Mistral-Large-4's NVFP4 release: fp8 attention and experts, nvfp4 shared
# experts. `dense_reference.pt` holds the bf16 weights it was quantized from.
REPO = "hf-internal-testing/tiny-random-Mistral4-fp8-nvfp4"


@require_torch
class QuantizationPlanTest(unittest.TestCase):
    def dense_reference(self):
        return torch.load(hf_hub_download(REPO, "dense_reference.pt"))

    def check_logits(self, model):
        config = AutoConfig.from_pretrained(REPO)
        del config.quantization_config
        dense = Mistral4ForCausalLM(config).to("cuda", torch.bfloat16).eval()
        dense.load_state_dict(self.dense_reference())
        # One token: the sm120 NVFP4 GEMM fails for more rows, its decode GEMV works
        input_ids = torch.randint(0, config.vocab_size, (1, 1), generator=torch.Generator().manual_seed(0)).cuda()
        with torch.no_grad():
            expected, logits = dense(input_ids).logits.float(), model(input_ids).logits.float()
        self.assertLess(((logits - expected).norm() / expected.norm()).item(), 0.2)

    def test_dequantized(self):
        plan = {
            "*.self_attn.*_proj*": FineGrainedFP8Config(weight_block_size=(32, 32), dequantize=True),
            "*.mlp.experts.*": FineGrainedFP8Config(weight_block_size=(128, 128), dequantize=True),
            "*.mlp.shared_experts.*": NVFP4Config(dequantize=True),
        }
        model = Mistral4ForCausalLM.from_pretrained(REPO, dtype=torch.bfloat16, quantization_config=plan)
        # Each weight dequantized from its own format: fp8 ~2.6%, nvfp4 ~9.5% away from the bf16 it came from
        for name, weight in self.dense_reference().items():
            loaded = model.state_dict()[name]
            self.assertEqual(loaded.dtype, torch.bfloat16, name)
            self.assertLess(((loaded.float() - weight.float()).norm() / weight.float().norm()).item(), 0.12, name)

    @slow
    @require_torch_gpu
    @require_kernels
    def test_quantized(self):
        if torch.cuda.get_device_capability()[0] < 10:
            self.skipTest("NVFP4 needs a Blackwell GPU")
        plan = {
            "*.self_attn.*_proj*": FineGrainedFP8Config(weight_block_size=(32, 32)),
            "*.mlp.experts.*": FineGrainedFP8Config(weight_block_size=(128, 128)),
            "*.mlp.shared_experts.*": NVFP4Config(),
        }
        model = Mistral4ForCausalLM.from_pretrained(
            REPO, dtype=torch.bfloat16, device_map="cuda", quantization_config=plan
        )
        layer = model.model.layers[0]
        self.assertEqual(type(layer.self_attn.q_b_proj).__name__, "FP8Linear")
        self.assertEqual(type(layer.mlp.experts).__name__, "FP8Experts")
        self.assertEqual(type(layer.mlp.shared_experts.gate_proj).__name__, "NVFP4Linear")
        self.assertEqual(type(model.lm_head), torch.nn.Linear)
        self.check_logits(model)

    @slow
    @require_torch_gpu
    @require_kernels
    def test_fp8_quantized_nvfp4_dequantized(self):
        plan = {
            "*.self_attn.*_proj*": FineGrainedFP8Config(weight_block_size=(32, 32)),
            "*.mlp.experts.*": FineGrainedFP8Config(weight_block_size=(128, 128)),
            "*.mlp.shared_experts.*": NVFP4Config(dequantize=True),
        }
        model = Mistral4ForCausalLM.from_pretrained(
            REPO, dtype=torch.bfloat16, device_map="cuda", quantization_config=plan
        )
        layer = model.model.layers[0]
        self.assertEqual(type(layer.mlp.experts).__name__, "FP8Experts")
        self.assertEqual(type(layer.mlp.shared_experts.gate_proj), torch.nn.Linear)
        self.check_logits(model)
