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
"""Testing suite for the PyTorch H-Net model."""

import unittest

from transformers import is_torch_available
from transformers.testing_utils import require_torch, slow, torch_device

from ...causal_lm_tester import CausalLMModelTest, CausalLMModelTester


if is_torch_available():
    import torch

    from transformers import HNetConfig, HNetForCausalLM, HNetModel


class HNetModelTester(CausalLMModelTester):
    if is_torch_available():
        base_model_class = HNetModel

    def get_config(self):
        # 2-stage H-Net: bytes (32) -> stage 1 (48) -> main network (64), mixing Mamba2 and attention layers
        return HNetConfig(
            vocab_size=self.vocab_size,
            hidden_sizes=[self.hidden_size, 48, 64],
            intermediate_sizes=[0, 64, 96],
            num_heads_per_stage=[2, 2, 2],
            sliding_windows=[4, 4, None],
            encoder_layer_types=[["mamba", "attention"], ["attention", "mamba"]],
            main_layer_types=["attention", "attention"],
            decoder_layer_types=[["attention", "mamba"], ["mamba", "attention"]],
            mamba_headdim=16,
            mamba_d_state=8,
            chunk_size=8,
            # The default (1M) makes common tests build inputs of `1.5 * max_position_embeddings` tokens
            max_position_embeddings=self.max_position_embeddings,
            pad_token_id=self.pad_token_id,
            bos_token_id=self.bos_token_id,
            eos_token_id=self.eos_token_id,
        )


@require_torch
class HNetModelTest(CausalLMModelTest, unittest.TestCase):
    model_tester_class = HNetModelTester


@require_torch
@slow
class HNetIntegrationTest(unittest.TestCase):
    def test_greedy_generation(self):
        model = HNetForCausalLM.from_pretrained("ArthurZ/hnet_1stage_L").to(torch_device)
        input_ids = torch.tensor([[254, *"Hello, my name is".encode()]], device=torch_device)
        output = model.generate(input_ids, max_new_tokens=20, do_sample=False)
        # Same bytes as the original implementation: " Maria and I am a st"
        EXPECTED = [32, 77, 97, 114, 105, 97, 32, 97, 110, 100, 32, 73, 32, 97, 109, 32, 97, 32, 115, 116]
        self.assertEqual(output[0, input_ids.shape[1] :].tolist(), EXPECTED)
