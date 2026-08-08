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
"""Init-only tests for Qwen3-Next GatedDeltaNet (no full model-tester imports)."""

import unittest

from transformers import is_torch_available
from transformers.testing_utils import require_torch


if is_torch_available():
    import torch

    from transformers import Qwen3NextConfig, Qwen3NextModel
    from transformers.models.qwen3_next.modeling_qwen3_next import Qwen3NextGatedDeltaNet


@require_torch
class Qwen3NextGatedDeltaNetInitTest(unittest.TestCase):
    def test_gated_delta_net_a_log_finite_under_bf16(self):
        """Regression for #47831: bf16 uniform init must not produce -inf A_log.

        Sampling A in the module dtype can round to 0 under bfloat16; log(0) is
        -inf and freezes that head. Init must keep A_log finite for every head,
        including after _init_weights re-initialization.
        """
        config = Qwen3NextConfig(
            hidden_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            intermediate_size=128,
            vocab_size=256,
            head_dim=16,
            linear_conv_kernel_dim=2,
            linear_key_head_dim=16,
            linear_value_head_dim=16,
            linear_num_key_heads=16,
            # Large enough that the old bf16 path almost always hit dead heads.
            linear_num_value_heads=32,
            layer_types=["linear_attention", "full_attention"],
        )

        prev_dtype = torch.get_default_dtype()
        try:
            torch.set_default_dtype(torch.bfloat16)
            torch.manual_seed(0)
            layer = Qwen3NextGatedDeltaNet(config, layer_idx=0)
            self.assertTrue(
                torch.isfinite(layer.A_log).all(),
                f"A_log has non-finite values after bf16 construction: {layer.A_log}",
            )

            torch.manual_seed(1)
            model = Qwen3NextModel(config)
            model._init_weights(layer)
            self.assertTrue(
                torch.isfinite(layer.A_log).all(),
                f"A_log has non-finite values after _init_weights under bf16: {layer.A_log}",
            )
        finally:
            torch.set_default_dtype(prev_dtype)
