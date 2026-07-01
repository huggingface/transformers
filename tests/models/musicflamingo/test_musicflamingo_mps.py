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

from transformers import is_torch_available
from transformers.testing_utils import require_torch


if is_torch_available():
    import torch

    from transformers.models.musicflamingo.modeling_musicflamingo import apply_rotary_time_emb


@require_torch
class MusicFlamingoMPSTest(unittest.TestCase):
    def test_apply_rotary_time_emb_mps(self):
        if not torch.backends.mps.is_available():
            self.skipTest("MPS is not available")

        # Create inputs on MPS
        hidden_states = torch.randn(1, 2, 4, 8, device="mps", dtype=torch.float32)
        cos = torch.randn(1, 2, 4, 4, device="mps", dtype=torch.float32)
        sin = torch.randn(1, 2, 4, 4, device="mps", dtype=torch.float32)

        # This should execute without throwing "TypeError: Cannot convert a MPS Tensor to float64 dtype"
        try:
            output = apply_rotary_time_emb(hidden_states, cos, sin)
        except TypeError as e:
            self.fail(f"apply_rotary_time_emb failed on MPS with TypeError: {e}")

        self.assertEqual(output.device.type, "mps")
        self.assertEqual(output.dtype, torch.float32)
        self.assertEqual(output.shape, hidden_states.shape)
