# Copyright 2026 The HuggingFace Team. All rights reserved.
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
"""Tests for MXFP8 (`Mxfp8Config`), its config and the checkpoint layouts it loads and saves back."""

import unittest

import torch

from transformers.testing_utils import (
    TestCasePlus,
    require_torch,
)
from transformers.utils.quantization_config import Mxfp8Config

from ...integrations.finegrained.utils import (
    _ChainAssertions,
    _load_and_save,
    _per_expert,
)
from ..block_scaled_checkpoint_tester import TinyCheckpointTesterMixin
from ..block_scaled_parallel_tester import LoadPathEquivalenceTesterMixin
from ..block_scaled_tester import (
    MXFP8,
    BlockScaledFormatTesterMixin,
)


@require_torch
class Mxfp8ChainTest(_ChainAssertions, unittest.TestCase):
    """MXFP8, as MiniMax-M3 ships it."""

    def test_minimax_m3_e8m0_scale_bytes(self):
        """Check that MiniMax-M3's raw `uint8` E8M0 scales reach the kernels unchanged and save back as shipped."""
        torch.manual_seed(0)
        checkpoint = _per_expert(
            lambda n, k: {
                "weight": torch.randn(n, k).to(torch.float8_e4m3fn),
                "weight_scale_inv": torch.randint(120, 130, (n, k // 32), dtype=torch.uint8),
            }
        )
        model, info, saved = _load_and_save(checkpoint, Mxfp8Config())
        self.assertLoadsCleanly(info)
        experts = model.model.layers[0].mlp.experts
        self.assertEqual(experts.gate_up_proj_scale_inv.dtype, torch.uint8)
        self.assertHoldsInterleaved(experts, checkpoint, "weight_scale_inv", experts.gate_up_proj_scale_inv)
        self.assertSavesBack(saved, checkpoint)


class Mxfp8Test(BlockScaledFormatTesterMixin, unittest.TestCase):
    spec = MXFP8


class Mxfp8TinyCheckpointTest(TinyCheckpointTesterMixin, TestCasePlus):
    spec = MXFP8


class Mxfp8LoadPathEquivalenceTest(LoadPathEquivalenceTesterMixin, TestCasePlus):
    spec = MXFP8
