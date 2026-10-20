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
"""Check that producer configs read without changing module targets or quantization formats."""

import unittest

from transformers.testing_utils import require_torch
from transformers.utils.quantization_config import read_compressed_tensors

from ...integrations.finegrained.utils import _compressed_tensors


@require_torch
class BlockScaledCheckpointReaderTest(unittest.TestCase):
    """How modelopt and compressed-tensors exports read into block-scaled configs."""

    def test_compressed_tensors_checkpoints_we_do_not_serve_stay_with_it(self):
        """Check that integer, per-channel FP8 and sparse checkpoints stay with compressed-tensors."""

        int4 = {"num_bits": 4, "type": "int", "group_size": 128, "strategy": "group"}
        channel = {"num_bits": 8, "type": "float", "strategy": "channel"}
        nvfp4 = {"num_bits": 4, "type": "float", "group_size": 16}
        for checkpoint in (
            _compressed_tensors(int4, fmt="pack-quantized"),
            _compressed_tensors(channel, fmt="float-quantized"),
            _compressed_tensors(nvfp4, sparsity_config={"format": "sparse-24-bitmask"}),
        ):
            self.assertIsNone(read_compressed_tensors(checkpoint))


class FineGrainedConfigGroupReaderTest(unittest.TestCase):
    def _groups(self, reverse=False, identical=False):
        nvfp4 = {"targets": ["Linear"], "weights": {"num_bits": 4, "type": "float", "group_size": 16}}
        fp8 = {
            "targets": ["Linear"],
            "weights": {"num_bits": 8, "type": "float", "strategy": "block", "block_structure": [128, 128]},
            "input_activations": {"num_bits": 8, "type": "float", "dynamic": True},
        }
        groups = [("nvfp4", nvfp4), ("other", nvfp4 if identical else fp8)]
        return dict(reversed(groups) if reverse else groups)

    def test_compressed_tensors_keeps_its_backend_for_conflicting_group_targets(self):
        for reverse in (False, True):
            with self.subTest(reverse=reverse):
                self.assertIsNone(
                    read_compressed_tensors(
                        {"format": "float-quantized", "config_groups": self._groups(reverse=reverse)}
                    )
                )


NVFP4 = {"num_bits": 4, "type": "float", "group_size": 16}
