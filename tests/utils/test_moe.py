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
from contextlib import contextmanager
from unittest.mock import MagicMock, patch

from transformers.testing_utils import require_torch
from transformers.utils import is_torch_available


if is_torch_available():
    import torch

    from transformers.integrations.moe import _can_use_grouped_mm


@contextmanager
def mock_torch_ops(has_functional_grouped_mm: bool, has_torch_grouped_mm: bool):
    """Context manager to simulate presence/absence of grouped_mm ops."""
    orig_f_grouped_mm = getattr(torch.nn.functional, "grouped_mm", None)
    orig_grouped_mm = getattr(torch, "_grouped_mm", None)

    try:
        if has_functional_grouped_mm:
            torch.nn.functional.grouped_mm = MagicMock()
        elif hasattr(torch.nn.functional, "grouped_mm"):
            delattr(torch.nn.functional, "grouped_mm")

        if has_torch_grouped_mm:
            torch._grouped_mm = MagicMock()
        elif hasattr(torch, "_grouped_mm"):
            delattr(torch, "_grouped_mm")

        yield
    finally:
        if orig_f_grouped_mm is not None:
            torch.nn.functional.grouped_mm = orig_f_grouped_mm
        elif hasattr(torch.nn.functional, "grouped_mm"):
            delattr(torch.nn.functional, "grouped_mm")

        if orig_grouped_mm is not None:
            torch._grouped_mm = orig_grouped_mm
        elif hasattr(torch, "_grouped_mm"):
            delattr(torch, "_grouped_mm")


@require_torch
class MoEUtilsTester(unittest.TestCase):
    def test_can_use_grouped_mm_cuda_compute_capabilities(self):
        """Test that _can_use_grouped_mm correctly respects device compute capabilities across torch versions."""
        mock_input = MagicMock()
        mock_weight = MagicMock()
        mock_offs = MagicMock()
        mock_weight.device.type = "cuda"
        mock_weight.dtype = torch.bfloat16

        # Case 1: torch <= 2.8 (where torch._grouped_mm is Hopper-only, major == 9)
        with patch("transformers.integrations.moe.is_torchdynamo_compiling", return_value=False), \
             patch("transformers.integrations.moe.is_torch_greater_or_equal", return_value=False), \
             mock_torch_ops(has_functional_grouped_mm=False, has_torch_grouped_mm=True):

            # Hopper (9, 0) should be supported
            with patch("torch.cuda.get_device_capability", return_value=(9, 0)):
                self.assertTrue(_can_use_grouped_mm(mock_input, mock_weight, mock_offs))

            # Blackwell (10, 0) and (12, 0) must NOT be admitted on torch <= 2.8
            with patch("torch.cuda.get_device_capability", return_value=(10, 0)):
                self.assertFalse(_can_use_grouped_mm(mock_input, mock_weight, mock_offs))

            with patch("torch.cuda.get_device_capability", return_value=(12, 0)):
                self.assertFalse(_can_use_grouped_mm(mock_input, mock_weight, mock_offs))

            # Ampere/Ada (8, 0)/(8, 9) not supported on torch <= 2.8
            with patch("torch.cuda.get_device_capability", return_value=(8, 9)):
                self.assertFalse(_can_use_grouped_mm(mock_input, mock_weight, mock_offs))

        # Case 2: torch >= 2.9 (where torch._grouped_mm supports SM80+ via in-torch fallback)
        with patch("transformers.integrations.moe.is_torchdynamo_compiling", return_value=False), \
             patch("transformers.integrations.moe.is_torch_greater_or_equal", return_value=True), \
             mock_torch_ops(has_functional_grouped_mm=False, has_torch_grouped_mm=True):

            # Ampere (8, 0), Ada (8, 9), Hopper (9, 0), Blackwell (10, 0), (12, 0) all supported
            for cap in [(8, 0), (8, 9), (9, 0), (10, 0), (12, 0)]:
                with patch("torch.cuda.get_device_capability", return_value=cap):
                    self.assertTrue(_can_use_grouped_mm(mock_input, mock_weight, mock_offs))

            # Turing / Volta (< 8.0) not supported
            with patch("torch.cuda.get_device_capability", return_value=(7, 5)):
                self.assertFalse(_can_use_grouped_mm(mock_input, mock_weight, mock_offs))

        # Case 3: torch.nn.functional.grouped_mm available (torch >= 2.10)
        with patch("transformers.integrations.moe.is_torchdynamo_compiling", return_value=False), \
             mock_torch_ops(has_functional_grouped_mm=True, has_torch_grouped_mm=True):

            # SM80+ supported
            for cap in [(8, 0), (8, 9), (9, 0), (10, 0), (12, 0)]:
                with patch("torch.cuda.get_device_capability", return_value=cap):
                    self.assertTrue(_can_use_grouped_mm(mock_input, mock_weight, mock_offs))

            with patch("torch.cuda.get_device_capability", return_value=(7, 5)):
                self.assertFalse(_can_use_grouped_mm(mock_input, mock_weight, mock_offs))
