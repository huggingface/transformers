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

import unittest

from transformers import is_torch_available
from transformers.testing_utils import require_torch


if is_torch_available():
    import torch

    from transformers import initialization as init


@require_torch
class TestInitialization(unittest.TestCase):
    def test_orthogonal_low_precision(self):
        # The QR decomposition (`geqrf`) used by `torch.nn.init.orthogonal_` is not implemented for
        # low-precision dtypes, in which case the init runs in float32 and is copied back (see #47225)
        gain = 2.0
        for dtype in (torch.bfloat16, torch.float16):
            with self.subTest(dtype=dtype):
                tensor = torch.empty(16, 16, dtype=dtype)
                init.orthogonal_(tensor, gain=gain)
                self.assertEqual(tensor.dtype, dtype)
                self.assertTrue(torch.isfinite(tensor).all())
                # Rows should be orthogonal with norm equal to `gain`, up to low-precision rounding
                product = (tensor.float() @ tensor.float().T) / gain**2
                torch.testing.assert_close(product, torch.eye(16), atol=5e-2, rtol=0)

    def test_orthogonal_full_precision_matches_torch(self):
        tensor = torch.empty(16, 16)
        init.orthogonal_(tensor, generator=torch.Generator().manual_seed(0))
        expected = torch.empty(16, 16)
        torch.nn.init.orthogonal_(expected, generator=torch.Generator().manual_seed(0))
        torch.testing.assert_close(tensor, expected, atol=0, rtol=0)

    def test_orthogonal_respects_hf_initialized_flag(self):
        tensor = torch.zeros(16, 16, dtype=torch.bfloat16)
        tensor._is_hf_initialized = True
        init.orthogonal_(tensor)
        self.assertTrue((tensor == 0).all())
