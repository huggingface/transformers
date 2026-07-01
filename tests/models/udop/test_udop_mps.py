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

    from transformers.models.udop.modeling_udop import combine_image_text_embeddings


@require_torch
class UdopMPSTest(unittest.TestCase):
    def test_combine_image_text_embeddings_mps(self):
        if not torch.backends.mps.is_available():
            self.skipTest("MPS is not available")

        # Create inputs on MPS
        image_embeddings = torch.randn(1, 196, 8, device="mps", dtype=torch.float32)
        inputs_embeds = torch.randn(1, 5, 8, device="mps", dtype=torch.float32)
        bbox = torch.rand(1, 5, 4, device="mps", dtype=torch.float32)
        visual_bbox = None
        attention_mask = torch.ones(1, 5, device="mps", dtype=torch.float32)

        # This should execute without throwing "TypeError: Cannot convert a MPS Tensor to float64 dtype"
        try:
            combine_image_text_embeddings(
                image_embeddings,
                inputs_embeds,
                bbox,
                visual_bbox,
                attention_mask,
                num_patches=14,
            )
        except TypeError as e:
            self.fail(f"combine_image_text_embeddings failed on MPS with TypeError: {e}")
