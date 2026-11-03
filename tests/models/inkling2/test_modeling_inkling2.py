# Copyright 2026 the HuggingFace Team. All rights reserved.
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
"""Testing suite for the PyTorch Inkling2 model."""

import importlib.util
import os
import re
import unittest

from transformers import Inkling2VisionConfig, is_torch_available
from transformers.testing_utils import require_torch


if is_torch_available():
    import torch

    from transformers import Inkling2VisionModel


REFERENCE_VISION_PATH = os.environ.get("INKLING2_REFERENCE_VISION_PATH")
REFERENCE_TO_TRANSFORMERS = [
    (r"^backbone\.embeddings\.position_embedding$", "embeddings.position_embedding.weight"),
    (r"^backbone\.embeddings\.", "embeddings."),
    (r"^backbone\.ln_pre\.", "pre_layernorm."),
    (r"^backbone\.layers\.", "layers."),
    (r"\.ln_1\.", ".layernorm_before."),
    (r"\.ln_2\.", ".layernorm_after."),
    (r"\.mlp\.c_fc\.", ".mlp.fc1."),
    (r"\.mlp\.c_proj\.", ".mlp.fc2."),
    (r"\.ls_1\.gamma$", ".lambda_1"),
    (r"\.ls_2\.gamma$", ".lambda_2"),
    (r"^proj_in\.", "merger.linear_1."),
    (r"^proj_out\.", "merger.linear_2."),
]


@require_torch
@unittest.skipUnless(REFERENCE_VISION_PATH, "set INKLING2_REFERENCE_VISION_PATH to the reference `tml/vision.py`")
class Inkling2VisionReferenceParityTest(unittest.TestCase):
    vision_config = {
        "temporal_patch_size": 2,
        "native_patch_size": 4,
        "adapter_merge_size": 2,
        "projection_hidden_dim": 96,
        "projector_bias": False,
        "hidden_size": 64,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "mlp_ratio": 4.0,
        "posemb_grid_size": 8,
        "layer_norm_eps": 1e-5,
    }
    text_hidden_size = 48
    # Merged-token grids: native grids 6x4 and 4x10 shrink the 8x8 position grid, 10x12 enlarges it.
    image_grid_thw = [[1, 3, 2], [1, 2, 5], [1, 5, 6]]

    def setUp(self):
        specification = importlib.util.spec_from_file_location("inkling2_reference_vision", REFERENCE_VISION_PATH)
        reference_vision = importlib.util.module_from_spec(specification)
        specification.loader.exec_module(reference_vision)
        # Same math as the reference's `torch.compile`d function, run eagerly
        reference_vision._renormalize = lambda x, mean, std: 2.0 * (x * std.to(x.dtype) + mean.to(x.dtype)) - 1.0

        torch.manual_seed(0)
        self.reference = reference_vision.Inkling2VisionEncoder(self.vision_config, self.text_hidden_size).eval()
        with torch.no_grad():
            for name, parameter in self.reference.named_parameters():
                base = 1.0 if re.search(r"ln_|gamma", name) else 0.0
                parameter.copy_(base + 0.1 * torch.randn_like(parameter, dtype=torch.float32))

        self.grid = torch.tensor(self.image_grid_thw)
        side = self.vision_config["native_patch_size"] * self.vision_config["adapter_merge_size"]
        num_tokens = int((self.grid[:, 1] * self.grid[:, 2]).sum())
        frame = torch.randn(num_tokens, 1, side, side, 3).bfloat16()
        self.clip_patches = frame.expand(-1, self.vision_config["temporal_patch_size"], -1, -1, -1).contiguous()
        mean, std = self.reference._clip_mean.bfloat16(), self.reference._clip_std.bfloat16()
        self.unit_patches = 2.0 * (self.clip_patches * std + mean) - 1.0

    def get_model(self, dtype):
        config = Inkling2VisionConfig(
            **self.vision_config, text_hidden_size=self.text_hidden_size, attn_implementation="eager"
        )
        model = Inkling2VisionModel(config).to(dtype).eval()
        state_dict = {}
        for name, value in self.reference.state_dict().items():
            for pattern, replacement in REFERENCE_TO_TRANSFORMERS:
                name = re.sub(pattern, replacement, name)
            state_dict[name] = value.to(dtype)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        self.assertEqual(unexpected, [])
        self.assertEqual([key for key in missing if "inv_freq" not in key], [])
        return model

    def test_matches_reference_float32(self):
        reference = self.reference.float()
        model = self.get_model(torch.float32)
        with torch.no_grad():
            expected = reference(self.clip_patches, self.grid)
            actual = model(self.unit_patches[:, 0].float(), self.grid).pooler_output
        self.assertEqual(actual.shape, expected.shape)
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5 * expected.abs().max().item())

    def test_matches_reference_bfloat16(self):
        model = self.get_model(torch.bfloat16)
        with torch.no_grad():
            expected = self.reference(self.clip_patches, self.grid).float()
            actual = model(self.unit_patches[:, 0], self.grid).pooler_output.float()
        cosine = torch.nn.functional.cosine_similarity(actual.flatten(), expected.flatten(), dim=0)
        self.assertGreater(cosine.item(), 0.9999)
