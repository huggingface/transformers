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

from parameterized import parameterized

from transformers import Inkling2Config, Inkling2TextConfig, Inkling2VisionConfig, is_torch_available
from transformers.testing_utils import require_torch, torch_device

from ...causal_lm_tester import CausalLMModelTest, CausalLMModelTester
from ...test_modeling_common import (
    TEST_EAGER_MATCHES_BATCHED_AND_GROUPED_INFERENCE_PARAMETERIZATION,
    _test_eager_matches_batched_and_grouped_inference,
    floats_tensor,
)
from ...vlm_tester import VLMModelTest, VLMModelTester


if is_torch_available():
    import torch

    from transformers import (
        Inkling2ForCausalLM,
        Inkling2ForConditionalGeneration,
        Inkling2Model,
        Inkling2TextModel,
        Inkling2VisionModel,
    )


class Inkling2TextModelTester(CausalLMModelTester):
    if is_torch_available():
        config_class = Inkling2TextConfig
        base_model_class = Inkling2TextModel
        causal_lm_class = Inkling2ForCausalLM

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.layer_types = ["hybrid_sliding", "hybrid"]
        self.swa_num_attention_heads = self.num_attention_heads
        self.swa_num_key_value_heads = self.num_key_value_heads
        self.swa_head_dim = self.head_dim
        self.n_routed_experts = 8


@require_torch
class Inkling2TextModelTest(CausalLMModelTest, unittest.TestCase):
    model_tester_class = Inkling2TextModelTester

    def _get_conv_state_shape(self, batch_size: int, config):
        # key and value convolutions, then the attention output and MLP output convolutions
        kv_size = config.num_key_value_heads * config.head_dim
        return [(batch_size, kv_size, config.conv_kernel_size)] * 2 + [
            (batch_size, config.hidden_size, config.conv_kernel_size)
        ] * 2

    @unittest.skip(
        "The dequantizing converter reads scales a dense save never writes, and chained renames leave no "
        "intermediate key after the reverse mapping"
    )
    def test_reverse_loading_mapping(self):
        pass

    @parameterized.expand(TEST_EAGER_MATCHES_BATCHED_AND_GROUPED_INFERENCE_PARAMETERIZATION)
    def test_eager_matches_batched_and_grouped_inference(self, name, dtype):
        # batched_mm runs one matrix-vector product per token, whose float32 sums round differently in low precision
        if dtype in ("fp16", "bf16"):
            self.skipTest("Low-precision rounding differences across expert kernels exceed the 1e-4 tolerance")
        _test_eager_matches_batched_and_grouped_inference(self, name, dtype)

    @unittest.skip("Decoder layers read earlier block states from a list, which reentrant checkpointing cannot track")
    def test_training_gradient_checkpointing_use_reentrant_true(self):
        pass


class Inkling2VisionText2TextModelTester(VLMModelTester):
    base_model_class = Inkling2Model
    config_class = Inkling2Config
    text_config_class = Inkling2TextConfig
    vision_config_class = Inkling2VisionConfig
    conditional_generation_class = Inkling2ForConditionalGeneration

    def __init__(self, parent, **kwargs):
        kwargs.setdefault("image_token_id", 4)
        kwargs.setdefault("patch_size", 2)
        kwargs.setdefault("spatial_merge_size", 2)
        kwargs.setdefault("num_image_tokens", 4)
        kwargs.setdefault("posemb_grid_size", 4)
        kwargs.setdefault("projection_hidden_dim", 32)
        kwargs.setdefault("layer_types", ["hybrid_sliding", "hybrid"])
        kwargs.setdefault("moe_intermediate_size", 16)
        kwargs.setdefault("n_routed_experts", 8)
        kwargs.setdefault("swa_num_attention_heads", 2)
        kwargs.setdefault("swa_num_key_value_heads", 2)
        kwargs.setdefault("swa_head_dim", 16)
        super().__init__(parent, **kwargs)
        self.text_hidden_size = self.hidden_size

    def create_pixel_values(self, batch_size: int | None = None):
        batch_size = batch_size if batch_size is not None else self.batch_size
        side = self.patch_size * self.spatial_merge_size
        return floats_tensor([batch_size * self.num_image_tokens, side, side, self.num_channels])

    def get_additional_inputs(self, config, input_ids, pixel_values, batch_size: int | None = None):
        batch_size = batch_size if batch_size is not None else self.batch_size
        return {"image_grid_thw": torch.tensor([[1, 2, 2]] * batch_size, device=torch_device)}


@require_torch
class Inkling2VisionText2TextModelTest(VLMModelTest, unittest.TestCase):
    model_tester_class = Inkling2VisionText2TextModelTester
    # each image adds a class token and `spatial_merge_size**2` vision tokens per `pixel_values` row
    skip_test_image_features_output_shape = True

    def _get_conv_state_shape(self, batch_size: int, config):
        # key and value convolutions, then the attention output and MLP output convolutions
        kv_size = config.num_key_value_heads * config.head_dim
        return [(batch_size, kv_size, config.conv_kernel_size)] * 2 + [
            (batch_size, config.hidden_size, config.conv_kernel_size)
        ] * 2

    @unittest.skip(
        "The dequantizing converter reads scales a dense save never writes, and chained renames leave no "
        "intermediate key after the reverse mapping"
    )
    def test_reverse_loading_mapping(self):
        pass

    @parameterized.expand(TEST_EAGER_MATCHES_BATCHED_AND_GROUPED_INFERENCE_PARAMETERIZATION)
    def test_eager_matches_batched_and_grouped_inference(self, name, dtype):
        # batched_mm runs one matrix-vector product per token, whose float32 sums round differently in low precision
        if dtype in ("fp16", "bf16"):
            self.skipTest("Low-precision rounding differences across expert kernels exceed the 1e-4 tolerance")
        _test_eager_matches_batched_and_grouped_inference(self, name, dtype)

    @unittest.skip("Decoder layers read earlier block states from a list, which reentrant checkpointing cannot track")
    def test_training_gradient_checkpointing_use_reentrant_true(self):
        pass

    @unittest.skip("Inkling2 attention is eager only: it adds a relative position bias and a sink")
    def test_can_set_attention_dynamically_composite_model(self):
        pass


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
