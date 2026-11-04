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
    )


class Inkling2ModelTestMixin:
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
class Inkling2TextModelTest(Inkling2ModelTestMixin, CausalLMModelTest, unittest.TestCase):
    model_tester_class = Inkling2TextModelTester


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
class Inkling2VisionText2TextModelTest(Inkling2ModelTestMixin, VLMModelTest, unittest.TestCase):
    model_tester_class = Inkling2VisionText2TextModelTester
    # each image adds a class token and `spatial_merge_size**2` vision tokens per `pixel_values` row
    skip_test_image_features_output_shape = True

    @unittest.skip("Inkling2 attention is eager only: it adds a relative position bias and a sink")
    def test_can_set_attention_dynamically_composite_model(self):
        pass
