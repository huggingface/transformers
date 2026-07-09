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
"""Testing suite for the PyTorch SAM 3.1 tracker video model."""

import gc
import tempfile
import unittest

from transformers.testing_utils import (
    backend_empty_cache,
    require_torch,
    torch_device,
)
from transformers.utils import is_torch_available


if is_torch_available():
    import torch

    from transformers import (
        Sam31MultiplexController,
        Sam31TrackerVideoConfig,
        Sam31TrackerVideoInferenceSession,
        Sam31TrackerVideoModel,
        Sam31VisionModel,
    )


class Sam31TrackerVideoModelTester:
    def __init__(
        self,
        parent,
        batch_size=1,
        num_channels=3,
        image_size=112,
        patch_size=14,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=64,
        num_multimask_outputs=3,
        multiplex_count=2,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.num_channels = num_channels
        self.image_size = image_size
        self.patch_size = patch_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.num_multimask_outputs = num_multimask_outputs
        self.multiplex_count = multiplex_count

    def get_config(self, image_size=None):
        image_size = self.image_size if image_size is None else image_size
        patch_grid_size = image_size // self.patch_size
        backbone_feature_sizes = [
            [patch_grid_size * 4, patch_grid_size * 4],
            [patch_grid_size * 2, patch_grid_size * 2],
            [patch_grid_size, patch_grid_size],
        ]

        vision_config = {
            "model_type": "sam3_1_vision_model",
            "backbone_config": {
                "model_type": "sam3_vit_model",
                "hidden_size": self.hidden_size,
                "num_hidden_layers": self.num_hidden_layers,
                "num_attention_heads": self.num_attention_heads,
                "intermediate_size": self.intermediate_size,
                "image_size": image_size,
                "patch_size": self.patch_size,
                "window_size": patch_grid_size,
                "global_attn_indexes": list(range(self.num_hidden_layers)),
            },
            "fpn_hidden_size": self.hidden_size,
            "backbone_feature_sizes": backbone_feature_sizes,
            "scale_factors": [4.0, 2.0, 1.0],
        }

        return Sam31TrackerVideoConfig(
            vision_config=vision_config,
            prompt_encoder_config={
                "hidden_size": self.hidden_size,
                "image_size": image_size,
                "patch_size": self.patch_size,
            },
            mask_decoder_config={
                "hidden_size": self.hidden_size,
                "mlp_dim": self.intermediate_size,
                "num_hidden_layers": 2,
                "num_attention_heads": self.num_attention_heads,
                "iou_head_hidden_dim": self.hidden_size,
                "num_multimask_outputs": self.num_multimask_outputs,
                "multiplex_count": self.multiplex_count,
            },
            image_size=image_size,
            num_maskmem=3,
            memory_attention_hidden_size=self.hidden_size,
            memory_attention_num_layers=2,
            memory_attention_num_attention_heads=2,
            memory_attention_feed_forward_hidden_size=self.intermediate_size,
            memory_encoder_hidden_size=self.hidden_size,
            memory_encoder_output_channels=self.hidden_size // 2,
            mask_downsampler_embed_dim=self.hidden_size,
            memory_fuser_embed_dim=self.hidden_size,
            memory_fuser_intermediate_dim=self.intermediate_size,
            multiplex_count=self.multiplex_count,
            memory_attention_rope_feat_sizes=[patch_grid_size, patch_grid_size],
        )


@require_torch
class Sam31TrackerVideoModelTest(unittest.TestCase):
    def setUp(self):
        self.model_tester = Sam31TrackerVideoModelTester(self)

    def tearDown(self):
        gc.collect()
        backend_empty_cache(torch_device)

    def test_config_round_trip(self):
        config = self.model_tester.get_config()
        with tempfile.TemporaryDirectory() as tmp_dir:
            config.save_pretrained(tmp_dir)
            reloaded = Sam31TrackerVideoConfig.from_pretrained(tmp_dir)
        self.assertEqual(reloaded.image_size, config.image_size)
        self.assertEqual(reloaded.multiplex_count, config.multiplex_count)
        self.assertEqual(reloaded.memory_attention_num_attention_heads, 2)

    def test_model_init_has_multiplex_components(self):
        config = self.model_tester.get_config()
        model = Sam31TrackerVideoModel(config).to(torch_device).eval()
        self.assertIsNotNone(model.vision_encoder)
        self.assertIsInstance(model.multiplex_controller, Sam31MultiplexController)
        self.assertTrue(hasattr(model, "propagation_mask_decoder"))
        self.assertTrue(hasattr(model, "interactive_object_pointer_proj"))
        self.assertEqual(model.multiplex_controller.multiplex_count, self.model_tester.multiplex_count)
        # eval_multiplex_count=1 → single-object buckets at eval time (Meta parity)
        self.assertEqual(model.multiplex_controller.allowed_bucket_capacity, 1)

    def test_vision_model_forward(self):
        config = self.model_tester.get_config()
        model = Sam31VisionModel(config.vision_config).to(torch_device).eval()
        pixel_values = torch.randn(
            self.model_tester.batch_size,
            self.model_tester.num_channels,
            self.model_tester.image_size,
            self.model_tester.image_size,
            device=torch_device,
        )
        with torch.no_grad():
            outputs = model(pixel_values=pixel_values)
        self.assertIsNotNone(outputs.last_hidden_state)
        self.assertIsNotNone(outputs.sam3_fpn_hidden_states)
        self.assertEqual(len(outputs.sam3_fpn_hidden_states), 3)
        self.assertIsNotNone(outputs.propagation_fpn_hidden_states)
        self.assertIsNotNone(outputs.interactive_fpn_hidden_states)

    def test_model_can_be_saved_and_reloaded(self):
        config = self.model_tester.get_config()
        model = Sam31TrackerVideoModel(config).to(torch_device).eval()
        with tempfile.TemporaryDirectory() as tmp_dir:
            model.save_pretrained(tmp_dir)
            reloaded = Sam31TrackerVideoModel.from_pretrained(tmp_dir).to(torch_device).eval()
        self.assertEqual(
            model.multiplex_controller.multiplex_count,
            reloaded.multiplex_controller.multiplex_count,
        )
        # Weight parity on a multiplex-specific parameter
        torch.testing.assert_close(
            model.propagation_mask_decoder.mask_tokens.weight,
            reloaded.propagation_mask_decoder.mask_tokens.weight,
        )

    def test_inference_session_dtype_string(self):
        session = Sam31TrackerVideoInferenceSession(
            video_height=64, video_width=64, dtype="float32", inference_device="cpu"
        )
        self.assertEqual(session.dtype, torch.float32)
        frame = torch.randn(1, 3, 64, 64)
        idx = session.add_new_frame(frame, frame_idx=0)
        self.assertEqual(idx, 0)
        self.assertEqual(session.get_frame(0).shape[-2:], (64, 64))

    def test_multiplex_controller_buckets(self):
        controller = Sam31MultiplexController(multiplex_count=4, eval_multiplex_count=1)
        controller.eval()
        state = controller.get_state(num_valid_entries=3, device=torch.device("cpu"), dtype=torch.float32, random=False)
        self.assertEqual(state.num_buckets, 3)  # one object per bucket in eval
        self.assertEqual(state.allowed_bucket_capacity, 1)
        self.assertEqual(state.total_valid_entries, 3)
