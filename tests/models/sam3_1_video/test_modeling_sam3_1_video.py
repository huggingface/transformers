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
"""Testing suite for the PyTorch SAM 3.1 video model."""

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

    from transformers import Sam31VideoConfig, Sam31VideoModel


class Sam31VideoModelTester:
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

    def get_tracker_config_dict(self, image_size=None):
        image_size = self.image_size if image_size is None else image_size
        patch_grid_size = image_size // self.patch_size
        backbone_feature_sizes = [
            [patch_grid_size * 4, patch_grid_size * 4],
            [patch_grid_size * 2, patch_grid_size * 2],
            [patch_grid_size, patch_grid_size],
        ]
        return {
            "model_type": "sam3_1_tracker_video",
            "vision_config": {
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
            },
            "prompt_encoder_config": {
                "hidden_size": self.hidden_size,
                "image_size": image_size,
                "patch_size": self.patch_size,
            },
            "mask_decoder_config": {
                "hidden_size": self.hidden_size,
                "mlp_dim": self.intermediate_size,
                "num_hidden_layers": 2,
                "num_attention_heads": self.num_attention_heads,
                "iou_head_hidden_dim": self.hidden_size,
                "num_multimask_outputs": self.num_multimask_outputs,
                "multiplex_count": self.multiplex_count,
            },
            "image_size": image_size,
            "num_maskmem": 3,
            "memory_attention_hidden_size": self.hidden_size,
            "memory_attention_num_layers": 2,
            "memory_attention_num_attention_heads": 2,
            "memory_attention_feed_forward_hidden_size": self.intermediate_size,
            "memory_encoder_hidden_size": self.hidden_size,
            "memory_encoder_output_channels": self.hidden_size // 2,
            "mask_downsampler_embed_dim": self.hidden_size,
            "memory_fuser_embed_dim": self.hidden_size,
            "memory_fuser_intermediate_dim": self.intermediate_size,
            "multiplex_count": self.multiplex_count,
            "memory_attention_rope_feat_sizes": [patch_grid_size, patch_grid_size],
        }

    def get_detector_config_dict(self, image_size=None):
        image_size = self.image_size if image_size is None else image_size
        patch_grid_size = image_size // self.patch_size
        return {
            "model_type": "sam3",
            "vision_config": {
                "model_type": "sam3_vision_model",
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
            },
            "text_config": {
                "model_type": "clip_text_model",
                "hidden_size": self.hidden_size,
                "intermediate_size": self.intermediate_size,
                "num_hidden_layers": 2,
                "num_attention_heads": self.num_attention_heads,
                "vocab_size": 100,
                "max_position_embeddings": 16,
            },
            "geometry_encoder_config": {
                "hidden_size": self.hidden_size,
                "num_attention_heads": self.num_attention_heads,
                "intermediate_size": self.intermediate_size,
            },
            "detr_encoder_config": {
                "hidden_size": self.hidden_size,
                "num_attention_heads": self.num_attention_heads,
                "intermediate_size": self.intermediate_size,
                "num_hidden_layers": 1,
            },
            "detr_decoder_config": {
                "hidden_size": self.hidden_size,
                "num_attention_heads": self.num_attention_heads,
                "intermediate_size": self.intermediate_size,
                "num_hidden_layers": 1,
            },
            "mask_decoder_config": {
                "hidden_size": self.hidden_size,
                "num_attention_heads": self.num_attention_heads,
                "intermediate_size": self.intermediate_size,
            },
            "image_size": image_size,
        }

    def get_config(self, image_size=None):
        image_size = self.image_size if image_size is None else image_size
        return Sam31VideoConfig(
            detector_config=self.get_detector_config_dict(image_size),
            tracker_config=self.get_tracker_config_dict(image_size),
            low_res_mask_size=image_size // 4,
            hotstart_delay=0,  # disable hotstart buffering in unit tests
        )


@require_torch
class Sam31VideoModelTest(unittest.TestCase):
    def setUp(self):
        self.model_tester = Sam31VideoModelTester(self)

    def tearDown(self):
        gc.collect()
        backend_empty_cache(torch_device)

    def test_config_round_trip(self):
        config = self.model_tester.get_config()
        with tempfile.TemporaryDirectory() as tmp_dir:
            config.save_pretrained(tmp_dir)
            reloaded = Sam31VideoConfig.from_pretrained(tmp_dir)
        self.assertEqual(reloaded.low_res_mask_size, config.low_res_mask_size)
        self.assertTrue(reloaded.det_nms_use_iom)
        self.assertTrue(reloaded.use_iom_recondition)
        self.assertFalse(reloaded.suppress_unmatched_only_within_hotstart)

    def test_model_init(self):
        config = self.model_tester.get_config()
        model = Sam31VideoModel(config).to(torch_device).eval()
        # Detector should not own a vision encoder (shared TriNeck lives on tracker)
        self.assertIsNone(model.detector_model.vision_encoder)
        self.assertIsNotNone(model.tracker_model.vision_encoder)
        self.assertTrue(hasattr(model.tracker_model, "multiplex_controller"))
        self.assertTrue(hasattr(model.tracker_model, "propagation_mask_decoder"))

    def test_model_can_be_saved_and_reloaded(self):
        config = self.model_tester.get_config()
        model = Sam31VideoModel(config).to(torch_device).eval()
        with tempfile.TemporaryDirectory() as tmp_dir:
            model.save_pretrained(tmp_dir)
            reloaded = Sam31VideoModel.from_pretrained(tmp_dir).to(torch_device).eval()
        self.assertIsNone(reloaded.detector_model.vision_encoder)
        self.assertEqual(
            model.tracker_model.multiplex_controller.multiplex_count,
            reloaded.tracker_model.multiplex_controller.multiplex_count,
        )
