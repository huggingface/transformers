import unittest

import numpy as np

from transformers import is_torch_available, is_vision_available
from transformers.testing_utils import (
    require_torch,
    require_torchvision,
    require_vision,
)


if is_torch_available() and is_vision_available():
    import torch

    from transformers import FuyuImageProcessor

if is_vision_available():
    from PIL import Image


@require_torch
@require_vision
@require_torchvision
class TestFuyuImageProcessor(unittest.TestCase):
    def setUp(self):
        self.processor = FuyuImageProcessor(target_height=160, target_width=320, padding_value=1.0)
        self.batch_size = 3
        self.channels = 3
        self.height = 300
        self.width = 300

        self.image_input = torch.rand(self.batch_size, self.channels, self.height, self.width)

        self.image_patch_dim_h = 30
        self.image_patch_dim_w = 30
        self.sample_image = np.zeros((450, 210, 3), dtype=np.uint8)
        self.sample_image_pil = Image.fromarray(self.sample_image)

    def test_patches(self):
        expected_num_patches = self.processor.get_num_patches(
            img_h=self.height, img_w=self.width, patch_dim_h=self.image_patch_dim_h, patch_dim_w=self.image_patch_dim_w
        )

        patches_final = self.processor.patchify_image(
            image=self.image_input, patch_dim_h=self.image_patch_dim_h, patch_dim_w=self.image_patch_dim_w
        )
        assert (
            patches_final.shape[1] == expected_num_patches
        ), f"Expected {expected_num_patches} patches, got {patches_final.shape[1]}."

    def test_scale_to_target_aspect_ratio(self):
        # (h:450, w:210) fitting (160, 320) -> (160, 210*160/450)
        scaled_image = self.processor._scale_to_target_aspect_ratio(self.sample_image)
        self.assertEqual(scaled_image.shape[0], 160)
        self.assertEqual(scaled_image.shape[1], 74)

    def test_apply_transformation_numpy(self):
        transformed_image = self.processor.apply_transformation(self.sample_image)
        self.assertEqual(transformed_image.shape[0], 160)
        self.assertEqual(transformed_image.shape[1], 320)

    def test_apply_transformation_pil(self):
        transformed_image = self.processor.apply_transformation(self.sample_image_pil)
        self.assertEqual(transformed_image.shape[0], 160)
        self.assertEqual(transformed_image.shape[1], 320)
