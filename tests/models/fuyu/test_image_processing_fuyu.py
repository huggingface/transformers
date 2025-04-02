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
    @classmethod
    def setUpClass(cls):
        cls.size = {"height": 160, "width": 320}
        cls.processor = FuyuImageProcessor(size=cls.size, padding_value=1.0)
        cls.batch_size = 3
        cls.channels = 3
        cls.height = 300
        cls.width = 300

        cls.image_input = torch.rand(cls.batch_size, cls.channels, cls.height, cls.width)

        cls.image_patch_dim_h = 30
        cls.image_patch_dim_w = 30
        cls.sample_image = np.zeros((450, 210, 3), dtype=np.uint8)
        cls.sample_image_pil = Image.fromarray(cls.sample_image)

    def test_patches(self):
        expected_num_patches = self.processor.get_num_patches(image_height=self.height, image_width=self.width)

        patches_final = self.processor.patchify_image(image=self.image_input)
        assert patches_final.shape[1] == expected_num_patches, (
            f"Expected {expected_num_patches} patches, got {patches_final.shape[1]}."
        )

    def test_scale_to_target_aspect_ratio(self):
        # (h:450, w:210) fitting (160, 320) -> (160, 210*160/450)
        scaled_image = self.processor.resize(self.sample_image, size=self.size)
        self.assertEqual(scaled_image.shape[0], 160)
        self.assertEqual(scaled_image.shape[1], 74)

    def test_apply_transformation_numpy(self):
        transformed_image = self.processor.preprocess(self.sample_image).images[0][0]
        self.assertEqual(transformed_image.shape[1], 160)
        self.assertEqual(transformed_image.shape[2], 320)

    def test_apply_transformation_pil(self):
        transformed_image = self.processor.preprocess(self.sample_image_pil).images[0][0]
        self.assertEqual(transformed_image.shape[1], 160)
        self.assertEqual(transformed_image.shape[2], 320)
