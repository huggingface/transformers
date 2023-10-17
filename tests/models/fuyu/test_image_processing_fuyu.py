import unittest

import torch

from transformers.models.fuyu.image_processing_fuyu import FuyuImageProcessor


class TestFuyuImageProcessor(unittest.TestCase):
    def setUp(self):
        self.processor = FuyuImageProcessor()
        self.batch_size = 3
        self.channels = 3
        self.height = 300
        self.width = 300

        self.image_input = torch.rand(self.batch_size, self.channels, self.height, self.width)

        self.image_patch_dim_h = 30
        self.image_patch_dim_w = 30

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
