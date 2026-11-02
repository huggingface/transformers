# Copyright 2026 HuggingFace Inc.
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

import numpy as np
import torch
import torchvision.transforms.v2.functional as tvF
from PIL import Image

from transformers.image_processing_backends import TorchvisionBackend
from transformers.testing_utils import require_torch, require_vision


@require_torch
@require_vision
class TorchvisionBackendRescaleAndNormalizeTest(unittest.TestCase):
    def setUp(self):
        self.backend = TorchvisionBackend()
        self.mean = (0.48145466, 0.4578275, 0.40821073)
        self.std = (0.26862954, 0.26130258, 0.27577711)
        self.rescale_factor = 1 / 255.0

    def _reference_rescale_and_normalize(self, images: torch.Tensor) -> torch.Tensor:
        image_mean = torch.tensor(self.mean, device=images.device) * (1.0 / self.rescale_factor)
        image_std = torch.tensor(self.std, device=images.device) * (1.0 / self.rescale_factor)
        images_f32 = images.to(torch.float32)
        return tvF.normalize(images_f32, image_mean, image_std, inplace=False)

    def test_multi_image_uint8_matches_out_of_place_reference(self):
        """Regression for #50100: outputs must match explicit out-of-place normalization."""
        rng = np.random.default_rng(0)
        batch = torch.stack([torch.from_numpy(rng.integers(0, 256, (3, 512, 512), dtype=np.uint8)) for _ in range(30)])
        expected = self._reference_rescale_and_normalize(batch)
        actual = self.backend.rescale_and_normalize(
            batch,
            do_rescale=True,
            rescale_factor=self.rescale_factor,
            do_normalize=True,
            image_mean=self.mean,
            image_std=self.std,
        )
        torch.testing.assert_close(actual, expected)

    def test_uint8_inplace_and_out_of_place_normalize_are_equivalent(self):
        rng = np.random.default_rng(1)
        batch = torch.stack([torch.from_numpy(rng.integers(0, 256, (3, 128, 128), dtype=np.uint8)) for _ in range(4)])
        image_mean = torch.tensor(self.mean) * (1.0 / self.rescale_factor)
        image_std = torch.tensor(self.std) * (1.0 / self.rescale_factor)
        out_of_place = tvF.normalize(batch.to(torch.float32), image_mean, image_std, inplace=False)
        in_place = batch.to(torch.float32)
        tvF.normalize(in_place, image_mean, image_std, inplace=True)
        torch.testing.assert_close(
            self.backend.rescale_and_normalize(
                batch,
                do_rescale=True,
                rescale_factor=self.rescale_factor,
                do_normalize=True,
                image_mean=self.mean,
                image_std=self.std,
            ),
            out_of_place,
        )
        torch.testing.assert_close(in_place, out_of_place)

    def test_qwen_multi_image_output_matches_pre_48897_inplace_path(self):
        """Full Qwen2-VL preprocessing output is unchanged vs the #48897 in-place normalize path."""
        from transformers import AutoImageProcessor

        rng = np.random.default_rng(2)
        images = [Image.fromarray(rng.integers(0, 256, (512, 512, 3), dtype=np.uint8)) for _ in range(4)]
        proc = AutoImageProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")
        original = TorchvisionBackend.rescale_and_normalize

        def rescale_and_normalize_inplace(
            self, images, do_rescale, rescale_factor, do_normalize, image_mean, image_std
        ):
            image_mean_t, image_std_t, do_rescale = self._fuse_mean_std_and_rescale_factor(
                do_normalize=do_normalize,
                image_mean=image_mean,
                image_std=image_std,
                do_rescale=do_rescale,
                rescale_factor=rescale_factor,
                device=images.device,
            )
            if do_normalize:
                inplace = images.dtype != torch.float32
                images = self.normalize(images.to(torch.float32), image_mean_t, image_std_t, inplace=inplace)
            elif do_rescale:
                images = self.rescale(images, rescale_factor)
            return images

        TorchvisionBackend.rescale_and_normalize = rescale_and_normalize_inplace
        try:
            expected = proc(images=images, return_tensors="pt")
        finally:
            TorchvisionBackend.rescale_and_normalize = original

        actual = proc(images=images, return_tensors="pt")
        torch.testing.assert_close(actual["pixel_values"], expected["pixel_values"])
        torch.testing.assert_close(actual["image_grid_thw"], expected["image_grid_thw"])
